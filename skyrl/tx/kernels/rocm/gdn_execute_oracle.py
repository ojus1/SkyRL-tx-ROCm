"""Import-light NumPy oracle for the exact S512 GDN execute stage.

This module consumes the output of :mod:`gdn_prepare_oracle` but deliberately
does not import or reimplement preparation.  ``prepared_u``, ``prepared_w``,
and ``gamma`` must use the canonical token-major layouts returned by
``gdn_prepare_s512_numpy``: ``[1,512,32,128]``, ``[1,512,32,128]``, and
``[1,512,32]``.  The eight ``[64]`` chunks are views formed internally;
chunk-major external arrays are not accepted.

Query and key are already masked, FP32-normalized, and query-scaled at this
boundary.  Value head ``hv`` reads key head ``hv // 2``.  The oracle keeps all
execute arithmetic and the boundary state in FP32.  It optionally performs
the intended BF16 output-boundary cast through a lazy ``ml_dtypes`` import.

This is a CPU-only semantic gate, not a runtime fallback or a GPU kernel.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


GDN_EXECUTE_S512_QUERY_SHAPE = (1, 512, 16, 128)
GDN_EXECUTE_S512_PREPARED_SHAPE = (1, 512, 32, 128)
GDN_EXECUTE_S512_GAMMA_SHAPE = (1, 512, 32)
GDN_EXECUTE_S512_STATE_SHAPE = (1, 32, 128, 128)
GDN_EXECUTE_S512_OUTPUT_SHAPE = GDN_EXECUTE_S512_PREPARED_SHAPE

GDN_EXECUTE_S512_INPUT_BYTES = 27_328_512
GDN_EXECUTE_S512_BF16_OUTPUT_BYTES = 4_194_304
GDN_EXECUTE_S512_STATE_BYTES = 2_097_152
GDN_EXECUTE_S512_OUTPUT_TENSOR_BYTES = 6_291_456
# A future two-result XLA custom call also has a 16-byte tuple root.  The CPU
# oracle returns ordinary arrays and never materializes or counts that root.
GDN_EXECUTE_S512_COMPILED_OUTPUT_BYTES = 6_291_472

_CHUNKS = 8
_CHUNK_SIZE = 64
_VALUE_HEADS = 32
_VALUE_TO_KEY_HEAD = np.arange(_VALUE_HEADS, dtype=np.intp) // 2

_INPUT_SPECS = (
    ("query", GDN_EXECUTE_S512_QUERY_SHAPE),
    ("key", GDN_EXECUTE_S512_QUERY_SHAPE),
    ("prepared_u", GDN_EXECUTE_S512_PREPARED_SHAPE),
    ("prepared_w", GDN_EXECUTE_S512_PREPARED_SHAPE),
    ("gamma", GDN_EXECUTE_S512_GAMMA_SHAPE),
    ("initial_state", GDN_EXECUTE_S512_STATE_SHAPE),
)


def _require_f32_c_array(array: np.ndarray, name: str, shape: tuple[int, ...]) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.shape != shape:
        raise ValueError(f"{name} shape must be exactly {shape}, got {array.shape}")
    if array.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} dtype must be exactly float32")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")


def _validate_inputs(arrays: tuple[np.ndarray, ...]) -> None:
    for array, (name, shape) in zip(arrays, _INPUT_SPECS, strict=True):
        _require_f32_c_array(array, name, shape)

    observed_bytes = sum(array.nbytes for array in arrays)
    if observed_bytes != GDN_EXECUTE_S512_INPUT_BYTES:
        raise ValueError(
            "exact S512 execute inputs must total "
            f"{GDN_EXECUTE_S512_INPUT_BYTES} bytes, got {observed_bytes}"
        )

    for left_index, (left_name, _) in enumerate(_INPUT_SPECS):
        for right_index in range(left_index + 1, len(arrays)):
            right_name = _INPUT_SPECS[right_index][0]
            if np.shares_memory(arrays[left_index], arrays[right_index]):
                raise ValueError(
                    "exact S512 execute inputs must use distinct, non-overlapping "
                    f"buffers; {left_name} overlaps {right_name}"
                )


def _bfloat16_dtype() -> np.dtype:
    try:
        module = importlib.import_module("ml_dtypes")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "BF16 output was requested but ml_dtypes is not installed"
        ) from error
    dtype = getattr(module, "bfloat16", None)
    if dtype is None:
        raise RuntimeError("installed ml_dtypes does not expose bfloat16")
    return np.dtype(dtype)


def gdn_execute_s512_numpy(
    query: "NDArray[np.float32]",
    key: "NDArray[np.float32]",
    prepared_u: "NDArray[np.float32]",
    prepared_w: "NDArray[np.float32]",
    gamma: "NDArray[np.float32]",
    initial_state: "NDArray[np.float32]",
    *,
    output_bfloat16: bool = True,
) -> tuple[np.ndarray, "NDArray[np.float32]"]:
    """Execute eight exact 64-token GDN chunks.

    ``output_bfloat16=True`` matches the planned operation boundary and lazily
    requires ``ml_dtypes``.  Passing ``False`` retains the pre-cast FP32 output
    for tight oracle comparisons.  The returned state is always a fresh,
    C-contiguous FP32 array, and no input is mutated.
    """
    if type(output_bfloat16) is not bool:
        raise TypeError("output_bfloat16 must be an exact bool")
    arrays = (query, key, prepared_u, prepared_w, gamma, initial_state)
    _validate_inputs(arrays)
    output_dtype = _bfloat16_dtype() if output_bfloat16 else None

    query_chunks = query.reshape(1, _CHUNKS, _CHUNK_SIZE, 16, 128).transpose(
        0, 1, 3, 2, 4
    )
    key_chunks = key.reshape(1, _CHUNKS, _CHUNK_SIZE, 16, 128).transpose(0, 1, 3, 2, 4)
    u_chunks = prepared_u.reshape(1, _CHUNKS, _CHUNK_SIZE, _VALUE_HEADS, 128).transpose(
        0, 1, 3, 2, 4
    )
    w_chunks = prepared_w.reshape(1, _CHUNKS, _CHUNK_SIZE, _VALUE_HEADS, 128).transpose(
        0, 1, 3, 2, 4
    )
    gamma_chunks = gamma.reshape(1, _CHUNKS, _CHUNK_SIZE, _VALUE_HEADS).transpose(
        0, 1, 3, 2
    )

    state = initial_state.copy()
    output_chunks = np.empty(
        (1, _CHUNKS, _VALUE_HEADS, _CHUNK_SIZE, 128), dtype=np.float32
    )

    for chunk_index in range(_CHUNKS):
        query_chunk = np.take(query_chunks[:, chunk_index], _VALUE_TO_KEY_HEAD, axis=1)
        key_chunk = np.take(key_chunks[:, chunk_index], _VALUE_TO_KEY_HEAD, axis=1)
        gamma_chunk = gamma_chunks[:, chunk_index]

        # Gamma[i,j] = gamma[i] / gamma[j] on and below the diagonal.
        decay = np.tril(
            np.asarray(
                gamma_chunk[..., :, None] / gamma_chunk[..., None, :],
                dtype=np.float32,
            )
        )
        corrected = np.asarray(
            u_chunks[:, chunk_index] - np.matmul(w_chunks[:, chunk_index], state),
            dtype=np.float32,
        )
        attention = np.asarray(
            np.matmul(query_chunk, np.swapaxes(key_chunk, -1, -2)) * decay,
            dtype=np.float32,
        )
        inter_chunk = np.matmul(query_chunk * gamma_chunk[..., :, None], state).astype(
            np.float32, copy=False
        )
        output_chunks[:, chunk_index] = inter_chunk + np.matmul(attention, corrected)

        reverse_decay = decay[..., -1, :, None]
        state = np.asarray(
            gamma_chunk[..., -1, None, None] * state
            + np.matmul(np.swapaxes(key_chunk * reverse_decay, -1, -2), corrected),
            dtype=np.float32,
        )

    output = np.ascontiguousarray(
        output_chunks.transpose(0, 1, 3, 2, 4).reshape(GDN_EXECUTE_S512_OUTPUT_SHAPE)
    )
    state = np.ascontiguousarray(state, dtype=np.float32)
    if output_dtype is not None:
        output = np.ascontiguousarray(output.astype(output_dtype))
    return output, state
