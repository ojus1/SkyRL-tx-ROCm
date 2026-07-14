"""Import-light composed reverse oracle for one exact S512 GDN superblock.

This is the transpose of the transformed native operation boundary, not of the
raw model-side inputs.  Query and key are already masked, normalized in the
model input dtype, promoted to FP32, and query is already scaled.  Value, g,
and beta are already masked and promoted to FP32.  Normalization and masking
remain outside this boundary and are differentiated by their caller.

The reverse deliberately does not accept or save prepared U, W, or gamma.  It
recomputes them with :mod:`gdn_prepare_oracle`, applies the execute transpose in
:mod:`gdn_execute_reverse_oracle`, applies the prepare transpose in
:mod:`gdn_prepare_reverse_oracle`, and combines execute dK followed by prepare
dK in a pinned FP32 order.  The exact public boundary accepts a BF16 output
cotangent and an FP32 final-state cotangent; the execute transpose widens BF16
before doing any reverse arithmetic.  Every returned boundary gradient is
FP32.

The module imports no JAX and does not initialize an accelerator backend, load
or register native code, or dispatch a device operation.  It is a CPU semantic
gate and is not a runtime fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from skyrl.tx.kernels.rocm import gdn_execute_reverse_oracle as execute_reverse
from skyrl.tx.kernels.rocm import gdn_prepare_oracle as prepare_forward
from skyrl.tx.kernels.rocm import gdn_prepare_reverse_oracle as prepare_reverse

if TYPE_CHECKING:
    from numpy.typing import NDArray


GDN_SUPERBLOCK_S512_QUERY_SHAPE = (1, 512, 16, 128)
GDN_SUPERBLOCK_S512_VALUE_SHAPE = (1, 512, 32, 128)
GDN_SUPERBLOCK_S512_GATE_SHAPE = (1, 512, 32)
GDN_SUPERBLOCK_S512_STATE_SHAPE = (1, 32, 128, 128)
GDN_SUPERBLOCK_S512_OUTPUT_SHAPE = GDN_SUPERBLOCK_S512_VALUE_SHAPE

GDN_SUPERBLOCK_S512_PRIMAL_BYTES = 19_005_440
GDN_SUPERBLOCK_S512_COTANGENT_BYTES = 6_291_456
GDN_SUPERBLOCK_S512_REVERSE_INPUT_BYTES = 25_296_896
GDN_SUPERBLOCK_S512_GRADIENT_BYTES = 19_005_440

_CHUNKS = 8
_CHUNK = 64
_KEY_HEADS = 16
_VALUE_HEADS = 32
_HEADS_PER_KEY = 2
_HEAD_DIMENSION = 128

_F32_INPUT_SPECS = (
    ("query", GDN_SUPERBLOCK_S512_QUERY_SHAPE),
    ("key", GDN_SUPERBLOCK_S512_QUERY_SHAPE),
    ("value", GDN_SUPERBLOCK_S512_VALUE_SHAPE),
    ("g", GDN_SUPERBLOCK_S512_GATE_SHAPE),
    ("beta", GDN_SUPERBLOCK_S512_GATE_SHAPE),
    ("initial_state", GDN_SUPERBLOCK_S512_STATE_SHAPE),
)


@dataclass(frozen=True, slots=True)
class GDNSuperblockBoundaryGradients:
    """FP32 cotangents for the six transformed superblock primals."""

    query: "NDArray[np.float32]"
    key: "NDArray[np.float32]"
    value: "NDArray[np.float32]"
    g: "NDArray[np.float32]"
    beta: "NDArray[np.float32]"
    initial_state: "NDArray[np.float32]"


def _require_f32_c_array(
    array: np.ndarray,
    name: str,
    shape: tuple[int, ...],
) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.shape != shape:
        raise ValueError(f"{name} shape must be exactly {shape}, got {array.shape}")
    if array.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} dtype must be exactly float32")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")


def _require_bf16_c_array(
    array: np.ndarray,
    name: str,
    shape: tuple[int, ...],
) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.shape != shape:
        raise ValueError(f"{name} shape must be exactly {shape}, got {array.shape}")
    if array.dtype != execute_reverse._bfloat16_dtype():
        raise TypeError(f"{name} dtype must be exactly bfloat16")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")


def _validate_exact_inputs(arrays: tuple[np.ndarray, ...]) -> None:
    for array, (name, shape) in zip(arrays[:6], _F32_INPUT_SPECS, strict=True):
        _require_f32_c_array(array, name, shape)
    _require_bf16_c_array(
        arrays[6],
        "output_cotangent",
        GDN_SUPERBLOCK_S512_OUTPUT_SHAPE,
    )
    _require_f32_c_array(
        arrays[7],
        "final_state_cotangent",
        GDN_SUPERBLOCK_S512_STATE_SHAPE,
    )

    observed_bytes = sum(array.nbytes for array in arrays)
    if observed_bytes != GDN_SUPERBLOCK_S512_REVERSE_INPUT_BYTES:
        raise ValueError(
            "exact S512 superblock reverse inputs must total "
            f"{GDN_SUPERBLOCK_S512_REVERSE_INPUT_BYTES} bytes, got "
            f"{observed_bytes}"
        )

    names = (
        *(name for name, _ in _F32_INPUT_SPECS),
        "output_cotangent",
        "final_state_cotangent",
    )
    for left_index, (left_name, left) in enumerate(zip(names, arrays, strict=True)):
        for right_name, right in zip(
            names[left_index + 1 :],
            arrays[left_index + 1 :],
            strict=True,
        ):
            if np.shares_memory(left, right):
                raise ValueError(
                    "exact S512 superblock reverse inputs must use distinct, "
                    f"non-overlapping buffers; {left_name} overlaps {right_name}"
                )


def _validate_generic_inputs(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    initial_state: np.ndarray,
    output_cotangent: np.ndarray,
    final_state_cotangent: np.ndarray,
    chunk_size: int,
) -> tuple[int, int, int, int, int, int, int]:
    arrays = (
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        output_cotangent,
        final_state_cotangent,
    )
    if any(not isinstance(item, np.ndarray) for item in arrays):
        raise TypeError("generic superblock reverse inputs must be NumPy arrays")
    if any(item.dtype != np.dtype(np.float32) for item in arrays):
        raise TypeError("generic superblock reverse inputs must be exactly float32")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive exact int")
    if query.ndim != 4 or key.shape != query.shape:
        raise ValueError("query and key must have one common rank-4 shape")

    batch, tokens, key_heads, key_dimension = query.shape
    if batch != 1:
        raise ValueError("generic superblock reverse currently requires batch one")
    if min(tokens, key_heads, key_dimension) <= 0:
        raise ValueError("generic superblock reverse dimensions must be positive")
    if tokens % chunk_size:
        raise ValueError("token count must be divisible by chunk_size")
    if value.ndim != 4 or value.shape[:2] != (batch, tokens):
        raise ValueError("value must have shape [1,T,Hv,Dv]")
    value_heads, value_dimension = value.shape[2:]
    if min(value_heads, value_dimension) <= 0:
        raise ValueError("generic superblock reverse dimensions must be positive")
    if value_heads % key_heads:
        raise ValueError("value heads must be divisible by key heads")
    if g.shape != (batch, tokens, value_heads) or beta.shape != g.shape:
        raise ValueError("g and beta must have shape [1,T,Hv]")
    state_shape = (batch, value_heads, key_dimension, value_dimension)
    if initial_state.shape != state_shape:
        raise ValueError("initial_state has an incompatible shape")
    if output_cotangent.shape != value.shape:
        raise ValueError("output_cotangent must have the value shape")
    if final_state_cotangent.shape != state_shape:
        raise ValueError("final_state_cotangent must have the state shape")
    return (
        batch,
        tokens,
        key_heads,
        value_heads,
        key_dimension,
        value_dimension,
        tokens // chunk_size,
    )


def _token_to_prepare_chunks(
    key: np.ndarray,
    value: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    *,
    chunks: int,
    chunk_size: int,
    key_heads: int,
    heads_per_key: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key_dimension = key.shape[-1]
    value_dimension = value.shape[-1]
    key_chunks = key.reshape(
        chunks,
        chunk_size,
        key_heads,
        key_dimension,
    ).transpose(0, 2, 1, 3)
    value_chunks = value.reshape(
        chunks,
        chunk_size,
        key_heads,
        heads_per_key,
        value_dimension,
    ).transpose(0, 2, 3, 1, 4)
    g_chunks = g.reshape(
        chunks,
        chunk_size,
        key_heads,
        heads_per_key,
    ).transpose(0, 2, 3, 1)
    beta_chunks = beta.reshape(
        chunks,
        chunk_size,
        key_heads,
        heads_per_key,
    ).transpose(0, 2, 3, 1)
    return key_chunks, value_chunks, g_chunks, beta_chunks


def _prepared_chunks_to_tokens(
    prepared_u: np.ndarray,
    prepared_w: np.ndarray,
    gamma: np.ndarray,
    *,
    tokens: int,
    value_heads: int,
    value_dimension: int,
    key_dimension: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_tokens = prepared_u.transpose(0, 3, 1, 2, 4).reshape(
        1,
        tokens,
        value_heads,
        value_dimension,
    )
    w_tokens = prepared_w.transpose(0, 3, 1, 2, 4).reshape(
        1,
        tokens,
        value_heads,
        key_dimension,
    )
    gamma_tokens = gamma.transpose(0, 3, 1, 2).reshape(
        1,
        tokens,
        value_heads,
    )
    return (
        np.ascontiguousarray(u_tokens, dtype=np.float32),
        np.ascontiguousarray(w_tokens, dtype=np.float32),
        np.ascontiguousarray(gamma_tokens, dtype=np.float32),
    )


def _prepare_gradients_to_tokens(
    gradients: prepare_reverse.GDNPrepareBoundaryGradients,
    *,
    tokens: int,
    key_heads: int,
    value_heads: int,
    key_dimension: int,
    value_dimension: int,
) -> prepare_reverse.GDNPrepareBoundaryGradients:
    return prepare_reverse.GDNPrepareBoundaryGradients(
        key=np.ascontiguousarray(
            gradients.key.transpose(0, 2, 1, 3).reshape(
                1,
                tokens,
                key_heads,
                key_dimension,
            ),
            dtype=np.float32,
        ),
        value=np.ascontiguousarray(
            gradients.value.transpose(0, 3, 1, 2, 4).reshape(
                1,
                tokens,
                value_heads,
                value_dimension,
            ),
            dtype=np.float32,
        ),
        g=np.ascontiguousarray(
            gradients.g.transpose(0, 3, 1, 2).reshape(
                1,
                tokens,
                value_heads,
            ),
            dtype=np.float32,
        ),
        beta=np.ascontiguousarray(
            gradients.beta.transpose(0, 3, 1, 2).reshape(
                1,
                tokens,
                value_heads,
            ),
            dtype=np.float32,
        ),
    )


def _gdn_superblock_reverse_numpy(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    initial_state: np.ndarray,
    output_cotangent: np.ndarray,
    final_state_cotangent: np.ndarray,
    *,
    chunk_size: int,
) -> GDNSuperblockBoundaryGradients:
    """Compose prepare and execute transposes at reduced batch-one geometry."""
    (
        _batch,
        tokens,
        key_heads,
        value_heads,
        key_dimension,
        value_dimension,
        chunks,
    ) = _validate_generic_inputs(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        output_cotangent,
        final_state_cotangent,
        chunk_size,
    )
    heads_per_key = value_heads // key_heads
    key_chunks, value_chunks, g_chunks, beta_chunks = _token_to_prepare_chunks(
        key,
        value,
        g,
        beta,
        chunks=chunks,
        chunk_size=chunk_size,
        key_heads=key_heads,
        heads_per_key=heads_per_key,
    )
    prepared_u, prepared_w, gamma = prepare_forward._dense_prepare_chunks_numpy(
        key_chunks,
        value_chunks,
        g_chunks,
        beta_chunks,
    )
    u_tokens, w_tokens, gamma_tokens = _prepared_chunks_to_tokens(
        prepared_u,
        prepared_w,
        gamma,
        tokens=tokens,
        value_heads=value_heads,
        value_dimension=value_dimension,
        key_dimension=key_dimension,
    )
    execute_gradients = execute_reverse._gdn_execute_reverse_chunks_numpy(
        query,
        key,
        u_tokens,
        w_tokens,
        gamma_tokens,
        initial_state,
        output_cotangent,
        final_state_cotangent,
        chunk_size=chunk_size,
    )
    u_bar_chunks = execute_gradients.prepared_u.reshape(
        chunks,
        chunk_size,
        key_heads,
        heads_per_key,
        value_dimension,
    ).transpose(0, 2, 3, 1, 4)
    w_bar_chunks = execute_gradients.prepared_w.reshape(
        chunks,
        chunk_size,
        key_heads,
        heads_per_key,
        key_dimension,
    ).transpose(0, 2, 3, 1, 4)
    gamma_bar_chunks = execute_gradients.gamma.reshape(
        chunks,
        chunk_size,
        key_heads,
        heads_per_key,
    ).transpose(0, 2, 3, 1)
    prepare_gradients = prepare_reverse._dense_prepare_reverse_chunks_numpy(
        key_chunks,
        value_chunks,
        g_chunks,
        beta_chunks,
        u_bar_chunks,
        w_bar_chunks,
        gamma_bar_chunks,
    )
    prepare_tokens = _prepare_gradients_to_tokens(
        prepare_gradients,
        tokens=tokens,
        key_heads=key_heads,
        value_heads=value_heads,
        key_dimension=key_dimension,
        value_dimension=value_dimension,
    )
    combined_key = np.add(
        execute_gradients.key,
        prepare_tokens.key,
        dtype=np.float32,
    )
    return GDNSuperblockBoundaryGradients(
        query=np.ascontiguousarray(execute_gradients.query, dtype=np.float32),
        key=np.ascontiguousarray(combined_key, dtype=np.float32),
        value=np.ascontiguousarray(prepare_tokens.value, dtype=np.float32),
        g=np.ascontiguousarray(prepare_tokens.g, dtype=np.float32),
        beta=np.ascontiguousarray(prepare_tokens.beta, dtype=np.float32),
        initial_state=np.ascontiguousarray(
            execute_gradients.initial_state,
            dtype=np.float32,
        ),
    )


def gdn_superblock_s512_reverse_numpy(
    query: "NDArray[np.float32]",
    key: "NDArray[np.float32]",
    value: "NDArray[np.float32]",
    g: "NDArray[np.float32]",
    beta: "NDArray[np.float32]",
    initial_state: "NDArray[np.float32]",
    output_cotangent: np.ndarray,
    final_state_cotangent: "NDArray[np.float32]",
) -> GDNSuperblockBoundaryGradients:
    """Return exact transformed-boundary FP32 gradients for one S512 block."""
    arrays = (
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        output_cotangent,
        final_state_cotangent,
    )
    _validate_exact_inputs(arrays)

    prepared_u, prepared_w, gamma = prepare_forward.gdn_prepare_s512_numpy(
        key,
        value,
        g,
        beta,
    )
    execute_gradients = execute_reverse.gdn_execute_s512_reverse_numpy(
        query,
        key,
        prepared_u,
        prepared_w,
        gamma,
        initial_state,
        output_cotangent,
        final_state_cotangent,
    )
    prepare_gradients = prepare_reverse.gdn_prepare_s512_reverse_numpy(
        key,
        value,
        g,
        beta,
        execute_gradients.prepared_u,
        execute_gradients.prepared_w,
        execute_gradients.gamma,
    )
    combined_key = np.add(
        execute_gradients.key,
        prepare_gradients.key,
        dtype=np.float32,
    )
    gradients = GDNSuperblockBoundaryGradients(
        query=np.array(execute_gradients.query, dtype=np.float32, order="C", copy=True),
        key=np.array(combined_key, dtype=np.float32, order="C", copy=True),
        value=np.array(prepare_gradients.value, dtype=np.float32, order="C", copy=True),
        g=np.array(prepare_gradients.g, dtype=np.float32, order="C", copy=True),
        beta=np.array(prepare_gradients.beta, dtype=np.float32, order="C", copy=True),
        initial_state=np.array(
            execute_gradients.initial_state,
            dtype=np.float32,
            order="C",
            copy=True,
        ),
    )
    observed_gradient_bytes = sum(
        array.nbytes
        for array in (
            gradients.query,
            gradients.key,
            gradients.value,
            gradients.g,
            gradients.beta,
            gradients.initial_state,
        )
    )
    if observed_gradient_bytes != GDN_SUPERBLOCK_S512_GRADIENT_BYTES:
        raise RuntimeError(
            "exact S512 superblock gradients must total "
            f"{GDN_SUPERBLOCK_S512_GRADIENT_BYTES} bytes, got "
            f"{observed_gradient_bytes}"
        )
    return gradients
