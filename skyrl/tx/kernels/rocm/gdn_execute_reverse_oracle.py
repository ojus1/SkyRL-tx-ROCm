"""Import-light NumPy reverse oracle for the exact S512 GDN execute stage.

This module is the transpose of the FP32 algebra in
``gdn_execute_oracle.gdn_execute_s512_numpy``.  It is a CPU-only semantic gate:
it does not import JAX, load native code, dispatch a device operation, or
differentiate the preceding WY preparation or Q/K normalization stages.

For one value head and one 64-token chunk, with incoming state ``S``, define

``D[i,j] = gamma[i] / gamma[j]`` for ``j <= i`` (zero otherwise),
``C = U - W @ S``, ``A = (Q @ K.T) * D``, and
``P = K * D[-1,:,None]``.  The exact execute equations are

``O = (Q * gamma[:,None]) @ S + A @ C`` and
``S_next = gamma[-1] * S + P.T @ C``.

The reverse applies the matrix transposes of those equations from chunk seven
through chunk zero.  Value head ``hv`` reads, and contributes Q/K cotangents
to, key head ``hv // 2``.  Diagonal entries of ``D`` are the constant one, so
their analytically cancelling numerator/denominator derivatives are omitted.

The qualified forward returns BF16 output and FP32 state, so this exact public
reverse boundary accepts a BF16 output cotangent and an FP32 final-state
cotangent.  The BF16 operand is widened internally before any reverse
arithmetic.  Every returned boundary gradient is FP32.

No additional internal forward residual is required.  The reverse must have
the six primal boundary inputs, whose exact footprint is 27,328,512 bytes
(26.0625 MiB), but an enclosing checkpoint may rematerialize rather than save
them.  Those primals are sufficient to replay the eight chunk-start states.
Retaining the seven internal FP32 states instead would cost exactly 14 MiB at
S512; retaining all eight chunk starts including the already-available initial
state would cost 16 MiB.  This oracle replays once and holds those starts
transiently.  A production custom VJP can make the same checkpoint/replay
choice without a persistent internal-state residual.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


GDN_EXECUTE_S512_QUERY_SHAPE = (1, 512, 16, 128)
GDN_EXECUTE_S512_PREPARED_SHAPE = (1, 512, 32, 128)
GDN_EXECUTE_S512_GAMMA_SHAPE = (1, 512, 32)
GDN_EXECUTE_S512_STATE_SHAPE = (1, 32, 128, 128)
GDN_EXECUTE_S512_OUTPUT_SHAPE = GDN_EXECUTE_S512_PREPARED_SHAPE

GDN_EXECUTE_S512_OUTPUT_COTANGENT_BYTES = 4_194_304
GDN_EXECUTE_S512_STATE_COTANGENT_BYTES = 2_097_152
GDN_EXECUTE_S512_COTANGENT_INPUT_BYTES = 6_291_456
GDN_EXECUTE_S512_PRIMAL_BYTES = 27_328_512
GDN_EXECUTE_S512_GRADIENT_BYTES = 27_328_512
GDN_EXECUTE_S512_INTERNAL_STATE_COUNT = 7
GDN_EXECUTE_S512_INTERNAL_STATE_RESIDUAL_BYTES = 14_680_064
GDN_EXECUTE_S512_ALL_CHUNK_START_BYTES = 16_777_216

_CHUNKS = 8
_CHUNK_SIZE = 64
_KEY_HEADS = 16
_VALUE_HEADS = 32
_HEAD_DIMENSION = 128

_PRIMAL_SPECS = (
    ("query", GDN_EXECUTE_S512_QUERY_SHAPE),
    ("key", GDN_EXECUTE_S512_QUERY_SHAPE),
    ("prepared_u", GDN_EXECUTE_S512_PREPARED_SHAPE),
    ("prepared_w", GDN_EXECUTE_S512_PREPARED_SHAPE),
    ("gamma", GDN_EXECUTE_S512_GAMMA_SHAPE),
    ("initial_state", GDN_EXECUTE_S512_STATE_SHAPE),
)
_COTANGENT_SPECS = (
    ("output_cotangent", GDN_EXECUTE_S512_OUTPUT_SHAPE),
    ("final_state_cotangent", GDN_EXECUTE_S512_STATE_SHAPE),
)


@dataclass(frozen=True, slots=True)
class GDNExecuteBoundaryGradients:
    """FP32 cotangents for all six exact execute boundary inputs."""

    query: "NDArray[np.float32]"
    key: "NDArray[np.float32]"
    prepared_u: "NDArray[np.float32]"
    prepared_w: "NDArray[np.float32]"
    gamma: "NDArray[np.float32]"
    initial_state: "NDArray[np.float32]"


def _require_f32_c_array(array: np.ndarray, name: str, shape: tuple[int, ...]) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.shape != shape:
        raise ValueError(f"{name} shape must be exactly {shape}, got {array.shape}")
    if array.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} dtype must be exactly float32")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")


def _bfloat16_dtype() -> np.dtype:
    try:
        module = importlib.import_module("ml_dtypes")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "the exact BF16 output cotangent requires ml_dtypes"
        ) from error
    dtype = getattr(module, "bfloat16", None)
    if dtype is None:
        raise RuntimeError("installed ml_dtypes does not expose bfloat16")
    return np.dtype(dtype)


def _require_bf16_c_array(
    array: np.ndarray, name: str, shape: tuple[int, ...], bfloat16: np.dtype
) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.shape != shape:
        raise ValueError(f"{name} shape must be exactly {shape}, got {array.shape}")
    if array.dtype != bfloat16:
        raise TypeError(f"{name} dtype must be exactly bfloat16")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")


def _validate_exact_inputs(arrays: tuple[np.ndarray, ...]) -> None:
    specs = (*_PRIMAL_SPECS, *_COTANGENT_SPECS)
    for array, (name, shape) in zip(arrays[:6], _PRIMAL_SPECS, strict=True):
        _require_f32_c_array(array, name, shape)
    _require_bf16_c_array(
        arrays[6],
        _COTANGENT_SPECS[0][0],
        _COTANGENT_SPECS[0][1],
        _bfloat16_dtype(),
    )
    _require_f32_c_array(arrays[7], *_COTANGENT_SPECS[1])

    for left_index, (left_name, _) in enumerate(specs):
        for right_index in range(left_index + 1, len(arrays)):
            right_name = specs[right_index][0]
            if np.shares_memory(arrays[left_index], arrays[right_index]):
                raise ValueError(
                    "exact S512 reverse inputs must use distinct, non-overlapping "
                    f"buffers; {left_name} overlaps {right_name}"
                )

    gamma = arrays[4]
    if not np.all(np.isfinite(gamma)) or not np.all(gamma > np.float32(0.0)):
        raise ValueError("gamma must contain only finite, strictly positive values")

    observed_primal_bytes = sum(array.nbytes for array in arrays[:6])
    if observed_primal_bytes != GDN_EXECUTE_S512_PRIMAL_BYTES:
        raise ValueError(
            "exact S512 reverse primals must total "
            f"{GDN_EXECUTE_S512_PRIMAL_BYTES} bytes, got {observed_primal_bytes}"
        )
    observed_cotangent_bytes = arrays[6].nbytes + arrays[7].nbytes
    if observed_cotangent_bytes != GDN_EXECUTE_S512_COTANGENT_INPUT_BYTES:
        raise ValueError(
            "exact S512 reverse cotangents must total "
            f"{GDN_EXECUTE_S512_COTANGENT_INPUT_BYTES} bytes, got "
            f"{observed_cotangent_bytes}"
        )


def _chunk_views(
    query: np.ndarray,
    key: np.ndarray,
    prepared_u: np.ndarray,
    prepared_w: np.ndarray,
    gamma: np.ndarray,
    output_cotangent: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, ...]:
    batch, tokens, key_heads, key_dimension = query.shape
    value_heads = prepared_u.shape[2]
    value_dimension = prepared_u.shape[3]
    chunks = tokens // chunk_size
    heads_per_key = value_heads // key_heads
    head_map = np.arange(value_heads, dtype=np.intp) // heads_per_key

    query_chunks = query.reshape(
        batch, chunks, chunk_size, key_heads, key_dimension
    ).transpose(0, 1, 3, 2, 4)
    key_chunks = key.reshape(
        batch, chunks, chunk_size, key_heads, key_dimension
    ).transpose(0, 1, 3, 2, 4)
    u_chunks = prepared_u.reshape(
        batch, chunks, chunk_size, value_heads, value_dimension
    ).transpose(0, 1, 3, 2, 4)
    w_chunks = prepared_w.reshape(
        batch, chunks, chunk_size, value_heads, key_dimension
    ).transpose(0, 1, 3, 2, 4)
    gamma_chunks = gamma.reshape(batch, chunks, chunk_size, value_heads).transpose(
        0, 1, 3, 2
    )
    output_bar_chunks = output_cotangent.reshape(
        batch, chunks, chunk_size, value_heads, value_dimension
    ).transpose(0, 1, 3, 2, 4)
    return (
        np.take(query_chunks, head_map, axis=2),
        np.take(key_chunks, head_map, axis=2),
        u_chunks,
        w_chunks,
        gamma_chunks,
        output_bar_chunks,
    )


def _chunk_decay_and_corrected(
    key: np.ndarray,
    prepared_u: np.ndarray,
    prepared_w: np.ndarray,
    gamma: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return state-dependent values shared by replay and reverse."""
    decay = np.tril(
        np.asarray(
            gamma[..., :, None] / gamma[..., None, :],
            dtype=np.float32,
        )
    )
    corrected = np.asarray(prepared_u - np.matmul(prepared_w, state), dtype=np.float32)
    reverse_key = np.asarray(key * decay[..., -1, :, None], dtype=np.float32)
    return decay, corrected, reverse_key


def _replay_next_state(
    key: np.ndarray,
    prepared_u: np.ndarray,
    prepared_w: np.ndarray,
    gamma: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """Replay only the query-independent chunk state transition."""
    _, corrected, reverse_key = _chunk_decay_and_corrected(
        key, prepared_u, prepared_w, gamma, state
    )
    return np.asarray(
        gamma[..., -1, None, None] * state
        + np.matmul(np.swapaxes(reverse_key, -1, -2), corrected),
        dtype=np.float32,
    )


def _chunk_reverse_values(
    query: np.ndarray,
    key: np.ndarray,
    prepared_u: np.ndarray,
    prepared_w: np.ndarray,
    gamma: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    decay, corrected, reverse_key = _chunk_decay_and_corrected(
        key, prepared_u, prepared_w, gamma, state
    )
    query_key = np.matmul(query, np.swapaxes(key, -1, -2)).astype(
        np.float32, copy=False
    )
    return decay, corrected, query_key, reverse_key


def _gdn_execute_reverse_chunks_numpy(
    query: np.ndarray,
    key: np.ndarray,
    prepared_u: np.ndarray,
    prepared_w: np.ndarray,
    gamma: np.ndarray,
    initial_state: np.ndarray,
    output_cotangent: np.ndarray,
    final_state_cotangent: np.ndarray,
    *,
    chunk_size: int,
) -> GDNExecuteBoundaryGradients:
    """Generic grouped-head reverse used for reduced CPU-only validation."""
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive exact int")
    if query.ndim != 4 or key.shape != query.shape:
        raise ValueError("query and key must have one common rank-4 shape")
    batch, tokens, key_heads, key_dimension = query.shape
    if tokens % chunk_size:
        raise ValueError("token count must be divisible by chunk_size")
    if prepared_u.ndim != 4 or prepared_u.shape[:2] != (batch, tokens):
        raise ValueError("prepared_u must have shape [B,T,Hv,Dv]")
    value_heads, value_dimension = prepared_u.shape[2:]
    if value_heads % key_heads:
        raise ValueError("value heads must be divisible by key heads")
    if prepared_w.shape != (batch, tokens, value_heads, key_dimension):
        raise ValueError("prepared_w must have shape [B,T,Hv,Dk]")
    if gamma.shape != (batch, tokens, value_heads):
        raise ValueError("gamma must have shape [B,T,Hv]")
    state_shape = (batch, value_heads, key_dimension, value_dimension)
    if initial_state.shape != state_shape:
        raise ValueError("initial_state has an incompatible shape")
    if output_cotangent.shape != prepared_u.shape:
        raise ValueError("output_cotangent must have the prepared_u shape")
    if final_state_cotangent.shape != state_shape:
        raise ValueError("final_state_cotangent must have the state shape")
    arrays = (
        query,
        key,
        prepared_u,
        prepared_w,
        gamma,
        initial_state,
        output_cotangent,
        final_state_cotangent,
    )
    if any(item.dtype != np.dtype(np.float32) for item in arrays):
        raise TypeError("generic reverse inputs must all be exactly float32")

    (
        query_chunks,
        key_chunks,
        u_chunks,
        w_chunks,
        gamma_chunks,
        output_bar_chunks,
    ) = _chunk_views(
        query,
        key,
        prepared_u,
        prepared_w,
        gamma,
        output_cotangent,
        chunk_size=chunk_size,
    )
    chunks = tokens // chunk_size

    # Replay only the state recurrence.  S0 is already a primal input, so the
    # seven internal entries are the only optional persistent residuals at
    # exact S512 geometry.
    state_starts = [initial_state]
    state = initial_state
    for chunk_index in range(chunks - 1):
        state = _replay_next_state(
            key_chunks[:, chunk_index],
            u_chunks[:, chunk_index],
            w_chunks[:, chunk_index],
            gamma_chunks[:, chunk_index],
            state,
        )
        state_starts.append(state)

    query_bar_chunks = np.zeros_like(query_chunks)
    key_bar_chunks = np.zeros_like(key_chunks)
    u_bar_chunks = np.empty_like(u_chunks)
    w_bar_chunks = np.empty_like(w_chunks)
    gamma_bar_chunks = np.empty_like(gamma_chunks)
    state_bar = final_state_cotangent.copy()
    strict_lower = np.tril(np.ones((chunk_size, chunk_size), dtype=np.bool_), k=-1)

    for chunk_index in range(chunks - 1, -1, -1):
        query_chunk = query_chunks[:, chunk_index]
        key_chunk = key_chunks[:, chunk_index]
        u_chunk = u_chunks[:, chunk_index]
        w_chunk = w_chunks[:, chunk_index]
        gamma_chunk = gamma_chunks[:, chunk_index]
        output_bar = output_bar_chunks[:, chunk_index]
        state = state_starts[chunk_index]
        decay, corrected, query_key, reverse_key = _chunk_reverse_values(
            query_chunk,
            key_chunk,
            u_chunk,
            w_chunk,
            gamma_chunk,
            state,
        )
        attention = np.asarray(query_key * decay, dtype=np.float32)
        scaled_query = np.asarray(
            query_chunk * gamma_chunk[..., :, None], dtype=np.float32
        )
        reverse_decay = decay[..., -1, :]

        attention_bar = np.matmul(output_bar, np.swapaxes(corrected, -1, -2)).astype(
            np.float32, copy=False
        )
        corrected_bar = np.matmul(np.swapaxes(attention, -1, -2), output_bar).astype(
            np.float32, copy=False
        )
        scaled_query_bar = np.matmul(output_bar, np.swapaxes(state, -1, -2)).astype(
            np.float32, copy=False
        )
        incoming_state_bar = np.matmul(
            np.swapaxes(scaled_query, -1, -2), output_bar
        ).astype(np.float32, copy=False)

        incoming_state_bar += gamma_chunk[..., -1, None, None] * state_bar
        gamma_bar = np.zeros_like(gamma_chunk)
        gamma_bar[..., -1] = np.sum(state_bar * state, axis=(-2, -1), dtype=np.float32)
        reverse_key_bar = np.matmul(corrected, np.swapaxes(state_bar, -1, -2)).astype(
            np.float32, copy=False
        )
        corrected_bar += np.matmul(reverse_key, state_bar).astype(
            np.float32, copy=False
        )

        u_bar_chunks[:, chunk_index] = corrected_bar
        w_bar_chunks[:, chunk_index] = -np.matmul(
            corrected_bar, np.swapaxes(state, -1, -2)
        ).astype(np.float32, copy=False)
        incoming_state_bar -= np.matmul(
            np.swapaxes(w_chunk, -1, -2), corrected_bar
        ).astype(np.float32, copy=False)

        query_key_bar = np.asarray(attention_bar * decay, dtype=np.float32)
        decay_bar = np.asarray(attention_bar * query_key, dtype=np.float32)
        query_bar = np.matmul(query_key_bar, key_chunk).astype(np.float32, copy=False)
        key_bar = np.matmul(np.swapaxes(query_key_bar, -1, -2), query_chunk).astype(
            np.float32, copy=False
        )

        query_bar += scaled_query_bar * gamma_chunk[..., :, None]
        gamma_bar += np.sum(scaled_query_bar * query_chunk, axis=-1, dtype=np.float32)

        key_bar += reverse_key_bar * reverse_decay[..., :, None]
        decay_bar[..., -1, :] += np.sum(
            reverse_key_bar * key_chunk, axis=-1, dtype=np.float32
        )

        # D[i,j] = gamma[i] / gamma[j] below the diagonal.  D[i,i] is
        # identically one, so excluding it avoids relying on floating-point
        # cancellation of two mathematically equal terms.
        decay_bar = np.where(strict_lower, decay_bar, np.float32(0.0))
        gamma_bar += np.sum(
            decay_bar / gamma_chunk[..., None, :],
            axis=-1,
            dtype=np.float32,
        )
        gamma_bar -= np.sum(
            decay_bar * decay / gamma_chunk[..., None, :],
            axis=-2,
            dtype=np.float32,
        )

        query_bar_chunks[:, chunk_index] = query_bar
        key_bar_chunks[:, chunk_index] = key_bar
        gamma_bar_chunks[:, chunk_index] = gamma_bar
        state_bar = incoming_state_bar

    heads_per_key = value_heads // key_heads
    query_bar_chunks = query_bar_chunks.reshape(
        batch,
        chunks,
        key_heads,
        heads_per_key,
        chunk_size,
        key_dimension,
    ).sum(axis=3, dtype=np.float32)
    key_bar_chunks = key_bar_chunks.reshape(
        batch,
        chunks,
        key_heads,
        heads_per_key,
        chunk_size,
        key_dimension,
    ).sum(axis=3, dtype=np.float32)

    query_bar = query_bar_chunks.transpose(0, 1, 3, 2, 4).reshape(query.shape)
    key_bar = key_bar_chunks.transpose(0, 1, 3, 2, 4).reshape(key.shape)
    u_bar = u_bar_chunks.transpose(0, 1, 3, 2, 4).reshape(prepared_u.shape)
    w_bar = w_bar_chunks.transpose(0, 1, 3, 2, 4).reshape(prepared_w.shape)
    gamma_bar = gamma_bar_chunks.transpose(0, 1, 3, 2).reshape(gamma.shape)
    return GDNExecuteBoundaryGradients(
        query=np.ascontiguousarray(query_bar, dtype=np.float32),
        key=np.ascontiguousarray(key_bar, dtype=np.float32),
        prepared_u=np.ascontiguousarray(u_bar, dtype=np.float32),
        prepared_w=np.ascontiguousarray(w_bar, dtype=np.float32),
        gamma=np.ascontiguousarray(gamma_bar, dtype=np.float32),
        initial_state=np.ascontiguousarray(state_bar, dtype=np.float32),
    )


def gdn_execute_s512_reverse_numpy(
    query: "NDArray[np.float32]",
    key: "NDArray[np.float32]",
    prepared_u: "NDArray[np.float32]",
    prepared_w: "NDArray[np.float32]",
    gamma: "NDArray[np.float32]",
    initial_state: "NDArray[np.float32]",
    output_cotangent: np.ndarray,
    final_state_cotangent: "NDArray[np.float32]",
) -> GDNExecuteBoundaryGradients:
    """Return the exact FP32 transpose for one S512 execute boundary.

    Inputs are immutable, pairwise-disjoint, and C-contiguous.  The six
    primals and final-state cotangent are FP32; the output cotangent is BF16.
    Gamma must be finite and strictly positive, as produced by WY preparation.
    The result owns six fresh C-contiguous FP32 arrays in the same order and
    shapes as the primal execute inputs.
    """
    arrays = (
        query,
        key,
        prepared_u,
        prepared_w,
        gamma,
        initial_state,
        output_cotangent,
        final_state_cotangent,
    )
    _validate_exact_inputs(arrays)
    gradients = _gdn_execute_reverse_chunks_numpy(
        *arrays[:6],
        np.ascontiguousarray(output_cotangent.astype(np.float32)),
        final_state_cotangent,
        chunk_size=_CHUNK_SIZE,
    )
    observed_gradient_bytes = sum(
        array.nbytes
        for array in (
            gradients.query,
            gradients.key,
            gradients.prepared_u,
            gradients.prepared_w,
            gradients.gamma,
            gradients.initial_state,
        )
    )
    if observed_gradient_bytes != GDN_EXECUTE_S512_GRADIENT_BYTES:
        raise RuntimeError(
            "exact S512 execute gradients must total "
            f"{GDN_EXECUTE_S512_GRADIENT_BYTES} bytes, got "
            f"{observed_gradient_bytes}"
        )
    return gradients
