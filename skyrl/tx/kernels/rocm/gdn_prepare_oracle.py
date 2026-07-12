"""Independent NumPy oracle for exact S512 GDN WY preparation.

The oracle intentionally uses batched dense ``numpy.linalg.solve`` rather than
the HIP kernel's tiled forward substitution.  It has no JAX dependency and is
not a runtime fallback.  Its only purpose is semantic gating of the exact
prepare-only operation boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_KEY_SHAPE = (1, 512, 16, 128)
_VALUE_SHAPE = (1, 512, 32, 128)
_GATE_SHAPE = (1, 512, 32)
_MASK_SHAPE = (1, 512)
_CHUNKS = 8
_CHUNK = 64
_KEY_HEADS = 16
_HEADS_PER_KEY = 2
_HEAD_DIMENSION = 128


def _require_f32(array: np.ndarray, name: str, shape: tuple[int, ...]) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.shape != shape:
        raise ValueError(f"{name} shape must be exactly {shape}, got {array.shape}")
    if array.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} dtype must be exactly float32")


def transform_gdn_prepare_s512_inputs_numpy(
    key: "NDArray[np.float32]",
    value: "NDArray[np.float32]",
    g: "NDArray[np.float32]",
    beta: "NDArray[np.float32]",
    attention_mask: "NDArray[np.bool_]",
) -> tuple[
    "NDArray[np.float32]",
    "NDArray[np.float32]",
    "NDArray[np.float32]",
    "NDArray[np.float32]",
]:
    """Apply the exact pre-FFI mask transformation to fresh FP32 arrays."""
    _require_f32(key, "key", _KEY_SHAPE)
    _require_f32(value, "value", _VALUE_SHAPE)
    _require_f32(g, "g", _GATE_SHAPE)
    _require_f32(beta, "beta", _GATE_SHAPE)
    if not isinstance(attention_mask, np.ndarray):
        raise TypeError("attention_mask must be a NumPy array")
    if attention_mask.shape != _MASK_SHAPE:
        raise ValueError(
            f"attention_mask shape must be exactly {_MASK_SHAPE}, "
            f"got {attention_mask.shape}"
        )
    if attention_mask.dtype != np.dtype(np.bool_):
        raise TypeError("attention_mask dtype must be exactly bool")

    token_mask = attention_mask.astype(np.float32, copy=False)
    return (
        np.asarray(key * token_mask[:, :, None, None], dtype=np.float32),
        np.asarray(value * token_mask[:, :, None, None], dtype=np.float32),
        np.asarray(g * token_mask[:, :, None], dtype=np.float32),
        np.asarray(beta * token_mask[:, :, None], dtype=np.float32),
    )


def _dense_prepare_chunks_numpy(
    key_chunks: "NDArray[np.float32]",
    value_chunks: "NDArray[np.float32]",
    g_chunks: "NDArray[np.float32]",
    beta_chunks: "NDArray[np.float32]",
) -> tuple[
    "NDArray[np.float32]",
    "NDArray[np.float32]",
    "NDArray[np.float32]",
]:
    """Dense-solve equations on generic ``[C,H,(P),L,D]`` chunk arrays."""
    if any(
        not isinstance(item, np.ndarray)
        for item in (key_chunks, value_chunks, g_chunks, beta_chunks)
    ):
        raise TypeError("dense oracle chunk inputs must be NumPy arrays")
    if any(
        item.dtype != np.dtype(np.float32)
        for item in (key_chunks, value_chunks, g_chunks, beta_chunks)
    ):
        raise TypeError("dense oracle chunk inputs must be exactly float32")
    if key_chunks.ndim != 4 or value_chunks.ndim != 5:
        raise ValueError("dense oracle requires rank-4 K and rank-5 V chunks")
    chunks, key_heads, chunk, key_dimension = key_chunks.shape
    if value_chunks.shape[:2] != (chunks, key_heads):
        raise ValueError("dense oracle K/V chunk and key-head axes must match")
    pairs = value_chunks.shape[2]
    if value_chunks.shape[3] != chunk:
        raise ValueError("dense oracle K/V chunk-token axes must match")
    if g_chunks.shape != (chunks, key_heads, pairs, chunk):
        raise ValueError("dense oracle g chunk shape does not match V")
    if beta_chunks.shape != g_chunks.shape:
        raise ValueError("dense oracle beta chunk shape must match g")
    if min(chunks, key_heads, pairs, chunk, key_dimension, value_chunks.shape[-1]) <= 0:
        raise ValueError("dense oracle dimensions must be positive")

    # [C,H,I,D] x [C,H,J,D] -> [C,H,I,J], shared by each pair.
    key_gram = np.einsum(
        "chid,chjd->chij",
        key_chunks,
        key_chunks,
        dtype=np.float32,
        optimize=False,
    )
    prefix = np.cumsum(g_chunks, axis=-1, dtype=np.float32)
    gamma = np.exp(prefix).astype(np.float32, copy=False)
    decay = np.exp(prefix[..., :, None] - prefix[..., None, :]).astype(
        np.float32,
        copy=False,
    )
    decay = np.tril(decay)
    strict_lower = np.tril(
        beta_chunks[..., :, None] * key_gram[:, :, None, :, :] * decay,
        k=-1,
    ).astype(np.float32, copy=False)

    rhs_u = beta_chunks[..., None] * value_chunks
    rhs_w = beta_chunks[..., None] * gamma[..., None] * key_chunks[:, :, None, :, :]
    rhs = np.concatenate((rhs_u, rhs_w), axis=-1).astype(np.float32, copy=False)
    unit_lower = strict_lower + np.eye(chunk, dtype=np.float32)
    solution = np.linalg.solve(unit_lower, rhs).astype(np.float32, copy=False)
    value_dimension = value_chunks.shape[-1]
    return (
        solution[..., :value_dimension],
        solution[..., value_dimension : value_dimension + key_dimension],
        gamma,
    )


def gdn_prepare_s512_numpy(
    key: "NDArray[np.float32]",
    value: "NDArray[np.float32]",
    g: "NDArray[np.float32]",
    beta: "NDArray[np.float32]",
    *,
    attention_mask: "NDArray[np.bool_] | None" = None,
) -> tuple[
    "NDArray[np.float32]",
    "NDArray[np.float32]",
    "NDArray[np.float32]",
]:
    """Return exact-shape ``(U, W, G)`` using independent dense solves.

    ``attention_mask`` is an oracle convenience, not an FFI argument.  When
    supplied, it applies the model-side transformed-input contract before the
    equations: masked K/V/g/beta entries become zero.  ``G`` remains the
    cumulative decay of the transformed g and therefore need not be zero at a
    masked token, while the corresponding U and W rows are exactly zero.
    """
    _require_f32(key, "key", _KEY_SHAPE)
    _require_f32(value, "value", _VALUE_SHAPE)
    _require_f32(g, "g", _GATE_SHAPE)
    _require_f32(beta, "beta", _GATE_SHAPE)
    if attention_mask is not None:
        key, value, g, beta = transform_gdn_prepare_s512_inputs_numpy(
            key,
            value,
            g,
            beta,
            attention_mask,
        )

    key_chunks = key.reshape(_CHUNKS, _CHUNK, _KEY_HEADS, _HEAD_DIMENSION).transpose(
        0, 2, 1, 3
    )
    value_chunks = value.reshape(
        _CHUNKS,
        _CHUNK,
        _KEY_HEADS,
        _HEADS_PER_KEY,
        _HEAD_DIMENSION,
    ).transpose(0, 2, 3, 1, 4)
    g_chunks = g.reshape(_CHUNKS, _CHUNK, _KEY_HEADS, _HEADS_PER_KEY).transpose(
        0, 2, 3, 1
    )
    beta_chunks = beta.reshape(_CHUNKS, _CHUNK, _KEY_HEADS, _HEADS_PER_KEY).transpose(
        0, 2, 3, 1
    )

    prepared_u, prepared_w, gamma = _dense_prepare_chunks_numpy(
        key_chunks,
        value_chunks,
        g_chunks,
        beta_chunks,
    )
    prepared_u = prepared_u.transpose(0, 3, 1, 2, 4)
    prepared_w = prepared_w.transpose(0, 3, 1, 2, 4)
    gamma_tokens = gamma.transpose(0, 3, 1, 2)
    return (
        np.ascontiguousarray(prepared_u.reshape(_VALUE_SHAPE)),
        np.ascontiguousarray(prepared_w.reshape(_VALUE_SHAPE)),
        np.ascontiguousarray(gamma_tokens.reshape(_GATE_SHAPE)),
    )
