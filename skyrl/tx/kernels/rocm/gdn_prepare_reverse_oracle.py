"""Import-light NumPy transpose for exact S512 GDN WY preparation.

This module differentiates the algebra in :mod:`gdn_prepare_oracle` without
importing JAX, loading native code, or dispatching a device operation.  For one
chunk and value head, preparation solves

``A X = R``, ``A = I + L``, and ``X = [U | W]``.

The reverse first solves ``A.T R_bar = X_bar`` and then forms
``L_bar = strictLower(-R_bar @ X.T)``.  It propagates through the right-hand
side, the shared key Gram matrix, the causal decay ratios, and
``gamma = exp(cumsum(g))``.  Its cumsum transpose is a reverse cumulative sum.
Contributions from both value heads are accumulated into the shared key-head
cotangent in FP32.

The optional attention mask mirrors the host oracle's pre-FFI convenience: it
multiplies K/V/g/beta before preparation and multiplies their cotangents on the
way out.  It is not part of the future native FFI ABI.  All seven exact S512
boundary arrays are FP32, C-contiguous, pairwise disjoint, and immutable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


GDN_PREPARE_S512_KEY_SHAPE = (1, 512, 16, 128)
GDN_PREPARE_S512_VALUE_SHAPE = (1, 512, 32, 128)
GDN_PREPARE_S512_GATE_SHAPE = (1, 512, 32)
GDN_PREPARE_S512_MASK_SHAPE = (1, 512)

GDN_PREPARE_S512_PRIMAL_BYTES = 12_713_984
GDN_PREPARE_S512_COTANGENT_BYTES = 16_842_752
GDN_PREPARE_S512_REVERSE_INPUT_BYTES = 29_556_736
GDN_PREPARE_S512_GRADIENT_BYTES = 12_713_984

_CHUNKS = 8
_CHUNK = 64
_KEY_HEADS = 16
_HEADS_PER_KEY = 2
_HEAD_DIMENSION = 128

_EXACT_INPUT_SPECS = (
    ("key", GDN_PREPARE_S512_KEY_SHAPE),
    ("value", GDN_PREPARE_S512_VALUE_SHAPE),
    ("g", GDN_PREPARE_S512_GATE_SHAPE),
    ("beta", GDN_PREPARE_S512_GATE_SHAPE),
    ("prepared_u_cotangent", GDN_PREPARE_S512_VALUE_SHAPE),
    ("prepared_w_cotangent", GDN_PREPARE_S512_VALUE_SHAPE),
    ("gamma_cotangent", GDN_PREPARE_S512_GATE_SHAPE),
)


@dataclass(frozen=True, slots=True)
class GDNPrepareBoundaryGradients:
    """FP32 cotangents for K, V, g, and beta at the prepare boundary."""

    key: "NDArray[np.float32]"
    value: "NDArray[np.float32]"
    g: "NDArray[np.float32]"
    beta: "NDArray[np.float32]"


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


def _validate_exact_inputs(
    arrays: tuple[np.ndarray, ...],
    attention_mask: np.ndarray | None,
) -> None:
    for array, (name, shape) in zip(arrays, _EXACT_INPUT_SPECS, strict=True):
        _require_f32_c_array(array, name, shape)

    observed_bytes = sum(array.nbytes for array in arrays)
    if observed_bytes != GDN_PREPARE_S512_REVERSE_INPUT_BYTES:
        raise ValueError(
            "exact S512 prepare reverse inputs must total "
            f"{GDN_PREPARE_S512_REVERSE_INPUT_BYTES} bytes, got {observed_bytes}"
        )

    named_arrays: tuple[tuple[str, np.ndarray], ...] = tuple(
        (name, array)
        for array, (name, _) in zip(arrays, _EXACT_INPUT_SPECS, strict=True)
    )
    if attention_mask is not None:
        if not isinstance(attention_mask, np.ndarray):
            raise TypeError("attention_mask must be a NumPy array")
        if attention_mask.shape != GDN_PREPARE_S512_MASK_SHAPE:
            raise ValueError(
                "attention_mask shape must be exactly "
                f"{GDN_PREPARE_S512_MASK_SHAPE}, got {attention_mask.shape}"
            )
        if attention_mask.dtype != np.dtype(np.bool_):
            raise TypeError("attention_mask dtype must be exactly bool")
        if not attention_mask.flags.c_contiguous:
            raise ValueError("attention_mask must be C-contiguous")
        named_arrays = (*named_arrays, ("attention_mask", attention_mask))

    for left_index, (left_name, left) in enumerate(named_arrays):
        for right_name, right in named_arrays[left_index + 1 :]:
            if np.shares_memory(left, right):
                raise ValueError(
                    "exact S512 prepare reverse inputs must use distinct, "
                    f"non-overlapping buffers; {left_name} overlaps {right_name}"
                )


def _validate_chunk_inputs(
    key: np.ndarray,
    value: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    prepared_u_cotangent: np.ndarray,
    prepared_w_cotangent: np.ndarray,
    gamma_cotangent: np.ndarray,
    attention_mask_chunks: np.ndarray | None,
) -> tuple[int, int, int, int, int, int]:
    arrays = (
        key,
        value,
        g,
        beta,
        prepared_u_cotangent,
        prepared_w_cotangent,
        gamma_cotangent,
    )
    if any(not isinstance(item, np.ndarray) for item in arrays):
        raise TypeError("prepare reverse chunk inputs must be NumPy arrays")
    if any(item.dtype != np.dtype(np.float32) for item in arrays):
        raise TypeError("prepare reverse chunk inputs must be exactly float32")
    if key.ndim != 4 or value.ndim != 5:
        raise ValueError("prepare reverse requires rank-4 K and rank-5 V chunks")

    chunks, key_heads, chunk, key_dimension = key.shape
    if value.shape[:2] != (chunks, key_heads) or value.shape[3] != chunk:
        raise ValueError("prepare reverse K/V chunk and key-head axes must match")
    pairs = value.shape[2]
    value_dimension = value.shape[-1]
    gate_shape = (chunks, key_heads, pairs, chunk)
    if g.shape != gate_shape or beta.shape != gate_shape:
        raise ValueError("prepare reverse g/beta chunk shapes must match V")
    if prepared_u_cotangent.shape != value.shape:
        raise ValueError("prepared_u_cotangent must have the V chunk shape")
    expected_w_shape = (chunks, key_heads, pairs, chunk, key_dimension)
    if prepared_w_cotangent.shape != expected_w_shape:
        raise ValueError("prepared_w_cotangent must end in the key feature dimension")
    if gamma_cotangent.shape != gate_shape:
        raise ValueError("gamma_cotangent must have the g chunk shape")
    if (
        min(
            chunks,
            key_heads,
            pairs,
            chunk,
            key_dimension,
            value_dimension,
        )
        <= 0
    ):
        raise ValueError("prepare reverse dimensions must be positive")

    if attention_mask_chunks is not None:
        if not isinstance(attention_mask_chunks, np.ndarray):
            raise TypeError("attention_mask_chunks must be a NumPy array")
        if attention_mask_chunks.shape != (chunks, chunk):
            raise ValueError("attention_mask_chunks must have shape [chunks, chunk]")
        if attention_mask_chunks.dtype != np.dtype(np.bool_):
            raise TypeError("attention_mask_chunks dtype must be exactly bool")
    return chunks, key_heads, pairs, chunk, key_dimension, value_dimension


def _transpose_unit_lower_solve(
    unit_lower: np.ndarray,
    solution_cotangent: np.ndarray,
) -> np.ndarray:
    """Solve ``unit_lower.T @ rhs_bar = solution_cotangent`` in FP32."""
    if not isinstance(unit_lower, np.ndarray) or not isinstance(
        solution_cotangent, np.ndarray
    ):
        raise TypeError("transpose solve inputs must be NumPy arrays")
    if unit_lower.dtype != np.float32 or solution_cotangent.dtype != np.float32:
        raise TypeError("transpose solve inputs must be exactly float32")
    if unit_lower.ndim < 2 or unit_lower.shape[-1] != unit_lower.shape[-2]:
        raise ValueError("unit_lower must end in one square matrix")
    if solution_cotangent.shape[:-2] != unit_lower.shape[:-2]:
        raise ValueError("transpose solve batch dimensions must match")
    if solution_cotangent.shape[-2] != unit_lower.shape[-1]:
        raise ValueError("transpose solve row dimensions must match")
    return np.asarray(
        np.linalg.solve(
            np.swapaxes(unit_lower, -1, -2),
            solution_cotangent,
        ),
        dtype=np.float32,
    )


def _masked_chunk_primals(
    key: np.ndarray,
    value: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    attention_mask_chunks: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if attention_mask_chunks is None:
        return key, value, g, beta
    key_mask = attention_mask_chunks[:, None, :, None]
    value_mask = attention_mask_chunks[:, None, None, :, None]
    gate_mask = attention_mask_chunks[:, None, None, :]
    return (
        np.asarray(key * key_mask, dtype=np.float32),
        np.asarray(value * value_mask, dtype=np.float32),
        np.asarray(g * gate_mask, dtype=np.float32),
        np.asarray(beta * gate_mask, dtype=np.float32),
    )


def _mask_chunk_gradients(
    gradients: GDNPrepareBoundaryGradients,
    attention_mask_chunks: np.ndarray | None,
) -> GDNPrepareBoundaryGradients:
    if attention_mask_chunks is None:
        return gradients
    key_mask = attention_mask_chunks[:, None, :, None]
    value_mask = attention_mask_chunks[:, None, None, :, None]
    gate_mask = attention_mask_chunks[:, None, None, :]
    return GDNPrepareBoundaryGradients(
        key=np.ascontiguousarray(gradients.key * key_mask, dtype=np.float32),
        value=np.ascontiguousarray(gradients.value * value_mask, dtype=np.float32),
        g=np.ascontiguousarray(gradients.g * gate_mask, dtype=np.float32),
        beta=np.ascontiguousarray(gradients.beta * gate_mask, dtype=np.float32),
    )


def _dense_prepare_reverse_chunks_numpy(
    key: np.ndarray,
    value: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    prepared_u_cotangent: np.ndarray,
    prepared_w_cotangent: np.ndarray,
    gamma_cotangent: np.ndarray,
    *,
    attention_mask_chunks: np.ndarray | None = None,
) -> GDNPrepareBoundaryGradients:
    """Transpose dense WY preparation on generic chunk-major FP32 arrays."""
    (
        _chunks,
        _key_heads,
        _pairs,
        chunk,
        _key_dimension,
        value_dimension,
    ) = _validate_chunk_inputs(
        key,
        value,
        g,
        beta,
        prepared_u_cotangent,
        prepared_w_cotangent,
        gamma_cotangent,
        attention_mask_chunks,
    )
    key, value, g, beta = _masked_chunk_primals(
        key,
        value,
        g,
        beta,
        attention_mask_chunks,
    )

    prefix = np.cumsum(g, axis=-1, dtype=np.float32)
    gamma = np.exp(prefix).astype(np.float32, copy=False)

    # Gamma-only losses do not depend on K, V, or beta.  This exact shortcut
    # also keeps ABI/identity tests from performing an irrelevant dense solve.
    if not np.any(prepared_u_cotangent) and not np.any(prepared_w_cotangent):
        prefix_cotangent = np.asarray(
            gamma_cotangent * gamma,
            dtype=np.float32,
        )
        g_cotangent = np.flip(
            np.cumsum(
                np.flip(prefix_cotangent, axis=-1),
                axis=-1,
                dtype=np.float32,
            ),
            axis=-1,
        )
        zeros = GDNPrepareBoundaryGradients(
            key=np.zeros_like(key),
            value=np.zeros_like(value),
            g=np.ascontiguousarray(g_cotangent, dtype=np.float32),
            beta=np.zeros_like(beta),
        )
        return _mask_chunk_gradients(zeros, attention_mask_chunks)

    key_gram = np.einsum(
        "chid,chjd->chij",
        key,
        key,
        dtype=np.float32,
        optimize=False,
    )
    decay = np.exp(prefix[..., :, None] - prefix[..., None, :]).astype(
        np.float32,
        copy=False,
    )
    decay = np.tril(decay)
    strict_lower_mask = np.tril(
        np.ones((chunk, chunk), dtype=np.bool_),
        k=-1,
    )
    strict_lower = np.where(
        strict_lower_mask,
        beta[..., :, None] * key_gram[:, :, None, :, :] * decay,
        np.float32(0.0),
    ).astype(np.float32, copy=False)

    rhs_u = beta[..., None] * value
    rhs_w = beta[..., None] * gamma[..., None] * key[:, :, None, :, :]
    rhs = np.concatenate((rhs_u, rhs_w), axis=-1).astype(np.float32, copy=False)
    unit_lower = strict_lower + np.eye(chunk, dtype=np.float32)
    solution = np.linalg.solve(unit_lower, rhs).astype(np.float32, copy=False)
    solution_cotangent = np.concatenate(
        (prepared_u_cotangent, prepared_w_cotangent),
        axis=-1,
    ).astype(np.float32, copy=False)
    rhs_cotangent = _transpose_unit_lower_solve(unit_lower, solution_cotangent)
    strict_lower_cotangent = -np.matmul(
        rhs_cotangent,
        np.swapaxes(solution, -1, -2),
    )
    strict_lower_cotangent = np.where(
        strict_lower_mask,
        strict_lower_cotangent,
        np.float32(0.0),
    ).astype(np.float32, copy=False)

    rhs_u_cotangent = rhs_cotangent[..., :value_dimension]
    rhs_w_cotangent = rhs_cotangent[..., value_dimension:]
    value_cotangent = np.asarray(
        beta[..., None] * rhs_u_cotangent,
        dtype=np.float32,
    )
    beta_cotangent = np.sum(
        rhs_u_cotangent * value,
        axis=-1,
        dtype=np.float32,
    )
    beta_cotangent += np.sum(
        rhs_w_cotangent * gamma[..., None] * key[:, :, None, :, :],
        axis=-1,
        dtype=np.float32,
    )
    gamma_total_cotangent = np.asarray(gamma_cotangent, dtype=np.float32).copy()
    gamma_total_cotangent += np.sum(
        rhs_w_cotangent * beta[..., None] * key[:, :, None, :, :],
        axis=-1,
        dtype=np.float32,
    )
    key_pair_cotangent = np.asarray(
        rhs_w_cotangent * beta[..., None] * gamma[..., None],
        dtype=np.float32,
    )

    beta_cotangent += np.sum(
        strict_lower_cotangent * key_gram[:, :, None, :, :] * decay,
        axis=-1,
        dtype=np.float32,
    )
    key_gram_cotangent = np.sum(
        strict_lower_cotangent * beta[..., :, None] * decay,
        axis=2,
        dtype=np.float32,
    )
    key_cotangent = np.sum(key_pair_cotangent, axis=2, dtype=np.float32)
    key_cotangent += np.matmul(
        key_gram_cotangent + np.swapaxes(key_gram_cotangent, -1, -2),
        key,
    ).astype(np.float32, copy=False)

    decay_cotangent = np.asarray(
        strict_lower_cotangent * beta[..., :, None] * key_gram[:, :, None, :, :],
        dtype=np.float32,
    )
    weighted_decay_cotangent = np.asarray(
        decay_cotangent * decay,
        dtype=np.float32,
    )
    prefix_cotangent = np.asarray(
        gamma_total_cotangent * gamma,
        dtype=np.float32,
    )
    prefix_cotangent += np.sum(
        weighted_decay_cotangent,
        axis=-1,
        dtype=np.float32,
    )
    prefix_cotangent -= np.sum(
        weighted_decay_cotangent,
        axis=-2,
        dtype=np.float32,
    )
    g_cotangent = np.flip(
        np.cumsum(
            np.flip(prefix_cotangent, axis=-1),
            axis=-1,
            dtype=np.float32,
        ),
        axis=-1,
    )

    gradients = GDNPrepareBoundaryGradients(
        key=np.ascontiguousarray(key_cotangent, dtype=np.float32),
        value=np.ascontiguousarray(value_cotangent, dtype=np.float32),
        g=np.ascontiguousarray(g_cotangent, dtype=np.float32),
        beta=np.ascontiguousarray(beta_cotangent, dtype=np.float32),
    )
    return _mask_chunk_gradients(gradients, attention_mask_chunks)


def gdn_prepare_s512_reverse_numpy(
    key: "NDArray[np.float32]",
    value: "NDArray[np.float32]",
    g: "NDArray[np.float32]",
    beta: "NDArray[np.float32]",
    prepared_u_cotangent: "NDArray[np.float32]",
    prepared_w_cotangent: "NDArray[np.float32]",
    gamma_cotangent: "NDArray[np.float32]",
    *,
    attention_mask: "NDArray[np.bool_] | None" = None,
) -> GDNPrepareBoundaryGradients:
    """Return the exact FP32 transpose of one S512 prepare boundary.

    ``attention_mask`` mirrors the NumPy forward oracle and returns gradients
    with respect to the pre-mask arrays.  The future FFI receives transformed
    arrays and therefore has no mask operand.
    """
    arrays = (
        key,
        value,
        g,
        beta,
        prepared_u_cotangent,
        prepared_w_cotangent,
        gamma_cotangent,
    )
    _validate_exact_inputs(arrays, attention_mask)

    key_chunks = key.reshape(
        _CHUNKS,
        _CHUNK,
        _KEY_HEADS,
        _HEAD_DIMENSION,
    ).transpose(0, 2, 1, 3)
    value_chunks = value.reshape(
        _CHUNKS,
        _CHUNK,
        _KEY_HEADS,
        _HEADS_PER_KEY,
        _HEAD_DIMENSION,
    ).transpose(0, 2, 3, 1, 4)
    g_chunks = g.reshape(
        _CHUNKS,
        _CHUNK,
        _KEY_HEADS,
        _HEADS_PER_KEY,
    ).transpose(0, 2, 3, 1)
    beta_chunks = beta.reshape(
        _CHUNKS,
        _CHUNK,
        _KEY_HEADS,
        _HEADS_PER_KEY,
    ).transpose(0, 2, 3, 1)
    u_bar_chunks = prepared_u_cotangent.reshape(
        _CHUNKS,
        _CHUNK,
        _KEY_HEADS,
        _HEADS_PER_KEY,
        _HEAD_DIMENSION,
    ).transpose(0, 2, 3, 1, 4)
    w_bar_chunks = prepared_w_cotangent.reshape(
        _CHUNKS,
        _CHUNK,
        _KEY_HEADS,
        _HEADS_PER_KEY,
        _HEAD_DIMENSION,
    ).transpose(0, 2, 3, 1, 4)
    gamma_bar_chunks = gamma_cotangent.reshape(
        _CHUNKS,
        _CHUNK,
        _KEY_HEADS,
        _HEADS_PER_KEY,
    ).transpose(0, 2, 3, 1)
    mask_chunks = (
        None if attention_mask is None else attention_mask.reshape(_CHUNKS, _CHUNK)
    )

    gradients = _dense_prepare_reverse_chunks_numpy(
        key_chunks,
        value_chunks,
        g_chunks,
        beta_chunks,
        u_bar_chunks,
        w_bar_chunks,
        gamma_bar_chunks,
        attention_mask_chunks=mask_chunks,
    )
    result = GDNPrepareBoundaryGradients(
        key=np.ascontiguousarray(
            gradients.key.transpose(0, 2, 1, 3).reshape(GDN_PREPARE_S512_KEY_SHAPE),
            dtype=np.float32,
        ),
        value=np.ascontiguousarray(
            gradients.value.transpose(0, 3, 1, 2, 4).reshape(
                GDN_PREPARE_S512_VALUE_SHAPE
            ),
            dtype=np.float32,
        ),
        g=np.ascontiguousarray(
            gradients.g.transpose(0, 3, 1, 2).reshape(GDN_PREPARE_S512_GATE_SHAPE),
            dtype=np.float32,
        ),
        beta=np.ascontiguousarray(
            gradients.beta.transpose(0, 3, 1, 2).reshape(GDN_PREPARE_S512_GATE_SHAPE),
            dtype=np.float32,
        ),
    )
    observed_gradient_bytes = sum(
        array.nbytes for array in (result.key, result.value, result.g, result.beta)
    )
    if observed_gradient_bytes != GDN_PREPARE_S512_GRADIENT_BYTES:
        raise RuntimeError(
            "exact S512 prepare gradients must total "
            f"{GDN_PREPARE_S512_GRADIENT_BYTES} bytes, got "
            f"{observed_gradient_bytes}"
        )
    return result
