"""Import-light NumPy forward oracles for grouped-head Qwen3.5 GDN.

This module is a CPU-only semantic gate.  It does not import or dispatch JAX,
PyTorch, TileLang, HIP, or CUDA, and it is not a runtime fallback.

The FlashQLA equations are pinned to QwenLM/FlashQLA commit
``40b7527f6c6e2ed8ed65350103e3ca64174f53f3``:

* ``flash_qla/ops/gated_delta_rule/chunk/hopper/kkt_solve.py`` constructs
  ``R = inv(I + strictLower(diag(beta) K K^T))``; and
* ``flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py`` forms
  ``Ag = decay * R * beta_column`` before the recurrent state/output update.

``skyrl_wy_gdn_forward_numpy`` independently mirrors the decay-folded WY
equations in :mod:`skyrl.tx.models.qwen3_5` and
:mod:`skyrl.tx.kernels.qwen3_5_gdn_superblock`.  The representations differ,
but are related by diagonal conjugation:

``inv(I + diag(gamma) L diag(1/gamma)) = diag(gamma) inv(I + L) diag(1/gamma)``.

The public forward functions accept raw FP32 Q/K, matching SkyRL's fallback:
Q/K are L2-normalized in FP32, Q is scaled by ``1 / sqrt(Dk)``, and value head
``hv`` reads key head ``hv // (Hv / Hk)`` without requiring callers to repeat
Q/K.  A false mask entry transforms Q/K/V/g/beta to zero, giving zero output
and an identity state transition.

This first gate intentionally covers forward equivalence only.  Comparing two
finite-difference estimates would not independently validate FlashQLA's
backward equations, so reverse-mode work remains a separate gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


FLASHQLA_REVISION = "40b7527f6c6e2ed8ed65350103e3ca64174f53f3"
FLASHQLA_KKT_SOURCE = (
    "https://github.com/QwenLM/FlashQLA/blob/"
    f"{FLASHQLA_REVISION}/flash_qla/ops/gated_delta_rule/chunk/hopper/kkt_solve.py"
)
FLASHQLA_FUSED_FORWARD_SOURCE = (
    "https://github.com/QwenLM/FlashQLA/blob/"
    f"{FLASHQLA_REVISION}/flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py"
)
SKYRL_GDN_EQUATION_SOURCE = "skyrl/tx/models/qwen3_5.py"
FLASHQLA_CHUNK_SIZE = 64


@dataclass(frozen=True)
class GDNChunkRepresentation:
    """Prepared matrices for one chunk representation.

    All arrays have arbitrary common leading dimensions.  Token axes are the
    final matrix axes.  ``basis_inverse`` is the inverse before the beta-column
    multiply: it is the no-decay inverse for FlashQLA and the decay-folded
    inverse for SkyRL.  ``effective_update`` is comparable between the two.
    """

    basis_inverse: "NDArray[np.float32]"
    effective_update: "NDArray[np.float32]"
    prepared_u: "NDArray[np.float32]"
    prepared_w: "NDArray[np.float32]"
    gamma: "NDArray[np.float32]"
    decay: "NDArray[np.float32]"


def _require_f32(array: np.ndarray, name: str) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} dtype must be exactly float32")


def _require_chunk_inputs(
    key: np.ndarray,
    value: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
) -> tuple[tuple[int, ...], int, int, int]:
    for name, array in (("key", key), ("value", value), ("g", g), ("beta", beta)):
        _require_f32(array, name)
    if (
        key.ndim < 2
        or value.ndim != key.ndim
        or g.ndim != key.ndim - 1
        or beta.ndim != g.ndim
    ):
        raise ValueError(
            "chunk inputs must have shapes [...,L,Dk], [...,L,Dv], [...,L], [...,L]"
        )
    leading = key.shape[:-2]
    chunk, key_dimension = key.shape[-2:]
    if value.shape[:-2] != leading or value.shape[-2] != chunk:
        raise ValueError("chunk K/V leading and token dimensions must match")
    if g.shape != (*leading, chunk) or beta.shape != g.shape:
        raise ValueError("chunk g/beta shapes must match K leading/token dimensions")
    value_dimension = value.shape[-1]
    if min(*leading, chunk, key_dimension, value_dimension) <= 0:
        raise ValueError("all chunk dimensions must be positive")
    return leading, chunk, key_dimension, value_dimension


def _chunk_decay(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prefix = np.cumsum(g, axis=-1, dtype=np.float32)
    gamma = np.exp(prefix).astype(np.float32, copy=False)
    decay = np.exp(prefix[..., :, None] - prefix[..., None, :]).astype(
        np.float32, copy=False
    )
    decay = np.tril(decay).astype(np.float32, copy=False)
    return gamma, decay


def flashqla_chunk_representation_numpy(
    key: "NDArray[np.float32]",
    value: "NDArray[np.float32]",
    g: "NDArray[np.float32]",
    beta: "NDArray[np.float32]",
) -> GDNChunkRepresentation:
    """Return FlashQLA's no-decay KKT inverse and effective WY values.

    Inputs use generic ``[...,L,D]`` chunk shapes.  This is the FP32 algebraic
    form of the pinned source, not an emulation of its BF16 storage, fast math,
    or NVIDIA execution schedule.
    """
    leading, chunk, _, _ = _require_chunk_inputs(key, value, g, beta)
    gamma, decay = _chunk_decay(g)
    gram = np.einsum("...id,...jd->...ij", key, key, dtype=np.float32, optimize=False)
    strict_lower = np.tril(beta[..., :, None] * gram, k=-1).astype(
        np.float32, copy=False
    )
    unit_lower = strict_lower + np.eye(chunk, dtype=np.float32)
    identity = np.broadcast_to(
        np.eye(chunk, dtype=np.float32), (*leading, chunk, chunk)
    )
    raw_inverse = np.linalg.solve(unit_lower, identity).astype(np.float32, copy=False)

    # fused_fwd.py: Ag[i,j] = exp(prefix[i]-prefix[j]) * R[i,j] * beta[j].
    effective_update = (decay * raw_inverse * beta[..., None, :]).astype(
        np.float32, copy=False
    )
    prepared_u = np.matmul(effective_update, value).astype(np.float32, copy=False)
    prepared_w = np.matmul(effective_update, gamma[..., :, None] * key).astype(
        np.float32, copy=False
    )
    return GDNChunkRepresentation(
        basis_inverse=raw_inverse,
        effective_update=effective_update,
        prepared_u=prepared_u,
        prepared_w=prepared_w,
        gamma=gamma,
        decay=decay,
    )


def skyrl_wy_chunk_representation_numpy(
    key: "NDArray[np.float32]",
    value: "NDArray[np.float32]",
    g: "NDArray[np.float32]",
    beta: "NDArray[np.float32]",
) -> GDNChunkRepresentation:
    """Return SkyRL's decay-folded triangular solve and effective WY values."""
    leading, chunk, key_dimension, value_dimension = _require_chunk_inputs(
        key, value, g, beta
    )
    gamma, decay = _chunk_decay(g)
    gram = np.einsum("...id,...jd->...ij", key, key, dtype=np.float32, optimize=False)
    strict_lower = np.tril(beta[..., :, None] * gram * decay, k=-1).astype(
        np.float32, copy=False
    )
    unit_lower = strict_lower + np.eye(chunk, dtype=np.float32)

    # The identity block exposes the decay-folded inverse for the conjugation
    # gate.  The U/W blocks mirror SkyRL's current concatenated RHS solve.
    identity = np.broadcast_to(
        np.eye(chunk, dtype=np.float32), (*leading, chunk, chunk)
    )
    rhs_u = beta[..., :, None] * value
    rhs_w = beta[..., :, None] * gamma[..., :, None] * key
    rhs = np.concatenate((identity, rhs_u, rhs_w), axis=-1).astype(
        np.float32, copy=False
    )
    solution = np.linalg.solve(unit_lower, rhs).astype(np.float32, copy=False)
    decay_folded_inverse = solution[..., :chunk]
    effective_update = (decay_folded_inverse * beta[..., None, :]).astype(
        np.float32, copy=False
    )
    prepared_u = solution[..., chunk : chunk + value_dimension]
    prepared_w = solution[
        ..., chunk + value_dimension : chunk + value_dimension + key_dimension
    ]
    return GDNChunkRepresentation(
        basis_inverse=decay_folded_inverse,
        effective_update=effective_update,
        prepared_u=prepared_u,
        prepared_w=prepared_w,
        gamma=gamma,
        decay=decay,
    )


def _validate_forward_inputs(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    attention_mask: np.ndarray | None,
    initial_state: np.ndarray | None,
) -> tuple[int, int, int, int, int, int]:
    for name, array in (
        ("query", query),
        ("key", key),
        ("value", value),
        ("g", g),
        ("beta", beta),
    ):
        _require_f32(array, name)
    if (
        query.ndim != 4
        or key.ndim != 4
        or value.ndim != 4
        or g.ndim != 3
        or beta.ndim != 3
    ):
        raise ValueError("forward inputs must have ranks Q/K/V=4 and g/beta=3")
    if query.shape != key.shape:
        raise ValueError("query and key shapes must match")
    batch, sequence, key_heads, key_dimension = query.shape
    if value.shape[:2] != (batch, sequence):
        raise ValueError("query/key and value batch/sequence dimensions must match")
    value_heads, value_dimension = value.shape[2:]
    if value_heads % key_heads:
        raise ValueError("num_value_heads must be divisible by num_key_heads")
    if g.shape != (batch, sequence, value_heads) or beta.shape != g.shape:
        raise ValueError("g and beta must have shape [B,T,Hv]")
    if (
        min(batch, sequence, key_heads, value_heads, key_dimension, value_dimension)
        <= 0
    ):
        raise ValueError("all forward dimensions must be positive")
    if attention_mask is not None:
        if not isinstance(attention_mask, np.ndarray):
            raise TypeError("attention_mask must be a NumPy array")
        if attention_mask.shape != (
            batch,
            sequence,
        ) or attention_mask.dtype != np.dtype(np.bool_):
            raise TypeError("attention_mask must have shape [B,T] and dtype bool")
    if initial_state is not None:
        _require_f32(initial_state, "initial_state")
        if initial_state.shape != (batch, value_heads, key_dimension, value_dimension):
            raise ValueError("initial_state must have shape [B,Hv,Dk,Dv]")
    return batch, sequence, key_heads, value_heads, key_dimension, value_dimension


def _prepare_forward_inputs(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    attention_mask: np.ndarray | None,
    initial_state: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch, sequence, _, value_heads, key_dimension, value_dimension = (
        _validate_forward_inputs(
            query,
            key,
            value,
            g,
            beta,
            attention_mask,
            initial_state,
        )
    )
    if attention_mask is None:
        attention_mask = np.ones((batch, sequence), dtype=np.bool_)
    mask = attention_mask.astype(np.float32, copy=False)
    query = np.asarray(query * mask[..., None, None], dtype=np.float32)
    key = np.asarray(key * mask[..., None, None], dtype=np.float32)
    value = np.asarray(value * mask[..., None, None], dtype=np.float32)
    g = np.asarray(g * mask[..., None], dtype=np.float32)
    beta = np.asarray(beta * mask[..., None], dtype=np.float32)

    query_norm = np.sqrt(
        np.sum(query * query, axis=-1, keepdims=True, dtype=np.float32)
        + np.float32(1e-6)
    )
    key_norm = np.sqrt(
        np.sum(key * key, axis=-1, keepdims=True, dtype=np.float32) + np.float32(1e-6)
    )
    query = np.asarray(query / query_norm, dtype=np.float32)
    key = np.asarray(key / key_norm, dtype=np.float32)
    query *= np.float32(1.0 / math.sqrt(key_dimension))

    if initial_state is None:
        state = np.zeros(
            (batch, value_heads, key_dimension, value_dimension), dtype=np.float32
        )
    else:
        state = initial_state.copy()
    return query, key, value, g, beta, state


def recurrent_gdn_forward_numpy(
    query: "NDArray[np.float32]",
    key: "NDArray[np.float32]",
    value: "NDArray[np.float32]",
    g: "NDArray[np.float32]",
    beta: "NDArray[np.float32]",
    *,
    attention_mask: "NDArray[np.bool_] | None" = None,
    initial_state: "NDArray[np.float32] | None" = None,
) -> tuple["NDArray[np.float32]", "NDArray[np.float32]"]:
    """Execute the independent token-by-token grouped-head GDN recurrence."""
    query, key, value, g, beta, state = _prepare_forward_inputs(
        query,
        key,
        value,
        g,
        beta,
        attention_mask,
        initial_state,
    )
    batch, sequence, key_heads, _ = query.shape
    value_heads, value_dimension = value.shape[2], value.shape[3]
    heads_per_key = value_heads // key_heads
    head_map = np.arange(value_heads) // heads_per_key
    output = np.empty((batch, sequence, value_heads, value_dimension), dtype=np.float32)

    for token in range(sequence):
        query_token = query[:, token, head_map]
        key_token = key[:, token, head_map]
        state *= np.exp(g[:, token]).astype(np.float32, copy=False)[..., None, None]
        memory = np.einsum(
            "bhkv,bhk->bhv", state, key_token, dtype=np.float32, optimize=False
        )
        delta = (value[:, token] - memory) * beta[:, token, :, None]
        state += key_token[..., :, None] * delta[..., None, :]
        output[:, token] = np.einsum(
            "bhkv,bhk->bhv", state, query_token, dtype=np.float32, optimize=False
        )
    return output, state


def _chunk_gdn_forward_numpy(
    representation: str,
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    *,
    attention_mask: np.ndarray | None,
    initial_state: np.ndarray | None,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if chunk_size != FLASHQLA_CHUNK_SIZE:
        raise ValueError(
            f"this pinned FlashQLA gate requires chunk_size={FLASHQLA_CHUNK_SIZE}"
        )
    query, key, value, g, beta, state = _prepare_forward_inputs(
        query,
        key,
        value,
        g,
        beta,
        attention_mask,
        initial_state,
    )
    batch, sequence, key_heads, key_dimension = query.shape
    value_heads, value_dimension = value.shape[2], value.shape[3]
    heads_per_key = value_heads // key_heads
    padding = (-sequence) % chunk_size
    padded_sequence = sequence + padding
    chunks = padded_sequence // chunk_size

    def pad_tokens(array: np.ndarray) -> np.ndarray:
        pad_width = [(0, 0)] * array.ndim
        pad_width[1] = (0, padding)
        return np.pad(array, pad_width)

    query = pad_tokens(query)
    key = pad_tokens(key)
    value = pad_tokens(value)
    g = pad_tokens(g)
    beta = pad_tokens(beta)

    query_chunks = query.reshape(
        batch, chunks, chunk_size, key_heads, key_dimension
    ).transpose(0, 1, 3, 2, 4)
    key_chunks = key.reshape(
        batch, chunks, chunk_size, key_heads, key_dimension
    ).transpose(0, 1, 3, 2, 4)
    value_chunks = value.reshape(
        batch, chunks, chunk_size, value_heads, value_dimension
    ).transpose(0, 1, 3, 2, 4)
    g_chunks = g.reshape(batch, chunks, chunk_size, value_heads).transpose(0, 1, 3, 2)
    beta_chunks = beta.reshape(g.shape[0], chunks, chunk_size, value_heads).transpose(
        0, 1, 3, 2
    )
    head_map = np.arange(value_heads) // heads_per_key
    query_groups = query_chunks[:, :, head_map]
    key_groups = key_chunks[:, :, head_map]

    if representation == "flashqla":
        prepared = flashqla_chunk_representation_numpy(
            key_groups, value_chunks, g_chunks, beta_chunks
        )
    elif representation == "skyrl":
        prepared = skyrl_wy_chunk_representation_numpy(
            key_groups, value_chunks, g_chunks, beta_chunks
        )
    else:
        raise ValueError(f"unknown chunk representation: {representation}")

    output_chunks = np.empty(
        (batch, chunks, value_heads, chunk_size, value_dimension), dtype=np.float32
    )
    for chunk in range(chunks):
        query_chunk = query_groups[:, chunk]
        key_chunk = key_groups[:, chunk]
        gamma = prepared.gamma[:, chunk]
        decay = prepared.decay[:, chunk]
        corrected = prepared.prepared_u[:, chunk] - np.matmul(
            prepared.prepared_w[:, chunk], state
        )
        intra = (
            np.einsum(
                "bhid,bhjd->bhij",
                query_chunk,
                key_chunk,
                dtype=np.float32,
                optimize=False,
            )
            * decay
        )
        inter_output = np.matmul(query_chunk * gamma[..., :, None], state)
        output_chunks[:, chunk] = inter_output + np.matmul(intra, corrected)
        reverse_decay = decay[..., -1, :, None]
        state = gamma[..., -1, None, None] * state + np.matmul(
            np.swapaxes(key_chunk * reverse_decay, -1, -2),
            corrected,
        )

    output = output_chunks.transpose(0, 1, 3, 2, 4).reshape(
        batch, padded_sequence, value_heads, value_dimension
    )
    return output[:, :sequence].astype(np.float32, copy=False), state.astype(
        np.float32, copy=False
    )


def flashqla_gdn_forward_numpy(
    query: "NDArray[np.float32]",
    key: "NDArray[np.float32]",
    value: "NDArray[np.float32]",
    g: "NDArray[np.float32]",
    beta: "NDArray[np.float32]",
    *,
    attention_mask: "NDArray[np.bool_] | None" = None,
    initial_state: "NDArray[np.float32] | None" = None,
    chunk_size: int = FLASHQLA_CHUNK_SIZE,
) -> tuple["NDArray[np.float32]", "NDArray[np.float32]"]:
    """Execute the pinned FlashQLA strict-lower KKT forward algebra."""
    return _chunk_gdn_forward_numpy(
        "flashqla",
        query,
        key,
        value,
        g,
        beta,
        attention_mask=attention_mask,
        initial_state=initial_state,
        chunk_size=chunk_size,
    )


def skyrl_wy_gdn_forward_numpy(
    query: "NDArray[np.float32]",
    key: "NDArray[np.float32]",
    value: "NDArray[np.float32]",
    g: "NDArray[np.float32]",
    beta: "NDArray[np.float32]",
    *,
    attention_mask: "NDArray[np.bool_] | None" = None,
    initial_state: "NDArray[np.float32] | None" = None,
    chunk_size: int = FLASHQLA_CHUNK_SIZE,
) -> tuple["NDArray[np.float32]", "NDArray[np.float32]"]:
    """Execute SkyRL's current decay-folded WY forward algebra."""
    return _chunk_gdn_forward_numpy(
        "skyrl",
        query,
        key,
        value,
        g,
        beta,
        attention_mask=attention_mask,
        initial_state=initial_state,
        chunk_size=chunk_size,
    )
