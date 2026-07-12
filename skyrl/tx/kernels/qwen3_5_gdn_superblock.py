"""Portable Qwen3.5 Gated DeltaNet superblock equations.

This module is an unwired CPU semantic and scheduling experiment.  It is not a
GPU kernel and is never selected by :mod:`skyrl.tx.models.qwen3_5`.  The
operation boundary models a future bounded ROCm implementation which processes
8 or 16 64-token chunks per dispatch-visible stage:

* Q/K stay in key-head form.  Value head ``hv`` reads key head
  ``hv // value_heads_per_key_head``; no explicit Q/K head repeat is formed.
* the outer scan carries the recurrent state in FP32 only at superblock
  boundaries;
* one superblock prepares and consumes its WY ``U``/``W`` values locally; and
* the complete superblock body is checkpointed so reverse-mode autodiff
  recomputes ``U``/``W`` instead of retaining them across all superblocks.

Portable JAX cannot promise a device buffer schedule or launch count.  In
particular, ``lax.map`` over the two Qwen3.5 value heads associated with each
key head only specifies the no-repeat dataflow; a production HIP kernel should
make that association inside one bounded launch.  Likewise, checkpointing
specifies reverse recomputation but does not substitute for an explicit FFI
custom VJP with a measured GPU buffer assignment.  The grouped transpose also
changes the reduction order of the two value-group contributions to BF16 Q/K
cotangents, so a production reverse kernel should accumulate those
contributions in FP32 before its boundary cast.  The oracle also pads a physical
tail to a complete configured superblock; a production scheduler should select
an 8-chunk tail bucket or fall back to the 64-token chunk path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Qwen35GDNSuperblockConfig:
    """Static scheduling geometry for the portable GDN oracle."""

    chunk_size: int = 64
    chunks_per_superblock: int = 16

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunks_per_superblock not in (8, 16):
            raise ValueError("chunks_per_superblock must be 8 or 16")

    @property
    def tokens_per_superblock(self) -> int:
        return self.chunk_size * self.chunks_per_superblock


def qwen35_gdn_superblock_logical_buffers(
    *,
    batch_size: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    config: Qwen35GDNSuperblockConfig,
) -> dict[str, int]:
    """Return byte sizes of important FP32 tensors visible in one superblock.

    The entries are individual logical tensor sizes, not an additive peak
    allocation estimate.  XLA may alias or fuse them, while the portable CPU
    lowering may retain buffers that a tiled GPU implementation would not.
    ``rhs`` and ``solution`` each contain both the U- and W-widths.
    """
    dimensions = (batch_size, num_value_heads, key_head_dim, value_head_dim)
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError("all batch/head dimensions must be positive")

    tokens = config.tokens_per_superblock
    chunks = config.chunks_per_superblock
    chunk = config.chunk_size
    fp32_bytes = 4
    state = batch_size * num_value_heads * key_head_dim * value_head_dim * fp32_bytes
    u = batch_size * tokens * num_value_heads * value_head_dim * fp32_bytes
    w = batch_size * tokens * num_value_heads * key_head_dim * fp32_bytes
    chunk_matrix = batch_size * num_value_heads * chunks * chunk * chunk * fp32_bytes
    return {
        "boundary_state": state,
        "u": u,
        "w": w,
        "u_plus_w": u + w,
        "gamma": batch_size * tokens * num_value_heads * fp32_bytes,
        "decay_mask": chunk_matrix,
        "strict_lower": chunk_matrix,
        "rhs": u + w,
        "solution": u + w,
        "intra_attention": chunk_matrix,
        "corrected_values": u,
        "output_block": u,
    }


def _l2norm_in_input_dtype(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    inverse_norm = jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)
    return x * inverse_norm


def _one_value_group_superblock(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    state: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Prepare and consume WY values for one value-head/key-head pairing.

    ``query`` and ``key`` are shared by every mapped value-head group and have
    shape ``[B,Hk,C,L,Dk]``.  The remaining inputs describe one of the
    ``Hv/Hk`` value-head groups, so no ``[B,T,Hv,Dk]`` Q/K tensor is needed.
    """
    value_head_dim = value.shape[-1]

    k_beta = key * beta[..., None]
    v_beta = value * beta[..., None]

    g_cumsum = jnp.cumsum(g, axis=-1)
    gamma = jnp.exp(g_cumsum)
    decay_mask = jnp.tril(
        jnp.exp(jnp.tril(g_cumsum[..., :, None] - g_cumsum[..., None, :]))
    )

    strict_lower = jnp.tril(
        (k_beta @ jnp.swapaxes(key, -1, -2)) * decay_mask,
        k=-1,
    )
    rhs = jnp.concatenate([v_beta, k_beta * gamma[..., None]], axis=-1)
    solution = jax.lax.linalg.triangular_solve(
        strict_lower,
        rhs,
        left_side=True,
        lower=True,
        unit_diagonal=True,
    )
    u = solution[..., :value_head_dim]
    w_decay = solution[..., value_head_dim:]

    def chunk_step(
        current_state: jax.Array,
        inputs: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, jax.Array]:
        q_chunk, k_chunk, u_chunk, w_chunk, gamma_chunk, decay_chunk = inputs
        intra_attention = q_chunk @ jnp.swapaxes(k_chunk, -1, -2) * decay_chunk
        corrected_values = u_chunk - w_chunk @ current_state
        inter_output = (q_chunk * gamma_chunk[..., None]) @ current_state
        output = inter_output + intra_attention @ corrected_values

        gamma_end = gamma_chunk[..., -1, None, None]
        key_to_end_decay = decay_chunk[..., -1, :][..., None]
        next_state = gamma_end * current_state + (
            jnp.swapaxes(k_chunk * key_to_end_decay, -1, -2) @ corrected_values
        )
        return next_state, output

    scan_inputs = tuple(
        jnp.moveaxis(item, 2, 0) for item in (query, key, u, w_decay, gamma, decay_mask)
    )
    final_state, outputs = jax.lax.scan(chunk_step, state, scan_inputs)
    return final_state, jnp.moveaxis(outputs, 0, 2)


def _validate_inputs(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    attention_mask: jax.Array | None,
    initial_state: jax.Array | None,
) -> tuple[int, int, int, int, int, int]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have rank 4")
    if g.ndim != 3 or beta.ndim != 3:
        raise ValueError("g and beta must have rank 3")
    if query.shape != key.shape:
        raise ValueError("query and key shapes must match")

    batch, sequence, key_heads, key_head_dim = query.shape
    value_batch, value_sequence, value_heads, value_head_dim = value.shape
    if any(
        dimension <= 0
        for dimension in (
            batch,
            sequence,
            key_heads,
            value_heads,
            key_head_dim,
            value_head_dim,
        )
    ):
        raise ValueError(
            "all batch/head dimensions and sequence length must be positive"
        )
    if (value_batch, value_sequence) != (batch, sequence):
        raise ValueError("query/key and value batch/sequence shapes must match")
    if value_heads % key_heads:
        raise ValueError("num_value_heads must be divisible by num_key_heads")
    if g.shape != (batch, sequence, value_heads):
        raise ValueError("g must have shape [B,T,Hv]")
    if beta.shape != g.shape:
        raise ValueError("beta must have the same shape as g")
    if attention_mask is not None and attention_mask.shape != (batch, sequence):
        raise ValueError("attention_mask must have shape [B,T]")
    if initial_state is not None and initial_state.shape != (
        batch,
        value_heads,
        key_head_dim,
        value_head_dim,
    ):
        raise ValueError("initial_state must have shape [B,Hv,Dk,Dv]")
    return batch, sequence, key_heads, value_heads, key_head_dim, value_head_dim


def qwen35_gdn_superblocks(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    *,
    attention_mask: jax.Array | None = None,
    initial_state: jax.Array | None = None,
    config: Qwen35GDNSuperblockConfig = Qwen35GDNSuperblockConfig(),
) -> tuple[jax.Array, jax.Array]:
    """Execute grouped-head gated-delta recurrence in bounded superblocks.

    Args:
        query: ``[B,T,Hk,Dk]`` Q tensor, before L2 normalization.
        key: ``[B,T,Hk,Dk]`` K tensor, before L2 normalization.
        value: ``[B,T,Hv,Dv]`` value tensor.
        g: ``[B,T,Hv]`` log decay values (normally non-positive).
        beta: ``[B,T,Hv]`` update gate values.
        attention_mask: optional ``[B,T]`` validity mask.  A masked token has
            zero output and an identity state transition.
        initial_state: optional FP32-compatible ``[B,Hv,Dk,Dv]`` state.
        config: chunk and superblock scheduling geometry.

    Returns:
        Output ``[B,T,Hv,Dv]`` in the query dtype and final recurrent state
        ``[B,Hv,Dk,Dv]`` in FP32.
    """
    (
        batch,
        sequence,
        key_heads,
        value_heads,
        key_head_dim,
        value_head_dim,
    ) = _validate_inputs(
        query,
        key,
        value,
        g,
        beta,
        attention_mask,
        initial_state,
    )
    output_dtype = query.dtype
    heads_per_key = value_heads // key_heads
    superblock_tokens = config.tokens_per_superblock
    padding = (-sequence) % superblock_tokens
    padded_sequence = sequence + padding
    num_superblocks = padded_sequence // superblock_tokens

    if attention_mask is None:
        attention_mask = jnp.ones((batch, sequence), dtype=jnp.bool_)
    else:
        attention_mask = attention_mask.astype(jnp.bool_)

    def pad_sequence_axis(item: jax.Array, axis: int) -> jax.Array:
        pad_width = [(0, 0)] * item.ndim
        pad_width[axis] = (0, padding)
        return jnp.pad(item, pad_width)

    query = pad_sequence_axis(query, 1)
    key = pad_sequence_axis(key, 1)
    value = pad_sequence_axis(value, 1)
    g = pad_sequence_axis(g, 1)
    beta = pad_sequence_axis(beta, 1)
    attention_mask = pad_sequence_axis(attention_mask, 1)

    query_blocks = query.transpose(0, 2, 1, 3).reshape(
        batch,
        key_heads,
        num_superblocks,
        config.chunks_per_superblock,
        config.chunk_size,
        key_head_dim,
    )
    key_blocks = key.transpose(0, 2, 1, 3).reshape(query_blocks.shape)

    value_blocks = value.reshape(
        batch,
        padded_sequence,
        key_heads,
        heads_per_key,
        value_head_dim,
    ).transpose(0, 2, 3, 1, 4)
    value_blocks = value_blocks.reshape(
        batch,
        key_heads,
        heads_per_key,
        num_superblocks,
        config.chunks_per_superblock,
        config.chunk_size,
        value_head_dim,
    )
    g_blocks = g.reshape(
        batch,
        padded_sequence,
        key_heads,
        heads_per_key,
    ).transpose(0, 2, 3, 1)
    g_blocks = g_blocks.reshape(
        batch,
        key_heads,
        heads_per_key,
        num_superblocks,
        config.chunks_per_superblock,
        config.chunk_size,
    )
    beta_blocks = beta.reshape(
        batch,
        padded_sequence,
        key_heads,
        heads_per_key,
    ).transpose(0, 2, 3, 1)
    beta_blocks = beta_blocks.reshape(g_blocks.shape)
    mask_blocks = attention_mask.reshape(
        batch,
        num_superblocks,
        config.chunks_per_superblock,
        config.chunk_size,
    )

    scan_inputs = (
        jnp.moveaxis(query_blocks, 2, 0),
        jnp.moveaxis(key_blocks, 2, 0),
        jnp.moveaxis(value_blocks, 3, 0).transpose(0, 3, 1, 2, 4, 5, 6),
        jnp.moveaxis(g_blocks, 3, 0).transpose(0, 3, 1, 2, 4, 5),
        jnp.moveaxis(beta_blocks, 3, 0).transpose(0, 3, 1, 2, 4, 5),
        jnp.moveaxis(mask_blocks, 1, 0),
    )

    if initial_state is None:
        initial_state = jnp.zeros(
            (batch, value_heads, key_head_dim, value_head_dim),
            dtype=jnp.float32,
        )
    grouped_state = (
        initial_state.astype(jnp.float32)
        .reshape(
            batch,
            key_heads,
            heads_per_key,
            key_head_dim,
            value_head_dim,
        )
        .transpose(2, 0, 1, 3, 4)
    )

    def superblock_step(
        state: jax.Array,
        inputs: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, jax.Array]:
        q_block, k_block, v_groups, g_groups, beta_groups, mask = inputs

        qk_mask = mask[:, None, :, :, None]
        q_block = _l2norm_in_input_dtype(
            q_block * qk_mask.astype(q_block.dtype)
        ).astype(jnp.float32)
        q_block = q_block * (1.0 / math.sqrt(key_head_dim))
        k_block = _l2norm_in_input_dtype(
            k_block * qk_mask.astype(k_block.dtype)
        ).astype(jnp.float32)

        def value_group_step(
            group_inputs: tuple[jax.Array, ...],
        ) -> tuple[jax.Array, jax.Array]:
            v_group, g_group, beta_group, group_state = group_inputs
            group_mask = mask[:, None, :, :]
            v_group = (v_group * group_mask[..., None].astype(v_group.dtype)).astype(
                jnp.float32
            )
            g_group = g_group.astype(jnp.float32) * group_mask.astype(jnp.float32)
            beta_group = beta_group.astype(jnp.float32) * group_mask.astype(jnp.float32)
            return _one_value_group_superblock(
                q_block,
                k_block,
                v_group,
                g_group,
                beta_group,
                group_state,
            )

        next_state, grouped_outputs = jax.lax.map(
            value_group_step,
            (v_groups, g_groups, beta_groups, state),
        )
        # [R,B,Hk,C,L,Dv] -> [B,C,L,Hk,R,Dv] -> [B,S,Hv,Dv]
        outputs = grouped_outputs.transpose(1, 3, 4, 2, 0, 5).reshape(
            batch,
            superblock_tokens,
            value_heads,
            value_head_dim,
        )
        # FP32 is local to the recurrent stage.  Cast at the superblock output
        # boundary so the outer scan does not accumulate a full-sequence FP32
        # output buffer.
        return next_state, outputs.astype(output_dtype)

    # This is the executable statement that U/W are reverse-recomputed per
    # superblock.  It deliberately checkpoints normalization and WY preparation
    # together, rather than retaining full-sequence prepared tensors.
    rematerialized_step = jax.checkpoint(
        superblock_step,
        prevent_cse=True,
        policy=jax.checkpoint_policies.nothing_saveable,
    )
    final_grouped_state, output_blocks = jax.lax.scan(
        rematerialized_step,
        grouped_state,
        scan_inputs,
    )

    outputs = output_blocks.transpose(1, 0, 2, 3, 4).reshape(
        batch,
        padded_sequence,
        value_heads,
        value_head_dim,
    )
    final_state = final_grouped_state.transpose(1, 2, 0, 3, 4).reshape(
        batch,
        value_heads,
        key_head_dim,
        value_head_dim,
    )
    return outputs[:, :sequence].astype(output_dtype), final_state
