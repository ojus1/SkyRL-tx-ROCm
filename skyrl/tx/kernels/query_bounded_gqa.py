"""Experimental query-bounded native grouped-query attention.

This module is an isolated Pallas prototype.  It is deliberately not wired
into :mod:`skyrl.tx.layers.attention`: the non-interpret lowering still needs
an isolated gfx1100 compile and stress matrix before it is eligible for the
model path.

The important difference from JAX's current GPU MHA helper is that every
Pallas call covers only a fixed query range and maps each query head directly
to its KV head.  It therefore neither repeats K/V nor places all 32K queries in
one forward/backward dispatch.  Right padding is represented as a key mask;
padded query outputs retain the same semantics as ``jax.nn`` attention and
must still be excluded by the loss mask.
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

_DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)


def _forward_kernel(
    q_ref,
    k_ref,
    v_ref,
    key_mask_ref,
    out_ref,
    lse_ref,
    *,
    query_start: int,
    scale: float,
    block_q: int,
    block_k: int,
    head_dim: int,
):
    """FlashAttention forward for one bounded query range and one Q head."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    local_query_block = pl.program_id(0)
    key_sequence_length = k_ref.shape[0]
    q = plgpu.load(q_ref)

    row_max = jnp.full((block_q,), -jnp.inf, dtype=jnp.float32)
    row_sum = jnp.zeros((block_q,), dtype=jnp.float32)
    accumulator = jnp.zeros((block_q, head_dim), dtype=jnp.float32)
    query_positions = query_start + local_query_block * block_q + jnp.arange(block_q)

    def consume_key_block(key_block, carry):
        previous_output, previous_max, previous_sum = carry
        key_slice = pl.dslice(key_block * block_k, block_k)
        key_positions = key_block * block_k + jnp.arange(block_k)
        k = plgpu.load(k_ref.at[key_slice, :])
        logits = plgpu.dot(q, k.T).astype(jnp.float32)
        logits *= scale * math.log2(math.e)
        valid = (key_positions[None, :] <= query_positions[:, None]) & (key_mask_ref[key_slice][None, :] != 0)
        logits = jnp.where(valid, logits, _DEFAULT_MASK_VALUE)

        current_max = jnp.max(logits, axis=-1)
        next_max = jnp.maximum(previous_max, current_max)
        correction = jnp.exp2(previous_max - next_max)
        probabilities = jnp.exp2(logits - next_max[:, None])
        next_sum = correction * previous_sum + jnp.sum(probabilities, axis=-1)
        v = plgpu.load(v_ref.at[key_slice, :])
        next_output = correction[:, None] * previous_output + plgpu.dot(probabilities.astype(v.dtype), v)
        return next_output, next_max, next_sum

    # Causality uses global query positions.  This is the key offset that is
    # lost if a later query chunk is naively treated as a standalone sequence.
    key_block_limit = lax.min(
        key_sequence_length // block_k,
        lax.div(
            query_start + (local_query_block + 1) * block_q + block_k - 1,
            block_k,
        ),
    )
    accumulator, row_max, row_sum = lax.fori_loop(
        0,
        key_block_limit,
        consume_key_block,
        (accumulator, row_max, row_sum),
    )
    output = accumulator / row_sum[:, None]
    plgpu.store(out_ref, output.astype(out_ref.dtype))
    lse_ref[...] = row_max + jnp.log2(row_sum)


def _dq_kernel(
    q_ref,
    k_ref,
    v_ref,
    key_mask_ref,
    out_ref,
    dout_ref,
    lse_ref,
    dq_ref,
    *,
    query_start: int,
    scale: float,
    block_q: int,
    block_k: int,
    head_dim: int,
):
    """Compute dQ for one bounded query range without any cross-grid writes."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    local_query_block = pl.program_id(0)
    key_sequence_length = k_ref.shape[0]
    query_positions = query_start + local_query_block * block_q + jnp.arange(block_q)
    q = plgpu.load(q_ref)
    output = plgpu.load(out_ref).astype(jnp.float32)
    dout = plgpu.load(dout_ref)
    delta = jnp.sum(output * dout.astype(jnp.float32), axis=-1)
    lse = lse_ref[...]
    dq = jnp.zeros((block_q, head_dim), dtype=jnp.float32)

    def consume_key_block(key_block, previous_dq):
        key_slice = pl.dslice(key_block * block_k, block_k)
        key_positions = key_block * block_k + jnp.arange(block_k)
        k = plgpu.load(k_ref.at[key_slice, :])
        v = plgpu.load(v_ref.at[key_slice, :])
        logits = plgpu.dot(q, k.T).astype(jnp.float32)
        logits *= scale * math.log2(math.e)
        valid = (key_positions[None, :] <= query_positions[:, None]) & (key_mask_ref[key_slice][None, :] != 0)
        logits = jnp.where(valid, logits, _DEFAULT_MASK_VALUE)
        probabilities = jnp.exp2(logits - lse[:, None])
        dprobabilities = plgpu.dot(dout, v.T).astype(jnp.float32)
        dscores = probabilities * (dprobabilities - delta[:, None]) * scale
        return previous_dq + plgpu.dot(dscores.astype(k.dtype), k).astype(jnp.float32)

    key_block_limit = lax.min(
        key_sequence_length // block_k,
        lax.div(
            query_start + (local_query_block + 1) * block_q + block_k - 1,
            block_k,
        ),
    )
    dq = lax.fori_loop(0, key_block_limit, consume_key_block, dq)
    plgpu.store(dq_ref, dq.astype(dq_ref.dtype))


def _dkdv_kernel(
    q_ref,
    k_ref,
    v_ref,
    key_mask_ref,
    out_ref,
    dout_ref,
    lse_ref,
    previous_dk_ref,
    previous_dv_ref,
    dk_ref,
    dv_ref,
    *,
    query_start: int,
    scale: float,
    query_chunk_size: int,
    group_size: int,
    block_q: int,
    block_k: int,
    head_dim: int,
):
    """Accumulate one query range into dK/dV, one program per KV tile.

    Each program is the sole writer for its KV tile.  Query blocks and the Q
    heads belonging to a KV head are visited in a fixed lexical order; chunk
    calls are connected by an explicit input/output dependency.  No atomics or
    nondeterministic cross-program reduction is used.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    key_block = pl.program_id(0)
    key_slice = pl.dslice(key_block * block_k, block_k)
    key_positions = key_block * block_k + jnp.arange(block_k)
    k = plgpu.load(k_ref)
    v = plgpu.load(v_ref)
    dk = plgpu.load(previous_dk_ref).astype(jnp.float32)
    dv = plgpu.load(previous_dv_ref).astype(jnp.float32)

    def consume_query_block(local_query_block, carry):
        next_dk, next_dv = carry
        query_slice = pl.dslice(local_query_block * block_q, block_q)
        query_positions = query_start + local_query_block * block_q + jnp.arange(block_q)
        valid = (key_positions[None, :] <= query_positions[:, None]) & (key_mask_ref[key_slice][None, :] != 0)

        # Static unrolling is intentional: it gives a fixed, reproducible
        # reduction order for the four Q heads in each Qwen3.5 KV group.
        for query_head_in_group in range(group_size):
            # A scalar JAX index works in both Triton lowering and Pallas's HLO
            # interpreter.  A bare Python ``int`` currently trips the latter's
            # state-discharge path (JAX 0.10.2).
            head_index = jnp.asarray(query_head_in_group, dtype=jnp.int32)
            q = plgpu.load(q_ref.at[query_slice, head_index, :])
            output = plgpu.load(out_ref.at[query_slice, head_index, :]).astype(jnp.float32)
            dout = plgpu.load(dout_ref.at[query_slice, head_index, :])
            lse = lse_ref[head_index, query_slice]
            delta = jnp.sum(output * dout.astype(jnp.float32), axis=-1)
            logits = plgpu.dot(q, k.T).astype(jnp.float32)
            logits *= scale * math.log2(math.e)
            logits = jnp.where(valid, logits, _DEFAULT_MASK_VALUE)
            probabilities = jnp.exp2(logits - lse[:, None])
            next_dv += plgpu.dot(probabilities.astype(dout.dtype).T, dout).astype(jnp.float32)
            dprobabilities = plgpu.dot(dout, v.T).astype(jnp.float32)
            dscores = probabilities * (dprobabilities - delta[:, None]) * scale
            next_dk += plgpu.dot(dscores.astype(q.dtype).T, q).astype(jnp.float32)
        return next_dk, next_dv

    dk, dv = lax.fori_loop(
        0,
        query_chunk_size // block_q,
        consume_query_block,
        (dk, dv),
    )
    plgpu.store(dk_ref, dk.astype(dk_ref.dtype))
    plgpu.store(dv_ref, dv.astype(dv_ref.dtype))


def _compiler_params():
    from jax.experimental.pallas import triton as plgpu

    return plgpu.CompilerParams(num_warps=4, num_stages=1)


def _forward_chunk(
    q,
    k,
    v,
    key_mask,
    *,
    query_start: int,
    scale: float,
    block_q: int,
    block_k: int,
    interpret: bool,
):
    from jax.experimental import pallas as pl

    batch_size, query_chunk_size, query_heads, head_dim = q.shape
    sequence_length = k.shape[1]
    group_size = query_heads // k.shape[2]
    output, lse = pl.pallas_call(
        partial(
            _forward_kernel,
            query_start=query_start,
            scale=scale,
            block_q=block_q,
            block_k=block_k,
            head_dim=head_dim,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct((batch_size, query_heads, query_chunk_size), jnp.float32),
        ),
        grid=(query_chunk_size // block_q, batch_size, query_heads),
        in_specs=(
            pl.BlockSpec((None, block_q, None, head_dim), lambda qb, b, qh: (b, qb, qh, 0)),
            pl.BlockSpec(
                (None, sequence_length, None, head_dim),
                lambda qb, b, qh: (b, 0, qh // group_size, 0),
            ),
            pl.BlockSpec(
                (None, sequence_length, None, head_dim),
                lambda qb, b, qh: (b, 0, qh // group_size, 0),
            ),
            pl.BlockSpec((None, sequence_length), lambda qb, b, qh: (b, 0)),
        ),
        out_specs=(
            pl.BlockSpec((None, block_q, None, head_dim), lambda qb, b, qh: (b, qb, qh, 0)),
            pl.BlockSpec((None, None, block_q), lambda qb, b, qh: (b, qh, qb)),
        ),
        compiler_params=_compiler_params(),
        interpret=interpret,
        name=f"query_bounded_gqa_forward_q{query_start}",
        metadata={"query_start": str(query_start), "query_size": str(query_chunk_size)},
    )(q, k, v, key_mask)
    return output, lse


def _dq_chunk(
    q,
    k,
    v,
    key_mask,
    output,
    dout,
    lse,
    *,
    query_start: int,
    scale: float,
    block_q: int,
    block_k: int,
    interpret: bool,
):
    from jax.experimental import pallas as pl

    batch_size, query_chunk_size, query_heads, head_dim = q.shape
    sequence_length = k.shape[1]
    group_size = query_heads // k.shape[2]
    return pl.pallas_call(
        partial(
            _dq_kernel,
            query_start=query_start,
            scale=scale,
            block_q=block_q,
            block_k=block_k,
            head_dim=head_dim,
        ),
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(query_chunk_size // block_q, batch_size, query_heads),
        in_specs=(
            pl.BlockSpec((None, block_q, None, head_dim), lambda qb, b, qh: (b, qb, qh, 0)),
            pl.BlockSpec(
                (None, sequence_length, None, head_dim),
                lambda qb, b, qh: (b, 0, qh // group_size, 0),
            ),
            pl.BlockSpec(
                (None, sequence_length, None, head_dim),
                lambda qb, b, qh: (b, 0, qh // group_size, 0),
            ),
            pl.BlockSpec((None, sequence_length), lambda qb, b, qh: (b, 0)),
            pl.BlockSpec((None, block_q, None, head_dim), lambda qb, b, qh: (b, qb, qh, 0)),
            pl.BlockSpec((None, block_q, None, head_dim), lambda qb, b, qh: (b, qb, qh, 0)),
            pl.BlockSpec((None, None, block_q), lambda qb, b, qh: (b, qh, qb)),
        ),
        out_specs=pl.BlockSpec((None, block_q, None, head_dim), lambda qb, b, qh: (b, qb, qh, 0)),
        compiler_params=_compiler_params(),
        interpret=interpret,
        name=f"query_bounded_gqa_dq_q{query_start}",
        metadata={"query_start": str(query_start), "query_size": str(query_chunk_size)},
    )(q, k, v, key_mask, output, dout, lse)


def _accumulate_dkdv_chunk(
    q,
    k,
    v,
    key_mask,
    output,
    dout,
    lse,
    previous_dk,
    previous_dv,
    *,
    query_start: int,
    scale: float,
    block_q: int,
    block_k: int,
    interpret: bool,
):
    from jax.experimental import pallas as pl

    batch_size, query_chunk_size, query_heads, head_dim = q.shape
    sequence_length, kv_heads = k.shape[1:3]
    group_size = query_heads // kv_heads
    return pl.pallas_call(
        partial(
            _dkdv_kernel,
            query_start=query_start,
            scale=scale,
            query_chunk_size=query_chunk_size,
            group_size=group_size,
            block_q=block_q,
            block_k=block_k,
            head_dim=head_dim,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(previous_dk.shape, previous_dk.dtype),
            jax.ShapeDtypeStruct(previous_dv.shape, previous_dv.dtype),
        ),
        grid=(sequence_length // block_k, batch_size, kv_heads),
        in_specs=(
            pl.BlockSpec(
                (None, query_chunk_size, group_size, head_dim),
                lambda kb, b, kvh: (b, 0, kvh, 0),
            ),
            pl.BlockSpec((None, block_k, None, head_dim), lambda kb, b, kvh: (b, kb, kvh, 0)),
            pl.BlockSpec((None, block_k, None, head_dim), lambda kb, b, kvh: (b, kb, kvh, 0)),
            pl.BlockSpec((None, sequence_length), lambda kb, b, kvh: (b, 0)),
            pl.BlockSpec(
                (None, query_chunk_size, group_size, head_dim),
                lambda kb, b, kvh: (b, 0, kvh, 0),
            ),
            pl.BlockSpec(
                (None, query_chunk_size, group_size, head_dim),
                lambda kb, b, kvh: (b, 0, kvh, 0),
            ),
            pl.BlockSpec(
                (None, group_size, query_chunk_size),
                lambda kb, b, kvh: (b, kvh, 0),
            ),
            pl.BlockSpec((None, block_k, None, head_dim), lambda kb, b, kvh: (b, kb, kvh, 0)),
            pl.BlockSpec((None, block_k, None, head_dim), lambda kb, b, kvh: (b, kb, kvh, 0)),
        ),
        out_specs=(
            pl.BlockSpec((None, block_k, None, head_dim), lambda kb, b, kvh: (b, kb, kvh, 0)),
            pl.BlockSpec((None, block_k, None, head_dim), lambda kb, b, kvh: (b, kb, kvh, 0)),
        ),
        input_output_aliases={7: 0, 8: 1},
        compiler_params=_compiler_params(),
        interpret=interpret,
        name=f"query_bounded_gqa_dkdv_q{query_start}",
        metadata={"query_start": str(query_start), "query_size": str(query_chunk_size)},
    )(
        q,
        k,
        v,
        key_mask,
        output,
        dout,
        lse,
        previous_dk,
        previous_dv,
    )


def _validate_inputs(
    q,
    k,
    v,
    key_mask,
    *,
    query_chunk_size: int,
    block_q: int,
    block_k: int,
    backward_block_q: int,
    backward_block_k: int,
):
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or key_mask.ndim != 2:
        raise ValueError("q/k/v must be rank four and key_mask rank two")
    if q.dtype != k.dtype or q.dtype != v.dtype or not jnp.issubdtype(q.dtype, jnp.floating):
        raise TypeError("q, k, and v must have the same floating-point dtype")
    batch, sequence_length, query_heads, head_dim = q.shape
    if batch != 1:
        raise ValueError("the prototype is restricted to batch size one")
    if sequence_length <= 0 or query_heads <= 0 or head_dim <= 0:
        raise ValueError("sequence, query-head, and head dimensions must be positive")
    if k.shape[0] != batch or v.shape[0] != batch:
        raise ValueError("q, k, and v batch dimensions must match")
    if k.shape[1] != sequence_length or v.shape[1] != sequence_length:
        raise ValueError("the prototype only supports causal self-attention")
    if k.shape[2] <= 0 or k.shape[2] != v.shape[2] or query_heads % k.shape[2]:
        raise ValueError("query heads must be divisible by the matching K/V head count")
    if k.shape[3] != head_dim or v.shape[3] != head_dim:
        raise ValueError("q, k, and v head dimensions must match")
    if key_mask.shape != (batch, sequence_length):
        raise ValueError("key_mask must have shape [batch, sequence_length]")
    if not (jnp.issubdtype(key_mask.dtype, jnp.bool_) or jnp.issubdtype(key_mask.dtype, jnp.integer)):
        raise TypeError("key_mask must have boolean or integer dtype")
    for name, value in (
        ("query_chunk_size", query_chunk_size),
        ("block_q", block_q),
        ("block_k", block_k),
        ("backward_block_q", backward_block_q),
        ("backward_block_k", backward_block_k),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if sequence_length % query_chunk_size:
        raise ValueError("sequence length must be divisible by query_chunk_size")
    if query_chunk_size % block_q:
        raise ValueError("query_chunk_size must be divisible by block_q")
    if sequence_length % block_k:
        raise ValueError("sequence length must be divisible by block_k")
    if query_chunk_size % backward_block_q:
        raise ValueError("query_chunk_size must be divisible by backward_block_q")
    if sequence_length % backward_block_k:
        raise ValueError("sequence length must be divisible by backward_block_k")


def _validate_forward_chunk_inputs(
    q_chunk,
    k,
    v,
    key_mask,
    *,
    query_start: int,
    block_q: int,
    block_k: int,
):
    """Validate the standalone forward-chunk contract without full-Q assumptions."""
    if type(query_start) is not int:
        raise TypeError("query_start must be an exact concrete Python int")
    for name, value in (("block_q", block_q), ("block_k", block_k)):
        if type(value) is not int:
            raise TypeError(f"{name} must be an exact concrete Python int")
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    if q_chunk.ndim != 4 or k.ndim != 4 or v.ndim != 4 or key_mask.ndim != 2:
        raise ValueError("q_chunk/k/v must be rank four and key_mask rank two")
    if q_chunk.dtype != k.dtype or q_chunk.dtype != v.dtype or not jnp.issubdtype(q_chunk.dtype, jnp.floating):
        raise TypeError("q_chunk, k, and v must have the same floating-point dtype")

    batch, query_chunk_size, query_heads, head_dim = q_chunk.shape
    key_batch, sequence_length, kv_heads, key_head_dim = k.shape
    if batch != 1 or key_batch != 1 or v.shape[0] != 1:
        raise ValueError("the forward-chunk prototype is restricted to batch size one")
    if query_chunk_size <= 0 or sequence_length <= 0:
        raise ValueError("query-chunk and key sequence lengths must be positive")
    if query_heads <= 0 or kv_heads <= 0 or head_dim <= 0:
        raise ValueError("query-head, K/V-head, and head dimensions must be positive")
    if v.shape[1] != sequence_length or v.shape[2] != kv_heads:
        raise ValueError("k and v sequence and head dimensions must match")
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by the K/V head count")
    if key_head_dim != head_dim or v.shape[3] != head_dim:
        raise ValueError("q_chunk, k, and v head dimensions must match")
    if key_mask.shape != (1, sequence_length):
        raise ValueError("key_mask must have shape [1, key_sequence_length]")
    if not (jnp.issubdtype(key_mask.dtype, jnp.bool_) or jnp.issubdtype(key_mask.dtype, jnp.integer)):
        raise TypeError("key_mask must have boolean or integer dtype")

    if query_chunk_size % block_q:
        raise ValueError("query chunk length must be divisible by block_q")
    if sequence_length % block_k:
        raise ValueError("key sequence length must be divisible by block_k")
    if query_start < 0:
        raise ValueError("query_start must be nonnegative")
    if query_start % block_q:
        raise ValueError("query_start must be aligned to block_q")
    if query_start + query_chunk_size > sequence_length:
        raise ValueError("query chunk must fit within the key sequence length")


def _forward_impl(
    q,
    k,
    v,
    key_mask,
    scale: float,
    query_chunk_size: int,
    block_q: int,
    block_k: int,
    interpret: bool,
):
    outputs = []
    lses = []
    for query_start in range(0, q.shape[1], query_chunk_size):
        output, lse = _forward_chunk(
            q[:, query_start : query_start + query_chunk_size],
            k,
            v,
            key_mask,
            query_start=query_start,
            scale=scale,
            block_q=block_q,
            block_k=block_k,
            interpret=interpret,
        )
        outputs.append(output)
        lses.append(lse)
    return jnp.concatenate(outputs, axis=1), jnp.concatenate(lses, axis=2)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 9, 10))
def _query_bounded_gqa(
    q,
    k,
    v,
    key_mask,
    scale: float,
    query_chunk_size: int,
    block_q: int,
    block_k: int,
    backward_block_q: int,
    backward_block_k: int,
    interpret: bool,
):
    del backward_block_q, backward_block_k
    return _forward_impl(
        q,
        k,
        v,
        key_mask,
        scale,
        query_chunk_size,
        block_q,
        block_k,
        interpret,
    )[0]


def _query_bounded_gqa_fwd(
    q,
    k,
    v,
    key_mask,
    scale: float,
    query_chunk_size: int,
    block_q: int,
    block_k: int,
    backward_block_q: int,
    backward_block_k: int,
    interpret: bool,
):
    del backward_block_q, backward_block_k
    output, lse = _forward_impl(
        q,
        k,
        v,
        key_mask,
        scale,
        query_chunk_size,
        block_q,
        block_k,
        interpret,
    )
    return output, (q, k, v, key_mask, output, lse)


def _query_bounded_gqa_bwd(
    scale: float,
    query_chunk_size: int,
    block_q: int,
    block_k: int,
    backward_block_q: int,
    backward_block_k: int,
    interpret: bool,
    residuals,
    dout,
):
    del block_q, block_k
    q, k, v, key_mask, output, lse = residuals
    dq_chunks = []
    # FP32 persistent accumulators give one deterministic reduction chain over
    # query chunks.  Aliasing is a performance contract, not a semantic one.
    dk = jnp.zeros(k.shape, dtype=jnp.float32)
    dv = jnp.zeros(v.shape, dtype=jnp.float32)
    for query_start in range(0, q.shape[1], query_chunk_size):
        query_stop = query_start + query_chunk_size
        q_chunk = q[:, query_start:query_stop]
        output_chunk = output[:, query_start:query_stop]
        dout_chunk = dout[:, query_start:query_stop]
        lse_chunk = lse[:, :, query_start:query_stop]
        dq_chunks.append(
            _dq_chunk(
                q_chunk,
                k,
                v,
                key_mask,
                output_chunk,
                dout_chunk,
                lse_chunk,
                query_start=query_start,
                scale=scale,
                block_q=backward_block_q,
                block_k=backward_block_k,
                interpret=interpret,
            )
        )
        dk, dv = _accumulate_dkdv_chunk(
            q_chunk,
            k,
            v,
            key_mask,
            output_chunk,
            dout_chunk,
            lse_chunk,
            dk,
            dv,
            query_start=query_start,
            scale=scale,
            block_q=backward_block_q,
            block_k=backward_block_k,
            interpret=interpret,
        )
    return (
        jnp.concatenate(dq_chunks, axis=1),
        dk.astype(k.dtype),
        dv.astype(v.dtype),
        None,
    )


_query_bounded_gqa.defvjp(_query_bounded_gqa_fwd, _query_bounded_gqa_bwd)


def query_bounded_gqa_forward_chunk(
    q_chunk,
    k,
    v,
    key_mask,
    *,
    query_start: int,
    scale: float | None = None,
    block_q: int = 64,
    block_k: int = 64,
    interpret: bool = False,
):
    """Run one experimental, forward-only query chunk against a longer K/V.

    ``q_chunk`` has shape ``[1, C, Hq, D]`` while ``k`` and ``v`` have shape
    ``[1, L, Hkv, D]``.  ``query_start`` is the chunk's global position in the
    causal sequence, must be an exact Python ``int`` aligned to ``block_q``,
    and the complete chunk must lie inside ``[0, L)``.  No query chunks are
    concatenated and no custom VJP is installed around this entry point.

    As in :func:`query_bounded_gqa`, ``key_mask`` must be a nonempty prefix of
    ones followed by zeros.  Shape and dtype are checked here, but the mask's
    data-dependent right-padding structure cannot be inspected while JAX is
    tracing.  ``interpret=True`` is the CPU correctness mode.
    """
    _validate_forward_chunk_inputs(
        q_chunk,
        k,
        v,
        key_mask,
        query_start=query_start,
        block_q=block_q,
        block_k=block_k,
    )
    if scale is None:
        scale = q_chunk.shape[-1] ** -0.5
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    return _forward_chunk(
        q_chunk,
        k,
        v,
        key_mask,
        query_start=query_start,
        scale=float(scale),
        block_q=block_q,
        block_k=block_k,
        interpret=interpret,
    )[0]


def query_bounded_gqa(
    q,
    k,
    v,
    key_mask,
    *,
    scale: float | None = None,
    query_chunk_size: int = 512,
    block_q: int = 64,
    block_k: int = 64,
    backward_block_q: int = 32,
    backward_block_k: int = 32,
    interpret: bool = False,
):
    """Run the isolated native-GQA prototype.

    ``key_mask`` must describe right padding (a nonempty prefix of ones followed
    by zeros).  This structural mask precondition is not inspected while JAX is
    tracing.  The mask is applied only to keys, matching the portable SkyRL
    path.  Sequence and tile dimensions must divide exactly; partial final
    query or key tiles are intentionally rejected.  ``interpret=True`` is the
    CPU-only correctness mode.  Calling the non-interpret path on ROCm is
    intentionally not part of the production dispatch selector yet.
    """
    _validate_inputs(
        q,
        k,
        v,
        key_mask,
        query_chunk_size=query_chunk_size,
        block_q=block_q,
        block_k=block_k,
        backward_block_q=backward_block_q,
        backward_block_k=backward_block_k,
    )
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    return _query_bounded_gqa(
        q,
        k,
        v,
        key_mask,
        float(scale),
        query_chunk_size,
        block_q,
        block_k,
        backward_block_q,
        backward_block_k,
        interpret,
    )


def qwen35_32k_dispatch_plan(
    *,
    query_chunk_size: int = 512,
    block_q: int = 64,
    block_k: int = 64,
    backward_block_q: int = 32,
    backward_block_k: int = 32,
) -> dict[str, int]:
    """Return the static dispatch bounds for B1/T32K/Hq16/Hkv4/D256."""
    sequence_length = 32_768
    if sequence_length % query_chunk_size or query_chunk_size % block_q:
        raise ValueError("32K and query_chunk_size must be exactly tiled")
    if sequence_length % block_k:
        raise ValueError("32K must be exactly tiled by block_k")
    if query_chunk_size % backward_block_q or sequence_length % backward_block_k:
        raise ValueError("32K backward dimensions must be exactly tiled")
    query_chunks = sequence_length // query_chunk_size
    return {
        "query_chunks": query_chunks,
        "forward_dispatches": query_chunks,
        "dq_dispatches": query_chunks,
        "dkdv_dispatches": query_chunks,
        "forward_programs_per_dispatch": query_chunk_size // block_q * 16,
        "dq_programs_per_dispatch": query_chunk_size // backward_block_q * 16,
        "dkdv_programs_per_dispatch": sequence_length // backward_block_k * 4,
        "max_key_blocks_per_forward_program": sequence_length // block_k,
        "max_key_blocks_per_dq_program": sequence_length // backward_block_k,
        "query_blocks_per_dkdv_program": query_chunk_size // backward_block_q,
    }
