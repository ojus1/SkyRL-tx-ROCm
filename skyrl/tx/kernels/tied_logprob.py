"""Portable semantics for a split-vocabulary tied target-logprob stage.

This module is a portable, default-off JAX oracle, not an accelerated kernel.
Qwen3.5 can select it only through the explicit experimental backend option;
it also specifies the operation that a future bounded ROCm HIP/Pallas stage
should implement for a frozen tied embedding:

``logits = hidden @ embedding.T``
``logprob = logits[target] - logsumexp(logits)``

The vocabulary is visited in fixed-size superblocks and tokens are visited in
M64/M128/M256 chunks, so the largest logical logits buffer is
``M x min(V, vocab_superblock_size)``.  Forward updates a numerically stable
online ``(max, sumexp)`` state while each vocabulary tile is live, so every
forward tile is formed once.  Non-finite maxima are classified explicitly so
infinities and NaNs retain ``jax.nn.logsumexp``'s finite-max convention.
Backward recomputes every tile once.  This removes one logical embedding pass,
but the portable JAX scans remain an equation/lowering experiment rather than a
measured speed kernel.  A production kernel still needs a tiled ROCm
implementation of the same bounded online reduction and custom VJP.

The custom VJP treats ``embedding`` as frozen, exactly as SkyRL's tied output
head is used during LoRA training.  It returns only ``dHidden`` and streams the
softmax-weighted embedding by vocabulary superblock rather than retaining
logits or probabilities from forward.
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp

SUPPORTED_TOKEN_CHUNK_SIZES = (64, 128, 256)


def _validate_inputs(
    hidden: jax.Array,
    embedding: jax.Array,
    target_ids: jax.Array,
    active_mask: jax.Array,
    token_chunk_size: int,
    vocab_superblock_size: int,
) -> None:
    if token_chunk_size not in SUPPORTED_TOKEN_CHUNK_SIZES:
        raise ValueError(
            "token_chunk_size must be one of "
            f"{SUPPORTED_TOKEN_CHUNK_SIZES}, got {token_chunk_size}"
        )
    if vocab_superblock_size <= 0:
        raise ValueError(
            f"vocab_superblock_size must be positive, got {vocab_superblock_size}"
        )
    if hidden.ndim < 1:
        raise ValueError(f"hidden must have at least one dimension, got {hidden.shape}")
    if embedding.ndim != 2:
        raise ValueError(f"embedding must have shape [V, H], got {embedding.shape}")
    if hidden.shape[-1] <= 0 or embedding.shape[0] <= 0:
        raise ValueError(
            "hidden size and vocabulary size must be positive, got "
            f"hidden={hidden.shape} and embedding={embedding.shape}"
        )
    if hidden.shape[-1] != embedding.shape[-1]:
        raise ValueError(
            f"hidden H={hidden.shape[-1]} does not match embedding H={embedding.shape[-1]}"
        )
    if target_ids.shape != hidden.shape[:-1]:
        raise ValueError(
            f"target shape {target_ids.shape} must match hidden leading shape {hidden.shape[:-1]}"
        )
    if active_mask.shape != target_ids.shape:
        raise ValueError(
            f"active_mask shape {active_mask.shape} must match target shape {target_ids.shape}"
        )
    if math.prod(target_ids.shape) <= 0:
        raise ValueError("the token domain must be nonempty")
    if not jnp.issubdtype(hidden.dtype, jnp.floating) or not jnp.issubdtype(
        embedding.dtype, jnp.floating
    ):
        raise TypeError(
            "hidden and embedding must be floating point, got "
            f"{hidden.dtype} and {embedding.dtype}"
        )
    if hidden.dtype != embedding.dtype:
        raise TypeError(
            "the tied hidden and embedding dtypes must match, got "
            f"{hidden.dtype} and {embedding.dtype}"
        )
    if not jnp.issubdtype(target_ids.dtype, jnp.integer):
        raise TypeError(f"target_ids must be integral, got {target_ids.dtype}")
    if active_mask.dtype != jnp.bool_:
        raise TypeError(f"active_mask must be boolean, got {active_mask.dtype}")


def dense_tied_target_logprobs_reference(
    hidden: jax.Array,
    embedding: jax.Array,
    target_ids: jax.Array,
) -> jax.Array:
    """Compose the current dense SkyRL tied-head equations directly.

    This function intentionally materializes the full logits tensor.  It is an
    independent convenience reference for small CPU cases, not the split path.
    Unlike :func:`split_vocab_tied_target_logprobs`, ordinary autodiff through
    this helper differentiates the embedding.
    """
    active_mask = jnp.ones(target_ids.shape, dtype=jnp.bool_)
    _validate_inputs(hidden, embedding, target_ids, active_mask, 64, 1)
    logits = hidden @ embedding.T
    target_logits = jnp.take_along_axis(logits, target_ids[..., None], axis=-1)
    return (target_logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)).squeeze(
        -1
    )


def _vocab_block_descriptors(
    vocab_size: int, vocab_superblock_size: int
) -> tuple[int, jax.Array, jax.Array]:
    """Describe fixed-width slices without padding or copying the embedding.

    ``dynamic_slice`` requires every slice to lie within its operand.  When the
    vocabulary is not divisible by the block width, the final slice therefore
    overlaps the previous slice.  ``assigned_starts`` identifies the disjoint
    conceptual range owned by each slice; the overlap is masked later.
    """
    block_width = min(vocab_size, vocab_superblock_size)
    num_blocks = (vocab_size + block_width - 1) // block_width
    assigned_starts = jnp.arange(num_blocks, dtype=jnp.int32) * block_width
    slice_starts = jnp.minimum(assigned_starts, vocab_size - block_width)
    return block_width, slice_starts, assigned_starts


def _masked_block_logits(
    chunk_hidden: jax.Array,
    embedding_block: jax.Array,
    slice_start: jax.Array,
    assigned_start: jax.Array,
    vocab_size: int,
) -> jax.Array:
    """Form one MxVB tile and mask columns not owned by this block."""
    block_logits = chunk_hidden @ embedding_block.T
    vocab_indices = slice_start + jnp.arange(embedding_block.shape[0], dtype=jnp.int32)
    assigned_end = jnp.minimum(assigned_start + embedding_block.shape[0], vocab_size)
    valid_columns = (vocab_indices >= assigned_start) & (vocab_indices < assigned_end)
    return jnp.where(
        valid_columns[None, :],
        block_logits,
        jnp.asarray(-jnp.inf, dtype=block_logits.dtype),
    )


def _normalize_target_ids(
    target_ids: jax.Array, vocab_size: int
) -> tuple[jax.Array, jax.Array]:
    """Match ``take_along_axis`` negative-index and fill-mode semantics.

    JAX accepts indices in ``[-V, V)`` for a vocabulary axis of length ``V``.
    More distant negative indices and positive indices at or above ``V`` select
    the floating fill value (NaN).  Returning normalized IDs separately keeps
    the block-ownership tests independent of the caller's signed spelling.
    """
    normalized = jnp.where(target_ids < 0, target_ids + vocab_size, target_ids)
    valid = (normalized >= 0) & (normalized < vocab_size)
    return normalized, valid


def _chunk_forward_with_normalizer(
    chunk_hidden: jax.Array,
    chunk_targets: jax.Array,
    chunk_active: jax.Array,
    embedding: jax.Array,
    block_width: int,
    slice_starts: jax.Array,
    assigned_starts: jax.Array,
    vocab_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return logprobs plus the online max and denominator needed by VJP."""
    output_dtype = chunk_hidden.dtype
    chunk_size = chunk_hidden.shape[0]
    initial_max = jnp.full((chunk_size,), -jnp.inf, dtype=output_dtype)
    initial_denominator_f32 = jnp.zeros((chunk_size,), dtype=jnp.float32)
    initial_target = jnp.zeros((chunk_size,), dtype=output_dtype)

    normalized_targets, valid_targets = _normalize_target_ids(chunk_targets, vocab_size)

    def online_normalizer_and_target_scan(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        block_values: tuple[jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        running_max, running_denominator_f32, target_logits = carry
        slice_start, assigned_start = block_values
        embedding_block = jax.lax.dynamic_slice_in_dim(
            embedding, slice_start, block_width, axis=0
        )
        block_logits = _masked_block_logits(
            chunk_hidden,
            embedding_block,
            slice_start,
            assigned_start,
            vocab_size,
        )
        next_max = jnp.maximum(running_max, jnp.max(block_logits, axis=-1))

        # For ordinary finite inputs, rescale the previous denominator to the
        # new maximum and add this block.  The initial ``-inf`` state contributes
        # exact zero.  A non-finite final maximum is classified after the scan;
        # in that case the finite-path denominator is deliberately ignored.
        finite_next_max = jnp.where(jnp.isfinite(next_max), next_max, 0)
        previous_scale = jnp.where(
            jnp.isfinite(running_max) & jnp.isfinite(next_max),
            jnp.exp(running_max - finite_next_max),
            jnp.zeros_like(next_max),
        ).astype(jnp.float32)
        shifted_exponentials = jnp.exp(block_logits - finite_next_max[:, None])
        block_denominator_f32 = jnp.sum(
            shifted_exponentials.astype(jnp.float32), axis=-1
        )
        running_denominator_f32 = (
            running_denominator_f32 * previous_scale + block_denominator_f32
        )

        local_targets = normalized_targets - slice_start.astype(
            normalized_targets.dtype
        )
        assigned_end = jnp.minimum(
            assigned_start + embedding_block.shape[0], vocab_size
        )
        target_in_block = (
            valid_targets
            & (normalized_targets >= assigned_start.astype(normalized_targets.dtype))
            & (normalized_targets < assigned_end)
        )
        safe_local_targets = jnp.clip(local_targets, 0, embedding_block.shape[0] - 1)
        block_target_logits = jnp.take_along_axis(
            block_logits, safe_local_targets[:, None], axis=-1
        ).squeeze(-1)
        target_logits = jnp.where(target_in_block, block_target_logits, target_logits)
        return (
            next_max,
            running_denominator_f32,
            target_logits,
        ), None

    (
        (
            global_max,
            online_denominator_f32,
            target_logits,
        ),
        _,
    ) = jax.lax.scan(
        online_normalizer_and_target_scan,
        (
            initial_max,
            initial_denominator_f32,
            initial_target,
        ),
        (slice_starts, assigned_starts),
    )
    # This is the finite-max convention used by jax.nn.logsumexp.  A normal
    # finite input has a finite maximum; retaining the convention keeps the
    # semantic oracle well defined for exceptional values as well.
    finite_global_max = jnp.isfinite(global_max)
    finite_max = jnp.where(finite_global_max, global_max, 0)
    exceptional_denominator_f32 = jnp.where(
        jnp.isposinf(global_max),
        jnp.asarray(jnp.inf, dtype=jnp.float32),
        jnp.where(
            jnp.isneginf(global_max),
            jnp.asarray(0, dtype=jnp.float32),
            jnp.asarray(jnp.nan, dtype=jnp.float32),
        ),
    )
    denominator_f32 = jnp.where(
        finite_global_max,
        online_denominator_f32,
        exceptional_denominator_f32,
    )
    # JAX promotes BF16/F16 reductions to FP32 internally, then casts the result
    # back before log().  Retain FP32 across superblocks and perform that cast
    # exactly once after the online reduction.
    rounded_denominator = denominator_f32.astype(output_dtype)
    log_normalizer = jnp.log(jnp.abs(rounded_denominator)) + finite_max
    raw_logprobs = target_logits - log_normalizer

    nan = jnp.asarray(jnp.nan, dtype=output_dtype)
    active_logprobs = jnp.where(valid_targets, raw_logprobs, nan)
    logprobs = jnp.where(chunk_active, active_logprobs, jnp.zeros_like(active_logprobs))
    return logprobs, finite_max, rounded_denominator


def _split_forward_with_normalizer(
    hidden: jax.Array,
    embedding: jax.Array,
    target_ids: jax.Array,
    active_mask: jax.Array,
    token_chunk_size: int,
    vocab_superblock_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    token_shape = target_ids.shape
    token_count = math.prod(token_shape)
    hidden_size = hidden.shape[-1]
    num_chunks = (token_count + token_chunk_size - 1) // token_chunk_size
    padded_token_count = num_chunks * token_chunk_size
    token_padding = padded_token_count - token_count

    flat_hidden = hidden.reshape(token_count, hidden_size)
    flat_targets = target_ids.reshape(token_count)
    flat_active = active_mask.reshape(token_count)
    flat_hidden = jnp.pad(flat_hidden, ((0, token_padding), (0, 0)))
    flat_targets = jnp.pad(flat_targets, (0, token_padding))
    flat_active = jnp.pad(flat_active, (0, token_padding), constant_values=False)
    chunked_hidden = flat_hidden.reshape(num_chunks, token_chunk_size, hidden_size)
    chunked_targets = flat_targets.reshape(num_chunks, token_chunk_size)
    chunked_active = flat_active.reshape(num_chunks, token_chunk_size)

    block_width, slice_starts, assigned_starts = _vocab_block_descriptors(
        embedding.shape[0], vocab_superblock_size
    )

    def one_chunk(
        chunk_values: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        chunk_hidden, chunk_targets, chunk_active = chunk_values
        return _chunk_forward_with_normalizer(
            chunk_hidden,
            chunk_targets,
            chunk_active,
            embedding,
            block_width,
            slice_starts,
            assigned_starts,
            embedding.shape[0],
        )

    logprobs, finite_max, rounded_denominator = jax.lax.map(
        one_chunk, (chunked_hidden, chunked_targets, chunked_active)
    )

    def restore_token_shape(value: jax.Array) -> jax.Array:
        return value.reshape(padded_token_count)[:token_count].reshape(token_shape)

    return tuple(
        restore_token_shape(value)
        for value in (logprobs, finite_max, rounded_denominator)
    )


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _split_vocab_tied_target_logprobs(
    hidden: jax.Array,
    embedding: jax.Array,
    target_ids: jax.Array,
    active_mask: jax.Array,
    token_chunk_size: int,
    vocab_superblock_size: int,
) -> jax.Array:
    logprobs, _, _ = _split_forward_with_normalizer(
        hidden,
        embedding,
        target_ids,
        active_mask,
        token_chunk_size,
        vocab_superblock_size,
    )
    return logprobs


def _split_vocab_tied_target_logprobs_fwd(
    hidden: jax.Array,
    embedding: jax.Array,
    target_ids: jax.Array,
    active_mask: jax.Array,
    token_chunk_size: int,
    vocab_superblock_size: int,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    logprobs, finite_max, rounded_denominator = _split_forward_with_normalizer(
        hidden,
        embedding,
        target_ids,
        active_mask,
        token_chunk_size,
        vocab_superblock_size,
    )
    # Save only O(tokens) normalizer state.  In particular, neither logits nor
    # probabilities appear in this residual; backward recomputes one MxVB tile.
    residual = (
        hidden,
        embedding,
        target_ids,
        active_mask,
        finite_max,
        rounded_denominator,
    )
    return logprobs, residual


def _split_vocab_tied_target_logprobs_bwd(
    token_chunk_size: int,
    vocab_superblock_size: int,
    residual: tuple[jax.Array, ...],
    output_cotangent: jax.Array,
) -> tuple[jax.Array | None, ...]:
    hidden, embedding, target_ids, active_mask, finite_max, denominator = residual
    token_shape = target_ids.shape
    token_count = math.prod(token_shape)
    hidden_size = hidden.shape[-1]
    num_chunks = (token_count + token_chunk_size - 1) // token_chunk_size
    padded_token_count = num_chunks * token_chunk_size
    token_padding = padded_token_count - token_count

    def pad_and_chunk(value: jax.Array, trailing_shape: tuple[int, ...] = ()):
        flat = value.reshape((token_count, *trailing_shape))
        pad_width = ((0, token_padding),) + tuple((0, 0) for _ in trailing_shape)
        return jnp.pad(flat, pad_width).reshape(
            (num_chunks, token_chunk_size, *trailing_shape)
        )

    chunked_hidden = pad_and_chunk(hidden, (hidden_size,))
    chunked_targets = pad_and_chunk(target_ids)
    chunked_active = pad_and_chunk(active_mask)
    chunked_max = pad_and_chunk(finite_max)
    chunked_denominator = pad_and_chunk(denominator)
    chunked_cotangent = pad_and_chunk(output_cotangent)
    block_width, slice_starts, assigned_starts = _vocab_block_descriptors(
        embedding.shape[0], vocab_superblock_size
    )
    vocab_size = embedding.shape[0]

    def one_chunk_input_vjp(
        chunk_values: tuple[jax.Array, ...],
    ) -> jax.Array:
        (
            chunk_hidden,
            chunk_targets,
            chunk_active,
            chunk_max,
            chunk_denominator,
            chunk_cotangent,
        ) = chunk_values
        normalized_targets, valid_targets = _normalize_target_ids(
            chunk_targets, vocab_size
        )
        dy_f32 = jnp.where(
            chunk_active, chunk_cotangent, jnp.zeros_like(chunk_cotangent)
        ).astype(jnp.float32)

        safe_targets = jnp.clip(normalized_targets, 0, vocab_size - 1)
        target_embedding_f32 = embedding[safe_targets].astype(jnp.float32)
        target_embedding_f32 = jnp.where(
            valid_targets[:, None],
            target_embedding_f32,
            jnp.zeros_like(target_embedding_f32),
        )

        def expected_embedding_scan(
            expected_f32: jax.Array,
            block_values: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, None]:
            slice_start, assigned_start = block_values
            embedding_block = jax.lax.dynamic_slice_in_dim(
                embedding, slice_start, block_width, axis=0
            )
            block_logits = _masked_block_logits(
                chunk_hidden,
                embedding_block,
                slice_start,
                assigned_start,
                vocab_size,
            )
            shifted_exponentials = jnp.exp(block_logits - chunk_max[:, None])
            probabilities_f32 = (
                shifted_exponentials.astype(jnp.float32)
                / (chunk_denominator.astype(jnp.float32)[:, None])
            )
            block_expected_f32 = probabilities_f32 @ embedding_block.astype(jnp.float32)
            return expected_f32 + block_expected_f32, None

        expected_embedding_f32, _ = jax.lax.scan(
            expected_embedding_scan,
            jnp.zeros((token_chunk_size, hidden_size), dtype=jnp.float32),
            (slice_starts, assigned_starts),
        )
        # An out-of-range ``take_along_axis`` gather has zero transpose, while
        # the separately evaluated logsumexp still contributes its negative
        # softmax cotangent.  Keeping that normalizer term is required for this
        # custom rule to remain the VJP of the dense equation even when the
        # observable primal result is NaN.
        dx_f32 = dy_f32[:, None] * (target_embedding_f32 - expected_embedding_f32)
        # Multiplying a zero cotangent by an exceptional softmax value still
        # produces NaN.  Predicate after the arithmetic as well so inactive and
        # padded rows uphold the public zero-dHidden contract for every input.
        dx_f32 = jnp.where(
            chunk_active[:, None],
            dx_f32,
            jnp.zeros_like(dx_f32),
        )
        return dx_f32.astype(hidden.dtype)

    chunked_dx = jax.lax.map(
        one_chunk_input_vjp,
        (
            chunked_hidden,
            chunked_targets,
            chunked_active,
            chunked_max,
            chunked_denominator,
            chunked_cotangent,
        ),
    )
    dx = chunked_dx.reshape(padded_token_count, hidden_size)[:token_count]
    dx = dx.reshape(hidden.shape)
    # ``None`` is a symbolic zero cotangent.  In particular the tied embedding
    # is frozen and no VxH dEmbedding allocation is requested from this VJP.
    return dx, None, None, None


_split_vocab_tied_target_logprobs.defvjp(
    _split_vocab_tied_target_logprobs_fwd,
    _split_vocab_tied_target_logprobs_bwd,
)


def split_vocab_tied_target_logprobs(
    hidden: jax.Array,
    embedding: jax.Array,
    target_ids: jax.Array,
    *,
    token_chunk_size: int = 128,
    vocab_superblock_size: int = 4096,
    active_mask: jax.Array | None = None,
) -> jax.Array:
    """Compute tied target logprobs without a logical token-by-vocab buffer.

    Args:
        hidden: Floating hidden states with shape ``[..., H]``.
        embedding: Frozen tied embedding with shape ``[V, H]`` and the same
            dtype as ``hidden``.
        target_ids: Integer target token IDs with shape ``hidden.shape[:-1]``.
        token_chunk_size: Static token tile, restricted to 64, 128, or 256.
        vocab_superblock_size: Static maximum vocabulary rows per tile.  Set it
            below ``V`` to obtain a genuinely split-vocabulary lowering.
        active_mask: Optional boolean study mask.  Inactive outputs and their
            ``dHidden`` rows are zero.  Leave this as ``None`` to match the
            current unmasked ``LogitsProcessorMixin.compute_logprobs`` contract.

    Returns:
        Target logprobs with shape ``hidden.shape[:-1]`` and hidden dtype.

    Notes:
        The output embedding is frozen by the explicit custom VJP. Qwen3.5's
        experimental model wiring is default-off and makes no performance
        claim.
    """
    if active_mask is None:
        active_mask = jnp.ones(target_ids.shape, dtype=jnp.bool_)
    _validate_inputs(
        hidden,
        embedding,
        target_ids,
        active_mask,
        token_chunk_size,
        vocab_superblock_size,
    )
    return _split_vocab_tied_target_logprobs(
        hidden,
        embedding,
        target_ids,
        active_mask,
        token_chunk_size,
        vocab_superblock_size,
    )


def compact_masked_tied_target_logprobs_study(
    hidden: jax.Array,
    embedding: jax.Array,
    target_ids: jax.Array,
    loss_mask: jax.Array,
    *,
    active_token_capacity: int,
    token_chunk_size: int = 128,
    vocab_superblock_size: int = 4096,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Study fixed-capacity active-token compaction before the tied head.

    This helper gathers only rows whose loss mask is nonzero, evaluates a
    fixed-capacity compact buffer, then scatters logprobs back with zeros at
    inactive positions.  The returned tuple is ``(logprobs, active_count,
    overflow)``.  Callers must reject ``overflow``; compiled JAX cannot vary
    the compact buffer shape with the runtime mask.

    This is intentionally separate from the main oracle because SkyRL
    currently exposes target logprobs even at loss-masked positions.  Replacing
    those auxiliary values by zero is loss/VJP-equivalent after
    ``safe_loss_mask``, but it is not the same observable forward contract.
    """
    if active_token_capacity <= 0:
        raise ValueError(
            f"active_token_capacity must be positive, got {active_token_capacity}"
        )
    if loss_mask.shape != target_ids.shape:
        raise ValueError(
            f"loss_mask shape {loss_mask.shape} must match target shape {target_ids.shape}"
        )
    if not (
        jnp.issubdtype(loss_mask.dtype, jnp.bool_)
        or jnp.issubdtype(loss_mask.dtype, jnp.number)
    ):
        raise TypeError(f"loss_mask must be boolean or numeric, got {loss_mask.dtype}")

    active = loss_mask != 0
    token_count = math.prod(target_ids.shape)
    hidden_size = hidden.shape[-1]
    flat_active = active.reshape(token_count)
    active_count = jnp.sum(flat_active, dtype=jnp.int32)
    active_indices = jnp.nonzero(flat_active, size=active_token_capacity, fill_value=0)[
        0
    ]
    valid_slots = jnp.arange(active_token_capacity, dtype=jnp.int32) < active_count
    compact_hidden = hidden.reshape(token_count, hidden_size)[active_indices]
    compact_targets = target_ids.reshape(token_count)[active_indices]
    compact_logprobs = split_vocab_tied_target_logprobs(
        compact_hidden,
        embedding,
        compact_targets,
        token_chunk_size=token_chunk_size,
        vocab_superblock_size=vocab_superblock_size,
        active_mask=valid_slots,
    )
    compact_logprobs = jnp.where(
        valid_slots, compact_logprobs, jnp.zeros_like(compact_logprobs)
    )
    flat_logprobs = jnp.zeros((token_count,), dtype=compact_logprobs.dtype)
    flat_logprobs = flat_logprobs.at[active_indices].add(compact_logprobs)
    overflow = active_count > active_token_capacity
    return flat_logprobs.reshape(target_ids.shape), active_count, overflow
