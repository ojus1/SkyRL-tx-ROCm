from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyrl.tx.kernels.tied_logprob import (
    _split_vocab_tied_target_logprobs_fwd,
    compact_masked_tied_target_logprobs_study,
    dense_tied_target_logprobs_reference,
    split_vocab_tied_target_logprobs,
)


def _independent_dense_equations(
    hidden: jax.Array, embedding: jax.Array, target_ids: jax.Array
) -> jax.Array:
    """Directly reproduce LogitsProcessorMixin without using the oracle."""
    logits = hidden @ embedding.T
    log_sum_exp = jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    target_logits = jnp.take_along_axis(logits, target_ids[..., None], axis=-1)
    return (target_logits - log_sum_exp).squeeze(-1)


def _inputs(
    token_shape: tuple[int, ...],
    *,
    hidden_size: int = 9,
    vocab_size: int = 37,
    dtype: jnp.dtype = jnp.float32,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    hidden = (
        jax.random.normal(
            jax.random.key(101), (*token_shape, hidden_size), dtype=jnp.float32
        )
        * 0.2
    ).astype(dtype)
    embedding = (
        jax.random.normal(
            jax.random.key(102), (vocab_size, hidden_size), dtype=jnp.float32
        )
        * 0.2
    ).astype(dtype)
    target_ids = jax.random.randint(
        jax.random.key(103), token_shape, 0, vocab_size, dtype=jnp.int32
    )
    return hidden, embedding, target_ids


@pytest.mark.parametrize("token_chunk_size", [64, 128, 256])
def test_forward_matches_independent_target_gather_and_logsumexp_equations(
    token_chunk_size: int,
) -> None:
    hidden, embedding, target_ids = _inputs((token_chunk_size + 3,))
    actual = split_vocab_tied_target_logprobs(
        hidden,
        embedding,
        target_ids,
        token_chunk_size=token_chunk_size,
        vocab_superblock_size=11,
    )
    expected = _independent_dense_equations(hidden, embedding, target_ids)

    assert actual.shape == target_ids.shape
    assert actual.dtype == hidden.dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_forward_handles_batch_shape_and_both_padding_boundaries() -> None:
    hidden, embedding, target_ids = _inputs((2, 35), hidden_size=7, vocab_size=41)
    actual = jax.jit(
        lambda h, e, t: split_vocab_tied_target_logprobs(
            h,
            e,
            t,
            token_chunk_size=64,
            vocab_superblock_size=13,
        )
    )(hidden, embedding, target_ids)
    expected = _independent_dense_equations(hidden, embedding, target_ids)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_overlapping_tail_slice_assigns_every_vocabulary_row_exactly_once() -> None:
    vocab_size = 37
    hidden, embedding, _ = _inputs((vocab_size,), hidden_size=7, vocab_size=vocab_size)
    target_ids = jnp.arange(vocab_size, dtype=jnp.int32)
    cotangent = jax.random.normal(jax.random.key(107), target_ids.shape)

    def split_objective(value: jax.Array) -> jax.Array:
        return jnp.sum(
            split_vocab_tied_target_logprobs(
                value,
                embedding,
                target_ids,
                token_chunk_size=64,
                vocab_superblock_size=16,
            )
            * cotangent
        )

    def dense_objective(value: jax.Array) -> jax.Array:
        return jnp.sum(
            _independent_dense_equations(value, embedding, target_ids) * cotangent
        )

    np.testing.assert_allclose(
        split_vocab_tied_target_logprobs(
            hidden,
            embedding,
            target_ids,
            token_chunk_size=64,
            vocab_superblock_size=16,
        ),
        _independent_dense_equations(hidden, embedding, target_ids),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        jax.grad(split_objective)(hidden),
        jax.grad(dense_objective)(hidden),
        rtol=3e-6,
        atol=3e-6,
    )


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_low_precision_forward_tracks_current_dense_equation(dtype: jnp.dtype) -> None:
    hidden, embedding, target_ids = _inputs(
        (67,), hidden_size=13, vocab_size=97, dtype=dtype
    )
    actual = split_vocab_tied_target_logprobs(
        hidden,
        embedding,
        target_ids,
        token_chunk_size=64,
        vocab_superblock_size=31,
    )
    expected = _independent_dense_equations(hidden, embedding, target_ids)

    # Splitting changes only FP32 reduction association.  BF16's final rounded
    # denominator can consequently move by one ULP; the equations and dtype
    # are otherwise the same as the current dense path.
    tolerance = 0.032 if dtype == jnp.bfloat16 else 0.002
    np.testing.assert_allclose(
        actual.astype(jnp.float32),
        expected.astype(jnp.float32),
        rtol=tolerance,
        atol=tolerance,
    )


def test_online_normalizer_rescales_when_later_blocks_raise_the_maximum() -> None:
    hidden = jnp.ones((3, 1), dtype=jnp.float32)
    embedding = jnp.arange(8, dtype=jnp.float32)[:, None]
    target_ids = jnp.asarray([0, 3, 7], dtype=jnp.int32)
    active_mask = jnp.ones(target_ids.shape, dtype=jnp.bool_)

    output, residual = _split_vocab_tied_target_logprobs_fwd(
        hidden,
        embedding,
        target_ids,
        active_mask,
        64,
        2,
    )
    expected = _independent_dense_equations(hidden, embedding, target_ids)
    np.testing.assert_allclose(output, expected, rtol=2e-6, atol=2e-6)

    finite_max, denominator = residual[-2:]
    expected_max = jnp.full_like(finite_max, 7.0)
    expected_denominator = jnp.full_like(
        denominator,
        jnp.sum(jnp.exp(jnp.arange(8, dtype=jnp.float32) - 7.0)),
    )
    np.testing.assert_array_equal(finite_max, expected_max)
    np.testing.assert_allclose(denominator, expected_denominator, rtol=1e-6)


@pytest.mark.parametrize(
    ("embedding_value", "expected_denominator"),
    [(-np.inf, 0.0), (np.inf, np.inf), (np.nan, np.nan)],
)
def test_online_normalizer_preserves_nonfinite_logsumexp_convention(
    embedding_value: float,
    expected_denominator: float,
) -> None:
    hidden = jnp.ones((3, 1), dtype=jnp.float32)
    embedding = jnp.full((5, 1), embedding_value, dtype=jnp.float32)
    target_ids = jnp.asarray([0, 2, 4], dtype=jnp.int32)
    active_mask = jnp.ones(target_ids.shape, dtype=jnp.bool_)

    output, residual = _split_vocab_tied_target_logprobs_fwd(
        hidden,
        embedding,
        target_ids,
        active_mask,
        64,
        2,
    )
    expected = _independent_dense_equations(hidden, embedding, target_ids)
    np.testing.assert_allclose(output, expected, equal_nan=True)

    finite_max, denominator = residual[-2:]
    np.testing.assert_array_equal(finite_max, jnp.zeros_like(finite_max))
    if np.isnan(expected_denominator):
        assert bool(jnp.all(jnp.isnan(denominator)))
    else:
        np.testing.assert_array_equal(
            denominator,
            jnp.full_like(denominator, expected_denominator),
        )


@pytest.mark.parametrize("embedding_value", [-np.inf, np.inf, np.nan])
def test_mixed_inactive_exceptional_rows_keep_zero_dhidden(
    embedding_value: float,
) -> None:
    hidden = jnp.ones((3, 1), dtype=jnp.float32)
    embedding = jnp.full((5, 1), embedding_value, dtype=jnp.float32)
    target_ids = jnp.asarray([0, 2, 4], dtype=jnp.int32)
    active_mask = jnp.asarray([True, False, False], dtype=jnp.bool_)

    def objective(value: jax.Array) -> jax.Array:
        return jnp.sum(
            split_vocab_tied_target_logprobs(
                value,
                embedding,
                target_ids,
                token_chunk_size=64,
                vocab_superblock_size=2,
                active_mask=active_mask,
            )
        )

    output = split_vocab_tied_target_logprobs(
        hidden,
        embedding,
        target_ids,
        token_chunk_size=64,
        vocab_superblock_size=2,
        active_mask=active_mask,
    )
    gradient = jax.grad(objective)(hidden)
    assert bool(jnp.isnan(output[0]))
    np.testing.assert_array_equal(output[~active_mask], jnp.zeros_like(output[1:]))
    np.testing.assert_array_equal(
        gradient[~active_mask],
        jnp.zeros_like(gradient[1:]),
    )


def test_custom_vjp_dhidden_matches_independent_dense_vjp() -> None:
    hidden, embedding, target_ids = _inputs((2, 37), hidden_size=11, vocab_size=53)
    cotangent = jax.random.normal(jax.random.key(104), target_ids.shape)

    def split_objective(hidden_arg: jax.Array) -> jax.Array:
        logprobs = split_vocab_tied_target_logprobs(
            hidden_arg,
            embedding,
            target_ids,
            token_chunk_size=64,
            vocab_superblock_size=17,
        )
        return jnp.sum(logprobs * cotangent)

    def dense_objective(hidden_arg: jax.Array) -> jax.Array:
        return jnp.sum(
            _independent_dense_equations(hidden_arg, embedding, target_ids) * cotangent
        )

    actual = jax.jit(jax.grad(split_objective))(hidden)
    expected = jax.jit(jax.grad(dense_objective))(hidden)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_low_precision_custom_vjp_tracks_current_dense_dhidden(
    dtype: jnp.dtype,
) -> None:
    hidden, embedding, target_ids = _inputs(
        (67,), hidden_size=13, vocab_size=97, dtype=dtype
    )
    cotangent = jax.random.normal(
        jax.random.key(106), target_ids.shape, dtype=jnp.float32
    )

    actual = jax.grad(
        lambda value: jnp.sum(
            split_vocab_tied_target_logprobs(
                value,
                embedding,
                target_ids,
                token_chunk_size=64,
                vocab_superblock_size=31,
            ).astype(jnp.float32)
            * cotangent
        )
    )(hidden)
    expected = jax.grad(
        lambda value: jnp.sum(
            _independent_dense_equations(value, embedding, target_ids).astype(
                jnp.float32
            )
            * cotangent
        )
    )(hidden)

    tolerance = 0.016 if dtype == jnp.bfloat16 else 0.002
    np.testing.assert_allclose(
        actual.astype(jnp.float32),
        expected.astype(jnp.float32),
        rtol=tolerance,
        atol=tolerance,
    )


def test_custom_vjp_freezes_tied_embedding_and_keeps_nonzero_dhidden() -> None:
    hidden, embedding, target_ids = _inputs((23,), hidden_size=8, vocab_size=29)

    def objective(hidden_arg: jax.Array, embedding_arg: jax.Array) -> jax.Array:
        output = split_vocab_tied_target_logprobs(
            hidden_arg,
            embedding_arg,
            target_ids,
            token_chunk_size=64,
            vocab_superblock_size=7,
        )
        return jnp.sum(output**2)

    dhidden, dembedding = jax.grad(objective, argnums=(0, 1))(hidden, embedding)
    assert float(jnp.linalg.norm(dhidden)) > 0
    np.testing.assert_array_equal(dembedding, jnp.zeros_like(embedding))

    dense_dembedding = jax.grad(
        lambda weight: jnp.sum(
            _independent_dense_equations(hidden, weight, target_ids) ** 2
        )
    )(embedding)
    assert float(jnp.linalg.norm(dense_dembedding)) > 0


def test_custom_vjp_residual_contains_only_token_state_not_logits() -> None:
    hidden, embedding, target_ids = _inputs((2, 33), hidden_size=7, vocab_size=43)
    active_mask = jnp.ones_like(target_ids, dtype=jnp.bool_)
    output, residual = _split_vocab_tied_target_logprobs_fwd(
        hidden,
        embedding,
        target_ids,
        active_mask,
        64,
        13,
    )

    assert output.shape == target_ids.shape
    residual_shapes = [value.shape for value in jax.tree.leaves(residual)]
    assert target_ids.shape in residual_shapes
    assert (*target_ids.shape, embedding.shape[0]) not in residual_shapes
    assert (64, 13) not in residual_shapes


def test_active_mask_zeroes_inactive_outputs_and_dhidden_rows() -> None:
    hidden, embedding, target_ids = _inputs((19,), hidden_size=7, vocab_size=31)
    active_mask = jnp.asarray(
        [True, False, True, True, False, False, True, False, True] * 2 + [False]
    )
    cotangent = jax.random.normal(jax.random.key(105), target_ids.shape)

    def split_objective(hidden_arg: jax.Array) -> jax.Array:
        output = split_vocab_tied_target_logprobs(
            hidden_arg,
            embedding,
            target_ids,
            token_chunk_size=64,
            vocab_superblock_size=9,
            active_mask=active_mask,
        )
        return jnp.sum(output * cotangent)

    actual_output = split_vocab_tied_target_logprobs(
        hidden,
        embedding,
        target_ids,
        token_chunk_size=64,
        vocab_superblock_size=9,
        active_mask=active_mask,
    )
    dense_output = _independent_dense_equations(hidden, embedding, target_ids)
    np.testing.assert_allclose(actual_output[active_mask], dense_output[active_mask])
    np.testing.assert_array_equal(
        actual_output[~active_mask], jnp.zeros_like(actual_output[~active_mask])
    )

    actual_gradient = jax.grad(split_objective)(hidden)
    expected_gradient = jax.grad(
        lambda value: jnp.sum(
            _independent_dense_equations(value, embedding, target_ids)
            * jnp.where(active_mask, cotangent, 0)
        )
    )(hidden)
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=3e-6, atol=3e-6)
    np.testing.assert_array_equal(
        actual_gradient[~active_mask], jnp.zeros_like(actual_gradient[~active_mask])
    )


def test_target_gather_matches_negative_index_and_out_of_bounds_semantics() -> None:
    hidden, embedding, _ = _inputs((6,), hidden_size=5, vocab_size=17)
    target_ids = jnp.asarray([-17, -1, -18, 17, 0, 16], dtype=jnp.int32)
    active_mask = jnp.asarray([True, True, True, True, False, True])
    output = split_vocab_tied_target_logprobs(
        hidden,
        embedding,
        target_ids,
        token_chunk_size=64,
        vocab_superblock_size=8,
        active_mask=active_mask,
    )
    dense_output = _independent_dense_equations(hidden, embedding, target_ids)
    valid_rows = jnp.asarray([0, 1, 5], dtype=jnp.int32)
    np.testing.assert_allclose(output[valid_rows], dense_output[valid_rows])
    assert output[4] == 0
    assert bool(jnp.isnan(output[2]))
    assert bool(jnp.isnan(output[3]))


@pytest.mark.parametrize("invalid_target", [-18, 17])
def test_active_out_of_bounds_target_vjp_keeps_logsumexp_cotangent(
    invalid_target: int,
) -> None:
    hidden, embedding, _ = _inputs((3,), hidden_size=5, vocab_size=17)
    target_ids = jnp.asarray([invalid_target, -1, -17], dtype=jnp.int32)
    cotangent = jnp.asarray([0.75, -0.2, 0.4], dtype=jnp.float32)

    def split_objective(value: jax.Array) -> jax.Array:
        return jnp.sum(
            split_vocab_tied_target_logprobs(
                value,
                embedding,
                target_ids,
                token_chunk_size=64,
                vocab_superblock_size=8,
            )
            * cotangent
        )

    def dense_objective(value: jax.Array) -> jax.Array:
        return jnp.sum(
            _independent_dense_equations(value, embedding, target_ids) * cotangent
        )

    actual = jax.grad(split_objective)(hidden)
    expected = jax.grad(dense_objective)(hidden)
    assert float(jnp.linalg.norm(expected[0])) > 0
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


def test_compact_active_study_is_masked_loss_and_vjp_equivalent() -> None:
    hidden, embedding, target_ids = _inputs((2, 11), hidden_size=7, vocab_size=31)
    loss_mask = jnp.asarray(
        [
            [0.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.25, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.25, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    def compact_objective(hidden_arg: jax.Array) -> jax.Array:
        output, active_count, overflow = compact_masked_tied_target_logprobs_study(
            hidden_arg,
            embedding,
            target_ids,
            loss_mask,
            active_token_capacity=12,
            token_chunk_size=64,
            vocab_superblock_size=9,
        )
        assert active_count.shape == ()
        assert overflow.shape == ()
        return -jnp.sum(jnp.where(loss_mask != 0, loss_mask * output, 0))

    output, active_count, overflow = compact_masked_tied_target_logprobs_study(
        hidden,
        embedding,
        target_ids,
        loss_mask,
        active_token_capacity=12,
        token_chunk_size=64,
        vocab_superblock_size=9,
    )
    dense_output = _independent_dense_equations(hidden, embedding, target_ids)
    active = loss_mask != 0
    assert int(active_count) == int(jnp.sum(active))
    assert not bool(overflow)
    np.testing.assert_allclose(output[active], dense_output[active])
    np.testing.assert_array_equal(output[~active], jnp.zeros_like(output[~active]))

    actual_gradient = jax.grad(compact_objective)(hidden)
    expected_gradient = jax.grad(
        lambda value: (
            -jnp.sum(
                jnp.where(
                    active,
                    loss_mask
                    * _independent_dense_equations(value, embedding, target_ids),
                    0,
                )
            )
        )
    )(hidden)
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=3e-6, atol=3e-6)


def test_all_inactive_compaction_fill_slots_keep_zero_exceptional_vjp() -> None:
    hidden = jnp.ones((5, 1), dtype=jnp.float32)
    embedding = jnp.full((7, 1), jnp.nan, dtype=jnp.float32)
    target_ids = jnp.arange(5, dtype=jnp.int32)
    loss_mask = jnp.zeros(target_ids.shape, dtype=jnp.float32)

    def objective(value: jax.Array) -> jax.Array:
        output, _, _ = compact_masked_tied_target_logprobs_study(
            value,
            embedding,
            target_ids,
            loss_mask,
            active_token_capacity=4,
            token_chunk_size=64,
            vocab_superblock_size=3,
        )
        return jnp.sum(output)

    output, active_count, overflow = compact_masked_tied_target_logprobs_study(
        hidden,
        embedding,
        target_ids,
        loss_mask,
        active_token_capacity=4,
        token_chunk_size=64,
        vocab_superblock_size=3,
    )
    np.testing.assert_array_equal(output, jnp.zeros_like(output))
    assert int(active_count) == 0
    assert not bool(overflow)
    np.testing.assert_array_equal(jax.grad(objective)(hidden), jnp.zeros_like(hidden))


def test_compact_active_study_reports_fixed_capacity_overflow() -> None:
    hidden, embedding, target_ids = _inputs((6,), hidden_size=5, vocab_size=19)
    _, active_count, overflow = compact_masked_tied_target_logprobs_study(
        hidden,
        embedding,
        target_ids,
        jnp.asarray([1, 1, 0, 1, 0, 1], dtype=jnp.int32),
        active_token_capacity=3,
        token_chunk_size=64,
        vocab_superblock_size=8,
    )
    assert int(active_count) == 4
    assert bool(overflow)


def _has_floating_tensor_shape(stablehlo: str, shape: tuple[int, ...]) -> bool:
    dimensions = "x".join(str(dimension) for dimension in shape)
    return re.search(rf"tensor<{dimensions}x(?:bf16|f16|f32)>", stablehlo) is not None


def _has_typed_tensor_shape(stablehlo: str, shape: tuple[int, ...], dtype: str) -> bool:
    dimensions = "x".join(str(dimension) for dimension in shape)
    return f"tensor<{dimensions}x{dtype}>" in stablehlo


def _stablehlo_op_count(stablehlo: str, operation: str) -> int:
    return len(
        re.findall(
            rf"(?m)^\s*%[A-Za-z0-9_]+\s*=\s*stablehlo\.{re.escape(operation)}\b",
            stablehlo,
        )
    )


def test_stablehlo_forward_and_vjp_use_tiles_not_full_logits() -> None:
    token_count, hidden_size, vocab_size = 65, 7, 97
    token_chunk_size, vocab_superblock_size = 64, 32
    hidden_spec = jax.ShapeDtypeStruct((token_count, hidden_size), jnp.float32)
    embedding_spec = jax.ShapeDtypeStruct((vocab_size, hidden_size), jnp.float32)
    target_spec = jax.ShapeDtypeStruct((token_count,), jnp.int32)

    dense_hlo = str(
        jax.jit(_independent_dense_equations)
        .lower(hidden_spec, embedding_spec, target_spec)
        .compiler_ir(dialect="stablehlo")
    )
    assert _has_floating_tensor_shape(dense_hlo, (token_count, vocab_size))

    def split_forward(h, e, t):
        return split_vocab_tied_target_logprobs(
            h,
            e,
            t,
            token_chunk_size=token_chunk_size,
            vocab_superblock_size=vocab_superblock_size,
        )

    split_hlo = str(
        jax.jit(split_forward)
        .lower(hidden_spec, embedding_spec, target_spec)
        .compiler_ir(dialect="stablehlo")
    )
    value_and_dhidden_hlo = str(
        jax.jit(
            jax.value_and_grad(
                lambda h, e, t: jnp.sum(split_forward(h, e, t)), argnums=0
            )
        )
        .lower(hidden_spec, embedding_spec, target_spec)
        .compiler_ir(dialect="stablehlo")
    )

    for stablehlo in (split_hlo, value_and_dhidden_hlo):
        assert _has_floating_tensor_shape(
            stablehlo, (token_chunk_size, vocab_superblock_size)
        )
        assert not _has_floating_tensor_shape(stablehlo, (token_count, vocab_size))
        assert not _has_floating_tensor_shape(stablehlo, (token_chunk_size, vocab_size))
    # For this pinned small FP32 lowering, the online scan has one syntactic
    # forward dot body. Value-plus-dHidden has three dot bodies rather than the
    # former four. This is an IR-structure guard, not a physical-read claim.
    assert _stablehlo_op_count(split_hlo, "dot_general") == 1
    assert _stablehlo_op_count(value_and_dhidden_hlo, "dot_general") == 3


def test_stablehlo_exact_qwen35_geometry_has_bounded_logit_tiles() -> None:
    token_chunk_size = 256
    hidden_size = 2560
    vocab_size = 248_320
    vocab_superblock_size = 4096
    hidden_spec = jax.ShapeDtypeStruct((token_chunk_size, hidden_size), jnp.bfloat16)
    embedding_spec = jax.ShapeDtypeStruct((vocab_size, hidden_size), jnp.bfloat16)
    target_spec = jax.ShapeDtypeStruct((token_chunk_size,), jnp.int32)

    def split_forward(h, e, t):
        return split_vocab_tied_target_logprobs(
            h,
            e,
            t,
            token_chunk_size=token_chunk_size,
            vocab_superblock_size=vocab_superblock_size,
        )

    forward_hlo = str(
        jax.jit(split_forward)
        .lower(hidden_spec, embedding_spec, target_spec)
        .compiler_ir(dialect="stablehlo")
    )
    value_and_dhidden_hlo = str(
        jax.jit(
            jax.value_and_grad(
                lambda h, e, t: jnp.sum(split_forward(h, e, t).astype(jnp.float32)),
                argnums=0,
            )
        )
        .lower(hidden_spec, embedding_spec, target_spec)
        .compiler_ir(dialect="stablehlo")
    )

    for stablehlo in (forward_hlo, value_and_dhidden_hlo):
        assert _has_typed_tensor_shape(
            stablehlo, (token_chunk_size, vocab_superblock_size), "bf16"
        )
        assert not _has_floating_tensor_shape(stablehlo, (token_chunk_size, vocab_size))
        assert not _has_floating_tensor_shape(stablehlo, (249_856, hidden_size))
    assert _has_typed_tensor_shape(
        value_and_dhidden_hlo, (token_chunk_size, hidden_size), "f32"
    )
    assert _stablehlo_op_count(forward_hlo, "dot_general") == 1
    assert _stablehlo_op_count(value_and_dhidden_hlo, "dot_general") == 3


def test_dense_convenience_reference_matches_independent_equations() -> None:
    hidden, embedding, target_ids = _inputs((3, 5))
    np.testing.assert_array_equal(
        dense_tied_target_logprobs_reference(hidden, embedding, target_ids),
        _independent_dense_equations(hidden, embedding, target_ids),
    )


def test_public_validation_rejects_ambiguous_or_unsupported_inputs() -> None:
    hidden, embedding, target_ids = _inputs((5,), hidden_size=7, vocab_size=13)
    with pytest.raises(ValueError, match="token_chunk_size must be one of"):
        split_vocab_tied_target_logprobs(
            hidden,
            embedding,
            target_ids,
            token_chunk_size=32,
            vocab_superblock_size=8,
        )
    with pytest.raises(ValueError, match="vocab_superblock_size must be positive"):
        split_vocab_tied_target_logprobs(
            hidden,
            embedding,
            target_ids,
            token_chunk_size=64,
            vocab_superblock_size=0,
        )
    with pytest.raises(ValueError, match="target shape"):
        split_vocab_tied_target_logprobs(
            hidden,
            embedding,
            target_ids[None],
            token_chunk_size=64,
            vocab_superblock_size=8,
        )
    with pytest.raises(TypeError, match="target_ids must be integral"):
        split_vocab_tied_target_logprobs(
            hidden,
            embedding,
            target_ids.astype(jnp.float32),
            token_chunk_size=64,
            vocab_superblock_size=8,
        )
    with pytest.raises(TypeError, match="active_mask must be boolean"):
        split_vocab_tied_target_logprobs(
            hidden,
            embedding,
            target_ids,
            token_chunk_size=64,
            vocab_superblock_size=8,
            active_mask=jnp.ones_like(target_ids),
        )
    with pytest.raises(TypeError, match="dtypes must match"):
        split_vocab_tied_target_logprobs(
            hidden.astype(jnp.bfloat16),
            embedding,
            target_ids,
            token_chunk_size=64,
            vocab_superblock_size=8,
        )
