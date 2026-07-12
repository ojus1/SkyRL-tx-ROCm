from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyrl.tx.kernels.query_bounded_gqa import (
    query_bounded_gqa,
    qwen35_32k_dispatch_plan,
)


def _inputs(dtype=jnp.float32, *, valid_length=13):
    q = jax.random.normal(jax.random.key(0), (1, 16, 4, 8), dtype=dtype)
    k = jax.random.normal(jax.random.key(1), (1, 16, 2, 8), dtype=dtype)
    v = jax.random.normal(jax.random.key(2), (1, 16, 2, 8), dtype=dtype)
    mask = (jnp.arange(16)[None, :] < valid_length).astype(jnp.int32)
    return q, k, v, mask


def _prototype(q, k, v, mask):
    return query_bounded_gqa(
        q,
        k,
        v,
        mask,
        query_chunk_size=8,
        block_q=4,
        block_k=4,
        backward_block_q=4,
        backward_block_k=4,
        interpret=True,
    )


def _reference(q, k, v, mask):
    return jax.nn.dot_product_attention(
        q,
        k,
        v,
        mask=mask[:, None, None, :].astype(bool),
        scale=8**-0.5,
        is_causal=True,
        implementation="xla",
    )


@pytest.mark.parametrize("valid_length", [1, 7, 13, 16])
def test_interpret_forward_and_vjp_match_portable_gqa(valid_length):
    q, k, v, mask = _inputs(valid_length=valid_length)
    dout = jax.random.normal(jax.random.key(3), q.shape)

    actual, actual_pullback = jax.vjp(lambda *args: _prototype(*args, mask), q, k, v)
    expected, expected_pullback = jax.vjp(
        lambda *args: _reference(*args, mask), q, k, v
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    for actual_gradient, expected_gradient in zip(
        actual_pullback(dout), expected_pullback(dout), strict=True
    ):
        np.testing.assert_allclose(
            actual_gradient, expected_gradient, rtol=3e-6, atol=3e-6
        )


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (jnp.bfloat16, 2e-2, 2e-2),
        (jnp.float16, 3e-3, 3e-3),
    ],
)
def test_low_precision_forward_and_vjp_match_dtype_semantics(dtype, rtol, atol):
    q, k, v, mask = _inputs(dtype)
    dout = jax.random.normal(jax.random.key(3), q.shape, dtype=dtype)
    actual, actual_pullback = jax.vjp(lambda *args: _prototype(*args, mask), q, k, v)
    expected, expected_pullback = jax.vjp(
        lambda *args: _reference(*args, mask), q, k, v
    )

    np.testing.assert_allclose(
        actual.astype(jnp.float32),
        expected.astype(jnp.float32),
        rtol=rtol,
        atol=atol,
    )
    for actual_gradient, expected_gradient in zip(
        actual_pullback(dout), expected_pullback(dout), strict=True
    ):
        np.testing.assert_allclose(
            actual_gradient.astype(jnp.float32),
            expected_gradient.astype(jnp.float32),
            rtol=rtol,
            atol=atol,
        )


def test_exact_qwen_head_geometry_and_dtype_forward_and_vjp():
    dtype = jnp.bfloat16
    q = jax.random.normal(jax.random.key(10), (1, 8, 16, 256), dtype=dtype)
    k = jax.random.normal(jax.random.key(11), (1, 8, 4, 256), dtype=dtype)
    v = jax.random.normal(jax.random.key(12), (1, 8, 4, 256), dtype=dtype)
    mask = (jnp.arange(8)[None, :] < 7).astype(jnp.int32)
    dout = jax.random.normal(jax.random.key(13), q.shape, dtype=dtype)

    def actual(q_arg, k_arg, v_arg):
        return query_bounded_gqa(
            q_arg,
            k_arg,
            v_arg,
            mask,
            query_chunk_size=4,
            block_q=4,
            block_k=4,
            backward_block_q=4,
            backward_block_k=4,
            interpret=True,
        )

    def expected(q_arg, k_arg, v_arg):
        return jax.nn.dot_product_attention(
            q_arg,
            k_arg,
            v_arg,
            mask=mask[:, None, None, :].astype(bool),
            scale=256**-0.5,
            is_causal=True,
            implementation="xla",
        )

    actual_output, actual_pullback = jax.vjp(actual, q, k, v)
    expected_output, expected_pullback = jax.vjp(expected, q, k, v)
    np.testing.assert_allclose(
        actual_output.astype(jnp.float32),
        expected_output.astype(jnp.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    for actual_gradient, expected_gradient in zip(
        actual_pullback(dout), expected_pullback(dout), strict=True
    ):
        np.testing.assert_allclose(
            actual_gradient.astype(jnp.float32),
            expected_gradient.astype(jnp.float32),
            rtol=2e-2,
            atol=2e-2,
        )


def test_later_query_chunk_uses_global_causal_offset_and_native_head_mapping():
    # Zero logits make each row an exact prefix average.  The second query
    # chunk must see keys from the first one; Q heads 0/1 map to KV head 0 and
    # Q heads 2/3 map to KV head 1 without physically repeating K or V.
    q = jnp.zeros((1, 16, 4, 8), dtype=jnp.float32)
    k = jnp.zeros((1, 16, 2, 8), dtype=jnp.float32)
    positions = jnp.arange(16, dtype=jnp.float32)
    v = jnp.zeros((1, 16, 2, 8), dtype=jnp.float32)
    v = v.at[0, :, 0, :].set(positions[:, None])
    v = v.at[0, :, 1, :].set(100 + positions[:, None])
    mask = (jnp.arange(16)[None, :] < 5).astype(jnp.int32)

    output = _prototype(q, k, v, mask)

    # Query 8 is in the second Pallas call.  Right padding limits its visible
    # keys to [0, 5), whose mean is 2.
    np.testing.assert_array_equal(output[0, 8, 0], jnp.full((8,), 2.0))
    np.testing.assert_array_equal(output[0, 8, 1], jnp.full((8,), 2.0))
    np.testing.assert_array_equal(output[0, 8, 2], jnp.full((8,), 102.0))
    np.testing.assert_array_equal(output[0, 8, 3], jnp.full((8,), 102.0))


def test_dkdv_reduction_is_bitwise_deterministic():
    q, k, v, mask = _inputs()
    dout = jax.random.normal(jax.random.key(4), q.shape)

    def gradients():
        return jax.vjp(lambda *args: _prototype(*args, mask), q, k, v)[1](dout)

    first = gradients()
    second = gradients()
    for first_gradient, second_gradient in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_gradient, second_gradient)


def _count_pallas_calls(closed_jaxpr) -> int:
    """Recursively count primitive boundaries, including custom-VJP jaxprs."""
    count = 0

    def visit(value):
        nonlocal count
        if hasattr(value, "jaxpr") and hasattr(value.jaxpr, "eqns"):
            visit(value.jaxpr)
        elif hasattr(value, "eqns"):
            for equation in value.eqns:
                if equation.primitive.name == "pallas_call":
                    count += 1
                for parameter in equation.params.values():
                    visit(parameter)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(closed_jaxpr)
    return count


def test_each_query_chunk_remains_a_separate_pallas_boundary():
    q, k, v, mask = _inputs()
    forward_jaxpr = jax.make_jaxpr(lambda *args: _prototype(*args, mask))(q, k, v)
    assert _count_pallas_calls(forward_jaxpr) == 2

    dout = jnp.ones_like(q)
    backward_jaxpr = jax.make_jaxpr(
        lambda q_arg, k_arg, v_arg: jax.vjp(
            lambda *args: _prototype(*args, mask), q_arg, k_arg, v_arg
        )[1](dout)
    )(q, k, v)
    # Two forward residual calls, then one dQ and one deterministic dK/dV
    # accumulation call per query chunk.
    assert _count_pallas_calls(backward_jaxpr) == 6


def test_qwen35_32k_plan_has_bounded_dispatches():
    plan = qwen35_32k_dispatch_plan()
    assert plan == {
        "query_chunks": 64,
        "forward_dispatches": 64,
        "dq_dispatches": 64,
        "dkdv_dispatches": 64,
        "forward_programs_per_dispatch": 128,
        "dq_programs_per_dispatch": 256,
        "dkdv_programs_per_dispatch": 4096,
        "max_key_blocks_per_forward_program": 512,
        "max_key_blocks_per_dq_program": 1024,
        "query_blocks_per_dkdv_program": 16,
    }


def test_qwen35_32k_forward_and_vjp_abstract_shapes():
    q = jax.ShapeDtypeStruct((1, 32_768, 16, 256), jnp.bfloat16)
    k = jax.ShapeDtypeStruct((1, 32_768, 4, 256), jnp.bfloat16)
    v = jax.ShapeDtypeStruct((1, 32_768, 4, 256), jnp.bfloat16)
    mask = jax.ShapeDtypeStruct((1, 32_768), jnp.int32)

    forward_jaxpr = jax.make_jaxpr(
        lambda q_arg, k_arg, v_arg, mask_arg: query_bounded_gqa(
            q_arg, k_arg, v_arg, mask_arg, interpret=True
        )
    )(q, k, v, mask)

    def pullback(q_arg, k_arg, v_arg, mask_arg, dout_arg):
        return jax.vjp(
            lambda q_item, k_item, v_item: query_bounded_gqa(
                q_item, k_item, v_item, mask_arg, interpret=True
            ),
            q_arg,
            k_arg,
            v_arg,
        )[1](dout_arg)

    backward_jaxpr = jax.make_jaxpr(pullback)(q, k, v, mask, q)
    assert tuple((aval.shape, aval.dtype) for aval in forward_jaxpr.out_avals) == (
        (q.shape, q.dtype),
    )
    assert tuple((aval.shape, aval.dtype) for aval in backward_jaxpr.out_avals) == (
        (q.shape, q.dtype),
        (k.shape, k.dtype),
        (v.shape, v.dtype),
    )
    assert _count_pallas_calls(forward_jaxpr) == 64
    # The pullback retains 64 forward calls for its residuals, then emits 64
    # bounded dQ calls and 64 deterministic dK/dV accumulation calls.
    assert _count_pallas_calls(backward_jaxpr) == 192


def test_partial_tail_chunks_are_explicitly_rejected():
    q, k, v, mask = _inputs()
    q = q[:, :-1]
    k = k[:, :-1]
    v = v[:, :-1]
    mask = mask[:, :-1]

    with pytest.raises(ValueError, match="divisible by query_chunk_size"):
        _prototype(q, k, v, mask)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda q, k, v, m: (jnp.concatenate((q, q), axis=0), k, v, m),
            "batch size one",
        ),
        (lambda q, k, v, m: (q, k[:, :-1], v, m), "self-attention"),
        (lambda q, k, v, m: (q, k[:, :, :1], v, m), "matching K/V"),
        (lambda q, k, v, m: (q, k, v, m.astype(jnp.float32)), "boolean or integer"),
    ],
)
def test_contract_rejects_unsupported_inputs(mutate, match):
    q, k, v, mask = _inputs()
    q, k, v, mask = mutate(q, k, v, mask)
    with pytest.raises((TypeError, ValueError), match=match):
        _prototype(q, k, v, mask)
