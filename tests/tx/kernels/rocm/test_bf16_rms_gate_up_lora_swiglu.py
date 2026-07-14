from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyrl.tx.kernels.rocm.bf16_rms_gate_up_lora_swiglu import (
    BF16_LORA_RANK,
    MAX_IN_FEATURES,
    MAX_PHYSICAL_FEATURES,
    RMS_NORM_EPSILON,
    _validate_inputs,
    bf16_rms_gate_up_lora_swiglu,
)


@pytest.fixture(autouse=True)
def _force_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _case(*, rows: int = 16, inputs: int = 64, products: int = 64):
    x = jax.random.normal(jax.random.key(1), (1, rows, inputs), dtype=jnp.bfloat16)
    rms_delta = (
        jax.random.normal(jax.random.key(2), (inputs,), dtype=jnp.bfloat16) * 0.05
    ).astype(jnp.bfloat16)
    frozen_weight = (
        jax.random.normal(jax.random.key(3), (inputs, 2 * products), dtype=jnp.bfloat16)
        * 0.1
    ).astype(jnp.bfloat16)
    lora_a = (
        jax.random.normal(
            jax.random.key(4), (inputs, BF16_LORA_RANK), dtype=jnp.bfloat16
        )
        * 0.1
    ).astype(jnp.bfloat16)
    lora_b = (
        jax.random.normal(
            jax.random.key(5),
            (BF16_LORA_RANK, 2 * products),
            dtype=jnp.bfloat16,
        )
        * 0.1
    ).astype(jnp.bfloat16)
    scaling = jnp.asarray(0.75, dtype=jnp.bfloat16)
    return x, rms_delta, frozen_weight, lora_a, lora_b, scaling


def _reference(
    x: jax.Array,
    rms_delta: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    scaling: jax.Array,
) -> jax.Array:
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
    normalized = x_f32 * jax.lax.rsqrt(variance + RMS_NORM_EPSILON)
    normalized = (normalized * (1.0 + rms_delta.astype(jnp.float32))).astype(
        jnp.bfloat16
    )
    flat_normalized = normalized.reshape((-1, normalized.shape[-1]))
    z = (flat_normalized @ lora_a).astype(jnp.bfloat16)
    base = (flat_normalized @ frozen_weight).astype(jnp.bfloat16)
    low_rank = (z @ lora_b).astype(jnp.bfloat16)
    fused = (base + (low_rank * scaling).astype(jnp.bfloat16)).astype(jnp.bfloat16)
    gate = fused[:, 0::2]
    up = fused[:, 1::2]
    product = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
    return product.reshape((*x.shape[:-1], gate.shape[-1]))


def _relative_l2(actual: jax.Array, expected: jax.Array) -> float:
    actual_f32 = actual.astype(jnp.float32)
    expected_f32 = expected.astype(jnp.float32)
    return float(
        jnp.linalg.norm(actual_f32 - expected_f32)
        / jnp.maximum(jnp.linalg.norm(expected_f32), 1e-12)
    )


def _equations_recursive(jaxpr):
    for equation in jaxpr.eqns:
        yield equation
        for value in equation.params.values():
            candidates = value if isinstance(value, (tuple, list)) else (value,)
            for candidate in candidates:
                nested = getattr(candidate, "jaxpr", candidate)
                if hasattr(nested, "eqns"):
                    yield from _equations_recursive(nested)


def test_default_off_rejects_before_pallas_execution() -> None:
    x, rms_delta, weight, lora_a, lora_b, scaling = _case()

    with pytest.raises(RuntimeError, match="disabled by default"):
        bf16_rms_gate_up_lora_swiglu(
            x,
            rms_delta,
            weight,
            lora_a,
            lora_b,
            scaling,
            interpret=True,
        )


@pytest.mark.parametrize(("name", "value"), [("enabled", 1), ("interpret", 0)])
def test_opt_in_flags_require_exact_boole(name: str, value: object) -> None:
    x, rms_delta, weight, lora_a, lora_b, scaling = _case()
    kwargs = {"enabled": True, "interpret": True, name: value}

    with pytest.raises(TypeError, match=f"{name} must be an exact bool"):
        bf16_rms_gate_up_lora_swiglu(
            x,
            rms_delta,
            weight,
            lora_a,
            lora_b,
            scaling,
            **kwargs,
        )


def test_scaling_and_epsilon_contracts_are_strict() -> None:
    x, rms_delta, weight, lora_a, lora_b, scaling = _case()

    with pytest.raises(TypeError, match="lora_scaling must be BF16"):
        bf16_rms_gate_up_lora_swiglu(
            x,
            rms_delta,
            weight,
            lora_a,
            lora_b,
            scaling.astype(jnp.float32),
            enabled=True,
            interpret=True,
        )
    with pytest.raises(ValueError, match="eps must be the Qwen3.5 value"):
        bf16_rms_gate_up_lora_swiglu(
            x,
            rms_delta,
            weight,
            lora_a,
            lora_b,
            scaling,
            enabled=True,
            eps=1e-5,
            interpret=True,
        )


def test_exact_production_geometry_passes_static_validation() -> None:
    bf16 = jnp.bfloat16
    _validate_inputs(
        jax.ShapeDtypeStruct((1, 64, MAX_IN_FEATURES), bf16),
        jax.ShapeDtypeStruct((MAX_IN_FEATURES,), bf16),
        jax.ShapeDtypeStruct((MAX_IN_FEATURES, MAX_PHYSICAL_FEATURES), bf16),
        jax.ShapeDtypeStruct((MAX_IN_FEATURES, BF16_LORA_RANK), bf16),
        jax.ShapeDtypeStruct((BF16_LORA_RANK, MAX_PHYSICAL_FEATURES), bf16),
        jax.ShapeDtypeStruct((), bf16),
        block_m=16,
        block_n=32,
        block_k=64,
    )


def test_interpret_forward_and_vjp_match_bf16_reference() -> None:
    x, rms_delta, weight, lora_a, lora_b, scaling = _case()
    cotangent = jax.random.normal(jax.random.key(6), (1, 16, 64), dtype=jnp.bfloat16)

    def candidate(x_arg, a_arg, b_arg):
        return bf16_rms_gate_up_lora_swiglu(
            x_arg,
            rms_delta,
            weight,
            a_arg,
            b_arg,
            scaling,
            enabled=True,
            interpret=True,
            block_m=16,
            block_n=32,
            block_k=64,
        )

    def reference(x_arg, a_arg, b_arg):
        return _reference(
            x_arg,
            rms_delta,
            weight,
            a_arg,
            b_arg,
            scaling,
        )

    actual_output, actual_pullback = jax.vjp(candidate, x, lora_a, lora_b)
    expected_output, expected_pullback = jax.vjp(reference, x, lora_a, lora_b)
    actual_gradients = actual_pullback(cotangent)
    expected_gradients = expected_pullback(cotangent)

    assert actual_output.dtype == jnp.bfloat16
    assert _relative_l2(actual_output, expected_output) < 0.01
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        assert actual.dtype == jnp.bfloat16
        assert _relative_l2(actual, expected) < 0.01


def test_jitted_checkpoint_backward_is_finite() -> None:
    x, rms_delta, weight, lora_a, lora_b, scaling = _case()

    def objective(x_arg, a_arg, b_arg):
        product = bf16_rms_gate_up_lora_swiglu(
            x_arg,
            rms_delta,
            weight,
            a_arg,
            b_arg,
            scaling,
            enabled=True,
            interpret=True,
            block_m=16,
            block_n=32,
            block_k=64,
        )
        return product.astype(jnp.float32).sum()

    gradients = jax.jit(jax.grad(jax.checkpoint(objective), argnums=(0, 1, 2)))(
        x, lora_a, lora_b
    )

    assert all(value.dtype == jnp.bfloat16 for value in gradients)
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in gradients)


def test_interpret_forward_is_bitwise_repeatable() -> None:
    x, rms_delta, weight, lora_a, lora_b, scaling = _case()

    def run():
        return bf16_rms_gate_up_lora_swiglu(
            x,
            rms_delta,
            weight,
            lora_a,
            lora_b,
            scaling,
            enabled=True,
            interpret=True,
            block_m=16,
            block_n=32,
            block_k=64,
        )

    np.testing.assert_array_equal(run(), run())


def test_trace_has_only_two_forward_dispatches_and_bounded_outputs() -> None:
    x, rms_delta, weight, lora_a, lora_b, scaling = _case()

    def objective(x_arg, a_arg, b_arg):
        product = bf16_rms_gate_up_lora_swiglu(
            x_arg,
            rms_delta,
            weight,
            a_arg,
            b_arg,
            scaling,
            enabled=True,
            interpret=True,
            block_m=16,
            block_n=64,
            block_k=64,
        )
        return product.astype(jnp.float32).sum()

    closed = jax.make_jaxpr(jax.grad(objective, argnums=(0, 1, 2)))(x, lora_a, lora_b)
    calls = {
        equation.params["name"]: equation
        for equation in _equations_recursive(closed.jaxpr)
        if equation.primitive.name == "pallas_call"
    }

    assert set(calls) == {
        "skyrl_qwen35_bf16_rms_lora_a_forward",
        "skyrl_qwen35_bf16_gate_up_lora_swiglu_forward",
    }
    first = calls["skyrl_qwen35_bf16_rms_lora_a_forward"]
    second = calls["skyrl_qwen35_bf16_gate_up_lora_swiglu_forward"]
    assert first.params["grid_mapping"].grid == (1,)
    assert second.params["grid_mapping"].grid == (1, 1)
    assert {tuple(variable.aval.shape) for variable in first.outvars} == {
        (16,),
        (16, BF16_LORA_RANK),
    }
    assert [tuple(variable.aval.shape) for variable in second.outvars] == [(16, 64)]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("partial_row_tile", "must be divisible by block_m"),
        ("odd_physical_width", "must be even"),
        ("wrong_rank", "lora_a must have shape"),
    ],
)
def test_invalid_geometry_rejects_before_execution(mutation: str, match: str) -> None:
    x, rms_delta, weight, lora_a, lora_b, scaling = _case()
    if mutation == "partial_row_tile":
        x = x[:, :-1]
    elif mutation == "odd_physical_width":
        weight = weight[:, :-1]
        lora_b = lora_b[:, :-1]
    else:
        lora_a = lora_a[:, :-1]

    with pytest.raises(ValueError, match=match):
        bf16_rms_gate_up_lora_swiglu(
            x,
            rms_delta,
            weight,
            lora_a,
            lora_b,
            scaling,
            enabled=True,
            interpret=True,
            block_m=16,
            block_n=64,
            block_k=64,
        )


def test_interleaved_pairing_is_gate_then_up() -> None:
    x = jnp.ones((1, 16, 64), dtype=jnp.bfloat16)
    rms_delta = jnp.zeros((64,), dtype=jnp.bfloat16)
    weight = jnp.zeros((64, 128), dtype=jnp.bfloat16)
    weight = weight.at[:, 0::2].set(jnp.asarray(0.02, dtype=jnp.bfloat16))
    weight = weight.at[:, 1::2].set(jnp.asarray(0.03, dtype=jnp.bfloat16))
    lora_a = jnp.zeros((64, BF16_LORA_RANK), dtype=jnp.bfloat16)
    lora_b = jnp.zeros((BF16_LORA_RANK, 128), dtype=jnp.bfloat16)
    scaling = jnp.asarray(1.0, dtype=jnp.bfloat16)

    actual = bf16_rms_gate_up_lora_swiglu(
        x,
        rms_delta,
        weight,
        lora_a,
        lora_b,
        scaling,
        enabled=True,
        interpret=True,
        block_m=16,
        block_n=64,
        block_k=64,
    )
    expected = _reference(x, rms_delta, weight, lora_a, lora_b, scaling)

    np.testing.assert_array_equal(actual, expected)
