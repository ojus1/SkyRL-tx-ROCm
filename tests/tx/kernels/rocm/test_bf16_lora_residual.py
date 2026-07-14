from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyrl.tx.kernels.rocm.bf16_lora_residual import (
    BF16_LORA_RANK,
    _padded_output_features,
    bf16_lora_residual,
)


@pytest.fixture(autouse=True)
def _force_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _case(*, rows: int = 3, inputs: int = 32, outputs: int = 17):
    x = jax.random.normal(jax.random.key(1), (rows, inputs), dtype=jnp.bfloat16)
    weight = jax.random.normal(jax.random.key(2), (inputs, outputs), dtype=jnp.bfloat16)
    lora_a = (
        jax.random.normal(
            jax.random.key(3), (inputs, BF16_LORA_RANK), dtype=jnp.bfloat16
        )
        * 0.1
    ).astype(jnp.bfloat16)
    lora_b = (
        jax.random.normal(
            jax.random.key(4), (BF16_LORA_RANK, outputs), dtype=jnp.bfloat16
        )
        * 0.1
    ).astype(jnp.bfloat16)
    residual = jax.random.normal(jax.random.key(5), (rows, outputs), dtype=jnp.bfloat16)
    return x, weight, lora_a, lora_b, residual


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
    x, weight, lora_a, lora_b, residual = _case()

    with pytest.raises(RuntimeError, match="disabled by default"):
        bf16_lora_residual(
            x,
            weight,
            lora_a,
            lora_b,
            0.75,
            residual,
            interpret=True,
            block_m=16,
            block_n=16,
            block_k=16,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [("enabled", 1), ("interpret", 0), ("pallas_input_vjp", 1)],
)
def test_opt_in_flags_require_exact_boole(name: str, value: object) -> None:
    x, weight, lora_a, lora_b, residual = _case()
    kwargs = {
        "enabled": True,
        "interpret": True,
        "block_m": 16,
        "block_n": 16,
        "block_k": 16,
        name: value,
    }

    with pytest.raises(TypeError, match=f"{name} must be an exact bool"):
        bf16_lora_residual(x, weight, lora_a, lora_b, 0.75, residual, **kwargs)


@pytest.mark.parametrize("pallas_input_vjp", [False, True])
def test_interpret_forward_and_vjp_match_dense_bf16_contract(
    pallas_input_vjp: bool,
) -> None:
    x, weight, lora_a, lora_b, residual = _case(rows=19, inputs=32, outputs=17)
    scaling = 0.75
    cotangent = jax.random.normal(jax.random.key(6), residual.shape, dtype=jnp.bfloat16)

    def candidate(x_arg, a_arg, b_arg, residual_arg):
        return bf16_lora_residual(
            x_arg,
            weight,
            a_arg,
            b_arg,
            scaling,
            residual_arg,
            enabled=True,
            interpret=True,
            block_m=16,
            block_n=16,
            block_k=16,
            row_superblock=16,
            pallas_input_vjp=pallas_input_vjp,
        )

    def reference(x_arg, a_arg, b_arg, residual_arg):
        base = x_arg @ weight
        low_rank = (x_arg @ a_arg) @ b_arg
        return base + low_rank * scaling + residual_arg

    actual_output, actual_pullback = jax.vjp(candidate, x, lora_a, lora_b, residual)
    expected_output, expected_pullback = jax.vjp(reference, x, lora_a, lora_b, residual)
    actual_gradients = actual_pullback(cotangent)
    expected_gradients = expected_pullback(cotangent)

    assert _relative_l2(actual_output, expected_output) < 0.005
    for actual, expected in zip(
        actual_gradients[:3], expected_gradients[:3], strict=True
    ):
        assert _relative_l2(actual, expected) < 0.01
    np.testing.assert_array_equal(actual_gradients[3], expected_gradients[3])


def test_python_float_scaling_survives_jitted_checkpoint_backward() -> None:
    x, weight, lora_a, lora_b, residual = _case()

    def objective(x_arg, a_arg, b_arg, residual_arg):
        output = bf16_lora_residual(
            x_arg,
            weight,
            a_arg,
            b_arg,
            0.75,
            residual_arg,
            enabled=True,
            interpret=True,
            block_m=16,
            block_n=16,
            block_k=16,
            row_superblock=16,
        )
        return output.astype(jnp.float32).sum()

    gradients = jax.jit(jax.grad(jax.checkpoint(objective), argnums=(0, 1, 2, 3)))(
        x, lora_a, lora_b, residual
    )

    assert all(bool(jnp.all(jnp.isfinite(value))) for value in gradients)


def test_optional_pallas_input_vjp_trace_has_two_named_full_tile_calls() -> None:
    x, weight, lora_a, lora_b, residual = _case()

    def objective(x_arg, a_arg, b_arg, residual_arg):
        output = bf16_lora_residual(
            x_arg,
            weight,
            a_arg,
            b_arg,
            0.75,
            residual_arg,
            enabled=True,
            interpret=True,
            block_m=16,
            block_n=16,
            block_k=16,
            row_superblock=16,
            pallas_input_vjp=True,
        )
        return output.astype(jnp.float32).sum()

    closed = jax.make_jaxpr(jax.grad(objective, argnums=(0, 1, 2, 3)))(
        x, lora_a, lora_b, residual
    )
    calls = {
        equation.params["name"]: equation
        for equation in _equations_recursive(closed.jaxpr)
        if equation.primitive.name == "pallas_call"
    }

    assert set(calls) == {
        "skyrl_qwen35_bf16_down_lora_residual_forward",
        "skyrl_qwen35_bf16_down_lora_input_vjp",
    }
    assert calls["skyrl_qwen35_bf16_down_lora_residual_forward"].params[
        "grid_mapping"
    ].grid == (1, 2)
    assert calls["skyrl_qwen35_bf16_down_lora_input_vjp"].params[
        "grid_mapping"
    ].grid == (1, 2)

    for call in calls.values():
        kernel = getattr(call.params["jaxpr"], "jaxpr", call.params["jaxpr"])
        primitives = {
            equation.primitive.name for equation in _equations_recursive(kernel)
        }
        assert primitives.isdisjoint({"lt", "and"})


@pytest.mark.parametrize(
    ("out_features", "block_n", "expected"),
    [(2560, 64, 2560), (17, 16, 32), (32, 16, 32)],
)
def test_output_feature_padding_geometry(
    out_features: int, block_n: int, expected: int
) -> None:
    assert _padded_output_features(out_features, block_n) == expected
