from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyrl.tx.kernels.bounded_lora import (
    MAX_TOKEN_TILE_SIZE,
    bounded_frozen_lora_linear,
)


def _inputs(dtype: jnp.dtype = jnp.float32):
    x = jax.random.normal(jax.random.key(1), (2, 5, 11), dtype=dtype)
    weight = jax.random.normal(jax.random.key(2), (11, 7), dtype=dtype)
    lora_a = jax.random.normal(jax.random.key(3), (11, 8), dtype=dtype) * 0.1
    lora_b = jax.random.normal(jax.random.key(4), (8, 7), dtype=dtype) * 0.1
    return x, weight, lora_a, lora_b


@pytest.mark.parametrize("token_tile_size", [1, 4, 6, 16])
def test_fp32_forward_matches_dense_equation_across_tiles_and_tail(
    token_tile_size: int,
) -> None:
    x, weight, lora_a, lora_b = _inputs()
    scaling = 0.375
    residual = jax.random.normal(jax.random.key(5), (2, 5, 7))

    actual = jax.jit(
        lambda: bounded_frozen_lora_linear(
            x,
            weight,
            lora_a,
            lora_b,
            scaling,
            residual=residual,
            token_tile_size=token_tile_size,
        )
    )()
    expected = x @ weight + scaling * ((x @ lora_a) @ lora_b) + residual

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_fp32_custom_vjp_matches_analytic_arbitrary_cotangent() -> None:
    x, weight, lora_a, lora_b = _inputs()
    scaling = jnp.asarray(-0.625, dtype=jnp.float32)
    cotangent = jax.random.normal(jax.random.key(6), (2, 5, 7))

    def operation(x_arg, weight_arg, a_arg, b_arg, scaling_arg):
        return bounded_frozen_lora_linear(
            x_arg,
            weight_arg,
            a_arg,
            b_arg,
            scaling_arg,
            token_tile_size=4,
        )

    _, pullback = jax.vjp(operation, x, weight, lora_a, lora_b, scaling)
    actual_dx, actual_dw, actual_da, actual_db, actual_dscale = pullback(cotangent)

    x_flat = x.reshape((-1, x.shape[-1]))
    dy_flat = cotangent.reshape((-1, cotangent.shape[-1]))
    intermediate = x_flat @ lora_a
    low_rank_input_cotangent = scaling * (dy_flat @ lora_b.T)
    expected_dx = (
        dy_flat @ weight.T + low_rank_input_cotangent @ lora_a.T
    ).reshape(x.shape)
    expected_da = x_flat.T @ low_rank_input_cotangent
    expected_db = scaling * (intermediate.T @ dy_flat)

    np.testing.assert_allclose(actual_dx, expected_dx, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(actual_dw, jnp.zeros_like(weight))
    np.testing.assert_allclose(actual_da, expected_da, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual_db, expected_db, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(actual_dscale, jnp.zeros_like(scaling))


def test_residual_cotangent_is_identity() -> None:
    x, weight, lora_a, lora_b = _inputs()
    residual = jax.random.normal(jax.random.key(7), (2, 5, 7))
    cotangent = jax.random.normal(jax.random.key(8), residual.shape)

    _, pullback = jax.vjp(
        lambda residual_arg: bounded_frozen_lora_linear(
            x,
            weight,
            lora_a,
            lora_b,
            0.75,
            residual=residual_arg,
            token_tile_size=6,
        ),
        residual,
    )
    (actual_dresidual,) = pullback(cotangent)

    np.testing.assert_array_equal(actual_dresidual, cotangent)


def test_bf16_forward_and_strengthened_vjp_follow_declared_precision() -> None:
    x, weight, lora_a, lora_b = _inputs(jnp.bfloat16)
    scaling = jnp.asarray(0.75, dtype=jnp.float32)
    cotangent = jax.random.normal(
        jax.random.key(9), (2, 5, 7), dtype=jnp.bfloat16
    )

    def operation(x_arg, a_arg, b_arg):
        return bounded_frozen_lora_linear(
            x_arg,
            weight,
            a_arg,
            b_arg,
            scaling,
            token_tile_size=4,
        )

    actual_output, pullback = jax.vjp(operation, x, lora_a, lora_b)
    actual_dx, actual_da, actual_db = pullback(cotangent)
    dense_output = x @ weight + scaling.astype(jnp.bfloat16) * (
        (x @ lora_a) @ lora_b
    )

    x_f32 = x.reshape((-1, x.shape[-1])).astype(jnp.float32)
    dy_f32 = cotangent.reshape((-1, cotangent.shape[-1])).astype(jnp.float32)
    weight_f32 = weight.astype(jnp.float32)
    a_f32 = lora_a.astype(jnp.float32)
    b_f32 = lora_b.astype(jnp.float32)
    lora_input_cotangent = scaling * (dy_f32 @ b_f32.T)
    expected_dx = (
        dy_f32 @ weight_f32.T + lora_input_cotangent @ a_f32.T
    ).reshape(x.shape).astype(jnp.bfloat16)
    expected_da = (x_f32.T @ lora_input_cotangent).astype(jnp.bfloat16)
    expected_db = (
        scaling * ((x_f32 @ a_f32).T @ dy_f32)
    ).astype(jnp.bfloat16)

    np.testing.assert_array_equal(actual_output, dense_output)
    np.testing.assert_array_equal(actual_dx, expected_dx)
    np.testing.assert_allclose(
        actual_da.astype(jnp.float32),
        expected_da.astype(jnp.float32),
        rtol=8e-3,
        atol=8e-3,
    )
    np.testing.assert_allclose(
        actual_db.astype(jnp.float32),
        expected_db.astype(jnp.float32),
        rtol=8e-3,
        atol=8e-3,
    )


def test_zero_padded_tail_has_no_effect_on_lora_gradients() -> None:
    x, weight, lora_a, lora_b = _inputs()
    cotangent = jax.random.normal(jax.random.key(10), (2, 5, 7))

    def gradients(tile_size: int):
        return jax.grad(
            lambda a_arg, b_arg: jnp.sum(
                bounded_frozen_lora_linear(
                    x,
                    weight,
                    a_arg,
                    b_arg,
                    0.5,
                    token_tile_size=tile_size,
                )
                * cotangent
            ),
            argnums=(0, 1),
        )(lora_a, lora_b)

    tail_gradients = gradients(6)
    exact_gradients = gradients(5)
    for actual, expected in zip(tail_gradients, exact_gradients, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    ("mutation", "error", "match"),
    [
        (lambda values: values | {"token_tile_size": 0}, ValueError, "in \\[1"),
        (
            lambda values: values | {"token_tile_size": MAX_TOKEN_TILE_SIZE + 1},
            ValueError,
            "in \\[1",
        ),
        (
            lambda values: values | {"token_tile_size": 1.5},
            TypeError,
            "must be an integer",
        ),
        (
            lambda values: values | {"lora_a": values["lora_a"][:, :7]},
            ValueError,
            "lora_a must have shape",
        ),
        (
            lambda values: values
            | {"lora_a": jnp.stack((values["lora_a"], values["lora_a"]))},
            ValueError,
            "single-adapter",
        ),
        (
            lambda values: values
            | {"residual": jnp.zeros((2, 5, 8), dtype=jnp.float32)},
            ValueError,
            "residual shape",
        ),
        (
            lambda values: values | {"x": values["x"].astype(jnp.float16)},
            TypeError,
            "BF16 or FP32",
        ),
    ],
)
def test_validation_rejects_unsupported_contracts(mutation, error, match) -> None:
    x, weight, lora_a, lora_b = _inputs()
    values = {
        "x": x,
        "frozen_weight": weight,
        "lora_a": lora_a,
        "lora_b": lora_b,
        "lora_scaling": 0.5,
        "residual": None,
        "token_tile_size": 4,
    }

    with pytest.raises(error, match=match):
        bounded_frozen_lora_linear(**mutation(values))
