from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyrl.tx.kernels.quantized_lora import (
    GroupQuantizedWeight,
    _quantized_frozen_linear_f32,
    _quantized_lora_linear,
    _quantized_lora_linear_fwd,
    dequantize_frozen_weight,
    pack_signed_int4,
    quantize_frozen_weight,
    quantized_frozen_linear,
    quantized_lora_linear,
    unpack_signed_int4,
)


def _relative_l2(actual: jax.Array, expected: jax.Array) -> float:
    numerator = jnp.linalg.norm(
        actual.astype(jnp.float32) - expected.astype(jnp.float32)
    )
    denominator = jnp.maximum(jnp.linalg.norm(expected.astype(jnp.float32)), 1e-12)
    return float(numerator / denominator)


def test_signed_int4_pack_roundtrip_preserves_axis_order() -> None:
    values = jnp.asarray(
        [[-7, -1, 0], [-6, 1, 7], [-5, 2, 6], [-4, 3, 5]],
        dtype=jnp.int8,
    )
    packed = pack_signed_int4(values)
    assert packed.shape == (2, 3)
    assert packed.dtype == jnp.uint8
    np.testing.assert_array_equal(
        packed,
        np.asarray([[0xA9, 0x1F, 0x70], [0xCB, 0x32, 0x56]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(unpack_signed_int4(packed), values)


def test_hand_coded_int4_codes_and_scales_dequantize_independently() -> None:
    weight = GroupQuantizedWeight(
        codes=jnp.asarray([[0x31, 0x4E], [0x0B, 0x96]], dtype=jnp.uint8),
        scales=jnp.asarray([[0.5, 2.0], [1.5, 0.25]], dtype=jnp.float32),
        original_in_features=3,
        padded_in_features=4,
        bits=4,
        group_size=2,
    )
    expected = np.asarray([[0.5, -4.0], [1.5, 8.0], [-7.5, 1.5]], dtype=np.float32)
    np.testing.assert_array_equal(dequantize_frozen_weight(weight), expected)


def test_hand_coded_w4a4_grouped_linear_matches_manual_result() -> None:
    weight = GroupQuantizedWeight(
        codes=jnp.asarray([[0x31, 0x4E], [0x0B, 0x96]], dtype=jnp.uint8),
        scales=jnp.asarray([[0.5, 2.0], [1.5, 0.25]], dtype=jnp.float32),
        original_in_features=3,
        padded_in_features=4,
        bits=4,
        group_size=2,
    )
    x = jnp.asarray([[7.0, -3.0, -7.0], [0.0, 0.0, 3.0]], dtype=jnp.float32)
    expected = np.asarray([[51.5, -62.5], [-22.5, 4.5]], dtype=np.float32)
    np.testing.assert_allclose(
        quantized_frozen_linear(x, weight, activation_bits=4),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("bits,max_relative_l2", [(8, 0.012), (4, 0.20)])
def test_group_quantized_weight_roundtrip(bits: int, max_relative_l2: float) -> None:
    weight = jax.random.normal(jax.random.key(bits), (70, 13), dtype=jnp.float32)
    quantized = quantize_frozen_weight(
        weight, bits=bits, group_size=32, scale_dtype=jnp.float32
    )
    restored = dequantize_frozen_weight(quantized)
    assert restored.shape == weight.shape
    assert quantized.padded_in_features == 96
    assert quantized.scales.shape == (3, 13)
    assert quantized.codes.dtype == (jnp.int8 if bits == 8 else jnp.uint8)
    assert _relative_l2(restored, weight) < max_relative_l2


@pytest.mark.parametrize(
    "weight_bits,activation_bits,max_relative_l2",
    [(8, 8, 0.025), (4, 8, 0.20), (8, 4, 0.20), (4, 4, 0.30)],
)
def test_quantized_frozen_linear_accuracy(
    weight_bits: int,
    activation_bits: int,
    max_relative_l2: float,
) -> None:
    x = jax.random.normal(
        jax.random.key(10 + activation_bits), (3, 5, 70), dtype=jnp.float32
    )
    weight = jax.random.normal(
        jax.random.key(20 + weight_bits), (70, 13), dtype=jnp.float32
    )
    quantized = quantize_frozen_weight(
        weight, bits=weight_bits, group_size=32, scale_dtype=jnp.float32
    )
    actual = quantized_frozen_linear(x, quantized, activation_bits=activation_bits)
    expected = x @ weight
    assert actual.shape == (3, 5, 13)
    assert _relative_l2(actual, expected) < max_relative_l2


@pytest.mark.parametrize("weight_bits", [4, 8])
def test_quantized_lora_custom_vjp_matches_documented_backward(
    weight_bits: int,
) -> None:
    x = jax.random.normal(jax.random.key(31), (2, 3, 10), dtype=jnp.float32)
    base_weight = jax.random.normal(jax.random.key(32), (10, 7), dtype=jnp.float32)
    lora_a = jax.random.normal(jax.random.key(33), (10, 3), dtype=jnp.float32) * 0.1
    lora_b = jax.random.normal(jax.random.key(34), (3, 7), dtype=jnp.float32) * 0.1
    scaling = jnp.asarray(0.75, dtype=jnp.float32)
    cotangent = jax.random.normal(jax.random.key(35), (2, 3, 7), dtype=jnp.float32)
    weight = quantize_frozen_weight(
        base_weight, bits=weight_bits, group_size=8, scale_dtype=jnp.float32
    )

    def objective(x_arg, a_arg, b_arg, scaling_arg):
        output = quantized_lora_linear(
            x_arg,
            weight,
            a_arg,
            b_arg,
            scaling_arg,
            activation_bits=8,
        )
        return jnp.sum(output * cotangent)

    actual_dx, actual_da, actual_db, actual_dscaling = jax.grad(
        objective, argnums=(0, 1, 2, 3)
    )(x, lora_a, lora_b, scaling)

    x_flat = x.reshape((-1, x.shape[-1]))
    dy_flat = cotangent.reshape((-1, cotangent.shape[-1]))
    dequantized_weight = dequantize_frozen_weight(weight)
    intermediate = x_flat @ lora_a
    low_rank_input_cotangent = (dy_flat @ lora_b.T) * scaling
    expected_dx = (
        dy_flat @ dequantized_weight.T + low_rank_input_cotangent @ lora_a.T
    ).reshape(x.shape)
    expected_da = x_flat.T @ low_rank_input_cotangent
    expected_db = (intermediate.T @ dy_flat) * scaling
    expected_dscaling = jnp.sum(dy_flat * (intermediate @ lora_b))

    np.testing.assert_allclose(actual_dx, expected_dx, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual_da, expected_da, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual_db, expected_db, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual_dscaling, expected_dscaling, rtol=2e-6, atol=2e-6)


def test_quantized_lora_is_jittable_and_zero_groups_are_finite() -> None:
    weight = quantize_frozen_weight(
        jnp.zeros((16, 5), dtype=jnp.bfloat16), bits=8, group_size=8
    )
    x = jnp.zeros((2, 16), dtype=jnp.bfloat16)
    lora_a = jnp.ones((16, 2), dtype=jnp.bfloat16)
    lora_b = jnp.zeros((2, 5), dtype=jnp.bfloat16)
    run = jax.jit(
        lambda value: quantized_lora_linear(value, weight, lora_a, lora_b, 1.0)
    )
    output = run(x)
    assert output.dtype == jnp.bfloat16
    assert bool(jnp.all(jnp.isfinite(output)))
    np.testing.assert_array_equal(output, jnp.zeros_like(output))


def test_quantized_lora_uses_one_bf16_cast_after_fp32_epilogue() -> None:
    x = jax.random.normal(jax.random.key(41), (4, 16), dtype=jnp.bfloat16)
    base_weight = jax.random.normal(jax.random.key(42), (16, 9), dtype=jnp.bfloat16)
    lora_a = jax.random.normal(jax.random.key(43), (16, 4), dtype=jnp.bfloat16) * 0.2
    lora_b = jax.random.normal(jax.random.key(44), (4, 9), dtype=jnp.bfloat16) * 0.2
    scaling = 0.75
    weight = quantize_frozen_weight(
        base_weight, bits=8, group_size=8, scale_dtype=jnp.bfloat16
    )

    actual = quantized_lora_linear(x, weight, lora_a, lora_b, scaling)
    base_f32 = _quantized_frozen_linear_f32(x, weight)
    low_rank_f32 = (x.astype(jnp.float32) @ lora_a.astype(jnp.float32)) @ lora_b.astype(
        jnp.float32
    )
    expected = (base_f32 + scaling * low_rank_f32).astype(jnp.bfloat16)
    early_cast = (
        base_f32.astype(jnp.bfloat16).astype(jnp.float32) + scaling * low_rank_f32
    ).astype(jnp.bfloat16)

    np.testing.assert_array_equal(actual, expected)
    assert np.count_nonzero(np.asarray(early_cast) != np.asarray(expected)) > 0


@pytest.mark.parametrize(
    "weight_bits,activation_bits", [(8, 8), (4, 8), (8, 4), (4, 4)]
)
def test_nonzero_bf16_jitted_value_and_grad(
    weight_bits: int, activation_bits: int
) -> None:
    x = jax.random.normal(jax.random.key(51), (3, 16), dtype=jnp.bfloat16)
    base = jax.random.normal(jax.random.key(52), (16, 7), dtype=jnp.bfloat16)
    lora_a = jax.random.normal(jax.random.key(53), (16, 3), dtype=jnp.bfloat16) * 0.1
    lora_b = jax.random.normal(jax.random.key(54), (3, 7), dtype=jnp.bfloat16) * 0.1
    weight = quantize_frozen_weight(base, bits=weight_bits, group_size=8)

    def objective(x_arg, a_arg, b_arg):
        output = quantized_lora_linear(
            x_arg,
            weight,
            a_arg,
            b_arg,
            0.75,
            activation_bits=activation_bits,
        )
        return jnp.sum(output.astype(jnp.float32) ** 2)

    value, gradients = jax.jit(jax.value_and_grad(objective, argnums=(0, 1, 2)))(
        x, lora_a, lora_b
    )
    assert bool(jnp.isfinite(value))
    for gradient in gradients:
        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert float(jnp.linalg.norm(gradient.astype(jnp.float32))) > 0


def test_frozen_weight_scales_receive_zero_cotangent() -> None:
    x = jnp.arange(16, dtype=jnp.float32).reshape(2, 8) / 16
    weight = quantize_frozen_weight(
        jnp.eye(8, 5, dtype=jnp.float32), bits=4, group_size=4
    )
    lora_a = jnp.ones((8, 2), dtype=jnp.float32) * 0.1
    lora_b = jnp.ones((2, 5), dtype=jnp.float32) * 0.1

    def objective(scales):
        return jnp.sum(
            _quantized_lora_linear(
                x,
                weight.codes,
                scales,
                lora_a,
                lora_b,
                jnp.asarray(1.0),
                weight.bits,
                8,
                weight.group_size,
                weight.original_in_features,
            )
        )

    np.testing.assert_array_equal(
        jax.grad(objective)(weight.scales), jnp.zeros_like(weight.scales)
    )


def test_custom_vjp_residual_keeps_compact_weight_not_fp32_matrix() -> None:
    x = jnp.ones((2, 16), dtype=jnp.bfloat16)
    weight = quantize_frozen_weight(
        jnp.ones((16, 7), dtype=jnp.bfloat16), bits=4, group_size=8
    )
    lora_a = jnp.ones((16, 2), dtype=jnp.bfloat16)
    lora_b = jnp.ones((2, 7), dtype=jnp.bfloat16)
    _, residual = _quantized_lora_linear_fwd(
        x,
        weight.codes,
        weight.scales,
        lora_a,
        lora_b,
        jnp.asarray(1.0),
        weight.bits,
        8,
        weight.group_size,
        weight.original_in_features,
    )

    _, saved_codes, saved_scales, _, _, _ = residual
    assert saved_codes is weight.codes
    assert saved_scales is weight.scales
    assert not any(
        getattr(value, "shape", None) == (16, 7) and value.dtype == jnp.float32
        for value in residual
    )


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0)])
def test_empty_weight_dimensions_are_rejected(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="weight dimensions must be positive"):
        quantize_frozen_weight(
            jnp.zeros(shape, dtype=jnp.float32), bits=4, group_size=4
        )


def test_nonfloating_inputs_and_scales_are_rejected() -> None:
    with pytest.raises(TypeError, match="scale_dtype must be floating point"):
        quantize_frozen_weight(
            jnp.ones((8, 5), dtype=jnp.float32), scale_dtype=jnp.int8
        )

    weight = quantize_frozen_weight(jnp.ones((8, 5), dtype=jnp.float32), group_size=4)
    with pytest.raises(TypeError, match="activations must be floating point"):
        quantized_frozen_linear(jnp.ones((2, 8), dtype=jnp.int32), weight)
    with pytest.raises(TypeError, match="LoRA parameters must be floating point"):
        quantized_lora_linear(
            jnp.ones((2, 8), dtype=jnp.float32),
            weight,
            jnp.ones((8, 2), dtype=jnp.int8),
            jnp.ones((2, 5), dtype=jnp.float32),
            1.0,
        )


def test_scalar_and_empty_leading_activation_shapes_are_handled_explicitly() -> None:
    weight = quantize_frozen_weight(jnp.ones((8, 5), dtype=jnp.float32), group_size=4)
    with pytest.raises(ValueError, match="at least one dimension"):
        quantized_frozen_linear(jnp.asarray(1.0, dtype=jnp.float32), weight)

    empty = jnp.empty((0, 3, 8), dtype=jnp.float32)
    output = quantized_frozen_linear(empty, weight)
    assert output.shape == (0, 3, 5)


def test_malformed_quantized_weight_is_rejected() -> None:
    malformed = GroupQuantizedWeight(
        codes=jnp.zeros((4, 5), dtype=jnp.int8),
        scales=jnp.ones((2, 5), dtype=jnp.int8),
        original_in_features=4,
        padded_in_features=4,
        bits=8,
        group_size=2,
    )
    with pytest.raises(TypeError, match="scales must be floating point"):
        dequantize_frozen_weight(malformed)


@pytest.mark.parametrize("bits", [3, 5, 16])
def test_invalid_quantization_bits_are_rejected(bits: int) -> None:
    with pytest.raises(ValueError, match="bits must be 4 or 8"):
        quantize_frozen_weight(jnp.ones((8, 4)), bits=bits)
