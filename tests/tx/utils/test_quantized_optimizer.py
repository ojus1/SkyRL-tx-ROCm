from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from skyrl.tx.utils.quantized_optimizer import (
    dequantize_affine_int8,
    dequantize_nonnegative_uint8,
    dequantize_symmetric_int8,
    init_quantized_adamw,
    moment_memory_accounting,
    quantize_affine_int8,
    quantize_nonnegative_uint8,
    quantize_symmetric_int8,
    quantized_adamw_update,
    quantized_moment_quality_gate,
)


@pytest.fixture(autouse=True)
def _force_cpu_default_device():
    """Keep this semantic prototype off the accelerator."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def test_symmetric_int8_has_explicit_tail_metadata_and_ties_to_even():
    source = jnp.asarray([1.0, 0.5, -0.5, -1.0, 0.25], dtype=jnp.float32)

    encoded = quantize_symmetric_int8(source, block_size=4)

    assert encoded.original_shape == (5,)
    assert encoded.original_size == 5
    assert encoded.padded_size == 8
    assert encoded.pad_elements == 3
    assert encoded.block_count == 2
    assert encoded.values.dtype == jnp.int8
    # 0.5 / (1 / 127) is the exact tie 63.5; nearest-even is 64.
    np.testing.assert_array_equal(
        np.asarray(encoded.values[:4]), np.asarray([127, 64, -64, -127], dtype=np.int8)
    )
    assert dequantize_symmetric_int8(encoded).shape == source.shape


def test_nonnegative_uint8_sqrt_encoding_preserves_zeros_and_outlier_order():
    source = jnp.asarray([0.0, 1e-12, 1e-8, 1e-4, 1.0e4], dtype=jnp.float32)

    encoded = quantize_nonnegative_uint8(source, block_size=4)
    restored = np.asarray(dequantize_nonnegative_uint8(encoded))

    assert encoded.values.dtype == jnp.uint8
    assert encoded.encoding == "sqrt_nonnegative"
    assert restored[0] == 0.0
    assert np.all(restored >= 0)
    assert np.all(np.diff(restored) >= 0)
    # The tail is in a separate padded block, so the outlier is represented exactly.
    np.testing.assert_allclose(restored[-1], 1.0e4, rtol=2e-4)


def test_sqrt_companding_avoids_a_linear_nu_zero_beside_an_outlier():
    source = jnp.asarray([0.0, 1e-4, 1e-6, 1.0], dtype=jnp.float32)

    encoded = quantize_nonnegative_uint8(source, block_size=4)
    linear_codes = jnp.rint(source / (jnp.max(source) / 255.0)).astype(jnp.uint8)

    assert int(linear_codes[1]) == 0
    assert int(encoded.values[1]) > 0
    assert float(dequantize_nonnegative_uint8(encoded)[1]) > 0.0


@pytest.mark.parametrize("scale_dtype", [jnp.float32, jnp.bfloat16])
def test_zero_blocks_roundtrip_exactly_with_positive_scales(scale_dtype):
    zeros = jnp.zeros((3, 257), dtype=jnp.float32)

    mu = quantize_symmetric_int8(zeros, scale_dtype=scale_dtype)
    nu = quantize_nonnegative_uint8(zeros, scale_dtype=scale_dtype)

    assert mu.scales.dtype == scale_dtype
    assert nu.scales.dtype == scale_dtype
    assert np.all(np.asarray(mu.scales) > 0)
    assert np.all(np.asarray(nu.scales) > 0)
    np.testing.assert_array_equal(
        np.asarray(dequantize_symmetric_int8(mu)), np.zeros((3, 257))
    )
    np.testing.assert_array_equal(
        np.asarray(dequantize_nonnegative_uint8(nu)), np.zeros((3, 257))
    )


def test_quantization_is_bitwise_deterministic():
    source = jnp.reshape(jnp.sin(jnp.arange(777, dtype=jnp.float32) * 0.173), (3, 259))

    first_mu = quantize_symmetric_int8(source)
    second_mu = quantize_symmetric_int8(source)
    first_nu = quantize_nonnegative_uint8(jnp.square(source))
    second_nu = quantize_nonnegative_uint8(jnp.square(source))

    np.testing.assert_array_equal(
        np.asarray(first_mu.values), np.asarray(second_mu.values)
    )
    np.testing.assert_array_equal(
        np.asarray(first_mu.scales), np.asarray(second_mu.scales)
    )
    np.testing.assert_array_equal(
        np.asarray(first_nu.values), np.asarray(second_nu.values)
    )
    np.testing.assert_array_equal(
        np.asarray(first_nu.scales), np.asarray(second_nu.scales)
    )


def test_affine_mu_uses_offsets_and_preserves_constant_outlier_and_tail_blocks():
    source = jnp.asarray([4.0, 4.001, 4.002, 9.0, -7.0], dtype=jnp.float32)

    first = quantize_affine_int8(source, block_size=4)
    second = quantize_affine_int8(source, block_size=4)
    restored = dequantize_affine_int8(first)

    assert first.encoding == "affine_signed"
    assert first.offsets.shape == (2,)
    assert first.pad_elements == 3
    np.testing.assert_array_equal(np.asarray(first.values), np.asarray(second.values))
    np.testing.assert_array_equal(np.asarray(first.scales), np.asarray(second.scales))
    np.testing.assert_array_equal(np.asarray(first.offsets), np.asarray(second.offsets))
    np.testing.assert_allclose(np.asarray(restored), np.asarray(source), atol=0.011)


def test_symmetric_error_is_bounded_by_half_a_stored_scale_away_from_clipping():
    source = jnp.sin(jnp.arange(513, dtype=jnp.float32) * 0.37) * 17.0
    encoded = quantize_symmetric_int8(source)
    restored = dequantize_symmetric_int8(encoded)
    error = jnp.abs(restored - source)
    per_element_scale = jnp.repeat(
        encoded.scales.astype(jnp.float32), encoded.block_size
    )[: source.size]

    assert float(jnp.max(error - 0.5001 * per_element_scale)) <= 2e-6


def test_exact_memory_accounting_for_qwen_lora_moments():
    fp32_scales = moment_memory_accounting(
        34_512_896, block_size=256, scale_dtype=jnp.float32
    )
    bf16_scales = moment_memory_accounting(
        34_512_896, block_size=256, scale_dtype=jnp.bfloat16
    )
    block64_candidate = moment_memory_accounting(
        34_512_896,
        block_size=64,
        scale_dtype=jnp.float32,
        affine_mu=True,
    )
    block32_candidate = moment_memory_accounting(
        34_512_896,
        block_size=32,
        scale_dtype=jnp.float32,
        affine_mu=True,
    )
    candidate = moment_memory_accounting(
        34_512_896,
        block_size=16,
        scale_dtype=jnp.float32,
        affine_mu=True,
    )

    assert fp32_scales.block_count_per_slot == 134_816
    assert fp32_scales.pad_elements_per_slot == 0
    assert fp32_scales.quantized_values_bytes == 69_025_792
    assert fp32_scales.quantized_scales_bytes == 1_078_528
    assert fp32_scales.quantized_offsets_bytes == 0
    assert fp32_scales.quantized_total_bytes == 70_104_320
    assert fp32_scales.bf16_total_bytes == 138_051_584
    assert fp32_scales.saved_bytes_vs_bf16 == 67_947_264
    assert bf16_scales.quantized_scales_bytes == 539_264
    assert bf16_scales.quantized_total_bytes == 69_565_056
    assert bf16_scales.bf16_total_bytes == 138_051_584
    assert block64_candidate.quantized_total_bytes == 75_496_960
    assert block64_candidate.saved_bytes_vs_bf16 == 62_554_624
    assert block32_candidate.quantized_total_bytes == 81_968_128
    assert block32_candidate.saved_bytes_vs_bf16 == 56_083_456
    assert candidate.block_count_per_slot == 2_157_056
    assert candidate.quantized_values_bytes == 69_025_792
    assert candidate.quantized_scales_bytes == 17_256_448
    assert candidate.quantized_offsets_bytes == 8_628_224
    assert candidate.mu_has_offsets is True
    assert candidate.quantized_total_bytes == 94_910_464
    assert candidate.saved_bytes_vs_bf16 == 43_141_120


def _cosine(left, right):
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def _relative_norm_error(actual, expected):
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(
        expected, dtype=np.float64
    )
    return float(
        np.linalg.norm(delta) / np.linalg.norm(np.asarray(expected, dtype=np.float64))
    )


def _run_100_step_semantic_comparison(param_dtype, shape):
    grid = jnp.reshape(jnp.arange(np.prod(shape), dtype=jnp.float32), shape)
    initial = (0.02 * jnp.sin(grid * 0.013) + 0.01 * jnp.cos(grid * 0.031)).astype(
        param_dtype
    )
    quantized_params = initial
    reference_params = initial
    # The candidate spends FP32 metadata on its 16-value blocks.  Its reference
    # is the exact Optax configuration used by SkyRL, not a hand-written hybrid.
    quantized_state = init_quantized_adamw(
        initial, block_size=16, scale_dtype=jnp.float32
    )
    reference_optimizer = optax.adamw(
        learning_rate=3e-4,
        b1=0.9,
        b2=0.99,
        eps=1e-8,
        weight_decay=0.01,
        mu_dtype=None,
    )
    reference_state = reference_optimizer.init(reference_params)
    kwargs = dict(
        learning_rate=3e-4, beta1=0.9, beta2=0.99, epsilon=1e-8, weight_decay=0.01
    )
    worst_relative_norm_error = 0.0
    first_cosine = None

    for step in range(100):
        phase = jnp.asarray(step + 1, dtype=jnp.float32)
        # Deterministically varying dense gradients with broad per-coordinate
        # magnitudes.  Dedicated tests above cover zero/outlier blocks.
        grads = 0.3 * jnp.sin(grid * 0.017 + phase * 0.11) + 0.2 * jnp.cos(
            grid * 0.007 - phase * 0.19
        )
        grads = grads.astype(param_dtype)

        quantized_params, quantized_state, quantized_update = quantized_adamw_update(
            quantized_params, grads, quantized_state, **kwargs
        )
        reference_update, reference_state = reference_optimizer.update(
            grads, reference_state, reference_params
        )
        reference_params = optax.apply_updates(reference_params, reference_update)

        assert bool(np.asarray(jnp.all(jnp.isfinite(quantized_update))))
        assert bool(np.asarray(jnp.all(jnp.isfinite(quantized_params))))
        assert bool(
            np.asarray(
                jnp.all(jnp.isfinite(dequantize_affine_int8(quantized_state.mu)))
            )
        )
        assert bool(
            np.asarray(
                jnp.all(jnp.isfinite(dequantize_nonnegative_uint8(quantized_state.nu)))
            )
        )
        error = _relative_norm_error(quantized_update, reference_update)
        worst_relative_norm_error = max(worst_relative_norm_error, error)
        if step == 0:
            first_cosine = _cosine(quantized_update, reference_update)

    return first_cosine, worst_relative_norm_error


@pytest.mark.parametrize(
    ("param_dtype", "shape", "expected_to_pass"),
    [
        (jnp.float32, (16, 512), True),  # LoRA-A-like.
        (jnp.bfloat16, (512, 16), False),  # LoRA-B-like.
    ],
)
def test_quantized_adamw_against_actual_optax_adamw_for_100_varying_lora_steps(
    param_dtype, shape, expected_to_pass, record_property
):
    first_cosine, worst_error = _run_100_step_semantic_comparison(param_dtype, shape)

    record_property("reference_optimizer", "optax.adamw(mu_dtype=None)")
    record_property("first_update_direction_cosine_sanity_only", first_cosine)
    record_property("worst_100_step_relative_update_norm_error", worst_error)
    quality = quantized_moment_quality_gate(first_cosine, worst_error)
    record_property("quality_gate_passed", quality.passed)
    record_property("quality_gate_reasons", "; ".join(quality.reasons))
    assert np.isfinite(first_cosine)
    assert np.isfinite(worst_error)
    assert first_cosine >= 0.999, (
        f"sanity-only first-update direction cosine={first_cosine:.9f}"
    )
    if expected_to_pass:
        assert quality.passed, quality.reasons
        assert worst_error <= 0.01
    else:
        assert worst_error > 0.01
        assert quality.passed is False
        assert any("exceeds 1.00%" in reason for reason in quality.reasons)


def test_quantized_moment_quality_gate_has_fixed_boundaries_and_structured_reasons():
    exact_boundary = quantized_moment_quality_gate(0.999, 0.01)
    rejected = quantized_moment_quality_gate(0.998, 0.011)
    nonfinite = quantized_moment_quality_gate(np.nan, np.inf)

    assert exact_boundary.passed is True
    assert exact_boundary.reasons == ()
    assert exact_boundary.minimum_direction_cosine == 0.999
    assert exact_boundary.maximum_relative_update_norm_error == 0.01
    assert rejected.passed is False
    assert len(rejected.reasons) == 2
    assert nonfinite.passed is False
    assert len(nonfinite.reasons) == 2


def test_adamw_preserves_parameter_dtype_and_state_geometry():
    params = jnp.ones((257,), dtype=jnp.bfloat16)
    state = init_quantized_adamw(params, block_size=128, scale_dtype=jnp.bfloat16)

    next_params, next_state, update = quantized_adamw_update(
        params, jnp.full_like(params, 0.25), state, learning_rate=1e-3
    )

    assert next_params.dtype == jnp.bfloat16
    assert update.dtype == jnp.float32
    assert next_state.count == 1
    assert next_state.mu.original_shape == params.shape
    assert next_state.mu.block_size == 128
    assert next_state.mu.scales.dtype == jnp.bfloat16


@pytest.mark.parametrize(
    "bad", [jnp.asarray([jnp.nan]), jnp.asarray([jnp.inf]), jnp.asarray([-jnp.inf])]
)
def test_rejects_nonfinite_quantization_input(bad):
    with pytest.raises(ValueError, match="finite"):
        quantize_symmetric_int8(bad)


def test_rejects_negative_second_moment_input():
    with pytest.raises(ValueError, match="nonnegative"):
        quantize_nonnegative_uint8(jnp.asarray([0.0, -1e-9]))


@pytest.mark.parametrize("block_size", [0, -1, 1.5, True])
def test_rejects_invalid_block_size(block_size):
    with pytest.raises((TypeError, ValueError), match="block_size"):
        quantize_symmetric_int8(jnp.ones((3,)), block_size=block_size)


@pytest.mark.parametrize("scale_dtype", [jnp.float16, jnp.float64, jnp.int32])
def test_rejects_invalid_scale_dtype(scale_dtype):
    with pytest.raises(TypeError, match="scale_dtype"):
        quantize_symmetric_int8(jnp.ones((3,)), scale_dtype=scale_dtype)


def test_rejects_malformed_metadata_and_payloads():
    valid = quantize_symmetric_int8(jnp.ones((5,)), block_size=4)
    affine = quantize_affine_int8(jnp.arange(5, dtype=jnp.float32), block_size=4)

    with pytest.raises(ValueError, match="original_size"):
        dequantize_symmetric_int8(replace(valid, original_size=4))
    with pytest.raises(ValueError, match="values shape"):
        dequantize_symmetric_int8(replace(valid, values=valid.values[:-1]))
    with pytest.raises(TypeError, match="dtype int8"):
        dequantize_symmetric_int8(replace(valid, values=valid.values.astype(jnp.uint8)))
    with pytest.raises(ValueError, match="strictly positive"):
        dequantize_symmetric_int8(replace(valid, scales=valid.scales.at[0].set(0)))
    with pytest.raises(TypeError, match="offsets"):
        dequantize_affine_int8(replace(affine, offsets=None))
    with pytest.raises(ValueError, match="offsets.*finite"):
        dequantize_affine_int8(
            replace(affine, offsets=affine.offsets.at[0].set(jnp.nan))
        )


def test_rejects_state_gradient_and_hyperparameter_errors():
    params = jnp.ones((8,), dtype=jnp.float32)
    state = init_quantized_adamw(params)

    with pytest.raises(ValueError, match="grads shape"):
        quantized_adamw_update(params, jnp.ones((7,)), state, learning_rate=1e-3)
    with pytest.raises(ValueError, match="moment shape"):
        quantized_adamw_update(
            jnp.ones((9,)), jnp.ones((9,)), state, learning_rate=1e-3
        )
    with pytest.raises(ValueError, match="beta1"):
        quantized_adamw_update(params, params, state, learning_rate=1e-3, beta1=1.0)
    with pytest.raises(ValueError, match="epsilon"):
        quantized_adamw_update(params, params, state, learning_rate=1e-3, epsilon=0.0)
    with pytest.raises(ValueError, match="finite"):
        quantized_adamw_update(
            params, jnp.full_like(params, jnp.nan), state, learning_rate=1e-3
        )


def test_rejects_fp16_parameters_explicitly():
    params = jnp.ones((8,), dtype=jnp.float16)

    with pytest.raises(TypeError, match="exactly float32 or bfloat16"):
        init_quantized_adamw(params)
    with pytest.raises(TypeError, match="exactly float32 or bfloat16"):
        quantized_adamw_update(
            params,
            params,
            init_quantized_adamw(params.astype(jnp.float32)),
            learning_rate=1e-3,
        )


def test_actual_optax_default_moment_state_follows_bf16_parameters():
    params = jnp.ones((8,), dtype=jnp.bfloat16)
    optimizer = optax.adamw(learning_rate=1e-3, mu_dtype=None)

    state = optimizer.init(params)

    assert state[0].mu.dtype == jnp.bfloat16
    assert state[0].nu.dtype == jnp.bfloat16


def test_pytree_roundtrip_keeps_static_metadata():
    state = init_quantized_adamw(jnp.ones((3, 7)), block_size=8)

    leaves, treedef = jax.tree_util.tree_flatten(state)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(restored, type(state))
    assert restored.count == 0
    assert restored.mu.original_shape == (3, 7)
    assert restored.mu.padded_size == 24
    assert len(leaves) == 5
