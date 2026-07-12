from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyrl.tx.kernels.qwen3_5_qkv_lora import (
    Qwen35QKVStageConfig,
    _qwen35_qkv_lora_rope_fwd,
    qwen35_qkv_lora_rope,
)
from skyrl.tx.layers.lora import FusedLoRALinear
from skyrl.tx.models.qwen3_5 import apply_partial_rope


def _config() -> Qwen35QKVStageConfig:
    return Qwen35QKVStageConfig(
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=4,
        rotary_dim=2,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
    )


def _inputs(dtype: jnp.dtype = jnp.float32) -> tuple[jax.Array, ...]:
    config = _config()
    keys = jax.random.split(jax.random.key(73), 7)
    hidden_size = 8
    rank = 3
    values = (
        jax.random.normal(keys[0], (2, 3, hidden_size), dtype=jnp.float32),
        jax.random.normal(keys[1], (hidden_size,), dtype=jnp.float32) * 0.1,
        jax.random.normal(keys[2], (config.head_dim,), dtype=jnp.float32) * 0.1,
        jax.random.normal(keys[3], (config.head_dim,), dtype=jnp.float32) * 0.1,
        jax.random.normal(keys[4], (hidden_size, config.fused_width), dtype=jnp.float32)
        * 0.2,
        jax.random.normal(keys[5], (hidden_size, rank), dtype=jnp.float32) * 0.2,
        jax.random.normal(keys[6], (rank, config.fused_width), dtype=jnp.float32) * 0.2,
    )
    return tuple(value.astype(dtype) for value in values) + (
        jnp.asarray(0.75, dtype=dtype),
        jnp.asarray([[0, 1, 7], [3, 4, 9]], dtype=jnp.int32),
    )


def _reference_delta_rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    output_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
    normalized = x_f32 * jax.lax.rsqrt(variance + eps)
    return (normalized * (1.0 + weight.astype(jnp.float32))).astype(output_dtype)


def _current_equation_reference(
    hidden: jax.Array,
    input_norm_weight: jax.Array,
    q_norm_weight: jax.Array,
    k_norm_weight: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    positions: jax.Array,
    *,
    config: Qwen35QKVStageConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compose the existing SkyRL operations at the proposed stage boundary."""
    normalized_hidden = _reference_delta_rms_norm(
        hidden, input_norm_weight, config.rms_norm_eps
    )
    fused = normalized_hidden @ frozen_weight
    fused = fused + ((normalized_hidden @ lora_a) @ lora_b) * lora_scaling
    q_raw, k_raw, v_raw = FusedLoRALinear.split(
        fused,
        group_sizes=(
            config.queries_per_kv * 2 * config.head_dim,
            config.head_dim,
            config.head_dim,
        ),
    )

    batch, sequence = hidden.shape[:2]
    q_and_gate = q_raw.reshape(
        batch, sequence, config.num_query_heads, 2 * config.head_dim
    )
    q_raw, gate = jnp.split(q_and_gate, 2, axis=-1)
    k_raw = k_raw.reshape(batch, sequence, config.num_kv_heads, config.head_dim)
    v = v_raw.reshape(batch, sequence, config.num_kv_heads, config.head_dim)
    q = _reference_delta_rms_norm(q_raw, q_norm_weight, config.rms_norm_eps)
    k = _reference_delta_rms_norm(k_raw, k_norm_weight, config.rms_norm_eps)
    q, k = apply_partial_rope(q, k, positions, config.rotary_dim, config.rope_theta)
    gate = gate.reshape(batch, sequence, config.num_query_heads * config.head_dim)
    return q, k, v, gate


def _stage(inputs: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    return qwen35_qkv_lora_rope(*inputs, config=_config())


def _reference(inputs: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    return _current_equation_reference(*inputs, config=_config())


def _weighted_loss(
    function: Callable[..., tuple[jax.Array, ...]],
    differentiable: tuple[jax.Array, ...],
    frozen_weight: jax.Array,
    positions: jax.Array,
    cotangents: tuple[jax.Array, ...],
) -> jax.Array:
    hidden, input_weight, q_weight, k_weight, lora_a, lora_b, scaling = differentiable
    outputs = function(
        hidden,
        input_weight,
        q_weight,
        k_weight,
        frozen_weight,
        lora_a,
        lora_b,
        scaling,
        positions,
        config=_config(),
    )
    return sum(
        jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32))
        for output, cotangent in zip(outputs, cotangents, strict=True)
    )


def test_forward_matches_current_qwen35_equations_and_interleaving() -> None:
    inputs = _inputs()
    actual = _stage(inputs)
    expected = _reference(inputs)
    expected_shapes = ((2, 3, 4, 4), (2, 3, 2, 4), (2, 3, 2, 4), (2, 3, 16))
    assert tuple(value.shape for value in actual) == expected_shapes
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_value, expected_value, rtol=1e-6, atol=1e-6)


def test_custom_vjp_matches_reference_for_input_norms_and_lora() -> None:
    inputs = _inputs()
    (
        hidden,
        input_weight,
        q_weight,
        k_weight,
        frozen_weight,
        lora_a,
        lora_b,
        scaling,
        positions,
    ) = inputs
    differentiable = (hidden, input_weight, q_weight, k_weight, lora_a, lora_b, scaling)
    output_shapes = jax.eval_shape(lambda: _stage(inputs))
    cotangent_keys = jax.random.split(jax.random.key(91), len(output_shapes))
    cotangents = tuple(
        jax.random.normal(key, shape.shape, dtype=jnp.float32)
        for key, shape in zip(cotangent_keys, output_shapes, strict=True)
    )

    actual_gradients = jax.grad(
        lambda *args: _weighted_loss(
            qwen35_qkv_lora_rope,
            args,
            frozen_weight,
            positions,
            cotangents,
        ),
        argnums=tuple(range(len(differentiable))),
    )(*differentiable)
    expected_gradients = jax.grad(
        lambda *args: _weighted_loss(
            _current_equation_reference,
            args,
            frozen_weight,
            positions,
            cotangents,
        ),
        argnums=tuple(range(len(differentiable))),
    )(*differentiable)

    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_custom_vjp_freezes_base_weight_but_not_lora() -> None:
    inputs = _inputs()
    (
        hidden,
        input_weight,
        q_weight,
        k_weight,
        frozen_weight,
        lora_a,
        lora_b,
        scaling,
        positions,
    ) = inputs

    def actual_loss(weight: jax.Array, a: jax.Array) -> jax.Array:
        outputs = qwen35_qkv_lora_rope(
            hidden,
            input_weight,
            q_weight,
            k_weight,
            weight,
            a,
            lora_b,
            scaling,
            positions,
            config=_config(),
        )
        return sum(jnp.sum(output.astype(jnp.float32) ** 2) for output in outputs)

    dweight, dlora_a = jax.grad(actual_loss, argnums=(0, 1))(frozen_weight, lora_a)
    np.testing.assert_array_equal(dweight, jnp.zeros_like(dweight))
    assert float(jnp.linalg.norm(dlora_a)) > 0

    reference_dweight = jax.grad(
        lambda weight: sum(
            jnp.sum(output.astype(jnp.float32) ** 2)
            for output in _current_equation_reference(
                hidden,
                input_weight,
                q_weight,
                k_weight,
                weight,
                lora_a,
                lora_b,
                scaling,
                positions,
                config=_config(),
            )
        )
    )(frozen_weight)
    assert float(jnp.linalg.norm(reference_dweight)) > 0


def test_bfloat16_forward_is_exact_and_precision_policy_gradients_are_close() -> None:
    inputs = _inputs(jnp.bfloat16)
    expected = _reference(inputs)

    def loss(*args: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        outputs = qwen35_qkv_lora_rope(*args, config=_config())
        scalar = sum(jnp.mean(output.astype(jnp.float32) ** 2) for output in outputs)
        return scalar, outputs

    gradient_argnums = (0, 1, 2, 3, 5, 6, 7)
    gradient_names = (
        "hidden",
        "input_norm_weight",
        "q_norm_weight",
        "k_norm_weight",
        "lora_a",
        "lora_b",
        "scaling",
    )
    (actual_loss, actual), gradients = jax.jit(
        jax.value_and_grad(loss, argnums=gradient_argnums, has_aux=True)
    )(*inputs)
    assert jnp.isfinite(actual_loss)
    assert all(output.dtype == jnp.bfloat16 for output in actual)
    assert all(jnp.all(jnp.isfinite(gradient)) for gradient in gradients)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            actual_value.astype(jnp.float32), expected_value.astype(jnp.float32)
        )

    def reference_loss(*args: jax.Array) -> jax.Array:
        outputs = _current_equation_reference(*args, config=_config())
        return sum(jnp.mean(output.astype(jnp.float32) ** 2) for output in outputs)

    expected_gradients = jax.grad(reference_loss, argnums=gradient_argnums)(*inputs)
    precision_limits = {
        # These are explicit acceptance gates for the deliberate FP32-backward
        # policy, not claims of BF16 bit-equivalence.  They cover every VJP
        # output even though SkyRL optimizes only LoRA A/B.
        "hidden": (0.012, 0.9999),
        "input_norm_weight": (0.008, 0.9999),
        "q_norm_weight": (0.006, 0.9999),
        "k_norm_weight": (0.006, 0.9999),
        "lora_a": (0.010, 0.9999),
        "lora_b": (0.010, 0.9999),
        "scaling": (0.008, 0.9999),
    }
    any_gradient_differs = False
    for name, actual_gradient, expected_gradient in zip(
        gradient_names, gradients, expected_gradients, strict=True
    ):
        actual_f32 = actual_gradient.astype(jnp.float32)
        expected_f32 = expected_gradient.astype(jnp.float32)
        relative_l2 = jnp.linalg.norm(actual_f32 - expected_f32) / jnp.maximum(
            jnp.linalg.norm(expected_f32), 1e-12
        )
        cosine = jnp.vdot(actual_f32.reshape(-1), expected_f32.reshape(-1)) / (
            jnp.linalg.norm(actual_f32) * jnp.linalg.norm(expected_f32)
        )
        max_relative_l2, min_cosine = precision_limits[name]
        assert float(relative_l2) < max_relative_l2, name
        assert float(cosine) > min_cosine, name
        any_gradient_differs |= not bool(
            jnp.array_equal(actual_gradient, expected_gradient)
        )

    # Guard the experiment's stated contract: the forward is exact for this
    # BF16 fixture, while the strengthened backward is intentionally distinct.
    assert any_gradient_differs


def test_residual_storage_contract_is_not_a_peak_memory_claim() -> None:
    inputs = _inputs()
    output, residual = _qwen35_qkv_lora_rope_fwd(*inputs, _config())
    raw_projection_shape = (*inputs[0].shape[:2], _config().fused_width)

    # This only verifies the Python residual contract.  It deliberately makes
    # no assertion about compiler buffer assignment, whole-layer remat, or peak
    # device memory.
    assert len(output) == 4
    assert len(residual) == len(inputs)
    assert all(value.shape != raw_projection_shape for value in residual)


def test_input_validation_rejects_incompatible_stage_geometry() -> None:
    with pytest.raises(ValueError, match="rotary_dim"):
        Qwen35QKVStageConfig(4, 2, 4, 3, 10_000.0, 1e-6)

    inputs = list(_inputs())
    inputs[6] = inputs[6][..., :-1]
    with pytest.raises(ValueError, match="lora_b output width"):
        _stage(tuple(inputs))

    inputs = list(_inputs())
    inputs[4] = inputs[4].astype(jnp.bfloat16)
    with pytest.raises(TypeError, match="hidden/model dtype"):
        _stage(tuple(inputs))
