from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyrl.tx.kernels.rocm.bf16_rms_gate_up_lora_swiglu_contiguous import (
    BF16_LORA_RANK,
    PRODUCTION_IN_FEATURES,
    PRODUCTION_PHYSICAL_FEATURES,
    PRODUCTION_X_SHAPE,
    RMS_NORM_EPSILON,
    _forward_with_residuals,
    _fp32_swiglu_from_interleaved,
    _fp32_swiglu_pullback_interleaved,
    _is_power_of_two,
    _rms_norm_pullback,
    _validate_inputs,
    bf16_rms_gate_up_lora_swiglu_contiguous,
)


@pytest.fixture(autouse=True)
def _force_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _case(*, rows: int = 16, inputs: int = 64, products: int = 64):
    x = jax.random.normal(jax.random.key(1), (1, rows, inputs), dtype=jnp.bfloat16)
    rms_delta = (jax.random.normal(jax.random.key(2), (inputs,), dtype=jnp.bfloat16) * 0.05).astype(jnp.bfloat16)
    frozen_weight = (jax.random.normal(jax.random.key(3), (inputs, 2 * products), dtype=jnp.bfloat16) * 0.1).astype(
        jnp.bfloat16
    )
    lora_a = (jax.random.normal(jax.random.key(4), (inputs, BF16_LORA_RANK), dtype=jnp.bfloat16) * 0.1).astype(
        jnp.bfloat16
    )
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


def _stage_reference(
    x: jax.Array,
    rms_delta: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    scaling: jax.Array,
    *,
    fp32_epilogue: bool = True,
) -> jax.Array:
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
    normalized = x_f32 * jax.lax.rsqrt(variance + RMS_NORM_EPSILON)
    normalized = (normalized * (1.0 + rms_delta.astype(jnp.float32))).astype(jnp.bfloat16)
    flat_normalized = normalized.reshape((-1, normalized.shape[-1]))
    z = (flat_normalized @ lora_a).astype(jnp.bfloat16)
    base = (flat_normalized @ frozen_weight).astype(jnp.bfloat16)
    low_rank = (z @ lora_b).astype(jnp.bfloat16)
    fused = (base + (low_rank * scaling).astype(jnp.bfloat16)).astype(jnp.bfloat16)
    pairs = fused.reshape((fused.shape[0], fused.shape[1] // 2, 2))
    gate, up = jnp.unstack(pairs, axis=-1)
    if fp32_epilogue:
        product = (jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)).astype(jnp.bfloat16)
    else:
        product = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
    return product.reshape((*x.shape[:-1], gate.shape[-1]))


def _candidate_contract_reference(*args: jax.Array) -> jax.Array:
    return _stage_reference(*args, fp32_epilogue=True)


def _model_bf16_reference(*args: jax.Array) -> jax.Array:
    return _stage_reference(*args, fp32_epilogue=False)


def _relative_l2(actual: jax.Array, expected: jax.Array) -> float:
    actual_f32 = actual.astype(jnp.float32)
    expected_f32 = expected.astype(jnp.float32)
    return float(jnp.linalg.norm(actual_f32 - expected_f32) / jnp.maximum(jnp.linalg.norm(expected_f32), 1e-12))


def _cosine(actual: jax.Array, expected: jax.Array) -> float:
    actual_f32 = actual.astype(jnp.float32).reshape(-1)
    expected_f32 = expected.astype(jnp.float32).reshape(-1)
    return float(
        jnp.vdot(actual_f32, expected_f32)
        / jnp.maximum(jnp.linalg.norm(actual_f32) * jnp.linalg.norm(expected_f32), 1e-12)
    )


def _candidate(
    x: jax.Array,
    rms_delta: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    scaling: jax.Array,
) -> jax.Array:
    return bf16_rms_gate_up_lora_swiglu_contiguous(
        x,
        rms_delta,
        frozen_weight,
        lora_a,
        lora_b,
        scaling,
        enabled=True,
        interpret=True,
        block_m=16,
        block_physical_n=64,
        block_k=32,
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
        bf16_rms_gate_up_lora_swiglu_contiguous(
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
        bf16_rms_gate_up_lora_swiglu_contiguous(
            x,
            rms_delta,
            weight,
            lora_a,
            lora_b,
            scaling,
            **kwargs,
        )


def test_noninterpreted_validation_requires_exact_production_shapes() -> None:
    x, rms_delta, weight, lora_a, lora_b, scaling = _case()

    with pytest.raises(ValueError, match="exact Qwen3.5 B1/T64 shapes"):
        bf16_rms_gate_up_lora_swiglu_contiguous(
            x,
            rms_delta,
            weight,
            lora_a,
            lora_b,
            scaling,
            enabled=True,
            interpret=False,
        )


def test_exact_production_geometry_passes_static_validation() -> None:
    bf16 = jnp.bfloat16
    _validate_inputs(
        jax.ShapeDtypeStruct(PRODUCTION_X_SHAPE, bf16),
        jax.ShapeDtypeStruct((PRODUCTION_IN_FEATURES,), bf16),
        jax.ShapeDtypeStruct((PRODUCTION_IN_FEATURES, PRODUCTION_PHYSICAL_FEATURES), bf16),
        jax.ShapeDtypeStruct((PRODUCTION_IN_FEATURES, BF16_LORA_RANK), bf16),
        jax.ShapeDtypeStruct((BF16_LORA_RANK, PRODUCTION_PHYSICAL_FEATURES), bf16),
        jax.ShapeDtypeStruct((), bf16),
        interpret=False,
        block_m=16,
        block_physical_n=64,
        block_k=32,
    )


def test_exact_production_trace_has_safe_grids_and_compiler_parameters() -> None:
    bf16 = jnp.bfloat16

    def operation(x, rms_delta, weight, lora_a, lora_b, scaling):
        return bf16_rms_gate_up_lora_swiglu_contiguous(
            x,
            rms_delta,
            weight,
            lora_a,
            lora_b,
            scaling,
            enabled=True,
            interpret=False,
        )

    closed = jax.make_jaxpr(operation)(
        jax.ShapeDtypeStruct(PRODUCTION_X_SHAPE, bf16),
        jax.ShapeDtypeStruct((PRODUCTION_IN_FEATURES,), bf16),
        jax.ShapeDtypeStruct((PRODUCTION_IN_FEATURES, PRODUCTION_PHYSICAL_FEATURES), bf16),
        jax.ShapeDtypeStruct((PRODUCTION_IN_FEATURES, BF16_LORA_RANK), bf16),
        jax.ShapeDtypeStruct((BF16_LORA_RANK, PRODUCTION_PHYSICAL_FEATURES), bf16),
        jax.ShapeDtypeStruct((), bf16),
    )
    calls = {
        equation.params["name"]: equation
        for equation in _equations_recursive(closed.jaxpr)
        if equation.primitive.name == "pallas_call"
    }

    stage_one = calls["skyrl_qwen35_bf16_rms_materialize_lora_a_forward"]
    stage_two = calls["skyrl_qwen35_bf16_contiguous_gate_up_lora_swiglu_forward"]
    assert stage_one.params["grid_mapping"].grid == (4,)
    assert stage_two.params["grid_mapping"].grid == (4, 288)
    for equation in (stage_one, stage_two):
        compiler = equation.params["compiler_params"]
        assert compiler.num_warps == 4
        assert compiler.num_stages == 1


@pytest.mark.parametrize("value", [16 * 32, 32 * 64, 16 * 64, 8 * 64, 16 * 32])
def test_first_compile_tile_operation_sizes_are_powers_of_two(value: int) -> None:
    assert _is_power_of_two(value)


@pytest.mark.parametrize("block_physical_n", [64, 128])
def test_interpret_forward_matches_bf16_reference(block_physical_n: int) -> None:
    x, rms_delta, weight, lora_a, lora_b, scaling = _case()

    actual = bf16_rms_gate_up_lora_swiglu_contiguous(
        x,
        rms_delta,
        weight,
        lora_a,
        lora_b,
        scaling,
        enabled=True,
        interpret=True,
        block_m=16,
        block_physical_n=block_physical_n,
        block_k=64,
    )
    expected = _candidate_contract_reference(x, rms_delta, weight, lora_a, lora_b, scaling)

    assert actual.dtype == jnp.bfloat16
    assert actual.shape == (1, 16, 64)
    assert _relative_l2(actual, expected) < 0.01


@pytest.mark.parametrize(("scale", "zero_b"), [(0.0, False), (4.0, True), (32.0, False)])
def test_interpret_custom_vjp_matches_candidate_contract(scale: float, zero_b: bool) -> None:
    x, rms_delta, weight, lora_a, lora_b, _ = _case()
    if zero_b:
        lora_b = jnp.zeros_like(lora_b)
    scaling = jnp.asarray(scale, dtype=jnp.bfloat16)
    cotangent = jax.random.normal(jax.random.key(6), (1, 16, 64), dtype=jnp.bfloat16)

    def candidate(x_arg, a_arg, b_arg):
        return _candidate(x_arg, rms_delta, weight, a_arg, b_arg, scaling)

    def reference(x_arg, a_arg, b_arg):
        return _candidate_contract_reference(x_arg, rms_delta, weight, a_arg, b_arg, scaling)

    actual_output, actual_pullback = jax.vjp(candidate, x, lora_a, lora_b)
    expected_output, expected_pullback = jax.vjp(reference, x, lora_a, lora_b)
    actual_gradients = actual_pullback(cotangent)
    expected_gradients = expected_pullback(cotangent)

    assert _relative_l2(actual_output, expected_output) < 1e-5
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        assert actual.dtype == jnp.bfloat16
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("case", ["random", "zero", "saturated"])
def test_manual_fp32_swiglu_pullback_is_bitwise_exact(case: str) -> None:
    if case == "random":
        fused = jax.random.normal(jax.random.key(20), (4, 16), dtype=jnp.bfloat16)
        cotangent = jax.random.normal(jax.random.key(21), (4, 8), dtype=jnp.bfloat16)
    elif case == "zero":
        fused = jnp.zeros((4, 16), dtype=jnp.bfloat16)
        cotangent = jnp.zeros((4, 8), dtype=jnp.bfloat16)
    else:
        gate = jnp.asarray([-80, -20, -8, -1, 1, 8, 20, 80], dtype=jnp.bfloat16)
        up = jnp.asarray([8, -7, 6, -5, 4, -3, 2, -1], dtype=jnp.bfloat16)
        fused = jnp.stack((gate, up), axis=-1).reshape(1, 16).repeat(4, axis=0)
        cotangent = jnp.arange(32, dtype=jnp.float32).reshape(4, 8).astype(jnp.bfloat16)

    _, pullback = jax.vjp(_fp32_swiglu_from_interleaved, fused)
    (expected,) = pullback(cotangent)
    actual = _fp32_swiglu_pullback_interleaved(fused, cotangent)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("case", ["random", "zero", "tiny", "large"])
def test_manual_rms_pullback_is_bitwise_exact(case: str) -> None:
    if case == "random":
        x = jax.random.normal(jax.random.key(30), (2, 16), dtype=jnp.bfloat16)
    elif case == "zero":
        x = jnp.zeros((2, 16), dtype=jnp.bfloat16)
    elif case == "tiny":
        x = jnp.full((2, 16), 2**-10, dtype=jnp.bfloat16)
    else:
        x = jnp.linspace(-32, 32, 32, dtype=jnp.float32).reshape(2, 16).astype(jnp.bfloat16)
    rms_delta = jnp.linspace(-0.1, 0.1, 16, dtype=jnp.float32).astype(jnp.bfloat16)
    cotangent = jax.random.normal(jax.random.key(31), x.shape, dtype=jnp.bfloat16)

    def rms_forward(x_arg):
        x_f32 = x_arg.astype(jnp.float32)
        denominator = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + RMS_NORM_EPSILON
        return (
            x_f32 * jax.lax.rsqrt(denominator) * (jnp.asarray(1.0, jnp.float32) + rms_delta.astype(jnp.float32))
        ).astype(jnp.bfloat16)

    _, pullback = jax.vjp(rms_forward, x)
    (expected,) = pullback(cotangent)
    x_f32 = x.astype(jnp.float32)
    denominator = jnp.mean(x_f32 * x_f32, axis=-1) + RMS_NORM_EPSILON
    inverse_rms = jax.lax.rsqrt(denominator)
    actual = _rms_norm_pullback(x, rms_delta, cotangent, denominator, inverse_rms)

    np.testing.assert_array_equal(actual, expected)


def test_interpret_output_and_vjp_pass_model_bf16_promotion_gate() -> None:
    x, rms_delta, weight, lora_a, lora_b, _ = _case()
    scaling = jnp.asarray(4.0, dtype=jnp.bfloat16)
    cotangent = jax.random.normal(jax.random.key(6), (1, 16, 64), dtype=jnp.bfloat16)

    def candidate(x_arg, a_arg, b_arg):
        return _candidate(x_arg, rms_delta, weight, a_arg, b_arg, scaling)

    def model_reference(x_arg, a_arg, b_arg):
        return _model_bf16_reference(
            x_arg,
            rms_delta,
            weight,
            a_arg,
            b_arg,
            scaling,
        )

    actual_output, actual_pullback = jax.vjp(candidate, x, lora_a, lora_b)
    expected_output, expected_pullback = jax.vjp(model_reference, x, lora_a, lora_b)
    actual_gradients = actual_pullback(cotangent)
    expected_gradients = expected_pullback(cotangent)

    assert _relative_l2(actual_output, expected_output) < 0.01
    assert _cosine(actual_output, expected_output) >= 0.9999
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        assert _relative_l2(actual, expected) < 0.01


def test_checkpointed_output_and_full_gradient_tree_are_bitwise_repeatable() -> None:
    x, rms_delta, weight, lora_a, lora_b, scaling = _case()

    def objective(x_arg, a_arg, b_arg):
        output = _candidate(x_arg, rms_delta, weight, a_arg, b_arg, scaling)
        return output.astype(jnp.float32).sum(), output

    plain_step = jax.jit(jax.value_and_grad(objective, argnums=(0, 1, 2), has_aux=True))
    remat_step = jax.jit(jax.value_and_grad(jax.checkpoint(objective), argnums=(0, 1, 2), has_aux=True))
    plain = plain_step(x, lora_a, lora_b)
    first = remat_step(x, lora_a, lora_b)
    second = remat_step(x, lora_a, lora_b)
    plain_leaves = jax.tree.leaves(plain)
    first_leaves = jax.tree.leaves(first)
    second_leaves = jax.tree.leaves(second)

    assert all(value.dtype in (jnp.float32, jnp.bfloat16) for value in first_leaves)
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in first_leaves)
    for expected, left, right in zip(plain_leaves, first_leaves, second_leaves, strict=True):
        np.testing.assert_array_equal(left, expected)
        np.testing.assert_array_equal(right, expected)


def test_output_only_and_residual_forward_kernels_are_bitwise_equal() -> None:
    x, rms_delta, weight, lora_a, lora_b, scaling = _case()
    output_only = _candidate(x, rms_delta, weight, lora_a, lora_b, scaling)
    with_residuals, _ = _forward_with_residuals(
        x,
        rms_delta,
        weight,
        lora_a,
        lora_b,
        scaling,
        RMS_NORM_EPSILON,
        16,
        64,
        32,
        True,
    )

    np.testing.assert_array_equal(output_only, with_residuals)


def test_training_residual_is_bounded_and_does_not_retain_normalized_x() -> None:
    x, rms_delta, weight, lora_a, lora_b, scaling = _case()
    output, residuals = _forward_with_residuals(
        x,
        rms_delta,
        weight,
        lora_a,
        lora_b,
        scaling,
        RMS_NORM_EPSILON,
        16,
        64,
        32,
        True,
    )
    z, denominator, inverse_rms, fused = residuals

    assert output.shape == (1, 16, 64)
    assert [value.shape for value in residuals] == [(16, 8), (16,), (16,), (16, 128)]
    assert sum(value.size * value.dtype.itemsize for value in residuals) == 4_480
    production_residual_bytes = 64 * BF16_LORA_RANK * 2 + 64 * 4 + 64 * 4 + 64 * PRODUCTION_PHYSICAL_FEATURES * 2
    assert production_residual_bytes == 2_360_832
    assert z.dtype == fused.dtype == jnp.bfloat16
    assert denominator.dtype == inverse_rms.dtype == jnp.float32


def test_backward_trace_has_five_gemms_and_no_dense_forward_recompute() -> None:
    x, rms_delta, weight, lora_a, lora_b, scaling = _case()

    def objective(x_arg, a_arg, b_arg):
        return _candidate(x_arg, rms_delta, weight, a_arg, b_arg, scaling).astype(jnp.float32).sum()

    closed = jax.make_jaxpr(jax.grad(objective, argnums=(0, 1, 2)))(x, lora_a, lora_b)
    dot_shapes = [
        tuple(tuple(variable.aval.shape) for variable in equation.invars)
        for equation in _equations_recursive(closed.jaxpr)
        if equation.primitive.name == "dot_general"
    ]

    assert len(dot_shapes) == 6
    assert ((16, 32), (32, 64)) in dot_shapes  # The single tiled forward dot.
    for backward_shape in (
        ((16, 128), (128, 8)),
        ((64, 16), (16, 8)),
        ((8, 16), (16, 128)),
        ((16, 128), (128, 64)),
        ((16, 8), (8, 64)),
    ):
        assert backward_shape in dot_shapes
    assert ((16, 64), (64, 128)) not in dot_shapes
    assert ((16, 8), (8, 128)) not in dot_shapes


def test_trace_materializes_normalized_x_then_uses_physical_tiles() -> None:
    x, rms_delta, weight, lora_a, lora_b, scaling = _case()

    closed = jax.make_jaxpr(
        lambda x_arg: bf16_rms_gate_up_lora_swiglu_contiguous(
            x_arg,
            rms_delta,
            weight,
            lora_a,
            lora_b,
            scaling,
            enabled=True,
            interpret=True,
            block_m=16,
            block_physical_n=64,
            block_k=64,
        )
    )(x)
    calls = {
        equation.params["name"]: equation
        for equation in _equations_recursive(closed.jaxpr)
        if equation.primitive.name == "pallas_call"
    }

    assert set(calls) == {
        "skyrl_qwen35_bf16_rms_materialize_lora_a_forward",
        "skyrl_qwen35_bf16_contiguous_gate_up_lora_swiglu_forward",
    }
    stage_one = calls["skyrl_qwen35_bf16_rms_materialize_lora_a_forward"]
    stage_two = calls["skyrl_qwen35_bf16_contiguous_gate_up_lora_swiglu_forward"]
    assert stage_one.params["grid_mapping"].grid == (1,)
    assert stage_two.params["grid_mapping"].grid == (1, 2)
    assert [tuple(variable.aval.shape) for variable in stage_one.outvars] == [
        (16, 64),
        (16, BF16_LORA_RANK),
        (16,),
        (16,),
    ]
    assert [tuple(variable.aval.shape) for variable in stage_two.outvars] == [(16, 64)]
    kernel_jaxpr = getattr(stage_two.params["jaxpr"], "jaxpr", stage_two.params["jaxpr"])
    kernel_equations = tuple(_equations_recursive(kernel_jaxpr))
    primitive_names = {equation.primitive.name for equation in kernel_equations}
    assert not primitive_names.intersection({"slice", "gather", "dynamic_slice", "concatenate"})
    assert {"reshape", "unstack"}.issubset(primitive_names)
    assert sum(equation.primitive.name == "dot_general" for equation in kernel_equations) == 1


def test_interleaved_epilogue_pairs_adjacent_gate_then_up_columns() -> None:
    x = jnp.ones((1, 16, 64), dtype=jnp.bfloat16)
    rms_delta = jnp.zeros((64,), dtype=jnp.bfloat16)
    weight = jnp.zeros((64, 128), dtype=jnp.bfloat16)
    weight = weight.at[:, 0::2].set(jnp.asarray(0.02, dtype=jnp.bfloat16))
    weight = weight.at[:, 1::2].set(jnp.asarray(0.03, dtype=jnp.bfloat16))
    lora_a = jnp.zeros((64, BF16_LORA_RANK), dtype=jnp.bfloat16)
    lora_b = jnp.zeros((BF16_LORA_RANK, 128), dtype=jnp.bfloat16)
    scaling = jnp.asarray(1.0, dtype=jnp.bfloat16)

    actual = bf16_rms_gate_up_lora_swiglu_contiguous(
        x,
        rms_delta,
        weight,
        lora_a,
        lora_b,
        scaling,
        enabled=True,
        interpret=True,
        block_m=16,
        block_physical_n=64,
        block_k=64,
    )
    expected = _candidate_contract_reference(x, rms_delta, weight, lora_a, lora_b, scaling)

    assert _relative_l2(actual, expected) < 0.01
    np.testing.assert_array_equal(actual, jnp.full_like(actual, actual[0, 0, 0]))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("partial_row_tile", "must be divisible by block_m"),
        ("odd_physical_width", "must be even"),
        ("wrong_rank", "lora_a must have shape"),
    ],
)
def test_invalid_interpreter_geometry_rejects_before_execution(mutation: str, match: str) -> None:
    x, rms_delta, weight, lora_a, lora_b, scaling = _case()
    if mutation == "partial_row_tile":
        x = x[:, :-1]
    elif mutation == "odd_physical_width":
        weight = weight[:, :-1]
        lora_b = lora_b[:, :-1]
    else:
        lora_a = lora_a[:, :-1]

    with pytest.raises(ValueError, match=match):
        bf16_rms_gate_up_lora_swiglu_contiguous(
            x,
            rms_delta,
            weight,
            lora_a,
            lora_b,
            scaling,
            enabled=True,
            interpret=True,
            block_m=16,
            block_physical_n=64,
            block_k=64,
        )
