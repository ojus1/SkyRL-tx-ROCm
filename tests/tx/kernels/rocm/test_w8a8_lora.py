from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyrl.tx.kernels.quantized_lora import (
    GroupQuantizedWeight,
    dequantize_frozen_weight,
    quantize_frozen_weight,
    quantized_frozen_linear,
    quantized_lora_linear,
)
from skyrl.tx.kernels.rocm.w8a8_lora import (
    W8A8_GROUP_SIZE,
    W8A8_LORA_RANK,
    _pad_output_features,
    _padded_output_features,
    _w8a8_lora_linear_fwd,
    w8a8_frozen_linear,
    w8a8_lora_linear,
)


@pytest.fixture(autouse=True)
def _force_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _case(*, rows: int = 3, inputs: int = 64, outputs: int = 17):
    x = jax.random.normal(jax.random.key(1), (rows, inputs), dtype=jnp.bfloat16)
    base = jax.random.normal(jax.random.key(2), (inputs, outputs), dtype=jnp.bfloat16)
    weight = quantize_frozen_weight(
        base,
        bits=8,
        group_size=W8A8_GROUP_SIZE,
        scale_dtype=jnp.bfloat16,
    )
    lora_a = (
        jax.random.normal(
            jax.random.key(3), (inputs, W8A8_LORA_RANK), dtype=jnp.bfloat16
        )
        * 0.1
    ).astype(jnp.bfloat16)
    lora_b = (
        jax.random.normal(
            jax.random.key(4), (W8A8_LORA_RANK, outputs), dtype=jnp.bfloat16
        )
        * 0.1
    ).astype(jnp.bfloat16)
    return x, weight, lora_a, lora_b


def _relative_l2(actual, expected) -> float:
    actual = actual.astype(jnp.float32)
    expected = expected.astype(jnp.float32)
    return float(
        jnp.linalg.norm(actual - expected)
        / jnp.maximum(jnp.linalg.norm(expected), 1e-12)
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
    x, weight, lora_a, lora_b = _case()

    with pytest.raises(RuntimeError, match="disabled by default"):
        w8a8_lora_linear(x, weight, lora_a, lora_b, 1.0, interpret=True)
    with pytest.raises(RuntimeError, match="disabled by default"):
        w8a8_frozen_linear(x, weight, interpret=True)


@pytest.mark.parametrize(("name", "value"), [("enabled", 1), ("interpret", 0)])
def test_opt_in_flags_require_exact_boole(name: str, value: object) -> None:
    x, weight, lora_a, lora_b = _case()
    kwargs = {"enabled": True, "interpret": True, name: value}
    with pytest.raises(TypeError, match=f"{name} must be an exact bool"):
        w8a8_lora_linear(x, weight, lora_a, lora_b, 1.0, **kwargs)


@pytest.mark.parametrize("inputs", [64, 128])
@pytest.mark.parametrize("outputs", [17, 19, 31, 33])
def test_interpret_forward_is_bitwise_equal_to_grouped_oracle_with_tails(
    inputs: int, outputs: int
) -> None:
    x, weight, lora_a, lora_b = _case(rows=3, inputs=inputs, outputs=outputs)

    actual = w8a8_lora_linear(
        x,
        weight,
        lora_a,
        lora_b,
        0.75,
        enabled=True,
        interpret=True,
        block_m=16,
        block_n=16,
    )
    expected = quantized_lora_linear(
        x,
        weight,
        lora_a,
        lora_b,
        0.75,
        activation_bits=8,
    )

    assert actual.shape == (3, outputs)
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(actual, expected)


def test_output_feature_padding_is_identity_when_aligned() -> None:
    value = jnp.zeros((64, 16), dtype=jnp.int8)

    assert _padded_output_features(16, 16) == 16
    assert _pad_output_features(value, 16) is value


@pytest.mark.parametrize(
    ("out_features", "block_n", "expected"),
    [(2560, 64, 2560), (18432, 64, 18432), (32, 64, 64), (32, 32, 32)],
)
def test_qwen_output_feature_geometry(
    out_features: int, block_n: int, expected: int
) -> None:
    assert _padded_output_features(out_features, block_n) == expected


def test_tail_forward_and_input_vjp_use_full_unmasked_physical_tiles() -> None:
    x, weight, lora_a, lora_b = _case(rows=3, inputs=64, outputs=17)

    def objective(x_arg, a_arg, b_arg):
        output = w8a8_lora_linear(
            x_arg,
            weight,
            a_arg,
            b_arg,
            1.0,
            enabled=True,
            interpret=True,
            block_m=16,
            block_n=16,
            row_superblock=16,
        )
        return jnp.sum(output.astype(jnp.float32))

    closed = jax.make_jaxpr(jax.grad(objective, argnums=(0, 1, 2)))(x, lora_a, lora_b)
    calls = {
        equation.params["name"]: equation
        for equation in _equations_recursive(closed.jaxpr)
        if equation.primitive.name == "pallas_call"
    }
    forward = calls["skyrl_qwen35_w8a8_lora_forward"]
    input_vjp = calls["skyrl_qwen35_w8a16_lora_input_vjp"]

    assert forward.params["grid_mapping"].grid == (1, 2)
    assert [value.aval.shape for value in forward.invars] == [
        (16, 64),
        (16, 1),
        (64, 32),
        (1, 32),
        (16, 8),
        (8, 32),
        (),
    ]
    assert [value.aval.shape for value in forward.outvars] == [(16, 32)]
    assert input_vjp.params["grid_mapping"].grid == (1, 1)
    assert [value.aval.shape for value in input_vjp.invars] == [
        (16, 32),
        (64, 32),
        (1, 32),
    ]
    assert [value.aval.shape for value in input_vjp.outvars] == [(16, 64)]

    for call in (forward, input_vjp):
        kernel = getattr(call.params["jaxpr"], "jaxpr", call.params["jaxpr"])
        primitives = {
            equation.primitive.name for equation in _equations_recursive(kernel)
        }
        assert primitives.isdisjoint({"lt", "and"})


def test_three_row_superblocks_match_forward_and_gradient_contract() -> None:
    x, weight, lora_a, lora_b = _case(rows=33, inputs=128, outputs=19)
    scaling = jnp.asarray(0.75, dtype=jnp.float32)
    cotangent = jax.random.normal(jax.random.key(7), (33, 19), dtype=jnp.bfloat16)

    def candidate(x_arg, a_arg, b_arg, scaling_arg):
        output = w8a8_lora_linear(
            x_arg,
            weight,
            a_arg,
            b_arg,
            scaling_arg,
            enabled=True,
            interpret=True,
            block_m=16,
            block_n=16,
            row_superblock=16,
        )
        objective = jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32))
        return objective, output

    def oracle(x_arg, a_arg, b_arg, scaling_arg):
        output = quantized_lora_linear(
            x_arg,
            weight,
            a_arg,
            b_arg,
            scaling_arg,
            activation_bits=8,
        )
        objective = jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32))
        return objective, output

    actual_objective, actual_pullback, actual_output = jax.vjp(
        candidate, x, lora_a, lora_b, scaling, has_aux=True
    )
    expected_objective, expected_pullback, expected_output = jax.vjp(
        oracle, x, lora_a, lora_b, scaling, has_aux=True
    )
    del actual_objective, expected_objective
    np.testing.assert_array_equal(actual_output, expected_output)

    actual = actual_pullback(jnp.asarray(1.0, dtype=jnp.float32))
    expected = expected_pullback(jnp.asarray(1.0, dtype=jnp.float32))
    errors = tuple(_relative_l2(left, right) for left, right in zip(actual, expected))
    assert errors[0] < 0.01
    assert errors[1] < 0.01
    assert errors[2] < 0.01
    assert errors[3] < 0.01


def test_base_only_route_matches_forward_and_relaxed_input_vjp_contract() -> None:
    x, weight, _, _ = _case(rows=3, inputs=128, outputs=19)
    cotangent = jax.random.normal(jax.random.key(6), (3, 19), dtype=jnp.bfloat16)

    actual = w8a8_frozen_linear(
        x,
        weight,
        enabled=True,
        interpret=True,
        block_m=16,
        block_n=16,
        row_superblock=16,
    )
    expected = quantized_frozen_linear(x, weight, activation_bits=8)
    np.testing.assert_array_equal(actual, expected)

    def objective(value):
        output = w8a8_frozen_linear(
            value,
            weight,
            enabled=True,
            interpret=True,
            block_m=16,
            block_n=16,
            row_superblock=16,
        )
        return jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32))

    actual_dx = jax.grad(objective)(x)
    expected_dx = (
        cotangent.astype(jnp.float32)
        @ dequantize_frozen_weight(weight, dtype=jnp.float32).T
    ).astype(jnp.bfloat16)
    assert _relative_l2(actual_dx, expected_dx) < 0.01


def test_interpret_forward_is_jittable_and_zero_groups_stay_finite() -> None:
    x = jnp.zeros((2, 64), dtype=jnp.bfloat16)
    weight = quantize_frozen_weight(
        jnp.zeros((64, 19), dtype=jnp.bfloat16),
        bits=8,
        group_size=64,
        scale_dtype=jnp.bfloat16,
    )
    lora_a = jnp.ones((64, 8), dtype=jnp.bfloat16)
    lora_b = jnp.zeros((8, 19), dtype=jnp.bfloat16)
    run = jax.jit(
        lambda value: w8a8_lora_linear(
            value,
            weight,
            lora_a,
            lora_b,
            1.0,
            enabled=True,
            interpret=True,
            block_m=16,
            block_n=16,
        )
    )

    output = run(x)

    assert bool(jnp.all(jnp.isfinite(output)))
    np.testing.assert_array_equal(output, jnp.zeros_like(output))


def test_1025_rows_trace_as_three_512_row_scan_iterations() -> None:
    x, weight, lora_a, lora_b = _case(rows=1025, outputs=64)

    closed = jax.make_jaxpr(
        lambda value: w8a8_lora_linear(
            value,
            weight,
            lora_a,
            lora_b,
            1.0,
            enabled=True,
            interpret=True,
            block_m=32,
            block_n=64,
            row_superblock=512,
        )
    )(x)
    equations = tuple(_equations_recursive(closed.jaxpr))
    row_scans = [
        equation
        for equation in equations
        if equation.primitive.name == "scan" and equation.params.get("length") == 3
    ]
    pallas_calls = [
        equation for equation in equations if equation.primitive.name == "pallas_call"
    ]

    assert len(row_scans) == 1
    assert len(pallas_calls) == 1
    grid = pallas_calls[0].params["grid_mapping"].grid
    assert grid == (16, 1)
    assert pallas_calls[0].params["name"] == "skyrl_qwen35_w8a8_lora_forward"


def test_relaxed_custom_vjp_stays_inside_one_percent_gate() -> None:
    x, weight, lora_a, lora_b = _case(rows=3, outputs=17)
    scaling = jnp.asarray(0.75, dtype=jnp.float32)
    cotangent = jax.random.normal(jax.random.key(5), (3, 17), dtype=jnp.bfloat16)

    def candidate(x_arg, a_arg, b_arg, scaling_arg):
        output = w8a8_lora_linear(
            x_arg,
            weight,
            a_arg,
            b_arg,
            scaling_arg,
            enabled=True,
            interpret=True,
            block_m=16,
            block_n=16,
        )
        return jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32))

    def oracle(x_arg, a_arg, b_arg, scaling_arg):
        output = quantized_lora_linear(
            x_arg,
            weight,
            a_arg,
            b_arg,
            scaling_arg,
            activation_bits=8,
        )
        return jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32))

    actual = jax.grad(candidate, argnums=(0, 1, 2, 3))(x, lora_a, lora_b, scaling)
    expected = jax.grad(oracle, argnums=(0, 1, 2, 3))(x, lora_a, lora_b, scaling)

    errors = tuple(_relative_l2(left, right) for left, right in zip(actual, expected))
    assert errors[0] < 0.01
    assert errors[1] < 0.01
    assert errors[2] < 0.01
    assert errors[3] < 0.01


def test_custom_vjp_residual_keeps_compact_weight_and_rank8_z() -> None:
    x, weight, lora_a, lora_b = _case(rows=5, inputs=128, outputs=31)
    output, residual = _w8a8_lora_linear_fwd(
        x,
        weight.codes,
        weight.scales,
        lora_a,
        lora_b,
        jnp.asarray(1.0, dtype=jnp.float32),
        64,
        16,
        16,
        16,
        True,
    )

    assert output.shape == (5, 31)
    _, saved_codes, saved_scales, _, _, _, z = residual
    assert saved_codes is weight.codes
    assert saved_scales is weight.scales
    assert z.shape == (5, 8)
    assert not any(
        getattr(value, "shape", None) == (128, 31)
        and jnp.issubdtype(getattr(value, "dtype", jnp.int8), jnp.floating)
        for value in residual
    )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda weight: GroupQuantizedWeight(
                weight.codes.astype(jnp.uint8),
                weight.scales,
                weight.original_in_features,
                weight.padded_in_features,
                weight.bits,
                weight.group_size,
            ),
            "INT8",
        ),
        (
            lambda weight: GroupQuantizedWeight(
                weight.codes,
                weight.scales.astype(jnp.float32),
                weight.original_in_features,
                weight.padded_in_features,
                weight.bits,
                weight.group_size,
            ),
            "BF16",
        ),
    ],
)
def test_storage_contract_fails_closed(mutator, message: str) -> None:
    x, weight, lora_a, lora_b = _case()
    with pytest.raises(TypeError, match=message):
        w8a8_lora_linear(
            x,
            mutator(weight),
            lora_a,
            lora_b,
            1.0,
            enabled=True,
            interpret=True,
        )


@pytest.mark.parametrize("row_superblock", [15, 48, 2064])
def test_row_superblock_contract_fails_before_pallas(row_superblock: int) -> None:
    x, weight, _, _ = _case()
    with pytest.raises(ValueError, match="row_superblock"):
        w8a8_frozen_linear(
            x,
            weight,
            enabled=True,
            interpret=True,
            block_m=32,
            row_superblock=row_superblock,
        )


@pytest.mark.parametrize(
    ("inputs", "outputs", "message"),
    [(9280, 1, "K must not exceed"), (64, 18433, "N must not exceed")],
)
def test_qwen_projection_dimension_caps_fail_closed(
    inputs: int, outputs: int, message: str
) -> None:
    weight = GroupQuantizedWeight(
        codes=jnp.zeros((inputs, outputs), dtype=jnp.int8),
        scales=jnp.ones((inputs // 64, outputs), dtype=jnp.bfloat16),
        original_in_features=inputs,
        padded_in_features=inputs,
        bits=8,
        group_size=64,
    )
    x = jnp.zeros((1, inputs), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match=message):
        w8a8_frozen_linear(
            x,
            weight,
            enabled=True,
            interpret=True,
        )


def test_source_has_no_capture_replay_or_graph_api() -> None:
    source = inspect.getsource(
        __import__("skyrl.tx.kernels.rocm.w8a8_lora", fromlist=["w8a8_lora_linear"])
    )
    forbidden = (
        "hipGraph",
        "cudaGraph",
        "command_buffer",
        "capture_begin",
        "capture_end",
    )
    assert not any(token in source for token in forbidden)
