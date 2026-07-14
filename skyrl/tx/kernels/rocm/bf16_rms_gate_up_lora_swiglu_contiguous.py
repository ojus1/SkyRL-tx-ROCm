"""CPU-semantics prototype for a contiguous gate/up Pallas projection.

This default-off experiment targets the exact Qwen3.5 B1/T64 MLP geometry:
``M=64``, ``K=2560``, and an interleaved gate/up physical width of ``18432``.
It is intentionally separate from the previously benchmarked strided
gate/up candidate.

The forward has two stages.  Stage one computes RMSNorm once and materializes
both the normalized BF16 activation and the rank-eight LoRA-A projection.
Stage two issues one Pallas dot per K tile against a contiguous physical-N
tile of the model's canonical adjacent ``(gate0, up0, gate1, up1, ...)``
storage; "contiguous" does not mean separate all-gate/all-up matrices. It adds
the paired LoRA-B projection at the model's BF16
boundaries, and writes the SwiGLU product. The training rule additionally
saves the BF16 gate/up projection, rank-eight ``z``, and two FP32 RMS vectors;
its backward reconstructs normalized X elementwise and uses five library
GEMMs without recomputing either forward projection. Smaller full-tile
geometries are accepted exclusively by Pallas's CPU interpreter for focused
semantic tests; non-interpreted calls require the exact production shapes.

This module is not wired into model code and makes no GPU compilation or
performance claim.
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

BF16_LORA_RANK = 8
RMS_NORM_EPSILON = 1e-6

PRODUCTION_X_SHAPE = (1, 64, 2560)
PRODUCTION_ROWS = 64
PRODUCTION_IN_FEATURES = 2560
PRODUCTION_PRODUCT_FEATURES = 9216
PRODUCTION_PHYSICAL_FEATURES = 2 * PRODUCTION_PRODUCT_FEATURES

DEFAULT_BLOCK_M = 16
DEFAULT_BLOCK_PHYSICAL_N = 64
DEFAULT_BLOCK_K = 32

_SUPPORTED_BLOCK_M = frozenset((16,))
_SUPPORTED_BLOCK_PHYSICAL_N = frozenset((64, 128))
_SUPPORTED_BLOCK_K = frozenset((32, 64))


def _validate_exact_bool(name: str, value: bool) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact bool")


def _validate_tile(name: str, value: int, supported: frozenset[int]) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value not in supported:
        raise ValueError(f"{name} must be one of {sorted(supported)}, got {value}")


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _validate_epsilon(eps: float) -> None:
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise TypeError("eps must be a real scalar")
    if float(eps) != RMS_NORM_EPSILON:
        raise ValueError(f"eps must be the Qwen3.5 value {RMS_NORM_EPSILON}, got {eps}")


def _validate_inputs(
    x: jax.Array,
    rms_delta: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    *,
    interpret: bool,
    block_m: int,
    block_physical_n: int,
    block_k: int,
) -> None:
    """Validate exact production geometry or interpreter-only full tiles."""

    _validate_tile("block_m", block_m, _SUPPORTED_BLOCK_M)
    _validate_tile("block_physical_n", block_physical_n, _SUPPORTED_BLOCK_PHYSICAL_N)
    _validate_tile("block_k", block_k, _SUPPORTED_BLOCK_K)

    if x.ndim < 2:
        raise ValueError(f"x must have a row domain and final K axis, got {x.shape}")
    row_count = math.prod(x.shape[:-1])
    if not 0 < row_count <= PRODUCTION_ROWS:
        raise ValueError(f"flattened M must be in [1, {PRODUCTION_ROWS}], got {row_count}")
    if row_count % block_m:
        raise ValueError(f"flattened M={row_count} must be divisible by block_m={block_m}")

    values = {
        "x": x,
        "rms_delta": rms_delta,
        "frozen_weight": frozen_weight,
        "lora_a": lora_a,
        "lora_b": lora_b,
        "lora_scaling": lora_scaling,
    }
    for name, value in values.items():
        if value.dtype != jnp.bfloat16:
            raise TypeError(f"{name} must be BF16, got {value.dtype}")

    if frozen_weight.ndim != 2:
        raise ValueError(
            "frozen_weight must have shape [K, 2 * N] with interleaved " f"gate/up columns, got {frozen_weight.shape}"
        )
    in_features, physical_features = frozen_weight.shape
    if not 0 < in_features <= PRODUCTION_IN_FEATURES:
        raise ValueError(f"K must be in [1, {PRODUCTION_IN_FEATURES}], got {in_features}")
    if in_features % block_k:
        raise ValueError(f"K={in_features} must be divisible by block_k={block_k}")
    if x.shape[-1] != in_features:
        raise ValueError(f"x K={x.shape[-1]} does not match weight K={in_features}")
    if rms_delta.shape != (in_features,):
        raise ValueError(f"rms_delta must have shape ({in_features},), got {rms_delta.shape}")

    if not 0 < physical_features <= PRODUCTION_PHYSICAL_FEATURES:
        raise ValueError(
            "physical gate/up width must be in " f"[2, {PRODUCTION_PHYSICAL_FEATURES}], got {physical_features}"
        )
    if physical_features % 2:
        raise ValueError(f"physical gate/up width must be even, got {physical_features}")
    if physical_features % block_physical_n:
        raise ValueError(f"physical N={physical_features} must be divisible by " f"block_physical_n={block_physical_n}")

    operation_sizes = {
        "normalized tile": block_m * block_k,
        "weight tile": block_k * block_physical_n,
        "accumulator tile": block_m * block_physical_n,
        "LoRA-B tile": BF16_LORA_RANK * block_physical_n,
        "product tile": block_m * (block_physical_n // 2),
    }
    for name, size in operation_sizes.items():
        if not _is_power_of_two(size):
            raise ValueError(f"{name} element count must be a power of two, got {size}")

    if lora_a.shape != (in_features, BF16_LORA_RANK):
        raise ValueError(f"lora_a must have shape ({in_features}, {BF16_LORA_RANK}), " f"got {lora_a.shape}")
    if lora_b.shape != (BF16_LORA_RANK, physical_features):
        raise ValueError(f"lora_b must have shape ({BF16_LORA_RANK}, {physical_features}), " f"got {lora_b.shape}")
    if lora_scaling.shape != ():
        raise ValueError(f"lora_scaling must be a BF16 scalar, got shape={lora_scaling.shape}")

    if not interpret:
        exact_shapes = (
            x.shape == PRODUCTION_X_SHAPE
            and frozen_weight.shape == (PRODUCTION_IN_FEATURES, PRODUCTION_PHYSICAL_FEATURES)
            and rms_delta.shape == (PRODUCTION_IN_FEATURES,)
            and lora_a.shape == (PRODUCTION_IN_FEATURES, BF16_LORA_RANK)
            and lora_b.shape == (BF16_LORA_RANK, PRODUCTION_PHYSICAL_FEATURES)
        )
        if not exact_shapes:
            raise ValueError(
                "non-interpreted execution requires exact Qwen3.5 B1/T64 "
                f"shapes: x={PRODUCTION_X_SHAPE}, "
                "rms_delta=(2560,), frozen_weight=(2560, 18432), "
                "lora_a=(2560, 8), and lora_b=(8, 18432)"
            )


def _compiler_params():
    from jax.experimental.pallas import triton as plgpu

    return plgpu.CompilerParams(num_warps=4, num_stages=1)


def _rms_materialize_lora_a_kernel(
    x_ref,
    rms_delta_ref,
    lora_a_ref,
    normalized_ref,
    z_ref,
    denominator_ref,
    inverse_rms_ref,
    *,
    block_m: int,
    block_k: int,
    eps: float,
) -> None:
    """Materialize normalized BF16 X and its rank-eight LoRA-A product."""

    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    row_start = pl.program_id(0) * block_m
    rows = row_start + jnp.arange(block_m)

    def consume_square(k_block, previous):
        k = k_block * block_k + jnp.arange(block_k)
        x = plgpu.load(x_ref.at[rows[:, None], k[None, :]]).astype(jnp.float32)
        return previous + jnp.sum(x * x, axis=1)

    squared_sum = lax.fori_loop(
        0,
        x_ref.shape[1] // block_k,
        consume_square,
        jnp.zeros((block_m,), dtype=jnp.float32),
    )
    denominator = squared_sum / x_ref.shape[1] + eps
    inverse_rms = lax.rsqrt(denominator)
    rank = jnp.arange(BF16_LORA_RANK)

    def normalize_and_consume_lora_a(k_block, previous):
        k = k_block * block_k + jnp.arange(block_k)
        x = plgpu.load(x_ref.at[rows[:, None], k[None, :]]).astype(jnp.float32)
        delta = plgpu.load(rms_delta_ref.at[k]).astype(jnp.float32)
        normalized = (x * inverse_rms[:, None] * (1.0 + delta[None, :])).astype(jnp.bfloat16)
        plgpu.store(normalized_ref.at[rows[:, None], k[None, :]], normalized)
        lora_a = plgpu.load(lora_a_ref.at[k[:, None], rank[None, :]]).astype(jnp.float32)
        return previous + jnp.sum(
            normalized.astype(jnp.float32)[:, :, None] * lora_a[None, :, :],
            axis=1,
        )

    z = lax.fori_loop(
        0,
        x_ref.shape[1] // block_k,
        normalize_and_consume_lora_a,
        jnp.zeros((block_m, BF16_LORA_RANK), dtype=jnp.float32),
    )
    plgpu.store(z_ref.at[rows[:, None], rank[None, :]], z.astype(jnp.bfloat16))
    plgpu.store(denominator_ref.at[rows], denominator)
    plgpu.store(inverse_rms_ref.at[rows], inverse_rms)


def _contiguous_gate_up_swiglu_values(
    normalized_ref,
    frozen_weight_ref,
    z_ref,
    lora_b_ref,
    lora_scaling_ref,
    *,
    block_m: int,
    block_physical_n: int,
    block_k: int,
) -> tuple[jax.Array, ...]:
    """Return one contiguous projection tile and its paired product."""

    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    row_start = pl.program_id(0) * block_m
    physical_start = pl.program_id(1) * block_physical_n
    rows = row_start + jnp.arange(block_m)
    physical_columns = physical_start + jnp.arange(block_physical_n)

    def consume_k(k_block, previous):
        k = k_block * block_k + jnp.arange(block_k)
        normalized = plgpu.load(normalized_ref.at[rows[:, None], k[None, :]])
        weight = plgpu.load(frozen_weight_ref.at[k[:, None], physical_columns[None, :]])
        # Unlike the rejected candidate, this is one dot over adjacent
        # physical gate/up columns, not separate even/odd strided dots.
        return previous + plgpu.dot(normalized, weight).astype(jnp.float32)

    base_accumulator = lax.fori_loop(
        0,
        normalized_ref.shape[1] // block_k,
        consume_k,
        jnp.zeros((block_m, block_physical_n), dtype=jnp.float32),
    )

    base = base_accumulator.astype(jnp.bfloat16)
    del base_accumulator
    low_rank = jnp.zeros((block_m, block_physical_n), dtype=jnp.float32)
    for rank_index in range(BF16_LORA_RANK):
        z_rank = plgpu.load(z_ref.at[rows, rank_index]).astype(jnp.float32)
        b_rank = plgpu.load(lora_b_ref.at[rank_index, physical_columns]).astype(jnp.float32)
        low_rank = low_rank + z_rank[:, None] * b_rank[None, :]

    scaling = lora_scaling_ref[...].astype(jnp.bfloat16)
    scaled_low_rank = (low_rank.astype(jnp.bfloat16) * scaling).astype(jnp.bfloat16)
    fused = (base + scaled_low_rank).astype(jnp.bfloat16)

    paired = fused.reshape((block_m, block_physical_n // 2, 2))
    gate, up = jnp.unstack(paired, axis=-1)
    product = (jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)).astype(jnp.bfloat16)
    return rows, physical_start, physical_columns, fused, product


def _contiguous_gate_up_swiglu_kernel(
    normalized_ref,
    frozen_weight_ref,
    z_ref,
    lora_b_ref,
    lora_scaling_ref,
    product_ref,
    *,
    block_m: int,
    block_physical_n: int,
    block_k: int,
) -> None:
    """Write only the paired product for an inference-style forward."""

    from jax.experimental.pallas import triton as plgpu

    rows, physical_start, _, _, product = _contiguous_gate_up_swiglu_values(
        normalized_ref,
        frozen_weight_ref,
        z_ref,
        lora_b_ref,
        lora_scaling_ref,
        block_m=block_m,
        block_physical_n=block_physical_n,
        block_k=block_k,
    )
    product_start = physical_start // 2
    product_columns = product_start + jnp.arange(block_physical_n // 2)
    plgpu.store(product_ref.at[rows[:, None], product_columns[None, :]], product)


def _contiguous_gate_up_swiglu_residual_kernel(
    normalized_ref,
    frozen_weight_ref,
    z_ref,
    lora_b_ref,
    lora_scaling_ref,
    product_ref,
    fused_ref,
    *,
    block_m: int,
    block_physical_n: int,
    block_k: int,
) -> None:
    """Write the product and the minimal projection residual for the VJP."""

    from jax.experimental.pallas import triton as plgpu

    rows, physical_start, physical_columns, fused, product = _contiguous_gate_up_swiglu_values(
        normalized_ref,
        frozen_weight_ref,
        z_ref,
        lora_b_ref,
        lora_scaling_ref,
        block_m=block_m,
        block_physical_n=block_physical_n,
        block_k=block_k,
    )
    product_start = physical_start // 2
    product_columns = product_start + jnp.arange(block_physical_n // 2)
    plgpu.store(product_ref.at[rows[:, None], product_columns[None, :]], product)
    plgpu.store(fused_ref.at[rows[:, None], physical_columns[None, :]], fused)


def _materialize_normalized_lora_a(
    flat_x: jax.Array,
    rms_delta: jax.Array,
    lora_a: jax.Array,
    *,
    eps: float,
    block_m: int,
    block_k: int,
    interpret: bool,
) -> tuple[jax.Array, ...]:
    from jax.experimental import pallas as pl

    row_count, in_features = flat_x.shape
    return pl.pallas_call(
        partial(
            _rms_materialize_lora_a_kernel,
            block_m=block_m,
            block_k=block_k,
            eps=eps,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((row_count, in_features), jnp.bfloat16),
            jax.ShapeDtypeStruct((row_count, BF16_LORA_RANK), jnp.bfloat16),
            jax.ShapeDtypeStruct((row_count,), jnp.float32),
            jax.ShapeDtypeStruct((row_count,), jnp.float32),
        ),
        grid=(row_count // block_m,),
        compiler_params=_compiler_params(),
        interpret=interpret,
        name="skyrl_qwen35_bf16_rms_materialize_lora_a_forward",
    )(flat_x, rms_delta, lora_a)


def _project_contiguous_gate_up(
    normalized: jax.Array,
    frozen_weight: jax.Array,
    z: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    *,
    block_m: int,
    block_physical_n: int,
    block_k: int,
    interpret: bool,
    save_fused: bool,
) -> tuple[jax.Array, jax.Array | None]:
    from jax.experimental import pallas as pl

    row_count, _ = normalized.shape
    physical_features = frozen_weight.shape[1]
    product_features = physical_features // 2
    common = {
        "grid": (
            row_count // block_m,
            physical_features // block_physical_n,
        ),
        "compiler_params": _compiler_params(),
        "interpret": interpret,
    }
    arguments = (normalized, frozen_weight, z, lora_b, lora_scaling)
    if save_fused:
        product, fused = pl.pallas_call(
            partial(
                _contiguous_gate_up_swiglu_residual_kernel,
                block_m=block_m,
                block_physical_n=block_physical_n,
                block_k=block_k,
            ),
            out_shape=(
                jax.ShapeDtypeStruct((row_count, product_features), jnp.bfloat16),
                jax.ShapeDtypeStruct((row_count, physical_features), jnp.bfloat16),
            ),
            name="skyrl_qwen35_bf16_contiguous_gate_up_lora_swiglu_residual_forward",
            **common,
        )(*arguments)
        return product, fused

    product = pl.pallas_call(
        partial(
            _contiguous_gate_up_swiglu_kernel,
            block_m=block_m,
            block_physical_n=block_physical_n,
            block_k=block_k,
        ),
        out_shape=jax.ShapeDtypeStruct((row_count, product_features), jnp.bfloat16),
        name="skyrl_qwen35_bf16_contiguous_gate_up_lora_swiglu_forward",
        **common,
    )(*arguments)
    return product, None


def _forward_impl(
    x: jax.Array,
    rms_delta: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    eps: float,
    block_m: int,
    block_physical_n: int,
    block_k: int,
    interpret: bool,
) -> jax.Array:
    row_count = math.prod(x.shape[:-1])
    flat_x = x.reshape((row_count, x.shape[-1]))
    normalized, z, _, _ = _materialize_normalized_lora_a(
        flat_x,
        rms_delta,
        lora_a,
        eps=eps,
        block_m=block_m,
        block_k=block_k,
        interpret=interpret,
    )
    product, _ = _project_contiguous_gate_up(
        normalized,
        frozen_weight,
        z,
        lora_b,
        lora_scaling,
        block_m=block_m,
        block_physical_n=block_physical_n,
        block_k=block_k,
        interpret=interpret,
        save_fused=False,
    )
    return product.reshape((*x.shape[:-1], product.shape[-1]))


def _forward_with_residuals(
    x: jax.Array,
    rms_delta: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    eps: float,
    block_m: int,
    block_physical_n: int,
    block_k: int,
    interpret: bool,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    row_count = math.prod(x.shape[:-1])
    flat_x = x.reshape((row_count, x.shape[-1]))
    normalized, z, denominator, inverse_rms = _materialize_normalized_lora_a(
        flat_x,
        rms_delta,
        lora_a,
        eps=eps,
        block_m=block_m,
        block_k=block_k,
        interpret=interpret,
    )
    product, fused = _project_contiguous_gate_up(
        normalized,
        frozen_weight,
        z,
        lora_b,
        lora_scaling,
        block_m=block_m,
        block_physical_n=block_physical_n,
        block_k=block_k,
        interpret=interpret,
        save_fused=True,
    )
    if fused is None:
        raise AssertionError("the training forward did not produce its projection residual")
    return product.reshape((*x.shape[:-1], product.shape[-1])), (
        z,
        denominator,
        inverse_rms,
        fused,
    )


def _fp32_swiglu_from_interleaved(fused: jax.Array) -> jax.Array:
    paired = fused.reshape((fused.shape[0], fused.shape[1] // 2, 2))
    gate, up = jnp.unstack(paired, axis=-1)
    return (jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)).astype(jnp.bfloat16)


def _fp32_swiglu_pullback_interleaved(fused: jax.Array, product_cotangent: jax.Array) -> jax.Array:
    paired = fused.reshape((fused.shape[0], fused.shape[1] // 2, 2))
    gate, up = jnp.unstack(paired, axis=-1)
    gate_f32 = gate.astype(jnp.float32)
    up_f32 = up.astype(jnp.float32)
    cotangent_f32 = product_cotangent.reshape(gate.shape).astype(jnp.float32)
    sigmoid = jax.nn.sigmoid(gate_f32)
    silu = gate_f32 * sigmoid
    d_up = (cotangent_f32 * silu).astype(jnp.bfloat16)
    weighted_cotangent = cotangent_f32 * up_f32
    sigmoid_prime = sigmoid * (jnp.asarray(1.0, jnp.float32) - sigmoid)
    d_gate = ((weighted_cotangent * sigmoid) + ((gate_f32 * weighted_cotangent) * sigmoid_prime)).astype(jnp.bfloat16)
    return jnp.stack((d_gate, d_up), axis=-1).reshape(fused.shape)


def _rms_norm_pullback(
    x: jax.Array,
    rms_delta: jax.Array,
    normalized_cotangent: jax.Array,
    denominator: jax.Array,
    inverse_rms: jax.Array,
) -> jax.Array:
    flat_x = x.reshape((-1, x.shape[-1]))
    x_f32 = flat_x.astype(jnp.float32)
    gamma = jnp.asarray(1.0, jnp.float32) + rms_delta.astype(jnp.float32)
    p = normalized_cotangent.reshape(flat_x.shape).astype(jnp.float32) * gamma[None, :]
    inverse = inverse_rms.reshape((-1, 1))
    denominator = denominator.reshape((-1, 1))
    inverse_cotangent = jnp.sum(x_f32 * p, axis=-1, keepdims=True)
    minus_half = jnp.asarray(-0.5, dtype=jnp.float32)
    denominator_cotangent = inverse_cotangent * (minus_half * (inverse / denominator))
    variance_cotangent = denominator_cotangent / jnp.asarray(x.shape[-1], dtype=jnp.float32)
    dx = (((p * inverse) + (x_f32 * variance_cotangent)) + (variance_cotangent * x_f32)).astype(jnp.bfloat16)
    return dx.reshape(x.shape).astype(x.dtype)


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10))
def _bf16_rms_gate_up_lora_swiglu_contiguous(
    x: jax.Array,
    rms_delta: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    eps: float,
    block_m: int,
    block_physical_n: int,
    block_k: int,
    interpret: bool,
) -> jax.Array:
    return _forward_impl(
        x,
        rms_delta,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
        eps,
        block_m,
        block_physical_n,
        block_k,
        interpret,
    )


def _bf16_rms_gate_up_lora_swiglu_contiguous_fwd(
    x: jax.Array,
    rms_delta: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    eps: float,
    block_m: int,
    block_physical_n: int,
    block_k: int,
    interpret: bool,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    output, intermediates = _forward_with_residuals(
        x,
        rms_delta,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
        eps,
        block_m,
        block_physical_n,
        block_k,
        interpret,
    )
    return output, (
        x,
        rms_delta,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
        *intermediates,
    )


def _bf16_rms_gate_up_lora_swiglu_contiguous_bwd(
    eps: float,
    block_m: int,
    block_physical_n: int,
    block_k: int,
    interpret: bool,
    saved: tuple[jax.Array, ...],
    product_cotangent: jax.Array,
) -> tuple[jax.Array | None, ...]:
    del eps, block_m, block_physical_n, block_k, interpret
    (
        x,
        rms_delta,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
        z,
        denominator,
        inverse_rms,
        fused,
    ) = saved

    dfused = _fp32_swiglu_pullback_interleaved(fused, product_cotangent)
    scaled_dfused = (dfused * lora_scaling).astype(jnp.bfloat16)

    dz = (scaled_dfused @ lora_b.T).astype(jnp.bfloat16)
    flat_x = x.reshape((-1, x.shape[-1]))
    x_f32 = flat_x.astype(jnp.float32)
    gamma = 1.0 + rms_delta.astype(jnp.float32)
    normalized = (x_f32 * inverse_rms[:, None] * gamma[None, :]).astype(jnp.bfloat16)
    da = (normalized.T @ dz).astype(jnp.bfloat16)
    db = (z.T @ scaled_dfused).astype(jnp.bfloat16)

    base_dnormalized = (dfused @ frozen_weight.T).astype(jnp.bfloat16)
    lora_dnormalized = (dz @ lora_a.T).astype(jnp.bfloat16)
    dnormalized = (base_dnormalized + lora_dnormalized).astype(jnp.bfloat16)

    dx = _rms_norm_pullback(x, rms_delta, dnormalized, denominator, inverse_rms)
    return (
        dx,
        None,
        None,
        da.astype(lora_a.dtype),
        db.astype(lora_b.dtype),
        None,
    )


_bf16_rms_gate_up_lora_swiglu_contiguous.defvjp(
    _bf16_rms_gate_up_lora_swiglu_contiguous_fwd,
    _bf16_rms_gate_up_lora_swiglu_contiguous_bwd,
)


def bf16_rms_gate_up_lora_swiglu_contiguous(
    x: jax.Array,
    rms_delta: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    *,
    enabled: bool = False,
    eps: float = RMS_NORM_EPSILON,
    interpret: bool = False,
    block_m: int = DEFAULT_BLOCK_M,
    block_physical_n: int = DEFAULT_BLOCK_PHYSICAL_N,
    block_k: int = DEFAULT_BLOCK_K,
) -> jax.Array:
    """Run the isolated two-stage forward after an explicit opt-in.

    All tensor inputs, including scalar ``lora_scaling``, must be BF16.  A
    non-interpreted call accepts only the exact production tensor shapes;
    smaller full tiles exist solely for ``interpret=True`` CPU semantics.
    """

    _validate_exact_bool("enabled", enabled)
    _validate_exact_bool("interpret", interpret)
    if not enabled:
        raise RuntimeError("the contiguous BF16 Pallas RMS/gate-up/LoRA/SwiGLU experiment " "is disabled by default")
    _validate_epsilon(eps)
    x = jnp.asarray(x)
    rms_delta = jnp.asarray(rms_delta)
    frozen_weight = jnp.asarray(frozen_weight)
    lora_a = jnp.asarray(lora_a)
    lora_b = jnp.asarray(lora_b)
    lora_scaling = jnp.asarray(lora_scaling)
    _validate_inputs(
        x,
        rms_delta,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
        interpret=interpret,
        block_m=block_m,
        block_physical_n=block_physical_n,
        block_k=block_k,
    )
    return _bf16_rms_gate_up_lora_swiglu_contiguous(
        x,
        rms_delta,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
        eps,
        block_m,
        block_physical_n,
        block_k,
        interpret,
    )
