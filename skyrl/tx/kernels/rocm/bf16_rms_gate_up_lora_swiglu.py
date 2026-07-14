"""Default-off BF16 Qwen3.5 RMSNorm + gate/up LoRA + SwiGLU stage.

This experimental Pallas candidate targets the exact B1/T64 Qwen3.5 MLP
geometry after flattening: M=64, K=2560, an interleaved gate/up projection of
physical width 18432, and a product width of 9216.  Smaller full-tile shapes
are accepted only so the bounded implementation can be interpreted on CPU.

The forward is deliberately split into two launches.  The first produces only
the per-row FP32 inverse RMS and the rank-eight BF16 LoRA-A projection.  The
second recomputes normalized input tiles, consumes paired gate/up columns, and
writes the BF16 SwiGLU product directly.  It never exposes a full normalized
activation or raw gate/up projection.

The custom VJP is correctness-first and experimental: it recomputes the dense
JAX reference intermediates and uses library dots for x/LoRA-A/LoRA-B
pullbacks.  Frozen projection weights, the RMS delta, and scaling are
nondifferentiable by contract.  Model code must retain an exact-shape ROCm gate
and must not opt in until GPU numerical and end-to-end performance gates pass.
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

BF16_LORA_RANK = 8
RMS_NORM_EPSILON = 1e-6
DEFAULT_BLOCK_M = 16
# Thirty-two logical pairs keep the initial production tile to 64 physical
# gate/up columns.  The paired accumulators make a 64-pair tile substantially
# more register-hungry; larger variants remain explicit benchmark candidates.
DEFAULT_BLOCK_N = 32
DEFAULT_BLOCK_K = 64
MAX_ROWS = 64
MAX_IN_FEATURES = 2560
MAX_PRODUCT_FEATURES = 9216
MAX_PHYSICAL_FEATURES = 2 * MAX_PRODUCT_FEATURES
_SUPPORTED_BLOCK_SIZES = frozenset((16, 32, 64, 128))


def _validate_exact_bool(name: str, value: bool) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact bool")


def _validate_block_size(name: str, value: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value not in _SUPPORTED_BLOCK_SIZES or value > maximum:
        supported = sorted(size for size in _SUPPORTED_BLOCK_SIZES if size <= maximum)
        raise ValueError(f"{name} must be one of {supported}, got {value}")


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
    block_m: int,
    block_n: int,
    block_k: int,
) -> None:
    _validate_block_size("block_m", block_m, 64)
    _validate_block_size("block_n", block_n, 128)
    _validate_block_size("block_k", block_k, 64)

    if x.ndim < 2:
        raise ValueError(f"x must have a row domain and final K axis, got {x.shape}")
    row_count = math.prod(x.shape[:-1])
    if not 0 < row_count <= MAX_ROWS:
        raise ValueError(f"flattened M must be in [1, {MAX_ROWS}], got {row_count}")
    if row_count % block_m:
        raise ValueError(
            f"flattened M={row_count} must be divisible by block_m={block_m}"
        )

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
            "frozen_weight must have shape [K, 2 * N] with interleaved "
            f"gate/up columns, got {frozen_weight.shape}"
        )
    in_features, physical_features = frozen_weight.shape
    if not 0 < in_features <= MAX_IN_FEATURES:
        raise ValueError(f"K must be in [1, {MAX_IN_FEATURES}], got {in_features}")
    if in_features % block_k:
        raise ValueError(f"K={in_features} must be divisible by block_k={block_k}")
    if x.shape[-1] != in_features:
        raise ValueError(f"x K={x.shape[-1]} does not match weight K={in_features}")
    if rms_delta.shape != (in_features,):
        raise ValueError(
            f"rms_delta must have shape ({in_features},), got {rms_delta.shape}"
        )
    if physical_features <= 0 or physical_features > MAX_PHYSICAL_FEATURES:
        raise ValueError(
            f"physical gate/up width must be in [2, {MAX_PHYSICAL_FEATURES}], "
            f"got {physical_features}"
        )
    if physical_features % 2:
        raise ValueError(
            f"physical gate/up width must be even, got {physical_features}"
        )
    product_features = physical_features // 2
    if product_features % block_n:
        raise ValueError(
            f"product N={product_features} must be divisible by block_n={block_n}"
        )
    if lora_a.shape != (in_features, BF16_LORA_RANK):
        raise ValueError(
            f"lora_a must have shape ({in_features}, {BF16_LORA_RANK}), "
            f"got {lora_a.shape}"
        )
    if lora_b.shape != (BF16_LORA_RANK, physical_features):
        raise ValueError(
            f"lora_b must have shape ({BF16_LORA_RANK}, {physical_features}), "
            f"got {lora_b.shape}"
        )
    if lora_scaling.shape != ():
        raise ValueError(
            f"lora_scaling must be a BF16 scalar, got shape={lora_scaling.shape}"
        )


def _compiler_params():
    from jax.experimental.pallas import triton as plgpu

    return plgpu.CompilerParams(num_warps=4, num_stages=1)


def _rms_lora_a_kernel(
    x_ref,
    rms_delta_ref,
    lora_a_ref,
    inverse_rms_ref,
    z_ref,
    *,
    block_m: int,
    block_k: int,
    eps: float,
) -> None:
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
    inverse_rms = lax.rsqrt(squared_sum / x_ref.shape[1] + eps)

    rank = jnp.arange(BF16_LORA_RANK)

    def consume_lora_a(k_block, previous):
        k = k_block * block_k + jnp.arange(block_k)
        x = plgpu.load(x_ref.at[rows[:, None], k[None, :]]).astype(jnp.float32)
        delta = plgpu.load(rms_delta_ref.at[k]).astype(jnp.float32)
        normalized = (x * inverse_rms[:, None] * (1.0 + delta[None, :])).astype(
            jnp.bfloat16
        )
        lora_a = plgpu.load(lora_a_ref.at[k[:, None], rank[None, :]]).astype(
            jnp.float32
        )
        return previous + jnp.sum(
            normalized.astype(jnp.float32)[:, :, None] * lora_a[None, :, :],
            axis=1,
        )

    z = lax.fori_loop(
        0,
        x_ref.shape[1] // block_k,
        consume_lora_a,
        jnp.zeros((block_m, BF16_LORA_RANK), dtype=jnp.float32),
    )
    plgpu.store(inverse_rms_ref.at[rows], inverse_rms)
    plgpu.store(z_ref.at[rows[:, None], rank[None, :]], z.astype(jnp.bfloat16))


def _gate_up_swiglu_kernel(
    x_ref,
    rms_delta_ref,
    frozen_weight_ref,
    z_ref,
    lora_b_ref,
    lora_scaling_ref,
    inverse_rms_ref,
    product_ref,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
) -> None:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    row_start = pl.program_id(0) * block_m
    product_start = pl.program_id(1) * block_n
    rows = row_start + jnp.arange(block_m)
    product_columns = product_start + jnp.arange(block_n)
    gate_columns = 2 * product_columns
    up_columns = gate_columns + 1
    inverse_rms = plgpu.load(inverse_rms_ref.at[rows]).astype(jnp.float32)

    def consume_k(k_block, previous):
        gate_accumulator, up_accumulator = previous
        k = k_block * block_k + jnp.arange(block_k)
        x = plgpu.load(x_ref.at[rows[:, None], k[None, :]]).astype(jnp.float32)
        delta = plgpu.load(rms_delta_ref.at[k]).astype(jnp.float32)
        normalized = (x * inverse_rms[:, None] * (1.0 + delta[None, :])).astype(
            jnp.bfloat16
        )
        gate_weight = plgpu.load(
            frozen_weight_ref.at[k[:, None], gate_columns[None, :]]
        )
        up_weight = plgpu.load(frozen_weight_ref.at[k[:, None], up_columns[None, :]])
        return (
            gate_accumulator + plgpu.dot(normalized, gate_weight).astype(jnp.float32),
            up_accumulator + plgpu.dot(normalized, up_weight).astype(jnp.float32),
        )

    initial = (
        jnp.zeros((block_m, block_n), dtype=jnp.float32),
        jnp.zeros((block_m, block_n), dtype=jnp.float32),
    )
    gate_accumulator, up_accumulator = lax.fori_loop(
        0,
        x_ref.shape[1] // block_k,
        consume_k,
        initial,
    )

    rank = jnp.arange(BF16_LORA_RANK)
    z = plgpu.load(z_ref.at[rows[:, None], rank[None, :]]).astype(jnp.float32)
    gate_lora_b = plgpu.load(
        lora_b_ref.at[rank[:, None], gate_columns[None, :]]
    ).astype(jnp.float32)
    up_lora_b = plgpu.load(lora_b_ref.at[rank[:, None], up_columns[None, :]]).astype(
        jnp.float32
    )
    low_rank_gate = jnp.sum(z[:, :, None] * gate_lora_b[None, :, :], axis=1)
    low_rank_up = jnp.sum(z[:, :, None] * up_lora_b[None, :, :], axis=1)
    scaling = lora_scaling_ref[...].astype(jnp.bfloat16)

    # Match the model's BF16 boundaries: both projection reductions and both
    # LoRA-B reductions round before BF16 scale/add and the BF16 SwiGLU.
    gate = gate_accumulator.astype(jnp.bfloat16)
    gate = (
        gate + (low_rank_gate.astype(jnp.bfloat16) * scaling).astype(jnp.bfloat16)
    ).astype(jnp.bfloat16)
    up = up_accumulator.astype(jnp.bfloat16)
    up = (
        up + (low_rank_up.astype(jnp.bfloat16) * scaling).astype(jnp.bfloat16)
    ).astype(jnp.bfloat16)
    product = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
    plgpu.store(
        product_ref.at[rows[:, None], product_columns[None, :]],
        product.astype(jnp.bfloat16),
    )


def _rms_norm_reference(x: jax.Array, rms_delta: jax.Array, eps: float) -> jax.Array:
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
    normalized = x_f32 * lax.rsqrt(variance + eps)
    return (normalized * (1.0 + rms_delta.astype(jnp.float32))).astype(jnp.bfloat16)


def _dense_reference(
    x: jax.Array,
    rms_delta: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    normalized = _rms_norm_reference(x, rms_delta, eps)
    flat_normalized = normalized.reshape((-1, normalized.shape[-1]))
    z = (flat_normalized @ lora_a).astype(jnp.bfloat16)
    base = (flat_normalized @ frozen_weight).astype(jnp.bfloat16)
    low_rank = (z @ lora_b).astype(jnp.bfloat16)
    fused = base + (low_rank * lora_scaling).astype(jnp.bfloat16)
    fused = fused.astype(jnp.bfloat16)
    gate = fused[:, 0::2]
    up = fused[:, 1::2]
    product = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
    return (
        product.reshape((*x.shape[:-1], gate.shape[-1])),
        normalized,
        z,
        fused,
    )


def _forward_impl(
    x: jax.Array,
    rms_delta: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    eps: float,
    block_m: int,
    block_n: int,
    block_k: int,
    interpret: bool,
) -> jax.Array:
    from jax.experimental import pallas as pl

    row_count = math.prod(x.shape[:-1])
    product_features = frozen_weight.shape[1] // 2
    flat_x = x.reshape((row_count, x.shape[-1]))
    inverse_rms, z = pl.pallas_call(
        partial(
            _rms_lora_a_kernel,
            block_m=block_m,
            block_k=block_k,
            eps=eps,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((row_count,), jnp.float32),
            jax.ShapeDtypeStruct((row_count, BF16_LORA_RANK), jnp.bfloat16),
        ),
        grid=(row_count // block_m,),
        compiler_params=_compiler_params(),
        interpret=interpret,
        name="skyrl_qwen35_bf16_rms_lora_a_forward",
    )(flat_x, rms_delta, lora_a)
    product = pl.pallas_call(
        partial(
            _gate_up_swiglu_kernel,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
        ),
        out_shape=jax.ShapeDtypeStruct((row_count, product_features), jnp.bfloat16),
        grid=(row_count // block_m, product_features // block_n),
        compiler_params=_compiler_params(),
        interpret=interpret,
        name="skyrl_qwen35_bf16_gate_up_lora_swiglu_forward",
    )(
        flat_x,
        rms_delta,
        frozen_weight,
        z,
        lora_b,
        lora_scaling,
        inverse_rms,
    )
    return product.reshape((*x.shape[:-1], product_features))


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10))
def _bf16_rms_gate_up_lora_swiglu(
    x: jax.Array,
    rms_delta: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    eps: float,
    block_m: int,
    block_n: int,
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
        block_n,
        block_k,
        interpret,
    )


def _bf16_rms_gate_up_lora_swiglu_fwd(
    x: jax.Array,
    rms_delta: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    eps: float,
    block_m: int,
    block_n: int,
    block_k: int,
    interpret: bool,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    output = _forward_impl(
        x,
        rms_delta,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
        eps,
        block_m,
        block_n,
        block_k,
        interpret,
    )
    return output, (x, rms_delta, frozen_weight, lora_a, lora_b, lora_scaling)


def _bf16_rms_gate_up_lora_swiglu_bwd(
    eps: float,
    block_m: int,
    block_n: int,
    block_k: int,
    interpret: bool,
    saved: tuple[jax.Array, ...],
    product_cotangent: jax.Array,
) -> tuple[jax.Array | None, ...]:
    del block_m, block_n, block_k, interpret
    x, rms_delta, frozen_weight, lora_a, lora_b, lora_scaling = saved

    # This intentionally favors the exact library equations over fusion.  It
    # keeps the forward candidate independently replaceable while its backward
    # numerical and performance behavior is still being qualified.
    product, normalized, z, fused = _dense_reference(
        x,
        rms_delta,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
        eps,
    )
    del product
    flat_normalized = normalized.reshape((-1, normalized.shape[-1]))

    def swiglu(fused_arg: jax.Array) -> jax.Array:
        return jax.nn.silu(fused_arg[:, 0::2]) * fused_arg[:, 1::2]

    _, swiglu_pullback = jax.vjp(swiglu, fused)
    (dfused,) = swiglu_pullback(
        product_cotangent.reshape((-1, product_cotangent.shape[-1]))
    )
    dfused = dfused.astype(jnp.bfloat16)
    scaled_dfused = (dfused * lora_scaling).astype(jnp.bfloat16)
    dz = (scaled_dfused @ lora_b.T).astype(jnp.bfloat16)
    base_dnormalized = (dfused @ frozen_weight.T).astype(jnp.bfloat16)
    lora_dnormalized = (dz @ lora_a.T).astype(jnp.bfloat16)
    dnormalized = (base_dnormalized + lora_dnormalized).astype(jnp.bfloat16)
    da = (flat_normalized.T @ dz).astype(jnp.bfloat16)
    db = (z.T @ scaled_dfused).astype(jnp.bfloat16)

    def rms_only(x_arg: jax.Array) -> jax.Array:
        return _rms_norm_reference(x_arg, rms_delta, eps)

    _, rms_pullback = jax.vjp(rms_only, x)
    (dx,) = rms_pullback(dnormalized.reshape(normalized.shape))
    return (
        dx.astype(x.dtype),
        None,
        None,
        da.astype(lora_a.dtype),
        db.astype(lora_b.dtype),
        None,
    )


_bf16_rms_gate_up_lora_swiglu.defvjp(
    _bf16_rms_gate_up_lora_swiglu_fwd,
    _bf16_rms_gate_up_lora_swiglu_bwd,
)


def bf16_rms_gate_up_lora_swiglu(
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
    block_n: int = DEFAULT_BLOCK_N,
    block_k: int = DEFAULT_BLOCK_K,
) -> jax.Array:
    """Run the bounded two-dispatch stage after an explicit opt-in.

    All tensor inputs, including scalar ``lora_scaling``, must be BF16.  Only
    ``x`` and the selected rank-eight LoRA A/B matrices are differentiable;
    ``rms_delta``, ``frozen_weight``, and ``lora_scaling`` are frozen.
    """
    _validate_exact_bool("enabled", enabled)
    _validate_exact_bool("interpret", interpret)
    if not enabled:
        raise RuntimeError(
            "the BF16 Pallas RMS/gate-up/LoRA/SwiGLU experiment is disabled by default"
        )
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
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )
    return _bf16_rms_gate_up_lora_swiglu(
        x,
        rms_delta,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
        eps,
        block_m,
        block_n,
        block_k,
        interpret,
    )
