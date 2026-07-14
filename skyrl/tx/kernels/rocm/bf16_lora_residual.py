"""Default-off bounded BF16 frozen-linear + LoRA + residual Pallas stage.

The production candidate in this module targets the Qwen3.5 MLP down
projection.  It keeps the library-independent row domain bounded, computes the
rank-8 LoRA-A projection outside the kernel, and owns the expensive frozen
projection, LoRA-B epilogue, and residual add in one Pallas launch.  Its custom
VJP defaults to the faster library-dot input pullback; a second, fully fused
Pallas input-pullback candidate remains available for isolated comparison.

This path is deliberately opt-in.  It accepts only BF16, rank eight, bounded
Qwen3.5 projection geometry, and full unmasked physical tiles.  Model code does
not select it until an exact-shape ROCm performance and numerical gate passes.
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

BF16_LORA_RANK = 8
DEFAULT_BLOCK_M = 16
DEFAULT_BLOCK_N = 64
DEFAULT_BLOCK_K = 64
DEFAULT_ROW_SUPERBLOCK = 256
MAX_ROW_SUPERBLOCK = 2048
MAX_IN_FEATURES = 9216
MAX_OUT_FEATURES = 2560
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


def _validate_row_superblock(value: int, block_m: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("row_superblock must be an integer")
    if not block_m <= value <= MAX_ROW_SUPERBLOCK:
        raise ValueError(
            f"row_superblock must be in [{block_m}, {MAX_ROW_SUPERBLOCK}], got {value}"
        )
    if value % block_m:
        raise ValueError("row_superblock must be divisible by block_m")


def _validate_inputs(
    x: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    residual: jax.Array,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    row_superblock: int,
) -> None:
    _validate_block_size("block_m", block_m, 64)
    _validate_block_size("block_n", block_n, 128)
    _validate_block_size("block_k", block_k, 64)
    _validate_row_superblock(row_superblock, block_m)

    if x.ndim < 2 or math.prod(x.shape[:-1]) <= 0:
        raise ValueError(
            f"x must have a nonempty row domain and final K axis, got {x.shape}"
        )
    values = {
        "x": x,
        "frozen_weight": frozen_weight,
        "lora_a": lora_a,
        "lora_b": lora_b,
        "residual": residual,
    }
    for name, value in values.items():
        if value.dtype != jnp.bfloat16:
            raise TypeError(f"{name} must be BF16, got {value.dtype}")
    if frozen_weight.ndim != 2:
        raise ValueError(
            f"frozen_weight must have shape [K, N], got {frozen_weight.shape}"
        )

    in_features, out_features = frozen_weight.shape
    if not 0 < in_features <= MAX_IN_FEATURES:
        raise ValueError(f"K must be in [1, {MAX_IN_FEATURES}], got {in_features}")
    if not 0 < out_features <= MAX_OUT_FEATURES:
        raise ValueError(f"N must be in [1, {MAX_OUT_FEATURES}], got {out_features}")
    if in_features % block_k:
        raise ValueError(f"K={in_features} must be divisible by block_k={block_k}")
    if x.shape[-1] != in_features:
        raise ValueError(f"x K={x.shape[-1]} does not match weight K={in_features}")
    if lora_a.shape != (in_features, BF16_LORA_RANK):
        raise ValueError(
            f"lora_a must have shape ({in_features}, {BF16_LORA_RANK}), got {lora_a.shape}"
        )
    if lora_b.shape != (BF16_LORA_RANK, out_features):
        raise ValueError(
            f"lora_b must have shape ({BF16_LORA_RANK}, {out_features}), got {lora_b.shape}"
        )
    expected_output_shape = (*x.shape[:-1], out_features)
    if residual.shape != expected_output_shape:
        raise ValueError(
            f"residual shape {residual.shape} must match output shape {expected_output_shape}"
        )
    if lora_scaling.shape != () or not jnp.issubdtype(lora_scaling.dtype, jnp.floating):
        raise TypeError(
            f"lora_scaling must be a floating scalar, got "
            f"shape={lora_scaling.shape}, dtype={lora_scaling.dtype}"
        )


def _compiler_params():
    from jax.experimental.pallas import triton as plgpu

    return plgpu.CompilerParams(num_warps=4, num_stages=1)


def _row_block_geometry(
    row_count: int, row_superblock: int, block_m: int
) -> tuple[int, int]:
    effective_rows = min(
        row_superblock,
        ((row_count + block_m - 1) // block_m) * block_m,
    )
    padded_rows = ((row_count + effective_rows - 1) // effective_rows) * effective_rows
    return effective_rows, padded_rows


def _pad_row_blocks(
    value: jax.Array, effective_rows: int, padded_rows: int
) -> jax.Array:
    return jnp.pad(value, ((0, padded_rows - value.shape[0]), (0, 0))).reshape(
        (-1, effective_rows, value.shape[1])
    )


def _padded_output_features(out_features: int, block_n: int) -> int:
    return ((out_features + block_n - 1) // block_n) * block_n


def _pad_output_features(value: jax.Array, padded_out_features: int) -> jax.Array:
    padding = padded_out_features - value.shape[-1]
    if padding < 0:
        raise ValueError("padded output-feature count is smaller than the input")
    if padding == 0:
        return value
    return jnp.pad(value, ((0, 0), (0, padding)))


def _forward_kernel(
    x_ref,
    frozen_weight_ref,
    z_ref,
    lora_b_ref,
    lora_scaling_ref,
    residual_ref,
    output_ref,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
) -> None:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    row_start = pl.program_id(0) * block_m
    column_start = pl.program_id(1) * block_n
    rows = row_start + jnp.arange(block_m)
    columns = column_start + jnp.arange(block_n)
    accumulator = jnp.zeros((block_m, block_n), dtype=jnp.float32)

    def consume_k(k_block, previous):
        k = k_block * block_k + jnp.arange(block_k)
        x = plgpu.load(x_ref.at[rows[:, None], k[None, :]])
        weight = plgpu.load(frozen_weight_ref.at[k[:, None], columns[None, :]])
        return previous + plgpu.dot(x, weight).astype(jnp.float32)

    accumulator = lax.fori_loop(
        0,
        x_ref.shape[1] // block_k,
        consume_k,
        accumulator,
    )

    rank = jnp.arange(BF16_LORA_RANK)
    z = plgpu.load(z_ref.at[rows[:, None], rank[None, :]]).astype(jnp.float32)
    lora_b = plgpu.load(lora_b_ref.at[rank[:, None], columns[None, :]]).astype(
        jnp.float32
    )
    low_rank = jnp.sum(z[:, :, None] * lora_b[None, :, :], axis=1).astype(jnp.bfloat16)
    scaling = lora_scaling_ref[...].astype(jnp.bfloat16)
    residual = plgpu.load(residual_ref.at[rows[:, None], columns[None, :]])

    # Preserve the model's BF16 operation boundaries: the frozen dot and LoRA-B
    # reduction each round before scaling and the two ordered additions.
    output = accumulator.astype(jnp.bfloat16)
    output = output + low_rank * scaling
    output = output + residual
    plgpu.store(output_ref.at[rows[:, None], columns[None, :]], output)


def _input_vjp_kernel(
    output_cotangent_ref,
    frozen_weight_ref,
    scaled_lora_input_cotangent_ref,
    lora_a_ref,
    input_cotangent_ref,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
) -> None:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    row_start = pl.program_id(0) * block_m
    k_start = pl.program_id(1) * block_k
    rows = row_start + jnp.arange(block_m)
    k = k_start + jnp.arange(block_k)
    accumulator = jnp.zeros((block_m, block_k), dtype=jnp.float32)

    def consume_n(n_block, previous):
        columns = n_block * block_n + jnp.arange(block_n)
        dy = plgpu.load(output_cotangent_ref.at[rows[:, None], columns[None, :]])
        weight = plgpu.load(frozen_weight_ref.at[k[:, None], columns[None, :]])
        return previous + plgpu.dot(dy, weight.T).astype(jnp.float32)

    accumulator = lax.fori_loop(
        0,
        output_cotangent_ref.shape[1] // block_n,
        consume_n,
        accumulator,
    )

    rank = jnp.arange(BF16_LORA_RANK)
    scaled_r = plgpu.load(
        scaled_lora_input_cotangent_ref.at[rows[:, None], rank[None, :]]
    ).astype(jnp.float32)
    lora_a = plgpu.load(lora_a_ref.at[k[:, None], rank[None, :]]).astype(jnp.float32)
    low_rank = jnp.sum(scaled_r[:, :, None] * lora_a.T[None, :, :], axis=1)
    output = accumulator.astype(jnp.bfloat16) + low_rank.astype(jnp.bfloat16)
    plgpu.store(input_cotangent_ref.at[rows[:, None], k[None, :]], output)


def _forward_impl(
    x: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    residual: jax.Array,
    block_m: int,
    block_n: int,
    block_k: int,
    row_superblock: int,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    from jax.experimental import pallas as pl

    row_count = math.prod(x.shape[:-1])
    out_features = frozen_weight.shape[1]
    padded_out_features = _padded_output_features(out_features, block_n)
    effective_rows, padded_rows = _row_block_geometry(
        row_count, row_superblock, block_m
    )
    blocked_x = _pad_row_blocks(
        x.reshape((row_count, x.shape[-1])), effective_rows, padded_rows
    )
    blocked_residual = _pad_row_blocks(
        residual.reshape((row_count, out_features)), effective_rows, padded_rows
    )
    blocked_residual = jnp.pad(
        blocked_residual,
        ((0, 0), (0, 0), (0, padded_out_features - out_features)),
    )
    padded_weight = _pad_output_features(frozen_weight, padded_out_features)
    padded_lora_b = _pad_output_features(lora_b, padded_out_features)

    def one_row_superblock(
        values: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        x_block, residual_block = values
        z_block = x_block @ lora_a
        run = pl.pallas_call(
            partial(
                _forward_kernel,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
            ),
            out_shape=jax.ShapeDtypeStruct(
                (effective_rows, padded_out_features), jnp.bfloat16
            ),
            grid=(effective_rows // block_m, padded_out_features // block_n),
            compiler_params=_compiler_params(),
            interpret=interpret,
            name="skyrl_qwen35_bf16_down_lora_residual_forward",
        )
        output_block = run(
            x_block,
            padded_weight,
            z_block,
            padded_lora_b,
            lora_scaling,
            residual_block,
        )
        return output_block, z_block

    blocked_output, blocked_z = lax.map(
        one_row_superblock, (blocked_x, blocked_residual)
    )
    output = blocked_output.reshape((-1, padded_out_features))[
        :row_count, :out_features
    ]
    z = blocked_z.reshape((-1, BF16_LORA_RANK))[:row_count]
    return output.reshape((*x.shape[:-1], out_features)), z


def _input_vjp_impl(
    output_cotangent: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    *,
    block_m: int,
    block_n: int,
    block_k: int,
    row_superblock: int,
    interpret: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    from jax.experimental import pallas as pl

    row_count = math.prod(output_cotangent.shape[:-1])
    out_features = output_cotangent.shape[-1]
    padded_out_features = _padded_output_features(out_features, block_n)
    effective_rows, padded_rows = _row_block_geometry(
        row_count, row_superblock, block_m
    )
    flat_dy = output_cotangent.reshape((row_count, out_features))
    blocked_dy = _pad_row_blocks(flat_dy, effective_rows, padded_rows)
    blocked_dy = jnp.pad(
        blocked_dy,
        ((0, 0), (0, 0), (0, padded_out_features - out_features)),
    )
    padded_weight = _pad_output_features(frozen_weight, padded_out_features)
    padded_lora_b = _pad_output_features(lora_b, padded_out_features)
    scaling_bf16 = jnp.asarray(lora_scaling, dtype=jnp.bfloat16)

    def one_row_superblock(
        dy_block: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        scaled_dy = (dy_block * scaling_bf16).astype(jnp.bfloat16)
        scaled_r = scaled_dy @ padded_lora_b.T
        run = pl.pallas_call(
            partial(
                _input_vjp_kernel,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
            ),
            out_shape=jax.ShapeDtypeStruct(
                (effective_rows, frozen_weight.shape[0]), jnp.bfloat16
            ),
            grid=(effective_rows // block_m, frozen_weight.shape[0] // block_k),
            compiler_params=_compiler_params(),
            interpret=interpret,
            name="skyrl_qwen35_bf16_down_lora_input_vjp",
        )
        return run(dy_block, padded_weight, scaled_r, lora_a), scaled_dy, scaled_r

    blocked_dx, blocked_scaled_dy, blocked_scaled_r = lax.map(
        one_row_superblock, blocked_dy
    )
    dx = blocked_dx.reshape((-1, frozen_weight.shape[0]))[:row_count]
    scaled_dy = blocked_scaled_dy.reshape((-1, padded_out_features))[
        :row_count, :out_features
    ]
    scaled_r = blocked_scaled_r.reshape((-1, BF16_LORA_RANK))[:row_count]
    return dx, scaled_dy, scaled_r


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10, 11))
def _bf16_lora_residual(
    x: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    residual: jax.Array,
    block_m: int,
    block_n: int,
    block_k: int,
    row_superblock: int,
    pallas_input_vjp: bool,
    interpret: bool,
) -> jax.Array:
    del pallas_input_vjp
    output, _ = _forward_impl(
        x,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
        residual,
        block_m,
        block_n,
        block_k,
        row_superblock,
        interpret,
    )
    return output


def _bf16_lora_residual_fwd(
    x: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    residual: jax.Array,
    block_m: int,
    block_n: int,
    block_k: int,
    row_superblock: int,
    pallas_input_vjp: bool,
    interpret: bool,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    del pallas_input_vjp
    output, z = _forward_impl(
        x,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
        residual,
        block_m,
        block_n,
        block_k,
        row_superblock,
        interpret,
    )
    return output, (x, frozen_weight, lora_a, lora_b, lora_scaling, z)


def _bf16_lora_residual_bwd(
    block_m: int,
    block_n: int,
    block_k: int,
    row_superblock: int,
    pallas_input_vjp: bool,
    interpret: bool,
    saved: tuple[jax.Array, ...],
    output_cotangent: jax.Array,
) -> tuple[jax.Array | None, ...]:
    x, frozen_weight, lora_a, lora_b, lora_scaling, z = saved
    if pallas_input_vjp:
        dx, scaled_dy, scaled_r = _input_vjp_impl(
            output_cotangent,
            frozen_weight,
            lora_a,
            lora_b,
            lora_scaling,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            row_superblock=row_superblock,
            interpret=interpret,
        )
    else:
        flat_dy = output_cotangent.reshape((-1, output_cotangent.shape[-1]))
        scaling_bf16 = jnp.asarray(lora_scaling, dtype=jnp.bfloat16)
        scaled_dy = (flat_dy * scaling_bf16).astype(jnp.bfloat16)
        scaled_r = scaled_dy @ lora_b.T
        base_dx = flat_dy @ frozen_weight.T
        lora_dx = scaled_r @ lora_a.T
        dx = base_dx + lora_dx
    flat_x = x.reshape((-1, x.shape[-1]))
    da = flat_x.T @ scaled_r
    db = z.T @ scaled_dy
    return (
        dx.reshape(x.shape),
        None,
        da.astype(lora_a.dtype),
        db.astype(lora_b.dtype),
        None,
        output_cotangent,
    )


_bf16_lora_residual.defvjp(_bf16_lora_residual_fwd, _bf16_lora_residual_bwd)


def bf16_lora_residual(
    x: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array | float,
    residual: jax.Array,
    *,
    enabled: bool = False,
    interpret: bool = False,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
    block_k: int = DEFAULT_BLOCK_K,
    row_superblock: int = DEFAULT_ROW_SUPERBLOCK,
    pallas_input_vjp: bool = False,
) -> jax.Array:
    """Run the bounded fused stage after an explicit opt-in.

    ``frozen_weight`` and ``lora_scaling`` are nondifferentiable by contract;
    the custom VJP returns cotangents for ``x``, LoRA A/B, and the residual.
    """
    _validate_exact_bool("enabled", enabled)
    _validate_exact_bool("interpret", interpret)
    _validate_exact_bool("pallas_input_vjp", pallas_input_vjp)
    if not enabled:
        raise RuntimeError(
            "the BF16 Pallas LoRA-residual experiment is disabled by default"
        )
    x = jnp.asarray(x)
    frozen_weight = jnp.asarray(frozen_weight)
    lora_a = jnp.asarray(lora_a)
    lora_b = jnp.asarray(lora_b)
    lora_scaling = jnp.asarray(lora_scaling, dtype=jnp.bfloat16)
    residual = jnp.asarray(residual)
    _validate_inputs(
        x,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
        residual,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        row_superblock=row_superblock,
    )
    return _bf16_lora_residual(
        x,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
        residual,
        block_m,
        block_n,
        block_k,
        row_superblock,
        pallas_input_vjp,
        interpret,
    )
