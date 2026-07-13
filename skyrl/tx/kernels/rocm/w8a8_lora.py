"""Default-off Pallas W8A8 frozen-linear plus rank-8 LoRA experiment.

This module is a bounded projection prototype for Qwen3.5-4B on gfx1100.  It
does not alter model construction or checkpoint loading.  The forward keeps a
group-64 INT8 weight canonical, dynamically quantizes each activation row and
group to INT8, uses signed integer matrix dots with INT32 accumulation, applies
the two FP32 group scales, and adds the rank-8 LoRA branch before one BF16
output cast.

The custom VJP deliberately uses a relaxed W8A16 base-input pullback: compact
codes are dequantized one tile at a time to BF16 and multiplied by the BF16
output cotangent with FP32 accumulation.  It never materializes a full
dequantized weight.  LoRA reductions also use BF16 inputs with FP32
accumulation.  This is close to, but not bit-identical to, the stronger FP32
backward oracle in :mod:`skyrl.tx.kernels.quantized_lora`; it therefore needs
an explicit numerical gate before any model integration.

Every forward Pallas program covers at most ``block_m`` rows and ``block_n``
output features while scanning the explicitly capped K-group domain.  Each
input-pullback program covers ``block_m`` rows and one 64-value output K group,
but scans the explicitly capped N domain.  Row scans bound the logical grid;
the physical duration of both inner scans remains a hardware qualification
item.  This
source calls no explicit graph, command-buffer, capture, replay,
persistent-kernel, cross-program synchronization, or atomic-reduction API.
Importing the module imports JAX but does not enumerate devices or initialize
an accelerator backend.
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from skyrl.tx.kernels.quantized_lora import GroupQuantizedWeight

W8A8_GROUP_SIZE = 64
W8A8_LORA_RANK = 8
DEFAULT_BLOCK_M = 32
DEFAULT_BLOCK_N = 64
DEFAULT_ROW_SUPERBLOCK = 512
MAX_BLOCK_M = 64
MAX_BLOCK_N = 128
MAX_ROW_SUPERBLOCK = 2048
MAX_IN_FEATURES = 9216
MAX_OUT_FEATURES = 18432
_SUPPORTED_BLOCK_SIZES = frozenset((16, 32, 64, 128))


def _validate_exact_bool(name: str, value: bool) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact bool")


def _validate_block_size(name: str, value: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value not in _SUPPORTED_BLOCK_SIZES or value > maximum:
        raise ValueError(
            f"{name} must be one of {sorted(size for size in _SUPPORTED_BLOCK_SIZES if size <= maximum)}, "
            f"got {value}"
        )


def _validate_row_superblock(value: int, block_m: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("row_superblock must be an integer")
    if not block_m <= value <= MAX_ROW_SUPERBLOCK:
        raise ValueError(
            f"row_superblock must be in [{block_m}, {MAX_ROW_SUPERBLOCK}], got {value}"
        )
    if value % block_m:
        raise ValueError("row_superblock must be divisible by block_m")


def _validate_base_inputs(
    x: jax.Array,
    weight: GroupQuantizedWeight,
    *,
    block_m: int,
    block_n: int,
    row_superblock: int,
) -> None:
    _validate_block_size("block_m", block_m, MAX_BLOCK_M)
    _validate_block_size("block_n", block_n, MAX_BLOCK_N)
    _validate_row_superblock(row_superblock, block_m)
    if x.ndim < 2 or math.prod(x.shape[:-1]) <= 0:
        raise ValueError(
            f"x must have a nonempty row domain and final K axis, got {x.shape}"
        )
    if x.dtype != jnp.bfloat16:
        raise TypeError(f"x must be BF16, got {x.dtype}")
    if weight.bits != 8:
        raise ValueError(f"the native prototype requires W8 codes, got W{weight.bits}")
    if weight.group_size != W8A8_GROUP_SIZE:
        raise ValueError(
            f"the native prototype requires group size {W8A8_GROUP_SIZE}, "
            f"got {weight.group_size}"
        )
    if weight.codes.ndim != 2 or weight.codes.dtype != jnp.int8:
        raise TypeError("W8 codes must be a rank-two INT8 array")
    if weight.scales.ndim != 2 or weight.scales.dtype != jnp.bfloat16:
        raise TypeError("W8 scales must be a rank-two BF16 array")
    if weight.original_in_features != weight.padded_in_features:
        raise ValueError("the native prototype does not accept padded K dimensions")
    if (
        weight.original_in_features <= 0
        or weight.original_in_features % W8A8_GROUP_SIZE
    ):
        raise ValueError("K must be a positive multiple of 64")
    if weight.original_in_features > MAX_IN_FEATURES:
        raise ValueError(
            f"K must not exceed the Qwen3.5 projection cap {MAX_IN_FEATURES}"
        )
    if weight.codes.shape[0] != weight.original_in_features:
        raise ValueError("W8 code K does not match weight metadata")
    out_features = weight.codes.shape[1]
    expected_scale_shape = (
        weight.original_in_features // W8A8_GROUP_SIZE,
        out_features,
    )
    if weight.scales.shape != expected_scale_shape:
        raise ValueError(
            f"W8 scale shape {weight.scales.shape} does not match {expected_scale_shape}"
        )
    if x.shape[-1] != weight.original_in_features:
        raise ValueError(
            f"x K={x.shape[-1]} does not match W8 K={weight.original_in_features}"
        )
    if out_features <= 0:
        raise ValueError("N must be positive")
    if out_features > MAX_OUT_FEATURES:
        raise ValueError(
            f"N must not exceed the Qwen3.5 projection cap {MAX_OUT_FEATURES}"
        )


def _validate_inputs(
    x: jax.Array,
    weight: GroupQuantizedWeight,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    *,
    block_m: int,
    block_n: int,
    row_superblock: int,
) -> None:
    _validate_base_inputs(
        x,
        weight,
        block_m=block_m,
        block_n=block_n,
        row_superblock=row_superblock,
    )
    out_features = weight.codes.shape[1]
    if lora_a.shape != (weight.original_in_features, W8A8_LORA_RANK):
        raise ValueError(
            f"lora_a must have shape ({weight.original_in_features}, {W8A8_LORA_RANK}), "
            f"got {lora_a.shape}"
        )
    if lora_b.shape != (W8A8_LORA_RANK, out_features):
        raise ValueError(
            f"lora_b must have shape ({W8A8_LORA_RANK}, {out_features}), got {lora_b.shape}"
        )
    if lora_a.dtype != jnp.bfloat16 or lora_b.dtype != jnp.bfloat16:
        raise TypeError("LoRA A and B must both be BF16")
    if lora_scaling.shape != () or lora_scaling.dtype != jnp.float32:
        raise TypeError("lora_scaling must be an FP32 scalar")


def _dynamic_group64_quantize(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Reproduce the portable oracle's symmetric A8 group quantizer."""
    grouped = x.astype(jnp.float32).reshape(
        (x.shape[0], x.shape[1] // W8A8_GROUP_SIZE, W8A8_GROUP_SIZE)
    )
    amax = jnp.max(jnp.abs(grouped), axis=-1)
    scales = jnp.where(amax > 0, amax / 127.0, 1.0)
    codes = jnp.clip(jnp.rint(grouped / scales[..., None]), -127, 127).astype(jnp.int8)
    return codes.reshape(x.shape), scales


def _w8a8_base_tile(
    x_codes_ref,
    x_scales_ref,
    weight_codes_ref,
    weight_scales_ref,
    *,
    block_m: int,
    block_n: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    row_start = pl.program_id(0) * block_m
    column_start = pl.program_id(1) * block_n
    rows = row_start + jnp.arange(block_m)
    columns = column_start + jnp.arange(block_n)
    row_mask = rows < x_codes_ref.shape[0]
    column_mask = columns < weight_codes_ref.shape[1]
    output_mask = row_mask[:, None] & column_mask[None, :]
    accumulator = jnp.zeros((block_m, block_n), dtype=jnp.float32)

    def consume_group(group_index, previous):
        k = group_index * W8A8_GROUP_SIZE + jnp.arange(W8A8_GROUP_SIZE)
        x_codes = plgpu.load(
            x_codes_ref.at[rows[:, None], k[None, :]],
            mask=row_mask[:, None],
            other=0,
        )
        weight_codes = plgpu.load(
            weight_codes_ref.at[k[:, None], columns[None, :]],
            mask=column_mask[None, :],
            other=0,
        )
        integer_product = plgpu.dot(x_codes, weight_codes).astype(jnp.float32)
        x_scale = plgpu.load(
            x_scales_ref.at[rows, group_index], mask=row_mask, other=0.0
        ).astype(jnp.float32)
        weight_scale = plgpu.load(
            weight_scales_ref.at[group_index, columns],
            mask=column_mask,
            other=0.0,
        ).astype(jnp.float32)
        return previous + integer_product * x_scale[:, None] * weight_scale[None, :]

    accumulator = lax.fori_loop(
        0,
        x_codes_ref.shape[1] // W8A8_GROUP_SIZE,
        consume_group,
        accumulator,
    )
    return rows, columns, row_mask, column_mask, output_mask, accumulator


def _w8a8_base_forward_kernel(
    x_codes_ref,
    x_scales_ref,
    weight_codes_ref,
    weight_scales_ref,
    output_ref,
    *,
    block_m: int,
    block_n: int,
) -> None:
    from jax.experimental.pallas import triton as plgpu

    rows, columns, _, _, output_mask, accumulator = _w8a8_base_tile(
        x_codes_ref,
        x_scales_ref,
        weight_codes_ref,
        weight_scales_ref,
        block_m=block_m,
        block_n=block_n,
    )
    plgpu.store(
        output_ref.at[rows[:, None], columns[None, :]],
        accumulator.astype(output_ref.dtype),
        mask=output_mask,
    )


def _w8a8_forward_kernel(
    x_codes_ref,
    x_scales_ref,
    weight_codes_ref,
    weight_scales_ref,
    z_ref,
    lora_b_ref,
    lora_scaling_ref,
    output_ref,
    *,
    block_m: int,
    block_n: int,
) -> None:
    from jax.experimental.pallas import triton as plgpu

    rows, columns, row_mask, column_mask, output_mask, accumulator = _w8a8_base_tile(
        x_codes_ref,
        x_scales_ref,
        weight_codes_ref,
        weight_scales_ref,
        block_m=block_m,
        block_n=block_n,
    )

    rank_indices = jnp.arange(W8A8_LORA_RANK)
    z = plgpu.load(
        z_ref.at[rows[:, None], rank_indices[None, :]],
        mask=row_mask[:, None],
        other=0.0,
    ).astype(jnp.float32)
    lora_b = plgpu.load(
        lora_b_ref.at[rank_indices[:, None], columns[None, :]],
        mask=column_mask[None, :],
        other=0.0,
    ).astype(jnp.float32)
    low_rank = jnp.sum(z[:, :, None] * lora_b[None, :, :], axis=1)
    scaling = lora_scaling_ref[...].astype(jnp.float32)
    output = accumulator + scaling * low_rank
    plgpu.store(
        output_ref.at[rows[:, None], columns[None, :]],
        output.astype(output_ref.dtype),
        mask=output_mask,
    )


def _w8a16_input_vjp_kernel(
    output_cotangent_ref,
    weight_codes_ref,
    weight_scales_ref,
    input_cotangent_ref,
    *,
    block_m: int,
    block_n: int,
) -> None:
    """Tiled relaxed ``dY @ dequant(W).T`` without a shadow weight."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    row_start = pl.program_id(0) * block_m
    group_index = pl.program_id(1)
    rows = row_start + jnp.arange(block_m)
    k = group_index * W8A8_GROUP_SIZE + jnp.arange(W8A8_GROUP_SIZE)
    row_mask = rows < output_cotangent_ref.shape[0]
    accumulator = jnp.zeros((block_m, W8A8_GROUP_SIZE), dtype=jnp.float32)
    column_blocks = pl.cdiv(output_cotangent_ref.shape[1], block_n)

    def consume_columns(column_block, previous):
        columns = column_block * block_n + jnp.arange(block_n)
        column_mask = columns < output_cotangent_ref.shape[1]
        dy = plgpu.load(
            output_cotangent_ref.at[rows[:, None], columns[None, :]],
            mask=row_mask[:, None] & column_mask[None, :],
            other=0.0,
        )
        codes = plgpu.load(
            weight_codes_ref.at[k[:, None], columns[None, :]],
            mask=column_mask[None, :],
            other=0,
        ).astype(jnp.float32)
        scales = plgpu.load(
            weight_scales_ref.at[group_index, columns],
            mask=column_mask,
            other=0.0,
        ).astype(jnp.float32)
        # This BF16 conversion is the experiment's explicit relaxed-backward
        # policy.  The portable semantic oracle instead contracts FP32
        # ``codes * scales`` values.
        weight = (codes * scales[None, :]).astype(jnp.bfloat16)
        return previous + plgpu.dot(dy, weight.T).astype(jnp.float32)

    accumulator = lax.fori_loop(0, column_blocks, consume_columns, accumulator)
    plgpu.store(
        input_cotangent_ref.at[rows[:, None], k[None, :]],
        accumulator.astype(input_cotangent_ref.dtype),
        mask=row_mask[:, None],
    )


def _compiler_params():
    from jax.experimental.pallas import triton as plgpu

    return plgpu.CompilerParams(num_warps=4, num_stages=1)


def _dot_general_f32(
    left: jax.Array,
    right: jax.Array,
    *,
    left_contract: int,
    right_contract: int,
) -> jax.Array:
    return lax.dot_general(
        left,
        right,
        dimension_numbers=(((left_contract,), (right_contract,)), ((), ())),
        preferred_element_type=jnp.float32,
    )


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


def _base_forward_impl(
    x: jax.Array,
    weight_codes: jax.Array,
    weight_scales: jax.Array,
    group_size: int,
    block_m: int,
    block_n: int,
    row_superblock: int,
    interpret: bool,
) -> jax.Array:
    del group_size
    from jax.experimental import pallas as pl

    row_count = math.prod(x.shape[:-1])
    flat_x = x.reshape((row_count, x.shape[-1]))
    effective_rows, padded_rows = _row_block_geometry(
        row_count, row_superblock, block_m
    )
    blocked_x = _pad_row_blocks(flat_x, effective_rows, padded_rows)

    def one_row_superblock(x_block: jax.Array) -> jax.Array:
        x_codes, x_scales = _dynamic_group64_quantize(x_block)
        run = pl.pallas_call(
            partial(
                _w8a8_base_forward_kernel,
                block_m=block_m,
                block_n=block_n,
            ),
            out_shape=jax.ShapeDtypeStruct(
                (effective_rows, weight_codes.shape[1]), x.dtype
            ),
            grid=(
                effective_rows // block_m,
                pl.cdiv(weight_codes.shape[1], block_n),
            ),
            compiler_params=_compiler_params(),
            interpret=interpret,
            name="skyrl_qwen35_w8a8_frozen_forward",
        )
        return run(x_codes, x_scales, weight_codes, weight_scales)

    blocked_output = lax.map(one_row_superblock, blocked_x)
    flat_output = blocked_output.reshape((-1, weight_codes.shape[1]))[:row_count]
    return flat_output.reshape((*x.shape[:-1], weight_codes.shape[1]))


def _forward_impl(
    x: jax.Array,
    weight_codes: jax.Array,
    weight_scales: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    group_size: int,
    block_m: int,
    block_n: int,
    row_superblock: int,
    interpret: bool,
) -> tuple[jax.Array, jax.Array]:
    del group_size
    from jax.experimental import pallas as pl

    row_count = math.prod(x.shape[:-1])
    flat_x = x.reshape((row_count, x.shape[-1]))
    # A small Pallas program does not by itself bound a GPU launch: its grid
    # could still cover all 32K rows.  Pad and map fixed row superblocks so the
    # activation quantizer, LoRA-A dot, and Pallas grid all see at most the
    # configured (and hard-capped) row domain per loop iteration.
    effective_rows, padded_rows = _row_block_geometry(
        row_count, row_superblock, block_m
    )
    blocked_x = _pad_row_blocks(flat_x, effective_rows, padded_rows)

    def one_row_superblock(x_block: jax.Array) -> tuple[jax.Array, jax.Array]:
        x_codes, x_scales = _dynamic_group64_quantize(x_block)
        z_block = _dot_general_f32(
            x_block,
            lora_a,
            left_contract=1,
            right_contract=0,
        )
        output_shape = (effective_rows, weight_codes.shape[1])
        run = pl.pallas_call(
            partial(
                _w8a8_forward_kernel,
                block_m=block_m,
                block_n=block_n,
            ),
            out_shape=jax.ShapeDtypeStruct(output_shape, x.dtype),
            grid=(
                effective_rows // block_m,
                pl.cdiv(weight_codes.shape[1], block_n),
            ),
            compiler_params=_compiler_params(),
            interpret=interpret,
            name="skyrl_qwen35_w8a8_lora_forward",
        )
        output_block = run(
            x_codes,
            x_scales,
            weight_codes,
            weight_scales,
            z_block,
            lora_b,
            lora_scaling,
        )
        return output_block, z_block

    blocked_output, blocked_z = lax.map(one_row_superblock, blocked_x)
    flat_output = blocked_output.reshape((-1, weight_codes.shape[1]))[:row_count]
    z = blocked_z.reshape((-1, W8A8_LORA_RANK))[:row_count]
    return flat_output.reshape((*x.shape[:-1], weight_codes.shape[1])), z


def _base_input_vjp(
    output_cotangent: jax.Array,
    weight_codes: jax.Array,
    weight_scales: jax.Array,
    *,
    block_m: int,
    block_n: int,
    row_superblock: int,
    interpret: bool,
) -> jax.Array:
    from jax.experimental import pallas as pl

    flat_dy = output_cotangent.reshape((-1, output_cotangent.shape[-1]))
    input_features = weight_codes.shape[0]
    row_count = flat_dy.shape[0]
    effective_rows, padded_rows = _row_block_geometry(
        row_count, row_superblock, block_m
    )
    blocked_dy = _pad_row_blocks(flat_dy, effective_rows, padded_rows)

    def one_row_superblock(dy_block: jax.Array) -> jax.Array:
        run = pl.pallas_call(
            partial(
                _w8a16_input_vjp_kernel,
                block_m=block_m,
                block_n=block_n,
            ),
            out_shape=jax.ShapeDtypeStruct(
                (effective_rows, input_features), output_cotangent.dtype
            ),
            grid=(
                effective_rows // block_m,
                input_features // W8A8_GROUP_SIZE,
            ),
            compiler_params=_compiler_params(),
            interpret=interpret,
            name="skyrl_qwen35_w8a16_lora_input_vjp",
        )
        return run(dy_block, weight_codes, weight_scales)

    blocked_dx = lax.map(one_row_superblock, blocked_dy)
    return blocked_dx.reshape((-1, input_features))[:row_count]


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def _w8a8_frozen_linear(
    x: jax.Array,
    weight_codes: jax.Array,
    weight_scales: jax.Array,
    group_size: int,
    block_m: int,
    block_n: int,
    row_superblock: int,
    interpret: bool,
) -> jax.Array:
    return _base_forward_impl(
        x,
        weight_codes,
        weight_scales,
        group_size,
        block_m,
        block_n,
        row_superblock,
        interpret,
    )


def _w8a8_frozen_linear_fwd(
    x: jax.Array,
    weight_codes: jax.Array,
    weight_scales: jax.Array,
    group_size: int,
    block_m: int,
    block_n: int,
    row_superblock: int,
    interpret: bool,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    output = _base_forward_impl(
        x,
        weight_codes,
        weight_scales,
        group_size,
        block_m,
        block_n,
        row_superblock,
        interpret,
    )
    return output, (weight_codes, weight_scales)


def _w8a8_frozen_linear_bwd(
    group_size: int,
    block_m: int,
    block_n: int,
    row_superblock: int,
    interpret: bool,
    saved: tuple[jax.Array, jax.Array],
    output_cotangent: jax.Array,
) -> tuple[jax.Array | None, ...]:
    del group_size
    weight_codes, weight_scales = saved
    dx = _base_input_vjp(
        output_cotangent,
        weight_codes,
        weight_scales,
        block_m=block_m,
        block_n=block_n,
        row_superblock=row_superblock,
        interpret=interpret,
    )
    return (
        dx.reshape((*output_cotangent.shape[:-1], weight_codes.shape[0])),
        None,
        None,
    )


_w8a8_frozen_linear.defvjp(
    _w8a8_frozen_linear_fwd,
    _w8a8_frozen_linear_bwd,
)


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10))
def _w8a8_lora_linear(
    x: jax.Array,
    weight_codes: jax.Array,
    weight_scales: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    group_size: int,
    block_m: int,
    block_n: int,
    row_superblock: int,
    interpret: bool,
) -> jax.Array:
    output, _ = _forward_impl(
        x,
        weight_codes,
        weight_scales,
        lora_a,
        lora_b,
        lora_scaling,
        group_size,
        block_m,
        block_n,
        row_superblock,
        interpret,
    )
    return output


def _w8a8_lora_linear_fwd(
    x: jax.Array,
    weight_codes: jax.Array,
    weight_scales: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    group_size: int,
    block_m: int,
    block_n: int,
    row_superblock: int,
    interpret: bool,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    output, z = _forward_impl(
        x,
        weight_codes,
        weight_scales,
        lora_a,
        lora_b,
        lora_scaling,
        group_size,
        block_m,
        block_n,
        row_superblock,
        interpret,
    )
    return output, (
        x,
        weight_codes,
        weight_scales,
        lora_a,
        lora_b,
        lora_scaling,
        z,
    )


def _w8a8_lora_linear_bwd(
    group_size: int,
    block_m: int,
    block_n: int,
    row_superblock: int,
    interpret: bool,
    saved: tuple[jax.Array, ...],
    output_cotangent: jax.Array,
) -> tuple[jax.Array | None, ...]:
    del group_size
    x, weight_codes, weight_scales, lora_a, lora_b, lora_scaling, z = saved
    flat_x = x.reshape((-1, x.shape[-1]))
    flat_dy = output_cotangent.reshape((-1, output_cotangent.shape[-1]))
    base_dx = _base_input_vjp(
        output_cotangent,
        weight_codes,
        weight_scales,
        block_m=block_m,
        block_n=block_n,
        row_superblock=row_superblock,
        interpret=interpret,
    )

    # BF16 products with FP32 accumulation avoid full FP32 copies of the
    # sequence-wide x/dY tensors.  Row scans also bound the logical row domain
    # of each LoRA contraction.  Physical dispatch duration remains a hardware
    # qualification item.  The only full-row FP32 values are saved z and
    # unscaled_r, both [rows, rank=8].
    row_count = flat_x.shape[0]
    effective_rows, padded_rows = _row_block_geometry(
        row_count, row_superblock, block_m
    )
    blocked_x = _pad_row_blocks(flat_x, effective_rows, padded_rows)
    blocked_dy = _pad_row_blocks(flat_dy, effective_rows, padded_rows)
    blocked_z = _pad_row_blocks(z, effective_rows, padded_rows)

    def one_r_block(dy_block: jax.Array) -> jax.Array:
        return _dot_general_f32(
            dy_block,
            lora_b,
            left_contract=1,
            right_contract=1,
        )

    blocked_unscaled_r = lax.map(one_r_block, blocked_dy)
    blocked_scaled_r_bf16 = (blocked_unscaled_r * lora_scaling).astype(jnp.bfloat16)

    def one_lora_dx_block(r_block: jax.Array) -> jax.Array:
        return _dot_general_f32(
            r_block,
            lora_a,
            left_contract=1,
            right_contract=1,
        ).astype(jnp.bfloat16)

    blocked_lora_dx = lax.map(one_lora_dx_block, blocked_scaled_r_bf16)
    lora_dx = blocked_lora_dx.reshape((-1, x.shape[-1]))[:row_count]
    dx = (base_dx + lora_dx).reshape(x.shape)

    def reduce_lora_partials(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        blocks: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        da, db, dscaling = carry
        x_block, dy_block, z_block, r_block = blocks
        da += _dot_general_f32(
            x_block,
            (r_block * lora_scaling).astype(jnp.bfloat16),
            left_contract=0,
            right_contract=0,
        )
        db += (
            _dot_general_f32(
                z_block.astype(jnp.bfloat16),
                dy_block,
                left_contract=0,
                right_contract=0,
            )
            * lora_scaling
        )
        dscaling += jnp.sum(r_block.astype(jnp.float32) * z_block)
        return (da, db, dscaling), None

    (da, db, dscaling), _ = lax.scan(
        reduce_lora_partials,
        (
            jnp.zeros(lora_a.shape, dtype=jnp.float32),
            jnp.zeros(lora_b.shape, dtype=jnp.float32),
            jnp.zeros(lora_scaling.shape, dtype=jnp.float32),
        ),
        (blocked_x, blocked_dy, blocked_z, blocked_unscaled_r),
    )
    return (
        dx.astype(x.dtype),
        None,
        None,
        da.astype(lora_a.dtype),
        db.astype(lora_b.dtype),
        dscaling.astype(lora_scaling.dtype),
    )


_w8a8_lora_linear.defvjp(_w8a8_lora_linear_fwd, _w8a8_lora_linear_bwd)


def w8a8_frozen_linear(
    x: jax.Array,
    weight: GroupQuantizedWeight,
    *,
    enabled: bool = False,
    interpret: bool = False,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
    row_superblock: int = DEFAULT_ROW_SUPERBLOCK,
) -> jax.Array:
    """Apply only the compact W8A8 frozen base with a relaxed input VJP.

    This base-only route is the composable fallback for no-adapter and
    multi-adapter model calls: a future compact NNX layer can feed its result
    into the existing ``LoRAMixin.apply_lora`` routing without choosing or
    truncating an adapter inside this kernel.
    """
    _validate_exact_bool("enabled", enabled)
    _validate_exact_bool("interpret", interpret)
    if not enabled:
        raise RuntimeError(
            "the W8A8 Pallas frozen-linear experiment is disabled by default"
        )
    x = jnp.asarray(x)
    _validate_base_inputs(
        x,
        weight,
        block_m=block_m,
        block_n=block_n,
        row_superblock=row_superblock,
    )
    return _w8a8_frozen_linear(
        x,
        weight.codes,
        weight.scales,
        weight.group_size,
        block_m,
        block_n,
        row_superblock,
        interpret,
    )


def w8a8_lora_linear(
    x: jax.Array,
    weight: GroupQuantizedWeight,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array | float,
    *,
    enabled: bool = False,
    interpret: bool = False,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
    row_superblock: int = DEFAULT_ROW_SUPERBLOCK,
) -> jax.Array:
    """Run the bounded W8A8+LoRA experiment when explicitly enabled.

    ``interpret=True`` uses Pallas's HLO interpreter and is intended only for
    portable semantic tests.  ``interpret=False`` asks the installed Triton
    backend to lower the signed INT8 dots; that route remains unqualified until
    a guarded gfx1100 probe has passed.
    """
    _validate_exact_bool("enabled", enabled)
    _validate_exact_bool("interpret", interpret)
    if not enabled:
        raise RuntimeError("the W8A8 Pallas LoRA experiment is disabled by default")
    x = jnp.asarray(x)
    lora_a = jnp.asarray(lora_a)
    lora_b = jnp.asarray(lora_b)
    scaling = jnp.asarray(lora_scaling, dtype=jnp.float32)
    _validate_inputs(
        x,
        weight,
        lora_a,
        lora_b,
        scaling,
        block_m=block_m,
        block_n=block_n,
        row_superblock=row_superblock,
    )
    return _w8a8_lora_linear(
        x,
        weight.codes,
        weight.scales,
        lora_a,
        lora_b,
        scaling,
        weight.group_size,
        block_m,
        block_n,
        row_superblock,
        interpret,
    )


__all__ = (
    "DEFAULT_BLOCK_M",
    "DEFAULT_BLOCK_N",
    "DEFAULT_ROW_SUPERBLOCK",
    "MAX_BLOCK_M",
    "MAX_BLOCK_N",
    "MAX_IN_FEATURES",
    "MAX_OUT_FEATURES",
    "MAX_ROW_SUPERBLOCK",
    "W8A8_GROUP_SIZE",
    "W8A8_LORA_RANK",
    "w8a8_frozen_linear",
    "w8a8_lora_linear",
)
