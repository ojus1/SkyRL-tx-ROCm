"""Reference semantics for grouped quantized frozen-weight LoRA linears.

This module is deliberately a portable JAX reference, not the production GPU
implementation.  It defines the storage format, forward arithmetic, and custom
VJP that a ROCm W8/W4 fused projection must reproduce.  The integer einsums in
this file are not enabled in model code: on gfx1100 the production path should
replace them with bounded HIP/CK/rocWMMA dispatches after isolated validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class GroupQuantizedWeight:
    """A KxN weight grouped along K, with one scale per group and output.

    Four-bit codes contain two signed two's-complement nibbles per byte along
    the K dimension.  Eight-bit codes are stored directly as ``int8``.
    ``scales`` has shape ``[ceil(K / group_size), N]``.
    """

    codes: jax.Array
    scales: jax.Array
    original_in_features: int
    padded_in_features: int
    bits: int
    group_size: int


jax.tree_util.register_dataclass(
    GroupQuantizedWeight,
    data_fields=("codes", "scales"),
    meta_fields=("original_in_features", "padded_in_features", "bits", "group_size"),
)


def _validate_bits(bits: int) -> None:
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")


def _validate_group_size(group_size: int) -> None:
    if group_size <= 0 or group_size % 2:
        raise ValueError(
            f"group_size must be a positive even integer, got {group_size}"
        )


def _validate_quantized_weight(weight: GroupQuantizedWeight) -> None:
    _validate_bits(weight.bits)
    _validate_group_size(weight.group_size)
    if weight.original_in_features <= 0 or weight.padded_in_features <= 0:
        raise ValueError("quantized weight input dimensions must be positive")
    if weight.padded_in_features < weight.original_in_features:
        raise ValueError("padded_in_features must be at least original_in_features")
    if weight.padded_in_features % weight.group_size:
        raise ValueError("padded_in_features must be divisible by group_size")
    if weight.codes.ndim != 2 or weight.scales.ndim != 2:
        raise ValueError("quantized weight codes and scales must both be rank-two")

    expected_code_rows = weight.padded_in_features // (2 if weight.bits == 4 else 1)
    expected_scale_shape = (
        weight.padded_in_features // weight.group_size,
        weight.codes.shape[1],
    )
    if weight.codes.shape[0] != expected_code_rows:
        raise ValueError(
            f"code rows {weight.codes.shape[0]} do not match expected {expected_code_rows}"
        )
    if weight.scales.shape != expected_scale_shape:
        raise ValueError(
            f"scale shape {weight.scales.shape} does not match expected {expected_scale_shape}"
        )
    if weight.codes.shape[1] <= 0:
        raise ValueError("quantized weight output dimension must be positive")
    expected_code_dtype = jnp.uint8 if weight.bits == 4 else jnp.int8
    if weight.codes.dtype != expected_code_dtype:
        raise TypeError(
            f"W{weight.bits} codes must have dtype {expected_code_dtype}, got {weight.codes.dtype}"
        )
    if not jnp.issubdtype(weight.scales.dtype, jnp.floating):
        raise TypeError(
            f"quantized weight scales must be floating point, got {weight.scales.dtype}"
        )


def pack_signed_int4(values: jax.Array) -> jax.Array:
    """Pack pairs from axis zero into low/high two's-complement nibbles.

    Inputs are expected to be in ``[-8, 7]``. As a low-level JAX packing
    primitive, out-of-range inputs retain only their low nibble. The public
    quantizer always supplies representable values.
    """
    values = jnp.asarray(values)
    if values.ndim < 1 or values.shape[0] == 0 or values.shape[0] % 2:
        raise ValueError("signed INT4 packing requires an even, nonempty axis zero")
    if not jnp.issubdtype(values.dtype, jnp.integer):
        raise TypeError(f"signed INT4 values must be integral, got {values.dtype}")
    low = jnp.bitwise_and(values[0::2].astype(jnp.int16), 0xF).astype(jnp.uint8)
    high = jnp.bitwise_and(values[1::2].astype(jnp.int16), 0xF).astype(jnp.uint8)
    return jnp.bitwise_or(low, jnp.left_shift(high, 4))


def unpack_signed_int4(packed: jax.Array) -> jax.Array:
    """Unpack low/high two's-complement nibbles along axis zero."""
    packed = jnp.asarray(packed)
    if packed.ndim < 1 or packed.shape[0] == 0:
        raise ValueError("signed INT4 input must have a nonempty axis zero")
    if packed.dtype != jnp.uint8:
        raise TypeError(f"packed signed INT4 values must be uint8, got {packed.dtype}")
    low_u8 = jnp.bitwise_and(packed, 0xF)
    high_u8 = jnp.right_shift(packed, 4)
    low = jnp.where(low_u8 >= 8, low_u8.astype(jnp.int16) - 16, low_u8).astype(jnp.int8)
    high = jnp.where(high_u8 >= 8, high_u8.astype(jnp.int16) - 16, high_u8).astype(
        jnp.int8
    )
    interleaved = jnp.stack((low, high), axis=1)
    return interleaved.reshape((packed.shape[0] * 2, *packed.shape[1:]))


def quantize_frozen_weight(
    weight: jax.Array,
    *,
    bits: int = 8,
    group_size: int = 64,
    scale_dtype: jnp.dtype = jnp.bfloat16,
) -> GroupQuantizedWeight:
    """Symmetrically quantize a KxN frozen weight with per-group/output scales."""
    _validate_bits(bits)
    _validate_group_size(group_size)
    weight = jnp.asarray(weight)
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [K, N], got {weight.shape}")
    if weight.shape[0] <= 0 or weight.shape[1] <= 0:
        raise ValueError(f"weight dimensions must be positive, got {weight.shape}")
    if not jnp.issubdtype(weight.dtype, jnp.floating):
        raise TypeError(f"weight must be floating point, got {weight.dtype}")
    if not jnp.issubdtype(jnp.dtype(scale_dtype), jnp.floating):
        raise TypeError(
            f"scale_dtype must be floating point, got {jnp.dtype(scale_dtype)}"
        )

    in_features, out_features = weight.shape
    padded_in_features = ((in_features + group_size - 1) // group_size) * group_size
    padded = jnp.pad(
        weight.astype(jnp.float32), ((0, padded_in_features - in_features), (0, 0))
    )
    grouped = padded.reshape((-1, group_size, out_features))
    qmax = (1 << (bits - 1)) - 1
    amax = jnp.max(jnp.abs(grouped), axis=1)
    scales_f32 = jnp.where(amax > 0, amax / qmax, 1.0)
    quantized = jnp.clip(
        jnp.rint(grouped / scales_f32[:, None, :]), -qmax, qmax
    ).astype(jnp.int8)
    quantized = quantized.reshape((padded_in_features, out_features))
    codes = quantized if bits == 8 else pack_signed_int4(quantized)
    return GroupQuantizedWeight(
        codes=codes,
        scales=scales_f32.astype(scale_dtype),
        original_in_features=in_features,
        padded_in_features=padded_in_features,
        bits=bits,
        group_size=group_size,
    )


def _unpacked_codes(weight: GroupQuantizedWeight) -> jax.Array:
    _validate_quantized_weight(weight)
    codes = weight.codes if weight.bits == 8 else unpack_signed_int4(weight.codes)
    return codes[: weight.padded_in_features].astype(jnp.int8)


def dequantize_frozen_weight(
    weight: GroupQuantizedWeight,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Materialize the KxN dequantized reference weight."""
    if not jnp.issubdtype(jnp.dtype(dtype), jnp.floating):
        raise TypeError(
            f"dequantized weight dtype must be floating point, got {jnp.dtype(dtype)}"
        )
    codes = _unpacked_codes(weight)
    out_features = codes.shape[1]
    grouped = codes.reshape((-1, weight.group_size, out_features)).astype(jnp.float32)
    dequantized = grouped * weight.scales.astype(jnp.float32)[:, None, :]
    return dequantized.reshape((weight.padded_in_features, out_features))[
        : weight.original_in_features
    ].astype(dtype)


def _dynamic_group_quantize(
    x: jax.Array, bits: int, group_size: int
) -> tuple[jax.Array, jax.Array]:
    """Return int8-held group codes and FP32 scales for a 2D activation."""
    qmax = (1 << (bits - 1)) - 1
    grouped = x.astype(jnp.float32).reshape(
        (x.shape[0], x.shape[1] // group_size, group_size)
    )
    amax = jnp.max(jnp.abs(grouped), axis=-1)
    scales = jnp.where(amax > 0, amax / qmax, 1.0)
    codes = jnp.clip(jnp.rint(grouped / scales[..., None]), -qmax, qmax).astype(
        jnp.int8
    )
    return codes, scales


def _quantized_frozen_linear_f32(
    x: jax.Array,
    weight: GroupQuantizedWeight,
    *,
    activation_bits: int = 8,
) -> jax.Array:
    """Return the FP32 epilogue input for the grouped quantized linear."""
    _validate_bits(activation_bits)
    _validate_quantized_weight(weight)
    if x.ndim < 1:
        raise ValueError(f"activations must have at least one dimension, got {x.shape}")
    if not jnp.issubdtype(x.dtype, jnp.floating):
        raise TypeError(f"activations must be floating point, got {x.dtype}")
    if x.shape[-1] != weight.original_in_features:
        raise ValueError(
            f"activation K={x.shape[-1]} does not match weight K={weight.original_in_features}"
        )
    flat = x.reshape((math.prod(x.shape[:-1]), x.shape[-1]))
    padded = jnp.pad(
        flat, ((0, 0), (0, weight.padded_in_features - weight.original_in_features))
    )
    x_codes, x_scales = _dynamic_group_quantize(
        padded, activation_bits, weight.group_size
    )
    w_codes = _unpacked_codes(weight).reshape(
        (-1, weight.group_size, weight.codes.shape[-1])
    )
    accum = jnp.einsum(
        "mgk,gkn->mgn",
        x_codes.astype(jnp.int32),
        w_codes.astype(jnp.int32),
        preferred_element_type=jnp.int32,
    )
    scaled = (
        accum.astype(jnp.float32)
        * x_scales[..., None]
        * weight.scales.astype(jnp.float32)[None]
    )
    output = jnp.sum(scaled, axis=1)
    return output.reshape((*x.shape[:-1], w_codes.shape[-1]))


def quantized_frozen_linear(
    x: jax.Array,
    weight: GroupQuantizedWeight,
    *,
    activation_bits: int = 8,
) -> jax.Array:
    """Portable grouped W8/W4-A8/A4 reference linear.

    Accumulation is INT32 within each group and FP32 across scaled groups.  The
    result is cast back to the activation dtype.  This operation's ordinary JAX
    derivative is intentionally not used; :func:`quantized_lora_linear` defines
    the training VJP.
    """
    return _quantized_frozen_linear_f32(
        x, weight, activation_bits=activation_bits
    ).astype(x.dtype)


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9))
def _quantized_lora_linear(
    x: jax.Array,
    weight_codes: jax.Array,
    weight_scales: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    weight_bits: int,
    activation_bits: int,
    group_size: int,
    original_in_features: int,
) -> jax.Array:
    weight = GroupQuantizedWeight(
        codes=weight_codes,
        scales=weight_scales,
        original_in_features=original_in_features,
        padded_in_features=(
            weight_codes.shape[0] * 2 if weight_bits == 4 else weight_codes.shape[0]
        ),
        bits=weight_bits,
        group_size=group_size,
    )
    # Keep the base rescale and LoRA add in one FP32 epilogue. Casting the base
    # to BF16 before the low-rank add would define a different training path.
    base = _quantized_frozen_linear_f32(x, weight, activation_bits=activation_bits)
    low_rank = (x.astype(jnp.float32) @ lora_a.astype(jnp.float32)) @ lora_b.astype(
        jnp.float32
    )
    return (base + low_rank * lora_scaling.astype(jnp.float32)).astype(x.dtype)


def _quantized_lora_linear_fwd(
    x: jax.Array,
    weight_codes: jax.Array,
    weight_scales: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    weight_bits: int,
    activation_bits: int,
    group_size: int,
    original_in_features: int,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    output = _quantized_lora_linear(
        x,
        weight_codes,
        weight_scales,
        lora_a,
        lora_b,
        lora_scaling,
        weight_bits,
        activation_bits,
        group_size,
        original_in_features,
    )
    # Retain the compact representation, not a full FP32 KxN pullback
    # residual. The portable backward unpacks group-by-group; production HIP
    # must likewise stream packed tiles directly into the dX contraction.
    return output, (x, weight_codes, weight_scales, lora_a, lora_b, lora_scaling)


def _quantized_lora_linear_bwd(
    weight_bits: int,
    activation_bits: int,
    group_size: int,
    original_in_features: int,
    residual: tuple[jax.Array, ...],
    output_cotangent: jax.Array,
) -> tuple[jax.Array | None, ...]:
    del activation_bits
    x, weight_codes, weight_scales, lora_a, lora_b, lora_scaling = residual
    x_f32 = x.astype(jnp.float32)
    dy_f32 = output_cotangent.astype(jnp.float32)
    a_f32 = lora_a.astype(jnp.float32)
    b_f32 = lora_b.astype(jnp.float32)
    scaling_f32 = lora_scaling.astype(jnp.float32)

    row_count = math.prod(x.shape[:-1])
    x_flat = x_f32.reshape((row_count, x.shape[-1]))
    dy_flat = dy_f32.reshape((row_count, output_cotangent.shape[-1]))
    intermediate = x_flat @ a_f32
    low_rank_input_cotangent = (dy_flat @ b_f32.T) * scaling_f32

    padded_in_features = (
        weight_codes.shape[0] * 2 if weight_bits == 4 else weight_codes.shape[0]
    )
    code_group_size = group_size // 2 if weight_bits == 4 else group_size
    grouped_codes = weight_codes.reshape((-1, code_group_size, weight_codes.shape[-1]))
    scales_f32 = weight_scales.astype(jnp.float32)

    def one_group_input_vjp(group_values: tuple[jax.Array, jax.Array]) -> jax.Array:
        group_codes, group_scales = group_values
        # Keep the compact dtype across the map boundary. In particular, W4 is
        # unpacked one K-group at a time rather than expanded into a full int8
        # KxN matrix before the loop.
        if weight_bits == 4:
            group_codes = unpack_signed_int4(group_codes)
        group_weight = group_codes.astype(jnp.float32) * group_scales[None, :]
        return dy_flat @ group_weight.T

    # [G, M, Kgroup] -> [M, padded_K], without retaining or explicitly
    # materializing the full dequantized KxN base matrix.
    grouped_dx = jax.lax.map(one_group_input_vjp, (grouped_codes, scales_f32))
    base_dx = jnp.transpose(grouped_dx, (1, 0, 2)).reshape(
        (dy_flat.shape[0], padded_in_features)
    )
    dx = base_dx[:, :original_in_features] + low_rank_input_cotangent @ a_f32.T
    da = x_flat.T @ low_rank_input_cotangent
    db = (intermediate.T @ dy_flat) * scaling_f32
    unscaled_lora = intermediate @ b_f32
    dscaling = jnp.sum(dy_flat * unscaled_lora).reshape(lora_scaling.shape)
    return (
        dx.reshape(x.shape).astype(x.dtype),
        None,
        None,
        da.astype(lora_a.dtype),
        db.astype(lora_b.dtype),
        dscaling.astype(lora_scaling.dtype),
    )


_quantized_lora_linear.defvjp(_quantized_lora_linear_fwd, _quantized_lora_linear_bwd)


def quantized_lora_linear(
    x: jax.Array,
    weight: GroupQuantizedWeight,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array | float,
    *,
    activation_bits: int = 8,
) -> jax.Array:
    """Apply a frozen grouped-quantized linear plus trainable LoRA.

    The base path uses dynamic grouped activation quantization in forward.  Its
    input VJP is a straight-through linear using the dequantized frozen weight;
    LoRA ``dX``, ``dA``, and ``dB`` follow the exact FP32 equations.  Weight
    codes and scales deliberately receive no gradients.
    """
    _validate_bits(activation_bits)
    _validate_quantized_weight(weight)
    if x.ndim < 1:
        raise ValueError(f"activations must have at least one dimension, got {x.shape}")
    if not jnp.issubdtype(x.dtype, jnp.floating):
        raise TypeError(f"activations must be floating point, got {x.dtype}")
    if lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError("lora_a and lora_b must both be rank-two")
    if not jnp.issubdtype(lora_a.dtype, jnp.floating) or not jnp.issubdtype(
        lora_b.dtype, jnp.floating
    ):
        raise TypeError(
            f"LoRA parameters must be floating point, got {lora_a.dtype} and {lora_b.dtype}"
        )
    if lora_a.shape != (weight.original_in_features, lora_b.shape[0]):
        raise ValueError(
            f"incompatible LoRA shapes: A={lora_a.shape}, B={lora_b.shape}, "
            f"weight K={weight.original_in_features}"
        )
    if lora_b.shape[1] != weight.codes.shape[-1]:
        raise ValueError(
            f"LoRA output N={lora_b.shape[1]} does not match weight N={weight.codes.shape[-1]}"
        )
    scaling = jnp.asarray(lora_scaling, dtype=jnp.float32)
    if scaling.ndim != 0:
        raise ValueError(f"lora_scaling must be scalar, got {scaling.shape}")
    return _quantized_lora_linear(
        x,
        weight.codes,
        weight.scales,
        lora_a,
        lora_b,
        scaling,
        weight.bits,
        activation_bits,
        weight.group_size,
        weight.original_in_features,
    )
