"""Portable semantics for a bounded frozen-linear plus rank-8 LoRA stage.

This module is an unwired CPU/JAX oracle.  It specifies the forward order and
explicit VJP for a future bounded ROCm HIP/FFI operation; it is not selected by
model code and makes no GPU performance claim.

The forward matches SkyRL's model-dtype projection order::

    Y = X W + scaling * (X A) B

Rows are processed in fixed-size tiles, including a zero-padded final tile.
The frozen base weight and LoRA scale receive no cotangent.  Backward evaluates
the analytic ``dX``, ``dA``, and ``dB`` equations in FP32, reduces per-tile LoRA
partials in lexical tile order, and casts each result once to its declared
dtype.  An optional residual is added outside the custom-VJP boundary, so its
ordinary JAX cotangent is exactly the incoming output cotangent.
"""

from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp

LORA_RANK = 8
MAX_TOKEN_TILE_SIZE = 2048
_SUPPORTED_DTYPES = (jnp.bfloat16, jnp.float32)


def _validate_token_tile_size(token_tile_size: int) -> None:
    if isinstance(token_tile_size, bool) or not isinstance(token_tile_size, int):
        raise TypeError("token_tile_size must be an integer")
    if not 1 <= token_tile_size <= MAX_TOKEN_TILE_SIZE:
        raise ValueError(
            f"token_tile_size must be in [1, {MAX_TOKEN_TILE_SIZE}], "
            f"got {token_tile_size}"
        )


def _validate_inputs(
    x: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    residual: jax.Array | None,
    token_tile_size: int,
) -> None:
    _validate_token_tile_size(token_tile_size)
    if x.ndim < 1 or math.prod(x.shape[:-1]) <= 0:
        raise ValueError(f"x must have a nonempty row domain and final K axis, got {x.shape}")
    if frozen_weight.ndim != 2:
        raise ValueError(
            f"frozen_weight must have shape [K, N], got {frozen_weight.shape}"
        )
    if lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError("single-adapter lora_a and lora_b must both be rank-two")

    dtype = x.dtype
    if dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"x dtype must be BF16 or FP32, got {dtype}")
    for name, value in (
        ("frozen_weight", frozen_weight),
        ("lora_a", lora_a),
        ("lora_b", lora_b),
    ):
        if value.dtype != dtype:
            raise TypeError(f"{name} dtype {value.dtype} must match x dtype {dtype}")

    in_features = x.shape[-1]
    if in_features <= 0:
        raise ValueError("x input dimension must be positive")
    if frozen_weight.shape[0] != in_features:
        raise ValueError(
            f"x K={in_features} does not match frozen_weight K={frozen_weight.shape[0]}"
        )
    if lora_a.shape != (in_features, LORA_RANK):
        raise ValueError(
            f"lora_a must have shape ({in_features}, {LORA_RANK}), got {lora_a.shape}"
        )
    out_features = frozen_weight.shape[1]
    if out_features <= 0:
        raise ValueError("frozen_weight output dimension must be positive")
    if lora_b.shape != (LORA_RANK, out_features):
        raise ValueError(
            f"lora_b must have shape ({LORA_RANK}, {out_features}), got {lora_b.shape}"
        )
    if lora_scaling.ndim != 0 or not jnp.issubdtype(
        lora_scaling.dtype, jnp.floating
    ):
        raise TypeError(
            f"lora_scaling must be a floating scalar, got "
            f"shape={lora_scaling.shape}, dtype={lora_scaling.dtype}"
        )

    expected_output_shape = (*x.shape[:-1], out_features)
    if residual is not None:
        if residual.shape != expected_output_shape:
            raise ValueError(
                f"residual shape {residual.shape} must match output shape "
                f"{expected_output_shape}"
            )
        if residual.dtype != dtype:
            raise TypeError(
                f"residual dtype {residual.dtype} must match x dtype {dtype}"
            )


def _pad_and_tile_rows(value: jax.Array, token_tile_size: int) -> tuple[jax.Array, int]:
    row_count = math.prod(value.shape[:-1])
    flat = value.reshape((row_count, value.shape[-1]))
    padded_row_count = (
        (row_count + token_tile_size - 1) // token_tile_size * token_tile_size
    )
    padded = jnp.pad(flat, ((0, padded_row_count - row_count), (0, 0)))
    return padded.reshape((-1, token_tile_size, value.shape[-1])), row_count


def _bounded_forward_impl(
    x: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    token_tile_size: int,
) -> jax.Array:
    tiled_x, row_count = _pad_and_tile_rows(x, token_tile_size)

    def one_tile(x_tile: jax.Array) -> jax.Array:
        base = x_tile @ frozen_weight
        low_rank = (x_tile @ lora_a) @ lora_b
        return base + low_rank * lora_scaling.astype(base.dtype)

    tiled_output = jax.lax.map(one_tile, tiled_x)
    out_features = frozen_weight.shape[1]
    return tiled_output.reshape((-1, out_features))[:row_count].reshape(
        (*x.shape[:-1], out_features)
    )


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def _bounded_frozen_lora_linear(
    x: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    token_tile_size: int,
) -> jax.Array:
    return _bounded_forward_impl(
        x,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
        token_tile_size,
    )


def _bounded_frozen_lora_linear_fwd(
    x: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    token_tile_size: int,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    output = _bounded_forward_impl(
        x,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
        token_tile_size,
    )
    return output, (x, frozen_weight, lora_a, lora_b, lora_scaling)


def _bounded_frozen_lora_linear_bwd(
    token_tile_size: int,
    saved: tuple[jax.Array, ...],
    output_cotangent: jax.Array,
) -> tuple[jax.Array | None, ...]:
    x, frozen_weight, lora_a, lora_b, lora_scaling = saved
    tiled_x, row_count = _pad_and_tile_rows(x, token_tile_size)
    tiled_dy, cotangent_row_count = _pad_and_tile_rows(
        output_cotangent, token_tile_size
    )
    if cotangent_row_count != row_count:
        raise ValueError("output cotangent row count must match x")

    weight_f32 = frozen_weight.astype(jnp.float32)
    a_f32 = lora_a.astype(jnp.float32)
    b_f32 = lora_b.astype(jnp.float32)
    scaling_f32 = lora_scaling.astype(jnp.float32)

    def one_tile(
        values: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        x_tile, dy_tile = values
        x_f32 = x_tile.astype(jnp.float32)
        dy_f32 = dy_tile.astype(jnp.float32)
        intermediate = x_f32 @ a_f32
        lora_input_cotangent = scaling_f32 * (dy_f32 @ b_f32.T)
        dx = dy_f32 @ weight_f32.T + lora_input_cotangent @ a_f32.T
        da_partial = x_f32.T @ lora_input_cotangent
        db_partial = scaling_f32 * (intermediate.T @ dy_f32)
        return dx, da_partial, db_partial

    tiled_dx, da_partials, db_partials = jax.lax.map(
        one_tile, (tiled_x, tiled_dy)
    )

    def reduce_partials(
        accumulated: tuple[jax.Array, jax.Array],
        partials: tuple[jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        accumulated_da, accumulated_db = accumulated
        partial_da, partial_db = partials
        return (accumulated_da + partial_da, accumulated_db + partial_db), None

    (da, db), _ = jax.lax.scan(
        reduce_partials,
        (
            jnp.zeros(lora_a.shape, dtype=jnp.float32),
            jnp.zeros(lora_b.shape, dtype=jnp.float32),
        ),
        (da_partials, db_partials),
    )
    dx = tiled_dx.reshape((-1, x.shape[-1]))[:row_count].reshape(x.shape)
    return (
        dx.astype(x.dtype),
        None,
        da.astype(lora_a.dtype),
        db.astype(lora_b.dtype),
        None,
    )


_bounded_frozen_lora_linear.defvjp(
    _bounded_frozen_lora_linear_fwd,
    _bounded_frozen_lora_linear_bwd,
)


def bounded_frozen_lora_linear(
    x: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array | float,
    *,
    residual: jax.Array | None = None,
    token_tile_size: int = 512,
) -> jax.Array:
    """Apply an unwired bounded BF16/FP32 frozen linear plus rank-8 LoRA.

    ``x`` may have arbitrary nonempty leading dimensions; they are flattened
    into the row domain and restored after processing.  ``lora_a`` and
    ``lora_b`` describe one adapter and must have rank exactly eight.  The
    frozen base weight and scalar scale have symbolic-zero cotangents.
    """
    x = jnp.asarray(x)
    frozen_weight = jnp.asarray(frozen_weight)
    lora_a = jnp.asarray(lora_a)
    lora_b = jnp.asarray(lora_b)
    scaling = jnp.asarray(lora_scaling, dtype=jnp.float32)
    if residual is not None:
        residual = jnp.asarray(residual)
    _validate_inputs(
        x,
        frozen_weight,
        lora_a,
        lora_b,
        scaling,
        residual,
        token_tile_size,
    )
    output = _bounded_frozen_lora_linear(
        x,
        frozen_weight,
        lora_a,
        lora_b,
        scaling,
        token_tile_size,
    )
    return output if residual is None else output + residual
