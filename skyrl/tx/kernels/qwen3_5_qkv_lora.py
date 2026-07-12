"""Portable equations for a fused Qwen3.5 full-attention input stage.

This module is an equation and precision-policy experiment, not an accelerated
GPU kernel or a memory/performance oracle.  It keeps the complete operation
boundary that a future ROCm FFI implementation should own: decoder RMSNorm,
the interleaved frozen QKV projection plus single-adapter LoRA, Q/gate
splitting, distinct Q/K RMSNorms, and partial RoPE.

The explicit VJP treats the base projection as frozen and deliberately defines
an FP32-strengthened projection/normalization backward.  That policy is close
to, but not bit-equivalent to, autodiff through the current model-dtype BF16
dots.  It also recomputes the raw QKV projection in backward.  The residual
tuple therefore omits a ``[batch, sequence, fused_width]`` activation, but that
tuple-level fact alone is not a peak-memory result.  Inside SkyRL's existing
whole-layer remat, a consumer that needs Q/K/V/gate in backward causes an
original QKV projection, a layer-remat projection, and this VJP's projection;
ordinary rematerialized autodiff needs two projections instead of three.

The portable implementation also materializes large FP32 backward operands.
At the exact 32K Qwen3.5-4B geometry, ``dy_f32`` is
``[32768, 10240]`` (1.25 GiB), ``weight_f32`` is ``[2560, 10240]``
(100 MiB), and ``x_f32`` is ``[32768, 2560]`` (320 MiB).  A production ROCm
kernel must tile those calculations and own the base GEMM epilogue before any
speed or memory gain can be claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Qwen35QKVStageConfig:
    """Static geometry for :func:`qwen35_qkv_lora_rope`.

    The frozen projection uses SkyRL's interleaved layout.  For every KV head,
    one group contains the corresponding query-head Q/gate chunks followed by
    one K chunk and one V chunk.
    """

    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    rope_theta: float
    rms_norm_eps: float

    def __post_init__(self) -> None:
        if self.num_query_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("num_query_heads and num_kv_heads must be positive")
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if (
            self.rotary_dim < 0
            or self.rotary_dim > self.head_dim
            or self.rotary_dim % 2
        ):
            raise ValueError("rotary_dim must be even and lie in [0, head_dim]")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")

    @property
    def queries_per_kv(self) -> int:
        return self.num_query_heads // self.num_kv_heads

    @property
    def q_projection_width(self) -> int:
        # Each query head emits both one Q and one sigmoid-gate vector.
        return self.num_query_heads * 2 * self.head_dim

    @property
    def kv_projection_width(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def fused_width(self) -> int:
        return self.q_projection_width + 2 * self.kv_projection_width


def _delta_rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    """Qwen3.5 RMSNorm with FP32 reduction and delta-weight semantics."""
    output_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    variance = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
    normalized = x_f32 * jax.lax.rsqrt(variance + eps)
    return (normalized * (1.0 + weight.astype(jnp.float32))).astype(output_dtype)


def _delta_rms_norm_vjp(
    x: jax.Array,
    weight: jax.Array,
    dy: jax.Array,
    eps: float,
) -> tuple[jax.Array, jax.Array]:
    """Analytic VJP for :func:`_delta_rms_norm`, accumulated in FP32."""
    x_f32 = x.astype(jnp.float32)
    dy_f32 = dy.astype(jnp.float32)
    effective_weight = 1.0 + weight.astype(jnp.float32)
    inverse_rms = jax.lax.rsqrt(jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + eps)
    normalized = x_f32 * inverse_rms
    weighted_dy = dy_f32 * effective_weight
    dx = inverse_rms * (
        weighted_dy
        - normalized * jnp.mean(weighted_dy * normalized, axis=-1, keepdims=True)
    )
    reduction_axes = tuple(range(x.ndim - 1))
    dweight = jnp.sum(dy_f32 * normalized, axis=reduction_axes)
    return dx.astype(x.dtype), dweight.astype(weight.dtype)


def _rope_sin_cos(
    positions: jax.Array,
    rotary_dim: int,
    rope_theta: float,
) -> tuple[jax.Array, jax.Array]:
    fraction = 2 * jnp.arange(0, rotary_dim // 2, dtype=jnp.float32) / rotary_dim
    timescale = jnp.pow(rope_theta, fraction)
    angles = (positions[..., None] / timescale[None, None, :])[..., None, :]
    return jnp.sin(angles), jnp.cos(angles)


def _apply_partial_rope(
    x: jax.Array,
    positions: jax.Array,
    rotary_dim: int,
    rope_theta: float,
) -> jax.Array:
    if rotary_dim == 0:
        return x
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    first, second = jnp.split(x_rot, 2, axis=-1)
    sin, cos = _rope_sin_cos(positions, rotary_dim, rope_theta)
    rotated = jnp.concatenate(
        [first * cos - second * sin, first * sin + second * cos],
        axis=-1,
    ).astype(x.dtype)
    return jnp.concatenate([rotated, x_pass], axis=-1)


def _apply_partial_rope_transpose(
    dy: jax.Array,
    positions: jax.Array,
    rotary_dim: int,
    rope_theta: float,
) -> jax.Array:
    """Apply the transpose of the partial-RoPE rotation to a cotangent."""
    if rotary_dim == 0:
        return dy
    dy_rot, dy_pass = dy[..., :rotary_dim], dy[..., rotary_dim:]
    first, second = jnp.split(dy_rot.astype(jnp.float32), 2, axis=-1)
    sin, cos = _rope_sin_cos(positions, rotary_dim, rope_theta)
    unrotated = jnp.concatenate(
        [first * cos + second * sin, -first * sin + second * cos],
        axis=-1,
    ).astype(dy.dtype)
    return jnp.concatenate([unrotated, dy_pass], axis=-1)


def _frozen_lora_projection(
    x: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
) -> jax.Array:
    """Match the current model-dtype forward order ``xW + s(xA)B``."""
    base = x @ frozen_weight
    low_rank = (x @ lora_a) @ lora_b
    return base + low_rank * lora_scaling.astype(base.dtype)


def _frozen_lora_projection_vjp(
    x: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    dy: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Frozen-base VJP using the experiment's FP32 backward policy.

    This portable expression expands ``x``, ``dy``, and the frozen weight to
    FP32.  It specifies numerical intent; it is not the storage strategy for a
    production tiled kernel.
    """
    input_shape = x.shape
    x_f32 = x.reshape(-1, x.shape[-1]).astype(jnp.float32)
    dy_f32 = dy.reshape(-1, dy.shape[-1]).astype(jnp.float32)
    weight_f32 = frozen_weight.astype(jnp.float32)
    a_f32 = lora_a.astype(jnp.float32)
    b_f32 = lora_b.astype(jnp.float32)
    scaling_f32 = lora_scaling.astype(jnp.float32)

    intermediate = x_f32 @ a_f32
    lora_input_cotangent = scaling_f32 * (dy_f32 @ b_f32.T)
    dx = dy_f32 @ weight_f32.T + lora_input_cotangent @ a_f32.T
    da = x_f32.T @ lora_input_cotangent
    db = scaling_f32 * (intermediate.T @ dy_f32)
    dscaling = jnp.sum(dy_f32 * (intermediate @ b_f32)).reshape(lora_scaling.shape)
    return (
        dx.reshape(input_shape).astype(x.dtype),
        da.astype(lora_a.dtype),
        db.astype(lora_b.dtype),
        dscaling.astype(lora_scaling.dtype),
    )


def _split_interleaved_qkv(
    fused: jax.Array,
    config: Qwen35QKVStageConfig,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    group_q_width = config.queries_per_kv * 2 * config.head_dim
    group_width = group_q_width + 2 * config.head_dim
    grouped = fused.reshape(*fused.shape[:-1], config.num_kv_heads, group_width)
    q = grouped[..., :group_q_width].reshape(
        *fused.shape[:-1], config.q_projection_width
    )
    k = grouped[..., group_q_width : group_q_width + config.head_dim].reshape(
        *fused.shape[:-1], config.kv_projection_width
    )
    v = grouped[..., group_q_width + config.head_dim :].reshape(
        *fused.shape[:-1], config.kv_projection_width
    )
    return q, k, v


def _fuse_interleaved_qkv(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    config: Qwen35QKVStageConfig,
) -> jax.Array:
    group_q_width = config.queries_per_kv * 2 * config.head_dim
    grouped_q = q.reshape(*q.shape[:-1], config.num_kv_heads, group_q_width)
    grouped_k = k.reshape(*k.shape[:-1], config.num_kv_heads, config.head_dim)
    grouped_v = v.reshape(*v.shape[:-1], config.num_kv_heads, config.head_dim)
    return jnp.concatenate([grouped_q, grouped_k, grouped_v], axis=-1).reshape(
        *q.shape[:-1], config.fused_width
    )


def _qwen35_qkv_lora_rope_impl(
    hidden: jax.Array,
    input_norm_weight: jax.Array,
    q_norm_weight: jax.Array,
    k_norm_weight: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    positions: jax.Array,
    config: Qwen35QKVStageConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    normalized_hidden = _delta_rms_norm(hidden, input_norm_weight, config.rms_norm_eps)
    fused = _frozen_lora_projection(
        normalized_hidden,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
    )
    q_raw, k_raw, v_raw = _split_interleaved_qkv(fused, config)

    batch, sequence = hidden.shape[:2]
    q_and_gate = q_raw.reshape(
        batch, sequence, config.num_query_heads, 2 * config.head_dim
    )
    q_raw, gate = jnp.split(q_and_gate, 2, axis=-1)
    k_raw = k_raw.reshape(batch, sequence, config.num_kv_heads, config.head_dim)
    v = v_raw.reshape(batch, sequence, config.num_kv_heads, config.head_dim)

    q = _delta_rms_norm(q_raw, q_norm_weight, config.rms_norm_eps)
    k = _delta_rms_norm(k_raw, k_norm_weight, config.rms_norm_eps)
    q = _apply_partial_rope(q, positions, config.rotary_dim, config.rope_theta)
    k = _apply_partial_rope(k, positions, config.rotary_dim, config.rope_theta)
    gate = gate.reshape(batch, sequence, config.num_query_heads * config.head_dim)
    return q, k, v, gate


@partial(jax.custom_vjp, nondiff_argnums=(9,))
def _qwen35_qkv_lora_rope(
    hidden: jax.Array,
    input_norm_weight: jax.Array,
    q_norm_weight: jax.Array,
    k_norm_weight: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    positions: jax.Array,
    config: Qwen35QKVStageConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    return _qwen35_qkv_lora_rope_impl(
        hidden,
        input_norm_weight,
        q_norm_weight,
        k_norm_weight,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
        positions,
        config,
    )


def _qwen35_qkv_lora_rope_fwd(
    hidden: jax.Array,
    input_norm_weight: jax.Array,
    q_norm_weight: jax.Array,
    k_norm_weight: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    positions: jax.Array,
    config: Qwen35QKVStageConfig,
) -> tuple[
    tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, ...],
]:
    output = _qwen35_qkv_lora_rope_impl(
        hidden,
        input_norm_weight,
        q_norm_weight,
        k_norm_weight,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
        positions,
        config,
    )
    # Save inputs only.  In particular, do not retain normalized_hidden or the
    # raw fused projection; both are recomputed by the analytic backward.  This
    # describes the residual tuple, not compiled peak memory, and the extra
    # recomputation composes with the model's existing whole-layer remat.
    residual = (
        hidden,
        input_norm_weight,
        q_norm_weight,
        k_norm_weight,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
        positions,
    )
    return output, residual


def _qwen35_qkv_lora_rope_bwd(
    config: Qwen35QKVStageConfig,
    residual: tuple[jax.Array, ...],
    output_cotangents: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array | None, ...]:
    (
        hidden,
        input_norm_weight,
        q_norm_weight,
        k_norm_weight,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
        positions,
    ) = residual
    dq, dk, dv, dgate = output_cotangents

    # Recompute the projection inputs needed by the two RMSNorm VJPs.  Under
    # whole-layer remat this is an additional QKV projection after the layer
    # recomputation; it is an equation experiment, not a demonstrated win.
    normalized_hidden = _delta_rms_norm(hidden, input_norm_weight, config.rms_norm_eps)
    fused = _frozen_lora_projection(
        normalized_hidden,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
    )
    q_projection, k_projection, _ = _split_interleaved_qkv(fused, config)

    batch, sequence = hidden.shape[:2]
    q_and_gate = q_projection.reshape(
        batch, sequence, config.num_query_heads, 2 * config.head_dim
    )
    q_raw, _ = jnp.split(q_and_gate, 2, axis=-1)
    k_raw = k_projection.reshape(batch, sequence, config.num_kv_heads, config.head_dim)

    dq = _apply_partial_rope_transpose(
        dq, positions, config.rotary_dim, config.rope_theta
    )
    dk = _apply_partial_rope_transpose(
        dk, positions, config.rotary_dim, config.rope_theta
    )
    dq_raw, dq_norm_weight = _delta_rms_norm_vjp(
        q_raw, q_norm_weight, dq, config.rms_norm_eps
    )
    dk_raw, dk_norm_weight = _delta_rms_norm_vjp(
        k_raw, k_norm_weight, dk, config.rms_norm_eps
    )

    dq_projection = jnp.concatenate(
        [
            dq_raw,
            dgate.reshape(batch, sequence, config.num_query_heads, config.head_dim),
        ],
        axis=-1,
    ).reshape(batch, sequence, config.q_projection_width)
    dk_projection = dk_raw.reshape(batch, sequence, config.kv_projection_width)
    dv_projection = dv.reshape(batch, sequence, config.kv_projection_width)
    dfused = _fuse_interleaved_qkv(
        dq_projection,
        dk_projection,
        dv_projection,
        config,
    )

    dnormalized_hidden, dlora_a, dlora_b, dlora_scaling = _frozen_lora_projection_vjp(
        normalized_hidden,
        frozen_weight,
        lora_a,
        lora_b,
        lora_scaling,
        dfused,
    )
    dhidden, dinput_norm_weight = _delta_rms_norm_vjp(
        hidden,
        input_norm_weight,
        dnormalized_hidden,
        config.rms_norm_eps,
    )
    return (
        dhidden,
        dinput_norm_weight,
        dq_norm_weight,
        dk_norm_weight,
        None,  # The QKV base weight is frozen by contract.
        dlora_a,
        dlora_b,
        dlora_scaling,
        None,  # Integer token positions are nondifferentiable.
    )


_qwen35_qkv_lora_rope.defvjp(_qwen35_qkv_lora_rope_fwd, _qwen35_qkv_lora_rope_bwd)


def _validate_inputs(
    hidden: jax.Array,
    input_norm_weight: jax.Array,
    q_norm_weight: jax.Array,
    k_norm_weight: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array,
    positions: jax.Array,
    config: Qwen35QKVStageConfig,
) -> None:
    if hidden.ndim != 3:
        raise ValueError(
            f"hidden must have shape [batch, sequence, hidden_size], got {hidden.shape}"
        )
    if not jnp.issubdtype(hidden.dtype, jnp.floating):
        raise TypeError(f"hidden must be floating point, got {hidden.dtype}")
    hidden_size = hidden.shape[-1]
    expected_shapes = {
        "input_norm_weight": (hidden_size,),
        "q_norm_weight": (config.head_dim,),
        "k_norm_weight": (config.head_dim,),
        "frozen_weight": (hidden_size, config.fused_width),
    }
    actual_arrays = {
        "input_norm_weight": input_norm_weight,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "frozen_weight": frozen_weight,
    }
    for name, expected_shape in expected_shapes.items():
        value = actual_arrays[name]
        if value.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {value.shape}"
            )
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError(f"{name} must be floating point, got {value.dtype}")
    if lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError("lora_a and lora_b must both be rank-two")
    if lora_a.shape[0] != hidden_size or lora_a.shape[1] != lora_b.shape[0]:
        raise ValueError(f"incompatible LoRA shapes A={lora_a.shape}, B={lora_b.shape}")
    if lora_b.shape[1] != config.fused_width:
        raise ValueError(
            f"lora_b output width must be {config.fused_width}, got {lora_b.shape[1]}"
        )
    if not jnp.issubdtype(lora_a.dtype, jnp.floating) or not jnp.issubdtype(
        lora_b.dtype, jnp.floating
    ):
        raise TypeError(
            f"LoRA parameters must be floating point, got {lora_a.dtype} and {lora_b.dtype}"
        )
    model_dtype_arrays = {
        **actual_arrays,
        "lora_a": lora_a,
        "lora_b": lora_b,
    }
    for name, value in model_dtype_arrays.items():
        if value.dtype != hidden.dtype:
            raise TypeError(
                f"{name} must use the hidden/model dtype {hidden.dtype}, got {value.dtype}"
            )
    if lora_scaling.shape != () or not jnp.issubdtype(lora_scaling.dtype, jnp.floating):
        raise ValueError(
            f"lora_scaling must be a floating-point scalar, got {lora_scaling.shape}"
        )
    if positions.shape != hidden.shape[:2]:
        raise ValueError(
            f"positions must have shape {hidden.shape[:2]}, got {positions.shape}"
        )
    if not jnp.issubdtype(positions.dtype, jnp.integer):
        raise TypeError(f"positions must be integer, got {positions.dtype}")


def qwen35_qkv_lora_rope(
    hidden: jax.Array,
    input_norm_weight: jax.Array,
    q_norm_weight: jax.Array,
    k_norm_weight: jax.Array,
    frozen_weight: jax.Array,
    lora_a: jax.Array,
    lora_b: jax.Array,
    lora_scaling: jax.Array | float,
    positions: jax.Array,
    *,
    config: Qwen35QKVStageConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run the CPU-portable fused-stage equations with an explicit VJP.

    This contract handles one already-selected LoRA adapter.  Adapter routing
    remains outside the stage, which matches the batch-size-one fast path in
    :class:`skyrl.tx.layers.lora.LoRAMixin`.  The base projection has no
    cotangent; input, normalization, LoRA A/B, and scaling cotangents are
    defined explicitly.  The backward follows this module's FP32-strengthened
    precision policy rather than promising bit-equivalence with BF16 autodiff.

    Returns:
        ``(q, k, v, gate)`` in canonical Qwen3.5 attention shapes.
    """
    # The live Qwen3.5 path stores scaling in the model dtype.  Normalizing a
    # Python scalar here also prevents accidental mixed-dtype promotion inside
    # the proposed kernel boundary.
    scaling = jnp.asarray(lora_scaling, dtype=hidden.dtype)
    _validate_inputs(
        hidden,
        input_norm_weight,
        q_norm_weight,
        k_norm_weight,
        frozen_weight,
        lora_a,
        lora_b,
        scaling,
        positions,
        config,
    )
    return _qwen35_qkv_lora_rope(
        hidden,
        input_norm_weight,
        q_norm_weight,
        k_norm_weight,
        frozen_weight,
        lora_a,
        lora_b,
        scaling,
        positions,
        config,
    )
