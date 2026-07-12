"""Default-off semantic prototype for blockwise 8-bit AdamW moments.

This module is intentionally not wired into the trainer.  It provides a small,
eagerly validated reference implementation for studying optimizer-state
storage.  AdamW arithmetic and bias correction are performed in FP32; only the
moments carried between steps are quantized.  The acceptance test compares this
behavior directly with SkyRL's actual ``optax.adamw(mu_dtype=None)`` behavior,
including Optax's native BF16 arithmetic when parameters are BF16.

The block-16 candidate is explicitly **not production-qualified**: its strict
100-step BF16-reference quality gate is rejected.  The trainer remains unwired,
and memory savings below are capacity arithmetic rather than a speed or quality
claim.

The public signed primitive includes a conventional symmetric INT8 encoding.
The optimizer candidate instead uses a deterministic affine INT8 encoding with
a per-block offset and 16-element blocks, using the full byte range when a
small block has a large nonzero mean.  Block-32, block-64, and block-256
accounting are retained for comparison.  The nonnegative second moment uses a
UINT8 square-root encoding:
the stored byte is
linear in ``sqrt(nu)`` and dequantization squares it.  This companding is
important for Adam because its denominator also consumes ``sqrt(nu)``.  A
linear UINT8 encoding of ``nu`` makes small, valid entries round to zero when a
block contains an outlier.  All encodings use deterministic round-to-nearest,
ties-to-even and one scale per flattened, row-major block.

The validation in this prototype is deliberately eager (including finite-value
checks) and therefore the high-level step is not a JIT API.  A production GPU
implementation would need checked staging and custom fused kernels rather than
materializing both moments in FP32 as this semantic implementation does.
"""

from dataclasses import dataclass
from math import prod
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

ScaleDType = Literal["float32", "bfloat16"]

_INT8_MAX = 127
_UINT8_MAX = 255
_SUPPORTED_SCALE_DTYPES = (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16))
_SUPPORTED_PARAM_DTYPES = (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16))


def _validate_block_size(block_size: object) -> int:
    if type(block_size) is not int:
        raise TypeError("block_size must be an exact integer")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return block_size


def _validate_scale_dtype(scale_dtype: Any) -> jnp.dtype:
    dtype = jnp.dtype(scale_dtype)
    if dtype not in _SUPPORTED_SCALE_DTYPES:
        raise TypeError("scale_dtype must be float32 or bfloat16")
    return dtype


def _validated_floating_array(value: object, name: str) -> jax.Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must have a floating dtype")
    if not bool(np.asarray(jnp.all(jnp.isfinite(array)))):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validated_params(value: object) -> jax.Array:
    array = _validated_floating_array(value, "params")
    if array.dtype not in _SUPPORTED_PARAM_DTYPES:
        raise TypeError("params dtype must be exactly float32 or bfloat16")
    return array


def _shape_size(shape: tuple[int, ...]) -> int:
    return prod(shape, start=1)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class BlockwiseQuantizedTensor:
    """A padded byte payload and its explicit block/shape metadata.

    ``encoding`` is ``"symmetric"`` for a signed linear INT8 tensor,
    ``"affine_signed"`` for ``offset + byte * scale``, or
    ``"sqrt_nonnegative"`` for a nonnegative UINT8 tensor whose reconstructed
    value is ``(byte * scale) ** 2``.  Only the affine encoding has offsets.
    """

    values: jax.Array
    scales: jax.Array
    offsets: jax.Array | None
    original_shape: tuple[int, ...]
    original_size: int
    padded_size: int
    block_size: int
    encoding: Literal["symmetric", "affine_signed", "sqrt_nonnegative"]

    @property
    def pad_elements(self) -> int:
        return self.padded_size - self.original_size

    @property
    def block_count(self) -> int:
        return self.padded_size // self.block_size

    @property
    def nbytes(self) -> int:
        offset_bytes = (
            0
            if self.offsets is None
            else self.block_count * self.offsets.dtype.itemsize
        )
        return (
            self.padded_size
            + self.block_count * self.scales.dtype.itemsize
            + offset_bytes
        )

    def tree_flatten(self):
        children = (self.values, self.scales, self.offsets)
        metadata = (
            self.original_shape,
            self.original_size,
            self.padded_size,
            self.block_size,
            self.encoding,
        )
        return children, metadata

    @classmethod
    def tree_unflatten(cls, metadata, children):
        original_shape, original_size, padded_size, block_size, encoding = metadata
        values, scales, offsets = children
        return cls(
            values=values,
            scales=scales,
            offsets=offsets,
            original_shape=original_shape,
            original_size=original_size,
            padded_size=padded_size,
            block_size=block_size,
            encoding=encoding,
        )


def _quantize_blocks(
    array: jax.Array,
    *,
    block_size: int,
    scale_dtype: jnp.dtype,
    encoding: Literal["symmetric", "affine_signed", "sqrt_nonnegative"],
) -> BlockwiseQuantizedTensor:
    original_shape = tuple(array.shape)
    original_size = array.size
    block_count = (original_size + block_size - 1) // block_size
    padded_size = block_count * block_size

    flat = jnp.ravel(array).astype(jnp.float32)
    if padded_size != original_size:
        flat = jnp.pad(flat, (0, padded_size - original_size))
    blocks = jnp.reshape(flat, (block_count, block_size))

    offsets = None
    if encoding == "symmetric":
        magnitude = jnp.max(jnp.abs(blocks), axis=1)
        scale_fp32 = jnp.where(magnitude > 0, magnitude / _INT8_MAX, 1.0)
        # Quantize using the stored scale so BF16-scale behavior is represented
        # exactly, rather than computing bytes with a hidden FP32 scale.
        scales = scale_fp32.astype(scale_dtype)
        scales = jnp.where(
            scales > 0, scales, jnp.asarray(jnp.finfo(scale_dtype).tiny, scale_dtype)
        )
        quantized = jnp.clip(
            jnp.rint(blocks / scales[:, None]), -_INT8_MAX, _INT8_MAX
        ).astype(jnp.int8)
    elif encoding == "affine_signed":
        minimum = jnp.min(blocks, axis=1)
        maximum = jnp.max(blocks, axis=1)
        nonconstant = maximum > minimum
        # Divide before subtracting so opposite-sign finite FP32 extrema do not
        # overflow while their affine range is being constructed.
        scale_fp32 = jnp.where(nonconstant, maximum / 255.0 - minimum / 255.0, 1.0)
        scales = scale_fp32.astype(scale_dtype)
        scales = jnp.where(
            scales > 0, scales, jnp.asarray(jnp.finfo(scale_dtype).tiny, scale_dtype)
        )
        offset_fp32 = jnp.where(
            nonconstant,
            minimum * (127.0 / 255.0) + maximum * (128.0 / 255.0),
            minimum,
        )
        offsets = offset_fp32.astype(scale_dtype)
        quantized = jnp.clip(
            jnp.rint((blocks - offsets.astype(jnp.float32)[:, None]) / scales[:, None]),
            -128,
            127,
        ).astype(jnp.int8)
    else:
        roots = jnp.sqrt(blocks)
        magnitude = jnp.max(roots, axis=1)
        scale_fp32 = jnp.where(magnitude > 0, magnitude / _UINT8_MAX, 1.0)
        scales = scale_fp32.astype(scale_dtype)
        scales = jnp.where(
            scales > 0, scales, jnp.asarray(jnp.finfo(scale_dtype).tiny, scale_dtype)
        )
        quantized = jnp.clip(jnp.rint(roots / scales[:, None]), 0, _UINT8_MAX).astype(
            jnp.uint8
        )

    return BlockwiseQuantizedTensor(
        values=jnp.ravel(quantized),
        scales=scales,
        offsets=offsets,
        original_shape=original_shape,
        original_size=original_size,
        padded_size=padded_size,
        block_size=block_size,
        encoding=encoding,
    )


def quantize_symmetric_int8(
    value: object,
    *,
    block_size: int = 256,
    scale_dtype: Any = jnp.float32,
) -> BlockwiseQuantizedTensor:
    """Quantize a finite floating tensor using symmetric, blockwise INT8."""
    array = _validated_floating_array(value, "value")
    block_size = _validate_block_size(block_size)
    scale_dtype = _validate_scale_dtype(scale_dtype)
    return _quantize_blocks(
        array, block_size=block_size, scale_dtype=scale_dtype, encoding="symmetric"
    )


def quantize_affine_int8(
    value: object,
    *,
    block_size: int = 16,
    scale_dtype: Any = jnp.float32,
) -> BlockwiseQuantizedTensor:
    """Quantize a finite tensor to affine signed bytes with one block offset."""
    array = _validated_floating_array(value, "value")
    block_size = _validate_block_size(block_size)
    scale_dtype = _validate_scale_dtype(scale_dtype)
    return _quantize_blocks(
        array, block_size=block_size, scale_dtype=scale_dtype, encoding="affine_signed"
    )


def quantize_nonnegative_uint8(
    value: object,
    *,
    block_size: int = 256,
    scale_dtype: Any = jnp.float32,
) -> BlockwiseQuantizedTensor:
    """Quantize a finite nonnegative tensor using blockwise sqrt-encoded UINT8."""
    array = _validated_floating_array(value, "value")
    if bool(np.asarray(jnp.any(array < 0))):
        raise ValueError("value must be nonnegative")
    block_size = _validate_block_size(block_size)
    scale_dtype = _validate_scale_dtype(scale_dtype)
    return _quantize_blocks(
        array,
        block_size=block_size,
        scale_dtype=scale_dtype,
        encoding="sqrt_nonnegative",
    )


def _validate_quantized_tensor(
    value: object, expected_encoding: str
) -> BlockwiseQuantizedTensor:
    if not isinstance(value, BlockwiseQuantizedTensor):
        raise TypeError("value must be a BlockwiseQuantizedTensor")
    if value.encoding != expected_encoding:
        raise ValueError(
            f"expected {expected_encoding!r} encoding, got {value.encoding!r}"
        )
    _validate_block_size(value.block_size)
    if not isinstance(value.original_shape, tuple) or any(
        type(dimension) is not int for dimension in value.original_shape
    ):
        raise TypeError("original_shape must be a tuple of exact integers")
    if any(dimension < 0 for dimension in value.original_shape):
        raise ValueError("original_shape dimensions must be nonnegative")
    if type(value.original_size) is not int or value.original_size != _shape_size(
        value.original_shape
    ):
        raise ValueError("original_size does not match original_shape")
    if type(value.padded_size) is not int or value.padded_size < value.original_size:
        raise ValueError("padded_size must be an integer no smaller than original_size")
    if value.padded_size % value.block_size != 0:
        raise ValueError("padded_size must be divisible by block_size")
    expected_padded_size = (
        (value.original_size + value.block_size - 1) // value.block_size
    ) * value.block_size
    if value.padded_size != expected_padded_size:
        raise ValueError("padded_size is not the minimal complete block payload")
    if not isinstance(value.values, jax.Array):
        raise TypeError("values must be a jax.Array")
    if value.values.ndim != 1 or value.values.shape != (value.padded_size,):
        raise ValueError("values shape does not match padded_size")
    expected_dtype = jnp.uint8 if expected_encoding == "sqrt_nonnegative" else jnp.int8
    if value.values.dtype != expected_dtype:
        raise TypeError(f"values must have dtype {jnp.dtype(expected_dtype).name}")
    if not isinstance(value.scales, jax.Array):
        raise TypeError("scales must be a jax.Array")
    if value.scales.ndim != 1 or value.scales.shape != (value.block_count,):
        raise ValueError("scales shape does not match the block count")
    _validate_scale_dtype(value.scales.dtype)
    if not bool(np.asarray(jnp.all(jnp.isfinite(value.scales)))):
        raise ValueError("scales must contain only finite values")
    if not bool(np.asarray(jnp.all(value.scales > 0))):
        raise ValueError("scales must be strictly positive")
    if expected_encoding == "affine_signed":
        if not isinstance(value.offsets, jax.Array):
            raise TypeError("affine offsets must be a jax.Array")
        if value.offsets.ndim != 1 or value.offsets.shape != (value.block_count,):
            raise ValueError("offsets shape does not match the block count")
        if value.offsets.dtype != value.scales.dtype:
            raise TypeError("offsets dtype must match scales dtype")
        if not bool(np.asarray(jnp.all(jnp.isfinite(value.offsets)))):
            raise ValueError("offsets must contain only finite values")
    elif value.offsets is not None:
        raise ValueError("only affine signed encoding may carry offsets")
    return value


def _dequantize(value: BlockwiseQuantizedTensor, expected_encoding: str) -> jax.Array:
    value = _validate_quantized_tensor(value, expected_encoding)
    blocks = jnp.reshape(value.values, (value.block_count, value.block_size)).astype(
        jnp.float32
    )
    decoded = blocks * value.scales.astype(jnp.float32)[:, None]
    if expected_encoding == "affine_signed":
        decoded = decoded + value.offsets.astype(jnp.float32)[:, None]
    elif expected_encoding == "sqrt_nonnegative":
        decoded = jnp.square(decoded)
    return jnp.reshape(jnp.ravel(decoded)[: value.original_size], value.original_shape)


def dequantize_symmetric_int8(value: BlockwiseQuantizedTensor) -> jax.Array:
    """Dequantize a validated symmetric INT8 tensor to FP32."""
    return _dequantize(value, "symmetric")


def dequantize_affine_int8(value: BlockwiseQuantizedTensor) -> jax.Array:
    """Dequantize a validated affine signed-byte tensor to FP32."""
    return _dequantize(value, "affine_signed")


def dequantize_nonnegative_uint8(value: BlockwiseQuantizedTensor) -> jax.Array:
    """Dequantize a validated sqrt-encoded UINT8 tensor to FP32."""
    return _dequantize(value, "sqrt_nonnegative")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class QuantizedAdamWState:
    """Quantized moments for one parameter tensor."""

    count: int
    mu: BlockwiseQuantizedTensor
    nu: BlockwiseQuantizedTensor

    def tree_flatten(self):
        return (self.mu, self.nu), self.count

    @classmethod
    def tree_unflatten(cls, count, children):
        mu, nu = children
        return cls(count=count, mu=mu, nu=nu)


def _validate_state(state: object, shape: tuple[int, ...]) -> QuantizedAdamWState:
    if not isinstance(state, QuantizedAdamWState):
        raise TypeError("state must be a QuantizedAdamWState")
    if type(state.count) is not int or state.count < 0:
        raise ValueError("state.count must be a nonnegative exact integer")
    _validate_quantized_tensor(state.mu, "affine_signed")
    _validate_quantized_tensor(state.nu, "sqrt_nonnegative")
    if state.mu.original_shape != shape or state.nu.original_shape != shape:
        raise ValueError("optimizer moment shape does not match the parameter shape")
    if state.mu.block_size != state.nu.block_size:
        raise ValueError("optimizer moments must use the same block size")
    if state.mu.scales.dtype != state.nu.scales.dtype:
        raise ValueError("optimizer moments must use the same scale dtype")
    return state


def init_quantized_adamw(
    params: object,
    *,
    block_size: int = 16,
    scale_dtype: Any = jnp.float32,
) -> QuantizedAdamWState:
    """Initialize default-off quantized AdamW moments for one parameter tensor."""
    params_array = _validated_params(params)
    zeros = jnp.zeros(params_array.shape, dtype=jnp.float32)
    return QuantizedAdamWState(
        count=0,
        mu=quantize_affine_int8(zeros, block_size=block_size, scale_dtype=scale_dtype),
        nu=quantize_nonnegative_uint8(
            zeros, block_size=block_size, scale_dtype=scale_dtype
        ),
    )


def _validate_hyperparameter(
    name: str, value: object, *, lower: float, upper: float | None = None
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < lower or (upper is not None and result >= upper):
        interval = (
            f"[{lower}, {upper})" if upper is not None else f"[{lower}, infinity)"
        )
        raise ValueError(f"{name} must be in {interval}")
    return result


def quantized_adamw_update(
    params: object,
    grads: object,
    state: QuantizedAdamWState,
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    weight_decay: float = 0.0,
) -> tuple[jax.Array, QuantizedAdamWState, jax.Array]:
    """Apply one FP32 AdamW step and requantize the moments carried forward.

    The returned tuple is ``(new_params, new_state, fp32_update)``.  Parameter
    storage dtype is preserved; the explicit FP32 update is useful for semantic
    comparisons without conflating optimizer error with BF16 parameter casts.
    """
    params_array = _validated_params(params)
    grads_array = _validated_floating_array(grads, "grads")
    if grads_array.shape != params_array.shape:
        raise ValueError("grads shape must match params shape")
    state = _validate_state(state, tuple(params_array.shape))

    learning_rate = _validate_hyperparameter("learning_rate", learning_rate, lower=0.0)
    beta1 = _validate_hyperparameter("beta1", beta1, lower=0.0, upper=1.0)
    beta2 = _validate_hyperparameter("beta2", beta2, lower=0.0, upper=1.0)
    epsilon = _validate_hyperparameter("epsilon", epsilon, lower=0.0)
    if epsilon == 0:
        raise ValueError("epsilon must be strictly positive")
    weight_decay = _validate_hyperparameter("weight_decay", weight_decay, lower=0.0)

    params_fp32 = params_array.astype(jnp.float32)
    grads_fp32 = grads_array.astype(jnp.float32)
    mu = beta1 * dequantize_affine_int8(state.mu) + (1.0 - beta1) * grads_fp32
    nu = beta2 * dequantize_nonnegative_uint8(state.nu) + (1.0 - beta2) * jnp.square(
        grads_fp32
    )
    count = state.count + 1

    mu_hat = mu / (1.0 - beta1**count)
    nu_hat = nu / (1.0 - beta2**count)
    direction = mu_hat / (jnp.sqrt(nu_hat) + epsilon)
    update = -learning_rate * (direction + weight_decay * params_fp32)
    if not bool(np.asarray(jnp.all(jnp.isfinite(update)))):
        raise FloatingPointError("AdamW produced a non-finite update")

    next_params = (params_fp32 + update).astype(params_array.dtype)
    scale_dtype = state.mu.scales.dtype
    block_size = state.mu.block_size
    next_state = QuantizedAdamWState(
        count=count,
        mu=quantize_affine_int8(mu, block_size=block_size, scale_dtype=scale_dtype),
        nu=quantize_nonnegative_uint8(
            nu, block_size=block_size, scale_dtype=scale_dtype
        ),
    )
    return next_params, next_state, update


@dataclass(frozen=True, slots=True)
class MomentMemoryAccounting:
    """Exact payload accounting for two quantized moment slots."""

    element_count_per_slot: int
    block_size: int
    block_count_per_slot: int
    pad_elements_per_slot: int
    scale_bytes: int
    mu_has_offsets: bool
    quantized_values_bytes: int
    quantized_scales_bytes: int
    quantized_offsets_bytes: int
    quantized_total_bytes: int
    bf16_total_bytes: int

    @property
    def saved_bytes_vs_bf16(self) -> int:
        return self.bf16_total_bytes - self.quantized_total_bytes

    @property
    def compression_ratio_vs_bf16(self) -> float:
        return self.bf16_total_bytes / self.quantized_total_bytes


@dataclass(frozen=True, slots=True)
class QuantizedMomentQualityGateResult:
    """Structured decision from the fixed optimizer-moment quality contract."""

    passed: bool
    first_update_direction_cosine_sanity_only: float
    worst_relative_update_norm_error: float
    minimum_direction_cosine: float
    maximum_relative_update_norm_error: float
    reasons: tuple[str, ...]


def quantized_moment_quality_gate(
    first_update_direction_cosine_sanity_only: float,
    worst_relative_update_norm_error: float,
) -> QuantizedMomentQualityGateResult:
    """Evaluate the fixed gate; first-step cosine is only a gross sanity check.

    Step one precedes any carried-state quantization, so its cosine cannot
    establish quantized-state quality.  The worst 100-step relative update
    error is the substantive acceptance metric.
    """
    if isinstance(
        first_update_direction_cosine_sanity_only, (bool, np.bool_)
    ) or not isinstance(
        first_update_direction_cosine_sanity_only, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            "first_update_direction_cosine_sanity_only must be a real scalar"
        )
    if isinstance(worst_relative_update_norm_error, (bool, np.bool_)) or not isinstance(
        worst_relative_update_norm_error, (int, float, np.integer, np.floating)
    ):
        raise TypeError("worst_relative_update_norm_error must be a real scalar")

    cosine = float(first_update_direction_cosine_sanity_only)
    error = float(worst_relative_update_norm_error)
    minimum_cosine = 0.999
    maximum_error = 0.01
    reasons: list[str] = []
    if not np.isfinite(cosine):
        reasons.append("first-update direction cosine is non-finite")
    elif cosine < minimum_cosine:
        reasons.append(
            f"first-update direction cosine {cosine:.9f} is below {minimum_cosine:.3f}"
        )
    if not np.isfinite(error):
        reasons.append("worst relative update-norm error is non-finite")
    elif error > maximum_error:
        reasons.append(
            f"worst relative update-norm error {error:.9%} exceeds {maximum_error:.2%}"
        )

    return QuantizedMomentQualityGateResult(
        passed=not reasons,
        first_update_direction_cosine_sanity_only=cosine,
        worst_relative_update_norm_error=error,
        minimum_direction_cosine=minimum_cosine,
        maximum_relative_update_norm_error=maximum_error,
        reasons=tuple(reasons),
    )


def moment_memory_accounting(
    element_count_per_slot: int,
    *,
    block_size: int = 256,
    scale_dtype: Any = jnp.float32,
    affine_mu: bool = False,
) -> MomentMemoryAccounting:
    """Account exactly for two byte payloads, scales, and optional mu offsets.

    The scalar Adam step counter and Python metadata are excluded because this
    function compares moment payloads only.  Padding is included.
    """
    if type(element_count_per_slot) is not int:
        raise TypeError("element_count_per_slot must be an exact integer")
    if element_count_per_slot < 0:
        raise ValueError("element_count_per_slot must be nonnegative")
    if type(affine_mu) is not bool:
        raise TypeError("affine_mu must be a bool")
    block_size = _validate_block_size(block_size)
    scale_dtype = _validate_scale_dtype(scale_dtype)
    block_count = (element_count_per_slot + block_size - 1) // block_size
    padded_size = block_count * block_size
    values_bytes = 2 * padded_size
    scales_bytes = 2 * block_count * scale_dtype.itemsize
    offsets_bytes = block_count * scale_dtype.itemsize if affine_mu else 0
    return MomentMemoryAccounting(
        element_count_per_slot=element_count_per_slot,
        block_size=block_size,
        block_count_per_slot=block_count,
        pad_elements_per_slot=padded_size - element_count_per_slot,
        scale_bytes=scale_dtype.itemsize,
        mu_has_offsets=affine_mu,
        quantized_values_bytes=values_bytes,
        quantized_scales_bytes=scales_bytes,
        quantized_offsets_bytes=offsets_bytes,
        quantized_total_bytes=values_bytes + scales_bytes + offsets_bytes,
        bf16_total_bytes=2 * element_count_per_slot * jnp.dtype(jnp.bfloat16).itemsize,
    )
