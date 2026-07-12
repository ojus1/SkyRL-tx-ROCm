"""Small, synchronous helpers for moving JAX arrays between memory kinds.

These primitives deliberately do not choose an offload policy.  Callers must
select the source leaf and the destination memory kind explicitly.
"""

from typing import TypeVar

import jax
from flax import nnx

_ArrayT = TypeVar("_ArrayT", bound=jax.Array)


def _is_deleted(array: jax.Array) -> bool:
    """Return deletion state through JAX's public Array API."""
    return bool(array.is_deleted())


def _partition_spec(sharding: jax.sharding.Sharding):
    """Return a sharding's partition spec, or a sentinel for non-named shardings."""
    return getattr(sharding, "spec", _NO_SPEC)


_NO_SPEC = object()


def _validate_source(array: jax.Array) -> None:
    if not isinstance(array, jax.Array):
        raise TypeError(f"Expected a jax.Array, got {type(array).__name__}")
    if _is_deleted(array):
        raise ValueError("Cannot move a deleted jax.Array")
    if not array.committed:
        raise ValueError("Offload requires a committed jax.Array")
    if not array.is_fully_addressable:
        raise ValueError("Offload requires a fully addressable jax.Array")


def _target_sharding(array: jax.Array, memory_kind: str) -> jax.sharding.Sharding:
    if not isinstance(memory_kind, str):
        raise TypeError(f"memory_kind must be a string, got {type(memory_kind).__name__}")
    if not memory_kind:
        raise ValueError("memory_kind must not be empty")

    source = array.sharding
    with_memory_kind = getattr(source, "with_memory_kind", None)
    if not callable(with_memory_kind):
        raise TypeError(f"{type(source).__name__} does not support memory-kind placement")

    try:
        target = with_memory_kind(memory_kind)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unsupported memory kind {memory_kind!r} for {source}") from error

    if target.memory_kind != memory_kind:
        raise RuntimeError(f"Target sharding reported memory kind {target.memory_kind!r}, expected {memory_kind!r}")
    if target.device_set != source.device_set:
        raise RuntimeError("Changing memory kind unexpectedly changed the device assignment")
    if _partition_spec(target) != _partition_spec(source):
        raise RuntimeError("Changing memory kind unexpectedly changed the partition spec")
    return target


def _validate_result(
    result: jax.Array,
    source: jax.Array,
    target: jax.sharding.Sharding,
    memory_kind: str,
) -> None:
    if not isinstance(result, jax.Array):
        raise RuntimeError(f"device_put returned {type(result).__name__}, not a jax.Array")
    if _is_deleted(result):
        raise RuntimeError("device_put returned a deleted jax.Array")
    if not result.committed:
        raise RuntimeError("device_put returned an uncommitted jax.Array")
    if not result.is_fully_addressable:
        raise RuntimeError("device_put returned a non-fully-addressable jax.Array")
    if result.shape != source.shape:
        raise RuntimeError(f"device_put changed shape from {source.shape} to {result.shape}")
    if result.dtype != source.dtype:
        raise RuntimeError(f"device_put changed dtype from {source.dtype} to {result.dtype}")
    if result.sharding.memory_kind != memory_kind:
        raise RuntimeError(
            "device_put placed the array in memory kind " f"{result.sharding.memory_kind!r}, expected {memory_kind!r}"
        )
    if result.sharding.device_set != source.sharding.device_set:
        raise RuntimeError("device_put changed the device assignment")
    if _partition_spec(result.sharding) != _partition_spec(source.sharding):
        raise RuntimeError("device_put changed the partition spec")
    if result.sharding != target:
        raise RuntimeError("device_put did not preserve the exact target sharding")


def move_array_to_memory_kind(array: _ArrayT, memory_kind: str) -> _ArrayT:
    """Synchronously copy a committed local JAX array to ``memory_kind``.

    The transfer is explicitly non-donating and non-aliasing.  A same-kind
    request validates and synchronizes the source, then returns it unchanged.
    No policy, traversal, or asynchronous prefetching is performed here.
    """
    _validate_source(array)
    target = _target_sharding(array, memory_kind)

    if array.sharding == target:
        jax.block_until_ready(array)
        _validate_result(array, array, target, memory_kind)
        return array

    result = jax.device_put(
        array,
        target,
        src=array.sharding,
        donate=False,
        may_alias=False,
    )
    jax.block_until_ready(result)
    _validate_result(result, array, target, memory_kind)
    if _is_deleted(array):
        raise RuntimeError("Non-donating device_put unexpectedly deleted the source array")
    return result


def move_variable_to_memory_kind(variable: nnx.Variable, memory_kind: str) -> jax.Array:
    """Atomically replace a basic NNX Variable's array with a moved copy.

    Preparation and synchronization finish before ``set_raw_value`` is called.
    Variables with custom setters are rejected because their external side
    effects cannot be rolled back safely by this primitive.
    """
    if not isinstance(variable, nnx.Variable):
        raise TypeError(f"Expected an nnx.Variable, got {type(variable).__name__}")
    if type(variable).set_raw_value is not nnx.Variable.set_raw_value:
        raise TypeError("Variables with a custom set_raw_value implementation are not supported")

    original = variable.get_raw_value()
    prepared = move_array_to_memory_kind(original, memory_kind)
    if prepared is original:
        if variable.get_raw_value() is not original:
            raise RuntimeError("Variable value changed while the replacement was being prepared")
        return original

    if variable.get_raw_value() is not original:
        raise RuntimeError("Variable value changed while the replacement was being prepared")

    try:
        variable.set_raw_value(prepared)
    except BaseException:
        # The supported NNX setter checks writability before its single raw
        # assignment.  Keep a defensive rollback in case that contract changes.
        if variable.get_raw_value() is not original:
            nnx.Variable.set_raw_value(variable, original, _unsafe_bypass_check=True)
        raise

    if variable.get_raw_value() is not prepared:
        nnx.Variable.set_raw_value(variable, original, _unsafe_bypass_check=True)
        raise RuntimeError("NNX Variable did not retain the prepared replacement")
    return prepared


__all__ = ["move_array_to_memory_kind", "move_variable_to_memory_kind"]
