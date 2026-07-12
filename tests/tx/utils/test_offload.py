import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec, SingleDeviceSharding

from skyrl.tx.utils import offload
from skyrl.tx.utils.offload import (
    move_array_to_memory_kind,
    move_variable_to_memory_kind,
)


@pytest.fixture(autouse=True)
def _force_cpu_default_device():
    """Keep every test operation on CPU even without JAX_PLATFORMS."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _cpu_sharding(memory_kind: str = "device") -> SingleDeviceSharding:
    return SingleDeviceSharding(jax.devices("cpu")[0], memory_kind=memory_kind)


def _committed_array(values, *, dtype=np.float32) -> jax.Array:
    return jax.device_put(np.asarray(values, dtype=dtype), _cpu_sharding())


def test_cpu_device_pinned_host_exact_roundtrip():
    source = _committed_array(np.arange(24).reshape(4, 6), dtype=np.int32)

    host = move_array_to_memory_kind(source, "pinned_host")
    restored = move_array_to_memory_kind(host, "device")

    assert host.sharding.memory_kind == "pinned_host"
    assert restored.sharding.memory_kind == "device"
    assert restored.shape == source.shape
    assert restored.dtype == source.dtype
    np.testing.assert_array_equal(np.asarray(restored), np.asarray(source))


def test_named_sharding_and_partition_spec_are_preserved():
    devices = jax.devices("cpu")
    mesh = Mesh(np.asarray(devices), ("data",))
    source_sharding = NamedSharding(mesh, PartitionSpec("data", None))
    source = jax.device_put(
        np.arange(len(devices) * 12, dtype=np.float32).reshape(len(devices), 12),
        source_sharding,
    )

    host = move_array_to_memory_kind(source, "pinned_host")

    assert isinstance(host.sharding, NamedSharding)
    assert host.sharding.mesh == source.sharding.mesh
    assert host.sharding.spec == source.sharding.spec
    assert host.sharding.device_set == source.sharding.device_set
    assert host.sharding == source.sharding.with_memory_kind("pinned_host")


@pytest.mark.parametrize("memory_kind", ["device", "pinned_host"])
def test_same_kind_move_is_identity_preserving(memory_kind):
    source = jax.device_put(np.arange(4), _cpu_sharding(memory_kind))

    result = move_array_to_memory_kind(source, memory_kind)

    assert result is source


class _AddressabilityStub(jax.Array):
    @property
    def committed(self):
        return True

    @property
    def is_fully_addressable(self):
        return False

    def is_deleted(self):
        return False


def test_rejects_non_array():
    with pytest.raises(TypeError, match="jax.Array"):
        move_array_to_memory_kind(np.arange(4), "pinned_host")


def test_rejects_uncommitted_array():
    with jax.default_device(jax.devices("cpu")[0]):
        source = jnp.arange(4)
    assert not source.committed

    with pytest.raises(ValueError, match="committed"):
        move_array_to_memory_kind(source, "pinned_host")


def test_rejects_non_fully_addressable_array():
    with pytest.raises(ValueError, match="fully addressable"):
        move_array_to_memory_kind(_AddressabilityStub(), "pinned_host")


@pytest.mark.parametrize("memory_kind", [None, "", "definitely_not_a_memory_kind"])
def test_rejects_invalid_or_unsupported_memory_kind(memory_kind):
    source = _committed_array([1.0])

    with pytest.raises((TypeError, ValueError), match="memory.kind|memory_kind"):
        move_array_to_memory_kind(source, memory_kind)


def test_rejects_deleted_array():
    source = _committed_array([1.0])
    source.delete()

    with pytest.raises(ValueError, match="deleted"):
        move_array_to_memory_kind(source, "pinned_host")


def test_transfer_does_not_delete_or_donate_source():
    source = _committed_array(np.arange(8))

    moved = move_array_to_memory_kind(source, "pinned_host")

    assert moved is not source
    assert not source.is_deleted()
    np.testing.assert_array_equal(np.asarray(source), np.arange(8, dtype=np.float32))


def test_transfer_passes_explicit_safe_device_put_arguments(monkeypatch):
    source = _committed_array(np.arange(8))
    calls = []
    real_device_put = jax.device_put

    def tracked_device_put(value, device=None, *, src=None, donate=False, may_alias=None):
        calls.append((value, device, src, donate, may_alias))
        return real_device_put(value, device, src=src, donate=donate, may_alias=may_alias)

    monkeypatch.setattr(jax, "device_put", tracked_device_put)

    move_array_to_memory_kind(source, "pinned_host")

    assert len(calls) == 1
    value, target, src, donate, may_alias = calls[0]
    assert value is source
    assert target == source.sharding.with_memory_kind("pinned_host")
    assert src is source.sharding
    assert donate is False
    assert may_alias is False


def test_variable_replacement_is_atomic_when_preparation_fails(monkeypatch):
    original = _committed_array([1.0, 2.0])
    variable = nnx.OptState(original)

    def fail_before_replacement(*args, **kwargs):
        raise RuntimeError("injected transfer failure")

    monkeypatch.setattr(offload, "move_array_to_memory_kind", fail_before_replacement)

    with pytest.raises(RuntimeError, match="injected transfer failure"):
        move_variable_to_memory_kind(variable, "pinned_host")
    assert variable.get_raw_value() is original


def test_variable_replacement_is_atomic_when_setter_rejects_update(monkeypatch):
    original = _committed_array([1.0, 2.0])
    variable = nnx.OptState(original)

    def reject_update(self):
        raise RuntimeError("injected setter failure")

    monkeypatch.setattr(nnx.Variable, "_check_can_update", reject_update)

    with pytest.raises(RuntimeError, match="injected setter failure"):
        move_variable_to_memory_kind(variable, "pinned_host")
    assert variable.get_raw_value() is original


def test_variable_replacement_commits_only_blocked_value(monkeypatch):
    original = _committed_array([1.0, 2.0])
    variable = nnx.OptState(original)
    events = []
    real_block_until_ready = jax.block_until_ready
    real_check_can_update = nnx.Variable._check_can_update

    def tracked_block_until_ready(value):
        events.append(("block", value))
        return real_block_until_ready(value)

    def tracked_check_can_update(self):
        events.append(("set", self))
        return real_check_can_update(self)

    monkeypatch.setattr(jax, "block_until_ready", tracked_block_until_ready)
    monkeypatch.setattr(nnx.Variable, "_check_can_update", tracked_check_can_update)

    moved = move_variable_to_memory_kind(variable, "pinned_host")

    assert variable.get_raw_value() is moved
    event_names = [event for event, _ in events]
    assert event_names[0] == "block"
    assert event_names[1:]
    assert set(event_names[1:]) == {"set"}


def test_variable_same_kind_move_preserves_raw_value_identity():
    original = _committed_array([1.0, 2.0])
    variable = nnx.OptState(original)

    moved = move_variable_to_memory_kind(variable, "device")

    assert moved is original
    assert variable.get_raw_value() is original


def test_variable_same_kind_move_detects_value_change_during_synchronization(monkeypatch):
    original = _committed_array([1.0, 2.0])
    concurrent = _committed_array([3.0, 4.0])
    variable = nnx.OptState(original)

    def synchronize_then_change(value, memory_kind):
        variable.set_raw_value(concurrent)
        return value

    monkeypatch.setattr(offload, "move_array_to_memory_kind", synchronize_then_change)

    with pytest.raises(RuntimeError, match="changed while the replacement was being prepared"):
        move_variable_to_memory_kind(variable, "device")
    assert variable.get_raw_value() is concurrent


class _CustomSetterVariable(nnx.Variable):
    def set_raw_value(self, value, **kwargs):
        return super().set_raw_value(value, **kwargs)


def test_variable_rejects_custom_setter_without_changing_identity():
    original = _committed_array([1.0, 2.0])
    variable = _CustomSetterVariable(original)

    with pytest.raises(TypeError, match="custom set_raw_value"):
        move_variable_to_memory_kind(variable, "pinned_host")
    assert variable.get_raw_value() is original


def _replace_adam_state(opt_state, adam_state):
    return (adam_state, *opt_state[1:])


def test_one_adamw_mu_leaf_can_be_offloaded_and_staged_bitwise():
    params = {"weight": _committed_array([1.5, -2.0, 0.25])}
    optimizer = optax.adamw(learning_rate=1e-2, b1=0.9, b2=0.99, weight_decay=1e-3)
    initial_state = optimizer.init(params)
    initial_adam = initial_state[0]
    initial_mu = jax.device_put(initial_adam.mu["weight"], _cpu_sharding())
    initial_adam = initial_adam._replace(mu={"weight": initial_mu})
    initial_state = _replace_adam_state(initial_state, initial_adam)

    control_params = params
    control_state = initial_state
    staged_params = params
    staged_state = _replace_adam_state(
        initial_state,
        initial_adam._replace(mu={"weight": move_array_to_memory_kind(initial_mu, "pinned_host")}),
    )

    def step(current_params, current_state, grads):
        updates, next_state = optimizer.update(grads, current_state, current_params)
        return optax.apply_updates(current_params, updates), next_state

    step = jax.jit(step)

    gradients = (
        np.asarray([0.25, -0.5, 0.125], dtype=np.float32),
        np.asarray([-0.75, 0.125, 0.5], dtype=np.float32),
        np.asarray([0.375, -0.25, -0.625], dtype=np.float32),
    )
    for gradient in gradients:
        grads = {"weight": jax.device_put(gradient, _cpu_sharding())}
        control_params, control_state = step(control_params, control_state, grads)

        staged_adam = staged_state[0]
        staged_mu = move_array_to_memory_kind(staged_adam.mu["weight"], "device")
        device_state = _replace_adam_state(
            staged_state,
            staged_adam._replace(mu={"weight": staged_mu}),
        )
        staged_params, staged_state = step(staged_params, device_state, grads)
        next_adam = staged_state[0]
        staged_state = _replace_adam_state(
            staged_state,
            next_adam._replace(mu={"weight": move_array_to_memory_kind(next_adam.mu["weight"], "pinned_host")}),
        )

    np.testing.assert_array_equal(np.asarray(staged_params["weight"]), np.asarray(control_params["weight"]))
    np.testing.assert_array_equal(np.asarray(staged_state[0].mu["weight"]), np.asarray(control_state[0].mu["weight"]))
    np.testing.assert_array_equal(np.asarray(staged_state[0].nu["weight"]), np.asarray(control_state[0].nu["weight"]))
    np.testing.assert_array_equal(np.asarray(staged_state[0].count), np.asarray(control_state[0].count))
    assert staged_state[0].mu["weight"].sharding.memory_kind == "pinned_host"
