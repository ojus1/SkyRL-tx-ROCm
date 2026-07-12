from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec, SingleDeviceSharding

from skyrl.tx.utils import offload
from skyrl.tx.utils.offload import (
    MomentTreeOffloadHandle,
    move_array_to_memory_kind,
    move_variable_to_memory_kind,
    offload_optimizer_moments,
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


_MU_PATH = ("opt_state", 0, "mu", "weight")
_NU_PATH = ("opt_state", 0, "nu", "weight")
_MOMENT_PATHS = frozenset((_MU_PATH, _NU_PATH))


def _moment_test_tree():
    return {
        "opt_state": (
            {
                "count": nnx.OptState(_committed_array(0, dtype=np.int32)),
                "mu": {"weight": nnx.OptState(_committed_array([1.0, 2.0]))},
                "nu": {"weight": nnx.OptState(_committed_array([3.0, 4.0]))},
                "momentum": {"weight": nnx.OptState(_committed_array([5.0, 6.0]))},
                "mu_extra": {"weight": nnx.OptState(_committed_array([7.0, 8.0]))},
            },
        )
    }


def _raw_at(tree, path):
    node = tree
    for component in path:
        node = node[component]
    return node.get_raw_value()


def test_moment_tree_exact_paths_select_only_literal_mu_and_nu():
    tree = _moment_test_tree()
    untouched_paths = (
        ("opt_state", 0, "count"),
        ("opt_state", 0, "momentum", "weight"),
        ("opt_state", 0, "mu_extra", "weight"),
    )
    untouched = {path: _raw_at(tree, path) for path in untouched_paths}

    handle = offload_optimizer_moments(tree, paths=_MOMENT_PATHS)

    assert isinstance(handle, MomentTreeOffloadHandle)
    assert tuple(leaf.path for leaf in handle.manifest) == (_MU_PATH, _NU_PATH)
    assert tuple(leaf.moment_slot for leaf in handle.manifest) == ("mu", "nu")
    assert all(_raw_at(tree, path).sharding.memory_kind == "pinned_host" for path in _MOMENT_PATHS)
    assert all(_raw_at(tree, path) is original for path, original in untouched.items())


@pytest.mark.parametrize(
    "path",
    [
        ("opt_state", 0, "count"),
        ("opt_state", 0, "momentum", "weight"),
        ("opt_state", 0, "adam_mu", "weight"),
        ("opt_state", 0, "mu_extra", "weight"),
    ],
)
def test_moment_tree_rejects_non_exact_or_non_moment_path(path):
    tree = _moment_test_tree()
    if path == ("opt_state", 0, "adam_mu", "weight"):
        tree["opt_state"][0]["adam_mu"] = {"weight": nnx.OptState(_committed_array([9.0]))}
    originals = {candidate: _raw_at(tree, candidate) for candidate in _MOMENT_PATHS}

    with pytest.raises(ValueError, match="exact 'mu' or 'nu'"):
        offload_optimizer_moments(tree, paths={path})

    assert all(_raw_at(tree, candidate) is original for candidate, original in originals.items())


def test_moment_tree_rejects_path_with_multiple_literal_moment_slots():
    tree = _moment_test_tree()
    ambiguous_path = ("opt_state", 0, "mu", "nu", "weight")
    ambiguous = nnx.OptState(_committed_array([9.0]))
    tree["opt_state"][0]["mu"]["nu"] = {"weight": ambiguous}
    original = ambiguous.get_raw_value()

    with pytest.raises(ValueError, match="ambiguous.*multiple exact"):
        offload_optimizer_moments(tree, paths={ambiguous_path})

    assert ambiguous.get_raw_value() is original


def test_moment_tree_rejects_param_even_below_exact_mu_slot():
    parameter = nnx.Param(_committed_array([1.0, 2.0]))
    tree = {"opt_state": ({"mu": {"weight": parameter}},)}
    original = parameter.get_raw_value()

    with pytest.raises(TypeError, match="not an nnx.OptState"):
        offload_optimizer_moments(tree, paths={_MU_PATH})

    assert parameter.get_raw_value() is original


class _CustomGetRawOptState(nnx.OptState):
    def get_raw_value(self):
        return super().get_raw_value()


class _CustomSetRawOptState(nnx.OptState):
    def set_raw_value(self, value, **kwargs):
        return super().set_raw_value(value, **kwargs)


class _CustomSetAttrOptState(nnx.OptState):
    def __setattr__(self, name, value):
        return super().__setattr__(name, value)


class _CustomCheckUpdateOptState(nnx.OptState):
    def _check_can_update(self):
        return super()._check_can_update()


class _CustomGetAttributeOptState(nnx.OptState):
    def __getattribute__(self, name):
        return super().__getattribute__(name)


@pytest.mark.parametrize(
    ("variable_type", "method_name"),
    [
        (_CustomGetRawOptState, "get_raw_value"),
        (_CustomSetRawOptState, "set_raw_value"),
        (_CustomSetAttrOptState, "__setattr__"),
        (_CustomCheckUpdateOptState, "_check_can_update"),
        (_CustomGetAttributeOptState, "__getattribute__"),
    ],
)
def test_moment_tree_rejects_malicious_optstate_method_override(variable_type, method_name):
    variable = variable_type(_committed_array([1.0, 2.0]))
    tree = {"opt_state": ({"mu": {"weight": variable}},)}
    original = variable.get_raw_value()

    with pytest.raises(TypeError, match=f"custom {method_name}"):
        offload_optimizer_moments(tree, paths={_MU_PATH})

    assert variable.get_raw_value() is original


def test_moment_tree_explicit_predicate_has_no_implicit_name_selection():
    tree = _moment_test_tree()
    selected = {_NU_PATH}

    handle = offload_optimizer_moments(tree, predicate=lambda path, _variable: path in selected)

    assert tuple(leaf.path for leaf in handle.manifest) == (_NU_PATH,)
    assert _raw_at(tree, _NU_PATH).sharding.memory_kind == "pinned_host"
    assert _raw_at(tree, _MU_PATH).sharding.memory_kind == "device"


def test_moment_tree_mid_batch_device_put_failure_changes_no_variable(monkeypatch):
    tree = _moment_test_tree()
    originals = {path: _raw_at(tree, path) for path in _MOMENT_PATHS}
    real_device_put = jax.device_put
    calls = []

    def fail_mid_batch(values, targets, *, src=None, donate=False, may_alias=None):
        calls.append((values, targets, src, donate, may_alias))
        # Materialize one destination leaf inside the mocked batch call, then
        # fail before returning the batch.  No Variable setter may have run.
        real_device_put(values[0], targets[0], src=src[0], donate=donate, may_alias=may_alias)
        raise RuntimeError("injected mid-batch device_put failure")

    monkeypatch.setattr(jax, "device_put", fail_mid_batch)

    with pytest.raises(RuntimeError, match="injected mid-batch device_put failure"):
        offload_optimizer_moments(tree, paths=_MOMENT_PATHS)

    assert len(calls) == 1
    values, targets, source_shardings, donate, may_alias = calls[0]
    assert len(values) == len(targets) == len(source_shardings) == 2
    assert donate is False
    assert may_alias is False
    assert all(_raw_at(tree, path) is original for path, original in originals.items())


def test_moment_tree_uses_one_device_put_and_one_batch_block_before_first_setter(monkeypatch):
    tree = _moment_test_tree()
    events = []
    device_put_calls = []
    block_calls = []
    real_device_put = jax.device_put
    real_block_until_ready = jax.block_until_ready
    real_check_can_update = nnx.Variable._check_can_update

    def tracked_device_put(values, targets, *, src=None, donate=False, may_alias=None):
        events.append("device_put")
        device_put_calls.append((values, targets, src, donate, may_alias))
        return real_device_put(values, targets, src=src, donate=donate, may_alias=may_alias)

    def tracked_block_until_ready(value):
        events.append("batch_block")
        block_calls.append(value)
        return real_block_until_ready(value)

    def tracked_check_can_update(self):
        events.append("setter")
        return real_check_can_update(self)

    monkeypatch.setattr(jax, "device_put", tracked_device_put)
    monkeypatch.setattr(jax, "block_until_ready", tracked_block_until_ready)
    monkeypatch.setattr(nnx.Variable, "_check_can_update", tracked_check_can_update)

    offload_optimizer_moments(tree, paths=_MOMENT_PATHS)

    assert len(device_put_calls) == 1
    assert len(block_calls) == 1
    values, targets, source_shardings, donate, may_alias = device_put_calls[0]
    assert isinstance(values, tuple) and len(values) == 2
    assert isinstance(targets, tuple) and len(targets) == 2
    assert isinstance(source_shardings, tuple) and len(source_shardings) == 2
    assert donate is False
    assert may_alias is False
    assert block_calls[0] is not values
    assert isinstance(block_calls[0], tuple) and len(block_calls[0]) == 2
    assert events.count("device_put") == 1
    assert events.count("batch_block") == 1
    assert events.index("batch_block") < events.index("setter")
    assert events.count("setter") >= 2


def test_moment_tree_concurrent_value_mutation_is_detected_before_commit(monkeypatch):
    tree = _moment_test_tree()
    original_mu = _raw_at(tree, _MU_PATH)
    original_nu = _raw_at(tree, _NU_PATH)
    concurrent_nu = move_array_to_memory_kind(_committed_array([30.0, 40.0]), "pinned_host")
    nu_variable = tree["opt_state"][0]["nu"]["weight"]
    real_device_put = jax.device_put
    copies = 0

    def mutate_during_batch_copy(values, targets, *, src=None, donate=False, may_alias=None):
        nonlocal copies
        copies += 1
        result = real_device_put(values, targets, src=src, donate=donate, may_alias=may_alias)
        nu_variable.set_raw_value(concurrent_nu)
        return result

    monkeypatch.setattr(jax, "device_put", mutate_during_batch_copy)

    with pytest.raises(RuntimeError, match="changed during staging"):
        offload_optimizer_moments(tree, paths=_MOMENT_PATHS)

    assert _raw_at(tree, _MU_PATH) is original_mu
    assert _raw_at(tree, _NU_PATH) is concurrent_nu
    assert _raw_at(tree, _NU_PATH) is not original_nu
    assert copies == 1


def test_moment_tree_mid_setter_failure_rolls_back_all_prior_replacements(monkeypatch):
    tree = _moment_test_tree()
    originals = {path: _raw_at(tree, path) for path in _MOMENT_PATHS}
    real_check_can_update = nnx.Variable._check_can_update
    real_rollback = offload._rollback_replacements
    setter_calls = 0
    rollback_observations = []

    def fail_second_setter(self):
        nonlocal setter_calls
        setter_calls += 1
        # A basic NNX setter checks once explicitly and once through
        # ``__setattr__``.  Failing the third check rejects the second leaf
        # after the first leaf has been fully replaced.
        if setter_calls == 3:
            raise RuntimeError("injected second-setter failure")
        return real_check_can_update(self)

    def observe_rollback(bindings, sources, indices):
        rollback_observations.append(indices)
        assert indices == (0,)
        assert bindings[0].variable.get_raw_value() is not sources[0]
        return real_rollback(bindings, sources, indices)

    monkeypatch.setattr(nnx.Variable, "_check_can_update", fail_second_setter)
    monkeypatch.setattr(offload, "_rollback_replacements", observe_rollback)

    with pytest.raises(RuntimeError, match="injected second-setter failure"):
        offload_optimizer_moments(tree, paths=_MOMENT_PATHS)

    assert setter_calls == 3
    assert rollback_observations == [(0,)]
    assert all(_raw_at(tree, path) is original for path, original in originals.items())


def test_moment_tree_rollback_attempts_every_leaf_after_first_restore_failure(monkeypatch):
    tree = _moment_test_tree()
    bias_path = ("opt_state", 0, "mu", "bias")
    tree["opt_state"][0]["mu"]["bias"] = nnx.OptState(_committed_array([9.0, 10.0]))
    paths = frozenset((*_MOMENT_PATHS, bias_path))
    originals = {path: _raw_at(tree, path) for path in paths}
    variables = {path: tree["opt_state"][0][path[2]][path[3]] for path in paths}
    real_check_can_update = nnx.Variable._check_can_update
    real_restore = offload._restore_raw_value
    setter_calls = 0
    restore_attempts = []

    def fail_third_setter(self):
        nonlocal setter_calls
        setter_calls += 1
        if setter_calls == 5:
            raise RuntimeError("injected third-setter failure")
        return real_check_can_update(self)

    def fail_first_restore(variable, source):
        restore_attempts.append(variable)
        if len(restore_attempts) == 1:
            raise RuntimeError("injected first-restore failure")
        return real_restore(variable, source)

    monkeypatch.setattr(nnx.Variable, "_check_can_update", fail_third_setter)
    monkeypatch.setattr(offload, "_restore_raw_value", fail_first_restore)

    with pytest.raises(RuntimeError, match="rollback was incomplete"):
        offload_optimizer_moments(tree, paths=paths)

    assert setter_calls == 5
    assert restore_attempts == [variables[_MU_PATH], variables[bias_path]]
    assert _raw_at(tree, bias_path) is originals[bias_path]
    assert _raw_at(tree, _NU_PATH) is originals[_NU_PATH]
    assert _raw_at(tree, _MU_PATH) is not originals[_MU_PATH]


def test_moment_tree_manifest_inventory_and_exact_named_sharding_roundtrip():
    devices = jax.devices("cpu")
    mesh = Mesh(np.asarray(devices), ("data",))
    source_sharding = NamedSharding(mesh, PartitionSpec("data", None))
    mu_array = jax.device_put(np.arange(len(devices) * 12, dtype=np.float32).reshape(len(devices), 12), source_sharding)
    nu_array = jax.device_put(np.arange(len(devices) * 6, dtype=np.int16).reshape(len(devices), 6), source_sharding)
    tree = {
        "opt_state": (
            {
                "mu": {"weight": nnx.OptState(mu_array)},
                "nu": {"weight": nnx.OptState(nu_array)},
            },
        )
    }

    handle = offload_optimizer_moments(tree, paths=_MOMENT_PATHS)

    assert handle.leaf_count == 2
    assert handle.total_bytes == mu_array.nbytes + nu_array.nbytes
    for leaf, original in zip(handle.manifest, (mu_array, nu_array), strict=True):
        assert leaf.shape == original.shape
        assert leaf.dtype == str(original.dtype)
        assert leaf.nbytes == original.nbytes
        assert leaf.device_sharding is original.sharding
        assert leaf.offload_sharding == original.sharding.with_memory_kind("pinned_host")
        assert leaf.device_memory_kind == "device"
        assert leaf.offload_memory_kind == "pinned_host"
        assert _raw_at(tree, leaf.path).sharding == leaf.offload_sharding

    handle.stage_back()

    assert all(_raw_at(tree, leaf.path).sharding == leaf.device_sharding for leaf in handle.manifest)
    np.testing.assert_array_equal(np.asarray(_raw_at(tree, _MU_PATH)), np.asarray(mu_array))
    np.testing.assert_array_equal(np.asarray(_raw_at(tree, _NU_PATH)), np.asarray(nu_array))


def test_moment_tree_manifest_is_frozen_and_handle_rejects_stale_binding():
    tree = _moment_test_tree()
    handle = offload_optimizer_moments(tree, paths=_MOMENT_PATHS)
    old_mu_variable = tree["opt_state"][0]["mu"]["weight"]
    old_mu_value = old_mu_variable.get_raw_value()

    with pytest.raises(FrozenInstanceError):
        handle.manifest[0].nbytes = 0

    replacement_value = move_array_to_memory_kind(_committed_array([10.0, 20.0]), "pinned_host")
    replacement_variable = nnx.OptState(replacement_value)
    tree["opt_state"][0]["mu"]["weight"] = replacement_variable

    with pytest.raises(RuntimeError, match="binding.*stale"):
        handle.stage_back()

    assert old_mu_variable.get_raw_value() is old_mu_value
    assert replacement_variable.get_raw_value() is replacement_value
    assert _raw_at(tree, _NU_PATH).sharding.memory_kind == "pinned_host"


@pytest.mark.parametrize("constructor_name", ["_HandleControl", "MomentTreeOffloadHandle"])
def test_moment_tree_initial_constructor_failure_precedes_all_mutation(monkeypatch, constructor_name):
    tree = _moment_test_tree()
    originals = {path: _raw_at(tree, path) for path in _MOMENT_PATHS}

    def fail_constructor(*args, **kwargs):
        raise RuntimeError(f"injected {constructor_name} failure")

    monkeypatch.setattr(offload, constructor_name, fail_constructor)

    with pytest.raises(RuntimeError, match=f"injected {constructor_name} failure"):
        offload_optimizer_moments(tree, paths=_MOMENT_PATHS)

    assert all(_raw_at(tree, path) is original for path, original in originals.items())


@pytest.mark.parametrize("constructor_name", ["_HandleControl", "MomentTreeOffloadHandle"])
def test_moment_tree_reoffload_constructor_failure_preserves_tree_and_old_phase(monkeypatch, constructor_name):
    tree = _moment_test_tree()
    handle = offload_optimizer_moments(tree, paths=_MOMENT_PATHS)
    handle.stage_back()
    staged_sources = {path: _raw_at(tree, path) for path in _MOMENT_PATHS}

    def fail_constructor(*args, **kwargs):
        raise RuntimeError(f"injected {constructor_name} failure")

    monkeypatch.setattr(offload, constructor_name, fail_constructor)

    with pytest.raises(RuntimeError, match=f"injected {constructor_name} failure"):
        handle.reoffload()

    assert handle.phase == "staged_back"
    assert all(_raw_at(tree, path) is source for path, source in staged_sources.items())
    assert all(source.sharding.memory_kind == "device" for source in staged_sources.values())


def test_moment_tree_handle_is_one_cycle_and_successor_owns_next_cycle():
    tree = _moment_test_tree()
    handle = offload_optimizer_moments(tree, paths=_MOMENT_PATHS)

    with pytest.raises(RuntimeError, match="requires a staged_back handle"):
        handle.reoffload()
    assert handle.stage_back() is handle
    assert handle.phase == "staged_back"
    with pytest.raises(RuntimeError, match="requires an offloaded handle"):
        handle.stage_back()

    successor = handle.reoffload()

    assert handle.phase == "complete"
    assert successor.phase == "offloaded"
    with pytest.raises(RuntimeError, match="got complete"):
        handle.reoffload()
    with pytest.raises(RuntimeError, match="got complete"):
        handle.stage_back()
    successor.stage_back()
    assert successor.phase == "staged_back"


def test_moment_tree_handle_rejects_stale_raw_value_without_partial_stage_back():
    tree = _moment_test_tree()
    handle = offload_optimizer_moments(tree, paths=_MOMENT_PATHS)
    original_nu = _raw_at(tree, _NU_PATH)
    replacement_mu = move_array_to_memory_kind(_committed_array([10.0, 20.0]), "pinned_host")
    tree["opt_state"][0]["mu"]["weight"].set_raw_value(replacement_mu)

    with pytest.raises(RuntimeError, match="changed during staging"):
        handle.stage_back()

    assert _raw_at(tree, _MU_PATH) is replacement_mu
    assert _raw_at(tree, _NU_PATH) is original_nu
    assert _raw_at(tree, _NU_PATH).sharding.memory_kind == "pinned_host"


class _TinyMomentModel(nnx.Module):
    def __init__(self, value):
        self.weight = nnx.Param(value)


def _tiny_optimizer_pair():
    control_model = _TinyMomentModel(_committed_array([1.5, -2.0, 0.25]))
    staged_model = _TinyMomentModel(_committed_array([1.5, -2.0, 0.25]))
    transformation = optax.adamw(learning_rate=1e-2, b1=0.9, b2=0.99, weight_decay=1e-3)
    return (
        control_model,
        nnx.Optimizer(control_model, transformation, wrt=nnx.Param),
        staged_model,
        nnx.Optimizer(staged_model, transformation, wrt=nnx.Param),
    )


def _optimizer_variable(optimizer, path):
    return dict(nnx.iter_graph(optimizer))[path].get_raw_value()


def test_three_step_nnx_adamw_moment_tree_offload_is_bitwise_equivalent():
    control_model, control_optimizer, staged_model, staged_optimizer = _tiny_optimizer_pair()
    paths = {
        ("opt_state", 0, "mu", "weight"),
        ("opt_state", 0, "nu", "weight"),
    }
    handle = offload_optimizer_moments(staged_optimizer, paths=paths)
    gradients = (
        np.asarray([0.25, -0.5, 0.125], dtype=np.float32),
        np.asarray([-0.75, 0.125, 0.5], dtype=np.float32),
        np.asarray([0.375, -0.25, -0.625], dtype=np.float32),
    )

    for gradient in gradients:
        control_grads = nnx.State({"weight": nnx.Param(_committed_array(gradient))})
        staged_grads = nnx.State({"weight": nnx.Param(_committed_array(gradient))})
        control_optimizer.update(control_model, control_grads)

        handle.stage_back()
        staged_optimizer.update(staged_model, staged_grads)
        handle = handle.reoffload()

        np.testing.assert_array_equal(np.asarray(staged_model.weight), np.asarray(control_model.weight))
        for path in paths:
            np.testing.assert_array_equal(
                np.asarray(_optimizer_variable(staged_optimizer, path)),
                np.asarray(_optimizer_variable(control_optimizer, path)),
            )
            assert _optimizer_variable(staged_optimizer, path).sharding.memory_kind == "pinned_host"
        np.testing.assert_array_equal(
            np.asarray(_optimizer_variable(staged_optimizer, ("opt_state", 0, "count"))),
            np.asarray(_optimizer_variable(control_optimizer, ("opt_state", 0, "count"))),
        )
        np.testing.assert_array_equal(
            np.asarray(_optimizer_variable(staged_optimizer, ("step",))),
            np.asarray(_optimizer_variable(control_optimizer, ("step",))),
        )
