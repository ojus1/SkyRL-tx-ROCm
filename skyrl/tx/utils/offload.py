"""Synchronous helpers for moving JAX arrays between memory kinds.

These primitives deliberately do not choose an offload policy.  Callers must
select the source leaf and the destination memory kind explicitly.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any, TypeAlias, TypeVar

import jax
from flax import nnx

_ArrayT = TypeVar("_ArrayT", bound=jax.Array)
MomentPath: TypeAlias = tuple[str | int, ...]
MomentPredicate: TypeAlias = Callable[[MomentPath, nnx.Variable], bool]

_BASE_VARIABLE_SET_RAW_VALUE = nnx.Variable.set_raw_value
_MOMENT_SLOTS = frozenset(("mu", "nu"))


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


def _validate_copy_target(array: jax.Array, target: jax.sharding.Sharding) -> None:
    if not isinstance(target, jax.sharding.Sharding):
        raise TypeError(f"Expected a jax.sharding.Sharding, got {type(target).__name__}")
    if not isinstance(target.memory_kind, str) or not target.memory_kind:
        raise ValueError("Target sharding must have a non-empty memory kind")
    if target.device_set != array.sharding.device_set:
        raise ValueError("Target sharding changes the source device assignment")
    if _partition_spec(target) != _partition_spec(array.sharding):
        raise ValueError("Target sharding changes the source partition spec")


def _copy_array_to_sharding(array: _ArrayT, target: jax.sharding.Sharding) -> _ArrayT:
    """Synchronously copy ``array`` to an exact, topology-preserving sharding."""
    _validate_source(array)
    _validate_copy_target(array, target)

    if array.sharding == target:
        jax.block_until_ready(array)
        _validate_result(array, array, target, target.memory_kind)
        return array

    result = jax.device_put(
        array,
        target,
        src=array.sharding,
        donate=False,
        may_alias=False,
    )
    jax.block_until_ready(result)
    _validate_result(result, array, target, target.memory_kind)
    if _is_deleted(array):
        raise RuntimeError("Non-donating device_put unexpectedly deleted the source array")
    return result


def move_array_to_memory_kind(array: _ArrayT, memory_kind: str) -> _ArrayT:
    """Synchronously copy a committed local JAX array to ``memory_kind``.

    The transfer is explicitly non-donating and non-aliasing.  A same-kind
    request validates and synchronizes the source, then returns it unchanged.
    No policy, traversal, or asynchronous prefetching is performed here.
    """
    _validate_source(array)
    target = _target_sharding(array, memory_kind)
    return _copy_array_to_sharding(array, target)


def move_variable_to_memory_kind(variable: nnx.Variable, memory_kind: str) -> jax.Array:
    """Atomically replace a basic NNX Variable's array with a moved copy.

    Preparation and synchronization finish before ``set_raw_value`` is called.
    Variables with custom setters are rejected because their external side
    effects cannot be rolled back safely by this primitive.
    """
    if not isinstance(variable, nnx.Variable):
        raise TypeError(f"Expected an nnx.Variable, got {type(variable).__name__}")
    if type(variable).set_raw_value is not _BASE_VARIABLE_SET_RAW_VALUE:
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


@dataclass(frozen=True, slots=True)
class MomentLeafManifest:
    """Immutable inventory for one explicitly selected optimizer-moment leaf."""

    path: MomentPath
    moment_slot: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    device_sharding: jax.sharding.Sharding
    offload_sharding: jax.sharding.Sharding

    @property
    def device_memory_kind(self) -> str | None:
        return self.device_sharding.memory_kind

    @property
    def offload_memory_kind(self) -> str | None:
        return self.offload_sharding.memory_kind


@dataclass(frozen=True, slots=True)
class _MomentBinding:
    variable: nnx.Variable
    manifest: MomentLeafManifest
    dtype: Any


class _HandlePhase(Enum):
    OFFLOADED = "offloaded"
    STAGED_BACK = "staged_back"
    COMPLETE = "complete"


class _HandleControl:
    """Private mutable state for a frozen, consumable public handle."""

    __slots__ = ("expected", "lock", "phase")

    def __init__(self, expected: tuple[jax.Array, ...]):
        self.expected = expected
        self.lock = RLock()
        self.phase = _HandlePhase.OFFLOADED


def _path_sort_key(path: tuple[object, ...]) -> tuple[tuple[str, str], ...]:
    """Return a total, deterministic order even for an unselected exotic path."""
    return tuple(
        (f"{type(component).__module__}.{type(component).__qualname__}", repr(component)) for component in path
    )


def _validate_explicit_path(path: object) -> MomentPath:
    if not isinstance(path, tuple) or not path:
        raise TypeError("Every optimizer-moment path must be a non-empty tuple")
    if any(type(component) not in (str, int) for component in path):
        raise TypeError("Optimizer-moment path components must be exact strings or integers")
    return path


def _moment_slot(path: MomentPath) -> str:
    # A terminal ``mu``/``nu`` may just be a parameter name.  Requiring the
    # exact component above the leaf matches Optax's moment subtrees without
    # relying on substring or suffix conventions.
    slots = [component for component in path[:-1] if type(component) is str and component in _MOMENT_SLOTS]
    if not slots:
        raise ValueError(f"Selected path {path!r} is not below an exact 'mu' or 'nu' optimizer slot")
    if len(slots) != 1:
        raise ValueError(f"Selected path {path!r} is ambiguous: it contains multiple exact 'mu'/'nu' components")
    return slots[0]


def _graph_variables(tree: object) -> dict[MomentPath, nnx.Variable]:
    variables: dict[MomentPath, nnx.Variable] = {}
    for raw_path, node in nnx.iter_graph(tree):
        if not isinstance(node, nnx.Variable):
            continue
        path = tuple(raw_path)
        if path in variables and variables[path] is not node:
            raise RuntimeError(f"NNX graph traversal returned two Variables at path {path!r}")
        variables[path] = node
    return variables


def _validate_basic_variable(path: MomentPath, variable: nnx.Variable) -> None:
    if not isinstance(variable, nnx.OptState):
        raise TypeError(f"Optimizer-moment path {path!r} is not an nnx.OptState Variable")
    for method_name in (
        "get_raw_value",
        "set_raw_value",
        "__setattr__",
        "_check_can_update",
        "__getattribute__",
    ):
        variable_method = getattr(type(variable), method_name, None)
        base_method = getattr(nnx.Variable, method_name, None)
        if variable_method is not base_method:
            raise TypeError(f"Optimizer-moment path {path!r} has an unsupported custom {method_name} implementation")


def _select_moment_variables(
    tree: object,
    *,
    paths: Iterable[MomentPath] | None,
    predicate: MomentPredicate | None,
) -> tuple[tuple[MomentPath, nnx.Variable, str], ...]:
    if (paths is None) == (predicate is None):
        raise TypeError("Provide exactly one explicit optimizer-moment selector: paths or predicate")

    variables = _graph_variables(tree)
    if paths is not None:
        requested_list = [_validate_explicit_path(path) for path in paths]
        requested = frozenset(requested_list)
        if len(requested) != len(requested_list):
            raise ValueError("Optimizer-moment paths must not contain duplicates")
        missing = requested.difference(variables)
        if missing:
            rendered = ", ".join(repr(path) for path in sorted(missing, key=_path_sort_key))
            raise ValueError(f"Optimizer-moment paths do not resolve to NNX Variables: {rendered}")
        selected = [(path, variables[path]) for path in sorted(requested, key=_path_sort_key)]
    else:
        assert predicate is not None
        if not callable(predicate):
            raise TypeError("predicate must be callable")
        selected = []
        for raw_path, variable in sorted(variables.items(), key=lambda item: _path_sort_key(item[0])):
            path = _validate_explicit_path(raw_path)
            if bool(predicate(path, variable)):
                selected.append((path, variable))

    if not selected:
        raise ValueError("The explicit optimizer-moment selector matched no Variables")

    validated = []
    for path, variable in selected:
        path = _validate_explicit_path(path)
        _validate_basic_variable(path, variable)
        validated.append((path, variable, _moment_slot(path)))
    return tuple(validated)


def _assert_bindings_current(root: object, bindings: tuple[_MomentBinding, ...]) -> None:
    current = _graph_variables(root)
    for binding in bindings:
        path = binding.manifest.path
        if current.get(path) is not binding.variable:
            raise RuntimeError(f"Optimizer-moment Variable binding at path {path!r} is stale")
        _validate_basic_variable(path, binding.variable)


def _assert_source_identities(bindings: tuple[_MomentBinding, ...], sources: tuple[jax.Array, ...]) -> None:
    for binding, source in zip(bindings, sources, strict=True):
        if binding.variable.get_raw_value() is not source:
            raise RuntimeError(f"Optimizer-moment value at path {binding.manifest.path!r} changed during staging")


def _validate_snapshot(
    bindings: tuple[_MomentBinding, ...],
    sources: tuple[jax.Array, ...],
    shardings: tuple[jax.sharding.Sharding, ...],
) -> None:
    for binding, source, sharding in zip(bindings, sources, shardings, strict=True):
        _validate_source(source)
        manifest = binding.manifest
        if source.shape != manifest.shape:
            raise RuntimeError(f"Optimizer-moment shape at path {manifest.path!r} changed")
        if source.dtype != binding.dtype:
            raise RuntimeError(f"Optimizer-moment dtype at path {manifest.path!r} changed")
        if source.sharding != sharding:
            raise RuntimeError(f"Optimizer-moment sharding at path {manifest.path!r} is stale")


def _rollback_replacements(
    bindings: tuple[_MomentBinding, ...],
    sources: tuple[jax.Array, ...],
    indices: tuple[int, ...],
) -> None:
    errors: list[BaseException] = []
    for index in reversed(indices):
        try:
            _validate_basic_variable(bindings[index].manifest.path, bindings[index].variable)
            _restore_raw_value(bindings[index].variable, sources[index])
        except BaseException as error:
            errors.append(error)
    for index in indices:
        try:
            if bindings[index].variable.get_raw_value() is not sources[index]:
                errors.append(
                    RuntimeError(f"Failed to roll back optimizer-moment path {bindings[index].manifest.path!r}")
                )
        except BaseException as error:
            errors.append(error)
    if errors:
        raise RuntimeError(
            f"Failed to roll back {len(errors)} optimizer-moment restoration/verification action(s)"
        ) from errors[0]


def _restore_raw_value(variable: nnx.Variable, source: jax.Array) -> None:
    """Restore a strictly validated basic Variable without invoking any NNX hook."""
    object.__setattr__(variable, "_raw_value", source)


def _commit_replacements(
    root: object,
    bindings: tuple[_MomentBinding, ...],
    sources: tuple[jax.Array, ...],
    destinations: tuple[jax.Array, ...],
) -> None:
    # These two checks are intentionally adjacent to the first setter: every
    # destination has already completed and no user code runs between the
    # all-source identity check and deterministic commit.
    _assert_bindings_current(root, bindings)
    _assert_source_identities(bindings, sources)

    committed: list[int] = []
    attempted: int | None = None
    try:
        for index, (binding, source, destination) in enumerate(zip(bindings, sources, destinations, strict=True)):
            attempted = index
            if binding.variable.get_raw_value() is not source:
                raise RuntimeError(f"Optimizer-moment value at path {binding.manifest.path!r} changed during commit")
            nnx.Variable.set_raw_value(binding.variable, destination)
            if binding.variable.get_raw_value() is not destination:
                raise RuntimeError(f"Optimizer-moment path {binding.manifest.path!r} rejected its replacement")
            committed.append(index)

        _assert_bindings_current(root, bindings)
        for binding, destination in zip(bindings, destinations, strict=True):
            if binding.variable.get_raw_value() is not destination:
                raise RuntimeError(f"Optimizer-moment value at path {binding.manifest.path!r} changed during commit")
    except BaseException:
        rollback_indices = list(committed)
        if attempted is not None and attempted not in rollback_indices:
            binding = bindings[attempted]
            if binding.variable.get_raw_value() is not sources[attempted]:
                rollback_indices.append(attempted)
        try:
            _rollback_replacements(bindings, sources, tuple(rollback_indices))
        except BaseException as rollback_error:
            raise RuntimeError("Optimizer-moment transaction failed and rollback was incomplete") from rollback_error
        raise


def _transactional_replace(
    root: object,
    bindings: tuple[_MomentBinding, ...],
    sources: tuple[jax.Array, ...],
    targets: tuple[jax.sharding.Sharding, ...],
) -> tuple[jax.Array, ...]:
    _assert_bindings_current(root, bindings)
    _assert_source_identities(bindings, sources)

    for source, target in zip(sources, targets, strict=True):
        _validate_source(source)
        _validate_copy_target(source, target)
    source_shardings = tuple(source.sharding for source in sources)
    prepared_tree = jax.device_put(
        sources,
        targets,
        src=source_shardings,
        donate=False,
        may_alias=False,
    )
    jax.block_until_ready(prepared_tree)
    if not isinstance(prepared_tree, tuple) or len(prepared_tree) != len(sources):
        raise RuntimeError("Batched device_put did not preserve the optimizer-moment tuple structure")
    prepared = prepared_tree
    for binding, source, target, destination in zip(bindings, sources, targets, prepared, strict=True):
        _validate_result(destination, source, target, target.memory_kind)
        if _is_deleted(source):
            raise RuntimeError(f"Preparing optimizer-moment path {binding.manifest.path!r} deleted its source")

    _commit_replacements(root, bindings, sources, prepared)
    return prepared


@dataclass(frozen=True, slots=True, eq=False)
class MomentTreeOffloadHandle:
    """Consumable handle for one stage-back/re-offload optimizer-moment cycle.

    ``manifest`` is immutable.  The private lock serializes phase methods made
    through this handle only: ``stage_back`` succeeds once, ``reoffload``
    succeeds once after that, and re-offload returns a fresh handle for a later
    training step.  It does not lock the external NNX tree.  The caller must
    guarantee exclusive ownership and no concurrent external tree writes for
    each complete method call.  A failed transaction does not consume the
    operation.
    """

    manifest: tuple[MomentLeafManifest, ...]
    _root: object = field(repr=False, compare=False)
    _bindings: tuple[_MomentBinding, ...] = field(repr=False, compare=False)
    _control: _HandleControl = field(repr=False, compare=False)

    @property
    def phase(self) -> str:
        with self._control.lock:
            return self._control.phase.value

    @property
    def leaf_count(self) -> int:
        return len(self.manifest)

    @property
    def total_bytes(self) -> int:
        return sum(leaf.nbytes for leaf in self.manifest)

    def stage_back(self) -> "MomentTreeOffloadHandle":
        """Synchronously restore every selected leaf to its exact device sharding.

        The caller must prevent concurrent external writes to the NNX tree for
        the duration of this call.
        """
        with self._control.lock:
            if self._control.phase is not _HandlePhase.OFFLOADED:
                raise RuntimeError(f"stage_back requires an offloaded handle, got {self._control.phase.value}")
            sources = self._control.expected
            targets = tuple(binding.manifest.device_sharding for binding in self._bindings)
            offload_shardings = tuple(binding.manifest.offload_sharding for binding in self._bindings)
            _assert_bindings_current(self._root, self._bindings)
            _assert_source_identities(self._bindings, sources)
            _validate_snapshot(self._bindings, sources, offload_shardings)
            _transactional_replace(self._root, self._bindings, sources, targets)
            self._control.expected = ()
            self._control.phase = _HandlePhase.STAGED_BACK
            return self

    def reoffload(self) -> "MomentTreeOffloadHandle":
        """Offload the current, possibly optimizer-updated values exactly once.

        Values may legitimately change between ``stage_back`` and this call.
        Their identities are snapshotted here and checked again after the whole
        destination batch is ready.  The returned fresh handle owns the next
        stage-back capability; this handle becomes permanently complete.  The
        caller must prevent concurrent external writes to the NNX tree for the
        duration of this call.
        """
        with self._control.lock:
            if self._control.phase is not _HandlePhase.STAGED_BACK:
                raise RuntimeError(f"reoffload requires a staged_back handle, got {self._control.phase.value}")
            _assert_bindings_current(self._root, self._bindings)
            sources = tuple(binding.variable.get_raw_value() for binding in self._bindings)
            device_shardings = tuple(binding.manifest.device_sharding for binding in self._bindings)
            targets = tuple(binding.manifest.offload_sharding for binding in self._bindings)
            _validate_snapshot(self._bindings, sources, device_shardings)
            successor_control = _HandleControl(())
            successor = MomentTreeOffloadHandle(
                manifest=self.manifest,
                _root=self._root,
                _bindings=self._bindings,
                _control=successor_control,
            )
            prepared = _transactional_replace(self._root, self._bindings, sources, targets)
            successor_control.expected = prepared
            self._control.expected = ()
            self._control.phase = _HandlePhase.COMPLETE
            return successor


def offload_optimizer_moments(
    tree: object,
    *,
    paths: Iterable[MomentPath] | None = None,
    predicate: MomentPredicate | None = None,
    memory_kind: str = "pinned_host",
) -> MomentTreeOffloadHandle:
    """Transactionally offload explicitly selected basic NNX ``mu``/``nu`` leaves.

    Exactly one of ``paths`` or ``predicate`` is required.  Paths are exact NNX
    graph paths and must sit below a literal ``mu`` or ``nu`` component; no
    name guessing, substring matching, or implicit traversal policy is used.
    Every copy is prepared and blocked before the first Variable is replaced.
    The caller must guarantee exclusive ownership and no concurrent external
    writes to ``tree`` for the duration of this call.
    """
    selected = _select_moment_variables(tree, paths=paths, predicate=predicate)
    originals = tuple(variable.get_raw_value() for _, variable, _ in selected)

    manifests = []
    for (path, _variable, slot), original in zip(selected, originals, strict=True):
        _validate_source(original)
        if original.sharding.memory_kind != "device":
            raise ValueError(f"Optimizer-moment source at path {path!r} is not in device memory")
        target = _target_sharding(original, memory_kind)
        if target == original.sharding:
            raise ValueError(f"Optimizer-moment destination at path {path!r} is already the source sharding")
        manifests.append(
            MomentLeafManifest(
                path=path,
                moment_slot=slot,
                shape=tuple(original.shape),
                dtype=str(original.dtype),
                nbytes=int(original.nbytes),
                device_sharding=original.sharding,
                offload_sharding=target,
            )
        )

    manifest = tuple(manifests)
    bindings = tuple(
        _MomentBinding(variable=variable, manifest=leaf, dtype=original.dtype)
        for (_path, variable, _slot), original, leaf in zip(selected, originals, manifest, strict=True)
    )
    targets = tuple(leaf.offload_sharding for leaf in manifest)
    control = _HandleControl(())
    handle = MomentTreeOffloadHandle(
        manifest=manifest,
        _root=tree,
        _bindings=bindings,
        _control=control,
    )
    prepared = _transactional_replace(tree, bindings, originals, targets)
    control.expected = prepared
    return handle


__all__ = [
    "MomentLeafManifest",
    "MomentPath",
    "MomentPredicate",
    "MomentTreeOffloadHandle",
    "move_array_to_memory_kind",
    "move_variable_to_memory_kind",
    "offload_optimizer_moments",
]
