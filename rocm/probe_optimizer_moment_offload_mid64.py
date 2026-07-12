#!/usr/bin/env python3
"""Guarded 64 MiB transactional optimizer-moment offload rung.

The default ``abstract`` mode emits a refusal manifest without importing JAX,
Flax, or SkyRL's offload implementation.  The accelerator path requires the
exact explicit invocation ``--platform rocm --allow-gpu --case mid64 --output``
in a fresh headless process under ``rocm/profile_rocm.py``.

``mid64`` constructs 64 distinct deterministic nonzero BF16 ``nnx.OptState``
leaves of exactly 1 MiB each: 32 below a literal ``mu`` component and 32 below
literal ``nu``.  It performs one initial offload, an untimed warmup
stage/re-offload pair, three measured stage/re-offload cycles, then a final
stage-back, one complete host bitwise oracle, and a final re-offload.  Every
handle method is synchronously bounded below 100 ms and its subsequent tuple
barrier below 10 ms.  Only the three middle cycles contribute performance
statistics.

Allocator ``bytes_in_use`` reversals are hard gates.  BFC pool statistics are
recorded exactly and physical VRAM/GTT plateaus remain separate informational
evidence because retained physical pages are permitted.  This probe performs
no optimizer update, model invocation, compilation, overlap experiment, or
larger/exact-state offload.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, ContextManager, TextIO

_CASE = "mid64"
_LEAVES_PER_SLOT = 32
_LEAF_COUNT = 2 * _LEAVES_PER_SLOT
_LEAF_SHAPE = (512, 1024)
_LEAF_BYTES = math.prod(_LEAF_SHAPE) * 2
_SELECTED_BYTES = _LEAF_COUNT * _LEAF_BYTES
_COUNT_SENTINEL = 64173
_MU_PATHS = tuple(
    ("opt_state", 0, "mu", f"leaf_{index:02d}") for index in range(_LEAVES_PER_SLOT)
)
_NU_PATHS = tuple(
    ("opt_state", 0, "nu", f"leaf_{index:02d}") for index in range(_LEAVES_PER_SLOT)
)
_SELECTED_PATHS = (*_MU_PATHS, *_NU_PATHS)
_COUNT_PATH = ("opt_state", 0, "count")
_TIMED_CYCLES = 3
_MAX_METHOD_SECONDS = 0.1
_MAX_POST_BARRIER_SECONDS = 0.01
_ALLOCATOR_DELTA_FRACTION = 0.95
_PHYSICAL_RELEASE_FRACTION = 0.90
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_COUNTER_PREFIXES = frozenset(
    (
        "warmup_stage_back",
        "warmup_reoffload",
        "timed_stage_back",
        "timed_reoffload",
        "final_stage_back",
        "final_reoffload",
    )
)


def _smoke_probe() -> Any:
    try:
        from rocm import probe_optimizer_moment_offload
    except ModuleNotFoundError:
        import probe_optimizer_moment_offload  # type: ignore[no-redef]

    return probe_optimizer_moment_offload


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _emit(record: dict[str, Any], output: TextIO) -> None:
    output.write(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    output.flush()


def _redacted_message_summary(error: BaseException) -> dict[str, Any]:
    encoded = str(error).encode("utf-8", errors="replace")
    return {
        "message_redacted": True,
        "message_utf8_bytes": len(encoded),
        "message_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    smoke_file = Path(_smoke_probe().__file__)
    return {
        "probe_source_sha256": _sha256_file(Path(__file__)),
        "delegated_smoke_probe_source_sha256": _sha256_file(smoke_file),
        "offload_source_sha256": _sha256_file(
            _SOURCE_ROOT / "skyrl" / "tx" / "utils" / "offload.py"
        ),
        "safety_source_sha256": _sha256_file(
            _SOURCE_ROOT / "rocm" / "amdgpu_safety.py"
        ),
    }


def _zero_counters() -> dict[str, int]:
    counters = {
        "construction_device_put_attempts": 0,
        "construction_device_put_completions": 0,
        "initial_offload_attempts": 0,
        "initial_offload_completions": 0,
        "post_barrier_attempts": 0,
        "post_barrier_completions": 0,
        "device_get_attempts": 0,
        "device_get_completions": 0,
    }
    for prefix in sorted(_COUNTER_PREFIXES):
        counters[f"{prefix}_attempts"] = 0
        counters[f"{prefix}_completions"] = 0
    return counters


def _completed_counters() -> dict[str, int]:
    counters = _zero_counters()
    counters.update(
        {
            "construction_device_put_attempts": 1,
            "construction_device_put_completions": 1,
            "initial_offload_attempts": 1,
            "initial_offload_completions": 1,
            "warmup_stage_back_attempts": 1,
            "warmup_stage_back_completions": 1,
            "warmup_reoffload_attempts": 1,
            "warmup_reoffload_completions": 1,
            "timed_stage_back_attempts": _TIMED_CYCLES,
            "timed_stage_back_completions": _TIMED_CYCLES,
            "timed_reoffload_attempts": _TIMED_CYCLES,
            "timed_reoffload_completions": _TIMED_CYCLES,
            "final_stage_back_attempts": 1,
            "final_stage_back_completions": 1,
            "final_reoffload_attempts": 1,
            "final_reoffload_completions": 1,
            "post_barrier_attempts": 2 * (1 + _TIMED_CYCLES + 1),
            "post_barrier_completions": 2 * (1 + _TIMED_CYCLES + 1),
            "device_get_attempts": 1,
            "device_get_completions": 1,
        }
    )
    return counters


def _exact_contract() -> dict[str, Any]:
    return {
        "case": _CASE,
        "operation": "transactional_optimizer_moment_offload_mid64",
        "selected_paths": [list(path) for path in _SELECTED_PATHS],
        "path_contract": {
            "mu_leaf_count": _LEAVES_PER_SLOT,
            "nu_leaf_count": _LEAVES_PER_SLOT,
            "literal_exact_slot_components": ["mu", "nu"],
            "selector": "exact_explicit_paths_only",
        },
        "unselected_sentinel_path": list(_COUNT_PATH),
        "leaves": {
            "count": _LEAF_COUNT,
            "shape_each": list(_LEAF_SHAPE),
            "dtype_each": "bfloat16",
            "bytes_each": _LEAF_BYTES,
            "selected_bytes_total": _SELECTED_BYTES,
            "host_construction": "distinct deterministic nonzero signed power-of-two grids",
            "all_leaf_sha256_values_required_unique": True,
        },
        "memory_kinds": {"source": "device", "offload": "pinned_host"},
        "transfer_plan": {
            "construction_device_put_batches": 1,
            "construction_tuple_leaves": _LEAF_COUNT + 1,
            "initial_offload_batches": 1,
            "warmup_stage_back_batches": 1,
            "warmup_reoffload_batches": 1,
            "timed_cycles": _TIMED_CYCLES,
            "timed_stage_back_batches": _TIMED_CYCLES,
            "timed_reoffload_batches": _TIMED_CYCLES,
            "final_stage_back_batches": 1,
            "final_reoffload_batches": 1,
            "transactional_manager_batches_total": 1 + 2 * (1 + _TIMED_CYCLES + 1),
            "selected_leaf_directional_copies_total": _LEAF_COUNT
            * (1 + 2 * (1 + _TIMED_CYCLES + 1)),
            "device_get_calls": 1,
            "device_get_tuple_leaves": _LEAF_COUNT + 1,
            "optimizer_updates": 0,
            "model_invocations": 0,
            "compiled_invocations": 0,
            "command_buffer_invocations": 0,
        },
        "per_handle_method_gate": {
            "method_seconds_strictly_below": _MAX_METHOD_SECONDS,
            "post_method_tuple_barrier_seconds_strictly_below": _MAX_POST_BARRIER_SECONDS,
            "warmup_and_final_methods_gated_but_excluded_from_performance_summary": True,
        },
        "performance_report": {
            "included_cycles": _TIMED_CYCLES,
            "per_direction_each_seconds": True,
            "per_direction_median_seconds": True,
            "per_direction_maximum_seconds": True,
            "throughput_unit": "binary_GiB_per_second",
            "bytes_per_direction": _SELECTED_BYTES,
        },
        "allocator_gate": {
            "metric": "bytes_in_use",
            "pool_metric_required": "pool_bytes",
            "minimum_directional_delta_fraction": _ALLOCATOR_DELTA_FRACTION,
            "minimum_directional_delta_bytes": math.ceil(
                _ALLOCATOR_DELTA_FRACTION * _SELECTED_BYTES
            ),
            "every_transition_required": True,
        },
        "physical_accounting": {
            "metrics": ["mem_info_vram_used", "mem_info_gtt_used"],
            "delegated_plateau_sampling": "50 ms cadence over 500 ms",
            "release_target_fraction": _PHYSICAL_RELEASE_FRACTION,
            "release_is_informational": True,
            "reason": "BFC may retain physical pages after logical buffers are released",
        },
        "oracle": {
            "location": "host_after_final_stage_back",
            "single_device_get": True,
            "all_selected_leaves": "shape/dtype and bitwise SHA-256",
            "sentinel": "stable identity/placement and exact scalar value",
        },
        "outer_profiler_required": {
            "max_vram_gib": 2,
            "max_junction_temp_c": 70,
            "max_power_w": 200,
            "min_host_available_gib": 8,
            "timeout_seconds": 90,
            "sensor_grace_seconds": 15,
            "swap_growth_permitted": False,
        },
        "not_implemented": [
            "optimizer_update",
            "model_state",
            "transfer_compute_overlap",
            "larger_rung",
        ],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=("abstract", "rocm"),
        default="abstract",
        help="refuse by default; ROCm requires --allow-gpu and exact --case mid64",
    )
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help="acknowledge the bounded mid64 ROCm transfers",
    )
    parser.add_argument(
        "--case", choices=(_CASE,), help="exact probe case; required for ROCm"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="exclusive mode-0600 JSONL artifact (required for ROCm)",
    )
    args = parser.parse_args(argv)
    if args.platform == "rocm" and not args.allow_gpu:
        parser.error(
            "--platform rocm requires the explicit --allow-gpu acknowledgement"
        )
    if args.platform == "abstract" and args.allow_gpu:
        parser.error("--allow-gpu is only valid with --platform rocm")
    if args.platform == "rocm" and args.case != _CASE:
        parser.error("--platform rocm requires the explicit --case mid64 scope")
    if args.platform == "abstract" and args.case is not None:
        parser.error("--case is only valid with --platform rocm")
    if args.platform == "rocm" and args.output is None:
        parser.error("--platform rocm requires --output for a private JSONL artifact")
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    return args


def _open_exclusive_output(path: Path) -> TextIO:
    return _smoke_probe()._open_exclusive_output(path)


def _configure_rocm_environment() -> dict[str, str | None]:
    return _smoke_probe()._configure_rocm_environment()


def _environment_manifest(environment: dict[str, str | None]) -> dict[str, Any]:
    return _smoke_probe()._environment_manifest(environment)


def _prove_command_buffers_disabled(
    environment: dict[str, str | None],
) -> dict[str, Any]:
    return _smoke_probe()._prove_command_buffers_disabled(environment)


def _assert_fresh_accelerator_process() -> None:
    _smoke_probe()._assert_fresh_accelerator_process()


def _load_safety_helpers() -> tuple[
    Callable[[], ContextManager[dict[str, Any]]],
    Callable[[], dict[str, Any]],
]:
    return _smoke_probe()._load_safety_helpers()


def _public_clean_safety(safety: Any, stage: str) -> dict[str, Any]:
    return _smoke_probe()._public_clean_safety(safety, stage)


def _public_safety_preflight(safety: Any) -> dict[str, Any]:
    return _smoke_probe()._public_safety_preflight(safety)


def _journal_checkpoint(
    require_clean_boot: Callable[[], dict[str, Any]],
    output: TextIO,
    stage: str,
    counters: dict[str, int],
) -> dict[str, Any]:
    safety = _public_clean_safety(require_clean_boot(), stage)
    _emit(
        {
            "record_type": "journal_checkpoint",
            "timestamp": _utc_now(),
            "stage": stage,
            "safety": safety,
            "counters": dict(counters),
        },
        output,
    )
    return safety


def _sample_physical_plateau(card: str) -> dict[str, Any]:
    return _smoke_probe()._sample_physical_plateau(card)


def _variable_at(tree: Any, path: tuple[str | int, ...]) -> Any:
    node = tree
    for component in path:
        node = node[component]
    return node


def _raw_at(tree: Any, path: tuple[str | int, ...]) -> Any:
    return _variable_at(tree, path).get_raw_value()


def _host_sha256(np: Any, value: Any) -> str:
    array = np.asarray(value)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _construct_tree(
    jax: Any,
    np: Any,
    ml_dtypes: Any,
    nnx: Any,
    counters: dict[str, int],
) -> tuple[Any, tuple[dict[str, Any], ...], dict[str, Any]]:
    element_count = math.prod(_LEAF_SHAPE)
    indices = np.arange(element_count, dtype=np.uint32)
    host_leaves = []
    expected = []
    for leaf_index, path in enumerate(_SELECTED_PATHS):
        multiplier = np.uint32(2 * leaf_index + 1)
        offset = np.uint32(7 * leaf_index + 3)
        grid = (
            ((indices * multiplier + offset) % np.uint32(251)) + np.uint32(1)
        ).astype(np.float32)
        sign = np.float32(1.0 if leaf_index < _LEAVES_PER_SLOT else -1.0)
        values = (
            (sign * grid / np.float32(256.0))
            .astype(ml_dtypes.bfloat16)
            .reshape(_LEAF_SHAPE)
        )
        if values.nbytes != _LEAF_BYTES or not bool(np.all(values != 0)):
            raise RuntimeError(
                "mid64 host leaf violated the exact nonzero 1 MiB contract"
            )
        digest = _host_sha256(np, values)
        host_leaves.append(values)
        expected.append({"path": path, "sha256": digest})
    if len({entry["sha256"] for entry in expected}) != _LEAF_COUNT:
        raise RuntimeError("mid64 deterministic leaf hashes are not all distinct")
    count_host = np.asarray(_COUNT_SENTINEL, dtype=np.int32)

    devices = jax.devices()
    if len(devices) != 1:
        raise RuntimeError("mid64 construction requires exactly one ROCm device")
    device_sharding = jax.sharding.SingleDeviceSharding(devices[0]).with_memory_kind(
        "device"
    )
    construction_values = (*host_leaves, count_host)
    construction_targets = (device_sharding,) * (_LEAF_COUNT + 1)
    counters["construction_device_put_attempts"] += 1
    placed = jax.device_put(
        construction_values,
        construction_targets,
        donate=False,
        may_alias=False,
    )
    placed = jax.block_until_ready(placed)
    counters["construction_device_put_completions"] += 1
    if not isinstance(placed, tuple) or len(placed) != _LEAF_COUNT + 1:
        raise RuntimeError(
            "mid64 construction did not preserve the exact 65-leaf tuple"
        )
    selected_device = placed[:-1]
    count_device = placed[-1]
    for value in selected_device:
        if (
            tuple(value.shape) != _LEAF_SHAPE
            or str(value.dtype) != "bfloat16"
            or int(value.nbytes) != _LEAF_BYTES
            or not value.committed
            or not value.is_fully_addressable
            or value.sharding != device_sharding
            or value.sharding.memory_kind != "device"
        ):
            raise RuntimeError(
                "mid64 construction produced an invalid committed device moment"
            )
    if (
        tuple(count_device.shape) != ()
        or str(count_device.dtype) != "int32"
        or not count_device.committed
        or not count_device.is_fully_addressable
        or count_device.sharding != device_sharding
    ):
        raise RuntimeError("mid64 construction produced an invalid count sentinel")

    mu = {
        f"leaf_{index:02d}": nnx.OptState(selected_device[index])
        for index in range(_LEAVES_PER_SLOT)
    }
    nu = {
        f"leaf_{index:02d}": nnx.OptState(selected_device[_LEAVES_PER_SLOT + index])
        for index in range(_LEAVES_PER_SLOT)
    }
    tree = {"opt_state": ({"count": nnx.OptState(count_device), "mu": mu, "nu": nu},)}
    variables = tuple(_variable_at(tree, path) for path in _SELECTED_PATHS)
    count_variable = _variable_at(tree, _COUNT_PATH)
    identity = {
        "variables": variables,
        "count_variable": count_variable,
        "count_raw": count_variable.get_raw_value(),
        "device_sharding": device_sharding,
    }
    del (
        indices,
        host_leaves,
        construction_values,
        placed,
        selected_device,
        count_device,
        count_host,
    )
    gc.collect()
    return tree, tuple(expected), identity


def _manifest_record(handle: Any) -> list[dict[str, Any]]:
    return [
        {
            "path": list(leaf.path),
            "moment_slot": leaf.moment_slot,
            "shape": list(leaf.shape),
            "dtype": leaf.dtype,
            "nbytes": leaf.nbytes,
            "device_memory_kind": leaf.device_memory_kind,
            "offload_memory_kind": leaf.offload_memory_kind,
            "device_and_offload_shardings_distinct": leaf.device_sharding
            != leaf.offload_sharding,
        }
        for leaf in handle.manifest
    ]


def _validate_tree_phase(
    tree: Any,
    handle: Any,
    identity: dict[str, Any],
    phase: str,
    *,
    include_manifest: bool = False,
) -> dict[str, Any]:
    if phase not in {"offloaded", "staged_back"} or handle.phase != phase:
        raise RuntimeError(f"mid64 handle phase does not match required {phase!r}")
    if tuple(leaf.path for leaf in handle.manifest) != _SELECTED_PATHS:
        raise RuntimeError("mid64 manifest paths differ from the 64 exact selectors")
    expected_slots = ("mu",) * _LEAVES_PER_SLOT + ("nu",) * _LEAVES_PER_SLOT
    if tuple(leaf.moment_slot for leaf in handle.manifest) != expected_slots:
        raise RuntimeError(
            "mid64 manifest does not contain exactly 32 mu then 32 nu leaves"
        )
    if handle.leaf_count != _LEAF_COUNT or handle.total_bytes != _SELECTED_BYTES:
        raise RuntimeError("mid64 manifest byte inventory is not exactly 64 MiB")
    expected_kind = "pinned_host" if phase == "offloaded" else "device"
    for index, (path, leaf, expected_variable) in enumerate(
        zip(_SELECTED_PATHS, handle.manifest, identity["variables"], strict=True)
    ):
        variable = _variable_at(tree, path)
        value = variable.get_raw_value()
        expected_sharding = (
            leaf.offload_sharding if phase == "offloaded" else leaf.device_sharding
        )
        if variable is not expected_variable:
            raise RuntimeError(
                f"mid64 selected Variable identity changed at index {index}"
            )
        if (
            tuple(value.shape) != _LEAF_SHAPE
            or str(value.dtype) != "bfloat16"
            or int(value.nbytes) != _LEAF_BYTES
            or not value.committed
            or not value.is_fully_addressable
            or value.sharding != expected_sharding
            or value.sharding.memory_kind != expected_kind
        ):
            raise RuntimeError(
                f"mid64 selected leaf {index} failed exact {phase} placement"
            )
        expected_slot = "mu" if index < _LEAVES_PER_SLOT else "nu"
        if (
            leaf.shape != _LEAF_SHAPE
            or leaf.dtype != "bfloat16"
            or leaf.nbytes != _LEAF_BYTES
            or leaf.moment_slot != expected_slot
            or leaf.device_sharding != identity["device_sharding"]
            or leaf.device_memory_kind != "device"
            or leaf.offload_memory_kind != "pinned_host"
        ):
            raise RuntimeError(
                f"mid64 manifest leaf {index} changed from its exact contract"
            )
    count_variable = _variable_at(tree, _COUNT_PATH)
    count_value = count_variable.get_raw_value()
    if (
        count_variable is not identity["count_variable"]
        or count_value is not identity["count_raw"]
        or tuple(count_value.shape) != ()
        or str(count_value.dtype) != "int32"
        or count_value.sharding != identity["device_sharding"]
        or count_value.sharding.memory_kind != "device"
    ):
        raise RuntimeError(
            "mid64 unselected count sentinel identity or placement changed"
        )
    proof = {
        "phase": phase,
        "selected_leaf_count": _LEAF_COUNT,
        "selected_bytes": _SELECTED_BYTES,
        "selected_memory_kind": expected_kind,
        "exact_selected_shardings": True,
        "variable_identities_stable": True,
        "count_sentinel_identity_and_placement_stable": True,
    }
    if include_manifest:
        proof["manifest"] = _manifest_record(handle)
    return proof


def _allocator_snapshot(device: Any) -> dict[str, Any]:
    raw = device.memory_stats()
    if (
        not isinstance(raw, dict)
        or "bytes_in_use" not in raw
        or "pool_bytes" not in raw
    ):
        raise RuntimeError(
            "mid64 requires exact ROCm bytes_in_use and pool_bytes statistics"
        )
    allowlisted = {}
    for name in ("bytes_in_use", "pool_bytes", "peak_bytes_in_use", "bytes_limit"):
        if name in raw:
            value = int(raw[name])
            if value < 0:
                raise RuntimeError(f"ROCm allocator returned negative {name}")
            allowlisted[name] = value
    return {
        "available": True,
        "bytes_in_use": allowlisted["bytes_in_use"],
        "pool_bytes": allowlisted["pool_bytes"],
        "memory_stats": allowlisted,
    }


def _allocator_transition(
    label: str,
    before: dict[str, Any],
    after: dict[str, Any],
    direction: str,
) -> dict[str, Any]:
    if direction not in {"allocate", "release"}:
        raise ValueError(f"invalid mid64 allocator direction: {direction}")
    before_bytes = int(before["bytes_in_use"])
    after_bytes = int(after["bytes_in_use"])
    delta = (
        after_bytes - before_bytes
        if direction == "allocate"
        else before_bytes - after_bytes
    )
    required = math.ceil(_ALLOCATOR_DELTA_FRACTION * _SELECTED_BYTES)
    passed = delta >= required
    result = {
        "label": label,
        "direction": direction,
        "metric": "bytes_in_use",
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "directional_delta_bytes": delta,
        "required_delta_bytes": required,
        "minimum_fraction": _ALLOCATOR_DELTA_FRACTION,
        "selected_bytes": _SELECTED_BYTES,
        "before_pool_bytes": int(before["pool_bytes"]),
        "after_pool_bytes": int(after["pool_bytes"]),
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(
            f"mid64 allocator {label} bytes_in_use delta failed the 95% gate"
        )
    return result


def _physical_transition(
    label: str,
    before: dict[str, Any],
    after: dict[str, Any],
    direction: str,
) -> dict[str, Any]:
    if direction not in {"allocate", "release"}:
        raise ValueError(f"invalid mid64 physical direction: {direction}")
    vram_observed = before["vram"]["observed"] and after["vram"]["observed"]
    gtt_observed = before["gtt"]["observed"] and after["gtt"]["observed"]
    if vram_observed:
        if direction == "release":
            vram_delta = int(before["vram"]["minimum_bytes"]) - int(
                after["vram"]["maximum_bytes"]
            )
        else:
            vram_delta = int(after["vram"]["minimum_bytes"]) - int(
                before["vram"]["maximum_bytes"]
            )
    else:
        vram_delta = None
    gtt_delta = (
        int(after["gtt"]["minimum_bytes"]) - int(before["gtt"]["maximum_bytes"])
        if gtt_observed
        else None
    )
    target = math.ceil(_PHYSICAL_RELEASE_FRACTION * _SELECTED_BYTES)
    return {
        "label": label,
        "direction": direction,
        "vram_observed": bool(vram_observed),
        "conservative_vram_directional_delta_bytes": vram_delta,
        "physical_release_target_bytes": target if direction == "release" else None,
        "physical_release_target_met": (
            vram_delta >= target
            if direction == "release" and vram_delta is not None
            else None
        ),
        "gtt_observed": bool(gtt_observed),
        "conservative_gtt_after_minus_before_bytes": gtt_delta,
        "gate_effect": "informational_only",
        "failure_effect": "none_bfc_may_retain_physical_pages",
    }


def _capture_state(
    output: TextIO,
    label: str,
    device: Any,
    card: str,
    counters: dict[str, int],
) -> dict[str, Any]:
    allocator = _allocator_snapshot(device)
    physical = _sample_physical_plateau(card)
    _emit(
        {
            "record_type": "allocator_state",
            "timestamp": _utc_now(),
            "label": label,
            "accounting_role": "hard_gate",
            "allocator": allocator,
            "counters": dict(counters),
        },
        output,
    )
    _emit(
        {
            "record_type": "physical_plateau",
            "timestamp": _utc_now(),
            "label": label,
            "accounting_role": "informational_only",
            "physical": physical,
            "counters": dict(counters),
        },
        output,
    )
    return {"allocator": allocator, "physical": physical}


def _emit_transition(
    output: TextIO,
    label: str,
    before: dict[str, Any],
    after: dict[str, Any],
    direction: str,
    counters: dict[str, int],
) -> None:
    allocator = _allocator_transition(
        label, before["allocator"], after["allocator"], direction
    )
    physical = _physical_transition(
        label, before["physical"], after["physical"], direction
    )
    _emit(
        {
            "record_type": "allocator_transition",
            "timestamp": _utc_now(),
            "status": "passed",
            "transition": allocator,
            "counters": dict(counters),
        },
        output,
    )
    _emit(
        {
            "record_type": "physical_transition",
            "timestamp": _utc_now(),
            "status": "informational",
            "transition": physical,
            "counters": dict(counters),
        },
        output,
    )


def _post_method_barrier(jax: Any, tree: Any, counters: dict[str, int]) -> float:
    values = tuple(_raw_at(tree, path) for path in _SELECTED_PATHS)
    counters["post_barrier_attempts"] += 1
    start = time.perf_counter()
    jax.block_until_ready(values)
    seconds = time.perf_counter() - start
    counters["post_barrier_completions"] += 1
    del values
    if (
        not math.isfinite(seconds)
        or seconds < 0
        or seconds >= _MAX_POST_BARRIER_SECONDS
    ):
        raise RuntimeError(
            "mid64 post-method tuple barrier did not complete strictly below 10 ms"
        )
    return seconds


def _checked_stage_back(
    jax: Any,
    tree: Any,
    handle: Any,
    counters: dict[str, int],
    counter_prefix: str,
) -> tuple[Any, dict[str, Any]]:
    if counter_prefix not in _COUNTER_PREFIXES or not counter_prefix.endswith(
        "stage_back"
    ):
        raise ValueError("invalid mid64 stage-back counter prefix")
    counters[f"{counter_prefix}_attempts"] += 1
    start = time.perf_counter()
    returned = handle.stage_back()
    seconds = time.perf_counter() - start
    counters[f"{counter_prefix}_completions"] += 1
    if returned is not handle or handle.phase != "staged_back":
        raise RuntimeError("mid64 stage_back did not return its staged existing handle")
    if not math.isfinite(seconds) or seconds <= 0 or seconds >= _MAX_METHOD_SECONDS:
        raise RuntimeError(
            "mid64 stage_back did not complete strictly between zero and 100 ms"
        )
    barrier_seconds = _post_method_barrier(jax, tree, counters)
    return handle, {
        "method": "stage_back",
        "direction": "pinned_host_to_device",
        "seconds": seconds,
        "post_barrier_seconds": barrier_seconds,
        "binary_gib_per_second": (_SELECTED_BYTES / 1024**3) / seconds,
    }


def _checked_reoffload(
    jax: Any,
    tree: Any,
    handle: Any,
    counters: dict[str, int],
    counter_prefix: str,
) -> tuple[Any, dict[str, Any]]:
    if counter_prefix not in _COUNTER_PREFIXES or not counter_prefix.endswith(
        "reoffload"
    ):
        raise ValueError("invalid mid64 reoffload counter prefix")
    counters[f"{counter_prefix}_attempts"] += 1
    start = time.perf_counter()
    successor = handle.reoffload()
    seconds = time.perf_counter() - start
    counters[f"{counter_prefix}_completions"] += 1
    if (
        successor is handle
        or handle.phase != "complete"
        or successor.phase != "offloaded"
    ):
        raise RuntimeError(
            "mid64 reoffload did not consume the old handle and return an offloaded successor"
        )
    if not math.isfinite(seconds) or seconds <= 0 or seconds >= _MAX_METHOD_SECONDS:
        raise RuntimeError(
            "mid64 reoffload did not complete strictly between zero and 100 ms"
        )
    barrier_seconds = _post_method_barrier(jax, tree, counters)
    return successor, {
        "method": "reoffload",
        "direction": "device_to_pinned_host",
        "seconds": seconds,
        "post_barrier_seconds": barrier_seconds,
        "binary_gib_per_second": (_SELECTED_BYTES / 1024**3) / seconds,
    }


def _timing_summary(timed_records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(timed_records) != 2 * _TIMED_CYCLES:
        raise RuntimeError("mid64 timing summary requires exactly six measured methods")
    result = {}
    for method in ("stage_back", "reoffload"):
        selected = [record for record in timed_records if record["method"] == method]
        if len(selected) != _TIMED_CYCLES:
            raise RuntimeError(
                f"mid64 timing summary requires exactly three {method} records"
            )
        seconds = [float(record["seconds"]) for record in selected]
        barriers = [float(record["post_barrier_seconds"]) for record in selected]
        throughputs = [float(record["binary_gib_per_second"]) for record in selected]
        result[method] = {
            "each": [
                {
                    "cycle": record["cycle"],
                    "seconds": record["seconds"],
                    "post_barrier_seconds": record["post_barrier_seconds"],
                    "binary_gib_per_second": record["binary_gib_per_second"],
                }
                for record in selected
            ],
            "median_seconds": statistics.median(seconds),
            "maximum_seconds": max(seconds),
            "median_post_barrier_seconds": statistics.median(barriers),
            "maximum_post_barrier_seconds": max(barriers),
            "median_binary_gib_per_second": statistics.median(throughputs),
            "minimum_binary_gib_per_second": min(throughputs),
            "maximum_binary_gib_per_second": max(throughputs),
        }
    return result


def _device_get_oracle(
    jax: Any,
    np: Any,
    tree: Any,
    expected: tuple[dict[str, Any], ...],
    counters: dict[str, int],
) -> dict[str, Any]:
    values = tuple(_raw_at(tree, path) for path in (*_SELECTED_PATHS, _COUNT_PATH))
    counters["device_get_attempts"] += 1
    host_values = jax.device_get(values)
    counters["device_get_completions"] += 1
    if not isinstance(host_values, tuple) or len(host_values) != _LEAF_COUNT + 1:
        raise RuntimeError(
            "mid64 device_get did not preserve the exact 65-leaf oracle tuple"
        )
    selected_host = host_values[:-1]
    count_host = host_values[-1]
    actual = []
    for path, value in zip(_SELECTED_PATHS, selected_host, strict=True):
        if tuple(value.shape) != _LEAF_SHAPE or str(value.dtype) != "bfloat16":
            raise RuntimeError(
                "mid64 host oracle received a selected leaf with wrong shape or dtype"
            )
        actual.append({"path": path, "sha256": _host_sha256(np, value)})
    passed = (
        tuple(actual) == expected
        and len({entry["sha256"] for entry in actual}) == _LEAF_COUNT
        and tuple(count_host.shape) == ()
        and str(count_host.dtype) == "int32"
        and int(np.asarray(count_host).item()) == _COUNT_SENTINEL
    )
    del values, host_values, selected_host, count_host
    if not passed:
        raise RuntimeError(
            "mid64 complete host bitwise SHA or count-sentinel oracle failed"
        )
    return {
        "passed": True,
        "selected_leaf_count": _LEAF_COUNT,
        "all_selected_sha256_exact": True,
        "all_selected_sha256_distinct": True,
        "expected": [
            {"path": list(entry["path"]), "sha256": entry["sha256"]}
            for entry in expected
        ],
        "actual": [
            {"path": list(entry["path"]), "sha256": entry["sha256"]} for entry in actual
        ],
        "count_sentinel_exact": True,
        "single_device_get_call": True,
    }


def _emit_method_record(
    output: TextIO,
    label: str,
    timing_class: str,
    timing: dict[str, Any],
    placement: dict[str, Any],
    counters: dict[str, int],
) -> None:
    _emit(
        {
            "record_type": "manager_method",
            "timestamp": _utc_now(),
            "label": label,
            "timing_class": timing_class,
            "status": "passed",
            "method_gate_seconds_strictly_below": _MAX_METHOD_SECONDS,
            "barrier_gate_seconds_strictly_below": _MAX_POST_BARRIER_SECONDS,
            "timing": timing,
            "placement": placement,
            "counters": dict(counters),
        },
        output,
    )


def _run_rocm(
    output: TextIO,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    *,
    environment: dict[str, str | None],
    amd_card: str,
    _dependencies: tuple[Any, Any, Any, Any, Any, Any, Any] | None = None,
) -> int:
    command_buffer_proof = _prove_command_buffers_disabled(environment)
    _emit(
        {
            "record_type": "command_buffer_environment_proof",
            "timestamp": _utc_now(),
            "status": "passed",
            "proof": command_buffer_proof,
            "counters": dict(counters),
        },
        output,
    )
    if _dependencies is None:
        import jax
        import jaxlib
        import ml_dtypes
        import numpy as np
        from flax import nnx
        from jax.extend import backend as jax_backend

        from skyrl.tx.utils.offload import offload_optimizer_moments
    else:
        jax, jaxlib, jax_backend, np, ml_dtypes, nnx, offload_optimizer_moments = (
            _dependencies
        )

    _journal_checkpoint(
        require_clean_boot, output, "before_backend_initialization", counters
    )
    try:
        resolved_backend = jax.default_backend()
        platform_version = str(jax_backend.get_backend().platform_version)
        devices = jax.devices()
        if resolved_backend != "gpu" or "rocm" not in platform_version.lower():
            raise RuntimeError(
                "mid64 did not resolve the explicitly requested ROCm backend"
            )
        if len(devices) != 1 or getattr(devices[0], "platform", None) != "gpu":
            raise RuntimeError(
                "mid64 requires exactly one visible physical ROCm device"
            )
        device = devices[0]
        backend_allocator = _allocator_snapshot(device)
        _emit(
            {
                "record_type": "backend_ready",
                "timestamp": _utc_now(),
                "jax_version_sha256": hashlib.sha256(
                    str(jax.__version__).encode()
                ).hexdigest(),
                "jaxlib_version_sha256": hashlib.sha256(
                    str(jaxlib.__version__).encode()
                ).hexdigest(),
                "platform_resolved": "gpu",
                "platform_family": "rocm",
                "platform_version_sha256": hashlib.sha256(
                    platform_version.encode()
                ).hexdigest(),
                "visible_device_count": 1,
                "raw_backend_strings_emitted": False,
                "allocator": backend_allocator,
                "bfc_growth_allocator": True,
                "unified_memory_disabled": True,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_backend_initialization_attempt", counters
        )

    try:
        tree, expected_hashes, identity = _construct_tree(
            jax, np, ml_dtypes, nnx, counters
        )
        gc.collect()
        baseline_state = _capture_state(
            output, "device_baseline", device, amd_card, counters
        )
        _emit(
            {
                "record_type": "tree_ready",
                "timestamp": _utc_now(),
                "selected_leaf_count": _LEAF_COUNT,
                "selected_bytes": _SELECTED_BYTES,
                "selected_paths": [list(path) for path in _SELECTED_PATHS],
                "expected_hashes": [
                    {"path": list(entry["path"]), "sha256": entry["sha256"]}
                    for entry in expected_hashes
                ],
                "all_hashes_distinct": True,
                "all_values_nonzero": True,
                "direct_selected_host_or_device_array_aliases_retained": False,
                "unselected_count_sentinel": _COUNT_SENTINEL,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_tree_construction_attempt", counters
        )

    try:
        counters["initial_offload_attempts"] += 1
        initial_start = time.perf_counter()
        handle = offload_optimizer_moments(
            tree, paths=_SELECTED_PATHS, memory_kind="pinned_host"
        )
        initial_seconds = time.perf_counter() - initial_start
        counters["initial_offload_completions"] += 1
        if not math.isfinite(initial_seconds) or initial_seconds < 0:
            raise RuntimeError(
                "mid64 initial offload returned an invalid informational duration"
            )
        initial_placement = _validate_tree_phase(
            tree, handle, identity, "offloaded", include_manifest=True
        )
        gc.collect()
        offloaded_state = _capture_state(
            output, "after_initial_offload", device, amd_card, counters
        )
        _emit_transition(
            output,
            "initial_offload",
            baseline_state,
            offloaded_state,
            "release",
            counters,
        )
        _emit(
            {
                "record_type": "initial_offload",
                "timestamp": _utc_now(),
                "status": "passed",
                "seconds_informational": initial_seconds,
                "placement": initial_placement,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_initial_offload_attempt", counters
        )

    try:
        handle, warmup_stage_timing = _checked_stage_back(
            jax, tree, handle, counters, "warmup_stage_back"
        )
        warmup_stage_placement = _validate_tree_phase(
            tree, handle, identity, "staged_back"
        )
        gc.collect()
        warmup_staged_state = _capture_state(
            output, "after_warmup_stage_back", device, amd_card, counters
        )
        _emit_transition(
            output,
            "warmup_stage_back",
            offloaded_state,
            warmup_staged_state,
            "allocate",
            counters,
        )
        _emit_method_record(
            output,
            "warmup_stage_back",
            "warmup_excluded_from_summary",
            warmup_stage_timing,
            warmup_stage_placement,
            counters,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_warmup_stage_back_attempt", counters
        )

    try:
        handle, warmup_reoffload_timing = _checked_reoffload(
            jax, tree, handle, counters, "warmup_reoffload"
        )
        warmup_reoffload_placement = _validate_tree_phase(
            tree, handle, identity, "offloaded"
        )
        gc.collect()
        current_state = _capture_state(
            output, "after_warmup_reoffload", device, amd_card, counters
        )
        _emit_transition(
            output,
            "warmup_reoffload",
            warmup_staged_state,
            current_state,
            "release",
            counters,
        )
        _emit_method_record(
            output,
            "warmup_reoffload",
            "warmup_excluded_from_summary",
            warmup_reoffload_timing,
            warmup_reoffload_placement,
            counters,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_warmup_reoffload_attempt", counters
        )

    timed_records = []
    for cycle in range(1, _TIMED_CYCLES + 1):
        stage_label = f"timed_cycle_{cycle}_stage_back"
        try:
            handle, stage_timing = _checked_stage_back(
                jax, tree, handle, counters, "timed_stage_back"
            )
            stage_timing = {**stage_timing, "cycle": cycle}
            stage_placement = _validate_tree_phase(
                tree, handle, identity, "staged_back"
            )
            gc.collect()
            stage_state = _capture_state(
                output, f"after_{stage_label}", device, amd_card, counters
            )
            _emit_transition(
                output, stage_label, current_state, stage_state, "allocate", counters
            )
            _emit_method_record(
                output,
                stage_label,
                "measured",
                stage_timing,
                stage_placement,
                counters,
            )
            timed_records.append(stage_timing)
        finally:
            _journal_checkpoint(
                require_clean_boot, output, f"after_{stage_label}_attempt", counters
            )

        reoffload_label = f"timed_cycle_{cycle}_reoffload"
        try:
            handle, reoffload_timing = _checked_reoffload(
                jax, tree, handle, counters, "timed_reoffload"
            )
            reoffload_timing = {**reoffload_timing, "cycle": cycle}
            reoffload_placement = _validate_tree_phase(
                tree, handle, identity, "offloaded"
            )
            gc.collect()
            current_state = _capture_state(
                output, f"after_{reoffload_label}", device, amd_card, counters
            )
            _emit_transition(
                output, reoffload_label, stage_state, current_state, "release", counters
            )
            _emit_method_record(
                output,
                reoffload_label,
                "measured",
                reoffload_timing,
                reoffload_placement,
                counters,
            )
            timed_records.append(reoffload_timing)
        finally:
            _journal_checkpoint(
                require_clean_boot, output, f"after_{reoffload_label}_attempt", counters
            )

    performance = _timing_summary(timed_records)
    _emit(
        {
            "record_type": "performance_summary",
            "timestamp": _utc_now(),
            "status": "passed",
            "timed_cycles": _TIMED_CYCLES,
            "selected_bytes_per_direction": _SELECTED_BYTES,
            "throughput_unit": "binary_GiB_per_second",
            "directions": performance,
            "warmup_excluded": True,
            "final_oracle_cycle_excluded": True,
            "counters": dict(counters),
        },
        output,
    )

    try:
        handle, final_stage_timing = _checked_stage_back(
            jax, tree, handle, counters, "final_stage_back"
        )
        final_stage_placement = _validate_tree_phase(
            tree, handle, identity, "staged_back"
        )
        gc.collect()
        final_staged_state = _capture_state(
            output, "after_final_stage_back", device, amd_card, counters
        )
        _emit_transition(
            output,
            "final_stage_back",
            current_state,
            final_staged_state,
            "allocate",
            counters,
        )
        _emit_method_record(
            output,
            "final_stage_back",
            "oracle_cycle_excluded_from_summary",
            final_stage_timing,
            final_stage_placement,
            counters,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_final_stage_back_attempt", counters
        )

    try:
        oracle = _device_get_oracle(jax, np, tree, expected_hashes, counters)
        _emit(
            {
                "record_type": "bitwise_oracle",
                "timestamp": _utc_now(),
                "status": "passed",
                "oracle": oracle,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_device_get_attempt", counters
        )

    try:
        handle, final_reoffload_timing = _checked_reoffload(
            jax, tree, handle, counters, "final_reoffload"
        )
        final_placement = _validate_tree_phase(tree, handle, identity, "offloaded")
        gc.collect()
        final_state = _capture_state(
            output, "after_final_reoffload", device, amd_card, counters
        )
        _emit_transition(
            output,
            "final_reoffload",
            final_staged_state,
            final_state,
            "release",
            counters,
        )
        _emit_method_record(
            output,
            "final_reoffload",
            "oracle_cycle_excluded_from_summary",
            final_reoffload_timing,
            final_placement,
            counters,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_final_reoffload_attempt", counters
        )

    if counters != _completed_counters():
        raise RuntimeError("mid64 transfer counter contract did not complete exactly")
    _emit(
        {
            "record_type": "mid64_passed",
            "timestamp": _utc_now(),
            "status": "passed",
            "scope": "64_MiB_transactional_placement_bandwidth_and_bitwise_roundtrip_only",
            "performance": performance,
            "allocator_gates_passed": True,
            "all_method_and_barrier_gates_passed": True,
            "physical_release_is_informational": True,
            "oracle_passed": True,
            "counters": dict(counters),
            "limitations": [
                "no optimizer update or training-equivalence check",
                "no model or exact optimizer-state integration",
                "no transfer/compute overlap measurement",
                "physical VRAM release may be hidden by BFC retention",
                "64 MiB synthetic moment tree only",
            ],
        },
        output,
    )
    return 0


def _execute(args: argparse.Namespace, output: TextIO) -> int:
    counters = _zero_counters()
    _emit(
        {
            "record_type": "manifest",
            "timestamp": _utc_now(),
            "platform_requested": args.platform,
            "allow_gpu": args.allow_gpu,
            "case": args.case,
            "scope": "abstract_refusal"
            if args.platform == "abstract"
            else "guarded_optimizer_moment_offload_mid64",
            "contract": _exact_contract(),
            "fresh_process_required": True,
            "jax_flax_offload_imported_by_abstract_path": False,
            "counters": dict(counters),
            **_source_hashes(),
        },
        output,
    )
    if args.platform == "abstract":
        _emit(
            {
                "record_type": "refused",
                "timestamp": _utc_now(),
                "status": "no_gpu_abstract_manifest_only",
                "reason": (
                    "pass --platform rocm --allow-gpu --case mid64 --output explicitly under "
                    "profile_rocm.py in a fresh process"
                ),
                "jax_imported": False,
                "flax_imported": False,
                "offload_module_imported": False,
                "counters": dict(counters),
            },
            output,
        )
        return 0

    stage = "fresh_process_preflight"
    try:
        _assert_fresh_accelerator_process()
        stage = "bounded_environment"
        environment = _configure_rocm_environment()
        environment_record = _environment_manifest(environment)
        if (
            not environment_record["bfc_growth_allocator"]
            or not environment_record["unified_memory_disabled"]
        ):
            raise RuntimeError(
                "mid64 environment did not prove BFC growth without unified memory"
            )
        _emit(
            {
                "record_type": "environment",
                "timestamp": _utc_now(),
                "stage": "bounded_rocm_environment_configured",
                "environment": environment_record,
                "counters": dict(counters),
            },
            output,
        )
        guarded_process, require_clean_boot = _load_safety_helpers()
        stage = "safety_preflight"
        with guarded_process() as raw_preflight:
            preflight = _public_safety_preflight(raw_preflight)
            _emit(
                {
                    "record_type": "safety_preflight",
                    "timestamp": _utc_now(),
                    "stage": "guard_acquired",
                    "safety": preflight,
                    "counters": dict(counters),
                },
                output,
            )
            stage = "runtime"
            try:
                result = _run_rocm(
                    output,
                    require_clean_boot,
                    counters,
                    environment=environment,
                    amd_card=preflight["amd_cards"][0],
                )
            finally:
                try:
                    postflight = _public_clean_safety(
                        require_clean_boot(), "safety_postflight"
                    )
                except Exception:
                    stage = "safety_postflight"
                    raise
                _emit(
                    {
                        "record_type": "safety_postflight",
                        "timestamp": _utc_now(),
                        "stage": "current_boot_rechecked",
                        "safety": postflight,
                        "counters": dict(counters),
                    },
                    output,
                )
        _emit(
            {
                "record_type": "completed",
                "timestamp": _utc_now(),
                "status": "passed",
                "counters": dict(counters),
            },
            output,
        )
        return result
    except Exception as error:
        _emit(
            {
                "record_type": "error",
                "timestamp": _utc_now(),
                "stage": stage,
                "status": "failed_closed",
                "error_type": type(error).__name__,
                **_redacted_message_summary(error),
                "counters": dict(counters),
            },
            output,
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.output is None:
        return _execute(args, sys.stdout)
    with _open_exclusive_output(args.output) as output:
        return _execute(args, output)


if __name__ == "__main__":
    sys.exit(main())
