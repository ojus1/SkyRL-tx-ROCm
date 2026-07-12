#!/usr/bin/env python3
"""Guarded 8 MiB transactional optimizer-moment offload smoke probe.

The default ``abstract`` case emits a refusal manifest without importing JAX,
Flax, or SkyRL's offload implementation.  The only accelerator case currently
implemented is the exact ``smoke8`` contract.  It requires an explicit
``--platform rocm --allow-gpu --case smoke8 --output`` invocation in a fresh,
headless process under ``rocm/profile_rocm.py``.

``smoke8`` constructs two distinct deterministic 4 MiB BF16 ``nnx.OptState``
leaves below literal ``mu`` and ``nu`` paths, plus an unselected count sentinel.
It exercises one initial offload, one timed stage-back/re-offload cycle, then an
untimed stage-back for a host bitwise oracle and one final re-offload.  The
transactional manager's synchronous methods are the timed operations; a second
tuple barrier is recorded after each timed method but is deliberately excluded
from the 100 ms transfer gate.

Allocator ``bytes_in_use`` deltas are hard accounting gates.  Independently,
50 ms sysfs samples over 500 ms report physical VRAM and GTT plateaus.  Physical
release is informational because BFC is allowed to retain physical pages.
There is no optimizer update, model, compilation, command buffer, replay,
``mid64`` case, or exact model-state offload in this probe.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import shlex
import stat
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, ContextManager, TextIO

_CASE = "smoke8"
_LEAF_SHAPE = (1024, 2048)
_BF16_ITEMSIZE = 2
_LEAF_BYTES = math.prod(_LEAF_SHAPE) * _BF16_ITEMSIZE
_SELECTED_BYTES = 2 * _LEAF_BYTES
_COUNT_SENTINEL = 173
_MU_PATH = ("opt_state", 0, "mu", "weight")
_NU_PATH = ("opt_state", 0, "nu", "weight")
_COUNT_PATH = ("opt_state", 0, "count")
_SELECTED_PATHS = (_MU_PATH, _NU_PATH)
_MAX_TIMED_TRANSFER_SECONDS = 0.1
_MAX_POST_METHOD_BARRIER_SECONDS = 0.01
_ALLOCATOR_DELTA_FRACTION = 0.95
_PHYSICAL_RELEASE_FRACTION = 0.90
_PLATEAU_INTERVAL_SECONDS = 0.05
_PLATEAU_DURATION_SECONDS = 0.50
_PLATEAU_SAMPLE_COUNT = 11
_DISABLED_COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="
_COMMAND_BUFFER_FLAG_NAME = _DISABLED_COMMAND_BUFFER_FLAG.removesuffix("=")
_NEGATED_COMMAND_BUFFER_FLAG_NAME = "--no" + _COMMAND_BUFFER_FLAG_NAME.removeprefix("--")
_AMD_VENDOR_ID = "0x1002"
_SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _emit(record: dict[str, Any], output: TextIO) -> None:
    output.write(json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n")
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
    return {
        "probe_source_sha256": _sha256_file(Path(__file__)),
        "offload_source_sha256": _sha256_file(_SOURCE_ROOT / "skyrl" / "tx" / "utils" / "offload.py"),
        "safety_source_sha256": _sha256_file(_SOURCE_ROOT / "rocm" / "amdgpu_safety.py"),
    }


def _zero_counters() -> dict[str, int]:
    return {
        "construction_device_put_attempts": 0,
        "construction_device_put_completions": 0,
        "initial_offload_attempts": 0,
        "initial_offload_completions": 0,
        "timed_stage_back_attempts": 0,
        "timed_stage_back_completions": 0,
        "timed_reoffload_attempts": 0,
        "timed_reoffload_completions": 0,
        "timed_tuple_barrier_attempts": 0,
        "timed_tuple_barrier_completions": 0,
        "final_stage_back_attempts": 0,
        "final_stage_back_completions": 0,
        "device_get_attempts": 0,
        "device_get_completions": 0,
        "final_reoffload_attempts": 0,
        "final_reoffload_completions": 0,
    }


def _completed_counters() -> dict[str, int]:
    return {
        "construction_device_put_attempts": 1,
        "construction_device_put_completions": 1,
        "initial_offload_attempts": 1,
        "initial_offload_completions": 1,
        "timed_stage_back_attempts": 1,
        "timed_stage_back_completions": 1,
        "timed_reoffload_attempts": 1,
        "timed_reoffload_completions": 1,
        "timed_tuple_barrier_attempts": 2,
        "timed_tuple_barrier_completions": 2,
        "final_stage_back_attempts": 1,
        "final_stage_back_completions": 1,
        "device_get_attempts": 1,
        "device_get_completions": 1,
        "final_reoffload_attempts": 1,
        "final_reoffload_completions": 1,
    }


def _exact_contract() -> dict[str, Any]:
    return {
        "case": _CASE,
        "operation": "transactional_optimizer_moment_offload_smoke",
        "selected_paths": [list(path) for path in _SELECTED_PATHS],
        "unselected_sentinel_path": list(_COUNT_PATH),
        "leaves": {
            "count": 2,
            "shape_each": list(_LEAF_SHAPE),
            "dtype_each": "bfloat16",
            "bytes_each": _LEAF_BYTES,
            "selected_bytes_total": _SELECTED_BYTES,
            "host_construction": "distinct deterministic nonzero power-of-two grids",
        },
        "memory_kinds": {"source": "device", "offload": "pinned_host"},
        "transfer_plan": {
            "construction_device_put_batches": 1,
            "initial_offload_batches": 1,
            "timed_stage_back_batches": 1,
            "timed_reoffload_batches": 1,
            "final_stage_back_batches": 1,
            "final_reoffload_batches": 1,
            "transactional_manager_batches_total": 5,
            "selected_leaf_directional_copies_total": 10,
            "device_get_calls": 1,
            "optimizer_updates": 0,
            "model_invocations": 0,
            "compiled_invocations": 0,
            "command_buffer_invocations": 0,
        },
        "timing_gate": {
            "stage_back_seconds_strictly_below": _MAX_TIMED_TRANSFER_SECONDS,
            "reoffload_seconds_strictly_below": _MAX_TIMED_TRANSFER_SECONDS,
            "post_method_tuple_barriers": 2,
            "barrier_seconds_strictly_below": _MAX_POST_METHOD_BARRIER_SECONDS,
            "barrier_latency_recorded_separately_from_transfer_time": True,
        },
        "allocator_gate": {
            "metric": "bytes_in_use",
            "minimum_directional_delta_fraction_of_selected_bytes": _ALLOCATOR_DELTA_FRACTION,
            "minimum_directional_delta_bytes": math.ceil(_ALLOCATOR_DELTA_FRACTION * _SELECTED_BYTES),
            "initial_offload_release_required": True,
            "stage_back_reversal_required": True,
            "reoffload_release_required": True,
        },
        "physical_accounting": {
            "metrics": ["mem_info_vram_used", "mem_info_gtt_used"],
            "sample_interval_seconds": _PLATEAU_INTERVAL_SECONDS,
            "sample_window_seconds": _PLATEAU_DURATION_SECONDS,
            "samples_per_plateau": _PLATEAU_SAMPLE_COUNT,
            "release_target_fraction": _PHYSICAL_RELEASE_FRACTION,
            "release_is_informational": True,
            "reason": "BFC may retain physical pages after logical buffers are released",
        },
        "oracle": {
            "location": "host_after_untimed_final_stage_back",
            "selected_leaf_check": "shape/dtype and bitwise SHA-256",
            "sentinel_check": "identity, placement, dtype, shape, and exact scalar value",
        },
        "outer_profiler_required": {
            "max_vram_gib": 2,
            "max_junction_temp_c": 70,
            "max_power_w": 200,
            "min_host_available_gib": 8,
            "swap_growth_permitted": False,
        },
        "not_implemented": ["mid64", "exact_model_state", "optimizer_update", "overlap", "throughput_sweep"],
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=("abstract", "rocm"),
        default="abstract",
        help="refuse by default; the guarded accelerator path requires rocm and --allow-gpu",
    )
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help="acknowledge the bounded smoke8 ROCm transfers",
    )
    parser.add_argument(
        "--case",
        choices=(_CASE,),
        help="exact probe case; required for ROCm",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="exclusive mode-0600 JSONL artifact (required for ROCm)",
    )
    args = parser.parse_args(argv)
    if args.platform == "rocm" and not args.allow_gpu:
        parser.error("--platform rocm requires the explicit --allow-gpu acknowledgement")
    if args.platform == "abstract" and args.allow_gpu:
        parser.error("--allow-gpu is only valid with --platform rocm")
    if args.platform == "rocm" and args.case != _CASE:
        parser.error("--platform rocm requires the explicit --case smoke8 scope")
    if args.platform == "abstract" and args.case is not None:
        parser.error("--case is only valid with --platform rocm")
    if args.platform == "rocm" and args.output is None:
        parser.error("--platform rocm requires --output for a private JSONL artifact")
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    return args


def _open_exclusive_output(path: Path) -> TextIO:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise RuntimeError("private output is not a regular user-owned file")
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise RuntimeError("private output mode is not exactly 0600")
        return os.fdopen(descriptor, "w", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise


def _command_buffer_assignments(tokens: list[str]) -> list[str]:
    assignments = []
    for token in tokens:
        if token == _COMMAND_BUFFER_FLAG_NAME or token.startswith(f"{_COMMAND_BUFFER_FLAG_NAME}="):
            assignments.append(token)
        elif token == _NEGATED_COMMAND_BUFFER_FLAG_NAME or token.startswith(f"{_NEGATED_COMMAND_BUFFER_FLAG_NAME}="):
            assignments.append(token)
    return assignments


def _validate_exact_or_unset(name: str, expected: str) -> None:
    inherited = os.environ.get(name)
    if inherited is not None and inherited != expected:
        raise RuntimeError(f"{name} conflicts with the exact smoke8 environment contract")


def _configure_rocm_environment() -> dict[str, str | None]:
    fixed = {
        "JAX_PLATFORMS": "rocm",
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
        "JAX_ROCM_VISIBLE_DEVICES": "0",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_CLIENT_MEM_FRACTION": "0.75",
    }
    for name, expected in fixed.items():
        if name == "XLA_PYTHON_CLIENT_PREALLOCATE":
            inherited = os.environ.get(name)
            if inherited is not None and inherited.strip().lower() not in {"0", "false", "no", "off"}:
                raise RuntimeError("preallocation conflicts with the required BFC growth allocator")
        else:
            _validate_exact_or_unset(name, expected)
    for forbidden in (
        "HSA_OVERRIDE_GFX_VERSION",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB",
        "JAX_PJRT_CLIENT_CREATE_OPTIONS",
        "JAX_MOCK_GPU_TOPOLOGY",
    ):
        if os.environ.get(forbidden, "").strip():
            raise RuntimeError(f"{forbidden} must be unset for the physical bounded smoke8 probe")
        os.environ.pop(forbidden, None)
    if os.environ.get("MOCK_NUM_GPU_PROCESSES", "").strip() not in {"", "0"}:
        raise RuntimeError("MOCK_NUM_GPU_PROCESSES must be unset or zero")
    os.environ.pop("MOCK_NUM_GPU_PROCESSES", None)
    unified = os.environ.get("TF_FORCE_UNIFIED_MEMORY")
    if unified is not None and unified.strip().lower() not in {"", "0", "false", "no", "off"}:
        raise RuntimeError("TF_FORCE_UNIFIED_MEMORY must be unset or false")
    os.environ.pop("TF_FORCE_UNIFIED_MEMORY", None)

    original_flags = os.environ.get("XLA_FLAGS", "")
    try:
        tokens = shlex.split(original_flags, posix=True)
    except ValueError as error:
        raise RuntimeError("invalid inherited XLA_FLAGS quoting") from error
    assignments = _command_buffer_assignments(tokens)
    if assignments not in ([], [_DISABLED_COMMAND_BUFFER_FLAG]):
        raise RuntimeError("inherited command-buffer flags conflict with the sole exact empty assignment")
    if not assignments:
        tokens.append(_DISABLED_COMMAND_BUFFER_FLAG)
    effective_flags = shlex.join(tokens)
    os.environ.update(fixed)
    os.environ["XLA_FLAGS"] = effective_flags
    return {
        **fixed,
        "TF_FORCE_UNIFIED_MEMORY": None,
        "XLA_FLAGS_original": original_flags,
        "XLA_FLAGS_effective": effective_flags,
    }


def _text_summary(value: str | None) -> dict[str, Any]:
    raw = "" if value is None else value
    try:
        token_count: int | None = len(shlex.split(raw, posix=True))
        quoting_valid = True
    except ValueError:
        token_count = None
        quoting_valid = False
    return {
        "present": value is not None,
        "utf8_bytes": len(raw.encode("utf-8")),
        "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "shlex_quoting_valid": quoting_valid,
        "token_count": token_count,
    }


def _environment_manifest(environment: dict[str, str | None]) -> dict[str, Any]:
    fixed_names = (
        "JAX_PLATFORMS",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "JAX_ROCM_VISIBLE_DEVICES",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_CLIENT_MEM_FRACTION",
    )
    return {
        "fixed_values": {name: environment.get(name) for name in fixed_names},
        "bfc_growth_allocator": (
            environment.get("XLA_PYTHON_CLIENT_ALLOCATOR") == "bfc"
            and environment.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false"
        ),
        "unified_memory_disabled": environment.get("TF_FORCE_UNIFIED_MEMORY") is None,
        "xla_flags_original": _text_summary(environment.get("XLA_FLAGS_original")),
        "xla_flags_effective": _text_summary(environment.get("XLA_FLAGS_effective")),
        "raw_xla_flags_emitted": False,
    }


def _prove_command_buffers_disabled(environment: dict[str, str | None]) -> dict[str, Any]:
    returned = environment.get("XLA_FLAGS_effective")
    process = os.environ.get("XLA_FLAGS")
    if not isinstance(returned, str) or process is None or returned != process:
        raise RuntimeError("effective XLA_FLAGS do not exactly match the process environment")
    try:
        tokens = shlex.split(process, posix=True)
    except ValueError as error:
        raise RuntimeError("invalid effective XLA_FLAGS quoting") from error
    assignments = _command_buffer_assignments(tokens)
    if assignments != [_DISABLED_COMMAND_BUFFER_FLAG]:
        raise RuntimeError("command buffers are not disabled by one sole exact empty assignment")
    return {
        "process_matches_returned": True,
        "command_buffer_assignment_count": 1,
        "sole_assignment_is_exact_empty": True,
        "token_count": len(tokens),
        "xla_flags_sha256": hashlib.sha256(process.encode("utf-8")).hexdigest(),
        "raw_xla_flags_emitted": False,
    }


def _assert_fresh_accelerator_process() -> None:
    imported = sorted(
        name
        for name in sys.modules
        if name in {"jax", "jaxlib", "flax"}
        or name.startswith(("jax.", "jaxlib.", "flax."))
        or name == "skyrl.tx.utils.offload"
    )
    if imported:
        raise RuntimeError(
            "ROCm smoke8 requires a fresh process before JAX/Flax/offload import; "
            f"already imported module count: {len(imported)}"
        )


def _load_safety_helpers() -> tuple[
    Callable[[], ContextManager[dict[str, Any]]],
    Callable[[], dict[str, Any]],
]:
    try:
        from rocm.amdgpu_safety import (
            guarded_qwen35_rocm_process,
            require_clean_amdgpu_boot,
        )
    except ModuleNotFoundError:
        from amdgpu_safety import (  # type: ignore[no-redef]
            guarded_qwen35_rocm_process,
            require_clean_amdgpu_boot,
        )

    return guarded_qwen35_rocm_process, require_clean_amdgpu_boot


def _public_clean_safety(safety: Any, stage: str) -> dict[str, Any]:
    if not isinstance(safety, dict) or safety.get("amdgpu_boot_clean") is not True:
        raise RuntimeError(f"{stage} did not prove a clean AMDGPU boot")
    if safety.get("fatal_amdgpu_events") != []:
        raise RuntimeError(f"{stage} returned invalid fatal-event evidence")
    return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}


def _public_safety_preflight(safety: Any) -> dict[str, Any]:
    public = _public_clean_safety(safety, "safety_preflight")
    amd_cards = safety.get("amd_cards")
    if (
        not isinstance(amd_cards, list)
        or len(amd_cards) != 1
        or not isinstance(amd_cards[0], str)
        or re.fullmatch(r"card[0-9]+", amd_cards[0]) is None
    ):
        raise RuntimeError("safety_preflight did not identify exactly one AMD DRM card")
    if safety.get("connected_amd_connectors") != []:
        raise RuntimeError("safety_preflight did not prove the AMD card headless")
    if safety.get("kfd_path") != "/dev/kfd":
        raise RuntimeError("safety_preflight did not prove the exact KFD path")
    if safety.get("kfd_accessible") is not True or safety.get("kfd_unowned") is not True:
        raise RuntimeError("safety_preflight did not prove accessible unowned KFD")
    return {
        **public,
        "amd_cards": list(amd_cards),
        "connected_amd_connectors": [],
        "kfd_path": "/dev/kfd",
        "kfd_accessible": True,
        "kfd_unowned": True,
    }


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


def _hash_host_array(value: Any) -> str:
    import numpy as np

    array = np.asarray(value)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _variable_at(tree: Any, path: tuple[str | int, ...]) -> Any:
    node = tree
    for component in path:
        node = node[component]
    return node


def _raw_at(tree: Any, path: tuple[str | int, ...]) -> Any:
    return _variable_at(tree, path).get_raw_value()


def _construct_tree(
    jax: Any, np: Any, ml_dtypes: Any, nnx: Any, counters: dict[str, int]
) -> tuple[Any, dict[str, str], dict[str, Any]]:
    element_count = math.prod(_LEAF_SHAPE)
    indices = np.arange(element_count, dtype=np.uint32)
    mu_host = (((indices % 251) + 1).astype(np.float32) / np.float32(256.0)).astype(ml_dtypes.bfloat16)
    nu_host = (-(((indices * 17 + 29) % 251) + 1).astype(np.float32) / np.float32(128.0)).astype(ml_dtypes.bfloat16)
    mu_host = mu_host.reshape(_LEAF_SHAPE)
    nu_host = nu_host.reshape(_LEAF_SHAPE)
    if (
        mu_host.nbytes != _LEAF_BYTES
        or nu_host.nbytes != _LEAF_BYTES
        or not bool(np.all(mu_host != 0))
        or not bool(np.all(nu_host != 0))
    ):
        raise RuntimeError("host moment construction violated the exact nonzero 4 MiB contract")
    expected_hashes = {"mu": _hash_host_array(mu_host), "nu": _hash_host_array(nu_host)}
    if expected_hashes["mu"] == expected_hashes["nu"]:
        raise RuntimeError("mu and nu host constructions are not distinct")
    count_host = np.asarray(_COUNT_SENTINEL, dtype=np.int32)

    devices = jax.devices()
    if len(devices) != 1:
        raise RuntimeError("tree construction requires exactly one ROCm device")
    device_sharding = jax.sharding.SingleDeviceSharding(devices[0]).with_memory_kind("device")
    counters["construction_device_put_attempts"] += 1
    placed = jax.device_put(
        (mu_host, nu_host, count_host),
        (device_sharding, device_sharding, device_sharding),
        donate=False,
        may_alias=False,
    )
    placed = jax.block_until_ready(placed)
    counters["construction_device_put_completions"] += 1
    if not isinstance(placed, tuple) or len(placed) != 3:
        raise RuntimeError("construction device_put did not preserve the exact tuple")
    mu_device, nu_device, count_device = placed
    for value, shape, dtype in (
        (mu_device, _LEAF_SHAPE, "bfloat16"),
        (nu_device, _LEAF_SHAPE, "bfloat16"),
        (count_device, (), "int32"),
    ):
        if (
            tuple(value.shape) != shape
            or str(value.dtype) != dtype
            or not value.committed
            or not value.is_fully_addressable
            or value.sharding != device_sharding
            or value.sharding.memory_kind != "device"
        ):
            raise RuntimeError("construction produced an invalid committed device leaf")
    tree = {
        "opt_state": (
            {
                "count": nnx.OptState(count_device),
                "mu": {"weight": nnx.OptState(mu_device)},
                "nu": {"weight": nnx.OptState(nu_device)},
            },
        )
    }
    variables = {
        "mu": _variable_at(tree, _MU_PATH),
        "nu": _variable_at(tree, _NU_PATH),
        "count": _variable_at(tree, _COUNT_PATH),
    }
    identity = {
        "variables": variables,
        "count_raw_id": id(_raw_at(tree, _COUNT_PATH)),
        "device_sharding": device_sharding,
    }
    del indices, mu_host, nu_host, count_host, placed, mu_device, nu_device, count_device
    gc.collect()
    return tree, expected_hashes, identity


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
            "device_and_offload_shardings_distinct": leaf.device_sharding != leaf.offload_sharding,
        }
        for leaf in handle.manifest
    ]


def _validate_tree_phase(tree: Any, handle: Any, identity: dict[str, Any], phase: str) -> dict[str, Any]:
    if phase not in {"offloaded", "staged_back"} or handle.phase != phase:
        raise RuntimeError(f"handle phase does not match required {phase!r} state")
    if tuple(leaf.path for leaf in handle.manifest) != _SELECTED_PATHS:
        raise RuntimeError("manager manifest paths differ from the two exact selectors")
    if tuple(leaf.moment_slot for leaf in handle.manifest) != ("mu", "nu"):
        raise RuntimeError("manager manifest moment slots are not exact mu and nu")
    if handle.leaf_count != 2 or handle.total_bytes != _SELECTED_BYTES:
        raise RuntimeError("manager manifest byte inventory is not exactly 8 MiB")
    expected_kind = "pinned_host" if phase == "offloaded" else "device"
    for key, path, leaf in zip(("mu", "nu"), _SELECTED_PATHS, handle.manifest, strict=True):
        variable = _variable_at(tree, path)
        value = variable.get_raw_value()
        expected_sharding = leaf.offload_sharding if phase == "offloaded" else leaf.device_sharding
        if variable is not identity["variables"][key]:
            raise RuntimeError("selected NNX Variable identity changed")
        if (
            tuple(value.shape) != _LEAF_SHAPE
            or str(value.dtype) != "bfloat16"
            or int(value.nbytes) != _LEAF_BYTES
            or not value.committed
            or not value.is_fully_addressable
            or value.sharding != expected_sharding
            or value.sharding.memory_kind != expected_kind
        ):
            raise RuntimeError(f"selected {key} leaf failed exact {phase} placement validation")
        if (
            leaf.shape != _LEAF_SHAPE
            or leaf.dtype != "bfloat16"
            or leaf.nbytes != _LEAF_BYTES
            or leaf.device_sharding != identity["device_sharding"]
            or leaf.device_memory_kind != "device"
            or leaf.offload_memory_kind != "pinned_host"
        ):
            raise RuntimeError("manager leaf manifest changed from the construction contract")
    count_variable = _variable_at(tree, _COUNT_PATH)
    count_value = count_variable.get_raw_value()
    if (
        count_variable is not identity["variables"]["count"]
        or id(count_value) != identity["count_raw_id"]
        or tuple(count_value.shape) != ()
        or str(count_value.dtype) != "int32"
        or count_value.sharding != identity["device_sharding"]
        or count_value.sharding.memory_kind != "device"
    ):
        raise RuntimeError("unselected count sentinel identity or placement changed")
    return {
        "phase": phase,
        "selected_memory_kind": expected_kind,
        "exact_selected_shardings": True,
        "variable_identities_stable": True,
        "unselected_count_identity_and_placement_stable": True,
        "manifest": _manifest_record(handle),
    }


def _allocator_snapshot(device: Any) -> dict[str, Any]:
    raw = device.memory_stats()
    if not isinstance(raw, dict) or "bytes_in_use" not in raw:
        raise RuntimeError("ROCm allocator did not expose bytes_in_use")
    bytes_in_use = int(raw["bytes_in_use"])
    if bytes_in_use < 0:
        raise RuntimeError("ROCm allocator returned negative bytes_in_use")
    allowlisted = {}
    for name in ("bytes_in_use", "peak_bytes_in_use", "pool_bytes", "bytes_limit"):
        if name in raw:
            value = int(raw[name])
            if value < 0:
                raise RuntimeError(f"ROCm allocator returned negative {name}")
            allowlisted[name] = value
    return {"available": True, "memory_stats": allowlisted, "bytes_in_use": bytes_in_use}


def _allocator_transition(label: str, before: dict[str, Any], after: dict[str, Any], direction: str) -> dict[str, Any]:
    if direction not in {"allocate", "release"}:
        raise ValueError(f"invalid allocator transition direction: {direction}")
    before_bytes = int(before["bytes_in_use"])
    after_bytes = int(after["bytes_in_use"])
    directional_delta = after_bytes - before_bytes if direction == "allocate" else before_bytes - after_bytes
    required = math.ceil(_ALLOCATOR_DELTA_FRACTION * _SELECTED_BYTES)
    passed = directional_delta >= required
    result = {
        "label": label,
        "metric": "bytes_in_use",
        "direction": direction,
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "directional_delta_bytes": directional_delta,
        "required_delta_bytes": required,
        "selected_bytes": _SELECTED_BYTES,
        "minimum_fraction": _ALLOCATOR_DELTA_FRACTION,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"allocator bytes_in_use {label} delta failed the 95% gate")
    return result


def _read_sysfs_counter(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if re.fullmatch(r"[0-9]+", raw) is None:
        return None
    return int(raw)


def _sample_physical_plateau(
    card: str,
    *,
    drm_root: Path = Path("/sys/class/drm"),
    read_counter: Callable[[Path], int | None] = _read_sysfs_counter,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if re.fullmatch(r"card[0-9]+", card) is None:
        raise RuntimeError("physical plateau received an invalid DRM card name")
    device_root = drm_root / card / "device"
    vendor_path = device_root / "vendor"
    try:
        vendor = vendor_path.read_text(encoding="utf-8").strip()
    except OSError:
        vendor = None
    if vendor is not None and vendor != _AMD_VENDOR_ID:
        raise RuntimeError("physical plateau DRM card is not AMD")
    start = clock()
    samples = []
    for index in range(_PLATEAU_SAMPLE_COUNT):
        target = start + index * _PLATEAU_INTERVAL_SECONDS
        remaining = target - clock()
        if remaining > 0:
            sleep(remaining)
        sample_time = clock()
        samples.append(
            {
                "index": index,
                "relative_seconds": max(0.0, sample_time - start),
                "vram_used_bytes": read_counter(device_root / "mem_info_vram_used"),
                "gtt_used_bytes": read_counter(device_root / "mem_info_gtt_used"),
            }
        )
    window = max(0.0, samples[-1]["relative_seconds"] - samples[0]["relative_seconds"])

    def summary(name: str) -> dict[str, Any]:
        values = [sample[name] for sample in samples if sample[name] is not None]
        if not values:
            return {"observed": False, "readable_samples": 0}
        return {
            "observed": True,
            "readable_samples": len(values),
            "minimum_bytes": min(values),
            "maximum_bytes": max(values),
            "last_bytes": values[-1],
        }

    return {
        "card": card,
        "requested_interval_seconds": _PLATEAU_INTERVAL_SECONDS,
        "requested_window_seconds": _PLATEAU_DURATION_SECONDS,
        "sample_count": len(samples),
        "observed_window_seconds": window,
        "samples": samples,
        "vram": summary("vram_used_bytes"),
        "gtt": summary("gtt_used_bytes"),
    }


def _physical_transition(label: str, before: dict[str, Any], after: dict[str, Any], direction: str) -> dict[str, Any]:
    if direction not in {"allocate", "release"}:
        raise ValueError(f"invalid physical transition direction: {direction}")
    vram_observed = before["vram"]["observed"] and after["vram"]["observed"]
    gtt_observed = before["gtt"]["observed"] and after["gtt"]["observed"]
    if vram_observed:
        if direction == "release":
            vram_delta = int(before["vram"]["minimum_bytes"]) - int(after["vram"]["maximum_bytes"])
        else:
            vram_delta = int(after["vram"]["minimum_bytes"]) - int(before["vram"]["maximum_bytes"])
    else:
        vram_delta = None
    if gtt_observed:
        gtt_delta = int(after["gtt"]["minimum_bytes"]) - int(before["gtt"]["maximum_bytes"])
    else:
        gtt_delta = None
    target = math.ceil(_PHYSICAL_RELEASE_FRACTION * _SELECTED_BYTES)
    return {
        "label": label,
        "direction": direction,
        "vram_observed": bool(vram_observed),
        "conservative_vram_directional_delta_bytes": vram_delta,
        "physical_release_target_bytes": target if direction == "release" else None,
        "physical_release_target_met": (
            (vram_delta >= target) if direction == "release" and vram_delta is not None else None
        ),
        "gtt_observed": bool(gtt_observed),
        "conservative_gtt_after_minus_before_bytes": gtt_delta,
        "gate_effect": "informational_only",
        "failure_effect": "none_bfc_may_retain_physical_pages",
    }


def _emit_memory_state(
    output: TextIO,
    label: str,
    allocator: dict[str, Any],
    physical: dict[str, Any],
    counters: dict[str, int],
) -> None:
    _emit(
        {
            "record_type": "memory_state",
            "timestamp": _utc_now(),
            "label": label,
            "allocator": allocator,
            "physical_plateau": physical,
            "accounting_distinction": "allocator hard gate; physical VRAM/GTT informational",
            "counters": dict(counters),
        },
        output,
    )


def _timed_tuple_barrier(jax: Any, tree: Any, counters: dict[str, int]) -> float:
    values = tuple(_raw_at(tree, path) for path in _SELECTED_PATHS)
    counters["timed_tuple_barrier_attempts"] += 1
    start = time.perf_counter()
    jax.block_until_ready(values)
    seconds = time.perf_counter() - start
    counters["timed_tuple_barrier_completions"] += 1
    del values
    if not math.isfinite(seconds) or seconds < 0 or seconds >= _MAX_POST_METHOD_BARRIER_SECONDS:
        raise RuntimeError("post-method tuple barrier did not prove synchronous completion")
    return seconds


def _effective_gib_per_second(byte_count: int, seconds: float) -> float:
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
        raise ValueError("throughput byte count must be a positive integer")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("throughput duration must be finite and positive")
    return byte_count / seconds / 1024**3


def _timed_stage_back(jax: Any, tree: Any, handle: Any, counters: dict[str, int]) -> tuple[Any, float, float]:
    counters["timed_stage_back_attempts"] += 1
    start = time.perf_counter()
    returned = handle.stage_back()
    seconds = time.perf_counter() - start
    counters["timed_stage_back_completions"] += 1
    if returned is not handle:
        raise RuntimeError("stage_back did not return the same consumable handle")
    if not math.isfinite(seconds) or seconds < 0 or seconds >= _MAX_TIMED_TRANSFER_SECONDS:
        raise RuntimeError("timed stage_back did not complete strictly below 100 ms")
    barrier_seconds = _timed_tuple_barrier(jax, tree, counters)
    return handle, seconds, barrier_seconds


def _timed_reoffload(jax: Any, tree: Any, handle: Any, counters: dict[str, int]) -> tuple[Any, float, float]:
    counters["timed_reoffload_attempts"] += 1
    start = time.perf_counter()
    successor = handle.reoffload()
    seconds = time.perf_counter() - start
    counters["timed_reoffload_completions"] += 1
    if successor is handle or handle.phase != "complete" or successor.phase != "offloaded":
        raise RuntimeError("reoffload did not consume the old handle and return an offloaded successor")
    if not math.isfinite(seconds) or seconds < 0 or seconds >= _MAX_TIMED_TRANSFER_SECONDS:
        raise RuntimeError("timed reoffload did not complete strictly below 100 ms")
    barrier_seconds = _timed_tuple_barrier(jax, tree, counters)
    return successor, seconds, barrier_seconds


def _device_get_oracle(
    jax: Any, np: Any, tree: Any, expected_hashes: dict[str, str], counters: dict[str, int]
) -> dict[str, Any]:
    values = tuple(_raw_at(tree, path) for path in (*_SELECTED_PATHS, _COUNT_PATH))
    counters["device_get_attempts"] += 1
    host_values = jax.device_get(values)
    counters["device_get_completions"] += 1
    if not isinstance(host_values, tuple) or len(host_values) != 3:
        raise RuntimeError("device_get did not preserve the exact oracle tuple")
    mu_host, nu_host, count_host = host_values
    actual_hashes = {"mu": _hash_host_array(mu_host), "nu": _hash_host_array(nu_host)}
    passed = (
        tuple(mu_host.shape) == _LEAF_SHAPE
        and tuple(nu_host.shape) == _LEAF_SHAPE
        and str(mu_host.dtype) == "bfloat16"
        and str(nu_host.dtype) == "bfloat16"
        and actual_hashes == expected_hashes
        and tuple(count_host.shape) == ()
        and str(count_host.dtype) == "int32"
        and int(np.asarray(count_host).item()) == _COUNT_SENTINEL
    )
    del values, host_values, mu_host, nu_host, count_host
    if not passed:
        raise RuntimeError("final host bitwise SHA or count-sentinel oracle failed")
    return {
        "passed": True,
        "expected_sha256": dict(expected_hashes),
        "actual_sha256": actual_hashes,
        "shape_dtype_exact": True,
        "count_sentinel_exact": True,
        "single_device_get_call": True,
    }


def _run_rocm(
    output: TextIO,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    *,
    environment: dict[str, str | None],
    amd_card: str,
    _dependencies: tuple[Any, Any, Any, Any, Any, Any, Any] | None = None,
) -> int:
    proof = _prove_command_buffers_disabled(environment)
    _emit(
        {
            "record_type": "command_buffer_environment_proof",
            "timestamp": _utc_now(),
            "status": "passed",
            "proof": proof,
            "counters": dict(counters),
        },
        output,
    )
    try:
        if _dependencies is None:
            import jax
            import jaxlib
            import ml_dtypes
            import numpy as np
            from flax import nnx
            from jax.extend import backend as jax_backend

            from skyrl.tx.utils.offload import offload_optimizer_moments
        else:
            jax, jaxlib, jax_backend, np, ml_dtypes, nnx, offload_optimizer_moments = _dependencies

        resolved_backend = jax.default_backend()
        backend = jax_backend.get_backend()
        platform_version = str(backend.platform_version)
        devices = jax.devices()
        if resolved_backend != "gpu" or "rocm" not in platform_version.lower():
            raise RuntimeError("JAX did not resolve the explicitly requested ROCm backend")
        if len(devices) != 1 or getattr(devices[0], "platform", None) != "gpu":
            raise RuntimeError("smoke8 requires exactly one visible physical ROCm device")
        device = devices[0]
        backend_allocator = _allocator_snapshot(device)
        _emit(
            {
                "record_type": "backend_ready",
                "timestamp": _utc_now(),
                "jax_version": _text_summary(str(jax.__version__)),
                "jaxlib_version": _text_summary(str(jaxlib.__version__)),
                "platform_resolved": resolved_backend,
                "platform_version": _text_summary(platform_version),
                "visible_device_count": 1,
                "raw_device_descriptions_emitted": False,
                "allocator": backend_allocator,
                "bfc_growth_allocator": True,
                "unified_memory_disabled": True,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(require_clean_boot, output, "after_backend_initialization_attempt", counters)

    try:
        tree, expected_hashes, identity = _construct_tree(jax, np, ml_dtypes, nnx, counters)
        gc.collect()
        baseline_allocator = _allocator_snapshot(device)
        baseline_physical = _sample_physical_plateau(amd_card)
        _emit(
            {
                "record_type": "tree_ready",
                "timestamp": _utc_now(),
                "selected_paths": [list(path) for path in _SELECTED_PATHS],
                "selected_bytes": _SELECTED_BYTES,
                "selected_leaf_host_hashes": expected_hashes,
                "distinct_deterministic_nonzero_host_data": True,
                "selected_host_and_device_array_aliases_retained": False,
                "unselected_count_sentinel": _COUNT_SENTINEL,
                "counters": dict(counters),
            },
            output,
        )
        _emit_memory_state(output, "device_baseline", baseline_allocator, baseline_physical, counters)
    finally:
        _journal_checkpoint(require_clean_boot, output, "after_tree_construction_attempt", counters)

    try:
        counters["initial_offload_attempts"] += 1
        handle = offload_optimizer_moments(tree, paths=_SELECTED_PATHS, memory_kind="pinned_host")
        counters["initial_offload_completions"] += 1
        initial_proof = _validate_tree_phase(tree, handle, identity, "offloaded")
        gc.collect()
        initial_allocator = _allocator_snapshot(device)
        initial_physical = _sample_physical_plateau(amd_card)
        initial_allocator_transition = _allocator_transition(
            "initial_offload", baseline_allocator, initial_allocator, "release"
        )
        initial_physical_transition = _physical_transition(
            "initial_offload", baseline_physical, initial_physical, "release"
        )
        _emit(
            {
                "record_type": "placement_validation",
                "timestamp": _utc_now(),
                "label": "initial_offload",
                "status": "passed",
                "proof": initial_proof,
                "counters": dict(counters),
            },
            output,
        )
        _emit_memory_state(output, "after_initial_offload", initial_allocator, initial_physical, counters)
        _emit(
            {
                "record_type": "memory_transition",
                "timestamp": _utc_now(),
                "allocator": initial_allocator_transition,
                "physical": initial_physical_transition,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(require_clean_boot, output, "after_initial_offload_attempt", counters)

    try:
        handle, stage_seconds, stage_barrier_seconds = _timed_stage_back(jax, tree, handle, counters)
        staged_proof = _validate_tree_phase(tree, handle, identity, "staged_back")
        gc.collect()
        staged_allocator = _allocator_snapshot(device)
        staged_physical = _sample_physical_plateau(amd_card)
        stage_allocator_transition = _allocator_transition(
            "timed_stage_back", initial_allocator, staged_allocator, "allocate"
        )
        stage_physical_transition = _physical_transition(
            "timed_stage_back", initial_physical, staged_physical, "allocate"
        )
        _emit(
            {
                "record_type": "timed_transfer",
                "timestamp": _utc_now(),
                "direction": "pinned_host_to_device",
                "method": "stage_back",
                "method_seconds": stage_seconds,
                "effective_gib_per_second": _effective_gib_per_second(_SELECTED_BYTES, stage_seconds),
                "strictly_below_seconds": _MAX_TIMED_TRANSFER_SECONDS,
                "post_method_tuple_barrier_seconds": stage_barrier_seconds,
                "barrier_excluded_from_method_gate": True,
                "placement": staged_proof,
                "status": "passed",
                "counters": dict(counters),
            },
            output,
        )
        _emit_memory_state(output, "after_timed_stage_back", staged_allocator, staged_physical, counters)
        _emit(
            {
                "record_type": "memory_transition",
                "timestamp": _utc_now(),
                "allocator": stage_allocator_transition,
                "physical": stage_physical_transition,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(require_clean_boot, output, "after_timed_stage_back_attempt", counters)

    try:
        successor, reoffload_seconds, reoffload_barrier_seconds = _timed_reoffload(jax, tree, handle, counters)
        reoffloaded_proof = _validate_tree_phase(tree, successor, identity, "offloaded")
        gc.collect()
        reoffloaded_allocator = _allocator_snapshot(device)
        reoffloaded_physical = _sample_physical_plateau(amd_card)
        reoffload_allocator_transition = _allocator_transition(
            "timed_reoffload", staged_allocator, reoffloaded_allocator, "release"
        )
        reoffload_physical_transition = _physical_transition(
            "timed_reoffload", staged_physical, reoffloaded_physical, "release"
        )
        _emit(
            {
                "record_type": "timed_transfer",
                "timestamp": _utc_now(),
                "direction": "device_to_pinned_host",
                "method": "reoffload",
                "method_seconds": reoffload_seconds,
                "effective_gib_per_second": _effective_gib_per_second(_SELECTED_BYTES, reoffload_seconds),
                "strictly_below_seconds": _MAX_TIMED_TRANSFER_SECONDS,
                "post_method_tuple_barrier_seconds": reoffload_barrier_seconds,
                "barrier_excluded_from_method_gate": True,
                "placement": reoffloaded_proof,
                "status": "passed",
                "counters": dict(counters),
            },
            output,
        )
        _emit_memory_state(output, "after_timed_reoffload", reoffloaded_allocator, reoffloaded_physical, counters)
        _emit(
            {
                "record_type": "memory_transition",
                "timestamp": _utc_now(),
                "allocator": reoffload_allocator_transition,
                "physical": reoffload_physical_transition,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(require_clean_boot, output, "after_timed_reoffload_attempt", counters)

    try:
        counters["final_stage_back_attempts"] += 1
        returned = successor.stage_back()
        counters["final_stage_back_completions"] += 1
        if returned is not successor:
            raise RuntimeError("final stage_back did not return its existing handle")
        final_staged_proof = _validate_tree_phase(tree, successor, identity, "staged_back")
        gc.collect()
        final_staged_allocator = _allocator_snapshot(device)
        final_staged_physical = _sample_physical_plateau(amd_card)
        final_stage_allocator_transition = _allocator_transition(
            "final_stage_back", reoffloaded_allocator, final_staged_allocator, "allocate"
        )
        final_stage_physical_transition = _physical_transition(
            "final_stage_back", reoffloaded_physical, final_staged_physical, "allocate"
        )
        _emit(
            {
                "record_type": "untimed_final_stage_back",
                "timestamp": _utc_now(),
                "status": "passed",
                "placement": final_staged_proof,
                "counters": dict(counters),
            },
            output,
        )
        _emit_memory_state(output, "after_final_stage_back", final_staged_allocator, final_staged_physical, counters)
        _emit(
            {
                "record_type": "memory_transition",
                "timestamp": _utc_now(),
                "allocator": final_stage_allocator_transition,
                "physical": final_stage_physical_transition,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(require_clean_boot, output, "after_final_stage_back_attempt", counters)

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
        _journal_checkpoint(require_clean_boot, output, "after_device_get_attempt", counters)

    try:
        counters["final_reoffload_attempts"] += 1
        final_handle = successor.reoffload()
        counters["final_reoffload_completions"] += 1
        if final_handle is successor or successor.phase != "complete":
            raise RuntimeError("final reoffload did not consume the staged handle")
        final_proof = _validate_tree_phase(tree, final_handle, identity, "offloaded")
        gc.collect()
        final_allocator = _allocator_snapshot(device)
        final_physical = _sample_physical_plateau(amd_card)
        final_allocator_transition = _allocator_transition(
            "final_reoffload", final_staged_allocator, final_allocator, "release"
        )
        final_physical_transition = _physical_transition(
            "final_reoffload", final_staged_physical, final_physical, "release"
        )
        _emit(
            {
                "record_type": "final_offload",
                "timestamp": _utc_now(),
                "status": "passed",
                "placement": final_proof,
                "counters": dict(counters),
            },
            output,
        )
        _emit_memory_state(output, "after_final_reoffload", final_allocator, final_physical, counters)
        _emit(
            {
                "record_type": "memory_transition",
                "timestamp": _utc_now(),
                "allocator": final_allocator_transition,
                "physical": final_physical_transition,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(require_clean_boot, output, "after_final_reoffload_attempt", counters)

    if counters != _completed_counters():
        raise RuntimeError("smoke8 transfer counter contract was not completed exactly")
    round_trip_seconds = stage_seconds + reoffload_seconds
    _emit(
        {
            "record_type": "smoke8_passed",
            "timestamp": _utc_now(),
            "status": "passed",
            "scope": "8_MiB_placement_accounting_roundtrip_only",
            "timed_stage_back_seconds": stage_seconds,
            "timed_reoffload_seconds": reoffload_seconds,
            "timed_round_trip_seconds": round_trip_seconds,
            "round_trip_effective_gib_per_second": _effective_gib_per_second(2 * _SELECTED_BYTES, round_trip_seconds),
            "allocator_gates_passed": True,
            "physical_release_is_informational": True,
            "oracle_passed": True,
            "counters": dict(counters),
            "limitations": [
                "no optimizer update or numerical training equivalence",
                "no transfer/compute overlap or throughput sweep",
                "physical VRAM release may be hidden by BFC retention",
                "smoke8 only; mid64 and exact model-state cases are not implemented",
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
            "scope": "abstract_refusal" if args.platform == "abstract" else "guarded_optimizer_moment_offload_smoke8",
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
                    "pass --platform rocm --allow-gpu --case smoke8 --output explicitly under "
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
        if not environment_record["bfc_growth_allocator"] or not environment_record["unified_memory_disabled"]:
            raise RuntimeError("environment manifest did not prove BFC growth without unified memory")
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
        with guarded_process() as safety_preflight_raw:
            safety_preflight = _public_safety_preflight(safety_preflight_raw)
            _emit(
                {
                    "record_type": "safety_preflight",
                    "timestamp": _utc_now(),
                    "stage": "guard_acquired",
                    "safety": safety_preflight,
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
                    amd_card=safety_preflight["amd_cards"][0],
                )
            finally:
                try:
                    postflight = _public_clean_safety(require_clean_boot(), "safety_postflight")
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
