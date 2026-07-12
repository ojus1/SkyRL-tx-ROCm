#!/usr/bin/env python3
"""Guarded one-shot ROCm typed-FFI copy prerequisite.

The default ``abstract`` mode emits a refusal manifest without importing JAX,
the SkyRL ROCm package, or any shared library.  The ROCm path is deliberately
an exact, narrow ABI/stream smoke: one deterministic BF16 ``[1,1024,32,128]``
array is copied by the committed typed-FFI handler into a distinct output.

ROCm execution requires an explicit canonical library path and its complete
SHA-256, a fresh process, the exact ``copy8`` case, a private JSONL output, a
clean headless/unowned AMD GPU, and outer ``profile_rocm.py`` supervision.
Lowering and compilation happen once.  Compilation may dispatch bounded GPU
work.  A private one-shot capability is created only after independent
StableHLO/optimized-HLO structural gates and exact compiled-memory gates pass.

This is not Gated DeltaNet math, a model path, a VJP, or a performance result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import operator
import os
import re
import shlex
import stat
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, ContextManager, TextIO

_CASE = "copy8"
_TARGET = "skyrl_gdn_ffi_smoke_bf16_copy_v1"
_LIBRARY_BASENAME = "libskyrl_gdn_ffi_smoke_gfx1100.so"
_SHAPE = (1, 1024, 32, 128)
_NBYTES = 8 * 1024**2
_MAX_TEMP_BYTES = 64 * 1024**2
_MAX_DISPATCH_SECONDS = 0.100
_MAX_PROMOTION_SECONDS = 0.075
_COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="
_REQUIRED_SNAPSHOT_SEALS = 0x000F
_EXPECTED_WRAPPER_SHA256 = (
    "66b868b7909a2279d5ddca0e1582f8563e8097723e970d09fce733aef2ba425a"
)
_EXPECTED_HIP_SHA256 = (
    "50518d3f31cad345ba746c660796025ac8971a5d0322ad2d4fe321b786ce9fc4"
)
_EXPECTED_BUILD_SHA256 = (
    "81cbae7f2b5854fa9673459152e8975789db17232e4530ed70ec34e3ca79b600"
)
_CHECKED_CAPABILITY_TOKEN = object()
_JOURNAL_STAGES = (
    "before_backend_initialization",
    "after_backend_initialization_attempt",
    "after_ffi_lower_attempt",
    "after_ffi_compile_attempt",
    "after_host_input_construction",
    "after_explicit_input_device_put_attempt",
    "after_candidate_dispatch_attempt",
    "after_candidate_device_get_attempt",
    "after_host_validation",
    "after_library_postcheck",
)
_COMPILE_CAVEAT = (
    "lowered.compile may dispatch bounded GPU compilation, profiling, or "
    "autotuning work; the compiled executable remains inaccessible until the "
    "two independent IR summaries and exact compiled-memory gate pass"
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _emit(record: dict[str, Any], output: TextIO) -> None:
    output.write(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    output.flush()


def _redacted_text_summary(text: str) -> dict[str, Any]:
    encoded = text.encode("utf-8", errors="replace")
    return {
        "text_redacted": True,
        "utf8_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _redacted_message_summary(error: BaseException) -> dict[str, Any]:
    summary = _redacted_text_summary(str(error))
    return {
        "message_redacted": True,
        "message_utf8_bytes": summary["utf8_bytes"],
        "message_sha256": summary["sha256"],
    }


def _source_files() -> dict[str, Path]:
    repo = Path(__file__).resolve().parent.parent
    return {
        "probe_source_sha256": Path(__file__),
        "gdn_ffi_python_source_sha256": repo
        / "skyrl"
        / "tx"
        / "kernels"
        / "rocm"
        / "gdn_ffi_smoke.py",
        "gdn_ffi_hip_source_sha256": repo
        / "skyrl"
        / "tx"
        / "kernels"
        / "rocm"
        / "ffi"
        / "gdn_ffi_smoke.hip",
        "gdn_ffi_build_source_sha256": repo
        / "skyrl"
        / "tx"
        / "kernels"
        / "rocm"
        / "ffi"
        / "build_gdn_ffi_smoke_gfx1100.sh",
        "safety_helper_source_sha256": repo / "rocm" / "amdgpu_safety.py",
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {name: _file_sha256(path) for name, path in _source_files().items()}


def _assert_bound_sources() -> dict[str, Any]:
    files = _source_files()
    observed = {
        "python": _file_sha256(files["gdn_ffi_python_source_sha256"]),
        "hip": _file_sha256(files["gdn_ffi_hip_source_sha256"]),
        "build": _file_sha256(files["gdn_ffi_build_source_sha256"]),
    }
    expected = {
        "python": _EXPECTED_WRAPPER_SHA256,
        "hip": _EXPECTED_HIP_SHA256,
        "build": _EXPECTED_BUILD_SHA256,
    }
    if observed != expected:
        raise RuntimeError("committed GDN FFI prerequisite source hash mismatch")
    return {
        "passed": True,
        "committed_sources_exact": True,
        "python_sha256": observed["python"],
        "hip_sha256": observed["hip"],
        "build_sha256": observed["build"],
    }


def _outer_profile_contract() -> dict[str, Any]:
    return {
        "profile_rocm_required": True,
        "operational_dependency": True,
        "internally_proven_by_child": False,
        "timeout_seconds": 90,
        "sensor_grace_seconds": 15,
        "maximum_vram_bytes": 2 * 1024**3,
        "maximum_junction_temperature_c": 70,
        "maximum_gpu_power_watts": 200,
        "minimum_host_available_bytes": 8 * 1024**3,
        "maximum_swap_bytes": 0,
        "required_flags": [
            "--timeout=90",
            "--sensor-grace-seconds=15",
            "--max-vram-gib=2",
            "--max-junction-temp-c=70",
            "--max-gpu-power-watts=200",
            "--min-host-available-gib=8",
            "--max-swap-gib=0",
        ],
    }


def _exact_contract() -> dict[str, Any]:
    return {
        "operation": "gdn_typed_ffi_distinct_output_copy_prerequisite",
        "case": _CASE,
        "target": _TARGET,
        "input": {
            "shape": list(_SHAPE),
            "dtype": "bfloat16",
            "nbytes": _NBYTES,
            "value": "deterministic_nonconstant_finite_host_grid",
        },
        "output": {
            "shape": list(_SHAPE),
            "dtype": "bfloat16",
            "nbytes": _NBYTES,
            "distinct_buffer_required": True,
        },
        "dispatch_plan": {
            "batched_device_put_calls": 1,
            "lower_calls": 1,
            "compile_calls": 1,
            "ffi_custom_calls_per_dialect": 1,
            "candidate_invocations": 1,
            "candidate_synchronizations": 1,
            "device_get_calls": 1,
            "replay_invocations": 0,
            "backward_invocations": 0,
            "model_invocations": 0,
        },
        "compiled_ir_gate": {
            "independent_dialects": ["stablehlo", "optimized_hlo"],
            "exact_custom_call_target": _TARGET,
            "other_custom_calls": 0,
            "while_calls": 0,
            "output_aliasing": False,
        },
        "compiled_memory_gate": {
            "exact_argument_bytes": _NBYTES,
            "exact_output_bytes": _NBYTES,
            "exact_alias_bytes": 0,
            "maximum_temporary_bytes": _MAX_TEMP_BYTES,
        },
        "host_oracle": "complete bitwise equality and complete SHA-256 equality",
        "library_load_gate": {
            "source_path_is_dlopen_target": False,
            "exact_lowercase_sha256_required": True,
            "one_pass_hashing_copy_to_memfd": True,
            "private_snapshot_mode": 0o600,
            "required_snapshot_seals_mask": _REQUIRED_SNAPSHOT_SEALS,
            "snapshot_fd_retained_for_process_lifetime": True,
            "cdll_retained_for_process_lifetime": True,
        },
        "duration_gate": {
            "candidate_seconds_strictly_below": _MAX_DISPATCH_SECONDS,
            "promotion_seconds_strictly_below": _MAX_PROMOTION_SECONDS,
        },
        "scope_exclusions": {
            "gdn_math": False,
            "model_path": False,
            "vjp": False,
            "training": False,
            "kernel_performance_claim": False,
        },
        "outer_supervision": _outer_profile_contract(),
    }


def _sha256_argument(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise argparse.ArgumentTypeError(
            "must be exactly 64 lowercase hexadecimal digits"
        )
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=("abstract", "rocm"),
        default="abstract",
        help="refusal-only by default; guarded FFI work requires explicit rocm",
    )
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--case", choices=(_CASE,))
    parser.add_argument("--library", type=Path)
    parser.add_argument("--library-sha256", type=_sha256_argument)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    gpu_fields = (args.case, args.library, args.library_sha256, args.output)
    if args.platform == "abstract":
        if args.allow_gpu or any(value is not None for value in gpu_fields):
            parser.error(
                "GPU case, library, hash, and output options require --platform rocm"
            )
        return args
    if not args.allow_gpu:
        parser.error("--platform rocm requires --allow-gpu")
    if args.case != _CASE:
        parser.error("--platform rocm requires --case copy8")
    if args.library is None:
        parser.error("--platform rocm requires --library")
    if args.library_sha256 is None:
        parser.error("--platform rocm requires --library-sha256")
    if args.output is None:
        parser.error("--platform rocm requires --output")
    if args.output.exists() or args.output.is_symlink():
        parser.error("refusing to overwrite an existing or symbolic-link output")
    return args


def _zero_counters() -> dict[str, int]:
    return {
        "lower_attempts": 0,
        "lower_completions": 0,
        "compile_attempts": 0,
        "compile_completions": 0,
        "ffi_python_trace_calls": 0,
        "input_device_put_attempts": 0,
        "input_device_put_completions": 0,
        "candidate_attempts": 0,
        "candidate_completions": 0,
        "candidate_synchronizations": 0,
        "device_get_attempts": 0,
        "device_get_completions": 0,
        "lowered_callable_invocations": 0,
        "replay_invocations": 0,
        "backward_invocations": 0,
        "model_invocations": 0,
    }


def _completed_counters() -> dict[str, int]:
    expected = _zero_counters()
    expected.update(
        {
            "lower_attempts": 1,
            "lower_completions": 1,
            "compile_attempts": 1,
            "compile_completions": 1,
            "ffi_python_trace_calls": 1,
            "input_device_put_attempts": 1,
            "input_device_put_completions": 1,
            "candidate_attempts": 1,
            "candidate_completions": 1,
            "candidate_synchronizations": 1,
            "device_get_attempts": 1,
            "device_get_completions": 1,
        }
    )
    return expected


def _validate_library_path(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_absolute():
        raise ValueError("FFI library path must be absolute")
    if path.name != _LIBRARY_BASENAME or path.suffix != ".so":
        raise ValueError("FFI library must use the exact audited .so name")
    try:
        info_before = path.lstat()
    except OSError as error:
        raise ValueError("FFI library cannot be inspected") from error
    if stat.S_ISLNK(info_before.st_mode) or not stat.S_ISREG(info_before.st_mode):
        raise ValueError("FFI library must be a real regular file")
    if info_before.st_uid != os.getuid():
        raise ValueError("FFI library must be owned by the current user")
    if info_before.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("FFI library must not be group- or world-writable")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError("FFI library cannot be resolved") from error
    if resolved != path:
        raise ValueError(
            "FFI library path must be absolute, canonical, and symlink-free"
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        descriptor_info = os.fstat(descriptor)
        if (
            descriptor_info.st_dev != info_before.st_dev
            or descriptor_info.st_ino != info_before.st_ino
            or descriptor_info.st_size != info_before.st_size
        ):
            raise RuntimeError("FFI library changed while being opened")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.close(descriptor)
    observed_sha256 = digest.hexdigest()
    info_after = path.lstat()
    identity = (
        int(info_before.st_dev),
        int(info_before.st_ino),
        int(info_before.st_size),
        int(info_before.st_mtime_ns),
    )
    if identity != (
        int(info_after.st_dev),
        int(info_after.st_ino),
        int(info_after.st_size),
        int(info_after.st_mtime_ns),
    ):
        raise RuntimeError("FFI library changed while being hashed")
    if observed_sha256 != expected_sha256:
        raise RuntimeError("FFI library SHA-256 does not match --library-sha256")
    return {
        "validated": True,
        "absolute": True,
        "canonical": True,
        "symlink_free": True,
        "regular_file": True,
        "owner_exact": True,
        "group_or_world_writable": False,
        "basename": _LIBRARY_BASENAME,
        "size_bytes": int(info_after.st_size),
        "sha256": observed_sha256,
        "identity": identity,
        "raw_path_emitted": False,
    }


def _assert_same_library(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    current = _validate_library_path(path, str(manifest["sha256"]))
    if tuple(current["identity"]) != tuple(manifest["identity"]):
        raise RuntimeError(
            "FFI library identity changed after registration or execution"
        )
    return current


def _assert_fresh_accelerator_process() -> None:
    imported = sorted(
        name
        for name in sys.modules
        if name in {"jax", "jaxlib", "skyrl.tx.kernels.rocm"}
        or name.startswith("jax.")
        or name.startswith("jaxlib.")
        or name.startswith("skyrl.tx.kernels.rocm.")
    )
    if imported:
        raise RuntimeError(
            "ROCm FFI smoke requires a fresh process before JAX/ROCm import"
        )


def _validate_exact_or_unset(name: str, expected: str) -> None:
    observed = os.environ.get(name)
    if observed is not None and observed != expected:
        raise RuntimeError(f"{name} conflicts with the exact FFI smoke environment")


def _configure_rocm_environment() -> dict[str, str | None]:
    fixed = {
        "JAX_PLATFORMS": "rocm",
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
        "JAX_ROCM_VISIBLE_DEVICES": "0",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_CLIENT_MEM_FRACTION": "0.075",
    }
    for name, expected in fixed.items():
        observed = os.environ.get(name)
        if name == "XLA_PYTHON_CLIENT_PREALLOCATE" and observed is not None:
            if observed.strip().lower() not in {"0", "false", "no", "off"}:
                raise RuntimeError("preallocation conflicts with bounded FFI smoke")
            continue
        _validate_exact_or_unset(name, expected)
    inherited_flags = os.environ.get("XLA_FLAGS")
    if inherited_flags is not None:
        try:
            tokens = shlex.split(inherited_flags, posix=True)
        except ValueError as error:
            raise RuntimeError("invalid inherited XLA_FLAGS quoting") from error
        if tokens != [_COMMAND_BUFFER_FLAG]:
            raise RuntimeError(
                "XLA_FLAGS must contain solely the disabled command-buffer flag"
            )
    for name in (
        "HSA_OVERRIDE_GFX_VERSION",
        "XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB",
        "JAX_PJRT_CLIENT_CREATE_OPTIONS",
        "JAX_MOCK_GPU_TOPOLOGY",
        "TF_FORCE_UNIFIED_MEMORY",
    ):
        if os.environ.get(name) not in {None, ""}:
            raise RuntimeError(f"{name} must be unset for the exact physical-GPU smoke")
        os.environ.pop(name, None)
    if os.environ.get("MOCK_NUM_GPU_PROCESSES", "").strip() not in {"", "0"}:
        raise RuntimeError("MOCK_NUM_GPU_PROCESSES must be unset or zero")
    os.environ.pop("MOCK_NUM_GPU_PROCESSES", None)
    os.environ.update(fixed)
    os.environ["XLA_FLAGS"] = _COMMAND_BUFFER_FLAG
    return {**fixed, "XLA_FLAGS_effective": _COMMAND_BUFFER_FLAG}


def _environment_manifest(environment: dict[str, str | None]) -> dict[str, Any]:
    expected = {
        "JAX_PLATFORMS": "rocm",
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
        "JAX_ROCM_VISIBLE_DEVICES": "0",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_CLIENT_MEM_FRACTION": "0.075",
        "XLA_FLAGS_effective": _COMMAND_BUFFER_FLAG,
    }
    return {
        "fixed_values": {
            name: value if environment.get(name) == value else "<unexpected>"
            for name, value in expected.items()
        },
        "fixed_values_match_expected": all(
            environment.get(name) == value for name, value in expected.items()
        ),
        "command_buffer_flag_is_sole_xla_flag": (
            environment.get("XLA_FLAGS_effective") == _COMMAND_BUFFER_FLAG
        ),
        "raw_unrelated_environment_emitted": False,
    }


def _prove_command_buffers_disabled(
    environment: dict[str, str | None],
) -> dict[str, Any]:
    effective = environment.get("XLA_FLAGS_effective")
    if (
        effective != _COMMAND_BUFFER_FLAG
        or os.environ.get("XLA_FLAGS") != _COMMAND_BUFFER_FLAG
    ):
        raise RuntimeError(
            "command buffers are not proven disabled by the sole exact flag"
        )
    if shlex.split(effective, posix=True) != [_COMMAND_BUFFER_FLAG]:
        raise RuntimeError("effective XLA_FLAGS contains an unauthorized token")
    return {
        "passed": True,
        "command_buffers_disabled": True,
        "sole_xla_flag": True,
        "effective_xla_flags_sha256": hashlib.sha256(effective.encode()).hexdigest(),
        "raw_xla_flags_emitted": False,
    }


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


def _open_exclusive_output(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "w", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise


def _public_clean_safety(safety: Any, stage: str) -> dict[str, Any]:
    if not isinstance(safety, dict) or safety.get("amdgpu_boot_clean") is not True:
        raise RuntimeError(f"{stage} did not prove a clean AMDGPU boot")
    if safety.get("fatal_amdgpu_events") != []:
        raise RuntimeError(f"{stage} returned invalid fatal-event evidence")
    return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}


def _public_safety_preflight(safety: Any) -> dict[str, Any]:
    public = _public_clean_safety(safety, "safety_preflight")
    cards = safety.get("amd_cards")
    if (
        not isinstance(cards, list)
        or not cards
        or cards != sorted(set(cards))
        or not all(
            isinstance(card, str) and re.fullmatch(r"card[0-9]+", card)
            for card in cards
        )
    ):
        raise RuntimeError("safety preflight returned invalid AMD card evidence")
    if safety.get("connected_amd_connectors") != []:
        raise RuntimeError("safety preflight did not prove every AMD connector idle")
    if safety.get("kfd_path") != "/dev/kfd":
        raise RuntimeError("safety preflight did not prove the exact KFD path")
    if (
        safety.get("kfd_accessible") is not True
        or safety.get("kfd_unowned") is not True
    ):
        raise RuntimeError("safety preflight did not prove accessible unowned KFD")
    return {
        **public,
        "amd_cards": list(cards),
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
    if stage not in _JOURNAL_STAGES:
        raise RuntimeError("refusing an undeclared GDN FFI journal stage")
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


def _custom_call_blocks(text: str, dialect: str) -> list[str]:
    lines = text.splitlines()
    if dialect == "stablehlo":
        start = re.compile(r"^(?P<indent>\s*)%[^=]+?=\s*stablehlo\.custom_call\b")
        boundary = re.compile(
            r"^\s*(?:%[^=]+?=|#[A-Za-z_][\w.-]*\s*=|"
            r"(?:stablehlo\.|func\.)?return\b|}\s*$)"
        )
    elif dialect == "optimized_hlo":
        start = re.compile(r"^(?P<indent>\s*)(?:ROOT\s+)?[^=]+?=\s*.*\bcustom-call\(")
        boundary = re.compile(r"^\s*(?:(?:ROOT\s+)?[^=]+?=|}\s*$)")
    else:
        raise ValueError("unsupported IR dialect")
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        match = start.search(lines[index])
        if match is None:
            index += 1
            continue
        base_indent = len(match.group("indent").expandtabs())
        block_lines = [lines[index]]
        index += 1
        while index < len(lines):
            candidate = lines[index]
            candidate_indent = len(candidate) - len(candidate.lstrip(" \t"))
            if (
                candidate.strip()
                and candidate_indent <= base_indent
                and boundary.match(candidate)
            ):
                break
            block_lines.append(candidate)
            index += 1
        blocks.append("\n".join(block_lines))
    return blocks


def _metadata_definitions(text: str) -> dict[str, str]:
    definitions: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*(#[A-Za-z_][\w.-]*)\s*=\s*(.*)$", line)
        if match is not None:
            definitions[match.group(1)] = match.group(2)
    return definitions


def _resolved_block_metadata(block: str, definitions: dict[str, str]) -> str:
    pieces = [block]
    pending = list(re.findall(r"#[A-Za-z_][\w.-]*", block))
    visited: set[str] = set()
    while pending:
        reference = pending.pop()
        if reference in visited or reference not in definitions:
            continue
        visited.add(reference)
        definition = definitions[reference]
        pieces.append(f"{reference} = {definition}")
        pending.extend(re.findall(r"#[A-Za-z_][\w.-]*", definition))
    return "\n".join(pieces)


def _custom_call_targets(block: str, dialect: str) -> set[str]:
    targets = set(
        re.findall(r'(?:call_target_name|custom_call_target)\s*=\s*"([^"]+)"', block)
    )
    if dialect == "stablehlo":
        for match in re.finditer(
            r'\bstablehlo\.custom_call\s+@(?:"([^"]+)"|([A-Za-z0-9_.$-]+))',
            block,
        ):
            targets.add(match.group(1) or match.group(2))
    elif dialect != "optimized_hlo":
        raise ValueError("unsupported IR dialect")
    return targets


def _ir_summary(text: str, dialect: str) -> dict[str, Any]:
    blocks = _custom_call_blocks(text, dialect)
    definitions = _metadata_definitions(text)
    if dialect == "stablehlo":
        textual_custom_calls = len(re.findall(r"\bstablehlo\.custom_call\b", text))
        while_count = len(re.findall(r"\bstablehlo\.while\b", text))
    elif dialect == "optimized_hlo":
        textual_custom_calls = len(re.findall(r"\bcustom-call\(", text))
        while_count = len(re.findall(r"\bwhile\(", text))
    else:
        raise ValueError("unsupported IR dialect")
    calls = []
    for block in blocks:
        resolved_block = _resolved_block_metadata(block, definitions)
        lowered = resolved_block.lower()
        alias_tokens_seen: list[str] = []
        alias_tokens_nonempty: list[str] = []
        for token in (
            "output_operand_aliases",
            "output_to_operand_aliasing",
            "input_output_aliases",
        ):
            if token not in lowered:
                continue
            alias_tokens_seen.append(token)
            empty_assignment = re.search(
                rf"{re.escape(token)}\s*=\s*(?:\[\s*\]|\{{\s*\}})",
                lowered,
            )
            if empty_assignment is None:
                alias_tokens_nonempty.append(token)
        # A singular alias record is intrinsically non-empty, including when
        # it is reached through a metadata reference resolved above.
        if "output_operand_alias<" in lowered:
            alias_tokens_seen.append("output_operand_alias")
            alias_tokens_nonempty.append("output_operand_alias")
        calls.append(
            {
                "targets": sorted(_custom_call_targets(resolved_block, dialect)),
                "alias_metadata_tokens_seen": sorted(set(alias_tokens_seen)),
                "alias_metadata_tokens": sorted(set(alias_tokens_nonempty)),
                "sha256": hashlib.sha256(resolved_block.encode()).hexdigest(),
                "utf8_bytes": len(resolved_block.encode()),
            }
        )
    checks = {
        "parser_count_matches_textual_custom_call_count": len(blocks)
        == textual_custom_calls,
        "exactly_one_custom_call_total": len(blocks) == 1,
        "sole_exact_gdn_ffi_target": len(calls) == 1
        and calls[0]["targets"] == [_TARGET],
        "no_outer_while": while_count == 0,
        "no_output_alias_metadata": len(calls) == 1
        and calls[0]["alias_metadata_tokens"] == [],
    }
    return {
        "dialect": dialect,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_utf8_bytes": len(text.encode()),
        "raw_ir_emitted": False,
        "custom_call_count": len(blocks),
        "textual_custom_call_count": textual_custom_calls,
        "while_count": while_count,
        "calls": calls,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _structural_gate(*summaries: dict[str, Any]) -> dict[str, Any]:
    per_dialect = {str(summary.get("dialect")): summary for summary in summaries}
    exact = set(per_dialect) == {"stablehlo", "optimized_hlo"} and len(summaries) == 2
    checks = {
        "exactly_one_independent_summary_per_required_dialect": exact,
        "stablehlo_passed": exact and per_dialect["stablehlo"].get("passed") is True,
        "optimized_hlo_passed": exact
        and per_dialect["optimized_hlo"].get("passed") is True,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _integer_stat(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = operator.index(value)
    except TypeError:
        return None
    return int(parsed) if parsed >= 0 else None


def _compiled_memory(compiled: Any) -> dict[str, Any]:
    stats = compiled.memory_analysis()
    if stats is None:
        return {"available": False}
    values: dict[str, int] = {}
    for name in (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "generated_code_size_in_bytes",
    ):
        if hasattr(stats, name):
            parsed = _integer_stat(getattr(stats, name))
            if parsed is not None:
                values[name] = parsed
    return {"available": True, **values}


def _compiled_memory_gate(memory: dict[str, Any]) -> dict[str, Any]:
    required = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
    )
    available = memory.get("available") is True and all(
        _integer_stat(memory.get(name)) is not None for name in required
    )
    argument = (
        _integer_stat(memory.get("argument_size_in_bytes")) if available else None
    )
    output = _integer_stat(memory.get("output_size_in_bytes")) if available else None
    alias = _integer_stat(memory.get("alias_size_in_bytes")) if available else None
    temporary = _integer_stat(memory.get("temp_size_in_bytes")) if available else None
    checks = {
        "memory_analysis_available": available,
        "argument_bytes_exactly_8_mib": argument == _NBYTES,
        "output_bytes_exactly_8_mib": output == _NBYTES,
        "alias_bytes_exactly_zero": alias == 0,
        "temporary_bytes_at_most_64_mib": temporary is not None
        and temporary <= _MAX_TEMP_BYTES,
    }
    combined = (
        argument + output + temporary
        if argument is not None and output is not None and temporary is not None
        else None
    )
    return {
        "expected_argument_bytes": _NBYTES,
        "expected_output_bytes": _NBYTES,
        "expected_alias_bytes": 0,
        "maximum_temporary_bytes": _MAX_TEMP_BYTES,
        "argument_output_temporary_bytes": combined,
        "checks": checks,
        "passed": all(checks.values()),
    }


class _CheckedExecutable:
    __slots__ = ("_compiled", "_consumed", "_counters", "proof")

    def __init__(
        self,
        compiled: Any,
        *,
        proof: dict[str, Any],
        counters: dict[str, int],
        token: object,
    ) -> None:
        if token is not _CHECKED_CAPABILITY_TOKEN or proof.get("passed") is not True:
            raise RuntimeError(
                "refusing an FFI executable without both passed compile gates"
            )
        self._compiled = compiled
        self._consumed = False
        self._counters = counters
        self.proof = proof

    def invoke(self, jax: Any, value: Any, on_started: Callable[[], None]) -> Any:
        if self._consumed:
            raise RuntimeError("checked FFI executable capability was already consumed")
        self._consumed = True
        self._counters["candidate_attempts"] += 1
        on_started()
        result = self._compiled(value)
        result = jax.block_until_ready(result)
        self._counters["candidate_synchronizations"] += 1
        self._counters["candidate_completions"] += 1
        return result


def _wrap_checked(
    compiled: Any, proof: dict[str, Any], counters: dict[str, int]
) -> _CheckedExecutable:
    return _CheckedExecutable(
        compiled,
        proof=proof,
        counters=counters,
        token=_CHECKED_CAPABILITY_TOKEN,
    )


def _backend_manifest(jax: Any, jaxlib: Any, jax_backend: Any) -> dict[str, Any]:
    resolved = jax.default_backend()
    platform_version = str(jax_backend.get_backend().platform_version)
    devices = jax.devices()
    if resolved != "gpu" or "rocm" not in platform_version.lower() or len(devices) != 1:
        raise RuntimeError(
            "requested ROCm but JAX did not resolve exactly one ROCm GPU"
        )
    return {
        "platform_resolved": "gpu",
        "platform_family": "rocm",
        "visible_device_count": 1,
        "jax_version": _redacted_text_summary(str(jax.__version__)),
        "jaxlib_version": _redacted_text_summary(str(jaxlib.__version__)),
        "platform_version": _redacted_text_summary(platform_version),
        "raw_device_descriptions_emitted": False,
    }


def _sealed_registration_manifest(
    registration: Any,
    *,
    library_path: Path,
    library_sha256: str,
    library_size_bytes: int,
) -> dict[str, Any]:
    observed_path = getattr(registration, "library_path", None)
    observed_sha256 = getattr(registration, "library_sha256", None)
    snapshot_sha256 = getattr(registration, "snapshot_sha256", None)
    snapshot_size = getattr(registration, "snapshot_size_bytes", None)
    snapshot_mode = getattr(registration, "snapshot_mode", None)
    snapshot_seals = getattr(registration, "snapshot_seals", None)
    checks = {
        "original_canonical_path_identity_exact": observed_path == library_path,
        "approved_library_sha256_exact": observed_sha256 == library_sha256,
        "loaded_snapshot_sha256_exact": snapshot_sha256 == library_sha256,
        "snapshot_size_exact": snapshot_size == library_size_bytes,
        "snapshot_mode_exactly_0600": snapshot_mode == 0o600,
        "all_write_grow_shrink_and_seal_seals_present": isinstance(snapshot_seals, int)
        and not isinstance(snapshot_seals, bool)
        and snapshot_seals & _REQUIRED_SNAPSHOT_SEALS == _REQUIRED_SNAPSHOT_SEALS,
        "sealed_snapshot_proven": getattr(registration, "sealed_snapshot", None)
        is True,
        "snapshot_fd_retained": getattr(registration, "snapshot_fd_retained", None)
        is True,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "typed FFI registration did not prove the exact sealed snapshot identity"
        )
    return {
        "passed": True,
        "checks": checks,
        "library_sha256": observed_sha256,
        "snapshot_sha256": snapshot_sha256,
        "snapshot_size_bytes": snapshot_size,
        "snapshot_mode": snapshot_mode,
        "snapshot_seals": snapshot_seals,
        "required_snapshot_seals_mask": _REQUIRED_SNAPSHOT_SEALS,
        "raw_library_path_emitted": False,
        "dlopen_source": "retained_sealed_memfd_snapshot_only",
    }


def _compile_checked(
    jax: Any,
    jnp: Any,
    gdn_ffi_smoke_copy: Callable[..., Any],
    register_gdn_ffi_smoke: Callable[..., Any],
    library_path: Path,
    library_sha256: str,
    library_size_bytes: int,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    output: TextIO,
) -> tuple[_CheckedExecutable, dict[str, Any]]:
    def ffi_copy(value: Any) -> Any:
        counters["ffi_python_trace_calls"] += 1
        return gdn_ffi_smoke_copy(
            value,
            enabled=True,
            library_path=library_path,
            library_sha256=library_sha256,
        )

    signature = jax.ShapeDtypeStruct(_SHAPE, jnp.bfloat16)
    counters["lower_attempts"] += 1
    _emit(
        {
            "record_type": "stage",
            "stage": "ffi_lower_started_lazy_registration",
            "timestamp": _utc_now(),
            "counters": dict(counters),
        },
        output,
    )
    try:
        lower_start = time.perf_counter()
        lowered = jax.jit(ffi_copy).lower(signature)
        lower_seconds = time.perf_counter() - lower_start
        counters["lower_completions"] += 1
        if counters["ffi_python_trace_calls"] != 1:
            raise RuntimeError("lowering did not construct exactly one typed FFI call")
        registration = register_gdn_ffi_smoke(
            library_path,
            library_sha256=library_sha256,
            enabled=True,
        )
        sealed_registration = _sealed_registration_manifest(
            registration,
            library_path=library_path,
            library_sha256=library_sha256,
            library_size_bytes=library_size_bytes,
        )
        stablehlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
        stablehlo = _ir_summary(stablehlo_text, "stablehlo")
        del stablehlo_text
        _emit(
            {
                "record_type": "lowered",
                "timestamp": _utc_now(),
                "lower_seconds": lower_seconds,
                "registration": "lazy_during_exact_single_lower_trace",
                "sealed_registration": sealed_registration,
                "stablehlo": stablehlo,
                "raw_ir_emitted": False,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_ffi_lower_attempt", counters
        )

    counters["compile_attempts"] += 1
    _emit(
        {
            "record_type": "stage",
            "stage": "ffi_compile_started",
            "timestamp": _utc_now(),
            "compile_may_dispatch_gpu_work": True,
            "counters": dict(counters),
        },
        output,
    )
    compiled = None
    release = False
    try:
        compile_start = time.perf_counter()
        compiled = lowered.compile()
        compile_seconds = time.perf_counter() - compile_start
        counters["compile_completions"] += 1
        optimized_text = compiled.as_text()
        optimized_hlo = _ir_summary(optimized_text, "optimized_hlo")
        del optimized_text
        memory = _compiled_memory(compiled)
        structural = _structural_gate(stablehlo, optimized_hlo)
        memory_gate = _compiled_memory_gate(memory)
        proof = {
            "structural_gate_passed": structural["passed"],
            "compiled_memory_gate_passed": memory_gate["passed"],
            "passed": structural["passed"] and memory_gate["passed"],
        }
        report = {
            "record_type": "ffi_compiled",
            "timestamp": _utc_now(),
            "lower_seconds": lower_seconds,
            "compile_seconds": compile_seconds,
            "stablehlo": stablehlo,
            "optimized_hlo": optimized_hlo,
            "sealed_registration": sealed_registration,
            "structural_gate": structural,
            "compiled_memory": memory,
            "compiled_memory_gate": memory_gate,
            "release_gate": proof,
            "raw_ir_emitted": False,
            "counters": dict(counters),
        }
        _emit(report, output)
        if not proof["passed"]:
            raise RuntimeError(
                "FFI executable failed structural or compiled-memory gate"
            )
        checked = _wrap_checked(compiled, proof, counters)
        release = True
    finally:
        if compiled is not None and not release:
            del compiled
        _journal_checkpoint(
            require_clean_boot, output, "after_ffi_compile_attempt", counters
        )
    del lowered
    return checked, report


def _construct_host_input(np: Any, ml_dtypes: Any) -> tuple[Any, dict[str, Any]]:
    count = math.prod(_SHAPE)
    indices = np.arange(count, dtype=np.uint32)
    integer_grid = (
        (indices * np.uint32(73) + (indices >> np.uint32(5)) * np.uint32(19))
        % np.uint32(4093)
    ).astype(np.int32)
    host = ((integer_grid - 2046).astype(np.float32) / np.float32(64.0)).astype(
        ml_dtypes.bfloat16
    )
    host = host.reshape(_SHAPE)
    host_fp32 = np.asarray(host, dtype=np.float32)
    if (
        tuple(host.shape) != _SHAPE
        or str(host.dtype) != "bfloat16"
        or int(host.nbytes) != _NBYTES
        or not bool(np.all(np.isfinite(host_fp32)))
        or not bool(np.any(host != host.reshape(-1)[0]))
    ):
        raise RuntimeError(
            "host FFI copy input violated the exact deterministic contract"
        )
    digest = hashlib.sha256(host.tobytes(order="C")).hexdigest()
    return host, {
        "shape": list(host.shape),
        "dtype": str(host.dtype),
        "nbytes": int(host.nbytes),
        "finite": True,
        "nonconstant": True,
        "randomness_used": False,
        "sha256": digest,
    }


def _device_put(
    jax: Any,
    host: Any,
    counters: dict[str, int],
) -> Any:
    counters["input_device_put_attempts"] += 1
    placed = jax.device_put((host,))
    placed = jax.block_until_ready(placed)
    counters["input_device_put_completions"] += 1
    if not isinstance(placed, tuple) or len(placed) != 1:
        raise RuntimeError("batched device_put did not preserve the singleton tuple")
    return placed[0]


def _dispatch(
    jax: Any,
    executable: _CheckedExecutable,
    value: Any,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    output: TextIO,
) -> tuple[Any, float]:
    started: float | None = None

    def on_started() -> None:
        nonlocal started
        _emit(
            {
                "record_type": "dispatch_started",
                "timestamp": _utc_now(),
                "label": "single_copy8_candidate",
                "counters": dict(counters),
            },
            output,
        )
        started = time.perf_counter()

    fallback = time.perf_counter()
    try:
        result = executable.invoke(jax, value, on_started)
    finally:
        seconds = time.perf_counter() - (started if started is not None else fallback)
        _journal_checkpoint(
            require_clean_boot, output, "after_candidate_dispatch_attempt", counters
        )
    _emit(
        {
            "record_type": "dispatch",
            "timestamp": _utc_now(),
            "label": "single_copy8_candidate",
            "seconds": seconds,
            "counters": dict(counters),
        },
        output,
    )
    return result, seconds


def _device_get(
    jax: Any,
    value: Any,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    output: TextIO,
) -> Any:
    counters["device_get_attempts"] += 1
    try:
        host = jax.device_get(value)
        counters["device_get_completions"] += 1
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_candidate_device_get_attempt", counters
        )
    return host


def _validate_output(
    np: Any,
    expected: Any,
    actual: Any,
    seconds: float,
    counters: dict[str, int],
    output: TextIO,
) -> dict[str, Any]:
    expected_array = np.asarray(expected)
    actual_array = np.asarray(actual)
    expected_bytes = expected_array.tobytes(order="C")
    actual_bytes = actual_array.tobytes(order="C")
    expected_sha = hashlib.sha256(expected_bytes).hexdigest()
    actual_sha = hashlib.sha256(actual_bytes).hexdigest()
    shape_dtype_nbytes_exact = (
        tuple(expected_array.shape) == _SHAPE
        and tuple(actual_array.shape) == _SHAPE
        and str(expected_array.dtype) == "bfloat16"
        and str(actual_array.dtype) == "bfloat16"
        and int(expected_array.nbytes) == _NBYTES
        and int(actual_array.nbytes) == _NBYTES
    )
    bitwise = expected_bytes == actual_bytes and expected_sha == actual_sha
    finite = bool(np.all(np.isfinite(np.asarray(actual_array, dtype=np.float32))))
    safety_duration = math.isfinite(seconds) and 0 <= seconds < _MAX_DISPATCH_SECONDS
    promotion_duration = (
        math.isfinite(seconds) and 0 <= seconds < _MAX_PROMOTION_SECONDS
    )
    passed = (
        shape_dtype_nbytes_exact
        and finite
        and bitwise
        and safety_duration
        and promotion_duration
    )
    record = {
        "record_type": "host_validation",
        "timestamp": _utc_now(),
        "status": "passed" if passed else "failed",
        "metrics": {
            "shape_dtype_nbytes_exact": shape_dtype_nbytes_exact,
            "finite": finite,
            "complete_bitwise_equal": bitwise,
            "complete_byte_count_compared": len(expected_bytes),
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "actual_shape": list(actual_array.shape),
            "actual_dtype": str(actual_array.dtype),
            "actual_nbytes": int(actual_array.nbytes),
        },
        "gates": {
            "safety_duration_passed": safety_duration,
            "promotion_duration_passed": promotion_duration,
            "promotion_passed": passed,
        },
        "candidate_seconds": seconds,
        "thresholds": {
            "candidate_seconds_strictly_below": _MAX_DISPATCH_SECONDS,
            "promotion_seconds_strictly_below": _MAX_PROMOTION_SECONDS,
        },
        "counters": dict(counters),
    }
    _emit(record, output)
    if not passed:
        raise RuntimeError("copy8 candidate failed exact oracle or duration gate")
    return record


def _run_rocm(
    args: argparse.Namespace,
    output: TextIO,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    *,
    environment: dict[str, str | None],
    library_manifest: dict[str, Any],
    _dependencies: tuple[
        Any,
        Any,
        Any,
        Any,
        Any,
        Any,
        Callable[..., Any],
        Callable[..., Any],
    ]
    | None = None,
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
    _journal_checkpoint(
        require_clean_boot, output, "before_backend_initialization", counters
    )
    try:
        if _dependencies is None:
            import jax
            import jax.numpy as jnp
            import jaxlib
            import ml_dtypes
            import numpy as np
            from jax.extend import backend as jax_backend

            from skyrl.tx.kernels.rocm.gdn_ffi_smoke import (
                gdn_ffi_smoke_copy,
                register_gdn_ffi_smoke,
            )
        else:
            (
                jax,
                jnp,
                jaxlib,
                jax_backend,
                np,
                ml_dtypes,
                gdn_ffi_smoke_copy,
                register_gdn_ffi_smoke,
            ) = _dependencies
        wrapper_path = _source_files()["gdn_ffi_python_source_sha256"].resolve()
        module = sys.modules.get("skyrl.tx.kernels.rocm.gdn_ffi_smoke")
        if _dependencies is None and (
            module is None
            or not isinstance(getattr(module, "__file__", None), str)
            or Path(module.__file__).resolve() != wrapper_path
        ):
            raise RuntimeError("loaded GDN FFI wrapper is not the exact committed file")
        backend = _backend_manifest(jax, jaxlib, jax_backend)
        _emit(
            {
                "record_type": "backend_ready",
                "timestamp": _utc_now(),
                "backend": backend,
                "typed_ffi_registration_deferred_until_lower_trace": True,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_backend_initialization_attempt", counters
        )

    executable, compile_report = _compile_checked(
        jax,
        jnp,
        gdn_ffi_smoke_copy,
        register_gdn_ffi_smoke,
        args.library,
        args.library_sha256,
        int(library_manifest["size_bytes"]),
        require_clean_boot,
        counters,
        output,
    )
    try:
        host_input, host_manifest = _construct_host_input(np, ml_dtypes)
        _emit(
            {
                "record_type": "host_input",
                "timestamp": _utc_now(),
                "construction": "deterministic finite nonconstant host-only grid",
                "manifest": host_manifest,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_host_input_construction", counters
        )
    try:
        device_input = _device_put(jax, host_input, counters)
    finally:
        _journal_checkpoint(
            require_clean_boot,
            output,
            "after_explicit_input_device_put_attempt",
            counters,
        )
    device_output, seconds = _dispatch(
        jax, executable, device_input, require_clean_boot, counters, output
    )
    actual_host = _device_get(jax, device_output, require_clean_boot, counters, output)
    try:
        validation = _validate_output(
            np, host_input, actual_host, seconds, counters, output
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_host_validation", counters
        )
    try:
        final_library = _assert_same_library(args.library, library_manifest)
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_library_postcheck", counters
        )
    if counters != _completed_counters():
        raise RuntimeError("copy8 one-shot counter contract was not completed exactly")
    _emit(
        {
            "record_type": "runtime_passed",
            "timestamp": _utc_now(),
            "status": "passed_exact_copy8_typed_ffi_prerequisite",
            "compile_release_gate": compile_report["release_gate"],
            "compiled_memory_gate": compile_report["compiled_memory_gate"],
            "sealed_registration": compile_report["sealed_registration"],
            "host_validation": validation,
            "library": {
                key: value for key, value in final_library.items() if key != "identity"
            },
            "counters": dict(counters),
            "gdn_math_validated": False,
            "model_path_connected": False,
            "vjp_validated": False,
            "performance_claim_authorized": False,
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
            else "guarded_copy8",
            "contract": _exact_contract(),
            "fresh_process_required": True,
            "compile_may_dispatch_gpu_work": args.platform == "rocm",
            "compile_dispatch_caveat": _COMPILE_CAVEAT,
            "outer_profile_rocm_supervision_required": True,
            "outer_profile_rocm_supervision_operational_not_internally_proven": True,
            "raw_library_path_emitted": False,
            "raw_ir_emitted": False,
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
                    "run a fresh process under profile_rocm.py with --platform rocm "
                    "--allow-gpu --case copy8 --library, --library-sha256, and --output"
                ),
                "jax_imported": False,
                "skyrl_rocm_package_imported": False,
                "shared_library_loaded": False,
                "counters": dict(counters),
            },
            output,
        )
        return 0

    stage = "fresh_process_preflight"
    try:
        _assert_fresh_accelerator_process()
        stage = "bound_sources"
        bound_sources = _assert_bound_sources()
        stage = "library_preflight"
        library_manifest = _validate_library_path(args.library, args.library_sha256)
        _emit(
            {
                "record_type": "prerequisite_proof",
                "timestamp": _utc_now(),
                "bound_sources": bound_sources,
                "library": {
                    key: value
                    for key, value in library_manifest.items()
                    if key != "identity"
                },
                "counters": dict(counters),
            },
            output,
        )
        stage = "bounded_environment"
        environment = _configure_rocm_environment()
        _emit(
            {
                "record_type": "environment",
                "timestamp": _utc_now(),
                "environment": _environment_manifest(environment),
                "counters": dict(counters),
            },
            output,
        )
        guarded_process, require_clean_boot = _load_safety_helpers()
        stage = "safety_preflight"
        with guarded_process() as raw_safety:
            safety = _public_safety_preflight(raw_safety)
            _emit(
                {
                    "record_type": "safety_preflight",
                    "timestamp": _utc_now(),
                    "safety": safety,
                    "counters": dict(counters),
                },
                output,
            )
            stage = "runtime"
            try:
                result = _run_rocm(
                    args,
                    output,
                    require_clean_boot,
                    counters,
                    environment=environment,
                    library_manifest=library_manifest,
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
