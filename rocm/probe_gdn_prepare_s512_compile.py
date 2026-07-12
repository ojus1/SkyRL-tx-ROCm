#!/usr/bin/env python3
"""Fail-closed compile-only gate for exact S512 GDN prepare typed FFI.

The default ``abstract`` mode emits a refusal manifest without importing JAX,
the SkyRL ROCm package, or a shared library.  The explicit ROCm path registers
one sealed external library snapshot, lowers ``gdn_prepare_s512`` once from
``ShapeDtypeStruct`` inputs, compiles once, extracts two independent IR
summaries and compiler memory accounting, then destroys the uninvoked
executable.

``lowered.compile()`` can submit bounded ROCm compiler work.  It therefore
requires the shared AMDGPU guard and outer ``profile_rocm.py`` supervision even
though this probe constructs no user arrays and invokes no lowered or compiled
callable.  This gate proves neither GDN numerics nor runtime launch safety.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

_CASE = "s512-compile"
_TARGET = "skyrl_gdn_prepare_s512_f32_v1"
_LIBRARY_BASENAME = "libskyrl_gdn_prepare_s512_gfx1100.so"
_KEY_SHAPE = (1, 512, 16, 128)
_VALUE_SHAPE = (1, 512, 32, 128)
_GATE_SHAPE = (1, 512, 32)
_ARGUMENT_BYTES = 12_713_984
_OUTPUT_BYTES = 16_842_752
_MAX_TEMP_BYTES = 64 * 1024**2
_MAX_TOTAL_BYTES = 96 * 1024**2
_COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="
_REQUIRED_SNAPSHOT_SEALS = 0x000F
_EXPECTED_WRAPPER_SHA256 = (
    "39204094b1d9e1e8caddcc833cc02edb9dab2b7e32bbee75c5462fd771a6052f"
)
_EXPECTED_HIP_SHA256 = (
    "8deaf58f5bf68e936472c434c985a52bea8ea1e26b75983af9ec8ae9c1e80f45"
)
_EXPECTED_SAFETY_SHA256 = (
    "8b9441e0147b35b000fe8340ea7b7c702372320c569ced71410b92b558953e91"
)
_EXPECTED_SEALED_LOADER_SHA256 = (
    "66b868b7909a2279d5ddca0e1582f8563e8097723e970d09fce733aef2ba425a"
)
_EXPECTED_PACKAGE_SHA256 = {
    "skyrl": "667d4a15b970b851e20d17510224670c14646cfb6d5a1e388ca6b9cc6da8bf41",
    "tx": "a7abb3e76d66df1f4472bb7a02b032ef31b959ca937fd351637b4e9b4a8fa95a",
    "kernels": "40abe638c7726fe5680b7c88321042016a0f695d86acfbef52337421e7257c1a",
    "rocm": "6d12a789cf1108538a04fbacd0b38a15dbcb8255cd0ca0fadf5a76c4191a4cfd",
}
_COMPILE_CAVEAT = (
    "lowered.compile may dispatch bounded ROCm compilation, profiling, or "
    "autotuning work; no numerical or runtime-launch claim follows from this "
    "compile-only result"
)
_JOURNAL_STAGES = frozenset(
    {
        "before_backend_initialization",
        "after_backend_initialization_attempt",
        "after_ffi_registration_attempt",
        "after_ffi_lower_attempt",
        "after_ffi_compile_attempt",
        "after_library_postcheck",
    }
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _emit(record: dict[str, Any], output: TextIO) -> None:
    output.write(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    output.flush()


def _redacted_message(error: BaseException) -> dict[str, Any]:
    encoded = str(error).encode("utf-8", errors="replace")
    return {
        "message_redacted": True,
        "message_utf8_bytes": len(encoded),
        "message_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _source_files() -> dict[str, Path]:
    repo = Path(__file__).resolve().parent.parent
    return {
        "probe_source_sha256": Path(__file__),
        "gdn_prepare_wrapper_source_sha256": repo
        / "skyrl"
        / "tx"
        / "kernels"
        / "rocm"
        / "gdn_prepare_ffi.py",
        "gdn_prepare_hip_source_sha256": repo
        / "skyrl"
        / "tx"
        / "kernels"
        / "rocm"
        / "ffi"
        / "gdn_prepare_s512.hip",
        "sealed_loader_source_sha256": repo
        / "skyrl"
        / "tx"
        / "kernels"
        / "rocm"
        / "gdn_ffi_smoke.py",
        "skyrl_package_source_sha256": repo / "skyrl" / "__init__.py",
        "tx_package_source_sha256": repo / "skyrl" / "tx" / "__init__.py",
        "kernels_package_source_sha256": repo
        / "skyrl"
        / "tx"
        / "kernels"
        / "__init__.py",
        "rocm_package_source_sha256": repo
        / "skyrl"
        / "tx"
        / "kernels"
        / "rocm"
        / "__init__.py",
        "safety_helper_source_sha256": repo / "rocm" / "amdgpu_safety.py",
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {name: _file_sha256(path) for name, path in _source_files().items()}


def _assert_bound_sources() -> dict[str, Any]:
    files = _source_files()
    observed = {
        "wrapper": _file_sha256(files["gdn_prepare_wrapper_source_sha256"]),
        "hip": _file_sha256(files["gdn_prepare_hip_source_sha256"]),
        "safety": _file_sha256(files["safety_helper_source_sha256"]),
        "sealed_loader": _file_sha256(files["sealed_loader_source_sha256"]),
        "package_skyrl": _file_sha256(files["skyrl_package_source_sha256"]),
        "package_tx": _file_sha256(files["tx_package_source_sha256"]),
        "package_kernels": _file_sha256(files["kernels_package_source_sha256"]),
        "package_rocm": _file_sha256(files["rocm_package_source_sha256"]),
    }
    expected = {
        "wrapper": _EXPECTED_WRAPPER_SHA256,
        "hip": _EXPECTED_HIP_SHA256,
        "safety": _EXPECTED_SAFETY_SHA256,
        "sealed_loader": _EXPECTED_SEALED_LOADER_SHA256,
        "package_skyrl": _EXPECTED_PACKAGE_SHA256["skyrl"],
        "package_tx": _EXPECTED_PACKAGE_SHA256["tx"],
        "package_kernels": _EXPECTED_PACKAGE_SHA256["kernels"],
        "package_rocm": _EXPECTED_PACKAGE_SHA256["rocm"],
    }
    if observed != expected:
        raise RuntimeError("committed GDN prepare or safety source hash mismatch")
    return {"passed": True, "committed_sources_exact": True, **observed}


def _outer_profile_contract() -> dict[str, Any]:
    return {
        "profile_rocm_required": True,
        "operational_dependency": True,
        "internally_proven_by_child": False,
        "timeout_seconds": 120,
        "sensor_grace_seconds": 15,
        "maximum_vram_bytes": 2 * 1024**3,
        "maximum_junction_temperature_c": 90,
        "maximum_gpu_power_watts": 315,
        "minimum_host_available_bytes": 8 * 1024**3,
        "maximum_swap_bytes": 0,
    }


def _exact_contract() -> dict[str, Any]:
    return {
        "model_family": "Qwen/Qwen3.5-4B",
        "operation": "gdn_prepare_s512_typed_ffi_compile_only",
        "case": _CASE,
        "target": _TARGET,
        "inputs": [
            {"name": "key", "shape": list(_KEY_SHAPE), "dtype": "float32"},
            {"name": "value", "shape": list(_VALUE_SHAPE), "dtype": "float32"},
            {"name": "g", "shape": list(_GATE_SHAPE), "dtype": "float32"},
            {"name": "beta", "shape": list(_GATE_SHAPE), "dtype": "float32"},
        ],
        "outputs": [
            {"name": "u", "shape": list(_VALUE_SHAPE), "dtype": "float32"},
            {"name": "w", "shape": list(_VALUE_SHAPE), "dtype": "float32"},
            {"name": "cumulative_g", "shape": list(_GATE_SHAPE), "dtype": "float32"},
        ],
        "dispatch_plan": {
            "shape_dtype_structs": 4,
            "registration_attempts": 1,
            "lower_calls": 1,
            "compile_calls": 1,
            "ffi_custom_calls_per_dialect": 1,
            "constructed_user_arrays": 0,
            "lowered_callable_invocations": 0,
            "compiled_executable_invocations": 0,
            "device_put_calls": 0,
            "device_get_calls": 0,
            "synchronizations": 0,
        },
        "compiled_ir_gate": {
            "independent_dialects": ["stablehlo", "optimized_hlo"],
            "exact_custom_call_target": _TARGET,
            "exact_custom_calls_total": 1,
            "while_calls": 0,
            "output_aliasing": False,
            "r4_minor_to_major": [3, 2, 1, 0],
            "r4_occurrences": 4,
            "r3_minor_to_major": [2, 1, 0],
            "r3_occurrences": 3,
        },
        "compiled_memory_gate": {
            "exact_argument_bytes": _ARGUMENT_BYTES,
            "exact_output_bytes": _OUTPUT_BYTES,
            "exact_alias_bytes": 0,
            "maximum_temporary_bytes": _MAX_TEMP_BYTES,
            "maximum_argument_output_temporary_bytes": _MAX_TOTAL_BYTES,
        },
        "library_load_gate": {
            "source_path_is_dlopen_target": False,
            "exact_lowercase_sha256_required": True,
            "private_snapshot_mode": 0o600,
            "required_snapshot_seals_mask": _REQUIRED_SNAPSHOT_SEALS,
            "snapshot_fd_retained_for_process_lifetime": True,
        },
        "scope_exclusions": {
            "constructed_arrays": False,
            "executable_invocation": False,
            "gdn_numerics": False,
            "launch_safety": False,
            "performance": False,
            "model_integration": False,
            "vjp": False,
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
    parser.add_argument("--platform", choices=("abstract", "rocm"), default="abstract")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--case", choices=(_CASE,))
    parser.add_argument("--library", type=Path)
    parser.add_argument("--library-sha256", type=_sha256_argument)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    gpu_fields = (args.case, args.library, args.library_sha256, args.output)
    if args.platform == "abstract":
        if args.allow_gpu or any(value is not None for value in gpu_fields):
            parser.error("GPU case, library, hash, and output require --platform rocm")
        return args
    if not args.allow_gpu:
        parser.error("--platform rocm requires --allow-gpu")
    if args.case != _CASE:
        parser.error(f"--platform rocm requires --case {_CASE}")
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
        "backend_initialization_attempts": 0,
        "backend_initialization_completions": 0,
        "registration_attempts": 0,
        "registration_completions": 0,
        "shape_dtype_structs": 0,
        "ffi_python_trace_calls": 0,
        "lower_attempts": 0,
        "lower_completions": 0,
        "compile_attempts": 0,
        "compile_completions": 0,
        "constructed_user_arrays": 0,
        "device_put_calls": 0,
        "device_get_calls": 0,
        "synchronizations": 0,
        "lowered_callable_invocations": 0,
        "compiled_executable_invocations": 0,
    }


def _completed_counters() -> dict[str, int]:
    result = _zero_counters()
    result.update(
        {
            "backend_initialization_attempts": 1,
            "backend_initialization_completions": 1,
            "registration_attempts": 1,
            "registration_completions": 1,
            "shape_dtype_structs": 4,
            "ffi_python_trace_calls": 1,
            "lower_attempts": 1,
            "lower_completions": 1,
            "compile_attempts": 1,
            "compile_completions": 1,
        }
    )
    return result


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
        raise RuntimeError("ROCm GDN compile gate requires a fresh process")


def _validate_library_path(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_absolute():
        raise ValueError("GDN prepare library path must be absolute")
    if path.name != _LIBRARY_BASENAME:
        raise ValueError("GDN prepare library must use the exact audited basename")
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError("GDN prepare library cannot be inspected") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("GDN prepare library must be a real regular file")
    if before.st_uid != os.getuid():
        raise ValueError("GDN prepare library must be owned by the current user")
    if before.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("GDN prepare library must not be group- or world-writable")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError("GDN prepare library cannot be resolved") from error
    if resolved != path:
        raise ValueError("GDN prepare library path must be canonical and symlink-free")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise RuntimeError("GDN prepare library changed while being opened")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    after = path.lstat()
    identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
    )
    if identity != (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
    ):
        raise RuntimeError("GDN prepare library changed while being hashed")
    observed_sha256 = digest.hexdigest()
    if observed_sha256 != expected_sha256:
        raise RuntimeError("GDN prepare library SHA-256 does not match")
    return {
        "validated": True,
        "canonical": True,
        "symlink_free": True,
        "owner_exact": True,
        "group_or_world_writable": False,
        "basename": _LIBRARY_BASENAME,
        "size_bytes": int(after.st_size),
        "sha256": observed_sha256,
        "identity": identity,
        "raw_path_emitted": False,
    }


def _assert_same_library(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    current = _validate_library_path(path, str(manifest["sha256"]))
    if tuple(current["identity"]) != tuple(manifest["identity"]):
        raise RuntimeError("GDN prepare library identity changed after registration")
    return current


def _validate_exact_or_unset(name: str, expected: str) -> None:
    observed = os.environ.get(name)
    if observed is not None and observed != expected:
        raise RuntimeError(f"{name} conflicts with the exact GDN compile environment")


def _configure_rocm_environment() -> dict[str, str]:
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
                raise RuntimeError("preallocation conflicts with bounded GDN compile")
            continue
        _validate_exact_or_unset(name, expected)
    inherited = os.environ.get("XLA_FLAGS")
    if inherited is not None:
        try:
            tokens = shlex.split(inherited, posix=True)
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
            raise RuntimeError(
                f"{name} must be unset for the physical-GPU compile gate"
            )
        os.environ.pop(name, None)
    if os.environ.get("MOCK_NUM_GPU_PROCESSES", "").strip() not in {"", "0"}:
        raise RuntimeError("MOCK_NUM_GPU_PROCESSES must be unset or zero")
    os.environ.pop("MOCK_NUM_GPU_PROCESSES", None)
    os.environ.update(fixed)
    os.environ["XLA_FLAGS"] = _COMMAND_BUFFER_FLAG
    return {**fixed, "XLA_FLAGS_effective": _COMMAND_BUFFER_FLAG}


def _prove_command_buffers_disabled(environment: dict[str, str]) -> dict[str, Any]:
    effective = environment.get("XLA_FLAGS_effective")
    if effective != _COMMAND_BUFFER_FLAG or os.environ.get("XLA_FLAGS") != effective:
        raise RuntimeError("command buffers are not proven disabled")
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
    Callable[[], ContextManager[dict[str, Any]]], Callable[[], dict[str, Any]]
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
) -> None:
    if stage not in _JOURNAL_STAGES:
        raise RuntimeError("refusing an undeclared GDN compile journal stage")
    _emit(
        {
            "record_type": "journal_checkpoint",
            "timestamp": _utc_now(),
            "stage": stage,
            "safety": _public_clean_safety(require_clean_boot(), stage),
            "counters": dict(counters),
        },
        output,
    )


def _custom_call_blocks(text: str, dialect: str) -> list[str]:
    lines = text.splitlines()
    if dialect == "stablehlo":
        start = re.compile(r"^(?P<indent>\s*)%[^=]+?=\s*stablehlo\.custom_call\b")
        boundary = re.compile(
            r"^\s*(?:%[^=]+?=|#[A-Za-z_][\w.-]*\s*=|(?:stablehlo\.|func\.)?return\b|}\s*$)"
        )
    elif dialect == "optimized_hlo":
        start = re.compile(
            r"^(?P<indent>\s*)(?:ROOT\s+)?[^=]+?=\s*.*\bcustom-call\s*\("
        )
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
            r'\bstablehlo\.custom_call\s+@(?:"([^"]+)"|([A-Za-z0-9_.$-]+))', block
        ):
            targets.add(match.group(1) or match.group(2))
    elif dialect != "optimized_hlo":
        raise ValueError("unsupported IR dialect")
    return targets


def _integer_sequence(value: str) -> tuple[int, ...] | None:
    pieces = [piece.strip() for piece in value.split(",")]
    if not pieces or any(re.fullmatch(r"[0-9]+", piece) is None for piece in pieces):
        return None
    return tuple(int(piece) for piece in pieces)


def _top_level_custom_call_attributes(
    block: str, *, after: int, before: int
) -> str | None:
    # Preserve byte offsets while masking strings so a backend_config payload
    # cannot impersonate structural custom-call attributes.
    masked = re.sub(
        r'"(?:\\.|[^"\\])*"',
        lambda match: '"' + " " * (len(match.group()) - 2) + '"',
        block,
    )
    opening = masked.find("{", after, before)
    if opening < 0:
        return None
    depth = 0
    closing = -1
    for index in range(opening, before):
        if masked[index] == "{":
            depth += 1
        elif masked[index] == "}":
            depth -= 1
            if depth == 0:
                closing = index
                break
            if depth < 0:
                return None
    if closing < 0:
        return None
    content = list(masked[opening + 1 : closing])
    nested = 0
    for index, character in enumerate(content):
        if character == "{":
            nested += 1
            content[index] = " "
        elif character == "}":
            content[index] = " "
            nested -= 1
            if nested < 0:
                return None
        elif nested:
            content[index] = " "
    if nested != 0:
        return None
    return "".join(content)


def _balanced_square_value(text: str, after_assignment: int) -> tuple[str, int] | None:
    opening = after_assignment
    while opening < len(text) and text[opening].isspace():
        opening += 1
    if opening >= len(text) or text[opening] != "[":
        return None
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "[":
            depth += 1
        elif text[index] == "]":
            depth -= 1
            if depth == 0:
                return text[opening : index + 1], index + 1
            if depth < 0:
                return None
    return None


def _dense_layout_list(value: str) -> tuple[tuple[int, ...], ...] | None:
    if len(value) < 2 or value[0] != "[" or value[-1] != "]":
        return None
    inner = value[1:-1]
    if not inner.strip():
        return ()
    position = 0
    layouts: list[tuple[int, ...]] = []
    entry = re.compile(
        r"\s*dense<\[\s*(?P<values>[0-9,\s]+?)\s*\]>\s*"
        r"(?::\s*tensor<\s*(?P<annotation>[0-9]+)\s*x\s*"
        r"(?:index|i64)\s*>)?\s*"
    )
    while position < len(inner):
        match = entry.match(inner, position)
        if match is None:
            return None
        layout = _integer_sequence(match.group("values"))
        if layout is None:
            return None
        annotation = match.group("annotation")
        if annotation is not None and int(annotation) != len(layout):
            return None
        layouts.append(layout)
        position = match.end()
        if position == len(inner):
            break
        if inner[position] != ",":
            return None
        position += 1
        if not inner[position:].strip():
            return None
    return tuple(layouts)


def _stablehlo_layout_proof(block: str) -> dict[str, Any]:
    input_types = (
        "tensor<1x512x16x128xf32>",
        "tensor<1x512x32x128xf32>",
        "tensor<1x512x32xf32>",
        "tensor<1x512x32xf32>",
    )
    output_types = (
        "tensor<1x512x32x128xf32>",
        "tensor<1x512x32x128xf32>",
        "tensor<1x512x32xf32>",
    )
    tensor = r"tensor<[^>]+>"
    tensor_list = rf"{tensor}(?:\s*,\s*{tensor})*"
    signature_pattern = re.compile(
        rf":\s*\((?P<inputs>{tensor_list})\)\s*->\s*"
        rf"(?:(?:tuple<|\()(?P<outputs>{tensor_list})(?:>|\))|"
        rf"(?P<single>{tensor}))",
        re.DOTALL,
    )
    signatures = list(signature_pattern.finditer(block))
    signature = signatures[-1] if signatures else None
    observed_inputs = (
        tuple(re.findall(tensor, signature.group("inputs")))
        if signature is not None
        else ()
    )
    if signature is None:
        observed_outputs: tuple[str, ...] = ()
    elif signature.group("outputs") is not None:
        observed_outputs = tuple(re.findall(tensor, signature.group("outputs")))
    else:
        observed_outputs = (str(signature.group("single")),)

    operand_match = re.search(
        r"\bstablehlo\.custom_call\s+@(?:\"[^\"]+\"|[A-Za-z0-9_.$-]+)"
        r"\((?P<operands>[^)]*)\)",
        block,
        re.DOTALL,
    )
    operand_names = (
        re.findall(r"%[A-Za-z0-9_.$-]+", operand_match.group("operands"))
        if operand_match is not None
        else []
    )
    signature_start = signature.start() if signature is not None else -1
    attributes = (
        _top_level_custom_call_attributes(
            block,
            after=operand_match.end() if operand_match is not None else 0,
            before=signature_start,
        )
        if signature_start >= 0
        else None
    )
    operand_labels = (
        list(re.finditer(r"\boperand_layouts\b\s*=", attributes))
        if attributes is not None
        else []
    )
    result_labels = (
        list(re.finditer(r"\bresult_layouts\b\s*=", attributes))
        if attributes is not None
        else []
    )
    operand_value = (
        _balanced_square_value(attributes, operand_labels[0].end())
        if attributes is not None and len(operand_labels) == 1
        else None
    )
    result_value = (
        _balanced_square_value(attributes, result_labels[0].end())
        if attributes is not None and len(result_labels) == 1
        else None
    )
    labels_ordered = (
        len(operand_labels) == 1
        and len(result_labels) == 1
        and operand_value is not None
        and result_value is not None
        and operand_value[1] <= result_labels[0].start()
    )

    if labels_ordered:
        parsed_operand_layouts = _dense_layout_list(operand_value[0])
        parsed_output_layouts = _dense_layout_list(result_value[0])
    else:
        parsed_operand_layouts = None
        parsed_output_layouts = None
    operand_layouts = parsed_operand_layouts or ()
    output_layouts = parsed_output_layouts or ()
    expected_input_layouts = ((3, 2, 1, 0), (3, 2, 1, 0), (2, 1, 0), (2, 1, 0))
    expected_output_layouts = ((3, 2, 1, 0), (3, 2, 1, 0), (2, 1, 0))
    checks = {
        "one_typed_signature": len(signatures) == 1,
        "four_call_operands": len(operand_names) == 4 and len(set(operand_names)) == 4,
        "input_shapes_and_dtypes_exact_in_order": observed_inputs == input_types,
        "output_shapes_and_dtypes_exact_in_order": observed_outputs == output_types,
        "layout_attributes_ordered_before_signature": labels_ordered,
        "layout_attribute_values_fully_parsed": parsed_operand_layouts is not None
        and parsed_output_layouts is not None,
        "input_shapes_bound_to_exact_layouts": operand_layouts
        == expected_input_layouts,
        "output_shapes_bound_to_exact_layouts": output_layouts
        == expected_output_layouts,
    }
    return {
        "representation": "xla_minor_to_major",
        "observed_input_types": list(observed_inputs),
        "observed_output_types": list(observed_outputs),
        "observed_input_layouts": [
            list(item) if item is not None else None for item in operand_layouts
        ],
        "observed_output_layouts": [
            list(item) if item is not None else None for item in output_layouts
        ],
        "expected_r4": [3, 2, 1, 0],
        "expected_r3": [2, 1, 0],
        "observed_r4_count": sum(
            item == (3, 2, 1, 0) for item in (*operand_layouts, *output_layouts)
        ),
        "observed_r3_count": sum(
            item == (2, 1, 0) for item in (*operand_layouts, *output_layouts)
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _optimized_hlo_layout_proof(block: str) -> dict[str, Any]:
    instruction = re.search(
        r"=\s*(?P<outputs>\([^)]*\))\s*custom-call\s*\("
        r"(?P<inputs>[^)]*)\)\s*,",
        block,
        re.DOTALL,
    )
    shape_pattern = re.compile(
        r"f32\[\s*(?P<shape>[0-9,\s]+)\s*\]"
        r"\{\s*(?P<layout>[0-9,\s]+)\s*\}"
    )

    def shapes_and_layouts(
        segment: str,
    ) -> tuple[tuple[tuple[int, ...] | None, tuple[int, ...] | None], ...]:
        return tuple(
            (
                _integer_sequence(match.group("shape")),
                _integer_sequence(match.group("layout")),
            )
            for match in shape_pattern.finditer(segment)
        )

    if instruction is None:
        inputs: tuple[tuple[tuple[int, ...] | None, tuple[int, ...] | None], ...] = ()
        outputs: tuple[tuple[tuple[int, ...] | None, tuple[int, ...] | None], ...] = ()
        input_text = ""
        output_text = ""
    else:
        input_text = instruction.group("inputs")
        output_text = instruction.group("outputs")
        inputs = shapes_and_layouts(input_text)
        outputs = shapes_and_layouts(output_text)
    expected_inputs = (
        ((1, 512, 16, 128), (3, 2, 1, 0)),
        ((1, 512, 32, 128), (3, 2, 1, 0)),
        ((1, 512, 32), (2, 1, 0)),
        ((1, 512, 32), (2, 1, 0)),
    )
    expected_outputs = (
        ((1, 512, 32, 128), (3, 2, 1, 0)),
        ((1, 512, 32, 128), (3, 2, 1, 0)),
        ((1, 512, 32), (2, 1, 0)),
    )
    generic_shape = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\[")
    input_names = re.findall(r"%[A-Za-z0-9_.$-]+", input_text)
    checks = {
        "one_tuple_typed_signature": instruction is not None,
        "four_named_call_operands": len(input_names) == 4
        and len(set(input_names)) == 4,
        "four_and_only_four_input_array_types": len(generic_shape.findall(input_text))
        == 4,
        "three_and_only_three_output_array_types": len(
            generic_shape.findall(output_text)
        )
        == 3,
        "input_shapes_bound_to_exact_layouts_in_order": inputs == expected_inputs,
        "output_shapes_bound_to_exact_layouts_in_order": outputs == expected_outputs,
    }
    all_layouts = tuple(layout for _shape, layout in (*inputs, *outputs))
    return {
        "representation": "xla_minor_to_major",
        "observed_inputs": [
            {
                "shape": list(shape) if shape is not None else None,
                "layout": list(layout) if layout is not None else None,
            }
            for shape, layout in inputs
        ],
        "observed_outputs": [
            {
                "shape": list(shape) if shape is not None else None,
                "layout": list(layout) if layout is not None else None,
            }
            for shape, layout in outputs
        ],
        "expected_r4": [3, 2, 1, 0],
        "expected_r3": [2, 1, 0],
        "observed_r4_count": sum(item == (3, 2, 1, 0) for item in all_layouts),
        "observed_r3_count": sum(item == (2, 1, 0) for item in all_layouts),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _layout_proof(block: str, dialect: str) -> dict[str, Any]:
    if dialect == "stablehlo":
        return _stablehlo_layout_proof(block)
    if dialect == "optimized_hlo":
        return _optimized_hlo_layout_proof(block)
    raise ValueError("unsupported IR dialect")


def _nonempty_alias_metadata(text: str) -> list[str]:
    lowered = text.lower()
    nonempty: list[str] = []
    for token in (
        "output_operand_aliases",
        "output_to_operand_aliasing",
        "input_output_aliases",
        "input_output_alias",
    ):
        for match in re.finditer(rf"\b{re.escape(token)}\b\s*=", lowered):
            remainder = lowered[match.end() :].lstrip()
            if re.match(r"(?:\[\s*\]|\{\s*\})", remainder) is None:
                nonempty.append(token)
    if "output_operand_alias<" in lowered:
        nonempty.append("output_operand_alias")
    return sorted(set(nonempty))


def _ir_summary(text: str, dialect: str) -> dict[str, Any]:
    blocks = _custom_call_blocks(text, dialect)
    definitions = _metadata_definitions(text)
    if dialect == "stablehlo":
        textual_count = len(re.findall(r"\bstablehlo\.custom_call\b", text))
        while_count = len(re.findall(r"\bstablehlo\.while\b", text))
    elif dialect == "optimized_hlo":
        textual_count = len(re.findall(r"\bcustom-call\s*\(", text))
        while_count = len(re.findall(r"\bwhile\s*\(", text))
    else:
        raise ValueError("unsupported IR dialect")
    calls = []
    for block in blocks:
        resolved = _resolved_block_metadata(block, definitions)
        calls.append(
            {
                "targets": sorted(_custom_call_targets(resolved, dialect)),
                "nonempty_alias_metadata": _nonempty_alias_metadata(resolved),
                "layouts": _layout_proof(block, dialect),
                "sha256": hashlib.sha256(resolved.encode()).hexdigest(),
                "utf8_bytes": len(resolved.encode()),
            }
        )
    checks = {
        "parser_count_matches_textual_count": len(blocks) == textual_count,
        "exactly_one_custom_call_total": len(blocks) == 1,
        "sole_exact_target_no_lookalikes": len(calls) == 1
        and calls[0]["targets"] == [_TARGET],
        "no_while": while_count == 0,
        "no_output_alias_metadata": len(calls) == 1
        and calls[0]["nonempty_alias_metadata"] == [],
        "no_module_or_referenced_alias_metadata": _nonempty_alias_metadata(text) == [],
        "physical_row_major_layouts_exact": len(calls) == 1
        and calls[0]["layouts"]["passed"] is True,
    }
    return {
        "dialect": dialect,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_utf8_bytes": len(text.encode()),
        "raw_ir_emitted": False,
        "custom_call_count": len(blocks),
        "textual_custom_call_count": textual_count,
        "while_count": while_count,
        "calls": calls,
        "nonempty_alias_metadata": _nonempty_alias_metadata(text),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _structural_gate(*summaries: dict[str, Any]) -> dict[str, Any]:
    by_dialect = {str(summary.get("dialect")): summary for summary in summaries}
    exact = len(summaries) == 2 and set(by_dialect) == {"stablehlo", "optimized_hlo"}
    checks = {
        "exactly_two_independent_required_dialects": exact,
        "stablehlo_passed": exact and by_dialect["stablehlo"].get("passed") is True,
        "optimized_hlo_passed": exact
        and by_dialect["optimized_hlo"].get("passed") is True,
        "independent_ir_hashes_present": exact
        and all(
            re.fullmatch(r"[0-9a-f]{64}", str(item.get("text_sha256"))) is not None
            for item in by_dialect.values()
        ),
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
    combined = (
        argument + output + temporary
        if argument is not None and output is not None and temporary is not None
        else None
    )
    checks = {
        "memory_analysis_available": available,
        "argument_bytes_exact": argument == _ARGUMENT_BYTES,
        "output_bytes_exact": output == _OUTPUT_BYTES,
        "alias_bytes_zero": alias == 0,
        "temporary_bytes_at_most_64_mib": temporary is not None
        and temporary <= _MAX_TEMP_BYTES,
        "argument_output_temporary_at_most_96_mib": combined is not None
        and combined <= _MAX_TOTAL_BYTES,
    }
    return {
        "expected_argument_bytes": _ARGUMENT_BYTES,
        "expected_output_bytes": _OUTPUT_BYTES,
        "expected_alias_bytes": 0,
        "maximum_temporary_bytes": _MAX_TEMP_BYTES,
        "maximum_combined_bytes": _MAX_TOTAL_BYTES,
        "argument_output_temporary_bytes": combined,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _sealed_registration_manifest(
    registration: Any,
    *,
    library_path: Path,
    library_sha256: str,
    library_size_bytes: int,
) -> dict[str, Any]:
    seals = getattr(registration, "snapshot_seals", None)
    checks = {
        "original_canonical_path_identity_exact": getattr(
            registration, "library_path", None
        )
        == library_path,
        "approved_library_sha256_exact": getattr(registration, "library_sha256", None)
        == library_sha256,
        "loaded_snapshot_sha256_exact": getattr(registration, "snapshot_sha256", None)
        == library_sha256,
        "snapshot_size_exact": getattr(registration, "snapshot_size_bytes", None)
        == library_size_bytes,
        "snapshot_mode_exactly_0600": getattr(registration, "snapshot_mode", None)
        == 0o600,
        "required_seals_present": isinstance(seals, int)
        and not isinstance(seals, bool)
        and seals & _REQUIRED_SNAPSHOT_SEALS == _REQUIRED_SNAPSHOT_SEALS,
        "sealed_snapshot_proven": getattr(registration, "sealed_snapshot", None)
        is True,
        "snapshot_fd_retained": getattr(registration, "snapshot_fd_retained", None)
        is True,
        "target_exact": getattr(registration, "target_name", None) == _TARGET,
        "platform_exact": getattr(registration, "platform", None) == "ROCM",
        "registration_api_exact": getattr(
            registration, "registration_api_version", None
        )
        == 1,
        "custom_call_api_exact": getattr(registration, "custom_call_api_version", None)
        == 4,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "typed FFI registration did not prove the exact sealed identity"
        )
    return {
        "passed": True,
        "checks": checks,
        "library_sha256": library_sha256,
        "snapshot_sha256": library_sha256,
        "snapshot_size_bytes": library_size_bytes,
        "snapshot_mode": 0o600,
        "snapshot_seals": seals,
        "required_snapshot_seals_mask": _REQUIRED_SNAPSHOT_SEALS,
        "raw_library_path_emitted": False,
        "dlopen_source": "retained_sealed_memfd_snapshot_only",
    }


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
        "jax_version_sha256": hashlib.sha256(str(jax.__version__).encode()).hexdigest(),
        "jaxlib_version_sha256": hashlib.sha256(
            str(jaxlib.__version__).encode()
        ).hexdigest(),
        "platform_version_sha256": hashlib.sha256(
            platform_version.encode()
        ).hexdigest(),
        "raw_device_descriptions_emitted": False,
    }


def _compile_exact(
    jax: Any,
    jnp: Any,
    gdn_prepare_s512: Callable[..., Any],
    register_gdn_prepare_s512: Callable[..., Any],
    library_path: Path,
    library_sha256: str,
    library_size_bytes: int,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    output: TextIO,
) -> dict[str, Any]:
    counters["registration_attempts"] += 1
    try:
        registration = register_gdn_prepare_s512(
            library_path, library_sha256=library_sha256, enabled=True
        )
        sealed = _sealed_registration_manifest(
            registration,
            library_path=library_path,
            library_sha256=library_sha256,
            library_size_bytes=library_size_bytes,
        )
        counters["registration_completions"] += 1
        _emit(
            {
                "record_type": "ffi_registered",
                "timestamp": _utc_now(),
                "sealed_registration": sealed,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_ffi_registration_attempt", counters
        )

    signatures = (
        jax.ShapeDtypeStruct(_KEY_SHAPE, jnp.float32),
        jax.ShapeDtypeStruct(_VALUE_SHAPE, jnp.float32),
        jax.ShapeDtypeStruct(_GATE_SHAPE, jnp.float32),
        jax.ShapeDtypeStruct(_GATE_SHAPE, jnp.float32),
    )
    counters["shape_dtype_structs"] += len(signatures)

    def prepare(key: Any, value: Any, g: Any, beta: Any) -> tuple[Any, Any, Any]:
        counters["ffi_python_trace_calls"] += 1
        return gdn_prepare_s512(
            key,
            value,
            g,
            beta,
            enabled=True,
            library_path=library_path,
            library_sha256=library_sha256,
        )

    counters["lower_attempts"] += 1
    try:
        lower_start = time.perf_counter()
        lowered = jax.jit(prepare).lower(*signatures)
        lower_seconds = time.perf_counter() - lower_start
        counters["lower_completions"] += 1
        if counters["ffi_python_trace_calls"] != 1:
            raise RuntimeError("lowering did not trace exactly one GDN typed FFI call")
        stable_text = str(lowered.compiler_ir(dialect="stablehlo"))
        stablehlo = _ir_summary(stable_text, "stablehlo")
        del stable_text
        stablehlo_precompile_gate = {
            "stablehlo_structural_signature_layout_gate_passed": stablehlo["passed"],
            "passed": stablehlo["passed"],
        }
        _emit(
            {
                "record_type": "lowered",
                "timestamp": _utc_now(),
                "lower_seconds": lower_seconds,
                "stablehlo": stablehlo,
                "stablehlo_precompile_gate": stablehlo_precompile_gate,
                "sealed_registration": sealed,
                "counters": dict(counters),
            },
            output,
        )
        if not stablehlo_precompile_gate["passed"]:
            raise RuntimeError(
                "StableHLO failed the precompile custom-call, ABI, layout, loop, "
                "or alias gate"
            )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_ffi_lower_attempt", counters
        )

    counters["compile_attempts"] += 1
    compiled = None
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
        release_gate = {
            "structural_gate_passed": structural["passed"],
            "compiled_memory_gate_passed": memory_gate["passed"],
            "passed": structural["passed"] and memory_gate["passed"],
        }
        report = {
            "record_type": "ffi_compiled",
            "timestamp": _utc_now(),
            "compile_seconds": compile_seconds,
            "stablehlo": stablehlo,
            "optimized_hlo": optimized_hlo,
            "structural_gate": structural,
            "compiled_memory": memory,
            "compiled_memory_gate": memory_gate,
            "sealed_registration": sealed,
            "release_gate": release_gate,
            "lowered_callable_invocations": 0,
            "compiled_executable_invocations": 0,
            "counters": dict(counters),
        }
        _emit(report, output)
        if not release_gate["passed"]:
            raise RuntimeError("GDN compile metadata failed structural or memory gate")
    finally:
        if compiled is not None:
            del compiled
        _journal_checkpoint(
            require_clean_boot, output, "after_ffi_compile_attempt", counters
        )
    del lowered
    return report


def _run_rocm(
    args: argparse.Namespace,
    output: TextIO,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    *,
    environment: dict[str, str],
    library_manifest: dict[str, Any],
    _dependencies: tuple[Any, Any, Any, Any, Callable[..., Any], Callable[..., Any]]
    | None = None,
) -> int:
    _emit(
        {
            "record_type": "command_buffer_environment_proof",
            "timestamp": _utc_now(),
            "proof": _prove_command_buffers_disabled(environment),
            "counters": dict(counters),
        },
        output,
    )
    _journal_checkpoint(
        require_clean_boot, output, "before_backend_initialization", counters
    )
    counters["backend_initialization_attempts"] += 1
    try:
        if _dependencies is None:
            import jax
            import jax.numpy as jnp
            import jaxlib
            from jax.extend import backend as jax_backend

            from skyrl.tx.kernels.rocm.gdn_prepare_ffi import (
                gdn_prepare_s512,
                register_gdn_prepare_s512,
            )
        else:
            (
                jax,
                jnp,
                jaxlib,
                jax_backend,
                gdn_prepare_s512,
                register_gdn_prepare_s512,
            ) = _dependencies
        wrapper_path = _source_files()["gdn_prepare_wrapper_source_sha256"].resolve()
        module = sys.modules.get("skyrl.tx.kernels.rocm.gdn_prepare_ffi")
        if _dependencies is None and (
            module is None
            or not isinstance(getattr(module, "__file__", None), str)
            or Path(module.__file__).resolve() != wrapper_path
        ):
            raise RuntimeError("loaded GDN prepare wrapper is not the committed file")
        backend = _backend_manifest(jax, jaxlib, jax_backend)
        counters["backend_initialization_completions"] += 1
        _emit(
            {
                "record_type": "backend_ready",
                "timestamp": _utc_now(),
                "backend": backend,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_backend_initialization_attempt", counters
        )

    report = _compile_exact(
        jax,
        jnp,
        gdn_prepare_s512,
        register_gdn_prepare_s512,
        args.library,
        args.library_sha256,
        int(library_manifest["size_bytes"]),
        require_clean_boot,
        counters,
        output,
    )
    try:
        final_library = _assert_same_library(args.library, library_manifest)
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_library_postcheck", counters
        )
    if counters != _completed_counters():
        raise RuntimeError("compile-only counter contract was not completed exactly")
    _emit(
        {
            "record_type": "compile_only_passed",
            "timestamp": _utc_now(),
            "status": "passed_exact_s512_compile_only",
            "release_gate": report["release_gate"],
            "library": {
                key: value for key, value in final_library.items() if key != "identity"
            },
            "constructed_user_arrays": 0,
            "compiled_executable_invocations": 0,
            "gdn_numerics_validated": False,
            "runtime_launch_validated": False,
            "performance_claim_authorized": False,
            "counters": dict(counters),
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
            else "guarded_s512_compile_only",
            "contract": _exact_contract(),
            "fresh_process_required": True,
            "compile_may_dispatch_gpu_work": args.platform == "rocm",
            "compile_dispatch_caveat": _COMPILE_CAVEAT,
            "outer_profile_rocm_supervision_required": True,
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
                    "use a fresh profile_rocm.py child with --platform rocm, "
                    f"--allow-gpu, --case {_CASE}, --library, --library-sha256, and --output"
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
        library = _validate_library_path(args.library, args.library_sha256)
        _emit(
            {
                "record_type": "prerequisite_proof",
                "timestamp": _utc_now(),
                "bound_sources": bound_sources,
                "library": {
                    key: value for key, value in library.items() if key != "identity"
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
                "command_buffers": _prove_command_buffers_disabled(environment),
                "raw_environment_emitted": False,
                "counters": dict(counters),
            },
            output,
        )
        guarded_process, require_clean_boot = _load_safety_helpers()
        stage = "safety_preflight"
        with guarded_process() as raw_safety:
            _emit(
                {
                    "record_type": "safety_preflight",
                    "timestamp": _utc_now(),
                    "safety": _public_safety_preflight(raw_safety),
                    "counters": dict(counters),
                },
                output,
            )
            stage = "compile_only"
            try:
                result = _run_rocm(
                    args,
                    output,
                    require_clean_boot,
                    counters,
                    environment=environment,
                    library_manifest=library,
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
                **_redacted_message(error),
                "counters": dict(counters),
            },
            output,
        )
        return 1


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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.output is None:
        return _execute(args, sys.stdout)
    with _open_exclusive_output(args.output) as output:
        return _execute(args, output)


if __name__ == "__main__":
    sys.exit(main())
