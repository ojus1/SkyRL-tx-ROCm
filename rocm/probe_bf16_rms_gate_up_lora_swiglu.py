"""Guarded gfx1100 gate for the BF16 RMS + gate/up LoRA + SwiGLU stage.

The compile-only mode proves that the exact production forward and
forward-plus-VJP programs compile without invoking either executable.  The
execute mode is a separate, explicit qualification rung that checks numerical
agreement, determinism, and isolated-stage speed.  Neither mode authorizes
default model enablement; that still requires a separate end-to-end gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import select
import shlex
import stat
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_DISABLE_COMMAND_BUFFERS = "--xla_gpu_enable_command_buffer="
_EXACT_GEOMETRY = (1, 64, 64, 2560, 18432, 9216, 8)
_EXACT_EPSILON = 1e-6
_SUPPORTED_BLOCK_SIZES = frozenset((16, 32, 64, 128))
_MIN_WARMUPS = 3
_MIN_ITERATIONS = 11
_RELATIVE_L2_LIMIT = 0.03
_OUTPUT_COSINE_LIMIT = 0.9999
_MIN_FORWARD_VJP_SPEEDUP = 1.10
_MIN_REMATERIALIZED_STAGE_SPEEDUP = 1.15
_FORWARD_ONCE_WATCHDOG_SECONDS = 5.0
_FORWARD_ONCE_OUTPUT_SHAPE = (1, 64, 9216)
_FORWARD_ONCE_SCOPE_PATTERN = (
    r"/user\.slice/user-[0-9]+\.slice/user@[0-9]+\.service/app\.slice/"
    r"(skyrl-bf16-forward-[0-9]+-[0-9a-f]+\.scope)"
)
_EXACT_PROFILE_LIMITS = {
    "--max-junction-temp-c": 90.0,
    "--max-gpu-power-watts": 400.0,
    "--max-vram-gib": 24.0,
    "--min-host-available-gib": 0.0,
    "--max-swap-gib": 8.0,
}


def _exact_contract() -> dict[str, Any]:
    return {
        "case": "qwen35_b1_t64_bf16_rms_gate_up_lora_swiglu",
        "geometry": {
            "batch_size": 1,
            "sequence_length": 64,
            "rows": 64,
            "in_features": 2560,
            "physical_gate_up_features": 18432,
            "product_features": 9216,
            "rank": 8,
            "dtype": "bfloat16",
            "eps": _EXACT_EPSILON,
        },
        "initial_tiles": {"block_m": 16, "pair_block_n": 32, "block_k": 64},
        "target": {
            "drm_card": "card1",
            "pci_id": "1002:744c",
            "architecture": "gfx1100",
            "device_kind": "Radeon RX 7900 XTX",
        },
        "profile_limits": dict(_EXACT_PROFILE_LIMITS),
        "capture": {
            "xla_flags": _DISABLE_COMMAND_BUFFERS,
            "command_buffers_enabled": False,
            "graph_capture_enabled": False,
        },
        "compile_only": {
            "reference_executable_invocations": 0,
            "candidate_executable_invocations": 0,
        },
        "forward_once": {
            "compiled_programs": ["candidate_forward"],
            "reference_executable_invocations": 0,
            "candidate_forward_executable_invocations": 1,
            "candidate_forward_and_vjp_executable_invocations": 0,
            "watchdog_seconds": _FORWARD_ONCE_WATCHDOG_SECONDS,
            "required_output_shape": list(_FORWARD_ONCE_OUTPUT_SHAPE),
            "required_output_dtype": "bfloat16",
            "requires_exact_clean_compile_evidence": True,
            "requires_completed_compile_profile_summary": True,
            "requires_cgroup_wide_timeout_kill": True,
            "warmups": 0,
            "iterations": 0,
            "requires_finite_host_output": True,
            "performance_qualification": False,
        },
        "execute_gates": {
            "relative_l2_limit_exclusive": _RELATIVE_L2_LIMIT,
            "output_cosine_limit_inclusive": _OUTPUT_COSINE_LIMIT,
            "minimum_forward_and_vjp_speedup": _MIN_FORWARD_VJP_SPEEDUP,
            "minimum_rematerialized_stage_speedup": _MIN_REMATERIALIZED_STAGE_SPEEDUP,
            "deterministic_repeat_required": True,
        },
        "authorizes_default_model_enablement": False,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-gpu", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--compile-only", dest="mode", action="store_const", const="compile_only"
    )
    mode.add_argument(
        "--forward-once", dest="mode", action="store_const", const="forward_once"
    )
    mode.add_argument("--execute", dest="mode", action="store_const", const="execute")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-output", type=Path)
    parser.add_argument("--compile-evidence", type=Path)
    parser.add_argument("--compile-profile-summary", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--in-features", type=int, default=2560)
    parser.add_argument("--physical-features", type=int, default=18432)
    parser.add_argument("--product-features", type=int, default=9216)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--eps", type=float, default=_EXACT_EPSILON)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=_MIN_WARMUPS)
    parser.add_argument("--iterations", type=int, default=_MIN_ITERATIONS)
    args = parser.parse_args(argv)

    if not args.allow_gpu:
        parser.error("this probe requires the explicit --allow-gpu acknowledgement")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if not args.output.is_absolute():
        parser.error("--output must be an absolute path")
    if args.mode == "forward_once":
        if args.progress_output is None:
            parser.error("forward-once mode requires --progress-output")
        if not args.progress_output.is_absolute():
            parser.error("--progress-output must be an absolute path")
        if args.progress_output.exists():
            parser.error(
                "refusing to overwrite existing progress output: "
                f"{args.progress_output}"
            )
        if args.progress_output.resolve() == args.output.resolve():
            parser.error("--progress-output must differ from --output")
        if args.compile_evidence is None:
            parser.error("forward-once mode requires --compile-evidence")
        if not args.compile_evidence.is_absolute():
            parser.error("--compile-evidence must be an absolute path")
        if not args.compile_evidence.is_file():
            parser.error("--compile-evidence must be an existing regular file")
        if args.compile_profile_summary is None:
            parser.error("forward-once mode requires --compile-profile-summary")
        if not args.compile_profile_summary.is_absolute():
            parser.error("--compile-profile-summary must be an absolute path")
        if not args.compile_profile_summary.is_file():
            parser.error("--compile-profile-summary must be an existing regular file")
    elif args.progress_output is not None:
        parser.error("--progress-output is only valid with --forward-once")
    elif args.compile_evidence is not None:
        parser.error("--compile-evidence is only valid with --forward-once")
    elif args.compile_profile_summary is not None:
        parser.error("--compile-profile-summary is only valid with --forward-once")

    geometry = (
        args.batch_size,
        args.sequence_length,
        args.rows,
        args.in_features,
        args.physical_features,
        args.product_features,
        args.rank,
    )
    if geometry != _EXACT_GEOMETRY:
        parser.error(
            "the production gate requires exact B/T/M/K/physical-N/product-N/rank="
            f"{_EXACT_GEOMETRY}, got {geometry}"
        )
    if args.batch_size * args.sequence_length != args.rows:
        parser.error("--batch-size * --sequence-length must equal --rows")
    if args.physical_features != 2 * args.product_features:
        parser.error("--physical-features must equal 2 * --product-features")
    if not math.isfinite(args.eps) or args.eps != _EXACT_EPSILON:
        parser.error(f"--eps must be exactly {_EXACT_EPSILON}")

    for name, value, maximum in (
        ("block-m", args.block_m, args.rows),
        ("block-n", args.block_n, 128),
        ("block-k", args.block_k, 64),
    ):
        if value not in _SUPPORTED_BLOCK_SIZES or value > maximum:
            parser.error(f"--{name} is not a supported bounded tile")
    if args.rows % args.block_m:
        parser.error("--rows must be divisible by --block-m")
    if args.product_features % args.block_n:
        parser.error("--product-features must be divisible by --block-n")
    if args.in_features % args.block_k:
        parser.error("--in-features must be divisible by --block-k")
    if args.mode == "forward_once" and (
        args.block_m,
        args.block_n,
        args.block_k,
    ) != (16, 32, 64):
        parser.error("forward-once mode requires exact BM16/BN32/BK64 tiles")
    if args.mode == "execute" and args.warmups < _MIN_WARMUPS:
        parser.error(f"execute mode requires --warmups of at least {_MIN_WARMUPS}")
    if args.mode == "execute" and args.iterations < _MIN_ITERATIONS:
        parser.error(
            f"execute mode requires --iterations of at least {_MIN_ITERATIONS}"
        )
    if args.warmups < 0 or args.iterations < 0:
        parser.error("--warmups and --iterations cannot be negative")
    if args.mode == "forward_once" and (args.warmups, args.iterations) != (0, 0):
        parser.error("forward-once mode requires --warmups 0 --iterations 0")
    return args


def _disable_command_buffers() -> tuple[str, str]:
    original = os.environ.get("XLA_FLAGS", "")
    try:
        tokens = shlex.split(original)
    except ValueError as error:
        raise RuntimeError(f"invalid XLA_FLAGS quoting: {error}") from error
    if tokens not in ([], [_DISABLE_COMMAND_BUFFERS]):
        raise RuntimeError(
            "the exact benchmark rejects inherited XLA_FLAGS other than the "
            "command-buffer disable token"
        )
    os.environ["XLA_FLAGS"] = _DISABLE_COMMAND_BUFFERS
    return original, _DISABLE_COMMAND_BUFFERS


def _configure_environment() -> dict[str, Any]:
    expected = {
        "JAX_PLATFORMS": "rocm",
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
    }
    allowed_names = {*expected, "XLA_FLAGS"}
    accelerator_prefixes = (
        "JAX_",
        "XLA_",
        "HSA_",
        "HIP_",
        "ROCM_",
        "ROCR_",
        "AMD_",
        "GPU_",
        "CUDA_",
        "TF_XLA_",
        "TRITON_",
        "PJRT_",
        "ROCBLAS_",
        "HIPBLASLT_",
        "TENSILE_",
    )
    inherited_accelerator_environment = {
        name: value
        for name, value in os.environ.items()
        if name.startswith(accelerator_prefixes)
    }
    unexpected = sorted(set(inherited_accelerator_environment) - allowed_names)
    if unexpected:
        raise RuntimeError(
            "the exact benchmark rejects inherited accelerator environment: "
            + ", ".join(unexpected)
        )
    for name in ("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH"):
        if os.environ.get(name):
            raise RuntimeError(f"the exact benchmark rejects inherited {name}")
    for name, value in expected.items():
        inherited = os.environ.get(name)
        if inherited is not None and inherited != value:
            raise RuntimeError(
                f"{name}={inherited!r} conflicts with required {value!r}"
            )
        os.environ[name] = value
    original, effective = _disable_command_buffers()
    return {
        **expected,
        "inherited": inherited_accelerator_environment,
        "XLA_FLAGS_original": original,
        "XLA_FLAGS_effective": effective,
        "command_buffers_enabled": False,
        "graph_capture_enabled": False,
    }


def _resolve_command_path(value: str, cwd: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else cwd / path).resolve(strict=True)


def _absolute_command_path(value: str, cwd: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else cwd / path).absolute()


def _single_flag_value(command: list[str], name: str) -> str:
    positions = [index for index, value in enumerate(command) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise RuntimeError(f"profile_rocm parent must contain one {name}")
    return command[positions[0] + 1]


def _require_profile_parent(repo: Path) -> dict[str, Any]:
    parent_pid = os.getppid()
    parent_root = Path("/proc") / str(parent_pid)
    parent_cwd = (parent_root / "cwd").resolve(strict=True)
    raw_command = (parent_root / "cmdline").read_bytes()
    if not raw_command or not raw_command.endswith(b"\0"):
        raise RuntimeError("profile_rocm parent command line is unavailable")
    try:
        command = [part.decode("utf-8") for part in raw_command[:-1].split(b"\0")]
    except UnicodeDecodeError as error:
        raise RuntimeError("profile_rocm parent command line is not UTF-8") from error
    if len(command) < 4 or command.count("--") != 1:
        raise RuntimeError("profile_rocm parent command line is malformed")
    separator = command.index("--")
    profiler_command = command[:separator]
    child_command = command[separator + 1 :]
    expected_python = (repo / ".venv" / "bin" / "python").resolve(strict=True)
    expected_python_command = (repo / ".venv" / "bin" / "python").absolute()
    expected_profiler = (repo / "rocm" / "profile_rocm.py").resolve(strict=True)
    expected_probe = Path(__file__).resolve(strict=True)
    if (parent_root / "exe").resolve(strict=True) != expected_python:
        raise RuntimeError("probe parent is not the exact project Python runtime")
    if (
        _absolute_command_path(profiler_command[0], parent_cwd)
        != expected_python_command
    ):
        raise RuntimeError("profile_rocm was not launched through the project venv")
    if (
        len(profiler_command) < 2
        or _resolve_command_path(profiler_command[1], parent_cwd) != expected_profiler
    ):
        raise RuntimeError("direct parent is not the exact profile_rocm source")
    if len(child_command) < 2:
        raise RuntimeError("profile_rocm child command is incomplete")
    if (
        _absolute_command_path(child_command[0], parent_cwd) != expected_python_command
        or _resolve_command_path(child_command[1], parent_cwd) != expected_probe
    ):
        raise RuntimeError("profile_rocm does not directly supervise this exact probe")
    if "--record-command" not in profiler_command:
        raise RuntimeError("profile_rocm must record its redacted command provenance")

    observed_limits: dict[str, float] = {}
    for name, expected in _EXACT_PROFILE_LIMITS.items():
        actual = float(_single_flag_value(profiler_command, name))
        if not math.isfinite(actual) or actual != expected:
            raise RuntimeError(
                f"profile_rocm {name} must be exactly {expected}, got {actual}"
            )
        observed_limits[name] = actual
    if _single_flag_value(profiler_command, "--card") != "card1":
        raise RuntimeError("profile_rocm must monitor exact AMD DRM device card1")
    timeout = float(_single_flag_value(profiler_command, "--timeout"))
    interval = float(_single_flag_value(profiler_command, "--interval"))
    baseline_seconds = float(_single_flag_value(profiler_command, "--baseline-seconds"))
    sensor_grace_seconds = float(
        _single_flag_value(profiler_command, "--sensor-grace-seconds")
    )
    if not 0 < timeout <= 1800:
        raise RuntimeError("profile_rocm timeout must be in (0, 1800]")
    if not 0 < interval <= 0.25:
        raise RuntimeError("profile_rocm interval must be in (0, 0.25]")
    if baseline_seconds < 2:
        raise RuntimeError("profile_rocm baseline must be at least 2 seconds")
    if not 0 <= sensor_grace_seconds <= 60:
        raise RuntimeError("profile_rocm sensor grace must be in [0, 60]")

    return {
        "validated": True,
        "parent_pid": parent_pid,
        "parent_executable": str(expected_python),
        "parent_command_python": str(expected_python_command),
        "parent_command_sha256": hashlib.sha256(raw_command).hexdigest(),
        "profiler_path": str(expected_profiler),
        "profiler_sha256": hashlib.sha256(expected_profiler.read_bytes()).hexdigest(),
        "limits": observed_limits,
        "timeout_seconds": timeout,
        "interval_seconds": interval,
        "baseline_seconds": baseline_seconds,
        "sensor_grace_seconds": sensor_grace_seconds,
    }


def _require_forward_once_scope(
    parent_pid: int,
    *,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> tuple[dict[str, Any], int]:
    def exact_unified_cgroup(pid: str) -> str:
        lines = [
            line
            for line in (proc_root / pid / "cgroup").read_text().splitlines()
            if line
        ]
        if len(lines) != 1 or not lines[0].startswith("0::"):
            raise RuntimeError(f"process {pid} is not in one exact unified cgroup")
        return lines[0][3:]

    cgroup = exact_unified_cgroup("self")
    parent_cgroup = exact_unified_cgroup(str(parent_pid))
    if parent_cgroup != cgroup:
        raise RuntimeError("probe and profile_rocm parent are not in the same cgroup")
    match = re.fullmatch(_FORWARD_ONCE_SCOPE_PATTERN, cgroup)
    if match is None:
        raise RuntimeError("forward-once probe is not in a private BF16 systemd scope")
    uid = os.getuid()
    expected_prefix = f"/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/"
    if not cgroup.startswith(expected_prefix) or ".." in cgroup:
        raise RuntimeError("forward-once cgroup user identity is not exact")

    kill_path = cgroup_root / cgroup.removeprefix("/") / "cgroup.kill"
    path_info = kill_path.lstat()
    if not stat.S_ISREG(path_info.st_mode) or stat.S_ISLNK(path_info.st_mode):
        raise RuntimeError("forward-once cgroup.kill is not an exact control file")
    descriptor = os.open(
        kill_path,
        os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
    )
    try:
        descriptor_info = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_info.st_mode) or (
            descriptor_info.st_dev,
            descriptor_info.st_ino,
        ) != (path_info.st_dev, path_info.st_ino):
            raise RuntimeError("forward-once cgroup.kill descriptor identity changed")
        os.set_inheritable(descriptor, False)
        if os.get_inheritable(descriptor):
            raise RuntimeError("forward-once cgroup.kill descriptor is inheritable")
    except BaseException:
        os.close(descriptor)
        raise
    return (
        {
            "validated": True,
            "scope_unit": match.group(1),
            "cgroup": cgroup,
            "profile_parent_same_cgroup": True,
            "cgroup_kill_path": str(kill_path),
            "cgroup_kill_device": descriptor_info.st_dev,
            "cgroup_kill_inode": descriptor_info.st_ino,
            "timeout_kill_scope": "entire_systemd_cgroup",
        },
        descriptor,
    )


def _require_exact_card_identity(
    card_root: Path = Path("/sys/class/drm/card1/device"),
) -> dict[str, str]:
    resolved = card_root.resolve(strict=True)
    vendor = (card_root / "vendor").read_text().strip().lower()
    device = (card_root / "device").read_text().strip().lower()
    driver = (card_root / "driver").resolve(strict=True).name
    if vendor != "0x1002" or device != "0x744c" or driver != "amdgpu":
        raise RuntimeError(
            "exact card1 identity must be AMD 1002:744c/gfx1100 using amdgpu, got "
            f"vendor={vendor}, device={device}, driver={driver}"
        )
    return {
        "drm_card": "card1",
        "pci_vendor": vendor,
        "pci_device": device,
        "driver": driver,
        "architecture": "gfx1100",
        "pci_bdf": resolved.name,
        "sysfs_device": str(resolved),
    }


def _block_tree(tree: Any) -> Any:
    import jax

    return jax.tree.map(
        lambda value: (
            value.block_until_ready() if hasattr(value, "block_until_ready") else value
        ),
        tree,
    )


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    actual_f64 = actual.astype(np.float64)
    expected_f64 = expected.astype(np.float64)
    denominator = max(float(np.linalg.norm(expected_f64)), 1e-12)
    return float(np.linalg.norm(actual_f64 - expected_f64) / denominator)


def _cosine_similarity(actual: np.ndarray, expected: np.ndarray) -> float:
    actual_f64 = actual.astype(np.float64).reshape(-1)
    expected_f64 = expected.astype(np.float64).reshape(-1)
    denominator = float(np.linalg.norm(actual_f64) * np.linalg.norm(expected_f64))
    if denominator == 0:
        return 1.0 if np.array_equal(actual_f64, expected_f64) else 0.0
    cosine = float(np.dot(actual_f64, expected_f64) / denominator)
    return min(1.0, max(-1.0, cosine))


def _error_manifest(actual_tree: Any, expected_tree: Any) -> dict[str, Any]:
    import jax

    names = ("output", "dx", "d_lora_a", "d_lora_b")
    actual_leaves = jax.tree.leaves(actual_tree)
    expected_leaves = jax.tree.leaves(expected_tree)
    if len(actual_leaves) != len(names) or len(expected_leaves) != len(names):
        raise RuntimeError("unexpected benchmark result tree")
    errors: dict[str, Any] = {}
    for name, actual, expected in zip(
        names, actual_leaves, expected_leaves, strict=True
    ):
        actual_host = np.asarray(actual, dtype=np.float32)
        expected_host = np.asarray(expected, dtype=np.float32)
        errors[name] = {
            "relative_l2": _relative_l2(actual_host, expected_host),
            "cosine_similarity": _cosine_similarity(actual_host, expected_host),
            "max_absolute": float(np.max(np.abs(actual_host - expected_host))),
            "finite": bool(
                np.all(np.isfinite(actual_host)) and np.all(np.isfinite(expected_host))
            ),
        }
    return errors


def _numerics_gate_passed(errors: dict[str, Any]) -> bool:
    return bool(
        all(manifest["finite"] for manifest in errors.values())
        and all(
            manifest["relative_l2"] < _RELATIVE_L2_LIMIT for manifest in errors.values()
        )
        and errors["output"]["cosine_similarity"] >= _OUTPUT_COSINE_LIMIT
    )


def _performance_gate_passed(
    *,
    forward_vjp_speedup: float,
    rematerialized_speedup: float,
    candidate_seconds: list[float],
    candidate_forward_seconds: list[float],
) -> bool:
    samples = [*candidate_seconds, *candidate_forward_seconds]
    return bool(
        math.isfinite(forward_vjp_speedup)
        and math.isfinite(rematerialized_speedup)
        and forward_vjp_speedup >= _MIN_FORWARD_VJP_SPEEDUP
        and rematerialized_speedup >= _MIN_REMATERIALIZED_STAGE_SPEEDUP
        and samples
        and all(math.isfinite(sample) and sample > 0 for sample in samples)
    )


def _open_private_file_descriptor(path: Path, *, append: bool) -> int:
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise RuntimeError("private output path must be an absolute file path")
    absolute_parent = path.parent.absolute()
    resolved_parent = path.parent.resolve(strict=True)
    if resolved_parent != absolute_parent:
        raise RuntimeError("private output parent must not traverse symlinks")
    parent_info = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or stat.S_ISLNK(parent_info.st_mode)
        or parent_info.st_uid != os.getuid()
        or stat.S_IMODE(parent_info.st_mode) != 0o700
    ):
        raise RuntimeError("private output parent must be an owner-only 0700 directory")
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        descriptor_info = os.fstat(parent_fd)
        if not stat.S_ISDIR(descriptor_info.st_mode) or (
            descriptor_info.st_dev,
            descriptor_info.st_ino,
        ) != (parent_info.st_dev, parent_info.st_ino):
            raise RuntimeError("private output parent descriptor identity changed")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        if append:
            flags |= os.O_APPEND
        descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_fd)
        try:
            os.fsync(parent_fd)
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise
    finally:
        os.close(parent_fd)


def _fsync_private_parent(path: Path) -> None:
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _write_reserved_private_output(
    path: Path, descriptor: int, payload: dict[str, Any]
) -> None:
    try:
        descriptor_info = os.fstat(descriptor)
        path_info = path.lstat()
        if (
            not stat.S_ISREG(descriptor_info.st_mode)
            or stat.S_IMODE(descriptor_info.st_mode) != 0o600
            or descriptor_info.st_uid != os.getuid()
            or (descriptor_info.st_dev, descriptor_info.st_ino)
            != (path_info.st_dev, path_info.st_ino)
        ):
            raise RuntimeError("reserved private output descriptor identity changed")
    except BaseException:
        os.close(descriptor)
        raise
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(payload, allow_nan=False, sort_keys=True))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    _fsync_private_parent(path)


def _private_write(path: Path, payload: dict[str, Any]) -> None:
    descriptor = _open_private_file_descriptor(path, append=False)
    _write_reserved_private_output(path, descriptor, payload)


def _open_private_progress(path: Path):
    descriptor = _open_private_file_descriptor(path, append=True)
    return os.fdopen(descriptor, "w", encoding="utf-8")


def _durable_progress_record(output: Any, payload: dict[str, Any]) -> bytes:
    line = json.dumps(payload, allow_nan=False, sort_keys=True) + "\n"
    output.write(line)
    output.flush()
    os.fsync(output.fileno())
    return line.encode("utf-8")


_FORWARD_ONCE_WATCHDOG_PROGRAM = r"""
import os
import select
import signal
import sys
import time

probe_pid_fd = int(sys.argv[1])
completion_fd = int(sys.argv[2])
arm_fd = int(sys.argv[3])
ready_fd = int(sys.argv[4])
armed_fd = int(sys.argv[5])
cgroup_kill_fd = int(sys.argv[6])
timeout_seconds = float(sys.argv[7])
os.write(ready_fd, b"R")
os.close(ready_fd)
armed = os.read(arm_fd, 1)
os.close(arm_fd)
if armed != b"A":
    os.close(cgroup_kill_fd)
    os.close(probe_pid_fd)
    raise SystemExit(0)
armed_ns = time.monotonic_ns()
deadline_ns = armed_ns + int(timeout_seconds * 1_000_000_000)
os.write(armed_fd, f"{armed_ns}:{deadline_ns}\n".encode("ascii"))
os.close(armed_fd)
remaining = max(0.0, (deadline_ns - time.monotonic_ns()) / 1_000_000_000)
readable, _, _ = select.select([completion_fd], [], [], remaining)
completion = os.read(completion_fd, 1) if readable else b""
completed_at_ns = time.monotonic_ns()
if completion == b"C" and completed_at_ns < deadline_ns:
    os.close(completion_fd)
    os.close(cgroup_kill_fd)
    os.close(probe_pid_fd)
    raise SystemExit(0)
os.close(completion_fd)
try:
    os.write(cgroup_kill_fd, b"1")
except OSError:
    pass
finally:
    os.close(cgroup_kill_fd)
try:
    signal.pidfd_send_signal(probe_pid_fd, signal.SIGKILL)
except ProcessLookupError:
    pass
finally:
    os.close(probe_pid_fd)
"""


def _start_forward_once_watchdog(cgroup_kill_fd: int) -> dict[str, Any]:
    if isinstance(cgroup_kill_fd, bool) or cgroup_kill_fd < 3:
        raise RuntimeError("forward-once cgroup.kill descriptor is invalid")
    cgroup_kill_info = os.fstat(cgroup_kill_fd)
    if not stat.S_ISREG(cgroup_kill_info.st_mode):
        raise RuntimeError("forward-once cgroup.kill descriptor is not a control file")
    descriptors: list[int] = []
    try:
        probe_pid_fd = os.pidfd_open(os.getpid(), 0)
        descriptors.append(probe_pid_fd)
        completion_read, completion_write = os.pipe()
        descriptors.extend((completion_read, completion_write))
        arm_read, arm_write = os.pipe()
        descriptors.extend((arm_read, arm_write))
        ready_read, ready_write = os.pipe()
        descriptors.extend((ready_read, ready_write))
        armed_read, armed_write = os.pipe()
        descriptors.extend((armed_read, armed_write))
    except BaseException:
        for descriptor in descriptors:
            os.close(descriptor)
        raise
    command = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-c",
        _FORWARD_ONCE_WATCHDOG_PROGRAM,
        str(probe_pid_fd),
        str(completion_read),
        str(arm_read),
        str(ready_write),
        str(armed_write),
        str(cgroup_kill_fd),
        str(_FORWARD_ONCE_WATCHDOG_SECONDS),
    ]
    try:
        process = subprocess.Popen(
            command,
            close_fds=True,
            pass_fds=(
                probe_pid_fd,
                completion_read,
                arm_read,
                ready_write,
                armed_write,
                cgroup_kill_fd,
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except BaseException:
        for descriptor in descriptors:
            os.close(descriptor)
        raise

    os.close(probe_pid_fd)
    os.close(completion_read)
    os.close(arm_read)
    os.close(ready_write)
    os.close(armed_write)
    try:
        readable, _, _ = select.select([ready_read], [], [], 5.0)
        ready = os.read(ready_read, 1) if readable else b""
    finally:
        os.close(ready_read)
    if ready != b"R":
        os.close(armed_read)
        os.close(arm_write)
        os.close(completion_write)
        process.terminate()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)
        raise RuntimeError("the external forward-once watchdog did not become ready")
    return {
        "process": process,
        "armed_read": armed_read,
        "arm_write": arm_write,
        "completion_write": completion_write,
        "arm_sent": False,
        "armed": False,
        "settled": False,
        "timeout_seconds": _FORWARD_ONCE_WATCHDOG_SECONDS,
        "cgroup_wide_timeout_kill": True,
        "cgroup_kill_device": cgroup_kill_info.st_dev,
        "cgroup_kill_inode": cgroup_kill_info.st_ino,
        "armed_monotonic_ns": None,
        "deadline_monotonic_ns": None,
    }


def _arm_forward_once_watchdog(watchdog: dict[str, Any]) -> None:
    if watchdog["arm_sent"] or watchdog["settled"]:
        raise RuntimeError("forward-once watchdog cannot be armed in its current state")
    os.write(watchdog["arm_write"], b"A")
    watchdog["arm_sent"] = True
    os.close(watchdog["arm_write"])
    watchdog["arm_write"] = None
    try:
        readable, _, _ = select.select([watchdog["armed_read"]], [], [], 1.0)
        armed = os.read(watchdog["armed_read"], 128) if readable else b""
    finally:
        os.close(watchdog["armed_read"])
        watchdog["armed_read"] = None
    try:
        armed_text = armed.decode("ascii")
        armed_raw, deadline_raw = armed_text.removesuffix("\n").split(":", 1)
        armed_monotonic_ns = int(armed_raw)
        deadline_monotonic_ns = int(deadline_raw)
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("the external forward-once watchdog did not arm") from error
    received_monotonic_ns = time.monotonic_ns()
    expected_duration_ns = int(watchdog["timeout_seconds"] * 1_000_000_000)
    if (
        armed_monotonic_ns <= 0
        or deadline_monotonic_ns - armed_monotonic_ns != expected_duration_ns
        or not armed_monotonic_ns <= received_monotonic_ns < deadline_monotonic_ns
    ):
        raise RuntimeError("the external forward-once watchdog did not arm")
    watchdog["armed"] = True
    watchdog["armed_monotonic_ns"] = armed_monotonic_ns
    watchdog["deadline_monotonic_ns"] = deadline_monotonic_ns
    watchdog["armed_ack_received_monotonic_ns"] = received_monotonic_ns


def _settle_forward_once_watchdog(
    watchdog: dict[str, Any], *, invocation_completed: bool
) -> None:
    if watchdog["settled"]:
        return
    if watchdog["arm_write"] is not None:
        os.close(watchdog["arm_write"])
        watchdog["arm_write"] = None
    if watchdog["armed_read"] is not None:
        os.close(watchdog["armed_read"])
        watchdog["armed_read"] = None
    completion_write = watchdog["completion_write"]
    try:
        if invocation_completed:
            if not watchdog["arm_sent"]:
                raise RuntimeError("cannot complete a watchdog that was not armed")
            os.write(completion_write, b"C")
    finally:
        os.close(completion_write)
        watchdog["completion_write"] = None
    process = watchdog["process"]
    try:
        return_code = process.wait(timeout=1.0)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.wait(timeout=1.0)
        watchdog["settled"] = True
        raise RuntimeError("the external forward-once watchdog did not exit") from error
    watchdog["settled"] = True
    if return_code != 0:
        raise RuntimeError(
            f"the external forward-once watchdog exited with status {return_code}"
        )


def _run_forward_once_workload(
    *,
    executable: Any,
    arguments: tuple[Any, ...],
    progress_output: Path,
    binding: dict[str, Any],
    cgroup_kill_fd: int,
) -> dict[str, Any]:
    counts = {
        "reference_forward": 0,
        "candidate_forward": 0,
        "reference_forward_and_vjp": 0,
        "candidate_forward_and_vjp": 0,
    }
    completion_counts = dict(counts)
    progress_digest = hashlib.sha256()
    progress_byte_count = 0
    progress_record_count = 0

    def progress_record(event: str, **fields: Any) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "event": event,
            "mode": "forward_once",
            "probe_pid": os.getpid(),
            "wall_time_ns": time.time_ns(),
            "monotonic_time_ns": time.monotonic_ns(),
            "binding": binding,
            "invocation_attempt_counts": dict(counts),
            "invocation_completion_counts": dict(completion_counts),
            **fields,
        }

    def write_progress(output: Any, payload: dict[str, Any]) -> None:
        nonlocal progress_byte_count, progress_record_count
        encoded = _durable_progress_record(output, payload)
        progress_digest.update(encoded)
        progress_byte_count += len(encoded)
        progress_record_count += 1

    with _open_private_progress(progress_output) as progress:
        watchdog = _start_forward_once_watchdog(cgroup_kill_fd)
        try:
            write_progress(
                progress,
                progress_record(
                    "compiled_input_ready",
                    compiled_programs=["candidate_forward"],
                    concrete_device_inputs_ready=True,
                    watchdog_seconds=_FORWARD_ONCE_WATCHDOG_SECONDS,
                ),
            )
            _arm_forward_once_watchdog(watchdog)
            counts["candidate_forward"] += 1
            write_progress(
                progress,
                progress_record(
                    "dispatch_started",
                    watchdog_seconds=_FORWARD_ONCE_WATCHDOG_SECONDS,
                    watchdog_armed_monotonic_ns=watchdog["armed_monotonic_ns"],
                    watchdog_deadline_monotonic_ns=watchdog["deadline_monotonic_ns"],
                    watchdog_armed_ack_received_monotonic_ns=watchdog[
                        "armed_ack_received_monotonic_ns"
                    ],
                ),
            )
        except BaseException:
            _settle_forward_once_watchdog(watchdog, invocation_completed=False)
            raise

        invocation_completed = False
        try:
            device_output = _block_tree(executable(*arguments))
            device_shape = tuple(getattr(device_output, "shape", ()))
            device_dtype = str(getattr(device_output, "dtype", ""))
            if device_shape != _FORWARD_ONCE_OUTPUT_SHAPE:
                raise RuntimeError(
                    "forward-once output shape is not exact: "
                    f"{device_shape} != {_FORWARD_ONCE_OUTPUT_SHAPE}"
                )
            if device_dtype != "bfloat16":
                raise RuntimeError(
                    "forward-once output dtype is not exact: "
                    f"{device_dtype!r} != 'bfloat16'"
                )
            host_output = np.asarray(device_output, dtype=np.float32)
            if host_output.shape != _FORWARD_ONCE_OUTPUT_SHAPE:
                raise RuntimeError("forward-once host transfer changed output shape")
            invocation_completed = True
        finally:
            _settle_forward_once_watchdog(
                watchdog, invocation_completed=invocation_completed
            )

        finite = bool(np.all(np.isfinite(host_output)))
        completion_counts["candidate_forward"] += 1
        host_manifest = {
            "shape": list(host_output.shape),
            "device_dtype": device_dtype,
            "host_dtype": str(host_output.dtype),
            "element_count": int(host_output.size),
            "finite": finite,
        }
        write_progress(
            progress,
            progress_record("dispatch_completed", host_output=host_manifest),
        )
        progress_info = os.fstat(progress.fileno())
        if (
            not stat.S_ISREG(progress_info.st_mode)
            or stat.S_IMODE(progress_info.st_mode) != 0o600
            or progress_info.st_uid != os.getuid()
            or progress_info.st_size != progress_byte_count
            or progress_record_count != 3
        ):
            raise RuntimeError("forward-once progress evidence is not exact")
    _fsync_private_parent(progress_output)
    return {
        "invocation_counts": counts,
        "compile_only_zero_candidate_reference_executable_invocations": False,
        "warmup_orders": [],
        "measurement_orders": [],
        "host_output": host_manifest,
        "invocation_completion_counts": completion_counts,
        "progress": {
            "path": str(progress_output),
            "protocol": "durable_fsync_jsonl_v1",
            "record_count": progress_record_count,
            "bytes": progress_byte_count,
            "sha256": progress_digest.hexdigest(),
            "mode": "0600",
            "directory_fsynced": True,
        },
        "watchdog": {
            "external_process": True,
            "timeout_action": "cgroup.kill_then_pidfd_SIGKILL_fallback",
            "cgroup_wide_timeout_kill": watchdog["cgroup_wide_timeout_kill"],
            "timeout_seconds": _FORWARD_ONCE_WATCHDOG_SECONDS,
            "armed_monotonic_ns": watchdog["armed_monotonic_ns"],
            "deadline_monotonic_ns": watchdog["deadline_monotonic_ns"],
            "armed_ack_received_monotonic_ns": watchdog[
                "armed_ack_received_monotonic_ns"
            ],
            "dispatch_completed": True,
        },
    }


def _package_versions() -> dict[str, str | None]:
    names = ("jax", "jaxlib", "jax-rocm7-pjrt", "jax-rocm7-plugin", "numpy")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_manifest(repo: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.rstrip("\n")

    status = run("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": status.splitlines(),
        "clean": not status,
    }


def _json_exact(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _json_exact(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_exact(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _read_strict_private_json(
    path: Path, *, maximum_bytes: int = 1 << 20
) -> tuple[Any, dict[str, Any]]:
    resolved = path.resolve(strict=True)
    if resolved != path.absolute():
        raise RuntimeError("private JSON path must not traverse symlinks")
    parent_info = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or stat.S_ISLNK(parent_info.st_mode)
        or parent_info.st_uid != os.getuid()
        or stat.S_IMODE(parent_info.st_mode) != 0o700
    ):
        raise RuntimeError("private JSON parent must be an owner-only 0700 directory")
    path_info = path.lstat()
    if (
        not stat.S_ISREG(path_info.st_mode)
        or stat.S_ISLNK(path_info.st_mode)
        or path_info.st_uid != os.getuid()
        or stat.S_IMODE(path_info.st_mode) != 0o600
        or path_info.st_size > maximum_bytes
    ):
        raise RuntimeError(
            "private JSON must be a bounded owner-only 0600 regular file"
        )
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (
            path_info.st_dev,
            path_info.st_ino,
        ) or before.st_size != path_info.st_size:
            raise RuntimeError("private JSON descriptor identity changed")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(raw) > maximum_bytes
            or len(raw) != before.st_size
            or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        ):
            raise RuntimeError("private JSON changed while it was read")
    finally:
        os.close(descriptor)

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        payload = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("private JSON is not strict JSON") from error
    return payload, {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "mode": "0600",
        "device": before.st_dev,
        "inode": before.st_ino,
        "mtime_ns": before.st_mtime_ns,
    }


def _validate_compile_evidence(
    path: Path,
    *,
    expected_contract: dict[str, Any],
    expected_geometry: dict[str, Any],
    expected_source: dict[str, Any],
    expected_git: dict[str, Any],
    expected_device: dict[str, Any],
    expected_packages: dict[str, Any],
) -> dict[str, Any]:
    evidence, file_manifest = _read_strict_private_json(path)
    if not isinstance(evidence, dict):
        raise RuntimeError("compile evidence root is not an object")
    if (
        type(evidence.get("schema_version")) is not int
        or evidence["schema_version"] != 1
    ):
        raise RuntimeError("compile evidence schema is not exact")
    if evidence.get("mode") != "compile_only" or evidence.get("passed") is not True:
        raise RuntimeError("compile evidence is not a passing compile-only result")
    if not _json_exact(evidence.get("contract"), expected_contract):
        raise RuntimeError("compile evidence contract does not match this probe")
    if not _json_exact(evidence.get("geometry"), expected_geometry):
        raise RuntimeError("compile evidence geometry/tiles are not exact")
    if not _json_exact(evidence.get("device"), expected_device):
        raise RuntimeError("compile evidence device/platform is not exact")
    compilation = evidence.get("compilation")
    expected_programs = {
        "reference_forward_and_vjp",
        "candidate_forward_and_vjp",
        "reference_forward",
        "candidate_forward",
    }
    if not isinstance(compilation, dict) or set(compilation) != expected_programs:
        raise RuntimeError("compile evidence program set is not exact")
    if any(
        not isinstance(manifest, dict)
        or type(manifest.get("lower_calls")) is not int
        or manifest["lower_calls"] != 1
        or type(manifest.get("compile_calls")) is not int
        or manifest["compile_calls"] != 1
        for manifest in compilation.values()
    ):
        raise RuntimeError("compile evidence does not prove one compile per program")
    invocation = evidence.get("invocation_contract")
    if (
        not isinstance(invocation, dict)
        or invocation.get(
            "compile_only_zero_candidate_reference_executable_invocations"
        )
        is not True
    ):
        raise RuntimeError("compile evidence does not prove zero executable calls")
    if (
        type(invocation.get("total_executable_invocations")) is not int
        or invocation["total_executable_invocations"] != 0
    ):
        raise RuntimeError("compile evidence records executable invocations")
    zero_counts = {
        "reference_forward": 0,
        "candidate_forward": 0,
        "reference_forward_and_vjp": 0,
        "candidate_forward_and_vjp": 0,
    }
    if not _json_exact(
        invocation.get("per_program_executable_invocations"), zero_counts
    ) or not _json_exact(
        invocation.get("per_program_executable_completions"), zero_counts
    ):
        raise RuntimeError("compile evidence per-program zero counts are not exact")
    for name in (
        "reference_executable_invocations",
        "candidate_executable_invocations",
    ):
        if type(invocation.get(name)) is not int or invocation[name] != 0:
            raise RuntimeError("compile evidence aggregate zero counts are not exact")
    source = evidence.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("compile evidence source manifest is missing")
    for name, expected in expected_source.items():
        if not _json_exact(source.get(name), expected):
            raise RuntimeError(f"compile evidence {name} source does not match")
    if (
        not _json_exact(source.get("git"), expected_git)
        or expected_git.get("clean") is not True
    ):
        raise RuntimeError("compile evidence Git state is not this exact clean tree")
    if not _json_exact(source.get("packages"), expected_packages):
        raise RuntimeError("compile evidence package versions are not exact")
    preflight = evidence.get("preflight")
    if not isinstance(preflight, dict):
        raise RuntimeError("compile evidence preflight is missing")
    environment = preflight.get("environment")
    hardware = preflight.get("hardware")
    profiler_parent = preflight.get("profiler_parent")
    if (
        not isinstance(environment, dict)
        or environment.get("XLA_FLAGS_effective") != _DISABLE_COMMAND_BUFFERS
        or environment.get("command_buffers_enabled") is not False
        or environment.get("graph_capture_enabled") is not False
    ):
        raise RuntimeError("compile evidence did not disable capture exactly")
    if (
        not isinstance(hardware, dict)
        or hardware.get("amdgpu_boot_clean") is not True
        or hardware.get("kfd_unowned") is not True
        or hardware.get("connected_amd_connectors") != []
    ):
        raise RuntimeError("compile evidence hardware preflight is not clean")
    if (
        not isinstance(profiler_parent, dict)
        or profiler_parent.get("validated") is not True
        or not _json_exact(profiler_parent.get("limits"), _EXACT_PROFILE_LIMITS)
    ):
        raise RuntimeError("compile evidence profiler limits are not exact")
    if not _json_exact(
        evidence.get("postflight"),
        {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []},
    ):
        raise RuntimeError("compile evidence AMDGPU postflight is not clean")
    return {
        **file_manifest,
        "commit": expected_git["commit"],
        "programs": sorted(expected_programs),
        "zero_executable_invocations": True,
        "clean_preflight_and_postflight": True,
    }


def _validate_compile_profile_summary(
    path: Path,
    *,
    evidence_path: Path,
    evidence_manifest: dict[str, Any],
) -> dict[str, Any]:
    if (
        path.parent != evidence_path.parent
        or path.name != "telemetry.jsonl.summary.json"
    ):
        raise RuntimeError("compile profile summary is not the exact evidence sibling")
    summary, file_manifest = _read_strict_private_json(path)
    if not isinstance(summary, dict):
        raise RuntimeError("compile profile summary root is not an object")
    if file_manifest["mtime_ns"] < evidence_manifest["mtime_ns"]:
        raise RuntimeError("compile profile summary predates child compile evidence")
    if (
        summary.get("record_type") != "summary"
        or summary.get("status") != "completed"
        or type(summary.get("returncode")) is not int
        or summary["returncode"] != 0
        or summary.get("received_signal") is not None
        or summary.get("kernel_log_available") is not True
    ):
        raise RuntimeError("compile profile did not complete cleanly")
    for name in ("baseline_samples", "measured_samples", "samples"):
        if type(summary.get(name)) is not int or summary[name] <= 0:
            raise RuntimeError("compile profile sensor sample coverage is invalid")
    if (
        summary["samples"]
        != summary["baseline_samples"] + summary["measured_samples"] + 1
    ):
        raise RuntimeError("compile profile sample accounting is not exact")
    if summary.get("safety_violation") is not None:
        raise RuntimeError("compile profile records a safety violation")
    if summary.get("kernel_driver_errors") not in (None, []):
        raise RuntimeError("compile profile records kernel driver errors")
    metrics = summary.get("metrics")
    limits = {
        "gpu_junction_temp_c": _EXACT_PROFILE_LIMITS["--max-junction-temp-c"],
        "gpu_power_watts": _EXACT_PROFILE_LIMITS["--max-gpu-power-watts"],
        "vram_used_bytes": int(_EXACT_PROFILE_LIMITS["--max-vram-gib"] * 1024**3),
        "host_swap_used_bytes": int(_EXACT_PROFILE_LIMITS["--max-swap-gib"] * 1024**3),
    }
    observed_maxima: dict[str, float] = {}
    if not isinstance(metrics, dict):
        raise RuntimeError("compile profile metrics are missing")
    for name, limit in limits.items():
        manifest = metrics.get(name)
        maximum = manifest.get("measured_max") if isinstance(manifest, dict) else None
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, (int, float))
            or not math.isfinite(float(maximum))
            or float(maximum) < 0
            or float(maximum) > float(limit)
        ):
            raise RuntimeError(f"compile profile {name} maximum is unsafe or missing")
        observed_maxima[name] = float(maximum)
    return {
        **file_manifest,
        "status": "completed",
        "returncode": 0,
        "sensor_samples": summary["samples"],
        "observed_maxima": observed_maxima,
        "limits": limits,
        "safety_violation": None,
        "kernel_driver_errors": None,
    }


def _compile_callable(
    jax: Any, callable_: Any, arguments: tuple[Any, ...]
) -> tuple[Any, float, float]:
    lower_started = time.perf_counter()
    lowered = jax.jit(callable_).lower(*arguments)
    lower_seconds = time.perf_counter() - lower_started
    compile_started = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - compile_started
    return executable, lower_seconds, compile_seconds


def _compilation_plan(
    *,
    mode: str,
    reference_step: Any,
    candidate_step: Any,
    reference_forward: Any,
    candidate_forward: Any,
    step_arguments: tuple[Any, ...],
    forward_arguments: tuple[Any, ...],
) -> dict[str, tuple[Any, tuple[Any, ...]]]:
    if mode == "forward_once":
        return {"candidate_forward": (candidate_forward, forward_arguments)}
    return {
        "reference_forward_and_vjp": (reference_step, step_arguments),
        "candidate_forward_and_vjp": (candidate_step, step_arguments),
        "reference_forward": (reference_forward, forward_arguments),
        "candidate_forward": (candidate_forward, forward_arguments),
    }


def _alternating_order(iteration: int) -> tuple[str, str]:
    return (
        ("reference", "candidate") if iteration % 2 == 0 else ("candidate", "reference")
    )


def _run_compiled_workload(
    *,
    mode: str,
    warmups: int,
    iterations: int,
    executables: dict[str, Any],
    step_arguments: tuple[Any, ...] | None,
    forward_arguments: tuple[Any, ...] | None,
) -> dict[str, Any]:
    counts = {
        "reference_forward": 0,
        "candidate_forward": 0,
        "reference_forward_and_vjp": 0,
        "candidate_forward_and_vjp": 0,
    }
    if mode == "compile_only":
        return {
            "invocation_counts": counts,
            "compile_only_zero_candidate_reference_executable_invocations": True,
            "warmup_orders": [],
            "measurement_orders": [],
        }
    if mode != "execute":
        raise ValueError(f"unsupported workload mode: {mode}")
    if step_arguments is None or forward_arguments is None:
        raise RuntimeError("execute mode requires concrete device arguments")

    def invoke(key: str, arguments: tuple[Any, ...]) -> Any:
        counts[key] += 1
        return _block_tree(executables[key](*arguments))

    warmup_orders: list[dict[str, Any]] = []
    for iteration in range(warmups):
        step_order = _alternating_order(iteration)
        forward_order = _alternating_order(iteration + 1)
        warmup_orders.append({"forward_and_vjp": step_order, "forward": forward_order})
        for name in step_order:
            invoke(f"{name}_forward_and_vjp", step_arguments)
        for name in forward_order:
            invoke(f"{name}_forward", forward_arguments)

    def measure_pair(suffix: str, arguments: tuple[Any, ...], offset: int):
        samples = {"reference": [], "candidate": []}
        orders: list[tuple[str, str]] = []
        for iteration in range(iterations):
            order = _alternating_order(iteration + offset)
            orders.append(order)
            for name in order:
                started = time.perf_counter()
                invoke(f"{name}_{suffix}", arguments)
                samples[name].append(time.perf_counter() - started)
        return samples, orders

    step_samples, step_orders = measure_pair("forward_and_vjp", step_arguments, warmups)
    forward_samples, forward_orders = measure_pair(
        "forward", forward_arguments, warmups + 1
    )
    expected = invoke("reference_forward_and_vjp", step_arguments)
    actual = invoke("candidate_forward_and_vjp", step_arguments)
    actual_repeat = invoke("candidate_forward_and_vjp", step_arguments)
    return {
        "invocation_counts": counts,
        "compile_only_zero_candidate_reference_executable_invocations": False,
        "warmup_orders": warmup_orders,
        "measurement_orders": {
            "forward_and_vjp": step_orders,
            "forward": forward_orders,
        },
        "step_samples": step_samples,
        "forward_samples": forward_samples,
        "expected": expected,
        "actual": actual,
        "actual_repeat": actual_repeat,
    }


def _run(
    args: argparse.Namespace,
    preflight: dict[str, Any],
    *,
    forward_once_cgroup_kill_fd: int | None = None,
) -> dict[str, Any]:
    probe_path = Path(__file__).resolve(strict=True)
    repo = probe_path.parent.parent
    expected_source_path = (
        repo / "skyrl" / "tx" / "kernels" / "rocm" / "bf16_rms_gate_up_lora_swiglu.py"
    ).resolve(strict=True)
    profiler_path = (repo / "rocm" / "profile_rocm.py").resolve(strict=True)
    source_preimport = {
        "kernel": {
            "path": str(expected_source_path),
            "sha256": hashlib.sha256(expected_source_path.read_bytes()).hexdigest(),
        },
        "probe": {
            "path": str(probe_path),
            "sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        },
        "profiler": {
            "path": str(profiler_path),
            "sha256": hashlib.sha256(profiler_path.read_bytes()).hexdigest(),
        },
    }
    git_preimport = _git_manifest(repo)

    import jax
    import jax.numpy as jnp
    from jax.extend import backend as jax_backend

    kernel_module = importlib.import_module(
        "skyrl.tx.kernels.rocm.bf16_rms_gate_up_lora_swiglu"
    )
    source_path = Path(kernel_module.__file__).resolve(strict=True)
    if source_path != expected_source_path:
        raise RuntimeError(
            f"imported kernel source {source_path} is not exact repo source "
            f"{expected_source_path}"
        )
    source_preflight = {
        "kernel": {
            "path": str(source_path),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        },
        "probe": {
            "path": str(probe_path),
            "sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        },
        "profiler": {
            "path": str(profiler_path),
            "sha256": hashlib.sha256(profiler_path.read_bytes()).hexdigest(),
        },
    }
    git_preflight = _git_manifest(repo)
    if source_preflight != source_preimport or git_preflight != git_preimport:
        raise RuntimeError("source/Git state changed across JAX/kernel import")
    candidate_operation = kernel_module.bf16_rms_gate_up_lora_swiglu

    platform_version = jax_backend.get_backend().platform_version
    if jax.default_backend() != "gpu" or "rocm" not in platform_version.lower():
        raise RuntimeError(
            f"expected a ROCm JAX GPU, got {jax.default_backend()!r}: {platform_version}"
        )
    devices = jax.devices()
    if len(devices) != 1:
        raise RuntimeError(f"expected exactly one visible ROCm device, got {devices}")
    if devices[0].device_kind != "Radeon RX 7900 XTX":
        raise RuntimeError(
            "the exact gate requires Radeon RX 7900 XTX/gfx1100, got "
            f"{devices[0].device_kind!r}"
        )
    device_manifest = {
        "backend": jax.default_backend(),
        "device_kind": devices[0].device_kind,
        "architecture": "gfx1100",
        "platform_version": platform_version,
    }
    packages_preflight = _package_versions()

    geometry_manifest = {
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "rows": args.rows,
        "in_features": args.in_features,
        "physical_gate_up_features": args.physical_features,
        "product_features": args.product_features,
        "rank": args.rank,
        "dtype": "bfloat16",
        "eps": args.eps,
        "block_m": args.block_m,
        "pair_block_n": args.block_n,
        "block_k": args.block_k,
    }
    compile_evidence: dict[str, Any] | None = None
    compile_profile_summary: dict[str, Any] | None = None
    if args.mode == "forward_once":
        if git_preflight.get("clean") is not True:
            raise RuntimeError("forward-once dispatch requires an exact clean Git tree")
        if args.compile_evidence is None:
            raise RuntimeError("forward-once compile evidence is missing")
        if args.compile_profile_summary is None:
            raise RuntimeError("forward-once compile profile summary is missing")
        compile_evidence = _validate_compile_evidence(
            args.compile_evidence,
            expected_contract=_exact_contract(),
            expected_geometry=geometry_manifest,
            expected_source=source_preflight,
            expected_git=git_preflight,
            expected_device=device_manifest,
            expected_packages=packages_preflight,
        )
        compile_profile_summary = _validate_compile_profile_summary(
            args.compile_profile_summary,
            evidence_path=args.compile_evidence,
            evidence_manifest=compile_evidence,
        )

    def reference_forward(x, rms_delta, weight, a, b, scale):
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
        normalized = (
            x_f32
            * jax.lax.rsqrt(variance + args.eps)
            * (1.0 + rms_delta.astype(jnp.float32))
        ).astype(jnp.bfloat16)
        flat_normalized = normalized.reshape((-1, normalized.shape[-1]))
        z = (flat_normalized @ a).astype(jnp.bfloat16)
        base = (flat_normalized @ weight).astype(jnp.bfloat16)
        low_rank = (z @ b).astype(jnp.bfloat16)
        projection = (base + (low_rank * scale).astype(jnp.bfloat16)).astype(
            jnp.bfloat16
        )
        gate = projection[:, 0::2]
        up = projection[:, 1::2]
        product = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
        return product.reshape((*x.shape[:-1], args.product_features))

    def candidate_forward(x, rms_delta, weight, a, b, scale):
        return candidate_operation(
            x,
            rms_delta,
            weight,
            a,
            b,
            scale,
            enabled=True,
            eps=args.eps,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
        )

    def make_step(forward):
        def step(x, rms_delta, weight, a, b, scale, cotangent):
            def objective(x_arg, a_arg, b_arg):
                output = forward(x_arg, rms_delta, weight, a_arg, b_arg, scale)
                loss = jnp.sum(
                    output.astype(jnp.float32) * cotangent.astype(jnp.float32)
                )
                return loss, output

            (_, output), gradients = jax.value_and_grad(
                objective, argnums=(0, 1, 2), has_aux=True
            )(x, a, b)
            return (output, *gradients)

        return step

    reference_step = make_step(reference_forward)
    candidate_step = make_step(candidate_forward)
    signature_arguments = (
        jax.ShapeDtypeStruct(
            (args.batch_size, args.sequence_length, args.in_features), jnp.bfloat16
        ),
        jax.ShapeDtypeStruct((args.in_features,), jnp.bfloat16),
        jax.ShapeDtypeStruct((args.in_features, args.physical_features), jnp.bfloat16),
        jax.ShapeDtypeStruct((args.in_features, args.rank), jnp.bfloat16),
        jax.ShapeDtypeStruct((args.rank, args.physical_features), jnp.bfloat16),
        jax.ShapeDtypeStruct((), jnp.bfloat16),
        jax.ShapeDtypeStruct(
            (args.batch_size, args.sequence_length, args.product_features), jnp.bfloat16
        ),
    )

    concrete_arguments: tuple[Any, ...] | None = None
    concrete_forward_arguments: tuple[Any, ...] | None = None
    if args.mode in ("forward_once", "execute"):
        rng = np.random.default_rng(20260714)

        def device_bf16(shape: tuple[int, ...], scale: float = 1.0):
            host = rng.standard_normal(shape, dtype=np.float32) * scale
            return jax.device_put(jnp.asarray(host, dtype=jnp.bfloat16), devices[0])

        concrete_forward_arguments = (
            device_bf16(
                (args.batch_size, args.sequence_length, args.in_features), 0.02
            ),
            device_bf16((args.in_features,), 0.01),
            device_bf16(
                (args.in_features, args.physical_features),
                1.0 / math.sqrt(args.in_features),
            ),
            device_bf16((args.in_features, args.rank), 0.01),
            device_bf16((args.rank, args.physical_features), 0.01),
            jax.device_put(jnp.asarray(2.0, dtype=jnp.bfloat16), devices[0]),
        )
        if args.mode == "execute":
            concrete_arguments = (
                *concrete_forward_arguments,
                device_bf16(
                    (args.batch_size, args.sequence_length, args.product_features),
                    0.02,
                ),
            )

    if args.mode == "forward_once" and concrete_forward_arguments is None:
        raise RuntimeError("forward-once mode requires concrete device arguments")
    if args.mode == "forward_once":
        compile_arguments = signature_arguments
        compile_forward_arguments = concrete_forward_arguments
    else:
        compile_arguments = (
            concrete_arguments
            if concrete_arguments is not None
            else signature_arguments
        )
        compile_forward_arguments = (
            concrete_forward_arguments
            if concrete_forward_arguments is not None
            else compile_arguments[:-1]
        )
    callables = _compilation_plan(
        mode=args.mode,
        reference_step=reference_step,
        candidate_step=candidate_step,
        reference_forward=reference_forward,
        candidate_forward=candidate_forward,
        step_arguments=compile_arguments,
        forward_arguments=compile_forward_arguments,
    )
    executables: dict[str, Any] = {}
    compilation: dict[str, Any] = {}
    for name, (callable_, callable_arguments) in callables.items():
        executable, lower_seconds, compile_seconds = _compile_callable(
            jax, callable_, callable_arguments
        )
        executables[name] = executable
        compilation[name] = {
            "lower_calls": 1,
            "compile_calls": 1,
            "lower_seconds": lower_seconds,
            "compile_seconds": compile_seconds,
        }

    forward_once_binding = {
        "contract": _exact_contract(),
        "geometry": geometry_manifest,
        "preflight": preflight,
        "source": source_preflight,
        "git": git_preflight,
        "compile_evidence": compile_evidence,
        "compile_profile_summary": compile_profile_summary,
    }

    if args.mode == "forward_once":
        if (
            args.progress_output is None
            or concrete_forward_arguments is None
            or forward_once_cgroup_kill_fd is None
        ):
            raise RuntimeError("forward-once mode is missing its required arguments")
        workload = _run_forward_once_workload(
            executable=executables["candidate_forward"],
            arguments=concrete_forward_arguments,
            progress_output=args.progress_output,
            binding=forward_once_binding,
            cgroup_kill_fd=forward_once_cgroup_kill_fd,
        )
    else:
        workload = _run_compiled_workload(
            mode=args.mode,
            warmups=args.warmups,
            iterations=args.iterations,
            executables=executables,
            step_arguments=concrete_arguments,
            forward_arguments=concrete_forward_arguments,
        )
    source_postflight = {
        "kernel": {
            "path": str(source_path),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        },
        "probe": {
            "path": str(probe_path),
            "sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        },
        "profiler": {
            "path": str(profiler_path),
            "sha256": hashlib.sha256(profiler_path.read_bytes()).hexdigest(),
        },
    }
    git_postflight = _git_manifest(repo)
    if args.mode == "forward_once" and (
        source_postflight != source_preflight or git_postflight != git_preflight
    ):
        raise RuntimeError("forward-once source/Git state changed across dispatch")
    numerics: dict[str, Any]
    determinism: dict[str, Any]
    measurement: dict[str, Any]
    performance_gate: dict[str, Any]
    recommendation = False
    if args.mode == "compile_only":
        numerics = {"executed": False, "passed": None}
        determinism = {"executed": False, "passed": None}
        measurement = {
            "executed": False,
            "warmups": 0,
            "iterations": 0,
            "requested_warmups_ignored": args.warmups,
            "requested_iterations_ignored": args.iterations,
        }
        performance_gate = {
            "executed": False,
            "passed": None,
            "minimum_forward_and_vjp_speedup": _MIN_FORWARD_VJP_SPEEDUP,
            "minimum_rematerialized_stage_speedup": _MIN_REMATERIALIZED_STAGE_SPEEDUP,
        }
        passed = all(
            manifest["lower_calls"] == 1 and manifest["compile_calls"] == 1
            for manifest in compilation.values()
        ) and not any(workload["invocation_counts"].values())
    elif args.mode == "forward_once":
        finite_host_output = workload["host_output"]["finite"]
        numerics = {
            "executed": True,
            "reference_compared": False,
            "passed": None,
            "finite_host_output": finite_host_output,
            "host_output": workload["host_output"],
        }
        determinism = {"executed": False, "passed": None}
        measurement = {
            "executed": False,
            "performance_measured": False,
            "warmups": 0,
            "iterations": 0,
            "single_candidate_forward_safety_dispatch": True,
        }
        performance_gate = {
            "executed": False,
            "passed": None,
            "reason": "forward-once is not a performance qualification",
        }
        passed = bool(
            finite_host_output
            and workload["host_output"]
            == {
                "shape": list(_FORWARD_ONCE_OUTPUT_SHAPE),
                "device_dtype": "bfloat16",
                "host_dtype": "float32",
                "element_count": math.prod(_FORWARD_ONCE_OUTPUT_SHAPE),
                "finite": True,
            }
            and set(compilation) == {"candidate_forward"}
            and compilation["candidate_forward"]["lower_calls"] == 1
            and compilation["candidate_forward"]["compile_calls"] == 1
            and workload["invocation_counts"]
            == {
                "reference_forward": 0,
                "candidate_forward": 1,
                "reference_forward_and_vjp": 0,
                "candidate_forward_and_vjp": 0,
            }
            and workload["invocation_completion_counts"]
            == workload["invocation_counts"]
            and workload["progress"]["record_count"] == 3
            and workload["watchdog"]["dispatch_completed"] is True
            and workload["watchdog"]["cgroup_wide_timeout_kill"] is True
            and compile_evidence is not None
            and compile_profile_summary is not None
            and preflight.get("forward_once_scope", {}).get("validated") is True
        )
    else:
        errors = _error_manifest(workload["actual"], workload["expected"])
        numerics_passed = _numerics_gate_passed(errors)
        actual_leaves = jax.tree.leaves(workload["actual"])
        repeat_leaves = jax.tree.leaves(workload["actual_repeat"])
        deterministic = all(
            np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(actual_leaves, repeat_leaves, strict=True)
        )
        step_samples = workload["step_samples"]
        forward_samples = workload["forward_samples"]
        reference_step_median = statistics.median(step_samples["reference"])
        candidate_step_median = statistics.median(step_samples["candidate"])
        reference_forward_median = statistics.median(forward_samples["reference"])
        candidate_forward_median = statistics.median(forward_samples["candidate"])
        forward_vjp_speedup = reference_step_median / candidate_step_median
        forward_speedup = reference_forward_median / candidate_forward_median
        rematerialized_reference = reference_step_median + reference_forward_median
        rematerialized_candidate = candidate_step_median + candidate_forward_median
        rematerialized_speedup = rematerialized_reference / rematerialized_candidate
        performance_passed = _performance_gate_passed(
            forward_vjp_speedup=forward_vjp_speedup,
            rematerialized_speedup=rematerialized_speedup,
            candidate_seconds=step_samples["candidate"],
            candidate_forward_seconds=forward_samples["candidate"],
        )
        numerics = {
            "executed": True,
            "passed": numerics_passed,
            "relative_l2_limit_exclusive": _RELATIVE_L2_LIMIT,
            "output_cosine_limit_inclusive": _OUTPUT_COSINE_LIMIT,
            "errors": errors,
        }
        determinism = {
            "executed": True,
            "passed": deterministic,
            "repeated_candidate_result_tree_bitwise_equal": deterministic,
        }
        measurement = {
            "executed": True,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "alternating_warmup_orders": workload["warmup_orders"],
            "alternating_measurement_orders": workload["measurement_orders"],
            "forward_and_vjp": {
                "reference_seconds": step_samples["reference"],
                "candidate_seconds": step_samples["candidate"],
                "reference_median_seconds": reference_step_median,
                "candidate_median_seconds": candidate_step_median,
                "speedup": forward_vjp_speedup,
            },
            "forward": {
                "reference_seconds": forward_samples["reference"],
                "candidate_seconds": forward_samples["candidate"],
                "reference_median_seconds": reference_forward_median,
                "candidate_median_seconds": candidate_forward_median,
                "speedup": forward_speedup,
            },
            "rematerialized_stage_estimate": {
                "reference_seconds": rematerialized_reference,
                "candidate_seconds": rematerialized_candidate,
                "speedup": rematerialized_speedup,
            },
        }
        performance_gate = {
            "executed": True,
            "passed": performance_passed,
            "minimum_forward_and_vjp_speedup": _MIN_FORWARD_VJP_SPEEDUP,
            "minimum_rematerialized_stage_speedup": _MIN_REMATERIALIZED_STAGE_SPEEDUP,
        }
        recommendation = bool(numerics_passed and deterministic and performance_passed)
        passed = recommendation

    invocation_counts = workload["invocation_counts"]
    reference_invocations = (
        invocation_counts["reference_forward"]
        + invocation_counts["reference_forward_and_vjp"]
    )
    candidate_invocations = (
        invocation_counts["candidate_forward"]
        + invocation_counts["candidate_forward_and_vjp"]
    )
    payload = {
        "schema_version": 1,
        "mode": args.mode,
        "qualification_scope": "isolated_stage_only",
        "authorizes_default_model_enablement": False,
        "recommend_for_opt_in_model_integration": recommendation,
        "passed": bool(passed),
        "contract": _exact_contract(),
        "preflight": preflight,
        "device": device_manifest,
        "geometry": geometry_manifest,
        "invocation_contract": {
            "per_program_executable_invocations": invocation_counts,
            "per_program_executable_completions": workload.get(
                "invocation_completion_counts", invocation_counts
            ),
            "reference_executable_invocations": reference_invocations,
            "candidate_executable_invocations": candidate_invocations,
            "total_executable_invocations": reference_invocations
            + candidate_invocations,
            "compile_only_zero_candidate_reference_executable_invocations": workload[
                "compile_only_zero_candidate_reference_executable_invocations"
            ],
        },
        "compilation": compilation,
        "measurement": measurement,
        "performance_gate": performance_gate,
        "numerics": numerics,
        "determinism": determinism,
        "source": {
            **source_postflight,
            "git": git_postflight,
            "packages": packages_preflight,
        },
    }
    if args.mode == "forward_once":
        payload["forward_once"] = {
            "compile_evidence": compile_evidence,
            "compile_profile_summary": compile_profile_summary,
            "progress": workload["progress"],
            "watchdog": workload["watchdog"],
            "invocation_attempt_counts": workload["invocation_counts"],
            "invocation_completion_counts": workload["invocation_completion_counts"],
            "source_and_git_unchanged_across_dispatch": True,
        }
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    preloaded = sorted(
        name
        for name in sys.modules
        if name == "jax"
        or name.startswith("jax.")
        or name == "jaxlib"
        or name.startswith("jaxlib.")
        or name == "flax"
        or name.startswith("flax.")
        or name.startswith("skyrl.tx.kernels")
    )
    if preloaded:
        raise RuntimeError(
            "the exact ROCm benchmark requires a fresh process before JAX/kernel "
            f"import; already loaded: {preloaded[:8]}"
        )
    repo = Path(__file__).resolve(strict=True).parent.parent
    expected_prefix = (repo / ".venv").resolve(strict=True)
    if Path(sys.prefix).resolve(strict=True) != expected_prefix:
        raise RuntimeError(
            f"exact benchmark requires project venv {expected_prefix}, got {sys.prefix}"
        )
    expected_executable = (repo / ".venv" / "bin" / "python").absolute()
    if Path(sys.executable).absolute() != expected_executable:
        raise RuntimeError(
            f"exact benchmark requires {expected_executable}, got {sys.executable}"
        )
    output_descriptor: int | None = _open_private_file_descriptor(
        args.output, append=False
    )
    try:
        profiler_parent = _require_profile_parent(repo)
        forward_once_scope: dict[str, Any] | None = None
        forward_once_cgroup_kill_fd: int | None = None
        try:
            if args.mode == "forward_once":
                forward_once_scope, forward_once_cgroup_kill_fd = (
                    _require_forward_once_scope(profiler_parent["parent_pid"])
                )
            card_identity = _require_exact_card_identity()
            environment = _configure_environment()
            try:
                safety_module = importlib.import_module("rocm.amdgpu_safety")
            except ModuleNotFoundError:
                safety_module = importlib.import_module("amdgpu_safety")
            safety_path = Path(safety_module.__file__).resolve(strict=True)
            expected_safety_path = (repo / "rocm" / "amdgpu_safety.py").resolve(
                strict=True
            )
            if safety_path != expected_safety_path:
                raise RuntimeError(
                    f"imported safety source {safety_path} is not exact repo source "
                    f"{expected_safety_path}"
                )

            with safety_module.guarded_qwen35_rocm_process() as hardware_preflight:
                preflight = {
                    "environment": environment,
                    "hardware": hardware_preflight,
                    "profiler_parent": profiler_parent,
                    "card_identity": card_identity,
                    "safety_source": {
                        "path": str(safety_path),
                        "sha256": hashlib.sha256(safety_path.read_bytes()).hexdigest(),
                    },
                }
                if forward_once_scope is not None:
                    preflight["forward_once_scope"] = forward_once_scope
                try:
                    payload = _run(
                        args,
                        preflight,
                        forward_once_cgroup_kill_fd=forward_once_cgroup_kill_fd,
                    )
                finally:
                    postflight = safety_module.require_clean_amdgpu_boot()
                payload["postflight"] = postflight
        finally:
            if forward_once_cgroup_kill_fd is not None:
                os.close(forward_once_cgroup_kill_fd)
        descriptor = output_descriptor
        output_descriptor = None
        _write_reserved_private_output(args.output, descriptor, payload)
    finally:
        if output_descriptor is not None:
            os.close(output_descriptor)
    print(json.dumps(payload, allow_nan=False, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
