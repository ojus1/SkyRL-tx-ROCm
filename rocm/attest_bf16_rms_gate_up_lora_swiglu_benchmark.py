"""CPU-only attestation for guarded BF16 RMS/gate-up benchmark runs.

The GPU child cannot attest its supervising profiler because the profiler's
terminal sample and summary are written only after that child exits.  This
module verifies the child result together with its durable progress stream and
the completed profiler telemetry, then emits the only ``passed: true``
composite artifact for the benchmark rung.

Importing this module does not import JAX or open an accelerator device.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

_ATTESTATION_TYPE = "bf16_rms_gate_up_lora_swiglu_contiguous_v1_benchmark_profile_attestation"
_DISABLE_COMMAND_BUFFERS = "--xla_gpu_enable_command_buffer="
_EVIDENCE_SCHEMA_VERSION = 2
_WATCHDOG_SECONDS = 5.0
_WATCHDOG_NS = 5_000_000_000
_MIN_FORWARD_VJP_SPEEDUP = 1.10
_MIN_REMATERIALIZED_STAGE_SPEEDUP = 1.15
_RELATIVE_L2_LIMIT = 0.03
_OUTPUT_COSINE_LIMIT = 0.9999
_PROGRAM_ORDER = (
    "reference_forward",
    "candidate_forward",
    "reference_forward_and_vjp",
    "candidate_forward_and_vjp",
)
_INPUT_SPECS = (
    ("x", (1, 64, 2560)),
    ("rms_delta", (2560,)),
    ("weight", (2560, 18432)),
    ("lora_a", (2560, 8)),
    ("lora_b", (8, 18432)),
    ("scale", ()),
    ("cotangent", (1, 64, 9216)),
)
_RESULT_LEAF_COUNTS = {
    "reference_forward": 1,
    "candidate_forward": 1,
    "reference_forward_and_vjp": 4,
    "candidate_forward_and_vjp": 4,
}
_RESULT_SPECS = {
    "reference_forward": (("output", (1, 64, 9216)),),
    "candidate_forward": (("output", (1, 64, 9216)),),
    "reference_forward_and_vjp": (
        ("output", (1, 64, 9216)),
        ("dx", (1, 64, 2560)),
        ("d_lora_a", (2560, 8)),
        ("d_lora_b", (8, 18432)),
    ),
    "candidate_forward_and_vjp": (
        ("output", (1, 64, 9216)),
        ("dx", (1, 64, 2560)),
        ("d_lora_a", (2560, 8)),
        ("d_lora_b", (8, 18432)),
    ),
}
_CANDIDATE_HASH_POSITIONS = (
    ("candidate_forward", "output"),
    ("candidate_forward_and_vjp", "output"),
    ("candidate_forward_and_vjp", "dx"),
    ("candidate_forward_and_vjp", "d_lora_a"),
    ("candidate_forward_and_vjp", "d_lora_b"),
)
_BENCHMARK_WARMUP_ORDERS = (
    (
        "reference_forward",
        "candidate_forward",
        "candidate_forward_and_vjp",
        "reference_forward_and_vjp",
    ),
    (
        "reference_forward_and_vjp",
        "candidate_forward_and_vjp",
        "candidate_forward",
        "reference_forward",
    ),
)
_BENCHMARK_MEASUREMENT_ROTATION = (
    (
        "reference_forward",
        "candidate_forward",
        "candidate_forward_and_vjp",
        "reference_forward_and_vjp",
    ),
    (
        "candidate_forward_and_vjp",
        "reference_forward_and_vjp",
        "reference_forward",
        "candidate_forward",
    ),
    (
        "candidate_forward",
        "reference_forward",
        "reference_forward_and_vjp",
        "candidate_forward_and_vjp",
    ),
    (
        "reference_forward_and_vjp",
        "candidate_forward_and_vjp",
        "candidate_forward",
        "reference_forward",
    ),
)
_BENCHMARK_MODE_COUNTS = {
    "benchmark_smoke": (2, 4),
    "benchmark": (8, 32),
}
_EXACT_PROFILE_LIMITS = {
    "--max-junction-temp-c": 90.0,
    "--max-gpu-power-watts": 400.0,
    "--max-vram-gib": 24.0,
    "--min-host-available-gib": 0.0,
    "--max-swap-gib": 8.0,
}
_SAFETY_METRIC_LIMITS = {
    "gpu_junction_temp_c": 90.0,
    "gpu_power_watts": 400.0,
    "vram_used_bytes": 24 * 1024**3,
    "host_swap_used_bytes": 8 * 1024**3,
}
_PACKAGE_NAMES = (
    "jax",
    "jaxlib",
    "jax-rocm7-pjrt",
    "jax-rocm7-plugin",
    "ml_dtypes",
    "numpy",
)
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_HEX_40 = re.compile(r"[0-9a-f]{40}")


def _expected_program_structures() -> dict[str, Any]:
    stage_one = {
        "name": "skyrl_qwen35_bf16_rms_materialize_lora_a_forward",
        "grid": [4],
        "outputs": [
            {"shape": [64, 2560], "dtype": "bfloat16"},
            {"shape": [64, 8], "dtype": "bfloat16"},
            {"shape": [64], "dtype": "float32"},
            {"shape": [64], "dtype": "float32"},
        ],
    }
    return {
        "reference_forward": {"dot_general_count": 3, "pallas_calls": []},
        "candidate_forward": {
            "dot_general_count": 1,
            "pallas_calls": [
                stage_one,
                {
                    "name": "skyrl_qwen35_bf16_contiguous_gate_up_lora_swiglu_forward",
                    "grid": [4, 288],
                    "outputs": [
                        {"shape": [64, 9216], "dtype": "bfloat16"},
                    ],
                },
            ],
        },
        "reference_forward_and_vjp": {
            "dot_general_count": 8,
            "pallas_calls": [],
        },
        "candidate_forward_and_vjp": {
            "dot_general_count": 6,
            "pallas_calls": [
                stage_one,
                {
                    "name": ("skyrl_qwen35_bf16_contiguous_gate_up_lora_swiglu_" "residual_forward"),
                    "grid": [4, 288],
                    "outputs": [
                        {"shape": [64, 9216], "dtype": "bfloat16"},
                        {"shape": [64, 18432], "dtype": "bfloat16"},
                    ],
                },
            ],
        },
    }


def _expected_contract() -> dict[str, Any]:
    benchmark_common = {
        "compiled_programs": list(_PROGRAM_ORDER),
        "programs_per_supercycle": 4,
        "watchdog_seconds_per_operation": _WATCHDOG_SECONDS,
        "guarded_device_input_setup": True,
        "guarded_device_input_teardown": True,
        "requires_prior_numerics_evidence": True,
        "raw_samples_only": True,
        "performance_qualification": False,
    }
    return {
        "case": "qwen35_b1_t64_bf16_rms_gate_up_lora_swiglu_contiguous_v1",
        "geometry": {
            "batch_size": 1,
            "sequence_length": 64,
            "rows": 64,
            "in_features": 2560,
            "physical_gate_up_features": 18432,
            "product_features": 9216,
            "rank": 8,
            "dtype": "bfloat16",
            "eps": 1e-6,
        },
        "initial_tiles": {
            "block_m": 16,
            "block_physical_n": 64,
            "block_k": 32,
        },
        "program_structures": _expected_program_structures(),
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
            "watchdog_seconds": _WATCHDOG_SECONDS,
            "required_output_shape": [1, 64, 9216],
            "required_output_dtype": "bfloat16",
            "requires_exact_clean_compile_evidence": True,
            "requires_completed_compile_profile_summary": True,
            "requires_cgroup_wide_timeout_kill": True,
            "warmups": 0,
            "iterations": 0,
            "requires_finite_host_output": True,
            "performance_qualification": False,
        },
        "numerics_once": {
            "compiled_programs": list(_PROGRAM_ORDER),
            "program_order": list(_PROGRAM_ORDER),
            "executable_invocations_per_program": 1,
            "watchdog_seconds_per_program": _WATCHDOG_SECONDS,
            "requires_host_bfloat16_inputs": True,
            "requires_readonly_host_inputs": True,
            "requires_post_dispatch_input_rehash": True,
            "requires_exact_clean_compile_evidence": True,
            "requires_completed_compile_profile_summary": True,
            "requires_cgroup_wide_timeout_kill": True,
            "relative_l2_limit_exclusive": _RELATIVE_L2_LIMIT,
            "output_cosine_limit_inclusive": _OUTPUT_COSINE_LIMIT,
            "gradient_cosine_similarity_report_only": True,
            "warmups": 0,
            "iterations": 0,
            "performance_qualification": False,
        },
        "benchmark_smoke": {
            **benchmark_common,
            "warmup_supercycles": 2,
            "measured_supercycles": 4,
            "total_program_invocations": 24,
            "executable_invocations_per_program": 6,
            "independent_dispatch_watchdogs": 24,
            "independent_operation_watchdogs": 26,
            "requires_prior_smoke_profile_attestation": False,
        },
        "benchmark": {
            **benchmark_common,
            "warmup_supercycles": 8,
            "measured_supercycles": 32,
            "total_program_invocations": 160,
            "executable_invocations_per_program": 40,
            "independent_dispatch_watchdogs": 160,
            "independent_operation_watchdogs": 162,
            "requires_prior_smoke_profile_attestation": True,
        },
        "execute_gates": {
            "relative_l2_limit_exclusive": _RELATIVE_L2_LIMIT,
            "output_cosine_limit_inclusive": _OUTPUT_COSINE_LIMIT,
            "gradient_cosine_similarity_report_only": True,
            "minimum_forward_and_vjp_speedup": _MIN_FORWARD_VJP_SPEEDUP,
            "minimum_rematerialized_stage_speedup": (_MIN_REMATERIALIZED_STAGE_SPEEDUP),
            "deterministic_repeat_required": True,
        },
        "authorizes_default_model_enablement": False,
    }


def _exact_geometry() -> dict[str, Any]:
    return {
        **_expected_contract()["geometry"],
        "block_m": 16,
        "block_physical_n": 64,
        "block_k": 32,
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
            _json_exact(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _strict_json_loads(raw: bytes) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(
        raw,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )


def _read_private_bytes(path: Path, *, maximum_bytes: int) -> tuple[bytes, dict[str, Any]]:
    if not path.is_absolute():
        raise RuntimeError("attestation input paths must be absolute")
    resolved = path.resolve(strict=True)
    if resolved != path.absolute():
        raise RuntimeError("attestation input path must not traverse symlinks")
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or stat.S_ISLNK(parent.st_mode)
        or parent.st_uid != os.getuid()
        or stat.S_IMODE(parent.st_mode) != 0o700
    ):
        raise RuntimeError("attestation input parent must be owner-only mode 0700")
    path_info = path.lstat()
    if (
        not stat.S_ISREG(path_info.st_mode)
        or stat.S_ISLNK(path_info.st_mode)
        or path_info.st_uid != os.getuid()
        or stat.S_IMODE(path_info.st_mode) != 0o600
        or path_info.st_size > maximum_bytes
    ):
        raise RuntimeError("attestation input must be bounded owner-only mode 0600")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ) != (path_info.st_dev, path_info.st_ino, path_info.st_size):
            raise RuntimeError("attestation input descriptor identity changed")
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
            raise RuntimeError("attestation input changed while it was read")
    finally:
        os.close(descriptor)
    return raw, {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "mode": "0600",
        "device": before.st_dev,
        "inode": before.st_ino,
        "mtime_ns": before.st_mtime_ns,
    }


def _read_private_json(path: Path, *, maximum_bytes: int = 16 << 20) -> tuple[Any, dict[str, Any]]:
    raw, manifest = _read_private_bytes(path, maximum_bytes=maximum_bytes)
    try:
        payload = _strict_json_loads(raw)
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("attestation input is not strict JSON") from error
    return payload, manifest


def _open_private_output(path: Path) -> int:
    if not path.is_absolute() or path.exists():
        raise RuntimeError("attestation output must be a fresh absolute path")
    resolved_parent = path.parent.resolve(strict=True)
    if resolved_parent != path.parent.absolute():
        raise RuntimeError("attestation output parent must not traverse symlinks")
    parent_info = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or stat.S_ISLNK(parent_info.st_mode)
        or parent_info.st_uid != os.getuid()
        or stat.S_IMODE(parent_info.st_mode) != 0o700
    ):
        raise RuntimeError("attestation output parent must be owner-only mode 0700")
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return descriptor


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor = _open_private_output(path)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(payload, allow_nan=False, sort_keys=True))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _require_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise RuntimeError(f"{label} is not an exact integer")
    return value


def _require_number(
    value: Any,
    *,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric")
    result = float(value)
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise RuntimeError(f"{label} is outside its exact finite bounds")
    return result


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise RuntimeError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _current_git_manifest(repo: Path) -> dict[str, Any]:
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


def _current_package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in _PACKAGE_NAMES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _current_rocm_version() -> str:
    try:
        return Path("/opt/rocm/.info/version").read_text().strip()
    except OSError as error:
        raise RuntimeError("determinism ROCm runtime version is unavailable") from error


def _benchmark_schedule(mode: str) -> tuple[dict[str, Any], ...]:
    if mode not in _BENCHMARK_MODE_COUNTS:
        raise RuntimeError("unsupported benchmark attestation mode")
    warmups, measurements = _BENCHMARK_MODE_COUNTS[mode]
    schedule: list[dict[str, Any]] = []
    for index in range(warmups):
        schedule.append(
            {
                "phase": "warmup",
                "phase_supercycle": index,
                "global_supercycle": index,
                "order": _BENCHMARK_WARMUP_ORDERS[index % len(_BENCHMARK_WARMUP_ORDERS)],
            }
        )
    for index in range(measurements):
        schedule.append(
            {
                "phase": "measurement",
                "phase_supercycle": index,
                "global_supercycle": warmups + index,
                "order": _BENCHMARK_MEASUREMENT_ROTATION[index % len(_BENCHMARK_MEASUREMENT_ROTATION)],
            }
        )
    return tuple(schedule)


def _validate_source_and_runtime(source: Any, *, repo: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not isinstance(source, dict) or set(source) != {
        "kernel",
        "probe",
        "profiler",
        "git",
        "packages",
    }:
        raise RuntimeError("benchmark source/runtime manifest is not exact")
    expected_paths = {
        "kernel": repo / "skyrl" / "tx" / "kernels" / "rocm" / "bf16_rms_gate_up_lora_swiglu_contiguous.py",
        "probe": repo / "rocm" / "probe_bf16_rms_gate_up_lora_swiglu.py",
        "profiler": repo / "rocm" / "profile_rocm.py",
    }
    source_only: dict[str, Any] = {}
    for name, expected_path in expected_paths.items():
        resolved = expected_path.resolve(strict=True)
        expected = {
            "path": str(resolved),
            "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        }
        if not _json_exact(source.get(name), expected):
            raise RuntimeError(f"benchmark {name} source is not this exact source")
        source_only[name] = expected
    git = source.get("git")
    if (
        not isinstance(git, dict)
        or set(git) != {"commit", "branch", "status_porcelain", "clean"}
        or not isinstance(git.get("commit"), str)
        or _HEX_40.fullmatch(git["commit"]) is None
        or not isinstance(git.get("branch"), str)
        or not git["branch"]
        or git.get("status_porcelain") != []
        or git.get("clean") is not True
        or not _json_exact(git, _current_git_manifest(repo))
    ):
        raise RuntimeError("benchmark is not bound to this exact clean Git tree")
    packages = source.get("packages")
    if not _json_exact(packages, _current_package_versions()):
        raise RuntimeError("benchmark package-version binding is not exact")
    return source_only, git, packages


def _validate_embedded_file_manifest(manifest: Any, *, label: str, maximum_bytes: int) -> tuple[bytes, dict[str, Any]]:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("path"), str):
        raise RuntimeError(f"{label} file manifest is missing")
    path = Path(manifest["path"])
    raw, actual = _read_private_bytes(path, maximum_bytes=maximum_bytes)
    for name in ("path", "bytes", "sha256", "mode", "device", "inode", "mtime_ns"):
        if not _json_exact(manifest.get(name), actual[name]):
            raise RuntimeError(f"{label} file binding changed: {name}")
    return raw, actual


def _validate_compact_profile_manifest(manifest: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise RuntimeError(f"{label} compact profile manifest is missing")
    _validate_embedded_file_manifest(manifest, label=f"{label} summary", maximum_bytes=1 << 20)
    if (
        manifest.get("status") != "completed"
        or type(manifest.get("returncode")) is not int
        or manifest["returncode"] != 0
        or manifest.get("safety_violation") is not None
        or manifest.get("kernel_driver_errors") is not None
    ):
        raise RuntimeError(f"{label} compact profile did not pass safely")
    limits = {
        "gpu_junction_temp_c": 90.0,
        "gpu_power_watts": 400.0,
        "vram_used_bytes": 24 * 1024**3,
        "host_swap_used_bytes": 8 * 1024**3,
    }
    if not _json_exact(manifest.get("limits"), limits):
        raise RuntimeError(f"{label} compact profile limits are not exact")
    maxima = manifest.get("observed_maxima")
    if not isinstance(maxima, dict) or set(maxima) != set(limits):
        raise RuntimeError(f"{label} compact profile maxima are missing")
    for name, limit in limits.items():
        _require_number(maxima[name], label=f"{label} {name}", minimum=0, maximum=limit)
    telemetry = manifest.get("telemetry")
    if not isinstance(telemetry, dict):
        raise RuntimeError(f"{label} compact telemetry binding is missing")
    _validate_embedded_file_manifest(telemetry, label=f"{label} telemetry", maximum_bytes=32 << 20)
    _require_int(
        telemetry.get("record_count"),
        label=f"{label} telemetry record count",
        minimum=2,
    )
    _require_sha256(telemetry.get("command_sha256"), label=f"{label} command digest")
    _require_sha256(telemetry.get("profiler_sha256"), label=f"{label} profiler digest")
    if not isinstance(telemetry.get("evidence_output_path"), str):
        raise RuntimeError(f"{label} evidence output path is missing")
    return manifest


def _validate_host_input_manifest(host_inputs: Any) -> dict[str, Any]:
    if not isinstance(host_inputs, dict) or tuple(host_inputs) != tuple(sorted(name for name, _shape in _INPUT_SPECS)):
        # The result JSON is written with sort_keys=True.
        raise RuntimeError("benchmark host input set is not exact")
    expected = dict(_INPUT_SPECS)
    for name, shape in _INPUT_SPECS:
        manifest = host_inputs.get(name)
        element_count = math.prod(shape)
        if (
            not isinstance(manifest, dict)
            or set(manifest) != {"shape", "dtype", "element_count", "bytes", "sha256"}
            or not _json_exact(manifest.get("shape"), list(shape))
            or manifest.get("dtype") != "bfloat16"
            or type(manifest.get("element_count")) is not int
            or manifest["element_count"] != element_count
            or type(manifest.get("bytes")) is not int
            or manifest["bytes"] != element_count * 2
        ):
            raise RuntimeError(f"benchmark host input {name} is not exact")
        _require_sha256(manifest.get("sha256"), label=f"benchmark host input {name}")
        if tuple(manifest["shape"]) != expected[name]:
            raise RuntimeError(f"benchmark host input {name} shape changed")
    return host_inputs


def _validate_error_manifest(errors: Any) -> dict[str, Any]:
    expected_names = {"forward_output", "output", "dx", "d_lora_a", "d_lora_b"}
    if not isinstance(errors, dict) or set(errors) != expected_names:
        raise RuntimeError("prior numerics error set is not exact")
    for name, manifest in errors.items():
        if not isinstance(manifest, dict):
            raise RuntimeError(f"prior numerics {name} error manifest is malformed")
        relative_l2 = _require_number(
            manifest.get("relative_l2"),
            label=f"prior numerics {name} relative L2",
            minimum=0,
        )
        cosine = _require_number(
            manifest.get("cosine_similarity"),
            label=f"prior numerics {name} cosine",
            minimum=-1,
            maximum=1,
        )
        _require_number(
            manifest.get("max_absolute"),
            label=f"prior numerics {name} max absolute",
            minimum=0,
        )
        if manifest.get("finite") is not True or relative_l2 >= _RELATIVE_L2_LIMIT:
            raise RuntimeError(f"prior numerics {name} misses the numerical gate")
        if name in {"forward_output", "output"} and cosine < _OUTPUT_COSINE_LIMIT:
            raise RuntimeError(f"prior numerics {name} misses the cosine gate")
    return errors


def _validate_prior_chain(
    gated: dict[str, Any],
    *,
    contract: dict[str, Any],
    source: dict[str, Any],
    geometry: dict[str, Any],
    device: dict[str, Any],
) -> dict[str, Any]:
    compile_manifest = gated.get("compile_evidence")
    if not isinstance(compile_manifest, dict):
        raise RuntimeError("embedded compile evidence manifest is missing")
    compile_raw, _ = _validate_embedded_file_manifest(compile_manifest, label="compile evidence", maximum_bytes=4 << 20)
    try:
        compile_result = _strict_json_loads(compile_raw)
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("embedded compile evidence is not strict JSON") from error
    zero = dict.fromkeys(_PROGRAM_ORDER, 0)
    invocation = compile_result.get("invocation_contract") if isinstance(compile_result, dict) else None
    if (
        not isinstance(compile_result, dict)
        or type(compile_result.get("schema_version")) is not int
        or compile_result["schema_version"] != _EVIDENCE_SCHEMA_VERSION
        or compile_result.get("mode") != "compile_only"
        or compile_result.get("passed") is not True
        or not _json_exact(compile_result.get("contract"), contract)
        or not _json_exact(compile_result.get("geometry"), geometry)
        or not _json_exact(compile_result.get("device"), device)
        or not _json_exact(compile_result.get("source"), source)
        or not isinstance(invocation, dict)
        or not _json_exact(invocation.get("per_program_executable_invocations"), zero)
        or not _json_exact(invocation.get("per_program_executable_completions"), zero)
        or invocation.get("total_executable_invocations") != 0
    ):
        raise RuntimeError("embedded compile evidence is not exact")
    _validate_compilation(compile_result.get("compilation"))
    if (
        compile_manifest.get("commit") != source["git"]["commit"]
        or compile_manifest.get("zero_executable_invocations") is not True
        or compile_manifest.get("clean_preflight_and_postflight") is not True
        or compile_manifest.get("programs") != sorted(_PROGRAM_ORDER)
    ):
        raise RuntimeError("embedded compact compile binding is not exact")
    compile_profile = _validate_compact_profile_manifest(gated.get("compile_profile_summary"), label="compile")
    if compile_profile["telemetry"].get("evidence_output_path") != compile_manifest["path"]:
        raise RuntimeError("compile profile is not bound to compile evidence")

    numerics_manifest = gated.get("numerics_evidence")
    if not isinstance(numerics_manifest, dict):
        raise RuntimeError("embedded numerics evidence manifest is missing")
    numerics_raw, _ = _validate_embedded_file_manifest(
        numerics_manifest, label="numerics evidence", maximum_bytes=16 << 20
    )
    try:
        numerics_result = _strict_json_loads(numerics_raw)
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("embedded numerics evidence is not strict JSON") from error
    exact_one = dict.fromkeys(_PROGRAM_ORDER, 1)
    numerics_invocation = numerics_result.get("invocation_contract") if isinstance(numerics_result, dict) else None
    numerics = numerics_result.get("numerics") if isinstance(numerics_result, dict) else None
    numerics_gated = numerics_result.get("numerics_once") if isinstance(numerics_result, dict) else None
    errors = numerics.get("errors") if isinstance(numerics, dict) else None
    _validate_error_manifest(errors)
    if (
        not isinstance(numerics_result, dict)
        or type(numerics_result.get("schema_version")) is not int
        or numerics_result["schema_version"] != _EVIDENCE_SCHEMA_VERSION
        or numerics_result.get("mode") != "numerics_once"
        or numerics_result.get("passed") is not True
        or not _json_exact(numerics_result.get("contract"), contract)
        or not _json_exact(numerics_result.get("geometry"), geometry)
        or not _json_exact(numerics_result.get("device"), device)
        or not _json_exact(numerics_result.get("source"), source)
        or not isinstance(numerics_invocation, dict)
        or not _json_exact(numerics_invocation.get("per_program_executable_invocations"), exact_one)
        or not _json_exact(numerics_invocation.get("per_program_executable_completions"), exact_one)
        or numerics_invocation.get("total_executable_invocations") != 4
        or not isinstance(numerics, dict)
        or numerics.get("executed") is not True
        or numerics.get("reference_compared") is not True
        or numerics.get("passed") is not True
        or not isinstance(numerics_gated, dict)
        or not _json_exact(numerics_gated.get("compile_evidence"), compile_manifest)
        or not _json_exact(numerics_gated.get("compile_profile_summary"), compile_profile)
        or numerics_gated.get("host_inputs_unchanged") is not True
    ):
        raise RuntimeError("embedded numerics evidence is not exact")
    _validate_compilation(numerics_result.get("compilation"))
    prior_host_inputs = _validate_host_input_manifest(numerics_gated.get("host_inputs"))
    if not _json_exact(prior_host_inputs, gated.get("host_inputs")):
        raise RuntimeError("benchmark inputs do not match prior numerics inputs")
    if (
        numerics_manifest.get("commit") != source["git"]["commit"]
        or numerics_manifest.get("program_order") != list(_PROGRAM_ORDER)
        or numerics_manifest.get("one_attempt_and_completion_per_program") is not True
        or numerics_manifest.get("numerics_passed") is not True
        or not _json_exact(numerics_manifest.get("errors"), errors)
        or numerics_manifest.get("clean_preflight_and_postflight") is not True
    ):
        raise RuntimeError("embedded compact numerics binding is not exact")
    progress_manifest = numerics_manifest.get("progress")
    _validate_embedded_file_manifest(progress_manifest, label="numerics progress", maximum_bytes=16 << 20)
    if numerics_manifest.get("progress_record_count") != 10:
        raise RuntimeError("embedded numerics progress count is not exact")
    numerics_profile = _validate_compact_profile_manifest(gated.get("numerics_profile_summary"), label="numerics")
    if numerics_profile["telemetry"].get("evidence_output_path") != numerics_manifest["path"]:
        raise RuntimeError("numerics profile is not bound to numerics evidence")
    return {
        "compile_evidence": compile_manifest,
        "compile_profile_summary": compile_profile,
        "numerics_evidence": numerics_manifest,
        "numerics_profile_summary": numerics_profile,
    }


def _validate_preflight(
    preflight: Any,
    *,
    mode: str,
    repo: Path,
    source_only: dict[str, Any],
) -> dict[str, Any]:
    scope_key = f"{mode}_scope"
    if not isinstance(preflight, dict) or set(preflight) != {
        "environment",
        "hardware",
        "profiler_parent",
        "card_identity",
        "safety_source",
        scope_key,
    }:
        raise RuntimeError("benchmark preflight schema is not exact")
    environment = preflight["environment"]
    if not isinstance(environment, dict):
        raise RuntimeError("benchmark environment preflight is missing")
    required_environment = {
        "JAX_PLATFORMS": "rocm",
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
        "XLA_FLAGS_effective": _DISABLE_COMMAND_BUFFERS,
        "command_buffers_enabled": False,
        "graph_capture_enabled": False,
    }
    if any(not _json_exact(environment.get(name), value) for name, value in required_environment.items()):
        raise RuntimeError("benchmark environment/capture binding is not exact")
    if environment.get("XLA_FLAGS_original") not in ("", _DISABLE_COMMAND_BUFFERS):
        raise RuntimeError("benchmark inherited XLA flags are not exact")
    inherited = environment.get("inherited")
    if not isinstance(inherited, dict) or any(
        name
        not in {
            "JAX_PLATFORMS",
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
            "XLA_FLAGS",
        }
        for name in inherited
    ):
        raise RuntimeError("benchmark inherited accelerator environment is not exact")

    hardware = preflight["hardware"]
    if (
        not isinstance(hardware, dict)
        or hardware.get("amdgpu_boot_clean") is not True
        or hardware.get("kfd_accessible") is not True
        or hardware.get("kfd_unowned") is not True
        or hardware.get("connected_amd_connectors") != []
        or hardware.get("fatal_amdgpu_events") != []
        or hardware.get("amd_cards") != ["card1"]
        or hardware.get("kfd_path") != "/dev/kfd"
    ):
        raise RuntimeError("benchmark hardware preflight is not exact")
    card = preflight["card_identity"]
    if (
        not isinstance(card, dict)
        or card.get("drm_card") != "card1"
        or card.get("pci_vendor") != "0x1002"
        or card.get("pci_device") != "0x744c"
        or card.get("driver") != "amdgpu"
        or card.get("architecture") != "gfx1100"
        or not isinstance(card.get("pci_bdf"), str)
        or not isinstance(card.get("sysfs_device"), str)
    ):
        raise RuntimeError("benchmark card identity is not exact")
    safety_path = (repo / "rocm" / "amdgpu_safety.py").resolve(strict=True)
    safety_source = {
        "path": str(safety_path),
        "sha256": hashlib.sha256(safety_path.read_bytes()).hexdigest(),
    }
    if not _json_exact(preflight["safety_source"], safety_source):
        raise RuntimeError("benchmark safety source is not exact")

    profiler_parent = preflight["profiler_parent"]
    expected_python = str((repo / ".venv" / "bin" / "python").absolute())
    if (
        not isinstance(profiler_parent, dict)
        or profiler_parent.get("validated") is not True
        or profiler_parent.get("parent_command_python") != expected_python
        or profiler_parent.get("profiler_path") != source_only["profiler"]["path"]
        or profiler_parent.get("profiler_sha256") != source_only["profiler"]["sha256"]
        or not _json_exact(profiler_parent.get("limits"), _EXACT_PROFILE_LIMITS)
    ):
        raise RuntimeError("benchmark profiler-parent binding is not exact")
    _require_int(profiler_parent.get("parent_pid"), label="profiler parent PID", minimum=1)
    if not isinstance(profiler_parent.get("parent_executable"), str):
        raise RuntimeError("benchmark profiler executable is missing")
    _require_sha256(
        profiler_parent.get("parent_command_sha256"),
        label="profiler parent command",
    )
    timings = {
        "timeout_seconds": _require_number(
            profiler_parent.get("timeout_seconds"),
            label="profiler timeout",
            minimum=1e-12,
            maximum=1800,
        ),
        "interval_seconds": _require_number(
            profiler_parent.get("interval_seconds"),
            label="profiler interval",
            minimum=1e-12,
            maximum=0.25,
        ),
        "baseline_seconds": _require_number(
            profiler_parent.get("baseline_seconds"),
            label="profiler baseline",
            minimum=2,
        ),
        "sensor_grace_seconds": _require_number(
            profiler_parent.get("sensor_grace_seconds"),
            label="profiler sensor grace",
            minimum=0,
            maximum=60,
        ),
    }

    scope = preflight[scope_key]
    mode_label = mode.replace("_", "-")
    unit_pattern = rf"skyrl-bf16-{mode_label}-[0-9]+-[0-9a-f]+\.scope"
    if (
        not isinstance(scope, dict)
        or scope.get("validated") is not True
        or scope.get("profile_parent_same_cgroup") is not True
        or scope.get("timeout_kill_scope") != "entire_systemd_cgroup"
        or not isinstance(scope.get("scope_unit"), str)
        or re.fullmatch(unit_pattern, scope["scope_unit"]) is None
        or not isinstance(scope.get("cgroup"), str)
        or not scope["cgroup"].endswith("/" + scope["scope_unit"])
        or not isinstance(scope.get("cgroup_kill_path"), str)
        or not scope["cgroup_kill_path"].endswith(scope["cgroup"] + "/cgroup.kill")
    ):
        raise RuntimeError("benchmark private systemd scope is not exact")
    _require_int(scope.get("cgroup_kill_device"), label="cgroup.kill device", minimum=0)
    _require_int(scope.get("cgroup_kill_inode"), label="cgroup.kill inode", minimum=1)
    return {"profiler_timings": timings, "scope": scope, "safety_source": safety_source}


def _validate_compilation(compilation: Any) -> dict[str, Any]:
    if not isinstance(compilation, dict) or set(compilation) != set(_PROGRAM_ORDER):
        raise RuntimeError("benchmark compilation program set is not exact")
    for program, manifest in compilation.items():
        if (
            not isinstance(manifest, dict)
            or set(manifest)
            != {
                "lower_calls",
                "compile_calls",
                "lower_seconds",
                "compile_seconds",
                "program_structure",
            }
            or type(manifest.get("lower_calls")) is not int
            or manifest["lower_calls"] != 1
            or type(manifest.get("compile_calls")) is not int
            or manifest["compile_calls"] != 1
        ):
            raise RuntimeError(f"benchmark {program} compilation is not exact")
        _require_number(manifest["lower_seconds"], label=f"{program} lower time", minimum=0)
        _require_number(manifest["compile_seconds"], label=f"{program} compile time", minimum=0)
        if not _json_exact(manifest["program_structure"], _expected_program_structures()[program]):
            raise RuntimeError(f"benchmark {program} program structure is not exact")
    return compilation


def _validate_watchdog(
    watchdog: Any,
    *,
    label: str,
    expected_fields: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(watchdog, dict):
        raise RuntimeError(f"{label} watchdog is missing")
    for name, expected in expected_fields.items():
        if not _json_exact(watchdog.get(name), expected):
            raise RuntimeError(f"{label} watchdog {name} is not exact")
    if (
        watchdog.get("external_process") is not True
        or watchdog.get("timeout_action") != "cgroup.kill_then_pidfd_SIGKILL_fallback"
        or watchdog.get("cgroup_wide_timeout_kill") is not True
        or watchdog.get("timeout_seconds") != _WATCHDOG_SECONDS
        or watchdog.get("operation_completed") is not True
    ):
        raise RuntimeError(f"{label} watchdog containment is not exact")
    _require_int(watchdog.get("watchdog_pid"), label=f"{label} watchdog PID", minimum=1)
    armed = _require_int(watchdog.get("armed_monotonic_ns"), label=f"{label} armed time", minimum=1)
    deadline = _require_int(
        watchdog.get("deadline_monotonic_ns"),
        label=f"{label} deadline",
        minimum=1,
    )
    ack = _require_int(
        watchdog.get("armed_ack_received_monotonic_ns"),
        label=f"{label} armed acknowledgement",
        minimum=1,
    )
    if deadline - armed != _WATCHDOG_NS or not armed <= ack < deadline:
        raise RuntimeError(f"{label} watchdog deadline ordering is not exact")
    return watchdog


def _validate_device_input_evidence(setup_completed: dict[str, Any], host_inputs: dict[str, Any]) -> None:
    device_inputs = setup_completed.get("device_inputs")
    roundtrip = setup_completed.get("roundtrip")
    if (
        not isinstance(device_inputs, dict)
        or set(device_inputs) != set(host_inputs)
        or not isinstance(roundtrip, dict)
        or set(roundtrip) != set(host_inputs)
    ):
        raise RuntimeError("benchmark device input/roundtrip sets are not exact")
    for name, host in host_inputs.items():
        device = device_inputs[name]
        copied = roundtrip[name]
        if not _json_exact(
            device,
            {
                "shape": host["shape"],
                "dtype": "bfloat16",
                "element_count": host["element_count"],
                "bytes": host["bytes"],
                "host_sha256": host["sha256"],
                "exact_device": True,
            },
        ):
            raise RuntimeError(f"benchmark staged device input {name} is not exact")
        if not _json_exact(
            copied,
            {
                "shape": host["shape"],
                "dtype": "bfloat16",
                "bytes": host["bytes"],
                "sha256": host["sha256"],
                "matches_host_input": True,
            },
        ):
            raise RuntimeError(f"benchmark input roundtrip {name} is not exact")


def _expected_progress_binding(
    raw_result: dict[str, Any], gated: dict[str, Any], smoke_attestation: Any
) -> dict[str, Any]:
    source = raw_result["source"]
    expected = {
        "contract": raw_result["contract"],
        "geometry": raw_result["geometry"],
        "preflight": raw_result["preflight"],
        "source": {name: source[name] for name in ("kernel", "probe", "profiler")},
        "git": source["git"],
        "compile_evidence": gated["compile_evidence"],
        "compile_profile_summary": gated["compile_profile_summary"],
        "numerics_evidence": gated["numerics_evidence"],
        "numerics_profile_summary": gated["numerics_profile_summary"],
    }
    if smoke_attestation is not None:
        expected["smoke_attestation"] = smoke_attestation
    return expected


def _validate_progress(
    *,
    mode: str,
    raw_result: dict[str, Any],
    result_file: dict[str, Any],
    gated: dict[str, Any],
    smoke_attestation: Any,
) -> dict[str, Any]:
    progress_manifest = gated.get("progress")
    if (
        not isinstance(progress_manifest, dict)
        or set(progress_manifest)
        != {
            "path",
            "protocol",
            "record_count",
            "bytes",
            "sha256",
            "mode",
            "directory_fsynced",
        }
        or progress_manifest.get("protocol") != "durable_fsync_jsonl_v1"
        or progress_manifest.get("mode") != "0600"
        or progress_manifest.get("directory_fsynced") is not True
        or not isinstance(progress_manifest.get("path"), str)
    ):
        raise RuntimeError("benchmark progress manifest is not exact")
    progress_path = Path(progress_manifest["path"])
    progress_raw, progress_file = _read_private_bytes(progress_path, maximum_bytes=32 << 20)
    if (
        not progress_raw.endswith(b"\n")
        or progress_manifest.get("bytes") != len(progress_raw)
        or progress_manifest.get("sha256") != hashlib.sha256(progress_raw).hexdigest()
        or progress_file["mtime_ns"] >= result_file["mtime_ns"]
    ):
        raise RuntimeError("benchmark progress bytes/digest/order are not exact")
    try:
        records = [_strict_json_loads(line) for line in progress_raw.splitlines()]
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("benchmark progress is not strict JSONL") from error

    warmups, measurements = _BENCHMARK_MODE_COUNTS[mode]
    schedule = _benchmark_schedule(mode)
    dispatches = [
        {
            **supercycle,
            "supercycle_position": position,
            "program": program,
        }
        for supercycle in schedule
        for position, program in enumerate(supercycle["order"])
    ]
    expected_record_count = 2 * len(dispatches) + 6
    if (
        type(progress_manifest.get("record_count")) is not int
        or progress_manifest["record_count"] != expected_record_count
        or len(records) != expected_record_count
        or any(not isinstance(record, dict) for record in records)
    ):
        raise RuntimeError("benchmark progress record count is not exact")
    expected_events = [
        "host_inputs_ready",
        "device_input_setup_started",
        "device_input_setup_completed",
        *(event for _dispatch in dispatches for event in ("dispatch_started", "dispatch_completed")),
        "benchmark_samples_completed",
        "device_input_teardown_started",
        "device_input_teardown_completed",
    ]
    if [record.get("event") for record in records] != expected_events:
        raise RuntimeError("benchmark progress event protocol is not exact")
    probe_pid = _require_int(records[0].get("probe_pid"), label="probe PID", minimum=1)
    expected_binding = _expected_progress_binding(raw_result, gated, smoke_attestation)
    empty_counts = dict.fromkeys(_PROGRAM_ORDER, 0)
    attempt_counts = dict(empty_counts)
    completion_counts = dict(empty_counts)
    setup_counts = {"attempts": 0, "completions": 0}
    teardown_counts = {"attempts": 0, "completions": 0}
    previous_monotonic = 0

    common_keys = {
        "schema_version",
        "event",
        "mode",
        "probe_pid",
        "wall_time_ns",
        "monotonic_time_ns",
        "binding",
        "invocation_attempt_counts",
        "invocation_completion_counts",
        "device_input_setup_counts",
        "device_input_teardown_counts",
    }

    def validate_common(record: dict[str, Any]) -> None:
        nonlocal previous_monotonic
        if (
            type(record.get("schema_version")) is not int
            or record["schema_version"] != _EVIDENCE_SCHEMA_VERSION
            or record.get("mode") != mode
            or record.get("probe_pid") != probe_pid
            or not _json_exact(record.get("binding"), expected_binding)
            or not _json_exact(record.get("invocation_attempt_counts"), attempt_counts)
            or not _json_exact(record.get("invocation_completion_counts"), completion_counts)
            or not _json_exact(record.get("device_input_setup_counts"), setup_counts)
            or not _json_exact(record.get("device_input_teardown_counts"), teardown_counts)
        ):
            raise RuntimeError("benchmark progress common binding/counts are not exact")
        _require_int(record.get("wall_time_ns"), label="progress wall time", minimum=1)
        monotonic = _require_int(record.get("monotonic_time_ns"), label="progress monotonic time", minimum=1)
        if monotonic <= previous_monotonic:
            raise RuntimeError("benchmark progress monotonic order is not strict")
        previous_monotonic = monotonic

    host_inputs = _validate_host_input_manifest(gated.get("host_inputs"))
    first = records[0]
    if set(first) != common_keys | {
        "host_inputs",
        "warmup_supercycles",
        "measured_supercycles",
        "total_program_invocations",
        "watchdog_seconds_per_operation",
    }:
        raise RuntimeError("benchmark host-input progress schema is not exact")
    validate_common(first)
    if (
        not _json_exact(first.get("host_inputs"), host_inputs)
        or first.get("warmup_supercycles") != warmups
        or first.get("measured_supercycles") != measurements
        or first.get("total_program_invocations") != len(dispatches)
        or first.get("watchdog_seconds_per_operation") != _WATCHDOG_SECONDS
    ):
        raise RuntimeError("benchmark host-input progress values are not exact")

    setup_started = records[1]
    setup_counts["attempts"] = 1
    if set(setup_started) != common_keys | {
        "watchdog_armed_monotonic_ns",
        "watchdog_deadline_monotonic_ns",
    }:
        raise RuntimeError("benchmark setup-start progress schema is not exact")
    validate_common(setup_started)
    setup_watchdog = _validate_watchdog(
        gated.get("setup_watchdog"),
        label="device input setup",
        expected_fields={
            "operation": "device_input_setup",
            "guarded_device_put": True,
            "guarded_block": True,
            "guarded_device_get_roundtrip": True,
        },
    )
    if (
        setup_started.get("watchdog_armed_monotonic_ns") != setup_watchdog["armed_monotonic_ns"]
        or setup_started.get("watchdog_deadline_monotonic_ns") != setup_watchdog["deadline_monotonic_ns"]
    ):
        raise RuntimeError("benchmark setup watchdog/progress binding changed")

    setup_completed = records[2]
    setup_counts["completions"] = 1
    if set(setup_completed) != common_keys | {
        "completed_monotonic_ns",
        "one_device_put_call",
        "one_device_get_roundtrip_call",
        "all_input_leaves_blocked",
        "device_inputs",
        "roundtrip",
    }:
        raise RuntimeError("benchmark setup-completion progress schema is not exact")
    validate_common(setup_completed)
    completed_ns = _require_int(
        setup_completed.get("completed_monotonic_ns"),
        label="device setup completion",
        minimum=1,
    )
    if (
        completed_ns >= setup_watchdog["deadline_monotonic_ns"]
        or setup_completed.get("one_device_put_call") is not True
        or setup_completed.get("one_device_get_roundtrip_call") is not True
        or setup_completed.get("all_input_leaves_blocked") is not True
    ):
        raise RuntimeError("benchmark guarded device setup did not complete exactly")
    _validate_device_input_evidence(setup_completed, host_inputs)

    watchdogs = gated.get("dispatch_watchdogs")
    if not isinstance(watchdogs, list) or len(watchdogs) != len(dispatches):
        raise RuntimeError("benchmark dispatch watchdog count is not exact")
    record_index = 3
    expected_samples: list[dict[str, Any]] = []
    for ordinal, (dispatch, watchdog) in enumerate(zip(dispatches, watchdogs, strict=True), start=1):
        program = dispatch["program"]
        attempt_counts[program] += 1
        started = records[record_index]
        record_index += 1
        dispatch_fields = {
            "dispatch_ordinal": ordinal,
            "phase": dispatch["phase"],
            "phase_supercycle": dispatch["phase_supercycle"],
            "global_supercycle": dispatch["global_supercycle"],
            "supercycle_position": dispatch["supercycle_position"],
            "program": program,
        }
        if set(started) != common_keys | set(dispatch_fields) | {
            "watchdog_armed_monotonic_ns",
            "watchdog_deadline_monotonic_ns",
        }:
            raise RuntimeError("benchmark dispatch-start progress schema is not exact")
        validate_common(started)
        if any(not _json_exact(started.get(name), value) for name, value in dispatch_fields.items()):
            raise RuntimeError("benchmark dispatch-start schedule is not exact")
        validated_watchdog = _validate_watchdog(
            watchdog,
            label=f"dispatch {ordinal}",
            expected_fields={
                "operation": "program_dispatch",
                "dispatch_ordinal": ordinal,
                "phase": dispatch["phase"],
                "phase_supercycle": dispatch["phase_supercycle"],
                "supercycle_position": dispatch["supercycle_position"],
                "program": program,
            },
        )
        if (
            started.get("watchdog_armed_monotonic_ns") != validated_watchdog["armed_monotonic_ns"]
            or started.get("watchdog_deadline_monotonic_ns") != validated_watchdog["deadline_monotonic_ns"]
        ):
            raise RuntimeError("benchmark dispatch watchdog/progress binding changed")

        completion_counts[program] += 1
        completed = records[record_index]
        record_index += 1
        if set(completed) != common_keys | set(dispatch_fields) | {
            "elapsed_seconds",
            "result_leaves_blocked_before_timer_stop",
            "result_leaves_explicitly_deleted",
            "result_references_released_before_completion",
        }:
            raise RuntimeError("benchmark dispatch-completion progress schema is not exact")
        validate_common(completed)
        if any(not _json_exact(completed.get(name), value) for name, value in dispatch_fields.items()):
            raise RuntimeError("benchmark dispatch-completion schedule is not exact")
        elapsed = _require_number(
            completed.get("elapsed_seconds"),
            label=f"dispatch {ordinal} elapsed time",
            minimum=1e-15,
            maximum=_WATCHDOG_SECONDS,
        )
        if (
            completed.get("result_leaves_blocked_before_timer_stop") is not True
            or completed.get("result_leaves_explicitly_deleted") != _RESULT_LEAF_COUNTS[program]
            or completed.get("result_references_released_before_completion") is not True
        ):
            raise RuntimeError("benchmark dispatch result deletion is not exact")
        if dispatch["phase"] == "measurement":
            expected_samples.append({**dispatch_fields, "elapsed_seconds": elapsed})

    samples_completed = records[record_index]
    record_index += 1
    if set(samples_completed) != common_keys | {
        "raw_measurement_sample_count",
        "raw_samples",
        "performance_qualification",
    }:
        raise RuntimeError("benchmark sample-terminal progress schema is not exact")
    validate_common(samples_completed)
    if (
        samples_completed.get("raw_measurement_sample_count") != len(expected_samples)
        or not _json_exact(samples_completed.get("raw_samples"), expected_samples)
        or samples_completed.get("performance_qualification") is not False
    ):
        raise RuntimeError("benchmark sample-terminal evidence is not exact")

    teardown_started = records[record_index]
    record_index += 1
    teardown_counts["attempts"] = 1
    if set(teardown_started) != common_keys | {
        "watchdog_armed_monotonic_ns",
        "watchdog_deadline_monotonic_ns",
    }:
        raise RuntimeError("benchmark teardown-start progress schema is not exact")
    validate_common(teardown_started)
    teardown_watchdog = _validate_watchdog(
        gated.get("teardown_watchdog"),
        label="device input teardown",
        expected_fields={
            "operation": "device_input_teardown",
            "device_inputs_explicitly_deleted": True,
            "executable_references_cleared": True,
            "jax_caches_cleared": True,
            "all_device_inputs_ready_before_delete": True,
            "all_dispatch_results_blocked": True,
            "effects_barrier_completed": True,
        },
    )
    if (
        teardown_started.get("watchdog_armed_monotonic_ns") != teardown_watchdog["armed_monotonic_ns"]
        or teardown_started.get("watchdog_deadline_monotonic_ns") != teardown_watchdog["deadline_monotonic_ns"]
    ):
        raise RuntimeError("benchmark teardown watchdog/progress binding changed")

    teardown_completed = records[record_index]
    teardown_counts["completions"] = 1
    if set(teardown_completed) != common_keys | {
        "completed_monotonic_ns",
        "explicitly_deleted_unique_input_leaves",
        "executable_references_cleared",
        "jax_caches_cleared",
        "garbage_collection_completed",
        "all_device_inputs_ready_before_delete",
        "all_dispatch_results_blocked",
        "effects_barrier_completed",
    }:
        raise RuntimeError("benchmark teardown-completion progress schema is not exact")
    validate_common(teardown_completed)
    teardown_completed_ns = _require_int(
        teardown_completed.get("completed_monotonic_ns"),
        label="device teardown completion",
        minimum=1,
    )
    if (
        teardown_completed_ns >= teardown_watchdog["deadline_monotonic_ns"]
        or teardown_completed.get("explicitly_deleted_unique_input_leaves") != len(_INPUT_SPECS)
        or teardown_completed.get("executable_references_cleared") is not True
        or teardown_completed.get("jax_caches_cleared") is not True
        or teardown_completed.get("garbage_collection_completed") is not True
        or teardown_completed.get("all_device_inputs_ready_before_delete") is not True
        or teardown_completed.get("all_dispatch_results_blocked") is not True
        or teardown_completed.get("effects_barrier_completed") is not True
    ):
        raise RuntimeError("benchmark guarded device teardown is not exact")
    if record_index != len(records) - 1:
        raise RuntimeError("benchmark progress parser did not consume exact records")

    exact_per_program = warmups + measurements
    exact_counts = dict.fromkeys(_PROGRAM_ORDER, exact_per_program)
    if (
        attempt_counts != exact_counts
        or completion_counts != exact_counts
        or gated.get("device_setup_counts") != {"attempts": 1, "completions": 1}
        or gated.get("device_teardown_counts") != {"attempts": 1, "completions": 1}
        or not _json_exact(gated.get("invocation_attempt_counts"), exact_counts)
        or not _json_exact(gated.get("invocation_completion_counts"), exact_counts)
    ):
        raise RuntimeError("benchmark progress terminal counts are not exact")
    return {
        "file": progress_file,
        "record_count": len(records),
        "probe_pid": probe_pid,
        "expected_samples": expected_samples,
        "setup_watchdog": setup_watchdog,
        "dispatch_watchdogs": watchdogs,
        "teardown_watchdog": teardown_watchdog,
    }


def _validate_measurement(
    *,
    mode: str,
    raw_result: dict[str, Any],
    progress: dict[str, Any],
) -> dict[str, Any]:
    warmups, measurements = _BENCHMARK_MODE_COUNTS[mode]
    schedule = _benchmark_schedule(mode)
    expected_warmup_orders = [list(item["order"]) for item in schedule[:warmups]]
    expected_measurement_orders = [list(item["order"]) for item in schedule[warmups:]]
    measurement = raw_result.get("measurement")
    if (
        not isinstance(measurement, dict)
        or measurement.get("executed") is not True
        or measurement.get("performance_measured") is not True
        or measurement.get("performance_qualification") is not False
        or measurement.get("raw_samples_only") is not True
        or measurement.get("warmup_supercycles") != warmups
        or measurement.get("measured_supercycles") != measurements
        or not _json_exact(measurement.get("warmup_orders"), expected_warmup_orders)
        or not _json_exact(measurement.get("measurement_orders"), expected_measurement_orders)
        or not _json_exact(measurement.get("raw_samples"), progress["expected_samples"])
    ):
        raise RuntimeError("benchmark measurement manifest is not exact")
    by_program = measurement.get("raw_samples_by_program")
    if not isinstance(by_program, dict) or set(by_program) != set(_PROGRAM_ORDER):
        raise RuntimeError("benchmark per-program samples are not exact")
    expected_by_program = {
        program: [sample["elapsed_seconds"] for sample in progress["expected_samples"] if sample["program"] == program]
        for program in _PROGRAM_ORDER
    }
    if not _json_exact(by_program, expected_by_program) or any(
        len(samples) != measurements for samples in expected_by_program.values()
    ):
        raise RuntimeError("benchmark per-program sample binding changed")

    medians = {program: statistics.median(samples) for program, samples in expected_by_program.items()}
    forward_speedup = medians["reference_forward"] / medians["candidate_forward"]
    vjp_speedup = medians["reference_forward_and_vjp"] / medians["candidate_forward_and_vjp"]
    reference_rematerialized = medians["reference_forward"] + medians["reference_forward_and_vjp"]
    candidate_rematerialized = medians["candidate_forward"] + medians["candidate_forward_and_vjp"]
    rematerialized_speedup = reference_rematerialized / candidate_rematerialized
    if not all(
        math.isfinite(value) and value > 0
        for value in (
            *medians.values(),
            forward_speedup,
            vjp_speedup,
            rematerialized_speedup,
        )
    ):
        raise RuntimeError("benchmark derived performance is not finite")
    gates_passed = bool(
        vjp_speedup >= _MIN_FORWARD_VJP_SPEEDUP and rematerialized_speedup >= _MIN_REMATERIALIZED_STAGE_SPEEDUP
    )
    return {
        "raw_samples_by_program": expected_by_program,
        "median_seconds_by_program": medians,
        "forward_speedup": forward_speedup,
        "forward_and_vjp_speedup": vjp_speedup,
        "rematerialized_stage_speedup": rematerialized_speedup,
        "minimum_forward_and_vjp_speedup": _MIN_FORWARD_VJP_SPEEDUP,
        "minimum_rematerialized_stage_speedup": (_MIN_REMATERIALIZED_STAGE_SPEEDUP),
        "performance_gates_passed": gates_passed,
        "qualifying_mode": mode == "benchmark",
        "determinism_required_for_integration": True,
        "determinism_attested": False,
    }


def _expected_profile_command(
    *,
    repo: Path,
    mode: str,
    result_path: Path,
    gated: dict[str, Any],
    smoke_attestation: dict[str, Any] | None,
) -> list[str]:
    warmups, measurements = _BENCHMARK_MODE_COUNTS[mode]
    command = [
        str((repo / ".venv" / "bin" / "python").absolute()),
        str((repo / "rocm" / "probe_bf16_rms_gate_up_lora_swiglu.py").resolve(strict=True)),
        "--allow-gpu",
        "--" + mode.replace("_", "-"),
        "--output",
        str(result_path),
        "--progress-output",
        gated["progress"]["path"],
        "--compile-evidence",
        gated["compile_evidence"]["path"],
        "--compile-profile-summary",
        gated["compile_profile_summary"]["path"],
        "--numerics-evidence",
        gated["numerics_evidence"]["path"],
        "--numerics-profile-summary",
        gated["numerics_profile_summary"]["path"],
    ]
    if mode == "benchmark":
        if smoke_attestation is None:
            raise RuntimeError("full benchmark has no smoke attestation")
        command.extend(
            (
                "--smoke-evidence",
                smoke_attestation["result"]["path"],
                "--smoke-profile-summary",
                smoke_attestation["profile"]["summary"]["path"],
            )
        )
    command.extend(
        (
            "--block-m",
            "16",
            "--block-physical-n",
            "64",
            "--block-k",
            "32",
            "--warmups",
            str(warmups),
            "--iterations",
            str(measurements),
        )
    )
    return command


def _validate_profile(
    *,
    profile_summary_path: Path,
    result_path: Path,
    result_file: dict[str, Any],
    raw_result: dict[str, Any],
    mode: str,
    repo: Path,
    gated: dict[str, Any],
    smoke_attestation: dict[str, Any] | None,
) -> dict[str, Any]:
    if profile_summary_path.parent != result_path.parent or profile_summary_path.name != "telemetry.jsonl.summary.json":
        raise RuntimeError("benchmark profile summary is not the exact result sibling")
    summary, summary_file = _read_private_json(profile_summary_path, maximum_bytes=1 << 20)
    telemetry_path = profile_summary_path.with_name("telemetry.jsonl")
    telemetry_raw, telemetry_file = _read_private_bytes(telemetry_path, maximum_bytes=64 << 20)
    if not telemetry_raw.endswith(b"\n"):
        raise RuntimeError("benchmark telemetry is not complete JSONL")
    try:
        telemetry = [_strict_json_loads(line) for line in telemetry_raw.splitlines()]
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("benchmark telemetry is not strict JSONL") from error
    if (
        not isinstance(summary, dict)
        or not telemetry
        or not isinstance(telemetry[0], dict)
        or max(result_file["mtime_ns"], telemetry_file["mtime_ns"]) >= summary_file["mtime_ns"]
    ):
        raise RuntimeError("benchmark result/telemetry/summary ordering is not exact")
    manifest = telemetry[0]
    samples = telemetry[1:]
    if any(not isinstance(sample, dict) or sample.get("record_type") != "sample" for sample in samples):
        raise RuntimeError("benchmark telemetry sample stream is not exact")
    profiler = raw_result["preflight"]["profiler_parent"]
    expected_limits = {
        "max_junction_temp_c": 90.0,
        "max_gpu_power_watts": 400.0,
        "max_vram_bytes": float(24 * 1024**3),
        "min_host_available_bytes": 0.0,
        "max_swap_bytes": float(8 * 1024**3),
    }
    expected_command = _expected_profile_command(
        repo=repo,
        mode=mode,
        result_path=result_path,
        gated=gated,
        smoke_attestation=smoke_attestation,
    )
    runtime = manifest.get("runtime")
    gpu = manifest.get("gpu")
    if (
        manifest.get("record_type") != "manifest"
        or manifest.get("interval_seconds") != profiler["interval_seconds"]
        or manifest.get("baseline_seconds") != profiler["baseline_seconds"]
        or manifest.get("duration_seconds") is not None
        or manifest.get("timeout_seconds") != profiler["timeout_seconds"]
        or manifest.get("sensor_grace_seconds") != profiler["sensor_grace_seconds"]
        or manifest.get("terminate_included_on_safety") is not False
        or manifest.get("terminate_included_on_abort") is not False
        or not _json_exact(manifest.get("safety_limits"), expected_limits)
        or manifest.get("command_recorded") is not True
        or manifest.get("passed_file_descriptor_count") != 0
        or not _json_exact(manifest.get("command"), expected_command)
        or not isinstance(runtime, dict)
        or runtime.get("script_sha256") != raw_result["source"]["profiler"]["sha256"]
        or not _json_exact(
            runtime.get("accelerator_environment"),
            {
                "HIP_VISIBLE_DEVICES": "0",
                "JAX_PLATFORMS": "rocm",
                "XLA_FLAGS": _DISABLE_COMMAND_BUFFERS,
            },
        )
        or not isinstance(gpu, dict)
        or gpu.get("card") != "card1"
        or gpu.get("vendor_id") != "0x1002"
        or gpu.get("device_id") != "0x744c"
    ):
        raise RuntimeError("benchmark profiler telemetry manifest is not exact")

    phases = [sample.get("phase") for sample in samples]
    baseline_count = phases.count("baseline")
    measured_count = phases.count("measured")
    if (
        any(phase not in {"baseline", "preflight", "measured"} for phase in phases)
        or phases.count("preflight") != 1
        or baseline_count <= 0
        or measured_count <= 0
        or summary.get("record_type") != "summary"
        or summary.get("status") != "completed"
        or type(summary.get("returncode")) is not int
        or summary["returncode"] != 0
        or summary.get("received_signal") is not None
        or summary.get("safety_violation") is not None
        or summary.get("kernel_driver_errors") is not None
        or summary.get("kernel_log_available") is not True
        or summary.get("baseline_samples") != baseline_count
        or summary.get("measured_samples") != measured_count
        or summary.get("samples") != len(samples)
        or len(samples) != baseline_count + measured_count + 1
    ):
        raise RuntimeError("benchmark profiler did not complete safely and exactly")
    previous_wall = 0
    for sample in samples:
        wall = _require_int(sample.get("wall_time_ns"), label="telemetry wall time", minimum=1)
        if wall <= previous_wall:
            raise RuntimeError("benchmark telemetry wall-time order is not strict")
        previous_wall = wall

    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("benchmark profiler metrics are missing")
    maxima: dict[str, float] = {}
    for name, limit in _SAFETY_METRIC_LIMITS.items():
        measured_values = [
            _require_number(sample[name], label=f"telemetry {name}", minimum=0)
            for sample in samples
            if sample.get("phase") == "measured" and sample.get(name) is not None
        ]
        if not measured_values:
            raise RuntimeError(f"benchmark telemetry has no measured {name}")
        observed = max(measured_values)
        metric = metrics.get(name)
        if (
            observed > float(limit)
            or not isinstance(metric, dict)
            or _require_number(
                metric.get("measured_max"),
                label=f"summary {name} maximum",
                minimum=0,
            )
            != observed
        ):
            raise RuntimeError(f"benchmark telemetry/summary {name} is unsafe")
        maxima[name] = observed
    return {
        "summary": summary_file,
        "telemetry": {
            **telemetry_file,
            "record_count": len(telemetry),
            "sample_count": len(samples),
            "baseline_sample_count": baseline_count,
            "measured_sample_count": measured_count,
            "command_sha256": hashlib.sha256(
                json.dumps(expected_command, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
        "observed_maxima": maxima,
        "limits": dict(_SAFETY_METRIC_LIMITS),
        "status": "completed",
        "returncode": 0,
        "safety_violation": None,
        "kernel_driver_errors": None,
    }


def _validate_host_result_manifest(host_results: Any) -> dict[str, Any]:
    if not isinstance(host_results, dict) or set(host_results) != set(_RESULT_SPECS):
        raise RuntimeError("determinism host-result program set is not exact")
    for program, specs in _RESULT_SPECS.items():
        result = host_results.get(program)
        if not isinstance(result, dict) or set(result) != {name for name, _shape in specs}:
            raise RuntimeError(f"determinism {program} result set is not exact")
        for name, shape in specs:
            manifest = result[name]
            element_count = math.prod(shape)
            if (
                not isinstance(manifest, dict)
                or set(manifest)
                != {
                    "shape",
                    "device_dtype",
                    "host_dtype",
                    "element_count",
                    "bf16_bytes",
                    "bf16_sha256",
                    "finite",
                }
                or not _json_exact(manifest.get("shape"), list(shape))
                or manifest.get("device_dtype") != "bfloat16"
                or manifest.get("host_dtype") != "float32"
                or type(manifest.get("element_count")) is not int
                or manifest["element_count"] != element_count
                or type(manifest.get("bf16_bytes")) is not int
                or manifest["bf16_bytes"] != 2 * element_count
                or manifest.get("finite") is not True
            ):
                raise RuntimeError(f"determinism {program}/{name} result manifest is not exact")
            _require_sha256(
                manifest.get("bf16_sha256"),
                label=f"determinism {program}/{name} BF16 digest",
            )
    return host_results


def _candidate_result_hashes(host_results: dict[str, Any]) -> dict[str, str]:
    _validate_host_result_manifest(host_results)
    return {
        f"{program}/{name}": host_results[program][name]["bf16_sha256"] for program, name in _CANDIDATE_HASH_POSITIONS
    }


def _expected_numerics_progress_binding(result: dict[str, Any]) -> dict[str, Any]:
    source = result["source"]
    gated = result["numerics_once"]
    return {
        "contract": result["contract"],
        "geometry": result["geometry"],
        "preflight": result["preflight"],
        "source": {name: source[name] for name in ("kernel", "probe", "profiler")},
        "git": source["git"],
        "compile_evidence": gated["compile_evidence"],
        "compile_profile_summary": gated["compile_profile_summary"],
        "numerics_evidence": None,
        "numerics_profile_summary": None,
    }


def _validate_numerics_watchdog(
    watchdog: Any,
    *,
    ordinal: int,
    program: str,
) -> dict[str, Any]:
    if (
        not isinstance(watchdog, dict)
        or set(watchdog)
        != {
            "dispatch_ordinal",
            "program",
            "external_process",
            "watchdog_pid",
            "timeout_action",
            "cgroup_wide_timeout_kill",
            "timeout_seconds",
            "armed_monotonic_ns",
            "deadline_monotonic_ns",
            "armed_ack_received_monotonic_ns",
            "dispatch_completed",
        }
        or watchdog.get("dispatch_ordinal") != ordinal
        or watchdog.get("program") != program
        or watchdog.get("external_process") is not True
        or watchdog.get("timeout_action") != "cgroup.kill_then_pidfd_SIGKILL_fallback"
        or watchdog.get("cgroup_wide_timeout_kill") is not True
        or watchdog.get("timeout_seconds") != _WATCHDOG_SECONDS
        or watchdog.get("dispatch_completed") is not True
    ):
        raise RuntimeError(f"determinism watchdog {ordinal} is not exact")
    _require_int(
        watchdog.get("watchdog_pid"),
        label=f"determinism watchdog {ordinal} PID",
        minimum=1,
    )
    armed = _require_int(
        watchdog.get("armed_monotonic_ns"),
        label=f"determinism watchdog {ordinal} armed time",
        minimum=1,
    )
    deadline = _require_int(
        watchdog.get("deadline_monotonic_ns"),
        label=f"determinism watchdog {ordinal} deadline",
        minimum=1,
    )
    ack = _require_int(
        watchdog.get("armed_ack_received_monotonic_ns"),
        label=f"determinism watchdog {ordinal} acknowledgement",
        minimum=1,
    )
    if deadline - armed != _WATCHDOG_NS or not armed <= ack < deadline:
        raise RuntimeError(f"determinism watchdog {ordinal} ordering is not exact")
    return watchdog


def _validate_numerics_progress(
    *,
    result: dict[str, Any],
    result_file: dict[str, Any],
    host_inputs: dict[str, Any],
    host_results: dict[str, Any],
    errors: dict[str, Any],
) -> dict[str, Any]:
    gated = result["numerics_once"]
    progress = gated.get("progress")
    if (
        not isinstance(progress, dict)
        or set(progress)
        != {
            "path",
            "protocol",
            "record_count",
            "bytes",
            "sha256",
            "mode",
            "directory_fsynced",
        }
        or progress.get("protocol") != "durable_fsync_jsonl_v1"
        or progress.get("record_count") != 10
        or progress.get("mode") != "0600"
        or progress.get("directory_fsynced") is not True
        or not isinstance(progress.get("path"), str)
    ):
        raise RuntimeError("determinism progress manifest is not exact")
    progress_path = Path(progress["path"])
    if progress_path.parent != Path(result_file["path"]).parent or progress_path.name != "progress.jsonl":
        raise RuntimeError("determinism progress is not the exact result sibling")
    progress_raw, progress_file = _read_private_bytes(progress_path, maximum_bytes=16 << 20)
    if (
        not progress_raw.endswith(b"\n")
        or progress.get("bytes") != len(progress_raw)
        or progress.get("sha256") != hashlib.sha256(progress_raw).hexdigest()
        or progress_file["mtime_ns"] >= result_file["mtime_ns"]
    ):
        raise RuntimeError("determinism progress bytes/digest/order are not exact")
    try:
        records = [_strict_json_loads(line) for line in progress_raw.splitlines()]
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("determinism progress is not strict JSONL") from error
    expected_events = ["host_inputs_ready"]
    for _program in _PROGRAM_ORDER:
        expected_events.extend(("dispatch_started", "dispatch_completed"))
    expected_events.append("numerics_completed")
    if (
        len(records) != 10
        or any(not isinstance(record, dict) for record in records)
        or [record.get("event") for record in records] != expected_events
    ):
        raise RuntimeError("determinism progress event protocol is not exact")

    common_keys = {
        "schema_version",
        "event",
        "mode",
        "probe_pid",
        "wall_time_ns",
        "monotonic_time_ns",
        "binding",
        "invocation_attempt_counts",
        "invocation_completion_counts",
    }
    zero = dict.fromkeys(_PROGRAM_ORDER, 0)
    attempts = dict(zero)
    completions = dict(zero)
    expected_binding = _expected_numerics_progress_binding(result)
    probe_pid = _require_int(records[0].get("probe_pid"), label="determinism probe PID", minimum=1)
    previous_wall = 0
    previous_monotonic = 0

    def validate_common(record: dict[str, Any]) -> None:
        nonlocal previous_wall, previous_monotonic
        if (
            type(record.get("schema_version")) is not int
            or record["schema_version"] != _EVIDENCE_SCHEMA_VERSION
            or record.get("mode") != "numerics_once"
            or record.get("probe_pid") != probe_pid
            or not _json_exact(record.get("binding"), expected_binding)
            or not _json_exact(record.get("invocation_attempt_counts"), attempts)
            or not _json_exact(record.get("invocation_completion_counts"), completions)
        ):
            raise RuntimeError("determinism progress common binding/counts are not exact")
        wall = _require_int(record.get("wall_time_ns"), label="determinism progress wall time", minimum=1)
        monotonic = _require_int(
            record.get("monotonic_time_ns"),
            label="determinism progress monotonic time",
            minimum=1,
        )
        if wall <= previous_wall or monotonic <= previous_monotonic:
            raise RuntimeError("determinism progress ordering is not strict")
        previous_wall = wall
        previous_monotonic = monotonic

    first = records[0]
    if set(first) != common_keys | {
        "compiled_programs",
        "program_order",
        "host_inputs",
        "watchdog_seconds_per_program",
    }:
        raise RuntimeError("determinism host-input progress schema is not exact")
    validate_common(first)
    if (
        not _json_exact(first.get("compiled_programs"), list(_PROGRAM_ORDER))
        or not _json_exact(first.get("program_order"), list(_PROGRAM_ORDER))
        or not _json_exact(first.get("host_inputs"), host_inputs)
        or first.get("watchdog_seconds_per_program") != _WATCHDOG_SECONDS
    ):
        raise RuntimeError("determinism host-input progress values are not exact")

    watchdogs = gated.get("watchdogs")
    if not isinstance(watchdogs, list) or len(watchdogs) != len(_PROGRAM_ORDER):
        raise RuntimeError("determinism watchdog count is not exact")
    validated_watchdogs: list[dict[str, Any]] = []
    record_index = 1
    for ordinal, program in enumerate(_PROGRAM_ORDER, start=1):
        watchdog = _validate_numerics_watchdog(
            watchdogs[ordinal - 1],
            ordinal=ordinal,
            program=program,
        )
        validated_watchdogs.append(watchdog)
        attempts[program] += 1
        started = records[record_index]
        record_index += 1
        if set(started) != common_keys | {
            "dispatch_ordinal",
            "program",
            "watchdog_seconds",
            "watchdog_armed_monotonic_ns",
            "watchdog_deadline_monotonic_ns",
            "watchdog_armed_ack_received_monotonic_ns",
        }:
            raise RuntimeError(f"determinism dispatch {ordinal} start schema is not exact")
        validate_common(started)
        if (
            started.get("dispatch_ordinal") != ordinal
            or started.get("program") != program
            or started.get("watchdog_seconds") != _WATCHDOG_SECONDS
            or not watchdog["armed_monotonic_ns"]
            <= watchdog["armed_ack_received_monotonic_ns"]
            <= started["monotonic_time_ns"]
            < watchdog["deadline_monotonic_ns"]
            or started.get("watchdog_armed_monotonic_ns") != watchdog["armed_monotonic_ns"]
            or started.get("watchdog_deadline_monotonic_ns") != watchdog["deadline_monotonic_ns"]
            or started.get("watchdog_armed_ack_received_monotonic_ns") != watchdog["armed_ack_received_monotonic_ns"]
        ):
            raise RuntimeError(f"determinism dispatch {ordinal} watchdog binding changed")

        completions[program] += 1
        completed = records[record_index]
        record_index += 1
        if set(completed) != common_keys | {
            "dispatch_ordinal",
            "program",
            "invocation_started_monotonic_ns",
            "invocation_completed_monotonic_ns",
            "invocation_elapsed_seconds",
            "device_results_released_before_completion",
            "result",
        }:
            raise RuntimeError(f"determinism dispatch {ordinal} completion schema is not exact")
        validate_common(completed)
        invocation_started = _require_int(
            completed.get("invocation_started_monotonic_ns"),
            label=f"determinism dispatch {ordinal} invocation start",
            minimum=1,
        )
        invocation_completed = _require_int(
            completed.get("invocation_completed_monotonic_ns"),
            label=f"determinism dispatch {ordinal} invocation completion",
            minimum=1,
        )
        elapsed = _require_number(
            completed.get("invocation_elapsed_seconds"),
            label=f"determinism dispatch {ordinal} elapsed time",
            minimum=0,
            maximum=_WATCHDOG_SECONDS,
        )
        if (
            completed.get("dispatch_ordinal") != ordinal
            or completed.get("program") != program
            or invocation_started < started["monotonic_time_ns"]
            or invocation_completed < invocation_started
            or invocation_completed >= watchdog["deadline_monotonic_ns"]
            or completed["monotonic_time_ns"] < invocation_completed
            or not math.isclose(
                elapsed,
                (invocation_completed - invocation_started) / 1_000_000_000,
                rel_tol=0,
                abs_tol=1e-15,
            )
            or completed.get("device_results_released_before_completion") is not True
            or not _json_exact(completed.get("result"), host_results[program])
        ):
            raise RuntimeError(f"determinism dispatch {ordinal} completion binding changed")

    terminal = records[record_index]
    if set(terminal) != common_keys | {
        "relative_l2_limit_exclusive",
        "output_cosine_limit_inclusive",
        "gradient_cosine_similarity_report_only",
        "errors",
        "host_inputs_unchanged",
        "passed",
    }:
        raise RuntimeError("determinism terminal progress schema is not exact")
    validate_common(terminal)
    exact_one = dict.fromkeys(_PROGRAM_ORDER, 1)
    if (
        record_index != len(records) - 1
        or attempts != exact_one
        or completions != exact_one
        or terminal.get("relative_l2_limit_exclusive") != _RELATIVE_L2_LIMIT
        or terminal.get("output_cosine_limit_inclusive") != _OUTPUT_COSINE_LIMIT
        or terminal.get("gradient_cosine_similarity_report_only") is not True
        or not _json_exact(terminal.get("errors"), errors)
        or terminal.get("host_inputs_unchanged") is not True
        or terminal.get("passed") is not True
    ):
        raise RuntimeError("determinism terminal progress binding is not exact")
    watchdog_pids = [watchdog["watchdog_pid"] for watchdog in validated_watchdogs]
    if len(set(watchdog_pids)) != len(watchdog_pids) or probe_pid in watchdog_pids:
        raise RuntimeError("determinism watchdog process evidence is not distinct")
    return {
        "file": progress_file,
        "record_count": len(records),
        "probe_pid": probe_pid,
        "watchdog_pids": watchdog_pids,
    }


def _expected_numerics_profile_command(
    *,
    result_path: Path,
    result: dict[str, Any],
) -> list[str]:
    gated = result["numerics_once"]
    profiler_parent = result["preflight"]["profiler_parent"]
    return [
        profiler_parent["parent_command_python"],
        result["source"]["probe"]["path"],
        "--allow-gpu",
        "--numerics-once",
        "--output",
        str(result_path),
        "--progress-output",
        gated["progress"]["path"],
        "--compile-evidence",
        gated["compile_evidence"]["path"],
        "--compile-profile-summary",
        gated["compile_profile_summary"]["path"],
        "--block-m",
        "16",
        "--block-physical-n",
        "64",
        "--block-k",
        "32",
        "--warmups",
        "0",
        "--iterations",
        "0",
    ]


def _validate_numerics_profile(
    *,
    profile_summary_path: Path,
    result_path: Path,
    result_file: dict[str, Any],
    result: dict[str, Any],
    progress: dict[str, Any],
) -> dict[str, Any]:
    if profile_summary_path.parent != result_path.parent or profile_summary_path.name != "telemetry.jsonl.summary.json":
        raise RuntimeError("determinism profile summary is not the exact result sibling")
    summary, summary_file = _read_private_json(profile_summary_path, maximum_bytes=1 << 20)
    telemetry_path = profile_summary_path.with_name("telemetry.jsonl")
    telemetry_raw, telemetry_file = _read_private_bytes(telemetry_path, maximum_bytes=64 << 20)
    if not telemetry_raw.endswith(b"\n"):
        raise RuntimeError("determinism telemetry is not complete JSONL")
    try:
        telemetry = [_strict_json_loads(line) for line in telemetry_raw.splitlines()]
    except (UnicodeDecodeError, ValueError) as error:
        raise RuntimeError("determinism telemetry is not strict JSONL") from error
    if (
        not isinstance(summary, dict)
        or not telemetry
        or not isinstance(telemetry[0], dict)
        or not (result_file["mtime_ns"] < telemetry_file["mtime_ns"] < summary_file["mtime_ns"])
    ):
        raise RuntimeError("determinism result/telemetry/summary ordering is not exact")
    manifest = telemetry[0]
    samples = telemetry[1:]
    if any(not isinstance(sample, dict) or sample.get("record_type") != "sample" for sample in samples):
        raise RuntimeError("determinism telemetry sample stream is not exact")
    profiler_parent = result["preflight"]["profiler_parent"]
    expected_command = _expected_numerics_profile_command(result_path=result_path, result=result)
    expected_limits = {
        "max_junction_temp_c": 90.0,
        "max_gpu_power_watts": 400.0,
        "max_vram_bytes": float(24 * 1024**3),
        "min_host_available_bytes": 0.0,
        "max_swap_bytes": float(8 * 1024**3),
    }
    packages = result["source"]["packages"]
    runtime = manifest.get("runtime")
    gpu = manifest.get("gpu")
    rocm_version = _current_rocm_version()
    if (
        manifest.get("record_type") != "manifest"
        or manifest.get("interval_seconds") != profiler_parent["interval_seconds"]
        or manifest.get("baseline_seconds") != profiler_parent["baseline_seconds"]
        or manifest.get("duration_seconds") is not None
        or manifest.get("timeout_seconds") != profiler_parent["timeout_seconds"]
        or manifest.get("sensor_grace_seconds") != profiler_parent["sensor_grace_seconds"]
        or manifest.get("terminate_included_on_safety") is not False
        or manifest.get("terminate_included_on_abort") is not False
        or manifest.get("explicit_processes") != {}
        or not _json_exact(manifest.get("safety_limits"), expected_limits)
        or manifest.get("command_recorded") is not True
        or manifest.get("passed_file_descriptor_count") != 0
        or not _json_exact(manifest.get("command"), expected_command)
        or not isinstance(runtime, dict)
        or runtime.get("python") != sys.version
        or runtime.get("platform") != platform.platform()
        or runtime.get("rocm") != rocm_version
        or runtime.get("jax") != packages["jax"]
        or runtime.get("jaxlib") != packages["jaxlib"]
        or runtime.get("jax_rocm_plugin") != packages["jax-rocm7-plugin"]
        or runtime.get("jax_rocm_pjrt") != packages["jax-rocm7-pjrt"]
        or runtime.get("script_sha256") != result["source"]["profiler"]["sha256"]
        or not _json_exact(
            runtime.get("accelerator_environment"),
            {
                "HIP_VISIBLE_DEVICES": "0",
                "JAX_PLATFORMS": "rocm",
                "XLA_FLAGS": _DISABLE_COMMAND_BUFFERS,
            },
        )
        or not isinstance(gpu, dict)
        or gpu.get("card") != "card1"
        or gpu.get("pci_bdf") != result["preflight"]["card_identity"]["pci_bdf"]
        or gpu.get("vendor_id") != "0x1002"
        or gpu.get("device_id") != "0x744c"
        or gpu.get("hwmon_name") != "amdgpu"
    ):
        raise RuntimeError("determinism profiler telemetry manifest is not exact")

    phases = [sample.get("phase") for sample in samples]
    baseline_count = phases.count("baseline")
    measured_count = phases.count("measured")
    if (
        any(phase not in {"baseline", "preflight", "measured"} for phase in phases)
        or phases.count("preflight") != 1
        or baseline_count <= 0
        or measured_count <= 0
        or summary.get("record_type") != "summary"
        or summary.get("status") != "completed"
        or type(summary.get("returncode")) is not int
        or summary["returncode"] != 0
        or summary.get("received_signal") is not None
        or summary.get("safety_violation") is not None
        or summary.get("kernel_driver_errors") is not None
        or summary.get("kernel_log_available") is not True
        or summary.get("baseline_samples") != baseline_count
        or summary.get("measured_samples") != measured_count
        or summary.get("samples") != len(samples)
        or len(samples) != baseline_count + measured_count + 1
    ):
        raise RuntimeError("determinism profiler did not complete safely and exactly")
    previous_wall = 0
    observed_command_pid = False
    for sample in samples:
        wall = _require_int(sample.get("wall_time_ns"), label="determinism telemetry wall time", minimum=1)
        if wall <= previous_wall:
            raise RuntimeError("determinism telemetry wall-time order is not strict")
        previous_wall = wall
        processes = sample.get("processes")
        command_process = processes.get("command") if isinstance(processes, dict) else None
        if isinstance(command_process, dict) and command_process.get("process_count", 0) > 0:
            if command_process.get("root_pid") != progress["probe_pid"]:
                raise RuntimeError("determinism telemetry command PID is not the probe PID")
            observed_command_pid = True
    summary_processes = summary.get("processes")
    summary_command = summary_processes.get("command") if isinstance(summary_processes, dict) else None
    if (
        not observed_command_pid
        or not isinstance(summary_command, dict)
        or summary_command.get("pid") != progress["probe_pid"]
    ):
        raise RuntimeError("determinism profiler process evidence is incomplete")

    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("determinism profiler metrics are missing")
    maxima: dict[str, float] = {}
    for name, limit in _SAFETY_METRIC_LIMITS.items():
        measured_values = [
            _require_number(sample[name], label=f"determinism telemetry {name}", minimum=0)
            for sample in samples
            if sample.get("phase") == "measured" and sample.get(name) is not None
        ]
        if not measured_values:
            raise RuntimeError(f"determinism telemetry has no measured {name}")
        observed = max(measured_values)
        metric = metrics.get(name)
        if (
            observed > float(limit)
            or not isinstance(metric, dict)
            or _require_number(
                metric.get("measured_max"),
                label=f"determinism summary {name} maximum",
                minimum=0,
            )
            != observed
        ):
            raise RuntimeError(f"determinism telemetry/summary {name} is unsafe")
        maxima[name] = observed
    return {
        "summary": summary_file,
        "telemetry": {
            **telemetry_file,
            "record_count": len(telemetry),
            "sample_count": len(samples),
            "command_sha256": hashlib.sha256(
                json.dumps(expected_command, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
        "runtime": runtime,
        "gpu": gpu,
        "observed_maxima": maxima,
        "probe_pid": progress["probe_pid"],
        "profiler_pid": profiler_parent["parent_pid"],
        "profiler_command_sha256": profiler_parent["parent_command_sha256"],
    }


def _validate_numerics_determinism_run(
    *,
    result_path: Path,
    profile_summary_path: Path,
    repo: Path,
    benchmark_result: dict[str, Any],
    benchmark_gated: dict[str, Any],
) -> dict[str, Any]:
    result, result_file = _read_private_json(result_path, maximum_bytes=64 << 20)
    expected_root_keys = {
        "schema_version",
        "mode",
        "qualification_scope",
        "authorizes_default_model_enablement",
        "recommend_for_opt_in_model_integration",
        "passed",
        "contract",
        "preflight",
        "device",
        "geometry",
        "invocation_contract",
        "compilation",
        "measurement",
        "performance_gate",
        "numerics",
        "determinism",
        "source",
        "numerics_once",
        "postflight",
    }
    if (
        not isinstance(result, dict)
        or set(result) != expected_root_keys
        or type(result.get("schema_version")) is not int
        or result["schema_version"] != _EVIDENCE_SCHEMA_VERSION
        or result.get("mode") != "numerics_once"
        or result.get("qualification_scope") != "isolated_stage_only"
        or result.get("authorizes_default_model_enablement") is not False
        or result.get("recommend_for_opt_in_model_integration") is not False
        or result.get("passed") is not True
        or not _json_exact(result.get("contract"), benchmark_result["contract"])
        or not _json_exact(result.get("geometry"), benchmark_result["geometry"])
        or not _json_exact(result.get("device"), benchmark_result["device"])
        or not _json_exact(
            result.get("postflight"),
            {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []},
        )
    ):
        raise RuntimeError("determinism numerics result is not exact")
    if not _json_exact(result.get("source"), benchmark_result["source"]):
        raise RuntimeError("determinism source/runtime manifest changed")
    source_only, git, packages = _validate_source_and_runtime(result["source"], repo=repo)
    preflight = _validate_preflight(
        result.get("preflight"),
        mode="numerics_once",
        repo=repo,
        source_only=source_only,
    )
    for name in ("environment", "hardware", "card_identity", "safety_source"):
        if not _json_exact(result["preflight"].get(name), benchmark_result["preflight"].get(name)):
            raise RuntimeError(f"determinism {name} preflight changed")
    _validate_compilation(result.get("compilation"))
    exact_one = dict.fromkeys(_PROGRAM_ORDER, 1)
    invocation = result.get("invocation_contract")
    if (
        not isinstance(invocation, dict)
        or not _json_exact(invocation.get("per_program_executable_invocations"), exact_one)
        or not _json_exact(invocation.get("per_program_executable_completions"), exact_one)
        or invocation.get("reference_executable_invocations") != 2
        or invocation.get("candidate_executable_invocations") != 2
        or invocation.get("total_executable_invocations") != 4
        or invocation.get("compile_only_zero_candidate_reference_executable_invocations") is not False
    ):
        raise RuntimeError("determinism invocation contract is not exact")
    if (
        not _json_exact(
            result.get("measurement"),
            {
                "executed": False,
                "performance_measured": False,
                "warmups": 0,
                "iterations": 0,
                "reason": "numerics-once dispatch durations are safety evidence only",
            },
        )
        or not _json_exact(
            result.get("performance_gate"),
            {
                "executed": False,
                "passed": None,
                "reason": "numerics-once is not a performance qualification",
            },
        )
        or not _json_exact(
            result.get("determinism"),
            {
                "executed": False,
                "passed": None,
                "reason": "numerics-once invokes each program exactly once",
            },
        )
    ):
        raise RuntimeError("determinism inherited non-performance gates are not exact")

    numerics = result.get("numerics")
    errors = numerics.get("errors") if isinstance(numerics, dict) else None
    host_results = numerics.get("host_results") if isinstance(numerics, dict) else None
    _validate_error_manifest(errors)
    _validate_host_result_manifest(host_results)
    if (
        not isinstance(numerics, dict)
        or set(numerics)
        != {
            "executed",
            "reference_compared",
            "passed",
            "relative_l2_limit_exclusive",
            "output_cosine_limit_inclusive",
            "gradient_cosine_similarity_report_only",
            "errors",
            "host_results",
        }
        or numerics.get("executed") is not True
        or numerics.get("reference_compared") is not True
        or numerics.get("passed") is not True
        or numerics.get("relative_l2_limit_exclusive") != _RELATIVE_L2_LIMIT
        or numerics.get("output_cosine_limit_inclusive") != _OUTPUT_COSINE_LIMIT
        or numerics.get("gradient_cosine_similarity_report_only") is not True
    ):
        raise RuntimeError("determinism numerics manifest is not exact")

    gated = result.get("numerics_once")
    if (
        not isinstance(gated, dict)
        or set(gated)
        != {
            "compile_evidence",
            "compile_profile_summary",
            "progress",
            "watchdogs",
            "host_inputs",
            "host_inputs_unchanged",
            "host_results",
            "invocation_attempt_counts",
            "invocation_completion_counts",
            "source_and_git_unchanged_across_dispatch",
        }
        or not _json_exact(gated.get("compile_evidence"), benchmark_gated["compile_evidence"])
        or not _json_exact(gated.get("compile_profile_summary"), benchmark_gated["compile_profile_summary"])
        or gated.get("host_inputs_unchanged") is not True
        or not _json_exact(gated.get("host_results"), host_results)
        or not _json_exact(gated.get("invocation_attempt_counts"), exact_one)
        or not _json_exact(gated.get("invocation_completion_counts"), exact_one)
        or gated.get("source_and_git_unchanged_across_dispatch") is not True
    ):
        raise RuntimeError("determinism guarded numerics manifest is not exact")
    host_inputs = _validate_host_input_manifest(gated.get("host_inputs"))
    if not _json_exact(host_inputs, benchmark_gated.get("host_inputs")):
        raise RuntimeError("determinism input manifest changed from benchmark evidence")
    progress = _validate_numerics_progress(
        result=result,
        result_file=result_file,
        host_inputs=host_inputs,
        host_results=host_results,
        errors=errors,
    )
    profile = _validate_numerics_profile(
        profile_summary_path=profile_summary_path,
        result_path=result_path,
        result_file=result_file,
        result=result,
        progress=progress,
    )
    scope = preflight["scope"]
    return {
        "result": result_file,
        "progress": progress,
        "profile": profile,
        "source": result["source"],
        "git": git,
        "packages": packages,
        "device": result["device"],
        "geometry": result["geometry"],
        "host_inputs": host_inputs,
        "candidate_hashes": _candidate_result_hashes(host_results),
        "stable_preflight": {
            name: result["preflight"][name] for name in ("environment", "hardware", "card_identity", "safety_source")
        },
        "profiler_timings": preflight["profiler_timings"],
        "profiler_parent": result["preflight"]["profiler_parent"],
        "scope": scope,
    }


def _manifest_matches_file(compact: Any, actual: dict[str, Any], *, label: str) -> None:
    if not isinstance(compact, dict):
        raise RuntimeError(f"{label} compact manifest is missing")
    for name in ("path", "bytes", "sha256", "mode", "device", "inode", "mtime_ns"):
        if not _json_exact(compact.get(name), actual.get(name)):
            raise RuntimeError(f"{label} compact manifest changed: {name}")


def _compare_determinism_runs(
    runs: tuple[dict[str, Any], dict[str, Any]],
    *,
    benchmark_gated: dict[str, Any],
) -> dict[str, Any]:
    first, second = runs
    _manifest_matches_file(
        benchmark_gated.get("numerics_evidence"),
        first["result"],
        label="first determinism result",
    )
    _manifest_matches_file(
        benchmark_gated.get("numerics_profile_summary"),
        first["profile"]["summary"],
        label="first determinism profile",
    )
    _manifest_matches_file(
        benchmark_gated.get("numerics_evidence", {}).get("progress"),
        first["progress"]["file"],
        label="first determinism progress",
    )
    _manifest_matches_file(
        benchmark_gated.get("numerics_profile_summary", {}).get("telemetry"),
        first["profile"]["telemetry"],
        label="first determinism telemetry",
    )
    for name in (
        "source",
        "git",
        "packages",
        "device",
        "geometry",
        "host_inputs",
        "stable_preflight",
        "profiler_timings",
    ):
        if not _json_exact(first[name], second[name]):
            raise RuntimeError(f"determinism runs differ in {name}")
    if not _json_exact(first["profile"]["runtime"], second["profile"]["runtime"]):
        raise RuntimeError("determinism runs differ in profiler runtime")
    if not _json_exact(first["profile"]["gpu"], second["profile"]["gpu"]):
        raise RuntimeError("determinism runs differ in profiler GPU identity")
    expected_hash_positions = {f"{program}/{name}" for program, name in _CANDIDATE_HASH_POSITIONS}
    for ordinal, run in enumerate(runs, start=1):
        candidate_hashes = run.get("candidate_hashes")
        if not isinstance(candidate_hashes, dict) or set(candidate_hashes) != expected_hash_positions:
            raise RuntimeError(f"determinism run {ordinal} candidate BF16 result hash positions are not exact")
        if candidate_hashes["candidate_forward/output"] != candidate_hashes["candidate_forward_and_vjp/output"]:
            raise RuntimeError(f"determinism run {ordinal} candidate primal BF16 outputs differ")
    if not _json_exact(first["candidate_hashes"], second["candidate_hashes"]):
        raise RuntimeError("candidate BF16 result hashes are not deterministic")

    normalized_artifacts = [
        artifact
        for run in runs
        for artifact in (
            run["result"],
            run["progress"]["file"],
            run["profile"]["summary"],
            run["profile"]["telemetry"],
        )
    ]
    paths = [manifest["path"] for manifest in normalized_artifacts]
    identities = [(manifest["device"], manifest["inode"]) for manifest in normalized_artifacts]
    if len(set(paths)) != len(paths) or len(set(identities)) != len(identities):
        raise RuntimeError("determinism artifacts are not distinct")
    process_ids = [
        process_id
        for run in runs
        for process_id in (
            run["profile"]["profiler_pid"],
            run["progress"]["probe_pid"],
            *run["progress"]["watchdog_pids"],
        )
    ]
    if (
        len(set(process_ids)) != len(process_ids)
        or first["profile"]["profiler_command_sha256"] == second["profile"]["profiler_command_sha256"]
        or first["scope"]["scope_unit"] == second["scope"]["scope_unit"]
        or (
            first["scope"]["cgroup_kill_device"],
            first["scope"]["cgroup_kill_inode"],
        )
        == (
            second["scope"]["cgroup_kill_device"],
            second["scope"]["cgroup_kill_inode"],
        )
    ):
        raise RuntimeError("determinism process/scope evidence is not distinct")
    return {
        "provided": True,
        "passed": True,
        "attested": True,
        "protocol": "two_guarded_processes_five_candidate_bf16_sha256_v1",
        "candidate_hash_positions": [f"{program}/{name}" for program, name in _CANDIDATE_HASH_POSITIONS],
        "candidate_bf16_sha256": first["candidate_hashes"],
        "runs": [
            {
                "result": run["result"],
                "progress": run["progress"]["file"],
                "profile_summary": run["profile"]["summary"],
                "profile_telemetry": run["profile"]["telemetry"],
                "probe_pid": run["progress"]["probe_pid"],
                "watchdog_pids": run["progress"]["watchdog_pids"],
                "profiler_pid": run["profile"]["profiler_pid"],
                "profiler_command_sha256": run["profile"]["profiler_command_sha256"],
                "scope_unit": run["scope"]["scope_unit"],
                "cgroup_kill_device": run["scope"]["cgroup_kill_device"],
                "cgroup_kill_inode": run["scope"]["cgroup_kill_inode"],
            }
            for run in runs
        ],
    }


def _validate_cross_process_determinism(
    *,
    pairs: tuple[tuple[Path, Path], tuple[Path, Path]],
    repo: Path,
    benchmark_result: dict[str, Any],
    benchmark_gated: dict[str, Any],
) -> dict[str, Any]:
    if len(pairs) != 2 or any(len(pair) != 2 for pair in pairs):
        raise ValueError("determinism evidence requires exactly two result/profile pairs")
    runs = tuple(
        _validate_numerics_determinism_run(
            result_path=Path(result_path),
            profile_summary_path=Path(profile_path),
            repo=repo,
            benchmark_result=benchmark_result,
            benchmark_gated=benchmark_gated,
        )
        for result_path, profile_path in pairs
    )
    return _compare_determinism_runs((runs[0], runs[1]), benchmark_gated=benchmark_gated)


def _validate_raw_result(
    *,
    result_path: Path,
    expected_mode: str | None,
    repo: Path,
) -> tuple[
    str,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any],
]:
    raw_result, result_file = _read_private_json(result_path, maximum_bytes=64 << 20)
    if not isinstance(raw_result, dict):
        raise RuntimeError("benchmark result root is not an object")
    mode = raw_result.get("mode")
    if mode not in _BENCHMARK_MODE_COUNTS or (expected_mode is not None and mode != expected_mode):
        raise RuntimeError("benchmark result mode is not exact")
    if (
        type(raw_result.get("schema_version")) is not int
        or raw_result["schema_version"] != _EVIDENCE_SCHEMA_VERSION
        or raw_result.get("passed") is not False
        or raw_result.get("probe_completed") is not True
        or raw_result.get("profile_attested") is not False
        or raw_result.get("qualification_scope") != "isolated_stage_only"
        or raw_result.get("authorizes_default_model_enablement") is not False
        or raw_result.get("recommend_for_opt_in_model_integration") is not False
        or not _json_exact(raw_result.get("contract"), _expected_contract())
        or not _json_exact(raw_result.get("geometry"), _exact_geometry())
        or not _json_exact(
            raw_result.get("postflight"),
            {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []},
        )
    ):
        raise RuntimeError("benchmark raw result is not an exact completed probe")
    device = raw_result.get("device")
    if (
        not isinstance(device, dict)
        or set(device)
        != {
            "backend",
            "device_kind",
            "architecture",
            "platform_version",
        }
        or device.get("backend") != "gpu"
        or device.get("device_kind") != "Radeon RX 7900 XTX"
        or device.get("architecture") != "gfx1100"
        or not isinstance(device.get("platform_version"), str)
        or "rocm" not in device["platform_version"].lower()
    ):
        raise RuntimeError("benchmark device identity is not exact")
    source_only, git, packages = _validate_source_and_runtime(raw_result.get("source"), repo=repo)
    _validate_preflight(
        raw_result.get("preflight"),
        mode=mode,
        repo=repo,
        source_only=source_only,
    )
    _validate_compilation(raw_result.get("compilation"))
    gated = raw_result.get(mode)
    if not isinstance(gated, dict):
        raise RuntimeError("benchmark guarded detail is missing")
    smoke_attestation: dict[str, Any] | None = None
    if mode == "benchmark":
        candidate = gated.get("smoke_attestation")
        if not isinstance(candidate, dict):
            raise RuntimeError("full benchmark smoke attestation is missing")
        smoke_attestation = candidate
    elif "smoke_attestation" in gated:
        raise RuntimeError("smoke benchmark cannot contain a prior smoke attestation")
    _validate_prior_chain(
        gated,
        contract=raw_result["contract"],
        source=raw_result["source"],
        geometry=raw_result["geometry"],
        device=device,
    )
    progress = _validate_progress(
        mode=mode,
        raw_result=raw_result,
        result_file=result_file,
        gated=gated,
        smoke_attestation=smoke_attestation,
    )
    warmups, measurements = _BENCHMARK_MODE_COUNTS[mode]
    exact_counts = dict.fromkeys(_PROGRAM_ORDER, warmups + measurements)
    invocation = raw_result.get("invocation_contract")
    if (
        not isinstance(invocation, dict)
        or not _json_exact(invocation.get("per_program_executable_invocations"), exact_counts)
        or not _json_exact(invocation.get("per_program_executable_completions"), exact_counts)
        or invocation.get("reference_executable_invocations") != 2 * (warmups + measurements)
        or invocation.get("candidate_executable_invocations") != 2 * (warmups + measurements)
        or invocation.get("total_executable_invocations") != 4 * (warmups + measurements)
        or invocation.get("compile_only_zero_candidate_reference_executable_invocations") is not False
    ):
        raise RuntimeError("benchmark invocation contract is not exact")
    if (
        not _json_exact(gated.get("invocation_attempt_counts"), exact_counts)
        or not _json_exact(gated.get("invocation_completion_counts"), exact_counts)
        or gated.get("raw_samples_only") is not True
        or gated.get("performance_qualification") is not False
        or not _json_exact(gated.get("progress"), raw_result[mode]["progress"])
    ):
        raise RuntimeError("benchmark guarded terminal evidence is not exact")
    numerics = raw_result.get("numerics")
    determinism = raw_result.get("determinism")
    performance_gate = raw_result.get("performance_gate")
    if (
        not isinstance(numerics, dict)
        or numerics.get("executed") is not False
        or numerics.get("passed") is not True
        or numerics.get("prior_numerics_evidence_verified") is not True
        or not _json_exact(numerics.get("prior_numerics_evidence"), gated["numerics_evidence"])
        or not _json_exact(
            numerics.get("prior_numerics_profile_summary"),
            gated["numerics_profile_summary"],
        )
        or not isinstance(determinism, dict)
        or determinism.get("executed") is not False
        or not isinstance(performance_gate, dict)
        or performance_gate.get("executed") is not False
        or performance_gate.get("passed") is not None
    ):
        raise RuntimeError("benchmark inherited gates are not exact")
    return (
        mode,
        raw_result,
        result_file,
        gated,
        smoke_attestation,
        {
            "source": source_only,
            "git": git,
            "packages": packages,
            "progress": progress,
        },
    )


def validate_and_attest_benchmark(
    result_path: Path,
    profile_summary_path: Path,
    output_path: Path | None = None,
    expected_mode: str | None = None,
    write_output: bool = True,
    determinism_pairs: tuple[tuple[Path, Path], tuple[Path, Path]] | None = None,
) -> dict[str, Any]:
    """Validate one benchmark rung without importing JAX or opening the GPU."""
    result_path = Path(result_path)
    profile_summary_path = Path(profile_summary_path)
    if expected_mode is not None and expected_mode not in _BENCHMARK_MODE_COUNTS:
        raise ValueError("expected_mode is not a guarded benchmark mode")
    if determinism_pairs is not None and (
        len(determinism_pairs) != 2 or any(len(pair) != 2 for pair in determinism_pairs)
    ):
        raise ValueError("determinism evidence requires exactly two result/profile pairs")
    repo = Path(__file__).resolve(strict=True).parent.parent
    (
        mode,
        raw_result,
        result_file,
        gated,
        embedded_smoke,
        validated,
    ) = _validate_raw_result(
        result_path=result_path,
        expected_mode=expected_mode,
        repo=repo,
    )
    if determinism_pairs is not None and mode != "benchmark":
        raise ValueError("cross-process determinism can qualify only the full benchmark")
    smoke_attestation: dict[str, Any] | None = None
    if mode == "benchmark":
        assert embedded_smoke is not None
        result_manifest = embedded_smoke.get("result")
        profile_manifest = embedded_smoke.get("profile")
        if (
            not isinstance(result_manifest, dict)
            or not isinstance(result_manifest.get("path"), str)
            or not isinstance(profile_manifest, dict)
            or not isinstance(profile_manifest.get("summary"), dict)
            or not isinstance(profile_manifest["summary"].get("path"), str)
        ):
            raise RuntimeError("embedded smoke attestation paths are missing")
        smoke_attestation = validate_and_attest_benchmark(
            Path(result_manifest["path"]),
            Path(profile_manifest["summary"]["path"]),
            output_path=None,
            expected_mode="benchmark_smoke",
            write_output=False,
        )
        if not _json_exact(embedded_smoke, smoke_attestation):
            raise RuntimeError("embedded smoke attestation changed or is not exact")
    profile = _validate_profile(
        profile_summary_path=profile_summary_path,
        result_path=result_path,
        result_file=result_file,
        raw_result=raw_result,
        mode=mode,
        repo=repo,
        gated=gated,
        smoke_attestation=smoke_attestation,
    )
    performance = _validate_measurement(
        mode=mode,
        raw_result=raw_result,
        progress=validated["progress"],
    )
    if determinism_pairs is None:
        determinism = {
            "provided": False,
            "passed": None,
            "attested": False,
            "reason": "two guarded numerics-once result/profile pairs were not provided",
        }
    else:
        determinism = _validate_cross_process_determinism(
            pairs=determinism_pairs,
            repo=repo,
            benchmark_result=raw_result,
            benchmark_gated=gated,
        )
    determinism_attested = bool(determinism["attested"] is True)
    timing_qualified = bool(mode == "benchmark" and performance["performance_gates_passed"])
    performance_qualified = bool(timing_qualified and determinism_attested)
    performance = {
        **performance,
        "determinism_attested": determinism_attested,
    }
    attestor_path = Path(__file__).resolve(strict=True)
    composite = {
        "schema_version": _EVIDENCE_SCHEMA_VERSION,
        "attestation_type": _ATTESTATION_TYPE,
        "mode": mode,
        "passed": True,
        "attestation_passed": True,
        "timing_qualified": timing_qualified,
        "performance_qualified": performance_qualified,
        "determinism_attested": determinism_attested,
        "probe_completed": True,
        "profile_attested": True,
        "authorizes_default_model_enablement": False,
        "recommend_for_opt_in_integration": performance_qualified,
        "result": result_file,
        "progress": {
            **validated["progress"]["file"],
            "record_count": validated["progress"]["record_count"],
        },
        "profile": profile,
        "performance": performance,
        "determinism": determinism,
        "source": validated["source"],
        "git": validated["git"],
        "packages": validated["packages"],
        "prior_evidence": {
            "compile": gated["compile_evidence"],
            "compile_profile": gated["compile_profile_summary"],
            "numerics": gated["numerics_evidence"],
            "numerics_profile": gated["numerics_profile_summary"],
        },
        "numerics_errors": gated["numerics_evidence"]["errors"],
        "attestor": {
            "path": str(attestor_path),
            "sha256": hashlib.sha256(attestor_path.read_bytes()).hexdigest(),
        },
        "smoke_attestation": smoke_attestation,
    }
    if write_output:
        if output_path is None:
            raise ValueError("write_output=True requires output_path")
        _write_private_json(Path(output_path), composite)
    elif output_path is not None:
        raise ValueError("output_path is only valid when write_output=True")
    return composite


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--profile-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-mode", choices=tuple(_BENCHMARK_MODE_COUNTS), required=True)
    parser.add_argument(
        "--determinism-result",
        action="append",
        default=[],
        type=Path,
        help="guarded numerics-once result; repeat exactly twice, prior evidence first",
    )
    parser.add_argument(
        "--determinism-profile-summary",
        action="append",
        default=[],
        type=Path,
        help="matching guarded profiler summary; repeat exactly twice",
    )
    args = parser.parse_args(argv)
    if len(args.determinism_result) != len(args.determinism_profile_summary) or len(args.determinism_result) not in {
        0,
        2,
    }:
        parser.error(
            "determinism evidence requires exactly two --determinism-result and "
            "two --determinism-profile-summary arguments"
        )
    args.determinism_pairs = (
        tuple(zip(args.determinism_result, args.determinism_profile_summary, strict=True))
        if args.determinism_result
        else None
    )
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = validate_and_attest_benchmark(
        args.result,
        args.profile_summary,
        args.output,
        expected_mode=args.expected_mode,
        write_output=True,
        determinism_pairs=args.determinism_pairs,
    )
    print(json.dumps(payload, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
