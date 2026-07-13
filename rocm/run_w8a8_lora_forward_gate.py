#!/usr/bin/env python3
"""Supervise one exact W8A8+LoRA compile or one-shot runtime rung.

The default invocation is an abstract refusal and imports no JAX.  The ROCm
path creates a fresh private run directory, holds SkyRL's global ROCm lock,
captures an exact suspended/unowned RX 7900 XTX baseline, wraps the child
directly in ``profile_rocm.py``, then requires the exact three-sample idle
handoff and a final whole-boot journal check before releasing the lock.

All limits and paths are fixed for the first ``M=3,K=64,N=17`` rung.  The
runtime phase has a separate five-second dispatch watchdog and exactly one
compiled invocation.  This controller exposes no shape, repetition, warmup,
backward, graph, or replay option.  The child separately refuses a hardware
power cap above 315 W.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import signal
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

_COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="
_EXPECTED_DEVICE_ID = "0x744c"
_EXPECTED_SUBSYSTEM_VENDOR_ID = "0x1043"
_EXPECTED_SUBSYSTEM_DEVICE_ID = "0x0506"
_PROFILE_TIMEOUT_SECONDS = 300.0
_PROFILE_INTERVAL_SECONDS = 0.05
_PROFILE_BASELINE_SECONDS = 5.0
_PROFILE_SENSOR_GRACE_SECONDS = 15.0
_PROFILE_TERMINATE_GRACE_SECONDS = 0.5
_OUTER_WATCHDOG_SECONDS = 330.0
_OUTER_TERMINATE_GRACE_SECONDS = 10.0
_DISPATCH_WATCHDOG_SECONDS = 5.0
_DISPATCH_KILL_REAP_SECONDS = 1.0
_EXECUTE_FINAL_SCOPE_QUERY_SECONDS = 0.1
_MAX_TELEMETRY_DISPATCH_BRACKET_SECONDS = 0.20
_MAX_JUNCTION_TEMP_C = 90.0
_MAX_GPU_POWER_WATTS = 315.0
_MAX_VRAM_GIB = 2.0
_MIN_HOST_AVAILABLE_GIB = 0.0
_MAX_SWAP_GIB = 8.0
_HANDOFF_TIMEOUT_SECONDS = 120.0
_EXPECTED_VRAM_TOTAL_BYTES = 25_753_026_560
_EXPECTED_KERNEL_NAME = "skyrl_qwen35_w8a8_lora_forward"
_EXPECTED_NESTED_ELF_BYTES = 8_440
_EXPECTED_NESTED_ELF_SHA256 = (
    "606a80a508317af303966e5c2ca357d138d08828949c0dbfdcd73ccde1726389"
)
_EXPECTED_NESTED_ELF_TARGET = "amdgcn--amdhsa-amdgiz-gfx1100"
_EXPECTED_SYSTEMD_RUN_SHA256 = (
    "49f0bf95eb8a781b93853bf9fc981b4929dd0009f55a3e6db95534c0a2d11716"
)
_EXPECTED_SYSTEMCTL_SHA256 = (
    "7ba82b5ba146759c710e1b80fadaa3fdbc0f9b85c8fb2c8c3196b7b1a0037ef8"
)
_EXPECTED_PROFILE_RUNTIME_SHA256 = {
    "psutil/__init__.py": "d138a5786b163b56ba86ea0b2d5589dfca37e1bcdf8de1057fe1e933d6ab808a",
    "psutil/_common.py": "1bbce9fd97e6f5439a0d385f28401f527c5a9d978d9d062593d7ef6fe215fc75",
    "psutil/_ntuples.py": "9b8e8b938abae2786a515f686f4d8132218be897eb6b459b9324ed17d88a31c7",
    "psutil/_pslinux.py": "a8a09bab87848075b691c46112b00692745785a8c9d5a2c87037ae124b469749",
    "psutil/_psposix.py": "288887b58f5939e3b00e021f6e7b490c2f7f4852804a359918ab7d9c6fb39839",
    "psutil/_psutil_linux.abi3.so": "8085c743a77f6059be6d18d54317ccce2e715425bb5f15aa8835770463d6b9fd",
}
_EXPECTED_HOST_MANIFESTS = [
    {
        "name": "x",
        "shape": [3, 64],
        "dtype": "bfloat16",
        "nbytes": 384,
        "sha256": "97f28a640f7747f5a18725c2dffc67baff13068673b2edf0d600671d6132c765",
    },
    {
        "name": "weight_codes",
        "shape": [64, 17],
        "dtype": "int8",
        "nbytes": 1088,
        "sha256": "7b2d71ade420170140481aea83796639238df7cb0cd4c85ecc12f197654dd5be",
    },
    {
        "name": "weight_scales",
        "shape": [1, 17],
        "dtype": "bfloat16",
        "nbytes": 34,
        "sha256": "5aa21b6c4171062f63d5cae19f23d9d2874e3a173ce0238454e156cca6f843fb",
    },
    {
        "name": "lora_a",
        "shape": [64, 8],
        "dtype": "bfloat16",
        "nbytes": 1024,
        "sha256": "d293daaf4d695061960a610bc0c03cebcb45907094cfbfecb4cef28a1dccf857",
    },
    {
        "name": "lora_b",
        "shape": [8, 17],
        "dtype": "bfloat16",
        "nbytes": 272,
        "sha256": "1e9f8813f39862777be04e019879000d772b7c8ef2e87768deef64cc2134200d",
    },
    {
        "name": "lora_scaling",
        "shape": [],
        "dtype": "float32",
        "nbytes": 4,
        "sha256": "9a8208635e00348ab64aac2b759e76391fd47089e9a749bbcec770d9eb5c6421",
    },
]
_EXPECTED_HOST_OUTPUT = {
    "name": "expected",
    "shape": [3, 17],
    "dtype": "bfloat16",
    "nbytes": 102,
    "sha256": "964b0c1fe5f5c4cdbd60658717fee50a835006adf03011717b4914061ab4f88f",
}
_EXPECTED_STACK_VERSIONS = {
    "amdgpu": "6.16.13",
    "jax": "0.10.2",
    "jax-rocm7-pjrt": "0.10.2",
    "jax-rocm7-plugin": "0.10.2",
    "jaxlib": "0.10.2",
    "ml-dtypes": "0.5.4",
    "numpy": "2.3.5",
    "rocm": "7.2.4",
}
_EXPECTED_JAX_RUNTIME_SHA256 = {
    "jax_plugins/xla_rocm7/xla_rocm_plugin.so": "8e8319b48fcc771b5221771dc3da0d24bc6950c872ee07ac6f5200743b3b4fa7",
    "jax_rocm7_plugin/_triton.so": "168b21fa4d9bfd27c93c61a25ad0d9b4f2bbc718348c3c33251e3df0bc1ac6c8",
    "jax_rocm7_plugin/rocm_plugin_extension.so": "555f2d4915ae7e19277a9435d3f6fd55cb9137e981fa5431648cbe8ca8ff3f1c",
    "jaxlib/_jax.so": "1b8ba599253bddc480606ab30d6a47731ac9fe3be4e6082dec7a85e82778857f",
}
_ISOLATED_PROFILE_BOOTSTRAP = """\
import hashlib
import json
import pathlib
import sys

site_packages, repo, profile, expected_sha256, runtime_json, cache_root, *arguments = sys.argv[1:]
expected_stdlib = [
    "/usr/lib/python312.zip",
    "/usr/lib/python3.12",
    "/usr/lib/python3.12/lib-dynload",
]
if sys.path != expected_stdlib:
    raise RuntimeError("refusing non-standard isolated Python import path")
cache_path = pathlib.Path(cache_root)
if (
    sys.pycache_prefix != cache_root
    or cache_path.is_symlink()
    or not cache_path.is_dir()
    or cache_path.resolve() != cache_path
    or any(cache_path.iterdir())
):
    raise RuntimeError("refusing active or ambiguous Python bytecode cache")
payload = pathlib.Path(profile).read_bytes()
if hashlib.sha256(payload).hexdigest() != expected_sha256:
    raise RuntimeError("refusing changed profiler source")
site_root = pathlib.Path(site_packages)
package_root = site_root / "psutil"
repo_root = pathlib.Path(repo)
if (
    package_root.is_symlink()
    or not package_root.is_dir()
    or package_root.resolve() != package_root
    or (site_root / "psutil.py").exists()
    or (repo_root / "psutil.py").exists()
    or (repo_root / "psutil").exists()
):
    raise RuntimeError("refusing ambiguous profiler runtime")
runtime = json.loads(runtime_json)
for relative, expected in runtime.items():
    path = site_root / relative
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("refusing changed profiler runtime")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("refusing changed profiler runtime")
sys.path.extend([repo, site_packages])
sys.argv = [profile, *arguments]
namespace = {
    "__name__": "__main__",
    "__file__": profile,
    "__package__": None,
    "__cached__": None,
    "__builtins__": __builtins__,
}
exec(compile(payload, profile, "exec"), namespace)
"""
_EXPECTED_SOURCE_SHA256 = {
    "child": "4916a81b56fc161041404c69cf5af5298f1d581b080e82c2cd6d6bca4d56157a",
    "isa_inspector": "1a46c05700e5fbe6b5747632cfb23e872919f3b39fa77119c701836ce92cda07",
    "safety": "7ad79b9b9b54089add72dff65ea18505a794c51f0c4bafe231fbd3b745f23ba6",
    "handoff": "4d6c7e665219ce125d840e68b0e2cb7e8b1b5f98552ff65a2d07a153b3cd1392",
    "profiler": "ed230758101a2a540b3a09e7f84ac92256d2bb41c70dbc399b9466fe0b979684",
    "kernel": "1dec42508856d5e38a656bc4292dcb7033d4437bf8458f7b88030be4b4b4490a",
    "quantized_reference": "91a89055ea18b16d64bd32c2eac32a2361e52b4a56b23721b41ffeb413ccc0de",
    "package_skyrl": "667d4a15b970b851e20d17510224670c14646cfb6d5a1e388ca6b9cc6da8bf41",
    "package_tx": "a7abb3e76d66df1f4472bb7a02b032ef31b959ca937fd351637b4e9b4a8fa95a",
    "package_kernels": "40abe638c7726fe5680b7c88321042016a0f695d86acfbef52337421e7257c1a",
    "package_rocm": "6d12a789cf1108538a04fbacd0b38a15dbcb8255cd0ca0fadf5a76c4191a4cfd",
    "sealed_loader": "66b868b7909a2279d5ddca0e1582f8563e8097723e970d09fce733aef2ba425a",
    "package_top_rocm": "4db843597a8ef2b43b84c1e1d5b9f4e5dcd39d2fd8eead3dde07ac02d77d4ad9",
}
_RUNTIME_PROBE_PROTOCOL = (
    ("manifest", None),
    ("static_preflight", None),
    ("hardware_preflight", None),
    ("host_oracle", None),
    ("backend_ready", None),
    ("journal_checkpoint", "after_backend_initialization"),
    ("lowered", None),
    ("compiled_unreleased", None),
    ("fresh_isa_qualification", None),
    ("executable_released", None),
    ("input_device_put", None),
    ("journal_checkpoint", "after_input_device_put"),
    ("dispatch_preflight", None),
    ("dispatch_started", None),
    ("journal_checkpoint", "after_candidate_dispatch_attempt"),
    ("dispatch", None),
    ("device_get", None),
    ("journal_checkpoint", "after_candidate_device_get"),
    ("numerical_validation", None),
    ("completed", None),
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _emit(record: dict[str, Any], output: TextIO) -> None:
    output.write(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    output.flush()


def _strict_json_loads(payload: str | bytes) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    def reject_nonfinite_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def parse_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON number: {value}")
        return parsed

    return json.loads(
        payload,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonfinite_constant,
        parse_float=parse_finite_float,
    )


def _finite_number(value: Any, *, minimum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    converted = float(value)
    return math.isfinite(converted) and (minimum is None or converted >= minimum)


def _json_exact(value: Any, expected: Any) -> bool:
    """Compare decoded JSON without Python's bool/int numeric equivalence."""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(
            _json_exact(value[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _json_exact(observed, wanted)
            for observed, wanted in zip(value, expected, strict=True)
        )
    if isinstance(expected, float):
        return math.isfinite(value) and value == expected
    return value == expected


def _exact_int(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _clean_journal(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"amdgpu_boot_clean", "fatal_amdgpu_events"}
        and value.get("amdgpu_boot_clean") is True
        and value.get("fatal_amdgpu_events") == []
    )


class _DispatchWatchdogProtocolError(RuntimeError):
    def __init__(self, message: str, *, dispatch_started: bool) -> None:
        super().__init__(message)
        self.dispatch_started = dispatch_started


def _require_isolated_controller(run_dir: Path) -> dict[str, Any]:
    repo = Path(__file__).resolve().parent.parent
    expected_executable = repo / ".venv" / "bin" / "python"
    expected_cache = run_dir / "python-cache"
    expected_stdlib = [
        "/usr/lib/python312.zip",
        "/usr/lib/python3.12",
        "/usr/lib/python3.12/lib-dynload",
    ]
    flags = sys.flags
    checks = {
        "isolated": flags.isolated == 1,
        "ignore_environment": flags.ignore_environment == 1,
        "no_user_site": flags.no_user_site == 1,
        "no_site": flags.no_site == 1,
        "safe_path": flags.safe_path is True,
        "dont_write_bytecode": flags.dont_write_bytecode == 1,
        "exact_venv_python": Path(sys.executable) == expected_executable,
        "fresh_run_scoped_pycache_prefix": sys.pycache_prefix == str(expected_cache),
        "exact_initial_stdlib_path": sys.path == expected_stdlib,
    }
    if not all(checks.values()):
        failed = ", ".join(
            sorted(name for name, passed in checks.items() if not passed)
        )
        raise RuntimeError(
            "operational controller requires exact -I -S -B -X isolation: " + failed
        )
    return {
        "checks": checks,
        "sys_executable": sys.executable,
        "pycache_prefix": sys.pycache_prefix,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("abstract", "rocm"), default="abstract")
    parser.add_argument("--phase", choices=("compile", "execute"), default="compile")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    if args.platform == "rocm" and not args.allow_gpu:
        parser.error(
            "--platform rocm requires the explicit --allow-gpu acknowledgement"
        )
    if args.platform == "abstract" and args.allow_gpu:
        parser.error("--allow-gpu is only valid with --platform rocm")
    if args.platform == "rocm" and args.run_dir is None:
        parser.error("--platform rocm requires a fresh absolute --run-dir")
    if args.platform == "abstract" and args.run_dir is not None:
        parser.error("--run-dir is only valid with --platform rocm")
    if args.run_dir is not None and args.run_dir.exists():
        parser.error(f"refusing existing --run-dir: {args.run_dir}")
    if args.platform == "rocm":
        try:
            _require_isolated_controller(args.run_dir)
        except RuntimeError as error:
            parser.error(str(error))
    return args


def _contract(phase: str) -> dict[str, Any]:
    if phase not in {"compile", "execute"}:
        raise RuntimeError(f"unsupported W8 qualification phase: {phase}")
    execute = phase == "execute"
    return {
        "phase": phase,
        "child_case": "M3_K64_N17_W8A8_group64_rank8_LoRA_forward",
        "compiled_executable_invocations": 1 if execute else 0,
        "compiled_executable_completions": 1 if execute else 0,
        "runtime_promotion": False,
        "execute_rung_enabled": execute,
        "host_oracle_attempts": 1,
        "host_oracle_completions": 1,
        "host_sensitivity_comparisons": 35 if execute else 0,
        "backend_initialization_attempts": 1,
        "backend_initialization_completions": 1,
        "lower_attempts": 1,
        "lower_completions": 1,
        "compile_attempts": 1,
        "compile_completions": 1,
        "isa_qualification_attempts": 1 if execute else 0,
        "isa_qualification_completions": 1 if execute else 0,
        "tuple_device_put_attempts": 1 if execute else 0,
        "tuple_device_put_completions": 1 if execute else 0,
        "device_put_leaves": 6 if execute else 0,
        "input_readiness_invocations": 1 if execute else 0,
        "output_readiness_invocations": 1 if execute else 0,
        "device_get_attempts": 1 if execute else 0,
        "device_get_completions": 1 if execute else 0,
        "device_get_leaves": 1 if execute else 0,
        "backward_invocations": 0,
        "warmup_invocations": 0,
        "replay_invocations": 0,
        "gpu_reference_invocations": 0,
        "device_error_reduction_invocations": 0,
        "model_invocations": 0,
        "dispatch_watchdog_seconds": _DISPATCH_WATCHDOG_SECONDS if execute else None,
        "maximum_telemetry_dispatch_bracket_seconds": (
            _MAX_TELEMETRY_DISPATCH_BRACKET_SECONDS if execute else None
        ),
        "global_lock_held_through_idle_handoff": True,
        "profile": {
            "interval_seconds": _PROFILE_INTERVAL_SECONDS,
            "baseline_seconds": _PROFILE_BASELINE_SECONDS,
            "timeout_seconds": _PROFILE_TIMEOUT_SECONDS,
            "independent_outer_watchdog_seconds": _OUTER_WATCHDOG_SECONDS,
            "sensor_grace_seconds": _PROFILE_SENSOR_GRACE_SECONDS,
            "terminate_grace_seconds": _PROFILE_TERMINATE_GRACE_SECONDS,
            "maximum_sampled_junction_temperature_c": _MAX_JUNCTION_TEMP_C,
            "maximum_sampled_average_gpu_power_watts": _MAX_GPU_POWER_WATTS,
            "maximum_sampled_sysfs_vram_gib": _MAX_VRAM_GIB,
            "minimum_host_available_gib": _MIN_HOST_AVAILABLE_GIB,
            "maximum_swap_gib": _MAX_SWAP_GIB,
        },
        "handoff": {
            "timeout_seconds": _HANDOFF_TIMEOUT_SECONDS,
            "poll_interval_seconds": 1.0,
            "required_consecutive_ready_samples": 3,
            "vram_gtt_tolerance_bytes": 0,
        },
        "power_evidence_caveat": (
            "profile_rocm measures power1_average reactively; the child also "
            "requires the hardware power1_cap to be no greater than 315 W"
        ),
    }


def _repo() -> Path:
    return Path(__file__).resolve().parent.parent


def _source_files() -> dict[str, Path]:
    repo = _repo()
    return {
        "controller": Path(__file__).resolve(),
        "child": repo / "rocm" / "probe_w8a8_lora_forward.py",
        "isa_inspector": repo / "rocm" / "inspect_w8a8_lora_isa.py",
        "safety": repo / "rocm" / "amdgpu_safety.py",
        "handoff": repo / "rocm" / "qwen35_prewarm_handoff.py",
        "profiler": repo / "rocm" / "profile_rocm.py",
        "kernel": repo / "skyrl" / "tx" / "kernels" / "rocm" / "w8a8_lora.py",
        "quantized_reference": repo / "skyrl" / "tx" / "kernels" / "quantized_lora.py",
        "package_skyrl": repo / "skyrl" / "__init__.py",
        "package_tx": repo / "skyrl" / "tx" / "__init__.py",
        "package_kernels": repo / "skyrl" / "tx" / "kernels" / "__init__.py",
        "package_rocm": repo / "skyrl" / "tx" / "kernels" / "rocm" / "__init__.py",
        "sealed_loader": repo
        / "skyrl"
        / "tx"
        / "kernels"
        / "rocm"
        / "gdn_ffi_smoke.py",
        "package_top_rocm": repo / "rocm" / "__init__.py",
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"source is not a regular file: {path}")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _source_manifest() -> dict[str, str]:
    observed = {name: _file_sha256(path) for name, path in _source_files().items()}
    for name, expected in _EXPECTED_SOURCE_SHA256.items():
        if observed.get(name) != expected:
            raise RuntimeError(f"refusing changed {name} source")
    return observed


def _create_run_directory(path: Path) -> int:
    if not path.is_absolute():
        raise RuntimeError("--run-dir must be absolute")
    parent = path.parent
    if parent != Path(os.path.realpath(parent)):
        raise RuntimeError("--run-dir parent must not contain a symlink")
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.mkdir(path.name, mode=0o700, dir_fd=parent_fd)
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    finally:
        os.close(parent_fd)
    info = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        os.close(descriptor)
        raise RuntimeError("new run directory is not private")
    return descriptor


def _open_private_file(directory_fd: int, name: str) -> TextIO:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_nlink != 1
    ):
        os.close(descriptor)
        raise RuntimeError("new controller artifact is not a private regular file")
    return os.fdopen(descriptor, "w", encoding="utf-8")


def _create_private_subdirectory(directory_fd: int, name: str) -> None:
    if not name or "/" in name or name in {".", ".."}:
        raise RuntimeError("invalid private subdirectory name")
    os.mkdir(name, mode=0o700, dir_fd=directory_fd)
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=directory_fd,
    )
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
            or os.listdir(descriptor)
        ):
            raise RuntimeError("new private subdirectory is invalid")
    finally:
        os.close(descriptor)


def _child_environment(run_dir: Path | None = None) -> dict[str, str]:
    inherited_environment = os.environ.copy()
    required = {
        "JAX_PLATFORMS": "rocm",
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
        "JAX_ROCM_VISIBLE_DEVICES": "0",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_CLIENT_MEM_FRACTION": "0.075",
        "XLA_FLAGS": _COMMAND_BUFFER_FLAG,
    }
    prohibited = (
        "AMDGCN_ENABLE_DUMP",
        "HSA_OVERRIDE_GFX_VERSION",
        "JAX_COMPILATION_CACHE_DIR",
        "JAX_MOCK_GPU_TOPOLOGY",
        "JAX_PJRT_CLIENT_CREATE_OPTIONS",
        "LD_PRELOAD",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PROFILE",
        "MOCK_NUM_GPU_PROCESSES",
        "TEST_UNDECLARED_OUTPUTS_DIR",
        "TF_FORCE_UNIFIED_MEMORY",
        "TF_XLA_HSACO_BITCODE_SIZE_THRESHOLD",
        "TF_XLA_HSACO_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "TRITON_DUMP_DIR",
        "TRITON_KERNEL_DUMP",
        "XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
    )
    present = [name for name in prohibited if name in inherited_environment]
    if present:
        raise RuntimeError(
            "refusing inherited accelerator/compiler overrides: " + ", ".join(present)
        )
    accelerator_prefixes = (
        "AMDGCN_",
        "CUDA_",
        "GPU_DEVICE_ORDINAL",
        "HIP_",
        "HSA_",
        "JAX_",
        "ROCM_",
        "ROCR_",
        "TF_XLA_",
        "TRITON_",
        "XLA_",
    )
    unexpected = sorted(
        name
        for name in inherited_environment
        if name not in required
        and any(name.startswith(prefix) for prefix in accelerator_prefixes)
    )
    if unexpected:
        raise RuntimeError(
            "refusing unexpected accelerator environment: " + ", ".join(unexpected)
        )
    original_flags = inherited_environment.get("XLA_FLAGS")
    if original_flags not in (None, "", _COMMAND_BUFFER_FLAG):
        raise RuntimeError(
            "XLA_FLAGS must be unset or the exact sole command-buffer-disable flag"
        )
    for name, expected in required.items():
        inherited = inherited_environment.get(name)
        if inherited is not None and inherited != expected:
            raise RuntimeError(
                f"{name}={inherited!r} conflicts with required {expected!r}"
            )
    if run_dir is not None:
        info = run_dir.stat(follow_symlinks=False)
        if (
            run_dir != Path(os.path.realpath(run_dir))
            or run_dir.is_symlink()
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise RuntimeError("child run directory is not exact and private")
    private_home = "/nonexistent" if run_dir is None else str(run_dir)
    return {
        **required,
        "PATH": "/opt/rocm/bin:/usr/bin:/bin",
        "HOME": private_home,
        "TMPDIR": private_home,
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        **_systemd_user_environment(),
    }


def _read_vram_total(device_root: Path) -> int:
    try:
        value = int((device_root / "mem_info_vram_total").read_text().strip())
    except (OSError, ValueError) as error:
        raise RuntimeError("could not read exact AMD VRAM total") from error
    if value != _EXPECTED_VRAM_TOTAL_BYTES:
        raise RuntimeError(
            f"VRAM total {value} does not match expected {_EXPECTED_VRAM_TOTAL_BYTES}"
        )
    return value


def _systemd_user_environment() -> dict[str, str]:
    uid = os.getuid()
    runtime = Path("/run/user") / str(uid)
    bus = runtime / "bus"
    runtime_info = runtime.lstat()
    bus_info = bus.lstat()
    if (
        stat.S_ISLNK(runtime_info.st_mode)
        or not stat.S_ISDIR(runtime_info.st_mode)
        or runtime_info.st_uid != uid
        or stat.S_IMODE(runtime_info.st_mode) != 0o700
        or runtime != Path(os.path.realpath(runtime))
        or stat.S_ISLNK(bus_info.st_mode)
        or not stat.S_ISSOCK(bus_info.st_mode)
        or bus_info.st_uid != uid
        or bus != Path(os.path.realpath(bus))
    ):
        raise RuntimeError("private user-systemd bus identity is invalid")
    return {
        "XDG_RUNTIME_DIR": str(runtime),
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path={bus}",
    }


def _profile_runtime_manifest() -> dict[str, Any]:
    repo = _repo()
    site_packages = (
        repo
        / ".venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    ).resolve(strict=True)
    package_root = site_packages / "psutil"
    if (
        package_root.is_symlink()
        or not package_root.is_dir()
        or package_root.resolve(strict=True) != package_root
        or (site_packages / "psutil.py").exists()
        or (repo / "psutil.py").exists()
        or (repo / "psutil").exists()
    ):
        raise RuntimeError("ambiguous profiler psutil runtime")
    observed = {}
    for relative, expected in _EXPECTED_PROFILE_RUNTIME_SHA256.items():
        path = site_packages / relative
        digest = _file_sha256(path)
        if digest != expected:
            raise RuntimeError(f"profiler runtime mismatch: {relative}")
        observed[relative] = {
            "bytes": path.stat(follow_symlinks=False).st_size,
            "sha256": digest,
        }
    return {
        "site_packages": str(site_packages),
        "psutil_version": "7.2.2",
        "files": observed,
    }


def _systemctl_environment() -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        **_systemd_user_environment(),
    }


def _systemd_runtime_manifest() -> dict[str, Any]:
    binaries = {
        "/usr/bin/systemd-run": _EXPECTED_SYSTEMD_RUN_SHA256,
        "/usr/bin/systemctl": _EXPECTED_SYSTEMCTL_SHA256,
    }
    observed = {}
    for raw_path, expected in binaries.items():
        path = Path(raw_path)
        info = path.stat(follow_symlinks=False)
        digest = _file_sha256(path)
        if (
            path.is_symlink()
            or not stat.S_ISREG(info.st_mode)
            or info.st_uid != 0
            or not info.st_mode & stat.S_IXUSR
            or digest != expected
        ):
            raise RuntimeError(f"systemd containment binary mismatch: {path}")
        observed[raw_path] = {"bytes": info.st_size, "sha256": digest}
    result = subprocess.run(
        ["/usr/bin/systemctl", "--user", "is-system-running"],
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
        env=_systemctl_environment(),
    )
    if result.returncode != 0 or result.stdout != "running\n" or result.stderr:
        raise RuntimeError("user systemd manager is not exactly running")
    return {
        "user_manager": "running",
        "binaries": observed,
        "sealed_bus_environment": _systemd_user_environment(),
    }


def _scope_name(phase: str = "compile") -> str:
    if phase not in {"compile", "execute"}:
        raise RuntimeError(f"unsupported W8 qualification phase: {phase}")
    label = "compile" if phase == "compile" else "runtime"
    return f"skyrl-w8a8-{label}-{os.getpid()}-{time.monotonic_ns():x}"


def _scope_command(unit: str, command: list[str]) -> list[str]:
    if not unit.startswith(("skyrl-w8a8-compile-", "skyrl-w8a8-runtime-")) or not all(
        character.isalnum() or character in "-_" for character in unit
    ):
        raise RuntimeError("invalid private systemd scope name")
    return [
        "/usr/bin/systemd-run",
        "--user",
        "--scope",
        "--collect",
        "--quiet",
        f"--unit={unit}",
        "--property=KillMode=control-group",
        "--property=SendSIGKILL=yes",
        f"--property=TimeoutStopSec={_OUTER_TERMINATE_GRACE_SECONDS:g}s",
        f"--property=RuntimeMaxSec={_OUTER_WATCHDOG_SECONDS:g}s",
        "--",
        *command,
    ]


def _scope_state(unit: str, timeout_seconds: float = 2.0) -> dict[str, Any]:
    if not _finite_number(timeout_seconds, minimum=0.001):
        raise RuntimeError("systemd scope inspection timeout is invalid")
    result = subprocess.run(
        [
            "/usr/bin/systemctl",
            "--user",
            "show",
            f"{unit}.scope",
            "--property=Id",
            "--property=LoadState",
            "--property=ActiveState",
            "--property=SubState",
            "--property=ControlGroup",
            "--property=KillMode",
            "--property=SendSIGKILL",
            "--property=TimeoutStopUSec",
            "--property=RuntimeMaxUSec",
        ],
        capture_output=True,
        text=True,
        timeout=float(timeout_seconds),
        check=False,
        env=_systemctl_environment(),
    )
    if result.returncode != 0 or result.stderr:
        raise RuntimeError("could not inspect the private systemd scope")
    fields = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            raise RuntimeError("malformed systemd scope state")
        key, value = line.split("=", 1)
        if key in fields:
            raise RuntimeError("duplicate systemd scope state field")
        fields[key] = value
    required = {
        "Id",
        "LoadState",
        "ActiveState",
        "SubState",
        "ControlGroup",
        "KillMode",
        "SendSIGKILL",
        "TimeoutStopUSec",
        "RuntimeMaxUSec",
    }
    if set(fields) != required:
        raise RuntimeError("incomplete systemd scope state")
    if fields["Id"] != f"{unit}.scope":
        raise RuntimeError("systemd scope identity mismatch")
    if fields["KillMode"] != "control-group":
        raise RuntimeError("systemd scope does not use control-group kill mode")
    if fields["LoadState"] not in {"loaded", "not-found"}:
        raise RuntimeError("unexpected systemd scope load state")
    if fields["LoadState"] == "not-found" and (
        fields["ActiveState"] != "inactive"
        or fields["SubState"] != "dead"
        or fields["ControlGroup"]
    ):
        raise RuntimeError("malformed unobserved systemd scope state")
    control_group = fields["ControlGroup"]
    if control_group and (
        fields["SendSIGKILL"] != "yes"
        or fields["TimeoutStopUSec"] != "10s"
        or fields["RuntimeMaxUSec"] != "5min 30s"
    ):
        raise RuntimeError("systemd scope timeout/kill properties are not exact")
    pids: list[int] = []
    if control_group:
        if not control_group.startswith("/user.slice/") or ".." in control_group:
            raise RuntimeError("unexpected systemd scope cgroup path")
        cgroup_procs = (
            Path("/sys/fs/cgroup") / control_group.removeprefix("/") / "cgroup.procs"
        )
        try:
            raw_pids = cgroup_procs.read_text().splitlines()
        except FileNotFoundError:
            raw_pids = []
        if any(not value.isdecimal() or int(value) <= 0 for value in raw_pids):
            raise RuntimeError("malformed scope cgroup.procs")
        pids = sorted(int(value) for value in raw_pids)
    return {
        **fields,
        "observed": fields["LoadState"] == "loaded" and bool(control_group),
        "pids": pids,
    }


def _terminate_scope(unit: str) -> dict[str, Any]:
    before = _scope_state(unit)
    signals_sent = []
    if before["pids"] or before["ActiveState"] not in {"inactive", "failed"}:
        for signal_name in ("SIGTERM", "SIGKILL"):
            result = subprocess.run(
                [
                    "/usr/bin/systemctl",
                    "--user",
                    "kill",
                    "--kill-whom=all",
                    f"--signal={signal_name}",
                    f"{unit}.scope",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
                env=_systemctl_environment(),
            )
            signals_sent.append(
                {
                    "signal": signal_name,
                    "returncode": result.returncode,
                    "stderr": result.stderr.strip(),
                }
            )
            deadline = time.monotonic() + _OUTER_TERMINATE_GRACE_SECONDS
            state = _scope_state(unit)
            while time.monotonic() < deadline:
                if not state["pids"] and state["ActiveState"] in {
                    "inactive",
                    "failed",
                }:
                    break
                time.sleep(0.05)
                state = _scope_state(unit)
            if not state["pids"] and state["ActiveState"] in {
                "inactive",
                "failed",
            }:
                break
    after = _scope_state(unit)
    passed = not after["pids"] and after["ActiveState"] in {
        "inactive",
        "failed",
    }
    return {
        "unit": f"{unit}.scope",
        "before": before,
        "signals_sent": signals_sent,
        "after": after,
        "passed": passed,
    }


def _direct_cgroup_cleanup(unit: str) -> dict[str, Any]:
    uid = os.getuid()
    expected_control_group = (
        f"/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{unit}.scope"
    )
    path = Path("/sys/fs/cgroup") / expected_control_group.removeprefix("/")
    observed = False
    signals_sent = 0
    initial_pids: list[int] = []
    discovery_deadline = time.monotonic() + 1.0
    while True:
        try:
            descriptor = os.open(
                path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
            )
        except FileNotFoundError:
            if time.monotonic() >= discovery_deadline:
                return {
                    "control_group": expected_control_group,
                    "observed": observed,
                    "initial_pids": initial_pids,
                    "signals_sent": signals_sent,
                    "final_pids": [],
                    "passed": True,
                }
            time.sleep(0.05)
            continue
        break
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != uid:
            raise RuntimeError("direct cgroup directory identity mismatch")
        observed = True

        def read_pids() -> list[int]:
            try:
                proc_fd = os.open(
                    "cgroup.procs",
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                return []
            try:
                raw = os.read(proc_fd, 1024 * 1024).decode("ascii").splitlines()
            finally:
                os.close(proc_fd)
            if any(not value.isdecimal() or int(value) <= 0 for value in raw):
                raise RuntimeError("direct cgroup process list is malformed")
            return sorted(int(value) for value in raw)

        initial_pids = read_pids()
        if initial_pids:
            kill_fd = os.open(
                "cgroup.kill",
                os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            try:
                if os.write(kill_fd, b"1") != 1:
                    raise RuntimeError("direct cgroup kill was a short write")
                signals_sent = 1
            finally:
                os.close(kill_fd)
        deadline = time.monotonic() + _OUTER_TERMINATE_GRACE_SECONDS
        final_pids = read_pids()
        while final_pids and time.monotonic() < deadline:
            time.sleep(0.05)
            final_pids = read_pids()
    finally:
        os.close(descriptor)
    return {
        "control_group": expected_control_group,
        "observed": observed,
        "initial_pids": initial_pids,
        "signals_sent": signals_sent,
        "final_pids": final_pids,
        "passed": not final_pids,
    }


def _profile_command(
    *,
    phase: str,
    run_dir: Path,
    card: str,
    lock_fd: int,
) -> list[str]:
    if phase not in {"compile", "execute"}:
        raise RuntimeError(f"unsupported W8 qualification phase: {phase}")
    repo = _repo()
    site_packages = (
        repo
        / ".venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    ).resolve(strict=True)
    profiler = repo / "rocm" / "profile_rocm.py"
    return [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-X",
        f"pycache_prefix={run_dir / 'python-cache'}",
        "-c",
        _ISOLATED_PROFILE_BOOTSTRAP,
        str(site_packages),
        str(repo),
        str(profiler),
        _EXPECTED_SOURCE_SHA256["profiler"],
        json.dumps(
            _EXPECTED_PROFILE_RUNTIME_SHA256,
            separators=(",", ":"),
            sort_keys=True,
        ),
        str(run_dir / "python-cache"),
        "--output",
        str(run_dir / "telemetry.jsonl"),
        "--card",
        card,
        "--interval",
        str(_PROFILE_INTERVAL_SECONDS),
        "--baseline-seconds",
        str(_PROFILE_BASELINE_SECONDS),
        "--timeout",
        str(_PROFILE_TIMEOUT_SECONDS),
        "--terminate-grace-seconds",
        str(_PROFILE_TERMINATE_GRACE_SECONDS),
        "--sensor-grace-seconds",
        str(_PROFILE_SENSOR_GRACE_SECONDS),
        "--max-junction-temp-c",
        str(_MAX_JUNCTION_TEMP_C),
        "--max-gpu-power-watts",
        str(_MAX_GPU_POWER_WATTS),
        "--max-vram-gib",
        str(_MAX_VRAM_GIB),
        "--min-host-available-gib",
        str(_MIN_HOST_AVAILABLE_GIB),
        "--max-swap-gib",
        str(_MAX_SWAP_GIB),
        "--record-command",
        "--pass-fd",
        str(lock_fd),
        "--",
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-X",
        f"pycache_prefix={run_dir / 'python-cache'}",
        str(repo / "rocm" / "probe_w8a8_lora_forward.py"),
        "--platform",
        "rocm",
        "--phase",
        phase,
        "--allow-gpu",
        "--output",
        str(run_dir / "probe.jsonl"),
        "--artifact-dir",
        str(run_dir / "compiler-artifacts"),
        "--launcher-lock-fd",
        str(lock_fd),
    ]


def _dispatch_watchdog_state(path: Path) -> dict[str, Any]:
    """Read a strict flushed prefix of the one-shot runtime protocol."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except FileNotFoundError:
        return {
            "probe_observed": False,
            "dispatch_started": False,
            "post_attempt_observed": False,
        }
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 1
            or info.st_size < 0
            or info.st_size > 8 * 1024**2
        ):
            raise RuntimeError("dispatch watchdog probe log is not exact and private")
        remaining = info.st_size
        chunks = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError("dispatch watchdog probe log was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    complete_lines = payload.split(b"\n")
    if complete_lines and complete_lines[-1]:
        complete_lines.pop()
    else:
        complete_lines = complete_lines[:-1]
    records: list[dict[str, Any]] = []

    def fail(message: str) -> None:
        raise _DispatchWatchdogProtocolError(
            message,
            dispatch_started=len(records) > 13
            and records[13].get("record_type") == "dispatch_started",
        )

    for index, raw in enumerate(complete_lines):
        if not raw.strip():
            fail("dispatch watchdog observed an empty protocol record")
        try:
            record = _strict_json_loads(raw)
        except (UnicodeDecodeError, ValueError) as error:
            raise _DispatchWatchdogProtocolError(
                f"dispatch watchdog observed malformed JSON at record {index}",
                dispatch_started=len(records) > 13,
            ) from error
        if not isinstance(record, dict):
            fail(f"dispatch watchdog record {index} is not an object")
        records.append(record)
        if index >= len(_RUNTIME_PROBE_PROTOCOL):
            fail("dispatch watchdog observed records beyond the runtime protocol")
        expected_type, expected_stage = _RUNTIME_PROBE_PROTOCOL[index]
        if record.get("record_type") != expected_type:
            fail(f"dispatch watchdog protocol mismatch at record {index}")
        if expected_stage is None:
            if expected_type != "journal_checkpoint" and "stage" in record:
                fail(f"dispatch watchdog alien stage at record {index}")
        elif record.get("stage") != expected_stage:
            fail(f"dispatch watchdog journal stage mismatch at record {index}")

    def positive_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    start = records[13] if len(records) > 13 else None
    checkpoint = records[14] if len(records) > 14 else None
    dispatch = records[15] if len(records) > 15 else None
    started_wall_time_ns: int | None = None
    started_monotonic_ns: int | None = None
    if start is not None:
        started_wall_time_ns = start.get("wall_time_ns")
        started_monotonic_ns = start.get("monotonic_ns")
        if not positive_int(started_wall_time_ns) or not positive_int(
            started_monotonic_ns
        ):
            fail("dispatch watchdog start timestamps are malformed")
        if not _json_exact(
            start.get("one_shot_capability"),
            {
                "consumed": True,
                "compiled_executable_attempts": 1,
                "compiled_executable_completions": 0,
            },
        ) or not _exact_int(start.get("output_readiness_invocations"), 0):
            fail("dispatch watchdog start capability is not exact")

    checkpoint_wall_time_ns: int | None = None
    checkpoint_monotonic_ns: int | None = None
    if checkpoint is not None:
        checkpoint_wall_time_ns = checkpoint.get("dispatch_completed_wall_time_ns")
        checkpoint_monotonic_ns = checkpoint.get("dispatch_completed_monotonic_ns")
        if not positive_int(checkpoint_wall_time_ns) or not positive_int(
            checkpoint_monotonic_ns
        ):
            fail("dispatch watchdog checkpoint timestamps are malformed")
        assert started_wall_time_ns is not None and started_monotonic_ns is not None
        if (
            checkpoint_wall_time_ns < started_wall_time_ns
            or checkpoint_monotonic_ns < started_monotonic_ns
            or not _exact_int(checkpoint.get("compiled_executable_attempts"), 1)
            or not _exact_int(checkpoint.get("compiled_executable_completions"), 0)
            or not _clean_journal(checkpoint.get("safety"))
        ):
            fail("dispatch watchdog post-attempt checkpoint is not exact")

    post_attempt_observed = dispatch is not None
    if dispatch is not None:
        assert start is not None and checkpoint is not None
        if (
            dispatch.get("dispatch_started_wall_time_ns") != started_wall_time_ns
            or dispatch.get("dispatch_started_monotonic_ns") != started_monotonic_ns
            or dispatch.get("dispatch_completed_wall_time_ns")
            != checkpoint_wall_time_ns
            or dispatch.get("dispatch_completed_monotonic_ns")
            != checkpoint_monotonic_ns
            or not _json_exact(
                dispatch.get("one_shot_capability"),
                {
                    "consumed": True,
                    "compiled_executable_attempts": 1,
                    "compiled_executable_completions": 1,
                },
            )
            or not _exact_int(dispatch.get("output_readiness_invocations"), 1)
        ):
            fail("dispatch watchdog completion record is not exact")
        dispatch_seconds = dispatch.get("dispatch_seconds_including_output_readiness")
        assert started_monotonic_ns is not None and checkpoint_monotonic_ns is not None
        if (
            not _finite_number(dispatch_seconds, minimum=0.0)
            or float(dispatch_seconds) >= 1.0
            or abs(
                float(dispatch_seconds)
                - (checkpoint_monotonic_ns - started_monotonic_ns) / 1_000_000_000
            )
            > 0.01
        ):
            fail("dispatch watchdog completion duration is not exact")
    return {
        "probe_observed": True,
        "protocol_prefix_exact": True,
        "validated_protocol_prefix_length": len(records),
        "dispatch_started": start is not None,
        "dispatch_started_wall_time_ns": started_wall_time_ns,
        "dispatch_started_monotonic_ns": started_monotonic_ns,
        "post_attempt_checkpoint_observed": checkpoint is not None,
        "post_attempt_observed": post_attempt_observed,
        "post_attempt_wall_time_ns": (
            checkpoint_wall_time_ns if post_attempt_observed else None
        ),
        "post_attempt_monotonic_ns": (
            checkpoint_monotonic_ns if post_attempt_observed else None
        ),
        "post_attempt_record_types": (
            ["journal_checkpoint", "dispatch"] if post_attempt_observed else []
        ),
    }


def _write_cgroup_kill(path: Path) -> dict[str, Any]:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError("dispatch cgroup.kill is not a regular control file")
        written = os.write(descriptor, b"1")
        if written != 1:
            raise RuntimeError("dispatch cgroup.kill write was short")
    finally:
        os.close(descriptor)
    return {"path": str(path), "bytes_written": 1}


def _kill_dispatch_scope_immediately(
    unit: str,
    process: subprocess.Popen[Any],
    *,
    validated_control_group: str | None = None,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> tuple[int | None, dict[str, Any]]:
    started = time.monotonic()
    deadline = started + _DISPATCH_KILL_REAP_SECONDS
    uid = os.getuid()
    expected_control_group = (
        f"/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{unit}.scope"
    )
    kill_path = cgroup_root / expected_control_group.removeprefix("/") / "cgroup.kill"
    initial_scope_exact = validated_control_group == expected_control_group
    direct_kill: dict[str, Any] = {"attempted": True, "passed": False}
    if initial_scope_exact:
        try:
            direct_kill.update(_write_cgroup_kill(kill_path))
            direct_kill["passed"] = True
        except BaseException as error:
            direct_kill.update(
                {"error_type": type(error).__name__, "error": str(error)}
            )
    else:
        direct_kill["refused_reason"] = "initial_scope_identity_not_exact"
    systemctl_kill: dict[str, Any] = {"attempted": True, "passed": False}
    kill_command = [
        "/usr/bin/systemctl",
        "--user",
        "kill",
        "--kill-whom=all",
        "--signal=SIGKILL",
        f"{unit}.scope",
    ]
    try:
        result = subprocess.run(
            kill_command,
            capture_output=True,
            text=True,
            timeout=max(0.001, deadline - time.monotonic()),
            check=False,
            env=_systemctl_environment(),
        )
        systemctl_kill.update(
            {
                "command": kill_command,
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "passed": result.returncode == 0 and not result.stderr,
            }
        )
    except BaseException as error:
        systemctl_kill.update({"error_type": type(error).__name__, "error": str(error)})
    wrapper_kill_fallback = False
    returncode: int | None = None
    returncode = process.poll()
    if returncode is None:
        try:
            returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            wrapper_kill_fallback = True
            process.kill()
            try:
                returncode = process.wait(
                    timeout=max(0.001, deadline - time.monotonic())
                )
            except subprocess.TimeoutExpired:
                returncode = process.poll()
    final_scope_state: dict[str, Any] | None = None
    scope_state_errors: list[dict[str, str]] = []
    while time.monotonic() < deadline:
        try:
            state = _scope_state(unit, max(0.001, deadline - time.monotonic()))
        except BaseException as error:
            scope_state_errors.append(
                {"error_type": type(error).__name__, "error": str(error)}
            )
            break
        final_scope_state = state
        if not state.get("pids") and state.get("ActiveState") in {
            "inactive",
            "failed",
        }:
            break
        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
    if returncode is None and process.poll() is None:
        wrapper_kill_fallback = True
        process.kill()
        returncode = process.poll()
    elapsed = time.monotonic() - started
    kill_issued = direct_kill["passed"] is True or systemctl_kill["passed"] is True
    scope_empty = (
        isinstance(final_scope_state, dict)
        and final_scope_state.get("ControlGroup") in {"", expected_control_group}
        and not final_scope_state.get("pids")
        and final_scope_state.get("ActiveState") in {"inactive", "failed"}
    )
    passed = (
        kill_issued
        and scope_empty
        and returncode is not None
        and elapsed <= _DISPATCH_KILL_REAP_SECONDS + 0.01
    )
    return returncode, {
        "method": "immediate_cgroup_wide_sigkill_and_bounded_reap",
        "control_group": expected_control_group,
        "validated_control_group": validated_control_group,
        "initial_scope_exact": initial_scope_exact,
        "direct_cgroup_kill": direct_kill,
        "systemctl_cgroup_kill": systemctl_kill,
        "kill_issued": kill_issued,
        "wrapper_kill_fallback": wrapper_kill_fallback,
        "wrapper_reaped": returncode is not None,
        "final_scope_state": final_scope_state,
        "scope_state_errors": scope_state_errors,
        "scope_empty": scope_empty,
        "maximum_kill_and_reap_seconds": _DISPATCH_KILL_REAP_SECONDS,
        "kill_and_reap_seconds": elapsed,
        "returncode": returncode,
        "passed": passed,
    }


def _wait_profile(
    process: subprocess.Popen[Any],
    unit: str,
    stop_signal: Any | None = None,
    *,
    phase: str = "compile",
    probe_path: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    if phase not in {"compile", "execute"}:
        raise RuntimeError(f"unsupported W8 qualification phase: {phase}")
    if phase == "execute" and probe_path is None:
        raise RuntimeError("runtime wait requires the private probe log path")
    deadline = time.monotonic() + _OUTER_WATCHDOG_SECONDS
    expected_control_group = (
        f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/"
        f"app.slice/{unit}.scope"
    )
    scope_observed = False
    observed_control_group: str | None = None
    dispatch_watchdog: dict[str, Any] = {
        "enabled": phase == "execute",
        "timeout_seconds": _DISPATCH_WATCHDOG_SECONDS if phase == "execute" else None,
        "dispatch_started": False,
        "post_attempt_observed": False,
        "passed": phase == "compile",
    }
    if stop_signal is None:

        def no_stop_signal() -> None:
            return None

        stop_signal = no_stop_signal

    def observe_scope(timeout_seconds: float = 2.0) -> None:
        nonlocal scope_observed, observed_control_group
        state = _scope_state(unit, timeout_seconds)
        if state["observed"]:
            if state["ControlGroup"] != expected_control_group:
                raise RuntimeError("systemd scope identity mismatch")
            scope_observed = True
            observed_control_group = state["ControlGroup"]

    def terminate_execute(
        reason: str,
        received_signal: int | None,
        *,
        final_scope_state: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        returncode, immediate_cleanup = _kill_dispatch_scope_immediately(
            unit,
            process,
            validated_control_group=observed_control_group,
        )
        audit = {
            "outer_watchdog_seconds": _OUTER_WATCHDOG_SECONDS,
            "received_signal": received_signal,
            "termination_reason": reason,
            "scope_observed": scope_observed,
            "control_group": observed_control_group,
            "dispatch_watchdog": dispatch_watchdog,
            "cleanup": immediate_cleanup,
            "passed": False,
        }
        if final_scope_state is not None:
            audit["final_scope_state"] = final_scope_state
        return -signal.SIGKILL if returncode is None else returncode, audit

    try:
        while True:
            if phase == "execute":
                pending_signal = stop_signal()
                if pending_signal is not None:
                    return terminate_execute(f"signal_{pending_signal}", pending_signal)
            if phase == "execute":
                assert probe_path is not None
                try:
                    observed_dispatch = _dispatch_watchdog_state(probe_path)
                except _DispatchWatchdogProtocolError as error:
                    dispatch_watchdog["evidence_error"] = str(error)
                    dispatch_watchdog["dispatch_started"] = (
                        dispatch_watchdog.get("dispatch_started") is True
                        or error.dispatch_started
                    )
                    reason = "dispatch_watchdog_malformed_evidence"
                except RuntimeError as error:
                    dispatch_watchdog["evidence_error"] = str(error)
                    reason = "dispatch_watchdog_malformed_evidence"
                else:
                    dispatch_watchdog.update(observed_dispatch)
                    dispatch_in_flight = bool(
                        observed_dispatch.get("dispatch_started") is True
                        and observed_dispatch.get("post_attempt_observed") is not True
                    )
                    if not scope_observed and dispatch_in_flight:
                        started_ns = observed_dispatch.get(
                            "dispatch_started_monotonic_ns"
                        )
                        now_ns = time.monotonic_ns()
                        remaining_seconds = (
                            _DISPATCH_WATCHDOG_SECONDS
                            - (now_ns - started_ns) / 1_000_000_000
                            if isinstance(started_ns, int)
                            and not isinstance(started_ns, bool)
                            else 0.0
                        )
                        if remaining_seconds > 0:
                            observe_scope(max(0.001, min(2.0, remaining_seconds)))
                    elif not dispatch_in_flight:
                        observe_scope()
                    started_ns = observed_dispatch.get("dispatch_started_monotonic_ns")
                    post_attempt_ns = observed_dispatch.get("post_attempt_monotonic_ns")
                    now_ns = time.monotonic_ns()
                    if started_ns is not None and started_ns > now_ns + 100_000_000:
                        reason = "dispatch_watchdog_malformed_future_start"
                    elif (
                        started_ns is not None
                        and post_attempt_ns is not None
                        and post_attempt_ns - started_ns
                        >= int(_DISPATCH_WATCHDOG_SECONDS * 1_000_000_000)
                    ):
                        reason = "dispatch_watchdog_late_post_attempt"
                    elif (
                        started_ns is not None
                        and not observed_dispatch.get("post_attempt_observed")
                        and now_ns - started_ns
                        >= int(_DISPATCH_WATCHDOG_SECONDS * 1_000_000_000)
                    ):
                        reason = "dispatch_watchdog_timeout"
                    else:
                        reason = None
                    if (
                        reason is None
                        and observed_dispatch.get("dispatch_started") is True
                        and (
                            not scope_observed
                            or observed_control_group != expected_control_group
                        )
                    ):
                        reason = "dispatch_watchdog_unbound_scope_at_start"
                    dispatch_watchdog["passed"] = bool(
                        observed_dispatch.get("protocol_prefix_exact") is True
                        and observed_dispatch.get("dispatch_started")
                        and observed_dispatch.get("post_attempt_observed")
                    )
            else:
                observe_scope()
                reason = None
            received_signal = stop_signal()
            if received_signal is not None and reason is None:
                reason = f"signal_{received_signal}"
            elif time.monotonic() >= deadline and reason is None:
                reason = "outer_watchdog_timeout"
            if reason is not None:
                if phase == "execute":
                    return terminate_execute(reason, received_signal)
                cleanup = _terminate_scope(unit)
                try:
                    returncode = process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    returncode = process.wait(timeout=1.0)
                return (
                    returncode,
                    {
                        "outer_watchdog_seconds": _OUTER_WATCHDOG_SECONDS,
                        "received_signal": received_signal,
                        "termination_reason": reason,
                        "scope_observed": scope_observed,
                        "control_group": observed_control_group,
                        "dispatch_watchdog": dispatch_watchdog,
                        "cleanup": cleanup,
                        "passed": False,
                    },
                )
            returncode = process.poll()
            if returncode is None:
                time.sleep(0.05)
                continue
            final_state = _scope_state(
                unit,
                (_EXECUTE_FINAL_SCOPE_QUERY_SECONDS if phase == "execute" else 2.0),
            )
            if final_state["observed"]:
                if final_state["ControlGroup"] != expected_control_group:
                    raise RuntimeError("systemd scope identity mismatch")
                scope_observed = True
                observed_control_group = final_state["ControlGroup"]
            populated = bool(final_state["pids"]) or final_state["ActiveState"] not in {
                "inactive",
                "failed",
            }
            if phase == "execute" and populated:
                return terminate_execute(
                    "scope_still_populated_after_profile_exit",
                    None,
                    final_scope_state=final_state,
                )
            transition_deadline = time.monotonic() + 1.0
            while (
                bool(final_state["pids"])
                or final_state["ActiveState"] not in {"inactive", "failed"}
            ) and time.monotonic() < transition_deadline:
                time.sleep(0.05)
                final_state = _scope_state(unit)
                if final_state["observed"]:
                    scope_observed = True
                    observed_control_group = final_state["ControlGroup"]
            populated = bool(final_state["pids"]) or final_state["ActiveState"] not in {
                "inactive",
                "failed",
            }
            cleanup = _terminate_scope(unit) if populated else None
            termination_reason = None
            if populated:
                termination_reason = "scope_still_populated_after_profile_exit"
            elif not scope_observed:
                termination_reason = "scope_never_observed"
            return (
                returncode,
                {
                    "outer_watchdog_seconds": _OUTER_WATCHDOG_SECONDS,
                    "received_signal": None,
                    "termination_reason": termination_reason,
                    "scope_observed": scope_observed,
                    "control_group": observed_control_group,
                    "dispatch_watchdog": dispatch_watchdog,
                    "final_scope_state": final_state,
                    "cleanup": cleanup,
                    "passed": scope_observed
                    and not populated
                    and (phase == "compile" or dispatch_watchdog["passed"] is True),
                },
            )
    except BaseException as error:
        if phase == "execute":
            returncode, immediate_cleanup = _kill_dispatch_scope_immediately(
                unit,
                process,
                validated_control_group=observed_control_group,
            )
            dispatch_watchdog["supervisor_error_type"] = type(error).__name__
            dispatch_watchdog["supervisor_error"] = str(error)
            return (
                -signal.SIGKILL if returncode is None else returncode,
                {
                    "outer_watchdog_seconds": _OUTER_WATCHDOG_SECONDS,
                    "received_signal": None,
                    "termination_reason": "dispatch_watchdog_supervisor_error",
                    "scope_observed": scope_observed,
                    "control_group": observed_control_group,
                    "dispatch_watchdog": dispatch_watchdog,
                    "cleanup": immediate_cleanup,
                    "passed": False,
                },
            )
        try:
            _terminate_scope(unit)
        except BaseException:
            pass
        try:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=1.0)
        except BaseException:
            pass
        raise


def _private_json_lines(path: Path, maximum_bytes: int) -> list[dict[str, Any]]:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 1
            or info.st_size <= 0
            or info.st_size > maximum_bytes
        ):
            raise RuntimeError(f"invalid private JSON artifact: {path.name}")
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as source:
            records = [_strict_json_loads(line) for line in source if line.strip()]
    finally:
        os.close(descriptor)
    if not records or not all(isinstance(record, dict) for record in records):
        raise RuntimeError(f"empty or malformed JSON artifact: {path.name}")
    return records


def _private_json_object(path: Path, maximum_bytes: int) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 1
            or info.st_size <= 0
            or info.st_size > maximum_bytes
        ):
            raise RuntimeError(f"invalid private JSON artifact: {path.name}")
        chunks = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError(f"truncated JSON artifact: {path.name}")
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    value = _strict_json_loads(b"".join(chunks))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact is not one object: {path.name}")
    return value


def _summarize_private_telemetry(path: Path, maximum_bytes: int) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 1
            or info.st_size <= 0
            or info.st_size > maximum_bytes
        ):
            raise RuntimeError("invalid private telemetry artifact")
        manifest: dict[str, Any] | None = None
        sample_count = 0
        baseline_count = 0
        preflight_count = 0
        measured_count = 0
        previous_elapsed: float | None = None
        first_measured_elapsed: float | None = None
        maximum_gap = 0.0
        timestamps_monotonic = True
        measured_mandatory_fields_valid = True
        post_grace_gpu_sensors_present = True
        present = {
            "power": False,
            "temperature": False,
            "vram": False,
            "host_memory": False,
            "host_swap": False,
        }
        within = {
            "power": True,
            "temperature": True,
            "vram": True,
            "host_memory": True,
            "host_swap": True,
        }
        measured_maximum: dict[str, float | None] = {
            "power": None,
            "temperature": None,
            "vram": None,
            "host_swap": None,
        }
        measured_wall_time_ns: list[int] = []
        fully_observed_safety_wall_time_ns: list[int] = []
        measured_wall_timestamps_valid = True
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as source:
            for line in source:
                if not line.strip():
                    continue
                record = _strict_json_loads(line)
                if not isinstance(record, dict):
                    raise RuntimeError("telemetry line is not a JSON object")
                if manifest is None:
                    if record.get("record_type") != "manifest":
                        raise RuntimeError(
                            "telemetry_record_sequence_exact: first record is not the manifest"
                        )
                    manifest = record
                    continue
                if record.get("record_type") != "sample":
                    raise RuntimeError(
                        "telemetry_record_sequence_exact: non-sample record after manifest"
                    )
                phase = record.get("phase")
                if phase == "baseline":
                    if preflight_count or measured_count:
                        raise RuntimeError(
                            "telemetry_record_sequence_exact: baseline phase order regressed"
                        )
                    baseline_count += 1
                elif phase == "preflight":
                    if baseline_count < 1 or preflight_count or measured_count:
                        raise RuntimeError(
                            "telemetry_record_sequence_exact: preflight phase is not unique and ordered"
                        )
                    preflight_count = 1
                elif phase == "measured":
                    if baseline_count < 1 or preflight_count != 1:
                        raise RuntimeError(
                            "telemetry_record_sequence_exact: measured phase precedes preflight"
                        )
                    measured_count += 1
                else:
                    raise RuntimeError(
                        "telemetry_record_sequence_exact: sample phase is not recognized"
                    )
                sample_count += 1
                if phase != "measured":
                    continue
                wall_time_ns = record.get("wall_time_ns")
                if (
                    isinstance(wall_time_ns, bool)
                    or not isinstance(wall_time_ns, int)
                    or wall_time_ns <= 0
                ):
                    measured_wall_timestamps_valid = False
                    measured_mandatory_fields_valid = False
                else:
                    measured_wall_time_ns.append(wall_time_ns)
                elapsed = record.get("elapsed_seconds")
                if not _finite_number(elapsed, minimum=0.0):
                    timestamps_monotonic = False
                    measured_mandatory_fields_valid = False
                else:
                    elapsed = float(elapsed)
                    if first_measured_elapsed is None:
                        first_measured_elapsed = elapsed
                    if previous_elapsed is not None:
                        gap = elapsed - previous_elapsed
                        timestamps_monotonic &= gap > 0
                        maximum_gap = max(maximum_gap, gap)
                    previous_elapsed = elapsed
                complete_safety_sample = (
                    isinstance(wall_time_ns, int)
                    and not isinstance(wall_time_ns, bool)
                    and wall_time_ns > 0
                    and _finite_number(elapsed, minimum=0.0)
                )
                for label, key, limit, limit_kind in (
                    (
                        "power",
                        "gpu_power_watts",
                        _MAX_GPU_POWER_WATTS,
                        "maximum",
                    ),
                    (
                        "temperature",
                        "gpu_junction_temp_c",
                        _MAX_JUNCTION_TEMP_C,
                        "maximum",
                    ),
                    (
                        "vram",
                        "vram_used_bytes",
                        _MAX_VRAM_GIB * 1024**3,
                        "maximum",
                    ),
                    (
                        "host_memory",
                        "host_memory_available_bytes",
                        0.0,
                        "minimum",
                    ),
                    (
                        "host_swap",
                        "host_swap_used_bytes",
                        _MAX_SWAP_GIB * 1024**3,
                        "maximum",
                    ),
                ):
                    value = record.get(key)
                    if _finite_number(value, minimum=0.0):
                        present[label] = True
                        value_within = (
                            float(value) >= limit
                            if limit_kind == "minimum"
                            else float(value) <= limit
                        )
                        within[label] &= value_within
                        measured_value = float(value)
                        if label in measured_maximum:
                            current_maximum = measured_maximum[label]
                            measured_maximum[label] = (
                                measured_value
                                if current_maximum is None
                                else max(current_maximum, measured_value)
                            )
                        complete_safety_sample &= value_within
                    elif value is not None:
                        within[label] = False
                        complete_safety_sample = False
                    else:
                        complete_safety_sample = False
                    if label in {"vram", "host_memory", "host_swap"} and not (
                        _finite_number(value, minimum=0.0)
                    ):
                        measured_mandatory_fields_valid = False
                if (
                    first_measured_elapsed is not None
                    and _finite_number(elapsed, minimum=0.0)
                    and float(elapsed) - first_measured_elapsed
                    >= _PROFILE_SENSOR_GRACE_SECONDS
                    and not (
                        _finite_number(record.get("gpu_power_watts"), minimum=0.0)
                        and _finite_number(
                            record.get("gpu_junction_temp_c"), minimum=0.0
                        )
                    )
                ):
                    post_grace_gpu_sensors_present = False
                if complete_safety_sample:
                    fully_observed_safety_wall_time_ns.append(wall_time_ns)
    finally:
        os.close(descriptor)
    if manifest is None:
        raise RuntimeError("telemetry artifact contains no records")
    if baseline_count < 1 or preflight_count != 1:
        raise RuntimeError(
            "telemetry_record_sequence_exact: baseline/preflight sequence is incomplete"
        )
    return {
        "manifest": manifest,
        "sample_count": sample_count,
        "baseline_count": baseline_count,
        "preflight_count": preflight_count,
        "measured_count": measured_count,
        "maximum_measured_gap_seconds": maximum_gap,
        "timestamps_monotonic": timestamps_monotonic,
        "present": present,
        "within": within,
        "measured_maximum": measured_maximum,
        "measured_wall_time_ns": measured_wall_time_ns,
        "fully_observed_safety_wall_time_ns": fully_observed_safety_wall_time_ns,
        "measured_wall_timestamps_valid": measured_wall_timestamps_valid,
        "measured_mandatory_fields_valid": measured_mandatory_fields_valid,
        "post_grace_gpu_sensors_present": post_grace_gpu_sensors_present,
    }


def _private_text_artifact(
    path: Path, maximum_bytes: int
) -> tuple[str, dict[str, Any]]:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 1
            or info.st_size <= 0
            or info.st_size > maximum_bytes
        ):
            raise RuntimeError(f"invalid private evidence artifact: {path.name}")
        remaining = info.st_size
        chunks = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise RuntimeError(f"truncated evidence artifact: {path.name}")
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"evidence artifact is not UTF-8: {path.name}") from error
    return text, {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _independent_mask_ir_strings_and_comments(text: str) -> str:
    masked = list(text)
    state = "plain"
    escaped = False
    index = 0
    while index < len(text):
        character = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "quoted":
            if character == "\n":
                raise RuntimeError("compiler IR contains a multiline quoted string")
            masked[index] = " "
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                masked[index] = '"'
                state = "plain"
        elif state == "line_comment":
            if character == "\n":
                state = "plain"
            else:
                masked[index] = " "
        elif state == "block_comment":
            if character != "\n":
                masked[index] = " "
            if character == "/" and following == "*":
                raise RuntimeError("compiler IR contains a nested block comment")
            if character == "*" and following == "/":
                masked[index + 1] = " "
                state = "plain"
                index += 1
        elif character == '"':
            state = "quoted"
            masked[index] = '"'
        elif character == "/" and following == "/":
            state = "line_comment"
            masked[index] = masked[index + 1] = " "
            index += 1
        elif character == "/" and following == "*":
            state = "block_comment"
            masked[index] = masked[index + 1] = " "
            index += 1
        elif character == "*" and following == "/":
            raise RuntimeError("compiler IR contains an orphan block-comment close")
        index += 1
    if state == "quoted":
        raise RuntimeError("compiler IR contains an unterminated quoted string")
    if state == "block_comment":
        raise RuntimeError("compiler IR contains an unterminated block comment")
    return "".join(masked)


def _independent_custom_call_blocks(text: str, dialect: str) -> list[str]:
    raw_lines = text.splitlines()
    masked_lines = _independent_mask_ir_strings_and_comments(text).splitlines()
    if dialect == "stablehlo":
        start = re.compile(r"^(?P<indent>\s*)%[^=]+?=\s*stablehlo\.custom_call\b")
        boundary = re.compile(
            r"^\s*(?:%[^=]+?=|#[A-Za-z_][\w.-]*\s*=|"
            r"(?:stablehlo\.|func\.)?return\b|}\s*$)"
        )
    elif dialect == "optimized_hlo":
        start = re.compile(
            r"^(?P<indent>\s*)(?:ROOT\s+)?[^=]+?=\s*.*\bcustom-call\s*\("
        )
        boundary = re.compile(r"^\s*(?:(?:ROOT\s+)?[^=]+?=|}\s*$)")
    else:
        raise ValueError(f"unsupported IR dialect: {dialect}")
    blocks = []
    index = 0
    while index < len(masked_lines):
        match = start.search(masked_lines[index])
        if match is None:
            index += 1
            continue
        base_indent = len(match.group("indent").expandtabs())
        block = [raw_lines[index]]
        index += 1
        while index < len(masked_lines):
            candidate = masked_lines[index]
            candidate_indent = len(candidate) - len(candidate.lstrip(" \t"))
            if (
                candidate.strip()
                and candidate_indent <= base_indent
                and boundary.match(candidate)
            ):
                break
            block.append(raw_lines[index])
            index += 1
        blocks.append("\n".join(block))
    return blocks


def _independent_top_level_string_attribute_values(
    text: str, attribute: str
) -> list[str]:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", attribute) is None:
        raise ValueError(f"invalid IR attribute name: {attribute}")
    structural = _independent_mask_ir_strings_and_comments(text)
    assignment = re.compile(
        rf"(?<![A-Za-z0-9_.$-]){re.escape(attribute)}"
        rf"(?![A-Za-z0-9_.$-])\s*=\s*\""
    )
    values: list[str] = []
    depths = {"{": 0, "[": 0, "(": 0}
    closing = {"}": "{", "]": "[", ")": "("}
    index = 0
    while index < len(structural):
        if not any(depths.values()):
            match = assignment.match(structural, index)
            if match is not None:
                opening_quote = structural.find('"', match.start(), match.end())
                closing_quote = structural.find('"', opening_quote + 1)
                if opening_quote < 0 or closing_quote < 0:
                    raise RuntimeError("compiler IR string attribute is malformed")
                value = text[opening_quote + 1 : closing_quote]
                if re.fullmatch(r"[A-Za-z0-9_.$-]+", value) is None:
                    raise RuntimeError("compiler IR string attribute is not canonical")
                values.append(value)
                index = closing_quote + 1
                continue
        character = structural[index]
        if character in depths:
            depths[character] += 1
        elif character in closing:
            opening = closing[character]
            depths[opening] -= 1
            if depths[opening] < 0:
                raise RuntimeError("compiler IR delimiters are unbalanced")
        index += 1
    if any(depths.values()):
        raise RuntimeError("compiler IR delimiters are unbalanced")
    return values


def _independent_map_attribute_contents(
    text: str, attribute: str, *, required_brace_depth: int
) -> list[str]:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", attribute) is None:
        raise ValueError(f"invalid IR map attribute name: {attribute}")
    structural = _independent_mask_ir_strings_and_comments(text)
    assignment = re.compile(
        rf"(?<![A-Za-z0-9_.$-]){re.escape(attribute)}"
        rf"(?![A-Za-z0-9_.$-])\s*=\s*\{{"
    )
    contents: list[str] = []
    depths = {"{": 0, "[": 0, "(": 0}
    closing = {"}": "{", "]": "[", ")": "("}
    index = 0
    while index < len(structural):
        if (
            depths["{"] == required_brace_depth
            and depths["["] == 0
            and depths["("] == 0
        ):
            match = assignment.match(structural, index)
            if match is not None:
                opening = structural.rfind("{", match.start(), match.end())
                depth = 1
                cursor = opening + 1
                while cursor < len(structural) and depth:
                    if structural[cursor] == "{":
                        depth += 1
                    elif structural[cursor] == "}":
                        depth -= 1
                    cursor += 1
                if depth:
                    raise RuntimeError("compiler IR map attribute is unterminated")
                contents.append(text[opening + 1 : cursor - 1])
                index = cursor
                continue
        character = structural[index]
        if character in depths:
            depths[character] += 1
        elif character in closing:
            opening = closing[character]
            depths[opening] -= 1
            if depths[opening] < 0:
                raise RuntimeError("compiler IR delimiters are unbalanced")
        index += 1
    if any(depths.values()):
        raise RuntimeError("compiler IR delimiters are unbalanced")
    return contents


def _independent_call_targets(block: str, dialect: str) -> list[str]:
    if dialect == "optimized_hlo":
        return _independent_top_level_string_attribute_values(
            block, "custom_call_target"
        )
    if dialect != "stablehlo":
        raise ValueError(f"unsupported IR dialect: {dialect}")
    structural = _independent_mask_ir_strings_and_comments(block)
    return re.findall(r"\bstablehlo\.custom_call\s+@([A-Za-z0-9_.$-]+)", structural)


def _independent_kernel_name_binding(block: str, dialect: str) -> dict[str, Any]:
    attribute = "mhlo.backend_config" if dialect == "stablehlo" else "backend_config"
    maps = _independent_map_attribute_contents(
        block,
        attribute,
        required_brace_depth=1 if dialect == "stablehlo" else 0,
    )
    names = (
        _independent_top_level_string_attribute_values(maps[0], "name")
        if len(maps) == 1
        else []
    )
    return {"backend_config_map_count": len(maps), "names": names}


def _independent_entry_regions(
    text: str, dialect: str
) -> list[dict[str, tuple[int, int]]]:
    prefix = re.compile(
        r"\bfunc\.func\s+public\s+@main\s*\("
        if dialect == "stablehlo"
        else r"\bENTRY\s+%?main(?:\.[A-Za-z0-9_.-]+)?\s*\("
    )
    regions: list[dict[str, tuple[int, int]]] = []
    for match in prefix.finditer(text):
        parenthesis_depth = 1
        square_depth = 0
        cursor = match.end()
        body_opening = -1
        while cursor < len(text):
            character = text[cursor]
            if character == "(":
                parenthesis_depth += 1
            elif character == ")":
                parenthesis_depth -= 1
                if parenthesis_depth < 0:
                    raise RuntimeError("compiler IR entry header is malformed")
            elif character == "[":
                square_depth += 1
            elif character == "]":
                square_depth -= 1
                if square_depth < 0:
                    raise RuntimeError("compiler IR entry header is malformed")
            elif character == "{" and parenthesis_depth == 0 and square_depth == 0:
                body_opening = cursor
                break
            cursor += 1
        if body_opening < 0:
            raise RuntimeError("compiler IR entry body has no opening brace")
        depth = 1
        cursor = body_opening + 1
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth:
            raise RuntimeError("compiler IR entry body is unterminated")
        regions.append(
            {
                "header": (match.start(), body_opening),
                "body": (body_opening + 1, cursor - 1),
            }
        )
    return regions


def _independent_entry_result_flow(
    text: str, entry: dict[str, tuple[int, int]] | None, dialect: str
) -> dict[str, Any]:
    if entry is None:
        return {"call_results": [], "output_sinks": [], "passed": False}
    body = text[slice(*entry["body"])]
    definition = re.compile(
        r"^\s*(?P<root>ROOT\s+)?(?P<lhs>%[A-Za-z0-9_.-]+)\s*=\s*(?P<rhs>.*)$"
    )
    reference = re.compile(r"%[A-Za-z0-9_.-]+")
    call_pattern = re.compile(
        r"\bstablehlo\.custom_call\b"
        if dialect == "stablehlo"
        else r"\bcustom-call\s*\("
    )
    graph: dict[str, list[str]] = {}
    call_results: list[str] = []
    output_sinks: list[str] = []

    def data_references(rhs: str) -> list[str]:
        if dialect == "stablehlo":
            return reference.findall(rhs.split(" : ", maxsplit=1)[0])
        opening = rhs.find("(")
        if opening < 0:
            return []
        depth = 1
        cursor = opening + 1
        while cursor < len(rhs) and depth:
            if rhs[cursor] == "(":
                depth += 1
            elif rhs[cursor] == ")":
                depth -= 1
            cursor += 1
        if depth:
            raise RuntimeError("optimized HLO opcode operands are malformed")
        return reference.findall(rhs[opening + 1 : cursor - 1])

    for line in body.splitlines():
        match = definition.match(line)
        if match is not None:
            lhs = match.group("lhs")
            rhs = match.group("rhs")
            graph[lhs] = data_references(rhs)
            if call_pattern.search(rhs) is not None:
                call_results.append(lhs)
            if dialect == "optimized_hlo" and match.group("root") is not None:
                output_sinks.append(lhs)
        elif dialect == "stablehlo" and re.match(r"^\s*(?:stablehlo\.)?return\b", line):
            output_sinks.extend(reference.findall(line.split(" : ", maxsplit=1)[0]))

    def reaches_call(node: str, call_result: str, seen: set[str]) -> bool:
        if node == call_result:
            return True
        if node in seen:
            return False
        return any(
            reaches_call(dependency, call_result, seen | {node})
            for dependency in graph.get(node, [])
        )

    passed = (
        len(call_results) == 1
        and len(output_sinks) == 1
        and reaches_call(output_sinks[0], call_results[0], set())
    )
    return {
        "call_results": call_results,
        "output_sinks": output_sinks,
        "passed": passed,
    }


def _independent_ir_gate(stable_text: str, optimized_text: str) -> dict[str, Any]:
    stable_structural = _independent_mask_ir_strings_and_comments(stable_text)
    optimized_structural = _independent_mask_ir_strings_and_comments(optimized_text)
    marker = "skyrl_qwen35_w8a8_lora_forward"
    stable_blocks = _independent_custom_call_blocks(stable_text, "stablehlo")
    optimized_blocks = _independent_custom_call_blocks(optimized_text, "optimized_hlo")
    forbidden_patterns = (
        r"(?i)(?:hip|cuda)?graph(?:launch|exec|instantiate)?|capture",
        r"(?i)command[_ -]?buffer",
        r"(?i)replay",
    )
    combined = f"{stable_text}\n{optimized_text}"
    stable_block = stable_blocks[0] if len(stable_blocks) == 1 else ""
    optimized_block = optimized_blocks[0] if len(optimized_blocks) == 1 else ""
    expected_stable_signature = [
        "tensor<3x64xbf16>",
        "tensor<64x17xi8>",
        "tensor<1x17xbf16>",
        "tensor<64x8xbf16>",
        "tensor<8x17xbf16>",
        "tensor<f32>",
        "tensor<3x17xbf16>",
    ]
    expected_optimized_signature = [
        "bf16[3,64]",
        "s8[64,17]",
        "bf16[1,17]",
        "bf16[64,8]",
        "bf16[8,17]",
        "f32[]",
        "bf16[3,17]",
    ]
    stable_entries = _independent_entry_regions(stable_structural, "stablehlo")
    optimized_entries = _independent_entry_regions(
        optimized_structural, "optimized_hlo"
    )
    stable_entry = stable_entries[0] if len(stable_entries) == 1 else None
    optimized_entry = optimized_entries[0] if len(optimized_entries) == 1 else None
    stable_signature = (
        re.findall(
            r"tensor<[^>]+>",
            stable_structural[slice(*stable_entry["header"])],
        )
        if stable_entry is not None
        else []
    )
    optimized_signature = (
        re.findall(
            r"(?:bf16|s8|f32)\[[^\]]*\]",
            optimized_structural[slice(*optimized_entry["header"])],
        )
        if optimized_entry is not None
        else []
    )
    stable_call_positions = [
        match.start()
        for match in re.finditer(r"\bstablehlo\.custom_call\b", stable_structural)
    ]
    optimized_call_positions = [
        match.start()
        for match in re.finditer(r"\bcustom-call\s*\(", optimized_structural)
    ]
    stable_entry_span = stable_entry["body"] if stable_entry is not None else None
    optimized_entry_span = (
        optimized_entry["body"] if optimized_entry is not None else None
    )
    stable_result_flow = _independent_entry_result_flow(
        stable_structural, stable_entry, "stablehlo"
    )
    optimized_result_flow = _independent_entry_result_flow(
        optimized_structural, optimized_entry, "optimized_hlo"
    )
    checks = {
        "stablehlo_parser_matches_raw_custom_calls": len(stable_blocks)
        == len(stable_call_positions)
        == 1,
        "optimized_hlo_parser_matches_raw_custom_calls": len(optimized_blocks)
        == len(optimized_call_positions)
        == 1,
        "stablehlo_call_binds_exact_target": _independent_call_targets(
            stable_block, "stablehlo"
        )
        == ["__gpu$xla.gpu.triton"],
        "optimized_hlo_call_binds_exact_target": _independent_call_targets(
            optimized_block, "optimized_hlo"
        )
        == ["__gpu$xla.gpu.triton"],
        "stablehlo_call_binds_exact_forward_kernel_name": (
            _independent_kernel_name_binding(stable_block, "stablehlo")
            == {"backend_config_map_count": 1, "names": [marker]}
        ),
        "optimized_hlo_call_binds_exact_forward_kernel_name": (
            _independent_kernel_name_binding(optimized_block, "optimized_hlo")
            == {"backend_config_map_count": 1, "names": [marker]}
        ),
        "stablehlo_unique_public_main": len(stable_entries) == 1,
        "optimized_hlo_unique_entry": len(optimized_entries) == 1,
        "stablehlo_exact_public_main_signature": stable_signature
        == expected_stable_signature,
        "optimized_hlo_exact_entry_signature": optimized_signature
        == expected_optimized_signature,
        "stablehlo_call_owned_by_public_main": stable_entry_span is not None
        and len(stable_call_positions) == 1
        and stable_entry_span[0] <= stable_call_positions[0] < stable_entry_span[1],
        "optimized_hlo_call_owned_by_entry": optimized_entry_span is not None
        and len(optimized_call_positions) == 1
        and optimized_entry_span[0]
        <= optimized_call_positions[0]
        < optimized_entry_span[1],
        "stablehlo_entry_result_depends_on_call": stable_result_flow["passed"] is True,
        "optimized_hlo_entry_result_depends_on_call": optimized_result_flow["passed"]
        is True,
        "no_backward_marker": "w8a16_lora_input_vjp" not in combined,
        "no_stablehlo_outer_loop": re.search(r"\bstablehlo\.while\b", stable_structural)
        is None,
        "no_optimized_hlo_outer_loop": re.search(r"\bwhile\s*\(", optimized_structural)
        is None,
        "no_graph_capture_command_buffer_or_replay_marker": not any(
            re.search(pattern, combined) for pattern in forbidden_patterns
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _final_run_inventory(root: Path) -> dict[str, Any]:
    entries = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise RuntimeError(f"run evidence contains a symlink: {relative}")
        info = path.stat(follow_symlinks=False)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise RuntimeError(f"run evidence is not private: {relative}")
        if path.is_dir():
            continue
        if relative == Path("controller.jsonl"):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"run evidence is not a regular file: {relative}")
        if len(entries) >= 512 or info.st_size < 0 or info.st_size > 128 * 1024**2:
            raise RuntimeError("run evidence exceeds its per-file or file-count cap")
        total_bytes += info.st_size
        if total_bytes > 1024**3:
            raise RuntimeError("run evidence exceeds its one-GiB total cap")
        entries.append(
            {
                "path": str(relative),
                "bytes": info.st_size,
                "sha256": _file_sha256(path),
            }
        )
    return {
        "file_count_excluding_open_controller_log": len(entries),
        "total_bytes_excluding_open_controller_log": total_bytes,
        "maximum_file_count": 512,
        "maximum_file_bytes": 128 * 1024**2,
        "maximum_total_bytes": 1024**3,
        "files": entries,
    }


def _independent_compiler_artifact_inventory(root: Path) -> dict[str, Any]:
    paths = sorted(root.rglob("*"))
    if len(paths) > 1024:
        raise RuntimeError("compiler artifact node count exceeds controller cap")
    entries = []
    total_bytes = 0
    for path in paths:
        relative = path.relative_to(root)
        if path.is_symlink():
            raise RuntimeError(f"compiler artifact is a symlink: {relative}")
        info = path.stat(follow_symlinks=False)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise RuntimeError(f"compiler artifact is not private: {relative}")
        if path.is_dir():
            continue
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size < 0
            or info.st_size > 128 * 1024**2
            or len(entries) >= 256
        ):
            raise RuntimeError(f"invalid compiler artifact: {relative}")
        total_bytes += info.st_size
        if total_bytes > 512 * 1024**2:
            raise RuntimeError("compiler artifacts exceed controller total cap")
        entries.append(
            {
                "path": str(relative),
                "bytes": info.st_size,
                "sha256": _file_sha256(path),
            }
        )
    return {
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "maximum_file_count": 256,
        "maximum_file_bytes": 128 * 1024**2,
        "maximum_total_bytes": 512 * 1024**2,
        "files": entries,
    }


def _independent_fresh_isa_qualification(
    artifact_root: Path, artifact_inventory: dict[str, Any]
) -> dict[str, Any]:
    from rocm.inspect_w8a8_lora_isa import inspect_cache

    candidates = [
        item
        for item in artifact_inventory.get("files", [])
        if re.fullmatch(
            r"jax-cache/jit_candidate-[0-9a-f]{64}-cache", item.get("path", "")
        )
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"controller expected one candidate cache, observed {len(candidates)}"
        )
    cache_manifest = candidates[0]
    cache_path = artifact_root / cache_manifest["path"]
    evidence = inspect_cache(
        cache_path,
        expected_cache_sha256=cache_manifest["sha256"],
        expected_elf_sha256=_EXPECTED_NESTED_ELF_SHA256,
    )
    isa = evidence.get("isa", {})
    resources = isa.get("resources", {})
    candidate = evidence.get("candidate", {})
    elf_inventory = evidence.get("elf_inventory", {})
    checks = {
        "status_exact": evidence.get("status") == "passed_offline_isa_verification",
        "fresh_cache_sha_exact": evidence.get("cache", {}).get("path")
        == str(cache_path)
        and evidence.get("cache", {}).get("sha256") == cache_manifest["sha256"]
        and evidence.get("cache", {}).get("expected_sha256_matched") is True,
        "candidate_exact": _exact_int(
            candidate.get("bytes"), _EXPECTED_NESTED_ELF_BYTES
        )
        and candidate.get("sha256") == _EXPECTED_NESTED_ELF_SHA256
        and candidate.get("expected_sha256_matched") is True
        and candidate.get("written_elf") is None,
        "unique_candidate_exact": _exact_int(elf_inventory.get("elf_count"), 6)
        and _exact_int(elf_inventory.get("unique_exact_symbol_candidate_count"), 1),
        "isa_identity_exact": isa.get("symbol") == _EXPECTED_KERNEL_NAME
        and isa.get("amdgpu_target") == _EXPECTED_NESTED_ELF_TARGET,
        "wmma_exact": isa.get("instruction") == "v_wmma_i32_16x16x16_iu8"
        and _exact_int(isa.get("static_instruction_count"), 4)
        and _json_exact(isa.get("signed_neg_lo"), [1, 1, 0]),
        "resources_exact": _exact_int(resources.get("sgpr_count"), 34)
        and _exact_int(resources.get("vgpr_count"), 62)
        and _exact_int(resources.get("sgpr_spill_count"), 0)
        and _exact_int(resources.get("vgpr_spill_count"), 0)
        and _exact_int(resources.get("private_segment_fixed_size"), 0),
        "offline_no_device_access": evidence.get("offline_only") is True
        and evidence.get("device_access_performed") is False
        and evidence.get("jax_modules_imported_by_verifier") is False,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(
            "controller fresh ISA qualification failed: " + ", ".join(failed)
        )
    return {"checks": checks, "passed": True, "evidence": evidence}


def _dispatch_telemetry_bracket(
    measured_wall_time_ns: list[int], start_ns: int, completion_ns: int
) -> dict[str, Any]:
    pre = [value for value in measured_wall_time_ns if value <= start_ns]
    post = [value for value in measured_wall_time_ns if value >= completion_ns]
    before_seconds = (start_ns - max(pre)) / 1e9 if pre else None
    after_seconds = (min(post) - completion_ns) / 1e9 if post else None
    passed = (
        before_seconds is not None
        and after_seconds is not None
        and 0 <= before_seconds <= _MAX_TELEMETRY_DISPATCH_BRACKET_SECONDS
        and 0 <= after_seconds <= _MAX_TELEMETRY_DISPATCH_BRACKET_SECONDS
    )
    return {
        "sample_before_dispatch_wall_time_ns": max(pre) if pre else None,
        "sample_after_dispatch_wall_time_ns": min(post) if post else None,
        "seconds_before_dispatch": before_seconds,
        "seconds_after_dispatch": after_seconds,
        "maximum_seconds": _MAX_TELEMETRY_DISPATCH_BRACKET_SECONDS,
        "passed": passed,
    }


def _audit_runtime_profile_outputs(
    run_dir: Path,
    *,
    profile_returncode: int,
    wait_audit: dict[str, Any],
    expected_lock_fd: int,
    expected_scope_unit: str,
    expected_device_identity: dict[str, Any],
) -> dict[str, Any]:
    expected_lock_stat = os.fstat(expected_lock_fd)
    telemetry = _summarize_private_telemetry(run_dir / "telemetry.jsonl", 64 * 1024**2)
    summary = _private_json_object(run_dir / "telemetry.jsonl.summary.json", 1024**2)
    probe = _private_json_lines(run_dir / "probe.jsonl", 8 * 1024**2)
    manifest = telemetry["manifest"]
    limits = manifest.get("safety_limits")
    gpu = manifest.get("gpu")
    expected_profile_command = _profile_command(
        phase="execute", run_dir=run_dir, card="card1", lock_fd=expected_lock_fd
    )
    expected_child_command = expected_profile_command[
        expected_profile_command.index("--") + 1 :
    ]
    expected_parent_command_sha256 = hashlib.sha256(
        b"\0".join(argument.encode() for argument in expected_profile_command) + b"\0"
    ).hexdigest()
    expected_probe_types = [
        "manifest",
        "static_preflight",
        "hardware_preflight",
        "host_oracle",
        "backend_ready",
        "journal_checkpoint",
        "lowered",
        "compiled_unreleased",
        "fresh_isa_qualification",
        "executable_released",
        "input_device_put",
        "journal_checkpoint",
        "dispatch_preflight",
        "dispatch_started",
        "journal_checkpoint",
        "dispatch",
        "device_get",
        "journal_checkpoint",
        "numerical_validation",
        "completed",
    ]
    probe_types = [record.get("record_type") for record in probe]
    by_type: dict[str, list[dict[str, Any]]] = {}
    for record in probe:
        by_type.setdefault(str(record.get("record_type")), []).append(record)

    def only(record_type: str) -> dict[str, Any]:
        records = by_type.get(record_type, [])
        return records[0] if len(records) == 1 else {}

    journal_records = by_type.get("journal_checkpoint", [])
    journal_stages = [record.get("stage") for record in journal_records]
    journal_by_stage = {str(record.get("stage")): record for record in journal_records}
    probe_manifest = only("manifest")
    static_preflight = only("static_preflight")
    hardware_preflight = only("hardware_preflight")
    hardware = hardware_preflight.get("hardware", {})
    host_oracle = only("host_oracle")
    backend_ready = only("backend_ready")
    lowered = only("lowered")
    compiled = only("compiled_unreleased")
    fresh_isa = only("fresh_isa_qualification")
    released = only("executable_released")
    input_put = only("input_device_put")
    dispatch_preflight = only("dispatch_preflight")
    dispatch_hardware = dispatch_preflight.get("hardware_limits", {})
    dispatch_started = only("dispatch_started")
    dispatch = only("dispatch")
    device_get = only("device_get")
    numerical_record = only("numerical_validation")
    completed = only("completed")

    stable_text, stable_ir = _private_text_artifact(
        run_dir / "w8a8-forward.stablehlo.mlir", 8 * 1024**2
    )
    optimized_text, optimized_ir = _private_text_artifact(
        run_dir / "w8a8-forward.optimized.hlo", 32 * 1024**2
    )
    independent_ir = _independent_ir_gate(stable_text, optimized_text)
    artifact_root = run_dir / "compiler-artifacts"
    independent_artifacts = _independent_compiler_artifact_inventory(artifact_root)
    independent_isa = _independent_fresh_isa_qualification(
        artifact_root, independent_artifacts
    )

    start_wall_ns = dispatch.get("dispatch_started_wall_time_ns")
    completion_wall_ns = dispatch.get("dispatch_completed_wall_time_ns")
    start_monotonic_ns = dispatch.get("dispatch_started_monotonic_ns")
    completion_monotonic_ns = dispatch.get("dispatch_completed_monotonic_ns")
    dispatch_seconds = dispatch.get("dispatch_seconds_including_output_readiness")
    dispatch_times_valid = all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in (
            start_wall_ns,
            completion_wall_ns,
            start_monotonic_ns,
            completion_monotonic_ns,
        )
    )
    telemetry_bracket = (
        _dispatch_telemetry_bracket(
            telemetry["measured_wall_time_ns"], start_wall_ns, completion_wall_ns
        )
        if dispatch_times_valid
        else {"passed": False}
    )
    sensitivity = host_oracle.get("sensitivity", {})
    sensitivity_comparisons = sensitivity.get("comparisons", [])
    numerical = numerical_record.get("validation", {})
    release_gate = compiled.get("release_gate", {})
    child_isa = fresh_isa.get("qualification", {})
    child_isa_evidence = child_isa.get("evidence", {})
    independent_isa_evidence = independent_isa.get("evidence", {})
    one_shot_started = dispatch_started.get("one_shot_capability", {})
    one_shot_dispatch = dispatch.get("one_shot_capability", {})
    one_shot_get = device_get.get("one_shot_capability", {})
    one_shot_complete = completed.get("one_shot_capability", {})
    maximum_measured_gap = telemetry["maximum_measured_gap_seconds"]
    expected_scope = f"{expected_scope_unit}.scope"
    expected_control_group = (
        f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/"
        f"app.slice/{expected_scope}"
    )
    expected_probe_sources = {
        "probe": _EXPECTED_SOURCE_SHA256["child"],
        "isa_inspector": _EXPECTED_SOURCE_SHA256["isa_inspector"],
        "kernel": _EXPECTED_SOURCE_SHA256["kernel"],
        "quantized_reference": _EXPECTED_SOURCE_SHA256["quantized_reference"],
        "safety": _EXPECTED_SOURCE_SHA256["safety"],
        "handoff": _EXPECTED_SOURCE_SHA256["handoff"],
        "profiler": _EXPECTED_SOURCE_SHA256["profiler"],
        "package_skyrl": _EXPECTED_SOURCE_SHA256["package_skyrl"],
        "package_tx": _EXPECTED_SOURCE_SHA256["package_tx"],
        "package_kernels": _EXPECTED_SOURCE_SHA256["package_kernels"],
        "package_rocm": _EXPECTED_SOURCE_SHA256["package_rocm"],
        "sealed_loader": _EXPECTED_SOURCE_SHA256["sealed_loader"],
        "package_top_rocm": _EXPECTED_SOURCE_SHA256["package_top_rocm"],
    }
    expected_probe_environment = {
        "JAX_PLATFORMS": "rocm",
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
        "JAX_ROCM_VISIBLE_DEVICES": "0",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_CLIENT_MEM_FRACTION": "0.075",
        "XLA_FLAGS": _COMMAND_BUFFER_FLAG,
        "JAX_COMPILATION_CACHE_DIR": str(run_dir / "compiler-artifacts/jax-cache"),
        "JAX_ENABLE_COMPILATION_CACHE": "true",
        "JAX_RAISE_PERSISTENT_CACHE_ERRORS": "true",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
        "TF_XLA_HSACO_CACHE_DIR": str(run_dir / "compiler-artifacts/hsaco-cache"),
        "TRITON_CACHE_DIR": str(run_dir / "compiler-artifacts/triton-cache"),
        "TRITON_DUMP_DIR": str(run_dir / "compiler-artifacts/triton-dump"),
        "TRITON_KERNEL_DUMP": "1",
        "AMDGCN_ENABLE_DUMP": "1",
        "TEST_UNDECLARED_OUTPUTS_DIR": str(
            run_dir / "compiler-artifacts/compiler-dump"
        ),
    }
    expected_profile_environment = {
        "HIP_VISIBLE_DEVICES": "0",
        "JAX_PLATFORMS": "rocm",
        "JAX_ROCM_VISIBLE_DEVICES": "0",
        "XLA_CLIENT_MEM_FRACTION": "0.075",
        "XLA_FLAGS": _COMMAND_BUFFER_FLAG,
        "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    expected_module_origins = {
        "jax": str(
            (_repo() / ".venv/lib/python3.12/site-packages/jax/__init__.py").resolve(
                strict=True
            )
        ),
        "jaxlib": str(
            (_repo() / ".venv/lib/python3.12/site-packages/jaxlib/__init__.py").resolve(
                strict=True
            )
        ),
        "numpy": str(
            (_repo() / ".venv/lib/python3.12/site-packages/numpy/__init__.py").resolve(
                strict=True
            )
        ),
        "quantized_lora": str(
            (_repo() / "skyrl/tx/kernels/quantized_lora.py").resolve(strict=True)
        ),
        "w8a8_lora": str(
            (_repo() / "skyrl/tx/kernels/rocm/w8a8_lora.py").resolve(strict=True)
        ),
    }

    def probe_timestamps_are_exact() -> bool:
        previous: datetime | None = None
        for record in probe:
            raw = record.get("timestamp")
            if not isinstance(raw, str):
                return False
            try:
                parsed = datetime.fromisoformat(raw)
            except ValueError:
                return False
            if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
                return False
            if previous is not None and parsed < previous:
                return False
            previous = parsed
        return True

    def stack_is_exact(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        if set(value) != {*_EXPECTED_STACK_VERSIONS, "runtime_binaries"}:
            return False
        if {name: value.get(name) for name in _EXPECTED_STACK_VERSIONS} != (
            _EXPECTED_STACK_VERSIONS
        ):
            return False
        binaries = value.get("runtime_binaries")
        return (
            isinstance(binaries, dict)
            and set(binaries) == set(_EXPECTED_JAX_RUNTIME_SHA256)
            and all(
                isinstance(binaries.get(name), dict)
                and binaries[name].get("sha256") == digest
                and type(binaries[name].get("bytes")) is int
                and binaries[name]["bytes"] > 0
                and type(binaries[name].get("mode")) is int
                and binaries[name]["mode"] >= 0
                for name, digest in _EXPECTED_JAX_RUNTIME_SHA256.items()
            )
        )

    def hardware_limits_are_exact(value: Any) -> bool:
        if not isinstance(value, dict) or set(value) != {
            "power_cap_path",
            "power_cap_uw",
            "maximum_power_cap_uw",
            "junction_path",
            "junction_temperature_millic",
            "maximum_launch_junction_millic",
        }:
            return False
        device_root = expected_device_identity.get("pci_sysfs_path")
        if not isinstance(device_root, str) or not device_root.startswith(
            "/sys/devices/"
        ):
            return False
        power_path = value.get("power_cap_path")
        junction_path = value.get("junction_path")
        if not isinstance(power_path, str) or not isinstance(junction_path, str):
            return False
        power_match = re.fullmatch(
            re.escape(device_root) + r"/hwmon/(hwmon[0-9]+)/power1_cap", power_path
        )
        junction_match = re.fullmatch(
            re.escape(device_root) + r"/hwmon/(hwmon[0-9]+)/temp[0-9]+_input",
            junction_path,
        )
        return (
            power_match is not None
            and junction_match is not None
            and power_match.group(1) == junction_match.group(1)
            and type(value.get("power_cap_uw")) is int
            and value.get("power_cap_uw") > 0
            and value.get("power_cap_uw") <= 315_000_000
            and type(value.get("maximum_power_cap_uw")) is int
            and value.get("maximum_power_cap_uw") == 315_000_000
            and type(value.get("junction_temperature_millic")) is int
            and value.get("junction_temperature_millic") >= 0
            and value.get("junction_temperature_millic") <= 85_000
            and type(value.get("maximum_launch_junction_millic")) is int
            and value.get("maximum_launch_junction_millic") == 85_000
        )

    def profile_runtime_is_exact(value: Any) -> bool:
        return _json_exact(
            value,
            {
                "python": sys.version,
                "platform": platform.platform(),
                "rocm": _EXPECTED_STACK_VERSIONS["rocm"],
                "jax": _EXPECTED_STACK_VERSIONS["jax"],
                "jaxlib": _EXPECTED_STACK_VERSIONS["jaxlib"],
                "jax_rocm_plugin": _EXPECTED_STACK_VERSIONS["jax-rocm7-plugin"],
                "jax_rocm_pjrt": _EXPECTED_STACK_VERSIONS["jax-rocm7-pjrt"],
                "script_sha256": _EXPECTED_SOURCE_SHA256["profiler"],
                "accelerator_environment": expected_profile_environment,
            },
        )

    def profile_gpu_identity_is_exact(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        device_root = expected_device_identity.get("pci_sysfs_path")
        launch_limits = hardware.get("pre_backend_hardware_limits")
        if not isinstance(device_root, str) or not isinstance(launch_limits, dict):
            return False
        power_path = launch_limits.get("power_cap_path")
        junction_path = launch_limits.get("junction_path")
        if not isinstance(power_path, str) or not isinstance(junction_path, str):
            return False
        power_match = re.fullmatch(
            re.escape(device_root) + r"/hwmon/(hwmon[0-9]+)/power1_cap",
            power_path,
        )
        junction_match = re.fullmatch(
            re.escape(device_root) + r"/hwmon/(hwmon[0-9]+)/temp[0-9]+_input",
            junction_path,
        )
        if (
            power_match is None
            or junction_match is None
            or power_match.group(1) != junction_match.group(1)
        ):
            return False
        card = expected_device_identity.get("drm_card")
        return isinstance(card, str) and _json_exact(
            value,
            {
                "card": card,
                "card_path": f"/sys/class/drm/{card}",
                "pci_bdf": expected_device_identity.get("pci_bdf"),
                "vendor_id": expected_device_identity.get("vendor_id"),
                "device_id": expected_device_identity.get("device_id"),
                "subsystem_vendor_id": _EXPECTED_SUBSYSTEM_VENDOR_ID,
                "subsystem_device_id": _EXPECTED_SUBSYSTEM_DEVICE_ID,
                "hwmon": (f"/sys/class/drm/{card}/device/hwmon/{power_match.group(1)}"),
                "hwmon_name": "amdgpu",
            },
        )

    expected_runtime_dispatch_plan = {
        "host_oracle_attempts": 1,
        "host_oracle_completions": 1,
        "host_sensitivity_comparisons": 35,
        "backend_initialization_attempts": 1,
        "backend_initialization_completions": 1,
        "lower_attempts": 1,
        "lower_completions": 1,
        "compile_attempts": 1,
        "compile_completions": 1,
        "isa_qualification_attempts": 1,
        "isa_qualification_completions": 1,
        "tuple_device_put_attempts": 1,
        "tuple_device_put_completions": 1,
        "device_put_leaves": 6,
        "input_readiness_invocations": 1,
        "compiled_executable_invocations": 1,
        "compiled_executable_completions": 1,
        "output_readiness_invocations": 1,
        "device_get_attempts": 1,
        "device_get_completions": 1,
        "device_get_leaves": 1,
        "backward_invocations": 0,
        "warmup_invocations": 0,
        "replay_invocations": 0,
        "gpu_reference_invocations": 0,
        "device_error_reduction_invocations": 0,
        "model_invocations": 0,
    }
    expected_probe_contract = {
        "model_family": "Qwen/Qwen3.5-4B",
        "operation": "w8a8_group64_rank8_lora_forward_only",
        "phase": "execute",
        "inputs": [
            {"name": "x", "shape": [3, 64], "dtype": "bfloat16"},
            {"name": "weight_codes", "shape": [64, 17], "dtype": "int8"},
            {
                "name": "weight_scales",
                "shape": [1, 17],
                "dtype": "bfloat16",
            },
            {"name": "lora_a", "shape": [64, 8], "dtype": "bfloat16"},
            {"name": "lora_b", "shape": [8, 17], "dtype": "bfloat16"},
            {"name": "lora_scaling", "shape": [], "dtype": "float32"},
        ],
        "output": {"shape": [3, 17], "dtype": "bfloat16"},
        "tiles": {
            "group_size": 64,
            "block_m": 16,
            "block_n": 16,
            "row_superblock": 16,
        },
        "dispatch_plan": expected_runtime_dispatch_plan,
        "runtime_numerical_gate_evaluated": True,
        "runtime_promotion": False,
        "execute_rung_enabled": True,
        "outer_profile_rocm_required": True,
        "exact_idle_handoff_required": True,
    }
    expected_sensitivity_labels = {
        *(
            f"global_zero_{name}"
            for name in (
                "x",
                "weight_codes",
                "weight_scales",
                "lora_a",
                "lora_b",
                "lora_scaling",
            )
        ),
        "global_zero_output",
        *(f"omitted_output_row_{index}" for index in range(3)),
        *(f"omitted_output_column_{index}" for index in range(17)),
        *(f"omitted_lora_rank_{index}" for index in range(8)),
    }
    expected_numerical_checks = {
        "actual_shape_exact",
        "expected_shape_exact",
        "actual_dtype_bfloat16",
        "expected_dtype_bfloat16",
        "actual_nbytes_exact",
        "expected_nbytes_exact",
        "actual_c_contiguous",
        "expected_c_contiguous",
        "all_51_actual_elements_finite",
        "all_51_expected_elements_finite",
        "reference_norm_finite_nonzero",
        "relative_l2_below_one_percent",
        "cosine_at_least_0_9999",
        "maximum_absolute_error_at_most_0_25",
        "dispatch_finite_below_one_second",
    }
    expected_numerical_limits = {
        "maximum_relative_l2_error_exclusive": 0.01,
        "minimum_cosine_similarity_inclusive": 0.9999,
        "maximum_absolute_error_inclusive": 0.25,
        "maximum_dispatch_seconds_exclusive": 1.0,
    }
    expected_child_isa_checks = {
        "status_exact",
        "offline_inspector",
        "caller_bound_fresh_cache",
        "one_unique_exact_symbol_candidate",
        "candidate_bytes_exact",
        "candidate_sha256_exact",
        "candidate_not_written_to_disk",
        "symbol_exact",
        "target_exact",
        "four_static_signed_iu8_wmma",
        "registers_exact",
        "zero_spills_and_private_segment",
    }
    checks = {
        "wait_supervisor_passed": wait_audit.get("passed") is True,
        "dispatch_watchdog_passed": wait_audit.get("dispatch_watchdog", {}).get(
            "enabled"
        )
        is True
        and wait_audit.get("dispatch_watchdog", {}).get("timeout_seconds")
        == _DISPATCH_WATCHDOG_SECONDS
        and wait_audit.get("dispatch_watchdog", {}).get("dispatch_started") is True
        and wait_audit.get("dispatch_watchdog", {}).get("post_attempt_observed") is True
        and wait_audit.get("dispatch_watchdog", {}).get(
            "validated_protocol_prefix_length"
        )
        == len(_RUNTIME_PROBE_PROTOCOL)
        and wait_audit.get("dispatch_watchdog", {}).get("protocol_prefix_exact") is True
        and wait_audit.get("dispatch_watchdog", {}).get("post_attempt_record_types")
        == ["journal_checkpoint", "dispatch"]
        and wait_audit.get("dispatch_watchdog", {}).get("dispatch_started_wall_time_ns")
        == start_wall_ns
        and wait_audit.get("dispatch_watchdog", {}).get("dispatch_started_monotonic_ns")
        == start_monotonic_ns
        and wait_audit.get("dispatch_watchdog", {}).get("post_attempt_wall_time_ns")
        == completion_wall_ns
        and wait_audit.get("dispatch_watchdog", {}).get("post_attempt_monotonic_ns")
        == completion_monotonic_ns
        and "evidence_error" not in wait_audit.get("dispatch_watchdog", {})
        and wait_audit.get("dispatch_watchdog", {}).get("passed") is True,
        "profile_returncode_zero": type(profile_returncode) is int
        and profile_returncode == 0,
        "telemetry_manifest_exact": manifest.get("record_type") == "manifest",
        "telemetry_protocol_exact": manifest.get("interval_seconds")
        == _PROFILE_INTERVAL_SECONDS
        and manifest.get("baseline_seconds") == _PROFILE_BASELINE_SECONDS
        and manifest.get("duration_seconds") is None
        and manifest.get("timeout_seconds") == _PROFILE_TIMEOUT_SECONDS
        and manifest.get("sensor_grace_seconds") == _PROFILE_SENSOR_GRACE_SECONDS
        and manifest.get("terminate_included_on_safety") is False
        and manifest.get("command_recorded") is True
        and _exact_int(manifest.get("passed_file_descriptor_count"), 1),
        "telemetry_profiler_source_exact": manifest.get("runtime", {}).get(
            "script_sha256"
        )
        == _EXPECTED_SOURCE_SHA256["profiler"],
        "telemetry_profiler_runtime_exact": profile_runtime_is_exact(
            manifest.get("runtime")
        ),
        "telemetry_wrapped_child_command_exact": manifest.get("command")
        == expected_child_command,
        "gpu_identity_exact": profile_gpu_identity_is_exact(gpu),
        "limits_exact": isinstance(limits, dict)
        and _finite_number(limits.get("max_junction_temp_c"), minimum=0.0)
        and limits.get("max_junction_temp_c") == _MAX_JUNCTION_TEMP_C
        and _finite_number(limits.get("max_gpu_power_watts"), minimum=0.0)
        and limits.get("max_gpu_power_watts") == _MAX_GPU_POWER_WATTS
        and _finite_number(limits.get("max_vram_bytes"), minimum=0.0)
        and limits.get("max_vram_bytes") == _MAX_VRAM_GIB * 1024**3
        and _finite_number(limits.get("min_host_available_bytes"), minimum=0.0)
        and limits.get("min_host_available_bytes") == 0.0
        and _finite_number(limits.get("max_swap_bytes"), minimum=0.0)
        and limits.get("max_swap_bytes") == _MAX_SWAP_GIB * 1024**3,
        "summary_completed_cleanly": summary.get("record_type") == "summary"
        and summary.get("status") == "completed"
        and _exact_int(summary.get("returncode"), 0)
        and summary.get("received_signal") is None
        and summary.get("kernel_log_available") is True
        and "kernel_driver_errors" not in summary
        and "safety_violation" not in summary
        and all(
            name not in summary
            for name in (
                "error_type",
                "error",
                "terminated_explicit_pids",
                "surviving_explicit_pids",
            )
        ),
        "telemetry_record_sequence_exact": telemetry["preflight_count"] == 1
        and telemetry["sample_count"]
        == telemetry["baseline_count"]
        + telemetry["preflight_count"]
        + telemetry["measured_count"],
        "summary_sample_counts_match": _exact_int(
            summary.get("samples"), telemetry["sample_count"]
        )
        and _exact_int(summary.get("baseline_samples"), telemetry["baseline_count"])
        and _exact_int(summary.get("measured_samples"), telemetry["measured_count"]),
        "summary_measured_maxima_match": _finite_number(
            summary.get("metrics", {}).get("gpu_power_watts", {}).get("measured_max"),
            minimum=0.0,
        )
        and summary.get("metrics", {}).get("gpu_power_watts", {}).get("measured_max")
        == telemetry["measured_maximum"]["power"]
        and _finite_number(
            summary.get("metrics", {})
            .get("gpu_junction_temp_c", {})
            .get("measured_max"),
            minimum=0.0,
        )
        and summary.get("metrics", {})
        .get("gpu_junction_temp_c", {})
        .get("measured_max")
        == telemetry["measured_maximum"]["temperature"]
        and _finite_number(
            summary.get("metrics", {}).get("vram_used_bytes", {}).get("measured_max"),
            minimum=0.0,
        )
        and summary.get("metrics", {}).get("vram_used_bytes", {}).get("measured_max")
        == telemetry["measured_maximum"]["vram"]
        and _finite_number(
            summary.get("metrics", {})
            .get("host_swap_used_bytes", {})
            .get("measured_max"),
            minimum=0.0,
        )
        and summary.get("metrics", {})
        .get("host_swap_used_bytes", {})
        .get("measured_max")
        == telemetry["measured_maximum"]["host_swap"],
        "telemetry_samples_sufficient": telemetry["baseline_count"] >= 1
        and telemetry["preflight_count"] == 1
        and telemetry["measured_count"] >= 2
        and telemetry["timestamps_monotonic"] is True
        and maximum_measured_gap <= 2.0,
        "telemetry_wall_times_exact": telemetry["measured_wall_timestamps_valid"]
        is True
        and len(telemetry["measured_wall_time_ns"]) == telemetry["measured_count"]
        and all(
            right > left
            for left, right in zip(
                telemetry["measured_wall_time_ns"],
                telemetry["measured_wall_time_ns"][1:],
                strict=False,
            )
        ),
        "telemetry_limits_present_and_within": all(telemetry["present"].values())
        and all(telemetry["within"].values())
        and telemetry["measured_mandatory_fields_valid"] is True
        and telemetry["post_grace_gpu_sensors_present"] is True,
        "telemetry_brackets_dispatch": telemetry_bracket.get("passed") is True
        and _finite_number(
            telemetry_bracket.get("seconds_before_dispatch"), minimum=0.0
        )
        and telemetry_bracket.get("seconds_before_dispatch") <= 0.20
        and _finite_number(telemetry_bracket.get("seconds_after_dispatch"), minimum=0.0)
        and telemetry_bracket.get("seconds_after_dispatch") <= 0.20,
        "telemetry_dispatch_bracket_metrics_complete": telemetry_bracket.get("passed")
        is True
        and telemetry_bracket.get("sample_before_dispatch_wall_time_ns")
        in telemetry["fully_observed_safety_wall_time_ns"]
        and telemetry_bracket.get("sample_after_dispatch_wall_time_ns")
        in telemetry["fully_observed_safety_wall_time_ns"],
        "probe_record_sequence_exact": probe_types == expected_probe_types,
        "probe_timestamps_exact_and_ordered": probe_timestamps_are_exact(),
        "probe_no_error_record": all(
            record.get("record_type") != "error" for record in probe
        ),
        "runtime_scope_exact": static_preflight.get("controller_supervision", {}).get(
            "validated"
        )
        is True
        and static_preflight.get("controller_supervision", {}).get("scope")
        == expected_scope
        and static_preflight.get("controller_supervision", {}).get("cgroup")
        == expected_control_group
        and wait_audit.get("control_group") == expected_control_group
        and static_preflight.get("controller_supervision", {}).get("parent_executable")
        == str((_repo() / ".venv/bin/python").resolve(strict=True))
        and type(static_preflight.get("controller_supervision", {}).get("parent_pid"))
        is int
        and static_preflight.get("controller_supervision", {}).get("parent_pid") > 0
        and re.fullmatch(
            r"[0-9a-f]{64}",
            str(
                static_preflight.get("controller_supervision", {}).get(
                    "parent_command_sha256", ""
                )
            ),
        )
        is not None
        and static_preflight.get("controller_supervision", {}).get(
            "parent_command_sha256"
        )
        == expected_parent_command_sha256,
        "manifest_runtime_contract_exact": probe_manifest.get("platform_requested")
        == "rocm"
        and probe_manifest.get("allow_gpu") is True
        and _json_exact(probe_manifest.get("contract"), expected_probe_contract)
        and probe_manifest.get("graph_api_used") is False
        and probe_manifest.get("command_buffer_used") is False
        and probe_manifest.get("jax_imported") is False,
        "static_source_binding_exact": _json_exact(
            static_preflight.get("sources"), expected_probe_sources
        ),
        "static_lock_binding_exact": static_preflight.get("inherited_lock", {}).get(
            "validated"
        )
        is True
        and static_preflight.get("inherited_lock", {}).get("close_on_exec") is True
        and _exact_int(
            static_preflight.get("inherited_lock", {}).get("fd"), expected_lock_fd
        )
        and type(static_preflight.get("inherited_lock", {}).get("device")) is int
        and static_preflight.get("inherited_lock", {}).get("device")
        == expected_lock_stat.st_dev
        and type(static_preflight.get("inherited_lock", {}).get("inode")) is int
        and static_preflight.get("inherited_lock", {}).get("inode")
        == expected_lock_stat.st_ino,
        "static_environment_binding_exact": _json_exact(
            {
                key: value
                for key, value in static_preflight.get("environment", {}).items()
                if key != "XLA_FLAGS_original"
            },
            expected_probe_environment,
        )
        and static_preflight.get("environment", {}).get("XLA_FLAGS_original")
        == _COMMAND_BUFFER_FLAG,
        "static_isolated_python_exact": _json_exact(
            static_preflight.get("isolated_python", {}).get("checks"),
            {
                "isolated": True,
                "ignore_environment": True,
                "no_user_site": True,
                "no_site": True,
                "safe_path": True,
                "dont_write_bytecode": True,
            },
        )
        and static_preflight.get("isolated_python", {}).get("repo_root") == str(_repo())
        and static_preflight.get("isolated_python", {}).get("site_packages")
        == str((_repo() / ".venv/lib/python3.12/site-packages").resolve(strict=True))
        and static_preflight.get("isolated_python", {}).get("initial_stdlib_path")
        == [
            "/usr/lib/python312.zip",
            "/usr/lib/python3.12",
            "/usr/lib/python3.12/lib-dynload",
        ]
        and static_preflight.get("isolated_python", {}).get("sys_executable")
        == str(_repo() / ".venv/bin/python")
        and static_preflight.get("isolated_python", {}).get("pycache_prefix")
        == str(run_dir / "python-cache")
        and static_preflight.get("isolated_python", {}).get(
            "pycache_empty_before_bound_imports"
        )
        is True
        and static_preflight.get("isolated_python", {}).get("main_cached_artifact")
        is None,
        "static_git_stack_binding_exact": isinstance(static_preflight.get("git"), dict)
        and static_preflight.get("git", {}).get("worktree_clean") is True
        and re.fullmatch(
            r"[0-9a-f]{40}", str(static_preflight.get("git", {}).get("head", ""))
        )
        is not None
        and re.fullmatch(
            r"[0-9a-f]{40}", str(static_preflight.get("git", {}).get("tree", ""))
        )
        is not None
        and stack_is_exact(static_preflight.get("stack")),
        "hardware_preflight_exact": hardware_preflight.get("jax_imported") is False
        and isinstance(hardware, dict)
        and set(hardware)
        == {
            "amdgpu_boot_clean",
            "fatal_amdgpu_events",
            "generic_headless_preflight",
            "device",
            "kfd_owner_pids",
            "render_owner_pids",
            "controlled_render_wake_only",
            "pre_backend_hardware_limits",
            "journal_after_controlled_wake",
        }
        and hardware.get("amdgpu_boot_clean") is True
        and hardware.get("fatal_amdgpu_events") == []
        and hardware.get("kfd_owner_pids") == []
        and hardware.get("render_owner_pids") == []
        and hardware.get("controlled_render_wake_only") is True
        and _json_exact(
            hardware.get("generic_headless_preflight"),
            {
                "amd_cards": ["card1"],
                "connected_amd_connectors": [],
                "kfd_accessible": True,
                "kfd_path": "/dev/kfd",
                "kfd_unowned": True,
            },
        )
        and _json_exact(hardware.get("device"), expected_device_identity)
        and hardware_limits_are_exact(hardware.get("pre_backend_hardware_limits"))
        and _clean_journal(hardware.get("journal_after_controlled_wake")),
        "host_oracle_exact": _json_exact(
            host_oracle.get("inputs"), _EXPECTED_HOST_MANIFESTS
        )
        and _json_exact(host_oracle.get("expected"), _EXPECTED_HOST_OUTPUT)
        and host_oracle.get("verified_against_bound_hashes") is True
        and host_oracle.get("compile_signature_kind") == "ShapeDtypeStruct"
        and host_oracle.get("compile_abstract_signature_derived_from_host_metadata")
        is True
        and host_oracle.get("lowering_consumed_host_values") is False
        and host_oracle.get("runtime_comparison_evaluated") is False
        and _exact_int(host_oracle.get("oracle_attempts"), 1)
        and _exact_int(host_oracle.get("oracle_completions"), 1)
        and _exact_int(host_oracle.get("compiled_executable_invocations"), 0),
        "host_sensitivity_exact": sensitivity.get("passed") is True
        and _exact_int(sensitivity.get("comparison_count"), 35)
        and _json_exact(
            sensitivity.get("groups"),
            {
                "global": 7,
                "omitted_rows": 3,
                "omitted_columns": 17,
                "omitted_ranks": 8,
            },
        )
        and isinstance(sensitivity_comparisons, list)
        and len(sensitivity_comparisons) == 35
        and {item.get("label") for item in sensitivity_comparisons}
        == expected_sensitivity_labels
        and all(
            item.get("passed") is True
            and _finite_number(item.get("relative_l2_error"), minimum=0.0)
            and item["relative_l2_error"] > 0.03
            for item in sensitivity_comparisons
        )
        and _json_exact(
            sensitivity.get("required_minimum_relative_l2_error_exclusive"),
            0.03,
        )
        and _json_exact(
            sensitivity.get("minimum_observed_relative_l2_error"),
            min(item["relative_l2_error"] for item in sensitivity_comparisons),
        ),
        "backend_once": _exact_int(
            backend_ready.get("backend_initialization_attempts"), 1
        )
        and _exact_int(backend_ready.get("backend_initialization_completions"), 1)
        and _exact_int(backend_ready.get("compiled_executable_invocations"), 0),
        "backend_identity_and_stack_exact": backend_ready.get("jax") == "0.10.2"
        and backend_ready.get("jaxlib") == "0.10.2"
        and backend_ready.get("platform") == "gpu"
        and backend_ready.get("platform_version") == "PJRT C API\nrocm 70200"
        and _json_exact(backend_ready.get("devices"), ["rocm:0"])
        and _json_exact(backend_ready.get("module_origins"), expected_module_origins),
        "all_hardware_limit_checkpoints_exact": hardware_limits_are_exact(
            backend_ready.get("hardware_limits")
        )
        and hardware_limits_are_exact(lowered.get("hardware_limits_before_lower"))
        and hardware_limits_are_exact(compiled.get("hardware_limits_before_compile"))
        and hardware_limits_are_exact(compiled.get("hardware_limits_after_compile"))
        and hardware_limits_are_exact(dispatch_hardware),
        "journal_record_sequence_exact": journal_stages
        == [
            "after_backend_initialization",
            "after_input_device_put",
            "after_candidate_dispatch_attempt",
            "after_candidate_device_get",
        ]
        and all(_clean_journal(record.get("safety")) for record in journal_records),
        "embedded_journal_evidence_clean": _clean_journal(
            lowered.get("journal_checkpoint")
        )
        and _clean_journal(compiled.get("journal_checkpoint"))
        and _clean_journal(fresh_isa.get("journal_checkpoint"))
        and _clean_journal(dispatch_preflight.get("journal_checkpoint")),
        "lower_once_and_structural_gate": _exact_int(lowered.get("lower_attempts"), 1)
        and _exact_int(lowered.get("lower_completions"), 1)
        and _exact_int(lowered.get("compiled_executable_invocations"), 0)
        and lowered.get("stablehlo_precompile_gate", {}).get("passed") is True,
        "compile_once_unreleased": _exact_int(compiled.get("compile_attempts"), 1)
        and _exact_int(compiled.get("compile_completions"), 1)
        and _exact_int(compiled.get("compiled_executable_invocations"), 0)
        and compiled.get("executable_released") is False,
        "compiled_release_prerequisites": release_gate.get("passed") is True
        and release_gate.get("structural_gate", {}).get("passed") is True
        and release_gate.get("memory_gate", {}).get("passed") is True
        and release_gate.get("artifact_gate", {}).get("nonempty") is True
        and release_gate.get("artifact_gate", {}).get("isa_qualified") is False
        and release_gate.get("runtime_promotion") is False,
        "independent_ir_gate_passed": independent_ir.get("passed") is True,
        "compiler_artifacts_exact": independent_artifacts["file_count"] > 0
        and independent_artifacts["total_bytes"] > 0
        and _json_exact(compiled.get("artifact_inventory"), independent_artifacts),
        "ir_artifacts_match": lowered.get("stablehlo_artifact", {}).get("sha256")
        == stable_ir["sha256"]
        and compiled.get("optimized_hlo_artifact", {}).get("sha256")
        == optimized_ir["sha256"],
        "child_fresh_isa_exact": child_isa.get("passed") is True
        and set(child_isa.get("checks", {})) == expected_child_isa_checks
        and all(value is True for value in child_isa.get("checks", {}).values())
        and _exact_int(fresh_isa.get("isa_qualification_attempts"), 1)
        and _exact_int(fresh_isa.get("isa_qualification_completions"), 1)
        and _exact_int(fresh_isa.get("compiled_executable_invocations"), 0),
        "controller_fresh_isa_exact": independent_isa.get("passed") is True
        and all(independent_isa.get("checks", {}).values()),
        "child_and_controller_isa_match": child_isa_evidence.get("cache", {}).get(
            "sha256"
        )
        == independent_isa_evidence.get("cache", {}).get("sha256")
        and child_isa_evidence.get("cache", {}).get("path")
        == independent_isa_evidence.get("cache", {}).get("path")
        and child_isa_evidence.get("candidate", {}).get("sha256")
        == independent_isa_evidence.get("candidate", {}).get("sha256")
        == _EXPECTED_NESTED_ELF_SHA256
        and _exact_int(
            child_isa_evidence.get("candidate", {}).get("bytes"),
            _EXPECTED_NESTED_ELF_BYTES,
        )
        and _exact_int(
            independent_isa_evidence.get("candidate", {}).get("bytes"),
            _EXPECTED_NESTED_ELF_BYTES,
        )
        and _exact_int(child_isa_evidence.get("elf_inventory", {}).get("elf_count"), 6)
        and _exact_int(
            child_isa_evidence.get("elf_inventory", {}).get(
                "unique_exact_symbol_candidate_count"
            ),
            1,
        )
        and child_isa_evidence.get("status") == "passed_offline_isa_verification"
        and child_isa_evidence.get("offline_only") is True
        and child_isa_evidence.get("device_access_performed") is False
        and child_isa_evidence.get("jax_modules_imported_by_verifier") is False
        and child_isa_evidence.get("runtime_promotion") is False
        and _json_exact(
            child_isa_evidence.get("isa"), independent_isa_evidence.get("isa")
        )
        and _json_exact(
            child_isa_evidence.get("serialization"),
            independent_isa_evidence.get("serialization"),
        )
        and _json_exact(
            child_isa_evidence.get("toolchain"),
            independent_isa_evidence.get("toolchain"),
        ),
        "executable_release_exact": released.get("released_for")
        == "one_exact_runtime_correctness_invocation"
        and released.get("structural_gate_passed") is True
        and released.get("memory_gate_passed") is True
        and released.get("fresh_isa_gate_passed") is True
        and _exact_int(released.get("compiled_executable_invocations"), 0)
        and released.get("runtime_promotion") is False,
        "one_tuple_device_put_exact": _exact_int(
            input_put.get("tuple_device_put_attempts"), 1
        )
        and _exact_int(input_put.get("tuple_device_put_completions"), 1)
        and _exact_int(input_put.get("device_put_leaves"), 6)
        and _exact_int(input_put.get("input_readiness_invocations"), 1)
        and _exact_int(input_put.get("compiled_executable_invocations"), 0),
        "dispatch_preflight_exact": dispatch_preflight.get("xla_flags")
        == _COMMAND_BUFFER_FLAG
        and dispatch_preflight.get("required_xla_flags") == _COMMAND_BUFFER_FLAG
        and all(
            _exact_int(dispatch_preflight.get(name), 0)
            for name in (
                "compiled_executable_invocations",
                "warmup_invocations",
                "replay_invocations",
                "backward_invocations",
                "gpu_reference_invocations",
                "device_error_reduction_invocations",
                "model_invocations",
            )
        ),
        "dispatch_hardware_launch_gate_exact": hardware_limits_are_exact(
            dispatch_hardware
        ),
        "dispatch_capability_initially_unused": _json_exact(
            dispatch_preflight.get("one_shot_capability"),
            {
                "consumed": False,
                "compiled_executable_attempts": 0,
                "compiled_executable_completions": 0,
            },
        ),
        "one_shot_consumed_before_dispatch": _json_exact(
            one_shot_started,
            {
                "consumed": True,
                "compiled_executable_attempts": 1,
                "compiled_executable_completions": 0,
            },
        )
        and _exact_int(dispatch_started.get("output_readiness_invocations"), 0)
        and dispatch_started.get("wall_time_ns") == start_wall_ns
        and dispatch_started.get("monotonic_ns") == start_monotonic_ns,
        "dispatch_post_attempt_checkpoint_exact": _exact_int(
            journal_by_stage.get("after_candidate_dispatch_attempt", {}).get(
                "compiled_executable_attempts"
            ),
            1,
        )
        and _exact_int(
            journal_by_stage.get("after_candidate_dispatch_attempt", {}).get(
                "compiled_executable_completions"
            ),
            0,
        )
        and journal_by_stage.get("after_candidate_dispatch_attempt", {}).get(
            "dispatch_completed_wall_time_ns"
        )
        == completion_wall_ns
        and journal_by_stage.get("after_candidate_dispatch_attempt", {}).get(
            "dispatch_completed_monotonic_ns"
        )
        == completion_monotonic_ns,
        "dispatch_exactly_once": _json_exact(
            one_shot_dispatch,
            {
                "consumed": True,
                "compiled_executable_attempts": 1,
                "compiled_executable_completions": 1,
            },
        )
        and _exact_int(dispatch.get("output_readiness_invocations"), 1)
        and dispatch_times_valid
        and completion_wall_ns >= start_wall_ns
        and completion_monotonic_ns >= start_monotonic_ns
        and _finite_number(dispatch_seconds, minimum=0.0)
        and 0 <= dispatch_seconds < 1.0
        and abs(
            dispatch_seconds
            - (completion_monotonic_ns - start_monotonic_ns) / 1_000_000_000
        )
        <= 0.01,
        "device_get_exactly_once": _exact_int(device_get.get("device_get_attempts"), 1)
        and _exact_int(device_get.get("device_get_completions"), 1)
        and _exact_int(device_get.get("device_get_leaves"), 1)
        and _json_exact(one_shot_get, one_shot_dispatch),
        "numerical_validation_exact": numerical_record.get("host_only_comparison")
        is True
        and _exact_int(numerical_record.get("gpu_reference_invocations"), 0)
        and _exact_int(numerical_record.get("device_error_reduction_invocations"), 0)
        and numerical.get("passed") is True
        and set(numerical.get("checks", {})) == expected_numerical_checks
        and all(value is True for value in numerical.get("checks", {}).values())
        and type(numerical.get("element_count")) is int
        and numerical.get("element_count") == 51
        and type(numerical.get("finite_actual_count")) is int
        and numerical.get("finite_actual_count") == 51
        and type(numerical.get("finite_expected_count")) is int
        and numerical.get("finite_expected_count") == 51
        and _finite_number(numerical.get("relative_l2_error"), minimum=0.0)
        and 0 <= numerical.get("relative_l2_error") < 0.01
        and _finite_number(numerical.get("cosine_similarity"))
        and numerical.get("cosine_similarity") >= 0.9999
        and _finite_number(numerical.get("maximum_absolute_error"), minimum=0.0)
        and 0 <= numerical.get("maximum_absolute_error") <= 0.25
        and _finite_number(numerical.get("dispatch_seconds"), minimum=0.0)
        and 0 <= numerical.get("dispatch_seconds") < 1.0
        and _json_exact(numerical.get("expected"), _EXPECTED_HOST_OUTPUT)
        and _json_exact(numerical.get("actual", {}).get("shape"), [3, 17])
        and numerical.get("actual", {}).get("dtype") == "bfloat16"
        and _exact_int(numerical.get("actual", {}).get("nbytes"), 102)
        and re.fullmatch(
            r"[0-9a-f]{64}", str(numerical.get("actual", {}).get("sha256", ""))
        )
        is not None
        and isinstance(numerical.get("bitwise_equal_diagnostic"), bool)
        and _json_exact(numerical.get("limits"), expected_numerical_limits),
        "numerical_hash_diagnostic_consistent": numerical.get(
            "bitwise_equal_diagnostic"
        )
        is (
            numerical.get("actual", {}).get("sha256")
            == numerical.get("expected", {}).get("sha256")
        )
        and numerical.get("dispatch_seconds") == dispatch_seconds
        and (
            numerical.get("bitwise_equal_diagnostic") is not True
            or (
                _json_exact(numerical.get("relative_l2_error"), 0.0)
                and _json_exact(numerical.get("cosine_similarity"), 1.0)
                and _json_exact(numerical.get("maximum_absolute_error"), 0.0)
            )
        ),
        "terminal_runtime_correctness_only": completed.get("status")
        == "passed_exact_m3_k64_n17_w8a8_lora_forward_runtime_correctness_only"
        and completed.get("runtime_promotion") is False
        and completed.get("isa_qualified") is True
        and _exact_int(completed.get("compiled_executable_invocations"), 1)
        and _exact_int(completed.get("compiled_executable_completions"), 1)
        and all(
            _exact_int(completed.get(name), 0)
            for name in (
                "warmup_invocations",
                "replay_invocations",
                "backward_invocations",
                "gpu_reference_invocations",
                "device_error_reduction_invocations",
                "model_invocations",
            )
        )
        and _json_exact(one_shot_complete, one_shot_dispatch)
        and _json_exact(completed.get("source_postflight"), expected_probe_sources)
        and _json_exact(
            completed.get("source_postflight"), static_preflight.get("sources")
        )
        and _json_exact(completed.get("git_postflight"), static_preflight.get("git"))
        and stack_is_exact(completed.get("stack_postflight"))
        and _json_exact(
            completed.get("stack_postflight"), static_preflight.get("stack")
        )
        and _clean_journal(completed.get("journal_postflight")),
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(
            "runtime profile/probe evidence failed: " + ", ".join(failed)
        )
    return {
        "passed": True,
        "checks": checks,
        "telemetry_sha256": _file_sha256(run_dir / "telemetry.jsonl"),
        "summary_sha256": _file_sha256(run_dir / "telemetry.jsonl.summary.json"),
        "probe_sha256": _file_sha256(run_dir / "probe.jsonl"),
        "stablehlo": stable_ir,
        "optimized_hlo": optimized_ir,
        "controller_independent_raw_ir_gate": independent_ir,
        "controller_independent_compiler_artifacts": independent_artifacts,
        "controller_independent_fresh_isa": independent_isa,
        "dispatch_telemetry_bracket": telemetry_bracket,
        "sample_count": telemetry["sample_count"],
        "measured_sample_count": telemetry["measured_count"],
        "maximum_measured_sample_gap_seconds": maximum_measured_gap,
    }


def _audit_profile_outputs(
    run_dir: Path,
    *,
    phase: str,
    profile_returncode: int,
    wait_audit: dict[str, Any],
    expected_lock_fd: int,
    expected_scope_unit: str | None = None,
    expected_device_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if phase == "execute":
        if expected_scope_unit is None or expected_device_identity is None:
            raise RuntimeError(
                "runtime audit requires the exact controller scope and device identity"
            )
        return _audit_runtime_profile_outputs(
            run_dir,
            profile_returncode=profile_returncode,
            wait_audit=wait_audit,
            expected_lock_fd=expected_lock_fd,
            expected_scope_unit=expected_scope_unit,
            expected_device_identity=expected_device_identity,
        )
    if phase != "compile":
        raise RuntimeError(f"unsupported W8 qualification phase: {phase}")
    telemetry = _summarize_private_telemetry(run_dir / "telemetry.jsonl", 64 * 1024**2)
    summary = _private_json_object(run_dir / "telemetry.jsonl.summary.json", 1024**2)
    probe = _private_json_lines(run_dir / "probe.jsonl", 8 * 1024**2)
    manifest = telemetry["manifest"]
    maximum_measured_gap = telemetry["maximum_measured_gap_seconds"]
    limits = manifest.get("safety_limits")
    gpu = manifest.get("gpu")
    expected_profile_command = _profile_command(
        phase="compile",
        run_dir=run_dir,
        card="card1",
        lock_fd=expected_lock_fd,
    )
    expected_child_command = expected_profile_command[
        expected_profile_command.index("--") + 1 :
    ]
    probe_types = [record.get("record_type") for record in probe]
    expected_probe_types = [
        "manifest",
        "static_preflight",
        "hardware_preflight",
        "host_oracle",
        "backend_ready",
        "journal_checkpoint",
        "lowered",
        "compiled",
        "completed",
    ]
    probe_by_type = {record.get("record_type"): record for record in probe}
    stable_text, stable_ir = _private_text_artifact(
        run_dir / "w8a8-forward.stablehlo.mlir", 8 * 1024**2
    )
    optimized_text, optimized_ir = _private_text_artifact(
        run_dir / "w8a8-forward.optimized.hlo", 32 * 1024**2
    )
    independent_ir = _independent_ir_gate(stable_text, optimized_text)
    independent_compiler_artifacts = _independent_compiler_artifact_inventory(
        run_dir / "compiler-artifacts"
    )
    compiled_record = probe_by_type.get("compiled", {})
    lowered_record = probe_by_type.get("lowered", {})
    static_preflight_record = probe_by_type.get("static_preflight", {})
    release_gate = compiled_record.get("release_gate", {})
    host_oracle = probe_by_type.get("host_oracle", {})
    checks = {
        "wait_supervisor_passed": wait_audit.get("passed") is True,
        "profile_returncode_zero": _exact_int(profile_returncode, 0),
        "telemetry_manifest_exact": manifest.get("record_type") == "manifest",
        "telemetry_protocol_exact": manifest.get("interval_seconds")
        == _PROFILE_INTERVAL_SECONDS
        and manifest.get("baseline_seconds") == _PROFILE_BASELINE_SECONDS
        and manifest.get("duration_seconds") is None
        and manifest.get("timeout_seconds") == _PROFILE_TIMEOUT_SECONDS
        and manifest.get("sensor_grace_seconds") == _PROFILE_SENSOR_GRACE_SECONDS
        and manifest.get("terminate_included_on_safety") is False
        and manifest.get("command_recorded") is True
        and _exact_int(manifest.get("passed_file_descriptor_count"), 1),
        "telemetry_profiler_source_exact": manifest.get("runtime", {}).get(
            "script_sha256"
        )
        == _EXPECTED_SOURCE_SHA256["profiler"],
        "telemetry_wrapped_child_command_exact": manifest.get("command")
        == expected_child_command,
        "gpu_card_exact": isinstance(gpu, dict) and gpu.get("card") == "card1",
        "gpu_device_exact": isinstance(gpu, dict)
        and str(gpu.get("device_id", "")).lower() == _EXPECTED_DEVICE_ID,
        "limits_exact": _json_exact(
            limits,
            {
                "max_junction_temp_c": _MAX_JUNCTION_TEMP_C,
                "max_gpu_power_watts": _MAX_GPU_POWER_WATTS,
                "max_vram_bytes": _MAX_VRAM_GIB * 1024**3,
                "min_host_available_bytes": 0.0,
                "max_swap_bytes": _MAX_SWAP_GIB * 1024**3,
            },
        ),
        "summary_completed": summary.get("record_type") == "summary"
        and summary.get("status") == "completed",
        "summary_returncode_zero": _exact_int(summary.get("returncode"), 0),
        "summary_no_signal": summary.get("received_signal") is None,
        "summary_kernel_log_available": summary.get("kernel_log_available") is True,
        "summary_no_driver_errors": "kernel_driver_errors" not in summary,
        "summary_no_safety_violation": "safety_violation" not in summary,
        "summary_no_error_fields": all(
            name not in summary
            for name in (
                "error_type",
                "error",
                "terminated_explicit_pids",
                "surviving_explicit_pids",
            )
        ),
        "telemetry_record_sequence_exact": telemetry["preflight_count"] == 1
        and telemetry["sample_count"]
        == telemetry["baseline_count"]
        + telemetry["preflight_count"]
        + telemetry["measured_count"],
        "summary_sample_counts_match_raw": _exact_int(
            summary.get("samples"), telemetry["sample_count"]
        )
        and _exact_int(summary.get("baseline_samples"), telemetry["baseline_count"])
        and _exact_int(summary.get("measured_samples"), telemetry["measured_count"]),
        "summary_measured_maxima_match_raw": _json_exact(
            summary.get("metrics", {}).get("gpu_power_watts", {}).get("measured_max"),
            telemetry["measured_maximum"]["power"],
        )
        and _json_exact(
            summary.get("metrics", {})
            .get("gpu_junction_temp_c", {})
            .get("measured_max"),
            telemetry["measured_maximum"]["temperature"],
        )
        and _json_exact(
            summary.get("metrics", {}).get("vram_used_bytes", {}).get("measured_max"),
            telemetry["measured_maximum"]["vram"],
        )
        and _json_exact(
            summary.get("metrics", {})
            .get("host_swap_used_bytes", {})
            .get("measured_max"),
            telemetry["measured_maximum"]["host_swap"],
        ),
        "at_least_two_measured_samples": telemetry["measured_count"] >= 2,
        "at_least_one_baseline_sample": telemetry["baseline_count"] >= 1,
        "exactly_one_preflight_sample": telemetry["preflight_count"] == 1,
        "measured_timestamps_monotonic": telemetry["timestamps_monotonic"] is True,
        "maximum_measured_gap_at_most_two_seconds": maximum_measured_gap <= 2.0,
        "measured_mandatory_fields_valid": telemetry["measured_mandatory_fields_valid"]
        is True,
        "post_grace_gpu_sensors_present": telemetry["post_grace_gpu_sensors_present"]
        is True,
        "measured_power_present": telemetry["present"]["power"],
        "measured_temperature_present": telemetry["present"]["temperature"],
        "measured_vram_present": telemetry["present"]["vram"],
        "measured_host_memory_present": telemetry["present"]["host_memory"],
        "measured_host_swap_present": telemetry["present"]["host_swap"],
        "measured_power_within_limit": telemetry["within"]["power"],
        "measured_temperature_within_limit": telemetry["within"]["temperature"],
        "measured_vram_within_limit": telemetry["within"]["vram"],
        "measured_host_memory_within_limit": telemetry["within"]["host_memory"],
        "measured_host_swap_within_limit": telemetry["within"]["host_swap"],
        "probe_terminal_compile_diagnostic": probe[-1].get("record_type") == "completed"
        and probe[-1].get("status") == "passed_compile_diagnostic_unpromoted"
        and _exact_int(probe[-1].get("compiled_executable_invocations"), 0),
        "probe_never_invoked": all(
            _exact_int(record.get("compiled_executable_invocations", 0), 0)
            for record in probe
        ),
        "probe_no_error_record": all(
            record.get("record_type") != "error" for record in probe
        ),
        "probe_record_sequence_exact": probe_types == expected_probe_types,
        "probe_controller_supervision_exact": static_preflight_record.get(
            "controller_supervision", {}
        ).get("validated")
        is True
        and re.fullmatch(
            r"skyrl-w8a8-compile-[0-9]+-[0-9a-f]+\.scope",
            str(
                static_preflight_record.get("controller_supervision", {}).get(
                    "scope", ""
                )
            ),
        )
        is not None,
        "host_oracle_exact_bound_manifests": _json_exact(
            host_oracle.get("inputs"), _EXPECTED_HOST_MANIFESTS
        )
        and _json_exact(host_oracle.get("expected"), _EXPECTED_HOST_OUTPUT)
        and host_oracle.get("verified_against_bound_hashes") is True
        and host_oracle.get("compile_signature_kind") == "ShapeDtypeStruct"
        and host_oracle.get("compile_abstract_signature_derived_from_host_metadata")
        is True
        and host_oracle.get("lowering_consumed_host_values") is False
        and host_oracle.get("runtime_comparison_evaluated") is False
        and _exact_int(host_oracle.get("compiled_executable_invocations"), 0),
        "probe_manifest_compile_only": _exact_int(
            probe[0]
            .get("contract", {})
            .get("dispatch_plan", {})
            .get("compiled_executable_invocations"),
            0,
        )
        and probe[0].get("contract", {}).get("execute_rung_enabled") is False,
        "stablehlo_precompile_gate_passed": lowered_record.get(
            "stablehlo_precompile_gate", {}
        ).get("passed")
        is True,
        "controller_independent_raw_ir_gate_passed": independent_ir["passed"] is True,
        "compiled_gate_passed_unpromoted": release_gate.get("passed") is True
        and release_gate.get("structural_gate", {}).get("passed") is True
        and release_gate.get("memory_gate", {}).get("passed") is True
        and release_gate.get("artifact_gate", {}).get("nonempty") is True
        and release_gate.get("runtime_promotion") is False
        and release_gate.get("artifact_gate", {}).get("isa_qualified") is False,
        "compiler_artifact_inventory_nonempty_and_exact": independent_compiler_artifacts[
            "file_count"
        ]
        > 0
        and independent_compiler_artifacts["total_bytes"] > 0
        and _json_exact(
            compiled_record.get("artifact_inventory"),
            independent_compiler_artifacts,
        ),
        "stablehlo_artifact_matches_record": lowered_record.get(
            "stablehlo_artifact", {}
        ).get("sha256")
        == stable_ir["sha256"],
        "optimized_hlo_artifact_matches_record": compiled_record.get(
            "optimized_hlo_artifact", {}
        ).get("sha256")
        == optimized_ir["sha256"],
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError("profile/probe evidence failed: " + ", ".join(failed))
    return {
        "passed": True,
        "checks": checks,
        "telemetry_sha256": _file_sha256(run_dir / "telemetry.jsonl"),
        "summary_sha256": _file_sha256(run_dir / "telemetry.jsonl.summary.json"),
        "probe_sha256": _file_sha256(run_dir / "probe.jsonl"),
        "stablehlo": stable_ir,
        "optimized_hlo": optimized_ir,
        "controller_independent_raw_ir_gate": independent_ir,
        "controller_independent_compiler_artifacts": independent_compiler_artifacts,
        "sample_count": telemetry["sample_count"],
        "measured_sample_count": telemetry["measured_count"],
        "maximum_measured_sample_gap_seconds": maximum_measured_gap,
    }


def _run_gate(args: argparse.Namespace, output: TextIO, run_dir_fd: int) -> int:
    os.environ["PATH"] = "/opt/rocm/bin:/usr/bin:/bin"

    sources: dict[str, str] | None = None
    systemd_runtime: dict[str, Any] | None = None
    profile_runtime: dict[str, Any] | None = None
    environment: dict[str, str] | None = None
    lock_fd: int | None = None
    profile_returncode: int | None = None
    wait_audit: dict[str, Any] | None = None
    evidence_audit: dict[str, Any] | None = None
    settle_result: dict[str, Any] | None = None
    final_journal: dict[str, Any] | None = None
    final_source_manifest: dict[str, str] | None = None
    run_inventory: dict[str, Any] | None = None
    operation_error: BaseException | None = None
    cleanup_errors: list[dict[str, str]] = []
    profile_stdout = profile_stderr = None
    process: subprocess.Popen[Any] | None = None
    scope_unit: str | None = None
    pre_reap_scope_cleanup: dict[str, Any] | None = None
    final_scope_cleanup: dict[str, Any] | None = None
    direct_cgroup_cleanup: dict[str, Any] | None = None
    handoff_required = False
    acquire_qwen35_rocm_launch_lock: Any | None = None
    require_clean_amdgpu_boot: Any | None = None
    _discover_device: Any | None = None
    capture_baseline: Any | None = None
    settle_handoff: Any | None = None
    previous_pycache_prefix = sys.pycache_prefix
    previous_dont_write_bytecode = sys.dont_write_bytecode
    controller_cache_bound = False
    deferred_signal: int | None = None
    handled_signals = tuple(
        value
        for value in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None))
        if value is not None
    )
    previous_signal_handlers = {
        value: signal.getsignal(value) for value in handled_signals
    }

    def defer_signal(signum: int, _frame: Any) -> None:
        nonlocal deferred_signal
        deferred_signal = signum

    for value in handled_signals:
        signal.signal(value, defer_signal)

    def remember_cleanup_error(stage: str, error: BaseException) -> None:
        cleanup_errors.append(
            {
                "stage": stage,
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )

    try:
        sources = _source_manifest()
        _create_private_subdirectory(run_dir_fd, "python-cache")
        sys.pycache_prefix = str(args.run_dir / "python-cache")
        sys.dont_write_bytecode = True
        controller_cache_bound = True
        repo_path = str(_repo())
        if repo_path not in sys.path:
            sys.path.append(repo_path)
        from rocm.amdgpu_safety import (
            acquire_qwen35_rocm_launch_lock,
            require_clean_amdgpu_boot,
        )
        from rocm.qwen35_prewarm_handoff import (
            _discover_device,
            capture_baseline,
            settle_handoff,
        )

        systemd_runtime = _systemd_runtime_manifest()
        profile_runtime = _profile_runtime_manifest()
        environment = _child_environment(args.run_dir)
        lock_fd = acquire_qwen35_rocm_launch_lock()
        if deferred_signal is not None:
            raise RuntimeError(f"deferred operational signal {deferred_signal}")
        device = _discover_device()
        if device.device_id != _EXPECTED_DEVICE_ID:
            raise RuntimeError("exact RX 7900 XTX device identity was not found")
        vram_total = _read_vram_total(device.device_root)
        baseline_path = args.run_dir / "handoff.jsonl"
        handoff_required = True
        baseline = capture_baseline(baseline_path)
        if deferred_signal is not None:
            raise RuntimeError(f"deferred operational signal {deferred_signal}")
        command = _profile_command(
            phase=args.phase,
            run_dir=args.run_dir,
            card=device.drm_card,
            lock_fd=lock_fd,
        )
        scope_unit = (
            _scope_name() if args.phase == "compile" else _scope_name("execute")
        )
        contained_command = _scope_command(scope_unit, command)
        _emit(
            {
                "record_type": "operational_manifest",
                "timestamp": _utc_now(),
                "sources": sources,
                "device": device.identity(),
                "vram_total_bytes": vram_total,
                "baseline": baseline,
                "profile_command_sha256": hashlib.sha256(
                    "\0".join(command).encode()
                ).hexdigest(),
                "contained_command_sha256": hashlib.sha256(
                    "\0".join(contained_command).encode()
                ).hexdigest(),
                "systemd_containment": systemd_runtime,
                "isolated_profiler_runtime": profile_runtime,
                "controller_pycache_prefix": sys.pycache_prefix,
                "systemd_scope_unit": f"{scope_unit}.scope",
                "environment": {
                    name: environment[name]
                    for name in (
                        "JAX_PLATFORMS",
                        "ROCR_VISIBLE_DEVICES",
                        "HIP_VISIBLE_DEVICES",
                        "GPU_DEVICE_ORDINAL",
                        "JAX_ROCM_VISIBLE_DEVICES",
                        "XLA_PYTHON_CLIENT_ALLOCATOR",
                        "XLA_PYTHON_CLIENT_PREALLOCATE",
                        "XLA_CLIENT_MEM_FRACTION",
                        "XLA_FLAGS",
                    )
                },
                "lock_fd": lock_fd,
            },
            output,
        )
        profile_stdout = _open_private_file(run_dir_fd, "profile.stdout")
        profile_stderr = _open_private_file(run_dir_fd, "profile.stderr")
        previous_umask = os.umask(0o077)
        try:
            process = subprocess.Popen(
                contained_command,
                cwd=args.run_dir,
                env=environment,
                stdout=profile_stdout,
                stderr=profile_stderr,
                pass_fds=(lock_fd,),
                start_new_session=True,
            )
        finally:
            os.umask(previous_umask)
        if deferred_signal is not None and args.phase == "compile":
            raise RuntimeError(f"deferred operational signal {deferred_signal}")
        if args.phase == "compile":
            profile_returncode, wait_audit = _wait_profile(
                process, scope_unit, lambda: deferred_signal
            )
        else:
            profile_returncode, wait_audit = _wait_profile(
                process,
                scope_unit,
                lambda: deferred_signal,
                phase="execute",
                probe_path=args.run_dir / "probe.jsonl",
            )
        if wait_audit.get("received_signal") is not None:
            deferred_signal = int(wait_audit["received_signal"])
        if wait_audit.get("passed") is not True:
            raise RuntimeError("profile supervisor watchdog or descendant audit failed")
        evidence_audit = _audit_profile_outputs(
            args.run_dir,
            phase=args.phase,
            profile_returncode=profile_returncode,
            wait_audit=wait_audit,
            expected_lock_fd=lock_fd,
            expected_scope_unit=scope_unit,
            expected_device_identity=device.identity(),
        )
    except BaseException as error:
        operation_error = error
    finally:
        immediate_scope_cleanup = bool(
            args.phase == "execute"
            and isinstance(wait_audit, dict)
            and isinstance(wait_audit.get("cleanup"), dict)
            and wait_audit.get("cleanup", {}).get("method")
            == "immediate_cgroup_wide_sigkill_and_bounded_reap"
        )
        if scope_unit is not None:
            try:
                if immediate_scope_cleanup:
                    pre_reap_scope_cleanup = wait_audit.get("cleanup")
                    if not isinstance(pre_reap_scope_cleanup, dict):
                        raise RuntimeError(
                            "dispatch watchdog omitted immediate cleanup evidence"
                        )
                else:
                    pre_reap_scope_cleanup = _terminate_scope(scope_unit)
                if pre_reap_scope_cleanup.get("passed") is not True:
                    raise RuntimeError(
                        "immediate or ordinary scope cleanup left surviving processes"
                    )
                if wait_audit is None:
                    wait_audit = {
                        "passed": False,
                        "termination_reason": "controller_finally_cleanup",
                        "cleanup": pre_reap_scope_cleanup,
                    }
            except BaseException as error:
                remember_cleanup_error("pre_reap_systemd_scope", error)
        if process is not None:
            try:
                alive = process.poll() is None
            except BaseException as error:
                alive = True
                remember_cleanup_error("poll_systemd_run", error)
            if alive:
                try:
                    process.kill()
                except BaseException as error:
                    remember_cleanup_error("kill_systemd_run", error)
            try:
                process.wait(timeout=1.0)
            except BaseException as error:
                remember_cleanup_error("reap_systemd_run", error)
        if scope_unit is not None:
            try:
                if immediate_scope_cleanup:
                    state = _scope_state(scope_unit)
                    final_scope_cleanup = {
                        "method": "post_reap_scope_empty_reproof",
                        "unit": f"{scope_unit}.scope",
                        "state": state,
                        "passed": not state.get("pids")
                        and state.get("ActiveState") in {"inactive", "failed"},
                    }
                else:
                    final_scope_cleanup = _terminate_scope(scope_unit)
                if final_scope_cleanup.get("passed") is not True:
                    raise RuntimeError(
                        "fresh post-reap systemd scope proof found surviving processes"
                    )
            except BaseException as error:
                remember_cleanup_error("post_reap_systemd_scope", error)
            try:
                direct_cgroup_cleanup = _direct_cgroup_cleanup(scope_unit)
                if direct_cgroup_cleanup.get("passed") is not True:
                    raise RuntimeError("direct cgroup cleanup left surviving processes")
            except BaseException as error:
                remember_cleanup_error("direct_cgroup_cleanup", error)
        for stream in (profile_stdout, profile_stderr):
            if stream is not None:
                try:
                    stream.flush()
                    os.fsync(stream.fileno())
                except BaseException as error:
                    remember_cleanup_error("flush_profile_stream", error)
                try:
                    stream.close()
                except BaseException as error:
                    remember_cleanup_error("close_profile_stream", error)
        if handoff_required and settle_handoff is not None:
            try:
                settle_result = settle_handoff(
                    args.run_dir / "handoff.jsonl",
                    timeout_seconds=_HANDOFF_TIMEOUT_SECONDS,
                    poll_interval_seconds=1.0,
                )
            except BaseException as error:
                remember_cleanup_error("settle_idle_handoff", error)
        if require_clean_amdgpu_boot is not None:
            try:
                final_journal = require_clean_amdgpu_boot()
            except BaseException as error:
                remember_cleanup_error("final_amdgpu_journal", error)
        if sources is not None:
            try:
                final_source_manifest = _source_manifest()
                if final_source_manifest != sources:
                    raise RuntimeError(
                        "controller dependency sources changed during the run"
                    )
            except BaseException as error:
                remember_cleanup_error("final_source_manifest", error)
        try:
            run_inventory = _final_run_inventory(args.run_dir)
        except BaseException as error:
            remember_cleanup_error("final_run_inventory", error)
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except BaseException as error:
                remember_cleanup_error("close_global_lock", error)

    if deferred_signal is not None and operation_error is None:
        operation_error = RuntimeError(f"deferred operational signal {deferred_signal}")

    signal_before_postflight_emit = deferred_signal
    try:
        _emit(
            {
                "record_type": "controller_postflight",
                "timestamp": _utc_now(),
                "profile_returncode": profile_returncode,
                "profile_wait_audit": wait_audit,
                "pre_reap_scope_cleanup": pre_reap_scope_cleanup,
                "final_scope_cleanup": final_scope_cleanup,
                "direct_cgroup_cleanup": direct_cgroup_cleanup,
                "profile_evidence_audit": evidence_audit,
                "idle_handoff": settle_result,
                "final_journal": final_journal,
                "final_source_manifest": final_source_manifest,
                "final_run_inventory": run_inventory,
                "operation_error_type": (
                    None if operation_error is None else type(operation_error).__name__
                ),
                "operation_error": (
                    None if operation_error is None else str(operation_error)
                ),
                "cleanup_errors": cleanup_errors,
                "deferred_signal": deferred_signal,
            },
            output,
        )
        if deferred_signal != signal_before_postflight_emit:
            _emit(
                {
                    "record_type": "controller_late_signal",
                    "timestamp": _utc_now(),
                    "deferred_signal": deferred_signal,
                },
                output,
            )
    finally:
        for value, handler in previous_signal_handlers.items():
            signal.signal(value, handler)
        if controller_cache_bound:
            sys.pycache_prefix = previous_pycache_prefix
            sys.dont_write_bytecode = previous_dont_write_bytecode
    if deferred_signal is not None and operation_error is None:
        operation_error = RuntimeError(f"deferred operational signal {deferred_signal}")
    if cleanup_errors:
        return 2
    if operation_error is not None:
        if deferred_signal is not None:
            return 128 + deferred_signal
        return 1
    if profile_returncode is None:
        return 1
    return profile_returncode


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = {
        "record_type": "controller_manifest",
        "timestamp": _utc_now(),
        "platform_requested": args.platform,
        "allow_gpu": args.allow_gpu,
        "contract": _contract(args.phase),
        "controller_isolation": (
            _require_isolated_controller(args.run_dir)
            if args.platform == "rocm"
            else None
        ),
        "jax_imported": False,
        "graph_api_used": False,
        "command_buffer_used": False,
    }
    if args.platform == "abstract":
        print(
            json.dumps(manifest, allow_nan=False, separators=(",", ":"), sort_keys=True)
        )
        print(
            json.dumps(
                {
                    "record_type": "refused",
                    "timestamp": _utc_now(),
                    "status": "no_gpu_abstract_manifest_only",
                    "jax_imported": False,
                },
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0

    run_dir_fd = _create_run_directory(args.run_dir)
    try:
        with _open_private_file(run_dir_fd, "controller.jsonl") as output:
            _emit(manifest, output)
            try:
                result = _run_gate(args, output, run_dir_fd)
            except BaseException as error:
                _emit(
                    {
                        "record_type": "controller_error",
                        "timestamp": _utc_now(),
                        "status": "failed_closed",
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                    output,
                )
                return 1
            _emit(
                {
                    "record_type": "controller_complete",
                    "timestamp": _utc_now(),
                    "status": (
                        (
                            "passed_compile_diagnostic_unpromoted"
                            if args.phase == "compile"
                            else "passed_exact_m3_k64_n17_w8a8_lora_forward_runtime_correctness_only"
                        )
                        if result == 0
                        else "failed_closed"
                    ),
                    "returncode": result,
                },
                output,
            )
            return result
    finally:
        os.close(run_dir_fd)


if __name__ == "__main__":
    raise SystemExit(main())
