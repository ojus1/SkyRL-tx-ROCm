#!/usr/bin/env python3
"""Fail-closed gfx1100 compile diagnostic for one bounded W8A8+LoRA tile.

The default abstract mode emits a refusal without importing JAX.  The sole
ROCm contract is one ``M=3,K=64,N=17`` BF16/W8-group64/rank-8 forward.  A
controller must hold and pass SkyRL's global ROCm lock through
``profile_rocm.py``.  The child validates that inherited descriptor, the exact
RX 7900 XTX and render node, a clean boot, a headless AMD card, the exact sole
command-buffer-disable flag, and a hardware power cap no greater than 315 W.

The only enabled phase lowers and compiles but never invokes the executable.
The execute phase is rejected until a later source revision qualifies retained
gfx1100 ISA.  There is no runtime dispatch, backward, warmup, replay, benchmark,
model dispatch, graph API, or command-buffer API in this probe. Compilation may
still run bounded compiler profiling work, so it requires the outer telemetry
and idle-handoff controller.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime
from functools import partial
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, TextIO

_ROWS = 3
_IN_FEATURES = 64
_OUT_FEATURES = 17
_RANK = 8
_GROUP_SIZE = 64
_BLOCK_M = 16
_BLOCK_N = 16
_ROW_SUPERBLOCK = 16
_SCALING = 0.75
_COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="
_EXPECTED_DEVICE_ID = "0x744c"
_MAX_HARDWARE_POWER_CAP_UW = 315_000_000
_MAX_TEMP_BYTES = 256 * 1024**2
_MAX_TOTAL_BYTES = 512 * 1024**2
_MAX_ARTIFACT_FILES = 256
_MAX_ARTIFACT_FILE_BYTES = 128 * 1024**2
_MAX_ARTIFACT_TOTAL_BYTES = 512 * 1024**2
_PALLAS_TARGETS = frozenset(
    {
        "pallas",
        "pallas_call",
        "triton",
        "triton_kernel_call",
        "xla.gpu.triton",
        "__gpu$xla.gpu.triton",
    }
)
_EXPECTED_KERNEL_NAME = "skyrl_qwen35_w8a8_lora_forward"
_EXPECTED_PROFILE_BOOTSTRAP_SHA256 = (
    "d8d3589a8c160853f87960aa1e3a1571e0646428710e570579455339e8a65c28"
)
_EXPECTED_PROFILE_RUNTIME_JSON_SHA256 = (
    "17bda5dd82d71c612ab5f7e2ca3cfe1ffb2505bc5a3b408311d1b1054d46f04d"
)
_EXPECTED_STACK = {
    "jax": "0.10.2",
    "jaxlib": "0.10.2",
    "jax-rocm7-plugin": "0.10.2",
    "jax-rocm7-pjrt": "0.10.2",
    "numpy": "2.3.5",
    "ml-dtypes": "0.5.4",
}
_EXPECTED_RUNTIME_BINARY_SHA256 = {
    "jax_plugins/xla_rocm7/xla_rocm_plugin.so": "8e8319b48fcc771b5221771dc3da0d24bc6950c872ee07ac6f5200743b3b4fa7",
    "jax_rocm7_plugin/_triton.so": "168b21fa4d9bfd27c93c61a25ad0d9b4f2bbc718348c3c33251e3df0bc1ac6c8",
    "jax_rocm7_plugin/rocm_plugin_extension.so": "555f2d4915ae7e19277a9435d3f6fd55cb9137e981fa5431648cbe8ca8ff3f1c",
    "jaxlib/_jax.so": "1b8ba599253bddc480606ab30d6a47731ac9fe3be4e6082dec7a85e82778857f",
}
_EXPECTED_HOST_SHA256 = {
    "x": "97f28a640f7747f5a18725c2dffc67baff13068673b2edf0d600671d6132c765",
    "weight_codes": "7b2d71ade420170140481aea83796639238df7cb0cd4c85ecc12f197654dd5be",
    "weight_scales": "5aa21b6c4171062f63d5cae19f23d9d2874e3a173ce0238454e156cca6f843fb",
    "lora_a": "d293daaf4d695061960a610bc0c03cebcb45907094cfbfecb4cef28a1dccf857",
    "lora_b": "1e9f8813f39862777be04e019879000d772b7c8ef2e87768deef64cc2134200d",
    "lora_scaling": "9a8208635e00348ab64aac2b759e76391fd47089e9a749bbcec770d9eb5c6421",
    "expected": "964b0c1fe5f5c4cdbd60658717fee50a835006adf03011717b4914061ab4f88f",
}
_EXPECTED_SOURCE_SHA256 = {
    "kernel": "1dec42508856d5e38a656bc4292dcb7033d4437bf8458f7b88030be4b4b4490a",
    "quantized_reference": "91a89055ea18b16d64bd32c2eac32a2361e52b4a56b23721b41ffeb413ccc0de",
    "safety": "7ad79b9b9b54089add72dff65ea18505a794c51f0c4bafe231fbd3b745f23ba6",
    "handoff": "4d6c7e665219ce125d840e68b0e2cb7e8b1b5f98552ff65a2d07a153b3cd1392",
    "profiler": "ed230758101a2a540b3a09e7f84ac92256d2bb41c70dbc399b9466fe0b979684",
    "package_skyrl": "667d4a15b970b851e20d17510224670c14646cfb6d5a1e388ca6b9cc6da8bf41",
    "package_tx": "a7abb3e76d66df1f4472bb7a02b032ef31b959ca937fd351637b4e9b4a8fa95a",
    "package_kernels": "40abe638c7726fe5680b7c88321042016a0f695d86acfbef52337421e7257c1a",
    "package_rocm": "6d12a789cf1108538a04fbacd0b38a15dbcb8255cd0ca0fadf5a76c4191a4cfd",
    "sealed_loader": "66b868b7909a2279d5ddca0e1582f8563e8097723e970d09fce733aef2ba425a",
    "package_top_rocm": "4db843597a8ef2b43b84c1e1d5b9f4e5dcd39d2fd8eead3dde07ac02d77d4ad9",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _emit(record: dict[str, Any], output: TextIO) -> None:
    output.write(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    output.flush()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"bound source is not a regular file: {path}")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _source_files() -> dict[str, Path]:
    repo = Path(__file__).resolve().parent.parent
    return {
        "probe": Path(__file__).resolve(),
        "kernel": repo / "skyrl" / "tx" / "kernels" / "rocm" / "w8a8_lora.py",
        "quantized_reference": repo / "skyrl" / "tx" / "kernels" / "quantized_lora.py",
        "safety": repo / "rocm" / "amdgpu_safety.py",
        "handoff": repo / "rocm" / "qwen35_prewarm_handoff.py",
        "profiler": repo / "rocm" / "profile_rocm.py",
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


def _assert_bound_sources() -> dict[str, str]:
    observed = {name: _file_sha256(path) for name, path in _source_files().items()}
    for name, expected in _EXPECTED_SOURCE_SHA256.items():
        if observed.get(name) != expected:
            raise RuntimeError(f"refusing changed {name} source")
    return observed


def _venv_root() -> Path:
    repo = Path(__file__).resolve().parent.parent
    expected = repo / ".venv"
    executable = Path(sys.executable)
    if executable.parent != expected / "bin" or executable.name not in {
        "python",
        "python3",
        "python3.12",
    }:
        raise RuntimeError("probe is not running from the exact project venv")
    return expected


def _establish_isolated_import_path(run_dir: Path | None = None) -> dict[str, Any]:
    flags = sys.flags
    checks = {
        "isolated": flags.isolated == 1,
        "ignore_environment": flags.ignore_environment == 1,
        "no_user_site": flags.no_user_site == 1,
        "no_site": flags.no_site == 1,
        "safe_path": flags.safe_path is True,
        "dont_write_bytecode": flags.dont_write_bytecode == 1,
    }
    if not all(checks.values()):
        raise RuntimeError("probe must run under the exact isolated Python contract")
    expected_stdlib = [
        "/usr/lib/python312.zip",
        "/usr/lib/python3.12",
        "/usr/lib/python3.12/lib-dynload",
    ]
    if sys.path != expected_stdlib:
        raise RuntimeError("probe initial import path is not exact isolated stdlib")
    if run_dir is None:
        raise RuntimeError("probe requires the exact private run directory")
    expected_cache = run_dir / "python-cache"
    try:
        cache_info = expected_cache.lstat()
        resolved_cache = expected_cache.resolve(strict=True)
    except OSError as error:
        raise RuntimeError("isolated Python cache root is unavailable") from error
    if (
        sys.pycache_prefix != str(expected_cache)
        or resolved_cache != expected_cache
        or stat.S_ISLNK(cache_info.st_mode)
        or not stat.S_ISDIR(cache_info.st_mode)
        or cache_info.st_uid != os.getuid()
        or stat.S_IMODE(cache_info.st_mode) != 0o700
        or any(expected_cache.iterdir())
    ):
        raise RuntimeError(
            "isolated Python cache root is not exact, private, and empty"
        )
    repo = Path(__file__).resolve().parent.parent
    site_packages = (
        _venv_root()
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    ).resolve(strict=True)
    expected_site = (repo / ".venv" / "lib" / "python3.12" / "site-packages").resolve(
        strict=True
    )
    if site_packages != expected_site or str(site_packages) in sys.path:
        raise RuntimeError(
            "venv site-packages was unexpectedly active before explicit binding"
        )
    if str(repo) in sys.path:
        raise RuntimeError(
            "repository root was unexpectedly present before source binding"
        )
    for external_module in ("jax", "jaxlib", "numpy", "ml_dtypes", "psutil"):
        if (repo / external_module).exists() or (
            repo / f"{external_module}.py"
        ).exists():
            raise RuntimeError(
                f"repository shadows external runtime module: {external_module}"
            )
    sys.path.extend([str(repo), str(site_packages)])
    return {
        "checks": checks,
        "repo_root": str(repo),
        "site_packages": str(site_packages),
        "initial_stdlib_path": expected_stdlib,
        "sys_executable": sys.executable,
        "pycache_prefix": sys.pycache_prefix,
        "pycache_empty_before_bound_imports": True,
        "main_cached_artifact": globals().get("__cached__"),
    }


def _git_manifest() -> dict[str, str | bool]:
    repo = Path(__file__).resolve().parent.parent
    git_environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/nonexistent",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "XDG_CONFIG_HOME": "/nonexistent",
    }

    def run(*arguments: str) -> str:
        result = subprocess.run(
            [
                "/usr/bin/git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                "-C",
                str(repo),
                *arguments,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=git_environment,
        )
        if result.returncode != 0 or result.stderr:
            raise RuntimeError(f"git {' '.join(arguments)} failed")
        return result.stdout.strip()

    status = run("status", "--porcelain=v1")
    if status:
        raise RuntimeError("refusing GPU work from a dirty or untracked worktree")
    return {
        "head": run("rev-parse", "HEAD"),
        "tree": run("rev-parse", "HEAD^{tree}"),
        "worktree_clean": True,
    }


def _runtime_binary_manifest() -> dict[str, dict[str, Any]]:
    site_packages = (
        _venv_root()
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    manifest = {}
    for relative, expected in _EXPECTED_RUNTIME_BINARY_SHA256.items():
        path = site_packages / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"runtime binary is not a regular file: {relative}")
        info = path.stat(follow_symlinks=False)
        digest = _file_sha256(path)
        if digest != expected:
            raise RuntimeError(f"runtime binary hash mismatch: {relative}")
        manifest[relative] = {
            "bytes": info.st_size,
            "mode": stat.S_IMODE(info.st_mode),
            "sha256": digest,
        }
    return manifest


def _stack_manifest() -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for package, expected in _EXPECTED_STACK.items():
        try:
            value = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required package is missing: {package}") from error
        if value != expected:
            raise RuntimeError(
                f"{package} version {value!r} does not match required {expected!r}"
            )
        observed[package] = value
    rocm_version = Path("/opt/rocm/.info/version").read_text().strip()
    if rocm_version != "7.2.4":
        raise RuntimeError(f"ROCm version {rocm_version!r} does not match '7.2.4'")
    observed["rocm"] = rocm_version
    amdgpu_version = Path("/sys/module/amdgpu/version").read_text().strip()
    if amdgpu_version != "6.16.13":
        raise RuntimeError(
            f"AMDGPU version {amdgpu_version!r} does not match '6.16.13'"
        )
    observed["amdgpu"] = amdgpu_version
    observed["runtime_binaries"] = _runtime_binary_manifest()
    return observed


def _module_origin_manifest(modules: dict[str, Any]) -> dict[str, str]:
    repo = Path(__file__).resolve().parent.parent
    site = (
        _venv_root()
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    expected = {
        "jax": site / "jax" / "__init__.py",
        "jaxlib": site / "jaxlib" / "__init__.py",
        "numpy": site / "numpy" / "__init__.py",
        "quantized_lora": repo / "skyrl" / "tx" / "kernels" / "quantized_lora.py",
        "w8a8_lora": repo / "skyrl" / "tx" / "kernels" / "rocm" / "w8a8_lora.py",
    }
    observed = {}
    for name, module in modules.items():
        raw = getattr(module, "__file__", None)
        if not isinstance(raw, str):
            raise RuntimeError(f"loaded module has no filesystem origin: {name}")
        origin = Path(raw).resolve(strict=True)
        if origin != expected[name].resolve(strict=True):
            raise RuntimeError(f"loaded module origin mismatch: {name}")
        observed[name] = str(origin)
    return observed


def _exact_contract(phase: str) -> dict[str, Any]:
    if phase != "compile":
        raise RuntimeError("only the compile diagnostic contract is enabled")
    return {
        "model_family": "Qwen/Qwen3.5-4B",
        "operation": "w8a8_group64_rank8_lora_forward_only",
        "phase": phase,
        "inputs": [
            {"name": "x", "shape": [_ROWS, _IN_FEATURES], "dtype": "bfloat16"},
            {
                "name": "weight_codes",
                "shape": [_IN_FEATURES, _OUT_FEATURES],
                "dtype": "int8",
            },
            {
                "name": "weight_scales",
                "shape": [_IN_FEATURES // _GROUP_SIZE, _OUT_FEATURES],
                "dtype": "bfloat16",
            },
            {
                "name": "lora_a",
                "shape": [_IN_FEATURES, _RANK],
                "dtype": "bfloat16",
            },
            {
                "name": "lora_b",
                "shape": [_RANK, _OUT_FEATURES],
                "dtype": "bfloat16",
            },
            {"name": "lora_scaling", "shape": [], "dtype": "float32"},
        ],
        "output": {"shape": [_ROWS, _OUT_FEATURES], "dtype": "bfloat16"},
        "tiles": {
            "group_size": _GROUP_SIZE,
            "block_m": _BLOCK_M,
            "block_n": _BLOCK_N,
            "row_superblock": _ROW_SUPERBLOCK,
        },
        "dispatch_plan": {
            "compiled_executable_invocations": 0,
            "backward_invocations": 0,
            "warmup_invocations": 0,
            "replay_invocations": 0,
            "gpu_reference_invocations": 0,
            "device_error_reduction_invocations": 0,
        },
        "runtime_numerical_gate_evaluated": False,
        "runtime_promotion": False,
        "execute_rung_enabled": False,
        "outer_profile_rocm_required": True,
        "exact_idle_handoff_required": True,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("abstract", "rocm"), default="abstract")
    parser.add_argument("--phase", choices=("compile", "execute"), default="compile")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--launcher-lock-fd", type=int)
    args = parser.parse_args(argv)
    if args.platform == "rocm" and not args.allow_gpu:
        parser.error(
            "--platform rocm requires the explicit --allow-gpu acknowledgement"
        )
    if args.platform == "abstract" and args.allow_gpu:
        parser.error("--allow-gpu is only valid with --platform rocm")
    if args.platform == "rocm" and args.output is None:
        parser.error("--platform rocm requires --output")
    if args.platform == "rocm" and args.artifact_dir is None:
        parser.error("--platform rocm requires --artifact-dir")
    if args.platform == "rocm" and args.launcher_lock_fd is None:
        parser.error("--platform rocm requires --launcher-lock-fd")
    if args.platform == "abstract" and args.launcher_lock_fd is not None:
        parser.error("--launcher-lock-fd is only valid with --platform rocm")
    if args.phase == "execute":
        parser.error(
            "the execute rung is disabled until exact retained gfx1100 ISA is qualified"
        )
    for path in (args.output, args.artifact_dir):
        if path is not None and path.exists():
            parser.error(f"refusing to overwrite existing path: {path}")
    return args


def _private_parent(path: Path) -> tuple[Path, int]:
    if not path.is_absolute():
        raise RuntimeError("ROCm evidence paths must be absolute")
    parent = path.parent
    if parent != Path(os.path.realpath(parent)):
        raise RuntimeError("ROCm evidence path parent must not contain a symlink")
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    info = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        os.close(descriptor)
        raise RuntimeError(
            "ROCm evidence parent must be a real mode-0700 directory owned by the user"
        )
    return parent, descriptor


def _open_private_output(path: Path) -> TextIO:
    _parent, parent_fd = _private_parent(path)
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
    finally:
        os.close(parent_fd)
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_nlink != 1
    ):
        os.close(descriptor)
        raise RuntimeError("ROCm audit output is not a private regular file")
    return os.fdopen(descriptor, "w", encoding="utf-8")


def _write_private_ir(path: Path, text: str, maximum_bytes: int) -> dict[str, Any]:
    encoded = text.encode("utf-8")
    if not encoded or len(encoded) > maximum_bytes:
        raise RuntimeError(f"IR artifact size is outside its cap: {path.name}")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
    )
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 1
        ):
            raise RuntimeError("IR audit artifact is not a private regular file")
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError("short write while persisting IR audit artifact")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {
        "path": str(path),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _create_private_artifact_dir(path: Path) -> dict[str, Path]:
    _parent, parent_fd = _private_parent(path)
    try:
        os.mkdir(path.name, mode=0o700, dir_fd=parent_fd)
        directory_fd = os.open(
            path.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    finally:
        os.close(parent_fd)
    try:
        info = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            raise RuntimeError("compiler artifact directory is not private")
        children = {
            "jax_cache": "jax-cache",
            "hsaco_cache": "hsaco-cache",
            "triton_cache": "triton-cache",
            "triton_dump": "triton-dump",
            "compiler_dump": "compiler-dump",
        }
        for child in children.values():
            os.mkdir(child, mode=0o700, dir_fd=directory_fd)
        return {name: path / child for name, child in children.items()}
    finally:
        os.close(directory_fd)


def _validate_inherited_lock(lock_fd: int) -> dict[str, Any]:
    if isinstance(lock_fd, bool) or lock_fd < 3:
        raise RuntimeError("launcher lock descriptor must be an open fd of at least 3")
    lock_dir = Path("/run/user") / str(os.getuid()) / f"skyrl-qwen35-rocm-{os.getuid()}"
    path_info = lock_dir.lstat()
    descriptor_info = os.fstat(lock_fd)
    if (
        not stat.S_ISDIR(path_info.st_mode)
        or stat.S_ISLNK(path_info.st_mode)
        or path_info.st_uid != os.getuid()
        or stat.S_IMODE(path_info.st_mode) != 0o700
        or not stat.S_ISDIR(descriptor_info.st_mode)
        or (path_info.st_dev, path_info.st_ino)
        != (descriptor_info.st_dev, descriptor_info.st_ino)
    ):
        raise RuntimeError("inherited descriptor is not the private global launch lock")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(
            "another ROCm process owns the global launch lock"
        ) from error
    os.set_inheritable(lock_fd, False)
    if os.get_inheritable(lock_fd):
        raise RuntimeError("failed to restore close-on-exec on the launch lock")
    return {
        "validated": True,
        "fd": lock_fd,
        "device": descriptor_info.st_dev,
        "inode": descriptor_info.st_ino,
        "close_on_exec": True,
    }


def _validate_controller_supervision(
    run_dir: Path,
    lock_fd: int,
    *,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    cgroup_lines = [
        line
        for line in (proc_root / "self" / "cgroup").read_text().splitlines()
        if line
    ]
    if len(cgroup_lines) != 1 or not cgroup_lines[0].startswith("0::"):
        raise RuntimeError("probe is not in one exact unified cgroup")
    cgroup = cgroup_lines[0][3:]
    scope_match = re.fullmatch(
        r"/user\.slice/user-[0-9]+\.slice/user@[0-9]+\.service/"
        r"app\.slice/(skyrl-w8a8-compile-[0-9]+-[0-9a-f]+\.scope)",
        cgroup,
    )
    if scope_match is None:
        raise RuntimeError("probe is not in the private W8 compile scope")

    parent_pid = os.getppid()
    parent_root = proc_root / str(parent_pid)
    expected_python = (
        Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
    ).resolve(strict=True)
    if (parent_root / "exe").resolve(strict=True) != expected_python:
        raise RuntimeError("probe parent is not the exact project Python runtime")
    raw_command = (parent_root / "cmdline").read_bytes()
    if not raw_command or not raw_command.endswith(b"\0"):
        raise RuntimeError("probe parent command line is unavailable")
    command = [part.decode("utf-8") for part in raw_command[:-1].split(b"\0")]
    repo = Path(__file__).resolve().parent.parent
    site_packages = repo / ".venv" / "lib" / "python3.12" / "site-packages"
    cache_root = run_dir / "python-cache"
    expected_command = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-X",
        f"pycache_prefix={cache_root}",
        "-c",
        "<hash-bound-bootstrap>",
        str(site_packages),
        str(repo),
        str(repo / "rocm" / "profile_rocm.py"),
        _EXPECTED_SOURCE_SHA256["profiler"],
        "<hash-bound-profile-runtime>",
        str(cache_root),
        "--output",
        str(run_dir / "telemetry.jsonl"),
        "--card",
        "card1",
        "--interval",
        "0.05",
        "--baseline-seconds",
        "5.0",
        "--timeout",
        "300.0",
        "--terminate-grace-seconds",
        "0.5",
        "--sensor-grace-seconds",
        "15.0",
        "--max-junction-temp-c",
        "90.0",
        "--max-gpu-power-watts",
        "315.0",
        "--max-vram-gib",
        "2.0",
        "--min-host-available-gib",
        "0.0",
        "--max-swap-gib",
        "8.0",
        "--record-command",
        "--pass-fd",
        str(lock_fd),
        "--",
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-X",
        f"pycache_prefix={cache_root}",
        str(repo / "rocm" / "probe_w8a8_lora_forward.py"),
        "--platform",
        "rocm",
        "--phase",
        "compile",
        "--allow-gpu",
        "--output",
        str(run_dir / "probe.jsonl"),
        "--artifact-dir",
        str(run_dir / "compiler-artifacts"),
        "--launcher-lock-fd",
        str(lock_fd),
    ]
    if len(command) != len(expected_command):
        raise RuntimeError("probe parent command length is not exact")
    if (
        hashlib.sha256(command[7].encode()).hexdigest()
        != _EXPECTED_PROFILE_BOOTSTRAP_SHA256
        or hashlib.sha256(command[12].encode()).hexdigest()
        != _EXPECTED_PROFILE_RUNTIME_JSON_SHA256
    ):
        raise RuntimeError("probe parent bootstrap/runtime binding is not exact")
    expected_command[7] = command[7]
    expected_command[12] = command[12]
    if command != expected_command:
        raise RuntimeError("probe parent profiler/safety command is not exact")
    return {
        "validated": True,
        "scope": scope_match.group(1),
        "cgroup": cgroup,
        "parent_pid": parent_pid,
        "parent_executable": str(expected_python),
        "parent_command_sha256": hashlib.sha256(raw_command).hexdigest(),
    }


def _require_exact_or_unset(name: str, expected: str) -> None:
    value = os.environ.get(name)
    if value is not None and value != expected:
        raise RuntimeError(f"{name}={value!r} conflicts with required {expected!r}")


def _require_unset(name: str) -> None:
    if name in os.environ:
        raise RuntimeError(f"{name} must be unset for this one-shot probe")


def _configure_environment(paths: dict[str, Path]) -> dict[str, str | None]:
    required = {
        "JAX_PLATFORMS": "rocm",
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
        "JAX_ROCM_VISIBLE_DEVICES": "0",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_CLIENT_MEM_FRACTION": "0.075",
    }
    for name, expected in required.items():
        _require_exact_or_unset(name, expected)
    prohibited = (
        "HSA_OVERRIDE_GFX_VERSION",
        "JAX_PJRT_CLIENT_CREATE_OPTIONS",
        "JAX_MOCK_GPU_TOPOLOGY",
        "MOCK_NUM_GPU_PROCESSES",
        "TF_FORCE_UNIFIED_MEMORY",
        "XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB",
        "JAX_COMPILATION_CACHE_DIR",
        "TEST_UNDECLARED_OUTPUTS_DIR",
        "TF_XLA_HSACO_BITCODE_SIZE_THRESHOLD",
        "TF_XLA_HSACO_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "TRITON_DUMP_DIR",
        "TRITON_KERNEL_DUMP",
        "AMDGCN_ENABLE_DUMP",
        "LD_PRELOAD",
    )
    for name in prohibited:
        _require_unset(name)
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
    allowed_inherited = {*required, "XLA_FLAGS"}
    unexpected = sorted(
        name
        for name in os.environ
        if name not in allowed_inherited
        and any(name.startswith(prefix) for prefix in accelerator_prefixes)
    )
    if unexpected:
        raise RuntimeError(
            "refusing unexpected accelerator environment: " + ", ".join(unexpected)
        )
    original_xla_flags = os.environ.get("XLA_FLAGS")
    if original_xla_flags not in (None, "", _COMMAND_BUFFER_FLAG):
        raise RuntimeError(
            "XLA_FLAGS must be unset or the exact sole command-buffer-disable flag"
        )
    configured = {
        **required,
        "XLA_FLAGS": _COMMAND_BUFFER_FLAG,
        "JAX_COMPILATION_CACHE_DIR": str(paths["jax_cache"]),
        "JAX_ENABLE_COMPILATION_CACHE": "true",
        "JAX_RAISE_PERSISTENT_CACHE_ERRORS": "true",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
        "TF_XLA_HSACO_CACHE_DIR": str(paths["hsaco_cache"]),
        "TRITON_CACHE_DIR": str(paths["triton_cache"]),
        "TRITON_DUMP_DIR": str(paths["triton_dump"]),
        "TRITON_KERNEL_DUMP": "1",
        "AMDGCN_ENABLE_DUMP": "1",
        "TEST_UNDECLARED_OUTPUTS_DIR": str(paths["compiler_dump"]),
    }
    os.environ.update(configured)
    if os.environ.get("XLA_FLAGS") != _COMMAND_BUFFER_FLAG:
        raise RuntimeError("failed to establish the exact sole XLA_FLAGS value")
    return {"XLA_FLAGS_original": original_xla_flags, **configured}


def _hardware_preflight() -> tuple[dict[str, Any], Callable[[], dict[str, Any]]]:
    from rocm.amdgpu_safety import (
        require_clean_amdgpu_boot,
        require_headless_unowned_amdgpu,
    )
    from rocm.qwen35_prewarm_handoff import _discover_device, _fuser_owner_pids

    clean = require_clean_amdgpu_boot()
    generic = require_headless_unowned_amdgpu()
    device = _discover_device()
    if device.device_id != _EXPECTED_DEVICE_ID:
        raise RuntimeError("exact RX 7900 XTX PCI device identity was not found")
    kfd_owners = list(_fuser_owner_pids(device.kfd_node))
    render_owners = list(_fuser_owner_pids(device.render_node))
    if kfd_owners or render_owners:
        raise RuntimeError("KFD or the exact AMD render node is already owned")
    render_descriptor = os.open(
        device.render_node,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
    )
    try:
        render_info = os.fstat(render_descriptor)
        if not stat.S_ISCHR(render_info.st_mode):
            raise RuntimeError("exact AMD render node is not a character device")
        observed_rdev = (
            f"{os.major(render_info.st_rdev)}:{os.minor(render_info.st_rdev)}"
        )
        if observed_rdev != device.render_sysfs_dev:
            raise RuntimeError("opened AMD render node identity changed")
        pre_backend_limits = _read_hardware_limits(device.device_root)
        wake_journal = require_clean_amdgpu_boot()
    finally:
        os.close(render_descriptor)
    return (
        {
            **clean,
            "generic_headless_preflight": generic,
            "device": device.identity(),
            "kfd_owner_pids": kfd_owners,
            "render_owner_pids": render_owners,
            "controlled_render_wake_only": True,
            "pre_backend_hardware_limits": pre_backend_limits,
            "journal_after_controlled_wake": wake_journal,
        },
        require_clean_amdgpu_boot,
    )


def _read_int_with_wake_retry(path: Path, label: str) -> int:
    last_error: BaseException | None = None
    for _ in range(50):
        try:
            return int(path.read_text().strip())
        except (OSError, ValueError) as error:
            last_error = error
            time.sleep(0.1)
    raise RuntimeError(
        f"{label} is unavailable after a controlled device wake"
    ) from last_error


def _read_hardware_limits(device: Path) -> dict[str, Any]:
    hwmons = sorted((device / "hwmon").glob("hwmon*"))
    if len(hwmons) != 1:
        raise RuntimeError(f"expected one AMD hwmon directory, observed {len(hwmons)}")
    cap_path = hwmons[0] / "power1_cap"
    power_cap = _read_int_with_wake_retry(cap_path, "hardware power cap")
    if power_cap <= 0 or power_cap > _MAX_HARDWARE_POWER_CAP_UW:
        raise RuntimeError(
            f"hardware power cap {power_cap} uW exceeds the 315 W contract"
        )
    junction_path: Path | None = None
    for label_path in sorted(hwmons[0].glob("temp[0-9]*_label")):
        if label_path.read_text().strip() == "junction":
            junction_path = label_path.with_name(
                label_path.name.removesuffix("_label") + "_input"
            )
            break
    if junction_path is None:
        raise RuntimeError("junction temperature sensor is unavailable")
    junction_millic = _read_int_with_wake_retry(junction_path, "junction temperature")
    if junction_millic < 0 or junction_millic > 85_000:
        raise RuntimeError(
            f"junction temperature {junction_millic / 1000:g} C is outside the <=85 C launch gate"
        )
    return {
        "power_cap_path": str(cap_path),
        "power_cap_uw": power_cap,
        "maximum_power_cap_uw": _MAX_HARDWARE_POWER_CAP_UW,
        "junction_path": str(junction_path),
        "junction_temperature_millic": junction_millic,
        "maximum_launch_junction_millic": 85_000,
    }


def _array_manifest(name: str, value: Any) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "nbytes": int(value.nbytes),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _construct_host_case() -> tuple[tuple[Any, ...], list[dict[str, Any]], Any]:
    import ml_dtypes
    import numpy as np

    row = np.arange(_ROWS, dtype=np.int32)[:, None]
    k = np.arange(_IN_FEATURES, dtype=np.int32)[None, :]
    x = (((row * 11 + k * 5) % 33) - 16).astype(np.float32) / 8.0
    x = x.astype(ml_dtypes.bfloat16)

    k_index = np.arange(_IN_FEATURES, dtype=np.int32)[:, None]
    n_index = np.arange(_OUT_FEATURES, dtype=np.int32)[None, :]
    dense_weight = (((k_index * 7 + n_index * 3) % 41) - 20).astype(np.float32) / 16.0
    grouped_weight = dense_weight.reshape((-1, _GROUP_SIZE, _OUT_FEATURES))
    weight_amax = np.max(np.abs(grouped_weight), axis=1)
    weight_scales_f32 = np.where(weight_amax > 0, weight_amax / 127.0, 1.0).astype(
        np.float32
    )
    weight_codes = (
        np.clip(np.rint(grouped_weight / weight_scales_f32[:, None, :]), -127, 127)
        .astype(np.int8)
        .reshape((_IN_FEATURES, _OUT_FEATURES))
    )
    weight_scales = weight_scales_f32.astype(ml_dtypes.bfloat16)

    rank_index = np.arange(_RANK, dtype=np.int32)
    lora_a = (
        (
            ((np.arange(_IN_FEATURES)[:, None] * 3 + rank_index[None, :]) % 17) - 8
        ).astype(np.float32)
        / 16.0
    ).astype(ml_dtypes.bfloat16)
    lora_b = (
        (
            ((rank_index[:, None] * 5 + np.arange(_OUT_FEATURES)[None, :]) % 19) - 9
        ).astype(np.float32)
        / 16.0
    ).astype(ml_dtypes.bfloat16)
    scaling = np.asarray(_SCALING, dtype=np.float32)

    x_f32 = np.asarray(x, dtype=np.float32)
    x_grouped = x_f32.reshape((_ROWS, -1, _GROUP_SIZE))
    x_amax = np.max(np.abs(x_grouped), axis=-1)
    x_scales = np.where(x_amax > 0, x_amax / 127.0, 1.0).astype(np.float32)
    x_codes = np.clip(np.rint(x_grouped / x_scales[..., None]), -127, 127).astype(
        np.int8
    )
    integer = np.einsum(
        "mgk,gkn->mgn",
        x_codes.astype(np.int32),
        weight_codes.reshape((-1, _GROUP_SIZE, _OUT_FEATURES)).astype(np.int32),
        dtype=np.int32,
    )
    base = np.sum(
        integer.astype(np.float32)
        * x_scales[..., None]
        * np.asarray(weight_scales, dtype=np.float32)[None, :, :],
        axis=1,
        dtype=np.float32,
    )
    z = np.matmul(x_f32, np.asarray(lora_a, dtype=np.float32), dtype=np.float32)
    low_rank = np.matmul(z, np.asarray(lora_b, dtype=np.float32), dtype=np.float32)
    expected = (base + scaling * low_rank).astype(ml_dtypes.bfloat16)
    arguments = (x, weight_codes, weight_scales, lora_a, lora_b, scaling)
    manifests = [
        _array_manifest(name, value)
        for name, value in zip(
            ("x", "weight_codes", "weight_scales", "lora_a", "lora_b", "lora_scaling"),
            arguments,
            strict=True,
        )
    ]
    observed_hashes = {manifest["name"]: manifest["sha256"] for manifest in manifests}
    observed_hashes["expected"] = _array_manifest("expected", expected)["sha256"]
    if observed_hashes != _EXPECTED_HOST_SHA256:
        raise RuntimeError("deterministic host boundary or oracle hash changed")
    return arguments, manifests, expected


def _mask_ir_strings_and_comments(text: str) -> str:
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


def _custom_call_blocks(text: str, dialect: str) -> list[str]:
    raw_lines = text.splitlines()
    masked_lines = _mask_ir_strings_and_comments(text).splitlines()
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
        raise ValueError(f"unsupported IR dialect: {dialect}")
    blocks: list[str] = []
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


def _top_level_string_attribute_values(text: str, attribute: str) -> list[str]:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", attribute) is None:
        raise ValueError(f"invalid IR attribute name: {attribute}")
    structural = _mask_ir_strings_and_comments(text)
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


def _map_attribute_contents(
    text: str, attribute: str, *, required_brace_depth: int
) -> list[str]:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", attribute) is None:
        raise ValueError(f"invalid IR map attribute name: {attribute}")
    structural = _mask_ir_strings_and_comments(text)
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


def _call_targets(block: str, dialect: str) -> list[str]:
    if dialect == "optimized_hlo":
        return _top_level_string_attribute_values(block, "custom_call_target")
    if dialect != "stablehlo":
        raise ValueError(f"unsupported IR dialect: {dialect}")
    structural = _mask_ir_strings_and_comments(block)
    return re.findall(r"\bstablehlo\.custom_call\s+@([A-Za-z0-9_.$-]+)", structural)


def _kernel_name_binding(block: str, dialect: str) -> dict[str, Any]:
    attribute = "mhlo.backend_config" if dialect == "stablehlo" else "backend_config"
    maps = _map_attribute_contents(
        block,
        attribute,
        required_brace_depth=1 if dialect == "stablehlo" else 0,
    )
    names = (
        _top_level_string_attribute_values(maps[0], "name") if len(maps) == 1 else []
    )
    return {"backend_config_map_count": len(maps), "names": names}


def _entry_regions(text: str, dialect: str) -> list[dict[str, tuple[int, int]]]:
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


def _entry_result_flow(
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


def _ir_summary(text: str, dialect: str) -> dict[str, Any]:
    structural_text = _mask_ir_strings_and_comments(text)
    blocks = _custom_call_blocks(text, dialect)
    pallas = [
        block
        for block in blocks
        if any(target in _PALLAS_TARGETS for target in _call_targets(block, dialect))
    ]
    raw_opcode_count = len(
        re.findall(
            r"\bstablehlo\.custom_call\b"
            if dialect == "stablehlo"
            else r"\bcustom-call\s*\(",
            structural_text,
        )
    )
    call_positions = [
        match.start()
        for match in re.finditer(
            r"\bstablehlo\.custom_call\b"
            if dialect == "stablehlo"
            else r"\bcustom-call\s*\(",
            structural_text,
        )
    ]
    entries = _entry_regions(structural_text, dialect)
    entry = entries[0] if len(entries) == 1 else None
    entry_body = entry["body"] if entry is not None else None
    result_flow = _entry_result_flow(structural_text, entry, dialect)
    marker_pattern = re.compile(
        rf"(?<![A-Za-z0-9_.$-]){re.escape(_EXPECTED_KERNEL_NAME)}"
        r"(?![A-Za-z0-9_.$-])"
    )
    forbidden_patterns = {
        "graph_or_capture": r"(?i)(?:hip|cuda)?graph(?:launch|exec|instantiate)?|capture",
        "command_buffer": r"(?i)command[_ -]?buffer",
        "replay": r"(?i)replay",
    }
    forbidden = {
        name: bool(re.search(pattern, text))
        for name, pattern in forbidden_patterns.items()
    }
    kernel_name_bindings = [_kernel_name_binding(block, dialect) for block in pallas]
    return {
        "dialect": dialect,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "characters": len(text),
        "lines": len(text.splitlines()),
        "custom_call_count": len(blocks),
        "raw_custom_call_opcode_count": raw_opcode_count,
        "custom_call_parser_consistent": raw_opcode_count == len(blocks),
        "entry_count": len(entries),
        "sole_custom_call_owned_by_entry": entry_body is not None
        and len(call_positions) == 1
        and entry_body[0] <= call_positions[0] < entry_body[1],
        "entry_result_flow": result_flow,
        "pallas_custom_call_count": len(pallas),
        "expected_kernel_marker_count": sum(
            len(marker_pattern.findall(block)) for block in pallas
        ),
        "kernel_name_bindings": kernel_name_bindings,
        "pallas_targets": [_call_targets(block, dialect) for block in pallas],
        "backward_marker_present": "w8a16_lora_input_vjp" in text,
        "while_count": len(
            re.findall(
                r"\bstablehlo\.while\b" if dialect == "stablehlo" else r"\bwhile\s*\(",
                structural_text,
            )
        ),
        "fusion_opcode_count": (
            0
            if dialect == "stablehlo"
            else len(re.findall(r"\bfusion\s*\(", structural_text))
        ),
        "forbidden_markers": forbidden,
    }


def _stablehlo_gate(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "dialect_is_stablehlo": summary.get("dialect") == "stablehlo",
        "parser_matches_raw_opcode_count": summary.get("custom_call_parser_consistent")
        is True,
        "exactly_one_custom_call_total": summary.get("custom_call_count") == 1,
        "unique_public_entry": summary.get("entry_count") == 1,
        "sole_custom_call_owned_by_entry": summary.get(
            "sole_custom_call_owned_by_entry"
        )
        is True,
        "entry_result_depends_on_custom_call": summary.get("entry_result_flow", {}).get(
            "passed"
        )
        is True,
        "exactly_one_pallas_call": summary.get("pallas_custom_call_count") == 1,
        "exactly_one_exact_forward_kernel_name": summary.get("kernel_name_bindings")
        == [{"backend_config_map_count": 1, "names": [_EXPECTED_KERNEL_NAME]}],
        "exact_triton_target": summary.get("pallas_targets")
        == [["__gpu$xla.gpu.triton"]],
        "no_backward_marker": summary.get("backward_marker_present") is False,
        "no_graph_capture_command_buffer_or_replay": not any(
            summary.get("forbidden_markers", {}).values()
        ),
        "no_outer_loop": summary.get("while_count") == 0,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _structural_gate(*summaries: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "exact_dialects": sorted(summary["dialect"] for summary in summaries)
        == ["optimized_hlo", "stablehlo"]
    }
    for summary in summaries:
        dialect = str(summary["dialect"])
        checks[f"{dialect}_one_pallas_call"] = summary["pallas_custom_call_count"] == 1
        checks[f"{dialect}_one_custom_call_total"] = summary["custom_call_count"] == 1
        checks[f"{dialect}_unique_public_entry"] = summary["entry_count"] == 1
        checks[f"{dialect}_call_owned_by_entry"] = (
            summary["sole_custom_call_owned_by_entry"] is True
        )
        checks[f"{dialect}_entry_result_depends_on_call"] = (
            summary["entry_result_flow"]["passed"] is True
        )
        checks[f"{dialect}_parser_matches_raw_opcodes"] = (
            summary["custom_call_parser_consistent"] is True
        )
        checks[f"{dialect}_one_forward_kernel_name"] = summary[
            "kernel_name_bindings"
        ] == [{"backend_config_map_count": 1, "names": [_EXPECTED_KERNEL_NAME]}]
        checks[f"{dialect}_no_backward_marker"] = (
            summary["backward_marker_present"] is False
        )
        checks[f"{dialect}_exact_triton_target"] = summary["pallas_targets"] == [
            ["__gpu$xla.gpu.triton"]
        ]
        checks[f"{dialect}_no_outer_loop"] = summary["while_count"] == 0
        checks[f"{dialect}_no_forbidden_runtime_marker"] = not any(
            summary["forbidden_markers"].values()
        )
    return {"checks": checks, "passed": all(checks.values())}


def _compiled_memory(compiled: Any) -> dict[str, Any]:
    stats = compiled.memory_analysis()
    if stats is None:
        return {"available": False}
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "generated_code_size_in_bytes",
    )
    return {
        "available": True,
        **{name: int(getattr(stats, name)) for name in names if hasattr(stats, name)},
    }


def _memory_gate(memory: dict[str, Any]) -> dict[str, Any]:
    names = ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes")
    available = memory.get("available") is True and all(
        isinstance(memory.get(name), int) and memory[name] >= 0 for name in names
    )
    total = sum(memory[name] for name in names) if available else None
    checks = {
        "memory_analysis_available": available,
        "temporary_at_most_256_mib": available
        and memory["temp_size_in_bytes"] <= _MAX_TEMP_BYTES,
        "total_at_most_512_mib": available
        and total is not None
        and total <= _MAX_TOTAL_BYTES,
    }
    return {"checks": checks, "total_bytes": total, "passed": all(checks.values())}


def _artifact_inventory(root: Path) -> dict[str, Any]:
    paths = sorted(root.rglob("*"))
    if len(paths) > _MAX_ARTIFACT_FILES * 4:
        raise RuntimeError("compiler artifact node count exceeds the diagnostic cap")
    entries = []
    total = 0
    for path in paths:
        if path.is_symlink():
            raise RuntimeError(f"refusing compiler artifact symlink: {path}")
        if path.is_dir():
            info = path.stat(follow_symlinks=False)
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise RuntimeError(
                    f"compiler artifact directory is not private: {path}"
                )
            continue
        if len(entries) >= _MAX_ARTIFACT_FILES:
            raise RuntimeError(
                "compiler artifact file count exceeds the diagnostic cap"
            )
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.getuid()
                or stat.S_IMODE(before.st_mode) & 0o077
                or before.st_nlink != 1
                or before.st_size < 0
                or before.st_size > _MAX_ARTIFACT_FILE_BYTES
            ):
                raise RuntimeError(f"invalid compiler artifact: {path}")
            digest = hashlib.sha256()
            bytes_read = 0
            while chunk := os.read(descriptor, 1024 * 1024):
                bytes_read += len(chunk)
                if bytes_read > _MAX_ARTIFACT_FILE_BYTES:
                    raise RuntimeError("compiler artifact grew beyond its cap")
                digest.update(chunk)
            after = os.fstat(descriptor)
            if bytes_read != before.st_size or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise RuntimeError("compiler artifact changed while it was hashed")
        finally:
            os.close(descriptor)
        size = before.st_size
        total += size
        if total > _MAX_ARTIFACT_TOTAL_BYTES:
            raise RuntimeError("compiler artifact total exceeds the diagnostic cap")
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": size,
                "sha256": digest.hexdigest(),
            }
        )
    return {
        "file_count": len(entries),
        "total_bytes": total,
        "maximum_file_count": _MAX_ARTIFACT_FILES,
        "maximum_file_bytes": _MAX_ARTIFACT_FILE_BYTES,
        "maximum_total_bytes": _MAX_ARTIFACT_TOTAL_BYTES,
        "files": entries,
    }


def _json_scalars(values: Any) -> Any:
    if values is None or isinstance(values, (bool, int, float, str)):
        return values
    if isinstance(values, dict):
        return {str(key): _json_scalars(value) for key, value in values.items()}
    if isinstance(values, (tuple, list)):
        return [_json_scalars(value) for value in values]
    if hasattr(values, "item"):
        try:
            return _json_scalars(values.item())
        except (TypeError, ValueError):
            pass
    return str(values)


def _run_rocm(
    args: argparse.Namespace,
    output: TextIO,
    artifacts: Path,
    device_root: Path,
    require_clean_boot: Callable[[], dict[str, Any]],
) -> int:
    if args.phase != "compile":
        raise RuntimeError("only the compile diagnostic backend path is enabled")
    import jax
    import jax.numpy as jnp
    import jaxlib
    import numpy as np
    from jax.extend import backend as jax_backend

    import skyrl.tx.kernels.quantized_lora as quantized_lora_module
    import skyrl.tx.kernels.rocm.w8a8_lora as w8a8_lora_module

    host_arguments, host_manifests, host_expected = _construct_host_case()
    signature = tuple(
        jax.ShapeDtypeStruct(value.shape, value.dtype) for value in host_arguments
    )
    host_expected_manifest = _array_manifest("expected", host_expected)
    module_origins = _module_origin_manifest(
        {
            "jax": jax,
            "jaxlib": jaxlib,
            "numpy": np,
            "quantized_lora": quantized_lora_module,
            "w8a8_lora": w8a8_lora_module,
        }
    )

    backend = jax.default_backend()
    platform_version = str(jax_backend.get_backend().platform_version)
    devices = jax.devices()
    if backend != "gpu" or "rocm" not in platform_version.lower() or len(devices) != 1:
        raise RuntimeError(
            f"requested one ROCm GPU, resolved {backend!r}, {platform_version!r}, {devices!r}"
        )
    _emit(
        {
            "record_type": "host_oracle",
            "timestamp": _utc_now(),
            "inputs": host_manifests,
            "expected": host_expected_manifest,
            "verified_against_bound_hashes": True,
            "compile_signature_kind": "ShapeDtypeStruct",
            "compile_abstract_signature_derived_from_host_metadata": True,
            "lowering_consumed_host_values": False,
            "runtime_comparison_evaluated": False,
            "compiled_executable_invocations": 0,
        },
        output,
    )
    del host_arguments, host_expected
    hardware_limits = _read_hardware_limits(device_root)
    _emit(
        {
            "record_type": "backend_ready",
            "timestamp": _utc_now(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "platform": backend,
            "platform_version": platform_version,
            "devices": [str(device) for device in devices],
            "hardware_limits": hardware_limits,
            "module_origins": module_origins,
            "compiled_executable_invocations": 0,
        },
        output,
    )
    _emit(
        {
            "record_type": "journal_checkpoint",
            "stage": "after_backend_initialization",
            "timestamp": _utc_now(),
            "safety": require_clean_boot(),
        },
        output,
    )

    def candidate(
        x: Any, codes: Any, scales: Any, lora_a: Any, lora_b: Any, scaling: Any
    ) -> Any:
        from jax.experimental import pallas as pl

        padded_x = jnp.pad(x, ((0, _BLOCK_M - _ROWS), (0, 0)))
        x_codes, x_scales = w8a8_lora_module._dynamic_group64_quantize(padded_x)
        z = w8a8_lora_module._dot_general_f32(
            padded_x,
            lora_a,
            left_contract=1,
            right_contract=0,
        )
        run = pl.pallas_call(
            partial(
                w8a8_lora_module._w8a8_forward_kernel,
                block_m=_BLOCK_M,
                block_n=_BLOCK_N,
            ),
            out_shape=jax.ShapeDtypeStruct((_BLOCK_M, _OUT_FEATURES), x.dtype),
            grid=(1, 2),
            compiler_params=w8a8_lora_module._compiler_params(),
            interpret=False,
            name=_EXPECTED_KERNEL_NAME,
        )
        return run(
            x_codes,
            x_scales,
            codes,
            scales,
            z,
            lora_b,
            scaling,
        )[:_ROWS]

    limits_before_lower = _read_hardware_limits(device_root)
    lower_start = time.perf_counter()
    try:
        lowered = jax.jit(candidate).lower(*signature)
    finally:
        lower_end = time.perf_counter()
        lower_checkpoint = require_clean_boot()
    lower_seconds = lower_end - lower_start
    stable_text = str(lowered.compiler_ir(dialect="stablehlo"))
    stable = _ir_summary(stable_text, "stablehlo")
    stable_release = _stablehlo_gate(stable)
    stable_artifact = _write_private_ir(
        artifacts.parent / "w8a8-forward.stablehlo.mlir",
        stable_text,
        8 * 1024**2,
    )
    del stable_text
    _emit(
        {
            "record_type": "lowered",
            "timestamp": _utc_now(),
            "lower_seconds": lower_seconds,
            "hardware_limits_before_lower": limits_before_lower,
            "stablehlo": stable,
            "stablehlo_artifact": stable_artifact,
            "stablehlo_precompile_gate": stable_release,
            "journal_checkpoint": lower_checkpoint,
            "compiled_executable_invocations": 0,
        },
        output,
    )
    if not stable_release["passed"]:
        raise RuntimeError("StableHLO failed before-compile structural gate")

    limits_before_compile = _read_hardware_limits(device_root)
    compile_start = time.perf_counter()
    try:
        compiled = lowered.compile()
    finally:
        compile_end = time.perf_counter()
        compile_checkpoint = require_clean_boot()
    compile_seconds = compile_end - compile_start
    limits_after_compile = _read_hardware_limits(device_root)
    optimized_text = compiled.as_text()
    if optimized_text is None:
        raise RuntimeError("optimized HLO text is unavailable")
    optimized = _ir_summary(optimized_text, "optimized_hlo")
    optimized_artifact = _write_private_ir(
        artifacts.parent / "w8a8-forward.optimized.hlo",
        optimized_text,
        32 * 1024**2,
    )
    del optimized_text
    memory = _compiled_memory(compiled)
    structural = _structural_gate(stable, optimized)
    memory_proof = _memory_gate(memory)
    artifact_inventory = _artifact_inventory(artifacts)
    artifact_proof = {
        "nonempty": artifact_inventory["file_count"] > 0
        and artifact_inventory["total_bytes"] > 0,
        "isa_qualified": False,
        "reason": "retained artifacts require separate exact gfx1100 disassembly",
    }
    release = {
        "structural_gate": structural,
        "memory_gate": memory_proof,
        "artifact_gate": artifact_proof,
        "runtime_promotion": False,
        "passed": structural["passed"]
        and memory_proof["passed"]
        and artifact_proof["nonempty"],
    }
    _emit(
        {
            "record_type": "compiled",
            "timestamp": _utc_now(),
            "compile_seconds": compile_seconds,
            "hardware_limits_before_compile": limits_before_compile,
            "hardware_limits_after_compile": limits_after_compile,
            "optimized_hlo": optimized,
            "optimized_hlo_artifact": optimized_artifact,
            "compiled_memory": memory,
            "release_gate": release,
            "journal_checkpoint": compile_checkpoint,
            "artifact_inventory": artifact_inventory,
            "compiled_executable_invocations": 0,
        },
        output,
    )
    if not release["passed"]:
        raise RuntimeError(
            "compiled executable failed structural or memory release gate"
        )

    del compiled
    source_postflight = _assert_bound_sources()
    git_postflight = _git_manifest()
    stack_postflight = _stack_manifest()
    _emit(
        {
            "record_type": "completed",
            "timestamp": _utc_now(),
            "status": "passed_compile_diagnostic_unpromoted",
            "runtime_promotion": False,
            "isa_qualified": False,
            "compiled_executable_invocations": 0,
            "source_postflight": source_postflight,
            "git_postflight": git_postflight,
            "stack_postflight": stack_postflight,
            "journal_postflight": require_clean_boot(),
        },
        output,
    )
    return 0


def _execute(args: argparse.Namespace, output: TextIO) -> int:
    _emit(
        {
            "record_type": "manifest",
            "timestamp": _utc_now(),
            "platform_requested": args.platform,
            "allow_gpu": args.allow_gpu,
            "contract": _exact_contract(args.phase),
            "graph_api_used": False,
            "command_buffer_used": False,
            "jax_imported": False,
        },
        output,
    )
    if args.platform == "abstract":
        _emit(
            {
                "record_type": "refused",
                "timestamp": _utc_now(),
                "status": "no_gpu_abstract_manifest_only",
                "jax_imported": False,
            },
            output,
        )
        return 0

    os.umask(0o077)
    stage = "source_binding"
    require_clean_boot: Callable[[], dict[str, Any]] | None = None
    try:
        isolated_python = _establish_isolated_import_path(args.output.parent)
        sources = _assert_bound_sources()
        git = _git_manifest()
        stack = _stack_manifest()
        stage = "artifact_directory"
        artifact_paths = _create_private_artifact_dir(args.artifact_dir)
        stage = "inherited_lock"
        lock = _validate_inherited_lock(args.launcher_lock_fd)
        stage = "environment"
        environment = _configure_environment(artifact_paths)
        stage = "controller_supervision"
        supervision = _validate_controller_supervision(
            args.output.parent, args.launcher_lock_fd
        )
        _emit(
            {
                "record_type": "static_preflight",
                "timestamp": _utc_now(),
                "sources": sources,
                "isolated_python": isolated_python,
                "git": git,
                "stack": stack,
                "inherited_lock": lock,
                "controller_supervision": supervision,
                "environment": environment,
                "jax_imported": False,
            },
            output,
        )
        stage = "hardware_preflight"
        hardware, require_clean_boot = _hardware_preflight()
        _emit(
            {
                "record_type": "hardware_preflight",
                "timestamp": _utc_now(),
                "hardware": hardware,
                "jax_imported": False,
            },
            output,
        )
        stage = "rocm_compile_diagnostic"
        return _run_rocm(
            args,
            output,
            args.artifact_dir,
            Path(hardware["device"]["pci_sysfs_path"]),
            require_clean_boot,
        )
    except BaseException as error:
        postflight: dict[str, Any] | None = None
        if require_clean_boot is None:
            postflight = {
                "available": False,
                "reason": "verified hardware preflight did not complete",
            }
        else:
            try:
                postflight = require_clean_boot()
            except BaseException as postflight_error:
                postflight = {
                    "failed": True,
                    "error_type": type(postflight_error).__name__,
                    "message": str(postflight_error),
                }
        _emit(
            {
                "record_type": "error",
                "timestamp": _utc_now(),
                "stage": stage,
                "status": "failed_closed",
                "error_type": type(error).__name__,
                "message": str(error),
                "journal_postflight": postflight,
            },
            output,
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.output is None:
        return _execute(args, sys.stdout)
    with _open_private_output(args.output) as output:
        return _execute(args, output)


if __name__ == "__main__":
    raise SystemExit(main())
