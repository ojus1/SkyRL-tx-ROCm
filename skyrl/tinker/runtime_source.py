"""Fail-closed source-origin checks for commit-keyed Qwen ROCm servers.

This module is deliberately stdlib-only and never imports JAX. Generic Tinker
launches remain unchanged when the runtime-source claims are absent. When the
hardened ROCm launcher supplies them, both API and engine must resolve from an
exact full-HEAD private snapshot and inherit the graph-free cache policy used
by the compile-only prewarm.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Mapping

_SOURCE_ROOT_ENV = "SKYRL_QWEN35_RUNTIME_SOURCE_ROOT"
_GIT_HEAD_ENV = "SKYRL_QWEN35_RUNTIME_GIT_HEAD"
_REPO_ROOT_ENV = "SKYRL_QWEN35_RUNTIME_REPO_ROOT"
_UV_EXECUTABLE_ENV = "SKYRL_QWEN35_RUNTIME_UV_EXECUTABLE"
_MEMORY_MODE_ENV = "SKYRL_QWEN35_RUNTIME_MEMORY_MODE"
_LAUNCH_LOCK_FD_ENV = "SKYRL_QWEN35_LAUNCH_LOCK_FD"
_T64_CACHE_ATTEST_ENV = "SKYRL_QWEN35_RUNTIME_T64_CACHE_ATTEST"
_PREWARM_AUDIT_PATH_ENV = "SKYRL_QWEN35_RUNTIME_PREWARM_AUDIT_PATH"
_PREWARM_AUDIT_SHA256_ENV = "SKYRL_QWEN35_RUNTIME_PREWARM_AUDIT_SHA256"
_PREWARM_HANDOFF_PATH_ENV = "SKYRL_QWEN35_RUNTIME_PREWARM_HANDOFF_PATH"
_PREWARM_HANDOFF_SHA256_ENV = "SKYRL_QWEN35_RUNTIME_PREWARM_HANDOFF_SHA256"
_CACHE_ATTESTATION_ENVIRONMENT = (
    _T64_CACHE_ATTEST_ENV,
    _PREWARM_AUDIT_PATH_ENV,
    _PREWARM_AUDIT_SHA256_ENV,
    _PREWARM_HANDOFF_PATH_ENV,
    _PREWARM_HANDOFF_SHA256_ENV,
)
_SOURCE_CACHE_NAMESPACE = "skyrl-source-snapshots-private-v1"
_UV_SIZE_BYTES = 59_853_688
_UV_SHA256 = "646adf5cf12ba17d1a41fa77c8dd6496f73651dcfeeed6b5f4ec019b36bc7153"
_ROLE_PATHS = {
    "api": Path("skyrl/tinker/api.py"),
    "engine": Path("skyrl/tinker/engine.py"),
}
_REQUIRED_ENVIRONMENT = {
    "HF_XET_HIGH_PERFORMANCE": "1",
    "JAX_COMPILATION_CACHE_EXPECT_PGLE": "false",
    "JAX_COMPILATION_CACHE_MAX_SIZE": "17179869184",
    "JAX_ENABLE_COMPILATION_CACHE": "true",
    "JAX_ENABLE_PGLE": "false",
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": (
        "xla_gpu_per_fusion_autotune_cache_dir"
    ),
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
    "JAX_PLATFORMS": "rocm",
    "JAX_RAISE_PERSISTENT_CACHE_ERRORS": "true",
    "LLVM_PATH": "/opt/rocm/llvm",
    "PYTHONDONTWRITEBYTECODE": "1",
    "ROCR_VISIBLE_DEVICES": "0",
    "XLA_FLAGS": "--xla_gpu_enable_command_buffer=",
}
_MEMORY_MODE_ENVIRONMENT = {
    "growth": {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    },
    "preallocate85": {
        "GPU_DEVICE_ORDINAL": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "XLA_CLIENT_MEM_FRACTION": "0.85",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
    },
}
_ACCELERATOR_ENVIRONMENT_PREFIXES = (
    "AMD_",
    "GPU_",
    "HIP_",
    "HSA_",
    "JAX_",
    "PJRT_",
    "RCCL_",
    "ROCM_",
    "ROCR_",
    "XLA_",
)
_INTERPRETER_INJECTION_ENVIRONMENT = {
    "BASH_ENV",
    "ENV",
    "LD_AUDIT",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "__PYVENV_LAUNCHER__",
}


class RuntimeSourceError(RuntimeError):
    """The claimed runtime source or accelerator policy is not exact."""


def _canonical_directory(
    path: Path, label: str, *, exact_mode: int | None = None
) -> Path:
    if not path.is_absolute():
        raise RuntimeSourceError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeSourceError(f"cannot inspect {label}: {error}") from error
    if resolved != path or stat.S_ISLNK(metadata.st_mode):
        raise RuntimeSourceError(f"{label} must be canonical and non-symlinked")
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise RuntimeSourceError(f"{label} must be an owned directory")
    observed_mode = stat.S_IMODE(metadata.st_mode)
    if exact_mode is not None and observed_mode != exact_mode:
        raise RuntimeSourceError(
            f"{label} must have exact mode {exact_mode:04o}, observed {observed_mode:04o}"
        )
    return resolved


def _canonical_private_directory(path: Path, label: str) -> Path:
    return _canonical_directory(path, label, exact_mode=0o700)


def _canonical_private_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise RuntimeSourceError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeSourceError(f"cannot inspect {label}: {error}") from error
    if resolved != path or stat.S_ISLNK(metadata.st_mode):
        raise RuntimeSourceError(f"{label} must be canonical and non-symlinked")
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise RuntimeSourceError(
            f"{label} must be an owned, singly linked file with exact mode 0600"
        )
    return resolved


def _canonical_uv_executable(path: Path, expected: Path) -> Path:
    if path != expected or not path.is_absolute():
        raise RuntimeSourceError("runtime uv executable is not the fixed account path")
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeSourceError(f"cannot inspect runtime uv executable: {error}") from error
    if resolved != path or stat.S_ISLNK(metadata.st_mode):
        raise RuntimeSourceError(
            "runtime uv executable must be canonical and non-symlinked"
        )
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o755
        or not os.access(path, os.X_OK)
    ):
        raise RuntimeSourceError(
            "runtime uv executable must be an owned, singly linked 0755 executable"
        )
    _validate_uv_payload(path)
    return resolved


def _validate_uv_payload(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise RuntimeSourceError(f"cannot hash runtime uv executable: {error}") from error
    finally:
        if "descriptor" in locals():
            os.close(descriptor)
    fingerprint_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_uid,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    fingerprint_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_uid,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    observed_sha256 = digest.hexdigest()
    if (
        fingerprint_after != fingerprint_before
        or after.st_size != _UV_SIZE_BYTES
        or observed_sha256 != _UV_SHA256
    ):
        raise RuntimeSourceError(
            "runtime uv executable does not match qualified uv 0.11.8 payload"
        )
    return observed_sha256


def _canonical_head(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
        raise RuntimeSourceError("runtime Git HEAD must be one full lowercase object ID")
    return value


def validate_runtime_launch_lock(
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate the inherited whole-process ROCm launch-lock descriptor."""
    observed = os.environ if environment is None else environment
    raw_descriptor = observed.get(_LAUNCH_LOCK_FD_ENV)
    if raw_descriptor is None:
        return {"status": "not_required"}
    try:
        import fcntl
    except ImportError as error:  # pragma: no cover - hardened ROCm is Linux-only.
        raise RuntimeSourceError(
            "runtime launch-lock validation requires POSIX flock"
        ) from error
    if not re.fullmatch(r"[1-9][0-9]*", raw_descriptor):
        raise RuntimeSourceError("runtime launch-lock descriptor is malformed")
    descriptor = int(raw_descriptor)
    if descriptor < 3 or str(descriptor) != raw_descriptor:
        raise RuntimeSourceError("runtime launch-lock descriptor is not canonical")
    # Production always uses one per-UID namespace. An explicit mapping is a
    # dependency-injection seam for filesystem-isolated unit tests only.
    runtime_root = (
        Path("/run/user") / str(os.getuid())
        if environment is None
        else Path(observed.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    )
    runtime_root = _canonical_private_directory(
        runtime_root, "runtime launch-lock parent"
    )
    lock_path = _canonical_private_directory(
        runtime_root / f"skyrl-qwen35-rocm-{os.getuid()}",
        "runtime launch-lock directory",
    )
    try:
        descriptor_metadata = os.fstat(descriptor)
        descriptor_path = Path(f"/proc/self/fd/{descriptor}").resolve(strict=True)
    except OSError as error:
        raise RuntimeSourceError(
            f"cannot inspect inherited runtime launch lock: {error}"
        ) from error
    lock_metadata = lock_path.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(descriptor_metadata.st_mode)
        or descriptor_metadata.st_uid != os.getuid()
        or stat.S_IMODE(descriptor_metadata.st_mode) != 0o700
        or descriptor_path != lock_path
        or (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
        != (lock_metadata.st_dev, lock_metadata.st_ino)
        or not os.get_inheritable(descriptor)
    ):
        raise RuntimeSourceError(
            "runtime launch-lock descriptor is not the inherited private lock directory"
        )
    probe = None
    try:
        probe = os.open(
            lock_path,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise RuntimeSourceError("runtime launch-lock descriptor is not locked")
    finally:
        if probe is not None:
            os.close(probe)
    return {
        "status": "passed",
        "descriptor": descriptor,
        "path": str(lock_path),
        "inheritable": True,
        "exclusive_lock_observed": True,
    }


def revalidate_runtime_launch_lock(
    initial: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the inherited launch lock to remain unchanged during startup."""
    current = validate_runtime_launch_lock()
    if current != initial:
        raise RuntimeSourceError(
            "runtime launch-lock attestation changed after module import: "
            f"initial={dict(initial)!r}, current={current!r}"
        )
    return current


def _validate_full_source_cache(
    *,
    repo_root: Path,
    head: str,
    account_home: Path,
    source_root: Path,
    jax_cache: Path,
) -> dict[str, Any]:
    """Bind the runtime claim to Git, the archive, and every snapshot node."""
    try:
        from rocm import prepare_jax_cache_dir as cache_policy
        from rocm import verified_source_bootstrap as bootstrap
    except Exception as error:
        raise RuntimeSourceError(
            f"cannot import full source-cache validators: {error}"
        ) from error

    bootstrap_origin = _canonical_private_file(
        Path(bootstrap.__file__), "runtime source validator"
    )
    cache_policy_origin = _canonical_private_file(
        Path(cache_policy.__file__), "runtime JAX-cache policy"
    )
    if bootstrap_origin != source_root / "rocm/verified_source_bootstrap.py":
        raise RuntimeSourceError(
            "full source-cache validator did not resolve from the source snapshot"
        )
    if cache_policy_origin != source_root / "rocm/prepare_jax_cache_dir.py":
        raise RuntimeSourceError(
            "JAX-cache policy did not resolve from the source snapshot"
        )
    try:
        result = bootstrap.validate_source_cache(
            repo_root=repo_root,
            git_head=head,
            account_home=account_home,
        )
    except Exception as error:
        raise RuntimeSourceError(
            f"full runtime source-cache validation failed: {error}"
        ) from error
    if result.get("source_snapshot_root") != str(source_root):
        raise RuntimeSourceError(
            "full source-cache validation selected a different snapshot"
        )
    expected_jax_cache = (
        account_home
        / ".cache"
        / cache_policy._CACHE_BASE
        / cache_policy._CACHE_NAMESPACE
    )
    if jax_cache != expected_jax_cache:
        raise RuntimeSourceError(
            "runtime JAX compilation cache is not the exact stack namespace"
        )
    try:
        validated_jax_cache = cache_policy.validate_existing_cache(
            jax_cache, 4_294_967_296
        )
    except Exception as error:
        raise RuntimeSourceError(
            f"runtime JAX-cache content validation failed: {error}"
        ) from error
    if validated_jax_cache != expected_jax_cache:
        raise RuntimeSourceError(
            "JAX-cache policy selected a different stack namespace"
        )
    return {**result, "expected_jax_cache": str(expected_jax_cache)}


def _validate_startup_cache_claim(
    *,
    claims: Mapping[str, str],
    source_root: Path,
    git_head: str,
    git_tree: str,
    jax_cache: Path,
    attention_backend: str,
    memory_mode: str,
) -> dict[str, object]:
    """Validate both private prewarm artifacts without importing JAX."""
    try:
        from rocm import qwen35_cache_attestation as cache_attestation
    except Exception as error:
        raise RuntimeSourceError(
            f"cannot import startup cache-attestation validator: {error}"
        ) from error
    helper_origin = _canonical_private_file(
        Path(cache_attestation.__file__), "startup cache-attestation validator"
    )
    if helper_origin != source_root / "rocm/qwen35_cache_attestation.py":
        raise RuntimeSourceError(
            "startup cache-attestation validator did not resolve from the source snapshot"
        )
    expected_attention = "pallas" if attention_backend == "1" else "xla"
    try:
        result = cache_attestation.build_startup_cache_claim(
            prewarm_path=Path(claims[_PREWARM_AUDIT_PATH_ENV]),
            prewarm_sha256=claims[_PREWARM_AUDIT_SHA256_ENV],
            handoff_path=Path(claims[_PREWARM_HANDOFF_PATH_ENV]),
            handoff_sha256=claims[_PREWARM_HANDOFF_SHA256_ENV],
            expected_git_head=git_head,
            expected_git_tree=git_tree,
            expected_cache_path=str(jax_cache),
            expected_attention_backend=expected_attention,
        )
    except Exception as error:
        raise RuntimeSourceError(
            f"startup T64 cache-attestation artifacts are invalid: {error}"
        ) from error
    if result.get("status") != "required-v1":
        raise RuntimeSourceError(
            "startup T64 cache-attestation validator returned an invalid status"
        )
    seed = result.get("seed")
    expected_construction = (
        "abstract-load" if memory_mode == "preallocate85" else "eager"
    )
    if not isinstance(seed, dict) or seed.get("construction") != (
        expected_construction
    ):
        raise RuntimeSourceError(
            "startup T64 cache-attestation construction does not match memory mode"
        )
    return result


def validate_runtime_source(
    *,
    role: str,
    module_file: Path | str,
    package_file: Path | str,
    environment: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    dont_write_bytecode: bool | None = None,
) -> dict[str, Any]:
    """Validate one API/engine origin before backend construction."""
    if role not in _ROLE_PATHS:
        raise RuntimeSourceError(f"unknown runtime-source role: {role!r}")
    using_process_environment = environment is None
    observed = os.environ if using_process_environment else environment
    claim_names = (
        _SOURCE_ROOT_ENV,
        _GIT_HEAD_ENV,
        _REPO_ROOT_ENV,
        _UV_EXECUTABLE_ENV,
        _MEMORY_MODE_ENV,
    )
    claims = {name: observed.get(name) for name in claim_names}
    cache_claims = {name: observed.get(name) for name in _CACHE_ATTESTATION_ENVIRONMENT}
    cache_claim_present = any(value is not None for value in cache_claims.values())
    if cache_claim_present:
        missing_cache_claims = [
            name for name, value in cache_claims.items() if not value
        ]
        if missing_cache_claims:
            raise RuntimeSourceError(
                "runtime T64 cache-attestation claims are incomplete: "
                f"{missing_cache_claims!r}"
            )
        if cache_claims[_T64_CACHE_ATTEST_ENV] != "required-v1":
            raise RuntimeSourceError(
                "runtime T64 cache-attestation mode must be required-v1"
            )
    if all(value is None for value in claims.values()):
        if cache_claim_present:
            raise RuntimeSourceError(
                "runtime T64 cache attestation requires full hardened source claims"
            )
        return {"status": "not_required", "role": role}
    if any(not value for value in claims.values()):
        raise RuntimeSourceError("runtime source claims are incomplete")
    launch_lock = validate_runtime_launch_lock(
        None if using_process_environment else observed
    )
    if launch_lock.get("status") != "passed":
        raise RuntimeSourceError(
            "claimed runtime source requires the inherited ROCm launch lock"
        )

    bytecode_disabled = (
        bool(sys.flags.dont_write_bytecode)
        if dont_write_bytecode is None
        else dont_write_bytecode
    )
    memory_mode = claims[_MEMORY_MODE_ENV]
    assert memory_mode is not None
    if memory_mode not in _MEMORY_MODE_ENVIRONMENT:
        raise RuntimeSourceError("runtime memory mode must be growth or preallocate85")
    expected_environment = {
        **_REQUIRED_ENVIRONMENT,
        **_MEMORY_MODE_ENVIRONMENT[memory_mode],
    }
    mismatches = {
        name: {"expected": expected, "observed": observed.get(name)}
        for name, expected in expected_environment.items()
        if observed.get(name) != expected
    }
    if not bytecode_disabled:
        mismatches["sys.flags.dont_write_bytecode"] = {
            "expected": True,
            "observed": False,
        }
    allowed_accelerator_names = {
        *expected_environment,
        "JAX_COMPILATION_CACHE_DIR",
    }
    unexpected_accelerator_names = sorted(
        name
        for name in observed
        if name.startswith(_ACCELERATOR_ENVIRONMENT_PREFIXES)
        and name not in allowed_accelerator_names
    )
    if unexpected_accelerator_names:
        mismatches["unexpected_accelerator_environment"] = {
            "expected": [],
            "observed": unexpected_accelerator_names,
        }
    inherited_injection = sorted(
        name
        for name in _INTERPRETER_INJECTION_ENVIRONMENT
        if observed.get(name)
    )
    if inherited_injection:
        mismatches["interpreter_injection_environment"] = {
            "expected": [],
            "observed": inherited_injection,
        }
    if mismatches:
        raise RuntimeSourceError(
            f"runtime accelerator/source policy mismatch: {mismatches!r}"
        )

    head = _canonical_head(claims[_GIT_HEAD_ENV])
    root = _canonical_private_directory(
        Path(claims[_SOURCE_ROOT_ENV]), "runtime source root"
    )
    commit_root = _canonical_private_directory(
        root.parent, "runtime source commit root"
    )
    cache_root = _canonical_private_directory(
        commit_root.parent, "runtime source cache root"
    )
    account_cache = _canonical_private_directory(
        cache_root.parent, "private account cache"
    )
    account_home = _canonical_directory(
        account_cache.parent, "runtime account home"
    )
    if (
        root.name != "source-head"
        or commit_root.name != head
        or cache_root.name != _SOURCE_CACHE_NAMESPACE
        or account_cache.name != ".cache"
    ):
        raise RuntimeSourceError("runtime source root does not match its commit key")

    repo_root = _canonical_directory(
        Path(claims[_REPO_ROOT_ENV]), "runtime original repository"
    )
    current_directory = Path.cwd() if cwd is None else Path(cwd)
    try:
        current_directory = current_directory.resolve(strict=True)
    except OSError as error:
        raise RuntimeSourceError(
            f"cannot resolve runtime working directory: {error}"
        ) from error
    if current_directory != root:
        raise RuntimeSourceError(
            "runtime working directory is not the source snapshot"
        )

    module = _canonical_private_file(Path(module_file), f"{role} module source")
    package = _canonical_private_file(Path(package_file), "skyrl package source")
    if module != root / _ROLE_PATHS[role]:
        raise RuntimeSourceError(
            f"{role} module did not resolve from the source snapshot"
        )
    if package != root / "skyrl/__init__.py":
        raise RuntimeSourceError(
            "skyrl package did not resolve from the source snapshot"
        )
    if (root / ".venv").exists() or (root / ".venv").is_symlink():
        raise RuntimeSourceError(
            "runtime source snapshot must not contain a local .venv"
        )

    expected_venv = repo_root / ".venv"
    venv_bin = f"{expected_venv}/bin"
    uv_depth = "1" if role == "api" else "2"
    expected_path = ":".join(
        [venv_bin] * int(uv_depth) + ["/opt/rocm/bin", "/usr/bin", "/bin"]
    )
    uv_executable = _canonical_uv_executable(
        Path(claims[_UV_EXECUTABLE_ENV]), account_home / ".local/bin/uv"
    )
    process_mismatches = {
        name: {"expected": expected, "observed": observed.get(name)}
        for name, expected in {
            "HOME": str(account_home),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": expected_path,
            "UV": str(uv_executable),
            "UV_RUN_RECURSION_DEPTH": uv_depth,
            "VIRTUAL_ENV": str(expected_venv),
        }.items()
        if observed.get(name) != expected
    }
    if process_mismatches:
        raise RuntimeSourceError(
            f"runtime process environment mismatch: {process_mismatches!r}"
        )
    unexpected_uv_names = sorted(
        name
        for name in observed
        if (name == "UV" or name.startswith("UV_"))
        and name not in {"UV", "UV_RUN_RECURSION_DEPTH"}
    )
    if unexpected_uv_names:
        raise RuntimeSourceError(
            "runtime uv environment contains unexpected names: "
            f"{unexpected_uv_names!r}"
        )

    raw_jax_cache = observed.get("JAX_COMPILATION_CACHE_DIR")
    if not raw_jax_cache:
        raise RuntimeSourceError(
            "runtime JAX compilation cache claim is absent"
        )
    jax_cache = _canonical_private_directory(
        Path(raw_jax_cache), "runtime JAX compilation cache"
    )
    full_source = _validate_full_source_cache(
        repo_root=repo_root,
        head=head,
        account_home=account_home,
        source_root=root,
        jax_cache=jax_cache,
    )
    if str(jax_cache) != full_source.get("expected_jax_cache"):
        raise RuntimeSourceError(
            "runtime JAX compilation cache is not the exact stack namespace"
        )
    _canonical_private_directory(jax_cache.parent, "runtime JAX cache base")
    attention_backend = observed.get("SKYRL_ROCM_PALLAS_ATTENTION")
    if attention_backend not in {"0", "1"}:
        raise RuntimeSourceError(
            "SKYRL_ROCM_PALLAS_ATTENTION must be exactly 0 or 1 at runtime"
        )
    startup_cache_attestation: dict[str, object]
    if cache_claim_present:
        startup_cache_attestation = _validate_startup_cache_claim(
            claims={name: str(value) for name, value in cache_claims.items()},
            source_root=root,
            git_head=head,
            git_tree=str(full_source["git_tree"]),
            jax_cache=jax_cache,
            attention_backend=attention_backend,
            memory_mode=memory_mode,
        )
    else:
        startup_cache_attestation = {"status": "not_required"}

    return {
        "status": "passed",
        "role": role,
        "git_head": head,
        "git_tree": full_source["git_tree"],
        "source_archive_path": full_source["source_archive_path"],
        "source_archive_sha256": full_source["source_archive_sha256"],
        "source_file_count": full_source["source_file_count"],
        "source_total_bytes": full_source["source_total_bytes"],
        "full_head_tree_validated": full_source["full_head_tree_validated"],
        "source_root": str(root),
        "repo_root": str(repo_root),
        "working_directory": str(current_directory),
        "module_origin": str(module),
        "package_origin": str(package),
        "uv_executable": str(uv_executable),
        "uv_sha256": _UV_SHA256,
        "launch_lock": launch_lock,
        "jax_compilation_cache": str(jax_cache),
        "memory_mode": memory_mode,
        "xla_flags": observed["XLA_FLAGS"],
        "jax_enable_pgle": observed["JAX_ENABLE_PGLE"],
        "jax_compilation_cache_expect_pgle": observed[
            "JAX_COMPILATION_CACHE_EXPECT_PGLE"
        ],
        "pallas_attention": attention_backend,
        "startup_cache_attestation": startup_cache_attestation,
        "dont_write_bytecode": True,
    }


def revalidate_runtime_source(
    *,
    initial: Mapping[str, Any],
    role: str,
    module_file: Path | str,
    package_file: Path | str,
) -> dict[str, Any]:
    """Require source and policy claims to remain identical during startup."""
    current = validate_runtime_source(
        role=role,
        module_file=module_file,
        package_file=package_file,
    )
    if current != initial:
        raise RuntimeSourceError(
            "runtime source attestation changed after module import: "
            f"initial={dict(initial)!r}, current={current!r}"
        )
    return current
