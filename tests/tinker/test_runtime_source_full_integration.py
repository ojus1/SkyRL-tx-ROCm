"""CPU-only end-to-end coverage for full runtime source attestation."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
from pathlib import Path

from rocm import prepare_jax_cache_dir, verified_source_bootstrap

_ROOT = Path(__file__).resolve().parents[2]
_TRACKED_FIXTURE_PATHS = (
    ".gitignore",
    "rocm/__init__.py",
    "rocm/prepare_jax_cache_dir.py",
    "rocm/start_qwen35.sh",
    "rocm/verified_source_bootstrap.py",
    "skyrl/__init__.py",
    "skyrl/tinker/__init__.py",
    "skyrl/tinker/api.py",
    "skyrl/tinker/engine.py",
    "skyrl/tinker/runtime_source.py",
)
_VALIDATION_PROBE = r"""
import json
import pathlib
import sys

snapshot = pathlib.Path(sys.argv[1])
environment = json.loads(sys.argv[2])
sys.path.insert(0, str(snapshot))
from skyrl.tinker.runtime_source import validate_runtime_source

result = validate_runtime_source(
    role="api",
    module_file=snapshot / "skyrl/tinker/api.py",
    package_file=snapshot / "skyrl/__init__.py",
    environment=environment,
    cwd=snapshot,
    dont_write_bytecode=True,
)
print(json.dumps(result, sort_keys=True))
"""


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _make_runtime_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir(mode=0o700)
    for relative in _TRACKED_FIXTURE_PATHS:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative == ".gitignore":
            destination.write_text(".venv/\n", encoding="utf-8")
        else:
            shutil.copyfile(_ROOT / relative, destination)
        source_mode = (_ROOT / relative).stat().st_mode if relative != ".gitignore" else 0
        destination.chmod(0o755 if source_mode & 0o111 else 0o644)
    _git(repo, "init", "-q")
    _git(repo, "add", "--all")
    _git(
        repo,
        "-c",
        "user.name=Runtime Source Integration",
        "-c",
        "user.email=runtime-source@example.invalid",
        "commit",
        "-qm",
        "runtime source fixture",
    )
    return repo.resolve(), _git(repo, "rev-parse", "HEAD")


def _run_probe(
    snapshot: Path, environment: dict[str, str], lock_fd: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "/usr/bin/python3.12",
            "-I",
            "-S",
            "-B",
            "-P",
            "-c",
            _VALIDATION_PROBE,
            str(snapshot),
            json.dumps(environment, sort_keys=True),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
        pass_fds=(lock_fd,),
    )


def test_runtime_attestation_revalidates_archive_git_and_full_snapshot_tree(
    tmp_path: Path,
) -> None:
    repo, head = _make_runtime_repo(tmp_path)
    (repo / ".venv/bin").mkdir(parents=True, mode=0o700)
    account_home = tmp_path / "home"
    account_home.mkdir(mode=0o700)
    prepared = verified_source_bootstrap.prepare_source_cache(
        repo_root=repo,
        git_head=head,
        account_home=account_home,
    )
    snapshot = Path(prepared["source_snapshot_root"])
    uv_executable = account_home / ".local/bin/uv"
    uv_executable.parent.mkdir(parents=True, mode=0o700)
    qualified_uv = shutil.which("uv")
    assert qualified_uv is not None
    shutil.copyfile(qualified_uv, uv_executable)
    uv_executable.chmod(0o755)
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(mode=0o700)
    lock_path = runtime_root / f"skyrl-qwen35-rocm-{os.getuid()}"
    lock_path.mkdir(mode=0o700)
    lock_fd = os.open(lock_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jax_cache = (
        account_home
        / ".cache"
        / prepare_jax_cache_dir._CACHE_BASE
        / prepare_jax_cache_dir._CACHE_NAMESPACE
    )
    jax_cache.mkdir(parents=True, mode=0o700)
    jax_cache.parent.chmod(0o700)
    environment = {
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HOME": str(account_home),
        "JAX_COMPILATION_CACHE_DIR": str(jax_cache),
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
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LLVM_PATH": "/opt/rocm/llvm",
        "PATH": f"{repo}/.venv/bin:/opt/rocm/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "ROCR_VISIBLE_DEVICES": "0",
        "SKYRL_QWEN35_LAUNCH_LOCK_FD": str(lock_fd),
        "SKYRL_QWEN35_RUNTIME_GIT_HEAD": head,
        "SKYRL_QWEN35_RUNTIME_MEMORY_MODE": "growth",
        "SKYRL_QWEN35_RUNTIME_REPO_ROOT": str(repo),
        "SKYRL_QWEN35_RUNTIME_SOURCE_ROOT": str(snapshot),
        "SKYRL_QWEN35_RUNTIME_UV_EXECUTABLE": str(uv_executable),
        "SKYRL_ROCM_PALLAS_ATTENTION": "1",
        "UV": str(uv_executable),
        "UV_RUN_RECURSION_DEPTH": "1",
        "VIRTUAL_ENV": str(repo / ".venv"),
        "XDG_RUNTIME_DIR": str(runtime_root),
        "XLA_FLAGS": "--xla_gpu_enable_command_buffer=",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }

    try:
        passed = _run_probe(snapshot, environment, lock_fd)

        assert passed.returncode == 0, passed.stderr
        attestation = json.loads(passed.stdout)
        assert attestation["status"] == "passed"
        assert attestation["launch_lock"]["exclusive_lock_observed"] is True
        assert attestation["full_head_tree_validated"] is True
        assert attestation["git_head"] == head
        assert attestation["git_tree"] == _git(repo, "rev-parse", "HEAD^{tree}")
        assert attestation["source_archive_sha256"] == prepared[
            "source_archive_sha256"
        ]
        assert attestation["source_file_count"] == len(_TRACKED_FIXTURE_PATHS)

        shadow_module = snapshot / "fastapi.py"
        shadow_module.write_text("raise RuntimeError('shadowed')\n", encoding="utf-8")
        shadow_module.chmod(0o600)
        refused = _run_probe(snapshot, environment, lock_fd)

        assert refused.returncode != 0
        assert "snapshot file layout differs from HEAD" in refused.stderr
        assert not tuple(snapshot.rglob("__pycache__"))
        assert not tuple(snapshot.rglob("*.pyc"))
    finally:
        os.close(lock_fd)
