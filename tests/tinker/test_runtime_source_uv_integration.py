"""CPU-only integration coverage for snapshot-bound ``uv run`` imports.

The probes deliberately use ``find_spec`` instead of importing the API or
engine modules.  This exercises Python's source selection while keeping JAX
and every backend constructor out of the process.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]

_ORIGIN_PROBE = r"""
import importlib.util
import json
import pathlib
import sys

def origin(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"no source origin for {module_name}")
    return str(pathlib.Path(spec.origin).resolve())

print(json.dumps({
    "cwd": str(pathlib.Path.cwd().resolve()),
    "package_origin": origin("skyrl"),
    "api_origin": origin("skyrl.tinker.api"),
    "engine_origin": origin("skyrl.tinker.engine"),
    "api_imported": "skyrl.tinker.api" in sys.modules,
    "engine_imported": "skyrl.tinker.engine" in sys.modules,
    "jax_imported": any(name == "jax" or name.startswith("jax.") for name in sys.modules),
    "dont_write_bytecode": sys.flags.dont_write_bytecode,
}))
"""

_NESTED_ENGINE_PROBE = r"""
import importlib.util
import json
import pathlib
import sys

spec = importlib.util.find_spec("skyrl.tinker.engine")
if spec is None or spec.origin is None:
    raise RuntimeError("no source origin for skyrl.tinker.engine")

print(json.dumps({
    "cwd": str(pathlib.Path.cwd().resolve()),
    "engine_origin": str(pathlib.Path(spec.origin).resolve()),
    "engine_imported": "skyrl.tinker.engine" in sys.modules,
    "jax_imported": any(name == "jax" or name.startswith("jax.") for name in sys.modules),
    "dont_write_bytecode": sys.flags.dont_write_bytecode,
}))
"""


def _api_probe_source() -> str:
    return f"""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

def origin(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"no source origin for {{module_name}}")
    return str(pathlib.Path(spec.origin).resolve())

snapshot = os.environ["SKYRL_TEST_SOURCE_SNAPSHOT"]
nested_command = [
    os.environ["SKYRL_TEST_UV_EXECUTABLE"],
    "run",
    "--active",
    "--no-sync",
    "--no-env-file",
    "--no-config",
    "--directory",
    snapshot,
    "--project",
    snapshot,
    "python",
    "-c",
    {_NESTED_ENGINE_PROBE!r},
]
nested = subprocess.run(
    nested_command,
    check=True,
    capture_output=True,
    env=os.environ.copy(),
    text=True,
)

print(json.dumps({{
    "cwd": str(pathlib.Path.cwd().resolve()),
    "package_origin": origin("skyrl"),
    "api_origin": origin("skyrl.tinker.api"),
    "api_imported": "skyrl.tinker.api" in sys.modules,
    "jax_imported": any(name == "jax" or name.startswith("jax.") for name in sys.modules),
    "dont_write_bytecode": sys.flags.dont_write_bytecode,
    "nested": json.loads(nested.stdout),
}}))
"""


def _run_json(command: list[str], *, cwd: Path, environment: dict[str, str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"command failed with status {result.returncode}: {command!r}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return json.loads(result.stdout)


def _extract_git_snapshot(snapshot: Path, archive: Path, git_head: str) -> None:
    snapshot.mkdir(parents=True, mode=0o700)
    subprocess.run(
        [
            "git",
            "-C",
            str(_REPO_ROOT),
            "archive",
            "--format=tar",
            f"--output={archive}",
            git_head,
        ],
        check=True,
        timeout=30,
    )
    with tarfile.open(archive, mode="r:") as source_archive:
        source_archive.extractall(snapshot, filter="data")


def _tree_manifest(root: Path) -> tuple[tuple[str, str, int, int, str], ...]:
    manifest: list[tuple[str, str, int, int, str]] = []
    for path in sorted(root.rglob("*")):
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(metadata.st_mode):
            kind = "symlink"
            payload = os.readlink(path)
        elif stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
            payload = ""
        elif stat.S_ISREG(metadata.st_mode):
            kind = "file"
            payload = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            kind = "other"
            payload = ""
        manifest.append((relative, kind, mode, metadata.st_size, payload))
    return tuple(manifest)


def _write_hostile_project(root: Path) -> None:
    (root / "skyrl/tinker").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'hostile-source-redirect'\nversion = '0.0.0'\n",
        encoding="utf-8",
    )
    (root / "skyrl/__init__.py").write_text("HOSTILE = True\n", encoding="utf-8")
    (root / "skyrl/tinker/__init__.py").write_text("", encoding="utf-8")
    (root / "skyrl/tinker/api.py").write_text("HOSTILE = True\n", encoding="utf-8")
    (root / "skyrl/tinker/engine.py").write_text("HOSTILE = True\n", encoding="utf-8")


def test_uv_api_and_nested_engine_resolve_from_disposable_commit_snapshot(
    tmp_path: Path,
) -> None:
    """Explicit source flags beat both editable mappings and hostile UV claims."""
    uv_executable = shutil.which("uv")
    assert uv_executable is not None, "the runtime-source integration test requires uv"
    assert sys.prefix != sys.base_prefix, "the integration test requires an active virtualenv"

    git_head = subprocess.check_output(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    account_cache = tmp_path / "account-cache"
    snapshot = (
        account_cache
        / "skyrl-source-snapshots-private-v1"
        / git_head
        / "source-head"
    )
    archive = tmp_path / "source-head.tar"
    hostile_project = tmp_path / "hostile-project"
    neutral_directory = tmp_path / "neutral-cwd"
    uv_cache = tmp_path / "uv-cache"
    neutral_directory.mkdir()
    uv_cache.mkdir()
    _write_hostile_project(hostile_project)
    _extract_git_snapshot(snapshot, archive, git_head)

    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("PYTHON") and not name.startswith("UV_")
    }
    environment.update(
        {
            "PATH": f"{Path(sys.prefix) / 'bin'}:{environment.get('PATH', '')}",
            "PYTHONDONTWRITEBYTECODE": "1",
            "SKYRL_TEST_SOURCE_SNAPSHOT": str(snapshot),
            "SKYRL_TEST_UV_EXECUTABLE": uv_executable,
            "UV_CACHE_DIR": str(uv_cache),
            "UV_PROJECT": str(hostile_project),
            "UV_WORKING_DIR": str(hostile_project),
            "VIRTUAL_ENV": sys.prefix,
        }
    )

    # Outside the checkout, the active environment still maps SkyRL back to
    # the mutable worktree.  This is the editable-install condition the
    # commit snapshot must override.
    editable = _run_json(
        [sys.executable, "-c", _ORIGIN_PROBE],
        cwd=neutral_directory,
        environment=environment,
    )
    assert editable["package_origin"] == str((_REPO_ROOT / "skyrl/__init__.py").resolve())
    assert editable["api_origin"] == str((_REPO_ROOT / "skyrl/tinker/api.py").resolve())
    assert editable["engine_origin"] == str((_REPO_ROOT / "skyrl/tinker/engine.py").resolve())
    assert editable["api_imported"] is False
    assert editable["engine_imported"] is False
    assert editable["jax_imported"] is False
    assert editable["dont_write_bytecode"] == 1

    before = _tree_manifest(snapshot)
    result = _run_json(
        [
            uv_executable,
            "run",
            "--active",
            "--no-sync",
            "--no-env-file",
            "--no-config",
            "--directory",
            str(snapshot),
            "--project",
            str(snapshot),
            "python",
            "-c",
            _api_probe_source(),
        ],
        cwd=_REPO_ROOT,
        environment=environment,
    )
    after = _tree_manifest(snapshot)

    assert result == {
        "cwd": str(snapshot),
        "package_origin": str(snapshot / "skyrl/__init__.py"),
        "api_origin": str(snapshot / "skyrl/tinker/api.py"),
        "api_imported": False,
        "jax_imported": False,
        "dont_write_bytecode": 1,
        "nested": {
            "cwd": str(snapshot),
            "engine_origin": str(snapshot / "skyrl/tinker/engine.py"),
            "engine_imported": False,
            "jax_imported": False,
            "dont_write_bytecode": 1,
        },
    }
    assert before == after
    assert not (snapshot / ".venv").exists()
    assert not tuple(snapshot.rglob("__pycache__"))
    assert not tuple(snapshot.rglob("*.pyc"))
