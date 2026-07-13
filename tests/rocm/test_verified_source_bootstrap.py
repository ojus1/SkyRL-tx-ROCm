from __future__ import annotations

import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _ROOT / "rocm" / "verified_source_bootstrap.py"
_SPEC = importlib.util.spec_from_file_location("verified_source_bootstrap", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
bootstrap = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bootstrap
_SPEC.loader.exec_module(bootstrap)


def _git(repo: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), *arguments],
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    return result.stdout


def _write(path: Path, data: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(mode)


def _make_repo(
    tmp_path: Path,
    *,
    target_source: bytes = b"VALUE = 'verified-target'\n",
    extra_files: dict[str, tuple[bytes, int]] | None = None,
) -> tuple[Path, str]:
    repo = tmp_path / "original"
    repo.mkdir(mode=0o700)
    _git(repo, "init", "-q")
    files = {
        "rocm/__init__.py": (b'"""Verified test package."""\n', 0o644),
        "rocm/start_qwen35.sh": (b"#!/usr/bin/env bash\nexit 0\n", 0o755),
        "rocm/verified_source_bootstrap.py": (_TOOL.read_bytes(), 0o644),
        "rocm/amdgpu_safety.py": (target_source, 0o644),
        "nested/data.bin": (b"\x00tracked-data\xff\n", 0o644),
        "tools/tracked-helper": (b"#!/bin/sh\nexit 0\n", 0o755),
    }
    if extra_files:
        files.update(extra_files)
    for relative, (data, mode) in files.items():
        _write(repo / relative, data, mode)
    _git(repo, "add", "--all")
    _git(
        repo,
        "-c",
        "user.name=Verified Source Test",
        "-c",
        "user.email=verified-source@example.invalid",
        "commit",
        "-qm",
        "verified source fixture",
    )
    head = _git(repo, "rev-parse", "HEAD").decode("ascii").strip()
    return repo, head


def _make_snapshot(repo: Path, destination: Path) -> Path:
    archive = _git(repo, "archive", "--format=tar", "HEAD")
    destination.mkdir(mode=0o700)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
        source.extractall(destination, filter="data")
    for current_root, directories, files in os.walk(destination):
        root = Path(current_root)
        root.chmod(0o700)
        for directory in directories:
            (root / directory).chmod(0o700)
        for name in files:
            path = root / name
            source_mode = stat.S_IMODE(path.stat().st_mode)
            path.chmod(0o700 if source_mode & 0o111 else 0o600)
    return destination


def _make_venv_site(tmp_path: Path) -> Path:
    venv = tmp_path / "test-venv"
    venv.mkdir(mode=0o700)
    _write(venv / "pyvenv.cfg", b"home = /usr/bin\n", 0o600)
    site_packages = (
        venv
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    site_packages.mkdir(parents=True)
    return site_packages.resolve()


def _validate(
    repo: Path,
    head: str,
    snapshot: Path,
    site_packages: Path,
    *,
    target_module: str = "rocm.amdgpu_safety",
) -> dict[str, object]:
    return bootstrap.validate_snapshot(
        repo_root=repo,
        git_head=head,
        snapshot_root=snapshot,
        venv_site_packages=site_packages,
        target_module=target_module,
        require_runtime_policy=False,
    )


def test_full_tree_snapshot_manifest_is_exact_and_deterministic(tmp_path: Path) -> None:
    repo, head = _make_repo(tmp_path)
    snapshot = _make_snapshot(repo, tmp_path / "snapshot")
    site_packages = _make_venv_site(tmp_path)

    first = _validate(repo, head, snapshot, site_packages)
    second = _validate(repo, head, snapshot, site_packages)

    assert first == second
    assert first["status"] == "passed"
    assert first["format"] == "skyrl-verified-source-v1"
    assert first["git_head"] == head
    assert len(first["git_tree"]) == 40
    assert first["git_object_format"] == "sha1"
    assert first["file_count"] == 6
    assert first["total_source_bytes"] == sum(
        record["size_bytes"] for record in first["files"]
    )
    assert len(first["source_manifest_sha256"]) == 64
    assert [record["path"] for record in first["files"]] == sorted(
        record["path"] for record in first["files"]
    )
    assert {record["snapshot_mode"] for record in first["files"]} == {
        "0600",
        "0700",
    }
    assert first["threat_model_excludes"] == [
        "malicious process running as the same UID",
        "parent process or pre-Python dynamic-loader environment",
        "root, kernel, privileged OS, or compromised Git/Python binary",
    ]


def test_target_module_is_a_fixed_allowlist(tmp_path: Path) -> None:
    repo, head = _make_repo(tmp_path)
    snapshot = _make_snapshot(repo, tmp_path / "snapshot")
    site_packages = _make_venv_site(tmp_path)

    with pytest.raises(bootstrap.SourceVerificationError, match="not allowlisted"):
        _validate(
            repo,
            head,
            snapshot,
            site_packages,
            target_module="rocm.arbitrary_probe",
        )


def test_allowlisted_target_must_exist_in_head(tmp_path: Path) -> None:
    repo, head = _make_repo(tmp_path)
    snapshot = _make_snapshot(repo, tmp_path / "snapshot")
    site_packages = _make_venv_site(tmp_path)

    with pytest.raises(bootstrap.SourceVerificationError, match="target is absent"):
        _validate(
            repo,
            head,
            snapshot,
            site_packages,
            target_module="rocm.profile_rocm",
        )


def test_dirty_original_worktree_is_rejected(tmp_path: Path) -> None:
    repo, head = _make_repo(tmp_path)
    snapshot = _make_snapshot(repo, tmp_path / "snapshot")
    site_packages = _make_venv_site(tmp_path)
    (repo / "rocm" / "amdgpu_safety.py").write_text("dirty = True\n")

    with pytest.raises(bootstrap.SourceVerificationError, match="not exactly clean"):
        _validate(repo, head, snapshot, site_packages)


def test_changed_head_claim_is_rejected(tmp_path: Path) -> None:
    repo, head = _make_repo(tmp_path)
    snapshot = _make_snapshot(repo, tmp_path / "snapshot")
    site_packages = _make_venv_site(tmp_path)

    with pytest.raises(bootstrap.SourceVerificationError, match="HEAD changed"):
        _validate(repo, "0" * len(head), snapshot, site_packages)


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_every_hidden_index_flag_is_rejected_including_launcher(
    tmp_path: Path, flag: str
) -> None:
    repo, head = _make_repo(tmp_path)
    snapshot = _make_snapshot(repo, tmp_path / "snapshot")
    site_packages = _make_venv_site(tmp_path)
    _git(repo, "update-index", flag, "--", "rocm/start_qwen35.sh")
    assert _git(repo, "status", "--porcelain") == b""

    with pytest.raises(
        bootstrap.SourceVerificationError,
        match="assume-unchanged, skip-worktree",
    ):
        _validate(repo, head, snapshot, site_packages)


def test_git_replace_refs_cannot_change_claimed_head_tree(tmp_path: Path) -> None:
    repo, first_head = _make_repo(tmp_path)
    first_snapshot = _make_snapshot(repo, tmp_path / "snapshot")
    site_packages = _make_venv_site(tmp_path)
    target = repo / "rocm" / "amdgpu_safety.py"
    target.write_text("VALUE = 'replacement-tree'\n")
    _git(repo, "add", "--", "rocm/amdgpu_safety.py")
    _git(
        repo,
        "-c",
        "user.name=Verified Source Test",
        "-c",
        "user.email=verified-source@example.invalid",
        "commit",
        "-qm",
        "replacement tree",
    )
    replacement_head = _git(repo, "rev-parse", "HEAD").decode().strip()
    _git(repo, "checkout", "-q", "--detach", first_head)
    _git(repo, "replace", first_head, replacement_head)

    manifest = _validate(repo, first_head, first_snapshot, site_packages)

    target_record = next(
        record
        for record in manifest["files"]
        if record["path"] == "rocm/amdgpu_safety.py"
    )
    assert target_record["sha256"] == __import__("hashlib").sha256(
        b"VALUE = 'verified-target'\n"
    ).hexdigest()
    assert bootstrap._GIT_ENVIRONMENT["GIT_NO_REPLACE_OBJECTS"] == "1"


def test_tracked_git_symlink_is_rejected_before_snapshot_use(tmp_path: Path) -> None:
    repo, _head = _make_repo(tmp_path)
    link = repo / "tracked-link"
    link.symlink_to("nested/data.bin")
    _git(repo, "add", "--", "tracked-link")
    _git(
        repo,
        "-c",
        "user.name=Verified Source Test",
        "-c",
        "user.email=verified-source@example.invalid",
        "commit",
        "-qm",
        "tracked symlink",
    )
    head = _git(repo, "rev-parse", "HEAD").decode().strip()
    with pytest.raises(bootstrap.SourceVerificationError, match="unsupported tracked node"):
        bootstrap._inspect_git_repository(repo, head)


@pytest.mark.parametrize("extra_kind", ["file", "directory"])
def test_snapshot_extras_are_rejected_even_when_git_ignored(
    tmp_path: Path, extra_kind: str
) -> None:
    repo, head = _make_repo(
        tmp_path, extra_files={".gitignore": (b"ignored-runtime.py\n", 0o644)}
    )
    snapshot = _make_snapshot(repo, tmp_path / "snapshot")
    site_packages = _make_venv_site(tmp_path)
    extra = snapshot / "ignored-runtime.py"
    if extra_kind == "file":
        _write(extra, b"malicious ignored source\n", 0o600)
    else:
        extra.mkdir(mode=0o700)

    with pytest.raises(bootstrap.SourceVerificationError, match="layout differs"):
        _validate(repo, head, snapshot, site_packages)


def test_missing_snapshot_file_is_rejected(tmp_path: Path) -> None:
    repo, head = _make_repo(tmp_path)
    snapshot = _make_snapshot(repo, tmp_path / "snapshot")
    site_packages = _make_venv_site(tmp_path)
    (snapshot / "nested" / "data.bin").unlink()

    with pytest.raises(bootstrap.SourceVerificationError, match="file layout differs"):
        _validate(repo, head, snapshot, site_packages)


@pytest.mark.parametrize("node_kind", ["symlink", "fifo"])
def test_snapshot_symlink_and_nonregular_node_are_rejected(
    tmp_path: Path, node_kind: str
) -> None:
    repo, head = _make_repo(tmp_path)
    snapshot = _make_snapshot(repo, tmp_path / "snapshot")
    site_packages = _make_venv_site(tmp_path)
    target = snapshot / "nested" / "data.bin"
    original = tmp_path / "outside-data"
    original.write_bytes(target.read_bytes())
    target.unlink()
    if node_kind == "symlink":
        target.symlink_to(original)
    else:
        os.mkfifo(target, 0o600)

    with pytest.raises(bootstrap.SourceVerificationError, match="symlink or nonregular"):
        _validate(repo, head, snapshot, site_packages)


def test_snapshot_hardlinked_file_is_rejected(tmp_path: Path) -> None:
    repo, head = _make_repo(tmp_path)
    snapshot = _make_snapshot(repo, tmp_path / "snapshot")
    site_packages = _make_venv_site(tmp_path)
    target = snapshot / "nested" / "data.bin"
    os.link(target, tmp_path / "second-link")

    with pytest.raises(bootstrap.SourceVerificationError, match="must not be hardlinked"):
        _validate(repo, head, snapshot, site_packages)


@pytest.mark.parametrize(
    ("relative", "mode", "message"),
    [
        ("nested/data.bin", 0o620, "group/other writable"),
        ("nested/data.bin", 0o700, "mode mismatch"),
        ("tools/tracked-helper", 0o600, "mode mismatch"),
    ],
)
def test_snapshot_file_mode_must_match_normalized_git_mode(
    tmp_path: Path, relative: str, mode: int, message: str
) -> None:
    repo, head = _make_repo(tmp_path)
    snapshot = _make_snapshot(repo, tmp_path / "snapshot")
    site_packages = _make_venv_site(tmp_path)
    (snapshot / relative).chmod(mode)

    with pytest.raises(bootstrap.SourceVerificationError, match=message):
        _validate(repo, head, snapshot, site_packages)


def test_every_snapshot_directory_must_be_owner_private(tmp_path: Path) -> None:
    repo, head = _make_repo(tmp_path)
    snapshot = _make_snapshot(repo, tmp_path / "snapshot")
    site_packages = _make_venv_site(tmp_path)
    (snapshot / "nested").chmod(0o770)

    with pytest.raises(bootstrap.SourceVerificationError, match="mode must be 0700"):
        _validate(repo, head, snapshot, site_packages)


def test_snapshot_content_must_match_head_blob_and_sha256(tmp_path: Path) -> None:
    repo, head = _make_repo(tmp_path)
    snapshot = _make_snapshot(repo, tmp_path / "snapshot")
    site_packages = _make_venv_site(tmp_path)
    target = snapshot / "nested" / "data.bin"
    target.write_bytes(b"same-size-bad!!\xff\n")
    target.chmod(0o600)

    with pytest.raises(bootstrap.SourceVerificationError, match="does not equal HEAD blob"):
        _validate(repo, head, snapshot, site_packages)


def test_snapshot_file_wrong_owner_is_rejected_before_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "owned-source"
    _write(path, b"source\n", 0o600)
    real_uid = os.getuid()
    monkeypatch.setattr(bootstrap.os, "getuid", lambda: real_uid + 1)

    with pytest.raises(bootstrap.SourceVerificationError, match="owned"):
        bootstrap._read_snapshot_file(path, 0o600)


def test_path_inode_substitution_during_fd_read_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "raced-source"
    data = b"stable source bytes\n"
    _write(path, data, 0o600)
    moved = tmp_path / "original-inode"
    first_read = True

    def racing_read(descriptor: int, length: int) -> bytes:
        nonlocal first_read
        chunk = os.read(descriptor, length)
        if first_read and chunk:
            first_read = False
            path.rename(moved)
            _write(path, data, 0o600)
        return chunk

    with pytest.raises(
        bootstrap.SourceVerificationError,
        match="changed while being read|path changed during read",
    ):
        bootstrap._read_snapshot_file(path, 0o600, read_fn=racing_read)


def test_in_place_mutation_during_fd_read_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "raced-source"
    _write(path, b"stable source bytes\n", 0o600)
    first_read = True

    def racing_read(descriptor: int, length: int) -> bytes:
        nonlocal first_read
        chunk = os.read(descriptor, length)
        if first_read and chunk:
            first_read = False
            with path.open("r+b", buffering=0) as stream:
                stream.write(b"X")
                os.fsync(stream.fileno())
        return chunk

    with pytest.raises(bootstrap.SourceVerificationError, match="changed while being read"):
        bootstrap._read_snapshot_file(path, 0o600, read_fn=racing_read)


def test_noncanonical_or_relative_roots_are_rejected(tmp_path: Path) -> None:
    repo, head = _make_repo(tmp_path)
    snapshot = _make_snapshot(repo, tmp_path / "snapshot")
    site_packages = _make_venv_site(tmp_path)

    with pytest.raises(bootstrap.SourceVerificationError, match="must be absolute"):
        _validate(Path("relative-repo"), head, snapshot, site_packages)
    with pytest.raises(bootstrap.SourceVerificationError, match="canonical"):
        _validate(repo / ".." / repo.name, head, snapshot, site_packages)


def test_snapshot_must_be_disjoint_from_original_repository(tmp_path: Path) -> None:
    repo, head = _make_repo(tmp_path)
    site_packages = _make_venv_site(tmp_path)
    nested_snapshot = repo / "ignored-snapshot"
    nested_snapshot.mkdir(mode=0o700)
    _write(repo / ".git" / "info" / "exclude", b"ignored-snapshot/\n")

    with pytest.raises(bootstrap.SourceVerificationError, match="must be disjoint"):
        _validate(repo, head, nested_snapshot, site_packages)


@pytest.mark.parametrize(
    "site_builder",
    [
        lambda tmp: Path("relative/site-packages"),
        lambda tmp: (tmp / "not-a-venv-site"),
    ],
)
def test_venv_site_packages_must_be_absolute_canonical_venv_layout(
    tmp_path: Path, site_builder
) -> None:
    path = site_builder(tmp_path)
    if path.is_absolute():
        path.mkdir()
    with pytest.raises(bootstrap.SourceVerificationError, match="absolute|canonical"):
        bootstrap._validate_venv_site_packages(path)


def _integration_fixture(tmp_path: Path) -> dict[str, Path | str]:
    result_path = tmp_path / "target-result.json"
    pth_marker = tmp_path / "pth-executed"
    target_source = """
import json
import os
from pathlib import Path
import sys
import demo_dependency

Path(os.environ["TARGET_RESULT"]).write_text(json.dumps({
    "args": sys.argv[1:],
    "dependency": demo_dependency.VALUE,
    "site_imported": "site" in sys.modules,
    "sys_path": sys.path,
    "git_tree": os.environ["SKYRL_VERIFIED_SOURCE_GIT_TREE"],
    "manifest_sha256": os.environ["SKYRL_VERIFIED_SOURCE_MANIFEST_SHA256"],
    "runtime_policy": os.environ["SKYRL_VERIFIED_SOURCE_RUNTIME_POLICY"],
}), encoding="utf-8")
""".lstrip().encode()
    repo, head = _make_repo(tmp_path, target_source=target_source)
    snapshot = _make_snapshot(repo, tmp_path / "snapshot")
    site_packages = _make_venv_site(tmp_path)
    _write(site_packages / "demo_dependency.py", b"VALUE = 'venv-import-ok'\n", 0o600)
    _write(
        site_packages / "must-not-run.pth",
        f"import pathlib; pathlib.Path({str(pth_marker)!r}).write_text('bad')\n".encode(),
        0o600,
    )
    pycache = tmp_path / "empty-pycache"
    pycache.mkdir(mode=0o700)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(mode=0o700)
    return {
        "repo": repo,
        "head": head,
        "snapshot": snapshot,
        "site_packages": site_packages,
        "pycache": pycache,
        "manifest": artifacts / "source-manifest.jsonl",
        "result": result_path,
        "pth_marker": pth_marker,
    }


def _bootstrap_command(fixture: dict[str, Path | str]) -> list[str]:
    snapshot = fixture["snapshot"]
    assert isinstance(snapshot, Path)
    return [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-P",
        "-X",
        f"pycache_prefix={fixture['pycache']}",
        str(snapshot / "rocm" / "verified_source_bootstrap.py"),
        "--repo-root",
        str(fixture["repo"]),
        "--git-head",
        str(fixture["head"]),
        "--snapshot-root",
        str(snapshot),
        "--venv-site-packages",
        str(fixture["site_packages"]),
        "--module",
        "rocm.amdgpu_safety",
        "--manifest",
        str(fixture["manifest"]),
        "--",
        "alpha",
        "--beta",
    ]


def test_isolated_cli_runs_only_verified_module_without_site_or_pth(
    tmp_path: Path,
) -> None:
    fixture = _integration_fixture(tmp_path)
    environment = os.environ.copy()
    environment["TARGET_RESULT"] = str(fixture["result"])
    environment["PYTHONPATH"] = str(tmp_path / "must-be-ignored")

    result = subprocess.run(
        _bootstrap_command(fixture),
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert not fixture["pth_marker"].exists()
    assert list(fixture["pycache"].iterdir()) == []
    target = json.loads(fixture["result"].read_text())
    manifest = json.loads(fixture["manifest"].read_text())
    assert target["args"] == ["alpha", "--beta"]
    assert target["dependency"] == "venv-import-ok"
    assert target["site_imported"] is False
    assert target["sys_path"][0] == str(fixture["snapshot"])
    assert target["sys_path"][-1] == str(fixture["site_packages"])
    assert "" not in target["sys_path"]
    assert str(fixture["repo"]) not in target["sys_path"]
    assert target["manifest_sha256"] == manifest["source_manifest_sha256"]
    assert target["git_tree"] == manifest["git_tree"]
    assert target["runtime_policy"] == "true"
    assert manifest["runtime_policy"]["isolated"] is True
    assert manifest["runtime_policy"]["no_site"] is True
    assert manifest["runtime_policy"]["dont_write_bytecode"] is True
    assert manifest["runtime_policy"]["pycache_prefix_empty"] is True


def test_regular_site_package_cannot_override_verified_rocm_target(
    tmp_path: Path,
) -> None:
    fixture = _integration_fixture(tmp_path)
    site_packages = fixture["site_packages"]
    assert isinstance(site_packages, Path)
    hostile_marker = tmp_path / "hostile-rocm-imported"
    _write(
        site_packages / "rocm" / "__init__.py",
        f"from pathlib import Path\nPath({str(hostile_marker)!r}).touch()\n".encode(),
    )
    _write(
        site_packages / "rocm" / "amdgpu_safety.py",
        b"raise RuntimeError('hostile site target executed')\n",
    )
    environment = os.environ.copy()
    environment["TARGET_RESULT"] = str(fixture["result"])

    result = subprocess.run(
        _bootstrap_command(fixture),
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not hostile_marker.exists()
    assert json.loads(fixture["result"].read_text())["dependency"] == "venv-import-ok"


@pytest.mark.parametrize(
    ("remove", "message"),
    [
        ({"-I"}, "requires python -I -S -B -P"),
        ({"-S"}, "requires python -I -S -B -P"),
        ({"-B"}, "requires python -I -S -B -P"),
        ({"-X", "pycache"}, "requires -X pycache_prefix"),
    ],
)
def test_cli_rejects_missing_isolation_or_pycache_flags(
    tmp_path: Path, remove: set[str], message: str
) -> None:
    fixture = _integration_fixture(tmp_path)
    command = _bootstrap_command(fixture)
    if "pycache" in remove:
        x_index = command.index("-X")
        del command[x_index : x_index + 2]
    else:
        for flag in remove:
            command.remove(flag)

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert not fixture["result"].exists()
    assert not fixture["manifest"].exists()


@pytest.mark.parametrize("condition", ["nonempty", "public"])
def test_cli_rejects_nonempty_or_nonprivate_pycache_prefix(
    tmp_path: Path, condition: str
) -> None:
    fixture = _integration_fixture(tmp_path)
    pycache = fixture["pycache"]
    assert isinstance(pycache, Path)
    if condition == "nonempty":
        _write(pycache / "unexpected", b"data", 0o600)
    else:
        pycache.chmod(0o755)

    result = subprocess.run(
        _bootstrap_command(fixture),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert "pycache prefix" in result.stderr
    assert not fixture["result"].exists()


def test_cli_refuses_bootstrap_loaded_outside_snapshot(tmp_path: Path) -> None:
    fixture = _integration_fixture(tmp_path)
    command = _bootstrap_command(fixture)
    snapshot_tool = str(
        Path(fixture["snapshot"]) / "rocm" / "verified_source_bootstrap.py"
    )
    command[command.index(snapshot_tool)] = str(_TOOL)

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert "copy inside the validated snapshot" in result.stderr
    assert not fixture["result"].exists()


def test_manifest_must_be_new_private_and_outside_snapshot(tmp_path: Path) -> None:
    fixture = _integration_fixture(tmp_path)
    manifest = fixture["manifest"]
    assert isinstance(manifest, Path)
    _write(manifest, b"existing\n", 0o600)

    result = subprocess.run(
        _bootstrap_command(fixture),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert "cannot write source manifest" in result.stderr
    assert manifest.read_bytes() == b"existing\n"
    assert not fixture["result"].exists()


def test_fixed_git_command_is_sanitized_and_disables_replace_objects() -> None:
    assert bootstrap._GIT == "/usr/bin/git"
    assert bootstrap._GIT_PREFIX == (
        "/usr/bin/git",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
    )
    assert bootstrap._GIT_ENVIRONMENT == {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/nonexistent",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "XDG_CONFIG_HOME": "/nonexistent",
    }


def test_source_file_mentions_explicit_same_uid_and_os_exclusions() -> None:
    source = _TOOL.read_text(encoding="utf-8")
    assert "malicious process running as the same UID" in source
    assert "root, the kernel" in source
    assert "GIT_NO_REPLACE_OBJECTS" in source
    assert '"ls-tree", "-rz", "--full-tree", "HEAD"' in source
    assert "O_NOFOLLOW" in source
    assert "runpy.run_module" in source
