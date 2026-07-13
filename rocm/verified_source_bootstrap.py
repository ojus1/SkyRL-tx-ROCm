#!/usr/bin/env python3
"""Run a small ROCm helper from an exact, private Git source snapshot.

The bootstrap is intentionally stdlib-only.  Its operational invocation is::

    python -I -S -B -P -X pycache_prefix=/absolute/private/empty-dir \
      SNAPSHOT/rocm/verified_source_bootstrap.py \
      --repo-root ORIGINAL --git-head COMMIT --snapshot-root SNAPSHOT \
      --venv-site-packages /absolute/venv/lib/pythonX.Y/site-packages \
      --module rocm.prewarm_qwen35_buckets -- [module arguments]

The snapshot must be a normalized, owner-private copy of the *entire* tracked
tree: directories are mode 0700, Git 100644 blobs are mode 0600, and Git
100755 blobs are mode 0700.  No other node is accepted.  A separate preparation
mode creates or revalidates that snapshot at a stable private path keyed only by
the exact Git commit.  Stable source filenames are required for JAX executable
cache keys to survive process restarts.  The source manifest is checked against
fixed, sanitized ``/usr/bin/git`` commands before and after the snapshot read.
Python's site initialization and ``.pth`` processing remain disabled; the one
explicitly validated venv site-packages directory is added to ``sys.path``
directly.

Threat model: these checks fail closed for accidental source changes, Git index
flags, path substitution by a different UID, and ordinary filesystem races.
They explicitly do not defend against a malicious process running as the same
UID as this process, or against root, the kernel, compromised Git/Python
binaries, or a compromised operating system.  A same-UID process can modify
this already-loaded interpreter and is outside a meaningful self-attestation
boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import runpy
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence


class SourceVerificationError(RuntimeError):
    """The source snapshot or isolated interpreter contract was not satisfied."""


_GIT = "/usr/bin/git"
_GIT_TIMEOUT_SECONDS = 120
_TAR = "/usr/bin/tar"
_TAR_TIMEOUT_SECONDS = 120
_GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "HOME": "/nonexistent",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
    "XDG_CONFIG_HOME": "/nonexistent",
}
_GIT_PREFIX = (
    _GIT,
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.untrackedCache=false",
)
_MODE_TO_SNAPSHOT_MODE = {"100644": 0o600, "100755": 0o700}
_LAUNCHER_PATH = "rocm/start_qwen35.sh"
_BOOTSTRAP_PATH = "rocm/verified_source_bootstrap.py"
_PACKAGE_INIT_PATH = "rocm/__init__.py"
_ALLOWED_MODULES = {
    "rocm.amdgpu_safety": "rocm/amdgpu_safety.py",
    "rocm.prepare_jax_cache_dir": "rocm/prepare_jax_cache_dir.py",
    "rocm.prewarm_qwen35_buckets": "rocm/prewarm_qwen35_buckets.py",
    "rocm.profile_rocm": "rocm/profile_rocm.py",
    "rocm.qwen35_prewarm_handoff": "rocm/qwen35_prewarm_handoff.py",
}
_SOURCE_MANIFEST_FORMAT = "skyrl-verified-source-v1"
_SOURCE_CACHE_NAMESPACE = "skyrl-source-snapshots-private-v1"
_EXCLUDED_THREATS = (
    "malicious process running as the same UID",
    "parent process or pre-Python dynamic-loader environment",
    "root, kernel, privileged OS, or compromised Git/Python binary",
)
_INITIAL_ISOLATED_PATHS = tuple(sys.path)


@dataclass(frozen=True)
class GitBlob:
    path: str
    path_bytes: bytes
    mode: str
    oid: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class GitState:
    repo_root: Path
    head: str
    tree: str
    object_format: str
    blobs: tuple[GitBlob, ...]
    index_paths: tuple[str, ...]


@dataclass(frozen=True)
class SourceCachePaths:
    """Canonical locations for one commit-keyed private source snapshot."""

    cache_root: Path
    commit_root: Path
    archive: Path
    snapshot: Path


def _canonical_directory(
    raw_path: Path | str,
    label: str,
    *,
    exact_mode: int | None = None,
) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        raise SourceVerificationError(f"{label} must be absolute")
    try:
        path_metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SourceVerificationError(f"cannot inspect {label}: {error}") from error
    if resolved != path or stat.S_ISLNK(path_metadata.st_mode):
        raise SourceVerificationError(f"{label} must be a canonical, non-symlink path")
    if not stat.S_ISDIR(path_metadata.st_mode):
        raise SourceVerificationError(f"{label} must be a directory")
    if path_metadata.st_uid != os.getuid():
        raise SourceVerificationError(f"{label} must be owned by the current user")
    mode = stat.S_IMODE(path_metadata.st_mode)
    if exact_mode is not None and mode != exact_mode:
        raise SourceVerificationError(
            f"{label} mode must be {exact_mode:04o}, observed {mode:04o}"
        )
    return resolved


def _run_git(
    repo_root: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
    run_fn: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> bytes:
    command = [*_GIT_PREFIX, "-C", str(repo_root), *arguments]
    try:
        result = run_fn(
            command,
            input=input_bytes,
            capture_output=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
            env=dict(_GIT_ENVIRONMENT),
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SourceVerificationError("fixed Git source inspection failed") from error
    if result.returncode != 0 or result.stderr != b"":
        stderr = result.stderr[:512].decode("utf-8", "backslashreplace")
        raise SourceVerificationError(
            "fixed Git source inspection returned a nonzero or noisy result"
            + (f": {stderr}" if stderr else "")
        )
    return result.stdout


def _canonical_oid(raw: bytes, object_format: str, label: str) -> str:
    expected_length = 40 if object_format == "sha1" else 64
    try:
        value = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise SourceVerificationError(f"{label} is not ASCII") from error
    if len(value) != expected_length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SourceVerificationError(f"{label} is not one canonical object ID")
    return value


def _parse_git_tree(raw_tree: bytes, object_format: str) -> list[tuple[str, bytes, str, str]]:
    if not raw_tree.endswith(b"\0"):
        raise SourceVerificationError("Git tree output is not NUL terminated")
    parsed: list[tuple[str, bytes, str, str]] = []
    seen: set[bytes] = set()
    for record in raw_tree[:-1].split(b"\0"):
        try:
            metadata, path_bytes = record.split(b"\t", 1)
            mode_bytes, kind, oid_bytes = metadata.split(b" ", 2)
            mode = mode_bytes.decode("ascii")
        except (UnicodeDecodeError, ValueError) as error:
            raise SourceVerificationError("Git tree output is malformed") from error
        if kind != b"blob" or mode not in _MODE_TO_SNAPSHOT_MODE:
            display = os.fsdecode(path_bytes)
            raise SourceVerificationError(
                f"unsupported tracked node at {display!r}: mode={mode!r}, type={kind!r}"
            )
        if path_bytes in seen:
            raise SourceVerificationError("Git tree contains a duplicate path")
        seen.add(path_bytes)
        path = os.fsdecode(path_bytes)
        pure_path = PurePosixPath(path)
        if (
            not path
            or pure_path.is_absolute()
            or any(part in {"", ".", ".."} for part in pure_path.parts)
            or pure_path.as_posix() != path
        ):
            raise SourceVerificationError(f"Git tree path is not canonical: {path!r}")
        oid = _canonical_oid(oid_bytes, object_format, f"Git blob ID for {path!r}")
        parsed.append((path, path_bytes, mode, oid))
    if not parsed:
        raise SourceVerificationError("Git tree must contain at least one tracked blob")
    return parsed


def _git_blob_contents(
    repo_root: Path,
    tree: Sequence[tuple[str, bytes, str, str]],
    *,
    run_fn: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> list[bytes]:
    request = b"".join(oid.encode("ascii") + b"\n" for _, _, _, oid in tree)
    response = _run_git(
        repo_root,
        "cat-file",
        "--batch",
        input_bytes=request,
        run_fn=run_fn,
    )
    contents: list[bytes] = []
    offset = 0
    for path, _path_bytes, _mode, oid in tree:
        newline = response.find(b"\n", offset)
        if newline < 0:
            raise SourceVerificationError("Git cat-file batch response is truncated")
        header = response[offset:newline]
        fields = header.split(b" ")
        if len(fields) != 3 or fields[0] != oid.encode("ascii") or fields[1] != b"blob":
            raise SourceVerificationError(
                f"Git cat-file returned a malformed header for {path!r}"
            )
        try:
            size = int(fields[2])
        except ValueError as error:
            raise SourceVerificationError("Git cat-file returned a malformed size") from error
        if size < 0:
            raise SourceVerificationError("Git cat-file returned a negative size")
        content_start = newline + 1
        content_end = content_start + size
        if content_end >= len(response) or response[content_end : content_end + 1] != b"\n":
            raise SourceVerificationError(
                f"Git cat-file content framing failed for {path!r}"
            )
        contents.append(response[content_start:content_end])
        offset = content_end + 1
    if offset != len(response):
        raise SourceVerificationError("Git cat-file returned trailing output")
    return contents


def _inspect_git_repository(
    repo_root: Path | str,
    claimed_head: str,
    *,
    run_fn: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> GitState:
    root = _canonical_directory(repo_root, "original repository root")
    top_level = _run_git(root, "rev-parse", "--show-toplevel", run_fn=run_fn)
    if top_level != os.fsencode(root) + b"\n":
        raise SourceVerificationError("Git top-level does not equal the claimed repository")

    raw_format = _run_git(
        root, "rev-parse", "--show-object-format=storage", run_fn=run_fn
    )
    if raw_format not in {b"sha1\n", b"sha256\n"}:
        raise SourceVerificationError("Git object format must be exactly sha1 or sha256")
    object_format = raw_format[:-1].decode("ascii")

    raw_head = _run_git(
        root, "rev-parse", "--verify", "HEAD^{commit}", run_fn=run_fn
    )
    if not raw_head.endswith(b"\n") or raw_head.count(b"\n") != 1:
        raise SourceVerificationError("Git HEAD output is malformed")
    head = _canonical_oid(raw_head[:-1], object_format, "Git HEAD")
    try:
        claimed_head_bytes = claimed_head.encode("ascii", "strict")
    except UnicodeEncodeError as error:
        raise SourceVerificationError("claimed Git HEAD is not ASCII") from error
    claimed = _canonical_oid(
        claimed_head_bytes, object_format, "claimed Git HEAD"
    )
    if head != claimed:
        raise SourceVerificationError(
            f"Git HEAD changed: expected {claimed}, observed {head}"
        )

    raw_tree_oid = _run_git(
        root, "rev-parse", "--verify", "HEAD^{tree}", run_fn=run_fn
    )
    if not raw_tree_oid.endswith(b"\n") or raw_tree_oid.count(b"\n") != 1:
        raise SourceVerificationError("Git tree object output is malformed")
    tree_oid = _canonical_oid(
        raw_tree_oid[:-1], object_format, "Git HEAD tree"
    )

    status = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
        run_fn=run_fn,
    )
    if status != b"":
        raise SourceVerificationError("original repository worktree is not exactly clean")

    raw_tree = _run_git(
        root, "ls-tree", "-rz", "--full-tree", "HEAD", run_fn=run_fn
    )
    tree = _parse_git_tree(raw_tree, object_format)
    contents = _git_blob_contents(root, tree, run_fn=run_fn)
    blobs = tuple(
        GitBlob(
            path=path,
            path_bytes=path_bytes,
            mode=mode,
            oid=oid,
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        for (path, path_bytes, mode, oid), content in zip(tree, contents, strict=True)
    )

    raw_index = _run_git(
        root, "ls-files", "-v", "-z", "--full-name", run_fn=run_fn
    )
    if not raw_index.endswith(b"\0"):
        raise SourceVerificationError("Git index listing is not NUL terminated")
    index_paths: list[str] = []
    for record in raw_index[:-1].split(b"\0"):
        if not record.startswith(b"H "):
            display = os.fsdecode(record[2:] if len(record) >= 2 else record)
            raise SourceVerificationError(
                "Git index contains assume-unchanged, skip-worktree, or non-canonical "
                f"state at {display!r}"
            )
        index_paths.append(os.fsdecode(record[2:]))
    tree_paths = tuple(blob.path for blob in blobs)
    if tuple(index_paths) != tree_paths:
        raise SourceVerificationError("Git index paths do not exactly match HEAD")
    for required_path in (_LAUNCHER_PATH, _BOOTSTRAP_PATH, _PACKAGE_INIT_PATH):
        if required_path not in index_paths:
            raise SourceVerificationError(
                f"required bootstrap source is absent from HEAD/index: {required_path}"
            )
    return GitState(
        repo_root=root,
        head=head,
        tree=tree_oid,
        object_format=object_format,
        blobs=blobs,
        index_paths=tuple(index_paths),
    )


def _directory_metadata(path: Path, label: str) -> os.stat_result:
    required_flags = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required_flags):
        raise SourceVerificationError("directory verification requires Linux O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SourceVerificationError(f"cannot open {label}: {error}") from error
    try:
        descriptor_metadata = os.fstat(descriptor)
        path_metadata = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise SourceVerificationError(f"cannot inspect {label}: {error}") from error
    finally:
        os.close(descriptor)
    if not stat.S_ISDIR(descriptor_metadata.st_mode):
        raise SourceVerificationError(f"{label} is not a directory")
    if (descriptor_metadata.st_dev, descriptor_metadata.st_ino) != (
        path_metadata.st_dev,
        path_metadata.st_ino,
    ):
        raise SourceVerificationError(f"{label} changed while being inspected")
    if descriptor_metadata.st_uid != os.getuid():
        raise SourceVerificationError(f"{label} must be owned by the current user")
    mode = stat.S_IMODE(descriptor_metadata.st_mode)
    if mode != 0o700:
        raise SourceVerificationError(
            f"{label} mode must be 0700, observed {mode:04o}"
        )
    return descriptor_metadata


def _file_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_snapshot_file(
    path: Path,
    expected_mode: int,
    *,
    read_fn: Callable[[int, int], bytes] | None = None,
) -> bytes:
    required_flags = ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK")
    if any(not hasattr(os, name) for name in required_flags):
        raise SourceVerificationError("file verification requires Linux O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SourceVerificationError(f"cannot safely open snapshot file {path}: {error}") from error
    reader = os.read if read_fn is None else read_fn
    try:
        before = os.fstat(descriptor)
        path_before = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise SourceVerificationError(f"snapshot node is not a regular file: {path}")
        if before.st_uid != os.getuid():
            raise SourceVerificationError(
                f"snapshot file must be owned by the current user: {path}"
            )
        if before.st_nlink != 1:
            raise SourceVerificationError(f"snapshot file must not be hardlinked: {path}")
        observed_mode = stat.S_IMODE(before.st_mode)
        if observed_mode & 0o022:
            raise SourceVerificationError(
                f"snapshot file is group/other writable: {path}"
            )
        if observed_mode != expected_mode:
            raise SourceVerificationError(
                f"snapshot file mode mismatch at {path}: expected {expected_mode:04o}, "
                f"observed {observed_mode:04o}"
            )
        if (before.st_dev, before.st_ino) != (
            path_before.st_dev,
            path_before.st_ino,
        ):
            raise SourceVerificationError(f"snapshot file path changed before read: {path}")

        chunks: list[bytes] = []
        while True:
            chunk = reader(descriptor, 1024 * 1024)
            if not isinstance(chunk, bytes):
                raise SourceVerificationError("snapshot read returned a non-bytes value")
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        path_after = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise SourceVerificationError(f"snapshot file read failed for {path}: {error}") from error
    finally:
        os.close(descriptor)

    if _file_fingerprint(before) != _file_fingerprint(after):
        raise SourceVerificationError(f"snapshot file changed while being read: {path}")
    if (after.st_dev, after.st_ino) != (path_after.st_dev, path_after.st_ino):
        raise SourceVerificationError(f"snapshot file path changed during read: {path}")
    data = b"".join(chunks)
    if len(data) != after.st_size:
        raise SourceVerificationError(f"snapshot file size changed during read: {path}")
    return data


def _scan_snapshot(snapshot_root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    pending: list[tuple[Path, PurePosixPath | None]] = [(snapshot_root, None)]
    while pending:
        directory, relative = pending.pop()
        label = "snapshot root" if relative is None else f"snapshot directory {relative}"
        _directory_metadata(directory, label)
        try:
            with os.scandir(directory) as iterator:
                entries = list(iterator)
        except OSError as error:
            raise SourceVerificationError(f"cannot enumerate {label}: {error}") from error
        for entry in entries:
            child_relative = (
                PurePosixPath(entry.name)
                if relative is None
                else relative / entry.name
            )
            relative_name = child_relative.as_posix()
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise SourceVerificationError(
                    f"cannot inspect snapshot node {relative_name!r}: {error}"
                ) from error
            child_path = directory / entry.name
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative_name)
                pending.append((child_path, child_relative))
            elif stat.S_ISREG(metadata.st_mode):
                files.add(relative_name)
            else:
                raise SourceVerificationError(
                    f"snapshot contains a symlink or nonregular node: {relative_name!r}"
                )
    return files, directories


def _git_blob_oid(data: bytes, object_format: str) -> str:
    payload_header = f"blob {len(data)}\0".encode("ascii")
    if object_format == "sha1":
        digest = hashlib.new("sha1", usedforsecurity=False)
    elif object_format == "sha256":
        digest = hashlib.sha256()
    else:  # Defensive: _inspect_git_repository already rejects this.
        raise SourceVerificationError("unsupported Git object format")
    digest.update(payload_header)
    digest.update(data)
    return digest.hexdigest()


def _validate_snapshot_tree(snapshot_root: Path, state: GitState) -> tuple[list[dict[str, Any]], int]:
    observed_files, observed_directories = _scan_snapshot(snapshot_root)
    expected_files = {blob.path for blob in state.blobs}
    expected_directories: set[str] = set()
    for blob in state.blobs:
        parent = PurePosixPath(blob.path).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if observed_files != expected_files:
        missing = sorted(expected_files - observed_files)
        extra = sorted(observed_files - expected_files)
        raise SourceVerificationError(
            f"snapshot file layout differs from HEAD: missing={missing!r}, extra={extra!r}"
        )
    if observed_directories != expected_directories:
        missing = sorted(expected_directories - observed_directories)
        extra = sorted(observed_directories - expected_directories)
        raise SourceVerificationError(
            f"snapshot directory layout differs from HEAD: missing={missing!r}, extra={extra!r}"
        )

    records: list[dict[str, Any]] = []
    total_bytes = 0
    for blob in state.blobs:
        path = snapshot_root.joinpath(*PurePosixPath(blob.path).parts)
        data = _read_snapshot_file(path, _MODE_TO_SNAPSHOT_MODE[blob.mode])
        sha256 = hashlib.sha256(data).hexdigest()
        working_oid = _git_blob_oid(data, state.object_format)
        if len(data) != blob.size_bytes or sha256 != blob.sha256 or working_oid != blob.oid:
            raise SourceVerificationError(
                f"snapshot content does not equal HEAD blob at {blob.path!r}"
            )
        total_bytes += len(data)
        records.append(
            {
                "git_mode": blob.mode,
                "git_oid": blob.oid,
                "path": blob.path,
                "sha256": sha256,
                "size_bytes": len(data),
                "snapshot_mode": f"{_MODE_TO_SNAPSHOT_MODE[blob.mode]:04o}",
            }
        )
    return records, total_bytes


def _canonical_commit_key(value: str) -> str:
    try:
        encoded = value.encode("ascii", "strict")
    except UnicodeEncodeError as error:
        raise SourceVerificationError("source-cache commit key is not ASCII") from error
    if len(encoded) == 40:
        object_format = "sha1"
    elif len(encoded) == 64:
        object_format = "sha256"
    else:
        raise SourceVerificationError(
            "source-cache commit key must be one full SHA-1 or SHA-256 object ID"
        )
    return _canonical_oid(encoded, object_format, "source-cache commit key")


def source_cache_paths(account_home: Path | str, git_head: str) -> SourceCachePaths:
    """Return paths whose source filenames depend only on the account and commit."""
    home = Path(account_home)
    if not home.is_absolute():
        raise SourceVerificationError("source-cache account home must be absolute")
    if any(character in str(home) for character in "\r\n\t"):
        raise SourceVerificationError(
            "source-cache account home must not contain line-control characters"
        )
    commit = _canonical_commit_key(git_head)
    cache_root = home / ".cache" / _SOURCE_CACHE_NAMESPACE
    commit_root = cache_root / commit
    return SourceCachePaths(
        cache_root=cache_root,
        commit_root=commit_root,
        archive=commit_root / "source-head.tar",
        snapshot=commit_root / "source-head",
    )


def _create_or_validate_private_directory(path: Path, label: str) -> Path:
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    except OSError as error:
        raise SourceVerificationError(f"cannot create {label}: {error}") from error
    return _canonical_directory(path, label, exact_mode=0o700)


def _create_private_directory(path: Path, label: str) -> Path:
    try:
        os.mkdir(path, 0o700)
    except OSError as error:
        raise SourceVerificationError(f"cannot create new {label}: {error}") from error
    return _canonical_directory(path, label, exact_mode=0o700)


def _write_private_file_exclusive(path: Path, payload: bytes, label: str) -> None:
    required_flags = ("O_CLOEXEC", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required_flags):
        raise SourceVerificationError("private file creation requires Linux O_NOFOLLOW")
    _directory_metadata(path.parent, f"{label} parent")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise SourceVerificationError(f"cannot create {label}: {error}") from error
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short private source-cache write")
            view = view[written:]
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        endpoint = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or _file_fingerprint(metadata) != _file_fingerprint(endpoint)
        ):
            raise OSError("created private source-cache file failed metadata checks")
    except OSError as error:
        raise SourceVerificationError(f"cannot populate {label}: {error}") from error
    finally:
        os.close(descriptor)


def _extract_private_source_archive(archive: Path, snapshot: Path) -> None:
    environment = {"LC_ALL": "C", "PATH": "/usr/bin:/bin"}
    previous_umask = os.umask(0o077)
    try:
        result = subprocess.run(
            [
                _TAR,
                "--extract",
                "--no-same-owner",
                "--no-same-permissions",
                f"--file={archive}",
                f"--directory={snapshot}",
            ],
            capture_output=True,
            check=False,
            timeout=_TAR_TIMEOUT_SECONDS,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SourceVerificationError(
            "fixed private source-cache extraction failed"
        ) from error
    finally:
        os.umask(previous_umask)
    if result.returncode != 0 or result.stdout != b"" or result.stderr != b"":
        raise SourceVerificationError(
            "fixed private source-cache extraction returned a nonzero or noisy result"
        )


def _validate_source_cache_pair(
    paths: SourceCachePaths,
    state: GitState,
    expected_archive: bytes,
) -> tuple[list[dict[str, Any]], int, str]:
    _canonical_directory(paths.commit_root, "source-cache commit root", exact_mode=0o700)
    try:
        with os.scandir(paths.commit_root) as iterator:
            entry_names = sorted(entry.name for entry in iterator)
    except OSError as error:
        raise SourceVerificationError(
            f"cannot enumerate source-cache commit root: {error}"
        ) from error
    if entry_names != ["source-head", "source-head.tar"]:
        raise SourceVerificationError(
            "source-cache commit root must contain exactly source-head and "
            "source-head.tar"
        )

    snapshot = _canonical_directory(
        paths.snapshot, "source-cache snapshot", exact_mode=0o700
    )
    archive = _read_snapshot_file(paths.archive, 0o600)
    expected_sha256 = hashlib.sha256(expected_archive).hexdigest()
    observed_sha256 = hashlib.sha256(archive).hexdigest()
    if len(archive) != len(expected_archive) or observed_sha256 != expected_sha256:
        raise SourceVerificationError(
            "source-cache archive does not equal fixed Git archive output"
        )
    records, total_bytes = _validate_snapshot_tree(snapshot, state)
    return records, total_bytes, observed_sha256


def prepare_source_cache(
    *, repo_root: Path, git_head: str, account_home: Path
) -> dict[str, Any]:
    """Create or revalidate one stable, private, full-HEAD source snapshot."""
    home = _canonical_directory(account_home, "source-cache account home")
    before = _inspect_git_repository(repo_root, git_head)
    paths = source_cache_paths(home, before.head)

    cache_parent = _create_or_validate_private_directory(
        home / ".cache", "private account cache directory"
    )
    if paths.cache_root.parent != cache_parent:
        raise SourceVerificationError("source-cache root escaped the private account cache")
    _create_or_validate_private_directory(paths.cache_root, "source-cache root")

    expected_archive = _run_git(
        before.repo_root, "archive", "--format=tar", before.head
    )
    try:
        paths.commit_root.lstat()
    except FileNotFoundError:
        cache_status = "created"
        _create_private_directory(paths.commit_root, "source-cache commit root")
        _create_private_directory(paths.snapshot, "source-cache snapshot")
        _write_private_file_exclusive(
            paths.archive, expected_archive, "source-cache archive"
        )
        _extract_private_source_archive(paths.archive, paths.snapshot)
    except OSError as error:
        raise SourceVerificationError(
            f"cannot inspect source-cache commit root: {error}"
        ) from error
    else:
        cache_status = "reused"

    records, total_bytes, archive_sha256 = _validate_source_cache_pair(
        paths, before, expected_archive
    )
    after = _inspect_git_repository(repo_root, git_head)
    if after != before:
        raise SourceVerificationError(
            "Git HEAD, index, status, or blob contents changed during source-cache "
            "preparation"
        )
    return {
        "cache_status": cache_status,
        "format": "skyrl-private-source-cache-v1",
        "git_head": before.head,
        "git_tree": before.tree,
        "source_archive_path": str(paths.archive),
        "source_archive_sha256": archive_sha256,
        "source_file_count": len(records),
        "source_snapshot_root": str(paths.snapshot),
        "source_total_bytes": total_bytes,
        "full_head_tree_validated": True,
    }


def _validate_venv_site_packages(raw_path: Path | str) -> Path:
    site_packages = _canonical_directory(raw_path, "venv site-packages")
    expected_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    if (
        site_packages.name != "site-packages"
        or site_packages.parent.name != expected_version
        or site_packages.parent.parent.name != "lib"
    ):
        raise SourceVerificationError(
            "venv site-packages must have the canonical lib/pythonX.Y/site-packages layout"
        )
    venv_root = site_packages.parents[2]
    configuration = venv_root / "pyvenv.cfg"
    try:
        config_metadata = configuration.lstat()
    except OSError as error:
        raise SourceVerificationError("venv site-packages has no pyvenv.cfg") from error
    if (
        stat.S_ISLNK(config_metadata.st_mode)
        or not stat.S_ISREG(config_metadata.st_mode)
        or config_metadata.st_uid != os.getuid()
        or config_metadata.st_nlink != 1
    ):
        raise SourceVerificationError("venv pyvenv.cfg must be an owned regular file")
    return site_packages


def _validate_runtime_policy() -> dict[str, Any]:
    required = {
        "isolated": sys.flags.isolated == 1,
        "no_site": sys.flags.no_site == 1,
        "dont_write_bytecode": sys.flags.dont_write_bytecode == 1,
        "ignore_environment": sys.flags.ignore_environment == 1,
        "safe_path": bool(sys.flags.safe_path),
        "site_not_imported": "site" not in sys.modules,
    }
    failed = sorted(name for name, passed in required.items() if not passed)
    if failed:
        raise SourceVerificationError(
            "bootstrap requires python -I -S -B -P; failed checks: " + ", ".join(failed)
        )
    raw_prefix = sys.pycache_prefix
    if raw_prefix is None:
        raise SourceVerificationError(
            "bootstrap requires -X pycache_prefix=/absolute/private/empty-dir"
        )
    prefix = _canonical_directory(raw_prefix, "pycache prefix", exact_mode=0o700)
    try:
        with os.scandir(prefix) as iterator:
            prefix_nonempty = next(iterator, None) is not None
        if prefix_nonempty:
            raise SourceVerificationError("pycache prefix must be empty")
    except OSError as error:
        raise SourceVerificationError(f"cannot inspect pycache prefix: {error}") from error
    return {
        **required,
        "pycache_prefix": str(prefix),
        "pycache_prefix_empty": True,
        "pycache_prefix_private": True,
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_snapshot(
    *,
    repo_root: Path,
    git_head: str,
    snapshot_root: Path,
    venv_site_packages: Path,
    target_module: str,
    require_runtime_policy: bool = True,
) -> dict[str, Any]:
    """Validate an exact full-tree source snapshot and return its manifest.

    This function has no GPU/JAX imports or accesses.  It is deliberately
    reusable by an allowlisted target for exact first/final record
    revalidation.
    """
    if target_module not in _ALLOWED_MODULES:
        raise SourceVerificationError(
            f"target module is not allowlisted: {target_module!r}"
        )
    root = _canonical_directory(repo_root, "original repository root")
    snapshot = _canonical_directory(snapshot_root, "source snapshot root", exact_mode=0o700)
    site_packages = _validate_venv_site_packages(venv_site_packages)
    runtime_policy = (
        _validate_runtime_policy()
        if require_runtime_policy
        else {"enforced": False, "pycache_prefix": None}
    )
    if _is_relative_to(snapshot, root) or _is_relative_to(root, snapshot):
        raise SourceVerificationError(
            "source snapshot and original repository must be disjoint"
        )

    before = _inspect_git_repository(root, git_head)
    required_target_path = _ALLOWED_MODULES[target_module]
    if required_target_path not in before.index_paths:
        raise SourceVerificationError(
            f"allowlisted target is absent from HEAD: {required_target_path}"
        )
    file_records, total_bytes = _validate_snapshot_tree(snapshot, before)
    after = _inspect_git_repository(root, git_head)
    if after != before:
        raise SourceVerificationError(
            "Git HEAD, index, status, or blob contents changed during snapshot validation"
        )

    source_payload = {
        "format": _SOURCE_MANIFEST_FORMAT,
        "files": file_records,
        "git_head": before.head,
        "git_tree": before.tree,
        "git_object_format": before.object_format,
    }
    canonical_payload = json.dumps(
        source_payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    source_manifest_sha256 = hashlib.sha256(canonical_payload).hexdigest()
    return {
        **source_payload,
        "allowed_target_modules": sorted(_ALLOWED_MODULES),
        "file_count": len(file_records),
        "original_repo_root": str(root),
        "runtime_policy": runtime_policy,
        "snapshot_root": str(snapshot),
        "source_manifest_sha256": source_manifest_sha256,
        "status": "passed",
        "target_module": target_module,
        "target_source_path": required_target_path,
        "threat_model_excludes": list(_EXCLUDED_THREATS),
        "total_source_bytes": total_bytes,
        "venv_site_packages": str(site_packages),
    }


def _validate_bootstrap_location(snapshot_root: Path) -> None:
    expected = snapshot_root / _BOOTSTRAP_PATH
    try:
        actual = Path(__file__).resolve(strict=True)
    except OSError as error:
        raise SourceVerificationError("cannot resolve loaded bootstrap source") from error
    if actual != expected:
        raise SourceVerificationError(
            "operational bootstrap must execute the copy inside the validated snapshot"
        )


def _write_manifest(path: Path, snapshot_root: Path, manifest: dict[str, Any]) -> None:
    if not path.is_absolute():
        raise SourceVerificationError("manifest path must be absolute")
    if _is_relative_to(path, snapshot_root):
        raise SourceVerificationError("manifest must be outside the immutable source snapshot")
    parent = _canonical_directory(path.parent, "manifest parent", exact_mode=0o700)
    if parent != path.parent:
        raise SourceVerificationError("manifest parent must be canonical")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    payload = (
        json.dumps(manifest, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short manifest write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise SourceVerificationError(f"cannot write source manifest: {error}") from error


def _install_verified_sys_path(snapshot_root: Path, site_packages: Path) -> None:
    trusted_stdlib: list[str] = []
    for raw_entry in _INITIAL_ISOLATED_PATHS:
        if not raw_entry:
            raise SourceVerificationError("isolated interpreter sys.path contains an empty entry")
        entry = Path(raw_entry)
        if not entry.is_absolute():
            raise SourceVerificationError("isolated interpreter sys.path contains a relative entry")
        lowered_parts = {part.lower() for part in entry.parts}
        if "site-packages" in lowered_parts or "dist-packages" in lowered_parts:
            raise SourceVerificationError(
                "isolated interpreter unexpectedly initialized a package directory"
            )
        trusted_stdlib.append(str(entry))
    sys.path[:] = [str(snapshot_root), *trusted_stdlib, str(site_packages)]
    sys.path_importer_cache.clear()


def _validate_target_resolution(snapshot_root: Path, target_module: str) -> None:
    if "rocm" in sys.modules or target_module in sys.modules:
        raise SourceVerificationError(
            "verified target package must not be imported before origin validation"
        )
    package_source = snapshot_root / _PACKAGE_INIT_PATH
    package_directory = package_source.parent
    package_spec = importlib.util.find_spec("rocm")
    package_locations = (
        []
        if package_spec is None or package_spec.submodule_search_locations is None
        else list(package_spec.submodule_search_locations)
    )
    try:
        package_origin = (
            None
            if package_spec is None or package_spec.origin is None
            else Path(package_spec.origin).resolve(strict=True)
        )
    except OSError as error:
        raise SourceVerificationError(
            "cannot resolve the selected rocm package source"
        ) from error
    if package_origin != package_source or package_locations != [str(package_directory)]:
        raise SourceVerificationError(
            "rocm package resolution does not bind exclusively to the source snapshot"
        )
    target_spec = importlib.util.find_spec(target_module)
    expected_target = snapshot_root / _ALLOWED_MODULES[target_module]
    try:
        target_origin = (
            None
            if target_spec is None or target_spec.origin is None
            else Path(target_spec.origin).resolve(strict=True)
        )
    except OSError as error:
        raise SourceVerificationError(
            "cannot resolve the selected allowlisted target source"
        ) from error
    if (
        target_spec is None
        or target_spec.loader is None
        or target_origin != expected_target
    ):
        raise SourceVerificationError(
            "allowlisted target resolution does not bind to its validated source file"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and run one fixed ROCm helper from a private Git snapshot.",
        allow_abbrev=False,
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--git-head", required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--venv-site-packages", type=Path, required=True)
    parser.add_argument("--module", choices=sorted(_ALLOWED_MODULES), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("module_args", nargs=argparse.REMAINDER)
    return parser


def _source_cache_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create or revalidate a stable, private, commit-keyed source snapshot."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--git-head", required=True)
    parser.add_argument("--account-home", type=Path, required=True)
    return parser


def _prepare_source_cache_cli(argv: Sequence[str]) -> int:
    args = _source_cache_parser().parse_args(argv)
    try:
        _validate_runtime_policy()
        if __file__ != "<stdin>" or sys.argv[0] != "-":
            raise SourceVerificationError(
                "source-cache preparation must execute the exact Git bootstrap blob "
                "from standard input"
            )
        result = prepare_source_cache(
            repo_root=args.repo_root,
            git_head=args.git_head,
            account_home=args.account_home,
        )
        fields = (
            result["source_archive_path"],
            result["source_archive_sha256"],
            result["source_snapshot_root"],
            result["cache_status"],
        )
        if any(
            not isinstance(value, str)
            or any(character in value for character in "\r\n\t")
            for value in fields
        ):
            raise SourceVerificationError(
                "source-cache result cannot be represented as fixed launcher lines"
            )
    except SourceVerificationError as error:
        print(f"verified-source cache preparation refused: {error}", file=sys.stderr)
        return 2
    for value in fields:
        print(value)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--prepare-source-cache"]:
        return _prepare_source_cache_cli(arguments[1:])
    args = _parser().parse_args(arguments)
    module_arguments = list(args.module_args)
    if module_arguments[:1] == ["--"]:
        module_arguments = module_arguments[1:]
    try:
        manifest = validate_snapshot(
            repo_root=args.repo_root,
            git_head=args.git_head,
            snapshot_root=args.snapshot_root,
            venv_site_packages=args.venv_site_packages,
            target_module=args.module,
            require_runtime_policy=True,
        )
        _validate_bootstrap_location(args.snapshot_root)
        if args.manifest is not None:
            _write_manifest(args.manifest, args.snapshot_root, manifest)
        os.environ["SKYRL_VERIFIED_SOURCE_GIT_HEAD"] = manifest["git_head"]
        os.environ["SKYRL_VERIFIED_SOURCE_GIT_TREE"] = manifest["git_tree"]
        os.environ["SKYRL_VERIFIED_SOURCE_MANIFEST_SHA256"] = manifest[
            "source_manifest_sha256"
        ]
        os.environ["SKYRL_VERIFIED_SOURCE_RUNTIME_POLICY"] = "true"
        os.environ["SKYRL_VERIFIED_SOURCE_SNAPSHOT_ROOT"] = manifest["snapshot_root"]
        _install_verified_sys_path(args.snapshot_root, args.venv_site_packages)
        _validate_target_resolution(args.snapshot_root, args.module)
        sys.argv = [args.module, *module_arguments]
        runpy.run_module(args.module, run_name="__main__", alter_sys=True)
    except SourceVerificationError as error:
        print(f"verified-source bootstrap refused execution: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
