#!/usr/bin/env python3
"""Create and validate SkyRL's private, stack-versioned JAX cache directory.

JAX treats persistent-cache entries as trusted executable content.  This
helper deliberately accepts no arbitrary cache path: it walks from the user's
real home directory with no-follow directory descriptors, rejects preexisting
writable/private-directory mismatches instead of repairing them, and checks
the separately unbounded per-fusion autotune subtree at every server startup.
It imports no JAX and never opens an accelerator device.
"""

from __future__ import annotations

import argparse
import os
import pwd
import stat
import sys
from pathlib import Path

_CACHE_BASE = "skyrl-jax-rocm-private-v1"
_CACHE_NAMESPACE = (
    "jax0.10.2-jaxlib0.10.2-rocm-plugin0.10.2-pjrt0.10.2-"
    "rocm7.2.4-amdgpu6.16.13-gfx1100-v1"
)
_AUTOTUNE_SUBDIRECTORY = "xla_gpu_per_fusion_autotune_cache_dir"
_NOFOLLOW_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _require_owned_directory(
    directory_fd: int,
    label: str,
    *,
    exact_mode: int | None,
) -> None:
    metadata = os.fstat(directory_fd)
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"{label} is not a directory")
    if metadata.st_uid != os.getuid():
        raise RuntimeError(f"{label} is not owned by the current user")
    mode = stat.S_IMODE(metadata.st_mode)
    if exact_mode is not None and mode != exact_mode:
        raise RuntimeError(
            f"{label} mode is {mode:04o}; refusing preexisting trusted content unless mode is {exact_mode:04o}"
        )
    if mode & 0o022:
        raise RuntimeError(f"{label} is writable by group or other users")


def _open_or_create_private_child(parent_fd: int, name: str, label: str) -> int:
    try:
        child_fd = os.open(name, _NOFOLLOW_DIRECTORY, dir_fd=parent_fd)
    except FileNotFoundError:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        child_fd = os.open(name, _NOFOLLOW_DIRECTORY, dir_fd=parent_fd)
    _require_owned_directory(child_fd, label, exact_mode=0o700)
    return child_fd


def _tree_bytes_no_symlinks(path: Path) -> int:
    total = 0
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_symlink():
                raise RuntimeError(f"refusing symlink in trusted cache: {entry.path}")
            if entry.is_dir(follow_symlinks=False):
                total += _tree_bytes_no_symlinks(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                total += entry.stat(follow_symlinks=False).st_size
            else:
                raise RuntimeError(
                    f"refusing non-regular object in trusted cache: {entry.path}"
                )
    return total


def _open_account_home() -> tuple[Path, int]:
    uid = os.getuid()
    home = Path(pwd.getpwuid(uid).pw_dir)
    if not home.is_absolute() or any(part in {".", ".."} for part in home.parts):
        raise RuntimeError("account database returned a noncanonical home directory")

    current_fd = os.open("/", _NOFOLLOW_DIRECTORY)
    try:
        components = home.parts[1:]
        for index, component in enumerate(components):
            next_fd = os.open(component, _NOFOLLOW_DIRECTORY, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
            metadata = os.fstat(current_fd)
            mode = stat.S_IMODE(metadata.st_mode)
            if not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError("account-home ancestor is not a directory")
            if metadata.st_uid not in {0, uid}:
                raise RuntimeError("account-home ancestor has an untrusted owner")
            if mode & 0o022:
                raise RuntimeError(
                    "account-home ancestor is writable by group or other users"
                )
            if index == len(components) - 1 and metadata.st_uid != uid:
                raise RuntimeError("account home is not owned by the current user")
        return home, current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _prepare_cache_in_open_home(
    home: Path,
    home_fd: int,
    max_autotune_bytes: int,
) -> Path:
    if max_autotune_bytes < 0:
        raise ValueError("max_autotune_bytes must be nonnegative")
    cache_fd = base_fd = namespace_fd = None
    try:
        _require_owned_directory(home_fd, "home directory", exact_mode=None)
        cache_fd = _open_or_create_private_child(
            home_fd, ".cache", "home cache directory"
        )
        base_fd = _open_or_create_private_child(
            cache_fd, _CACHE_BASE, "SkyRL trusted cache base"
        )
        namespace_fd = _open_or_create_private_child(
            base_fd, _CACHE_NAMESPACE, "SkyRL trusted cache namespace"
        )

        namespace = home / ".cache" / _CACHE_BASE / _CACHE_NAMESPACE
        # Reject symlinks, sockets, devices, and other non-regular objects
        # anywhere JAX may read trusted executable or autotune content.
        _tree_bytes_no_symlinks(namespace)
        autotune = namespace / _AUTOTUNE_SUBDIRECTORY
        if autotune.exists():
            if autotune.is_symlink() or not autotune.is_dir():
                raise RuntimeError("autotune cache path is not a real directory")
            autotune_bytes = _tree_bytes_no_symlinks(autotune)
            if autotune_bytes > max_autotune_bytes:
                raise RuntimeError(
                    "per-fusion autotune cache uses "
                    f"{autotune_bytes} bytes, above startup maximum {max_autotune_bytes}"
                )
        return namespace
    finally:
        for directory_fd in (namespace_fd, base_fd, cache_fd):
            if directory_fd is not None:
                os.close(directory_fd)


def prepare_cache(max_autotune_bytes: int) -> Path:
    home, home_fd = _open_account_home()
    try:
        return _prepare_cache_in_open_home(home, home_fd, max_autotune_bytes)
    finally:
        os.close(home_fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-autotune-bytes", required=True, type=int)
    args = parser.parse_args()
    try:
        cache = prepare_cache(args.max_autotune_bytes)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"JAX cache validation failed: {error}", file=sys.stderr)
        return 2
    print(cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
