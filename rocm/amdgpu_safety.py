"""Fail-closed current-boot AMDGPU fatal-event quarantine.

An illegal command-stream opcode can leave a GPU context or driver state
suspect even when the offending process exits and VRAM returns to idle.  All
full-model ROCm probes call this module before importing JAX.  A reboot starts
a new journal boot and is the only way to clear the quarantine.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

AMDGPU_FATAL_PATTERN = re.compile(
    r"(?=.*\bamdgpu(?:\b|_)).*(?:ring\s+\S+\s+timeout|illegal opcode|page fault|"
    r"vm fault|protection[_. -]?fault|gpu fault|device wedged|gpu reset|"
    r"ring reset|job[ _]timed[ _]?out|failed to reset)",
    re.IGNORECASE,
)

_AMD_PCI_VENDOR_ID = "0x1002"
_QWEN35_LOCK_DIRECTORY_PREFIX = "skyrl-qwen35-rocm-"


def acquire_qwen35_rocm_launch_lock(*, runtime_dir: Path | None = None) -> int:
    """Hold the launch lock shared by every guarded Qwen3.5 ROCm process.

    The lock is taken on the same per-user directory inode used by the shell
    server launcher and the older Python probes.  Returning the descriptor
    makes its lifetime explicit: the caller must keep it open for the entire
    GPU process and close it during cleanup.
    """
    lock_parent = (
        runtime_dir
        if runtime_dir is not None
        else Path("/run/user") / str(os.getuid())
    )
    if not lock_parent.is_absolute():
        raise RuntimeError(f"launch-lock parent must be absolute: {lock_parent}")
    try:
        parent_metadata = lock_parent.lstat()
        resolved_parent = lock_parent.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(
            f"could not inspect fixed launch-lock parent: {lock_parent}"
        ) from error
    if (
        resolved_parent != lock_parent
        or stat.S_ISLNK(parent_metadata.st_mode)
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.getuid()
        or stat.S_IMODE(parent_metadata.st_mode) != 0o700
    ):
        raise RuntimeError(
            f"refusing unsafe fixed launch-lock parent: {lock_parent}"
        )
    lock_dir = lock_parent / f"{_QWEN35_LOCK_DIRECTORY_PREFIX}{os.getuid()}"
    try:
        lock_dir.mkdir(mode=0o700)
    except FileExistsError:
        pass
    except OSError as error:
        raise RuntimeError(
            f"could not create global launch-lock directory: {lock_dir}"
        ) from error

    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    try:
        lock_fd = os.open(lock_dir, flags)
    except OSError as error:
        raise RuntimeError(
            f"refusing unsafe global launch-lock directory: {lock_dir}"
        ) from error
    try:
        info = os.fstat(lock_fd)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise RuntimeError(
                f"refusing unsafe global launch-lock directory: {lock_dir}"
            )
        os.fchmod(lock_fd, 0o700)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(lock_fd)
        raise RuntimeError(
            "another Qwen3.5 ROCm process holds the global launch lock"
        ) from error
    except BaseException:
        os.close(lock_fd)
        raise
    return lock_fd


def require_headless_unowned_amdgpu(
    drm_root: Path = Path("/sys/class/drm"),
    kfd_path: Path = Path("/dev/kfd"),
    *,
    stat_fn: Any = os.stat,
    access_fn: Any = os.access,
    which_fn: Any = shutil.which,
    run_fn: Any = subprocess.run,
) -> dict[str, Any]:
    """Require an AMD DRM card, no AMD display, and an unowned KFD node."""
    amd_cards: list[str] = []
    connected_connectors: list[str] = []
    for card_path in sorted(drm_root.glob("card[0-9]*")):
        if re.fullmatch(r"card[0-9]+", card_path.name) is None:
            continue
        try:
            vendor = (card_path / "device" / "vendor").read_text().strip()
        except OSError:
            continue
        if vendor != _AMD_PCI_VENDOR_ID:
            continue
        amd_cards.append(card_path.name)
        for status_path in sorted(drm_root.glob(f"{card_path.name}-*/status")):
            if "Writeback" in status_path.parent.name:
                continue
            try:
                status_text = status_path.read_text().strip()
            except OSError as error:
                raise RuntimeError(
                    f"cannot verify AMD connector state at {status_path}: {error}"
                ) from error
            if status_text == "connected":
                connected_connectors.append(status_path.parent.name)

    if not amd_cards:
        raise RuntimeError("no AMD DRM card was found")
    if connected_connectors:
        raise RuntimeError(
            "refusing ROCm work while an AMD display connector is active: "
            + ", ".join(connected_connectors)
        )

    try:
        kfd_stat = stat_fn(kfd_path)
    except OSError as error:
        raise RuntimeError(f"{kfd_path} is missing or inaccessible: {error}") from error
    if not stat.S_ISCHR(kfd_stat.st_mode) or not access_fn(kfd_path, os.R_OK | os.W_OK):
        raise RuntimeError(f"{kfd_path} must be an accessible character device")

    fuser = which_fn("fuser")
    if fuser is None:
        raise RuntimeError("fuser is required to verify exclusive /dev/kfd ownership")
    try:
        ownership = run_fn(
            [fuser, str(kfd_path)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(
            f"could not verify exclusive {kfd_path} ownership with fuser"
        ) from error
    ownership_output = " ".join(
        part.strip() for part in (ownership.stdout, ownership.stderr) if part.strip()
    )
    if ownership.returncode == 0:
        raise RuntimeError(
            f"refusing ROCm work while {kfd_path} is already owned: "
            f"{ownership_output or 'owner reported without a PID'}"
        )
    if ownership.returncode != 1 or ownership_output:
        raise RuntimeError(
            f"could not verify exclusive {kfd_path} ownership with fuser: "
            f"{ownership_output or f'return code {ownership.returncode}'}"
        )

    return {
        "amd_cards": amd_cards,
        "connected_amd_connectors": connected_connectors,
        "kfd_path": str(kfd_path),
        "kfd_accessible": True,
        "kfd_unowned": True,
    }


def amdgpu_fatal_events_since_boot(*, run_fn: Any = subprocess.run) -> list[str]:
    """Return fatal AMDGPU journal lines from the current boot, or fail closed."""
    try:
        result = run_fn(
            ["journalctl", "-k", "-b", "--no-pager", "-o", "short-iso"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "journalctl is required to verify a clean AMDGPU boot"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "timed out while verifying the AMDGPU boot journal"
        ) from error
    if result.returncode != 0:
        detail = " ".join(
            part.strip() for part in (result.stdout, result.stderr) if part.strip()
        )
        raise RuntimeError(
            "could not verify the AMDGPU boot journal: "
            f"{detail or f'return code {result.returncode}'}"
        )

    matches = []
    for line in result.stdout.splitlines():
        if AMDGPU_FATAL_PATTERN.search(line):
            matches.append(line.strip())
    return matches


def require_clean_amdgpu_boot(*, run_fn: Any = subprocess.run) -> dict[str, Any]:
    """Return a manifest fragment or require a reboot after a fatal event."""
    events = amdgpu_fatal_events_since_boot(run_fn=run_fn)
    if events:
        preview = " | ".join(events[-3:])
        raise RuntimeError(
            "refusing ROCm work because this boot contains a fatal AMDGPU event; "
            f"reboot before retrying: {preview}"
        )
    return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}


@contextmanager
def guarded_qwen35_rocm_process(
    *, runtime_dir: Path | None = None
) -> Iterator[dict[str, Any]]:
    """Hold the shared lock and complete fail-closed hardware preflight."""
    launch_lock = acquire_qwen35_rocm_launch_lock(runtime_dir=runtime_dir)
    try:
        yield {
            **require_clean_amdgpu_boot(),
            **require_headless_unowned_amdgpu(),
        }
    finally:
        os.close(launch_lock)


def main() -> int:
    try:
        result = require_clean_amdgpu_boot()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
