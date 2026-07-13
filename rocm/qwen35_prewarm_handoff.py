#!/usr/bin/env python3
"""Prove an idle AMDGPU handoff around Qwen3.5 startup prewarm.

``capture`` records the exact, private idle baseline before the compile-only
prewarm child starts. ``settle`` reopens that artifact without following
symlinks and polls read-only kernel interfaces until the child has released
KFD and the render node, VRAM and GTT are no higher than their exact baseline,
the PCI device is runtime-suspended, and the current boot journal is clean.

This helper imports no JAX or ROCm userspace library and opens no accelerator
device. It does not compile, execute, capture, or replay GPU work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, TextIO

_AMD_VENDOR_ID = "0x1002"
_GPU_DEVICE_ID = "0x744c"
_SCHEMA_VERSION = 1
_DEFAULT_TIMEOUT_SECONDS = 120.0
_DEFAULT_POLL_INTERVAL_SECONDS = 1.0
_REQUIRED_STABLE_SAMPLES = 3
_BOOT_ID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
_PCI_BDF_PATTERN = re.compile(
    r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]"
)
_UNSIGNED_DECIMAL_PATTERN = re.compile(r"0|[1-9][0-9]*")
_DEVICE_NUMBER_PATTERN = re.compile(
    r"(?:0|[1-9][0-9]*):(?:0|[1-9][0-9]*)"
)
_NOFOLLOW_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_BASELINE_RECORD_FIELDS = frozenset(
    {
        "accelerator_device_opened",
        "amdgpu_boot_clean",
        "baseline",
        "command_buffer_used",
        "device",
        "fatal_amdgpu_events",
        "graph_api_used",
        "record_type",
        "release_contract",
        "schema_version",
        "script_sha256",
        "status",
        "timestamp",
    }
)
_BASELINE_FIELDS = frozenset(
    {
        "boot_id",
        "device_identity",
        "gtt_used_bytes",
        "kfd_owner_pids",
        "render_owner_pids",
        "runtime_status",
        "vram_used_bytes",
    }
)


class HandoffError(RuntimeError):
    """A fail-closed baseline or handoff validation error."""


@dataclass(frozen=True)
class DevicePaths:
    drm_card: str
    vendor_id: str
    device_id: str
    pci_bdf: str
    pci_sysfs_path: str
    drm_sysfs_path: str
    drm_sysfs_dev: str
    render_sysfs_path: str
    render_sysfs_dev: str
    device_root: Path
    drm_node: Path
    kfd_node: Path
    render_node: Path
    drm_node_identity: dict[str, str]
    kfd_node_identity: dict[str, str]
    render_node_identity: dict[str, str]
    vram_used: Path
    gtt_used: Path
    runtime_status: Path

    def identity(self) -> dict[str, str]:
        return {
            "drm_card": self.drm_card,
            "vendor_id": self.vendor_id,
            "device_id": self.device_id,
            "pci_bdf": self.pci_bdf,
            "pci_sysfs_path": self.pci_sysfs_path,
            "drm_sysfs_path": self.drm_sysfs_path,
            "drm_sysfs_dev": self.drm_sysfs_dev,
            "render_sysfs_path": self.render_sysfs_path,
            "render_sysfs_dev": self.render_sysfs_dev,
            "drm_node": self.drm_node_identity,
            "kfd_node": self.kfd_node_identity,
            "render_node": self.render_node_identity,
        }


OwnerProbe = Callable[[Path], tuple[int, ...]]
BootValidator = Callable[[], dict[str, Any]]
NodeIdentityProbe = Callable[[Path], dict[str, str]]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _emit(record: dict[str, Any], output: TextIO) -> None:
    output.write(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    output.flush()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number is forbidden: {value}")
    return parsed


def _strict_json_loads(value: str) -> Any:
    return json.loads(
        value,
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
        parse_float=_parse_finite_json_float,
    )


def _release_contract() -> dict[str, float | int | str]:
    return {
        "timeout_seconds": _DEFAULT_TIMEOUT_SECONDS,
        "poll_interval_seconds": _DEFAULT_POLL_INTERVAL_SECONDS,
        "required_consecutive_ready_samples": _REQUIRED_STABLE_SAMPLES,
        "vram_tolerance_bytes": 0,
        "gtt_tolerance_bytes": 0,
        "runtime_status_required": "suspended",
    }


def _validate_private_parent(path: Path) -> tuple[Path, int]:
    if not path.is_absolute():
        raise HandoffError("handoff artifact path must be absolute")
    parent = path.parent
    if parent != Path(os.path.realpath(parent)):
        raise HandoffError("handoff artifact parent must not contain a symlink")
    try:
        parent_fd = os.open(parent, _NOFOLLOW_DIRECTORY)
    except OSError as error:
        raise HandoffError(f"cannot open handoff artifact parent: {error}") from error
    metadata = os.fstat(parent_fd)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        os.close(parent_fd)
        raise HandoffError(
            "handoff artifact parent must be a real mode-0700 directory owned by the current user"
        )
    return parent, parent_fd


def _open_new_private_artifact(path: Path) -> int:
    _parent, parent_fd = _validate_private_parent(path)
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
        ):
            os.close(descriptor)
            os.unlink(path.name, dir_fd=parent_fd)
            raise HandoffError("new handoff artifact is not a private regular file")
        return descriptor
    finally:
        os.close(parent_fd)


def _read_descriptor(descriptor: int, maximum_bytes: int = 1024 * 1024) -> bytes:
    metadata = os.fstat(descriptor)
    if metadata.st_size > maximum_bytes:
        raise HandoffError("existing handoff artifact is unexpectedly large")
    os.lseek(descriptor, 0, os.SEEK_SET)
    remaining = metadata.st_size
    chunks: list[bytes] = []
    while remaining:
        chunk = os.read(descriptor, min(remaining, 65536))
        if not chunk:
            raise HandoffError("existing handoff artifact ended unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _open_existing_private_artifact(path: Path) -> tuple[int, dict[str, Any]]:
    _parent, parent_fd = _validate_private_parent(path)
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                path.name,
                os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        except OSError as error:
            raise HandoffError(f"cannot reopen handoff artifact: {error}") from error
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 1
            ):
                raise HandoffError(
                    "existing handoff artifact is not a private regular file"
                )
            payload = _read_descriptor(descriptor)
            if not payload.endswith(b"\n"):
                raise HandoffError(
                    "existing handoff artifact is not newline terminated"
                )
            try:
                lines = payload.decode("utf-8").splitlines()
                records = [_strict_json_loads(line) for line in lines]
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                RecursionError,
                ValueError,
            ) as error:
                raise HandoffError("existing handoff artifact is malformed") from error
            if (
                len(records) != 1
                or not isinstance(records[0], dict)
                or records[0].get("record_type") != "prewarm_handoff_baseline"
            ):
                raise HandoffError(
                    "handoff settle requires exactly one baseline record"
                )
            return descriptor, records[0]
        except BaseException:
            os.close(descriptor)
            raise
    finally:
        os.close(parent_fd)


def _read_exact_text(path: Path, label: str) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise HandoffError(f"cannot read {label}: {error}") from error
    if not value:
        raise HandoffError(f"{label} is empty")
    return value


def _read_counter(path: Path, label: str) -> int:
    value = _read_exact_text(path, label)
    if _UNSIGNED_DECIMAL_PATTERN.fullmatch(value) is None:
        raise HandoffError(f"{label} is not an unsigned decimal integer")
    return int(value)


def _read_boot_id(path: Path) -> str:
    value = _read_exact_text(path, "current boot ID")
    if _BOOT_ID_PATTERN.fullmatch(value) is None:
        raise HandoffError("current boot ID is malformed")
    return value


def _character_node_identity(
    path: Path,
    *,
    stat_fn: Callable[[Path], os.stat_result] = os.lstat,
    access_fn: Callable[[Path, int], bool] = os.access,
    sys_dev_char_root: Path = Path("/sys/dev/char"),
) -> dict[str, str]:
    try:
        metadata = stat_fn(path)
    except OSError as error:
        raise HandoffError(
            f"accelerator node {path} is inaccessible: {error}"
        ) from error
    if not stat.S_ISCHR(metadata.st_mode) or not access_fn(path, os.R_OK | os.W_OK):
        raise HandoffError(
            f"accelerator node {path} is not an accessible character device"
        )
    device_number = f"{os.major(metadata.st_rdev)}:{os.minor(metadata.st_rdev)}"
    sysfs_link = sys_dev_char_root / device_number
    try:
        sysfs_target = sysfs_link.resolve(strict=True)
    except OSError as error:
        raise HandoffError(
            f"accelerator node {path} has no resolvable sysfs character-device identity"
        ) from error
    sysfs_device_number = _read_exact_text(
        sysfs_target / "dev", f"sysfs character-device number for {path}"
    )
    if sysfs_device_number != device_number:
        raise HandoffError(
            f"accelerator node {path} rdev {device_number} does not match sysfs {sysfs_device_number}"
        )
    return {
        "path": str(path),
        "rdev": device_number,
        "sysfs_dev": sysfs_device_number,
        "sysfs_target": str(sysfs_target),
    }


def _validated_node_identity(
    path: Path, node_identity_probe: NodeIdentityProbe
) -> dict[str, str]:
    identity = node_identity_probe(path)
    expected_fields = {"path", "rdev", "sysfs_dev", "sysfs_target"}
    if (
        not isinstance(identity, dict)
        or set(identity) != expected_fields
        or any(type(identity[field]) is not str for field in expected_fields)
        or identity["path"] != str(path)
        or _DEVICE_NUMBER_PATTERN.fullmatch(identity["rdev"]) is None
        or identity["sysfs_dev"] != identity["rdev"]
        or not identity["sysfs_target"].startswith("/")
    ):
        raise HandoffError(f"accelerator node {path} returned an invalid identity")
    return identity


def _discover_device(
    *,
    drm_root: Path = Path("/sys/class/drm"),
    dev_root: Path = Path("/dev"),
    node_identity_probe: NodeIdentityProbe = _character_node_identity,
) -> DevicePaths:
    matches: list[tuple[Path, str]] = []
    for card in sorted(drm_root.glob("card[0-9]*")):
        if re.fullmatch(r"card[0-9]+", card.name) is None:
            continue
        try:
            vendor = (card / "device" / "vendor").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if vendor != _AMD_VENDOR_ID:
            continue
        device_id = _read_exact_text(
            card / "device" / "device", f"{card.name} PCI device ID"
        )
        matches.append((card, device_id))
    if len(matches) != 1 or matches[0][1] != _GPU_DEVICE_ID:
        observed = [(card.name, device_id) for card, device_id in matches]
        raise HandoffError(
            "handoff requires exactly one AMD DRM GPU with PCI device "
            f"{_GPU_DEVICE_ID}; observed {observed!r}"
        )

    card, device_id = matches[0]
    try:
        pci_root = (card / "device").resolve(strict=True)
        drm_sysfs_path = card.resolve(strict=True)
    except OSError as error:
        raise HandoffError("cannot resolve the AMD DRM/PCI sysfs identity") from error
    pci_bdf = pci_root.name
    if _PCI_BDF_PATTERN.fullmatch(pci_bdf) is None:
        raise HandoffError(f"resolved AMD PCI BDF is malformed: {pci_bdf!r}")
    if drm_sysfs_path.parent != pci_root / "drm":
        raise HandoffError("AMD DRM card is not rooted beneath the resolved PCI device")
    vendor_id = _read_exact_text(pci_root / "vendor", "resolved AMD PCI vendor ID")
    resolved_device_id = _read_exact_text(
        pci_root / "device", "resolved AMD PCI device ID"
    )
    if vendor_id != _AMD_VENDOR_ID or resolved_device_id != device_id:
        raise HandoffError("resolved AMD PCI identity changed during discovery")
    drm_sysfs_dev = _read_exact_text(card / "dev", f"{card.name} sysfs dev")
    render_names = sorted(
        path.name
        for path in (card / "device" / "drm").glob("renderD[0-9]*")
        if re.fullmatch(r"renderD[0-9]+", path.name) is not None
    )
    if len(render_names) != 1:
        raise HandoffError(
            f"handoff requires exactly one render node for {card.name}; observed {render_names!r}"
        )

    render_sysfs_path = (pci_root / "drm" / render_names[0]).resolve(strict=True)
    if render_sysfs_path.parent != pci_root / "drm":
        raise HandoffError("AMD render node is not rooted beneath the resolved PCI device")
    render_sysfs_dev = _read_exact_text(
        render_sysfs_path / "dev", f"{render_names[0]} sysfs dev"
    )

    drm_node = dev_root / "dri" / card.name
    kfd_node = dev_root / "kfd"
    render_node = dev_root / "dri" / render_names[0]
    drm_node_identity = _validated_node_identity(drm_node, node_identity_probe)
    kfd_node_identity = _validated_node_identity(kfd_node, node_identity_probe)
    render_node_identity = _validated_node_identity(render_node, node_identity_probe)
    if drm_node_identity["rdev"] != drm_sysfs_dev:
        raise HandoffError("DRM card-node rdev does not match its PCI sysfs dev")
    if render_node_identity["rdev"] != render_sysfs_dev:
        raise HandoffError("render-node rdev does not match its PCI sysfs dev")
    device_root = card / "device"
    return DevicePaths(
        drm_card=card.name,
        vendor_id=vendor_id,
        device_id=device_id,
        pci_bdf=pci_bdf,
        pci_sysfs_path=str(pci_root),
        drm_sysfs_path=str(drm_sysfs_path),
        drm_sysfs_dev=drm_sysfs_dev,
        render_sysfs_path=str(render_sysfs_path),
        render_sysfs_dev=render_sysfs_dev,
        device_root=device_root,
        drm_node=drm_node,
        kfd_node=kfd_node,
        render_node=render_node,
        drm_node_identity=drm_node_identity,
        kfd_node_identity=kfd_node_identity,
        render_node_identity=render_node_identity,
        vram_used=device_root / "mem_info_vram_used",
        gtt_used=device_root / "mem_info_gtt_used",
        runtime_status=device_root / "power" / "runtime_status",
    )


def _fuser_owner_pids(
    path: Path,
    *,
    which_fn: Callable[[str], str | None] = shutil.which,
    run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[int, ...]:
    executable = which_fn("fuser")
    if executable is None:
        raise HandoffError("fuser is required for prewarm handoff ownership checks")
    try:
        result = run_fn(
            [executable, str(path)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise HandoffError(f"cannot inspect owners of {path}") from error
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode == 1 and not stdout and not stderr:
        return ()
    if result.returncode != 0:
        raise HandoffError(
            f"fuser returned an indeterminate result for {path}: return code {result.returncode}"
        )
    # procps fuser writes PIDs to stdout and the path label to stderr. Parsing
    # the combined text would misread the ``128`` in ``renderD128`` as a PID.
    if stderr != f"{path}:" or re.fullmatch(r"[0-9]+(?:\s+[0-9]+)*", stdout) is None:
        raise HandoffError(f"fuser returned malformed owner output for {path}")
    pids = tuple(sorted({int(value) for value in stdout.split()}))
    if not pids or pids[0] <= 0:
        raise HandoffError(f"fuser returned an invalid owner PID for {path}")
    return pids


def _require_clean_boot() -> dict[str, Any]:
    try:
        from rocm.amdgpu_safety import require_clean_amdgpu_boot
    except ModuleNotFoundError:
        from amdgpu_safety import require_clean_amdgpu_boot

    return require_clean_amdgpu_boot()


def _validated_boot_manifest(boot_validator: BootValidator) -> dict[str, Any]:
    manifest = boot_validator()
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"amdgpu_boot_clean", "fatal_amdgpu_events"}
        or manifest.get("amdgpu_boot_clean") is not True
        or manifest.get("fatal_amdgpu_events") != []
    ):
        raise HandoffError(
            f"clean-boot validator returned an invalid manifest: {manifest!r}"
        )
    return manifest


def _snapshot(
    paths: DevicePaths,
    *,
    boot_id_path: Path,
    owner_probe: OwnerProbe,
) -> dict[str, Any]:
    runtime_status = _read_exact_text(paths.runtime_status, "AMDGPU runtime status")
    if runtime_status not in {"active", "suspending", "suspended", "resuming"}:
        raise HandoffError(f"AMDGPU runtime status is invalid: {runtime_status!r}")
    return {
        "boot_id": _read_boot_id(boot_id_path),
        "device_identity": paths.identity(),
        "vram_used_bytes": _read_counter(paths.vram_used, "VRAM used"),
        "gtt_used_bytes": _read_counter(paths.gtt_used, "GTT used"),
        "runtime_status": runtime_status,
        "kfd_owner_pids": list(owner_probe(paths.kfd_node)),
        "render_owner_pids": list(owner_probe(paths.render_node)),
    }


def _ready_checks(
    snapshot: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, bool]:
    return {
        "same_boot": snapshot["boot_id"] == baseline["boot_id"],
        "same_device_identity": (
            snapshot["device_identity"] == baseline["device_identity"]
        ),
        "kfd_unowned": snapshot["kfd_owner_pids"] == [],
        "render_unowned": snapshot["render_owner_pids"] == [],
        # A lower counter also proves that the prewarm child retained no memory.
        # No positive tolerance is accepted: even one residual byte must settle.
        "vram_no_higher_than_exact_baseline": (
            snapshot["vram_used_bytes"] <= baseline["vram_used_bytes"]
        ),
        "gtt_no_higher_than_exact_baseline": (
            snapshot["gtt_used_bytes"] <= baseline["gtt_used_bytes"]
        ),
        "runtime_suspended": snapshot["runtime_status"] == "suspended",
    }


def _stable_snapshot(
    expected_paths: DevicePaths,
    *,
    drm_root: Path,
    dev_root: Path,
    boot_id_path: Path,
    node_identity_probe: NodeIdentityProbe,
    owner_probe: OwnerProbe,
) -> dict[str, Any]:
    """Read one snapshot bracketed by exact PCI/DRM node-identity checks."""
    expected_identity = expected_paths.identity()
    before = _discover_device(
        drm_root=drm_root,
        dev_root=dev_root,
        node_identity_probe=node_identity_probe,
    )
    if before.identity() != expected_identity:
        raise HandoffError("AMDGPU device identity changed before the handoff snapshot")
    snapshot = _snapshot(
        before, boot_id_path=boot_id_path, owner_probe=owner_probe
    )
    after = _discover_device(
        drm_root=drm_root,
        dev_root=dev_root,
        node_identity_probe=node_identity_probe,
    )
    if after.identity() != expected_identity:
        raise HandoffError("AMDGPU device identity changed during the handoff snapshot")
    return snapshot


def capture_baseline(
    output_path: Path,
    *,
    drm_root: Path = Path("/sys/class/drm"),
    dev_root: Path = Path("/dev"),
    boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id"),
    node_identity_probe: NodeIdentityProbe = _character_node_identity,
    owner_probe: OwnerProbe = _fuser_owner_pids,
    boot_validator: BootValidator = _require_clean_boot,
) -> dict[str, Any]:
    paths = _discover_device(
        drm_root=drm_root,
        dev_root=dev_root,
        node_identity_probe=node_identity_probe,
    )
    clean_boot = _validated_boot_manifest(boot_validator)
    snapshot = _stable_snapshot(
        paths,
        drm_root=drm_root,
        dev_root=dev_root,
        boot_id_path=boot_id_path,
        node_identity_probe=node_identity_probe,
        owner_probe=owner_probe,
    )
    if snapshot["kfd_owner_pids"] or snapshot["render_owner_pids"]:
        raise HandoffError("cannot capture baseline while an accelerator node is owned")
    if snapshot["runtime_status"] != "suspended":
        raise HandoffError(
            "cannot capture baseline until the AMDGPU device is runtime-suspended"
        )

    record = {
        "record_type": "prewarm_handoff_baseline",
        "schema_version": _SCHEMA_VERSION,
        "timestamp": _utc_now(),
        "status": "passed",
        "device": paths.identity(),
        "baseline": snapshot,
        "release_contract": _release_contract(),
        "script_sha256": _script_sha256(),
        "graph_api_used": False,
        "command_buffer_used": False,
        "accelerator_device_opened": False,
        **clean_boot,
    }
    descriptor = _open_new_private_artifact(output_path)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        _emit(record, output)
        os.fsync(output.fileno())
    return record


def _validate_baseline_record(
    record: dict[str, Any], paths: DevicePaths, boot_id_path: Path
) -> dict[str, Any]:
    if set(record) != _BASELINE_RECORD_FIELDS:
        raise HandoffError("baseline record has an unexpected top-level schema")
    if (
        type(record.get("schema_version")) is not int
        or record["schema_version"] != _SCHEMA_VERSION
    ):
        raise HandoffError("baseline schema version mismatch")
    if not isinstance(record.get("timestamp"), str) or not record["timestamp"]:
        raise HandoffError("baseline timestamp is invalid")
    if record.get("script_sha256") != _script_sha256():
        raise HandoffError("handoff helper changed after baseline capture")
    if (
        record.get("record_type") != "prewarm_handoff_baseline"
        or record.get("status") != "passed"
        or record.get("device") != paths.identity()
    ):
        raise HandoffError("baseline device identity does not match current hardware")
    if (
        record.get("graph_api_used") is not False
        or record.get("command_buffer_used") is not False
        or record.get("accelerator_device_opened") is not False
        or record.get("amdgpu_boot_clean") is not True
        or record.get("fatal_amdgpu_events") != []
    ):
        raise HandoffError("baseline safety evidence is invalid")
    expected_contract = _release_contract()
    contract = record.get("release_contract")
    if (
        not isinstance(contract, dict)
        or set(contract) != set(expected_contract)
        or any(
            type(contract[name]) is not type(expected) or contract[name] != expected
            for name, expected in expected_contract.items()
        )
    ):
        raise HandoffError("baseline release contract is invalid")
    baseline = record.get("baseline")
    if not isinstance(baseline, dict) or set(baseline) != _BASELINE_FIELDS:
        raise HandoffError("baseline payload has an unexpected schema")
    if (
        type(baseline["boot_id"]) is not str
        or type(baseline["device_identity"]) is not dict
        or type(baseline["vram_used_bytes"]) is not int
        or type(baseline["gtt_used_bytes"]) is not int
        or type(baseline["runtime_status"]) is not str
        or type(baseline["kfd_owner_pids"]) is not list
        or type(baseline["render_owner_pids"]) is not list
    ):
        raise HandoffError("baseline payload has invalid field types")
    if (
        _BOOT_ID_PATTERN.fullmatch(baseline["boot_id"]) is None
        or baseline["device_identity"] != paths.identity()
        or baseline["vram_used_bytes"] < 0
        or baseline["gtt_used_bytes"] < 0
        or baseline["runtime_status"] != "suspended"
        or baseline["kfd_owner_pids"] != []
        or baseline["render_owner_pids"] != []
    ):
        raise HandoffError("baseline was not an idle, unowned device")
    if baseline["boot_id"] != _read_boot_id(boot_id_path):
        raise HandoffError("current boot changed after baseline capture")
    return baseline


def settle_handoff(
    output_path: Path,
    *,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
    drm_root: Path = Path("/sys/class/drm"),
    dev_root: Path = Path("/dev"),
    boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id"),
    node_identity_probe: NodeIdentityProbe = _character_node_identity,
    owner_probe: OwnerProbe = _fuser_owner_pids,
    boot_validator: BootValidator = _require_clean_boot,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise HandoffError("handoff timeout must be finite and positive")
    if timeout_seconds > _DEFAULT_TIMEOUT_SECONDS:
        raise HandoffError("handoff timeout must be no greater than 120 seconds")
    if (
        not math.isfinite(poll_interval_seconds)
        or poll_interval_seconds != _DEFAULT_POLL_INTERVAL_SECONDS
        or poll_interval_seconds > timeout_seconds
    ):
        raise HandoffError(
            "handoff poll interval must be exactly 1 second and no larger than the timeout"
        )

    descriptor, baseline_record = _open_existing_private_artifact(output_path)
    with os.fdopen(descriptor, "a", encoding="utf-8") as output:
        terminal_recorded = False
        try:
            paths = _discover_device(
                drm_root=drm_root,
                dev_root=dev_root,
                node_identity_probe=node_identity_probe,
            )
            baseline = _validate_baseline_record(baseline_record, paths, boot_id_path)
            _validated_boot_manifest(boot_validator)
            start = monotonic_fn()
            sample_index = 0
            last_snapshot: dict[str, Any] | None = None
            last_checks: dict[str, bool] | None = None
            ready_streak = 0
            while True:
                elapsed = monotonic_fn() - start
                snapshot = _stable_snapshot(
                    paths,
                    drm_root=drm_root,
                    dev_root=dev_root,
                    boot_id_path=boot_id_path,
                    node_identity_probe=node_identity_probe,
                    owner_probe=owner_probe,
                )
                checks = _ready_checks(snapshot, baseline)
                last_snapshot = snapshot
                last_checks = checks
                ready_streak = ready_streak + 1 if all(checks.values()) else 0
                _emit(
                    {
                        "record_type": "prewarm_handoff_sample",
                        "schema_version": _SCHEMA_VERSION,
                        "timestamp": _utc_now(),
                        "sample_index": sample_index,
                        "elapsed_seconds": elapsed,
                        "snapshot": snapshot,
                        "checks": checks,
                        "ready_streak": ready_streak,
                        "required_ready_streak": _REQUIRED_STABLE_SAMPLES,
                        "status": (
                            "ready_candidate" if all(checks.values()) else "waiting"
                        ),
                        "accelerator_device_opened": False,
                    },
                    output,
                )
                sample_index += 1

                if ready_streak >= _REQUIRED_STABLE_SAMPLES:
                    clean_boot = _validated_boot_manifest(boot_validator)
                    final_snapshot = _stable_snapshot(
                        paths,
                        drm_root=drm_root,
                        dev_root=dev_root,
                        boot_id_path=boot_id_path,
                        node_identity_probe=node_identity_probe,
                        owner_probe=owner_probe,
                    )
                    final_checks = _ready_checks(final_snapshot, baseline)
                    if all(final_checks.values()):
                        result = {
                            "record_type": "prewarm_handoff_complete",
                            "schema_version": _SCHEMA_VERSION,
                            "timestamp": _utc_now(),
                            "status": "passed",
                            "elapsed_seconds": monotonic_fn() - start,
                            "sample_count": sample_index,
                            "final_ready_streak": ready_streak,
                            "baseline": baseline,
                            "final_snapshot": final_snapshot,
                            "checks": final_checks,
                            "vram_tolerance_bytes": 0,
                            "gtt_tolerance_bytes": 0,
                            "graph_api_used": False,
                            "command_buffer_used": False,
                            "accelerator_device_opened": False,
                            **clean_boot,
                        }
                        _emit(result, output)
                        os.fsync(output.fileno())
                        terminal_recorded = True
                        return result
                    ready_streak = 0

                if elapsed >= timeout_seconds:
                    result = {
                        "record_type": "prewarm_handoff_timeout",
                        "schema_version": _SCHEMA_VERSION,
                        "timestamp": _utc_now(),
                        "status": "failed",
                        "elapsed_seconds": elapsed,
                        "sample_count": sample_index,
                        "final_ready_streak": ready_streak,
                        "last_snapshot": last_snapshot,
                        "checks": last_checks,
                        "accelerator_device_opened": False,
                    }
                    _emit(result, output)
                    os.fsync(output.fileno())
                    terminal_recorded = True
                    raise HandoffError(
                        f"AMDGPU did not return to its exact idle baseline within {timeout_seconds:g} seconds"
                    )
                remaining = timeout_seconds - elapsed
                sleep_fn(min(poll_interval_seconds, remaining))
        except BaseException as error:
            if not terminal_recorded:
                _emit(
                    {
                        "record_type": "prewarm_handoff_error",
                        "schema_version": _SCHEMA_VERSION,
                        "timestamp": _utc_now(),
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "message": str(error),
                        "accelerator_device_opened": False,
                    },
                    output,
                )
                os.fsync(output.fileno())
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--output", required=True, type=Path)
    settle = subparsers.add_parser("settle")
    settle.add_argument("--output", required=True, type=Path)
    settle.add_argument(
        "--timeout-seconds", type=float, default=_DEFAULT_TIMEOUT_SECONDS
    )
    settle.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=_DEFAULT_POLL_INTERVAL_SECONDS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.operation == "capture":
            capture_baseline(args.output)
        else:
            settle_handoff(
                args.output,
                timeout_seconds=args.timeout_seconds,
                poll_interval_seconds=args.poll_interval_seconds,
            )
    except (HandoffError, OSError, subprocess.SubprocessError) as error:
        print(f"Qwen3.5 prewarm handoff failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
