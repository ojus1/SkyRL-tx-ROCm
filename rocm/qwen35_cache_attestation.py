#!/usr/bin/env python3
"""Exact, graph-free Qwen3.5 persistent-cache attestation helpers.

This module is deliberately stdlib-only.  In particular, importing it never
imports JAX or opens an accelerator.  Callers pass a JAX-monitoring compatible
object only after their existing GPU safety gates have succeeded.

JAX 0.10.2's bounded local LRU rewrites an eight-byte ``<key>-atime`` file on
every successful ``get(key)``.  Pairing that write with the public cache-hit
events and byte-stable ``<key>-cache`` files gives us exact-key attribution
without reaching into JAX private runtime state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCHEMA_NAME = "skyrl.qwen35.persistent-cache-attestation"
SCHEMA_VERSION = 1
PREWARM_SEED_HIT = "strict_aot_seed_hit_v1"
PREWARM_SEED_MISS = "strict_aot_seed_miss_v1"
RUNTIME_HIT_KIND = "strict_aot_t64_persistent_cache_hit_v1"
HIT_EVIDENCE_LIMIT = (
    "proves one exact AOT persistent-cache key lookup/deserialization; it does not "
    "invoke the executable or seed normal PjitFunction first-call dispatch"
)
MISS_EVIDENCE_LIMIT = (
    "proves one exact paired AOT persistent-cache executable/atime population; "
    "auxiliary/autotune mutations are permitted and only their before/after "
    "manifest hashes are bound; it does not invoke the executable or seed normal "
    "PjitFunction first-call dispatch"
)
REQUIREMENT = "required-v1"
MODEL = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
COMPILE_TARGET = "train_bucket_forward_backward_accumulate"
BUCKET = 64
BATCH_SIZE = 1
DISABLE_COMMAND_BUFFERS = "--xla_gpu_enable_command_buffer="

_CACHE_SUFFIX = "-cache"
_ATIME_SUFFIX = "-atime"
_AUTOTUNE_DIRECTORY = "xla_gpu_per_fusion_autotune_cache_dir"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CACHE_KEY_PATTERN = re.compile(r"jit_forward_backward_and_accumulate-[0-9a-f]{64}")
_BOOT_ID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
_PCI_BDF_PATTERN = re.compile(
    r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]"
)
_DEVICE_NUMBER_PATTERN = re.compile(
    r"(?:0|[1-9][0-9]*):(?:0|[1-9][0-9]*)"
)
_DRM_CARD_PATTERN = re.compile(r"card(?:0|[1-9][0-9]*)")
_EVENT_REQUEST = "/jax/compilation_cache/compile_requests_use_cache"
_EVENT_HIT = "/jax/compilation_cache/cache_hits"
_EVENT_MISS = "/jax/compilation_cache/cache_misses"
_DURATION_SAVED = "/jax/compilation_cache/compile_time_saved_sec"
_DURATION_RETRIEVAL = "/jax/compilation_cache/cache_retrieval_time_sec"
_EVENT_NAMES = frozenset({_EVENT_REQUEST, _EVENT_HIT, _EVENT_MISS})
_DURATION_NAMES = frozenset({_DURATION_SAVED, _DURATION_RETRIEVAL})
_MAX_PREWARM_ARTIFACT_BYTES = 16 * 1024 * 1024
_MAX_HANDOFF_ARTIFACT_BYTES = 4 * 1024 * 1024


class CacheAttestationError(RuntimeError):
    """A cache snapshot, artifact, or exact-hit proof was not strict."""


def _is_sha256(value: object) -> bool:
    return type(value) is str and _SHA256_PATTERN.fullmatch(value) is not None


def _sibling_source_sha256(name: str) -> str:
    parent, parent_fd = _canonical_owned_directory(
        Path(__file__).resolve().parent, "cache-attestation source directory"
    )
    try:
        fingerprint, _payload = _read_stable_regular_at(
            parent_fd,
            name,
            label=f"cache-attestation sibling source {name}",
            retain_payload=False,
        )
    finally:
        os.close(parent_fd)
    if parent / name != Path(__file__).resolve().with_name(name):
        raise CacheAttestationError("cache-attestation sibling source escaped")
    return fingerprint.sha256


@dataclass(frozen=True)
class FileFingerprint:
    name: str
    device: int
    inode: int
    mode: int
    uid: int
    link_count: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "uid": self.uid,
            "link_count": self.link_count,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class CachePair:
    key: str
    executable: FileFingerprint
    logical_atime: FileFingerprint
    logical_atime_ns: int


@dataclass(frozen=True)
class CacheSnapshot:
    cache_path: str
    pairs: Mapping[str, CachePair]
    executable_manifest_sha256: str
    logical_atime_manifest_sha256: str
    auxiliary_manifest_sha256: str


@dataclass(frozen=True)
class MonitoringTrace:
    ordered_events: tuple[str, ...]
    events: Mapping[str, int]
    durations: Mapping[str, tuple[float, ...]]
    schema_issues: tuple[str, ...]


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise CacheAttestationError(
            f"value is not canonical JSON data: {error}"
        ) from error


def canonical_json_sha256(value: object, *, domain: str) -> str:
    if not domain or "\x00" in domain:
        raise ValueError("canonical JSON hash domain must be nonempty and NUL-free")
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\x00")
    digest.update(_canonical_json_bytes(value))
    return digest.hexdigest()


def _metadata_tuple(metadata: os.stat_result) -> tuple[int, ...]:
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


def _fingerprint_matches_metadata(
    fingerprint: FileFingerprint, metadata: os.stat_result
) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and fingerprint.device == metadata.st_dev
        and fingerprint.inode == metadata.st_ino
        and fingerprint.mode == stat.S_IMODE(metadata.st_mode)
        and fingerprint.uid == metadata.st_uid
        and fingerprint.link_count == metadata.st_nlink
        and fingerprint.size_bytes == metadata.st_size
        and fingerprint.mtime_ns == metadata.st_mtime_ns
        and fingerprint.ctime_ns == metadata.st_ctime_ns
    )


def _read_stable_regular_at(
    directory_fd: int,
    name: str,
    *,
    label: str,
    maximum_bytes: int | None = None,
    retain_payload: bool = True,
) -> tuple[FileFingerprint, bytes]:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as error:
        raise CacheAttestationError(f"cannot open {label}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
        ):
            raise CacheAttestationError(
                f"{label} must be an owned, singly linked regular file"
            )
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise CacheAttestationError(
                f"{label} exceeds its {maximum_bytes}-byte size limit"
            )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        payload_size = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            payload_size += len(chunk)
            if maximum_bytes is not None and payload_size > maximum_bytes:
                raise CacheAttestationError(
                    f"{label} exceeded its {maximum_bytes}-byte size limit while read"
                )
            if retain_payload:
                chunks.append(chunk)
        after = os.fstat(descriptor)
        try:
            endpoint = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise CacheAttestationError(
                f"cannot revalidate {label} endpoint: {error}"
            ) from error
        if _metadata_tuple(before) != _metadata_tuple(after) or (
            _metadata_tuple(after) != _metadata_tuple(endpoint)
        ):
            raise CacheAttestationError(f"{label} changed while it was read")
        payload = b"".join(chunks) if retain_payload else b""
        if payload_size != after.st_size:
            raise CacheAttestationError(f"{label} size changed while it was read")
        fingerprint = FileFingerprint(
            name=name,
            device=after.st_dev,
            inode=after.st_ino,
            mode=stat.S_IMODE(after.st_mode),
            uid=after.st_uid,
            link_count=after.st_nlink,
            size_bytes=after.st_size,
            mtime_ns=after.st_mtime_ns,
            ctime_ns=after.st_ctime_ns,
            sha256=digest.hexdigest(),
        )
        return fingerprint, payload
    finally:
        os.close(descriptor)


def _canonical_owned_directory(path: Path, label: str) -> tuple[Path, int]:
    if not path.is_absolute():
        raise CacheAttestationError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CacheAttestationError(f"cannot inspect {label}: {error}") from error
    if resolved != path or stat.S_ISLNK(metadata.st_mode):
        raise CacheAttestationError(f"{label} must be canonical and non-symlinked")
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o022
    ):
        raise CacheAttestationError(
            f"{label} must be an owned directory not writable by group or other"
        )
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise CacheAttestationError(f"cannot open {label}: {error}") from error
    if _metadata_tuple(os.fstat(descriptor)) != _metadata_tuple(metadata):
        os.close(descriptor)
        raise CacheAttestationError(f"{label} changed while it was opened")
    return resolved, descriptor


def _snapshot_auxiliary_tree(root: Path) -> str:
    records: list[dict[str, object]] = []
    observed_nodes: list[str] = []
    directory_metadata: dict[str, tuple[int, ...]] = {}
    file_fingerprints: dict[str, FileFingerprint] = {}

    def raise_walk_error(error: OSError) -> None:
        raise CacheAttestationError(
            f"cannot traverse JAX per-fusion autotune cache: {error}"
        ) from error

    try:
        root_before = root.lstat()
    except FileNotFoundError:
        records.append({"kind": "root", "status": "absent"})
        digest = canonical_json_sha256(
            records, domain="skyrl-qwen35-cache-auxiliary-manifest-v1"
        )
        try:
            root.lstat()
        except FileNotFoundError:
            return digest
        except OSError as error:
            raise CacheAttestationError(
                f"cannot revalidate absent JAX autotune cache: {error}"
            ) from error
        raise CacheAttestationError(
            "JAX autotune cache appeared during its snapshot"
        )
    except OSError as error:
        raise CacheAttestationError(
            f"cannot inspect JAX per-fusion autotune cache: {error}"
        ) from error
    if stat.S_ISLNK(root_before.st_mode) or not stat.S_ISDIR(root_before.st_mode):
        raise CacheAttestationError(
            "JAX per-fusion autotune cache is not a real directory"
        )
    root_mode = stat.S_IMODE(root_before.st_mode)
    if (
        root_before.st_uid != os.getuid()
        or root_mode & 0o022
        or root_mode & 0o500 != 0o500
    ):
        raise CacheAttestationError(
            "JAX per-fusion autotune cache must be owned and private"
        )
    records.append(
        {
            "kind": "root",
            "status": "present",
            "name": ".",
            "device": root_before.st_dev,
            "inode": root_before.st_ino,
            "mode": stat.S_IMODE(root_before.st_mode),
            "uid": root_before.st_uid,
            "link_count": root_before.st_nlink,
            "mtime_ns": root_before.st_mtime_ns,
            "ctime_ns": root_before.st_ctime_ns,
        }
    )
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, onerror=raise_walk_error, followlinks=False
    ):
        directory_names.sort()
        file_names.sort()
        current = Path(directory)
        for name in directory_names:
            child = current / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise CacheAttestationError(
                    f"invalid object in JAX autotune cache: {child}"
                )
            child_mode = stat.S_IMODE(metadata.st_mode)
            if (
                metadata.st_uid != os.getuid()
                or child_mode & 0o022
                or child_mode & 0o500 != 0o500
            ):
                raise CacheAttestationError(
                    f"untrusted directory in JAX autotune cache: {child}"
                )
            relative = str(child.relative_to(root))
            observed_nodes.append(f"d:{relative}")
            directory_metadata[relative] = _metadata_tuple(metadata)
            records.append(
                {
                    "kind": "directory",
                    "name": relative,
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                    "mode": stat.S_IMODE(metadata.st_mode),
                    "uid": metadata.st_uid,
                    "link_count": metadata.st_nlink,
                    "mtime_ns": metadata.st_mtime_ns,
                    "ctime_ns": metadata.st_ctime_ns,
                }
            )
        directory_fd = os.open(current, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for name in file_names:
                observed_nodes.append(f"f:{(current / name).relative_to(root)}")
                fingerprint, _payload = _read_stable_regular_at(
                    directory_fd,
                    name,
                    label=f"JAX autotune cache entry {current / name}",
                    retain_payload=False,
                )
                record = fingerprint.as_dict()
                relative = str((current / name).relative_to(root))
                file_fingerprints[relative] = fingerprint
                record["name"] = relative
                record["kind"] = "file"
                records.append(record)
        finally:
            os.close(directory_fd)
    try:
        root_after = root.lstat()
    except OSError as error:
        raise CacheAttestationError(
            f"JAX autotune cache changed during its snapshot: {error}"
        ) from error
    if _metadata_tuple(root_before) != _metadata_tuple(root_after):
        raise CacheAttestationError("JAX autotune cache changed during its snapshot")
    final_nodes: list[str] = []
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, onerror=raise_walk_error, followlinks=False
    ):
        directory_names.sort()
        file_names.sort()
        current = Path(directory)
        for name in directory_names:
            child = current / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise CacheAttestationError(
                    f"invalid object in JAX autotune cache: {child}"
                )
            relative = str(child.relative_to(root))
            if directory_metadata.get(relative) != _metadata_tuple(metadata):
                raise CacheAttestationError(
                    f"JAX autotune cache directory changed: {child}"
                )
            final_nodes.append(f"d:{relative}")
        for name in file_names:
            child = current / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise CacheAttestationError(
                    f"invalid object in JAX autotune cache: {child}"
                )
            relative = str(child.relative_to(root))
            fingerprint = file_fingerprints.get(relative)
            if fingerprint is None or not _fingerprint_matches_metadata(
                fingerprint, metadata
            ):
                raise CacheAttestationError(
                    f"JAX autotune cache file changed: {child}"
                )
            final_nodes.append(f"f:{relative}")
    if observed_nodes != final_nodes:
        raise CacheAttestationError(
            "JAX autotune cache directory changed during its snapshot"
        )
    try:
        root_final = root.lstat()
    except OSError as error:
        raise CacheAttestationError(
            f"JAX autotune cache changed during final revalidation: {error}"
        ) from error
    if _metadata_tuple(root_before) != _metadata_tuple(root_final):
        raise CacheAttestationError("JAX autotune cache changed during its snapshot")
    return canonical_json_sha256(
        records, domain="skyrl-qwen35-cache-auxiliary-manifest-v1"
    )


def snapshot_cache(cache_path: Path) -> CacheSnapshot:
    """Hash all paired executable entries and their logical LRU atimes."""
    canonical, directory_fd = _canonical_owned_directory(
        cache_path, "JAX compilation cache"
    )
    try:
        names_before: list[str] = []
        auxiliary_present_before = False
        with os.scandir(canonical) as entries:
            for entry in entries:
                if entry.name == ".lockfile":
                    # filelock may create this transiently while JAX is active.
                    continue
                if entry.name == _AUTOTUNE_DIRECTORY:
                    if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                        raise CacheAttestationError(
                            "JAX autotune cache entry is not a real directory"
                        )
                    auxiliary_present_before = True
                    continue
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise CacheAttestationError(
                        f"unexpected top-level object in JAX cache: {entry.name}"
                    )
                if not entry.name.endswith((_CACHE_SUFFIX, _ATIME_SUFFIX)):
                    raise CacheAttestationError(
                        f"unexpected top-level file in JAX cache: {entry.name}"
                    )
                names_before.append(entry.name)
        names_before.sort()
        cache_names = {
            name.removesuffix(_CACHE_SUFFIX)
            for name in names_before
            if name.endswith(_CACHE_SUFFIX)
        }
        atime_names = {
            name.removesuffix(_ATIME_SUFFIX)
            for name in names_before
            if name.endswith(_ATIME_SUFFIX)
        }
        if cache_names != atime_names:
            raise CacheAttestationError(
                "JAX executable cache contains orphaned cache/atime pairs"
            )

        pairs: dict[str, CachePair] = {}
        executable_records: list[dict[str, object]] = []
        atime_records: list[dict[str, object]] = []
        for key in sorted(cache_names):
            if _CACHE_KEY_PATTERN.fullmatch(key) is None and (
                not key.startswith("jit_") or len(key) > 512
            ):
                raise CacheAttestationError(f"invalid JAX cache key name: {key!r}")
            executable, _executable_payload = _read_stable_regular_at(
                directory_fd,
                f"{key}{_CACHE_SUFFIX}",
                label=f"JAX executable cache entry {key}",
                retain_payload=False,
            )
            if executable.size_bytes <= 0:
                raise CacheAttestationError(
                    f"JAX executable cache entry {key!r} is empty"
                )
            logical_atime, atime_payload = _read_stable_regular_at(
                directory_fd,
                f"{key}{_ATIME_SUFFIX}",
                label=f"JAX logical-atime sidecar {key}",
                maximum_bytes=8,
            )
            if len(atime_payload) != 8:
                raise CacheAttestationError(
                    f"JAX logical-atime sidecar {key!r} is not exactly 8 bytes"
                )
            logical_atime_ns = int.from_bytes(atime_payload, "little", signed=False)
            if logical_atime_ns <= 0:
                raise CacheAttestationError(
                    f"JAX logical-atime sidecar {key!r} is not positive"
                )
            pairs[key] = CachePair(
                key=key,
                executable=executable,
                logical_atime=logical_atime,
                logical_atime_ns=logical_atime_ns,
            )
            executable_records.append(executable.as_dict())
            atime_record = logical_atime.as_dict()
            atime_record["logical_atime_ns"] = logical_atime_ns
            atime_records.append(atime_record)

        auxiliary_manifest = _snapshot_auxiliary_tree(
            canonical / _AUTOTUNE_DIRECTORY
        )
        for pair in pairs.values():
            for fingerprint, label in (
                (pair.executable, "executable"),
                (pair.logical_atime, "logical-atime"),
            ):
                try:
                    metadata = os.stat(
                        fingerprint.name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                except OSError as error:
                    raise CacheAttestationError(
                        f"cannot revalidate JAX {label} cache entry: {error}"
                    ) from error
                if not _fingerprint_matches_metadata(fingerprint, metadata):
                    raise CacheAttestationError(
                        f"JAX {label} cache entry changed during its snapshot"
                    )
        names_after: list[str] = []
        auxiliary_present_after = False
        with os.scandir(canonical) as entries:
            for entry in entries:
                if entry.name == ".lockfile":
                    continue
                if entry.name == _AUTOTUNE_DIRECTORY:
                    if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                        raise CacheAttestationError(
                            "JAX autotune cache entry changed type during snapshot"
                        )
                    auxiliary_present_after = True
                    continue
                names_after.append(entry.name)
        names_after.sort()
        if (
            names_after != names_before
            or auxiliary_present_after is not auxiliary_present_before
        ):
            raise CacheAttestationError(
                "JAX executable cache directory changed during its snapshot"
            )
        return CacheSnapshot(
            cache_path=str(canonical),
            pairs=pairs,
            executable_manifest_sha256=canonical_json_sha256(
                executable_records,
                domain="skyrl-qwen35-cache-executable-manifest-v1",
            ),
            logical_atime_manifest_sha256=canonical_json_sha256(
                atime_records,
                domain="skyrl-qwen35-cache-logical-atime-manifest-v1",
            ),
            auxiliary_manifest_sha256=auxiliary_manifest,
        )
    finally:
        os.close(directory_fd)


def _pair_delta(before: CacheSnapshot, after: CacheSnapshot) -> dict[str, list[str]]:
    before_keys = set(before.pairs)
    after_keys = set(after.pairs)
    shared = before_keys & after_keys
    return {
        "executable_added": sorted(after_keys - before_keys),
        "executable_removed": sorted(before_keys - after_keys),
        "executable_changed": sorted(
            key
            for key in shared
            if before.pairs[key].executable != after.pairs[key].executable
        ),
        "logical_atime_added": sorted(after_keys - before_keys),
        "logical_atime_removed": sorted(before_keys - after_keys),
        "logical_atime_changed": sorted(
            key
            for key in shared
            if before.pairs[key].logical_atime != after.pairs[key].logical_atime
            or before.pairs[key].logical_atime_ns != after.pairs[key].logical_atime_ns
        ),
    }


def compare_cache_transition(
    before: CacheSnapshot,
    after: CacheSnapshot,
    trace: MonitoringTrace,
    *,
    operation_wall_start_ns: int,
    operation_wall_end_ns: int,
    expected_key: str | None = None,
    require_hit: bool = False,
) -> dict[str, object]:
    """Classify one monitored lower/compile cache transition."""
    if before.cache_path != after.cache_path:
        raise CacheAttestationError("JAX cache path changed during compilation")
    if (
        type(operation_wall_start_ns) is not int
        or type(operation_wall_end_ns) is not int
        or operation_wall_start_ns <= 0
        or operation_wall_end_ns < operation_wall_start_ns
    ):
        raise CacheAttestationError("cache operation wall-clock bracket is invalid")
    if expected_key is not None and _CACHE_KEY_PATTERN.fullmatch(expected_key) is None:
        raise CacheAttestationError("expected cache key is not canonical")

    delta = _pair_delta(before, after)
    requests = trace.events.get(_EVENT_REQUEST, 0)
    hits = trace.events.get(_EVENT_HIT, 0)
    misses = trace.events.get(_EVENT_MISS, 0)
    saved = list(trace.durations.get(_DURATION_SAVED, ()))
    retrieval = list(trace.durations.get(_DURATION_RETRIEVAL, ()))
    hit_event_order = (
        _EVENT_REQUEST,
        _EVENT_HIT,
        _DURATION_SAVED,
        _DURATION_RETRIEVAL,
    )
    miss_event_order = (_EVENT_REQUEST, _EVENT_MISS)
    exact_hit_events = (
        requests == 1
        and hits == 1
        and misses == 0
        and trace.ordered_events == hit_event_order
        and len(saved) == 1
        and len(retrieval) == 1
        and saved[0] > 0
        and retrieval[0] >= 0
    )
    exact_miss_events = (
        requests == 1
        and hits == 0
        and misses == 1
        and trace.ordered_events == miss_event_order
        and not saved
        and not retrieval
    )
    no_executable_delta = not any(
        delta[name]
        for name in (
            "executable_added",
            "executable_removed",
            "executable_changed",
        )
    )
    no_atime_add_remove = not (
        delta["logical_atime_added"] or delta["logical_atime_removed"]
    )
    hit_keys = delta["logical_atime_changed"]
    miss_keys = delta["executable_added"]

    target_key: str | None = None
    classification: str
    if trace.schema_issues:
        classification = "malformed_public_monitoring_evidence"
    elif (
        exact_hit_events
        and no_executable_delta
        and no_atime_add_remove
        and len(hit_keys) == 1
    ):
        target_key = hit_keys[0]
        target_before = before.pairs[target_key]
        target_after = after.pairs[target_key]
        if target_after.logical_atime_ns <= target_before.logical_atime_ns:
            classification = "hit_with_nonincreasing_logical_atime"
        elif not (
            operation_wall_start_ns
            <= target_after.logical_atime_ns
            <= operation_wall_end_ns
        ):
            classification = "hit_with_logical_atime_outside_operation"
        elif expected_key is not None and target_key != expected_key:
            classification = "hit_for_unexpected_cache_key"
        elif before.auxiliary_manifest_sha256 != after.auxiliary_manifest_sha256:
            classification = "hit_with_auxiliary_cache_mutation"
        else:
            classification = PREWARM_SEED_HIT
    elif (
        exact_miss_events
        and len(miss_keys) == 1
        and delta["logical_atime_added"] == miss_keys
        and not delta["executable_removed"]
        and not delta["executable_changed"]
        and not delta["logical_atime_removed"]
        and not delta["logical_atime_changed"]
    ):
        target_key = miss_keys[0]
        target_after = after.pairs[target_key]
        if not (
            operation_wall_start_ns
            <= target_after.logical_atime_ns
            <= operation_wall_end_ns
        ):
            classification = "miss_with_logical_atime_outside_operation"
        elif expected_key is not None and target_key != expected_key:
            classification = "miss_for_unexpected_cache_key"
        else:
            classification = PREWARM_SEED_MISS
    elif requests == 1 and hits == 1 and misses == 0 and not no_executable_delta:
        classification = "ambiguous_hit_with_executable_cache_mutation"
    elif requests == 1 and hits == 1 and misses == 0:
        classification = "hit_without_exact_logical_atime_evidence"
    elif requests == 1 and hits == 0 and misses == 1:
        classification = "miss_without_exact_paired_cache_add"
    elif saved or retrieval:
        classification = "duration_events_without_single_cache_hit"
    elif hits or misses or requests:
        classification = "mixed_or_incomplete_public_monitoring_events"
    else:
        classification = "no_public_hit_or_miss_event_observed"

    if require_hit and classification != PREWARM_SEED_HIT:
        raise CacheAttestationError(
            f"runtime persistent-cache lookup was not an exact hit: {classification}"
        )

    target_entry: dict[str, object] | None = None
    target_atime: dict[str, object] | None = None
    if target_key is not None:
        pair_after = after.pairs[target_key]
        pair_before = before.pairs.get(target_key)
        target_entry = {
            "key": target_key,
            **pair_after.executable.as_dict(),
        }
        target_atime = {
            "name": pair_after.logical_atime.name,
            "before_logical_atime_ns": (
                None if pair_before is None else pair_before.logical_atime_ns
            ),
            "after_logical_atime_ns": pair_after.logical_atime_ns,
            "before_sha256": (
                None if pair_before is None else pair_before.logical_atime.sha256
            ),
            "after_sha256": pair_after.logical_atime.sha256,
            "transition": "added" if pair_before is None else "rewritten",
        }

    if classification == PREWARM_SEED_HIT:
        evidence_limit = HIT_EVIDENCE_LIMIT
    elif classification == PREWARM_SEED_MISS:
        evidence_limit = MISS_EVIDENCE_LIMIT
    else:
        evidence_limit = (
            "does not prove an accepted persistent-cache transition or invoke the "
            "executable"
        )

    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "target_cache_entry": target_entry,
        "target_atime_transition": target_atime,
        "monitoring": {
            "ordered_events": list(trace.ordered_events),
            "compile_requests_use_cache": requests,
            "cache_hits": hits,
            "cache_misses": misses,
            "compile_time_saved_sec": saved,
            "cache_retrieval_time_sec": retrieval,
            "schema_issues": list(trace.schema_issues),
        },
        "snapshots": {
            "executable_manifest_before_sha256": (before.executable_manifest_sha256),
            "executable_manifest_after_sha256": after.executable_manifest_sha256,
            "logical_atime_manifest_before_sha256": (
                before.logical_atime_manifest_sha256
            ),
            "logical_atime_manifest_after_sha256": (
                after.logical_atime_manifest_sha256
            ),
            "auxiliary_manifest_before_sha256": before.auxiliary_manifest_sha256,
            "auxiliary_manifest_after_sha256": after.auxiliary_manifest_sha256,
            **delta,
        },
        "operation_wall_start_ns": operation_wall_start_ns,
        "operation_wall_end_ns": operation_wall_end_ns,
        "evidence_limit": evidence_limit,
    }


class PublicCacheMonitoringCapture:
    """Capture only JAX's five public persistent-cache monitoring signals."""

    def __init__(self, monitoring: Any):
        self._monitoring = monitoring
        self._ordered_events: list[str] = []
        self._events: dict[str, int] = {}
        self._durations: dict[str, list[float]] = {}
        self._issues: list[str] = []
        self._event_registered = False
        self._duration_registered = False

    def _event_listener(self, event: str, **metadata: object) -> None:
        if event not in _EVENT_NAMES:
            return
        self._ordered_events.append(event)
        if metadata:
            self._issues.append(f"unexpected event metadata for {event}")
        self._events[event] = self._events.get(event, 0) + 1

    def _duration_listener(
        self, event: str, duration_secs: object, **metadata: object
    ) -> None:
        if event not in _DURATION_NAMES:
            return
        self._ordered_events.append(event)
        if metadata:
            self._issues.append(f"unexpected duration metadata for {event}")
        if (
            isinstance(duration_secs, bool)
            or not isinstance(duration_secs, (int, float))
            or not math.isfinite(duration_secs)
        ):
            self._issues.append(f"invalid numeric duration for {event}")
            return
        numeric = float(duration_secs)
        if event == _DURATION_RETRIEVAL and numeric < 0:
            self._issues.append(f"negative retrieval duration for {event}")
            return
        self._durations.setdefault(event, []).append(numeric)

    def __enter__(self) -> "PublicCacheMonitoringCapture":
        try:
            self._monitoring.register_event_listener(self._event_listener)
            self._event_registered = True
            self._monitoring.register_event_duration_secs_listener(
                self._duration_listener
            )
            self._duration_registered = True
        except BaseException:
            self.close()
            raise
        return self

    def close(self) -> None:
        errors: list[BaseException] = []
        if self._duration_registered:
            self._duration_registered = False
            try:
                self._monitoring.unregister_event_duration_listener(
                    self._duration_listener
                )
            except BaseException as error:
                errors.append(error)
        if self._event_registered:
            self._event_registered = False
            try:
                self._monitoring.unregister_event_listener(self._event_listener)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise CacheAttestationError(
                "could not unregister JAX cache monitoring listeners"
            ) from errors[0]

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        try:
            self.close()
        except BaseException:
            if exc is None:
                raise
        return False

    def trace(self) -> MonitoringTrace:
        if self._event_registered or self._duration_registered:
            raise CacheAttestationError(
                "monitoring trace requested before listener cleanup"
            )
        return MonitoringTrace(
            ordered_events=tuple(self._ordered_events),
            events=dict(self._events),
            durations={name: tuple(values) for name, values in self._durations.items()},
            schema_issues=tuple(self._issues),
        )


def stable_private_artifact(
    path: Path,
    expected_sha256: str,
    *,
    maximum_bytes: int,
) -> tuple[FileFingerprint, bytes]:
    """Read one immutable-looking 0600 artifact under a private 0700 parent."""
    if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise CacheAttestationError("artifact SHA-256 claim is not canonical")
    parent, parent_fd = _canonical_owned_directory(
        path.parent, "cache-attestation artifact parent"
    )
    try:
        parent_metadata = os.fstat(parent_fd)
        if stat.S_IMODE(parent_metadata.st_mode) != 0o700:
            raise CacheAttestationError(
                "cache-attestation artifact parent must have exact mode 0700"
            )
        canonical = parent / path.name
        if canonical != path or path.name in {"", ".", ".."}:
            raise CacheAttestationError(
                "cache-attestation artifact path must be canonical"
            )
        fingerprint, payload = _read_stable_regular_at(
            parent_fd,
            path.name,
            label=f"cache-attestation artifact {path.name}",
            maximum_bytes=maximum_bytes,
        )
        if fingerprint.mode != 0o600:
            raise CacheAttestationError(
                "cache-attestation artifact must have exact mode 0600"
            )
        if fingerprint.sha256 != expected_sha256:
            raise CacheAttestationError(
                "cache-attestation artifact SHA-256 does not match its claim"
            )
        if not payload or not payload.endswith(b"\n"):
            raise CacheAttestationError(
                "cache-attestation artifact must be nonempty and newline terminated"
            )
        return fingerprint, payload
    finally:
        os.close(parent_fd)


def strict_jsonl(payload: bytes) -> tuple[dict[str, object], ...]:
    """Decode bounded JSONL while rejecting duplicate keys and nonfinite values."""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CacheAttestationError("artifact is not valid UTF-8") from error
    if not text.endswith("\n"):
        raise CacheAttestationError("artifact is not newline terminated")

    def reject_constant(value: str) -> object:
        raise CacheAttestationError(
            f"artifact contains nonfinite JSON constant {value!r}"
        )

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise CacheAttestationError(
                f"artifact contains nonfinite JSON number {value!r}"
            )
        return parsed

    def unique_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise CacheAttestationError(f"artifact JSON object repeats key {key!r}")
            result[key] = value
        return result

    records: list[dict[str, object]] = []
    for index, line in enumerate(text.splitlines()):
        if not line:
            raise CacheAttestationError(
                f"artifact contains an empty JSONL record at index {index}"
            )
        try:
            record = json.loads(
                line,
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
                parse_float=finite_float,
            )
        except CacheAttestationError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise CacheAttestationError(
                f"artifact JSONL record {index} is malformed"
            ) from error
        if not isinstance(record, dict):
            raise CacheAttestationError(
                f"artifact JSONL record {index} is not an object"
            )
        records.append(record)
    if not records:
        raise CacheAttestationError("artifact contains no records")
    return tuple(records)


def _require_zero_invocations(record: Mapping[str, object]) -> None:
    fields = (
        "compiled_callable_invocations",
        "model_pass_executable_invocations",
        "optimizer_step_invocations",
    )
    if "optimizer_step_invocations" not in record or not any(
        field in record for field in fields[:2]
    ):
        raise CacheAttestationError("artifact invocation counters are incomplete")
    for field in fields:
        if field in record and (type(record[field]) is not int or record[field] != 0):
            raise CacheAttestationError(f"artifact reports a nonzero {field} count")


def _one_record(
    records: Sequence[Mapping[str, object]],
    record_type: str,
    *,
    bucket: int | None = None,
) -> Mapping[str, object]:
    matches = [
        record
        for record in records
        if record.get("record_type") == record_type
        and (
            bucket is None
            or (type(record.get("bucket")) is int and record.get("bucket") == bucket)
        )
    ]
    if len(matches) != 1:
        suffix = "" if bucket is None else f" for bucket {bucket}"
        raise CacheAttestationError(
            f"artifact requires exactly one {record_type!r} record{suffix}"
        )
    return matches[0]


def validate_prewarm_t64_artifact(
    payload: bytes,
    *,
    expected_git_head: str,
    expected_git_tree: str,
    expected_cache_path: str,
    expected_attention_backend: str,
) -> dict[str, object]:
    """Extract the exact T64 seed from one successful operational prewarm."""
    records = strict_jsonl(payload)
    allowed_record_types = {
        "manifest",
        "backend_ready",
        "bucket_postflight",
        "bucket_cache_evidence",
        "bucket_compiled",
        "optimizer_postflight",
        "optimizer_cache_evidence",
        "optimizer_compiled",
        "hardware_postflight",
        "complete",
        "error",
    }
    if any(record.get("record_type") not in allowed_record_types for record in records):
        raise CacheAttestationError("prewarm artifact has an unknown record type")
    if any(
        record.get("record_type") == "error" or record.get("status") == "failed"
        for record in records
    ):
        raise CacheAttestationError("prewarm artifact contains a failure record")
    manifest = records[0]
    manifest_fields = {
        "record_type",
        "artifact_schema_name",
        "artifact_schema_version",
        "timestamp",
        "mode",
        "model",
        "model_revision",
        "construction",
        "buckets",
        "batch_size",
        "attention_backend",
        "optimizer_compile_requested",
        "train_bucket_lower_calls_planned",
        "train_bucket_compile_calls_planned",
        "optimizer_lower_calls_planned",
        "optimizer_compile_calls_planned",
        "command_buffers_required_disabled",
        "compiled_callable_invocations",
        "optimizer_step_invocations",
        "executable_export_used",
        "graph_api_used",
        "source_attestation",
    }
    if (
        set(manifest) != manifest_fields
        or _one_record(records, "manifest") is not manifest
        or manifest.get("record_type") != "manifest"
        or manifest.get("artifact_schema_name") != SCHEMA_NAME
        or type(manifest.get("artifact_schema_version")) is not int
        or manifest.get("artifact_schema_version") != SCHEMA_VERSION
        or type(manifest.get("timestamp")) is not str
        or not manifest["timestamp"]
        or manifest.get("mode") != "rocm_compile_only"
        or manifest.get("model") != MODEL
        or manifest.get("model_revision") != MODEL_REVISION
        or type(manifest.get("batch_size")) is not int
        or manifest.get("batch_size") != BATCH_SIZE
        or type(manifest.get("optimizer_compile_requested")) is not bool
        or manifest.get("command_buffers_required_disabled") != DISABLE_COMMAND_BUFFERS
        or manifest.get("executable_export_used") is not False
        or manifest.get("graph_api_used") is not False
    ):
        raise CacheAttestationError(
            "prewarm manifest schema or safety policy is invalid"
        )
    buckets = manifest.get("buckets")
    if (
        not isinstance(buckets, list)
        or any(type(value) is not int for value in buckets)
        or buckets != sorted(set(buckets))
        or buckets.count(BUCKET) != 1
    ):
        raise CacheAttestationError("prewarm manifest does not contain exact T64")
    construction = manifest.get("construction")
    if construction not in {"eager", "abstract-load"}:
        raise CacheAttestationError("prewarm construction route is invalid")
    if manifest.get("attention_backend") != expected_attention_backend:
        raise CacheAttestationError("prewarm attention backend changed before runtime")
    optimizer_requested = manifest["optimizer_compile_requested"]
    expected_optimizer_calls = 1 if optimizer_requested else 0
    if (
        type(manifest.get("train_bucket_lower_calls_planned")) is not int
        or manifest["train_bucket_lower_calls_planned"] != len(buckets)
        or type(manifest.get("train_bucket_compile_calls_planned")) is not int
        or manifest["train_bucket_compile_calls_planned"] != len(buckets)
        or type(manifest.get("optimizer_lower_calls_planned")) is not int
        or manifest["optimizer_lower_calls_planned"] != expected_optimizer_calls
        or type(manifest.get("optimizer_compile_calls_planned")) is not int
        or manifest["optimizer_compile_calls_planned"] != expected_optimizer_calls
    ):
        raise CacheAttestationError("prewarm manifest compile plan is invalid")
    expected_record_types = ["manifest", "backend_ready"]
    for _bucket in buckets:
        expected_record_types.extend(
            ["bucket_postflight", "bucket_cache_evidence", "bucket_compiled"]
        )
    if optimizer_requested:
        expected_record_types.extend(
            ["optimizer_postflight", "optimizer_cache_evidence", "optimizer_compiled"]
        )
    expected_record_types.extend(["hardware_postflight", "complete"])
    if [record.get("record_type") for record in records] != expected_record_types:
        raise CacheAttestationError("prewarm artifact record structure is invalid")
    for record_type in (
        "bucket_postflight",
        "bucket_cache_evidence",
        "bucket_compiled",
    ):
        observed_buckets = [
            record.get("bucket")
            for record in records
            if record.get("record_type") == record_type
        ]
        if (
            any(type(bucket) is not int for bucket in observed_buckets)
            or observed_buckets != buckets
        ):
            raise CacheAttestationError(
                f"prewarm {record_type} bucket structure is invalid"
            )
    _require_zero_invocations(manifest)
    source = manifest.get("source_attestation")
    if (
        not isinstance(source, dict)
        or source.get("status") != "passed"
        or source.get("git_head") != expected_git_head
        or source.get("git_tree") != expected_git_tree
        or source.get("full_head_tree_validated") is not True
    ):
        raise CacheAttestationError("prewarm source attestation does not match runtime")

    backend_ready = _one_record(records, "backend_ready")
    backend_ready_fields = {
        "record_type",
        "timestamp",
        "model",
        "model_revision",
        "model_path",
        "construction",
        "platform_resolved",
        "platform_version",
        "jax_version",
        "jaxlib_version",
        "cache_path",
        "backend_config",
        "backend_config_sha256",
        "adapter_index",
        "setup_seconds",
        "setup_dispatch_caveat",
        "optimizer_compile_requested",
        "train_bucket_lower_calls",
        "train_bucket_compile_calls",
        "optimizer_lower_calls",
        "optimizer_compile_calls",
        "model_pass_executable_invocations",
        "optimizer_step_invocations",
        "hardware_preflight",
    }
    setup_seconds = backend_ready.get("setup_seconds")
    if (
        set(backend_ready) != backend_ready_fields
        or type(backend_ready.get("timestamp")) is not str
        or not backend_ready["timestamp"]
        or backend_ready.get("model") != MODEL
        or backend_ready.get("model_revision") != MODEL_REVISION
        or backend_ready.get("cache_path") != expected_cache_path
        or backend_ready.get("platform_resolved") != "gpu"
        or backend_ready.get("jax_version") != "0.10.2"
        or backend_ready.get("jaxlib_version") != "0.10.2"
        or type(backend_ready.get("adapter_index")) is not int
        or backend_ready.get("adapter_index") != 1
        or backend_ready.get("construction") != construction
        or type(backend_ready.get("platform_version")) is not str
        or not backend_ready["platform_version"]
        or isinstance(setup_seconds, bool)
        or not isinstance(setup_seconds, (int, float))
        or not math.isfinite(setup_seconds)
        or setup_seconds < 0
        or type(backend_ready.get("setup_dispatch_caveat")) is not str
        or not backend_ready["setup_dispatch_caveat"]
        or backend_ready.get("optimizer_compile_requested") is not optimizer_requested
        or not isinstance(backend_ready.get("hardware_preflight"), dict)
        or any(
            type(backend_ready.get(field)) is not int
            or backend_ready[field] != 0
            for field in (
                "train_bucket_lower_calls",
                "train_bucket_compile_calls",
                "optimizer_lower_calls",
                "optimizer_compile_calls",
            )
        )
    ):
        raise CacheAttestationError("prewarm backend identity is invalid")
    backend_config = backend_ready.get("backend_config")
    backend_config_sha256 = backend_ready.get("backend_config_sha256")
    if not isinstance(backend_config, dict) or (
        backend_config_sha256
        != canonical_json_sha256(
            backend_config, domain="skyrl-qwen35-resolved-jax-backend-config-v1"
        )
    ):
        raise CacheAttestationError("prewarm resolved backend config hash is invalid")
    if backend_config.get("enforce_eager") is not False:
        raise CacheAttestationError("prewarm backend did not use JIT")
    if backend_config.get("qwen35_bf16_down_lora_residual") is not True:
        raise CacheAttestationError(
            "prewarm backend did not enable the qualified BF16 down fusion"
        )
    if backend_config.get("abstract_model_load") is not (
        construction == "abstract-load"
    ):
        raise CacheAttestationError(
            "prewarm resolved backend config does not match its construction route"
        )
    model_path = backend_ready.get("model_path")
    if not isinstance(model_path, str) or not model_path.startswith("/"):
        raise CacheAttestationError("prewarm canonical model path is absent")
    _require_zero_invocations(backend_ready)

    postflight = _one_record(records, "bucket_postflight", bucket=BUCKET)
    postflight_fields = {
        "record_type",
        "timestamp",
        "compile_target",
        "bucket",
        "status",
        "compile_succeeded",
        "cache_revalidated",
        "amdgpu_boot_clean",
        "fatal_amdgpu_events",
        "model_pass_executable_invocations",
        "optimizer_step_invocations",
    }
    if (
        set(postflight) != postflight_fields
        or type(postflight.get("timestamp")) is not str
        or not postflight["timestamp"]
        or type(postflight.get("bucket")) is not int
        or postflight.get("compile_target") != COMPILE_TARGET
        or postflight.get("status") != "clean"
        or postflight.get("compile_succeeded") is not True
        or postflight.get("cache_revalidated") is not True
        or postflight.get("amdgpu_boot_clean") is not True
        or postflight.get("fatal_amdgpu_events") != []
    ):
        raise CacheAttestationError("T64 prewarm postflight is not clean")
    _require_zero_invocations(postflight)

    cache_record = _one_record(records, "bucket_cache_evidence", bucket=BUCKET)
    cache_record_fields = {
        "record_type",
        "timestamp",
        "compile_target",
        "bucket",
        "status",
        "evidence",
        "model_pass_executable_invocations",
        "optimizer_step_invocations",
    }
    evidence = cache_record.get("evidence")
    if (
        set(cache_record) != cache_record_fields
        or type(cache_record.get("timestamp")) is not str
        or not cache_record["timestamp"]
        or type(cache_record.get("bucket")) is not int
        or cache_record.get("compile_target") != COMPILE_TARGET
        or cache_record.get("status") != "accepted"
        or not isinstance(evidence, dict)
        or evidence.get("schema_name") != SCHEMA_NAME
        or type(evidence.get("schema_version")) is not int
        or evidence.get("schema_version") != SCHEMA_VERSION
        or evidence.get("classification") not in {PREWARM_SEED_HIT, PREWARM_SEED_MISS}
    ):
        raise CacheAttestationError("T64 prewarm cache evidence is not strict")
    expected_evidence_fields = {
        "schema_name",
        "schema_version",
        "classification",
        "target_cache_entry",
        "target_atime_transition",
        "monitoring",
        "snapshots",
        "operation_wall_start_ns",
        "operation_wall_end_ns",
        "evidence_limit",
        "public_monitoring_events",
        "public_monitoring_duration_events",
        "public_monitoring_schema_issues",
        "top_level_executable_cache",
    }
    if set(evidence) != expected_evidence_fields:
        raise CacheAttestationError("T64 prewarm evidence schema is invalid")
    target_entry = evidence.get("target_cache_entry")
    target_atime = evidence.get("target_atime_transition")
    if not isinstance(target_entry, dict) or not isinstance(target_atime, dict):
        raise CacheAttestationError("T64 prewarm target cache identity is absent")
    if set(target_entry) != {
        "key",
        "name",
        "device",
        "inode",
        "mode",
        "uid",
        "link_count",
        "size_bytes",
        "mtime_ns",
        "ctime_ns",
        "sha256",
    } or set(target_atime) != {
        "name",
        "before_logical_atime_ns",
        "after_logical_atime_ns",
        "before_sha256",
        "after_sha256",
        "transition",
    }:
        raise CacheAttestationError("T64 prewarm target cache schema is invalid")
    key = target_entry.get("key")
    operation_wall_start_ns = evidence.get("operation_wall_start_ns")
    operation_wall_end_ns = evidence.get("operation_wall_end_ns")
    if (
        not isinstance(key, str)
        or _CACHE_KEY_PATTERN.fullmatch(key) is None
        or target_entry.get("name") != f"{key}{_CACHE_SUFFIX}"
        or target_atime.get("name") != f"{key}{_ATIME_SUFFIX}"
        or not _is_sha256(target_entry.get("sha256"))
        or type(target_entry.get("size_bytes")) is not int
        or target_entry["size_bytes"] <= 0
        or any(
            type(target_entry.get(field)) is not int
            or target_entry[field] < minimum
            for field, minimum in (
                ("device", 0),
                ("inode", 1),
                ("mode", 0),
                ("uid", 0),
                ("link_count", 1),
                ("mtime_ns", 0),
                ("ctime_ns", 0),
            )
        )
        or target_entry["mode"] > 0o7777
        or target_entry["uid"] != os.getuid()
        or target_entry["link_count"] != 1
        or type(target_atime.get("after_logical_atime_ns")) is not int
        or not 0 < target_atime["after_logical_atime_ns"] < 2**64
        or not _is_sha256(target_atime.get("after_sha256"))
        or target_atime.get("after_sha256")
        != hashlib.sha256(
            target_atime["after_logical_atime_ns"].to_bytes(8, "little")
        ).hexdigest()
        or type(operation_wall_start_ns) is not int
        or type(operation_wall_end_ns) is not int
        or not (
            0
            < operation_wall_start_ns
            <= target_atime["after_logical_atime_ns"]
            <= operation_wall_end_ns
        )
        or evidence.get("evidence_limit")
        != (
            HIT_EVIDENCE_LIMIT
            if evidence["classification"] == PREWARM_SEED_HIT
            else MISS_EVIDENCE_LIMIT
        )
    ):
        raise CacheAttestationError("T64 prewarm target cache fields are invalid")
    if evidence["classification"] == PREWARM_SEED_HIT:
        if (
            target_atime.get("transition") != "rewritten"
            or type(target_atime.get("before_logical_atime_ns")) is not int
            or not 0 < target_atime["before_logical_atime_ns"] < 2**64
            or target_atime["after_logical_atime_ns"]
            <= target_atime["before_logical_atime_ns"]
            or not _is_sha256(target_atime.get("before_sha256"))
            or target_atime.get("before_sha256")
            != hashlib.sha256(
                target_atime["before_logical_atime_ns"].to_bytes(8, "little")
            ).hexdigest()
        ):
            raise CacheAttestationError("T64 prewarm hit atime transition is invalid")
    elif (
        target_atime.get("transition") != "added"
        or target_atime.get("before_logical_atime_ns") is not None
        or target_atime.get("before_sha256") is not None
    ):
        raise CacheAttestationError("T64 prewarm miss atime transition is invalid")
    monitoring = evidence.get("monitoring")
    monitoring_fields = {
        "ordered_events",
        "compile_requests_use_cache",
        "cache_hits",
        "cache_misses",
        "compile_time_saved_sec",
        "cache_retrieval_time_sec",
        "schema_issues",
    }
    if evidence["classification"] == PREWARM_SEED_HIT:
        expected_counts = (1, 1, 0)
        expected_order = [
            _EVENT_REQUEST,
            _EVENT_HIT,
            _DURATION_SAVED,
            _DURATION_RETRIEVAL,
        ]
    else:
        expected_counts = (1, 0, 1)
        expected_order = [_EVENT_REQUEST, _EVENT_MISS]
    if not isinstance(monitoring, dict) or set(monitoring) != monitoring_fields:
        raise CacheAttestationError("T64 prewarm monitoring schema is invalid")
    observed_counts = (
        monitoring.get("compile_requests_use_cache"),
        monitoring.get("cache_hits"),
        monitoring.get("cache_misses"),
    )
    if (
        any(type(value) is not int for value in observed_counts)
        or observed_counts != expected_counts
        or monitoring.get("ordered_events") != expected_order
        or monitoring.get("schema_issues") != []
    ):
        raise CacheAttestationError("T64 prewarm public monitoring counts are invalid")
    saved = monitoring.get("compile_time_saved_sec")
    retrieval = monitoring.get("cache_retrieval_time_sec")
    if evidence["classification"] == PREWARM_SEED_HIT:
        if (
            not isinstance(saved, list)
            or len(saved) != 1
            or isinstance(saved[0], bool)
            or not isinstance(saved[0], (int, float))
            or not math.isfinite(saved[0])
            or saved[0] <= 0
            or not isinstance(retrieval, list)
            or len(retrieval) != 1
            or isinstance(retrieval[0], bool)
            or not isinstance(retrieval[0], (int, float))
            or not math.isfinite(retrieval[0])
            or retrieval[0] < 0
        ):
            raise CacheAttestationError("T64 prewarm hit durations are invalid")
    elif saved != [] or retrieval != []:
        raise CacheAttestationError("T64 prewarm miss has hit-duration evidence")

    snapshots = evidence.get("snapshots")
    snapshot_fields = {
        "executable_manifest_before_sha256",
        "executable_manifest_after_sha256",
        "logical_atime_manifest_before_sha256",
        "logical_atime_manifest_after_sha256",
        "auxiliary_manifest_before_sha256",
        "auxiliary_manifest_after_sha256",
        "executable_added",
        "executable_removed",
        "executable_changed",
        "logical_atime_added",
        "logical_atime_removed",
        "logical_atime_changed",
    }
    if (
        not isinstance(snapshots, dict)
        or set(snapshots) != snapshot_fields
        or any(
            not _is_sha256(snapshots.get(field))
            for field in snapshot_fields
            if field.endswith("sha256")
        )
    ):
        raise CacheAttestationError("T64 prewarm cache snapshot schema is invalid")
    if evidence["classification"] == PREWARM_SEED_HIT:
        expected_deltas = {
            "executable_added": [],
            "executable_removed": [],
            "executable_changed": [],
            "logical_atime_added": [],
            "logical_atime_removed": [],
            "logical_atime_changed": [key],
        }
        if (
            snapshots["executable_manifest_before_sha256"]
            != snapshots["executable_manifest_after_sha256"]
            or snapshots["auxiliary_manifest_before_sha256"]
            != snapshots["auxiliary_manifest_after_sha256"]
            or snapshots["logical_atime_manifest_before_sha256"]
            == snapshots["logical_atime_manifest_after_sha256"]
        ):
            raise CacheAttestationError("T64 prewarm hit cache manifests are invalid")
    else:
        expected_deltas = {
            "executable_added": [key],
            "executable_removed": [],
            "executable_changed": [],
            "logical_atime_added": [key],
            "logical_atime_removed": [],
            "logical_atime_changed": [],
        }
        if (
            snapshots["executable_manifest_before_sha256"]
            == snapshots["executable_manifest_after_sha256"]
            or snapshots["logical_atime_manifest_before_sha256"]
            == snapshots["logical_atime_manifest_after_sha256"]
        ):
            raise CacheAttestationError("T64 prewarm miss cache manifests are invalid")
    if any(snapshots.get(field) != value for field, value in expected_deltas.items()):
        raise CacheAttestationError("T64 prewarm cache deltas are invalid")

    public_events = evidence.get("public_monitoring_events")
    public_durations = evidence.get("public_monitoring_duration_events")
    top_level = evidence.get("top_level_executable_cache")
    if (
        public_events
        != {
            "compile_requests_use_cache": expected_counts[0],
            "cache_hits": expected_counts[1],
            "cache_misses": expected_counts[2],
        }
        or public_durations
        != {
            "compile_time_saved_sec": saved,
            "cache_retrieval_time_sec": retrieval,
        }
        or evidence.get("public_monitoring_schema_issues") != []
        or not isinstance(top_level, dict)
        or set(top_level)
        != {
            "entries_before",
            "entries_after",
            "bytes_before",
            "bytes_after",
            "added_entries",
            "changed_entries",
            "removed_entries",
            "post_manifest_sha256",
        }
        or any(
            type(top_level.get(field)) is not int or top_level[field] < 0
            for field in (
                "entries_before",
                "entries_after",
                "bytes_before",
                "bytes_after",
            )
        )
        or top_level.get("added_entries")
        != [f"{value}{_CACHE_SUFFIX}" for value in expected_deltas["executable_added"]]
        or top_level.get("changed_entries")
        != [f"{value}{_CACHE_SUFFIX}" for value in expected_deltas["executable_changed"]]
        or top_level.get("removed_entries")
        != [f"{value}{_CACHE_SUFFIX}" for value in expected_deltas["executable_removed"]]
        or top_level.get("post_manifest_sha256")
        != snapshots["executable_manifest_after_sha256"]
        or (
            evidence["classification"] == PREWARM_SEED_HIT
            and (
                top_level["entries_before"] != top_level["entries_after"]
                or top_level["bytes_before"] != top_level["bytes_after"]
            )
        )
        or (
            evidence["classification"] == PREWARM_SEED_MISS
            and (
                top_level["entries_after"] != top_level["entries_before"] + 1
                or top_level["bytes_after"]
                != top_level["bytes_before"] + target_entry["size_bytes"]
            )
        )
    ):
        raise CacheAttestationError("T64 prewarm compatibility summaries are invalid")
    _require_zero_invocations(cache_record)

    compiled = _one_record(records, "bucket_compiled", bucket=BUCKET)
    compiled_fields = {
        "record_type",
        "timestamp",
        "compile_target",
        "bucket",
        "batch_size",
        "attention_backend",
        "lower_seconds",
        "compile_seconds",
        "compiled_memory",
        "persistent_cache_evidence",
        "train_bucket_lower_calls",
        "train_bucket_compile_calls",
        "optimizer_lower_calls",
        "optimizer_compile_calls",
        "model_pass_executable_invocations",
        "optimizer_step_invocations",
        "status",
    }
    lower_seconds = compiled.get("lower_seconds")
    compile_seconds = compiled.get("compile_seconds")
    if (
        set(compiled) != compiled_fields
        or type(compiled.get("timestamp")) is not str
        or not compiled["timestamp"]
        or type(compiled.get("bucket")) is not int
        or compiled.get("compile_target") != COMPILE_TARGET
        or compiled.get("status") != "passed"
        or type(compiled.get("batch_size")) is not int
        or compiled.get("batch_size") != BATCH_SIZE
        or compiled.get("attention_backend") != expected_attention_backend
        or compiled.get("persistent_cache_evidence") != evidence
        or isinstance(lower_seconds, bool)
        or not isinstance(lower_seconds, (int, float))
        or not math.isfinite(lower_seconds)
        or lower_seconds < 0
        or isinstance(compile_seconds, bool)
        or not isinstance(compile_seconds, (int, float))
        or not math.isfinite(compile_seconds)
        or compile_seconds < 0
        or not isinstance(compiled.get("compiled_memory"), dict)
        or type(compiled.get("train_bucket_lower_calls")) is not int
        or compiled["train_bucket_lower_calls"] != 1
        or type(compiled.get("train_bucket_compile_calls")) is not int
        or compiled["train_bucket_compile_calls"] != 1
        or type(compiled.get("optimizer_lower_calls")) is not int
        or compiled["optimizer_lower_calls"] != 0
        or type(compiled.get("optimizer_compile_calls")) is not int
        or compiled["optimizer_compile_calls"] != 0
    ):
        raise CacheAttestationError("T64 compiled record is incomplete")
    _require_zero_invocations(compiled)

    hardware_postflight = _one_record(records, "hardware_postflight")
    hardware_postflight_fields = {
        "record_type",
        "timestamp",
        "status",
        "operation_succeeded",
        "source_attestation_revalidated",
        "inherited_launcher_lock_validated",
        "amdgpu_boot_clean",
        "fatal_amdgpu_events",
        "model_pass_executable_invocations",
        "optimizer_step_invocations",
    }
    if (
        set(hardware_postflight) != hardware_postflight_fields
        or type(hardware_postflight.get("timestamp")) is not str
        or not hardware_postflight["timestamp"]
        or hardware_postflight.get("status") != "clean"
        or hardware_postflight.get("operation_succeeded") is not True
        or hardware_postflight.get("source_attestation_revalidated") is not True
        or hardware_postflight.get("inherited_launcher_lock_validated") is not True
        or hardware_postflight.get("amdgpu_boot_clean") is not True
        or hardware_postflight.get("fatal_amdgpu_events") != []
    ):
        raise CacheAttestationError("prewarm hardware postflight is incomplete")
    _require_zero_invocations(hardware_postflight)
    complete = records[-1]
    complete_fields = {
        "record_type",
        "artifact_schema_name",
        "artifact_schema_version",
        "timestamp",
        "buckets",
        "optimizer_compiled",
        "train_bucket_lower_calls",
        "train_bucket_compile_calls",
        "optimizer_lower_calls",
        "optimizer_compile_calls",
        "cache_revalidated_after_each_compile",
        "amdgpu_postflight_clean",
        "source_attestation_revalidated",
        "inherited_launcher_lock_validated",
        "status",
        "model_pass_executable_invocations",
        "optimizer_step_invocations",
    }
    if (
        set(complete) != complete_fields
        or _one_record(records, "complete") is not complete
        or complete.get("record_type") != "complete"
        or complete.get("artifact_schema_name") != SCHEMA_NAME
        or type(complete.get("artifact_schema_version")) is not int
        or complete.get("artifact_schema_version") != SCHEMA_VERSION
        or type(complete.get("timestamp")) is not str
        or not complete["timestamp"]
        or complete.get("status") != "passed"
        or complete.get("buckets") != buckets
        or complete.get("optimizer_compiled") is not optimizer_requested
        or type(complete.get("train_bucket_lower_calls")) is not int
        or complete["train_bucket_lower_calls"] != len(buckets)
        or type(complete.get("train_bucket_compile_calls")) is not int
        or complete["train_bucket_compile_calls"] != len(buckets)
        or type(complete.get("optimizer_lower_calls")) is not int
        or complete["optimizer_lower_calls"] != expected_optimizer_calls
        or type(complete.get("optimizer_compile_calls")) is not int
        or complete["optimizer_compile_calls"] != expected_optimizer_calls
        or complete.get("amdgpu_postflight_clean") is not True
        or complete.get("source_attestation_revalidated") is not True
        or complete.get("inherited_launcher_lock_validated") is not True
        or complete.get("cache_revalidated_after_each_compile") is not True
    ):
        raise CacheAttestationError("prewarm terminal record is incomplete")
    _require_zero_invocations(complete)
    selected_records = (
        backend_ready,
        postflight,
        cache_record,
        compiled,
        hardware_postflight,
        complete,
    )
    selected_indices = [
        next(index for index, record in enumerate(records) if record is selected)
        for selected in selected_records
    ]
    if selected_indices != sorted(selected_indices):
        raise CacheAttestationError("prewarm authoritative records are out of order")

    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "kind": "qwen35_t64_prewarm_cache_target_v1",
        "bucket": BUCKET,
        "batch_size": BATCH_SIZE,
        "compile_target": COMPILE_TARGET,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "model_path": model_path,
        "attention_backend": expected_attention_backend,
        "construction": construction,
        "cache_path": expected_cache_path,
        "source_git_head": expected_git_head,
        "source_git_tree": expected_git_tree,
        "backend_config": backend_config,
        "backend_config_sha256": backend_config_sha256,
        "prewarm_seed_kind": evidence["classification"],
        "target_cache_entry": target_entry,
        "target_atime_transition": target_atime,
        "xla_flags": DISABLE_COMMAND_BUFFERS,
        "graph_api_used": False,
        "command_buffer_used": False,
        "compiled_callable_invocations": 0,
        "model_pass_executable_invocations": 0,
        "optimizer_step_invocations": 0,
    }


def validate_completed_handoff_artifact(
    payload: bytes,
    *,
    expected_boot_id: str,
) -> dict[str, object]:
    """Require the prewarm child to have returned the exact GPU to idle."""
    if _BOOT_ID_PATTERN.fullmatch(expected_boot_id) is None:
        raise CacheAttestationError("current boot ID is not canonical")
    records = strict_jsonl(payload)
    baseline_record = records[0]
    terminal = records[-1]
    baseline_record_fields = {
        "record_type",
        "schema_version",
        "timestamp",
        "status",
        "device",
        "baseline",
        "release_contract",
        "script_sha256",
        "graph_api_used",
        "command_buffer_used",
        "accelerator_device_opened",
        "amdgpu_boot_clean",
        "fatal_amdgpu_events",
    }

    def validate_device_identity(identity: object) -> dict[str, object]:
        fields = {
            "drm_card",
            "vendor_id",
            "device_id",
            "pci_bdf",
            "pci_sysfs_path",
            "drm_sysfs_path",
            "drm_sysfs_dev",
            "render_sysfs_path",
            "render_sysfs_dev",
            "drm_node",
            "kfd_node",
            "render_node",
        }
        if not isinstance(identity, dict) or set(identity) != fields:
            raise CacheAttestationError("prewarm handoff device schema is invalid")
        if (
            type(identity.get("drm_card")) is not str
            or _DRM_CARD_PATTERN.fullmatch(identity["drm_card"]) is None
            or identity.get("vendor_id") != "0x1002"
            or identity.get("device_id") != "0x744c"
            or type(identity.get("pci_bdf")) is not str
            or _PCI_BDF_PATTERN.fullmatch(identity["pci_bdf"]) is None
            or any(
                type(identity.get(field)) is not str
                or not identity[field].startswith("/")
                for field in (
                    "pci_sysfs_path",
                    "drm_sysfs_path",
                    "render_sysfs_path",
                )
            )
            or type(identity.get("drm_sysfs_dev")) is not str
            or _DEVICE_NUMBER_PATTERN.fullmatch(identity["drm_sysfs_dev"]) is None
            or type(identity.get("render_sysfs_dev")) is not str
            or _DEVICE_NUMBER_PATTERN.fullmatch(identity["render_sysfs_dev"])
            is None
        ):
            raise CacheAttestationError("prewarm handoff device identity is invalid")
        for field, expected_device in (
            ("drm_node", identity["drm_sysfs_dev"]),
            ("kfd_node", None),
            ("render_node", identity["render_sysfs_dev"]),
        ):
            node = identity.get(field)
            if (
                not isinstance(node, dict)
                or set(node) != {"path", "rdev", "sysfs_dev", "sysfs_target"}
                or any(type(value) is not str for value in node.values())
                or not node["path"].startswith("/")
                or not node["sysfs_target"].startswith("/")
                or _DEVICE_NUMBER_PATTERN.fullmatch(node["rdev"]) is None
                or node["sysfs_dev"] != node["rdev"]
                or (expected_device is not None and node["rdev"] != expected_device)
            ):
                raise CacheAttestationError(
                    "prewarm handoff device-node identity is invalid"
                )
        return identity

    if (
        set(baseline_record) != baseline_record_fields
        or baseline_record.get("record_type") != "prewarm_handoff_baseline"
        or type(baseline_record.get("schema_version")) is not int
        or baseline_record.get("schema_version") != 1
        or not isinstance(baseline_record.get("timestamp"), str)
        or not baseline_record["timestamp"]
        or baseline_record.get("status") != "passed"
        or baseline_record.get("amdgpu_boot_clean") is not True
        or baseline_record.get("fatal_amdgpu_events") != []
        or baseline_record.get("graph_api_used") is not False
        or baseline_record.get("command_buffer_used") is not False
        or baseline_record.get("accelerator_device_opened") is not False
        or baseline_record.get("script_sha256")
        != _sibling_source_sha256("qwen35_prewarm_handoff.py")
    ):
        raise CacheAttestationError("prewarm handoff baseline is invalid")
    baseline = baseline_record.get("baseline")
    snapshot_fields = {
        "boot_id",
        "device_identity",
        "vram_used_bytes",
        "gtt_used_bytes",
        "runtime_status",
        "kfd_owner_pids",
        "render_owner_pids",
    }
    if (
        not isinstance(baseline, dict)
        or set(baseline) != snapshot_fields
        or baseline.get("boot_id") != expected_boot_id
        or validate_device_identity(baseline.get("device_identity"))
        != baseline.get("device_identity")
        or baseline_record.get("device") != baseline.get("device_identity")
        or type(baseline.get("vram_used_bytes")) is not int
        or baseline["vram_used_bytes"] < 0
        or type(baseline.get("gtt_used_bytes")) is not int
        or baseline["gtt_used_bytes"] < 0
        or baseline.get("runtime_status") != "suspended"
        or baseline.get("kfd_owner_pids") != []
        or baseline.get("render_owner_pids") != []
    ):
        raise CacheAttestationError("prewarm handoff baseline is not idle")
    expected_release_contract = {
        "timeout_seconds": 120.0,
        "poll_interval_seconds": 1.0,
        "required_consecutive_ready_samples": 3,
        "vram_tolerance_bytes": 0,
        "gtt_tolerance_bytes": 0,
        "runtime_status_required": "suspended",
    }
    release_contract = baseline_record.get("release_contract")
    if not isinstance(release_contract, dict) or any(
        type(release_contract.get(field)) is not type(expected)
        or release_contract.get(field) != expected
        for field, expected in expected_release_contract.items()
    ) or set(release_contract) != set(expected_release_contract):
        raise CacheAttestationError("prewarm handoff release contract is invalid")
    if any(
        record.get("record_type")
        in {"prewarm_handoff_error", "prewarm_handoff_timeout"}
        or record.get("status") == "failed"
        for record in records
    ):
        raise CacheAttestationError("prewarm handoff artifact contains a failure")
    samples = [
        record
        for record in records[1:-1]
        if record.get("record_type") == "prewarm_handoff_sample"
    ]
    sample_fields = {
        "record_type",
        "schema_version",
        "timestamp",
        "sample_index",
        "elapsed_seconds",
        "snapshot",
        "checks",
        "ready_streak",
        "required_ready_streak",
        "status",
        "accelerator_device_opened",
    }
    check_fields = {
        "same_boot",
        "same_device_identity",
        "kfd_unowned",
        "render_unowned",
        "vram_no_higher_than_exact_baseline",
        "gtt_no_higher_than_exact_baseline",
        "runtime_suspended",
    }

    def validate_snapshot(snapshot: object) -> dict[str, object]:
        if not isinstance(snapshot, dict) or set(snapshot) != snapshot_fields:
            raise CacheAttestationError("prewarm handoff snapshot schema is invalid")
        if (
            snapshot.get("boot_id") != expected_boot_id
            or snapshot.get("device_identity") != baseline["device_identity"]
            or type(snapshot.get("vram_used_bytes")) is not int
            or snapshot["vram_used_bytes"] < 0
            or type(snapshot.get("gtt_used_bytes")) is not int
            or snapshot["gtt_used_bytes"] < 0
            or snapshot.get("runtime_status")
            not in {"active", "suspending", "suspended", "resuming"}
        ):
            raise CacheAttestationError("prewarm handoff snapshot identity is invalid")
        for field in ("kfd_owner_pids", "render_owner_pids"):
            owners = snapshot.get(field)
            if (
                not isinstance(owners, list)
                or any(type(pid) is not int or pid <= 0 for pid in owners)
                or owners != sorted(set(owners))
            ):
                raise CacheAttestationError(
                    "prewarm handoff snapshot owner list is invalid"
                )
        return snapshot

    def computed_checks(snapshot: Mapping[str, object]) -> dict[str, bool]:
        return {
            "same_boot": snapshot["boot_id"] == baseline["boot_id"],
            "same_device_identity": (
                snapshot["device_identity"] == baseline["device_identity"]
            ),
            "kfd_unowned": snapshot["kfd_owner_pids"] == [],
            "render_unowned": snapshot["render_owner_pids"] == [],
            "vram_no_higher_than_exact_baseline": (
                snapshot["vram_used_bytes"] <= baseline["vram_used_bytes"]
            ),
            "gtt_no_higher_than_exact_baseline": (
                snapshot["gtt_used_bytes"] <= baseline["gtt_used_bytes"]
            ),
            "runtime_suspended": snapshot["runtime_status"] == "suspended",
        }

    expected_ready_streak = 0
    previous_elapsed = -1.0
    for index, sample in enumerate(samples):
        elapsed = sample.get("elapsed_seconds")
        snapshot = validate_snapshot(sample.get("snapshot"))
        checks = sample.get("checks")
        if (
            set(sample) != sample_fields
            or type(sample.get("schema_version")) is not int
            or sample.get("schema_version") != 1
            or not isinstance(sample.get("timestamp"), str)
            or not sample["timestamp"]
            or type(sample.get("sample_index")) is not int
            or sample.get("sample_index") != index
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
            or elapsed < previous_elapsed
            or not isinstance(checks, dict)
            or set(checks) != check_fields
            or any(type(value) is not bool for value in checks.values())
            or checks != computed_checks(snapshot)
            or type(sample.get("required_ready_streak")) is not int
            or sample.get("required_ready_streak") != 3
            or sample.get("accelerator_device_opened") is not False
        ):
            raise CacheAttestationError("prewarm handoff sample is invalid")
        previous_elapsed = float(elapsed)
        observed_ready_streak = sample.get("ready_streak")
        if all(checks.values()):
            allowed_ready_streaks = {expected_ready_streak + 1}
            # The producer takes an extra final snapshot after three ready
            # samples. If that snapshot fails, it resets the streak without
            # emitting a reset record, so the next ready sample restarts at 1.
            if expected_ready_streak >= 3:
                allowed_ready_streaks = {1}
            if (
                type(observed_ready_streak) is not int
                or observed_ready_streak not in allowed_ready_streaks
            ):
                raise CacheAttestationError(
                    "prewarm handoff ready-streak evidence is invalid"
                )
            expected_ready_streak = observed_ready_streak
        else:
            if type(observed_ready_streak) is not int or observed_ready_streak != 0:
                raise CacheAttestationError(
                    "prewarm handoff ready-streak evidence is invalid"
                )
            expected_ready_streak = 0
        if sample.get("status") != (
            "ready_candidate" if all(checks.values()) else "waiting"
        ):
            raise CacheAttestationError(
                "prewarm handoff ready-streak evidence is invalid"
            )
    if (
        len(samples) != len(records) - 2
        or len(samples) < 3
        or [record.get("sample_index") for record in samples]
        != list(range(len(samples)))
    ):
        raise CacheAttestationError("prewarm handoff samples are incomplete")
    terminal_fields = {
        "record_type",
        "schema_version",
        "timestamp",
        "status",
        "elapsed_seconds",
        "sample_count",
        "final_ready_streak",
        "baseline",
        "final_snapshot",
        "checks",
        "vram_tolerance_bytes",
        "gtt_tolerance_bytes",
        "graph_api_used",
        "command_buffer_used",
        "accelerator_device_opened",
        "amdgpu_boot_clean",
        "fatal_amdgpu_events",
    }
    final_snapshot = validate_snapshot(terminal.get("final_snapshot"))
    checks = terminal.get("checks")
    terminal_elapsed = terminal.get("elapsed_seconds")
    if (
        set(terminal) != terminal_fields
        or terminal.get("record_type") != "prewarm_handoff_complete"
        or type(terminal.get("schema_version")) is not int
        or terminal.get("schema_version") != 1
        or not isinstance(terminal.get("timestamp"), str)
        or not terminal["timestamp"]
        or terminal.get("status") != "passed"
        or isinstance(terminal_elapsed, bool)
        or not isinstance(terminal_elapsed, (int, float))
        or not math.isfinite(terminal_elapsed)
        or terminal_elapsed < 0
        or terminal_elapsed < previous_elapsed
        or terminal.get("baseline") != baseline
        or terminal.get("amdgpu_boot_clean") is not True
        or terminal.get("fatal_amdgpu_events") != []
        or terminal.get("graph_api_used") is not False
        or terminal.get("command_buffer_used") is not False
        or terminal.get("accelerator_device_opened") is not False
        or type(terminal.get("vram_tolerance_bytes")) is not int
        or terminal.get("vram_tolerance_bytes") != 0
        or type(terminal.get("gtt_tolerance_bytes")) is not int
        or terminal.get("gtt_tolerance_bytes") != 0
        or type(terminal.get("final_ready_streak")) is not int
        or terminal["final_ready_streak"] != 3
        or type(terminal.get("sample_count")) is not int
        or terminal.get("sample_count") != len(samples)
    ):
        raise CacheAttestationError("prewarm handoff terminal record is invalid")
    if (
        not isinstance(checks, dict)
        or set(checks) != check_fields
        or any(type(value) is not bool for value in checks.values())
        or checks != computed_checks(final_snapshot)
        or any(value is not True for value in checks.values())
        or terminal.get("final_ready_streak") != samples[-1].get("ready_streak")
    ):
        raise CacheAttestationError("prewarm handoff did not finish exactly idle")
    return {
        "schema_version": 1,
        "status": "passed",
        "boot_id": expected_boot_id,
        "device_identity": baseline.get("device_identity"),
        "sample_count": terminal.get("sample_count"),
        "final_ready_streak": terminal.get("final_ready_streak"),
        "graph_api_used": False,
        "command_buffer_used": False,
        "accelerator_device_opened": False,
    }


def build_startup_cache_claim(
    *,
    prewarm_path: Path,
    prewarm_sha256: str,
    handoff_path: Path,
    handoff_sha256: str,
    expected_git_head: str,
    expected_git_tree: str,
    expected_cache_path: str,
    expected_attention_backend: str,
    boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id"),
) -> dict[str, object]:
    """Validate both launcher artifacts and return one shared API/engine claim."""
    if prewarm_path.name != "prewarm.jsonl" or handoff_path.name != (
        "prewarm-handoff.jsonl"
    ):
        raise CacheAttestationError("startup cache artifact basenames are invalid")
    if prewarm_path.parent != handoff_path.parent:
        raise CacheAttestationError("startup cache artifacts do not share one run dir")
    prewarm_fingerprint, prewarm_payload = stable_private_artifact(
        prewarm_path,
        prewarm_sha256,
        maximum_bytes=_MAX_PREWARM_ARTIFACT_BYTES,
    )
    handoff_fingerprint, handoff_payload = stable_private_artifact(
        handoff_path,
        handoff_sha256,
        maximum_bytes=_MAX_HANDOFF_ARTIFACT_BYTES,
    )
    try:
        boot_id = boot_id_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise CacheAttestationError(f"cannot read current boot ID: {error}") from error
    seed = validate_prewarm_t64_artifact(
        prewarm_payload,
        expected_git_head=expected_git_head,
        expected_git_tree=expected_git_tree,
        expected_cache_path=expected_cache_path,
        expected_attention_backend=expected_attention_backend,
    )
    handoff = validate_completed_handoff_artifact(
        handoff_payload, expected_boot_id=boot_id
    )
    return {
        "status": REQUIREMENT,
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "artifact_parent": str(prewarm_path.parent),
        "prewarm_audit": {
            "path": str(prewarm_path),
            "sha256": prewarm_sha256,
            "fingerprint": prewarm_fingerprint.as_dict(),
        },
        "prewarm_handoff": {
            "path": str(handoff_path),
            "sha256": handoff_sha256,
            "fingerprint": handoff_fingerprint.as_dict(),
        },
        "seed": seed,
        "handoff": handoff,
    }


def revalidate_startup_cache_claim(
    claim: Mapping[str, object],
    *,
    boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id"),
) -> dict[str, object]:
    """Rebuild a previously accepted artifact claim and require byte identity."""
    seed = claim.get("seed")
    prewarm = claim.get("prewarm_audit")
    handoff = claim.get("prewarm_handoff")
    if (
        claim.get("status") != REQUIREMENT
        or claim.get("schema_name") != SCHEMA_NAME
        or claim.get("schema_version") != SCHEMA_VERSION
        or not isinstance(seed, dict)
        or not isinstance(prewarm, dict)
        or not isinstance(handoff, dict)
    ):
        raise CacheAttestationError("startup cache claim schema is invalid")
    required_seed_strings = (
        "source_git_head",
        "source_git_tree",
        "cache_path",
        "attention_backend",
    )
    if any(not isinstance(seed.get(name), str) for name in required_seed_strings):
        raise CacheAttestationError("startup cache seed identity is incomplete")
    try:
        rebuilt = build_startup_cache_claim(
            prewarm_path=Path(str(prewarm["path"])),
            prewarm_sha256=str(prewarm["sha256"]),
            handoff_path=Path(str(handoff["path"])),
            handoff_sha256=str(handoff["sha256"]),
            expected_git_head=str(seed["source_git_head"]),
            expected_git_tree=str(seed["source_git_tree"]),
            expected_cache_path=str(seed["cache_path"]),
            expected_attention_backend=str(seed["attention_backend"]),
            boot_id_path=boot_id_path,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CacheAttestationError(
            "startup cache claim artifact fields are invalid"
        ) from error
    if dict(claim) != rebuilt:
        raise CacheAttestationError(
            "startup cache claim or artifact fingerprints changed"
        )
    return rebuilt


def _shape_signature(
    jax: Any,
    jnp: Any,
    backend: Any,
    *,
    loss_fn_config_class: type | None = None,
) -> tuple[Any, ...]:
    if loss_fn_config_class is None:
        from skyrl.tinker.loss_fns import LossFnConfig

        loss_fn_config_class = LossFnConfig
    batch_2d = jax.NamedSharding(backend.mesh, jax.P("fsdp", None))
    batch_1d = jax.NamedSharding(backend.mesh, jax.P("fsdp"))

    def shape(dimensions: tuple[int, ...], dtype: Any, sharding: Any) -> Any:
        return jax.ShapeDtypeStruct(dimensions, dtype, sharding=sharding)

    return (
        shape((BATCH_SIZE, BUCKET), jnp.int32, batch_2d),
        shape((BATCH_SIZE, BUCKET), jnp.int32, batch_2d),
        shape((BATCH_SIZE,), jnp.int32, batch_1d),
        shape((BATCH_SIZE, BUCKET), jnp.int32, batch_2d),
        shape((BATCH_SIZE, BUCKET), jnp.float32, batch_2d),
        shape((BATCH_SIZE,), jnp.int32, batch_1d),
        shape((BATCH_SIZE, BUCKET), jnp.float32, batch_2d),
        shape((BATCH_SIZE, BUCKET), jnp.float32, batch_2d),
        loss_fn_config_class(
            clip_low_threshold=shape((BATCH_SIZE,), jnp.float32, batch_1d),
            clip_high_threshold=shape((BATCH_SIZE,), jnp.float32, batch_1d),
        ),
    )


def _resolved_backend_config(backend: Any) -> dict[str, object]:
    config = getattr(backend, "config", None)
    if config is None or not hasattr(config, "model_dump"):
        raise CacheAttestationError("runtime JAX backend config is unavailable")
    resolved = config.model_dump(mode="json")
    if not isinstance(resolved, dict):
        raise CacheAttestationError("runtime JAX backend config is not JSON data")
    return resolved


def _target_entry(snapshot: CacheSnapshot, key: str) -> dict[str, object]:
    try:
        pair = snapshot.pairs[key]
    except KeyError as error:
        raise CacheAttestationError(
            "prewarmed T64 executable is absent from the runtime cache"
        ) from error
    return {"key": key, **pair.executable.as_dict()}


def validate_runtime_cache_evidence(
    claim: Mapping[str, object], evidence: Mapping[str, object]
) -> dict[str, object]:
    """Fail closed on any READY evidence not bound to the exact source claim."""
    seed = claim.get("seed")
    prewarm = claim.get("prewarm_audit")
    handoff = claim.get("prewarm_handoff")
    if (
        claim.get("status") != REQUIREMENT
        or not isinstance(seed, dict)
        or not isinstance(prewarm, dict)
        or not isinstance(handoff, dict)
    ):
        raise CacheAttestationError("required runtime cache claim is invalid")
    expected_fields = {
        "schema_name",
        "schema_version",
        "kind",
        "status",
        "compile_target",
        "bucket",
        "batch_size",
        "model",
        "model_revision",
        "model_path",
        "source_git_head",
        "source_git_tree",
        "prewarm_audit_sha256",
        "prewarm_handoff_sha256",
        "backend_config_sha256",
        "attention_backend",
        "cache_path",
        "cache_key",
        "target_cache_entry_before",
        "target_cache_entry_after",
        "target_atime_transition",
        "monitoring",
        "snapshots",
        "jax_stack",
        "operation_wall_start_ns",
        "operation_wall_end_ns",
        "lower_seconds",
        "compile_seconds",
        "xla_flags",
        "graph_api_used",
        "command_buffer_used",
        "lower_calls",
        "compile_calls",
        "compiled_callable_invocations",
        "model_pass_executable_invocations",
        "optimizer_step_invocations",
        "normal_pjit_first_call_seeded",
    }
    if set(evidence) != expected_fields:
        raise CacheAttestationError("runtime cache evidence has an unexpected schema")
    exact_values = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "kind": RUNTIME_HIT_KIND,
        "status": "passed",
        "compile_target": COMPILE_TARGET,
        "bucket": BUCKET,
        "batch_size": BATCH_SIZE,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "model_path": seed.get("model_path"),
        "source_git_head": seed.get("source_git_head"),
        "source_git_tree": seed.get("source_git_tree"),
        "prewarm_audit_sha256": prewarm.get("sha256"),
        "prewarm_handoff_sha256": handoff.get("sha256"),
        "backend_config_sha256": seed.get("backend_config_sha256"),
        "attention_backend": seed.get("attention_backend"),
        "cache_path": seed.get("cache_path"),
        "xla_flags": DISABLE_COMMAND_BUFFERS,
        "graph_api_used": False,
        "command_buffer_used": False,
        "lower_calls": 1,
        "compile_calls": 1,
        "compiled_callable_invocations": 0,
        "model_pass_executable_invocations": 0,
        "optimizer_step_invocations": 0,
        "normal_pjit_first_call_seeded": False,
    }
    mismatches = {
        field: {"expected": expected, "observed": evidence.get(field)}
        for field, expected in exact_values.items()
        if type(evidence.get(field)) is not type(expected)
        or evidence.get(field) != expected
    }
    if mismatches:
        raise CacheAttestationError(
            f"runtime cache evidence does not match its source claim: {mismatches!r}"
        )
    seed_target = seed.get("target_cache_entry")
    if not isinstance(seed_target, dict):
        raise CacheAttestationError("prewarm target cache identity is absent")
    cache_key = seed_target.get("key")
    if (
        evidence.get("cache_key") != cache_key
        or evidence.get("target_cache_entry_before") != seed_target
        or evidence.get("target_cache_entry_after") != seed_target
    ):
        raise CacheAttestationError(
            "runtime executable bytes do not match the prewarm target"
        )
    atime = evidence.get("target_atime_transition")
    seed_atime = seed.get("target_atime_transition")
    if not isinstance(atime, dict) or not isinstance(seed_atime, dict):
        raise CacheAttestationError("runtime logical-atime evidence is absent")
    if set(atime) != {
        "name",
        "before_logical_atime_ns",
        "after_logical_atime_ns",
        "before_sha256",
        "after_sha256",
        "transition",
    }:
        raise CacheAttestationError("runtime logical-atime schema is invalid")
    before_logical_atime_ns = atime.get("before_logical_atime_ns")
    after_logical_atime_ns = atime.get("after_logical_atime_ns")
    operation_wall_start_ns = evidence.get("operation_wall_start_ns")
    operation_wall_end_ns = evidence.get("operation_wall_end_ns")
    if (
        atime.get("name") != f"{cache_key}{_ATIME_SUFFIX}"
        or atime.get("transition") != "rewritten"
        or type(before_logical_atime_ns) is not int
        or before_logical_atime_ns != seed_atime.get("after_logical_atime_ns")
        or type(after_logical_atime_ns) is not int
        or not 0 < before_logical_atime_ns < after_logical_atime_ns < 2**64
        or type(operation_wall_start_ns) is not int
        or type(operation_wall_end_ns) is not int
        or not (
            0
            < operation_wall_start_ns
            <= after_logical_atime_ns
            <= operation_wall_end_ns
        )
        or atime.get("before_sha256")
        != hashlib.sha256(before_logical_atime_ns.to_bytes(8, "little")).hexdigest()
        or atime.get("after_sha256")
        != hashlib.sha256(after_logical_atime_ns.to_bytes(8, "little")).hexdigest()
        or atime.get("before_sha256") != seed_atime.get("after_sha256")
    ):
        raise CacheAttestationError(
            "runtime logical-atime transition is not continuous with prewarm"
        )
    monitoring = evidence.get("monitoring")
    monitoring_fields = {
        "ordered_events",
        "compile_requests_use_cache",
        "cache_hits",
        "cache_misses",
        "compile_time_saved_sec",
        "cache_retrieval_time_sec",
        "schema_issues",
    }
    saved = monitoring.get("compile_time_saved_sec") if isinstance(monitoring, dict) else None
    retrieval = monitoring.get("cache_retrieval_time_sec") if isinstance(monitoring, dict) else None
    if (
        not isinstance(monitoring, dict)
        or set(monitoring) != monitoring_fields
        or monitoring.get("ordered_events")
        != [_EVENT_REQUEST, _EVENT_HIT, _DURATION_SAVED, _DURATION_RETRIEVAL]
        or type(monitoring.get("compile_requests_use_cache")) is not int
        or monitoring["compile_requests_use_cache"] != 1
        or type(monitoring.get("cache_hits")) is not int
        or monitoring["cache_hits"] != 1
        or type(monitoring.get("cache_misses")) is not int
        or monitoring["cache_misses"] != 0
        or monitoring.get("schema_issues") != []
        or not isinstance(saved, list)
        or len(saved) != 1
        or isinstance(saved[0], bool)
        or not isinstance(saved[0], (int, float))
        or not math.isfinite(saved[0])
        or saved[0] <= 0
        or not isinstance(retrieval, list)
        or len(retrieval) != 1
        or isinstance(retrieval[0], bool)
        or not isinstance(retrieval[0], (int, float))
        or not math.isfinite(retrieval[0])
        or retrieval[0] < 0
    ):
        raise CacheAttestationError("runtime public cache monitoring is not exact")
    snapshots = evidence.get("snapshots")
    snapshot_fields = {
        "executable_manifest_before_sha256",
        "executable_manifest_after_sha256",
        "logical_atime_manifest_before_sha256",
        "logical_atime_manifest_after_sha256",
        "auxiliary_manifest_before_sha256",
        "auxiliary_manifest_after_sha256",
        "executable_added",
        "executable_removed",
        "executable_changed",
        "logical_atime_added",
        "logical_atime_removed",
        "logical_atime_changed",
    }
    if (
        not isinstance(snapshots, dict)
        or set(snapshots) != snapshot_fields
        or any(
            not _is_sha256(snapshots.get(field))
            for field in (
                "executable_manifest_before_sha256",
                "executable_manifest_after_sha256",
                "logical_atime_manifest_before_sha256",
                "logical_atime_manifest_after_sha256",
                "auxiliary_manifest_before_sha256",
                "auxiliary_manifest_after_sha256",
            )
        )
        or any(
            snapshots.get(name) != []
            for name in (
                "executable_added",
                "executable_removed",
                "executable_changed",
                "logical_atime_added",
                "logical_atime_removed",
            )
        )
        or snapshots.get("logical_atime_changed") != [cache_key]
    ):
        raise CacheAttestationError("runtime cache manifests contain extra mutations")
    if snapshots.get("executable_manifest_before_sha256") != snapshots.get(
        "executable_manifest_after_sha256"
    ) or snapshots.get("auxiliary_manifest_before_sha256") != snapshots.get(
        "auxiliary_manifest_after_sha256"
    ) or snapshots.get("logical_atime_manifest_before_sha256") == snapshots.get(
        "logical_atime_manifest_after_sha256"
    ):
        raise CacheAttestationError("runtime executable or autotune bytes changed")
    for field in ("lower_seconds", "compile_seconds"):
        value = evidence.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise CacheAttestationError(f"runtime {field} is invalid")
    jax_stack = evidence.get("jax_stack")
    if not isinstance(jax_stack, dict) or jax_stack != {
        "jax": "0.10.2",
        "jaxlib": "0.10.2",
        "jax-rocm7-plugin": "0.10.2",
        "jax-rocm7-pjrt": "0.10.2",
    }:
        raise CacheAttestationError("runtime JAX stack evidence is invalid")
    return dict(evidence)


def run_engine_t64_cache_attestation(
    backend: Any,
    claim: Mapping[str, object],
    *,
    jax_module: Any | None = None,
    jnp_module: Any | None = None,
    loss_fn_config_class: type | None = None,
    snapshot_fn: Callable[[Path], CacheSnapshot] = snapshot_cache,
    wall_time_ns: Callable[[], int] = time.time_ns,
    perf_counter: Callable[[], float] = time.perf_counter,
    boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id"),
) -> dict[str, object]:
    """Prove one exact real-backend T64 AOT hit without invoking the result."""
    verified_claim = revalidate_startup_cache_claim(claim, boot_id_path=boot_id_path)
    seed = verified_claim["seed"]
    assert isinstance(seed, dict)
    if os.environ.get("XLA_FLAGS") != DISABLE_COMMAND_BUFFERS:
        raise CacheAttestationError("runtime XLA command-buffer policy changed")
    if os.environ.get("JAX_COMPILATION_CACHE_DIR") != seed.get("cache_path"):
        raise CacheAttestationError("runtime JAX cache path changed")
    expected_pallas = "1" if seed.get("attention_backend") == "pallas" else "0"
    if os.environ.get("SKYRL_ROCM_PALLAS_ATTENTION") != expected_pallas:
        raise CacheAttestationError("runtime Pallas attention policy changed")
    resolved_config = _resolved_backend_config(backend)
    if resolved_config != seed.get("backend_config") or (
        canonical_json_sha256(
            resolved_config,
            domain="skyrl-qwen35-resolved-jax-backend-config-v1",
        )
        != seed.get("backend_config_sha256")
    ):
        raise CacheAttestationError("runtime JAX backend config changed from prewarm")
    if resolved_config.get("enforce_eager") is not False:
        raise CacheAttestationError("runtime JAX backend has disabled JIT")
    try:
        runtime_model_path = str(Path(backend.base_model).resolve(strict=True))
    except (AttributeError, OSError) as error:
        raise CacheAttestationError("runtime model path is unavailable") from error
    if runtime_model_path != seed.get("model_path"):
        raise CacheAttestationError("runtime model revision path changed from prewarm")

    if jax_module is None or jnp_module is None:
        import jax
        import jax.numpy as jnp

        jax_module = jax
        jnp_module = jnp
    jax = jax_module
    jnp = jnp_module
    devices = tuple(jax.devices())
    if (
        jax.process_count() != 1
        or jax.device_count() != 1
        or len(devices) != 1
        or getattr(devices[0], "platform", None) != "gpu"
    ):
        raise CacheAttestationError(
            "T64 cache attestation requires one JAX process and one GPU device"
        )
    model_pass = getattr(backend, "_forward_backward_and_accumulate", None)
    if model_pass is None or not hasattr(model_pass, "lower"):
        raise CacheAttestationError("runtime JAX model pass does not expose lower()")

    # Finish every constructor dispatch before observing the cache transition.
    jax.block_until_ready(
        (
            backend.accumulated_grads,
            backend.lora_params,
            backend.non_lora_params,
        )
    )
    cache_path = Path(str(seed["cache_path"]))
    target = seed.get("target_cache_entry")
    seed_atime = seed.get("target_atime_transition")
    if not isinstance(target, dict) or not isinstance(seed_atime, dict):
        raise CacheAttestationError("prewarm target identity is incomplete")
    cache_key = target.get("key")
    if not isinstance(cache_key, str):
        raise CacheAttestationError("prewarm target cache key is absent")
    before = snapshot_fn(cache_path)
    entry_before = _target_entry(before, cache_key)
    if entry_before != target or before.pairs[cache_key].logical_atime_ns != (
        seed_atime.get("after_logical_atime_ns")
    ):
        raise CacheAttestationError(
            "runtime T64 cache entry is not continuous with prewarm"
        )

    signature = _shape_signature(
        jax,
        jnp,
        backend,
        loss_fn_config_class=loss_fn_config_class,
    )
    capture = PublicCacheMonitoringCapture(jax.monitoring)
    compiled: Any | None = None
    lower_start = perf_counter()
    operation_wall_start_ns = wall_time_ns()
    with capture:
        mesh_context = (
            jax.set_mesh(backend.mesh) if hasattr(jax, "set_mesh") else _NullContext()
        )
        with mesh_context:
            lowered = model_pass.lower(
                backend.accumulated_grads,
                backend.lora_params,
                backend.non_lora_params,
                *signature,
            )
        lower_seconds = perf_counter() - lower_start
        compile_start = perf_counter()
        compiled = lowered.compile()
        compile_seconds = perf_counter() - compile_start
    operation_wall_end_ns = wall_time_ns()
    trace = capture.trace()
    # Deliberately discard the AOT object.  It is never called or exported.
    del compiled

    after = snapshot_fn(cache_path)
    transition = compare_cache_transition(
        before,
        after,
        trace,
        operation_wall_start_ns=operation_wall_start_ns,
        operation_wall_end_ns=operation_wall_end_ns,
        expected_key=cache_key,
        require_hit=True,
    )
    entry_after = _target_entry(after, cache_key)
    if entry_before != target or entry_after != target:
        raise CacheAttestationError("runtime T64 executable bytes changed")

    from importlib import metadata

    evidence = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "kind": RUNTIME_HIT_KIND,
        "status": "passed",
        "compile_target": COMPILE_TARGET,
        "bucket": BUCKET,
        "batch_size": BATCH_SIZE,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "model_path": seed["model_path"],
        "source_git_head": seed["source_git_head"],
        "source_git_tree": seed["source_git_tree"],
        "prewarm_audit_sha256": verified_claim["prewarm_audit"]["sha256"],
        "prewarm_handoff_sha256": verified_claim["prewarm_handoff"]["sha256"],
        "backend_config_sha256": seed["backend_config_sha256"],
        "attention_backend": seed["attention_backend"],
        "cache_path": seed["cache_path"],
        "cache_key": cache_key,
        "target_cache_entry_before": entry_before,
        "target_cache_entry_after": entry_after,
        "target_atime_transition": transition["target_atime_transition"],
        "monitoring": transition["monitoring"],
        "snapshots": transition["snapshots"],
        "jax_stack": {
            package: metadata.version(package)
            for package in (
                "jax",
                "jaxlib",
                "jax-rocm7-plugin",
                "jax-rocm7-pjrt",
            )
        },
        "operation_wall_start_ns": transition["operation_wall_start_ns"],
        "operation_wall_end_ns": transition["operation_wall_end_ns"],
        "lower_seconds": lower_seconds,
        "compile_seconds": compile_seconds,
        "xla_flags": DISABLE_COMMAND_BUFFERS,
        "graph_api_used": False,
        "command_buffer_used": False,
        "lower_calls": 1,
        "compile_calls": 1,
        "compiled_callable_invocations": 0,
        "model_pass_executable_invocations": 0,
        "optimizer_step_invocations": 0,
        "normal_pjit_first_call_seeded": False,
    }
    return validate_runtime_cache_evidence(verified_claim, evidence)


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False
