"""Fail-closed local process attestation for the Qwen3.5 ROCm server.

This module is deliberately standard-library-only.  It accepts only the
hardened source-snapshot form of ``rocm/start_qwen35.sh`` (a nonempty prewarm
bucket list), binds its Python API child to the loopback listening socket, and
uses the private WAL readiness database to identify the engine launcher and
engine process.  It never imports JAX or any SkyRL server module.

The result is a point-in-time local contract, not proof of historical process
execution or remote attestation.  In particular, ``exec`` replaces the shell
launcher before this observer runs.  The seal independently revalidates the
full private source cache against the benchmark's clean Git HEAD/tree and
streams every required model blob twice using its repository content ID.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import sqlite3
import stat
import sys
import time
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import quote, urlsplit

API_MODULE = "skyrl.tinker.api"
ENGINE_MODULE = "skyrl.tinker.engine"
EXPECTED_XLA_FLAGS = "--xla_gpu_enable_command_buffer="
EXPECTED_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EXPECTED_MODEL_CACHE_DIRECTORY = "models--Qwen--Qwen3.5-4B"
EXPECTED_UV_SHA256 = "646adf5cf12ba17d1a41fa77c8dd6496f73651dcfeeed6b5f4ec019b36bc7153"
RUNTIME_CACHE_HIT_KIND = "strict_aot_t64_persistent_cache_hit_v1"

_EXPECTED_MODEL_BLOBS = {
    "chat_template.jinja": "a585dec894e63da457d9440ec6aa7caa16d20860",
    "config.json": "557d961b205319c6a7da5f757f565b69b3967b7d",
    "merges.txt": "a494e019ca1502219fd0128658b979e5f05ae8e8",
    "model.safetensors-00001-of-00002.safetensors": (
        "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61"
    ),
    "model.safetensors-00002-of-00002.safetensors": (
        "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188"
    ),
    "model.safetensors.index.json": "fddda6039f7c1d17260c9e923b8a72fd025d9a86",
    "preprocessor_config.json": "2ea84a437d448ff71b08df68fdd949d5cc4ebb64",
    "tokenizer.json": (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    ),
    "tokenizer_config.json": "eda48d3e75a8e59a8479ee4ec8b37f76e711d9c1",
    "video_preprocessor_config.json": ("3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe"),
    "vocab.json": "0aa0ce0658d60ac4a5d609f4eadb0e8e43514176",
}
_EXPECTED_OPTIONAL_INERT_MODEL_BLOBS = {
    ".gitattributes": "52373fe24473b1aa44333d318f578ae6bf04b49b",
    "LICENSE": "f938136e3adacfd92be087f6e113b5d6d97f678f",
    "README.md": "7950a3aadf378cd13758097bc52f0ed849a59007",
}

_REQUIRED_RUNTIME_ENVIRONMENT = {
    "HF_XET_HIGH_PERFORMANCE": "1",
    "JAX_COMPILATION_CACHE_EXPECT_PGLE": "false",
    "JAX_COMPILATION_CACHE_MAX_SIZE": "17179869184",
    "JAX_ENABLE_COMPILATION_CACHE": "true",
    "JAX_ENABLE_PGLE": "false",
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": ("xla_gpu_per_fusion_autotune_cache_dir"),
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
    "XLA_FLAGS": EXPECTED_XLA_FLAGS,
    "JAX_PLATFORMS": "rocm",
    "JAX_RAISE_PERSISTENT_CACHE_ERRORS": "true",
    "LLVM_PATH": "/opt/rocm/llvm",
    "PYTHONDONTWRITEBYTECODE": "1",
    "ROCR_VISIBLE_DEVICES": "0",
}
_GROWTH_ENVIRONMENT = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
_PREALLOCATE_ENVIRONMENT = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
    "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
    "XLA_CLIENT_MEM_FRACTION": "0.85",
    "HIP_VISIBLE_DEVICES": "0",
    "GPU_DEVICE_ORDINAL": "0",
}
_VARIABLE_RUNTIME_ENVIRONMENT_NAMES = frozenset(
    {"JAX_COMPILATION_CACHE_DIR", "SKYRL_ROCM_PALLAS_ATTENTION"}
)
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
_INTERPRETER_INJECTION_ENVIRONMENT = frozenset(
    {
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
)

_CORE_RUNTIME_SOURCE_ENVIRONMENT = frozenset(
    {
        "SKYRL_QWEN35_RUNTIME_GIT_HEAD",
        "SKYRL_QWEN35_RUNTIME_MEMORY_MODE",
        "SKYRL_QWEN35_RUNTIME_REPO_ROOT",
        "SKYRL_QWEN35_RUNTIME_SOURCE_ROOT",
        "SKYRL_QWEN35_RUNTIME_UV_EXECUTABLE",
    }
)
_CACHE_RUNTIME_SOURCE_ENVIRONMENT = frozenset(
    {
        "SKYRL_QWEN35_RUNTIME_T64_CACHE_ATTEST",
        "SKYRL_QWEN35_RUNTIME_PREWARM_AUDIT_PATH",
        "SKYRL_QWEN35_RUNTIME_PREWARM_AUDIT_SHA256",
        "SKYRL_QWEN35_RUNTIME_PREWARM_HANDOFF_PATH",
        "SKYRL_QWEN35_RUNTIME_PREWARM_HANDOFF_SHA256",
    }
)
_QUALIFIED_PROCESS_ENVIRONMENT = frozenset(
    {
        *_CORE_RUNTIME_SOURCE_ENVIRONMENT,
        *_CACHE_RUNTIME_SOURCE_ENVIRONMENT,
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "UV",
        "UV_RUN_RECURSION_DEPTH",
        "VIRTUAL_ENV",
        "XDG_RUNTIME_DIR",
    }
)

_BOOT_ID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
_LAUNCH_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
_ENVIRONMENT_NAME_PATTERN = re.compile(rb"[A-Za-z_][A-Za-z0-9_]*")
_NON_OPERATIONAL_STATES = frozenset({"T", "X", "Z", "t", "x"})

_API_OPTION_ORDER = (
    "--base-model",
    "--backend",
    "--backend-config",
    "--host",
    "--port",
    "--checkpoints-base",
    "--engine-startup-timeout-sec",
    "--database-url",
)
_ENGINE_OPTION_ORDER = (
    "--base-model",
    "--backend",
    "--backend-config",
    "--checkpoints-base",
    "--database-url",
    "--external-inference-api-key",
    "--external-inference-lora-base",
    "--session-cleanup-interval-sec",
    "--session-timeout-sec",
    "--startup-launch-id",
    "--engine-startup-timeout-sec",
)
_BACKEND_CONFIG_TYPES = {
    "max_lora_adapters": int,
    "max_lora_rank": int,
    "train_micro_batch_size": int,
    "sample_max_num_sequences": int,
    "gradient_checkpointing": bool,
    "loss_chunk_size": int,
    "abstract_model_load": bool,
}
_BACKEND_CONFIG_FIXED = {
    "max_lora_adapters": 2,
    "max_lora_rank": 8,
    "train_micro_batch_size": 1,
    "sample_max_num_sequences": 1,
    "gradient_checkpointing": True,
    "loss_chunk_size": 64,
}
_ENGINE_LAUNCH_SCHEMA = (
    (0, "launch_id", "VARCHAR", 1, None, 1),
    (1, "backend", "VARCHAR", 1, None, 0),
    (2, "status", "VARCHAR(12)", 1, None, 0),
    (3, "boot_id", "VARCHAR", 1, None, 0),
    (4, "api_pid", "INTEGER", 1, None, 0),
    (5, "api_start_ticks", "BIGINT", 1, None, 0),
    (6, "engine_launcher_pid", "INTEGER", 0, None, 0),
    (7, "engine_launcher_start_ticks", "BIGINT", 0, None, 0),
    (8, "engine_pid", "INTEGER", 0, None, 0),
    (9, "engine_start_ticks", "BIGINT", 0, None, 0),
    (10, "api_source_attestation", "JSON", 1, None, 0),
    (11, "api_launch_lock_attestation", "JSON", 1, None, 0),
    (12, "engine_source_attestation", "JSON", 1, None, 0),
    (13, "engine_launch_lock_attestation", "JSON", 1, None, 0),
    (14, "runtime_handoff_attestation", "JSON", 1, None, 0),
    (15, "cache_evidence_status", "VARCHAR", 1, None, 0),
    (16, "cache_evidence", "JSON", 1, None, 0),
    (17, "error_message", "VARCHAR", 0, None, 0),
    (18, "heartbeat_at", "DATETIME", 0, None, 0),
    (19, "heartbeat_monotonic_ns", "BIGINT", 0, None, 0),
    (20, "heartbeat_sequence", "BIGINT", 1, None, 0),
    (21, "created_at", "DATETIME", 1, None, 0),
    (22, "updated_at", "DATETIME", 1, None, 0),
    (23, "ready_at", "DATETIME", 0, None, 0),
)
_ENGINE_LAUNCH_SCHEMA_COLUMNS = tuple(row[1] for row in _ENGINE_LAUNCH_SCHEMA)
_ENGINE_LAUNCH_INDEXES = {
    "ix_engine_launches_status": (False, "c", False, "status"),
    "ix_engine_launches_cache_evidence_status": (
        False,
        "c",
        False,
        "cache_evidence_status",
    ),
    "sqlite_autoindex_engine_launches_1": (True, "pk", False, "launch_id"),
}
_DYNAMIC_LAUNCH_FIELDS = frozenset(
    {"heartbeat_at", "heartbeat_monotonic_ns", "heartbeat_sequence", "updated_at"}
)
_MAX_HEARTBEAT_AGE_SECONDS = 5.0

HealthProbe = Callable[[str, int], Mapping[str, Any]]
SourceCacheValidator = Callable[[Path, str, Path], Mapping[str, Any]]
CacheEvidenceValidator = Callable[
    [Path, Path, Mapping[str, Any], Mapping[str, Any], Path], None
]


class LocalServerAttestationError(RuntimeError):
    """The observed local server does not satisfy the exact launch contract."""


@dataclass(frozen=True, slots=True)
class _ProcessSnapshot:
    pid: int
    ppid: int
    process_group: int
    session: int
    state: str
    start_ticks: int
    boot_id: str
    uid: int
    network_namespace: tuple[int, int]
    argv: tuple[str, ...]
    cmdline_sha256: str
    executable_path: str
    executable_name: str
    executable_sha256: str
    working_directory: str
    launch_lock_fd: int
    xla_flags_sha256: str
    accelerator_environment: tuple[tuple[str, str], ...]
    qualified_environment: tuple[tuple[str, str], ...]

    def public_evidence(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "ppid": self.ppid,
            "process_group": self.process_group,
            "session": self.session,
            "start_ticks": self.start_ticks,
            "cmdline_sha256": self.cmdline_sha256,
            "executable_sha256": self.executable_sha256,
            "working_directory_sha256": _sha256_bytes(
                self.working_directory.encode("utf-8")
            ),
            "xla_flags_sha256": self.xla_flags_sha256,
            "accelerator_environment_sha256": _sha256_bytes(
                _canonical_json(dict(self.accelerator_environment)).encode("ascii")
            ),
            "qualified_environment_sha256": _sha256_bytes(
                _canonical_json(dict(self.qualified_environment)).encode("ascii")
            ),
        }


@dataclass(frozen=True, slots=True)
class _DatabaseIdentity:
    device: int
    inode: int
    mode: int
    uid: int
    links: int
    parent_device: int
    parent_inode: int
    parent_mode: int
    parent_uid: int


@dataclass(frozen=True, slots=True)
class LocalServerAttestationSeal:
    """Opaque stable contract plus its allowlisted public record."""

    server_pid: int
    base_url: str
    contract_sha256: str
    _contract_json: str
    _record_json: str
    _expected_git_head: str
    _expected_git_tree: str
    _expected_repo_root: Path
    _expected_python_sha256: str
    _verified_model_snapshot: Path
    _runtime_root: Path
    _require_startup_cache: bool
    _source_cache_validator: SourceCacheValidator
    _cache_evidence_validator: CacheEvidenceValidator

    def as_record(self) -> dict[str, Any]:
        """Return a fresh JSON-compatible copy of the public evidence."""
        result = json.loads(self._record_json)
        if not isinstance(result, dict):  # pragma: no cover - construction invariant
            raise AssertionError("attestation record is not an object")
        return result

    def verified_model_snapshot(self) -> Path:
        """Return the exact pinned snapshot whose required blobs were verified."""
        return self._verified_model_snapshot


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


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


def _hash_regular_file(path: Path, label: str) -> tuple[str, str]:
    try:
        resolved = path.resolve(strict=True)
        with path.open("rb", buffering=0) as source:
            metadata_before = os.fstat(source.fileno())
            endpoint_before = path.stat()
            if not stat.S_ISREG(metadata_before.st_mode):
                raise LocalServerAttestationError(f"{label} is not a regular file")
            if _file_fingerprint(metadata_before) != _file_fingerprint(endpoint_before):
                raise LocalServerAttestationError(
                    f"{label} path changed before hashing"
                )
            digest = hashlib.sha256()
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
            metadata_after = os.fstat(source.fileno())
            endpoint_after = path.stat()
            if _file_fingerprint(metadata_after) != _file_fingerprint(
                metadata_before
            ) or _file_fingerprint(endpoint_after) != _file_fingerprint(metadata_after):
                raise LocalServerAttestationError(f"{label} changed while hashing")
    except LocalServerAttestationError:
        raise
    except OSError as error:
        raise LocalServerAttestationError(f"cannot hash {label}: {error}") from error
    if resolved.name.endswith(" (deleted)"):
        raise LocalServerAttestationError(f"{label} executable has been deleted")
    return str(resolved), digest.hexdigest()


def _parse_nul_argv(raw: bytes, label: str) -> tuple[str, ...]:
    if not raw or not raw.endswith(b"\0"):
        raise LocalServerAttestationError(
            f"{label} command line is empty or unterminated"
        )
    fields = raw[:-1].split(b"\0")
    if not fields or any(not field for field in fields):
        raise LocalServerAttestationError(f"{label} command line has empty arguments")
    try:
        return tuple(field.decode("utf-8") for field in fields)
    except UnicodeDecodeError as error:
        raise LocalServerAttestationError(
            f"{label} command line is not canonical UTF-8"
        ) from error


def _parse_environment(raw: bytes, label: str) -> dict[bytes, bytes]:
    if not raw or not raw.endswith(b"\0"):
        raise LocalServerAttestationError(
            f"{label} environment is empty or unterminated"
        )
    environment: dict[bytes, bytes] = {}
    for entry in raw[:-1].split(b"\0"):
        if not entry or b"=" not in entry:
            raise LocalServerAttestationError(f"{label} environment is malformed")
        name, value = entry.split(b"=", 1)
        if _ENVIRONMENT_NAME_PATTERN.fullmatch(name) is None:
            raise LocalServerAttestationError(
                f"{label} environment contains a malformed name"
            )
        if name in environment:
            decoded = name.decode("ascii")
            raise LocalServerAttestationError(
                f"{label} environment contains duplicate {decoded} entries"
            )
        environment[name] = value
    return environment


def _parse_process_stat(raw: str, expected_pid: int) -> tuple[str, int, int, int, int]:
    record = raw.removesuffix("\n")
    open_parenthesis = record.find("(")
    close_parenthesis = record.rfind(")")
    if (
        open_parenthesis <= 0
        or close_parenthesis <= open_parenthesis
        or close_parenthesis + 2 > len(record)
        or record[close_parenthesis + 1] != " "
    ):
        raise LocalServerAttestationError("Linux process stat record is malformed")
    try:
        observed_pid = int(record[:open_parenthesis].strip())
    except ValueError as error:
        raise LocalServerAttestationError(
            "Linux process stat PID is malformed"
        ) from error
    if observed_pid != expected_pid:
        raise LocalServerAttestationError("Linux process stat PID changed")
    fields = record[close_parenthesis + 2 :].split()
    if len(fields) < 20 or len(fields[0]) != 1:
        raise LocalServerAttestationError("Linux process stat fields are truncated")
    try:
        ppid = int(fields[1])
        process_group = int(fields[2])
        session = int(fields[3])
        start_ticks = int(fields[19])
    except ValueError as error:
        raise LocalServerAttestationError(
            "Linux process stat numeric fields are malformed"
        ) from error
    if min(ppid, process_group, session) < 0 or start_ticks <= 0:
        raise LocalServerAttestationError("Linux process stat fields are out of range")
    return fields[0], ppid, process_group, session, start_ticks


def _read_uid(status_path: Path, expected_uid: int, label: str) -> int:
    try:
        lines = status_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise LocalServerAttestationError(
            f"cannot read {label} status: {error}"
        ) from error
    uid_lines = [line for line in lines if line.startswith("Uid:")]
    if len(uid_lines) != 1:
        raise LocalServerAttestationError(f"{label} status has no unique UID record")
    try:
        values = tuple(int(value) for value in uid_lines[0].split()[1:])
    except ValueError as error:
        raise LocalServerAttestationError(f"{label} UID record is malformed") from error
    if len(values) != 4 or any(value != expected_uid for value in values):
        raise LocalServerAttestationError(f"{label} is not wholly owned by this user")
    return expected_uid


def _read_boot_id(proc_root: Path) -> str:
    try:
        boot_id = (
            (proc_root / "sys/kernel/random/boot_id")
            .read_text(encoding="ascii")
            .strip()
        )
    except (OSError, UnicodeError) as error:
        raise LocalServerAttestationError(
            f"cannot read Linux boot ID: {error}"
        ) from error
    if _BOOT_ID_PATTERN.fullmatch(boot_id) is None:
        raise LocalServerAttestationError("Linux boot ID is malformed")
    return boot_id


def _read_process(
    pid: int,
    *,
    proc_root: Path,
    boot_id: str,
    expected_uid: int,
    label: str,
) -> _ProcessSnapshot:
    if type(pid) is not int or pid <= 1:
        raise LocalServerAttestationError(
            f"{label} PID must be an integer greater than 1"
        )
    process_root = proc_root / str(pid)
    try:
        directory_metadata = process_root.stat(follow_symlinks=False)
        raw_stat = (process_root / "stat").read_text(encoding="utf-8")
        raw_cmdline = (process_root / "cmdline").read_bytes()
        raw_environment = (process_root / "environ").read_bytes()
        namespace_metadata = (process_root / "ns/net").stat()
        working_directory = (process_root / "cwd").resolve(strict=True)
        working_metadata = working_directory.stat(follow_symlinks=False)
    except (OSError, UnicodeError) as error:
        raise LocalServerAttestationError(f"cannot inspect {label}: {error}") from error
    if not stat.S_ISDIR(directory_metadata.st_mode) or (
        directory_metadata.st_uid != expected_uid
    ):
        raise LocalServerAttestationError(f"{label} proc directory has the wrong owner")
    state, ppid, process_group, session, start_ticks = _parse_process_stat(
        raw_stat, pid
    )
    if state in _NON_OPERATIONAL_STATES:
        raise LocalServerAttestationError(f"{label} is not operational")
    uid = _read_uid(process_root / "status", expected_uid, label)
    argv = _parse_nul_argv(raw_cmdline, label)
    environment = _parse_environment(raw_environment, label)
    injection_names = sorted(
        name
        for name in _INTERPRETER_INJECTION_ENVIRONMENT
        if environment.get(name.encode("ascii"), b"")
    )
    if injection_names:
        raise LocalServerAttestationError(
            f"{label} has interpreter injection environment: {injection_names!r}"
        )
    environment_names = {name.decode("ascii") for name in environment}
    unexpected_uv = sorted(
        name
        for name in environment_names
        if (name == "UV" or name.startswith("UV_"))
        and name not in {"UV", "UV_RUN_RECURSION_DEPTH"}
    )
    unexpected_python = sorted(
        name
        for name in environment_names
        if name.startswith("PYTHON") and name != "PYTHONDONTWRITEBYTECODE"
    )
    if unexpected_uv or unexpected_python:
        raise LocalServerAttestationError(
            f"{label} has unexpected uv/Python environment; "
            f"uv={unexpected_uv!r}, python={unexpected_python!r}"
        )
    unexpected_runtime_source = sorted(
        name
        for name in environment_names
        if name.startswith("SKYRL_QWEN35_RUNTIME_")
        and name not in _QUALIFIED_PROCESS_ENVIRONMENT
    )
    if unexpected_runtime_source:
        raise LocalServerAttestationError(
            f"{label} has unexpected runtime-source environment: "
            f"{unexpected_runtime_source!r}"
        )
    if (
        not stat.S_ISDIR(working_metadata.st_mode)
        or working_metadata.st_uid != expected_uid
    ):
        raise LocalServerAttestationError(
            f"{label} working directory is not an owned directory"
        )
    accelerator_environment: dict[str, str] = {}
    for raw_name, raw_value in environment.items():
        name = raw_name.decode("ascii")
        if not (
            name in _REQUIRED_RUNTIME_ENVIRONMENT
            or name in _GROWTH_ENVIRONMENT
            or name in _PREALLOCATE_ENVIRONMENT
            or name in _VARIABLE_RUNTIME_ENVIRONMENT_NAMES
            or name.startswith(_ACCELERATOR_ENVIRONMENT_PREFIXES)
        ):
            continue
        try:
            accelerator_environment[name] = raw_value.decode("ascii")
        except UnicodeDecodeError as error:
            raise LocalServerAttestationError(
                f"{label} {name} is not canonical ASCII"
            ) from error
    qualified_environment: dict[str, str] = {}
    for name in sorted(_QUALIFIED_PROCESS_ENVIRONMENT):
        raw_value = environment.get(name.encode("ascii"))
        if raw_value is None:
            continue
        try:
            qualified_environment[name] = raw_value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise LocalServerAttestationError(
                f"{label} {name} is not canonical UTF-8"
            ) from error
    for name, expected in _REQUIRED_RUNTIME_ENVIRONMENT.items():
        if accelerator_environment.get(name) != expected:
            raise LocalServerAttestationError(
                f"{label} {name} is not exactly {expected!r}"
            )
    raw_lock_fd = environment.get(b"SKYRL_QWEN35_LAUNCH_LOCK_FD")
    try:
        launch_lock_fd = int(raw_lock_fd) if raw_lock_fd is not None else -1
    except ValueError as error:
        raise LocalServerAttestationError(
            f"{label} launch-lock descriptor is malformed"
        ) from error
    if (
        raw_lock_fd is None
        or launch_lock_fd < 3
        or raw_lock_fd != str(launch_lock_fd).encode("ascii")
    ):
        raise LocalServerAttestationError(
            f"{label} launch-lock descriptor is absent or noncanonical"
        )
    executable_path, executable_sha256 = _hash_regular_file(process_root / "exe", label)
    return _ProcessSnapshot(
        pid=pid,
        ppid=ppid,
        process_group=process_group,
        session=session,
        state=state,
        start_ticks=start_ticks,
        boot_id=boot_id,
        uid=uid,
        network_namespace=(namespace_metadata.st_dev, namespace_metadata.st_ino),
        argv=argv,
        cmdline_sha256=_sha256_bytes(raw_cmdline),
        executable_path=executable_path,
        executable_name=Path(executable_path).name,
        executable_sha256=executable_sha256,
        working_directory=str(working_directory),
        launch_lock_fd=launch_lock_fd,
        xla_flags_sha256=_sha256_bytes(
            b"XLA_FLAGS=" + EXPECTED_XLA_FLAGS.encode("ascii") + b"\0"
        ),
        accelerator_environment=tuple(sorted(accelerator_environment.items())),
        qualified_environment=tuple(sorted(qualified_environment.items())),
    )


def _require_runtime_environment(
    process: _ProcessSnapshot, *, abstract_model_load: bool, label: str
) -> tuple[str, str]:
    environment = dict(process.accelerator_environment)
    memory_environment = (
        _PREALLOCATE_ENVIRONMENT if abstract_model_load else _GROWTH_ENVIRONMENT
    )
    mode = "preallocate85" if abstract_model_load else "growth"
    expected_names = {
        *_REQUIRED_RUNTIME_ENVIRONMENT,
        *memory_environment,
        *_VARIABLE_RUNTIME_ENVIRONMENT_NAMES,
    }
    if environment.keys() != expected_names:
        unexpected = sorted(environment.keys() - expected_names)
        missing = sorted(expected_names - environment.keys())
        raise LocalServerAttestationError(
            f"{label} {mode} accelerator environment is not exact; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    for name, value in {**_REQUIRED_RUNTIME_ENVIRONMENT, **memory_environment}.items():
        if environment[name] != value:
            raise LocalServerAttestationError(
                f"{label} {name} does not match {mode} memory mode"
            )
    pallas_attention = environment["SKYRL_ROCM_PALLAS_ATTENTION"]
    if pallas_attention not in {"0", "1"}:
        raise LocalServerAttestationError(
            f"{label} SKYRL_ROCM_PALLAS_ATTENTION must be exactly 0 or 1"
        )
    cache_directory = environment["JAX_COMPILATION_CACHE_DIR"]
    cache_path = Path(cache_directory)
    try:
        cache_metadata = cache_path.stat(follow_symlinks=False)
        cache_resolved = cache_path.resolve(strict=True)
    except OSError as error:
        raise LocalServerAttestationError(
            f"{label} JAX compilation cache cannot be inspected: {error}"
        ) from error
    if (
        not cache_path.is_absolute()
        or cache_resolved != cache_path
        or not stat.S_ISDIR(cache_metadata.st_mode)
        or cache_metadata.st_uid != process.uid
        or stat.S_IMODE(cache_metadata.st_mode) != 0o700
    ):
        raise LocalServerAttestationError(
            f"{label} JAX compilation cache is not canonical and private"
        )
    return pallas_attention, cache_directory


def _require_python_payload(
    process: _ProcessSnapshot,
    *,
    expected_repo_root: Path,
    expected_python_sha256: str,
    label: str,
) -> None:
    argv_path = Path(process.argv[0])
    expected_bin = expected_repo_root / ".venv" / "bin"
    try:
        resolved = argv_path.resolve(strict=True)
        bin_resolved = argv_path.parent.resolve(strict=True)
    except OSError as error:
        raise LocalServerAttestationError(
            f"{label} Python argv executable cannot be resolved: {error}"
        ) from error
    if (
        not argv_path.is_absolute()
        or bin_resolved != expected_bin
        or argv_path.name not in {"python", "python3", "python3.12"}
        or resolved != Path(process.executable_path)
        or process.executable_sha256 != expected_python_sha256
    ):
        raise LocalServerAttestationError(
            f"{label} does not use the benchmark's exact venv Python payload"
        )


def _require_qualified_process_environment(
    process: _ProcessSnapshot,
    *,
    role: str,
    source: Mapping[str, Any],
    runtime_root: Path,
) -> None:
    source_root = Path(source["source_root"])
    account_home = source_root.parents[3]
    repo_root = Path(source["repo_root"])
    venv_bin = repo_root / ".venv" / "bin"
    expected = {
        "HOME": str(account_home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "VIRTUAL_ENV": str(repo_root / ".venv"),
        "XDG_RUNTIME_DIR": str(runtime_root),
        "SKYRL_QWEN35_RUNTIME_GIT_HEAD": source["git_head"],
        "SKYRL_QWEN35_RUNTIME_MEMORY_MODE": source["memory_mode"],
        "SKYRL_QWEN35_RUNTIME_REPO_ROOT": str(repo_root),
        "SKYRL_QWEN35_RUNTIME_SOURCE_ROOT": str(source_root),
        "SKYRL_QWEN35_RUNTIME_UV_EXECUTABLE": source["uv_executable"],
    }
    if role == "outer":
        expected["PATH"] = "/opt/rocm/bin:/usr/bin:/bin"
    else:
        uv_depth = "2" if role == "engine" else "1"
        expected.update(
            {
                "PATH": ":".join(
                    [str(venv_bin)] * int(uv_depth)
                    + ["/opt/rocm/bin", "/usr/bin", "/bin"]
                ),
                "UV": source["uv_executable"],
                "UV_RUN_RECURSION_DEPTH": uv_depth,
            }
        )
    startup_cache = source["startup_cache_attestation"]
    if startup_cache["status"] == "required-v1":
        prewarm = startup_cache.get("prewarm_audit")
        handoff = startup_cache.get("prewarm_handoff")
        if not isinstance(prewarm, dict) or not isinstance(handoff, dict):
            raise LocalServerAttestationError(
                "required startup-cache paths are absent from source evidence"
            )
        try:
            expected.update(
                {
                    "SKYRL_QWEN35_RUNTIME_T64_CACHE_ATTEST": "required-v1",
                    "SKYRL_QWEN35_RUNTIME_PREWARM_AUDIT_PATH": str(prewarm["path"]),
                    "SKYRL_QWEN35_RUNTIME_PREWARM_AUDIT_SHA256": str(prewarm["sha256"]),
                    "SKYRL_QWEN35_RUNTIME_PREWARM_HANDOFF_PATH": str(handoff["path"]),
                    "SKYRL_QWEN35_RUNTIME_PREWARM_HANDOFF_SHA256": str(
                        handoff["sha256"]
                    ),
                }
            )
        except KeyError as error:
            raise LocalServerAttestationError(
                "required startup-cache environment evidence is incomplete"
            ) from error
    observed = dict(process.qualified_environment)
    if observed != expected:
        raise LocalServerAttestationError(
            f"{role} process environment is not launcher-exact"
        )


def _module_index(argv: tuple[str, ...], module: str, label: str) -> int:
    matches = [
        index
        for index in range(len(argv) - 1)
        if argv[index] == "-m" and argv[index + 1] == module
    ]
    if len(matches) != 1:
        raise LocalServerAttestationError(
            f"{label} must contain exactly one '-m {module}'"
        )
    return matches[0]


def _validate_outer_uv(argv: tuple[str, ...]) -> tuple[int, Path]:
    if len(argv) < 12 or Path(argv[0]).name != "uv" or argv[1] != "run":
        raise LocalServerAttestationError(
            "outer server process is not canonical uv run"
        )
    module_index = _module_index(argv, API_MODULE, "outer uv command")
    observed_prefix = argv[2:module_index]
    if (
        len(observed_prefix) == 8
        and observed_prefix[:4]
        == ("--active", "--no-sync", "--no-env-file", "--no-config")
        and observed_prefix[4] == "--directory"
        and observed_prefix[6] == "--project"
        and observed_prefix[5] == observed_prefix[7]
        and Path(observed_prefix[5]).is_absolute()
    ):
        source_root = Path(observed_prefix[5])
        try:
            resolved = source_root.resolve(strict=True)
            metadata = source_root.stat(follow_symlinks=False)
        except OSError as error:
            raise LocalServerAttestationError(
                f"hardened source root cannot be inspected: {error}"
            ) from error
        if (
            resolved != source_root
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise LocalServerAttestationError(
                "hardened source root is not a canonical private directory"
            )
        return module_index, source_root
    raise LocalServerAttestationError(
        "outer uv flags do not match the hardened source-snapshot launcher"
    )


def _validate_python_command(argv: tuple[str, ...], module: str, label: str) -> int:
    module_index = _module_index(argv, module, label)
    if (
        module_index != 1
        or len(argv) < 4
        or not Path(argv[0]).name.startswith("python")
        or argv[1:3] != ("-m", module)
    ):
        raise LocalServerAttestationError(f"{label} is not exactly Python -m {module}")
    return module_index


def _parse_options(
    arguments: tuple[str, ...], expected_order: tuple[str, ...], label: str
) -> dict[str, str]:
    if len(arguments) != 2 * len(expected_order):
        raise LocalServerAttestationError(
            f"{label} does not have the exact option count"
        )
    result: dict[str, str] = {}
    for option_index, index in enumerate(range(0, len(arguments), 2)):
        name, value = arguments[index : index + 2]
        if name != expected_order[option_index]:
            raise LocalServerAttestationError(
                f"{label} option order is not launcher-exact"
            )
        if not value or value.startswith("--"):
            raise LocalServerAttestationError(
                f"{label} has an invalid value for {name}"
            )
        result[name] = value
    return result


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_backend_config(raw: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_json_pairs)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise LocalServerAttestationError(
            f"{label} backend config is invalid JSON"
        ) from error
    if not isinstance(value, dict) or tuple(value) != tuple(_BACKEND_CONFIG_TYPES):
        raise LocalServerAttestationError(
            f"{label} backend config does not have the exact launcher fields"
        )
    for name, expected_type in _BACKEND_CONFIG_TYPES.items():
        if type(value[name]) is not expected_type:
            raise LocalServerAttestationError(
                f"{label} backend config field {name!r} has the wrong exact type"
            )
    for name, expected in _BACKEND_CONFIG_FIXED.items():
        if value[name] != expected:
            raise LocalServerAttestationError(
                f"{label} backend config field {name!r} must be exactly {expected!r}"
            )
    return value


def _parse_base_url(base_url: str) -> tuple[str, int]:
    if not isinstance(base_url, str):
        raise LocalServerAttestationError("base URL must be a string")
    parsed = urlsplit(base_url)
    try:
        port = parsed.port
    except ValueError as error:
        raise LocalServerAttestationError("base URL port is malformed") from error
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or port is None
        or not 1 <= port <= 65535
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or base_url != f"http://127.0.0.1:{port}"
    ):
        raise LocalServerAttestationError(
            "base URL must be exactly http://127.0.0.1:<canonical-port>"
        )
    return "127.0.0.1", port


def _read_children(proc_root: Path, parent_pid: int) -> list[int]:
    children: list[int] = []
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as error:
        raise LocalServerAttestationError(
            f"cannot enumerate procfs: {error}"
        ) from error
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            _, ppid, _, _, _ = _parse_process_stat(raw, pid)
        except (OSError, UnicodeError, LocalServerAttestationError):
            continue
        if ppid == parent_pid:
            children.append(pid)
    return sorted(children)


def _find_api_child(
    *, proc_root: Path, outer: _ProcessSnapshot, boot_id: str, expected_uid: int
) -> _ProcessSnapshot:
    candidate_pids: list[int] = []
    for pid in _read_children(proc_root, outer.pid):
        try:
            raw_cmdline = (proc_root / str(pid) / "cmdline").read_bytes()
            argv = _parse_nul_argv(raw_cmdline, f"API child PID {pid}")
            _validate_python_command(argv, API_MODULE, f"API child PID {pid}")
        except (OSError, LocalServerAttestationError):
            continue
        candidate_pids.append(pid)
    if len(candidate_pids) != 1:
        raise LocalServerAttestationError(
            "outer uv process does not have exactly one API-module child"
        )
    pid = candidate_pids[0]
    return _read_process(
        pid,
        proc_root=proc_root,
        boot_id=boot_id,
        expected_uid=expected_uid,
        label=f"API child PID {pid}",
    )


def _database_identity(path: Path, expected_uid: int) -> _DatabaseIdentity:
    try:
        metadata = path.stat(follow_symlinks=False)
        parent_metadata = path.parent.stat(follow_symlinks=False)
    except OSError as error:
        raise LocalServerAttestationError(
            f"cannot inspect server SQLite identity: {error}"
        ) from error
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != expected_uid
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != expected_uid
        or stat.S_IMODE(parent_metadata.st_mode) != 0o700
    ):
        raise LocalServerAttestationError(
            "server SQLite file or parent identity is not private and owned"
        )
    return _DatabaseIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        mode=stat.S_IMODE(metadata.st_mode),
        uid=metadata.st_uid,
        links=metadata.st_nlink,
        parent_device=parent_metadata.st_dev,
        parent_inode=parent_metadata.st_ino,
        parent_mode=stat.S_IMODE(parent_metadata.st_mode),
        parent_uid=parent_metadata.st_uid,
    )


def _parse_sqlite_path(
    database_url: str, expected_uid: int
) -> tuple[Path, _DatabaseIdentity]:
    prefix = "sqlite:///"
    if (
        not database_url.startswith(prefix)
        or "?" in database_url
        or "#" in database_url
        or "%" in database_url
    ):
        raise LocalServerAttestationError(
            "server database URL is not exact local SQLite"
        )
    path = Path(database_url[len(prefix) :])
    if not path.is_absolute():
        raise LocalServerAttestationError("server SQLite path is not absolute")
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
        parent_metadata = path.parent.stat(follow_symlinks=False)
    except OSError as error:
        raise LocalServerAttestationError(
            f"cannot inspect server SQLite file: {error}"
        ) from error
    if (
        resolved != path
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != expected_uid
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != expected_uid
        or stat.S_IMODE(parent_metadata.st_mode) != 0o700
    ):
        raise LocalServerAttestationError(
            "server SQLite file or parent is not canonical, private, and owned"
        )
    if resolved.name != "tinker.db":
        raise LocalServerAttestationError(
            "server SQLite file must be exactly the launcher tinker.db"
        )
    return resolved, _database_identity(resolved, expected_uid)


def _validate_checkpoints_directory(
    raw: str, database_path: Path, expected_uid: int
) -> Path:
    path = Path(raw)
    expected = database_path.parent / "checkpoints"
    try:
        resolved = path.resolve(strict=True)
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise LocalServerAttestationError(
            f"cannot inspect checkpoints directory: {error}"
        ) from error
    if (
        not path.is_absolute()
        or path != expected
        or resolved != path
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != expected_uid
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise LocalServerAttestationError(
            "checkpoints directory is not the canonical private database sibling"
        )
    return resolved


def _verify_model_blob(path: Path, blob_id: str, expected_uid: int) -> int:
    if not all(hasattr(os, name) for name in ("O_CLOEXEC", "O_NOFOLLOW")):
        raise LocalServerAttestationError(
            "pinned model verification requires Linux O_NOFOLLOW"
        )
    try:
        path_before = path.stat(follow_symlinks=False)
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise LocalServerAttestationError(
            f"cannot open pinned model blob {blob_id!r}: {error}"
        ) from error
    try:
        descriptor_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_before.st_mode)
            or descriptor_before.st_uid != expected_uid
            or descriptor_before.st_nlink != 1
            or descriptor_before.st_mode & 0o002
            or _file_fingerprint(path_before) != _file_fingerprint(descriptor_before)
        ):
            raise LocalServerAttestationError(
                f"pinned model blob {blob_id!r} is not an exact owned regular file"
            )
        if re.fullmatch(r"[0-9a-f]{64}", blob_id):
            digest = hashlib.sha256()
        elif re.fullmatch(r"[0-9a-f]{40}", blob_id):
            digest = hashlib.sha1(usedforsecurity=False)
            digest.update(f"blob {descriptor_before.st_size}\0".encode("ascii"))
        else:  # pragma: no cover - fixed module constant invariant
            raise LocalServerAttestationError(
                f"pinned model blob ID {blob_id!r} is malformed"
            )
        observed_size = 0
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            observed_size += len(chunk)
            digest.update(chunk)
        descriptor_after = os.fstat(descriptor)
        path_after = path.stat(follow_symlinks=False)
        if (
            observed_size != descriptor_before.st_size
            or _file_fingerprint(descriptor_after)
            != _file_fingerprint(descriptor_before)
            or _file_fingerprint(path_after) != _file_fingerprint(descriptor_after)
        ):
            raise LocalServerAttestationError(
                f"pinned model blob {blob_id!r} changed while being hashed"
            )
        if digest.hexdigest() != blob_id:
            raise LocalServerAttestationError(
                f"pinned model blob {blob_id!r} failed content verification"
            )
        return observed_size
    except LocalServerAttestationError:
        raise
    except OSError as error:
        raise LocalServerAttestationError(
            f"cannot hash pinned model blob {blob_id!r}: {error}"
        ) from error
    finally:
        os.close(descriptor)


def _validate_model_snapshot(raw: str, expected_uid: int) -> tuple[Path, str]:
    snapshot = Path(raw)
    try:
        resolved = snapshot.resolve(strict=True)
        metadata = snapshot.stat(follow_symlinks=False)
    except OSError as error:
        raise LocalServerAttestationError(
            f"cannot inspect pinned model snapshot: {error}"
        ) from error
    expected_suffix = (
        EXPECTED_MODEL_CACHE_DIRECTORY,
        "snapshots",
        EXPECTED_MODEL_REVISION,
    )
    if (
        not snapshot.is_absolute()
        or resolved != snapshot
        or tuple(snapshot.parts[-3:]) != expected_suffix
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != expected_uid
        or stat.S_IMODE(metadata.st_mode) & 0o002
    ):
        raise LocalServerAttestationError(
            "base model is not the canonical pinned Qwen3.5-4B snapshot"
        )
    blob_directory = snapshot.parents[1] / "blobs"
    try:
        blob_directory_resolved = blob_directory.resolve(strict=True)
        blob_directory_metadata = blob_directory.stat(follow_symlinks=False)
    except OSError as error:
        raise LocalServerAttestationError(
            f"cannot inspect pinned model blob directory: {error}"
        ) from error
    if (
        blob_directory_resolved != blob_directory
        or not stat.S_ISDIR(blob_directory_metadata.st_mode)
        or blob_directory_metadata.st_uid != expected_uid
        or blob_directory_metadata.st_mode & 0o002
    ):
        raise LocalServerAttestationError(
            "pinned model blob directory is not canonical, owned, and non-publicly-writable"
        )
    try:
        snapshot_nodes_before = {entry.name for entry in os.scandir(snapshot)}
    except OSError as error:
        raise LocalServerAttestationError(
            f"cannot enumerate pinned model snapshot: {error}"
        ) from error
    required_nodes = set(_EXPECTED_MODEL_BLOBS)
    allowed_nodes = required_nodes | set(_EXPECTED_OPTIONAL_INERT_MODEL_BLOBS)
    if not required_nodes.issubset(snapshot_nodes_before) or not (
        snapshot_nodes_before <= allowed_nodes
    ):
        raise LocalServerAttestationError(
            "pinned model snapshot has missing or unexpected top-level nodes"
        )
    expected_blobs = {
        **_EXPECTED_MODEL_BLOBS,
        **{
            name: blob_id
            for name, blob_id in _EXPECTED_OPTIONAL_INERT_MODEL_BLOBS.items()
            if name in snapshot_nodes_before
        },
    }
    manifest: list[dict[str, Any]] = []
    for name, blob_id in sorted(expected_blobs.items()):
        entry = snapshot / name
        expected_target = f"../../blobs/{blob_id}"
        blob_path = blob_directory / blob_id
        try:
            link_metadata_before = entry.stat(follow_symlinks=False)
            link_target = os.readlink(entry)
        except OSError as error:
            raise LocalServerAttestationError(
                f"cannot inspect pinned model entry {name!r}: {error}"
            ) from error
        if (
            link_target != expected_target
            or not stat.S_ISLNK(link_metadata_before.st_mode)
            or link_metadata_before.st_uid != expected_uid
            or link_metadata_before.st_nlink != 1
        ):
            raise LocalServerAttestationError(
                f"pinned model entry {name!r} does not match its repository blob"
            )
        size = _verify_model_blob(blob_path, blob_id, expected_uid)
        try:
            link_metadata_after = entry.stat(follow_symlinks=False)
            link_target_after = os.readlink(entry)
        except OSError as error:
            raise LocalServerAttestationError(
                f"cannot recheck pinned model entry {name!r}: {error}"
            ) from error
        if link_target_after != expected_target or _file_fingerprint(
            link_metadata_after
        ) != _file_fingerprint(link_metadata_before):
            raise LocalServerAttestationError(
                f"pinned model entry {name!r} changed while being verified"
            )
        manifest.append(
            {
                "name": name,
                "blob_id": blob_id,
                "size": size,
                "required_for_runtime": name in _EXPECTED_MODEL_BLOBS,
            }
        )
    try:
        snapshot_nodes_after = {entry.name for entry in os.scandir(snapshot)}
    except OSError as error:
        raise LocalServerAttestationError(
            f"cannot re-enumerate pinned model snapshot: {error}"
        ) from error
    if snapshot_nodes_after != snapshot_nodes_before:
        raise LocalServerAttestationError(
            "pinned model snapshot nodes changed while being verified"
        )
    return snapshot, _sha256_bytes(_canonical_json(manifest).encode("ascii"))


_ENGINE_LAUNCH_COLUMNS = (
    "launch_id",
    "backend",
    "status",
    "boot_id",
    "api_pid",
    "api_start_ticks",
    "engine_launcher_pid",
    "engine_launcher_start_ticks",
    "engine_pid",
    "engine_start_ticks",
    "api_source_attestation",
    "api_launch_lock_attestation",
    "engine_source_attestation",
    "engine_launch_lock_attestation",
    "runtime_handoff_attestation",
    "cache_evidence_status",
    "cache_evidence",
    "error_message",
    "heartbeat_at",
    "heartbeat_monotonic_ns",
    "heartbeat_sequence",
    "created_at",
    "updated_at",
    "ready_at",
)
_REQUEST_DEDUP_SCHEMA = (
    (0, "request_key", "VARCHAR", 1, None, 1),
    (1, "request_type", "VARCHAR", 1, None, 0),
    (2, "payload_sha256", "VARCHAR", 1, None, 0),
    (3, "response_data", "JSON", 0, None, 0),
    (4, "created_at", "DATETIME", 1, None, 0),
)


def _read_engine_launch(
    database_path: Path,
    *,
    api_pid: int,
    api_start_ticks: int,
    expected_uid: int,
) -> tuple[dict[str, Any], _DatabaseIdentity]:
    identity_before = _database_identity(database_path, expected_uid)
    uri = f"file:{quote(str(database_path), safe='/')}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA busy_timeout=2000")
            journal_rows = connection.execute("PRAGMA journal_mode").fetchall()
            if journal_rows != [("wal",)]:
                raise LocalServerAttestationError(
                    "server SQLite journal mode is not exactly WAL"
                )
            database_rows = connection.execute("PRAGMA database_list").fetchall()
            if (
                len(database_rows) != 1
                or database_rows[0][1] != "main"
                or Path(database_rows[0][2]).resolve(strict=True) != database_path
            ):
                raise LocalServerAttestationError(
                    "SQLite connection is not bound to the attested database path"
                )
            schema_rows = connection.execute(
                "PRAGMA table_info(engine_launches)"
            ).fetchall()
            if tuple(schema_rows) != _ENGINE_LAUNCH_SCHEMA:
                raise LocalServerAttestationError(
                    "server engine-launch schema is not fresh and exact"
                )
            trigger_rows = connection.execute(
                "SELECT name FROM sqlite_schema "
                "WHERE type = 'trigger' AND tbl_name = 'engine_launches'"
            ).fetchall()
            if trigger_rows:
                raise LocalServerAttestationError(
                    "server engine-launch table unexpectedly has triggers"
                )
            index_rows = connection.execute(
                "PRAGMA index_list(engine_launches)"
            ).fetchall()
            observed_indexes: dict[str, tuple[bool, str, bool, str]] = {}
            for index_row in index_rows:
                index_name = index_row[1]
                index_info = connection.execute(
                    f'PRAGMA index_info("{index_name}")'  # noqa: S608 - SQLite-provided identifier
                ).fetchall()
                if len(index_info) != 1:
                    raise LocalServerAttestationError(
                        "server engine-launch index shape is not exact"
                    )
                index_xinfo = connection.execute(
                    f'PRAGMA index_xinfo("{index_name}")'  # noqa: S608 - SQLite-provided identifier
                ).fetchall()
                column_name = index_info[0][2]
                column_index = _ENGINE_LAUNCH_SCHEMA_COLUMNS.index(column_name)
                if index_xinfo != [
                    (0, column_index, column_name, 0, "BINARY", 1),
                    (1, -1, None, 0, "BINARY", 0),
                ]:
                    raise LocalServerAttestationError(
                        "server engine-launch index collation/order is not exact"
                    )
                observed_indexes[index_name] = (
                    bool(index_row[2]),
                    index_row[3],
                    bool(index_row[4]),
                    column_name,
                )
            if observed_indexes != _ENGINE_LAUNCH_INDEXES:
                raise LocalServerAttestationError(
                    "server engine-launch indexes are not fresh and exact"
                )
            dedup_schema = connection.execute(
                "PRAGMA table_info(request_deduplication)"
            ).fetchall()
            if tuple(dedup_schema) != _REQUEST_DEDUP_SCHEMA:
                raise LocalServerAttestationError(
                    "server request-deduplication schema is not fresh and exact"
                )
            dedup_triggers = connection.execute(
                "SELECT name FROM sqlite_schema "
                "WHERE type = 'trigger' AND tbl_name = 'request_deduplication'"
            ).fetchall()
            if dedup_triggers:
                raise LocalServerAttestationError(
                    "server request-deduplication table unexpectedly has triggers"
                )
            columns = ", ".join(_ENGINE_LAUNCH_COLUMNS)
            connection.execute("BEGIN")
            row_count = connection.execute(
                "SELECT COUNT(*) FROM engine_launches"
            ).fetchone()
            if row_count != (1,):
                raise LocalServerAttestationError(
                    "fresh server database does not contain exactly one launch row"
                )
            rows = connection.execute(
                f"SELECT {columns} FROM engine_launches "  # noqa: S608 - fixed names
                "WHERE api_pid = ? AND api_start_ticks = ?",
                (api_pid, api_start_ticks),
            ).fetchall()
            connection.rollback()
        finally:
            connection.close()
    except LocalServerAttestationError:
        raise
    except sqlite3.Error as error:
        raise LocalServerAttestationError(
            f"cannot read the server READY identity: {error}"
        ) from error
    identity_after = _database_identity(database_path, expected_uid)
    if identity_after != identity_before:
        raise LocalServerAttestationError(
            "server SQLite identity changed while reading readiness"
        )
    if len(rows) != 1 or rows[0][2] != "READY":
        raise LocalServerAttestationError(
            "server database does not contain one exact matching READY launch"
        )
    return (
        dict(zip(_ENGINE_LAUNCH_COLUMNS, rows[0], strict=True)),
        identity_before,
    )


def _parse_json_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise LocalServerAttestationError(f"{label} is not serialized JSON")
    try:
        result = json.loads(value, object_pairs_hook=_reject_duplicate_json_pairs)
    except (ValueError, json.JSONDecodeError) as error:
        raise LocalServerAttestationError(f"{label} is invalid JSON") from error
    if not isinstance(result, dict):
        raise LocalServerAttestationError(f"{label} is not a JSON object")
    return result


_PASSED_SOURCE_SHARED_FIELDS = frozenset(
    {
        "status",
        "git_head",
        "git_tree",
        "source_archive_path",
        "source_archive_sha256",
        "source_file_count",
        "source_total_bytes",
        "full_head_tree_validated",
        "source_root",
        "repo_root",
        "working_directory",
        "package_origin",
        "uv_executable",
        "uv_sha256",
        "launch_lock",
        "jax_compilation_cache",
        "memory_mode",
        "xla_flags",
        "jax_enable_pgle",
        "jax_compilation_cache_expect_pgle",
        "pallas_attention",
        "startup_cache_attestation",
        "dont_write_bytecode",
    }
)


def _require_canonical_timestamp(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise LocalServerAttestationError(f"READY row {label} is absent")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise LocalServerAttestationError(
            f"READY row {label} is not an ISO timestamp"
        ) from error
    if parsed.year < 2000:
        raise LocalServerAttestationError(f"READY row {label} is implausible")


def _validate_lock_record(
    record: dict[str, Any], expected_uid: int, runtime_root: Path
) -> None:
    if tuple(record) != (
        "status",
        "descriptor",
        "path",
        "inheritable",
        "exclusive_lock_observed",
    ):
        raise LocalServerAttestationError("runtime launch-lock schema is not exact")
    descriptor = record["descriptor"]
    path = record["path"]
    expected_path = runtime_root / f"skyrl-qwen35-rocm-{expected_uid}"
    if (
        record["status"] != "passed"
        or type(descriptor) is not int
        or descriptor < 3
        or not isinstance(path, str)
        or not Path(path).is_absolute()
        or Path(path) != expected_path
        or record["inheritable"] is not True
        or record["exclusive_lock_observed"] is not True
    ):
        raise LocalServerAttestationError("runtime launch-lock record is invalid")
    try:
        runtime_metadata = runtime_root.stat(follow_symlinks=False)
        runtime_resolved = runtime_root.resolve(strict=True)
        resolved = Path(path).resolve(strict=True)
        metadata = Path(path).stat(follow_symlinks=False)
    except OSError as error:
        raise LocalServerAttestationError(
            f"runtime launch-lock path cannot be inspected: {error}"
        ) from error
    if (
        runtime_resolved != runtime_root
        or not stat.S_ISDIR(runtime_metadata.st_mode)
        or runtime_metadata.st_uid != expected_uid
        or stat.S_IMODE(runtime_metadata.st_mode) != 0o700
        or resolved != Path(path)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != expected_uid
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise LocalServerAttestationError(
            "runtime launch-lock path is not canonical and private"
        )


def _validate_source_handoff(
    parsed: dict[str, dict[str, Any]],
    *,
    expected_uid: int,
    expected_git_head: str,
    expected_git_tree: str,
    expected_repo_root: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    api_source = parsed["api_source_attestation"]
    engine_source = parsed["engine_source_attestation"]
    api_lock = parsed["api_launch_lock_attestation"]
    engine_lock = parsed["engine_launch_lock_attestation"]
    handoff = parsed["runtime_handoff_attestation"]
    if api_lock != engine_lock:
        raise LocalServerAttestationError(
            "API and engine runtime launch-lock records differ"
        )
    _validate_lock_record(api_lock, expected_uid, runtime_root)
    if api_source.get("role") != "api" or engine_source.get("role") != "engine":
        raise LocalServerAttestationError("runtime source roles are invalid")
    api_shared = {
        key: value
        for key, value in api_source.items()
        if key not in {"role", "module_origin"}
    }
    engine_shared = {
        key: value
        for key, value in engine_source.items()
        if key not in {"role", "module_origin"}
    }
    if api_shared != engine_shared or api_shared.get("status") != "passed":
        raise LocalServerAttestationError(
            "hardened API and engine source attestations do not agree"
        )
    if api_shared.keys() != _PASSED_SOURCE_SHARED_FIELDS:
        raise LocalServerAttestationError(
            "hardened runtime source-attestation schema is not exact"
        )
    source_root = api_shared["source_root"]
    string_fields = (
        "git_head",
        "git_tree",
        "source_archive_path",
        "source_archive_sha256",
        "source_root",
        "repo_root",
        "working_directory",
        "package_origin",
        "uv_executable",
        "uv_sha256",
        "jax_compilation_cache",
        "memory_mode",
        "xla_flags",
        "jax_enable_pgle",
        "jax_compilation_cache_expect_pgle",
        "pallas_attention",
    )
    if (
        any(not isinstance(api_shared[name], str) for name in string_fields)
        or not isinstance(source_root, str)
        or not Path(source_root).is_absolute()
        or api_source.get("module_origin") != f"{source_root}/skyrl/tinker/api.py"
        or engine_source.get("module_origin") != f"{source_root}/skyrl/tinker/engine.py"
        or api_shared["working_directory"] != source_root
        or api_shared["package_origin"] != f"{source_root}/skyrl/__init__.py"
        or api_shared["launch_lock"] != api_lock
        or api_shared["full_head_tree_validated"] is not True
        or api_shared["dont_write_bytecode"] is not True
        or api_shared["memory_mode"] not in {"growth", "preallocate85"}
        or api_shared["xla_flags"] != EXPECTED_XLA_FLAGS
        or api_shared["jax_enable_pgle"] != "false"
        or api_shared["jax_compilation_cache_expect_pgle"] != "false"
        or api_shared["pallas_attention"] not in {"0", "1"}
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", api_shared["git_head"]) is None
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", api_shared["git_tree"]) is None
        or re.fullmatch(r"[0-9a-f]{64}", api_shared["uv_sha256"]) is None
        or re.fullmatch(r"[0-9a-f]{64}", api_shared["source_archive_sha256"]) is None
        or type(api_shared["source_file_count"]) is not int
        or api_shared["source_file_count"] <= 0
        or type(api_shared["source_total_bytes"]) is not int
        or api_shared["source_total_bytes"] <= 0
        or api_shared["git_head"] != expected_git_head
        or api_shared["git_tree"] != expected_git_tree
        or api_shared["repo_root"] != str(expected_repo_root)
    ):
        raise LocalServerAttestationError(
            "hardened runtime source attestation has invalid fields"
        )
    startup_cache = api_shared["startup_cache_attestation"]
    if not isinstance(startup_cache, dict):
        raise LocalServerAttestationError(
            "runtime startup-cache attestation is not an object"
        )
    cache_requirement = startup_cache.get("status")
    if cache_requirement == "not_required":
        if startup_cache != {"status": "not_required"}:
            raise LocalServerAttestationError(
                "runtime cache opt-out has unexpected fields"
            )
    elif cache_requirement == "required-v1":
        if (
            startup_cache.get("schema_name")
            != "skyrl.qwen35.persistent-cache-attestation"
            or startup_cache.get("schema_version") != 1
            or not isinstance(startup_cache.get("seed"), dict)
            or not isinstance(startup_cache.get("prewarm_audit"), dict)
            or not isinstance(startup_cache.get("prewarm_handoff"), dict)
        ):
            raise LocalServerAttestationError(
                "required runtime cache attestation has an invalid schema"
            )
    else:
        raise LocalServerAttestationError(
            "runtime startup-cache requirement is invalid"
        )
    expected_handoff = {
        "status": "passed",
        "source_status": "passed",
        "git_head": api_shared["git_head"],
        "git_tree": api_shared["git_tree"],
        "source_root": source_root,
        "jax_compilation_cache": api_shared["jax_compilation_cache"],
        "startup_cache_attestation": startup_cache,
        "launch_lock": api_lock,
    }
    if handoff != expected_handoff:
        raise LocalServerAttestationError(
            "runtime source/lock handoff is not internally consistent"
        )
    return api_shared


def _load_stdlib_helper(source_root: Path, filename: str) -> Any:
    helper_path = source_root / "rocm" / filename
    module_name = (
        "_skyrl_attestation_"
        + filename.removesuffix(".py")
        + "_"
        + hashlib.sha256(str(helper_path).encode("utf-8")).hexdigest()[:16]
    )
    try:
        resolved = helper_path.resolve(strict=True)
        metadata_before = helper_path.stat(follow_symlinks=False)
        with helper_path.open("rb", buffering=0) as source:
            descriptor_before = os.fstat(source.fileno())
            payload = source.read()
            descriptor_after = os.fstat(source.fileno())
        metadata_after = helper_path.stat(follow_symlinks=False)
    except OSError as error:
        raise LocalServerAttestationError(
            f"cannot read trusted source helper {filename!r}: {error}"
        ) from error
    if (
        resolved != helper_path
        or not stat.S_ISREG(metadata_before.st_mode)
        or metadata_before.st_nlink != 1
        or _file_fingerprint(metadata_before)
        != _file_fingerprint(descriptor_before)
        or _file_fingerprint(descriptor_after)
        != _file_fingerprint(descriptor_before)
        or _file_fingerprint(metadata_after)
        != _file_fingerprint(descriptor_after)
    ):
        raise LocalServerAttestationError(
            f"trusted source helper {filename!r} is not one stable regular file"
        )
    module = types.ModuleType(module_name)
    module.__file__ = str(helper_path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        code = compile(
            payload,
            str(helper_path),
            "exec",
            flags=0,
            dont_inherit=True,
            optimize=0,
        )
        exec(code, module.__dict__)
    except Exception as error:
        raise LocalServerAttestationError(
            f"trusted source helper {filename!r} failed to load: {type(error).__name__}"
        ) from error
    finally:
        sys.modules.pop(module_name, None)
    return module


def _default_source_cache_validator(
    repo_root: Path, git_head: str, account_home: Path
) -> Mapping[str, Any]:
    module = _load_stdlib_helper(repo_root, "verified_source_bootstrap.py")
    try:
        result = module.validate_source_cache(
            repo_root=repo_root,
            git_head=git_head,
            account_home=account_home,
        )
    except Exception as error:
        raise LocalServerAttestationError(
            f"full source-cache revalidation failed: {type(error).__name__}"
        ) from error
    if not isinstance(result, Mapping):
        raise LocalServerAttestationError(
            "full source-cache validator returned a non-mapping"
        )
    return result


def _default_cache_evidence_validator(
    repo_root: Path,
    source_root: Path,
    claim: Mapping[str, Any],
    evidence: Mapping[str, Any],
    boot_id_path: Path,
) -> None:
    del repo_root
    module = _load_stdlib_helper(source_root, "qwen35_cache_attestation.py")
    try:
        rebuilt = module.revalidate_startup_cache_claim(
            claim,
            boot_id_path=boot_id_path,
        )
        module.validate_runtime_cache_evidence(rebuilt, evidence)
    except Exception as error:
        raise LocalServerAttestationError(
            "required startup/runtime cache evidence failed full revalidation: "
            f"{type(error).__name__}"
        ) from error


def _validate_source_cache_evidence(
    source: Mapping[str, Any],
    *,
    expected_uid: int,
    expected_git_head: str,
    expected_git_tree: str,
    expected_repo_root: Path,
    validator: SourceCacheValidator,
) -> None:
    source_root = Path(source["source_root"])
    commit_root = source_root.parent
    cache_root = commit_root.parent
    account_cache = cache_root.parent
    account_home = account_cache.parent
    archive_path = commit_root / "source-head.tar"
    if (
        source_root.name != "source-head"
        or commit_root.name != expected_git_head
        or cache_root.name != "skyrl-source-snapshots-private-v1"
        or account_cache.name != ".cache"
        or source["source_archive_path"] != str(archive_path)
        or source["repo_root"] != str(expected_repo_root)
    ):
        raise LocalServerAttestationError(
            "runtime source cache does not use the exact commit-keyed layout"
        )
    for directory, label in (
        (account_cache, "account cache"),
        (cache_root, "source cache"),
        (commit_root, "source commit cache"),
        (source_root, "source snapshot"),
    ):
        try:
            metadata = directory.stat(follow_symlinks=False)
            resolved = directory.resolve(strict=True)
        except OSError as error:
            raise LocalServerAttestationError(
                f"cannot inspect {label}: {error}"
            ) from error
        if (
            resolved != directory
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != expected_uid
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise LocalServerAttestationError(
                f"{label} is not canonical, private, and owned"
            )
    for relative_path in (
        "skyrl/__init__.py",
        "skyrl/tinker/api.py",
        "skyrl/tinker/engine.py",
    ):
        path = source_root / relative_path
        try:
            metadata = path.stat(follow_symlinks=False)
        except OSError as error:
            raise LocalServerAttestationError(
                f"cannot inspect runtime source {relative_path!r}: {error}"
            ) from error
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != expected_uid
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise LocalServerAttestationError(
                f"runtime source {relative_path!r} is not an exact private file"
            )
    try:
        result = validator(expected_repo_root, expected_git_head, account_home)
    except LocalServerAttestationError:
        raise
    except Exception as error:
        raise LocalServerAttestationError(
            f"full source-cache validator failed: {type(error).__name__}"
        ) from error
    expected_result = {
        "cache_status": "validated",
        "format": "skyrl-private-source-cache-v1",
        "git_head": expected_git_head,
        "git_tree": expected_git_tree,
        "source_archive_path": str(archive_path),
        "source_archive_sha256": source["source_archive_sha256"],
        "source_file_count": source["source_file_count"],
        "source_snapshot_root": str(source_root),
        "source_total_bytes": source["source_total_bytes"],
        "full_head_tree_validated": True,
    }
    if dict(result) != expected_result:
        raise LocalServerAttestationError(
            "full source-cache result does not match the runtime source claim"
        )


def _validate_launch_row(
    row: dict[str, Any],
    *,
    api: _ProcessSnapshot,
    boot_id: str,
    expected_uid: int,
    expected_git_head: str,
    expected_git_tree: str,
    expected_repo_root: Path,
    runtime_root: Path,
    require_startup_cache: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    launch_id = row["launch_id"]
    if (
        not isinstance(launch_id, str)
        or _LAUNCH_ID_PATTERN.fullmatch(launch_id) is None
    ):
        raise LocalServerAttestationError("engine launch ID is malformed")
    if row["backend"] != "jax" or row["boot_id"] != boot_id:
        raise LocalServerAttestationError("READY row backend or boot identity is wrong")
    if (row["api_pid"], row["api_start_ticks"]) != (api.pid, api.start_ticks):
        raise LocalServerAttestationError("READY row API identity changed")
    for name in (
        "engine_launcher_pid",
        "engine_launcher_start_ticks",
        "engine_pid",
        "engine_start_ticks",
        "heartbeat_monotonic_ns",
        "heartbeat_sequence",
    ):
        value = row[name]
        if type(value) is not int or value <= 0:
            raise LocalServerAttestationError(f"READY row field {name!r} is absent")
    for name in ("heartbeat_at", "created_at", "updated_at", "ready_at"):
        _require_canonical_timestamp(row[name], name)
    heartbeat_age_seconds = (
        time.monotonic_ns() - row["heartbeat_monotonic_ns"]
    ) / 1_000_000_000
    if heartbeat_age_seconds < 0 or heartbeat_age_seconds > _MAX_HEARTBEAT_AGE_SECONDS:
        raise LocalServerAttestationError("READY row heartbeat is stale")
    if row["error_message"] is not None:
        raise LocalServerAttestationError("READY row unexpectedly has an error")
    parsed = {
        name: _parse_json_object(row[name], name)
        for name in (
            "api_source_attestation",
            "api_launch_lock_attestation",
            "engine_source_attestation",
            "engine_launch_lock_attestation",
            "runtime_handoff_attestation",
            "cache_evidence",
        )
    }
    source_shared = _validate_source_handoff(
        parsed,
        expected_uid=expected_uid,
        expected_git_head=expected_git_head,
        expected_git_tree=expected_git_tree,
        expected_repo_root=expected_repo_root,
        runtime_root=runtime_root,
    )
    cache_requirement = source_shared["startup_cache_attestation"]["status"]
    if require_startup_cache and cache_requirement != "required-v1":
        raise LocalServerAttestationError(
            "promotion gate requires full startup-cache attestation"
        )
    cache_evidence = parsed["cache_evidence"]
    if cache_requirement == "not_required":
        if row["cache_evidence_status"] != "not_required" or cache_evidence != {}:
            raise LocalServerAttestationError(
                "READY row cache opt-out evidence is not exact"
            )
    elif (
        row["cache_evidence_status"] != RUNTIME_CACHE_HIT_KIND
        or cache_evidence.get("kind") != RUNTIME_CACHE_HIT_KIND
    ):
        raise LocalServerAttestationError(
            "READY row required cache-hit evidence is absent"
        )
    return parsed, source_shared


def _validate_required_cache_evidence(
    parsed: Mapping[str, Mapping[str, Any]],
    source: Mapping[str, Any],
    *,
    expected_repo_root: Path,
    backend_config: Mapping[str, Any],
    model_path: Path,
    pallas_attention: str,
    cache_directory: str,
    validator: CacheEvidenceValidator,
    boot_id_path: Path,
) -> None:
    claim = source["startup_cache_attestation"]
    if claim["status"] == "not_required":
        return
    seed = claim.get("seed")
    seed_backend = seed.get("backend_config") if isinstance(seed, dict) else None
    expected_attention = "pallas" if pallas_attention == "1" else "xla"
    expected_construction = (
        "abstract-load" if backend_config["abstract_model_load"] else "eager"
    )
    if (
        not isinstance(seed, dict)
        or not isinstance(seed_backend, dict)
        or seed.get("source_git_head") != source["git_head"]
        or seed.get("source_git_tree") != source["git_tree"]
        or seed.get("cache_path") != cache_directory
        or seed.get("attention_backend") != expected_attention
        or seed.get("construction") != expected_construction
        or seed.get("model") != "Qwen/Qwen3.5-4B"
        or seed.get("model_revision") != EXPECTED_MODEL_REVISION
        or seed.get("model_path") != str(model_path)
        or seed.get("compile_target")
        != "train_bucket_forward_backward_accumulate"
        or seed.get("bucket") != 64
        or seed.get("batch_size") != 1
        or seed.get("xla_flags") != EXPECTED_XLA_FLAGS
        or seed.get("graph_api_used") is not False
        or seed.get("command_buffer_used") is not False
        or re.fullmatch(
            r"[0-9a-f]{64}", str(seed.get("backend_config_sha256", ""))
        )
        is None
        or any(seed_backend.get(name) != value for name, value in backend_config.items())
    ):
        raise LocalServerAttestationError(
            "required cache seed is not cross-bound to the observed server contract"
        )
    evidence = parsed["cache_evidence"]
    try:
        validator(
            expected_repo_root,
            Path(source["source_root"]),
            claim,
            evidence,
            boot_id_path,
        )
    except LocalServerAttestationError:
        raise
    except Exception as error:
        raise LocalServerAttestationError(
            f"required cache-evidence validator failed: {type(error).__name__}"
        ) from error


def _parse_tcp_listener(
    tcp_path: Path, *, host: str, port: int, expected_uid: int
) -> int:
    if host != "127.0.0.1":  # pragma: no cover - guarded by URL parser
        raise LocalServerAttestationError("listener host is not IPv4 loopback")
    expected_address = f"0100007F:{port:04X}"
    try:
        lines = tcp_path.read_text(encoding="ascii").splitlines()[1:]
    except (OSError, UnicodeError) as error:
        raise LocalServerAttestationError(
            f"cannot read API TCP listeners: {error}"
        ) from error
    matches: list[int] = []
    for line in lines:
        fields = line.split()
        if len(fields) < 10:
            continue
        if (
            fields[1].upper() == expected_address
            and fields[2] == "00000000:0000"
            and fields[3].upper() == "0A"
        ):
            try:
                uid = int(fields[7])
                inode = int(fields[9])
            except ValueError as error:
                raise LocalServerAttestationError(
                    "API TCP listener metadata is malformed"
                ) from error
            if uid != expected_uid or inode <= 0:
                raise LocalServerAttestationError(
                    "API TCP listener owner or inode is invalid"
                )
            matches.append(inode)
    if len(matches) != 1:
        raise LocalServerAttestationError(
            "API process namespace does not have exactly one matching listener"
        )
    return matches[0]


def _require_api_socket(api_root: Path, inode: int) -> None:
    expected = f"socket:[{inode}]"
    try:
        entries = tuple((api_root / "fd").iterdir())
    except OSError as error:
        raise LocalServerAttestationError(
            f"cannot inspect API descriptors: {error}"
        ) from error
    links: list[str] = []
    for entry in entries:
        try:
            links.append(os.readlink(entry))
        except FileNotFoundError:
            continue
        except OSError as error:
            raise LocalServerAttestationError(
                f"cannot inspect API descriptor {entry.name!r}: {error}"
            ) from error
    if links.count(expected) != 1:
        raise LocalServerAttestationError(
            "loopback listener is not held by exactly one API descriptor"
        )


def _require_exact_process_children(
    proc_root: Path,
    *,
    outer: _ProcessSnapshot,
    api: _ProcessSnapshot,
    engine_launcher: _ProcessSnapshot,
    engine: _ProcessSnapshot,
) -> None:
    if _read_children(proc_root, outer.pid) != [api.pid]:
        raise LocalServerAttestationError(
            "outer uv process does not have the exact API child set"
        )
    if _read_children(proc_root, api.pid) != [engine_launcher.pid]:
        raise LocalServerAttestationError(
            "API process does not have the exact engine-launcher child set"
        )
    if engine_launcher.pid != engine.pid and _read_children(
        proc_root, engine_launcher.pid
    ) != [engine.pid]:
        raise LocalServerAttestationError(
            "engine uv wrapper does not have the exact engine child set"
        )
    if _read_children(proc_root, engine.pid):
        raise LocalServerAttestationError(
            "engine process is not the expected leaf process"
        )


def _require_live_launch_lock(
    processes: Mapping[int, _ProcessSnapshot],
    lock_record: Mapping[str, Any],
    *,
    proc_root: Path,
    expected_uid: int,
) -> None:
    import fcntl

    descriptor = lock_record["descriptor"]
    lock_path = Path(lock_record["path"])
    expected_metadata = lock_path.stat(follow_symlinks=False)
    expected_identity = (expected_metadata.st_dev, expected_metadata.st_ino)
    for process in processes.values():
        if process.launch_lock_fd != descriptor:
            raise LocalServerAttestationError(
                f"process PID {process.pid} inherited a different launch-lock descriptor"
            )
        descriptor_path = proc_root / str(process.pid) / "fd" / str(descriptor)
        try:
            resolved = descriptor_path.resolve(strict=True)
            metadata = descriptor_path.stat()
        except OSError as error:
            raise LocalServerAttestationError(
                f"cannot inspect process PID {process.pid} launch lock: {error}"
            ) from error
        if (
            resolved != lock_path
            or (metadata.st_dev, metadata.st_ino) != expected_identity
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != expected_uid
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise LocalServerAttestationError(
                f"process PID {process.pid} launch-lock descriptor is not exact"
            )
    probe = os.open(
        lock_path,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise LocalServerAttestationError(
                "runtime launch-lock directory is no longer exclusively locked"
            )
    finally:
        os.close(probe)


def _default_health_probe(host: str, port: int) -> Mapping[str, Any]:
    connection = http.client.HTTPConnection(host, port, timeout=2.0)
    try:
        connection.request(
            "GET",
            "/api/v1/healthz",
            headers={"Host": f"{host}:{port}", "Connection": "close"},
        )
        response = connection.getresponse()
        payload = response.read(4097)
        if response.status != 200:
            raise LocalServerAttestationError(
                f"server health returned HTTP {response.status}"
            )
        if len(payload) > 4096:
            raise LocalServerAttestationError("server health response is oversized")
        try:
            value = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_pairs,
            )
        except (UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise LocalServerAttestationError(
                "server health response is invalid JSON"
            ) from error
        if not isinstance(value, dict):
            raise LocalServerAttestationError("server health response is not an object")
        return value
    except LocalServerAttestationError:
        raise
    except OSError as error:
        raise LocalServerAttestationError(
            f"server health request failed: {error}"
        ) from error
    finally:
        connection.close()


def _require_health(probe: HealthProbe, host: str, port: int) -> None:
    try:
        result = probe(host, port)
    except LocalServerAttestationError:
        raise
    except Exception as error:
        raise LocalServerAttestationError(
            f"server health probe failed: {type(error).__name__}"
        ) from error
    if type(result) is not dict or result != {"status": "ok"}:
        raise LocalServerAttestationError(
            "server health response must be exactly {'status': 'ok'}"
        )


def _same_process(first: _ProcessSnapshot, second: _ProcessSnapshot) -> bool:
    return (
        first.pid,
        first.start_ticks,
        first.boot_id,
        first.ppid,
        first.process_group,
        first.session,
        first.network_namespace,
        first.argv,
        first.cmdline_sha256,
        first.executable_path,
        first.executable_name,
        first.executable_sha256,
        first.working_directory,
        first.launch_lock_fd,
        first.xla_flags_sha256,
        first.accelerator_environment,
        first.qualified_environment,
    ) == (
        second.pid,
        second.start_ticks,
        second.boot_id,
        second.ppid,
        second.process_group,
        second.session,
        second.network_namespace,
        second.argv,
        second.cmdline_sha256,
        second.executable_path,
        second.executable_name,
        second.executable_sha256,
        second.working_directory,
        second.launch_lock_fd,
        second.xla_flags_sha256,
        second.accelerator_environment,
        second.qualified_environment,
    )


def _reread_process(
    original: _ProcessSnapshot,
    *,
    proc_root: Path,
    boot_id: str,
    expected_uid: int,
    label: str,
) -> None:
    current = _read_process(
        original.pid,
        proc_root=proc_root,
        boot_id=boot_id,
        expected_uid=expected_uid,
        label=label,
    )
    if not _same_process(original, current):
        raise LocalServerAttestationError(f"{label} changed during attestation")


def _collect_contract(
    *,
    server_pid: int,
    base_url: str,
    proc_root: Path,
    health_probe: HealthProbe,
    expected_uid: int,
    expected_git_head: str,
    expected_git_tree: str,
    expected_repo_root: Path,
    expected_python_sha256: str,
    runtime_root: Path,
    require_startup_cache: bool,
    source_cache_validator: SourceCacheValidator,
    cache_evidence_validator: CacheEvidenceValidator,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    host, port = _parse_base_url(base_url)
    boot_id = _read_boot_id(proc_root)
    outer = _read_process(
        server_pid,
        proc_root=proc_root,
        boot_id=boot_id,
        expected_uid=expected_uid,
        label="outer uv process",
    )
    outer_module_index, source_root = _validate_outer_uv(outer.argv)
    try:
        outer_argv_executable = Path(outer.argv[0]).resolve(strict=True)
    except OSError as error:
        raise LocalServerAttestationError(
            f"outer uv argv executable cannot be resolved: {error}"
        ) from error
    if (
        outer.executable_name != "uv"
        or outer_argv_executable != Path(outer.executable_path)
        or outer.executable_sha256 != EXPECTED_UV_SHA256
    ):
        raise LocalServerAttestationError(
            "outer server executable is not the qualified hardened uv payload"
        )
    outer_api_arguments = outer.argv[outer_module_index + 2 :]
    outer_options = _parse_options(
        outer_api_arguments, _API_OPTION_ORDER, "outer API command"
    )
    api = _find_api_child(
        proc_root=proc_root,
        outer=outer,
        boot_id=boot_id,
        expected_uid=expected_uid,
    )
    api_module_index = _validate_python_command(api.argv, API_MODULE, "API command")
    if not api.executable_name.startswith("python"):
        raise LocalServerAttestationError("API executable is not Python")
    _require_python_payload(
        api,
        expected_repo_root=expected_repo_root,
        expected_python_sha256=expected_python_sha256,
        label="API process",
    )
    api_arguments = api.argv[api_module_index + 2 :]
    api_options = _parse_options(api_arguments, _API_OPTION_ORDER, "API command")
    if outer_options != api_options or outer_api_arguments != api_arguments:
        raise LocalServerAttestationError(
            "outer uv and API child arguments are not byte-for-byte equal"
        )
    if (
        api.ppid != outer.pid
        or api.start_ticks < outer.start_ticks
        or api.process_group != outer.process_group
        or api.session != outer.session
    ):
        raise LocalServerAttestationError("API child process ancestry is invalid")
    if outer_options["--backend"] != "jax":
        raise LocalServerAttestationError("API backend is not exactly jax")
    if outer_options["--host"] != host or outer_options["--port"] != str(port):
        raise LocalServerAttestationError("API host or port does not match base URL")
    if outer_options["--engine-startup-timeout-sec"] != "3600":
        raise LocalServerAttestationError("API startup timeout is not launcher-exact")
    backend_config = _parse_backend_config(outer_options["--backend-config"], "API")
    if outer_options["--backend-config"] != json.dumps(
        backend_config, separators=(",", ":")
    ):
        raise LocalServerAttestationError(
            "API backend config is not launcher-canonical compact JSON"
        )
    model_path, model_manifest_sha256 = _validate_model_snapshot(
        outer_options["--base-model"], expected_uid
    )
    database_path, database_identity = _parse_sqlite_path(
        outer_options["--database-url"], expected_uid
    )
    checkpoints_path = _validate_checkpoints_directory(
        outer_options["--checkpoints-base"], database_path, expected_uid
    )

    self_namespace = (proc_root / "self/ns/net").stat()
    expected_namespace = (self_namespace.st_dev, self_namespace.st_ino)
    if outer.network_namespace != expected_namespace or api.network_namespace != (
        expected_namespace
    ):
        raise LocalServerAttestationError(
            "server and attestor are not in the same network namespace"
        )
    listener_inode = _parse_tcp_listener(
        proc_root / str(api.pid) / "net/tcp",
        host=host,
        port=port,
        expected_uid=expected_uid,
    )
    _require_api_socket(proc_root / str(api.pid), listener_inode)
    _require_health(health_probe, host, port)

    row, observed_database_identity = _read_engine_launch(
        database_path,
        api_pid=api.pid,
        api_start_ticks=api.start_ticks,
        expected_uid=expected_uid,
    )
    if observed_database_identity != database_identity:
        raise LocalServerAttestationError(
            "server SQLite identity changed before readiness collection"
        )
    attestations, source_shared = _validate_launch_row(
        row,
        api=api,
        boot_id=boot_id,
        expected_uid=expected_uid,
        expected_git_head=expected_git_head,
        expected_git_tree=expected_git_tree,
        expected_repo_root=expected_repo_root,
        runtime_root=runtime_root,
        require_startup_cache=require_startup_cache,
    )
    _validate_source_cache_evidence(
        source_shared,
        expected_uid=expected_uid,
        expected_git_head=expected_git_head,
        expected_git_tree=expected_git_tree,
        expected_repo_root=expected_repo_root,
        validator=source_cache_validator,
    )
    engine_launcher = _read_process(
        row["engine_launcher_pid"],
        proc_root=proc_root,
        boot_id=boot_id,
        expected_uid=expected_uid,
        label="engine launcher",
    )
    engine = (
        engine_launcher
        if row["engine_pid"] == row["engine_launcher_pid"]
        else _read_process(
            row["engine_pid"],
            proc_root=proc_root,
            boot_id=boot_id,
            expected_uid=expected_uid,
            label="engine process",
        )
    )
    if engine_launcher.start_ticks != row["engine_launcher_start_ticks"] or (
        engine.start_ticks != row["engine_start_ticks"]
    ):
        raise LocalServerAttestationError(
            "engine process start identity differs from READY row"
        )
    if not engine.executable_name.startswith("python"):
        raise LocalServerAttestationError("engine executable is not Python")
    _require_python_payload(
        engine,
        expected_repo_root=expected_repo_root,
        expected_python_sha256=expected_python_sha256,
        label="engine process",
    )
    if engine.pid != engine_launcher.pid and engine_launcher.executable_name != "uv":
        raise LocalServerAttestationError("engine wrapper executable is not uv")
    if (
        engine_launcher.ppid != api.pid
        or engine_launcher.process_group != engine_launcher.pid
        or engine_launcher.session != engine_launcher.pid
        or engine_launcher.start_ticks < api.start_ticks
    ):
        raise LocalServerAttestationError(
            "engine launcher ancestry or session is invalid"
        )
    if engine.pid != engine_launcher.pid and (
        engine.ppid != engine_launcher.pid
        or engine.start_ticks < engine_launcher.start_ticks
    ):
        raise LocalServerAttestationError("engine wrapper-child ancestry is invalid")
    if engine.process_group != engine_launcher.pid or engine.session != (
        engine_launcher.pid
    ):
        raise LocalServerAttestationError("engine escaped its dedicated session")
    if any(
        process.network_namespace != expected_namespace
        for process in (engine_launcher, engine)
    ):
        raise LocalServerAttestationError("engine escaped the server network namespace")
    _require_exact_process_children(
        proc_root,
        outer=outer,
        api=api,
        engine_launcher=engine_launcher,
        engine=engine,
    )

    direct_engine_exec = engine.pid == engine_launcher.pid
    if direct_engine_exec:
        launcher_engine_index = _validate_python_command(
            engine_launcher.argv, ENGINE_MODULE, "engine launcher command"
        )
    else:
        launcher_engine_index = _module_index(
            engine_launcher.argv, ENGINE_MODULE, "engine launcher command"
        )
        expected_wrapper_prefix = (
            outer.argv[0],
            "run",
            *outer.argv[2:outer_module_index],
            "--extra",
            "tinker",
            "--extra",
            "jax",
            "-m",
            ENGINE_MODULE,
        )
        if engine_launcher.argv[: launcher_engine_index + 2] != expected_wrapper_prefix:
            raise LocalServerAttestationError(
                "engine uv wrapper prefix is not inherited and launcher-exact"
            )
        if (
            engine_launcher.argv[0] != outer.argv[0]
            or engine_launcher.executable_path != outer.executable_path
            or engine_launcher.executable_sha256 != outer.executable_sha256
        ):
            raise LocalServerAttestationError(
                "engine uv wrapper does not use the qualified outer uv executable"
            )
    launcher_engine_options = _parse_options(
        engine_launcher.argv[launcher_engine_index + 2 :],
        _ENGINE_OPTION_ORDER,
        "engine launcher command",
    )
    engine_index = _validate_python_command(
        engine.argv, ENGINE_MODULE, "engine command"
    )
    engine_options = _parse_options(
        engine.argv[engine_index + 2 :], _ENGINE_OPTION_ORDER, "engine command"
    )
    if launcher_engine_options != engine_options or (
        engine_launcher.argv[launcher_engine_index + 2 :]
        != engine.argv[engine_index + 2 :]
    ):
        raise LocalServerAttestationError(
            "engine launcher and engine arguments are not byte-for-byte equal"
        )
    if (
        api.executable_path != engine.executable_path
        or api.executable_sha256 != engine.executable_sha256
    ):
        raise LocalServerAttestationError(
            "API and engine do not use the same Python interpreter payload"
        )
    if (
        engine_options["--backend"] != "jax"
        or engine_options["--base-model"] != outer_options["--base-model"]
        or engine_options["--database-url"] != outer_options["--database-url"]
        or engine_options["--checkpoints-base"] != outer_options["--checkpoints-base"]
        or engine_options["--engine-startup-timeout-sec"] != "3600"
        or engine_options["--startup-launch-id"] != row["launch_id"]
        or engine_options["--external-inference-api-key"] != "EMPTY"
        or engine_options["--external-inference-lora-base"] != "/tmp/lora_models"
        or engine_options["--session-cleanup-interval-sec"] != "60"
        or engine_options["--session-timeout-sec"] != "300"
    ):
        raise LocalServerAttestationError(
            "engine configuration differs from the API launch"
        )
    engine_backend_config = _parse_backend_config(
        engine_options["--backend-config"], "engine"
    )
    if engine_backend_config != backend_config or engine_options[
        "--backend-config"
    ] != json.dumps(engine_backend_config):
        raise LocalServerAttestationError("API and engine backend configs differ")

    unique_processes = {
        process.pid: process for process in (outer, api, engine_launcher, engine)
    }
    environment_contracts: set[tuple[str, str]] = set()
    for process in unique_processes.values():
        pallas_attention, cache_directory = _require_runtime_environment(
            process,
            abstract_model_load=backend_config["abstract_model_load"],
            label=f"process PID {process.pid}",
        )
        environment_contracts.add((pallas_attention, cache_directory))
        if process.working_directory != str(source_root):
            raise LocalServerAttestationError(
                f"process PID {process.pid} is not running from the source snapshot"
            )
    if len(environment_contracts) != 1:
        raise LocalServerAttestationError(
            "server roles do not share one Pallas/cache environment"
        )
    pallas_attention, cache_directory = next(iter(environment_contracts))
    expected_memory_mode = (
        "preallocate85" if backend_config["abstract_model_load"] else "growth"
    )
    if (
        source_shared["source_root"] != str(source_root)
        or source_shared["uv_executable"] != outer.argv[0]
        or source_shared["uv_sha256"] != outer.executable_sha256
        or source_shared["memory_mode"] != expected_memory_mode
        or source_shared["pallas_attention"] != pallas_attention
        or source_shared["jax_compilation_cache"] != cache_directory
    ):
        raise LocalServerAttestationError(
            "hardened source attestation does not match observed argv/environment"
        )
    role_processes = [
        (outer, "outer"),
        (api, "api"),
        *([] if direct_engine_exec else [(engine_launcher, "engine_launcher")]),
        (engine, "engine"),
    ]
    for process, role in role_processes:
        _require_qualified_process_environment(
            process,
            role=role,
            source=source_shared,
            runtime_root=runtime_root,
        )
    _validate_required_cache_evidence(
        attestations,
        source_shared,
        expected_repo_root=expected_repo_root,
        backend_config=backend_config,
        model_path=model_path,
        pallas_attention=pallas_attention,
        cache_directory=cache_directory,
        validator=cache_evidence_validator,
        boot_id_path=proc_root / "sys/kernel/random/boot_id",
    )
    _require_live_launch_lock(
        unique_processes,
        attestations["api_launch_lock_attestation"],
        proc_root=proc_root,
        expected_uid=expected_uid,
    )
    for process in unique_processes.values():
        _reread_process(
            process,
            proc_root=proc_root,
            boot_id=boot_id,
            expected_uid=expected_uid,
            label=f"process PID {process.pid}",
        )
    _require_exact_process_children(
        proc_root,
        outer=outer,
        api=api,
        engine_launcher=engine_launcher,
        engine=engine,
    )
    _require_api_socket(proc_root / str(api.pid), listener_inode)
    final_listener_inode = _parse_tcp_listener(
        proc_root / str(api.pid) / "net/tcp",
        host=host,
        port=port,
        expected_uid=expected_uid,
    )
    if final_listener_inode != listener_inode:
        raise LocalServerAttestationError("API listener changed during attestation")
    final_row, final_database_identity = _read_engine_launch(
        database_path,
        api_pid=api.pid,
        api_start_ticks=api.start_ticks,
        expected_uid=expected_uid,
    )
    final_attestations, final_source_shared = _validate_launch_row(
        final_row,
        api=api,
        boot_id=boot_id,
        expected_uid=expected_uid,
        expected_git_head=expected_git_head,
        expected_git_tree=expected_git_tree,
        expected_repo_root=expected_repo_root,
        runtime_root=runtime_root,
        require_startup_cache=require_startup_cache,
    )
    if final_database_identity != database_identity:
        raise LocalServerAttestationError(
            "server SQLite identity changed during attestation"
        )
    stable_row = {
        name: value for name, value in row.items() if name not in _DYNAMIC_LAUNCH_FIELDS
    }
    final_stable_row = {
        name: value
        for name, value in final_row.items()
        if name not in _DYNAMIC_LAUNCH_FIELDS
    }
    if final_stable_row != stable_row:
        raise LocalServerAttestationError(
            "stable READY row fields changed during attestation"
        )
    _validate_source_cache_evidence(
        final_source_shared,
        expected_uid=expected_uid,
        expected_git_head=expected_git_head,
        expected_git_tree=expected_git_tree,
        expected_repo_root=expected_repo_root,
        validator=source_cache_validator,
    )
    _validate_required_cache_evidence(
        final_attestations,
        final_source_shared,
        expected_repo_root=expected_repo_root,
        backend_config=backend_config,
        model_path=model_path,
        pallas_attention=pallas_attention,
        cache_directory=cache_directory,
        validator=cache_evidence_validator,
        boot_id_path=proc_root / "sys/kernel/random/boot_id",
    )
    final_model_path, final_model_manifest_sha256 = _validate_model_snapshot(
        outer_options["--base-model"], expected_uid
    )
    if (
        final_model_path != model_path
        or final_model_manifest_sha256 != model_manifest_sha256
    ):
        raise LocalServerAttestationError(
            "pinned model snapshot changed during attestation"
        )
    last_row, last_database_identity = _read_engine_launch(
        database_path,
        api_pid=api.pid,
        api_start_ticks=api.start_ticks,
        expected_uid=expected_uid,
    )
    _validate_launch_row(
        last_row,
        api=api,
        boot_id=boot_id,
        expected_uid=expected_uid,
        expected_git_head=expected_git_head,
        expected_git_tree=expected_git_tree,
        expected_repo_root=expected_repo_root,
        runtime_root=runtime_root,
        require_startup_cache=require_startup_cache,
    )
    last_stable_row = {
        name: value
        for name, value in last_row.items()
        if name not in _DYNAMIC_LAUNCH_FIELDS
    }
    if (
        last_database_identity != database_identity
        or last_stable_row != stable_row
    ):
        raise LocalServerAttestationError(
            "server READY identity or stable fields changed during final checks"
        )
    _require_live_launch_lock(
        unique_processes,
        attestations["api_launch_lock_attestation"],
        proc_root=proc_root,
        expected_uid=expected_uid,
    )
    for process in unique_processes.values():
        _reread_process(
            process,
            proc_root=proc_root,
            boot_id=boot_id,
            expected_uid=expected_uid,
            label=f"process PID {process.pid}",
        )
    _require_exact_process_children(
        proc_root,
        outer=outer,
        api=api,
        engine_launcher=engine_launcher,
        engine=engine,
    )
    _require_api_socket(proc_root / str(api.pid), listener_inode)
    last_listener_inode = _parse_tcp_listener(
        proc_root / str(api.pid) / "net/tcp",
        host=host,
        port=port,
        expected_uid=expected_uid,
    )
    if last_listener_inode != listener_inode:
        raise LocalServerAttestationError(
            "API listener changed during final attestation checks"
        )
    # This is deliberately the final external observation in the collection.
    _require_health(health_probe, host, port)

    process_evidence = {
        "uv_launcher": outer.public_evidence(),
        "api": api.public_evidence(),
        "engine_launcher": engine_launcher.public_evidence(),
        "engine": engine.public_evidence(),
    }
    backend_config_json = _canonical_json(backend_config)
    handoff_json = _canonical_json(attestations["runtime_handoff_attestation"])
    source_status = attestations["api_source_attestation"]["status"]
    source_json = _canonical_json(
        {
            "api": attestations["api_source_attestation"],
            "engine": attestations["engine_source_attestation"],
        }
    )
    lock_json = _canonical_json(attestations["api_launch_lock_attestation"])
    cache_evidence_json = _canonical_json(attestations["cache_evidence"])
    stable_contract = {
        "server_pid": server_pid,
        "base_url": base_url,
        "boot_id_sha256": _sha256_bytes(boot_id.encode("ascii")),
        "network_namespace": list(expected_namespace),
        "listener_inode_sha256": _sha256_bytes(str(listener_inode).encode("ascii")),
        "database_path_sha256": _sha256_bytes(str(database_path).encode("utf-8")),
        "database_identity": {
            "device": database_identity.device,
            "inode": database_identity.inode,
            "parent_device": database_identity.parent_device,
            "parent_inode": database_identity.parent_inode,
        },
        "base_model_sha256": _sha256_bytes(str(model_path).encode("utf-8")),
        "base_model_revision": EXPECTED_MODEL_REVISION,
        "base_model_manifest_sha256": model_manifest_sha256,
        "source_git_head": expected_git_head,
        "source_git_tree": expected_git_tree,
        "source_archive_sha256": source_shared["source_archive_sha256"],
        "expected_repo_root_sha256": _sha256_bytes(
            str(expected_repo_root).encode("utf-8")
        ),
        "expected_python_sha256": expected_python_sha256,
        "checkpoints_base_sha256": _sha256_bytes(str(checkpoints_path).encode("utf-8")),
        "launch_id_sha256": _sha256_bytes(row["launch_id"].encode("ascii")),
        "processes": process_evidence,
        "backend_config_sha256": _sha256_bytes(backend_config_json.encode("ascii")),
        "abstract_model_load": backend_config["abstract_model_load"],
        "runtime_handoff_sha256": _sha256_bytes(handoff_json.encode("ascii")),
        "runtime_source_sha256": _sha256_bytes(source_json.encode("ascii")),
        "runtime_launch_lock_sha256": _sha256_bytes(lock_json.encode("ascii")),
        "cache_evidence_status": row["cache_evidence_status"],
        "cache_evidence_sha256": _sha256_bytes(cache_evidence_json.encode("ascii")),
        "source_status": source_status,
        "pallas_attention": pallas_attention,
        "xla_flags": EXPECTED_XLA_FLAGS,
    }
    public = {
        "record_type": "server_attestation",
        "schema_version": 3,
        "status": "passed",
        "scope": "hardened_local_procfs_point_in_time_v3",
        "attested_at_unix_ns": time.time_ns(),
        "endpoint": {
            "host": host,
            "port": port,
            "listener_inode_sha256": stable_contract["listener_inode_sha256"],
        },
        "processes": process_evidence,
        "backend": {
            "name": "jax",
            "backend_config_sha256": stable_contract["backend_config_sha256"],
            "sample_max_num_sequences": 1,
            "abstract_model_load": backend_config["abstract_model_load"],
        },
        "model": {
            "repository": "Qwen/Qwen3.5-4B",
            "revision": EXPECTED_MODEL_REVISION,
            "required_blob_manifest_sha256": model_manifest_sha256,
            "all_required_blob_contents_verified": True,
        },
        "source": {
            "git_head": expected_git_head,
            "git_tree": expected_git_tree,
            "archive_sha256": source_shared["source_archive_sha256"],
            "full_cache_revalidated_initially_and_finally": True,
        },
        "environment": {
            "XLA_FLAGS": EXPECTED_XLA_FLAGS,
            "JAX_PLATFORMS": "rocm",
            "ROCR_VISIBLE_DEVICES": "0",
            "JAX_ENABLE_PGLE": "false",
            "JAX_COMPILATION_CACHE_EXPECT_PGLE": "false",
            "SKYRL_ROCM_PALLAS_ATTENTION": pallas_attention,
            "memory_mode": expected_memory_mode,
            "memory_environment": (
                _PREALLOCATE_ENVIRONMENT
                if backend_config["abstract_model_load"]
                else _GROWTH_ENVIRONMENT
            ),
            "roles_verified": [
                "uv_launcher",
                "api",
                "engine_launcher",
                "engine",
            ],
        },
        "health": {"status": "ok", "engine_launch_status": "ready"},
        "source_attestation_status": source_status,
        "runtime_source_sha256": stable_contract["runtime_source_sha256"],
        "runtime_handoff_sha256": stable_contract["runtime_handoff_sha256"],
        "runtime_launch_lock_sha256": stable_contract["runtime_launch_lock_sha256"],
        "cache": {
            "required_for_gate": require_startup_cache,
            "evidence_status": row["cache_evidence_status"],
            "evidence_sha256": stable_contract["cache_evidence_sha256"],
        },
        "storage": {
            "sqlite_journal_mode": "wal",
            "database_identity_sealed": True,
            "checkpoints_private_sibling": True,
            "ingress_sequence_deduplication": {
                "transactional_future_enqueue": True,
                "backend_effect_exactly_once_across_engine_crash": False,
                "external_sampling_dispatch_covered": False,
            },
        },
        "launch_id_sha256": stable_contract["launch_id_sha256"],
        "causal_launcher_history_proven": False,
    }
    return stable_contract, public, model_path


def attest_local_server(
    *,
    server_pid: int,
    base_url: str,
    expected_git_head: str,
    expected_git_tree: str,
    expected_repo_root: Path,
    expected_python_sha256: str | None = None,
    require_startup_cache: bool = False,
    proc_root: Path = Path("/proc"),
    health_probe: HealthProbe | None = None,
    expected_uid: int | None = None,
    runtime_root: Path | None = None,
    source_cache_validator: SourceCacheValidator | None = None,
    cache_evidence_validator: CacheEvidenceValidator | None = None,
) -> LocalServerAttestationSeal:
    """Attest one exact local server and return a revalidation seal.

    ``server_pid`` is the outer ``uv`` PID returned by ``$!`` when
    ``start_qwen35.sh`` is placed in the background.  The caller-supplied Git
    revision must be the clean benchmark checkout that is producing evidence.
    """
    selected_uid = os.getuid() if expected_uid is None else expected_uid
    if type(selected_uid) is not int or selected_uid < 0:
        raise LocalServerAttestationError("expected UID must be a nonnegative integer")
    selected_proc_root = Path(proc_root)
    if (
        not isinstance(expected_git_head, str)
        or not isinstance(expected_git_tree, str)
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", expected_git_head) is None
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", expected_git_tree) is None
        or len(expected_git_head) != len(expected_git_tree)
    ):
        raise LocalServerAttestationError(
            "expected Git HEAD/tree must be full matching-format object IDs"
        )
    selected_repo_root = Path(expected_repo_root)
    try:
        repo_metadata = selected_repo_root.stat(follow_symlinks=False)
        repo_resolved = selected_repo_root.resolve(strict=True)
    except OSError as error:
        raise LocalServerAttestationError(
            f"cannot inspect expected repository root: {error}"
        ) from error
    if (
        not selected_repo_root.is_absolute()
        or repo_resolved != selected_repo_root
        or not stat.S_ISDIR(repo_metadata.st_mode)
        or repo_metadata.st_uid != selected_uid
    ):
        raise LocalServerAttestationError(
            "expected repository root is not canonical and owned"
        )
    if expected_python_sha256 is None:
        _, selected_python_sha256 = _hash_regular_file(
            Path(sys.executable), "benchmark Python"
        )
    else:
        selected_python_sha256 = expected_python_sha256
    if (
        not isinstance(selected_python_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", selected_python_sha256) is None
    ):
        raise LocalServerAttestationError(
            "expected Python SHA-256 must be one lowercase full digest"
        )
    if type(require_startup_cache) is not bool:
        raise LocalServerAttestationError("startup-cache requirement must be boolean")
    default_runtime_root = Path("/run/user") / str(selected_uid)
    selected_runtime_root = (
        default_runtime_root if runtime_root is None else Path(runtime_root)
    )
    selected_source_validator = (
        _default_source_cache_validator
        if source_cache_validator is None
        else source_cache_validator
    )
    selected_cache_validator = (
        _default_cache_evidence_validator
        if cache_evidence_validator is None
        else cache_evidence_validator
    )
    if selected_proc_root == Path("/proc") and (
        selected_runtime_root != default_runtime_root
        or selected_source_validator is not _default_source_cache_validator
        or selected_cache_validator is not _default_cache_evidence_validator
    ):
        raise LocalServerAttestationError(
            "live procfs attestation cannot override provenance validators or runtime root"
        )
    selected_probe = _default_health_probe if health_probe is None else health_probe
    stable_contract, public, verified_model_snapshot = _collect_contract(
        server_pid=server_pid,
        base_url=base_url,
        proc_root=selected_proc_root,
        health_probe=selected_probe,
        expected_uid=selected_uid,
        expected_git_head=expected_git_head,
        expected_git_tree=expected_git_tree,
        expected_repo_root=selected_repo_root,
        expected_python_sha256=selected_python_sha256,
        runtime_root=selected_runtime_root,
        require_startup_cache=require_startup_cache,
        source_cache_validator=selected_source_validator,
        cache_evidence_validator=selected_cache_validator,
    )
    contract_json = _canonical_json(stable_contract)
    contract_sha256 = _sha256_bytes(contract_json.encode("ascii"))
    public["attestation_sha256"] = contract_sha256
    return LocalServerAttestationSeal(
        server_pid=server_pid,
        base_url=base_url,
        contract_sha256=contract_sha256,
        _contract_json=contract_json,
        _record_json=_canonical_json(public),
        _expected_git_head=expected_git_head,
        _expected_git_tree=expected_git_tree,
        _expected_repo_root=selected_repo_root,
        _expected_python_sha256=selected_python_sha256,
        _verified_model_snapshot=verified_model_snapshot,
        _runtime_root=selected_runtime_root,
        _require_startup_cache=require_startup_cache,
        _source_cache_validator=selected_source_validator,
        _cache_evidence_validator=selected_cache_validator,
    )


def revalidate_local_server(
    seal: LocalServerAttestationSeal,
    *,
    proc_root: Path = Path("/proc"),
    health_probe: HealthProbe | None = None,
    expected_uid: int | None = None,
) -> dict[str, Any]:
    """Require the same sealed process contract to remain live and healthy."""
    if not isinstance(seal, LocalServerAttestationSeal):
        raise TypeError("seal must be a LocalServerAttestationSeal")
    current = attest_local_server(
        server_pid=seal.server_pid,
        base_url=seal.base_url,
        expected_git_head=seal._expected_git_head,
        expected_git_tree=seal._expected_git_tree,
        expected_repo_root=seal._expected_repo_root,
        expected_python_sha256=seal._expected_python_sha256,
        require_startup_cache=seal._require_startup_cache,
        proc_root=proc_root,
        health_probe=health_probe,
        expected_uid=expected_uid,
        runtime_root=seal._runtime_root,
        source_cache_validator=seal._source_cache_validator,
        cache_evidence_validator=seal._cache_evidence_validator,
    )
    if current._contract_json != seal._contract_json or (
        current.contract_sha256 != seal.contract_sha256
    ):
        raise LocalServerAttestationError(
            "local server contract changed since initial attestation"
        )
    return {
        "record_type": "server_revalidation",
        "schema_version": 3,
        "status": "passed",
        "scope": "hardened_local_procfs_point_in_time_v3",
        "revalidated_at_unix_ns": time.time_ns(),
        "attestation_sha256": seal.contract_sha256,
    }
