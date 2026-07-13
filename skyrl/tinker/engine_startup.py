"""Launch-bound engine readiness and Linux process-identity helpers."""

from __future__ import annotations

import os
import re
import secrets
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

_BOOT_ID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
_LAUNCH_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
_PROCESS_STATES = {"D", "I", "K", "P", "R", "S", "T", "W", "X", "Z", "t", "x"}
_NON_OPERATIONAL_PROCESS_STATES = {"T", "X", "Z", "t", "x"}
_MAX_ENGINE_HEARTBEAT_AGE_SECONDS = 5.0
_ROLE_LOCAL_SOURCE_FIELDS = frozenset({"role", "module_origin"})


class ProcessIdentityError(RuntimeError):
    """A Linux process identity could not be read or did not validate."""


@dataclass(frozen=True)
class ProcessIdentity:
    """PID identity hardened against PID reuse and stale database rows."""

    pid: int
    start_ticks: int
    boot_id: str
    state: str

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


def new_engine_launch_id() -> str:
    """Return one unguessable canonical 128-bit launch identifier."""
    launch_id = secrets.token_hex(16)
    if _LAUNCH_ID_PATTERN.fullmatch(launch_id) is None:  # pragma: no cover
        raise RuntimeError("generated engine launch ID is not canonical")
    return launch_id


def validate_engine_launch_id(value: str) -> str:
    if _LAUNCH_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("engine launch ID must be 32 lowercase hexadecimal digits")
    return value


def parse_linux_process_stat(raw: str, expected_pid: int) -> tuple[str, int]:
    """Parse state and field-22 start ticks from one `/proc/PID/stat` record."""
    if type(expected_pid) is not int or expected_pid <= 0:
        raise ProcessIdentityError("expected PID must be a positive integer")
    record = raw.removesuffix("\n")
    open_parenthesis = record.find("(")
    close_parenthesis = record.rfind(")")
    if (
        open_parenthesis <= 0
        or close_parenthesis <= open_parenthesis
        or close_parenthesis + 2 > len(record)
        or record[close_parenthesis + 1] != " "
    ):
        raise ProcessIdentityError("Linux process stat record is malformed")
    try:
        observed_pid = int(record[:open_parenthesis].strip())
    except ValueError as error:
        raise ProcessIdentityError("Linux process stat PID is malformed") from error
    if observed_pid != expected_pid:
        raise ProcessIdentityError(
            f"Linux process stat PID mismatch: {observed_pid} != {expected_pid}"
        )
    fields = record[close_parenthesis + 2 :].split()
    # `fields[0]` is field 3 (state); index 19 is field 22 (starttime).
    if len(fields) < 20 or fields[0] not in _PROCESS_STATES:
        raise ProcessIdentityError("Linux process stat fields are truncated")
    try:
        start_ticks = int(fields[19])
    except ValueError as error:
        raise ProcessIdentityError("Linux process start ticks are malformed") from error
    if start_ticks <= 0:
        raise ProcessIdentityError("Linux process start ticks must be positive")
    return fields[0], start_ticks


def read_process_identity(
    pid: int | None = None,
    *,
    proc_root: Path = Path("/proc"),
    boot_id_path: Path | None = None,
) -> ProcessIdentity:
    """Read one process identity from procfs without importing psutil."""
    selected_pid = os.getpid() if pid is None else pid
    if type(selected_pid) is not int or selected_pid <= 0:
        raise ProcessIdentityError("PID must be a positive integer")
    selected_boot_path = (
        proc_root / "sys/kernel/random/boot_id"
        if boot_id_path is None
        else boot_id_path
    )
    try:
        boot_id = selected_boot_path.read_text(encoding="ascii").strip()
        stat_record = (proc_root / str(selected_pid) / "stat").read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeError) as error:
        raise ProcessIdentityError(
            f"cannot read Linux process identity for PID {selected_pid}: {error}"
        ) from error
    if _BOOT_ID_PATTERN.fullmatch(boot_id) is None:
        raise ProcessIdentityError("Linux boot ID is malformed")
    state, start_ticks = parse_linux_process_stat(stat_record, selected_pid)
    return ProcessIdentity(
        pid=selected_pid,
        start_ticks=start_ticks,
        boot_id=boot_id,
        state=state,
    )


def process_identity_is_live(
    expected: ProcessIdentity,
    *,
    proc_root: Path = Path("/proc"),
    boot_id_path: Path | None = None,
) -> bool:
    """Return true only for the same live process on the same boot."""
    try:
        current = read_process_identity(
            expected.pid,
            proc_root=proc_root,
            boot_id_path=boot_id_path,
        )
    except ProcessIdentityError:
        return False
    return (
        current.start_ticks == expected.start_ticks
        and current.boot_id == expected.boot_id
        and current.state not in _NON_OPERATIONAL_PROCESS_STATES
    )


def identities_from_launch_record(record: Any) -> dict[str, ProcessIdentity]:
    """Build the three required process identities from an engine-launch row."""
    required = {
        "api": (record.api_pid, record.api_start_ticks),
        "engine_launcher": (
            record.engine_launcher_pid,
            record.engine_launcher_start_ticks,
        ),
        "engine": (record.engine_pid, record.engine_start_ticks),
    }
    identities: dict[str, ProcessIdentity] = {}
    for label, (pid, start_ticks) in required.items():
        if type(pid) is not int or type(start_ticks) is not int:
            raise ProcessIdentityError(f"{label} process identity is incomplete")
        identities[label] = ProcessIdentity(
            pid=pid,
            start_ticks=start_ticks,
            boot_id=record.boot_id,
            state="?",
        )
    return identities


def _status_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _validate_engine_heartbeat(
    record: Any,
    *,
    monotonic_ns: int | None = None,
    maximum_age_seconds: float = _MAX_ENGINE_HEARTBEAT_AGE_SECONDS,
) -> None:
    heartbeat_at = getattr(record, "heartbeat_at", None)
    heartbeat_monotonic_ns = getattr(record, "heartbeat_monotonic_ns", None)
    heartbeat_sequence = getattr(record, "heartbeat_sequence", None)
    if (
        not isinstance(heartbeat_at, datetime)
        or (type(heartbeat_sequence) is not int or heartbeat_sequence < 1)
        or (type(heartbeat_monotonic_ns) is not int or heartbeat_monotonic_ns <= 0)
    ):
        raise RuntimeError("engine watchdog heartbeat is absent")
    current_monotonic_ns = time.monotonic_ns() if monotonic_ns is None else monotonic_ns
    age_seconds = (current_monotonic_ns - heartbeat_monotonic_ns) / 1_000_000_000
    if age_seconds < 0 or age_seconds > maximum_age_seconds:
        raise RuntimeError("engine watchdog heartbeat is stale")


def validate_engine_claim_record(
    record: Any,
    *,
    launch_id: str,
    backend: str,
    engine_identity: ProcessIdentity,
    parent_identity: ProcessIdentity,
    engine_source_attestation: Mapping[str, Any],
    engine_lock_attestation: Mapping[str, Any],
    identity_is_live: Callable[[ProcessIdentity], bool] = process_identity_is_live,
    process_group_reader: Callable[[int], int] = os.getpgid,
) -> dict[str, Any]:
    """Validate the API-created STARTING row before backend import/construction."""
    validate_engine_launch_id(launch_id)
    if record is None or record.launch_id != launch_id:
        raise RuntimeError("engine launch row is absent or has the wrong launch ID")
    if _status_value(record.status) != "starting":
        raise RuntimeError("engine launch row is not in STARTING state")
    if record.backend != backend:
        raise RuntimeError("engine launch backend does not match the engine config")
    if record.boot_id != engine_identity.boot_id or (
        parent_identity.boot_id != engine_identity.boot_id
    ):
        raise RuntimeError("engine launch process identities span different boots")
    if (
        type(record.engine_launcher_pid) is not int
        or type(record.engine_launcher_start_ticks) is not int
    ):
        raise RuntimeError("API has not bound the engine launcher identity")
    expected_launcher = ProcessIdentity(
        pid=record.engine_launcher_pid,
        start_ticks=record.engine_launcher_start_ticks,
        boot_id=record.boot_id,
        state="?",
    )
    direct_exec = (
        engine_identity.pid == expected_launcher.pid
        and engine_identity.start_ticks == expected_launcher.start_ticks
    )
    wrapper_child = (
        parent_identity.pid == expected_launcher.pid
        and parent_identity.start_ticks == expected_launcher.start_ticks
    )
    if not (direct_exec or wrapper_child):
        raise RuntimeError(
            "engine process is neither the bound launcher nor its direct child"
        )
    try:
        launcher_process_group = process_group_reader(expected_launcher.pid)
        engine_process_group = process_group_reader(engine_identity.pid)
    except OSError as error:
        raise RuntimeError("cannot validate the engine launch process group") from error
    if launcher_process_group != expected_launcher.pid or (
        engine_process_group != expected_launcher.pid
    ):
        raise RuntimeError("engine launch escaped its dedicated process group")
    api_identity = ProcessIdentity(
        pid=record.api_pid,
        start_ticks=record.api_start_ticks,
        boot_id=record.boot_id,
        state="?",
    )
    if direct_exec and (
        parent_identity.pid != api_identity.pid
        or parent_identity.start_ticks != api_identity.start_ticks
    ):
        raise RuntimeError("direct-exec engine parent is not the recorded API process")
    if not identity_is_live(api_identity) or not identity_is_live(expected_launcher):
        raise RuntimeError("API or engine-launcher process identity is no longer live")
    if record.engine_pid is not None or record.engine_start_ticks is not None:
        raise RuntimeError("engine launch row already contains an engine identity")
    if not record.api_source_attestation or not record.api_launch_lock_attestation:
        raise RuntimeError("API runtime attestations are absent from engine launch row")
    return validate_runtime_attestation_handoff(
        record.api_source_attestation,
        engine_source_attestation,
        record.api_launch_lock_attestation,
        engine_lock_attestation,
    )


def validate_ready_engine_launch(
    record: Any,
    *,
    launch_id: str,
    backend: str,
    api_identity: ProcessIdentity,
    launcher_identity: ProcessIdentity,
    expected_api_source_attestation: Mapping[str, Any],
    expected_api_lock_attestation: Mapping[str, Any],
    identity_is_live: Callable[[ProcessIdentity], bool] = process_identity_is_live,
    process_group_reader: Callable[[int], int] = os.getpgid,
) -> dict[str, ProcessIdentity]:
    """Validate the exact READY row and all three live process identities."""
    validate_engine_launch_id(launch_id)
    if record is None or record.launch_id != launch_id:
        raise RuntimeError("current engine launch row is absent")
    if _status_value(record.status) != "ready":
        raise RuntimeError("current engine launch is not READY")
    if record.backend != backend:
        raise RuntimeError("ready engine backend does not match API configuration")
    if record.boot_id != api_identity.boot_id or (
        launcher_identity.boot_id != api_identity.boot_id
    ):
        raise RuntimeError("ready engine launch is bound to a different boot")
    if (record.api_pid, record.api_start_ticks) != (
        api_identity.pid,
        api_identity.start_ticks,
    ):
        raise RuntimeError("ready engine launch is bound to a different API process")
    if (record.engine_launcher_pid, record.engine_launcher_start_ticks) != (
        launcher_identity.pid,
        launcher_identity.start_ticks,
    ):
        raise RuntimeError(
            "ready engine launch is bound to a different launcher process"
        )
    identities = identities_from_launch_record(record)
    if not all(identity_is_live(identity) for identity in identities.values()):
        raise RuntimeError("one or more ready engine process identities are stale")
    try:
        launcher_process_group = process_group_reader(identities["engine_launcher"].pid)
        engine_process_group = process_group_reader(identities["engine"].pid)
    except OSError as error:
        raise RuntimeError("cannot validate the ready engine process group") from error
    if launcher_process_group != launcher_identity.pid or (
        engine_process_group != launcher_identity.pid
    ):
        raise RuntimeError("ready engine escaped its dedicated process group")
    _validate_engine_heartbeat(record)
    if not isinstance(getattr(record, "ready_at", None), datetime):
        raise RuntimeError("ready engine publication timestamp is absent")
    if (
        not record.engine_source_attestation
        or not record.engine_launch_lock_attestation
    ):
        raise RuntimeError("ready engine runtime attestations are incomplete")
    if dict(record.api_source_attestation) != dict(
        expected_api_source_attestation
    ) or dict(record.api_launch_lock_attestation) != dict(
        expected_api_lock_attestation
    ):
        raise RuntimeError("ready engine API attestations changed after launch")
    recomputed_handoff = validate_runtime_attestation_handoff(
        record.api_source_attestation,
        record.engine_source_attestation,
        record.api_launch_lock_attestation,
        record.engine_launch_lock_attestation,
    )
    if dict(record.runtime_handoff_attestation) != recomputed_handoff:
        raise RuntimeError("ready engine runtime handoff attestation is inconsistent")
    if record.cache_evidence_status != "not_required" or record.cache_evidence != {}:
        raise RuntimeError("ready engine cache evidence is incomplete or rejected")
    return identities


def validate_initializing_engine_launch(
    record: Any,
    *,
    launch_id: str,
    backend: str,
    api_identity: ProcessIdentity,
    launcher_identity: ProcessIdentity,
    engine_identity: ProcessIdentity,
    engine_source_attestation: Mapping[str, Any],
    engine_lock_attestation: Mapping[str, Any],
    identity_is_live: Callable[[ProcessIdentity], bool] = process_identity_is_live,
    process_group_reader: Callable[[int], int] = os.getpgid,
) -> dict[str, Any]:
    """Revalidate the claimed launch immediately before READY publication."""
    validate_engine_launch_id(launch_id)
    if record is None or record.launch_id != launch_id:
        raise RuntimeError("claimed engine launch row is absent")
    if _status_value(record.status) != "initializing":
        raise RuntimeError("claimed engine launch is not INITIALIZING")
    if record.backend != backend:
        raise RuntimeError("claimed engine backend changed during initialization")
    if any(
        identity.boot_id != record.boot_id
        for identity in (api_identity, launcher_identity, engine_identity)
    ):
        raise RuntimeError("claimed engine process identities span different boots")
    expected_scalars = {
        "api_pid": api_identity.pid,
        "api_start_ticks": api_identity.start_ticks,
        "engine_launcher_pid": launcher_identity.pid,
        "engine_launcher_start_ticks": launcher_identity.start_ticks,
        "engine_pid": engine_identity.pid,
        "engine_start_ticks": engine_identity.start_ticks,
    }
    scalar_mismatches = {
        field: {"expected": expected, "observed": getattr(record, field)}
        for field, expected in expected_scalars.items()
        if getattr(record, field) != expected
    }
    if scalar_mismatches:
        raise RuntimeError(
            f"claimed engine process identities changed: {scalar_mismatches!r}"
        )
    if not all(
        identity_is_live(identity)
        for identity in (api_identity, launcher_identity, engine_identity)
    ):
        raise RuntimeError("one or more claimed engine process identities are stale")
    try:
        launcher_process_group = process_group_reader(launcher_identity.pid)
        engine_process_group = process_group_reader(engine_identity.pid)
    except OSError as error:
        raise RuntimeError(
            "cannot validate the claimed engine process group"
        ) from error
    if launcher_process_group != launcher_identity.pid or (
        engine_process_group != launcher_identity.pid
    ):
        raise RuntimeError("claimed engine escaped its dedicated process group")
    _validate_engine_heartbeat(record)
    if dict(record.engine_source_attestation) != dict(
        engine_source_attestation
    ) or dict(record.engine_launch_lock_attestation) != dict(engine_lock_attestation):
        raise RuntimeError("engine runtime attestations changed during initialization")
    recomputed_handoff = validate_runtime_attestation_handoff(
        record.api_source_attestation,
        engine_source_attestation,
        record.api_launch_lock_attestation,
        engine_lock_attestation,
    )
    if dict(record.runtime_handoff_attestation) != recomputed_handoff:
        raise RuntimeError("engine runtime handoff changed during initialization")
    return recomputed_handoff


def validate_runtime_attestation_handoff(
    api_source: Mapping[str, Any],
    engine_source: Mapping[str, Any],
    api_lock: Mapping[str, Any],
    engine_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Require API and engine to agree on every shared runtime invariant."""
    api_source_dict = dict(api_source)
    engine_source_dict = dict(engine_source)
    if (
        api_source_dict.get("role") != "api"
        or engine_source_dict.get("role") != "engine"
    ):
        raise RuntimeError("API/engine runtime source roles are invalid")
    api_shared = {
        key: value
        for key, value in api_source_dict.items()
        if key not in _ROLE_LOCAL_SOURCE_FIELDS
    }
    engine_shared = {
        key: value
        for key, value in engine_source_dict.items()
        if key not in _ROLE_LOCAL_SOURCE_FIELDS
    }
    if api_shared != engine_shared:
        raise RuntimeError(
            "API/engine runtime source attestation mismatch: "
            f"api={api_shared!r}, engine={engine_shared!r}"
        )
    api_lock_dict = dict(api_lock)
    engine_lock_dict = dict(engine_lock)
    if api_lock_dict != engine_lock_dict:
        raise RuntimeError("API/engine runtime launch-lock attestations do not match")
    source_status = api_shared.get("status")
    lock_status = api_lock_dict.get("status")
    if source_status == "passed":
        if lock_status != "passed":
            raise RuntimeError("validated runtime source requires a passed launch lock")
        source_root = api_shared.get("source_root")
        if not isinstance(source_root, str) or not source_root.startswith("/"):
            raise RuntimeError(
                "validated runtime source root is absent or noncanonical"
            )
        expected_origins = {
            "api": f"{source_root}/skyrl/tinker/api.py",
            "engine": f"{source_root}/skyrl/tinker/engine.py",
        }
        if api_source_dict.get("module_origin") != expected_origins["api"] or (
            engine_source_dict.get("module_origin") != expected_origins["engine"]
        ):
            raise RuntimeError("API/engine runtime module origins are invalid")
        if api_shared.get("launch_lock") != api_lock_dict:
            raise RuntimeError("runtime source and launch-lock attestations disagree")
        handoff_status = "passed"
    elif source_status == "not_required":
        if api_source_dict != {"status": "not_required", "role": "api"} or (
            engine_source_dict != {"status": "not_required", "role": "engine"}
        ):
            raise RuntimeError("unattested runtime source record has unexpected fields")
        if lock_status == "not_required":
            if api_lock_dict != {"status": "not_required"}:
                raise RuntimeError("unattested runtime launch-lock record is invalid")
            handoff_status = "matched_unattested"
        elif lock_status == "passed":
            handoff_status = "matched_lock_only"
        else:
            raise RuntimeError("unattested runtime launch-lock record is invalid")
    else:
        raise RuntimeError("runtime source attestation status is invalid")
    return {
        "status": handoff_status,
        "source_status": source_status,
        "git_head": api_shared.get("git_head"),
        "git_tree": api_shared.get("git_tree"),
        "source_root": api_shared.get("source_root"),
        "jax_compilation_cache": api_shared.get("jax_compilation_cache"),
        "launch_lock": api_lock_dict,
    }
