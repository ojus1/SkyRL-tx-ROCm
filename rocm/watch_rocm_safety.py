"""Fail-closed, high-cadence ROCm power and junction-temperature watcher.

This program intentionally does much less than :mod:`rocm.profile_rocm`.  Its
25 ms hot path reads only the two primary safety sensors, the configured power
cap, and the device runtime state.  Detailed telemetry belongs in one separate,
lower-cadence profiler process.

Safety enforcement is deliberately limited to a command launched by this
watcher.  The command is born in a private cgroup-v2 scope, so ordinary forks,
new sessions, double forks, and nested cgroups remain recursively killable.
Attaching to an external PID cannot provide the same tree-ownership proof and
is therefore rejected rather than represented as fail-closed containment.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import secrets
import signal
import stat
import sys
import time
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn

try:
    from rocm.process_supervision import (
        ChildProcessRuntime,
        ContainmentReport,
        ProcessSupervisionError,
        RuntimeReport,
        WrappedProcessSupervisor,
        _validated_pass_fds,
    )
except ModuleNotFoundError:  # Direct execution from the rocm directory.
    from process_supervision import (  # type: ignore[no-redef]
        ChildProcessRuntime,
        ContainmentReport,
        ProcessSupervisionError,
        RuntimeReport,
        WrappedProcessSupervisor,
        _validated_pass_fds,
    )


_DEFAULT_INTERVAL_SECONDS = 0.025
_DEFAULT_MAXIMUM_READ_GAP_SECONDS = 0.050
_DEFAULT_MAX_POWER_WATTS = 400.0
_DEFAULT_MAX_JUNCTION_TEMP_C = 90.0
_DEFAULT_EXPECTED_POWER_CAP_MICROWATTS = 315_000_000
_SAFETY_EXIT_CODE = 125
_TIMEOUT_EXIT_CODE = 124
_RENAME_NOREPLACE = 1
_LIBC = ctypes.CDLL(None, use_errno=True)
try:
    _LIBC_RENAMEAT2 = _LIBC.renameat2
except AttributeError:
    _LIBC_RENAMEAT2 = None
else:
    _LIBC_RENAMEAT2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    _LIBC_RENAMEAT2.restype = ctypes.c_int


def _json_dumps(value: Any, **kwargs: Any) -> str:
    return json.dumps(value, allow_nan=False, **kwargs)


def _rename_noreplace(
    source: str, destination: str, *, source_dir_fd: int, destination_dir_fd: int
) -> None:
    """Linux renameat2(RENAME_NOREPLACE), relative to pinned directories."""
    if _LIBC_RENAMEAT2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable")
    result = _LIBC_RENAMEAT2(
        source_dir_fd,
        os.fsencode(source),
        destination_dir_fd,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), source, destination)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _find_labeled_sensor(
    hwmon: Path, kind: str, label: str, suffix: str
) -> Path | None:
    for label_path in sorted(hwmon.glob(f"{kind}[0-9]*_label")):
        if _read_text(label_path) == label:
            return label_path.with_name(label_path.name.removesuffix("_label") + suffix)
    return None


@dataclass(frozen=True)
class SensorPaths:
    power_average: Path
    junction_temperature: Path
    power_cap: Path
    runtime_status: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "power_average": str(self.power_average),
            "junction_temperature": str(self.junction_temperature),
            "power_cap": str(self.power_cap),
            "runtime_status": str(self.runtime_status),
        }


def _find_gpu(card_name: str | None) -> tuple[SensorPaths, dict[str, Any]]:
    if card_name is not None and re.fullmatch(r"card\d+", card_name) is None:
        raise RuntimeError(f"Invalid DRM card name: {card_name!r}")
    drm_root = Path("/sys/class/drm")
    candidates = [drm_root / card_name] if card_name else sorted(drm_root.glob("card*"))
    for card in candidates:
        if re.fullmatch(r"card\d+", card.name) is None:
            continue
        device = card / "device"
        if (_read_text(device / "vendor") or "").lower() != "0x1002":
            continue
        hwmons = sorted((device / "hwmon").glob("hwmon*"))
        if not hwmons:
            continue
        hwmon = hwmons[0]
        junction = (
            _find_labeled_sensor(hwmon, "temp", "junction", "_input")
            or hwmon / "temp2_input"
        )
        paths = SensorPaths(
            power_average=hwmon / "power1_average",
            junction_temperature=junction,
            power_cap=hwmon / "power1_cap",
            runtime_status=device / "power" / "runtime_status",
        )
        if not paths.power_average.exists() or not paths.junction_temperature.exists():
            continue
        resolved_device = device.resolve()
        identity = {
            "card": card.name,
            "card_path": str(card),
            "pci_bdf": resolved_device.name,
            "vendor_id": _read_text(device / "vendor"),
            "device_id": _read_text(device / "device"),
            "hwmon": str(hwmon),
            "hwmon_name": _read_text(hwmon / "name"),
        }
        return paths, identity
    selected = f" {card_name!r}" if card_name else ""
    raise RuntimeError(
        f"No AMD DRM device{selected} with power and junction sensors was found"
    )


@dataclass(frozen=True)
class TimedRead:
    started_monotonic_ns: int
    completed_monotonic_ns: int
    completed_wall_time_ns: int
    value: float | int | str | None
    error_kind: str | None = None
    error_errno: int | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {
            "started_monotonic_ns": self.started_monotonic_ns,
            "completed_monotonic_ns": self.completed_monotonic_ns,
            "completed_wall_time_ns": self.completed_wall_time_ns,
            "value": self.value,
        }
        if self.error_kind is not None:
            result["error_kind"] = self.error_kind
        if self.error_errno is not None:
            result["error_errno"] = self.error_errno
        return result


def _parse_scaled_number(raw: str, scale: float) -> float:
    value = float(raw) / scale
    if not math.isfinite(value):
        raise ValueError("sensor value is not finite")
    return value


def _parse_exact_nonnegative_integer(raw: str) -> int:
    if re.fullmatch(r"[0-9]+", raw) is None:
        raise ValueError("sensor value is not a nonnegative decimal integer")
    return int(raw)


def _timed_read(
    path: Path,
    parser: Callable[[str], float | int | str],
    *,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    wall_time_ns: Callable[[], int] = time.time_ns,
) -> TimedRead:
    started = monotonic_ns()
    value: float | int | str | None = None
    error_kind: str | None = None
    error_errno: int | None = None
    try:
        raw = path.read_text().strip()
        value = parser(raw)
    except OSError as error:
        error_errno = error.errno
        error_kind = "busy" if error.errno == errno.EBUSY else "os_error"
    except (TypeError, ValueError, OverflowError):
        error_kind = "malformed"
    # The completion deadline must include the wall-clock stamp and all local
    # bookkeeping performed for this read.  Taking the monotonic timestamp
    # first would leave a deschedule inside ``wall_time_ns`` invisible.
    completed_wall_time = wall_time_ns()
    completed = monotonic_ns()
    return TimedRead(
        started_monotonic_ns=started,
        completed_monotonic_ns=completed,
        completed_wall_time_ns=completed_wall_time,
        value=value,
        error_kind=error_kind,
        error_errno=error_errno,
    )


@dataclass(frozen=True)
class SafetySample:
    sequence: int
    scheduled_monotonic_ns: int
    phase: str
    power: TimedRead
    junction: TimedRead
    power_cap: TimedRead
    runtime_status: TimedRead

    @property
    def completed_monotonic_ns(self) -> int:
        return self.runtime_status.completed_monotonic_ns

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "scheduled_monotonic_ns": self.scheduled_monotonic_ns,
            "phase": self.phase,
            "power": self.power.as_dict(),
            "junction": self.junction.as_dict(),
            "power_cap": self.power_cap.as_dict(),
            "runtime_status": self.runtime_status.as_dict(),
        }


def _read_safety_sample(
    paths: SensorPaths,
    *,
    sequence: int,
    scheduled_monotonic_ns: int,
    phase: str,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    wall_time_ns: Callable[[], int] = time.time_ns,
) -> SafetySample:
    # The ordering is safety-critical: power and junction always precede all
    # secondary reads.  Each primary read receives independent real timestamps.
    power = _timed_read(
        paths.power_average,
        lambda raw: _parse_scaled_number(raw, 1_000_000),
        monotonic_ns=monotonic_ns,
        wall_time_ns=wall_time_ns,
    )
    junction = _timed_read(
        paths.junction_temperature,
        lambda raw: _parse_scaled_number(raw, 1_000),
        monotonic_ns=monotonic_ns,
        wall_time_ns=wall_time_ns,
    )
    power_cap = _timed_read(
        paths.power_cap,
        _parse_exact_nonnegative_integer,
        monotonic_ns=monotonic_ns,
        wall_time_ns=wall_time_ns,
    )
    runtime_status = _timed_read(
        paths.runtime_status,
        lambda raw: raw,
        monotonic_ns=monotonic_ns,
        wall_time_ns=wall_time_ns,
    )
    return SafetySample(
        sequence=sequence,
        scheduled_monotonic_ns=scheduled_monotonic_ns,
        phase=phase,
        power=power,
        junction=junction,
        power_cap=power_cap,
        runtime_status=runtime_status,
    )


def _violation(
    metric: str,
    value: float | int | None,
    limit: float | int | None,
    limit_kind: str,
    reason: str,
    **details: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "metric": metric,
        "value": value,
        "limit": limit,
        "limit_kind": limit_kind,
        "reason": reason,
    }
    result.update(details)
    return result


@dataclass
class SensorGuard:
    started_monotonic_ns: int
    sensor_grace_ns: int
    maximum_read_gap_ns: int
    maximum_power_watts: float
    maximum_junction_temp_c: float
    expected_power_cap_microwatts: int
    first_power_observed_monotonic_ns: int | None = None
    first_junction_observed_monotonic_ns: int | None = None
    cap_attested_monotonic_ns: int | None = None
    previous_power_completed_monotonic_ns: int | None = None
    previous_junction_completed_monotonic_ns: int | None = None

    def scheduled_deadline_violation(
        self, scheduled_monotonic_ns: int, observed_monotonic_ns: int, source: str
    ) -> dict[str, Any] | None:
        """Reject work reached too late after the scheduled safety tick."""
        observed_ns = observed_monotonic_ns - scheduled_monotonic_ns
        if observed_ns <= self.maximum_read_gap_ns:
            return None
        return _violation(
            "sampler_read_gap_seconds",
            observed_ns / 1_000_000_000,
            self.maximum_read_gap_ns / 1_000_000_000,
            "maximum",
            "sampler_deadline",
            source=source,
        )

    def _deadline_violation(self, sample: SafetySample) -> dict[str, Any] | None:
        candidates: list[tuple[str, int]] = [
            (
                "power_scheduled_completion_lateness",
                sample.power.completed_monotonic_ns - sample.scheduled_monotonic_ns,
            ),
            (
                "junction_scheduled_completion_lateness",
                sample.junction.completed_monotonic_ns - sample.scheduled_monotonic_ns,
            ),
            (
                "power_cap_scheduled_completion_lateness",
                sample.power_cap.completed_monotonic_ns - sample.scheduled_monotonic_ns,
            ),
            (
                "sample_scheduled_completion_lateness",
                sample.completed_monotonic_ns - sample.scheduled_monotonic_ns,
            ),
            (
                "power_read_duration",
                sample.power.completed_monotonic_ns - sample.power.started_monotonic_ns,
            ),
            (
                "junction_read_duration",
                sample.junction.completed_monotonic_ns
                - sample.junction.started_monotonic_ns,
            ),
            (
                "power_cap_read_duration",
                sample.power_cap.completed_monotonic_ns
                - sample.power_cap.started_monotonic_ns,
            ),
            (
                "runtime_status_read_duration",
                sample.runtime_status.completed_monotonic_ns
                - sample.runtime_status.started_monotonic_ns,
            ),
        ]
        if self.previous_power_completed_monotonic_ns is not None:
            candidates.append(
                (
                    "power_completion_gap",
                    sample.power.completed_monotonic_ns
                    - self.previous_power_completed_monotonic_ns,
                )
            )
        if self.previous_junction_completed_monotonic_ns is not None:
            candidates.append(
                (
                    "junction_completion_gap",
                    sample.junction.completed_monotonic_ns
                    - self.previous_junction_completed_monotonic_ns,
                )
            )
        self.previous_power_completed_monotonic_ns = sample.power.completed_monotonic_ns
        self.previous_junction_completed_monotonic_ns = (
            sample.junction.completed_monotonic_ns
        )
        source, observed_ns = max(candidates, key=lambda item: item[1])
        if observed_ns <= self.maximum_read_gap_ns:
            return None
        return _violation(
            "sampler_read_gap_seconds",
            observed_ns / 1_000_000_000,
            self.maximum_read_gap_ns / 1_000_000_000,
            "maximum",
            "sampler_deadline",
            source=source,
        )

    def _sensor_unavailable_violation(
        self,
        name: str,
        reading: TimedRead,
        first_observed: int | None,
        now_ns: int,
        limit: float,
        runtime_state: str,
    ) -> dict[str, Any] | None:
        if reading.value is not None:
            return None
        if reading.error_kind != "busy":
            return _violation(
                name,
                None,
                limit,
                "maximum",
                "sensor_read_error",
                unavailable=True,
                error_kind=reading.error_kind,
                error_errno=reading.error_errno,
            )
        if first_observed is not None:
            return _violation(
                name,
                None,
                limit,
                "maximum",
                "sensor_disappeared_after_observation",
                unavailable=True,
                error_kind=reading.error_kind,
                error_errno=reading.error_errno,
            )
        if runtime_state not in {"suspended", "resuming"}:
            return _violation(
                name,
                None,
                limit,
                "maximum",
                "sensor_unavailable_for_runtime_state",
                unavailable=True,
                error_kind=reading.error_kind,
                error_errno=reading.error_errno,
                runtime_state=runtime_state,
            )
        if now_ns - self.started_monotonic_ns >= self.sensor_grace_ns:
            return _violation(
                name,
                None,
                limit,
                "maximum",
                "sensor_grace_expired",
                unavailable=True,
                error_kind=reading.error_kind,
                error_errno=reading.error_errno,
            )
        return None

    def evaluate(self, sample: SafetySample) -> dict[str, Any] | None:
        deadline = self._deadline_violation(sample)
        if deadline is not None:
            return deadline

        primary_is_readable = (
            sample.power.value is not None or sample.junction.value is not None
        )
        # This cap gate intentionally precedes runtime-state and primary-sensor
        # availability handling.  A readable mismatch is independently unsafe,
        # including while both primary sensors still report EBUSY.
        if sample.power_cap.value is not None:
            if int(sample.power_cap.value) != self.expected_power_cap_microwatts:
                return _violation(
                    "power1_cap_microwatts",
                    int(sample.power_cap.value),
                    self.expected_power_cap_microwatts,
                    "exact",
                    "power_cap_mismatch_after_resume",
                )
        elif primary_is_readable:
            return _violation(
                "power1_cap_microwatts",
                None,
                self.expected_power_cap_microwatts,
                "exact",
                "power_cap_unavailable_after_resume",
                unavailable=True,
                error_kind=sample.power_cap.error_kind,
                error_errno=sample.power_cap.error_errno,
            )

        runtime_state = sample.runtime_status.value
        accepted_runtime_states = {
            "active",
            "suspending",
            "suspended",
            "resuming",
            "unsupported",
        }
        if (
            not isinstance(runtime_state, str)
            or runtime_state not in accepted_runtime_states
            or sample.runtime_status.error_kind is not None
        ):
            return _violation(
                "gpu_runtime_status",
                None,
                None,
                "valid_state",
                "runtime_status_unavailable_or_invalid",
                unavailable=True,
                observed_value=runtime_state,
                error_kind=sample.runtime_status.error_kind,
                error_errno=sample.runtime_status.error_errno,
            )

        now_ns = max(
            sample.power.completed_monotonic_ns, sample.junction.completed_monotonic_ns
        )
        power_unavailable = self._sensor_unavailable_violation(
            "gpu_power_watts",
            sample.power,
            self.first_power_observed_monotonic_ns,
            now_ns,
            self.maximum_power_watts,
            runtime_state,
        )
        if power_unavailable is not None:
            return power_unavailable
        junction_unavailable = self._sensor_unavailable_violation(
            "gpu_junction_temp_c",
            sample.junction,
            self.first_junction_observed_monotonic_ns,
            now_ns,
            self.maximum_junction_temp_c,
            runtime_state,
        )
        if junction_unavailable is not None:
            return junction_unavailable

        if sample.power.value is not None:
            if self.first_power_observed_monotonic_ns is None:
                self.first_power_observed_monotonic_ns = (
                    sample.power.completed_monotonic_ns
                )
            if float(sample.power.value) > self.maximum_power_watts:
                return _violation(
                    "gpu_power_watts",
                    float(sample.power.value),
                    self.maximum_power_watts,
                    "maximum",
                    "limit_exceeded",
                )
        if sample.junction.value is not None:
            if self.first_junction_observed_monotonic_ns is None:
                self.first_junction_observed_monotonic_ns = (
                    sample.junction.completed_monotonic_ns
                )
            if float(sample.junction.value) > self.maximum_junction_temp_c:
                return _violation(
                    "gpu_junction_temp_c",
                    float(sample.junction.value),
                    self.maximum_junction_temp_c,
                    "maximum",
                    "limit_exceeded",
                )

        if primary_is_readable:
            if self.cap_attested_monotonic_ns is None:
                self.cap_attested_monotonic_ns = sample.power_cap.completed_monotonic_ns
        return None

    def completion_violation(self) -> dict[str, Any] | None:
        if self.first_power_observed_monotonic_ns is None:
            return _violation(
                "gpu_power_watts",
                None,
                self.maximum_power_watts,
                "maximum",
                "sensor_unobserved_before_completion",
                unavailable=True,
            )
        if self.first_junction_observed_monotonic_ns is None:
            return _violation(
                "gpu_junction_temp_c",
                None,
                self.maximum_junction_temp_c,
                "maximum",
                "sensor_unobserved_before_completion",
                unavailable=True,
            )
        if self.cap_attested_monotonic_ns is None:
            return _violation(
                "power1_cap_microwatts",
                None,
                self.expected_power_cap_microwatts,
                "exact",
                "power_cap_unattested_before_completion",
                unavailable=True,
            )
        return None


@dataclass
class AbsoluteScheduler:
    interval_ns: int
    next_tick_ns: int
    skipped_ticks: int = 0

    def advance(self, completed_monotonic_ns: int) -> int:
        """Advance on the original time grid and never issue catch-up samples."""
        self.next_tick_ns += self.interval_ns
        skipped = 0
        if self.next_tick_ns <= completed_monotonic_ns:
            skipped = (
                completed_monotonic_ns - self.next_tick_ns
            ) // self.interval_ns + 1
            self.next_tick_ns += skipped * self.interval_ns
        self.skipped_ticks += skipped
        return skipped


@dataclass
class HeartbeatAccumulator:
    heartbeat_ns: int
    window_started_monotonic_ns: int
    sample_count: int = 0
    total_sample_count: int = 0
    first_sample: SafetySample | None = None
    last_sample: SafetySample | None = None
    maximum_power_watts: float | None = None
    maximum_junction_temp_c: float | None = None
    maximum_power_completion_gap_ns: int = 0
    maximum_junction_completion_gap_ns: int = 0
    _previous_power_completed_ns: int | None = field(default=None, repr=False)
    _previous_junction_completed_ns: int | None = field(default=None, repr=False)

    def add(self, sample: SafetySample) -> None:
        if self.first_sample is None:
            self.first_sample = sample
        if self._previous_power_completed_ns is not None:
            self.maximum_power_completion_gap_ns = max(
                self.maximum_power_completion_gap_ns,
                sample.power.completed_monotonic_ns - self._previous_power_completed_ns,
            )
        if self._previous_junction_completed_ns is not None:
            self.maximum_junction_completion_gap_ns = max(
                self.maximum_junction_completion_gap_ns,
                sample.junction.completed_monotonic_ns
                - self._previous_junction_completed_ns,
            )
        self._previous_power_completed_ns = sample.power.completed_monotonic_ns
        self._previous_junction_completed_ns = sample.junction.completed_monotonic_ns
        self.last_sample = sample
        self.sample_count += 1
        self.total_sample_count += 1
        if sample.power.value is not None:
            value = float(sample.power.value)
            self.maximum_power_watts = (
                value
                if self.maximum_power_watts is None
                else max(self.maximum_power_watts, value)
            )
        if sample.junction.value is not None:
            value = float(sample.junction.value)
            self.maximum_junction_temp_c = (
                value
                if self.maximum_junction_temp_c is None
                else max(self.maximum_junction_temp_c, value)
            )

    def ready(self, now_ns: int) -> bool:
        return (
            self.sample_count > 0
            and now_ns - self.window_started_monotonic_ns >= self.heartbeat_ns
        )

    def pop_record(
        self, *, skipped_ticks: int, partial: bool = False
    ) -> dict[str, Any] | None:
        if self.first_sample is None or self.last_sample is None:
            return None
        first = self.first_sample
        last = self.last_sample
        record = {
            "record_type": "heartbeat",
            "partial": partial,
            "sample_count": self.sample_count,
            "sequence_first": first.sequence,
            "sequence_last": last.sequence,
            "scheduled_first_monotonic_ns": first.scheduled_monotonic_ns,
            "scheduled_last_monotonic_ns": last.scheduled_monotonic_ns,
            "power_read_first_started_monotonic_ns": first.power.started_monotonic_ns,
            "power_read_last_completed_monotonic_ns": last.power.completed_monotonic_ns,
            "power_read_last_completed_wall_time_ns": last.power.completed_wall_time_ns,
            "junction_read_first_started_monotonic_ns": first.junction.started_monotonic_ns,
            "junction_read_last_completed_monotonic_ns": last.junction.completed_monotonic_ns,
            "junction_read_last_completed_wall_time_ns": last.junction.completed_wall_time_ns,
            "maximum_power_completion_gap_seconds": self.maximum_power_completion_gap_ns
            / 1_000_000_000,
            "maximum_junction_completion_gap_seconds": self.maximum_junction_completion_gap_ns
            / 1_000_000_000,
            "maximum_power_watts": self.maximum_power_watts,
            "maximum_junction_temp_c": self.maximum_junction_temp_c,
            "last_power_cap_microwatts": last.power_cap.value,
            "last_runtime_status": last.runtime_status.value,
            "skipped_ticks_total": skipped_ticks,
        }
        self.window_started_monotonic_ns = last.completed_monotonic_ns
        self.sample_count = 0
        self.first_sample = None
        self.last_sample = None
        self.maximum_power_watts = None
        self.maximum_junction_temp_c = None
        self.maximum_power_completion_gap_ns = 0
        self.maximum_junction_completion_gap_ns = 0
        return record


class ExistingSummaryError(FileExistsError):
    def __init__(self, path: Path, prepared_expected_exit_code: int | None):
        super().__init__(
            errno.EEXIST,
            "refusing to replace an existing summary",
            str(path),
        )
        self.prepared_expected_exit_code = prepared_expected_exit_code


class EvidenceWriter:
    def __init__(self, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.output_path = output_path
        self.summary_path = output_path.with_suffix(
            output_path.suffix + ".summary.json"
        )
        self._directory_path = output_path.parent
        self._ancestor_path = output_path.parent.parent
        self._directory_entry_name = output_path.parent.name or "."
        self._ancestor_fd = -1
        self._ancestor_identity: tuple[int, int] | None = None
        self._directory_fd = -1
        self._directory_identity: tuple[int, int] | None = None
        self._directory_mode = 0
        self._output_fd = -1
        self._output_pin_fd = -1
        self._output_identity: tuple[int, int] | None = None
        self._summary_fd = -1
        self._summary_pin_fd = -1
        self._summary_stage_name: str | None = None
        self._summary_identity: tuple[int, int] | None = None
        self._output_hasher = hashlib.sha256()
        self.output_byte_count = 0
        self.output_line_count = 0
        self.output_sha256: str | None = None
        self.output_sealed = False
        self.summary_published = False
        self.unpublished_summary_name_invalidated = False
        self.unpublished_summary_interference_detected = False
        self.untrusted_summary_expected_exit_code: int | None = None
        self._summary_quarantine_names: dict[str, tuple[int, int]] = {}
        output_created = False
        try:
            self._ancestor_fd = os.open(
                self._ancestor_path,
                os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
            ancestor_stat = os.fstat(self._ancestor_fd)
            self._ancestor_identity = self._identity(ancestor_stat)
            self._verify_ancestor_binding("create_evidence_ancestor")
            self._directory_fd = os.open(
                output_path.parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
            )
            directory_stat = os.fstat(self._directory_fd)
            self._directory_identity = self._identity(directory_stat)
            self._directory_mode = stat.S_IMODE(directory_stat.st_mode)
            self._attest_private_directory(
                directory_stat,
                identity=self._directory_identity,
                operation="create_evidence_directory_fd",
            )
            self._verify_directory_binding("create_evidence_directory_path")
            try:
                os.stat(
                    self.summary_path.name,
                    dir_fd=self._directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise ExistingSummaryError(
                    self.summary_path,
                    self._read_untrusted_summary_expected_exit_code(),
                )
            self._output_fd = os.open(
                output_path.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=self._directory_fd,
            )
            output_created = True
            os.fchmod(self._output_fd, 0o600)
            created = os.fstat(self._output_fd)
            self._output_identity = (created.st_dev, created.st_ino)
            self._attest_regular(
                created,
                identity=self._output_identity,
                nlink=1,
                size=0,
                operation="create_jsonl",
            )
            # Make the initial evidence reservation durable.  The final summary
            # and its private stage remain absent until after containment.
            os.fsync(self._directory_fd)
        except BaseException:
            if output_created and self._output_identity is not None:
                try:
                    self._unlink_if_owned(output_path.name, self._output_identity)
                except OSError:
                    pass
            self.abort()
            # Never fall back to a pathname unlink after losing the pinned
            # directory/identity: a same-UID peer may have substituted it.
            raise

    @staticmethod
    def _identity(observed: os.stat_result) -> tuple[int, int]:
        return observed.st_dev, observed.st_ino

    @staticmethod
    def _attest_regular(
        observed: os.stat_result,
        *,
        identity: tuple[int, int],
        nlink: int,
        size: int,
        operation: str,
    ) -> None:
        if not stat.S_ISREG(observed.st_mode):
            raise RuntimeError(f"{operation}: evidence object is not a regular file")
        if EvidenceWriter._identity(observed) != identity:
            raise RuntimeError(f"{operation}: evidence inode identity changed")
        if observed.st_uid != os.geteuid():
            raise RuntimeError(f"{operation}: evidence owner is not the effective UID")
        if stat.S_IMODE(observed.st_mode) != 0o600:
            raise RuntimeError(f"{operation}: evidence mode is not exactly 0600")
        if observed.st_nlink != nlink:
            raise RuntimeError(
                f"{operation}: evidence link count {observed.st_nlink} != {nlink}"
            )
        if observed.st_size != size:
            raise RuntimeError(
                f"{operation}: evidence size {observed.st_size} != {size}"
            )

    @staticmethod
    def _attest_private_directory(
        observed: os.stat_result,
        *,
        identity: tuple[int, int],
        operation: str,
    ) -> None:
        if not stat.S_ISDIR(observed.st_mode):
            raise RuntimeError(f"{operation}: evidence parent is not a directory")
        if EvidenceWriter._identity(observed) != identity:
            raise RuntimeError(f"{operation}: evidence parent identity changed")
        if observed.st_uid != os.geteuid():
            raise RuntimeError(
                f"{operation}: evidence parent is not owned by the effective UID"
            )
        if stat.S_IMODE(observed.st_mode) & 0o077:
            raise RuntimeError(
                f"{operation}: evidence parent grants group/world permissions"
            )

    def _verify_directory_binding(self, operation: str) -> None:
        if self._directory_fd < 0 or self._directory_identity is None:
            raise RuntimeError(f"{operation}: evidence directory is unavailable")
        self._attest_private_directory(
            os.fstat(self._directory_fd),
            identity=self._directory_identity,
            operation=f"{operation}_fd",
        )
        self._verify_ancestor_binding(f"{operation}_ancestor")
        anchored = os.stat(
            self._directory_entry_name,
            dir_fd=self._ancestor_fd,
            follow_symlinks=False,
        )
        self._attest_private_directory(
            anchored,
            identity=self._directory_identity,
            operation=f"{operation}_anchored_path",
        )
        visible = os.stat(self._directory_path, follow_symlinks=False)
        self._attest_private_directory(
            visible,
            identity=self._directory_identity,
            operation=f"{operation}_path",
        )

    def _verify_ancestor_binding(self, operation: str) -> None:
        if self._ancestor_fd < 0 or self._ancestor_identity is None:
            raise RuntimeError(f"{operation}: evidence ancestor is unavailable")
        pinned = os.fstat(self._ancestor_fd)
        if not stat.S_ISDIR(pinned.st_mode):
            raise RuntimeError(f"{operation}: evidence ancestor is not a directory")
        if self._identity(pinned) != self._ancestor_identity:
            raise RuntimeError(f"{operation}: evidence ancestor identity changed")
        visible = os.stat(self._ancestor_path, follow_symlinks=False)
        if not stat.S_ISDIR(visible.st_mode):
            raise RuntimeError(
                f"{operation}: visible evidence ancestor is not a directory"
            )
        if self._identity(visible) != self._ancestor_identity:
            raise RuntimeError(
                f"{operation}: visible evidence ancestor identity changed"
            )

    def _lstat_attested(
        self,
        name: str,
        *,
        identity: tuple[int, int],
        nlink: int,
        size: int,
        operation: str,
    ) -> os.stat_result:
        if self._directory_fd < 0:
            raise RuntimeError(f"{operation}: evidence directory is unavailable")
        observed = os.stat(
            name,
            dir_fd=self._directory_fd,
            follow_symlinks=False,
        )
        self._attest_regular(
            observed,
            identity=identity,
            nlink=nlink,
            size=size,
            operation=operation,
        )
        return observed

    def _read_visible_bytes(
        self,
        name: str,
        *,
        identity: tuple[int, int],
        nlink: int,
        expected: bytes,
        operation: str,
    ) -> None:
        self._lstat_attested(
            name,
            identity=identity,
            nlink=nlink,
            size=len(expected),
            operation=f"{operation}_lstat_before",
        )
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=self._directory_fd,
        )
        try:
            self._attest_regular(
                os.fstat(descriptor),
                identity=identity,
                nlink=nlink,
                size=len(expected),
                operation=f"{operation}_fstat_before",
            )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
            if b"".join(chunks) != expected:
                raise RuntimeError(f"{operation}: evidence payload readback differs")
            self._attest_regular(
                os.fstat(descriptor),
                identity=identity,
                nlink=nlink,
                size=len(expected),
                operation=f"{operation}_fstat_after",
            )
        finally:
            closing = descriptor
            descriptor = -1
            os.close(closing)
        self._lstat_attested(
            name,
            identity=identity,
            nlink=nlink,
            size=len(expected),
            operation=f"{operation}_lstat_after",
        )

    def _read_visible_digest(
        self,
        name: str,
        *,
        identity: tuple[int, int],
        nlink: int,
        size: int,
        sha256: str,
        operation: str,
    ) -> None:
        self._lstat_attested(
            name,
            identity=identity,
            nlink=nlink,
            size=size,
            operation=f"{operation}_lstat_before",
        )
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=self._directory_fd,
        )
        observed_hash = hashlib.sha256()
        observed_size = 0
        try:
            self._attest_regular(
                os.fstat(descriptor),
                identity=identity,
                nlink=nlink,
                size=size,
                operation=f"{operation}_fstat_before",
            )
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                observed_hash.update(chunk)
                observed_size += len(chunk)
            if observed_size != size or observed_hash.hexdigest() != sha256:
                raise RuntimeError(f"{operation}: JSONL digest or byte count differs")
            self._attest_regular(
                os.fstat(descriptor),
                identity=identity,
                nlink=nlink,
                size=size,
                operation=f"{operation}_fstat_after",
            )
        finally:
            closing = descriptor
            descriptor = -1
            os.close(closing)
        self._lstat_attested(
            name,
            identity=identity,
            nlink=nlink,
            size=size,
            operation=f"{operation}_lstat_after",
        )

    def _unlink_if_owned(self, name: str, identity: tuple[int, int]) -> bool:
        if self._directory_fd < 0:
            return False
        try:
            observed = os.stat(
                name,
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return True
        if self._identity(observed) != identity:
            return False
        os.unlink(name, dir_fd=self._directory_fd)
        return True

    def _name_is_absent(self, name: str) -> bool:
        if self._directory_fd < 0:
            raise RuntimeError("evidence directory is unavailable")
        try:
            os.stat(name, dir_fd=self._directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return True
        return False

    def _read_expected_code_from_directory(
        self, directory_fd: int, name: str
    ) -> int | None:
        """Read a verifier-eligible stable regular entry without following links."""
        if directory_fd < 0:
            return None
        try:
            visible = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except (OSError, RuntimeError):
            return None
        if (
            not stat.S_ISREG(visible.st_mode)
            or visible.st_uid != os.geteuid()
            or stat.S_IMODE(visible.st_mode) != 0o600
            or visible.st_nlink != 1
            or visible.st_size > 1 << 20
        ):
            return None
        identity = self._identity(visible)
        descriptor = -1
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or self._identity(opened) != identity
                or opened.st_size != visible.st_size
                or opened.st_uid != os.geteuid()
                or stat.S_IMODE(opened.st_mode) != 0o600
                or opened.st_nlink != 1
            ):
                return None
            chunks: list[bytes] = []
            remaining = visible.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1 << 16))
                if not chunk:
                    return None
                chunks.append(chunk)
                remaining -= len(chunk)
            if os.read(descriptor, 1):
                return None
            after = os.fstat(descriptor)
            rebound = os.stat(
                name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if (
                self._identity(after) != identity
                or after.st_size != visible.st_size
                or after.st_nlink != 1
                or self._identity(rebound) != identity
                or rebound.st_size != visible.st_size
                or rebound.st_nlink != 1
            ):
                return None
            parsed = json.loads(b"".join(chunks))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        finally:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        if not isinstance(parsed, dict):
            return None
        if (
            parsed.get("record_type") != "summary"
            or parsed.get("schema_version") != 2
            or parsed.get("watcher_transaction") != "prepared"
            or parsed.get("commit_rule")
            != "observed_process_exit_must_match_expected_watcher_exit_code"
        ):
            return None
        expected = parsed.get("expected_watcher_exit_code")
        if isinstance(expected, bool) or not isinstance(expected, int):
            return None
        if not 0 <= expected <= 255:
            return None
        return expected

    def _read_untrusted_summary_expected_exit_code(self) -> int | None:
        """Read a stable final entry in the originally pinned output directory."""
        return self._read_expected_code_from_directory(
            self._directory_fd, self.summary_path.name
        )

    def read_visible_summary_expected_exit_code(self) -> int | None:
        """Boundedly read a replaced visible parent after terminal containment.

        The parent is resolved exactly once beneath the pinned ancestor and must
        itself be a private, same-euid directory.  Neither the parent nor final
        entry may be a symlink or other special object.
        """
        try:
            self._verify_ancestor_binding("read_visible_summary_ancestor")
            visible_parent = os.stat(
                self._directory_entry_name,
                dir_fd=self._ancestor_fd,
                follow_symlinks=False,
            )
        except (OSError, RuntimeError):
            return None
        if (
            not stat.S_ISDIR(visible_parent.st_mode)
            or visible_parent.st_uid != os.geteuid()
            or stat.S_IMODE(visible_parent.st_mode) & 0o077
        ):
            return None
        parent_identity = self._identity(visible_parent)
        descriptor = -1
        try:
            descriptor = os.open(
                self._directory_entry_name,
                os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=self._ancestor_fd,
            )
            opened_parent = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(opened_parent.st_mode)
                or self._identity(opened_parent) != parent_identity
                or opened_parent.st_uid != os.geteuid()
                or stat.S_IMODE(opened_parent.st_mode) & 0o077
            ):
                return None
            try:
                os.stat(
                    self.summary_path.name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                self.unpublished_summary_interference_detected = True
            expected = self._read_expected_code_from_directory(
                descriptor, self.summary_path.name
            )
            after_parent = os.fstat(descriptor)
            rebound_parent = os.stat(
                self._directory_entry_name,
                dir_fd=self._ancestor_fd,
                follow_symlinks=False,
            )
            if (
                self._identity(after_parent) != parent_identity
                or self._identity(rebound_parent) != parent_identity
            ):
                return None
            return expected
        except OSError:
            return None
        finally:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    def invalidate_unpublished_summary(self, *, terminal_containment: bool) -> None:
        """Quarantine a child-created final name after its scope is terminal.

        renameat2 operates on the directory entry itself, so regular files,
        hardlinks, symlinks, FIFOs, sockets, and non-empty directories are all
        moved without following a foreign target.  RENAME_NOREPLACE ensures a
        random-name collision cannot overwrite another directory entry.
        """
        if self.summary_published:
            raise RuntimeError("a published summary must never be invalidated")
        if not terminal_containment:
            raise RuntimeError(
                "unpublished summary invalidation requires terminal containment"
            )
        self.unpublished_summary_name_invalidated = False
        try:
            os.stat(
                self.summary_path.name,
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            self.unpublished_summary_interference_detected = True
        self._verify_directory_binding("invalidate_summary_directory_before")
        if stat.S_IMODE(os.fstat(self._directory_fd).st_mode) != self._directory_mode:
            os.fchmod(self._directory_fd, self._directory_mode)
            self._verify_directory_binding("invalidate_summary_directory_restored")

        observed_expected_exit_code = self._read_untrusted_summary_expected_exit_code()
        if observed_expected_exit_code is not None:
            self.untrusted_summary_expected_exit_code = observed_expected_exit_code
        moved_name: str | None = None
        moved_identity: tuple[int, int] | None = None
        rename_errors: list[OSError] = []
        for _attempt in range(16):
            try:
                source_stat = os.stat(
                    self.summary_path.name,
                    dir_fd=self._directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                break
            source_identity = self._identity(source_stat)
            quarantine_name = (
                f".{self.summary_path.name}.untrusted-{os.getpid()}-"
                f"{secrets.token_hex(16)}"
            )
            try:
                _rename_noreplace(
                    self.summary_path.name,
                    quarantine_name,
                    source_dir_fd=self._directory_fd,
                    destination_dir_fd=self._directory_fd,
                )
            except OSError as error:
                rename_errors.append(error)
                try:
                    source_absent = self._name_is_absent(self.summary_path.name)
                    quarantine_stat = os.stat(
                        quarantine_name,
                        dir_fd=self._directory_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    quarantine_stat = None
                if (
                    source_absent
                    and quarantine_stat is not None
                    and self._identity(quarantine_stat) == source_identity
                ):
                    moved_name = quarantine_name
                    moved_identity = source_identity
                    break
                if error.errno == errno.EEXIST:
                    continue
            else:
                moved_name = quarantine_name
                moved_identity = source_identity
                break
        if not self._name_is_absent(self.summary_path.name):
            # A non-directory fallback remains no-follow and is useful on a
            # filesystem that transiently rejects rename.  Directories stay
            # eligible for the type-agnostic rename retries above.
            try:
                os.unlink(self.summary_path.name, dir_fd=self._directory_fd)
            except FileNotFoundError:
                pass
            except OSError as unlink_error:
                detail = "; ".join(str(item) for item in rename_errors[-3:])
                raise RuntimeError(
                    "could not invalidate the unpublished summary final name; "
                    f"rename errors=[{detail}], unlink error={unlink_error}"
                ) from unlink_error
        if not self._name_is_absent(self.summary_path.name):
            raise RuntimeError("unpublished summary final name is still visible")

        os.fsync(self._directory_fd)
        self._verify_directory_binding("invalidate_summary_directory_after_fsync")
        if not self._name_is_absent(self.summary_path.name):
            raise RuntimeError(
                "unpublished summary final name reappeared after invalidation"
            )
        if moved_name is not None:
            if moved_identity is None:
                raise RuntimeError("quarantined summary identity is unavailable")
            quarantined = os.stat(
                moved_name,
                dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
            if self._identity(quarantined) != moved_identity:
                raise RuntimeError(
                    "quarantined summary identity changed after directory fsync"
                )
            self._summary_quarantine_names[moved_name] = moved_identity
        self.unpublished_summary_name_invalidated = True

    def _cleanup_summary_quarantines(self) -> None:
        if self._directory_fd < 0:
            return
        for quarantine_name, identity in tuple(self._summary_quarantine_names.items()):
            try:
                observed = os.stat(
                    quarantine_name,
                    dir_fd=self._directory_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                self._summary_quarantine_names.pop(quarantine_name, None)
                continue
            if self._identity(observed) != identity:
                continue
            try:
                os.unlink(quarantine_name, dir_fd=self._directory_fd)
            except IsADirectoryError:
                try:
                    os.rmdir(quarantine_name, dir_fd=self._directory_fd)
                except OSError:
                    continue
            except OSError:
                continue
            self._summary_quarantine_names.pop(quarantine_name, None)

    def _close_fd(self, attribute: str) -> None:
        descriptor = int(getattr(self, attribute))
        if descriptor < 0:
            return
        # close(2) may release a descriptor and still report an error.  Drop
        # numeric ownership first so no retry can close a reused descriptor.
        setattr(self, attribute, -1)
        os.close(descriptor)

    def _open_identity_pin(
        self,
        name: str,
        *,
        identity: tuple[int, int],
        nlink: int,
        size: int,
        operation: str,
    ) -> int:
        self._lstat_attested(
            name,
            identity=identity,
            nlink=nlink,
            size=size,
            operation=f"{operation}_lstat",
        )
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=self._directory_fd,
        )
        try:
            self._attest_regular(
                os.fstat(descriptor),
                identity=identity,
                nlink=nlink,
                size=size,
                operation=f"{operation}_fstat",
            )
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    @staticmethod
    def _write_all(descriptor: int, payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write to evidence file")
            view = view[written:]

    def write_record(self, value: Mapping[str, Any], *, durable: bool = False) -> None:
        if self._output_fd < 0 or self.output_sealed:
            raise RuntimeError("JSONL evidence is already sealed")
        payload = (_json_dumps(value, separators=(",", ":")) + "\n").encode()
        self._write_all(self._output_fd, payload)
        self._output_hasher.update(payload)
        self.output_byte_count += len(payload)
        self.output_line_count += 1
        if durable:
            os.fsync(self._output_fd)

    def seal_output(self) -> None:
        if self.output_sealed:
            return
        if self._output_fd < 0 or self._output_identity is None:
            raise RuntimeError("JSONL evidence descriptor is unavailable")
        self._verify_directory_binding("seal_jsonl_directory_before")
        expected_hash = self._output_hasher.hexdigest()
        os.fsync(self._output_fd)
        self._attest_regular(
            os.fstat(self._output_fd),
            identity=self._output_identity,
            nlink=1,
            size=self.output_byte_count,
            operation="seal_jsonl_write_fd",
        )
        self._read_visible_digest(
            self.output_path.name,
            identity=self._output_identity,
            nlink=1,
            size=self.output_byte_count,
            sha256=expected_hash,
            operation="seal_jsonl_readback",
        )
        self._output_pin_fd = self._open_identity_pin(
            self.output_path.name,
            identity=self._output_identity,
            nlink=1,
            size=self.output_byte_count,
            operation="seal_jsonl_pin",
        )
        self._close_fd("_output_fd")
        self._lstat_attested(
            self.output_path.name,
            identity=self._output_identity,
            nlink=1,
            size=self.output_byte_count,
            operation="seal_jsonl_after_close",
        )
        self._read_visible_digest(
            self.output_path.name,
            identity=self._output_identity,
            nlink=1,
            size=self.output_byte_count,
            sha256=expected_hash,
            operation="seal_jsonl_final_readback",
        )
        self._verify_directory_binding("seal_jsonl_directory_after")
        self.output_sha256 = expected_hash
        self.output_sealed = True

    def _verify_sealed_output(self, operation: str) -> None:
        if (
            not self.output_sealed
            or self.output_sha256 is None
            or self._output_identity is None
        ):
            raise RuntimeError(f"{operation}: JSONL evidence is not sealed")
        self._verify_directory_binding(f"{operation}_directory")
        self._read_visible_digest(
            self.output_path.name,
            identity=self._output_identity,
            nlink=1,
            size=self.output_byte_count,
            sha256=self.output_sha256,
            operation=operation,
        )

    def sealed_output_attestation(self) -> dict[str, Any]:
        self._verify_sealed_output("record_sealed_jsonl_attestation")
        if self._output_identity is None or self._directory_identity is None:
            raise RuntimeError("sealed JSONL identity is unavailable")
        output_stat = self._lstat_attested(
            self.output_path.name,
            identity=self._output_identity,
            nlink=1,
            size=self.output_byte_count,
            operation="record_sealed_jsonl_identity",
        )
        directory_stat = os.fstat(self._directory_fd)
        return {
            "sha256": self.output_sha256,
            "byte_count": self.output_byte_count,
            "line_count": self.output_line_count,
            "device": self._output_identity[0],
            "inode": self._output_identity[1],
            "owner_uid": output_stat.st_uid,
            "mode_octal": f"{stat.S_IMODE(output_stat.st_mode):04o}",
            "link_count": output_stat.st_nlink,
            "parent_directory": {
                "device": self._directory_identity[0],
                "inode": self._directory_identity[1],
                "owner_uid": directory_stat.st_uid,
                "mode_octal": f"{stat.S_IMODE(directory_stat.st_mode):04o}",
            },
        }

    def _reserve_summary_stage(self) -> None:
        if self._summary_fd >= 0 or self._summary_stage_name is not None:
            raise RuntimeError("private summary stage is already reserved")
        self._verify_directory_binding("reserve_summary_stage_directory")
        for _attempt in range(16):
            stage_name = (
                f".{self.summary_path.name}.prepared-{os.getpid()}-"
                f"{secrets.token_hex(8)}"
            )
            try:
                descriptor = os.open(
                    stage_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=self._directory_fd,
                )
            except FileExistsError:
                continue
            self._summary_fd = descriptor
            self._summary_stage_name = stage_name
            os.fchmod(descriptor, 0o600)
            observed = os.fstat(descriptor)
            self._summary_identity = self._identity(observed)
            self._attest_regular(
                observed,
                identity=self._summary_identity,
                nlink=1,
                size=0,
                operation="create_summary_stage",
            )
            return
        raise FileExistsError("could not reserve a private summary stage")

    def publish_summary(self, value: Mapping[str, Any]) -> None:
        if not self.output_sealed:
            raise RuntimeError(
                "JSONL evidence must be sealed before summary publication"
            )
        if self.summary_published:
            raise RuntimeError("summary is already published")
        self._verify_sealed_output("publish_summary_jsonl_before")
        payload = (_json_dumps(value, indent=2, sort_keys=True) + "\n").encode()
        self._reserve_summary_stage()
        if self._summary_identity is None or self._summary_stage_name is None:
            raise RuntimeError("private summary stage identity is unavailable")
        self._write_all(self._summary_fd, payload)
        os.fsync(self._summary_fd)
        self._attest_regular(
            os.fstat(self._summary_fd),
            identity=self._summary_identity,
            nlink=1,
            size=len(payload),
            operation="summary_stage_after_write",
        )
        self._summary_pin_fd = self._open_identity_pin(
            self._summary_stage_name,
            identity=self._summary_identity,
            nlink=1,
            size=len(payload),
            operation="summary_stage_pin",
        )
        self._close_fd("_summary_fd")
        self._read_visible_bytes(
            self._summary_stage_name,
            identity=self._summary_identity,
            nlink=1,
            expected=payload,
            operation="summary_stage_readback",
        )
        self._verify_sealed_output("publish_summary_jsonl_before_link")
        try:
            os.link(
                self._summary_stage_name,
                self.summary_path.name,
                src_dir_fd=self._directory_fd,
                dst_dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
        except BaseException:
            # If linkat succeeded but reporting success was interrupted, only
            # an inode-identical final artifact is treated as publication.
            try:
                self._lstat_attested(
                    self.summary_path.name,
                    identity=self._summary_identity,
                    nlink=2,
                    size=len(payload),
                    operation="summary_ambiguous_link",
                )
            except BaseException:
                pass
            else:
                self.summary_published = True
            raise
        # From this point on, every failure must produce an exit code different
        # from the summary's expected code.  The prepared artifact is visible but
        # is not committed until an observer sees the matching process exit.
        self.summary_published = True
        self._lstat_attested(
            self.summary_path.name,
            identity=self._summary_identity,
            nlink=2,
            size=len(payload),
            operation="summary_final_after_link",
        )
        self._lstat_attested(
            self._summary_stage_name,
            identity=self._summary_identity,
            nlink=2,
            size=len(payload),
            operation="summary_stage_after_link",
        )
        self._read_visible_bytes(
            self.summary_path.name,
            identity=self._summary_identity,
            nlink=2,
            expected=payload,
            operation="summary_final_linked_readback",
        )
        self._verify_directory_binding("publish_summary_directory_after_link")
        os.fsync(self._directory_fd)
        self._lstat_attested(
            self._summary_stage_name,
            identity=self._summary_identity,
            nlink=2,
            size=len(payload),
            operation="summary_stage_before_unlink",
        )
        os.unlink(self._summary_stage_name, dir_fd=self._directory_fd)
        self._summary_stage_name = None
        self._lstat_attested(
            self.summary_path.name,
            identity=self._summary_identity,
            nlink=1,
            size=len(payload),
            operation="summary_final_after_stage_unlink",
        )
        self._read_visible_bytes(
            self.summary_path.name,
            identity=self._summary_identity,
            nlink=1,
            expected=payload,
            operation="summary_final_readback",
        )
        self._verify_sealed_output("publish_summary_jsonl_final")
        self._verify_directory_binding("publish_summary_directory_final")
        self._close_fd("_summary_pin_fd")
        self._close_fd("_output_pin_fd")
        self._lstat_attested(
            self.summary_path.name,
            identity=self._summary_identity,
            nlink=1,
            size=len(payload),
            operation="summary_final_after_pin_close",
        )
        self._lstat_attested(
            self.output_path.name,
            identity=self._output_identity,
            nlink=1,
            size=self.output_byte_count,
            operation="jsonl_final_after_pin_close",
        )
        self._verify_directory_binding("publish_summary_directory_before_close")
        self._cleanup_summary_quarantines()
        os.fsync(self._directory_fd)
        self._close_fd("_directory_fd")
        self._close_fd("_ancestor_fd")

    def abort(self) -> None:
        """Best-effort pre-publication cleanup without replacing final evidence."""
        if (
            self._summary_stage_name is not None
            and self._summary_identity is not None
            and self._directory_fd >= 0
        ):
            try:
                if self._unlink_if_owned(
                    self._summary_stage_name, self._summary_identity
                ):
                    self._summary_stage_name = None
            except OSError:
                pass
        if self._directory_fd >= 0:
            self._cleanup_summary_quarantines()
        try:
            self._close_fd("_output_fd")
        except OSError:
            pass
        try:
            self._close_fd("_summary_fd")
        except OSError:
            pass
        try:
            self._close_fd("_summary_pin_fd")
        except OSError:
            pass
        try:
            self._close_fd("_output_pin_fd")
        except OSError:
            pass
        try:
            self._close_fd("_directory_fd")
        except OSError:
            pass
        try:
            self._close_fd("_ancestor_fd")
        except OSError:
            pass


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--card", help="DRM card name, for example card1 (default: auto-detect)"
    )
    parser.add_argument(
        "--interval-seconds", type=float, default=_DEFAULT_INTERVAL_SECONDS
    )
    parser.add_argument(
        "--maximum-read-gap-seconds",
        type=float,
        default=_DEFAULT_MAXIMUM_READ_GAP_SECONDS,
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=1.0)
    parser.add_argument("--sensor-grace-seconds", type=float, default=5.0)
    parser.add_argument(
        "--max-junction-temp-c", type=float, default=_DEFAULT_MAX_JUNCTION_TEMP_C
    )
    parser.add_argument(
        "--max-gpu-power-watts", type=float, default=_DEFAULT_MAX_POWER_WATTS
    )
    parser.add_argument(
        "--expected-power-cap-microwatts",
        type=int,
        default=_DEFAULT_EXPECTED_POWER_CAP_MICROWATTS,
    )
    parser.add_argument("--duration", type=float, help="attach-only duration")
    parser.add_argument("--timeout", type=float, help="wrapped-command timeout")
    parser.add_argument(
        "--terminate-grace-seconds",
        type=float,
        default=5.0,
        help=(
            "deprecated name for the containment proof/reap timeout; "
            "cgroup.kill is always written immediately"
        ),
    )
    parser.add_argument(
        "--include-pid",
        action="append",
        default=[],
        metavar="[LABEL=]PID",
        help=(
            "unsupported: external PID attachment cannot prove process-tree "
            "containment; launch the workload as the wrapped command"
        ),
    )
    parser.add_argument(
        "--terminate-included-on-safety",
        action="store_true",
        help=("unsupported: external PID trees cannot be safely adopted"),
    )
    parser.add_argument(
        "--pass-fd",
        action="append",
        default=[],
        type=int,
        metavar="FD",
        help="preserve this open descriptor in the wrapped command; repeat as needed",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]

    positive_floats = (
        ("--interval-seconds", args.interval_seconds),
        ("--maximum-read-gap-seconds", args.maximum_read_gap_seconds),
        ("--heartbeat-seconds", args.heartbeat_seconds),
        ("--terminate-grace-seconds", args.terminate_grace_seconds),
    )
    for name, value in positive_floats:
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.maximum_read_gap_seconds < args.interval_seconds:
        parser.error("--maximum-read-gap-seconds must be at least --interval-seconds")
    if not math.isfinite(args.sensor_grace_seconds) or args.sensor_grace_seconds < 0:
        parser.error("--sensor-grace-seconds must be finite and nonnegative")
    for name, value in (
        ("--max-junction-temp-c", args.max_junction_temp_c),
        ("--max-gpu-power-watts", args.max_gpu_power_watts),
    ):
        if not math.isfinite(value) or value < 0:
            parser.error(f"{name} must be finite and nonnegative")
    if args.expected_power_cap_microwatts < 0:
        parser.error("--expected-power-cap-microwatts must be nonnegative")
    if args.duration is not None and (
        not math.isfinite(args.duration) or args.duration <= 0
    ):
        parser.error("--duration must be finite and positive")
    if args.timeout is not None and (
        not math.isfinite(args.timeout) or args.timeout <= 0
    ):
        parser.error("--timeout must be finite and positive")
    if args.command and args.duration is not None:
        parser.error("--duration is attach-only; use --timeout with a command")
    if not args.command and args.timeout is not None:
        parser.error("--timeout requires a wrapped command")
    if args.include_pid:
        parser.error(
            "--include-pid is unsupported: external PID trees cannot be proven "
            "contained; launch the workload as the wrapped command"
        )
    if args.terminate_included_on_safety:
        parser.error(
            "--terminate-included-on-safety is unsupported: launch the workload "
            "as the wrapped command"
        )
    if not args.command and args.duration is None:
        parser.error("provide a wrapped command or --duration")
    if args.pass_fd and not args.command:
        parser.error("--pass-fd requires a wrapped command")
    try:
        args.pass_fds = _validated_pass_fds(tuple(args.pass_fd))
    except ValueError as error:
        parser.error(str(error))
    return args


def _manifest(
    args: argparse.Namespace,
    gpu_identity: Mapping[str, Any],
    paths: SensorPaths,
    started_monotonic_ns: int,
) -> dict[str, Any]:
    command = [args.command[0], "<arguments omitted>"] if args.command else []
    return {
        "record_type": "manifest",
        "schema_version": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "wall_time_ns": time.time_ns(),
        "started_monotonic_ns": started_monotonic_ns,
        "interval_seconds": args.interval_seconds,
        "maximum_read_gap_seconds": args.maximum_read_gap_seconds,
        "heartbeat_seconds": args.heartbeat_seconds,
        "sensor_grace_seconds": args.sensor_grace_seconds,
        "maximum_gpu_power_watts": args.max_gpu_power_watts,
        "maximum_junction_temp_c": args.max_junction_temp_c,
        "expected_power_cap_microwatts": args.expected_power_cap_microwatts,
        "process_containment": (
            "wrapped_command_private_cgroup_v2"
            if args.command
            else "none_duration_only_observation"
        ),
        "external_pid_attachment": "rejected",
        "gpu": dict(gpu_identity),
        "sensor_paths": paths.as_dict(),
        "command": command,
        "passed_file_descriptor_count": len(args.pass_fds),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _exit_code(status: str, returncode: int | None, received_signal: int | None) -> int:
    if status in {"safety_limit", "error"}:
        return _SAFETY_EXIT_CODE
    if status == "timeout":
        return _TIMEOUT_EXIT_CODE
    if received_signal is not None:
        return 128 + received_signal
    if returncode is None:
        return 0 if status == "completed" else 127
    return 128 - returncode if returncode < 0 else returncode


def _different_exit_code(expected: int | Collection[int]) -> int:
    """Return a valid shell status outside every untrusted prepared value."""
    excluded = {expected} if isinstance(expected, int) else set(expected)
    for candidate in (126, 127, 125, 124, 1, 255):
        if candidate not in excluded:
            return candidate
    for candidate in range(256):
        if candidate not in excluded:
            return candidate
    raise RuntimeError("all process exit codes were excluded")


def _report_returncode(report: ContainmentReport | None) -> int | None:
    if report is None:
        return None
    if report.reaped_returncode is not None:
        return report.reaped_returncode
    return report.observed_returncode


def _write_stdout_summary(summary: Mapping[str, Any]) -> None:
    payload = (_json_dumps(summary, indent=2, sort_keys=True) + "\n").encode()
    descriptor = sys.stdout.fileno()
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while emitting prepared summary")
        view = view[written:]
    sys.stdout.flush()


def _run(
    args: argparse.Namespace,
    *,
    keep_final_signals_blocked: bool = False,
) -> tuple[int, dict[str, Any]]:
    writer: EvidenceWriter | None = None
    runtime: ChildProcessRuntime | None = None
    wrapped: WrappedProcessSupervisor | None = None
    launch_containment: ContainmentReport | None = None
    initial_containment: ContainmentReport | None = None
    runtime_report: RuntimeReport | None = None
    status = "running"
    returncode: int | None = None
    received_signal: int | None = None
    stop_requested = False
    violation: dict[str, Any] | None = None
    violating_sample: SafetySample | None = None
    primary_error: BaseException | None = None
    watcher_errors: list[dict[str, str]] = []
    installed_signal_handlers: list[tuple[int, Any]] = []
    restored_signal_handlers: list[int] = []
    preflight_complete = False
    started_ns = time.monotonic_ns()
    scheduler: AbsoluteScheduler | None = None
    guard: SensorGuard | None = None
    accumulator: HeartbeatAccumulator | None = None
    sequence = 0
    old_signal_mask: set[signal.Signals] | None = None
    final_signals_blocked = False
    pending_signals: list[int] = []
    final_signal_cutoff_monotonic_ns: int | None = None
    containment_frozen_ns: int | None = None
    final_jsonl_record_written = False
    jsonl_sealed = False
    jsonl_attestation: dict[str, Any] | None = None
    unpublished_summary_name_safe = False
    untrusted_summary_expected_exit_codes: set[int] = set()
    preexisting_summary_expected_exit_code: int | None = None
    summary: dict[str, Any] = {}

    class _SetupInterruptedBySignal(Exception):
        pass

    class _PreSpawnRejected(Exception):
        pass

    def note_failure(phase: str, caught: BaseException | str) -> None:
        nonlocal primary_error
        if isinstance(caught, BaseException):
            if primary_error is None:
                primary_error = caught
            error_type = type(caught).__name__
            message = str(caught)
        else:
            error_type = "InvariantError"
            message = caught
        watcher_errors.append(
            {"phase": phase, "error_type": error_type, "message": message}
        )

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal received_signal, stop_requested
        if received_signal is None:
            received_signal = signum
        stop_requested = True

    def stop_if_setup_was_signaled() -> None:
        if stop_requested:
            raise _SetupInterruptedBySignal

    def collect_terminal_untrusted_summary_codes(
        evidence_writer: EvidenceWriter,
    ) -> None:
        for reader in (
            evidence_writer._read_untrusted_summary_expected_exit_code,
            evidence_writer.read_visible_summary_expected_exit_code,
        ):
            try:
                observed = reader()
            except BaseException:
                continue
            if observed is not None:
                untrusted_summary_expected_exit_codes.add(observed)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous = signal.signal(signum, handle_signal)
            installed_signal_handlers.append((signum, previous))
        stop_if_setup_was_signaled()

        paths, gpu_identity = _find_gpu(args.card)
        stop_if_setup_was_signaled()
        writer = EvidenceWriter(args.output)
        stop_if_setup_was_signaled()
        writer.write_record(
            _manifest(args, gpu_identity, paths, started_ns), durable=True
        )
        stop_if_setup_was_signaled()

        if args.command:
            runtime = ChildProcessRuntime(cgroup_prefix="skyrl-safety").start()
            stop_if_setup_was_signaled()

        # Runtime/cgroup setup is intentionally allowed to be slow.  The 50 ms
        # safety deadline starts only when the first actual sensor tick begins.
        sampling_started_ns = time.monotonic_ns()
        scheduler = AbsoluteScheduler(
            int(args.interval_seconds * 1_000_000_000), sampling_started_ns
        )
        guard = SensorGuard(
            started_monotonic_ns=sampling_started_ns,
            sensor_grace_ns=int(args.sensor_grace_seconds * 1_000_000_000),
            maximum_read_gap_ns=int(args.maximum_read_gap_seconds * 1_000_000_000),
            maximum_power_watts=args.max_gpu_power_watts,
            maximum_junction_temp_c=args.max_junction_temp_c,
            expected_power_cap_microwatts=args.expected_power_cap_microwatts,
        )
        accumulator = HeartbeatAccumulator(
            int(args.heartbeat_seconds * 1_000_000_000), sampling_started_ns
        )

        while not stop_requested:
            scheduled_ns = scheduler.next_tick_ns
            now_ns = time.monotonic_ns()
            if now_ns < scheduled_ns:
                time.sleep((scheduled_ns - now_ns) / 1_000_000_000)
            phase = "preflight" if sequence == 0 else "measured"
            sample = _read_safety_sample(
                paths,
                sequence=sequence,
                scheduled_monotonic_ns=scheduled_ns,
                phase=phase,
            )
            sequence += 1
            accumulator.add(sample)
            violation = guard.evaluate(sample)
            scheduler.advance(sample.completed_monotonic_ns)
            if violation is not None:
                violating_sample = sample
                status = "safety_limit"
                stop_requested = True
                break

            if phase == "preflight":
                preflight_complete = True
            if stop_requested:
                status = "signal"
                break
            if wrapped is None and args.command and phase == "preflight":
                if runtime is None:
                    raise RuntimeError("wrapped command runtime was not initialized")

                def pre_spawn_check() -> None:
                    nonlocal violation, violating_sample, status, stop_requested
                    if stop_requested:
                        status = "signal"
                        raise _PreSpawnRejected("signal received before launch")
                    violation = guard.scheduled_deadline_violation(
                        scheduled_ns,
                        time.monotonic_ns(),
                        "prelaunch_scheduled_lateness",
                    )
                    if violation is not None:
                        violating_sample = sample
                        status = "safety_limit"
                        stop_requested = True
                        raise _PreSpawnRejected("prelaunch safety deadline expired")

                try:
                    wrapped = runtime.launch(
                        args.command,
                        pass_fds=args.pass_fds,
                        pre_spawn_check=pre_spawn_check,
                    )
                except ProcessSupervisionError as caught:
                    launch_containment = caught.containment
                    if isinstance(caught.__cause__, _PreSpawnRejected):
                        if launch_containment is None:
                            note_failure(
                                "rejected_launch_scope_cleanup",
                                "rejected launch has no containment report",
                            )
                        break
                    raise
                # A signal can be delivered in the tiny boundary after the last
                # callback and before process creation returns.  The child is inside
                # its private scope and will be hard-contained below.
                if stop_requested:
                    status = "signal"
                    break

            if accumulator.ready(sample.completed_monotonic_ns):
                heartbeat = accumulator.pop_record(
                    skipped_ticks=scheduler.skipped_ticks
                )
                if heartbeat is not None:
                    writer.write_record(heartbeat)

            elapsed_seconds = (time.monotonic_ns() - started_ns) / 1_000_000_000
            if wrapped is not None:
                observed = wrapped.observe_returncode()
                if observed is not None:
                    returncode = observed
                    status = "completed" if observed == 0 else "command_failed"
                    break
            if args.timeout is not None and elapsed_seconds >= args.timeout:
                status = "timeout"
                stop_requested = True
                break
            if args.duration is not None and elapsed_seconds >= args.duration:
                status = "completed"
                break

        if received_signal is not None and status == "running":
            status = "signal"
        if status in {"completed", "command_failed"} and guard is not None:
            violation = guard.completion_violation()
            if violation is not None:
                violating_sample = (
                    None if accumulator is None else accumulator.last_sample
                )
                status = "safety_limit"
    except _SetupInterruptedBySignal:
        status = "signal"
    except BaseException as caught:
        if isinstance(caught, ExistingSummaryError):
            preexisting_summary_expected_exit_code = caught.prepared_expected_exit_code
        note_failure("monitor", caught)
        status = "error"

    # Containment precedes every final evidence operation.  No code below the
    # freeze point calls containment/runtime APIs or signals a process identity.
    if wrapped is not None:
        try:
            initial_containment = wrapped.contain(
                args.terminate_grace_seconds,
                graceful_term_seconds=0.0,
            )
        except BaseException as caught:
            note_failure("initial_containment", caught)

    try:
        old_signal_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM}
        )
        final_signals_blocked = True
    except BaseException as caught:
        note_failure("block_final_signals", caught)

    if runtime is not None:
        try:
            runtime_report = runtime.restore(
                timeout_seconds=args.terminate_grace_seconds
            )
        except BaseException as caught:
            note_failure("runtime_restore", caught)
    containment_frozen_ns = time.monotonic_ns()

    frozen_containment: ContainmentReport | None = (
        initial_containment if initial_containment is not None else launch_containment
    )
    runtime_owned_scopes = () if runtime_report is None else runtime_report.containment
    if runtime_owned_scopes:
        reference = initial_containment or launch_containment
        reference_pid = wrapped.pid if wrapped is not None else None
        if reference is not None:
            matching = [
                report
                for report in runtime_owned_scopes
                if (
                    report.cgroup_path == reference.cgroup_path
                    or report.pid == reference.pid
                )
            ]
        elif reference_pid is not None:
            matching = [
                report for report in runtime_owned_scopes if report.pid == reference_pid
            ]
        else:
            matching = []
        if matching:
            frozen_containment = matching[-1]
        elif wrapped is None:
            # Failed launch attestation can leave no wrapped supervisor object.
            # Runtime.restore() is authoritative for its newest owned scope.
            frozen_containment = runtime_owned_scopes[-1]

    containment_required = (
        wrapped is not None
        or launch_containment is not None
        or bool(runtime_owned_scopes)
    )
    if frozen_containment is None and runtime_owned_scopes:
        frozen_containment = runtime_owned_scopes[-1]
    if containment_required:
        if frozen_containment is None:
            note_failure("containment_freeze", "owned command scope has no report")
        elif not frozen_containment.ok:
            note_failure("containment_freeze", "final containment report is not OK")
    if wrapped is not None:
        observed_final = _report_returncode(frozen_containment)
        if observed_final is not None:
            if returncode is not None and returncode != observed_final:
                note_failure(
                    "returncode_consistency",
                    f"observed {returncode}, contained {observed_final}",
                )
            returncode = observed_final
    if runtime_report is not None and not runtime_report.ok:
        note_failure("runtime_freeze", "child runtime report is not OK")

    def proves_terminal_scope(report: ContainmentReport) -> bool:
        return report.terminal and report.cgroup_empty and report.leader_reaped

    terminal_owned_containment = not containment_required or (
        frozen_containment is not None
        and proves_terminal_scope(frozen_containment)
        and all(proves_terminal_scope(report) for report in runtime_owned_scopes)
    )
    if writer is not None and containment_required:
        if terminal_owned_containment:
            try:
                writer.invalidate_unpublished_summary(terminal_containment=True)
                unpublished_summary_name_safe = True
                if writer.untrusted_summary_expected_exit_code is not None:
                    untrusted_summary_expected_exit_codes.add(
                        writer.untrusted_summary_expected_exit_code
                    )
                if writer.unpublished_summary_interference_detected:
                    note_failure(
                        "unpublished_summary_interference",
                        "wrapped child created the reserved summary final name",
                    )
            except BaseException as caught:
                collect_terminal_untrusted_summary_codes(writer)
                note_failure("invalidate_unpublished_summary", caught)
        else:
            note_failure(
                "invalidate_unpublished_summary",
                "terminal containment is unavailable; a same-UID child may still "
                "control the final name, so it was not touched or trusted",
            )
    elif writer is not None:
        # No wrapped process ever received authority to create this name, and
        # EvidenceWriter proved it absent during initialization.
        unpublished_summary_name_safe = True

    containment_evidence: dict[str, Any] = {
        "required": containment_required,
        "mechanism": (
            "private_cgroup_v2_immediate_kill"
            if containment_required
            else "none_duration_only_observation"
        ),
        "graceful_termination_seconds": 0.0,
        "frozen_monotonic_ns": containment_frozen_ns,
        "initial_report": (
            None if initial_containment is None else initial_containment.as_dict()
        ),
        "launch_report": (
            None if launch_containment is None else launch_containment.as_dict()
        ),
        "final_report": (
            None if frozen_containment is None else frozen_containment.as_dict()
        ),
        "runtime_report": (
            None if runtime_report is None else runtime_report.as_dict()
        ),
    }

    if writer is not None:
        try:
            if status == "safety_limit" and violation is not None:
                writer.write_record(
                    {
                        "record_type": "safety_violation",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "wall_time_ns": time.time_ns(),
                        "violation": violation,
                        "sample": (
                            None
                            if violating_sample is None
                            else violating_sample.as_dict()
                        ),
                        "containment": containment_evidence,
                    }
                )
                final_jsonl_record_written = True
            elif accumulator is not None and scheduler is not None:
                partial = accumulator.pop_record(
                    skipped_ticks=scheduler.skipped_ticks, partial=True
                )
                if partial is not None:
                    partial["containment"] = containment_evidence
                    writer.write_record(partial)
                    final_jsonl_record_written = True
        except BaseException as caught:
            note_failure("final_jsonl_record", caught)
        try:
            writer.seal_output()
            jsonl_sealed = True
        except BaseException as caught:
            note_failure("seal_jsonl", caught)

    # Restore every handler independently while INT/TERM remain blocked.
    for signum, previous in reversed(installed_signal_handlers):
        try:
            signal.signal(signum, previous)
            restored_signal_handlers.append(signum)
        except BaseException as caught:
            note_failure(f"restore_signal_handler_{signum}", caught)

    if final_signals_blocked:
        try:
            final_signal_cutoff_monotonic_ns = time.monotonic_ns()
            pending_signals = sorted(
                int(item)
                for item in signal.sigpending()
                if item in {signal.SIGINT, signal.SIGTERM}
            )
        except BaseException as caught:
            note_failure("read_pending_signals", caught)
    if received_signal is None and pending_signals:
        received_signal = pending_signals[0]
    if received_signal is not None and status in {
        "running",
        "completed",
        "command_failed",
    }:
        status = "signal"

    if writer is not None and jsonl_sealed:
        try:
            jsonl_attestation = writer.sealed_output_attestation()
        except BaseException as caught:
            note_failure("attest_sealed_jsonl", caught)
            jsonl_sealed = False

    if watcher_errors and status == "running":
        status = "error"
    completed_ns = time.monotonic_ns()
    sample_count = 0 if accumulator is None else accumulator.total_sample_count
    skipped_ticks = 0 if scheduler is None else scheduler.skipped_ticks
    expected_code = (
        _SAFETY_EXIT_CODE
        if watcher_errors
        else _exit_code(status, returncode, received_signal)
    )
    summary = {
        "record_type": "summary",
        "schema_version": 2,
        "watcher_transaction": "prepared",
        "commit_rule": "observed_process_exit_must_match_expected_watcher_exit_code",
        "expected_watcher_exit_code": expected_code,
        "watcher_status": "error" if watcher_errors else "ok",
        "workload_outcome": {
            "status": status,
            "returncode": returncode,
            "received_signal": received_signal,
            "safety_violation": violation,
        },
        "started_monotonic_ns": started_ns,
        "prepared_monotonic_ns": completed_ns,
        "elapsed_seconds": (completed_ns - started_ns) / 1_000_000_000,
        "sample_count": sample_count,
        "skipped_ticks": skipped_ticks,
        "preflight_complete": preflight_complete,
        "first_power_observed_monotonic_ns": (
            None if guard is None else guard.first_power_observed_monotonic_ns
        ),
        "first_junction_observed_monotonic_ns": (
            None if guard is None else guard.first_junction_observed_monotonic_ns
        ),
        "power_cap_attested_monotonic_ns": (
            None if guard is None else guard.cap_attested_monotonic_ns
        ),
        "external_pid_attachment": "rejected",
        "containment": containment_evidence,
        "evidence": {
            "jsonl_final_record_written": final_jsonl_record_written,
            "jsonl_sealed": jsonl_sealed,
            "jsonl_attestation": jsonl_attestation,
            "unpublished_summary_name_invalidated_after_containment": (
                writer is not None and writer.unpublished_summary_name_invalidated
            ),
            "unpublished_summary_interference_detected": (
                writer is not None and writer.unpublished_summary_interference_detected
            ),
            "unpublished_summary_detected_expected_exit_code": (
                None if writer is None else writer.untrusted_summary_expected_exit_code
            ),
            "unproven_containment_same_uid_final_name_limitation": (
                containment_required and not terminal_owned_containment
            ),
            "terminal_untrusted_summary_expected_exit_codes": sorted(
                untrusted_summary_expected_exit_codes
            ),
        },
        "final_signal_cutoff": {
            "monotonic_ns": final_signal_cutoff_monotonic_ns,
            "signals_blocked": final_signals_blocked,
            "pending_signals": pending_signals,
            "handlers_restored": sorted(restored_signal_handlers),
        },
        "watcher_errors": watcher_errors,
    }
    if primary_error is not None:
        summary["primary_error_type"] = type(primary_error).__name__
        summary["primary_error"] = str(primary_error)

    actual_code = expected_code
    published = False
    if (
        writer is not None
        and jsonl_sealed
        and final_signals_blocked
        and unpublished_summary_name_safe
    ):
        try:
            writer.publish_summary(summary)
            published = True
        except BaseException:
            # The summary is immutable once linked.  A mismatch exit is the only
            # safe way to invalidate a visible prepared artifact.
            if writer.summary_published:
                actual_code = _different_exit_code(expected_code)
            else:
                try:
                    writer.invalidate_unpublished_summary(
                        terminal_containment=terminal_owned_containment
                    )
                except BaseException:
                    if terminal_owned_containment:
                        collect_terminal_untrusted_summary_codes(writer)
                actual_code = (
                    _SAFETY_EXIT_CODE
                    if writer.unpublished_summary_name_invalidated
                    else _different_exit_code(
                        untrusted_summary_expected_exit_codes or {expected_code}
                    )
                )
                writer.abort()
    else:
        if writer is None and preexisting_summary_expected_exit_code is not None:
            actual_code = _different_exit_code(preexisting_summary_expected_exit_code)
        else:
            actual_code = (
                _SAFETY_EXIT_CODE
                if unpublished_summary_name_safe
                else _different_exit_code(
                    untrusted_summary_expected_exit_codes or {expected_code}
                )
            )
        if writer is not None:
            writer.abort()

    if published:
        try:
            _write_stdout_summary(summary)
        except BaseException:
            actual_code = _different_exit_code(expected_code)

    if not keep_final_signals_blocked and final_signals_blocked:
        try:
            signal.pthread_sigmask(
                signal.SIG_SETMASK,
                set() if old_signal_mask is None else old_signal_mask,
            )
        except BaseException:
            if published:
                actual_code = _different_exit_code(expected_code)
            else:
                actual_code = _SAFETY_EXIT_CODE

    return actual_code, summary


def main(argv: list[str] | None = None) -> NoReturn:
    args = _parse_args(argv)
    try:
        code, _summary = _run(args, keep_final_signals_blocked=True)
    except FileExistsError as error:
        raise SystemExit(
            f"refusing to overwrite existing safety evidence: {error.filename}"
        ) from error
    except ValueError as error:
        raise SystemExit(str(error)) from error
    os._exit(code)


if __name__ == "__main__":
    main()
