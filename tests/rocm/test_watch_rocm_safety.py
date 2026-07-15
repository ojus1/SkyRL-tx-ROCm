from __future__ import annotations

import errno
import hashlib
import inspect
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from rocm import watch_rocm_safety as watcher
from rocm.process_supervision import SupervisionIssue


def _reading(
    value,
    *,
    started: int,
    completed: int | None = None,
    error_kind: str | None = None,
    error_errno: int | None = None,
) -> watcher.TimedRead:
    return watcher.TimedRead(
        started_monotonic_ns=started,
        completed_monotonic_ns=started + 1 if completed is None else completed,
        completed_wall_time_ns=10_000 + started,
        value=value,
        error_kind=error_kind,
        error_errno=error_errno,
    )


def _sample(
    *,
    sequence: int = 0,
    base: int = 1_000_000_000,
    power=200.0,
    junction=50.0,
    cap=315_000_000,
    power_error: str | None = None,
    junction_error: str | None = None,
    cap_error: str | None = None,
    runtime: str | None = "active",
    runtime_error: str | None = None,
) -> watcher.SafetySample:
    return watcher.SafetySample(
        sequence=sequence,
        scheduled_monotonic_ns=base,
        phase="preflight" if sequence == 0 else "measured",
        power=_reading(power, started=base + 1, error_kind=power_error),
        junction=_reading(junction, started=base + 3, error_kind=junction_error),
        power_cap=_reading(cap, started=base + 5, error_kind=cap_error),
        runtime_status=_reading(runtime, started=base + 7, error_kind=runtime_error),
    )


def _guard(
    *, started: int = 1_000_000_000, grace_ns: int = 60_000_000_000
) -> watcher.SensorGuard:
    return watcher.SensorGuard(
        started_monotonic_ns=started,
        sensor_grace_ns=grace_ns,
        maximum_read_gap_ns=50_000_000,
        maximum_power_watts=400.0,
        maximum_junction_temp_c=90.0,
        expected_power_cap_microwatts=315_000_000,
    )


def _forged_prepared_summary(expected_exit_code: int, *, nonce: str) -> dict:
    return {
        "record_type": "summary",
        "schema_version": 2,
        "watcher_transaction": "prepared",
        "commit_rule": ("observed_process_exit_must_match_expected_watcher_exit_code"),
        "expected_watcher_exit_code": expected_exit_code,
        "forged_nonce": nonce,
    }


class _OrderedPath:
    def __init__(self, name: str, value: str, reads: list[str]):
        self.name = name
        self.value = value
        self.reads = reads

    def read_text(self) -> str:
        self.reads.append(self.name)
        return self.value

    def __str__(self) -> str:
        return f"/mock/{self.name}"


def test_primary_reads_are_first_and_have_independent_actual_timestamps():
    order: list[str] = []
    paths = watcher.SensorPaths(
        power_average=_OrderedPath("power", "200000000", order),  # type: ignore[arg-type]
        junction_temperature=_OrderedPath("junction", "50000", order),  # type: ignore[arg-type]
        power_cap=_OrderedPath("cap", "315000000", order),  # type: ignore[arg-type]
        runtime_status=_OrderedPath("runtime", "active", order),  # type: ignore[arg-type]
    )
    monotonic_values: Iterator[int] = iter(range(100, 108))
    wall_values: Iterator[int] = iter(range(1_000, 1_004))

    sample = watcher._read_safety_sample(
        paths,
        sequence=3,
        scheduled_monotonic_ns=99,
        phase="measured",
        monotonic_ns=lambda: next(monotonic_values),
        wall_time_ns=lambda: next(wall_values),
    )

    assert order == ["power", "junction", "cap", "runtime"]
    assert (sample.power.started_monotonic_ns, sample.power.completed_monotonic_ns) == (
        100,
        101,
    )
    assert (
        sample.junction.started_monotonic_ns,
        sample.junction.completed_monotonic_ns,
    ) == (102, 103)
    assert sample.power.completed_wall_time_ns == 1_000
    assert sample.junction.completed_wall_time_ns == 1_001


def test_timed_read_completion_includes_delayed_wall_time_bookkeeping():
    now = 1_000_000_000

    def monotonic_ns() -> int:
        return now

    def delayed_wall_time_ns() -> int:
        nonlocal now
        now += 50_000_001
        return 123

    reading = watcher._timed_read(
        _OrderedPath("power", "200000000", []),  # type: ignore[arg-type]
        lambda raw: watcher._parse_scaled_number(raw, 1_000_000),
        monotonic_ns=monotonic_ns,
        wall_time_ns=delayed_wall_time_ns,
    )

    assert reading.completed_wall_time_ns == 123
    assert reading.completed_monotonic_ns == 1_050_000_001


def test_hot_sample_path_has_no_process_or_heavy_host_sampling():
    source = inspect.getsource(watcher._read_safety_sample)
    for forbidden in (
        "psutil",
        "smaps",
        "journalctl",
        "virtual_memory",
        "swap_memory",
        "cpu_percent",
    ):
        assert forbidden not in source


def test_absolute_scheduler_skips_missed_ticks_without_catch_up():
    scheduler = watcher.AbsoluteScheduler(interval_ns=25, next_tick_ns=100)
    assert scheduler.advance(101) == 0
    assert scheduler.next_tick_ns == 125
    assert scheduler.advance(181) == 2
    assert scheduler.next_tick_ns == 200
    assert scheduler.skipped_ticks == 2


@pytest.mark.parametrize(
    ("delta", "violates"), ((50_000_000, False), (50_000_001, True))
)
def test_primary_completion_gap_has_exact_50ms_boundary(delta: int, violates: bool):
    guard = _guard()
    assert guard.evaluate(_sample(base=1_000_000_000)) is None
    result = guard.evaluate(_sample(sequence=1, base=1_000_000_000 + delta))
    assert (result is not None) is violates


@pytest.mark.parametrize(
    ("power", "junction", "metric"),
    ((400.0001, 50.0, "gpu_power_watts"), (200.0, 90.0001, "gpu_junction_temp_c")),
)
def test_single_limit_breach_fails_without_debounce(power, junction, metric):
    violation = _guard().evaluate(_sample(power=power, junction=junction))
    assert violation is not None
    assert violation["metric"] == metric
    assert violation["reason"] == "limit_exceeded"


def test_exact_limits_and_power_cap_pass_and_are_attested():
    guard = _guard()
    assert guard.evaluate(_sample(power=400.0, junction=90.0)) is None
    assert guard.cap_attested_monotonic_ns is not None


@pytest.mark.parametrize(
    ("cap", "cap_error", "reason"),
    (
        (350_000_000, None, "power_cap_mismatch_after_resume"),
        (None, "busy", "power_cap_unavailable_after_resume"),
    ),
)
def test_first_readable_primary_requires_exact_power_cap(cap, cap_error, reason):
    violation = _guard().evaluate(_sample(cap=cap, cap_error=cap_error))
    assert violation is not None
    assert violation["reason"] == reason


def test_suspended_busy_sensor_gets_only_bounded_startup_grace():
    guard = _guard(grace_ns=10)
    busy = _sample(
        power=None,
        junction=None,
        power_error="busy",
        junction_error="busy",
        runtime="suspended",
    )
    assert guard.evaluate(busy) is None
    late = _sample(
        sequence=1,
        base=1_000_000_011,
        power=None,
        junction=None,
        power_error="busy",
        junction_error="busy",
        runtime="suspended",
    )
    assert guard.evaluate(late)["reason"] == "sensor_grace_expired"


def test_completion_requires_both_primary_sensors_and_cap_attestation():
    guard = _guard()
    assert (
        guard.evaluate(
            _sample(
                junction=None,
                junction_error="busy",
                runtime="resuming",
            )
        )
        is None
    )
    violation = guard.completion_violation()
    assert violation is not None
    assert violation["metric"] == "gpu_junction_temp_c"


def test_heartbeat_aggregates_and_resets_one_window():
    accumulator = watcher.HeartbeatAccumulator(
        heartbeat_ns=1_000_000_000,
        window_started_monotonic_ns=1_000_000_000,
    )
    accumulator.add(_sample(sequence=0, base=1_000_000_000, power=200.0))
    last = _sample(sequence=1, base=2_000_000_000, power=300.0, junction=70.0)
    accumulator.add(last)
    record = accumulator.pop_record(skipped_ticks=2)
    assert record is not None
    assert record["sample_count"] == 2
    assert record["maximum_power_watts"] == 300.0
    assert record["maximum_junction_temp_c"] == 70.0
    assert accumulator.total_sample_count == 2


def _args(tmp_path: Path, *extra: str):
    return watcher._parse_args(
        [
            "--output",
            str(tmp_path / "safety.jsonl"),
            "--interval-seconds",
            "0.001",
            "--maximum-read-gap-seconds",
            "1",
            "--terminate-grace-seconds",
            "0.2",
            *extra,
        ]
    )


def _mock_samples(monkeypatch, specifications):
    iterator = iter(specifications)
    monkeypatch.setattr(
        watcher,
        "_find_gpu",
        lambda _card: (
            watcher.SensorPaths(
                *(
                    Path(f"/mock/{name}")
                    for name in ("power", "junction", "cap", "runtime")
                )
            ),
            {"card": "mock"},
        ),
    )

    def read_sample(
        _paths,
        *,
        sequence: int,
        scheduled_monotonic_ns: int,
        phase: str,
    ):
        specification = next(iterator)
        sample = _sample(
            sequence=sequence,
            base=scheduled_monotonic_ns,
            **specification,
        )
        assert sample.phase == phase
        return sample

    monkeypatch.setattr(watcher, "_read_safety_sample", read_sample)


def _containment(
    *, pid: int = 4242, returncode: int = 0, terminal: bool = True
) -> watcher.ContainmentReport:
    return watcher.ContainmentReport(
        pid=pid,
        observed_returncode=returncode,
        reaped_returncode=returncode,
        cgroup_path=f"/fake/cgroup/{pid}",
        cgroup_kill_writes=1,
        process_group_term_sent=False,
        cgroup_empty=terminal,
        leader_reaped=terminal,
        terminal=terminal,
        issues=(),
    )


class _FakeSupervisor:
    def __init__(self, events: list[object], *, observations=None, report=None):
        self.pid = 4242
        self.events = events
        self.observations = iter([0] if observations is None else observations)
        self.report = _containment() if report is None else report
        self.contain_calls: list[tuple[float, float]] = []

    def observe_returncode(self):
        self.events.append("observe")
        return next(self.observations, None)

    def contain(self, timeout_seconds, *, graceful_term_seconds=0.0):
        self.events.append("contain")
        self.contain_calls.append((timeout_seconds, graceful_term_seconds))
        return self.report


class _FakeRuntime:
    def __init__(
        self,
        supervisor: _FakeSupervisor,
        events: list[object],
        *,
        runtime_report: watcher.RuntimeReport | None = None,
        restore_error: BaseException | None = None,
        launch_error: BaseException | None = None,
    ):
        self.supervisor = supervisor
        self.events = events
        self.runtime_report = runtime_report
        self.restore_error = restore_error
        self.launch_error = launch_error
        self.launched = False

    def start(self):
        self.events.append("runtime_start")
        return self

    def launch(self, command, *, pass_fds=(), pre_spawn_check=None):
        self.events.append(("launch", tuple(command), tuple(pass_fds)))
        if pre_spawn_check is not None:
            try:
                pre_spawn_check()
            except BaseException as error:
                report = _containment(pid=-1)
                raise watcher.ProcessSupervisionError(
                    "pre-spawn rejected",
                    containment=report,
                ) from error
        if self.launch_error is not None:
            raise self.launch_error
        self.launched = True
        return self.supervisor

    def restore(self, *, timeout_seconds):
        self.events.append(("restore", timeout_seconds))
        if self.restore_error is not None:
            raise self.restore_error
        if self.runtime_report is not None:
            return self.runtime_report
        return watcher.RuntimeReport(
            containment=(self.supervisor.report,) if self.launched else (),
            sigchld_restored=True,
            issues=(),
        )


def _install_fake_runtime(monkeypatch, runtime: _FakeRuntime):
    monkeypatch.setattr(watcher, "ChildProcessRuntime", lambda **_kwargs: runtime)


def test_evidence_stage_is_deferred_private_and_removed_after_publish(
    monkeypatch,
    tmp_path: Path,
):
    output = tmp_path / "safety.jsonl"
    writer = watcher.EvidenceWriter(output)
    try:
        assert stat.S_IMODE(output.stat().st_mode) == 0o600
        assert not writer.summary_path.exists()
        stages = list(tmp_path.glob(".safety.jsonl.summary.json.prepared-*"))
        assert stages == []
        writer.write_record({"record_type": "manifest"})
        writer.seal_output()
        assert not writer.summary_path.exists()
        assert list(tmp_path.glob(".safety.jsonl.summary.json.prepared-*")) == []
        real_link = watcher.os.link
        observed_stage: dict[str, int] = {}

        def inspect_stage(source, destination, **kwargs):
            stage = tmp_path / source
            stage_stat = stage.lstat()
            observed_stage["mode"] = stat.S_IMODE(stage_stat.st_mode)
            observed_stage["links"] = stage_stat.st_nlink
            observed_stage["owner"] = stage_stat.st_uid
            return real_link(source, destination, **kwargs)

        monkeypatch.setattr(watcher.os, "link", inspect_stage)
        writer.publish_summary(
            {"watcher_transaction": "prepared", "expected_watcher_exit_code": 0}
        )
        assert observed_stage == {
            "mode": 0o600,
            "links": 1,
            "owner": os.geteuid(),
        }
        assert stat.S_IMODE(writer.summary_path.stat().st_mode) == 0o600
        assert writer.summary_path.stat().st_nlink == 1
        assert list(tmp_path.glob(".safety.jsonl.summary.json.prepared-*")) == []
    finally:
        writer.abort()


def test_summary_publication_never_replaces_an_existing_path(tmp_path: Path):
    writer = watcher.EvidenceWriter(tmp_path / "safety.jsonl")
    writer.write_record({"record_type": "manifest"})
    writer.seal_output()
    writer.summary_path.write_text("owned\n")
    try:
        with pytest.raises(FileExistsError):
            writer.publish_summary({"watcher_transaction": "prepared"})
        assert writer.summary_path.read_text() == "owned\n"
        assert not writer.summary_published
    finally:
        writer.abort()


def test_evidence_writer_refuses_a_summary_existing_before_start(tmp_path: Path):
    output = tmp_path / "safety.jsonl"
    summary = tmp_path / "safety.jsonl.summary.json"
    payload = json.dumps(_forged_prepared_summary(125, nonce="preexisting"))
    summary.write_text(payload)
    summary.chmod(0o600)

    with pytest.raises(FileExistsError, match="existing summary"):
        watcher.EvidenceWriter(output)

    assert summary.read_text() == payload
    assert not output.exists()


@pytest.mark.parametrize("forged_expected", (0, 1, 124, 125, 126, 127, 143, 255))
def test_preexisting_valid_summary_is_preserved_but_cannot_commit_refusal_exit(
    monkeypatch, tmp_path: Path, forged_expected: int
):
    summary_path = tmp_path / "safety.jsonl.summary.json"
    payload = json.dumps(
        _forged_prepared_summary(forged_expected, nonce="preexisting-run")
    )
    summary_path.write_text(payload)
    summary_path.chmod(0o600)
    _mock_samples(monkeypatch, [{}])

    code, summary = watcher._run(_args(tmp_path, "--duration", "0.01"))

    assert summary_path.read_text() == payload
    assert not (tmp_path / "safety.jsonl").exists()
    assert code != forged_expected
    assert summary["watcher_status"] == "error"
    assert any(
        error["error_type"] == "ExistingSummaryError"
        for error in summary["watcher_errors"]
    )


@pytest.mark.parametrize(
    "kind",
    (
        "regular",
        "hardlink",
        "symlink",
        "fifo",
        "socket",
        "nonempty_directory",
    ),
)
def test_terminal_invalidation_quarantines_every_final_entry_without_following(
    tmp_path: Path, kind: str
):
    output = tmp_path / "safety.jsonl"
    writer = watcher.EvidenceWriter(output)
    final = writer.summary_path
    target = tmp_path / "foreign-target"
    target_payload = json.dumps(_forged_prepared_summary(126, nonce=kind))
    bound_socket: socket.socket | None = None
    try:
        if kind == "regular":
            final.write_text(target_payload)
            final.chmod(0o600)
        elif kind == "hardlink":
            target.write_text(target_payload)
            target.chmod(0o600)
            os.link(target, final)
        elif kind == "symlink":
            target.write_text(target_payload)
            target.chmod(0o600)
            final.symlink_to(target)
        elif kind == "fifo":
            os.mkfifo(final, 0o600)
        elif kind == "socket":
            bound_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            bound_socket.bind(str(final))
        else:
            final.mkdir(mode=0o700)
            (final / "child-owned").write_text("must not be traversed")

        writer.invalidate_unpublished_summary(terminal_containment=True)
        writer.invalidate_unpublished_summary(terminal_containment=True)

        assert not final.exists()
        assert not final.is_symlink()
        assert writer.unpublished_summary_name_invalidated
        assert writer.unpublished_summary_interference_detected
        if kind == "regular":
            assert writer.untrusted_summary_expected_exit_code == 126
        else:
            assert writer.untrusted_summary_expected_exit_code is None
        if kind in {"hardlink", "symlink"}:
            assert target.read_text() == target_payload
    finally:
        if bound_socket is not None:
            bound_socket.close()
        writer.abort()


def test_unpublished_invalidation_requires_terminal_scope_and_spares_published(
    tmp_path: Path,
):
    output = tmp_path / "safety.jsonl"
    writer = watcher.EvidenceWriter(output)
    forged = json.dumps(_forged_prepared_summary(125, nonce="live-child"))
    writer.summary_path.write_text(forged)
    writer.summary_path.chmod(0o600)
    try:
        with pytest.raises(RuntimeError, match="terminal containment"):
            writer.invalidate_unpublished_summary(terminal_containment=False)
        assert writer.summary_path.read_text() == forged
    finally:
        writer.abort()

    output = tmp_path / "published.jsonl"
    writer = watcher.EvidenceWriter(output)
    writer.write_record({"record_type": "manifest"})
    writer.seal_output()
    writer.publish_summary(_forged_prepared_summary(0, nonce="watcher-owned"))
    published = writer.summary_path.read_bytes()
    with pytest.raises(RuntimeError, match="published summary"):
        writer.invalidate_unpublished_summary(terminal_containment=True)
    assert writer.summary_path.read_bytes() == published


def test_invalidation_skips_a_colliding_random_quarantine_name(
    monkeypatch, tmp_path: Path
):
    writer = watcher.EvidenceWriter(tmp_path / "safety.jsonl")
    writer.summary_path.write_text(
        json.dumps(_forged_prepared_summary(125, nonce="child"))
    )
    writer.summary_path.chmod(0o600)
    collision = tmp_path / (
        f".{writer.summary_path.name}.untrusted-{os.getpid()}-collision"
    )
    collision.write_text("foreign quarantine entry")
    collision.chmod(0o600)
    tokens = iter(("collision", "fresh"))
    monkeypatch.setattr(watcher.secrets, "token_hex", lambda _size: next(tokens))
    try:
        writer.invalidate_unpublished_summary(terminal_containment=True)
        assert not writer.summary_path.exists()
        assert collision.read_text() == "foreign quarantine entry"
        assert writer.unpublished_summary_interference_detected
    finally:
        writer.abort()


def test_ambiguous_noreplace_rename_is_reconciled_by_inode(monkeypatch, tmp_path: Path):
    writer = watcher.EvidenceWriter(tmp_path / "safety.jsonl")
    writer.summary_path.write_text(
        json.dumps(_forged_prepared_summary(125, nonce="ambiguous-rename"))
    )
    writer.summary_path.chmod(0o600)
    original_identity = (
        writer.summary_path.lstat().st_dev,
        writer.summary_path.lstat().st_ino,
    )
    real_rename = watcher._rename_noreplace

    def rename_then_report_error(source, destination, **kwargs):
        real_rename(source, destination, **kwargs)
        raise OSError(errno.EINTR, "synthetic ambiguous rename")

    monkeypatch.setattr(watcher, "_rename_noreplace", rename_then_report_error)
    try:
        writer.invalidate_unpublished_summary(terminal_containment=True)
        assert not writer.summary_path.exists()
        assert original_identity in writer._summary_quarantine_names.values()
        assert writer.unpublished_summary_interference_detected
    finally:
        writer.abort()


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (None, 126),
        ("record_type", None),
        ("schema_version", None),
        ("watcher_transaction", None),
        ("commit_rule", None),
        ("boolean_exit", None),
        ("out_of_range_exit", None),
    ),
)
def test_untrusted_expected_code_requires_a_stable_regular_v2_prepared_schema(
    tmp_path: Path, mutation: str | None, expected: int | None
):
    writer = watcher.EvidenceWriter(tmp_path / "safety.jsonl")
    payload = _forged_prepared_summary(126, nonce="schema")
    if mutation == "boolean_exit":
        payload["expected_watcher_exit_code"] = True
    elif mutation == "out_of_range_exit":
        payload["expected_watcher_exit_code"] = 256
    elif mutation is not None:
        payload[mutation] = "invalid"
    writer.summary_path.write_text(json.dumps(payload))
    writer.summary_path.chmod(0o600)
    try:
        assert writer._read_untrusted_summary_expected_exit_code() == expected
    finally:
        writer.abort()


def test_untrusted_expected_code_does_not_follow_a_summary_symlink(tmp_path: Path):
    writer = watcher.EvidenceWriter(tmp_path / "safety.jsonl")
    target = tmp_path / "foreign-valid-summary"
    payload = json.dumps(_forged_prepared_summary(126, nonce="symlink-target"))
    target.write_text(payload)
    target.chmod(0o600)
    writer.summary_path.symlink_to(target)
    try:
        assert writer._read_untrusted_summary_expected_exit_code() is None
        assert target.read_text() == payload
    finally:
        writer.abort()


def test_post_link_directory_fsync_failure_leaves_only_uncommitted_prepared_file(
    monkeypatch, tmp_path: Path
):
    writer = watcher.EvidenceWriter(tmp_path / "safety.jsonl")
    writer.write_record({"record_type": "manifest"})
    writer.seal_output()
    real_fsync = watcher.os.fsync

    def fail_after_link(descriptor):
        if writer.summary_published and descriptor == writer._directory_fd:
            raise OSError("directory fsync failed")
        return real_fsync(descriptor)

    monkeypatch.setattr(watcher.os, "fsync", fail_after_link)
    try:
        with pytest.raises(OSError, match="directory fsync failed"):
            writer.publish_summary(
                {"watcher_transaction": "prepared", "expected_watcher_exit_code": 0}
            )
        assert writer.summary_published
        assert writer.summary_path.exists()
    finally:
        writer.abort()


def test_ambiguous_summary_fd_close_is_never_retried(monkeypatch, tmp_path: Path):
    writer = watcher.EvidenceWriter(tmp_path / "safety.jsonl")
    writer.write_record({"record_type": "manifest"})
    writer.seal_output()
    real_close = watcher.os.close
    real_fsync = watcher.os.fsync
    stage_fds: set[int] = set()
    failed = False

    def track_stage_fd(descriptor):
        if descriptor == writer._summary_fd and descriptor >= 0:
            stage_fds.add(descriptor)
        return real_fsync(descriptor)

    def close_then_fail(descriptor):
        nonlocal failed
        if descriptor in stage_fds and not failed:
            failed = True
            real_close(descriptor)
            raise OSError("ambiguous close")
        return real_close(descriptor)

    monkeypatch.setattr(watcher.os, "fsync", track_stage_fd)
    monkeypatch.setattr(watcher.os, "close", close_then_fail)
    with pytest.raises(OSError, match="ambiguous close"):
        writer.publish_summary({"watcher_transaction": "prepared"})
    assert writer._summary_fd == -1
    reused = os.open("/dev/null", os.O_RDONLY)
    try:
        writer.abort()
        os.fstat(reused)
    finally:
        os.close(reused)


@pytest.mark.parametrize(
    "mutation", ("unlink", "replace", "symlink", "hardlink", "content")
)
def test_jsonl_substitution_is_rejected_before_seal(tmp_path: Path, mutation: str):
    output = tmp_path / "safety.jsonl"
    writer = watcher.EvidenceWriter(output)
    writer.write_record({"record_type": "manifest", "nonce": "original"})
    alias = tmp_path / "jsonl-alias"
    target = tmp_path / "jsonl-target"
    try:
        original = output.read_bytes()
        if mutation == "unlink":
            output.unlink()
        elif mutation == "replace":
            output.unlink()
            output.write_bytes(original)
            output.chmod(0o600)
        elif mutation == "symlink":
            target.write_bytes(original)
            target.chmod(0o600)
            output.unlink()
            output.symlink_to(target)
        elif mutation == "hardlink":
            os.link(output, alias)
        else:
            output.write_bytes(b"x" * len(original))

        with pytest.raises((OSError, RuntimeError)):
            writer.seal_output()
        assert not writer.output_sealed
        assert not writer.summary_path.exists()
    finally:
        writer.abort()


@pytest.mark.parametrize(
    "mutation", ("unlink", "replace", "symlink", "hardlink", "content")
)
def test_sealed_jsonl_substitution_prevents_summary_publication(
    tmp_path: Path, mutation: str
):
    output = tmp_path / "safety.jsonl"
    writer = watcher.EvidenceWriter(output)
    writer.write_record({"record_type": "manifest", "nonce": "original"})
    writer.seal_output()
    alias = tmp_path / "sealed-jsonl-alias"
    target = tmp_path / "sealed-jsonl-target"
    try:
        original = output.read_bytes()
        if mutation == "unlink":
            output.unlink()
        elif mutation == "replace":
            output.unlink()
            output.write_bytes(original)
            output.chmod(0o600)
        elif mutation == "symlink":
            target.write_bytes(original)
            target.chmod(0o600)
            output.unlink()
            output.symlink_to(target)
        elif mutation == "hardlink":
            os.link(output, alias)
        else:
            output.write_bytes(b"x" * len(original))

        with pytest.raises((OSError, RuntimeError)):
            writer.publish_summary(
                {"watcher_transaction": "prepared", "expected_watcher_exit_code": 0}
            )
        assert not writer.summary_published
        assert not writer.summary_path.exists()
        assert list(tmp_path.glob(".safety.jsonl.summary.json.prepared-*")) == []
    finally:
        writer.abort()


@pytest.mark.parametrize(
    "mutation", ("unlink", "replace", "symlink", "hardlink", "content")
)
def test_summary_stage_substitution_is_rejected(
    monkeypatch, tmp_path: Path, mutation: str
):
    writer = watcher.EvidenceWriter(tmp_path / "safety.jsonl")
    writer.write_record({"record_type": "manifest"})
    writer.seal_output()
    real_link = watcher.os.link
    alias = tmp_path / "summary-stage-alias"
    target = tmp_path / "summary-stage-target"

    def mutate_stage_then_link(source, destination, **kwargs):
        stage = tmp_path / source
        payload = stage.read_bytes()
        if mutation == "unlink":
            stage.unlink()
        elif mutation == "replace":
            stage.unlink()
            stage.write_bytes(payload)
            stage.chmod(0o600)
        elif mutation == "symlink":
            target.write_bytes(payload)
            target.chmod(0o600)
            stage.unlink()
            stage.symlink_to(target)
        elif mutation == "hardlink":
            real_link(stage, alias)
        else:
            stage.write_bytes(b"x" * len(payload))
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(watcher.os, "link", mutate_stage_then_link)
    try:
        with pytest.raises((OSError, RuntimeError)):
            writer.publish_summary(
                {"watcher_transaction": "prepared", "expected_watcher_exit_code": 0}
            )
        if mutation == "unlink":
            assert not writer.summary_published
            assert not writer.summary_path.exists()
        else:
            assert writer.summary_published
    finally:
        writer.abort()


@pytest.mark.parametrize(
    "mutation", ("unlink", "replace", "symlink", "hardlink", "content")
)
def test_summary_final_substitution_after_link_is_rejected(
    monkeypatch, tmp_path: Path, mutation: str
):
    writer = watcher.EvidenceWriter(tmp_path / "safety.jsonl")
    writer.write_record({"record_type": "manifest"})
    writer.seal_output()
    real_link = watcher.os.link
    alias = tmp_path / "summary-final-alias"
    target = tmp_path / "summary-final-target"

    def link_then_mutate(source, destination, **kwargs):
        result = real_link(source, destination, **kwargs)
        final = tmp_path / destination
        payload = (tmp_path / source).read_bytes()
        if mutation == "unlink":
            final.unlink()
        elif mutation == "replace":
            final.unlink()
            final.write_bytes(payload)
            final.chmod(0o600)
        elif mutation == "symlink":
            target.write_bytes(payload)
            target.chmod(0o600)
            final.unlink()
            final.symlink_to(target)
        elif mutation == "hardlink":
            real_link(final, alias)
        else:
            final.write_bytes(b"x" * len(payload))
        return result

    monkeypatch.setattr(watcher.os, "link", link_then_mutate)
    try:
        with pytest.raises((OSError, RuntimeError)):
            writer.publish_summary(
                {"watcher_transaction": "prepared", "expected_watcher_exit_code": 0}
            )
        assert writer.summary_published
    finally:
        writer.abort()


def test_parent_directory_replacement_and_permission_relaxation_are_rejected(
    tmp_path: Path,
):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    writer = watcher.EvidenceWriter(private / "safety.jsonl")
    writer.write_record({"record_type": "manifest"})
    displaced = tmp_path / "displaced"
    private.rename(displaced)
    private.mkdir(mode=0o700)
    try:
        with pytest.raises(RuntimeError, match="identity changed"):
            writer.seal_output()
    finally:
        writer.abort()

    guarded = tmp_path / "guarded"
    guarded.mkdir(mode=0o700)
    writer = watcher.EvidenceWriter(guarded / "safety.jsonl")
    writer.write_record({"record_type": "manifest"})
    guarded.chmod(0o750)
    try:
        with pytest.raises(RuntimeError, match="group/world"):
            writer.seal_output()
    finally:
        guarded.chmod(0o700)
        writer.abort()


def test_insecure_evidence_parent_is_rejected_before_output_creation(tmp_path: Path):
    insecure = tmp_path / "insecure"
    insecure.mkdir(mode=0o755)
    output = insecure / "safety.jsonl"
    with pytest.raises(RuntimeError, match="group/world"):
        watcher.EvidenceWriter(output)
    assert not output.exists()


def test_parent_directory_replacement_during_summary_link_is_postpublication(
    monkeypatch, tmp_path: Path
):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    writer = watcher.EvidenceWriter(private / "safety.jsonl")
    writer.write_record({"record_type": "manifest"})
    writer.seal_output()
    displaced = tmp_path / "displaced"
    real_link = watcher.os.link

    def link_then_replace_parent(source, destination, **kwargs):
        result = real_link(source, destination, **kwargs)
        private.rename(displaced)
        private.mkdir(mode=0o700)
        return result

    monkeypatch.setattr(watcher.os, "link", link_then_replace_parent)
    try:
        with pytest.raises(RuntimeError, match="identity changed"):
            writer.publish_summary(
                {"watcher_transaction": "prepared", "expected_watcher_exit_code": 0}
            )
        assert writer.summary_published
        assert (displaced / "safety.jsonl.summary.json").exists()
    finally:
        writer.abort()


@pytest.mark.parametrize("stage", ("write", "fsync", "link"))
def test_prepublication_summary_failures_never_create_a_final_artifact(
    monkeypatch, tmp_path: Path, stage: str
):
    writer = watcher.EvidenceWriter(tmp_path / "safety.jsonl")
    writer.write_record({"record_type": "manifest"})
    writer.seal_output()
    if stage == "write":
        real_write = watcher.os.write

        def fail_write(descriptor, value):
            if descriptor == writer._summary_fd:
                raise OSError("stage write failed")
            return real_write(descriptor, value)

        monkeypatch.setattr(watcher.os, "write", fail_write)
    elif stage == "fsync":
        real_fsync = watcher.os.fsync

        def fail_fsync(descriptor):
            if descriptor == writer._summary_fd:
                raise OSError("stage fsync failed")
            return real_fsync(descriptor)

        monkeypatch.setattr(watcher.os, "fsync", fail_fsync)
    else:
        monkeypatch.setattr(
            watcher.os,
            "link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("stage link failed")
            ),
        )
    try:
        with pytest.raises(OSError):
            writer.publish_summary({"watcher_transaction": "prepared"})
        assert not writer.summary_published
        assert not writer.summary_path.exists()
    finally:
        writer.abort()


@pytest.mark.parametrize(
    "external_args",
    (
        ("--include-pid", "server=123"),
        ("--terminate-included-on-safety",),
        ("--include-pid", "123", "--terminate-included-on-safety"),
    ),
)
def test_external_pid_options_are_rejected(tmp_path: Path, external_args):
    with pytest.raises(SystemExit) as raised:
        watcher._parse_args(
            [
                "--output",
                str(tmp_path / "safety.jsonl"),
                *external_args,
                "--duration",
                "1",
            ]
        )
    assert raised.value.code == 2


def test_watcher_contains_no_raw_or_snapshot_process_termination_code():
    source = Path(watcher.__file__).read_text()
    for forbidden in (
        "import psutil",
        "subprocess.Popen",
        "os.killpg",
        ".poll(",
        "_terminate_process_trees",
        "_terminate_wrapped_process",
    ):
        assert forbidden not in source


def test_wrapped_success_is_prepared_with_terminal_cgroup_evidence(
    monkeypatch, tmp_path: Path
):
    events: list[object] = []
    supervisor = _FakeSupervisor(events)
    runtime = _FakeRuntime(supervisor, events)
    _install_fake_runtime(monkeypatch, runtime)
    _mock_samples(monkeypatch, [{}])

    code, summary = watcher._run(
        _args(tmp_path, "--timeout", "2", "--", sys.executable, "-c", "pass")
    )

    persisted = json.loads((tmp_path / "safety.jsonl.summary.json").read_text())
    assert persisted == summary
    assert code == summary["expected_watcher_exit_code"] == 0
    assert summary["watcher_transaction"] == "prepared"
    assert summary["workload_outcome"]["status"] == "completed"
    assert not summary["evidence"]["unpublished_summary_interference_detected"]
    assert (
        summary["evidence"]["unpublished_summary_detected_expected_exit_code"] is None
    )
    assert summary["containment"]["final_report"]["terminal"] is True
    assert summary["containment"]["final_report"]["cgroup_kill_writes"] == 1
    jsonl = (tmp_path / "safety.jsonl").read_bytes()
    jsonl_stat = (tmp_path / "safety.jsonl").stat()
    attestation = summary["evidence"]["jsonl_attestation"]
    assert attestation["sha256"] == hashlib.sha256(jsonl).hexdigest()
    assert attestation["byte_count"] == len(jsonl)
    assert attestation["line_count"] == len(jsonl.splitlines())
    assert (attestation["device"], attestation["inode"]) == (
        jsonl_stat.st_dev,
        jsonl_stat.st_ino,
    )
    assert attestation["owner_uid"] == os.geteuid()
    assert attestation["mode_octal"] == "0600"
    assert attestation["link_count"] == 1
    parent_stat = tmp_path.stat()
    assert (
        attestation["parent_directory"]["device"],
        attestation["parent_directory"]["inode"],
    ) == (parent_stat.st_dev, parent_stat.st_ino)
    assert supervisor.contain_calls == [(0.2, 0.0)]
    assert events.index("contain") < next(
        index
        for index, item in enumerate(events)
        if isinstance(item, tuple) and item[0] == "restore"
    )


@pytest.mark.parametrize("forged_expected", (0, 1, 124, 125, 126, 127, 143, 255))
def test_contained_child_forgery_is_replaced_for_every_fallback_candidate(
    monkeypatch, tmp_path: Path, forged_expected: int
):
    events: list[object] = []
    supervisor = _FakeSupervisor(events)
    runtime = _FakeRuntime(supervisor, events)
    original_launch = runtime.launch
    forged_inode: int | None = None
    summary_path = tmp_path / "safety.jsonl.summary.json"

    def launch_and_forge(command, *, pass_fds=(), pre_spawn_check=None):
        nonlocal forged_inode
        result = original_launch(
            command, pass_fds=pass_fds, pre_spawn_check=pre_spawn_check
        )
        summary_path.write_text(
            json.dumps(
                _forged_prepared_summary(
                    forged_expected, nonce=f"forged-{forged_expected}"
                )
            )
        )
        summary_path.chmod(0o600)
        forged_inode = summary_path.stat().st_ino
        return result

    runtime.launch = launch_and_forge  # type: ignore[method-assign]
    _install_fake_runtime(monkeypatch, runtime)
    _mock_samples(monkeypatch, [{}])

    code, summary = watcher._run(
        _args(tmp_path, "--timeout", "2", "--", sys.executable, "-c", "pass")
    )

    persisted = json.loads(summary_path.read_text())
    assert persisted == summary
    assert "forged_nonce" not in persisted
    assert forged_inode is not None
    assert summary_path.stat().st_ino != forged_inode
    assert code == persisted["expected_watcher_exit_code"] == 125
    assert persisted["watcher_status"] == "error"
    assert summary["evidence"]["unpublished_summary_name_invalidated_after_containment"]
    assert summary["evidence"]["unpublished_summary_interference_detected"]
    assert (
        summary["evidence"]["unpublished_summary_detected_expected_exit_code"]
        == forged_expected
    )
    assert any(
        error["phase"] == "unpublished_summary_interference"
        for error in summary["watcher_errors"]
    )
    assert not summary["evidence"][
        "unproven_containment_same_uid_final_name_limitation"
    ]


@pytest.mark.parametrize("forged_expected", (0, 1, 124, 125, 126, 127, 143, 255))
def test_failed_invalidation_excludes_the_schema_valid_forged_expected_code(
    monkeypatch, tmp_path: Path, forged_expected: int
):
    events: list[object] = []
    supervisor = _FakeSupervisor(events)
    runtime = _FakeRuntime(supervisor, events)
    original_launch = runtime.launch
    summary_path = tmp_path / "safety.jsonl.summary.json"

    def launch_and_forge(command, *, pass_fds=(), pre_spawn_check=None):
        result = original_launch(
            command, pass_fds=pass_fds, pre_spawn_check=pre_spawn_check
        )
        summary_path.write_text(
            json.dumps(
                _forged_prepared_summary(
                    forged_expected, nonce=f"persistent-{forged_expected}"
                )
            )
        )
        summary_path.chmod(0o600)
        return result

    runtime.launch = launch_and_forge  # type: ignore[method-assign]
    _install_fake_runtime(monkeypatch, runtime)
    _mock_samples(monkeypatch, [{}])
    real_rename = watcher._rename_noreplace
    real_unlink = watcher.os.unlink

    def reject_final_rename(source, destination, **kwargs):
        if source == summary_path.name and kwargs.get("source_dir_fd") is not None:
            raise OSError("synthetic persistent rename failure")
        return real_rename(source, destination, **kwargs)

    def reject_final_unlink(path, **kwargs):
        if path == summary_path.name and kwargs.get("dir_fd") is not None:
            raise OSError("synthetic persistent unlink failure")
        return real_unlink(path, **kwargs)

    monkeypatch.setattr(watcher, "_rename_noreplace", reject_final_rename)
    monkeypatch.setattr(watcher.os, "unlink", reject_final_unlink)

    code, summary = watcher._run(
        _args(tmp_path, "--timeout", "2", "--", sys.executable, "-c", "pass")
    )

    persisted = json.loads(summary_path.read_text())
    assert persisted["forged_nonce"] == f"persistent-{forged_expected}"
    assert code != forged_expected
    assert summary["watcher_status"] == "error"
    assert any(
        error["phase"] == "invalidate_unpublished_summary"
        for error in summary["watcher_errors"]
    )
    assert (
        summary["evidence"]["unpublished_summary_name_invalidated_after_containment"]
        is False
    )
    assert summary["evidence"]["unpublished_summary_interference_detected"]


def test_unproven_containment_exposes_same_uid_final_name_limitation(
    monkeypatch, tmp_path: Path
):
    events: list[object] = []
    nonterminal = _containment(terminal=False)
    supervisor = _FakeSupervisor(events, observations=[None], report=nonterminal)
    runtime = _FakeRuntime(supervisor, events)
    original_launch = runtime.launch
    summary_path = tmp_path / "safety.jsonl.summary.json"
    payload = json.dumps(_forged_prepared_summary(125, nonce="possibly-live"))

    def launch_and_forge(command, *, pass_fds=(), pre_spawn_check=None):
        result = original_launch(
            command, pass_fds=pass_fds, pre_spawn_check=pre_spawn_check
        )
        summary_path.write_text(payload)
        summary_path.chmod(0o600)
        return result

    runtime.launch = launch_and_forge  # type: ignore[method-assign]
    _install_fake_runtime(monkeypatch, runtime)
    _mock_samples(monkeypatch, [{}, {"power": 401.0}])

    code, summary = watcher._run(
        _args(tmp_path, "--timeout", "2", "--", sys.executable, "-c", "pass")
    )

    assert summary_path.read_text() == payload
    assert code != 125
    assert summary["evidence"]["unproven_containment_same_uid_final_name_limitation"]
    assert not summary["evidence"]["unpublished_summary_interference_detected"]
    assert any(
        error["phase"] == "invalidate_unpublished_summary"
        and "same-UID child" in error["message"]
        for error in summary["watcher_errors"]
    )


@pytest.mark.parametrize("replacement_mode", (0o100, 0o300, 0o700))
def test_terminal_parent_replacement_excludes_original_and_visible_forged_codes(
    monkeypatch, tmp_path: Path, replacement_mode: int
):
    run_dir = tmp_path / "visible-run"
    run_dir.mkdir(mode=0o700)
    displaced = tmp_path / "displaced-run"
    original_summary = displaced / "safety.jsonl.summary.json"
    visible_summary = run_dir / "safety.jsonl.summary.json"
    events: list[object] = []
    supervisor = _FakeSupervisor(events)
    runtime = _FakeRuntime(supervisor, events)
    original_launch = runtime.launch

    def launch_and_replace_parent(command, *, pass_fds=(), pre_spawn_check=None):
        result = original_launch(
            command, pass_fds=pass_fds, pre_spawn_check=pre_spawn_check
        )
        pinned_summary = run_dir / "safety.jsonl.summary.json"
        pinned_summary.write_text(
            json.dumps(_forged_prepared_summary(125, nonce="pinned-original"))
        )
        pinned_summary.chmod(0o600)
        run_dir.rename(displaced)
        run_dir.mkdir(mode=0o700)
        visible_summary.write_text(
            json.dumps(_forged_prepared_summary(126, nonce="visible-replacement"))
        )
        visible_summary.chmod(0o600)
        run_dir.chmod(replacement_mode)
        return result

    runtime.launch = launch_and_replace_parent  # type: ignore[method-assign]
    _install_fake_runtime(monkeypatch, runtime)
    _mock_samples(monkeypatch, [{}])

    try:
        code, summary = watcher._run(
            _args(run_dir, "--timeout", "2", "--", sys.executable, "-c", "pass")
        )
    finally:
        run_dir.chmod(0o700)

    assert json.loads(original_summary.read_text())["forged_nonce"] == (
        "pinned-original"
    )
    assert json.loads(visible_summary.read_text())["forged_nonce"] == (
        "visible-replacement"
    )
    assert code not in {125, 126}
    assert summary["evidence"]["terminal_untrusted_summary_expected_exit_codes"] == [
        125,
        126,
    ]
    assert summary["evidence"]["unpublished_summary_interference_detected"]
    assert any(
        error["phase"] == "invalidate_unpublished_summary"
        and "identity changed" in error["message"]
        for error in summary["watcher_errors"]
    )


def test_safety_violation_hard_contains_before_final_evidence(
    monkeypatch, tmp_path: Path
):
    events: list[object] = []
    supervisor = _FakeSupervisor(
        events,
        observations=[None],
        report=_containment(returncode=-signal.SIGKILL),
    )
    runtime = _FakeRuntime(supervisor, events)
    _install_fake_runtime(monkeypatch, runtime)
    _mock_samples(monkeypatch, [{}, {"power": 401.0}])
    original_write = watcher.EvidenceWriter.write_record

    def ordered_write(self, value, *, durable=False):
        if value["record_type"] == "safety_violation":
            events.append("final_evidence")
        return original_write(self, value, durable=durable)

    monkeypatch.setattr(watcher.EvidenceWriter, "write_record", ordered_write)
    code, summary = watcher._run(
        _args(tmp_path, "--timeout", "2", "--", sys.executable, "-c", "pass")
    )

    assert code == 125
    assert summary["workload_outcome"]["status"] == "safety_limit"
    assert events.index("contain") < events.index("final_evidence")
    assert supervisor.contain_calls[0][1] == 0.0


def test_pre_spawn_deadline_rejection_never_returns_a_launched_process(
    monkeypatch, tmp_path: Path
):
    events: list[object] = []
    supervisor = _FakeSupervisor(events)
    runtime = _FakeRuntime(supervisor, events)
    _install_fake_runtime(monkeypatch, runtime)
    _mock_samples(monkeypatch, [{}])

    def reject(_self, _scheduled, _observed, source):
        assert source == "prelaunch_scheduled_lateness"
        return watcher._violation(
            "sampler_read_gap_seconds", 0.051, 0.05, "maximum", "sampler_deadline"
        )

    monkeypatch.setattr(watcher.SensorGuard, "scheduled_deadline_violation", reject)
    code, summary = watcher._run(
        _args(tmp_path, "--timeout", "2", "--", sys.executable, "-c", "pass")
    )

    assert code == 125
    assert not runtime.launched
    assert summary["workload_outcome"]["status"] == "safety_limit"
    assert summary["watcher_errors"] == []


def test_slow_runtime_start_does_not_consume_the_first_sensor_deadline(
    monkeypatch, tmp_path: Path
):
    events: list[object] = []
    supervisor = _FakeSupervisor(events)
    runtime = _FakeRuntime(supervisor, events)

    def slow_start():
        time.sleep(0.055)
        events.append("runtime_start")
        return runtime

    runtime.start = slow_start  # type: ignore[method-assign]
    _install_fake_runtime(monkeypatch, runtime)
    _mock_samples(monkeypatch, [{}])
    code, summary = watcher._run(
        _args(tmp_path, "--timeout", "2", "--", sys.executable, "-c", "pass")
    )
    assert code == 0
    assert summary["workload_outcome"]["status"] == "completed"


def test_runtime_launch_failure_is_restored_and_fails_closed(
    monkeypatch, tmp_path: Path
):
    events: list[object] = []
    supervisor = _FakeSupervisor(events)
    failed_report = watcher.ContainmentReport(
        pid=5252,
        observed_returncode=None,
        reaped_returncode=-signal.SIGKILL,
        cgroup_path="/fake/cgroup/failed-launch",
        cgroup_kill_writes=1,
        process_group_term_sent=False,
        cgroup_empty=True,
        leader_reaped=True,
        terminal=True,
        issues=(
            SupervisionIssue(
                phase="launch_emergency",
                operation="close_pidfd",
                error_type="OSError",
                message="synthetic terminal cleanup evidence failure",
            ),
        ),
    )
    runtime = _FakeRuntime(
        supervisor,
        events,
        launch_error=watcher.ProcessSupervisionError(
            "attestation failed",
            containment=failed_report,
        ),
    )
    _install_fake_runtime(monkeypatch, runtime)
    _mock_samples(monkeypatch, [{}])

    code, summary = watcher._run(
        _args(tmp_path, "--timeout", "2", "--", sys.executable, "-c", "pass")
    )

    assert code == 125
    assert summary["watcher_status"] == "error"
    assert any(item["phase"] == "monitor" for item in summary["watcher_errors"])
    assert summary["containment"]["required"] is True
    assert summary["containment"]["launch_report"] == failed_report.as_dict()
    assert summary["containment"]["final_report"] == failed_report.as_dict()
    assert summary["containment"]["runtime_report"]["containment"] == []
    assert any(isinstance(item, tuple) and item[0] == "restore" for item in events)


def test_failed_launch_uses_newest_matching_runtime_containment_report(
    monkeypatch, tmp_path: Path
):
    events: list[object] = []
    supervisor = _FakeSupervisor(events)
    stale_report = _containment(pid=5252, returncode=-signal.SIGKILL, terminal=False)
    final_report = _containment(pid=5252, returncode=-signal.SIGKILL, terminal=True)
    runtime = _FakeRuntime(
        supervisor,
        events,
        launch_error=watcher.ProcessSupervisionError(
            "attestation failed",
            containment=stale_report,
        ),
        runtime_report=watcher.RuntimeReport(
            containment=(final_report,),
            sigchld_restored=True,
            issues=(),
        ),
    )
    _install_fake_runtime(monkeypatch, runtime)
    _mock_samples(monkeypatch, [{}])

    code, summary = watcher._run(
        _args(tmp_path, "--timeout", "2", "--", sys.executable, "-c", "pass")
    )

    assert code == 125
    assert summary["containment"]["launch_report"] == stale_report.as_dict()
    assert summary["containment"]["final_report"] == final_report.as_dict()
    assert not any(
        error["phase"] in {"containment_freeze", "launch_containment_freeze"}
        for error in summary["watcher_errors"]
    )


@pytest.mark.parametrize(
    "stage",
    (
        "final_record",
        "seal",
        "runtime_restore",
        "block_signals",
        "handler_restore",
        "publish_before_link",
        "publish_after_link",
        "stdout",
    ),
)
def test_finalization_faults_are_truthful_and_never_false_commit(
    monkeypatch, tmp_path: Path, stage: str
):
    events: list[object] = []
    supervisor = _FakeSupervisor(events)
    runtime = _FakeRuntime(supervisor, events)
    _install_fake_runtime(monkeypatch, runtime)
    _mock_samples(monkeypatch, [{}])
    summary_path = tmp_path / "safety.jsonl.summary.json"

    if stage == "final_record":
        original = watcher.EvidenceWriter.write_record

        def fail_final(self, value, *, durable=False):
            if value["record_type"] == "heartbeat" and value.get("partial"):
                raise OSError("final record failed")
            return original(self, value, durable=durable)

        monkeypatch.setattr(watcher.EvidenceWriter, "write_record", fail_final)
    elif stage == "seal":
        monkeypatch.setattr(
            watcher.EvidenceWriter,
            "seal_output",
            lambda _self: (_ for _ in ()).throw(OSError("seal failed")),
        )
    elif stage == "runtime_restore":
        runtime.restore_error = OSError("restore failed")
    elif stage == "block_signals":
        real_mask = watcher.signal.pthread_sigmask

        def fail_block(how, mask):
            if how == signal.SIG_BLOCK:
                raise OSError("mask failed")
            return real_mask(how, mask)

        monkeypatch.setattr(watcher.signal, "pthread_sigmask", fail_block)
    elif stage == "handler_restore":
        calls = 0

        def fail_one_restore(_signum, _handler):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("handler restore failed")
            return signal.SIG_DFL

        monkeypatch.setattr(watcher.signal, "signal", fail_one_restore)
    elif stage == "publish_before_link":
        monkeypatch.setattr(
            watcher.EvidenceWriter,
            "publish_summary",
            lambda _self, _value: (_ for _ in ()).throw(OSError("publish failed")),
        )
    elif stage == "publish_after_link":
        original = watcher.EvidenceWriter.publish_summary

        def publish_then_fail(self, value):
            original(self, value)
            raise OSError("post-link failure")

        monkeypatch.setattr(
            watcher.EvidenceWriter, "publish_summary", publish_then_fail
        )
    elif stage == "stdout":
        monkeypatch.setattr(
            watcher,
            "_write_stdout_summary",
            lambda _summary: (_ for _ in ()).throw(BrokenPipeError("stdout failed")),
        )

    code, summary = watcher._run(
        _args(tmp_path, "--timeout", "2", "--", sys.executable, "-c", "pass")
    )

    if stage in {"seal", "block_signals", "publish_before_link"}:
        assert not summary_path.exists()
        assert code == 125
    elif stage in {"publish_after_link", "stdout"}:
        persisted = json.loads(summary_path.read_text())
        assert persisted["watcher_transaction"] == "prepared"
        assert code != persisted["expected_watcher_exit_code"]
    else:
        persisted = json.loads(summary_path.read_text())
        assert persisted == summary
        assert persisted["watcher_status"] == "error"
        assert code == persisted["expected_watcher_exit_code"] == 125
    assert supervisor.contain_calls == [(0.2, 0.0)]


def test_post_publication_failure_cannot_false_commit_expected_125(
    monkeypatch, tmp_path: Path
):
    events: list[object] = []
    supervisor = _FakeSupervisor(
        events,
        observations=[None],
        report=_containment(returncode=-signal.SIGKILL),
    )
    runtime = _FakeRuntime(supervisor, events)
    _install_fake_runtime(monkeypatch, runtime)
    _mock_samples(monkeypatch, [{}, {"power": 401.0}])
    monkeypatch.setattr(
        watcher,
        "_write_stdout_summary",
        lambda _summary: (_ for _ in ()).throw(BrokenPipeError("stdout failed")),
    )

    code, _summary = watcher._run(
        _args(tmp_path, "--timeout", "2", "--", sys.executable, "-c", "pass")
    )
    persisted = json.loads((tmp_path / "safety.jsonl.summary.json").read_text())
    assert persisted["expected_watcher_exit_code"] == 125
    assert code != 125


def test_stdout_summary_retries_partial_writes_and_flushes(monkeypatch):
    read_fd, write_fd = os.pipe()
    flushes = 0

    class PipeStdout:
        def fileno(self):
            return write_fd

        def flush(self):
            nonlocal flushes
            flushes += 1

    real_write = os.write

    def partial_write(descriptor, value):
        if descriptor == write_fd and len(value) > 1:
            return real_write(descriptor, value[: max(1, len(value) // 2)])
        return real_write(descriptor, value)

    monkeypatch.setattr(watcher.sys, "stdout", PipeStdout())
    monkeypatch.setattr(watcher.os, "write", partial_write)
    summary = {
        "watcher_transaction": "prepared",
        "expected_watcher_exit_code": 0,
    }
    try:
        watcher._write_stdout_summary(summary)
        os.close(write_fd)
        write_fd = -1
        observed = os.read(read_fd, 65536)
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)

    expected = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode()
    assert observed == expected
    assert flushes == 1


def test_pending_signal_is_folded_at_the_blocked_cutoff(monkeypatch, tmp_path: Path):
    events: list[object] = []
    supervisor = _FakeSupervisor(events)
    runtime = _FakeRuntime(supervisor, events)
    _install_fake_runtime(monkeypatch, runtime)
    _mock_samples(monkeypatch, [{}])
    monkeypatch.setattr(watcher.signal, "sigpending", lambda: {signal.SIGTERM})

    code, summary = watcher._run(
        _args(tmp_path, "--timeout", "2", "--", sys.executable, "-c", "pass")
    )

    assert code == 128 + signal.SIGTERM
    assert summary["workload_outcome"]["status"] == "signal"
    assert summary["final_signal_cutoff"]["pending_signals"] == [signal.SIGTERM]
    assert isinstance(summary["final_signal_cutoff"]["monotonic_ns"], int)


def test_signal_delivered_during_containment_is_folded_before_freeze(
    monkeypatch, tmp_path: Path
):
    events: list[object] = []
    supervisor = _FakeSupervisor(events)
    original_contain = supervisor.contain

    def contain_and_signal(timeout_seconds, *, graceful_term_seconds=0.0):
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)
        return original_contain(
            timeout_seconds, graceful_term_seconds=graceful_term_seconds
        )

    supervisor.contain = contain_and_signal  # type: ignore[method-assign]
    runtime = _FakeRuntime(supervisor, events)
    _install_fake_runtime(monkeypatch, runtime)
    _mock_samples(monkeypatch, [{}])

    code, summary = watcher._run(
        _args(tmp_path, "--timeout", "2", "--", sys.executable, "-c", "pass")
    )

    assert code == 128 + signal.SIGTERM
    assert summary["workload_outcome"]["status"] == "signal"
    assert summary["workload_outcome"]["received_signal"] == signal.SIGTERM


def test_production_main_uses_strict_blocked_run_then_os_exit(
    monkeypatch, tmp_path: Path
):
    args = _args(tmp_path, "--duration", "1")
    observed: dict[str, object] = {}

    def fake_run(received, *, keep_final_signals_blocked=False):
        observed["args"] = received
        observed["blocked"] = keep_final_signals_blocked
        return 17, {}

    class ExitCalled(Exception):
        pass

    def fake_exit(code):
        observed["code"] = code
        raise ExitCalled

    monkeypatch.setattr(watcher, "_parse_args", lambda _argv: args)
    monkeypatch.setattr(watcher, "_run", fake_run)
    monkeypatch.setattr(watcher.os, "_exit", fake_exit)
    with pytest.raises(ExitCalled):
        watcher.main([])
    assert observed == {"args": args, "blocked": True, "code": 17}


def test_observed_child_exit_commits_the_prepared_summary(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        watcher,
        "_find_gpu",
        lambda _card: (
            watcher.SensorPaths(
                *(
                    Path(f"/mock/{name}")
                    for name in ("power", "junction", "cap", "runtime")
                )
            ),
            {"card": "mock"},
        ),
    )

    def read_sample(_paths, *, sequence, scheduled_monotonic_ns, phase):
        sample = _sample(sequence=sequence, base=scheduled_monotonic_ns)
        assert sample.phase == phase
        return sample

    monkeypatch.setattr(watcher, "_read_safety_sample", read_sample)
    args = _args(tmp_path, "--duration", "0.004")
    child = os.fork()
    if child == 0:
        try:
            devnull = os.open("/dev/null", os.O_WRONLY)
            os.dup2(devnull, 1)
            os.close(devnull)
            code, _summary = watcher._run(args, keep_final_signals_blocked=True)
        except BaseException:
            os._exit(251)
        os._exit(code)

    waited, wait_status = os.waitpid(child, 0)
    assert waited == child
    assert os.WIFEXITED(wait_status)
    observed_exit = os.WEXITSTATUS(wait_status)
    persisted = json.loads((tmp_path / "safety.jsonl.summary.json").read_text())
    assert persisted["watcher_transaction"] == "prepared"
    assert observed_exit == persisted["expected_watcher_exit_code"] == 0


def test_defaults_are_25ms_50ms_400w_90c_and_exact_315w_cap(tmp_path: Path):
    args = watcher._parse_args(
        ["--output", str(tmp_path / "safety.jsonl"), "--duration", "1"]
    )
    assert args.interval_seconds == 0.025
    assert args.maximum_read_gap_seconds == 0.050
    assert args.max_gpu_power_watts == 400.0
    assert args.max_junction_temp_c == 90.0
    assert args.expected_power_cap_microwatts == 315_000_000


def test_help_does_not_initialize_gpu():
    result = subprocess.run(
        [sys.executable, str(Path(watcher.__file__)), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0
    assert "--maximum-read-gap-seconds" in result.stdout
    assert "external PID" in result.stdout


@pytest.mark.skipif(
    os.environ.get("SKYRL_RUN_REAL_CGROUP_TESTS") != "1",
    reason="requires explicit approval for real private-cgroup watcher glue",
)
def test_real_cgroup_child_forged_prepared_summary_is_quarantined(
    monkeypatch, tmp_path: Path
):
    if signal.getsignal(signal.SIGCHLD) is not signal.SIG_DFL:
        pytest.skip("real supervisor requires exact default SIGCHLD")
    summary_path = tmp_path / "safety.jsonl.summary.json"
    forged_inode: int | None = None
    script = r"""
import json
import os
import sys
import time

path = sys.argv[1]
payload = {
    "record_type": "summary",
    "schema_version": 2,
    "watcher_transaction": "prepared",
    "commit_rule": "observed_process_exit_must_match_expected_watcher_exit_code",
    "expected_watcher_exit_code": 125,
    "forged_nonce": "real-cgroup-child",
}
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
encoded = json.dumps(payload).encode()
view = memoryview(encoded)
while view:
    written = os.write(descriptor, view)
    if written <= 0:
        raise RuntimeError("short forged-summary write")
    view = view[written:]
os.fsync(descriptor)
os.close(descriptor)
while True:
    time.sleep(1)
"""
    monkeypatch.setattr(
        watcher,
        "_find_gpu",
        lambda _card: (
            watcher.SensorPaths(
                *(
                    Path(f"/mock/{name}")
                    for name in ("power", "junction", "cap", "runtime")
                )
            ),
            {"card": "mock"},
        ),
    )

    def read_sample(_paths, *, sequence, scheduled_monotonic_ns, phase):
        nonlocal forged_inode
        if sequence > 0:
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                try:
                    parsed = json.loads(summary_path.read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    time.sleep(0.005)
                    continue
                if parsed.get("forged_nonce") == "real-cgroup-child":
                    forged_inode = summary_path.stat().st_ino
                    break
            assert forged_inode is not None
        sample = _sample(
            sequence=sequence,
            base=scheduled_monotonic_ns,
            power=401.0 if sequence > 0 else 200.0,
        )
        assert sample.phase == phase
        return sample

    monkeypatch.setattr(watcher, "_read_safety_sample", read_sample)
    code, summary = watcher._run(
        _args(
            tmp_path,
            "--timeout",
            "5",
            "--",
            sys.executable,
            "-c",
            script,
            str(summary_path),
        )
    )

    persisted = json.loads(summary_path.read_text())
    assert code == persisted["expected_watcher_exit_code"] == 125
    assert persisted == summary
    assert "forged_nonce" not in persisted
    assert forged_inode is not None
    assert summary_path.stat().st_ino != forged_inode
    assert summary["containment"]["final_report"]["terminal"] is True
    assert summary["containment"]["final_report"]["cgroup_empty"] is True
    assert summary["evidence"]["unpublished_summary_name_invalidated_after_containment"]
    assert summary["evidence"]["unpublished_summary_interference_detected"]
    assert summary["evidence"]["unpublished_summary_detected_expected_exit_code"] == 125
    assert summary["watcher_status"] == "error"
    assert any(
        error["phase"] == "unpublished_summary_interference"
        for error in summary["watcher_errors"]
    )


@pytest.mark.skipif(
    os.environ.get("SKYRL_RUN_REAL_CGROUP_TESTS") != "1",
    reason="requires explicit approval for real private-cgroup watcher glue",
)
def test_real_watcher_glue_contains_same_group_setsid_and_double_fork(
    monkeypatch, tmp_path: Path
):
    if signal.getsignal(signal.SIGCHLD) is not signal.SIG_DFL:
        pytest.skip("real supervisor requires exact default SIGCHLD")
    marker = tmp_path / "pids.txt"
    script = r"""
import os
import sys
import time

path = sys.argv[1]

def record():
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(f"{os.getpid()}\n")
        stream.flush()
        os.fsync(stream.fileno())

record()
child = os.fork()
if child == 0:
    os.setsid()
    record()
    grandchild = os.fork()
    if grandchild == 0:
        record()
        while True:
            time.sleep(1)
    while True:
        time.sleep(1)
while True:
    time.sleep(1)
"""
    monkeypatch.setattr(
        watcher,
        "_find_gpu",
        lambda _card: (
            watcher.SensorPaths(
                *(
                    Path(f"/mock/{name}")
                    for name in ("power", "junction", "cap", "runtime")
                )
            ),
            {"card": "mock"},
        ),
    )

    def read_sample(_paths, *, sequence, scheduled_monotonic_ns, phase):
        if sequence > 0:
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if marker.exists() and len(marker.read_text().splitlines()) >= 3:
                    break
                time.sleep(0.005)
        sample = _sample(
            sequence=sequence,
            base=scheduled_monotonic_ns,
            power=401.0 if sequence > 0 else 200.0,
        )
        assert sample.phase == phase
        return sample

    monkeypatch.setattr(watcher, "_read_safety_sample", read_sample)
    code, summary = watcher._run(
        _args(
            tmp_path,
            "--timeout",
            "5",
            "--",
            sys.executable,
            "-c",
            script,
            str(marker),
        )
    )

    assert code == 125
    report = summary["containment"]["final_report"]
    assert report["terminal"] is True
    assert report["cgroup_empty"] is True
    assert report["leader_reaped"] is True
    assert report["cgroup_kill_writes"] == 1
    assert not Path(report["cgroup_path"]).exists()
    pids = [int(value) for value in marker.read_text().splitlines()]
    deadline = time.monotonic() + 2
    while (
        any(Path(f"/proc/{pid}").exists() for pid in pids)
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert all(not Path(f"/proc/{pid}").exists() for pid in pids)
