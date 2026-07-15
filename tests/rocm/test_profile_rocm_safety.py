from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rocm import profile_rocm

_REPO = Path(__file__).parents[2]
_PROFILER = _REPO / "rocm" / "profile_rocm.py"


def test_gpu_power_limit_is_inclusive_and_reports_breach():
    limits = {
        "max_junction_temp_c": 90.0,
        "max_gpu_power_watts": 315.0,
    }

    assert profile_rocm._safety_violation({"gpu_junction_temp_c": 70.0, "gpu_power_watts": 315.0}, limits) is None
    assert profile_rocm._safety_violation({"gpu_junction_temp_c": 70.0, "gpu_power_watts": 315.25}, limits) == {
        "metric": "gpu_power_watts",
        "value": 315.25,
        "limit": 315.0,
        "limit_kind": "maximum",
    }


def test_temperature_has_priority_when_temperature_and_power_both_breach():
    assert profile_rocm._safety_violation(
        {"gpu_junction_temp_c": 91.0, "gpu_power_watts": 316.0},
        {"max_junction_temp_c": 90.0, "max_gpu_power_watts": 315.0},
    ) == {
        "metric": "gpu_junction_temp_c",
        "value": 91.0,
        "limit": 90.0,
        "limit_kind": "maximum",
    }


def test_required_power_sensor_fails_closed_after_grace_period():
    violation = profile_rocm._safety_violation(
        {"gpu_junction_temp_c": 70.0},
        {"max_gpu_power_watts": 315.0},
        require_available=True,
    )

    assert violation == {
        "metric": "gpu_power_watts",
        "value": None,
        "limit": 315.0,
        "limit_kind": "maximum",
        "unavailable": True,
    }


@pytest.mark.parametrize(
    ("junction", "power", "max_junction", "max_power", "expected_metric"),
    (
        (None, 200.0, 90.0, 315.0, "gpu_junction_temp_c"),
        (50.0, None, 90.0, 315.0, "gpu_power_watts"),
        (None, None, 90.0, 315.0, "gpu_junction_temp_c"),
        (90.0, 315.0, 90.0, 315.0, None),
        (None, None, None, None, None),
    ),
)
def test_completion_requires_each_configured_gpu_sensor(junction, power, max_junction, max_power, expected_metric):
    violation = profile_rocm._unobserved_required_sensor_violation(
        [
            {
                "phase": "measured",
                "gpu_junction_temp_c": junction,
                "gpu_power_watts": power,
            }
        ],
        {
            "max_junction_temp_c": max_junction,
            "max_gpu_power_watts": max_power,
        },
    )

    if expected_metric is None:
        assert violation is None
    else:
        assert violation is not None
        assert violation["metric"] == expected_metric
        assert violation["unavailable"] is True


def test_only_measured_sensor_readings_satisfy_completion_postcondition():
    limits = {"max_junction_temp_c": 90.0, "max_gpu_power_watts": 315.0}
    present = {"gpu_junction_temp_c": 50.0, "gpu_power_watts": 200.0}
    missing = {"gpu_junction_temp_c": None, "gpu_power_watts": None}

    baseline_only = [
        {"phase": "baseline", **present},
        {"phase": "measured", **missing},
    ]
    assert profile_rocm._unobserved_required_sensor_violation(baseline_only, limits)["metric"] == "gpu_junction_temp_c"

    observed_during_measurement = [
        {"phase": "baseline", **missing},
        {"phase": "measured", **present},
        {"phase": "measured", **missing},
    ]
    assert profile_rocm._unobserved_required_sensor_violation(observed_during_measurement, limits) is None


def test_help_exposes_gpu_power_guard_without_initializing_hardware():
    result = subprocess.run(
        [sys.executable, str(_PROFILER), "--help"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0
    assert "--max-gpu-power-watts" in result.stdout


def _run_mocked_profiler(
    monkeypatch,
    tmp_path: Path,
    *,
    power_for_phase,
    junction_for_phase=None,
    max_power: float | None = 315.0,
    max_junction: float | None = None,
    sensor_grace_seconds: float = 60.0,
    command: str | None = "import time; time.sleep(5)",
    duration: float | None = None,
    timeout_seconds: float | None = None,
    include_pid: tuple[str, int] | None = None,
    terminate_included_on_safety: bool = False,
    terminate_included_on_abort: bool = False,
    signal_on_measured: int | None = None,
    sample_error: BaseException | None = None,
    kernel_errors_for_call=None,
    signal_on_kernel_check_call: tuple[int, int] | None = None,
    explicit_termination=None,
    target_liveness=None,
    before_pre_spawn_check=None,
    signal_before_pre_spawn: int | None = None,
    incomplete_containment: bool = False,
    launch_attestation_failure: bool = False,
    inherited_fds: tuple[int, ...] = (),
):
    output = tmp_path / "telemetry.jsonl"
    termination = {"called": False, "was_alive": False}
    signal_handlers = {}
    signal_delivered = False

    class FakeWrappedProcessSupervisor:
        def __init__(self, command_values):
            self.pid = os.getpid() + 10_000
            source = command_values[-1]
            if "SystemExit(7)" in source:
                self._observed_returncode = 7
            elif source.strip() == "pass":
                self._observed_returncode = 0
            else:
                self._observed_returncode = None
            self._report = None

        def observe_returncode(self):
            return self._observed_returncode

        def contain(self, timeout_seconds):
            assert timeout_seconds == 0.2
            if self._report is None:
                termination["called"] = True
                termination["was_alive"] = self._observed_returncode is None
                reaped_returncode = -9 if self._observed_returncode is None else self._observed_returncode
                self._report = profile_rocm.ContainmentReport(
                    pid=self.pid,
                    observed_returncode=self._observed_returncode,
                    reaped_returncode=reaped_returncode,
                    cgroup_path="/mock/skyrl-profile",
                    cgroup_kill_writes=1,
                    process_group_term_sent=False,
                    cgroup_empty=True,
                    leader_reaped=True,
                    terminal=not incomplete_containment,
                    issues=(),
                )
            return self._report

    class FakeChildProcessRuntime:
        def __init__(self, *, cgroup_prefix):
            assert cgroup_prefix == "skyrl-profile"
            self.supervisors = []

        def start(self):
            return self

        def launch(self, command_values, *, pass_fds, pre_spawn_check):
            assert command_values == [sys.executable, "-c", command]
            assert pass_fds == inherited_fds
            if before_pre_spawn_check is not None:
                before_pre_spawn_check()
            if signal_before_pre_spawn is not None:
                signal_handlers[signal_before_pre_spawn](signal_before_pre_spawn, None)
            try:
                pre_spawn_check()
            except BaseException as error:
                empty_scope_report = profile_rocm.ContainmentReport(
                    pid=-1,
                    observed_returncode=None,
                    reaped_returncode=None,
                    cgroup_path="/mock/skyrl-profile",
                    cgroup_kill_writes=1,
                    process_group_term_sent=False,
                    cgroup_empty=True,
                    leader_reaped=True,
                    terminal=True,
                    issues=(),
                )
                raise profile_rocm.ProcessSupervisionError(
                    "wrapped process launch attestation failed",
                    containment=empty_scope_report,
                ) from error
            if launch_attestation_failure:
                failed_report = profile_rocm.ContainmentReport(
                    pid=len(self.supervisors) + os.getpid() + 20_000,
                    observed_returncode=None,
                    reaped_returncode=-9,
                    cgroup_path="/mock/skyrl-profile-failed-launch",
                    cgroup_kill_writes=1,
                    process_group_term_sent=False,
                    cgroup_empty=True,
                    leader_reaped=True,
                    terminal=True,
                    issues=(
                        profile_rocm.SupervisionIssue(
                            phase="launch_emergency",
                            operation="close_pidfd",
                            error_type="OSError",
                            message="synthetic terminal cleanup evidence failure",
                        ),
                    ),
                )
                raise profile_rocm.ProcessSupervisionError(
                    "synthetic launch attestation failure",
                    containment=failed_report,
                )
            supervisor = FakeWrappedProcessSupervisor(command_values)
            self.supervisors.append(supervisor)
            return supervisor

        def restore(self, *, timeout_seconds):
            reports = tuple(supervisor.contain(timeout_seconds) for supervisor in self.supervisors)
            return profile_rocm.RuntimeReport(
                containment=reports,
                sigchld_restored=True,
                issues=(),
            )

    def sample(_device, _hwmon, _targets, _create_times, start, phase):
        nonlocal signal_delivered
        if phase == "measured" and sample_error is not None:
            raise sample_error
        if phase == "measured" and signal_on_measured is not None and not signal_delivered:
            signal_delivered = True
            signal_handlers[signal_on_measured](signal_on_measured, None)
            assert termination["called"] is False
        return {
            "record_type": "sample",
            "phase": phase,
            "elapsed_seconds": time.monotonic() - start,
            "gpu_power_watts": power_for_phase(phase),
            "gpu_junction_temp_c": (50.0 if junction_for_phase is None else junction_for_phase(phase)),
            "vram_used_bytes": 0,
            "gtt_used_bytes": 0,
            "host_memory_used_bytes": 0,
            "host_memory_available_bytes": 64 * 1024**3,
            "host_swap_used_bytes": 0,
            "processes": {},
        }

    monkeypatch.setattr(
        profile_rocm,
        "_find_gpu",
        lambda _card: (Path("/mock/device"), None, {"card": "mock"}),
    )
    monkeypatch.setattr(profile_rocm, "_sample", sample)
    kernel_check_count = 0

    def kernel_driver_errors_since(_start):
        nonlocal kernel_check_count
        kernel_check_count += 1
        if (
            signal_on_kernel_check_call is not None
            and kernel_check_count == signal_on_kernel_check_call[0]
        ):
            signum = signal_on_kernel_check_call[1]
            signal_handlers[signum](signum, None)
        if kernel_errors_for_call is None:
            return []
        return kernel_errors_for_call(kernel_check_count)

    monkeypatch.setattr(
        profile_rocm,
        "_kernel_driver_errors_since",
        kernel_driver_errors_since,
    )
    monkeypatch.setattr(profile_rocm, "ChildProcessRuntime", FakeChildProcessRuntime)
    if target_liveness is not None:
        monkeypatch.setattr(
            profile_rocm,
            "_pid_matches_create_time",
            lambda _pid, _create_time: target_liveness(),
        )
    if explicit_termination is not None:

        def terminate_process_trees(pids, _grace_seconds, _expected_create_times):
            explicit_termination["calls"] = explicit_termination.get("calls", 0) + 1
            explicit_termination["pids"] = list(pids)
            return sorted(pids), []

        monkeypatch.setattr(
            profile_rocm,
            "_terminate_process_trees",
            terminate_process_trees,
        )
    monkeypatch.setattr(
        profile_rocm.signal,
        "signal",
        lambda signum, handler: signal_handlers.__setitem__(signum, handler),
    )

    arguments = [
        str(_PROFILER),
        "--output",
        str(output),
        "--interval",
        "0.01",
        "--terminate-grace-seconds",
        "0.2",
        "--sensor-grace-seconds",
        str(sensor_grace_seconds),
    ]
    if max_junction is not None:
        arguments.extend(("--max-junction-temp-c", str(max_junction)))
    if max_power is not None:
        arguments.extend(("--max-gpu-power-watts", str(max_power)))
    if include_pid is not None:
        label, pid = include_pid
        arguments.extend(("--include-pid", f"{label}={pid}"))
    if terminate_included_on_safety:
        arguments.append("--terminate-included-on-safety")
    if terminate_included_on_abort:
        arguments.append("--terminate-included-on-abort")
    if duration is not None:
        arguments.extend(("--duration", str(duration)))
    if timeout_seconds is not None:
        arguments.extend(("--timeout", str(timeout_seconds)))
    for descriptor in inherited_fds:
        arguments.extend(("--pass-fd", str(descriptor)))
    if command is not None:
        arguments.extend(("--", sys.executable, "-c", command))
    monkeypatch.setattr(sys, "argv", arguments)

    returncode = profile_rocm.main()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    summary = json.loads(output.with_suffix(output.suffix + ".summary.json").read_text())
    return returncode, records, summary, termination


@pytest.mark.parametrize(
    "status",
    (
        "safety_limit",
        "driver_error",
        "timeout",
        "signal",
        "command_failed",
        "error",
    ),
)
def test_abort_included_termination_policy_covers_every_abort_status(status):
    assert profile_rocm._included_termination_required(
        status,
        terminate_on_safety=False,
        terminate_on_abort=True,
    )


@pytest.mark.parametrize("status", ("running", "completed", "targets_exited"))
def test_abort_included_termination_policy_preserves_non_abort_statuses(status):
    assert not profile_rocm._included_termination_required(
        status,
        terminate_on_safety=False,
        terminate_on_abort=True,
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    (
        ("safety_limit", True),
        ("driver_error", True),
        ("timeout", False),
        ("signal", False),
        ("command_failed", False),
        ("error", False),
        ("completed", False),
    ),
)
def test_safety_only_included_termination_policy_is_backward_compatible(status, expected):
    assert (
        profile_rocm._included_termination_required(
            status,
            terminate_on_safety=True,
            terminate_on_abort=False,
        )
        is expected
    )


@pytest.mark.parametrize(
    "stage",
    (
        "gpu_discovery",
        "output_directory",
        "manifest",
        "output_open",
        "manifest_write",
        "manifest_flush",
    ),
)
def test_abort_option_terminates_included_target_for_every_setup_failure(
    monkeypatch,
    tmp_path,
    stage,
):
    output = tmp_path / f"{stage}-output" / "telemetry.jsonl"
    termination = {}

    def terminate_process_trees(pids, grace_seconds, expected_create_times):
        termination["calls"] = termination.get("calls", 0) + 1
        termination["pids"] = list(pids)
        termination["grace_seconds"] = grace_seconds
        termination["expected_create_times"] = dict(expected_create_times)
        return sorted(pids), []

    monkeypatch.setattr(
        profile_rocm,
        "_terminate_process_trees",
        terminate_process_trees,
    )
    monkeypatch.setattr(profile_rocm.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        profile_rocm,
        "_find_gpu",
        lambda _card: (Path("/mock/device"), None, {"card": "mock"}),
    )

    if stage == "gpu_discovery":
        monkeypatch.setattr(
            profile_rocm,
            "_find_gpu",
            lambda _card: (_ for _ in ()).throw(RuntimeError("setup failure: gpu discovery")),
        )
    elif stage == "output_directory":
        original_mkdir = profile_rocm.Path.mkdir

        def failing_mkdir(path, *args, **kwargs):
            if path == output.parent:
                raise OSError("setup failure: output directory")
            return original_mkdir(path, *args, **kwargs)

        monkeypatch.setattr(profile_rocm.Path, "mkdir", failing_mkdir)
    elif stage == "manifest":
        monkeypatch.setattr(
            profile_rocm,
            "_process_manifest",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("setup failure: manifest")),
        )
    elif stage == "output_open":
        original_open = profile_rocm.os.open

        def failing_open(path, *args, **kwargs):
            if Path(path) == output:
                raise OSError("setup failure: output open")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(profile_rocm.os, "open", failing_open)
    else:
        original_fdopen = profile_rocm.os.fdopen
        close_fd = profile_rocm.os.close

        class FailingManifestOutput:
            def __init__(self, descriptor):
                self.descriptor = descriptor
                self.closed = False

            def write(self, value):
                if stage == "manifest_write":
                    raise OSError("setup failure: manifest write")
                return len(value)

            def flush(self):
                if stage == "manifest_flush":
                    raise OSError("setup failure: manifest flush")

            def close(self):
                if not self.closed:
                    close_fd(self.descriptor)
                    self.closed = True

        def failing_fdopen(descriptor, *_args, **_kwargs):
            if stage in {"manifest_write", "manifest_flush"}:
                return FailingManifestOutput(descriptor)
            return original_fdopen(descriptor, *_args, **_kwargs)

        monkeypatch.setattr(profile_rocm.os, "fdopen", failing_fdopen)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_PROFILER),
            "--output",
            str(output),
            "--duration",
            "1",
            "--terminate-grace-seconds",
            "0.2",
            "--include-pid",
            f"server={os.getpid()}",
            "--terminate-included-on-abort",
        ],
    )

    with pytest.raises(Exception, match="setup failure"):
        profile_rocm.main()

    assert termination["calls"] == 1
    assert termination["pids"] == [os.getpid()]
    assert termination["grace_seconds"] == 0.2
    assert set(termination["expected_create_times"]) == {os.getpid()}
    assert not output.with_suffix(output.suffix + ".summary.json").exists()


def test_invalid_cli_does_not_terminate_included_target_before_pid_acceptance(monkeypatch, tmp_path):
    output = tmp_path / "invalid-cli.jsonl"
    monkeypatch.setattr(
        profile_rocm,
        "_terminate_process_trees",
        lambda *_args, **_kwargs: pytest.fail("CLI rejection must not terminate included targets"),
    )
    monkeypatch.setattr(
        profile_rocm,
        "_find_gpu",
        lambda _card: pytest.fail("CLI rejection must precede GPU discovery"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_PROFILER),
            "--output",
            str(output),
            "--duration",
            "1",
            "--interval",
            "0",
            "--include-pid",
            f"server={os.getpid()}",
            "--terminate-included-on-abort",
        ],
    )

    with pytest.raises(SystemExit) as raised:
        profile_rocm.main()

    assert raised.value.code == 2
    assert not output.exists()


def test_runtime_exception_terminates_included_target_exactly_once(monkeypatch, tmp_path):
    explicit_termination = {}

    with pytest.raises(RuntimeError, match="runtime sample failure"):
        _run_mocked_profiler(
            monkeypatch,
            tmp_path,
            power_for_phase=lambda _phase: 100.0,
            include_pid=("server", os.getpid()),
            terminate_included_on_abort=True,
            sample_error=RuntimeError("runtime sample failure"),
            explicit_termination=explicit_termination,
        )

    summary_path = tmp_path / "telemetry.jsonl.summary.json"
    summary = json.loads(summary_path.read_text())
    assert explicit_termination == {"calls": 1, "pids": [os.getpid()]}
    assert summary["status"] == "error"
    assert summary["error_type"] == "RuntimeError"
    assert summary["terminated_explicit_pids"] == [os.getpid()]
    assert summary["surviving_explicit_pids"] == []


@pytest.mark.parametrize(
    (
        "command",
        "timeout_seconds",
        "signal_on_measured",
        "expected_status",
        "expected_returncode",
    ),
    (
        ("raise SystemExit(7)", None, None, "command_failed", 7),
        ("import time; time.sleep(5)", 0.03, None, "timeout", 124),
        (
            "import time; time.sleep(5)",
            None,
            profile_rocm.signal.SIGTERM,
            "signal",
            143,
        ),
    ),
)
def test_abort_option_terminates_included_target_for_command_abort_statuses(
    monkeypatch,
    tmp_path,
    command,
    timeout_seconds,
    signal_on_measured,
    expected_status,
    expected_returncode,
):
    explicit_termination = {}
    returncode, records, summary, _termination = _run_mocked_profiler(
        monkeypatch,
        tmp_path,
        power_for_phase=lambda _phase: 100.0,
        command=command,
        timeout_seconds=timeout_seconds,
        include_pid=("server", os.getpid()),
        terminate_included_on_abort=True,
        signal_on_measured=signal_on_measured,
        explicit_termination=explicit_termination,
    )

    assert records[0]["terminate_included_on_abort"] is True
    assert records[0]["terminate_included_on_safety"] is False
    assert returncode == expected_returncode
    assert summary["status"] == expected_status
    assert explicit_termination == {"calls": 1, "pids": [os.getpid()]}
    assert summary["terminated_explicit_pids"] == [os.getpid()]
    assert summary["surviving_explicit_pids"] == []


def test_abort_option_terminates_included_target_for_safety_limit(monkeypatch, tmp_path):
    explicit_termination = {}
    returncode, _records, summary, _termination = _run_mocked_profiler(
        monkeypatch,
        tmp_path,
        power_for_phase=lambda phase: 315.0 if phase == "preflight" else 315.25,
        include_pid=("server", os.getpid()),
        terminate_included_on_abort=True,
        explicit_termination=explicit_termination,
    )

    assert returncode == 125
    assert summary["status"] == "safety_limit"
    assert explicit_termination == {"calls": 1, "pids": [os.getpid()]}


def test_signal_during_final_journal_query_matches_summary_and_exit(monkeypatch, tmp_path):
    returncode, _records, summary, termination = _run_mocked_profiler(
        monkeypatch,
        tmp_path,
        power_for_phase=lambda _phase: 100.0,
        command="pass",
        signal_on_kernel_check_call=(2, profile_rocm.signal.SIGTERM),
    )

    assert returncode == 128 + profile_rocm.signal.SIGTERM
    assert termination == {"called": True, "was_alive": False}
    assert summary["status"] == "signal"
    assert summary["received_signal"] == profile_rocm.signal.SIGTERM
    assert isinstance(summary["final_signal_cutoff_monotonic_ns"], int)
    assert summary["pending_final_signals_at_cutoff"] == []


def test_abort_option_terminates_included_target_for_driver_error(monkeypatch, tmp_path):
    explicit_termination = {}
    returncode, _records, summary, _termination = _run_mocked_profiler(
        monkeypatch,
        tmp_path,
        power_for_phase=lambda _phase: 100.0,
        include_pid=("server", os.getpid()),
        terminate_included_on_abort=True,
        kernel_errors_for_call=lambda _call: ["fatal amdgpu event"],
        explicit_termination=explicit_termination,
    )

    assert returncode == 126
    assert summary["status"] == "driver_error"
    assert explicit_termination == {"calls": 1, "pids": [os.getpid()]}


def test_abort_option_does_not_terminate_included_target_after_success(monkeypatch, tmp_path):
    explicit_termination = {}
    returncode, records, summary, _termination = _run_mocked_profiler(
        monkeypatch,
        tmp_path,
        power_for_phase=lambda _phase: 100.0,
        command="pass",
        include_pid=("server", os.getpid()),
        terminate_included_on_abort=True,
        explicit_termination=explicit_termination,
    )

    assert records[0]["terminate_included_on_abort"] is True
    assert returncode == 0
    assert summary["status"] == "completed"
    assert explicit_termination == {}
    assert "terminated_explicit_pids" not in summary
    assert "surviving_explicit_pids" not in summary


def test_safety_only_option_does_not_terminate_included_target_on_timeout(monkeypatch, tmp_path):
    explicit_termination = {}
    returncode, records, summary, _termination = _run_mocked_profiler(
        monkeypatch,
        tmp_path,
        power_for_phase=lambda _phase: 100.0,
        timeout_seconds=0.03,
        include_pid=("server", os.getpid()),
        terminate_included_on_safety=True,
        explicit_termination=explicit_termination,
    )

    assert records[0]["terminate_included_on_safety"] is True
    assert records[0]["terminate_included_on_abort"] is False
    assert returncode == 124
    assert summary["status"] == "timeout"
    assert explicit_termination == {}


def test_power_breach_is_manifested_terminates_command_and_returns_125(monkeypatch, tmp_path):
    returncode, records, summary, termination = _run_mocked_profiler(
        monkeypatch,
        tmp_path,
        power_for_phase=lambda phase: 315.0 if phase == "preflight" else 315.25,
    )

    assert records[0]["safety_limits"]["max_gpu_power_watts"] == 315.0
    assert returncode == 125
    assert summary["status"] == "safety_limit"
    assert summary["safety_violation"] == {
        "metric": "gpu_power_watts",
        "value": 315.25,
        "limit": 315.0,
        "limit_kind": "maximum",
    }
    assert termination == {"called": True, "was_alive": True}


def test_wrapped_command_records_terminal_private_cgroup_proof(monkeypatch, tmp_path):
    returncode, records, summary, termination = _run_mocked_profiler(
        monkeypatch,
        tmp_path,
        power_for_phase=lambda _phase: 100.0,
        command="pass",
    )

    assert returncode == 0
    assert records[0]["wrapped_process_supervision"] == "pidfd_waitid_private_cgroup_v2"
    assert termination == {"called": True, "was_alive": False}
    containment = summary["wrapped_process_containment"]
    assert containment["cgroup_kill_writes"] == 1
    assert containment["process_group_term_sent"] is False
    assert containment["cgroup_empty"] is True
    assert containment["leader_reaped"] is True
    assert containment["terminal"] is True
    assert summary["wrapped_process_runtime"]["sigchld_restored"] is True


def test_wrapped_command_preserves_validated_pass_fds(monkeypatch, tmp_path):
    read_fd, write_fd = os.pipe()
    try:
        returncode, records, summary, _termination = _run_mocked_profiler(
            monkeypatch,
            tmp_path,
            power_for_phase=lambda _phase: 100.0,
            command="pass",
            inherited_fds=(read_fd,),
        )
    finally:
        os.close(read_fd)
        os.close(write_fd)

    assert returncode == 0
    assert records[0]["passed_file_descriptor_count"] == 1
    assert summary["status"] == "completed"


def test_final_pre_spawn_gate_rejects_preflight_older_than_50ms(monkeypatch, tmp_path):
    returncode, _records, summary, termination = _run_mocked_profiler(
        monkeypatch,
        tmp_path,
        power_for_phase=lambda _phase: 100.0,
        command="pass",
        before_pre_spawn_check=lambda: time.sleep(0.06),
    )

    assert returncode == 125
    assert termination == {"called": False, "was_alive": False}
    assert summary["status"] == "safety_limit"
    assert summary["safety_violation"]["reason"] == "preflight_launch_deadline"
    assert summary["safety_violation"]["value"] > 0.05
    assert "command" not in summary["processes"]


def test_final_pre_spawn_gate_preserves_signal_exit_without_launch(monkeypatch, tmp_path):
    returncode, _records, summary, termination = _run_mocked_profiler(
        monkeypatch,
        tmp_path,
        power_for_phase=lambda _phase: 100.0,
        command="pass",
        signal_before_pre_spawn=profile_rocm.signal.SIGTERM,
    )

    assert returncode == 128 + profile_rocm.signal.SIGTERM
    assert termination == {"called": False, "was_alive": False}
    assert summary["status"] == "signal"
    assert summary["received_signal"] == profile_rocm.signal.SIGTERM
    assert summary["wrapped_process_containment"]["terminal"] is True
    assert "command" not in summary["processes"]


def test_incomplete_wrapped_containment_fails_closed(monkeypatch, tmp_path):
    with pytest.raises(profile_rocm.ProcessSupervisionError, match="runtime restoration failed"):
        _run_mocked_profiler(
            monkeypatch,
            tmp_path,
            power_for_phase=lambda _phase: 100.0,
            command="pass",
            incomplete_containment=True,
        )

    summary = json.loads((tmp_path / "telemetry.jsonl.summary.json").read_text())
    assert summary["status"] == "error"
    assert summary["error_type"] == "ProcessSupervisionError"
    assert summary["wrapped_process_containment"]["terminal"] is False
    assert summary["wrapped_process_runtime"]["containment"][0]["terminal"] is False


def test_terminal_non_ok_launch_failure_remains_in_summary(monkeypatch, tmp_path):
    with pytest.raises(
        profile_rocm.ProcessSupervisionError,
        match="synthetic launch attestation failure",
    ):
        _run_mocked_profiler(
            monkeypatch,
            tmp_path,
            power_for_phase=lambda _phase: 100.0,
            command="pass",
            launch_attestation_failure=True,
        )

    summary = json.loads((tmp_path / "telemetry.jsonl.summary.json").read_text())
    assert summary["status"] == "error"
    assert summary["wrapped_process_runtime"] == {
        "containment": [],
        "issues": [],
        "sigchld_restored": True,
    }
    containment = summary["wrapped_process_containment"]
    assert containment["terminal"] is True
    assert containment["issues"] == [
        {
            "error_type": "OSError",
            "message": "synthetic terminal cleanup evidence failure",
            "operation": "close_pidfd",
            "phase": "launch_emergency",
        }
    ]


@pytest.mark.skipif(
    os.environ.get("SKYRL_RUN_REAL_CGROUP_TESTS") != "1",
    reason="requires an explicitly enabled delegated cgroup-v2 CPU smoke test",
)
def test_real_profile_runtime_contains_and_reaps_direct_child(monkeypatch, tmp_path):
    output = tmp_path / "real-supervision.jsonl"

    def sample(_device, _hwmon, _targets, _create_times, start, phase):
        return {
            "record_type": "sample",
            "phase": phase,
            "elapsed_seconds": time.monotonic() - start,
            "gpu_power_watts": 100.0,
            "gpu_junction_temp_c": 50.0,
            "vram_used_bytes": 0,
            "gtt_used_bytes": 0,
            "host_memory_used_bytes": 0,
            "host_memory_available_bytes": 64 * 1024**3,
            "host_swap_used_bytes": 0,
            "processes": {},
        }

    monkeypatch.setattr(
        profile_rocm,
        "_find_gpu",
        lambda _card: (Path("/mock/device"), None, {"card": "mock"}),
    )
    monkeypatch.setattr(profile_rocm, "_sample", sample)
    monkeypatch.setattr(profile_rocm, "_kernel_driver_errors_since", lambda _start: [])
    monkeypatch.setattr(profile_rocm.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_PROFILER),
            "--output",
            str(output),
            "--interval",
            "0.01",
            "--terminate-grace-seconds",
            "1",
            "--sensor-grace-seconds",
            "0",
            "--max-junction-temp-c",
            "90",
            "--max-gpu-power-watts",
            "400",
            "--",
            sys.executable,
            "-c",
            "pass",
        ],
    )

    assert profile_rocm.main() == 0
    summary = json.loads(output.with_suffix(output.suffix + ".summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["wrapped_process_containment"]["cgroup_kill_writes"] == 1
    assert summary["wrapped_process_containment"]["terminal"] is True
    assert summary["wrapped_process_runtime"]["sigchld_restored"] is True


def test_exact_power_limit_does_not_stop_completed_command(monkeypatch, tmp_path):
    returncode, records, summary, _termination = _run_mocked_profiler(
        monkeypatch,
        tmp_path,
        power_for_phase=lambda _phase: 315.0,
        command="pass",
    )

    assert records[0]["safety_limits"]["max_gpu_power_watts"] == 315.0
    assert returncode == 0
    assert summary["status"] == "completed"
    assert "safety_violation" not in summary


def test_missing_power_is_tolerated_before_grace_then_fails_closed(monkeypatch, tmp_path):
    returncode, _records, summary, termination = _run_mocked_profiler(
        monkeypatch,
        tmp_path,
        power_for_phase=lambda _phase: None,
        sensor_grace_seconds=0.03,
    )

    assert summary["measured_samples"] >= 2
    assert returncode == 125
    assert summary["safety_violation"] == {
        "metric": "gpu_power_watts",
        "value": None,
        "limit": 315.0,
        "limit_kind": "maximum",
        "unavailable": True,
    }
    assert termination == {"called": True, "was_alive": True}


@pytest.mark.parametrize(
    ("power", "junction", "max_power", "max_junction", "metric", "limit"),
    (
        (None, 50.0, 315.0, None, "gpu_power_watts", 315.0),
        (200.0, None, None, 90.0, "gpu_junction_temp_c", 90.0),
    ),
)
def test_completed_short_command_fails_if_configured_sensor_was_never_observed(
    monkeypatch, tmp_path, power, junction, max_power, max_junction, metric, limit
):
    returncode, _records, summary, _termination = _run_mocked_profiler(
        monkeypatch,
        tmp_path,
        power_for_phase=lambda _phase: power,
        junction_for_phase=lambda _phase: junction,
        max_power=max_power,
        max_junction=max_junction,
        sensor_grace_seconds=60.0,
        command="pass",
    )

    assert returncode == 125
    assert summary["status"] == "safety_limit"
    assert summary["safety_violation"]["metric"] == metric
    assert summary["safety_violation"]["limit"] == limit
    assert summary["safety_violation"]["unavailable"] is True


def test_attach_only_target_exit_fails_if_power_was_never_observed(monkeypatch, tmp_path):
    returncode, _records, summary, _termination = _run_mocked_profiler(
        monkeypatch,
        tmp_path,
        power_for_phase=lambda _phase: None,
        sensor_grace_seconds=60.0,
        command=None,
        include_pid=("target", os.getpid()),
        target_liveness=lambda: False,
    )

    assert returncode == 125
    assert summary["status"] == "safety_limit"
    assert summary["returncode"] is None
    assert summary["safety_violation"]["metric"] == "gpu_power_watts"


def test_attach_duration_late_sensor_failure_terminates_included_target(monkeypatch, tmp_path):
    explicit_termination = {}
    returncode, _records, summary, _termination = _run_mocked_profiler(
        monkeypatch,
        tmp_path,
        power_for_phase=lambda _phase: None,
        sensor_grace_seconds=60.0,
        command=None,
        duration=0.03,
        include_pid=("target", os.getpid()),
        terminate_included_on_safety=True,
        explicit_termination=explicit_termination,
    )

    assert returncode == 125
    assert summary["status"] == "safety_limit"
    assert explicit_termination == {"calls": 1, "pids": [os.getpid()]}
    assert summary["terminated_explicit_pids"] == [os.getpid()]
    assert summary["surviving_explicit_pids"] == []


def test_final_driver_error_outranks_unobserved_sensor(monkeypatch, tmp_path):
    returncode, _records, summary, _termination = _run_mocked_profiler(
        monkeypatch,
        tmp_path,
        power_for_phase=lambda _phase: None,
        sensor_grace_seconds=60.0,
        command="pass",
        kernel_errors_for_call=lambda call: [] if call == 1 else ["fatal amdgpu event"],
    )

    assert returncode == 126
    assert summary["status"] == "driver_error"
    assert summary["kernel_driver_errors"] == ["fatal amdgpu event"]
    assert "safety_violation" not in summary


def test_failed_command_is_not_relabelled_for_unobserved_sensor(monkeypatch, tmp_path):
    returncode, _records, summary, _termination = _run_mocked_profiler(
        monkeypatch,
        tmp_path,
        power_for_phase=lambda _phase: None,
        sensor_grace_seconds=60.0,
        command="raise SystemExit(7)",
    )

    assert returncode == 7
    assert summary["status"] == "command_failed"
    assert summary["returncode"] == 7
    assert "safety_violation" not in summary


def test_real_temperature_breach_outranks_unobserved_power(monkeypatch, tmp_path):
    returncode, _records, summary, _termination = _run_mocked_profiler(
        monkeypatch,
        tmp_path,
        power_for_phase=lambda _phase: None,
        junction_for_phase=lambda phase: 50.0 if phase != "measured" else 91.0,
        max_power=315.0,
        max_junction=90.0,
        sensor_grace_seconds=60.0,
    )

    assert returncode == 125
    assert summary["status"] == "safety_limit"
    assert summary["safety_violation"] == {
        "metric": "gpu_junction_temp_c",
        "value": 91.0,
        "limit": 90.0,
        "limit_kind": "maximum",
    }


def test_real_power_breach_stops_during_sensor_grace(monkeypatch, tmp_path):
    returncode, _records, summary, _termination = _run_mocked_profiler(
        monkeypatch,
        tmp_path,
        power_for_phase=lambda phase: 315.0 if phase == "preflight" else 400.0,
        sensor_grace_seconds=60.0,
    )

    assert returncode == 125
    assert summary["measured_samples"] == 1
    assert summary["safety_violation"]["metric"] == "gpu_power_watts"
    assert summary["safety_violation"]["value"] == 400.0


@pytest.mark.parametrize("power", (None, 400.0))
def test_unconfigured_power_limit_preserves_existing_behavior(monkeypatch, tmp_path, power):
    returncode, records, summary, _termination = _run_mocked_profiler(
        monkeypatch,
        tmp_path,
        power_for_phase=lambda _phase: power,
        max_power=None,
        sensor_grace_seconds=0.0,
        command="pass",
    )

    assert records[0]["safety_limits"]["max_gpu_power_watts"] is None
    assert returncode == 0
    assert summary["status"] == "completed"
    assert "safety_violation" not in summary


@pytest.mark.parametrize("value", ("-1", "nan", "inf"))
def test_invalid_power_limit_is_rejected_before_hardware_or_output(monkeypatch, tmp_path, value):
    output = tmp_path / f"invalid-{value}.jsonl"
    monkeypatch.setattr(
        profile_rocm,
        "_find_gpu",
        lambda _card: pytest.fail("invalid CLI must be rejected before GPU discovery"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_PROFILER),
            "--output",
            str(output),
            "--duration",
            "1",
            "--max-gpu-power-watts",
            value,
        ],
    )

    with pytest.raises(SystemExit) as raised:
        profile_rocm.main()

    assert raised.value.code == 2
    assert not output.exists()


def test_abort_option_requires_an_included_pid_before_hardware_or_output(monkeypatch, tmp_path):
    output = tmp_path / "missing-included-pid.jsonl"
    monkeypatch.setattr(
        profile_rocm,
        "_find_gpu",
        lambda _card: pytest.fail("invalid CLI must be rejected before GPU discovery"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_PROFILER),
            "--output",
            str(output),
            "--duration",
            "1",
            "--terminate-included-on-abort",
        ],
    )

    with pytest.raises(SystemExit) as raised:
        profile_rocm.main()

    assert raised.value.code == 2
    assert not output.exists()
