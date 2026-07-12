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

    assert (
        profile_rocm._safety_violation(
            {"gpu_junction_temp_c": 70.0, "gpu_power_watts": 315.0}, limits
        )
        is None
    )
    assert profile_rocm._safety_violation(
        {"gpu_junction_temp_c": 70.0, "gpu_power_watts": 315.25}, limits
    ) == {
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
def test_completion_requires_each_configured_gpu_sensor(
    junction, power, max_junction, max_power, expected_metric
):
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
    assert profile_rocm._unobserved_required_sensor_violation(
        baseline_only, limits
    )["metric"] == "gpu_junction_temp_c"

    observed_during_measurement = [
        {"phase": "baseline", **missing},
        {"phase": "measured", **present},
        {"phase": "measured", **missing},
    ]
    assert (
        profile_rocm._unobserved_required_sensor_violation(
            observed_during_measurement, limits
        )
        is None
    )


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
    include_pid: tuple[str, int] | None = None,
    terminate_included_on_safety: bool = False,
    kernel_errors_for_call=None,
    explicit_termination=None,
    target_liveness=None,
):
    output = tmp_path / "telemetry.jsonl"
    original_terminate = profile_rocm._terminate
    termination = {"called": False, "was_alive": False}

    def sample(_device, _hwmon, _targets, _create_times, start, phase):
        return {
            "record_type": "sample",
            "phase": phase,
            "elapsed_seconds": time.monotonic() - start,
            "gpu_power_watts": power_for_phase(phase),
            "gpu_junction_temp_c": (
                50.0 if junction_for_phase is None else junction_for_phase(phase)
            ),
            "vram_used_bytes": 0,
            "gtt_used_bytes": 0,
            "host_memory_used_bytes": 0,
            "host_memory_available_bytes": 64 * 1024**3,
            "host_swap_used_bytes": 0,
            "processes": {},
        }

    def terminate(process, grace_seconds):
        termination["called"] = True
        termination["was_alive"] = process.poll() is None
        return original_terminate(process, grace_seconds)

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
        if kernel_errors_for_call is None:
            return []
        return kernel_errors_for_call(kernel_check_count)

    monkeypatch.setattr(
        profile_rocm,
        "_kernel_driver_errors_since",
        kernel_driver_errors_since,
    )
    monkeypatch.setattr(profile_rocm, "_terminate", terminate)
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
    monkeypatch.setattr(profile_rocm.signal, "signal", lambda *_args: None)

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
    if duration is not None:
        arguments.extend(("--duration", str(duration)))
    if command is not None:
        arguments.extend(("--", sys.executable, "-c", command))
    monkeypatch.setattr(sys, "argv", arguments)

    returncode = profile_rocm.main()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    summary = json.loads(
        output.with_suffix(output.suffix + ".summary.json").read_text()
    )
    return returncode, records, summary, termination


def test_power_breach_is_manifested_terminates_command_and_returns_125(
    monkeypatch, tmp_path
):
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


def test_missing_power_is_tolerated_before_grace_then_fails_closed(
    monkeypatch, tmp_path
):
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
def test_unconfigured_power_limit_preserves_existing_behavior(
    monkeypatch, tmp_path, power
):
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
def test_invalid_power_limit_is_rejected_before_hardware_or_output(
    monkeypatch, tmp_path, value
):
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
