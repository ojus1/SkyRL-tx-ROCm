from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
import platform
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).parents[2]
_CONTROLLER_PATH = _REPO / "rocm" / "run_w8a8_lora_forward_gate.py"
_SPEC = importlib.util.spec_from_file_location(
    "run_w8a8_lora_forward_gate_test", _CONTROLLER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_CONTROLLER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONTROLLER)

_ACCELERATOR_ENVIRONMENT = (
    "AMDGCN_ENABLE_DUMP",
    "GPU_DEVICE_ORDINAL",
    "HSA_OVERRIDE_GFX_VERSION",
    "HIP_VISIBLE_DEVICES",
    "JAX_COMPILATION_CACHE_DIR",
    "JAX_ENABLE_COMPILATION_CACHE",
    "JAX_MOCK_GPU_TOPOLOGY",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
    "JAX_PJRT_CLIENT_CREATE_OPTIONS",
    "JAX_PLATFORMS",
    "JAX_RAISE_PERSISTENT_CACHE_ERRORS",
    "JAX_ROCM_VISIBLE_DEVICES",
    "MOCK_NUM_GPU_PROCESSES",
    "ROCR_VISIBLE_DEVICES",
    "TEST_UNDECLARED_OUTPUTS_DIR",
    "TF_FORCE_UNIFIED_MEMORY",
    "TF_XLA_HSACO_BITCODE_SIZE_THRESHOLD",
    "TF_XLA_HSACO_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TRITON_DUMP_DIR",
    "TRITON_KERNEL_DUMP",
    "XLA_CLIENT_MEM_FRACTION",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)


def _clean_environment(monkeypatch) -> None:
    for name in _ACCELERATOR_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)


def _run(*arguments: str):
    environment = os.environ.copy()
    for name in _ACCELERATOR_ENVIRONMENT:
        environment.pop(name, None)
    return subprocess.run(
        [sys.executable, str(_CONTROLLER_PATH), *arguments],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def test_default_is_abstract_refusal_without_jax() -> None:
    result = _run()

    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert [record["record_type"] for record in records] == [
        "controller_manifest",
        "refused",
    ]
    assert records[0]["platform_requested"] == "abstract"
    assert records[0]["jax_imported"] is False
    assert records[1]["jax_imported"] is False


def test_abstract_execute_manifest_is_nonoperational_and_imports_no_jax() -> None:
    result = _run("--phase", "execute")

    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert [record["record_type"] for record in records] == [
        "controller_manifest",
        "refused",
    ]
    assert records[0]["contract"]["phase"] == "execute"
    assert records[0]["contract"]["compiled_executable_invocations"] == 1
    assert all(record.get("jax_imported") is False for record in records)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--platform", "rocm"), "requires the explicit --allow-gpu"),
        (("--platform", "rocm", "--allow-gpu"), "requires a fresh absolute"),
        (("--allow-gpu",), "only valid with --platform rocm"),
        (("--run-dir", "/tmp/example"), "only valid with --platform rocm"),
        (("--backward",), "unrecognized arguments"),
        (("--warmups", "1"), "unrecognized arguments"),
        (("--repeats", "2"), "unrecognized arguments"),
        (("--max-vram-gib", "24"), "unrecognized arguments"),
    ],
)
def test_scope_and_limits_cannot_be_broadened(
    arguments: tuple[str, ...], message: str
) -> None:
    result = _run(*arguments)

    assert result.returncode == 2
    assert result.stdout == ""
    assert message in result.stderr


def test_operational_controller_refuses_nonisolated_startup_before_run_dir(
    tmp_path: Path,
) -> None:
    run_dir = (tmp_path / "never-created").resolve()

    result = _run(
        "--platform",
        "rocm",
        "--allow-gpu",
        "--run-dir",
        str(run_dir),
    )

    assert result.returncode == 2
    assert "requires exact -I -S -B -X isolation" in result.stderr
    assert not run_dir.exists()


def test_operational_controller_accepts_exact_isolated_startup_contract(
    tmp_path: Path,
) -> None:
    run_dir = (tmp_path / "future-run").resolve()
    cache_root = run_dir / "python-cache"
    code = (
        "import importlib.util,json,pathlib,sys;"
        "path=pathlib.Path(sys.argv[1]);run_dir=pathlib.Path(sys.argv[2]);"
        "spec=importlib.util.spec_from_file_location('isolated_controller',path);"
        "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);"
        "print(json.dumps(module._require_isolated_controller(run_dir)))"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            f"pycache_prefix={cache_root}",
            "-c",
            code,
            str(_CONTROLLER_PATH),
            str(run_dir),
        ],
        cwd=_REPO,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert all(record["checks"].values())
    assert record["pycache_prefix"] == str(cache_root)
    assert not run_dir.exists()


def test_contract_fixes_profiler_handoff_and_one_shot_counts() -> None:
    compile_contract = _CONTROLLER._contract("compile")
    runtime_contract = _CONTROLLER._contract("execute")

    assert compile_contract["compiled_executable_invocations"] == 0
    assert compile_contract["execute_rung_enabled"] is False
    assert compile_contract["backward_invocations"] == 0
    assert compile_contract["warmup_invocations"] == 0
    assert compile_contract["replay_invocations"] == 0
    assert compile_contract["profile"] == {
        "interval_seconds": 0.05,
        "baseline_seconds": 5.0,
        "timeout_seconds": 300.0,
        "sensor_grace_seconds": 15.0,
        "terminate_grace_seconds": 0.5,
        "maximum_sampled_junction_temperature_c": 90.0,
        "maximum_sampled_average_gpu_power_watts": 315.0,
        "maximum_sampled_sysfs_vram_gib": 2.0,
        "minimum_host_available_gib": 0.0,
        "maximum_swap_gib": 8.0,
        "independent_outer_watchdog_seconds": 330.0,
    }
    assert compile_contract["handoff"]["required_consecutive_ready_samples"] == 3
    assert compile_contract["handoff"]["vram_gtt_tolerance_bytes"] == 0
    assert runtime_contract["compiled_executable_invocations"] == 1
    assert runtime_contract["compiled_executable_completions"] == 1
    assert runtime_contract["execute_rung_enabled"] is True
    assert runtime_contract["host_sensitivity_comparisons"] == 35
    assert runtime_contract["isa_qualification_attempts"] == 1
    assert runtime_contract["tuple_device_put_attempts"] == 1
    assert runtime_contract["device_put_leaves"] == 6
    assert runtime_contract["input_readiness_invocations"] == 1
    assert runtime_contract["output_readiness_invocations"] == 1
    assert runtime_contract["device_get_attempts"] == 1
    assert runtime_contract["device_get_leaves"] == 1
    assert runtime_contract["dispatch_watchdog_seconds"] == 5.0
    assert runtime_contract["maximum_telemetry_dispatch_bracket_seconds"] == 0.2
    assert runtime_contract["backward_invocations"] == 0
    assert runtime_contract["warmup_invocations"] == 0
    assert runtime_contract["replay_invocations"] == 0
    assert runtime_contract["gpu_reference_invocations"] == 0
    assert runtime_contract["device_error_reduction_invocations"] == 0
    assert runtime_contract["model_invocations"] == 0
    with pytest.raises(RuntimeError, match="unsupported W8 qualification phase"):
        _CONTROLLER._contract("benchmark")


def test_new_run_directory_and_controller_files_are_private(tmp_path: Path) -> None:
    run_dir = tmp_path / "fresh"
    descriptor = _CONTROLLER._create_run_directory(run_dir.resolve())
    try:
        assert stat.S_IMODE(run_dir.stat().st_mode) == 0o700
        with _CONTROLLER._open_private_file(descriptor, "audit.jsonl") as output:
            output.write("{}\n")
        assert stat.S_IMODE((run_dir / "audit.jsonl").stat().st_mode) == 0o600
        with pytest.raises(FileExistsError):
            _CONTROLLER._open_private_file(descriptor, "audit.jsonl")
    finally:
        os.close(descriptor)


def test_bound_source_hashes_match_current_files() -> None:
    manifest = _CONTROLLER._source_manifest()

    assert manifest["child"] == _CONTROLLER._EXPECTED_SOURCE_SHA256["child"]
    assert set(_CONTROLLER._EXPECTED_SOURCE_SHA256) <= set(manifest)


def test_child_environment_has_exact_sole_xla_flag(monkeypatch) -> None:
    _clean_environment(monkeypatch)

    environment = _CONTROLLER._child_environment()

    assert environment["XLA_FLAGS"] == "--xla_gpu_enable_command_buffer="
    assert environment["JAX_PLATFORMS"] == "rocm"
    assert environment["ROCR_VISIBLE_DEVICES"] == "0"
    assert environment["HIP_VISIBLE_DEVICES"] == "0"
    assert environment["XLA_CLIENT_MEM_FRACTION"] == "0.075"
    assert environment["XDG_RUNTIME_DIR"] == f"/run/user/{os.getuid()}"
    assert environment["DBUS_SESSION_BUS_ADDRESS"] == (
        f"unix:path=/run/user/{os.getuid()}/bus"
    )


def test_systemctl_uses_only_the_validated_user_bus_environment() -> None:
    environment = _CONTROLLER._systemctl_environment()

    assert environment == {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}",
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{os.getuid()}/bus",
    }
    result = subprocess.run(
        ["/usr/bin/systemctl", "--user", "is-system-running"],
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
        env=environment,
    )
    assert result.returncode == 0
    assert result.stdout == "running\n"
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        (
            "XLA_FLAGS",
            "--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=2",
            "exact sole",
        ),
        ("HSA_OVERRIDE_GFX_VERSION", "11.0.0", "refusing inherited"),
        ("JAX_DISABLE_JIT", "1", "unexpected accelerator environment"),
        ("JAX_PLATFORMS", "cpu", "conflicts with required"),
        ("XLA_CLIENT_MEM_FRACTION", "1.0", "conflicts with required"),
    ],
)
def test_child_environment_rejects_inherited_overrides(
    monkeypatch, name: str, value: str, message: str
) -> None:
    _clean_environment(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=message):
        _CONTROLLER._child_environment()


def test_profile_command_is_exact_and_passes_only_the_lock_fd(tmp_path: Path) -> None:
    run_dir = tmp_path.resolve()

    command = _CONTROLLER._profile_command(
        phase="compile", run_dir=run_dir, card="card1", lock_fd=41
    )

    def value(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert command[:8] == [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-X",
        f"pycache_prefix={run_dir / 'python-cache'}",
        "-c",
        _CONTROLLER._ISOLATED_PROFILE_BOOTSTRAP,
    ]
    assert command[8].endswith("/.venv/lib/python3.12/site-packages")
    assert command[9] == str(_REPO)
    assert command[10] == str(_REPO / "rocm" / "profile_rocm.py")
    assert command[11] == _CONTROLLER._EXPECTED_SOURCE_SHA256["profiler"]
    assert json.loads(command[12]) == _CONTROLLER._EXPECTED_PROFILE_RUNTIME_SHA256
    assert command[13] == str(run_dir / "python-cache")
    assert value("--card") == "card1"
    assert value("--interval") == "0.05"
    assert value("--baseline-seconds") == "5.0"
    assert value("--timeout") == "300.0"
    assert value("--sensor-grace-seconds") == "15.0"
    assert value("--terminate-grace-seconds") == "0.5"
    assert value("--max-junction-temp-c") == "90.0"
    assert value("--max-gpu-power-watts") == "315.0"
    assert value("--max-vram-gib") == "2.0"
    assert value("--min-host-available-gib") == "0.0"
    assert value("--max-swap-gib") == "8.0"
    assert value("--pass-fd") == "41"
    separator = command.index("--")
    child = command[separator + 1 :]
    assert child[:7] == [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-X",
        f"pycache_prefix={run_dir / 'python-cache'}",
        str(_REPO / "rocm" / "probe_w8a8_lora_forward.py"),
    ]
    assert child[child.index("--phase") + 1] == "compile"
    assert child[child.index("--launcher-lock-fd") + 1] == "41"
    assert "--allow-gpu" in child
    assert "--backward" not in child
    runtime = _CONTROLLER._profile_command(
        phase="execute", run_dir=run_dir, card="card1", lock_fd=41
    )
    runtime_child = runtime[runtime.index("--") + 1 :]
    assert runtime_child[runtime_child.index("--phase") + 1] == "execute"
    with pytest.raises(RuntimeError, match="unsupported W8 qualification phase"):
        _CONTROLLER._profile_command(
            phase="benchmark", run_dir=run_dir, card="card1", lock_fd=41
        )


def test_isolated_profiler_bootstrap_executes_only_hashed_source(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "profile.py"
    profile.write_text(
        "import json,sys\n"
        "print(json.dumps({'isolated':sys.flags.isolated,'no_site':sys.flags.no_site,"
        "'path':sys.path}))\n"
    )
    payload = profile.read_bytes()
    site_packages = _REPO / ".venv" / "lib" / "python3.12" / "site-packages"
    cache_root = (tmp_path / "python-cache").resolve()
    cache_root.mkdir(mode=0o700)
    command = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        "-X",
        f"pycache_prefix={cache_root}",
        "-c",
        _CONTROLLER._ISOLATED_PROFILE_BOOTSTRAP,
        str(site_packages),
        str(_REPO),
        str(profile),
        hashlib.sha256(payload).hexdigest(),
        json.dumps(_CONTROLLER._EXPECTED_PROFILE_RUNTIME_SHA256, sort_keys=True),
        str(cache_root),
    ]

    result = subprocess.run(
        command,
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record == {
        "isolated": 1,
        "no_site": 1,
        "path": [
            "/usr/lib/python312.zip",
            "/usr/lib/python3.12",
            "/usr/lib/python3.12/lib-dynload",
            str(_REPO),
            str(site_packages),
        ],
    }
    command[11] = "0" * 64
    refused = subprocess.run(
        command,
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )
    assert refused.returncode != 0
    assert "refusing changed profiler source" in refused.stderr


def _contained_process(command: list[str], *, pass_fds: tuple[int, ...] = ()):
    try:
        _CONTROLLER._systemd_runtime_manifest()
    except RuntimeError as error:
        pytest.skip(f"user systemd scope unavailable: {error}")
    unit = _CONTROLLER._scope_name()
    process = subprocess.Popen(
        _CONTROLLER._scope_command(unit, command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_CONTROLLER._systemctl_environment(),
        pass_fds=pass_fds,
        start_new_session=True,
    )
    return unit, process


def test_systemd_scope_preserves_explicit_inherited_fd(tmp_path: Path) -> None:
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    code = (
        "import os,sys,time;"
        "os.fstat(int(sys.argv[1]));"
        "print('LOCK_FD_SURVIVED',flush=True);"
        "time.sleep(0.25)"
    )
    unit = None
    process = None
    try:
        unit, process = _contained_process(
            [sys.executable, "-c", code, str(descriptor)], pass_fds=(descriptor,)
        )
        returncode, audit = _CONTROLLER._wait_profile(process, unit)
        assert process.stdout is not None
        assert process.stderr is not None
        assert process.stdout.read().strip() == "LOCK_FD_SURVIVED"
        assert process.stderr.read() == ""
        assert returncode == 0
        assert audit["passed"] is True
    finally:
        os.close(descriptor)
        if unit is not None:
            _CONTROLLER._terminate_scope(unit)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=1)


def test_phase_specific_scope_names_are_exact(monkeypatch) -> None:
    monkeypatch.setattr(_CONTROLLER.os, "getpid", lambda: 123)
    monkeypatch.setattr(_CONTROLLER.time, "monotonic_ns", lambda: 0xABCDEF)

    assert _CONTROLLER._scope_name("compile") == "skyrl-w8a8-compile-123-abcdef"
    assert _CONTROLLER._scope_name("execute") == "skyrl-w8a8-runtime-123-abcdef"
    with pytest.raises(RuntimeError, match="unsupported W8 qualification phase"):
        _CONTROLLER._scope_name("benchmark")


def _runtime_watchdog_prefix(
    *, started_monotonic_ns: int, include_post_attempt: bool
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for record_type, stage in _CONTROLLER._RUNTIME_PROBE_PROTOCOL[:13]:
        record: dict[str, object] = {"record_type": record_type}
        if stage is not None:
            record["stage"] = stage
        records.append(record)
    started_wall_ns = time.time_ns()
    records.append(
        {
            "record_type": "dispatch_started",
            "wall_time_ns": started_wall_ns,
            "monotonic_ns": started_monotonic_ns,
            "one_shot_capability": {
                "consumed": True,
                "compiled_executable_attempts": 1,
                "compiled_executable_completions": 0,
            },
            "output_readiness_invocations": 0,
        }
    )
    if include_post_attempt:
        completed_wall_ns = started_wall_ns + 1_000_000
        completed_monotonic_ns = started_monotonic_ns + 1_000_000
        records.extend(
            [
                {
                    "record_type": "journal_checkpoint",
                    "stage": "after_candidate_dispatch_attempt",
                    "safety": {
                        "amdgpu_boot_clean": True,
                        "fatal_amdgpu_events": [],
                    },
                    "dispatch_completed_wall_time_ns": completed_wall_ns,
                    "dispatch_completed_monotonic_ns": completed_monotonic_ns,
                    "compiled_executable_attempts": 1,
                    "compiled_executable_completions": 0,
                },
                {
                    "record_type": "dispatch",
                    "dispatch_started_wall_time_ns": started_wall_ns,
                    "dispatch_completed_wall_time_ns": completed_wall_ns,
                    "dispatch_started_monotonic_ns": started_monotonic_ns,
                    "dispatch_completed_monotonic_ns": completed_monotonic_ns,
                    "dispatch_seconds_including_output_readiness": 0.001,
                    "one_shot_capability": {
                        "consumed": True,
                        "compiled_executable_attempts": 1,
                        "compiled_executable_completions": 1,
                    },
                    "output_readiness_invocations": 1,
                },
            ]
        )
    return records


@pytest.mark.parametrize(
    ("start_age_ns", "watchdog_seconds", "received_signal", "expected_reason"),
    [
        (1_000_000_000, 0.01, None, "dispatch_watchdog_timeout"),
        (0, 5.0, signal.SIGTERM, f"signal_{signal.SIGTERM}"),
    ],
)
def test_dispatch_watchdog_immediately_kills_after_start_on_timeout_or_signal(
    monkeypatch,
    tmp_path: Path,
    start_age_ns: int,
    watchdog_seconds: float,
    received_signal: int | None,
    expected_reason: str,
) -> None:
    probe = tmp_path / "probe.jsonl"
    started_ns = time.monotonic_ns() - start_age_ns
    _write_private(
        probe,
        "".join(
            json.dumps(record) + "\n"
            for record in _runtime_watchdog_prefix(
                started_monotonic_ns=started_ns, include_post_attempt=False
            )
        ),
    )
    monkeypatch.setattr(_CONTROLLER, "_DISPATCH_WATCHDOG_SECONDS", watchdog_seconds)
    unit = "skyrl-w8a8-runtime-123-abcdef"
    control_group = (
        f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/"
        f"app.slice/{unit}.scope"
    )
    scope_states = iter(
        [
            {
                "observed": True,
                "ControlGroup": control_group,
                "pids": [123],
                "ActiveState": "active",
            },
            {
                "observed": True,
                "ControlGroup": control_group,
                "pids": [],
                "ActiveState": "failed",
            },
        ]
    )
    monkeypatch.setattr(
        _CONTROLLER,
        "_scope_state",
        lambda _unit, *_args: next(scope_states),
    )
    direct_kills = []
    monkeypatch.setattr(
        _CONTROLLER,
        "_write_cgroup_kill",
        lambda path: (
            direct_kills.append(path) or {"path": str(path), "bytes_written": 1}
        ),
    )
    systemctl_commands = []
    monkeypatch.setattr(
        _CONTROLLER.subprocess,
        "run",
        lambda command, **_kwargs: (
            systemctl_commands.append(command)
            or SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )
    monkeypatch.setattr(_CONTROLLER, "_systemctl_environment", lambda: {})
    monkeypatch.setattr(
        _CONTROLLER,
        "_terminate_scope",
        lambda _unit: (_ for _ in ()).throw(
            AssertionError("watchdog timeout must never use SIGTERM cleanup")
        ),
    )

    class Process:
        reaped = False

        def poll(self):
            return -signal.SIGKILL if self.reaped else None

        def wait(self, timeout):
            assert 0 < timeout <= 1.0
            self.reaped = True
            return -signal.SIGKILL

        def kill(self):
            raise AssertionError("cgroup-wide SIGKILL should reap the wrapper")

    stop_calls = 0

    def stop_signal():
        nonlocal stop_calls
        stop_calls += 1
        if received_signal is not None and stop_calls < 2:
            return None
        return received_signal

    process = Process()
    returncode, audit = _CONTROLLER._wait_profile(
        process,
        unit,
        stop_signal,
        phase="execute",
        probe_path=probe,
    )

    assert returncode == -signal.SIGKILL
    assert audit["termination_reason"] == expected_reason
    assert audit["received_signal"] == received_signal
    assert audit["dispatch_watchdog"]["dispatch_started"] is True
    assert audit["dispatch_watchdog"]["post_attempt_observed"] is False
    assert audit["passed"] is False
    assert stop_calls == 2
    assert process.reaped is True
    assert len(direct_kills) == (1 if received_signal is not None else 0)
    assert len(systemctl_commands) == 1
    assert "--kill-whom=all" in systemctl_commands[0]
    assert "--signal=SIGKILL" in systemctl_commands[0]
    assert "--signal=SIGTERM" not in systemctl_commands[0]
    assert audit["cleanup"]["direct_cgroup_kill"]["passed"] is (
        received_signal is not None
    )
    assert audit["cleanup"]["kill_issued"] is True
    assert audit["cleanup"]["scope_empty"] is True
    assert audit["cleanup"]["wrapper_reaped"] is True
    assert audit["cleanup"]["kill_and_reap_seconds"] <= 1.01
    assert audit["cleanup"]["passed"] is True


def test_execute_signal_race_immediately_kills_when_start_appears_after_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    probe = tmp_path / "probe.jsonl"
    started_ns = time.monotonic_ns()
    records_with_start = _runtime_watchdog_prefix(
        started_monotonic_ns=started_ns, include_post_attempt=False
    )
    _write_private(
        probe,
        "".join(json.dumps(record) + "\n" for record in records_with_start[:13]),
    )
    unit = "skyrl-w8a8-runtime-123-abcdef"
    control_group = (
        f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/"
        f"app.slice/{unit}.scope"
    )
    scope_calls = 0

    def scope_state(_unit, *_args):
        nonlocal scope_calls
        scope_calls += 1
        if scope_calls == 1:
            _write_private(
                probe,
                "".join(json.dumps(record) + "\n" for record in records_with_start),
            )
            return {
                "observed": True,
                "ControlGroup": control_group,
                "pids": [123],
                "ActiveState": "active",
            }
        assert scope_calls == 2
        return {
            "observed": True,
            "ControlGroup": control_group,
            "pids": [],
            "ActiveState": "failed",
        }

    monkeypatch.setattr(_CONTROLLER, "_scope_state", scope_state)
    direct_kills = []
    monkeypatch.setattr(
        _CONTROLLER,
        "_write_cgroup_kill",
        lambda path: (
            direct_kills.append(path) or {"path": str(path), "bytes_written": 1}
        ),
    )
    systemctl_commands = []
    monkeypatch.setattr(
        _CONTROLLER.subprocess,
        "run",
        lambda command, **_kwargs: (
            systemctl_commands.append(command)
            or SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )
    monkeypatch.setattr(_CONTROLLER, "_systemctl_environment", lambda: {})
    monkeypatch.setattr(
        _CONTROLLER,
        "_terminate_scope",
        lambda _unit: (_ for _ in ()).throw(
            AssertionError("an execute signal must never use SIGTERM cleanup")
        ),
    )

    class Process:
        reaped = False

        def poll(self):
            return -signal.SIGKILL if self.reaped else None

        def wait(self, timeout):
            assert 0 < timeout <= 1.0
            self.reaped = True
            return -signal.SIGKILL

        def kill(self):
            raise AssertionError("cgroup-wide SIGKILL should reap the wrapper")

    stop_calls = 0

    def signal_after_poll():
        nonlocal stop_calls
        stop_calls += 1
        return None if stop_calls == 1 else signal.SIGTERM

    process = Process()
    returncode, audit = _CONTROLLER._wait_profile(
        process,
        unit,
        signal_after_poll,
        phase="execute",
        probe_path=probe,
    )

    assert returncode == -signal.SIGKILL
    assert audit["termination_reason"] == f"signal_{signal.SIGTERM}"
    assert audit["received_signal"] == signal.SIGTERM
    assert audit["dispatch_watchdog"]["dispatch_started"] is False
    assert _CONTROLLER._dispatch_watchdog_state(probe)["dispatch_started"] is True
    assert audit["scope_observed"] is True
    assert audit["control_group"] == control_group
    assert process.reaped is True
    assert scope_calls == 2
    assert len(direct_kills) == 1
    assert len(systemctl_commands) == 1
    assert "--signal=SIGKILL" in systemctl_commands[0]
    assert all("SIGTERM" not in argument for argument in systemctl_commands[0])
    assert audit["cleanup"]["direct_cgroup_kill"]["passed"] is True
    assert audit["cleanup"]["systemctl_cgroup_kill"]["passed"] is True
    assert audit["cleanup"]["wrapper_reaped"] is True
    assert audit["cleanup"]["scope_empty"] is True
    assert audit["cleanup"]["passed"] is True
    assert stop_calls == 2


def test_dispatch_immediate_kill_uses_systemctl_and_wrapper_fallback(
    monkeypatch,
) -> None:
    unit = "skyrl-w8a8-runtime-123-abcdef"
    control_group = (
        f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/"
        f"app.slice/{unit}.scope"
    )
    states = iter(
        [
            {
                "observed": True,
                "ControlGroup": control_group,
                "pids": [],
                "ActiveState": "failed",
            }
        ]
    )
    monkeypatch.setattr(_CONTROLLER, "_scope_state", lambda _unit, *_args: next(states))
    monkeypatch.setattr(
        _CONTROLLER,
        "_write_cgroup_kill",
        lambda _path: (_ for _ in ()).throw(FileNotFoundError("cgroup.kill gone")),
    )
    commands = []
    monkeypatch.setattr(
        _CONTROLLER.subprocess,
        "run",
        lambda command, **_kwargs: (
            commands.append(command)
            or SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )
    monkeypatch.setattr(_CONTROLLER, "_systemctl_environment", lambda: {})

    class Process:
        killed = False
        waits = 0

        def poll(self):
            return -signal.SIGKILL if self.killed else None

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("systemd-run", timeout)
            return -signal.SIGKILL

        def kill(self):
            self.killed = True

    process = Process()
    returncode, cleanup = _CONTROLLER._kill_dispatch_scope_immediately(
        unit, process, validated_control_group=control_group
    )

    assert returncode == -signal.SIGKILL
    assert process.killed is True
    assert cleanup["direct_cgroup_kill"]["passed"] is False
    assert cleanup["systemctl_cgroup_kill"]["passed"] is True
    assert cleanup["wrapper_kill_fallback"] is True
    assert cleanup["scope_empty"] is True
    assert cleanup["passed"] is True
    assert len(commands) == 1
    assert "--kill-whom=all" in commands[0]
    assert "--signal=SIGKILL" in commands[0]
    assert all("SIGTERM" not in argument for argument in commands[0])


def test_execute_supervisor_exception_before_start_conservatively_sigkills(
    monkeypatch, tmp_path: Path
) -> None:
    unit = "skyrl-w8a8-runtime-123-abcdef"
    states = iter(
        [
            RuntimeError("scope inspection failed"),
            {
                "observed": False,
                "ControlGroup": "",
                "pids": [],
                "ActiveState": "failed",
            },
        ]
    )

    def scope_state(_unit, *_args):
        value = next(states)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(_CONTROLLER, "_scope_state", scope_state)
    commands = []
    monkeypatch.setattr(
        _CONTROLLER.subprocess,
        "run",
        lambda command, **_kwargs: (
            commands.append(command)
            or SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )
    monkeypatch.setattr(_CONTROLLER, "_systemctl_environment", lambda: {})
    monkeypatch.setattr(
        _CONTROLLER,
        "_terminate_scope",
        lambda _unit: (_ for _ in ()).throw(
            AssertionError("execute supervisor failure must not use SIGTERM")
        ),
    )

    class Process:
        reaped = False

        def poll(self):
            return -signal.SIGKILL if self.reaped else None

        def wait(self, timeout):
            self.reaped = True
            return -signal.SIGKILL

        def kill(self):
            self.reaped = True

    returncode, audit = _CONTROLLER._wait_profile(
        Process(),
        unit,
        phase="execute",
        probe_path=tmp_path / "not-created.jsonl",
    )

    assert returncode == -signal.SIGKILL
    assert audit["termination_reason"] == "dispatch_watchdog_supervisor_error"
    assert audit["cleanup"]["systemctl_cgroup_kill"]["passed"] is True
    assert audit["cleanup"]["scope_empty"] is True
    assert audit["cleanup"]["wrapper_reaped"] is True
    assert audit["cleanup"]["passed"] is True
    assert len(commands) == 1
    assert "--signal=SIGKILL" in commands[0]


def test_execute_fast_wrapper_exit_immediately_sigkills_populated_scope(
    monkeypatch, tmp_path: Path
) -> None:
    probe = tmp_path / "probe.jsonl"
    started_ns = time.monotonic_ns()
    _write_private(
        probe,
        "".join(
            json.dumps(record) + "\n"
            for record in _runtime_watchdog_prefix(
                started_monotonic_ns=started_ns, include_post_attempt=True
            )
        ),
    )
    unit = "skyrl-w8a8-runtime-123-abcdef"
    control_group = (
        f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/"
        f"app.slice/{unit}.scope"
    )
    scope_timeouts = []
    scope_states = iter(
        [
            {
                "observed": True,
                "ControlGroup": control_group,
                "pids": [123],
                "ActiveState": "active",
            },
            {
                "observed": True,
                "ControlGroup": control_group,
                "pids": [456],
                "ActiveState": "active",
            },
            {
                "observed": True,
                "ControlGroup": control_group,
                "pids": [],
                "ActiveState": "failed",
            },
        ]
    )

    def scope_state(_unit, timeout_seconds=2.0):
        scope_timeouts.append(timeout_seconds)
        return next(scope_states)

    monkeypatch.setattr(_CONTROLLER, "_scope_state", scope_state)
    direct_kills = []
    monkeypatch.setattr(
        _CONTROLLER,
        "_write_cgroup_kill",
        lambda path: (
            direct_kills.append(path) or {"path": str(path), "bytes_written": 1}
        ),
    )
    systemctl_commands = []
    monkeypatch.setattr(
        _CONTROLLER.subprocess,
        "run",
        lambda command, **_kwargs: (
            systemctl_commands.append(command)
            or SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )
    monkeypatch.setattr(_CONTROLLER, "_systemctl_environment", lambda: {})
    monkeypatch.setattr(
        _CONTROLLER,
        "_terminate_scope",
        lambda _unit: (_ for _ in ()).throw(
            AssertionError("execute populated-scope exit must never use SIGTERM")
        ),
    )

    class Process:
        def poll(self):
            return 0

        def wait(self, timeout):
            raise AssertionError("an exited wrapper must not require another wait")

        def kill(self):
            raise AssertionError("the already-exited wrapper must not be killed")

    returncode, audit = _CONTROLLER._wait_profile(
        Process(),
        unit,
        phase="execute",
        probe_path=probe,
    )

    assert returncode == 0
    assert audit["termination_reason"] == "scope_still_populated_after_profile_exit"
    assert audit["passed"] is False
    assert audit["final_scope_state"]["pids"] == [456]
    assert audit["cleanup"]["method"] == (
        "immediate_cgroup_wide_sigkill_and_bounded_reap"
    )
    assert audit["cleanup"]["direct_cgroup_kill"]["passed"] is True
    assert audit["cleanup"]["systemctl_cgroup_kill"]["passed"] is True
    assert audit["cleanup"]["scope_empty"] is True
    assert audit["cleanup"]["wrapper_reaped"] is True
    assert audit["cleanup"]["passed"] is True
    assert scope_timeouts[0] == 2.0
    assert scope_timeouts[1] == _CONTROLLER._EXECUTE_FINAL_SCOPE_QUERY_SECONDS
    assert 0 < scope_timeouts[2] <= _CONTROLLER._DISPATCH_KILL_REAP_SECONDS
    assert len(scope_timeouts) == 3
    assert len(direct_kills) == 1
    assert len(systemctl_commands) == 1
    assert "--signal=SIGKILL" in systemctl_commands[0]
    assert all("SIGTERM" not in argument for argument in systemctl_commands[0])


def test_execute_stalled_final_scope_query_falls_into_immediate_sigkill(
    monkeypatch, tmp_path: Path
) -> None:
    probe = tmp_path / "probe.jsonl"
    started_ns = time.monotonic_ns()
    _write_private(
        probe,
        "".join(
            json.dumps(record) + "\n"
            for record in _runtime_watchdog_prefix(
                started_monotonic_ns=started_ns, include_post_attempt=True
            )
        ),
    )
    unit = "skyrl-w8a8-runtime-123-abcdef"
    control_group = (
        f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/"
        f"app.slice/{unit}.scope"
    )
    scope_timeouts = []

    def scope_state(_unit, timeout_seconds=2.0):
        scope_timeouts.append(timeout_seconds)
        if len(scope_timeouts) == 1:
            return {
                "observed": True,
                "ControlGroup": control_group,
                "pids": [123],
                "ActiveState": "active",
            }
        if len(scope_timeouts) == 2:
            raise subprocess.TimeoutExpired("systemctl show", timeout_seconds)
        assert len(scope_timeouts) == 3
        return {
            "observed": True,
            "ControlGroup": control_group,
            "pids": [],
            "ActiveState": "failed",
        }

    monkeypatch.setattr(_CONTROLLER, "_scope_state", scope_state)
    direct_kills = []
    monkeypatch.setattr(
        _CONTROLLER,
        "_write_cgroup_kill",
        lambda path: (
            direct_kills.append(path) or {"path": str(path), "bytes_written": 1}
        ),
    )
    commands = []
    monkeypatch.setattr(
        _CONTROLLER.subprocess,
        "run",
        lambda command, **_kwargs: (
            commands.append(command)
            or SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )
    monkeypatch.setattr(_CONTROLLER, "_systemctl_environment", lambda: {})
    monkeypatch.setattr(
        _CONTROLLER,
        "_terminate_scope",
        lambda _unit: (_ for _ in ()).throw(
            AssertionError("execute scope-query failure must never use SIGTERM")
        ),
    )

    class Process:
        def poll(self):
            return 0

        def wait(self, timeout):
            raise AssertionError("an exited wrapper must not require another wait")

        def kill(self):
            raise AssertionError("the already-exited wrapper must not be killed")

    returncode, audit = _CONTROLLER._wait_profile(
        Process(),
        unit,
        phase="execute",
        probe_path=probe,
    )

    assert returncode == 0
    assert audit["termination_reason"] == "dispatch_watchdog_supervisor_error"
    assert audit["dispatch_watchdog"]["supervisor_error_type"] == "TimeoutExpired"
    assert audit["cleanup"]["method"] == (
        "immediate_cgroup_wide_sigkill_and_bounded_reap"
    )
    assert audit["cleanup"]["passed"] is True
    assert scope_timeouts[1] == _CONTROLLER._EXECUTE_FINAL_SCOPE_QUERY_SECONDS
    assert len(scope_timeouts) == 3
    assert len(direct_kills) == 1
    assert len(commands) == 1
    assert "--signal=SIGKILL" in commands[0]
    assert all("SIGTERM" not in argument for argument in commands[0])


def test_dispatch_watchdog_parser_rejects_retry_and_accepts_one_postflight(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "probe.jsonl"
    started = time.monotonic_ns()
    completed = started + 1_000_000
    records = _runtime_watchdog_prefix(
        started_monotonic_ns=started, include_post_attempt=True
    )
    _write_private(probe, "".join(json.dumps(record) + "\n" for record in records))

    state = _CONTROLLER._dispatch_watchdog_state(probe)

    assert state["dispatch_started"] is True
    assert state["post_attempt_observed"] is True
    assert state["post_attempt_monotonic_ns"] == completed
    assert state["validated_protocol_prefix_length"] == 16

    records.insert(14, dict(records[13]))
    _write_private(probe, "".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(RuntimeError, match="protocol mismatch"):
        _CONTROLLER._dispatch_watchdog_state(probe)

    baseline = _runtime_watchdog_prefix(
        started_monotonic_ns=started, include_post_attempt=True
    )
    mutations = [
        ((13, "one_shot_capability", "compiled_executable_attempts"), 2, "capability"),
        (
            (13, "one_shot_capability", "compiled_executable_attempts"),
            True,
            "capability",
        ),
        (
            (13, "one_shot_capability", "compiled_executable_completions"),
            False,
            "capability",
        ),
        ((13, "output_readiness_invocations"), False, "capability"),
        ((14, "dispatch_completed_monotonic_ns"), started - 1, "checkpoint"),
        ((14, "compiled_executable_attempts"), True, "checkpoint"),
        ((14, "compiled_executable_completions"), False, "checkpoint"),
        ((14, "compiled_executable_completions"), 1, "checkpoint"),
        ((14, "safety", "amdgpu_boot_clean"), False, "checkpoint"),
        ((15, "dispatch_started_monotonic_ns"), started + 1, "completion record"),
        (
            (15, "one_shot_capability", "compiled_executable_attempts"),
            True,
            "completion record",
        ),
        (
            (15, "one_shot_capability", "compiled_executable_completions"),
            True,
            "completion record",
        ),
        ((15, "output_readiness_invocations"), True, "completion record"),
        ((15, "dispatch_seconds_including_output_readiness"), True, "duration"),
    ]
    for path, replacement, message in mutations:
        mutated = copy.deepcopy(baseline)
        target = mutated
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        _write_private(probe, "".join(json.dumps(record) + "\n" for record in mutated))
        with pytest.raises(
            _CONTROLLER._DispatchWatchdogProtocolError, match=message
        ) as caught:
            _CONTROLLER._dispatch_watchdog_state(probe)
        assert caught.value.dispatch_started is True

    checkpoint_only = baseline[:15]
    _write_private(
        probe, "".join(json.dumps(record) + "\n" for record in checkpoint_only)
    )
    checkpoint_state = _CONTROLLER._dispatch_watchdog_state(probe)
    assert checkpoint_state["post_attempt_checkpoint_observed"] is True
    assert checkpoint_state["post_attempt_observed"] is False


@pytest.mark.parametrize(
    "payload",
    [
        '{"value":1,"value":2}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e400}',
    ],
)
def test_strict_json_parser_rejects_duplicate_keys_and_nonfinite_numbers(
    payload: str,
) -> None:
    with pytest.raises(ValueError):
        _CONTROLLER._strict_json_loads(payload)


def test_telemetry_dispatch_bracket_uses_strict_two_sided_200ms_window() -> None:
    start = 1_000_000_000
    completed = 1_010_000_000

    exact = _CONTROLLER._dispatch_telemetry_bracket(
        [800_000_000, 1_210_000_000], start, completed
    )
    late = _CONTROLLER._dispatch_telemetry_bracket(
        [799_999_999, 1_210_000_001], start, completed
    )

    assert exact["passed"] is True
    assert exact["seconds_before_dispatch"] == 0.2
    assert exact["seconds_after_dispatch"] == 0.2
    assert late["passed"] is False


def test_outer_watchdog_terminates_contained_new_session_descendant(
    monkeypatch,
) -> None:
    code = (
        "import subprocess,sys,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
        "start_new_session=True);"
        "print(child.pid,flush=True);"
        "time.sleep(60)"
    )
    unit = None
    process = None
    try:
        unit, process = _contained_process([sys.executable, "-c", code])
        monkeypatch.setattr(_CONTROLLER, "_OUTER_WATCHDOG_SECONDS", 0.25)
        monkeypatch.setattr(_CONTROLLER, "_OUTER_TERMINATE_GRACE_SECONDS", 0.25)
        assert process.stdout is not None
        descendant_pid = int(process.stdout.readline().strip())
        returncode, audit = _CONTROLLER._wait_profile(process, unit)

        assert returncode != 0
        assert audit["termination_reason"] == "outer_watchdog_timeout"
        assert audit["passed"] is False
        assert audit["cleanup"]["passed"] is True
        assert process.poll() is not None
        assert not Path(f"/proc/{descendant_pid}").exists()
    finally:
        if unit is not None:
            _CONTROLLER._terminate_scope(unit)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=1)


def test_fast_profile_exit_cleans_contained_setsid_descendant() -> None:
    code = (
        "import subprocess,sys;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
        "start_new_session=True);"
        "print(child.pid,flush=True)"
    )
    unit = None
    process = None
    try:
        unit, process = _contained_process([sys.executable, "-c", code])
        assert process.stdout is not None
        descendant_pid = int(process.stdout.readline().strip())
        returncode, audit = _CONTROLLER._wait_profile(process, unit)

        assert returncode == 0
        assert audit["termination_reason"] == "scope_still_populated_after_profile_exit"
        assert audit["passed"] is False
        assert audit["cleanup"]["passed"] is True
        assert audit["cleanup"]["after"]["pids"] == []
        assert not Path(f"/proc/{descendant_pid}").exists()
    finally:
        if unit is not None:
            _CONTROLLER._terminate_scope(unit)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=1)


def test_vram_identity_gate_is_exact(tmp_path: Path) -> None:
    device = tmp_path / "device"
    device.mkdir()
    total = device / "mem_info_vram_total"
    total.write_text("25753026560\n")
    assert _CONTROLLER._read_vram_total(device) == 25_753_026_560
    total.write_text("25753026559\n")
    with pytest.raises(RuntimeError, match="does not match expected"):
        _CONTROLLER._read_vram_total(device)


def _write_private(path: Path, payload: str) -> None:
    path.write_text(payload)
    path.chmod(0o600)


def _valid_independent_ir_pair() -> tuple[str, str]:
    marker = "skyrl_qwen35_w8a8_lora_forward"
    stable = (
        "module {\n"
        "  func.func public @main(%arg0: tensor<3x64xbf16>, "
        "%arg1: tensor<64x32xi8>, %arg2: tensor<1x32xbf16>, "
        "%arg3: tensor<64x8xbf16>, %arg4: tensor<8x32xbf16>, "
        "%arg5: tensor<f32>) -> (tensor<3x17xbf16> "
        '{jax.result_info = "result"}) {\n'
        "    %0 = stablehlo.custom_call @__gpu$xla.gpu.triton() "
        f'{{mhlo.backend_config = {{ir = "payload\\00{marker}\\00", '
        f'name = "{marker}"}}}} : () -> tensor<16x32xbf16>\n'
        "    %1 = stablehlo.slice %0 : tensor<16x32xbf16>\n"
        "    return %1 : tensor<3x17xbf16>\n"
        "  }\n"
        "}\n"
    )
    optimized = (
        "ENTRY %main.6 (%x: bf16[3,64], %codes: s8[64,32], "
        "%scales: bf16[1,32], %a: bf16[64,8], %b: bf16[8,32], "
        "%scale: f32[]) -> bf16[3,17] {\n"
        "  %0 = bf16[16,32] custom-call(), "
        'custom_call_target="__gpu$xla.gpu.triton", '
        f'metadata={{op_name="jit(candidate)/{marker}/pallas_call"}}, '
        f'backend_config={{ir="payload\\00{marker}\\00", name="{marker}"}}\n'
        "  ROOT %1 = bf16[3,17] fusion(%0)\n"
        "}\n"
    )
    return stable, optimized


def test_independent_ir_gate_rejects_signature_tokens_supplied_only_in_comments() -> (
    None
):
    stable = (
        "func.func public @main(%arg0: tensor<1xf32>) -> tensor<1xf32> {\n"
        "  // tensor<3x64xbf16> tensor<64x32xi8> tensor<1x32xbf16> "
        "tensor<64x8xbf16> tensor<8x32xbf16> tensor<f32> tensor<3x17xbf16>\n"
        "  %0 = stablehlo.custom_call @__gpu$xla.gpu.triton() "
        '{mhlo.backend_config={name="skyrl_qwen35_w8a8_lora_forward"}} : '
        "() -> tensor<1xf32>\n"
        "  return %0 : tensor<1xf32>\n"
        "}\n"
    )
    optimized = (
        "ENTRY %main (%x: f32[1]) -> f32[1] {\n"
        "  // bf16[3,64] s8[64,32] bf16[1,32] bf16[64,8] bf16[8,32] "
        "f32[] bf16[3,17]\n"
        '  ROOT %0 = f32[1] custom-call(), custom_call_target="__gpu$xla.gpu.triton", '
        'metadata={op_name="jit(candidate)/skyrl_qwen35_w8a8_lora_forward/pallas_call"}, '
        'backend_config={name="skyrl_qwen35_w8a8_lora_forward"}\n'
        "}\n"
    )

    result = _CONTROLLER._independent_ir_gate(stable, optimized)

    assert result["passed"] is False
    assert result["checks"]["stablehlo_exact_public_main_signature"] is False
    assert result["checks"]["optimized_hlo_exact_entry_signature"] is False


def test_independent_ir_gate_accepts_realistic_duplicate_metadata() -> None:
    stable, optimized = _valid_independent_ir_pair()

    result = _CONTROLLER._independent_ir_gate(stable, optimized)

    assert result["passed"] is True


def test_independent_ir_gate_rejects_backend_name_and_target_decoys() -> None:
    marker = "skyrl_qwen35_w8a8_lora_forward"
    stable, optimized = _valid_independent_ir_pair()
    exact_name = f'name="{marker}"'
    exact_target = 'custom_call_target="__gpu$xla.gpu.triton"'
    optimized_cases = [
        optimized.replace(exact_name, 'name="wrong"'),
        optimized.replace(exact_name, f'debug={{name="{marker}"}}'),
        optimized.replace(exact_name, f"{exact_name}, {exact_name}"),
        optimized.replace(exact_name, f"/* {exact_name} */"),
        optimized.replace(exact_name, f'fake-name="{marker}"'),
        optimized.replace(
            exact_target,
            'custom_call_target="wrong", '
            'target_decoy={custom_call_target="__gpu$xla.gpu.triton"}',
        ),
        optimized.replace(exact_target, f"{exact_target}, {exact_target}"),
    ]

    for candidate in optimized_cases:
        assert _CONTROLLER._independent_ir_gate(stable, candidate)["passed"] is False


def test_independent_ir_gate_rejects_backend_map_scope_decoys() -> None:
    marker = "skyrl_qwen35_w8a8_lora_forward"
    stable, optimized = _valid_independent_ir_pair()
    stable_map = (
        f'mhlo.backend_config = {{ir = "payload\\00{marker}\\00", name = "{marker}"}}'
    )
    optimized_map = f'backend_config={{ir="payload\\00{marker}\\00", name="{marker}"}}'
    nested_stable = stable.replace(stable_map, f"metadata = {{{stable_map}}}")
    nested_optimized = optimized.replace(
        optimized_map, f"metadata_decoy={{{optimized_map}}}"
    )

    assert _CONTROLLER._independent_ir_gate(nested_stable, optimized)["passed"] is False
    assert _CONTROLLER._independent_ir_gate(stable, nested_optimized)["passed"] is False


def test_independent_ir_gate_rejects_dead_call_and_control_dependency() -> None:
    stable, optimized = _valid_independent_ir_pair()
    dead_stable = stable.replace(
        "    %1 = stablehlo.slice %0 : tensor<16x32xbf16>\n",
        "    %1 = stablehlo.constant dense<0> : tensor<3x17xbf16>\n",
    )
    dead_optimized = optimized.replace(
        "  ROOT %1 = bf16[3,17] fusion(%0)\n",
        "  ROOT %1 = bf16[3,17] constant(0), control-predecessors={%0}\n",
    )

    assert _CONTROLLER._independent_ir_gate(dead_stable, optimized)["passed"] is False
    assert _CONTROLLER._independent_ir_gate(stable, dead_optimized)["passed"] is False


def test_independent_ir_gate_rejects_call_owned_only_by_dead_helper() -> None:
    stable, optimized = _valid_independent_ir_pair()
    stable_call = next(
        line for line in stable.splitlines() if "stablehlo.custom_call" in line
    )
    stable_without_live_call = stable.replace(
        stable_call,
        "    %0 = stablehlo.constant dense<0> : tensor<16x32xbf16>",
    )
    dead_stable = (
        stable_without_live_call.removesuffix("}\n")
        + "  func.func private @dead() -> tensor<16x32xbf16> {\n"
        + stable_call
        + "\n    return %0 : tensor<16x32xbf16>\n  }\n}\n"
    )
    optimized_call = next(
        line for line in optimized.splitlines() if "custom-call" in line
    )
    optimized_without_live_call = optimized.replace(
        optimized_call, "  %0 = bf16[16,32] constant(0)"
    )
    dead_optimized = (
        "%dead {\n"
        + optimized_call
        + "\n  ROOT %dead_root = bf16[16,32] copy(%0)\n}\n\n"
        + optimized_without_live_call
    )

    stable_result = _CONTROLLER._independent_ir_gate(dead_stable, optimized)
    optimized_result = _CONTROLLER._independent_ir_gate(stable, dead_optimized)

    assert stable_result["passed"] is False
    assert stable_result["checks"]["stablehlo_call_owned_by_public_main"] is False
    assert optimized_result["passed"] is False
    assert optimized_result["checks"]["optimized_hlo_call_owned_by_entry"] is False


def test_evidence_audit_reparses_raw_ir_and_requires_two_measured_samples(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path
    run_dir.chmod(0o700)
    stable_text, optimized_text = _valid_independent_ir_pair()
    stable_sha = hashlib.sha256(stable_text.encode()).hexdigest()
    optimized_sha = hashlib.sha256(optimized_text.encode()).hexdigest()
    telemetry_records = [
        {
            "record_type": "manifest",
            "interval_seconds": 0.05,
            "baseline_seconds": 5.0,
            "duration_seconds": None,
            "timeout_seconds": 300.0,
            "sensor_grace_seconds": 15.0,
            "terminate_included_on_safety": False,
            "command_recorded": True,
            "passed_file_descriptor_count": 1,
            "runtime": {
                "script_sha256": _CONTROLLER._EXPECTED_SOURCE_SHA256["profiler"]
            },
            "gpu": {"card": "card1", "device_id": "0x744c"},
            "safety_limits": {
                "max_junction_temp_c": 90.0,
                "max_gpu_power_watts": 315.0,
                "max_vram_bytes": 2.0 * 1024**3,
                "min_host_available_bytes": 0.0,
                "max_swap_bytes": 8.0 * 1024**3,
            },
        },
        {
            "record_type": "sample",
            "phase": "baseline",
            "elapsed_seconds": -0.05,
            "wall_time_ns": 900_000_000,
            "gpu_power_watts": 20.0,
            "gpu_junction_temp_c": 39.0,
            "vram_used_bytes": 0,
            "host_memory_available_bytes": 16 * 1024**3,
            "host_swap_used_bytes": 0,
        },
        {
            "record_type": "sample",
            "phase": "preflight",
            "elapsed_seconds": -0.01,
            "wall_time_ns": 925_000_000,
            "gpu_power_watts": None,
            "gpu_junction_temp_c": None,
            "vram_used_bytes": 0,
            "host_memory_available_bytes": 16 * 1024**3,
            "host_swap_used_bytes": 0,
        },
        {
            "record_type": "sample",
            "phase": "measured",
            "elapsed_seconds": 0.0,
            "wall_time_ns": 950_000_000,
            "gpu_power_watts": 25.0,
            "gpu_junction_temp_c": 40.0,
            "vram_used_bytes": 1024,
            "host_memory_available_bytes": 16 * 1024**3,
            "host_swap_used_bytes": 0,
        },
        {
            "record_type": "sample",
            "phase": "measured",
            "elapsed_seconds": 0.05,
            "wall_time_ns": 1_050_000_000,
            "gpu_power_watts": 30.0,
            "gpu_junction_temp_c": 41.0,
            "vram_used_bytes": 2048,
            "host_memory_available_bytes": 16 * 1024**3,
            "host_swap_used_bytes": 0,
        },
    ]
    expected_profile_command = _CONTROLLER._profile_command(
        phase="compile", run_dir=run_dir, card="card1", lock_fd=41
    )
    telemetry_records[0]["command"] = expected_profile_command[
        expected_profile_command.index("--") + 1 :
    ]
    probe_records = [
        {
            "record_type": "manifest",
            "contract": {
                "dispatch_plan": {"compiled_executable_invocations": 0},
                "execute_rung_enabled": False,
            },
        },
        {
            "record_type": "static_preflight",
            "controller_supervision": {
                "validated": True,
                "scope": "skyrl-w8a8-compile-123-abcdef.scope",
            },
        },
        {"record_type": "hardware_preflight"},
        {
            "record_type": "host_oracle",
            "inputs": [
                dict(manifest) for manifest in _CONTROLLER._EXPECTED_HOST_MANIFESTS
            ],
            "expected": dict(_CONTROLLER._EXPECTED_HOST_OUTPUT),
            "verified_against_bound_hashes": True,
            "compile_signature_kind": "ShapeDtypeStruct",
            "compile_abstract_signature_derived_from_host_metadata": True,
            "lowering_consumed_host_values": False,
            "runtime_comparison_evaluated": False,
            "compiled_executable_invocations": 0,
        },
        {"record_type": "backend_ready"},
        {"record_type": "journal_checkpoint"},
        {
            "record_type": "lowered",
            "stablehlo_precompile_gate": {"passed": True},
            "stablehlo_artifact": {"sha256": stable_sha},
        },
        {
            "record_type": "compiled",
            "optimized_hlo_artifact": {"sha256": optimized_sha},
            "release_gate": {
                "passed": True,
                "structural_gate": {"passed": True},
                "memory_gate": {"passed": True},
                "runtime_promotion": False,
                "artifact_gate": {"nonempty": True, "isa_qualified": False},
            },
        },
        {
            "record_type": "completed",
            "status": "passed_compile_diagnostic_unpromoted",
            "compiled_executable_invocations": 0,
        },
    ]
    compiler_artifacts = run_dir / "compiler-artifacts"
    compiler_artifacts.mkdir(mode=0o700)
    _write_private(compiler_artifacts / "kernel.hsaco", "bounded-code-object")
    artifact_inventory = _CONTROLLER._independent_compiler_artifact_inventory(
        compiler_artifacts
    )
    probe_records[7]["artifact_inventory"] = artifact_inventory
    summary = {
        "record_type": "summary",
        "status": "completed",
        "samples": 4,
        "baseline_samples": 1,
        "measured_samples": 2,
        "returncode": 0,
        "received_signal": None,
        "kernel_log_available": True,
        "metrics": {
            "gpu_power_watts": {"measured_max": 30.0},
            "gpu_junction_temp_c": {"measured_max": 41.0},
            "vram_used_bytes": {"measured_max": 2048.0},
            "host_swap_used_bytes": {"measured_max": 0.0},
        },
    }
    _write_private(
        run_dir / "telemetry.jsonl",
        "".join(json.dumps(record) + "\n" for record in telemetry_records),
    )
    _write_private(run_dir / "telemetry.jsonl.summary.json", json.dumps(summary))
    _write_private(
        run_dir / "probe.jsonl",
        "".join(json.dumps(record) + "\n" for record in probe_records),
    )
    _write_private(run_dir / "w8a8-forward.stablehlo.mlir", stable_text)
    _write_private(run_dir / "w8a8-forward.optimized.hlo", optimized_text)

    audit = _CONTROLLER._audit_profile_outputs(
        run_dir,
        phase="compile",
        profile_returncode=0,
        wait_audit={"passed": True},
        expected_lock_fd=41,
    )

    assert audit["passed"] is True
    assert audit["controller_independent_raw_ir_gate"]["passed"] is True

    compile_type_mutations = [
        (
            "probe",
            (7, "artifact_inventory", "file_count"),
            True,
            "compiler_artifact_inventory_nonempty_and_exact",
        ),
        (
            "probe",
            (8, "compiled_executable_invocations"),
            False,
            "probe_terminal_compile_diagnostic",
        ),
        (
            "probe",
            (3, "compiled_executable_invocations"),
            False,
            "host_oracle_exact_bound_manifests",
        ),
        (
            "probe",
            (0, "contract", "dispatch_plan", "compiled_executable_invocations"),
            False,
            "probe_manifest_compile_only",
        ),
        (
            "probe",
            (3, "inputs", 0, "nbytes"),
            True,
            "host_oracle_exact_bound_manifests",
        ),
        (
            "telemetry",
            (0, "passed_file_descriptor_count"),
            True,
            "telemetry_protocol_exact",
        ),
        (
            "telemetry",
            (0, "safety_limits", "min_host_available_bytes"),
            False,
            "limits_exact",
        ),
        ("summary", ("returncode",), False, "summary_returncode_zero"),
        (
            "summary",
            ("metrics", "vram_used_bytes", "measured_max"),
            2048,
            "summary_measured_maxima_match_raw",
        ),
        ("summary", ("error_type",), "RuntimeError", "summary_no_error_fields"),
    ]

    def mutate_path(value, path, replacement):
        target = value
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement

    for collection, path, replacement, failed_check in compile_type_mutations:
        mutated_probe = copy.deepcopy(probe_records)
        mutated_telemetry = copy.deepcopy(telemetry_records)
        mutated_summary = copy.deepcopy(summary)
        target = {
            "probe": mutated_probe,
            "telemetry": mutated_telemetry,
            "summary": mutated_summary,
        }[collection]
        mutate_path(target, path, replacement)
        _write_private(
            run_dir / "probe.jsonl",
            "".join(json.dumps(record) + "\n" for record in mutated_probe),
        )
        _write_private(
            run_dir / "telemetry.jsonl",
            "".join(json.dumps(record) + "\n" for record in mutated_telemetry),
        )
        _write_private(
            run_dir / "telemetry.jsonl.summary.json", json.dumps(mutated_summary)
        )
        with pytest.raises(RuntimeError, match=failed_check):
            _CONTROLLER._audit_profile_outputs(
                run_dir,
                phase="compile",
                profile_returncode=0,
                wait_audit={"passed": True},
                expected_lock_fd=41,
            )

    _write_private(
        run_dir / "probe.jsonl",
        "".join(json.dumps(record) + "\n" for record in probe_records),
    )
    _write_private(
        run_dir / "telemetry.jsonl",
        "".join(json.dumps(record) + "\n" for record in telemetry_records),
    )
    _write_private(run_dir / "telemetry.jsonl.summary.json", json.dumps(summary))

    with pytest.raises(RuntimeError, match="profile_returncode_zero"):
        _CONTROLLER._audit_profile_outputs(
            run_dir,
            phase="compile",
            profile_returncode=False,
            wait_audit={"passed": True},
            expected_lock_fd=41,
        )

    probe_records[3]["inputs"][0]["sha256"] = "0" * 64
    _write_private(
        run_dir / "probe.jsonl",
        "".join(json.dumps(record) + "\n" for record in probe_records),
    )
    with pytest.raises(RuntimeError, match="host_oracle_exact_bound_manifests"):
        _CONTROLLER._audit_profile_outputs(
            run_dir,
            phase="compile",
            profile_returncode=0,
            wait_audit={"passed": True},
            expected_lock_fd=41,
        )
    probe_records[3]["inputs"][0] = dict(_CONTROLLER._EXPECTED_HOST_MANIFESTS[0])
    _write_private(
        run_dir / "probe.jsonl",
        "".join(json.dumps(record) + "\n" for record in probe_records),
    )

    _write_private(compiler_artifacts / "kernel.hsaco", "mutated-code-object")
    with pytest.raises(
        RuntimeError, match="compiler_artifact_inventory_nonempty_and_exact"
    ):
        _CONTROLLER._audit_profile_outputs(
            run_dir,
            phase="compile",
            profile_returncode=0,
            wait_audit={"passed": True},
            expected_lock_fd=41,
        )
    _write_private(compiler_artifacts / "kernel.hsaco", "bounded-code-object")

    summary["metrics"]["gpu_power_watts"]["measured_max"] = 29.0
    _write_private(run_dir / "telemetry.jsonl.summary.json", json.dumps(summary))
    with pytest.raises(RuntimeError, match="summary_measured_maxima_match_raw"):
        _CONTROLLER._audit_profile_outputs(
            run_dir,
            phase="compile",
            profile_returncode=0,
            wait_audit={"passed": True},
            expected_lock_fd=41,
        )
    summary["metrics"]["gpu_power_watts"]["measured_max"] = 30.0
    _write_private(run_dir / "telemetry.jsonl.summary.json", json.dumps(summary))

    exact_command = telemetry_records[0]["command"]
    telemetry_records[0]["command"] = exact_command[:-1]
    _write_private(
        run_dir / "telemetry.jsonl",
        "".join(json.dumps(record) + "\n" for record in telemetry_records),
    )
    with pytest.raises(RuntimeError, match="telemetry_wrapped_child_command_exact"):
        _CONTROLLER._audit_profile_outputs(
            run_dir,
            phase="compile",
            profile_returncode=0,
            wait_audit={"passed": True},
            expected_lock_fd=41,
        )
    telemetry_records[0]["command"] = exact_command
    _write_private(
        run_dir / "telemetry.jsonl",
        "".join(json.dumps(record) + "\n" for record in telemetry_records),
    )

    _write_private(
        run_dir / "telemetry.jsonl",
        "".join(json.dumps(record) + "\n" for record in telemetry_records[:-1]),
    )
    with pytest.raises(RuntimeError, match="at_least_two_measured_samples"):
        _CONTROLLER._audit_profile_outputs(
            run_dir,
            phase="compile",
            profile_returncode=0,
            wait_audit={"passed": True},
            expected_lock_fd=41,
        )


def test_runtime_evidence_audit_requires_exact_one_shot_sequence_and_counters(
    monkeypatch, tmp_path: Path
) -> None:
    run_dir = tmp_path
    run_dir.chmod(0o700)
    lock_fd = os.open(run_dir, os.O_RDONLY | os.O_DIRECTORY)
    lock_stat = os.fstat(lock_fd)
    stable_text, optimized_text = _valid_independent_ir_pair()
    stable_sha = hashlib.sha256(stable_text.encode()).hexdigest()
    optimized_sha = hashlib.sha256(optimized_text.encode()).hexdigest()
    artifact_root = run_dir / "compiler-artifacts"
    cache_dir = artifact_root / "jax-cache"
    cache_dir.mkdir(parents=True, mode=0o700)
    artifact_root.chmod(0o700)
    cache_dir.chmod(0o700)
    cache = cache_dir / f"jit_candidate-{'a' * 64}-cache"
    cache.write_bytes(b"fresh-runtime-cache")
    cache.chmod(0o600)
    cache_sha = hashlib.sha256(b"fresh-runtime-cache").hexdigest()
    artifact_inventory = _CONTROLLER._independent_compiler_artifact_inventory(
        artifact_root
    )
    isa = {
        "symbol": _CONTROLLER._EXPECTED_KERNEL_NAME,
        "amdgpu_target": _CONTROLLER._EXPECTED_NESTED_ELF_TARGET,
        "instruction": "v_wmma_i32_16x16x16_iu8",
        "static_instruction_count": 4,
        "signed_neg_lo": [1, 1, 0],
        "resources": {
            "sgpr_count": 34,
            "vgpr_count": 62,
            "sgpr_spill_count": 0,
            "vgpr_spill_count": 0,
            "private_segment_fixed_size": 0,
        },
    }
    isa_evidence = {
        "status": "passed_offline_isa_verification",
        "offline_only": True,
        "device_access_performed": False,
        "jax_modules_imported_by_verifier": False,
        "runtime_promotion": False,
        "cache": {
            "path": str(cache),
            "sha256": cache_sha,
            "expected_sha256_matched": True,
        },
        "candidate": {
            "bytes": 8440,
            "sha256": _CONTROLLER._EXPECTED_NESTED_ELF_SHA256,
            "expected_sha256_matched": True,
            "written_elf": None,
        },
        "elf_inventory": {
            "elf_count": 6,
            "unique_exact_symbol_candidate_count": 1,
        },
        "isa": isa,
        "serialization": {"format": "synthetic-bound"},
        "toolchain": {"llvm": "synthetic-bound"},
    }
    monkeypatch.setattr(
        _CONTROLLER,
        "_independent_fresh_isa_qualification",
        lambda *_args: {
            "passed": True,
            "checks": {"synthetic_independent_exact": True},
            "evidence": isa_evidence,
        },
    )
    runtime_plan = {
        "host_oracle_attempts": 1,
        "host_oracle_completions": 1,
        "host_sensitivity_comparisons": 35,
        "backend_initialization_attempts": 1,
        "backend_initialization_completions": 1,
        "lower_attempts": 1,
        "lower_completions": 1,
        "compile_attempts": 1,
        "compile_completions": 1,
        "isa_qualification_attempts": 1,
        "isa_qualification_completions": 1,
        "tuple_device_put_attempts": 1,
        "tuple_device_put_completions": 1,
        "device_put_leaves": 6,
        "input_readiness_invocations": 1,
        "compiled_executable_invocations": 1,
        "compiled_executable_completions": 1,
        "output_readiness_invocations": 1,
        "device_get_attempts": 1,
        "device_get_completions": 1,
        "device_get_leaves": 1,
        "backward_invocations": 0,
        "warmup_invocations": 0,
        "replay_invocations": 0,
        "gpu_reference_invocations": 0,
        "device_error_reduction_invocations": 0,
        "model_invocations": 0,
    }
    expected_labels = [
        *(
            f"global_zero_{name}"
            for name in (
                "x",
                "weight_codes",
                "weight_scales",
                "lora_a",
                "lora_b",
                "lora_scaling",
            )
        ),
        "global_zero_output",
        *(f"omitted_output_row_{index}" for index in range(3)),
        *(f"omitted_output_column_{index}" for index in range(17)),
        *(f"omitted_lora_rank_{index}" for index in range(8)),
    ]
    child_isa_checks = {
        name: True
        for name in (
            "status_exact",
            "offline_inspector",
            "caller_bound_fresh_cache",
            "one_unique_exact_symbol_candidate",
            "candidate_bytes_exact",
            "candidate_sha256_exact",
            "candidate_not_written_to_disk",
            "symbol_exact",
            "target_exact",
            "four_static_signed_iu8_wmma",
            "registers_exact",
            "zero_spills_and_private_segment",
        )
    }
    numerical_checks = {
        name: True
        for name in (
            "actual_shape_exact",
            "expected_shape_exact",
            "actual_dtype_bfloat16",
            "expected_dtype_bfloat16",
            "actual_nbytes_exact",
            "expected_nbytes_exact",
            "actual_c_contiguous",
            "expected_c_contiguous",
            "all_51_actual_elements_finite",
            "all_51_expected_elements_finite",
            "reference_norm_finite_nonzero",
            "relative_l2_below_one_percent",
            "cosine_at_least_0_9999",
            "maximum_absolute_error_at_most_0_25",
            "dispatch_finite_below_one_second",
        )
    }
    unused = {
        "consumed": False,
        "compiled_executable_attempts": 0,
        "compiled_executable_completions": 0,
    }
    started = {
        "consumed": True,
        "compiled_executable_attempts": 1,
        "compiled_executable_completions": 0,
    }
    completed_once = {
        "consumed": True,
        "compiled_executable_attempts": 1,
        "compiled_executable_completions": 1,
    }
    start_wall = 1_000_000_000
    completion_wall = 1_010_000_000
    start_monotonic = 2_000_000_000
    completion_monotonic = 2_010_000_000
    clean_journal = {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}
    device_root = (
        "/sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0/0000:02:00.0/0000:03:00.0"
    )
    expected_device_identity = {
        "drm_card": "card1",
        "vendor_id": "0x1002",
        "device_id": "0x744c",
        "pci_bdf": "0000:03:00.0",
        "pci_sysfs_path": device_root,
        "drm_sysfs_path": f"{device_root}/drm/card1",
        "drm_sysfs_dev": "226:1",
        "render_sysfs_path": f"{device_root}/drm/renderD128",
        "render_sysfs_dev": "226:128",
        "drm_node": {
            "path": "/dev/dri/card1",
            "rdev": "226:1",
            "sysfs_dev": "226:1",
            "sysfs_target": f"{device_root}/drm/card1",
        },
        "kfd_node": {
            "path": "/dev/kfd",
            "rdev": "236:0",
            "sysfs_dev": "236:0",
            "sysfs_target": "/sys/devices/virtual/kfd/kfd",
        },
        "render_node": {
            "path": "/dev/dri/renderD128",
            "rdev": "226:128",
            "sysfs_dev": "226:128",
            "sysfs_target": f"{device_root}/drm/renderD128",
        },
    }
    hardware_limits = {
        "power_cap_path": f"{device_root}/hwmon/hwmon4/power1_cap",
        "power_cap_uw": 315_000_000,
        "maximum_power_cap_uw": 315_000_000,
        "junction_path": f"{device_root}/hwmon/hwmon4/temp2_input",
        "junction_temperature_millic": 85_000,
        "maximum_launch_junction_millic": 85_000,
    }
    expected_sources = {
        "probe": _CONTROLLER._EXPECTED_SOURCE_SHA256["child"],
        **{
            name: _CONTROLLER._EXPECTED_SOURCE_SHA256[name]
            for name in (
                "isa_inspector",
                "kernel",
                "quantized_reference",
                "safety",
                "handoff",
                "profiler",
                "package_skyrl",
                "package_tx",
                "package_kernels",
                "package_rocm",
                "sealed_loader",
                "package_top_rocm",
            )
        },
    }
    stack = {
        **_CONTROLLER._EXPECTED_STACK_VERSIONS,
        "runtime_binaries": {
            name: {"bytes": 4096, "mode": 0o755, "sha256": digest}
            for name, digest in _CONTROLLER._EXPECTED_JAX_RUNTIME_SHA256.items()
        },
    }
    git = {"head": "a" * 40, "tree": "b" * 40, "worktree_clean": True}
    scope_unit = "skyrl-w8a8-runtime-123-abcdef"
    scope = f"{scope_unit}.scope"
    control_group = (
        f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/"
        f"app.slice/{scope}"
    )
    expected_profile_command = _CONTROLLER._profile_command(
        phase="execute", run_dir=run_dir, card="card1", lock_fd=lock_fd
    )
    parent_command_sha256 = hashlib.sha256(
        b"\0".join(value.encode() for value in expected_profile_command) + b"\0"
    ).hexdigest()
    expected_environment = {
        "XLA_FLAGS_original": _CONTROLLER._COMMAND_BUFFER_FLAG,
        "JAX_PLATFORMS": "rocm",
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
        "JAX_ROCM_VISIBLE_DEVICES": "0",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_CLIENT_MEM_FRACTION": "0.075",
        "XLA_FLAGS": _CONTROLLER._COMMAND_BUFFER_FLAG,
        "JAX_COMPILATION_CACHE_DIR": str(run_dir / "compiler-artifacts/jax-cache"),
        "JAX_ENABLE_COMPILATION_CACHE": "true",
        "JAX_RAISE_PERSISTENT_CACHE_ERRORS": "true",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
        "TF_XLA_HSACO_CACHE_DIR": str(run_dir / "compiler-artifacts/hsaco-cache"),
        "TRITON_CACHE_DIR": str(run_dir / "compiler-artifacts/triton-cache"),
        "TRITON_DUMP_DIR": str(run_dir / "compiler-artifacts/triton-dump"),
        "TRITON_KERNEL_DUMP": "1",
        "AMDGCN_ENABLE_DUMP": "1",
        "TEST_UNDECLARED_OUTPUTS_DIR": str(
            run_dir / "compiler-artifacts/compiler-dump"
        ),
    }
    exact_contract = {
        "model_family": "Qwen/Qwen3.5-4B",
        "operation": "w8a8_group64_rank8_lora_forward_only",
        "phase": "execute",
        "inputs": [
            {"name": "x", "shape": [3, 64], "dtype": "bfloat16"},
            {"name": "weight_codes", "shape": [64, 32], "dtype": "int8"},
            {"name": "weight_scales", "shape": [1, 32], "dtype": "bfloat16"},
            {"name": "lora_a", "shape": [64, 8], "dtype": "bfloat16"},
            {"name": "lora_b", "shape": [8, 32], "dtype": "bfloat16"},
            {"name": "lora_scaling", "shape": [], "dtype": "float32"},
        ],
        "output": {"shape": [3, 17], "dtype": "bfloat16"},
        "tiles": {
            "group_size": 64,
            "block_m": 16,
            "block_n": 16,
            "row_superblock": 16,
            "logical_out_features": 17,
            "physical_out_features": 32,
            "full_output_tiles_only": True,
        },
        "dispatch_plan": runtime_plan,
        "runtime_numerical_gate_evaluated": True,
        "runtime_promotion": False,
        "execute_rung_enabled": True,
        "outer_profile_rocm_required": True,
        "exact_idle_handoff_required": True,
    }
    probe_records = [
        {
            "record_type": "manifest",
            "platform_requested": "rocm",
            "allow_gpu": True,
            "contract": exact_contract,
            "graph_api_used": False,
            "command_buffer_used": False,
            "jax_imported": False,
        },
        {
            "record_type": "static_preflight",
            "sources": expected_sources,
            "isolated_python": {
                "checks": {
                    "isolated": True,
                    "ignore_environment": True,
                    "no_user_site": True,
                    "no_site": True,
                    "safe_path": True,
                    "dont_write_bytecode": True,
                },
                "repo_root": str(_REPO),
                "site_packages": str(
                    (_REPO / ".venv/lib/python3.12/site-packages").resolve()
                ),
                "initial_stdlib_path": [
                    "/usr/lib/python312.zip",
                    "/usr/lib/python3.12",
                    "/usr/lib/python3.12/lib-dynload",
                ],
                "sys_executable": str(_REPO / ".venv/bin/python"),
                "pycache_prefix": str(run_dir / "python-cache"),
                "pycache_empty_before_bound_imports": True,
                "main_cached_artifact": None,
            },
            "git": git,
            "stack": stack,
            "inherited_lock": {
                "validated": True,
                "fd": lock_fd,
                "device": lock_stat.st_dev,
                "inode": lock_stat.st_ino,
                "close_on_exec": True,
            },
            "controller_supervision": {
                "validated": True,
                "scope": scope,
                "cgroup": control_group,
                "parent_pid": 123,
                "parent_executable": str((_REPO / ".venv/bin/python").resolve()),
                "parent_command_sha256": parent_command_sha256,
            },
            "environment": expected_environment,
        },
        {
            "record_type": "hardware_preflight",
            "jax_imported": False,
            "hardware": {
                **clean_journal,
                "generic_headless_preflight": {
                    "amd_cards": ["card1"],
                    "connected_amd_connectors": [],
                    "kfd_accessible": True,
                    "kfd_path": "/dev/kfd",
                    "kfd_unowned": True,
                },
                "device": expected_device_identity,
                "kfd_owner_pids": [],
                "render_owner_pids": [],
                "controlled_render_wake_only": True,
                "pre_backend_hardware_limits": hardware_limits,
                "journal_after_controlled_wake": clean_journal,
            },
        },
        {
            "record_type": "host_oracle",
            "inputs": [
                dict(manifest) for manifest in _CONTROLLER._EXPECTED_HOST_MANIFESTS
            ],
            "expected": dict(_CONTROLLER._EXPECTED_HOST_OUTPUT),
            "verified_against_bound_hashes": True,
            "compile_signature_kind": "ShapeDtypeStruct",
            "compile_abstract_signature_derived_from_host_metadata": True,
            "lowering_consumed_host_values": False,
            "runtime_comparison_evaluated": False,
            "compiled_executable_invocations": 0,
            "oracle_attempts": 1,
            "oracle_completions": 1,
            "sensitivity": {
                "passed": True,
                "comparison_count": 35,
                "required_minimum_relative_l2_error_exclusive": 0.03,
                "minimum_observed_relative_l2_error": 0.04,
                "groups": {
                    "global": 7,
                    "omitted_rows": 3,
                    "omitted_columns": 17,
                    "omitted_ranks": 8,
                },
                "comparisons": [
                    {
                        "label": label,
                        "relative_l2_error": 0.04,
                        "passed": True,
                    }
                    for label in expected_labels
                ],
            },
        },
        {
            "record_type": "backend_ready",
            "jax": "0.10.2",
            "jaxlib": "0.10.2",
            "platform": "gpu",
            "platform_version": "PJRT C API\nrocm 70200",
            "devices": ["rocm:0"],
            "module_origins": {
                "jax": str(
                    (
                        _REPO / ".venv/lib/python3.12/site-packages/jax/__init__.py"
                    ).resolve()
                ),
                "jaxlib": str(
                    (
                        _REPO / ".venv/lib/python3.12/site-packages/jaxlib/__init__.py"
                    ).resolve()
                ),
                "numpy": str(
                    (
                        _REPO / ".venv/lib/python3.12/site-packages/numpy/__init__.py"
                    ).resolve()
                ),
                "quantized_lora": str(
                    (_REPO / "skyrl/tx/kernels/quantized_lora.py").resolve()
                ),
                "w8a8_lora": str(
                    (_REPO / "skyrl/tx/kernels/rocm/w8a8_lora.py").resolve()
                ),
            },
            "backend_initialization_attempts": 1,
            "backend_initialization_completions": 1,
            "compiled_executable_invocations": 0,
            "hardware_limits": hardware_limits,
        },
        {
            "record_type": "journal_checkpoint",
            "stage": "after_backend_initialization",
            "safety": clean_journal,
        },
        {
            "record_type": "lowered",
            "lower_attempts": 1,
            "lower_completions": 1,
            "compiled_executable_invocations": 0,
            "stablehlo_precompile_gate": {"passed": True},
            "stablehlo_artifact": {"sha256": stable_sha},
            "hardware_limits_before_lower": hardware_limits,
            "journal_checkpoint": clean_journal,
        },
        {
            "record_type": "compiled_unreleased",
            "compile_attempts": 1,
            "compile_completions": 1,
            "compiled_executable_invocations": 0,
            "executable_released": False,
            "optimized_hlo_artifact": {"sha256": optimized_sha},
            "artifact_inventory": artifact_inventory,
            "hardware_limits_before_compile": hardware_limits,
            "hardware_limits_after_compile": hardware_limits,
            "journal_checkpoint": clean_journal,
            "release_gate": {
                "passed": True,
                "structural_gate": {"passed": True},
                "memory_gate": {"passed": True},
                "artifact_gate": {"nonempty": True, "isa_qualified": False},
                "runtime_promotion": False,
            },
        },
        {
            "record_type": "fresh_isa_qualification",
            "qualification": {
                "passed": True,
                "checks": child_isa_checks,
                "evidence": isa_evidence,
            },
            "isa_qualification_attempts": 1,
            "isa_qualification_completions": 1,
            "compiled_executable_invocations": 0,
            "journal_checkpoint": clean_journal,
        },
        {
            "record_type": "executable_released",
            "released_for": "one_exact_runtime_correctness_invocation",
            "structural_gate_passed": True,
            "memory_gate_passed": True,
            "fresh_isa_gate_passed": True,
            "compiled_executable_invocations": 0,
            "runtime_promotion": False,
        },
        {
            "record_type": "input_device_put",
            "tuple_device_put_attempts": 1,
            "tuple_device_put_completions": 1,
            "device_put_leaves": 6,
            "input_readiness_invocations": 1,
            "compiled_executable_invocations": 0,
        },
        {
            "record_type": "journal_checkpoint",
            "stage": "after_input_device_put",
            "safety": clean_journal,
        },
        {
            "record_type": "dispatch_preflight",
            "xla_flags": _CONTROLLER._COMMAND_BUFFER_FLAG,
            "required_xla_flags": _CONTROLLER._COMMAND_BUFFER_FLAG,
            "one_shot_capability": unused,
            "compiled_executable_invocations": 0,
            "warmup_invocations": 0,
            "replay_invocations": 0,
            "backward_invocations": 0,
            "gpu_reference_invocations": 0,
            "device_error_reduction_invocations": 0,
            "model_invocations": 0,
            "hardware_limits": hardware_limits,
            "journal_checkpoint": clean_journal,
        },
        {
            "record_type": "dispatch_started",
            "wall_time_ns": start_wall,
            "monotonic_ns": start_monotonic,
            "one_shot_capability": started,
            "output_readiness_invocations": 0,
        },
        {
            "record_type": "journal_checkpoint",
            "stage": "after_candidate_dispatch_attempt",
            "dispatch_completed_wall_time_ns": completion_wall,
            "dispatch_completed_monotonic_ns": completion_monotonic,
            "compiled_executable_attempts": 1,
            "compiled_executable_completions": 0,
            "safety": clean_journal,
        },
        {
            "record_type": "dispatch",
            "dispatch_started_wall_time_ns": start_wall,
            "dispatch_completed_wall_time_ns": completion_wall,
            "dispatch_started_monotonic_ns": start_monotonic,
            "dispatch_completed_monotonic_ns": completion_monotonic,
            "dispatch_seconds_including_output_readiness": 0.01,
            "one_shot_capability": completed_once,
            "output_readiness_invocations": 1,
        },
        {
            "record_type": "device_get",
            "device_get_attempts": 1,
            "device_get_completions": 1,
            "device_get_leaves": 1,
            "one_shot_capability": completed_once,
        },
        {
            "record_type": "journal_checkpoint",
            "stage": "after_candidate_device_get",
            "safety": clean_journal,
        },
        {
            "record_type": "numerical_validation",
            "host_only_comparison": True,
            "gpu_reference_invocations": 0,
            "device_error_reduction_invocations": 0,
            "validation": {
                "passed": True,
                "checks": numerical_checks,
                "element_count": 51,
                "finite_actual_count": 51,
                "finite_expected_count": 51,
                "relative_l2_error": 0.0,
                "cosine_similarity": 1.0,
                "maximum_absolute_error": 0.0,
                "dispatch_seconds": 0.01,
                "limits": {
                    "maximum_relative_l2_error_exclusive": 0.01,
                    "minimum_cosine_similarity_inclusive": 0.9999,
                    "maximum_absolute_error_inclusive": 0.25,
                    "maximum_dispatch_seconds_exclusive": 1.0,
                },
                "expected": dict(_CONTROLLER._EXPECTED_HOST_OUTPUT),
                "actual": {
                    **_CONTROLLER._EXPECTED_HOST_OUTPUT,
                    "name": "actual",
                },
                "bitwise_equal_diagnostic": True,
            },
        },
        {
            "record_type": "completed",
            "status": "passed_exact_m3_k64_n17_w8a8_lora_forward_runtime_correctness_only",
            "runtime_promotion": False,
            "isa_qualified": True,
            "one_shot_capability": completed_once,
            "compiled_executable_invocations": 1,
            "compiled_executable_completions": 1,
            "warmup_invocations": 0,
            "replay_invocations": 0,
            "backward_invocations": 0,
            "gpu_reference_invocations": 0,
            "device_error_reduction_invocations": 0,
            "model_invocations": 0,
            "source_postflight": expected_sources,
            "git_postflight": git,
            "stack_postflight": stack,
            "journal_postflight": clean_journal,
        },
    ]
    for index, record in enumerate(probe_records):
        record["timestamp"] = f"2026-07-14T00:00:00.{index:06d}+00:00"
    telemetry_records = [
        {
            "record_type": "manifest",
            "interval_seconds": 0.05,
            "baseline_seconds": 5.0,
            "duration_seconds": None,
            "timeout_seconds": 300.0,
            "sensor_grace_seconds": 15.0,
            "terminate_included_on_safety": False,
            "command_recorded": True,
            "passed_file_descriptor_count": 1,
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "rocm": "7.2.4",
                "jax": "0.10.2",
                "jaxlib": "0.10.2",
                "jax_rocm_plugin": "0.10.2",
                "jax_rocm_pjrt": "0.10.2",
                "script_sha256": _CONTROLLER._EXPECTED_SOURCE_SHA256["profiler"],
                "accelerator_environment": {
                    "HIP_VISIBLE_DEVICES": "0",
                    "JAX_PLATFORMS": "rocm",
                    "JAX_ROCM_VISIBLE_DEVICES": "0",
                    "XLA_CLIENT_MEM_FRACTION": "0.075",
                    "XLA_FLAGS": _CONTROLLER._COMMAND_BUFFER_FLAG,
                    "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                },
            },
            "gpu": {
                "card": "card1",
                "card_path": "/sys/class/drm/card1",
                "pci_bdf": "0000:03:00.0",
                "vendor_id": "0x1002",
                "device_id": "0x744c",
                "subsystem_vendor_id": "0x1043",
                "subsystem_device_id": "0x0506",
                "hwmon": "/sys/class/drm/card1/device/hwmon/hwmon4",
                "hwmon_name": "amdgpu",
            },
            "safety_limits": {
                "max_junction_temp_c": 90.0,
                "max_gpu_power_watts": 315.0,
                "max_vram_bytes": 2.0 * 1024**3,
                "min_host_available_bytes": 0.0,
                "max_swap_bytes": 8.0 * 1024**3,
            },
        },
        {
            "record_type": "sample",
            "phase": "baseline",
            "elapsed_seconds": -0.05,
            "wall_time_ns": 900_000_000,
            "gpu_power_watts": 20.0,
            "gpu_junction_temp_c": 39.0,
            "vram_used_bytes": 0,
            "host_memory_available_bytes": 16 * 1024**3,
            "host_swap_used_bytes": 0,
        },
        {
            "record_type": "sample",
            "phase": "preflight",
            "elapsed_seconds": -0.01,
            "wall_time_ns": 925_000_000,
            "gpu_power_watts": None,
            "gpu_junction_temp_c": None,
            "vram_used_bytes": 0,
            "host_memory_available_bytes": 16 * 1024**3,
            "host_swap_used_bytes": 0,
        },
        {
            "record_type": "sample",
            "phase": "measured",
            "elapsed_seconds": 0.0,
            "wall_time_ns": 950_000_000,
            "gpu_power_watts": 25.0,
            "gpu_junction_temp_c": 40.0,
            "vram_used_bytes": 1024,
            "host_memory_available_bytes": 16 * 1024**3,
            "host_swap_used_bytes": 0,
        },
        {
            "record_type": "sample",
            "phase": "measured",
            "elapsed_seconds": 0.05,
            "wall_time_ns": 1_050_000_000,
            "gpu_power_watts": 30.0,
            "gpu_junction_temp_c": 41.0,
            "vram_used_bytes": 2048,
            "host_memory_available_bytes": 16 * 1024**3,
            "host_swap_used_bytes": 0,
        },
    ]
    expected_command = _CONTROLLER._profile_command(
        phase="execute", run_dir=run_dir, card="card1", lock_fd=lock_fd
    )
    telemetry_records[0]["command"] = expected_command[
        expected_command.index("--") + 1 :
    ]
    summary = {
        "record_type": "summary",
        "status": "completed",
        "samples": 4,
        "baseline_samples": 1,
        "measured_samples": 2,
        "returncode": 0,
        "received_signal": None,
        "kernel_log_available": True,
        "metrics": {
            "gpu_power_watts": {"measured_max": 30.0},
            "gpu_junction_temp_c": {"measured_max": 41.0},
            "vram_used_bytes": {"measured_max": 2048.0},
            "host_swap_used_bytes": {"measured_max": 0.0},
        },
    }
    _write_private(
        run_dir / "telemetry.jsonl",
        "".join(json.dumps(record) + "\n" for record in telemetry_records),
    )
    _write_private(run_dir / "telemetry.jsonl.summary.json", json.dumps(summary))
    _write_private(
        run_dir / "probe.jsonl",
        "".join(json.dumps(record) + "\n" for record in probe_records),
    )
    _write_private(run_dir / "w8a8-forward.stablehlo.mlir", stable_text)
    _write_private(run_dir / "w8a8-forward.optimized.hlo", optimized_text)
    wait_audit = {
        "passed": True,
        "control_group": control_group,
        "dispatch_watchdog": {
            "enabled": True,
            "timeout_seconds": 5.0,
            "protocol_prefix_exact": True,
            "validated_protocol_prefix_length": len(
                _CONTROLLER._RUNTIME_PROBE_PROTOCOL
            ),
            "dispatch_started": True,
            "dispatch_started_wall_time_ns": start_wall,
            "dispatch_started_monotonic_ns": start_monotonic,
            "post_attempt_wall_time_ns": completion_wall,
            "post_attempt_monotonic_ns": completion_monotonic,
            "post_attempt_observed": True,
            "post_attempt_record_types": ["journal_checkpoint", "dispatch"],
            "passed": True,
        },
    }

    audit = _CONTROLLER._audit_profile_outputs(
        run_dir,
        phase="execute",
        profile_returncode=0,
        wait_audit=wait_audit,
        expected_lock_fd=lock_fd,
        expected_scope_unit=scope_unit,
        expected_device_identity=expected_device_identity,
    )

    assert audit["passed"] is True
    assert audit["dispatch_telemetry_bracket"]["passed"] is True

    delete = object()
    mutations = [
        ("probe", (0, "allow_gpu"), False, "manifest_runtime_contract_exact"),
        (
            "probe",
            (1, "sources", "kernel"),
            "0" * 64,
            "static_source_binding_exact",
        ),
        (
            "probe",
            (1, "git", "worktree_clean"),
            1,
            "static_git_stack_binding_exact",
        ),
        (
            "probe",
            (
                1,
                "stack",
                "runtime_binaries",
                next(iter(stack["runtime_binaries"])),
                "bytes",
            ),
            True,
            "static_git_stack_binding_exact",
        ),
        (
            "probe",
            (1, "inherited_lock", "device"),
            True,
            "static_lock_binding_exact",
        ),
        (
            "probe",
            (1, "inherited_lock", "inode"),
            lock_stat.st_ino + 1,
            "static_lock_binding_exact",
        ),
        (
            "probe",
            (1, "isolated_python", "checks", "isolated"),
            1,
            "static_isolated_python_exact",
        ),
        (
            "probe",
            (1, "environment", "XLA_FLAGS"),
            "--xla_gpu_enable_command_buffer= --alien=1",
            "static_environment_binding_exact",
        ),
        (
            "probe",
            (1, "controller_supervision", "parent_command_sha256"),
            "0" * 64,
            "runtime_scope_exact",
        ),
        (
            "probe",
            (2, "hardware", "device", "render_node", "rdev"),
            "226:129",
            "hardware_preflight_exact",
        ),
        (
            "probe",
            (2, "hardware", "pre_backend_hardware_limits", "power_cap_path"),
            "/tmp/power1_cap",
            "hardware_preflight_exact",
        ),
        (
            "probe",
            (4, "module_origins", "jax"),
            "/tmp/jax.py",
            "backend_identity_and_stack_exact",
        ),
        (
            "probe",
            (4, "platform_version"),
            "ROCm alien",
            "backend_identity_and_stack_exact",
        ),
        (
            "probe",
            (4, "devices"),
            ["AlienDevice"],
            "backend_identity_and_stack_exact",
        ),
        (
            "probe",
            (6, "hardware_limits_before_lower", "power_cap_uw"),
            True,
            "all_hardware_limit_checkpoints_exact",
        ),
        (
            "probe",
            (5, "safety", "amdgpu_boot_clean"),
            False,
            "journal_record_sequence_exact",
        ),
        (
            "probe",
            (7, "journal_checkpoint", "fatal_amdgpu_events"),
            ["amdgpu fault"],
            "embedded_journal_evidence_clean",
        ),
        (
            "probe",
            (7, "artifact_inventory", "file_count"),
            True,
            "compiler_artifacts_exact",
        ),
        (
            "probe",
            (8, "qualification", "evidence", "candidate", "sha256"),
            "0" * 64,
            "child_and_controller_isa_match",
        ),
        (
            "probe",
            (8, "qualification", "checks", "registers_exact"),
            1,
            "child_fresh_isa_exact",
        ),
        (
            "probe",
            (8, "qualification", "evidence", "isa", "signed_neg_lo"),
            [True, True, False],
            "child_and_controller_isa_match",
        ),
        (
            "probe",
            (3, "sensitivity", "comparisons", 0, "relative_l2_error"),
            True,
            "host_sensitivity_exact",
        ),
        (
            "probe",
            (
                3,
                "sensitivity",
                "required_minimum_relative_l2_error_exclusive",
            ),
            0.02,
            "host_sensitivity_exact",
        ),
        (
            "probe",
            (3, "sensitivity", "minimum_observed_relative_l2_error"),
            0.05,
            "host_sensitivity_exact",
        ),
        (
            "probe",
            (15, "one_shot_capability", "compiled_executable_attempts"),
            True,
            "dispatch_exactly_once",
        ),
        (
            "probe",
            (18, "validation", "relative_l2_error"),
            True,
            "numerical_validation_exact",
        ),
        (
            "probe",
            (18, "validation", "checks", "actual_shape_exact"),
            1,
            "numerical_validation_exact",
        ),
        (
            "probe",
            (18, "validation", "actual", "nbytes"),
            102.0,
            "numerical_validation_exact",
        ),
        (
            "probe",
            (18, "validation", "actual", "shape"),
            [3.0, 17.0],
            "numerical_validation_exact",
        ),
        (
            "probe",
            (18, "validation", "relative_l2_error"),
            0.005,
            "numerical_hash_diagnostic_consistent",
        ),
        (
            "probe",
            (18, "validation", "maximum_absolute_error"),
            0.125,
            "numerical_hash_diagnostic_consistent",
        ),
        (
            "probe",
            (18, "validation", "cosine_similarity"),
            0.9999,
            "numerical_hash_diagnostic_consistent",
        ),
        (
            "probe",
            (
                18,
                "validation",
                "limits",
                "maximum_relative_l2_error_exclusive",
            ),
            0.02,
            "numerical_validation_exact",
        ),
        (
            "probe",
            (19, "source_postflight", "kernel"),
            "0" * 64,
            "terminal_runtime_correctness_only",
        ),
        (
            "probe",
            (13, "timestamp"),
            delete,
            "probe_timestamps_exact_and_ordered",
        ),
        (
            "telemetry",
            (3, "gpu_power_watts"),
            True,
            "telemetry_dispatch_bracket_metrics_complete",
        ),
        (
            "telemetry",
            (0, "runtime", "jax"),
            "0.10.3",
            "telemetry_profiler_runtime_exact",
        ),
        (
            "telemetry",
            (0, "runtime", "accelerator_environment", "XLA_FLAGS"),
            "--xla_gpu_enable_command_buffer= --alien=1",
            "telemetry_profiler_runtime_exact",
        ),
        (
            "telemetry",
            (0, "gpu", "pci_bdf"),
            "0000:04:00.0",
            "gpu_identity_exact",
        ),
        (
            "telemetry",
            (0, "gpu", "hwmon"),
            "/sys/class/drm/card1/device/hwmon/hwmon5",
            "gpu_identity_exact",
        ),
        ("summary", ("returncode",), False, "summary_completed_cleanly"),
        (
            "summary",
            ("samples",),
            True,
            "summary_sample_counts_match",
        ),
        (
            "summary",
            ("kernel_driver_errors",),
            [],
            "summary_completed_cleanly",
        ),
        (
            "summary",
            ("error_type",),
            "RuntimeError",
            "summary_completed_cleanly",
        ),
        (
            "summary",
            ("error",),
            "synthetic completed-run error",
            "summary_completed_cleanly",
        ),
        (
            "summary",
            ("terminated_explicit_pids",),
            [],
            "summary_completed_cleanly",
        ),
        (
            "summary",
            ("surviving_explicit_pids",),
            [],
            "summary_completed_cleanly",
        ),
        (
            "wait",
            ("dispatch_watchdog", "post_attempt_monotonic_ns"),
            completion_monotonic + 1,
            "dispatch_watchdog_passed",
        ),
    ]

    def mutate(value, path, replacement):
        target = value
        for key in path[:-1]:
            target = target[key]
        if replacement is delete:
            del target[path[-1]]
        else:
            target[path[-1]] = replacement

    for collection, path, replacement, failed_check in mutations:
        mutated_probe = copy.deepcopy(probe_records)
        mutated_telemetry = copy.deepcopy(telemetry_records)
        mutated_summary = copy.deepcopy(summary)
        mutated_wait = copy.deepcopy(wait_audit)
        target = {
            "probe": mutated_probe,
            "telemetry": mutated_telemetry,
            "summary": mutated_summary,
            "wait": mutated_wait,
        }[collection]
        mutate(target, path, replacement)
        _write_private(
            run_dir / "probe.jsonl",
            "".join(json.dumps(record) + "\n" for record in mutated_probe),
        )
        _write_private(
            run_dir / "telemetry.jsonl",
            "".join(json.dumps(record) + "\n" for record in mutated_telemetry),
        )
        _write_private(
            run_dir / "telemetry.jsonl.summary.json", json.dumps(mutated_summary)
        )
        with pytest.raises(RuntimeError, match=failed_check):
            _CONTROLLER._audit_profile_outputs(
                run_dir,
                phase="execute",
                profile_returncode=0,
                wait_audit=mutated_wait,
                expected_lock_fd=lock_fd,
                expected_scope_unit=scope_unit,
                expected_device_identity=expected_device_identity,
            )

    telemetry_protocol_mutations = []
    alien_record = copy.deepcopy(telemetry_records)
    alien_record.insert(
        3,
        {"record_type": "alien_runtime_evidence", "phase": "measured"},
    )
    telemetry_protocol_mutations.append(alien_record)
    alien_phase = copy.deepcopy(telemetry_records)
    alien_phase[3]["phase"] = "alien"
    telemetry_protocol_mutations.append(alien_phase)
    phase_order_regression = copy.deepcopy(telemetry_records)
    phase_order_regression[4]["phase"] = "baseline"
    telemetry_protocol_mutations.append(phase_order_regression)
    duplicate_preflight = copy.deepcopy(telemetry_records)
    duplicate_preflight.insert(3, copy.deepcopy(duplicate_preflight[2]))
    telemetry_protocol_mutations.append(duplicate_preflight)

    _write_private(
        run_dir / "probe.jsonl",
        "".join(json.dumps(record) + "\n" for record in probe_records),
    )
    _write_private(run_dir / "telemetry.jsonl.summary.json", json.dumps(summary))
    for mutated_telemetry in telemetry_protocol_mutations:
        _write_private(
            run_dir / "telemetry.jsonl",
            "".join(json.dumps(record) + "\n" for record in mutated_telemetry),
        )
        with pytest.raises(RuntimeError, match="telemetry_record_sequence_exact"):
            _CONTROLLER._audit_profile_outputs(
                run_dir,
                phase="execute",
                profile_returncode=0,
                wait_audit=wait_audit,
                expected_lock_fd=lock_fd,
                expected_scope_unit=scope_unit,
                expected_device_identity=expected_device_identity,
            )

    missing_bracket_metric = copy.deepcopy(telemetry_records)
    del missing_bracket_metric[3]["gpu_power_watts"]
    _write_private(
        run_dir / "telemetry.jsonl",
        "".join(json.dumps(record) + "\n" for record in missing_bracket_metric),
    )
    with pytest.raises(
        RuntimeError, match="telemetry_dispatch_bracket_metrics_complete"
    ):
        _CONTROLLER._audit_profile_outputs(
            run_dir,
            phase="execute",
            profile_returncode=0,
            wait_audit=wait_audit,
            expected_lock_fd=lock_fd,
            expected_scope_unit=scope_unit,
            expected_device_identity=expected_device_identity,
        )

    excessive_swap = copy.deepcopy(telemetry_records)
    excessive_swap[4]["host_swap_used_bytes"] = 9 * 1024**3
    excessive_swap_summary = copy.deepcopy(summary)
    excessive_swap_summary["metrics"]["host_swap_used_bytes"]["measured_max"] = (
        9 * 1024**3
    )
    _write_private(
        run_dir / "telemetry.jsonl",
        "".join(json.dumps(record) + "\n" for record in excessive_swap),
    )
    _write_private(
        run_dir / "telemetry.jsonl.summary.json",
        json.dumps(excessive_swap_summary),
    )
    with pytest.raises(RuntimeError, match="telemetry_limits_present_and_within"):
        _CONTROLLER._audit_profile_outputs(
            run_dir,
            phase="execute",
            profile_returncode=0,
            wait_audit=wait_audit,
            expected_lock_fd=lock_fd,
            expected_scope_unit=scope_unit,
            expected_device_identity=expected_device_identity,
        )
    os.close(lock_fd)


def test_operational_flow_always_settles_and_rechecks_journal(
    monkeypatch, tmp_path: Path
) -> None:
    import rocm.amdgpu_safety as safety
    import rocm.qwen35_prewarm_handoff as handoff

    run_dir = (tmp_path / "run").resolve()
    run_dir_fd = _CONTROLLER._create_run_directory(run_dir)
    lock_dir = tmp_path / "fake-lock"
    lock_dir.mkdir()
    lock_fd = os.open(lock_dir, os.O_RDONLY | os.O_DIRECTORY)
    device_root = tmp_path / "device"
    device_root.mkdir()
    state = {"settled": 0, "journal": 0}
    device = SimpleNamespace(
        device_id="0x744c",
        drm_card="card1",
        device_root=device_root,
        identity=lambda: {"device_id": "0x744c", "drm_card": "card1"},
    )

    monkeypatch.setattr(_CONTROLLER, "_source_manifest", lambda: {"bound": "yes"})
    monkeypatch.setattr(
        _CONTROLLER,
        "_systemd_runtime_manifest",
        lambda: {"user_manager": "mocked-running"},
    )
    monkeypatch.setattr(
        _CONTROLLER,
        "_profile_runtime_manifest",
        lambda: {"psutil_version": "mocked"},
    )
    monkeypatch.setattr(
        _CONTROLLER,
        "_child_environment",
        lambda *_args: {
            "JAX_PLATFORMS": "rocm",
            "ROCR_VISIBLE_DEVICES": "0",
            "HIP_VISIBLE_DEVICES": "0",
            "GPU_DEVICE_ORDINAL": "0",
            "JAX_ROCM_VISIBLE_DEVICES": "0",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_CLIENT_MEM_FRACTION": "0.075",
            "XLA_FLAGS": "--xla_gpu_enable_command_buffer=",
        },
    )
    monkeypatch.setattr(_CONTROLLER, "_read_vram_total", lambda _root: 25_753_026_560)
    monkeypatch.setattr(
        _CONTROLLER,
        "_profile_command",
        lambda **_kwargs: [sys.executable, "-c", "import time;time.sleep(0.05)"],
    )
    monkeypatch.setattr(
        _CONTROLLER,
        "_scope_command",
        lambda _unit, command: command,
    )
    monkeypatch.setattr(_CONTROLLER, "_scope_name", lambda: "mocked-scope")

    def wait_profile(process, _unit, _stop_signal):
        return process.wait(timeout=1), {
            "passed": True,
            "received_signal": None,
            "termination_reason": None,
        }

    monkeypatch.setattr(_CONTROLLER, "_wait_profile", wait_profile)
    monkeypatch.setattr(
        _CONTROLLER,
        "_terminate_scope",
        lambda _unit: {
            "passed": True,
            "before": {"pids": []},
            "after": {"pids": []},
        },
    )
    monkeypatch.setattr(
        _CONTROLLER,
        "_direct_cgroup_cleanup",
        lambda _unit: {"passed": True, "final_pids": []},
    )
    monkeypatch.setattr(
        _CONTROLLER,
        "_audit_profile_outputs",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(safety, "acquire_qwen35_rocm_launch_lock", lambda: lock_fd)
    monkeypatch.setattr(handoff, "_discover_device", lambda: device)

    def capture(path):
        path.write_text("baseline\n")
        path.chmod(0o600)
        return {"status": "passed"}

    def settle(*_args, **_kwargs):
        state["settled"] += 1
        return {"status": "passed"}

    def journal():
        state["journal"] += 1
        return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}

    monkeypatch.setattr(handoff, "capture_baseline", capture)
    monkeypatch.setattr(handoff, "settle_handoff", settle)
    monkeypatch.setattr(safety, "require_clean_amdgpu_boot", journal)
    output = io.StringIO()
    args = SimpleNamespace(phase="compile", run_dir=run_dir)

    try:
        result = _CONTROLLER._run_gate(args, output, run_dir_fd)
    finally:
        os.close(run_dir_fd)

    assert result == 0
    assert state == {"settled": 1, "journal": 1}
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert records[-1]["record_type"] == "controller_postflight"
    assert records[-1]["idle_handoff"]["status"] == "passed"
    assert records[-1]["final_journal"]["amdgpu_boot_clean"] is True
    with pytest.raises(OSError):
        os.fstat(lock_fd)


@pytest.mark.parametrize(
    ("termination_reason", "received_signal", "expected_returncode", "terminate_calls"),
    [
        ("dispatch_watchdog_timeout", None, 1, 0),
        (f"signal_{signal.SIGTERM}", signal.SIGTERM, 128 + signal.SIGTERM, 0),
    ],
)
def test_timeout_and_signal_paths_share_handoff_journal_and_final_cleanup(
    monkeypatch,
    tmp_path: Path,
    termination_reason: str,
    received_signal: int | None,
    expected_returncode: int,
    terminate_calls: int,
) -> None:
    import rocm.amdgpu_safety as safety
    import rocm.qwen35_prewarm_handoff as handoff

    run_dir = (tmp_path / "run").resolve()
    run_dir_fd = _CONTROLLER._create_run_directory(run_dir)
    lock_dir = tmp_path / "fake-lock"
    lock_dir.mkdir()
    lock_fd = os.open(lock_dir, os.O_RDONLY | os.O_DIRECTORY)
    device_root = tmp_path / "device"
    device_root.mkdir()
    calls = {"terminate": 0, "direct": 0, "settle": 0, "journal": 0}
    device = SimpleNamespace(
        device_id="0x744c",
        drm_card="card1",
        device_root=device_root,
        identity=lambda: {"device_id": "0x744c", "drm_card": "card1"},
    )
    environment = {
        "JAX_PLATFORMS": "rocm",
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
        "JAX_ROCM_VISIBLE_DEVICES": "0",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_CLIENT_MEM_FRACTION": "0.075",
        "XLA_FLAGS": _CONTROLLER._COMMAND_BUFFER_FLAG,
    }
    monkeypatch.setattr(_CONTROLLER, "_source_manifest", lambda: {"bound": "yes"})
    monkeypatch.setattr(
        _CONTROLLER, "_systemd_runtime_manifest", lambda: {"user_manager": "mocked"}
    )
    monkeypatch.setattr(
        _CONTROLLER, "_profile_runtime_manifest", lambda: {"psutil": "mocked"}
    )
    monkeypatch.setattr(_CONTROLLER, "_child_environment", lambda *_args: environment)
    monkeypatch.setattr(_CONTROLLER, "_read_vram_total", lambda _root: 25_753_026_560)
    monkeypatch.setattr(
        _CONTROLLER,
        "_profile_command",
        lambda **_kwargs: [sys.executable, "-c", "raise SystemExit(0)"],
    )
    monkeypatch.setattr(
        _CONTROLLER, "_scope_name", lambda *_args: "skyrl-w8a8-runtime-mocked"
    )
    monkeypatch.setattr(_CONTROLLER, "_scope_command", lambda _unit, command: command)

    class Process:
        def poll(self):
            return -signal.SIGKILL

        def wait(self, timeout):
            return -signal.SIGKILL

        def kill(self):
            raise AssertionError("mocked wrapper is already reaped")

    monkeypatch.setattr(_CONTROLLER.subprocess, "Popen", lambda *_a, **_k: Process())
    immediate_cleanup = {
        "method": "immediate_cgroup_wide_sigkill_and_bounded_reap",
        "passed": True,
        "scope_empty": True,
        "wrapper_reaped": True,
    }

    def wait_profile(*_args, **_kwargs):
        return -signal.SIGKILL, {
            "passed": False,
            "received_signal": received_signal,
            "termination_reason": termination_reason,
            "cleanup": immediate_cleanup,
        }

    monkeypatch.setattr(_CONTROLLER, "_wait_profile", wait_profile)

    def terminate(_unit):
        calls["terminate"] += 1
        return {"passed": True, "before": {"pids": []}, "after": {"pids": []}}

    monkeypatch.setattr(_CONTROLLER, "_terminate_scope", terminate)
    monkeypatch.setattr(
        _CONTROLLER,
        "_scope_state",
        lambda _unit: {
            "pids": [],
            "ActiveState": "failed",
            "ControlGroup": "",
        },
    )

    def direct(_unit):
        calls["direct"] += 1
        return {"passed": True, "final_pids": []}

    monkeypatch.setattr(_CONTROLLER, "_direct_cgroup_cleanup", direct)
    monkeypatch.setattr(safety, "acquire_qwen35_rocm_launch_lock", lambda: lock_fd)
    monkeypatch.setattr(handoff, "_discover_device", lambda: device)

    def capture(path):
        _write_private(path, "baseline\n")
        return {"status": "passed"}

    def settle(*_args, **_kwargs):
        calls["settle"] += 1
        return {"status": "passed"}

    def journal():
        calls["journal"] += 1
        return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}

    monkeypatch.setattr(handoff, "capture_baseline", capture)
    monkeypatch.setattr(handoff, "settle_handoff", settle)
    monkeypatch.setattr(safety, "require_clean_amdgpu_boot", journal)
    output = io.StringIO()
    try:
        result = _CONTROLLER._run_gate(
            SimpleNamespace(phase="execute", run_dir=run_dir), output, run_dir_fd
        )
    finally:
        os.close(run_dir_fd)

    records = [json.loads(line) for line in output.getvalue().splitlines()]
    postflight = records[-1]
    assert result == expected_returncode
    assert calls == {
        "terminate": terminate_calls,
        "direct": 1,
        "settle": 1,
        "journal": 1,
    }
    assert postflight["record_type"] == "controller_postflight"
    assert postflight["idle_handoff"]["status"] == "passed"
    assert postflight["final_journal"]["amdgpu_boot_clean"] is True
    assert postflight["final_scope_cleanup"]["passed"] is True
    assert postflight["direct_cgroup_cleanup"]["passed"] is True
    with pytest.raises(OSError):
        os.fstat(lock_fd)


def test_execute_signal_after_popen_enters_immediate_wait_cleanup(
    monkeypatch, tmp_path: Path
) -> None:
    import rocm.amdgpu_safety as safety
    import rocm.qwen35_prewarm_handoff as handoff

    run_dir = (tmp_path / "run").resolve()
    run_dir_fd = _CONTROLLER._create_run_directory(run_dir)
    lock_dir = tmp_path / "fake-lock"
    lock_dir.mkdir()
    lock_fd = os.open(lock_dir, os.O_RDONLY | os.O_DIRECTORY)
    device_root = tmp_path / "device"
    device_root.mkdir()
    calls = {
        "popen": 0,
        "post_popen_signal": 0,
        "terminate": 0,
        "direct": 0,
        "settle": 0,
        "journal": 0,
    }
    device = SimpleNamespace(
        device_id="0x744c",
        drm_card="card1",
        device_root=device_root,
        identity=lambda: {"device_id": "0x744c", "drm_card": "card1"},
    )
    environment = {
        "JAX_PLATFORMS": "rocm",
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
        "JAX_ROCM_VISIBLE_DEVICES": "0",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_CLIENT_MEM_FRACTION": "0.075",
        "XLA_FLAGS": _CONTROLLER._COMMAND_BUFFER_FLAG,
    }
    unit = "skyrl-w8a8-runtime-mocked"
    monkeypatch.setattr(_CONTROLLER, "_source_manifest", lambda: {"bound": "yes"})
    monkeypatch.setattr(
        _CONTROLLER, "_systemd_runtime_manifest", lambda: {"user_manager": "mocked"}
    )
    monkeypatch.setattr(
        _CONTROLLER, "_profile_runtime_manifest", lambda: {"psutil": "mocked"}
    )
    monkeypatch.setattr(_CONTROLLER, "_child_environment", lambda *_args: environment)
    monkeypatch.setattr(_CONTROLLER, "_read_vram_total", lambda _root: 25_753_026_560)
    monkeypatch.setattr(
        _CONTROLLER,
        "_profile_command",
        lambda **_kwargs: [sys.executable, "-c", "raise SystemExit(0)"],
    )
    monkeypatch.setattr(_CONTROLLER, "_scope_name", lambda *_args: unit)
    monkeypatch.setattr(_CONTROLLER, "_scope_command", lambda _unit, command: command)

    class Process:
        reaped = False

        def poll(self):
            return -signal.SIGKILL if self.reaped else None

        def wait(self, timeout):
            assert 0 < timeout <= 1.0
            self.reaped = True
            return -signal.SIGKILL

        def kill(self):
            raise AssertionError("cgroup-wide SIGKILL should reap the wrapper")

    process = Process()

    def popen(*_args, **_kwargs):
        calls["popen"] += 1
        return process

    monkeypatch.setattr(_CONTROLLER.subprocess, "Popen", popen)

    installed_handlers = {}

    def install_signal(signum, handler):
        previous = installed_handlers.get(signum, signal.getsignal(signum))
        installed_handlers[signum] = handler
        return previous

    monkeypatch.setattr(_CONTROLLER.signal, "signal", install_signal)
    original_umask = os.umask
    umask_calls = 0

    def signal_after_popen_umask(mode):
        nonlocal umask_calls
        previous = original_umask(mode)
        umask_calls += 1
        if umask_calls == 2:
            assert calls["popen"] == 1
            handler = installed_handlers[signal.SIGTERM]
            assert callable(handler)
            handler(signal.SIGTERM, None)
            calls["post_popen_signal"] += 1
        return previous

    monkeypatch.setattr(_CONTROLLER.os, "umask", signal_after_popen_umask)

    cleanup_events = []
    scope_states = iter(
        [
            {
                "observed": False,
                "ControlGroup": "",
                "pids": [],
                "ActiveState": "failed",
            },
            {
                "observed": False,
                "ControlGroup": "",
                "pids": [],
                "ActiveState": "failed",
            },
        ]
    )

    def scope_state(_unit, *_args):
        assert cleanup_events and cleanup_events[0] == "systemctl_sigkill"
        cleanup_events.append("scope_state")
        return next(scope_states)

    monkeypatch.setattr(_CONTROLLER, "_scope_state", scope_state)
    direct_kills = []
    monkeypatch.setattr(
        _CONTROLLER,
        "_write_cgroup_kill",
        lambda path: (
            direct_kills.append(path) or {"path": str(path), "bytes_written": 1}
        ),
    )
    systemctl_commands = []
    monkeypatch.setattr(
        _CONTROLLER.subprocess,
        "run",
        lambda command, **_kwargs: (
            cleanup_events.append("systemctl_sigkill")
            or systemctl_commands.append(command)
            or SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
    )
    monkeypatch.setattr(_CONTROLLER, "_systemctl_environment", lambda: {})

    def terminate(_unit):
        calls["terminate"] += 1
        raise AssertionError("execute signal cleanup must never use SIGTERM")

    monkeypatch.setattr(_CONTROLLER, "_terminate_scope", terminate)

    def direct(_unit):
        calls["direct"] += 1
        return {"passed": True, "final_pids": []}

    monkeypatch.setattr(_CONTROLLER, "_direct_cgroup_cleanup", direct)
    monkeypatch.setattr(
        _CONTROLLER,
        "_audit_profile_outputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("failed signal supervision must not audit success evidence")
        ),
    )
    monkeypatch.setattr(safety, "acquire_qwen35_rocm_launch_lock", lambda: lock_fd)
    monkeypatch.setattr(handoff, "_discover_device", lambda: device)

    def capture(path):
        _write_private(path, "baseline\n")
        return {"status": "passed"}

    def settle(*_args, **_kwargs):
        calls["settle"] += 1
        return {"status": "passed"}

    def journal():
        calls["journal"] += 1
        return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}

    monkeypatch.setattr(handoff, "capture_baseline", capture)
    monkeypatch.setattr(handoff, "settle_handoff", settle)
    monkeypatch.setattr(safety, "require_clean_amdgpu_boot", journal)
    output = io.StringIO()
    try:
        result = _CONTROLLER._run_gate(
            SimpleNamespace(phase="execute", run_dir=run_dir), output, run_dir_fd
        )
    finally:
        os.close(run_dir_fd)

    records = [json.loads(line) for line in output.getvalue().splitlines()]
    postflight = records[-1]
    wait_audit = postflight["profile_wait_audit"]
    assert result == 128 + signal.SIGTERM
    assert calls == {
        "popen": 1,
        "post_popen_signal": 1,
        "terminate": 0,
        "direct": 1,
        "settle": 1,
        "journal": 1,
    }
    assert umask_calls == 2
    assert wait_audit["received_signal"] == signal.SIGTERM
    assert wait_audit["termination_reason"] == f"signal_{signal.SIGTERM}"
    assert wait_audit["dispatch_watchdog"]["dispatch_started"] is False
    assert wait_audit["cleanup"]["method"] == (
        "immediate_cgroup_wide_sigkill_and_bounded_reap"
    )
    assert wait_audit["cleanup"]["passed"] is True
    assert postflight["pre_reap_scope_cleanup"] == wait_audit["cleanup"]
    assert postflight["final_scope_cleanup"]["passed"] is True
    assert postflight["direct_cgroup_cleanup"]["passed"] is True
    assert postflight["idle_handoff"]["status"] == "passed"
    assert postflight["final_journal"]["amdgpu_boot_clean"] is True
    assert postflight["cleanup_errors"] == []
    assert process.reaped is True
    assert direct_kills == []
    assert len(systemctl_commands) == 1
    assert "--signal=SIGKILL" in systemctl_commands[0]
    assert all("SIGTERM" not in argument for argument in systemctl_commands[0])
    assert cleanup_events == [
        "systemctl_sigkill",
        "scope_state",
        "scope_state",
    ]
    assert wait_audit["scope_observed"] is False
    assert wait_audit["cleanup"]["initial_scope_exact"] is False
    assert wait_audit["cleanup"]["direct_cgroup_kill"]["passed"] is False
    assert wait_audit["cleanup"]["systemctl_cgroup_kill"]["passed"] is True
    with pytest.raises(OSError):
        os.fstat(lock_fd)


def test_signal_handlers_are_installed_before_global_lock_acquisition(
    monkeypatch, tmp_path: Path
) -> None:
    import rocm.amdgpu_safety as safety

    run_dir = (tmp_path / "run").resolve()
    run_dir_fd = _CONTROLLER._create_run_directory(run_dir)
    monkeypatch.setattr(_CONTROLLER, "_source_manifest", lambda: {"bound": "yes"})
    monkeypatch.setattr(
        _CONTROLLER,
        "_systemd_runtime_manifest",
        lambda: {"user_manager": "mocked-running"},
    )
    monkeypatch.setattr(
        _CONTROLLER,
        "_profile_runtime_manifest",
        lambda: {"psutil_version": "mocked"},
    )
    monkeypatch.setattr(_CONTROLLER, "_child_environment", lambda *_args: {})
    monkeypatch.setattr(
        safety,
        "require_clean_amdgpu_boot",
        lambda: {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []},
    )
    handled = {
        value
        for value in (
            signal.SIGINT,
            signal.SIGTERM,
            getattr(signal, "SIGHUP", None),
        )
        if value is not None
    }
    installed = set()
    original_signal = signal.signal

    def tracking_signal(signum, handler):
        if callable(handler) and getattr(handler, "__name__", "") == "defer_signal":
            installed.add(signum)
        return original_signal(signum, handler)

    monkeypatch.setattr(signal, "signal", tracking_signal)

    def acquire():
        assert installed == handled
        raise RuntimeError("intentional acquisition refusal")

    monkeypatch.setattr(safety, "acquire_qwen35_rocm_launch_lock", acquire)
    try:
        result = _CONTROLLER._run_gate(
            SimpleNamespace(phase="compile", run_dir=run_dir),
            io.StringIO(),
            run_dir_fd,
        )
    finally:
        os.close(run_dir_fd)

    assert result == 1
    assert installed == handled


def test_popen_failure_still_handoffs_rechecks_and_closes_lock(
    monkeypatch, tmp_path: Path
) -> None:
    import rocm.amdgpu_safety as safety
    import rocm.qwen35_prewarm_handoff as handoff

    run_dir = (tmp_path / "run").resolve()
    run_dir_fd = _CONTROLLER._create_run_directory(run_dir)
    lock_dir = tmp_path / "fake-lock"
    lock_dir.mkdir()
    lock_fd = os.open(lock_dir, os.O_RDONLY | os.O_DIRECTORY)
    device_root = tmp_path / "device"
    device_root.mkdir()
    calls = {"settle": 0, "journal": 0, "scope": 0, "direct": 0}
    device = SimpleNamespace(
        device_id="0x744c",
        drm_card="card1",
        device_root=device_root,
        identity=lambda: {"device_id": "0x744c", "drm_card": "card1"},
    )
    monkeypatch.setattr(_CONTROLLER, "_source_manifest", lambda: {"bound": "yes"})
    monkeypatch.setattr(
        _CONTROLLER,
        "_systemd_runtime_manifest",
        lambda: {"user_manager": "mocked-running"},
    )
    monkeypatch.setattr(
        _CONTROLLER,
        "_profile_runtime_manifest",
        lambda: {"psutil_version": "mocked"},
    )
    monkeypatch.setattr(
        _CONTROLLER,
        "_child_environment",
        lambda *_args: {
            "JAX_PLATFORMS": "rocm",
            "ROCR_VISIBLE_DEVICES": "0",
            "HIP_VISIBLE_DEVICES": "0",
            "GPU_DEVICE_ORDINAL": "0",
            "JAX_ROCM_VISIBLE_DEVICES": "0",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_CLIENT_MEM_FRACTION": "0.075",
            "XLA_FLAGS": "--xla_gpu_enable_command_buffer=",
        },
    )
    monkeypatch.setattr(_CONTROLLER, "_read_vram_total", lambda _root: 25_753_026_560)
    monkeypatch.setattr(
        _CONTROLLER,
        "_profile_command",
        lambda **_kwargs: [sys.executable, "-c", "raise SystemExit(0)"],
    )
    monkeypatch.setattr(_CONTROLLER, "_scope_name", lambda: "mocked-scope")
    monkeypatch.setattr(_CONTROLLER, "_scope_command", lambda _unit, command: command)

    def terminate(_unit):
        calls["scope"] += 1
        return {"passed": True, "before": {"pids": []}, "after": {"pids": []}}

    def direct(_unit):
        calls["direct"] += 1
        return {"passed": True, "final_pids": []}

    monkeypatch.setattr(_CONTROLLER, "_terminate_scope", terminate)
    monkeypatch.setattr(_CONTROLLER, "_direct_cgroup_cleanup", direct)
    monkeypatch.setattr(
        _CONTROLLER.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("intentional Popen failure")
        ),
    )
    monkeypatch.setattr(safety, "acquire_qwen35_rocm_launch_lock", lambda: lock_fd)
    monkeypatch.setattr(handoff, "_discover_device", lambda: device)

    def capture(path):
        _write_private(path, "baseline\n")
        return {"status": "passed"}

    def settle(*_args, **_kwargs):
        calls["settle"] += 1
        return {"status": "passed"}

    def journal():
        calls["journal"] += 1
        return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}

    monkeypatch.setattr(handoff, "capture_baseline", capture)
    monkeypatch.setattr(handoff, "settle_handoff", settle)
    monkeypatch.setattr(safety, "require_clean_amdgpu_boot", journal)
    output = io.StringIO()
    try:
        result = _CONTROLLER._run_gate(
            SimpleNamespace(phase="compile", run_dir=run_dir), output, run_dir_fd
        )
    finally:
        os.close(run_dir_fd)

    assert result == 1
    assert calls == {"settle": 1, "journal": 1, "scope": 2, "direct": 1}
    with pytest.raises(OSError):
        os.fstat(lock_fd)
    postflight = json.loads(output.getvalue().splitlines()[-1])
    assert postflight["operation_error_type"] == "OSError"
    assert postflight["cleanup_errors"] == []


def test_controller_top_level_is_standard_library_only() -> None:
    module = ast.parse(_CONTROLLER_PATH.read_text(encoding="utf-8"))
    imports = [
        node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    roots = {alias.name.partition(".")[0] for node in imports for alias in node.names}
    assert not roots & {"jax", "jaxlib", "numpy", "ml_dtypes", "skyrl", "rocm"}


def test_controller_source_has_no_graph_capture_or_replay_api() -> None:
    source = _CONTROLLER_PATH.read_text(encoding="utf-8")
    forbidden = (
        "hipGraph",
        "cudaGraph",
        "capture_begin",
        "capture_end",
        "stream_capture",
        "command_buffer(",
    )
    assert not any(token in source for token in forbidden)
