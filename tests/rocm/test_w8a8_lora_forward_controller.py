from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import os
import signal
import stat
import subprocess
import sys
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
        (("--phase", "execute"), "execute rung is disabled"),
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
    with pytest.raises(RuntimeError, match="only the compile diagnostic contract"):
        _CONTROLLER._contract("execute")


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
    with pytest.raises(RuntimeError, match="only the compile diagnostic command"):
        _CONTROLLER._profile_command(
            phase="execute", run_dir=run_dir, card="card1", lock_fd=41
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


def test_independent_ir_gate_rejects_signature_tokens_supplied_only_in_comments() -> (
    None
):
    stable = (
        "func.func public @main(%arg0: tensor<1xf32>) -> tensor<1xf32> {\n"
        "  // tensor<3x64xbf16> tensor<64x17xi8> tensor<1x17xbf16> "
        "tensor<64x8xbf16> tensor<8x17xbf16> tensor<f32> tensor<3x17xbf16>\n"
        "  %0 = stablehlo.custom_call @__gpu$xla.gpu.triton() "
        '{backend_config="skyrl_qwen35_w8a8_lora_forward"} : '
        "() -> tensor<1xf32>\n"
        "  return %0 : tensor<1xf32>\n"
        "}\n"
    )
    optimized = (
        "ENTRY %main (%x: f32[1]) -> f32[1] {\n"
        "  // bf16[3,64] s8[64,17] bf16[1,17] bf16[64,8] bf16[8,17] "
        "f32[] bf16[3,17]\n"
        '  ROOT %0 = f32[1] custom-call(), custom_call_target="__gpu$xla.gpu.triton", '
        'op_name="skyrl_qwen35_w8a8_lora_forward"\n'
        "}\n"
    )

    result = _CONTROLLER._independent_ir_gate(stable, optimized)

    assert result["passed"] is False
    assert result["checks"]["stablehlo_exact_public_main_signature"] is False
    assert result["checks"]["optimized_hlo_exact_entry_signature"] is False


def test_evidence_audit_reparses_raw_ir_and_requires_two_measured_samples(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path
    run_dir.chmod(0o700)
    stable_text = (
        "module {\n"
        "  func.func public @main(%arg0: tensor<3x64xbf16>, "
        "%arg1: tensor<64x17xi8>, %arg2: tensor<1x17xbf16>, "
        "%arg3: tensor<64x8xbf16>, %arg4: tensor<8x17xbf16>, "
        "%arg5: tensor<f32>) -> tensor<3x17xbf16> {\n"
        "    %0 = stablehlo.custom_call @__gpu$xla.gpu.triton() "
        '{backend_config = "skyrl_qwen35_w8a8_lora_forward"} : '
        "() -> tensor<3x17xbf16>\n"
        "    return %0 : tensor<3x17xbf16>\n"
        "  }\n"
        "}\n"
    )
    optimized_text = (
        "ENTRY %main (%x: bf16[3,64], %codes: s8[64,17], "
        "%scales: bf16[1,17], %a: bf16[64,8], %b: bf16[8,17], "
        "%scale: f32[]) -> bf16[3,17] {\n"
        '  ROOT %0 = bf16[3,17] custom-call(), custom_call_target="__gpu$xla.gpu.triton", '
        'op_name="skyrl_qwen35_w8a8_lora_forward"\n'
        "}\n"
    )
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
            "gpu_power_watts": 20.0,
            "gpu_junction_temp_c": 39.0,
            "vram_used_bytes": 0,
        },
        {
            "record_type": "sample",
            "phase": "measured",
            "elapsed_seconds": 0.0,
            "gpu_power_watts": 25.0,
            "gpu_junction_temp_c": 40.0,
            "vram_used_bytes": 1024,
        },
        {
            "record_type": "sample",
            "phase": "measured",
            "elapsed_seconds": 0.05,
            "gpu_power_watts": 30.0,
            "gpu_junction_temp_c": 41.0,
            "vram_used_bytes": 2048,
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
        "samples": 3,
        "baseline_samples": 1,
        "measured_samples": 2,
        "returncode": 0,
        "received_signal": None,
        "kernel_log_available": True,
        "kernel_driver_errors": [],
        "metrics": {
            "gpu_power_watts": {"measured_max": 30.0},
            "gpu_junction_temp_c": {"measured_max": 41.0},
            "vram_used_bytes": {"measured_max": 2048.0},
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
        "".join(json.dumps(record) + "\n" for record in telemetry_records[:2]),
    )
    with pytest.raises(RuntimeError, match="at_least_two_measured_samples"):
        _CONTROLLER._audit_profile_outputs(
            run_dir,
            phase="compile",
            profile_returncode=0,
            wait_audit={"passed": True},
            expected_lock_fd=41,
        )


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
