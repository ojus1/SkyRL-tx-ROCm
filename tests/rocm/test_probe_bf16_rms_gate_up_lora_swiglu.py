from __future__ import annotations

import inspect
import json
import math
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ml_dtypes import bfloat16

from rocm import probe_bf16_rms_gate_up_lora_swiglu as probe


def _valid_errors() -> dict:
    common = {
        "relative_l2": 0.0,
        "cosine_similarity": 1.0,
        "max_absolute": 0.0,
        "finite": True,
    }
    return {
        "output": dict(common),
        "dx": dict(common),
        "d_lora_a": dict(common),
        "d_lora_b": dict(common),
    }


def _clear_accelerator_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    prefixes = (
        "JAX_",
        "XLA_",
        "HSA_",
        "HIP_",
        "ROCM_",
        "ROCR_",
        "AMD_",
        "GPU_",
        "CUDA_",
        "TF_XLA_",
        "TRITON_",
        "PJRT_",
        "ROCBLAS_",
        "HIPBLASLT_",
        "TENSILE_",
    )
    for name in list(os.environ):
        if name.startswith(prefixes) or name in (
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
        ):
            monkeypatch.delenv(name, raising=False)


def test_exact_contract_fixes_geometry_target_limits_capture_and_gates() -> None:
    assert probe._exact_contract() == {
        "case": "qwen35_b1_t64_bf16_rms_gate_up_lora_swiglu",
        "geometry": {
            "batch_size": 1,
            "sequence_length": 64,
            "rows": 64,
            "in_features": 2560,
            "physical_gate_up_features": 18432,
            "product_features": 9216,
            "rank": 8,
            "dtype": "bfloat16",
            "eps": 1e-6,
        },
        "initial_tiles": {"block_m": 16, "pair_block_n": 32, "block_k": 64},
        "target": {
            "drm_card": "card1",
            "pci_id": "1002:744c",
            "architecture": "gfx1100",
            "device_kind": "Radeon RX 7900 XTX",
        },
        "profile_limits": {
            "--max-junction-temp-c": 90.0,
            "--max-gpu-power-watts": 400.0,
            "--max-vram-gib": 24.0,
            "--min-host-available-gib": 0.0,
            "--max-swap-gib": 8.0,
        },
        "capture": {
            "xla_flags": "--xla_gpu_enable_command_buffer=",
            "command_buffers_enabled": False,
            "graph_capture_enabled": False,
        },
        "compile_only": {
            "reference_executable_invocations": 0,
            "candidate_executable_invocations": 0,
        },
        "forward_once": {
            "compiled_programs": ["candidate_forward"],
            "reference_executable_invocations": 0,
            "candidate_forward_executable_invocations": 1,
            "candidate_forward_and_vjp_executable_invocations": 0,
            "watchdog_seconds": 5.0,
            "required_output_shape": [1, 64, 9216],
            "required_output_dtype": "bfloat16",
            "requires_exact_clean_compile_evidence": True,
            "requires_completed_compile_profile_summary": True,
            "requires_cgroup_wide_timeout_kill": True,
            "warmups": 0,
            "iterations": 0,
            "requires_finite_host_output": True,
            "performance_qualification": False,
        },
        "execute_gates": {
            "relative_l2_limit_exclusive": 0.03,
            "output_cosine_limit_inclusive": 0.9999,
            "minimum_forward_and_vjp_speedup": 1.10,
            "minimum_rematerialized_stage_speedup": 1.15,
            "deterministic_repeat_required": True,
        },
        "authorizes_default_model_enablement": False,
    }


def test_numerics_gate_uses_exclusive_three_percent_for_every_result() -> None:
    assert probe._numerics_gate_passed(_valid_errors())

    for name in ("output", "dx", "d_lora_a", "d_lora_b"):
        errors = _valid_errors()
        errors[name]["relative_l2"] = 0.03
        assert not probe._numerics_gate_passed(errors)

    output_cosine = _valid_errors()
    output_cosine["output"]["cosine_similarity"] = 0.9998
    assert not probe._numerics_gate_passed(output_cosine)

    gradient_cosine_is_report_only = _valid_errors()
    gradient_cosine_is_report_only["dx"]["cosine_similarity"] = 0.0
    assert probe._numerics_gate_passed(gradient_cosine_is_report_only)

    nonfinite = _valid_errors()
    nonfinite["d_lora_b"]["finite"] = False
    assert not probe._numerics_gate_passed(nonfinite)


def test_performance_gate_requires_forward_vjp_and_remat_thresholds() -> None:
    arguments = {
        "forward_vjp_speedup": 1.10,
        "rematerialized_speedup": 1.15,
        "candidate_seconds": [0.01, 0.02],
        "candidate_forward_seconds": [0.005, 0.006],
    }
    assert probe._performance_gate_passed(**arguments)

    for mutation in (
        {"forward_vjp_speedup": 1.0999},
        {"rematerialized_speedup": 1.1499},
        {"forward_vjp_speedup": math.nan},
        {"candidate_seconds": [0.0, 0.01]},
        {"candidate_forward_seconds": [math.inf]},
    ):
        assert not probe._performance_gate_passed(**(arguments | mutation))


def test_parser_requires_double_ack_and_exact_production_geometry(
    tmp_path: Path,
) -> None:
    output = (tmp_path / "probe.jsonl").resolve()
    args = probe._parse_args(["--allow-gpu", "--compile-only", "--output", str(output)])
    assert args.mode == "compile_only"
    assert (
        args.batch_size,
        args.sequence_length,
        args.rows,
        args.in_features,
        args.physical_features,
        args.product_features,
        args.rank,
    ) == (1, 64, 64, 2560, 18432, 9216, 8)
    assert (args.block_m, args.block_n, args.block_k) == (16, 32, 64)
    assert args.eps == 1e-6

    with pytest.raises(SystemExit):
        probe._parse_args(["--compile-only", "--output", str(output)])
    with pytest.raises(SystemExit):
        probe._parse_args(["--allow-gpu", "--output", str(output)])
    with pytest.raises(SystemExit):
        probe._parse_args(
            ["--allow-gpu", "--compile-only", "--execute", "--output", str(output)]
        )

    for flag, value in (
        ("--batch-size", "2"),
        ("--sequence-length", "32"),
        ("--rows", "32"),
        ("--in-features", "1280"),
        ("--physical-features", "9216"),
        ("--product-features", "4608"),
        ("--rank", "4"),
        ("--eps", "1e-5"),
    ):
        with pytest.raises(SystemExit):
            probe._parse_args(
                ["--allow-gpu", "--compile-only", "--output", str(output), flag, value]
            )


def test_parser_keeps_execute_sampling_floor_but_compile_only_runs_zero_samples(
    tmp_path: Path,
) -> None:
    output = (tmp_path / "probe.jsonl").resolve()
    compile_args = probe._parse_args(
        [
            "--allow-gpu",
            "--compile-only",
            "--output",
            str(output),
            "--warmups",
            "0",
            "--iterations",
            "0",
        ]
    )
    assert (compile_args.warmups, compile_args.iterations) == (0, 0)

    execute_args = probe._parse_args(
        ["--allow-gpu", "--execute", "--output", str(output)]
    )
    assert (execute_args.warmups, execute_args.iterations) == (3, 11)
    for flag, value in (("--warmups", "2"), ("--iterations", "10")):
        with pytest.raises(SystemExit):
            probe._parse_args(
                ["--allow-gpu", "--execute", "--output", str(output), flag, value]
            )


def test_parser_requires_exact_guarded_forward_once_contract(tmp_path: Path) -> None:
    output = (tmp_path / "probe.jsonl").resolve()
    progress = (tmp_path / "progress.jsonl").resolve()
    evidence = (tmp_path / "compile.jsonl").resolve()
    profile_summary = (tmp_path / "telemetry.jsonl.summary.json").resolve()
    evidence.write_text("{}\n")
    profile_summary.write_text("{}\n")
    base = [
        "--allow-gpu",
        "--forward-once",
        "--output",
        str(output),
        "--progress-output",
        str(progress),
        "--compile-evidence",
        str(evidence),
        "--compile-profile-summary",
        str(profile_summary),
        "--warmups",
        "0",
        "--iterations",
        "0",
    ]
    args = probe._parse_args(base)
    assert args.mode == "forward_once"
    assert args.progress_output == progress
    assert args.compile_evidence == evidence
    assert args.compile_profile_summary == profile_summary
    assert (args.block_m, args.block_n, args.block_k) == (16, 32, 64)

    for mutation in (
        ["--block-m", "32"],
        ["--block-n", "16"],
        ["--block-k", "32"],
        ["--warmups", "1"],
        ["--iterations", "1"],
    ):
        with pytest.raises(SystemExit):
            probe._parse_args([*base, *mutation])
    with pytest.raises(SystemExit):
        probe._parse_args(
            [
                "--allow-gpu",
                "--forward-once",
                "--output",
                str(output),
                "--warmups",
                "0",
                "--iterations",
                "0",
            ]
        )
    with pytest.raises(SystemExit):
        probe._parse_args(
            [
                "--allow-gpu",
                "--compile-only",
                "--output",
                str(output),
                "--progress-output",
                str(progress),
            ]
        )

    progress.write_text("occupied\n")
    with pytest.raises(SystemExit):
        probe._parse_args(base)
    progress.unlink()
    relative_progress = list(base)
    relative_progress[relative_progress.index(str(progress))] = "progress.jsonl"
    with pytest.raises(SystemExit):
        probe._parse_args(relative_progress)
    missing_evidence = list(base)
    evidence_index = missing_evidence.index("--compile-evidence")
    del missing_evidence[evidence_index : evidence_index + 2]
    with pytest.raises(SystemExit):
        probe._parse_args(missing_evidence)
    missing_summary = list(base)
    summary_index = missing_summary.index("--compile-profile-summary")
    del missing_summary[summary_index : summary_index + 2]
    with pytest.raises(SystemExit):
        probe._parse_args(missing_summary)


def test_parser_rejects_unsupported_or_nondivisible_tiles(tmp_path: Path) -> None:
    output = (tmp_path / "probe.jsonl").resolve()
    for flag, value in (
        ("--block-m", "128"),
        ("--block-n", "48"),
        ("--block-k", "128"),
    ):
        with pytest.raises(SystemExit):
            probe._parse_args(
                ["--allow-gpu", "--compile-only", "--output", str(output), flag, value]
            )

    block_m = probe._parse_args(
        ["--allow-gpu", "--compile-only", "--output", str(output), "--block-m", "32"]
    )
    assert block_m.block_m == 32
    block_k = probe._parse_args(
        ["--allow-gpu", "--compile-only", "--output", str(output), "--block-k", "32"]
    )
    assert block_k.block_k == 32


def test_forward_once_scope_requires_exact_shared_systemd_cgroup(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    cgroup_root = tmp_path / "cgroup"
    parent_pid = 4242
    scope = (
        f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/"
        "app.slice/skyrl-bf16-forward-4242-abc123.scope"
    )
    (proc_root / "self").mkdir(parents=True)
    (proc_root / str(parent_pid)).mkdir(parents=True)
    (proc_root / "self" / "cgroup").write_text(f"0::{scope}\n")
    (proc_root / str(parent_pid) / "cgroup").write_text(f"0::{scope}\n")
    kill_path = cgroup_root / scope.removeprefix("/") / "cgroup.kill"
    kill_path.parent.mkdir(parents=True)
    kill_path.touch()

    manifest, descriptor = probe._require_forward_once_scope(
        parent_pid, proc_root=proc_root, cgroup_root=cgroup_root
    )
    try:
        assert manifest["validated"] is True
        assert manifest["scope_unit"] == "skyrl-bf16-forward-4242-abc123.scope"
        assert manifest["cgroup"] == scope
        assert manifest["timeout_kill_scope"] == "entire_systemd_cgroup"
        assert not os.get_inheritable(descriptor)
    finally:
        os.close(descriptor)

    (proc_root / str(parent_pid) / "cgroup").write_text("0::/other.scope\n")
    with pytest.raises(RuntimeError, match="not in the same cgroup"):
        probe._require_forward_once_scope(
            parent_pid, proc_root=proc_root, cgroup_root=cgroup_root
        )


def test_compile_evidence_is_exactly_source_git_and_zero_invocations_bound(
    tmp_path: Path,
) -> None:
    geometry = {
        "batch_size": 1,
        "sequence_length": 64,
        "rows": 64,
        "in_features": 2560,
        "physical_gate_up_features": 18432,
        "product_features": 9216,
        "rank": 8,
        "dtype": "bfloat16",
        "eps": 1e-6,
        "block_m": 16,
        "pair_block_n": 32,
        "block_k": 64,
    }
    source = {
        name: {"path": f"/repo/{name}.py", "sha256": name * 8}
        for name in ("kernel", "probe", "profiler")
    }
    git = {
        "commit": "a" * 40,
        "branch": "main",
        "status_porcelain": [],
        "clean": True,
    }
    device = {
        "backend": "gpu",
        "device_kind": "Radeon RX 7900 XTX",
        "architecture": "gfx1100",
        "platform_version": "ROCm test",
    }
    packages = {
        "jax": "test",
        "jaxlib": "test",
        "jax-rocm7-pjrt": "test",
        "jax-rocm7-plugin": "test",
        "numpy": "test",
    }
    compilation = {
        name: {"lower_calls": 1, "compile_calls": 1}
        for name in (
            "reference_forward_and_vjp",
            "candidate_forward_and_vjp",
            "reference_forward",
            "candidate_forward",
        )
    }
    payload = {
        "schema_version": 1,
        "mode": "compile_only",
        "passed": True,
        "contract": probe._exact_contract(),
        "geometry": geometry,
        "device": device,
        "compilation": compilation,
        "invocation_contract": {
            "compile_only_zero_candidate_reference_executable_invocations": True,
            "total_executable_invocations": 0,
            "reference_executable_invocations": 0,
            "candidate_executable_invocations": 0,
            "per_program_executable_invocations": {
                name: 0
                for name in (
                    "reference_forward",
                    "candidate_forward",
                    "reference_forward_and_vjp",
                    "candidate_forward_and_vjp",
                )
            },
            "per_program_executable_completions": {
                name: 0
                for name in (
                    "reference_forward",
                    "candidate_forward",
                    "reference_forward_and_vjp",
                    "candidate_forward_and_vjp",
                )
            },
        },
        "source": {**source, "git": git, "packages": packages},
        "preflight": {
            "environment": {
                "XLA_FLAGS_effective": "--xla_gpu_enable_command_buffer=",
                "command_buffers_enabled": False,
                "graph_capture_enabled": False,
            },
            "hardware": {
                "amdgpu_boot_clean": True,
                "kfd_unowned": True,
                "connected_amd_connectors": [],
            },
            "profiler_parent": {
                "validated": True,
                "limits": dict(probe._EXACT_PROFILE_LIMITS),
            },
        },
        "postflight": {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []},
    }
    evidence = tmp_path / "compile.jsonl"
    evidence.write_text(json.dumps(payload) + "\n")
    evidence.chmod(0o600)

    manifest = probe._validate_compile_evidence(
        evidence,
        expected_contract=probe._exact_contract(),
        expected_geometry=geometry,
        expected_source=source,
        expected_git=git,
        expected_device=device,
        expected_packages=packages,
    )
    assert manifest["zero_executable_invocations"] is True
    assert manifest["commit"] == "a" * 40

    payload["invocation_contract"]["total_executable_invocations"] = 1
    evidence.unlink()
    evidence.write_text(json.dumps(payload) + "\n")
    evidence.chmod(0o600)
    with pytest.raises(RuntimeError, match="records executable invocations"):
        probe._validate_compile_evidence(
            evidence,
            expected_contract=probe._exact_contract(),
            expected_geometry=geometry,
            expected_source=source,
            expected_git=git,
            expected_device=device,
            expected_packages=packages,
        )

    payload["invocation_contract"]["total_executable_invocations"] = 0
    payload["compilation"]["candidate_forward"]["lower_calls"] = True
    evidence.unlink()
    evidence.write_text(json.dumps(payload) + "\n")
    evidence.chmod(0o600)
    with pytest.raises(RuntimeError, match="one compile per program"):
        probe._validate_compile_evidence(
            evidence,
            expected_contract=probe._exact_contract(),
            expected_geometry=geometry,
            expected_source=source,
            expected_git=git,
            expected_device=device,
            expected_packages=packages,
        )


def test_private_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    evidence = tmp_path / "duplicate.json"
    evidence.write_text('{"schema_version": 1, "schema_version": 1}\n')
    evidence.chmod(0o600)
    with pytest.raises(RuntimeError, match="not strict JSON"):
        probe._read_strict_private_json(evidence)


def test_compile_profile_summary_must_complete_with_observed_safe_limits(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "result.json"
    evidence.write_text("{}\n")
    evidence.chmod(0o600)
    evidence_time = 1_700_000_000_000_000_000
    os.utime(evidence, ns=(evidence_time, evidence_time))
    evidence_manifest = {
        "mtime_ns": evidence_time,
    }
    summary_path = tmp_path / "telemetry.jsonl.summary.json"
    summary = {
        "record_type": "summary",
        "status": "completed",
        "returncode": 0,
        "received_signal": None,
        "kernel_log_available": True,
        "baseline_samples": 10,
        "measured_samples": 20,
        "samples": 31,
        "metrics": {
            "gpu_junction_temp_c": {"measured_max": 60.0},
            "gpu_power_watts": {"measured_max": 200.0},
            "vram_used_bytes": {"measured_max": 20 * 1024**3},
            "host_swap_used_bytes": {"measured_max": 4096},
        },
    }
    summary_path.write_text(json.dumps(summary) + "\n")
    summary_path.chmod(0o600)
    summary_time = evidence_time + 1_000_000
    os.utime(summary_path, ns=(summary_time, summary_time))

    manifest = probe._validate_compile_profile_summary(
        summary_path,
        evidence_path=evidence,
        evidence_manifest=evidence_manifest,
    )
    assert manifest["status"] == "completed"
    assert manifest["returncode"] == 0
    assert manifest["observed_maxima"]["gpu_power_watts"] == 200.0

    summary["metrics"]["gpu_power_watts"]["measured_max"] = 400.1
    summary_path.unlink()
    summary_path.write_text(json.dumps(summary) + "\n")
    summary_path.chmod(0o600)
    os.utime(summary_path, ns=(summary_time, summary_time))
    with pytest.raises(RuntimeError, match="gpu_power_watts"):
        probe._validate_compile_profile_summary(
            summary_path,
            evidence_path=evidence,
            evidence_manifest=evidence_manifest,
        )


def test_environment_sets_only_exact_rocm_device_and_disables_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_accelerator_environment(monkeypatch)
    result = probe._configure_environment()
    assert result["JAX_PLATFORMS"] == "rocm"
    assert result["ROCR_VISIBLE_DEVICES"] == "0"
    assert result["HIP_VISIBLE_DEVICES"] == "0"
    assert result["GPU_DEVICE_ORDINAL"] == "0"
    assert result["XLA_FLAGS_effective"] == "--xla_gpu_enable_command_buffer="
    assert result["command_buffers_enabled"] is False
    assert result["graph_capture_enabled"] is False
    assert os.environ["XLA_FLAGS"] == "--xla_gpu_enable_command_buffer="


def test_environment_rejects_hidden_accelerator_or_library_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_accelerator_environment(monkeypatch)
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    with pytest.raises(RuntimeError, match="HSA_OVERRIDE_GFX_VERSION"):
        probe._configure_environment()

    _clear_accelerator_environment(monkeypatch)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/untrusted")
    with pytest.raises(RuntimeError, match="LD_LIBRARY_PATH"):
        probe._configure_environment()


@pytest.mark.parametrize(
    "flags",
    (
        "--xla_gpu_enable_triton_gemm=true",
        "--xla_gpu_enable_command_buffer= --xla_gpu_enable_command_buffer=",
        "--xla_gpu_enable_command_buffer= --xla_dump_to=/tmp/hlo",
    ),
)
def test_environment_rejects_every_nonexact_xla_flag_set(
    monkeypatch: pytest.MonkeyPatch, flags: str
) -> None:
    _clear_accelerator_environment(monkeypatch)
    monkeypatch.setenv("XLA_FLAGS", flags)
    with pytest.raises(RuntimeError, match="rejects inherited XLA_FLAGS"):
        probe._configure_environment()


def test_exact_card_identity_requires_card1_navi31_amdgpu(tmp_path: Path) -> None:
    device = tmp_path / "0000:03:00.0"
    device.mkdir()
    (device / "vendor").write_text("0x1002\n")
    (device / "device").write_text("0x744c\n")
    driver = tmp_path / "amdgpu"
    driver.mkdir()
    (device / "driver").symlink_to(driver, target_is_directory=True)

    identity = probe._require_exact_card_identity(device)
    assert identity["pci_vendor"] == "0x1002"
    assert identity["pci_device"] == "0x744c"
    assert identity["driver"] == "amdgpu"
    assert identity["architecture"] == "gfx1100"

    (device / "device").write_text("0x73bf\n")
    with pytest.raises(RuntimeError, match="1002:744c/gfx1100"):
        probe._require_exact_card_identity(device)


def test_compile_only_workload_returns_before_any_executable_invocation() -> None:
    class MustNotRun:
        def __call__(self, *_arguments):
            raise AssertionError("compile-only invoked an executable")

    result = probe._run_compiled_workload(
        mode="compile_only",
        warmups=100,
        iterations=100,
        executables={
            "reference_forward": MustNotRun(),
            "candidate_forward": MustNotRun(),
            "reference_forward_and_vjp": MustNotRun(),
            "candidate_forward_and_vjp": MustNotRun(),
        },
        step_arguments=None,
        forward_arguments=None,
    )
    assert (
        result["compile_only_zero_candidate_reference_executable_invocations"] is True
    )
    assert result["invocation_counts"] == {
        "reference_forward": 0,
        "candidate_forward": 0,
        "reference_forward_and_vjp": 0,
        "candidate_forward_and_vjp": 0,
    }
    assert result["warmup_orders"] == []
    assert result["measurement_orders"] == []


def test_forward_once_plan_and_workload_run_only_one_candidate_forward(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker = object()
    plan = probe._compilation_plan(
        mode="forward_once",
        reference_step=object(),
        candidate_step=object(),
        reference_forward=object(),
        candidate_forward=marker,
        step_arguments=(object(),),
        forward_arguments=("forward",),
    )
    assert plan == {"candidate_forward": (marker, ("forward",))}

    events: list[str] = []
    watchdog = {
        "armed": False,
        "arm_sent": False,
        "settled": False,
    }

    def start_watchdog(_cgroup_kill_fd):
        events.append("watchdog_started")
        return watchdog

    def arm_watchdog(state):
        state["arm_sent"] = True
        state["armed"] = True
        state["cgroup_wide_timeout_kill"] = True
        state["armed_monotonic_ns"] = 100
        state["deadline_monotonic_ns"] = 5_000_000_100
        state["armed_ack_received_monotonic_ns"] = 101
        events.append("watchdog_armed")

    def settle_watchdog(state, *, invocation_completed):
        assert invocation_completed is True
        state["settled"] = True
        events.append("watchdog_settled")

    calls = 0

    def executable(argument):
        nonlocal calls
        calls += 1
        assert argument == "input"
        events.append("candidate_invoked")
        return np.ones(probe._FORWARD_ONCE_OUTPUT_SHAPE, dtype=bfloat16)

    monkeypatch.setattr(probe, "_start_forward_once_watchdog", start_watchdog)
    monkeypatch.setattr(probe, "_arm_forward_once_watchdog", arm_watchdog)
    monkeypatch.setattr(probe, "_settle_forward_once_watchdog", settle_watchdog)
    monkeypatch.setattr(probe, "_block_tree", lambda value: value)
    progress = tmp_path / "progress.jsonl"
    kill_path = tmp_path / "cgroup.kill"
    kill_path.touch()
    kill_fd = os.open(kill_path, os.O_WRONLY)
    try:
        result = probe._run_forward_once_workload(
            executable=executable,
            arguments=("input",),
            progress_output=progress,
            binding={"tiles": {"block_m": 16, "pair_block_n": 32, "block_k": 64}},
            cgroup_kill_fd=kill_fd,
        )
    finally:
        os.close(kill_fd)

    assert calls == 1
    assert result["invocation_counts"] == {
        "reference_forward": 0,
        "candidate_forward": 1,
        "reference_forward_and_vjp": 0,
        "candidate_forward_and_vjp": 0,
    }
    assert result["invocation_completion_counts"] == result["invocation_counts"]
    assert result["host_output"] == {
        "shape": [1, 64, 9216],
        "device_dtype": "bfloat16",
        "host_dtype": "float32",
        "element_count": 589824,
        "finite": True,
    }
    assert result["watchdog"] == {
        "external_process": True,
        "timeout_action": "cgroup.kill_then_pidfd_SIGKILL_fallback",
        "cgroup_wide_timeout_kill": True,
        "timeout_seconds": 5.0,
        "armed_monotonic_ns": 100,
        "deadline_monotonic_ns": 5_000_000_100,
        "armed_ack_received_monotonic_ns": 101,
        "dispatch_completed": True,
    }
    assert result["progress"]["record_count"] == 3
    assert result["progress"]["protocol"] == "durable_fsync_jsonl_v1"
    assert events == [
        "watchdog_started",
        "watchdog_armed",
        "candidate_invoked",
        "watchdog_settled",
    ]
    records = [json.loads(line) for line in progress.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "compiled_input_ready",
        "dispatch_started",
        "dispatch_completed",
    ]
    assert records[1]["invocation_attempt_counts"]["candidate_forward"] == 1
    assert records[1]["invocation_completion_counts"]["candidate_forward"] == 0
    assert records[2]["invocation_completion_counts"]["candidate_forward"] == 1
    assert all(record["wall_time_ns"] > 0 for record in records)
    assert all(record["monotonic_time_ns"] > 0 for record in records)
    assert stat.S_IMODE(progress.stat().st_mode) == 0o600


def test_real_forward_once_watchdog_deadline_exists_before_armed_ack(
    tmp_path: Path,
) -> None:
    kill_path = tmp_path / "cgroup.kill"
    kill_path.touch()
    kill_fd = os.open(kill_path, os.O_WRONLY)
    try:
        watchdog = probe._start_forward_once_watchdog(kill_fd)
        probe._arm_forward_once_watchdog(watchdog)
        assert watchdog["armed"] is True
        assert (
            watchdog["deadline_monotonic_ns"] - watchdog["armed_monotonic_ns"]
            == 5_000_000_000
        )
        assert (
            watchdog["armed_monotonic_ns"]
            <= watchdog["armed_ack_received_monotonic_ns"]
            < watchdog["deadline_monotonic_ns"]
        )
        probe._settle_forward_once_watchdog(watchdog, invocation_completed=True)
        assert watchdog["settled"] is True
    finally:
        os.close(kill_fd)


def test_watchdog_pipe_allocation_failure_closes_every_partial_fd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    kill_path = tmp_path / "cgroup.kill"
    kill_path.touch()
    kill_fd = os.open(kill_path, os.O_WRONLY)
    real_pipe = os.pipe
    opened: list[int] = []
    calls = 0

    def failing_pipe():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("synthetic pipe allocation failure")
        pair = real_pipe()
        opened.extend(pair)
        return pair

    monkeypatch.setattr(probe.os, "pipe", failing_pipe)
    fd_count_before = len(list(Path("/proc/self/fd").iterdir()))
    try:
        with pytest.raises(OSError, match="synthetic pipe allocation failure"):
            probe._start_forward_once_watchdog(kill_fd)
        assert len(list(Path("/proc/self/fd").iterdir())) == fd_count_before
        for descriptor in opened:
            with pytest.raises(OSError):
                os.fstat(descriptor)
    finally:
        os.close(kill_fd)


def test_forward_once_exception_never_signals_invocation_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    watchdog = {"arm_sent": False, "settled": False}
    completions: list[bool] = []

    def start_watchdog(_cgroup_kill_fd):
        return watchdog

    def arm_watchdog(state):
        state.update(
            {
                "arm_sent": True,
                "armed": True,
                "armed_monotonic_ns": 100,
                "deadline_monotonic_ns": 5_000_000_100,
                "armed_ack_received_monotonic_ns": 101,
            }
        )

    def settle_watchdog(state, *, invocation_completed):
        completions.append(invocation_completed)
        state["settled"] = True

    def failing_executable(_argument):
        raise RuntimeError("synthetic executable failure")

    monkeypatch.setattr(probe, "_start_forward_once_watchdog", start_watchdog)
    monkeypatch.setattr(probe, "_arm_forward_once_watchdog", arm_watchdog)
    monkeypatch.setattr(probe, "_settle_forward_once_watchdog", settle_watchdog)
    kill_path = tmp_path / "cgroup.kill"
    kill_path.touch()
    kill_fd = os.open(kill_path, os.O_WRONLY)
    try:
        with pytest.raises(RuntimeError, match="synthetic executable failure"):
            probe._run_forward_once_workload(
                executable=failing_executable,
                arguments=(object(),),
                progress_output=tmp_path / "failed-progress.jsonl",
                binding={"test": "exception-is-not-completion"},
                cgroup_kill_fd=kill_fd,
            )
    finally:
        os.close(kill_fd)
    assert completions == [False]
    records = [
        json.loads(line)
        for line in (tmp_path / "failed-progress.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == [
        "compiled_input_ready",
        "dispatch_started",
    ]
    assert records[-1]["invocation_completion_counts"]["candidate_forward"] == 0


def test_forward_once_watchdog_timeout_sigkills_only_disposable_subprocess(
    tmp_path: Path,
) -> None:
    progress = tmp_path / "timeout-progress.jsonl"
    child_source = """
import os
import sys
import time
from pathlib import Path
from rocm import probe_bf16_rms_gate_up_lora_swiglu as probe

probe._FORWARD_ONCE_WATCHDOG_SECONDS = 0.15
probe._block_tree = lambda value: value
cgroup_kill_fd = os.open(sys.argv[2], os.O_WRONLY)

def blocked_executable(_argument):
    time.sleep(10.0)

probe._run_forward_once_workload(
    executable=blocked_executable,
    arguments=(object(),),
    progress_output=Path(sys.argv[1]),
    binding={"test": "disposable-watchdog-timeout"},
    cgroup_kill_fd=cgroup_kill_fd,
)
"""
    kill_path = tmp_path / "cgroup.kill"
    kill_path.touch()
    result = subprocess.run(
        [sys.executable, "-c", child_source, str(progress), str(kill_path)],
        cwd=Path(probe.__file__).resolve().parent.parent,
        check=False,
        capture_output=True,
        text=True,
        timeout=5.0,
    )

    assert result.returncode == -signal.SIGKILL
    records = [json.loads(line) for line in progress.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "compiled_input_ready",
        "dispatch_started",
    ]
    assert records[-1]["invocation_attempt_counts"]["candidate_forward"] == 1
    assert records[-1]["invocation_completion_counts"]["candidate_forward"] == 0
    assert stat.S_IMODE(progress.stat().st_mode) == 0o600


def test_execute_workload_alternates_warmups_and_samples_and_repeats_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class Executable:
        def __init__(self, name: str):
            self.name = name

        def __call__(self, *_arguments):
            calls.append(self.name)
            return self.name

    monkeypatch.setattr(probe, "_block_tree", lambda value: value)
    result = probe._run_compiled_workload(
        mode="execute",
        warmups=1,
        iterations=3,
        executables={
            name: Executable(name)
            for name in (
                "reference_forward",
                "candidate_forward",
                "reference_forward_and_vjp",
                "candidate_forward_and_vjp",
            )
        },
        step_arguments=(object(),),
        forward_arguments=(object(),),
    )
    assert result["warmup_orders"] == [
        {
            "forward_and_vjp": ("reference", "candidate"),
            "forward": ("candidate", "reference"),
        }
    ]
    assert result["measurement_orders"]["forward_and_vjp"] == [
        ("candidate", "reference"),
        ("reference", "candidate"),
        ("candidate", "reference"),
    ]
    assert result["measurement_orders"]["forward"] == [
        ("reference", "candidate"),
        ("candidate", "reference"),
        ("reference", "candidate"),
    ]
    assert result["invocation_counts"] == {
        "reference_forward": 4,
        "candidate_forward": 4,
        "reference_forward_and_vjp": 5,
        "candidate_forward_and_vjp": 6,
    }
    assert calls[-3:] == [
        "reference_forward_and_vjp",
        "candidate_forward_and_vjp",
        "candidate_forward_and_vjp",
    ]


def test_private_output_is_exclusive_and_owner_only(tmp_path: Path) -> None:
    output = tmp_path / "result.jsonl"
    probe._private_write(output, {"passed": True})
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        probe._private_write(output, {"passed": False})


def test_source_orders_fresh_environment_guard_and_clean_boot_postflight() -> None:
    main_source = inspect.getsource(probe.main)
    assert main_source.index("preloaded =") < main_source.index(
        "_configure_environment()"
    )
    assert main_source.index("_configure_environment()") < main_source.index("_run(")
    assert main_source.index("_open_private_file_descriptor(") < main_source.index(
        "_run("
    )
    assert main_source.index("_run(") < main_source.index(
        "_write_reserved_private_output("
    )
    assert "guarded_qwen35_rocm_process" in main_source
    assert "finally:" in main_source
    assert "postflight = safety_module.require_clean_amdgpu_boot()" in main_source

    workload_source = inspect.getsource(probe._run_compiled_workload)
    assert workload_source.index('if mode == "compile_only":') < workload_source.index(
        "def invoke("
    )
    assert "executables[key](*arguments)" in workload_source
