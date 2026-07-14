from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from rocm import attest_bf16_rms_gate_up_lora_swiglu_benchmark as attest
from rocm import probe_bf16_rms_gate_up_lora_swiglu as probe

_EXPECTED_PROGRAM_STRUCTURES = {
    "reference_forward": {"dot_general_count": 3, "pallas_calls": []},
    "candidate_forward": {
        "dot_general_count": 1,
        "pallas_calls": [
            {
                "name": "skyrl_qwen35_bf16_rms_materialize_lora_a_forward",
                "grid": [4],
                "outputs": [
                    {"shape": [64, 2560], "dtype": "bfloat16"},
                    {"shape": [64, 8], "dtype": "bfloat16"},
                    {"shape": [64], "dtype": "float32"},
                    {"shape": [64], "dtype": "float32"},
                ],
            },
            {
                "name": ("skyrl_qwen35_bf16_contiguous_gate_up_lora_swiglu_forward"),
                "grid": [4, 288],
                "outputs": [{"shape": [64, 9216], "dtype": "bfloat16"}],
            },
        ],
    },
    "reference_forward_and_vjp": {
        "dot_general_count": 8,
        "pallas_calls": [],
    },
    "candidate_forward_and_vjp": {
        "dot_general_count": 6,
        "pallas_calls": [
            {
                "name": "skyrl_qwen35_bf16_rms_materialize_lora_a_forward",
                "grid": [4],
                "outputs": [
                    {"shape": [64, 2560], "dtype": "bfloat16"},
                    {"shape": [64, 8], "dtype": "bfloat16"},
                    {"shape": [64], "dtype": "float32"},
                    {"shape": [64], "dtype": "float32"},
                ],
            },
            {
                "name": ("skyrl_qwen35_bf16_contiguous_gate_up_lora_swiglu_" "residual_forward"),
                "grid": [4, 288],
                "outputs": [
                    {"shape": [64, 9216], "dtype": "bfloat16"},
                    {"shape": [64, 18432], "dtype": "bfloat16"},
                ],
            },
        ],
    },
}


def _private_dir(path: Path) -> Path:
    path.chmod(0o700)
    return path


def _write_private_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, allow_nan=False, sort_keys=True) + "\n")
    path.chmod(0o600)
    return path


def _compilation_fixture() -> dict[str, Any]:
    structures = json.loads(json.dumps(_EXPECTED_PROGRAM_STRUCTURES))
    return {
        program: {
            "lower_calls": 1,
            "compile_calls": 1,
            "lower_seconds": 0.1,
            "compile_seconds": 0.2,
            "program_structure": structures[program],
        }
        for program in attest._PROGRAM_ORDER
    }


def _raw_result_fixture(schema_version: Any) -> dict[str, Any]:
    mode = "benchmark_smoke"
    warmups, measurements = attest._BENCHMARK_MODE_COUNTS[mode]
    count = warmups + measurements
    per_program = dict.fromkeys(attest._PROGRAM_ORDER, count)
    progress = {"path": "/tmp/benchmark-smoke.progress.jsonl"}
    gated = {
        "compile_evidence": {"path": "/tmp/compile.json"},
        "compile_profile_summary": {"path": "/tmp/compile-summary.json"},
        "numerics_evidence": {"path": "/tmp/numerics.json"},
        "numerics_profile_summary": {"path": "/tmp/numerics-summary.json"},
        "progress": progress,
        "invocation_attempt_counts": per_program,
        "invocation_completion_counts": per_program,
        "raw_samples_only": True,
        "performance_qualification": False,
    }
    return {
        "schema_version": schema_version,
        "mode": mode,
        "passed": False,
        "probe_completed": True,
        "profile_attested": False,
        "qualification_scope": "isolated_stage_only",
        "authorizes_default_model_enablement": False,
        "recommend_for_opt_in_model_integration": False,
        "contract": attest._expected_contract(),
        "geometry": attest._exact_geometry(),
        "postflight": {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []},
        "device": {
            "backend": "gpu",
            "device_kind": "Radeon RX 7900 XTX",
            "architecture": "gfx1100",
            "platform_version": "ROCm test",
        },
        "source": {},
        "preflight": {},
        "compilation": {},
        mode: gated,
        "invocation_contract": {
            "per_program_executable_invocations": per_program,
            "per_program_executable_completions": per_program,
            "reference_executable_invocations": 2 * count,
            "candidate_executable_invocations": 2 * count,
            "total_executable_invocations": 4 * count,
            "compile_only_zero_candidate_reference_executable_invocations": False,
        },
        "numerics": {
            "executed": False,
            "passed": True,
            "prior_numerics_evidence_verified": True,
            "prior_numerics_evidence": gated["numerics_evidence"],
            "prior_numerics_profile_summary": gated["numerics_profile_summary"],
        },
        "determinism": {"executed": False},
        "performance_gate": {"executed": False, "passed": None},
    }


def _measurement_fixture(mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    warmups, measurements = attest._BENCHMARK_MODE_COUNTS[mode]
    schedule = attest._benchmark_schedule(mode)
    elapsed = {
        "reference_forward": 2.0,
        "candidate_forward": 1.0,
        "reference_forward_and_vjp": 4.0,
        "candidate_forward_and_vjp": 2.0,
    }
    samples = [
        {
            "dispatch_ordinal": 4 * (warmups + item["phase_supercycle"]) + position + 1,
            "phase": "measurement",
            "phase_supercycle": item["phase_supercycle"],
            "global_supercycle": item["global_supercycle"],
            "supercycle_position": position,
            "program": program,
            "elapsed_seconds": elapsed[program],
        }
        for item in schedule[warmups:]
        for position, program in enumerate(item["order"])
    ]
    by_program = {
        program: [sample["elapsed_seconds"] for sample in samples if sample["program"] == program]
        for program in attest._PROGRAM_ORDER
    }
    raw = {
        "measurement": {
            "executed": True,
            "performance_measured": True,
            "performance_qualification": False,
            "raw_samples_only": True,
            "warmup_supercycles": warmups,
            "measured_supercycles": measurements,
            "warmup_orders": [list(item["order"]) for item in schedule[:warmups]],
            "measurement_orders": [list(item["order"]) for item in schedule[warmups:]],
            "raw_samples": samples,
            "raw_samples_by_program": by_program,
        }
    }
    return raw, {"expected_samples": json.loads(json.dumps(samples))}


def _public_stub_data(tmp_path: Path, mode: str, smoke: Any = None):
    result_file = {
        "path": str(tmp_path / f"{mode}.json"),
        "bytes": 1,
        "sha256": "1" * 64,
        "mode": "0600",
        "device": 1,
        "inode": 2,
        "mtime_ns": 3,
    }
    progress_file = {
        "path": str(tmp_path / f"{mode}.progress"),
        "bytes": 1,
        "sha256": "2" * 64,
        "mode": "0600",
        "device": 1,
        "inode": 3,
        "mtime_ns": 2,
    }
    errors = {
        name: {
            "relative_l2": 0.0,
            "cosine_similarity": 1.0,
            "max_absolute": 0.0,
            "finite": True,
        }
        for name in ("forward_output", "output", "dx", "d_lora_a", "d_lora_b")
    }
    gated = {
        "compile_evidence": {"path": str(tmp_path / "compile.json")},
        "compile_profile_summary": {"path": str(tmp_path / "compile-summary.json")},
        "numerics_evidence": {
            "path": str(tmp_path / "numerics.json"),
            "errors": errors,
        },
        "numerics_profile_summary": {"path": str(tmp_path / "numerics-summary.json")},
    }
    if smoke is not None:
        gated["smoke_attestation"] = smoke
    validated = {
        "source": {"probe": {"path": "probe", "sha256": "3" * 64}},
        "git": {"commit": "4" * 40},
        "packages": {"jax": "test"},
        "progress": {
            "file": progress_file,
            "record_count": 54 if mode == "benchmark_smoke" else 326,
        },
    }
    return {}, result_file, gated, smoke, validated


def _file_manifest(path: str, *, inode: int, digest_digit: str) -> dict[str, Any]:
    return {
        "path": path,
        "bytes": 100 + inode,
        "sha256": digest_digit * 64,
        "mode": "0600",
        "device": 7,
        "inode": inode,
        "mtime_ns": 1_000 + inode,
    }


@pytest.mark.parametrize(
    ("mode", "scope_unit"),
    (
        ("numerics_once", "skyrl-bf16-numerics-123-abc.scope"),
        ("benchmark_smoke", "skyrl-bf16-benchmark-smoke-123-abc.scope"),
        ("benchmark", "skyrl-bf16-benchmark-123-abc.scope"),
    ),
)
def test_preflight_accepts_exact_probe_scope_unit_contract(mode: str, scope_unit: str) -> None:
    repo = Path(attest.__file__).resolve().parent.parent
    profiler_path = (repo / "rocm" / "profile_rocm.py").resolve()
    safety_path = (repo / "rocm" / "amdgpu_safety.py").resolve()
    cgroup = f"/user.slice/user-1000.slice/user@1000.service/app.slice/{scope_unit}"
    source_only = {
        "profiler": {
            "path": str(profiler_path),
            "sha256": attest.hashlib.sha256(profiler_path.read_bytes()).hexdigest(),
        }
    }
    preflight = {
        "environment": {
            "JAX_PLATFORMS": "rocm",
            "ROCR_VISIBLE_DEVICES": "0",
            "HIP_VISIBLE_DEVICES": "0",
            "GPU_DEVICE_ORDINAL": "0",
            "XLA_FLAGS_effective": attest._DISABLE_COMMAND_BUFFERS,
            "XLA_FLAGS_original": attest._DISABLE_COMMAND_BUFFERS,
            "command_buffers_enabled": False,
            "graph_capture_enabled": False,
            "inherited": {
                "JAX_PLATFORMS": "rocm",
                "HIP_VISIBLE_DEVICES": "0",
                "XLA_FLAGS": attest._DISABLE_COMMAND_BUFFERS,
            },
        },
        "hardware": {
            "amdgpu_boot_clean": True,
            "kfd_accessible": True,
            "kfd_unowned": True,
            "connected_amd_connectors": [],
            "fatal_amdgpu_events": [],
            "amd_cards": ["card1"],
            "kfd_path": "/dev/kfd",
        },
        "profiler_parent": {
            "validated": True,
            "parent_pid": 123,
            "parent_executable": "/usr/bin/python3",
            "parent_command_python": str((repo / ".venv" / "bin" / "python").absolute()),
            "parent_command_sha256": "a" * 64,
            "profiler_path": str(profiler_path),
            "profiler_sha256": source_only["profiler"]["sha256"],
            "limits": dict(attest._EXACT_PROFILE_LIMITS),
            "timeout_seconds": 600.0,
            "interval_seconds": 0.05,
            "baseline_seconds": 5.0,
            "sensor_grace_seconds": 15.0,
        },
        "card_identity": {
            "drm_card": "card1",
            "pci_vendor": "0x1002",
            "pci_device": "0x744c",
            "driver": "amdgpu",
            "architecture": "gfx1100",
            "pci_bdf": "0000:03:00.0",
            "sysfs_device": "/sys/devices/pci0000:00/0000:03:00.0",
        },
        "safety_source": {
            "path": str(safety_path),
            "sha256": attest.hashlib.sha256(safety_path.read_bytes()).hexdigest(),
        },
        f"{mode}_scope": {
            "validated": True,
            "scope_unit": scope_unit,
            "cgroup": cgroup,
            "profile_parent_same_cgroup": True,
            "cgroup_kill_path": f"/sys/fs/cgroup{cgroup}/cgroup.kill",
            "cgroup_kill_device": 30,
            "cgroup_kill_inode": 456,
            "timeout_kill_scope": "entire_systemd_cgroup",
        },
    }

    validated = attest._validate_preflight(
        preflight,
        mode=mode,
        repo=repo,
        source_only=source_only,
    )

    assert validated["scope"]["scope_unit"] == scope_unit


def test_guarded_scope_pattern_rejects_old_numerics_alias_and_unknown_mode() -> None:
    pattern = attest._guarded_scope_unit_pattern("numerics_once")
    assert attest.re.fullmatch(pattern, "skyrl-bf16-numerics-123-abc.scope")
    assert not attest.re.fullmatch(pattern, "skyrl-bf16-numerics-once-123-abc.scope")
    with pytest.raises(ValueError, match="unsupported guarded mode"):
        attest._guarded_scope_unit_pattern("unknown")


def _determinism_run_fixture(run_id: int) -> dict[str, Any]:
    base = 100 * run_id
    hashes = {
        f"{program}/{name}": attest.hashlib.sha256(f"{program}/{name}".encode()).hexdigest()
        for program, name in attest._CANDIDATE_HASH_POSITIONS
    }
    primal_hash = attest.hashlib.sha256(b"candidate_primal/output").hexdigest()
    hashes["candidate_forward/output"] = primal_hash
    hashes["candidate_forward_and_vjp/output"] = primal_hash
    return {
        "result": _file_manifest(
            f"/private/run-{run_id}/result.json",
            inode=base + 1,
            digest_digit=str(run_id),
        ),
        "progress": {
            "file": _file_manifest(
                f"/private/run-{run_id}/progress.jsonl",
                inode=base + 2,
                digest_digit=str(run_id + 2),
            ),
            "record_count": 10,
            "probe_pid": 1_000 + run_id,
            "watchdog_pids": [2_000 + 10 * run_id + index for index in range(4)],
        },
        "profile": {
            "summary": _file_manifest(
                f"/private/run-{run_id}/telemetry.jsonl.summary.json",
                inode=base + 3,
                digest_digit=str(run_id + 4),
            ),
            "telemetry": _file_manifest(
                f"/private/run-{run_id}/telemetry.jsonl",
                inode=base + 4,
                digest_digit=str(run_id + 6),
            ),
            "runtime": {"python": "test", "rocm": "7.1.1"},
            "gpu": {"card": "card1", "pci_bdf": "0000:03:00.0"},
            "profiler_pid": 3_000 + run_id,
            "profiler_command_sha256": f"{run_id + 6:x}" * 64,
        },
        "source": {"probe": {"path": "/repo/probe.py", "sha256": "a" * 64}},
        "git": {"commit": "b" * 40, "clean": True},
        "packages": {"jax": "test"},
        "device": {"architecture": "gfx1100", "platform_version": "ROCm test"},
        "geometry": attest._exact_geometry(),
        "host_inputs": {"x": {"sha256": "c" * 64}},
        "candidate_hashes": hashes,
        "stable_preflight": {
            "environment": {"XLA_FLAGS_effective": attest._DISABLE_COMMAND_BUFFERS},
            "hardware": {"amdgpu_boot_clean": True},
            "card_identity": {"pci_bdf": "0000:03:00.0"},
            "safety_source": {"sha256": "d" * 64},
        },
        "profiler_timings": {
            "timeout_seconds": 600.0,
            "interval_seconds": 0.05,
            "baseline_seconds": 5.0,
            "sensor_grace_seconds": 15.0,
        },
        "scope": {
            "scope_unit": f"skyrl-bf16-numerics-123-run{run_id}.scope",
            "cgroup_kill_device": 9,
            "cgroup_kill_inode": 4_000 + run_id,
        },
    }


def _determinism_benchmark_gated(first: dict[str, Any]) -> dict[str, Any]:
    return {
        "numerics_evidence": {
            **first["result"],
            "progress": dict(first["progress"]["file"]),
        },
        "numerics_profile_summary": {
            **first["profile"]["summary"],
            "telemetry": dict(first["profile"]["telemetry"]),
        },
    }


def _host_results_fixture() -> dict[str, Any]:
    return {
        program: {
            name: {
                "shape": list(shape),
                "device_dtype": "bfloat16",
                "host_dtype": "float32",
                "element_count": attest.math.prod(shape),
                "bf16_bytes": 2 * attest.math.prod(shape),
                "bf16_sha256": attest.hashlib.sha256(f"{program}/{name}".encode()).hexdigest(),
                "finite": True,
            }
            for name, shape in specs
        }
        for program, specs in attest._RESULT_SPECS.items()
    }


def _watchdog_fixture() -> dict[str, Any]:
    armed = 1_000_000_000
    return {
        "dispatch_ordinal": 1,
        "program": "reference_forward",
        "external_process": True,
        "watchdog_pid": 1234,
        "timeout_action": "cgroup.kill_then_pidfd_SIGKILL_fallback",
        "cgroup_wide_timeout_kill": True,
        "timeout_seconds": 5.0,
        "armed_monotonic_ns": armed,
        "deadline_monotonic_ns": armed + attest._WATCHDOG_NS,
        "armed_ack_received_monotonic_ns": armed + 1,
        "dispatch_completed": True,
    }


def _numerics_profile_fixture(tmp_path: Path) -> dict[str, Any]:
    private = _private_dir(tmp_path)
    result_path = _write_private_json(private / "result.json", {})
    os.utime(result_path, ns=(100, 100))
    _, result_file = attest._read_private_json(result_path)
    packages = {
        "jax": "test-jax",
        "jaxlib": "test-jaxlib",
        "jax-rocm7-plugin": "test-plugin",
        "jax-rocm7-pjrt": "test-pjrt",
    }
    result = {
        "preflight": {
            "profiler_parent": {
                "parent_command_python": "/private/python",
                "interval_seconds": 0.05,
                "baseline_seconds": 2.0,
                "timeout_seconds": 600.0,
                "sensor_grace_seconds": 15.0,
                "parent_pid": 987,
                "parent_command_sha256": "9" * 64,
            },
            "card_identity": {"pci_bdf": "0000:03:00.0"},
        },
        "source": {
            "probe": {"path": "/repo/probe.py"},
            "profiler": {"sha256": "e" * 64},
            "packages": packages,
        },
        "numerics_once": {
            "progress": {"path": str(private / "progress.jsonl")},
            "compile_evidence": {"path": "/private/compile.json"},
            "compile_profile_summary": {"path": "/private/compile-summary.json"},
        },
    }
    progress = {"probe_pid": 1234}
    command = attest._expected_numerics_profile_command(
        result_path=result_path,
        result=result,
    )
    manifest = {
        "record_type": "manifest",
        "interval_seconds": 0.05,
        "baseline_seconds": 2.0,
        "duration_seconds": None,
        "timeout_seconds": 600.0,
        "sensor_grace_seconds": 15.0,
        "terminate_included_on_safety": False,
        "terminate_included_on_abort": False,
        "explicit_processes": {},
        "safety_limits": {
            "max_junction_temp_c": 90.0,
            "max_gpu_power_watts": 400.0,
            "max_vram_bytes": float(24 * 1024**3),
            "min_host_available_bytes": 0.0,
            "max_swap_bytes": float(8 * 1024**3),
        },
        "command_recorded": True,
        "passed_file_descriptor_count": 0,
        "command": command,
        "runtime": {
            "python": attest.sys.version,
            "platform": attest.platform.platform(),
            "rocm": "test-rocm",
            "jax": packages["jax"],
            "jaxlib": packages["jaxlib"],
            "jax_rocm_plugin": packages["jax-rocm7-plugin"],
            "jax_rocm_pjrt": packages["jax-rocm7-pjrt"],
            "script_sha256": result["source"]["profiler"]["sha256"],
            "accelerator_environment": {
                "HIP_VISIBLE_DEVICES": "0",
                "JAX_PLATFORMS": "rocm",
                "XLA_FLAGS": attest._DISABLE_COMMAND_BUFFERS,
            },
        },
        "gpu": {
            "card": "card1",
            "pci_bdf": "0000:03:00.0",
            "vendor_id": "0x1002",
            "device_id": "0x744c",
            "hwmon_name": "amdgpu",
        },
    }
    measured = {
        "record_type": "sample",
        "phase": "measured",
        "wall_time_ns": 3,
        "gpu_junction_temp_c": 70.0,
        "gpu_power_watts": 300.0,
        "vram_used_bytes": float(23 * 1024**3),
        "host_swap_used_bytes": float(2 * 1024**3),
        "processes": {"command": {"root_pid": progress["probe_pid"], "process_count": 1}},
    }
    samples = [
        {
            **measured,
            "phase": "baseline",
            "wall_time_ns": 1,
        },
        {
            **measured,
            "phase": "preflight",
            "wall_time_ns": 2,
        },
        measured,
    ]
    telemetry_path = private / "telemetry.jsonl"
    telemetry_path.write_text("\n".join(json.dumps(item, sort_keys=True) for item in (manifest, *samples)) + "\n")
    telemetry_path.chmod(0o600)
    os.utime(telemetry_path, ns=(200, 200))
    summary = {
        "record_type": "summary",
        "status": "completed",
        "samples": 3,
        "baseline_samples": 1,
        "measured_samples": 1,
        "returncode": 0,
        "received_signal": None,
        "kernel_log_available": True,
        "metrics": {name: {"measured_max": measured[name]} for name in attest._SAFETY_METRIC_LIMITS},
        "processes": {"command": {"pid": progress["probe_pid"]}},
    }
    summary_path = _write_private_json(
        private / "telemetry.jsonl.summary.json",
        summary,
    )
    os.utime(summary_path, ns=(300, 300))
    return {
        "result_path": result_path,
        "result_file": result_file,
        "result": result,
        "progress": progress,
        "manifest": manifest,
        "samples": samples,
        "telemetry_path": telemetry_path,
        "summary": summary,
        "summary_path": summary_path,
    }


def test_contract_matches_probe_and_import_does_not_load_jax() -> None:
    contract = attest._expected_contract()
    assert attest._EVIDENCE_SCHEMA_VERSION == 2
    assert contract == probe._exact_contract()
    assert contract["case"] == ("qwen35_b1_t64_bf16_rms_gate_up_lora_swiglu_contiguous_v1")
    assert contract["initial_tiles"] == {
        "block_m": 16,
        "block_physical_n": 64,
        "block_k": 32,
    }
    assert contract["program_structures"] == _EXPECTED_PROGRAM_STRUCTURES
    assert attest._expected_program_structures() == _EXPECTED_PROGRAM_STRUCTURES
    assert probe._expected_program_structures() == _EXPECTED_PROGRAM_STRUCTURES
    assert attest._exact_geometry() == {
        **contract["geometry"],
        "block_m": 16,
        "block_physical_n": 64,
        "block_k": 32,
    }
    repo = Path(attest.__file__).resolve().parent.parent
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(repo)!r}); "
        "import rocm.attest_bf16_rms_gate_up_lora_swiglu_benchmark; "
        "assert not any(n == 'jax' or n.startswith('jax.') for n in sys.modules)"
    )
    subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code],
        check=True,
        timeout=10,
    )


def test_raw_result_schema_version_is_strict_v2_and_rejects_bool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_result = _raw_result_fixture(2)
    result_file = {"path": str(tmp_path / "result.json")}
    monkeypatch.setattr(
        attest,
        "_read_private_json",
        lambda *_args, **_kwargs: (raw_result, result_file),
    )
    monkeypatch.setattr(
        attest,
        "_validate_source_and_runtime",
        lambda *_args, **_kwargs: ({}, {}, {}),
    )
    monkeypatch.setattr(attest, "_validate_preflight", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(attest, "_validate_compilation", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(attest, "_validate_prior_chain", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(attest, "_validate_progress", lambda *_args, **_kwargs: {})

    validated = attest._validate_raw_result(
        result_path=tmp_path / "result.json",
        expected_mode="benchmark_smoke",
        repo=tmp_path,
    )
    assert validated[0] == "benchmark_smoke"
    assert type(raw_result["schema_version"]) is int

    for invalid_version in (1, True, 2.0):
        raw_result["schema_version"] = invalid_version
        with pytest.raises(RuntimeError, match="raw result is not an exact"):
            attest._validate_raw_result(
                result_path=tmp_path / "result.json",
                expected_mode="benchmark_smoke",
                repo=tmp_path,
            )


def test_source_runtime_binds_exact_contiguous_v1_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = Path(attest.__file__).resolve().parent.parent
    paths = {
        "kernel": (repo / "skyrl" / "tx" / "kernels" / "rocm" / "bf16_rms_gate_up_lora_swiglu_contiguous.py"),
        "probe": repo / "rocm" / "probe_bf16_rms_gate_up_lora_swiglu.py",
        "profiler": repo / "rocm" / "profile_rocm.py",
    }
    git = {
        "commit": "4" * 40,
        "branch": "test",
        "status_porcelain": [],
        "clean": True,
    }
    packages = {"jax": "test"}
    source = {
        name: {
            "path": str(path.resolve(strict=True)),
            "sha256": attest.hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in paths.items()
    }
    source.update({"git": git, "packages": packages})
    monkeypatch.setattr(attest, "_current_git_manifest", lambda _repo: git)
    monkeypatch.setattr(attest, "_current_package_versions", lambda: packages)

    source_only, validated_git, validated_packages = attest._validate_source_and_runtime(source, repo=repo)
    assert source_only["kernel"] == source["kernel"]
    assert validated_git == git
    assert validated_packages == packages

    legacy_source = json.loads(json.dumps(source))
    legacy_source["kernel"] = {
        "path": str(repo / "skyrl" / "tx" / "kernels" / "rocm" / "bf16_rms_gate_up_lora_swiglu.py"),
        "sha256": "0" * 64,
    }
    with pytest.raises(RuntimeError, match="kernel source is not this exact source"):
        attest._validate_source_and_runtime(legacy_source, repo=repo)


def test_compilation_requires_exact_program_structure_manifests() -> None:
    compilation = _compilation_fixture()
    assert attest._validate_compilation(compilation) is compilation

    for program in attest._PROGRAM_ORDER:
        tampered = _compilation_fixture()
        tampered[program]["program_structure"]["dot_general_count"] += 1
        with pytest.raises(RuntimeError, match=f"{program} program structure is not exact"):
            attest._validate_compilation(tampered)

    missing = _compilation_fixture()
    del missing["candidate_forward"]["program_structure"]
    with pytest.raises(RuntimeError, match="candidate_forward compilation is not exact"):
        attest._validate_compilation(missing)


def test_private_reader_rejects_group_readable_input(tmp_path: Path) -> None:
    private = _private_dir(tmp_path)
    path = _write_private_json(private / "input.json", {"ok": True})
    payload, manifest = attest._read_private_json(path)
    assert payload == {"ok": True}
    assert manifest["mode"] == "0600"
    path.chmod(0o640)
    with pytest.raises(RuntimeError, match="owner-only"):
        attest._read_private_json(path)


@pytest.mark.parametrize("mode", ("benchmark_smoke", "benchmark"))
def test_measurement_derives_exact_speedups_and_thresholds(mode: str) -> None:
    raw, progress = _measurement_fixture(mode)
    result = attest._validate_measurement(mode=mode, raw_result=raw, progress=progress)
    assert result["forward_speedup"] == 2.0
    assert result["forward_and_vjp_speedup"] == 2.0
    assert result["rematerialized_stage_speedup"] == 2.0
    assert result["performance_gates_passed"] is True
    assert result["qualifying_mode"] is (mode == "benchmark")
    assert result["determinism_required_for_integration"] is True
    assert result["determinism_attested"] is False
    raw["measurement"]["raw_samples"][0]["elapsed_seconds"] = 9.0
    with pytest.raises(RuntimeError, match="measurement manifest"):
        attest._validate_measurement(mode=mode, raw_result=raw, progress=progress)


def test_candidate_bf16_hashes_cover_exact_five_result_positions() -> None:
    host_results = _host_results_fixture()
    hashes = attest._candidate_result_hashes(host_results)
    assert list(hashes) == [f"{program}/{name}" for program, name in attest._CANDIDATE_HASH_POSITIONS]
    assert len(hashes) == 5

    wrong_shape = json.loads(json.dumps(host_results))
    wrong_shape["candidate_forward_and_vjp"]["dx"]["shape"][-1] += 1
    with pytest.raises(RuntimeError, match="candidate_forward_and_vjp/dx result manifest"):
        attest._candidate_result_hashes(wrong_shape)

    missing = json.loads(json.dumps(host_results))
    del missing["candidate_forward_and_vjp"]["d_lora_b"]
    with pytest.raises(RuntimeError, match="candidate_forward_and_vjp result set"):
        attest._candidate_result_hashes(missing)

    malformed_hash = json.loads(json.dumps(host_results))
    malformed_hash["candidate_forward"]["output"]["bf16_sha256"] = "A" * 64
    with pytest.raises(RuntimeError, match="lowercase SHA-256"):
        attest._candidate_result_hashes(malformed_hash)


def test_numerics_watchdog_requires_completed_exact_deadline_binding() -> None:
    watchdog = _watchdog_fixture()
    assert (
        attest._validate_numerics_watchdog(
            watchdog,
            ordinal=1,
            program="reference_forward",
        )
        is watchdog
    )

    incomplete = {**watchdog, "dispatch_completed": False}
    with pytest.raises(RuntimeError, match="watchdog 1 is not exact"):
        attest._validate_numerics_watchdog(
            incomplete,
            ordinal=1,
            program="reference_forward",
        )

    wrong_deadline = {
        **watchdog,
        "deadline_monotonic_ns": watchdog["deadline_monotonic_ns"] + 1,
    }
    with pytest.raises(RuntimeError, match="watchdog 1 ordering"):
        attest._validate_numerics_watchdog(
            wrong_deadline,
            ordinal=1,
            program="reference_forward",
        )


def test_numerics_profile_requires_safe_completion_and_abort_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _numerics_profile_fixture(tmp_path)
    monkeypatch.setattr(attest, "_current_rocm_version", lambda: "test-rocm")
    profile = attest._validate_numerics_profile(
        profile_summary_path=fixture["summary_path"],
        result_path=fixture["result_path"],
        result_file=fixture["result_file"],
        result=fixture["result"],
        progress=fixture["progress"],
    )
    assert profile["observed_maxima"]["gpu_power_watts"] == 300.0
    assert profile["profiler_pid"] == 987
    assert profile["probe_pid"] == 1234

    fixture["manifest"]["terminate_included_on_abort"] = True
    fixture["telemetry_path"].write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in (fixture["manifest"], *fixture["samples"])) + "\n"
    )
    fixture["telemetry_path"].chmod(0o600)
    os.utime(fixture["telemetry_path"], ns=(200, 200))
    with pytest.raises(RuntimeError, match="telemetry manifest is not exact"):
        attest._validate_numerics_profile(
            profile_summary_path=fixture["summary_path"],
            result_path=fixture["result_path"],
            result_file=fixture["result_file"],
            result=fixture["result"],
            progress=fixture["progress"],
        )

    fixture["manifest"]["terminate_included_on_abort"] = False
    fixture["samples"][-1]["gpu_power_watts"] = 401.0
    fixture["summary"]["metrics"]["gpu_power_watts"]["measured_max"] = 401.0
    fixture["telemetry_path"].write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in (fixture["manifest"], *fixture["samples"])) + "\n"
    )
    fixture["telemetry_path"].chmod(0o600)
    os.utime(fixture["telemetry_path"], ns=(200, 200))
    _write_private_json(fixture["summary_path"], fixture["summary"])
    os.utime(fixture["summary_path"], ns=(300, 300))
    with pytest.raises(RuntimeError, match="gpu_power_watts is unsafe"):
        attest._validate_numerics_profile(
            profile_summary_path=fixture["summary_path"],
            result_path=fixture["result_path"],
            result_file=fixture["result_file"],
            result=fixture["result"],
            progress=fixture["progress"],
        )


def test_cross_process_determinism_compares_five_hashes_and_emits_run_evidence() -> None:
    first = _determinism_run_fixture(1)
    second = _determinism_run_fixture(2)
    benchmark_gated = _determinism_benchmark_gated(first)
    result = attest._compare_determinism_runs(
        (first, second),
        benchmark_gated=benchmark_gated,
    )
    assert result["provided"] is True
    assert result["passed"] is True
    assert result["attested"] is True
    assert result["candidate_hash_positions"] == [
        f"{program}/{name}" for program, name in attest._CANDIDATE_HASH_POSITIONS
    ]
    assert len(result["candidate_bf16_sha256"]) == 5
    assert [run["probe_pid"] for run in result["runs"]] == [1001, 1002]


@pytest.mark.parametrize("ordinal", (1, 2))
def test_cross_process_determinism_rejects_within_run_candidate_primal_mismatch(ordinal: int) -> None:
    first = _determinism_run_fixture(1)
    second = _determinism_run_fixture(2)
    runs = (first, second)
    runs[ordinal - 1]["candidate_hashes"]["candidate_forward_and_vjp/output"] = "f" * 64
    benchmark_gated = _determinism_benchmark_gated(first)

    with pytest.raises(RuntimeError, match=rf"run {ordinal} candidate primal BF16 outputs differ"):
        attest._compare_determinism_runs(
            runs,
            benchmark_gated=benchmark_gated,
        )


@pytest.mark.parametrize(
    "stable_field",
    (
        "source",
        "git",
        "packages",
        "device",
        "geometry",
        "host_inputs",
        "stable_preflight",
    ),
)
def test_cross_process_determinism_rejects_stable_manifest_changes(stable_field: str) -> None:
    first = _determinism_run_fixture(1)
    second = _determinism_run_fixture(2)
    second[stable_field] = {"tampered": True}
    benchmark_gated = _determinism_benchmark_gated(first)
    with pytest.raises(RuntimeError, match=rf"differ in {stable_field}"):
        attest._compare_determinism_runs(
            (first, second),
            benchmark_gated=benchmark_gated,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("runtime", "profiler runtime"),
        ("gpu", "profiler GPU identity"),
    ),
)
def test_cross_process_determinism_rejects_profile_identity_changes(
    field: str,
    message: str,
) -> None:
    first = _determinism_run_fixture(1)
    second = _determinism_run_fixture(2)
    second["profile"][field] = {"tampered": True}
    benchmark_gated = _determinism_benchmark_gated(first)
    with pytest.raises(RuntimeError, match=message):
        attest._compare_determinism_runs(
            (first, second),
            benchmark_gated=benchmark_gated,
        )


@pytest.mark.parametrize(
    "field",
    (
        "interval_seconds",
        "baseline_seconds",
        "timeout_seconds",
        "sensor_grace_seconds",
    ),
)
def test_cross_process_determinism_rejects_profiler_timing_changes(field: str) -> None:
    first = _determinism_run_fixture(1)
    second = _determinism_run_fixture(2)
    second["profiler_timings"][field] += 1.0
    benchmark_gated = _determinism_benchmark_gated(first)

    with pytest.raises(RuntimeError, match="differ in profiler_timings"):
        attest._compare_determinism_runs(
            (first, second),
            benchmark_gated=benchmark_gated,
        )


def test_cross_process_determinism_rejects_hash_change_and_wrong_first_pair() -> None:
    first = _determinism_run_fixture(1)
    second = _determinism_run_fixture(2)
    benchmark_gated = _determinism_benchmark_gated(first)
    key = "candidate_forward_and_vjp/d_lora_b"
    second["candidate_hashes"][key] = "f" * 64
    with pytest.raises(RuntimeError, match="candidate BF16 result hashes"):
        attest._compare_determinism_runs(
            (first, second),
            benchmark_gated=benchmark_gated,
        )

    second = _determinism_run_fixture(2)
    wrong_prior = json.loads(json.dumps(benchmark_gated))
    wrong_prior["numerics_evidence"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="first determinism result compact manifest"):
        attest._compare_determinism_runs(
            (first, second),
            benchmark_gated=wrong_prior,
        )

    wrong_progress = json.loads(json.dumps(benchmark_gated))
    wrong_progress["numerics_evidence"]["progress"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="first determinism progress compact manifest"):
        attest._compare_determinism_runs(
            (first, second),
            benchmark_gated=wrong_progress,
        )


@pytest.mark.parametrize(
    "reuse",
    (
        "artifact_path",
        "artifact_identity",
        "probe_pid",
        "profiler_pid",
        "cross_role_pid",
        "profiler_command_sha256",
        "scope_unit",
        "scope_identity",
    ),
)
def test_cross_process_determinism_rejects_reused_artifact_or_process_evidence(
    reuse: str,
) -> None:
    first = _determinism_run_fixture(1)
    second = _determinism_run_fixture(2)
    if reuse == "artifact_path":
        second["progress"]["file"]["path"] = first["result"]["path"]
    elif reuse == "artifact_identity":
        second["progress"]["file"]["device"] = first["result"]["device"]
        second["progress"]["file"]["inode"] = first["result"]["inode"]
    elif reuse in {"probe_pid", "profiler_pid", "profiler_command_sha256"}:
        owner = "progress" if reuse == "probe_pid" else "profile"
        second[owner][reuse] = first[owner][reuse]
    elif reuse == "cross_role_pid":
        second["progress"]["probe_pid"] = first["progress"]["watchdog_pids"][0]
    elif reuse == "scope_unit":
        second["scope"]["scope_unit"] = first["scope"]["scope_unit"]
    else:
        second["scope"]["cgroup_kill_device"] = first["scope"]["cgroup_kill_device"]
        second["scope"]["cgroup_kill_inode"] = first["scope"]["cgroup_kill_inode"]
    benchmark_gated = _determinism_benchmark_gated(first)
    message = "artifacts are not distinct" if reuse.startswith("artifact") else "process/scope evidence"
    with pytest.raises(RuntimeError, match=message):
        attest._compare_determinism_runs(
            (first, second),
            benchmark_gated=benchmark_gated,
        )


def test_cli_requires_exactly_two_complete_determinism_pairs() -> None:
    base = [
        "--result",
        "/private/result.json",
        "--profile-summary",
        "/private/summary.json",
        "--output",
        "/private/attestation.json",
        "--expected-mode",
        "benchmark",
    ]
    assert attest._parse_args(base).determinism_pairs is None
    paired = base + [
        "--determinism-result",
        "/private/run-1/result.json",
        "--determinism-profile-summary",
        "/private/run-1/summary.json",
        "--determinism-result",
        "/private/run-2/result.json",
        "--determinism-profile-summary",
        "/private/run-2/summary.json",
    ]
    assert attest._parse_args(paired).determinism_pairs == (
        (Path("/private/run-1/result.json"), Path("/private/run-1/summary.json")),
        (Path("/private/run-2/result.json"), Path("/private/run-2/summary.json")),
    )
    with pytest.raises(SystemExit) as incomplete:
        attest._parse_args(
            base
            + [
                "--determinism-result",
                "/private/run-1/result.json",
                "--determinism-profile-summary",
                "/private/run-1/summary.json",
            ]
        )
    assert incomplete.value.code == 2
    with pytest.raises(SystemExit) as unpaired:
        attest._parse_args(
            base
            + [
                "--determinism-result",
                "/private/run-1/result.json",
                "--determinism-result",
                "/private/run-2/result.json",
            ]
        )
    assert unpaired.value.code == 2


def test_profile_recomputes_maxima_and_allows_telemetry_before_result(
    tmp_path: Path,
) -> None:
    private = _private_dir(tmp_path)
    result_path = _write_private_json(private / "result.json", {})
    os.utime(result_path, ns=(100, 100))
    _, result_file = attest._read_private_json(result_path)
    repo = Path(attest.__file__).resolve().parent.parent
    gated = {
        "progress": {"path": str(private / "progress.jsonl")},
        "compile_evidence": {"path": str(private / "compile.json")},
        "compile_profile_summary": {"path": str(private / "compile-summary.json")},
        "numerics_evidence": {"path": str(private / "numerics.json")},
        "numerics_profile_summary": {"path": str(private / "numerics-summary.json")},
    }
    raw = {
        "preflight": {
            "profiler_parent": {
                "interval_seconds": 0.05,
                "baseline_seconds": 2.0,
                "timeout_seconds": 600.0,
                "sensor_grace_seconds": 15.0,
            }
        },
        "source": {
            "profiler": {"sha256": attest.hashlib.sha256((repo / "rocm" / "profile_rocm.py").read_bytes()).hexdigest()}
        },
    }
    command = attest._expected_profile_command(
        repo=repo,
        mode="benchmark_smoke",
        result_path=result_path,
        gated=gated,
        smoke_attestation=None,
    )
    manifest = {
        "record_type": "manifest",
        "interval_seconds": 0.05,
        "baseline_seconds": 2.0,
        "duration_seconds": None,
        "timeout_seconds": 600.0,
        "sensor_grace_seconds": 15.0,
        "terminate_included_on_safety": False,
        "terminate_included_on_abort": False,
        "safety_limits": {
            "max_junction_temp_c": 90.0,
            "max_gpu_power_watts": 400.0,
            "max_vram_bytes": float(24 * 1024**3),
            "min_host_available_bytes": 0.0,
            "max_swap_bytes": float(8 * 1024**3),
        },
        "command_recorded": True,
        "passed_file_descriptor_count": 0,
        "command": command,
        "runtime": {
            "script_sha256": raw["source"]["profiler"]["sha256"],
            "accelerator_environment": {
                "HIP_VISIBLE_DEVICES": "0",
                "JAX_PLATFORMS": "rocm",
                "XLA_FLAGS": attest._DISABLE_COMMAND_BUFFERS,
            },
        },
        "gpu": {"card": "card1", "vendor_id": "0x1002", "device_id": "0x744c"},
    }
    base = {
        "record_type": "sample",
        "gpu_junction_temp_c": 50.0,
        "gpu_power_watts": 100.0,
        "vram_used_bytes": 1024.0,
        "host_swap_used_bytes": 0.0,
    }
    samples = [
        {**base, "phase": "baseline", "wall_time_ns": 1},
        {**base, "phase": "preflight", "wall_time_ns": 2},
        {**base, "phase": "measured", "wall_time_ns": 3},
    ]
    telemetry_path = private / "telemetry.jsonl"
    telemetry_path.write_text("\n".join(json.dumps(item, sort_keys=True) for item in (manifest, *samples)) + "\n")
    telemetry_path.chmod(0o600)
    os.utime(telemetry_path, ns=(90, 90))
    summary = {
        "record_type": "summary",
        "status": "completed",
        "samples": 3,
        "baseline_samples": 1,
        "measured_samples": 1,
        "returncode": 0,
        "received_signal": None,
        "kernel_log_available": True,
        "metrics": {name: {"measured_max": base[name]} for name in attest._SAFETY_METRIC_LIMITS},
    }
    summary_path = _write_private_json(private / "telemetry.jsonl.summary.json", summary)
    os.utime(summary_path, ns=(110, 110))
    profile = attest._validate_profile(
        profile_summary_path=summary_path,
        result_path=result_path,
        result_file=result_file,
        raw_result=raw,
        mode="benchmark_smoke",
        repo=repo,
        gated=gated,
        smoke_attestation=None,
    )
    assert profile["observed_maxima"]["gpu_power_watts"] == 100.0
    summary["metrics"]["gpu_power_watts"]["measured_max"] = 99.0
    _write_private_json(private / "tampered-summary.json", summary)
    summary_path.unlink()
    (private / "tampered-summary.json").rename(summary_path)
    summary_path.chmod(0o600)
    os.utime(summary_path, ns=(120, 120))
    with pytest.raises(RuntimeError, match="gpu_power_watts is unsafe"):
        attest._validate_profile(
            profile_summary_path=summary_path,
            result_path=result_path,
            result_file=result_file,
            raw_result=raw,
            mode="benchmark_smoke",
            repo=repo,
            gated=gated,
            smoke_attestation=None,
        )


def test_full_profile_command_binds_smoke_before_tile_flags(tmp_path: Path) -> None:
    repo = Path(attest.__file__).resolve().parent.parent
    gated = {
        "progress": {"path": "/tmp/full-progress.jsonl"},
        "compile_evidence": {"path": "/tmp/compile.json"},
        "compile_profile_summary": {"path": "/tmp/compile-summary.json"},
        "numerics_evidence": {"path": "/tmp/numerics.json"},
        "numerics_profile_summary": {"path": "/tmp/numerics-summary.json"},
    }
    smoke = {
        "result": {"path": "/tmp/smoke.json"},
        "profile": {"summary": {"path": "/tmp/smoke-summary.json"}},
    }
    command = attest._expected_profile_command(
        repo=repo,
        mode="benchmark",
        result_path=tmp_path / "full.json",
        gated=gated,
        smoke_attestation=smoke,
    )
    smoke_index = command.index("--smoke-evidence")
    block_index = command.index("--block-m")
    assert command[smoke_index:block_index] == [
        "--smoke-evidence",
        "/tmp/smoke.json",
        "--smoke-profile-summary",
        "/tmp/smoke-summary.json",
    ]
    assert command[block_index : block_index + 6] == [
        "--block-m",
        "16",
        "--block-physical-n",
        "64",
        "--block-k",
        "32",
    ]
    assert command[-4:] == ["--warmups", "8", "--iterations", "32"]


def test_public_callable_is_only_passing_artifact_and_recurses_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = _private_dir(tmp_path)
    state: dict[str, Any] = {"smoke": None}
    performance_passes = {"value": True}

    def fake_raw(**kwargs):
        mode = kwargs["expected_mode"]
        smoke = state["smoke"] if mode == "benchmark" else None
        return (mode, *_public_stub_data(private, mode, smoke))

    monkeypatch.setattr(attest, "_validate_raw_result", fake_raw)
    monkeypatch.setattr(
        attest,
        "_validate_profile",
        lambda **kwargs: {
            "summary": {"path": str(private / f"{kwargs['mode']}-summary.json")},
            "status": "completed",
        },
    )
    monkeypatch.setattr(
        attest,
        "_validate_measurement",
        lambda **kwargs: {
            "performance_gates_passed": (performance_passes["value"] if kwargs["mode"] == "benchmark" else True),
            "qualifying_mode": kwargs["mode"] == "benchmark",
        },
    )
    smoke = attest.validate_and_attest_benchmark(
        private / "smoke.json",
        private / "smoke-summary.json",
        output_path=None,
        expected_mode="benchmark_smoke",
        write_output=False,
    )
    assert smoke["schema_version"] == 2
    assert type(smoke["schema_version"]) is int
    assert smoke["attestation_type"] == ("bf16_rms_gate_up_lora_swiglu_contiguous_v1_" "benchmark_profile_attestation")
    assert smoke["passed"] is True
    assert smoke["attestation_passed"] is True
    assert smoke["performance_qualified"] is False
    assert smoke["determinism_attested"] is False
    assert smoke["determinism"]["provided"] is False
    assert smoke["profile_attested"] is True
    assert smoke["recommend_for_opt_in_integration"] is False

    state["smoke"] = smoke
    output = private / "full-attestation.json"
    full = attest.validate_and_attest_benchmark(
        private / "full.json",
        private / "full-summary.json",
        output_path=output,
        expected_mode="benchmark",
        write_output=True,
    )
    assert full["passed"] is True
    assert full["timing_qualified"] is True
    assert full["performance_qualified"] is False
    assert full["determinism_attested"] is False
    assert full["determinism"]["provided"] is False
    assert full["smoke_attestation"] == smoke
    assert full["recommend_for_opt_in_integration"] is False
    assert json.loads(output.read_text()) == full
    assert output.stat().st_mode & 0o777 == 0o600

    determinism = {
        "provided": True,
        "passed": True,
        "attested": True,
        "protocol": "test",
    }
    monkeypatch.setattr(
        attest,
        "_validate_cross_process_determinism",
        lambda **_kwargs: determinism,
    )
    promoted = attest.validate_and_attest_benchmark(
        private / "full.json",
        private / "full-summary.json",
        output_path=None,
        expected_mode="benchmark",
        write_output=False,
        determinism_pairs=(
            (private / "numerics-1.json", private / "numerics-1-summary.json"),
            (private / "numerics-2.json", private / "numerics-2-summary.json"),
        ),
    )
    assert promoted["timing_qualified"] is True
    assert promoted["determinism_attested"] is True
    assert promoted["performance"]["determinism_attested"] is True
    assert promoted["performance_qualified"] is True
    assert promoted["recommend_for_opt_in_integration"] is True
    assert promoted["determinism"] == determinism

    performance_passes["value"] = False
    timing_failed = attest.validate_and_attest_benchmark(
        private / "full.json",
        private / "full-summary.json",
        output_path=None,
        expected_mode="benchmark",
        write_output=False,
        determinism_pairs=(
            (private / "numerics-1.json", private / "numerics-1-summary.json"),
            (private / "numerics-2.json", private / "numerics-2-summary.json"),
        ),
    )
    assert timing_failed["determinism_attested"] is True
    assert timing_failed["timing_qualified"] is False
    assert timing_failed["performance_qualified"] is False
    assert timing_failed["recommend_for_opt_in_integration"] is False

    state["smoke"] = {**smoke, "passed": False}
    with pytest.raises(RuntimeError, match="embedded smoke attestation changed"):
        attest.validate_and_attest_benchmark(
            private / "full.json",
            private / "full-summary.json",
            output_path=None,
            expected_mode="benchmark",
            write_output=False,
        )
