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
            "performance_gates_passed": True,
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
    assert full["smoke_attestation"] == smoke
    assert full["recommend_for_opt_in_integration"] is False
    assert json.loads(output.read_text()) == full
    assert output.stat().st_mode & 0o777 == 0o600

    state["smoke"] = {**smoke, "passed": False}
    with pytest.raises(RuntimeError, match="embedded smoke attestation changed"):
        attest.validate_and_attest_benchmark(
            private / "full.json",
            private / "full-summary.json",
            output_path=None,
            expected_mode="benchmark",
            write_output=False,
        )
