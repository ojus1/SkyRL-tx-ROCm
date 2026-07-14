from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import signal
import stat
import subprocess
import sys
import time
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
        "forward_output": dict(common),
        "output": dict(common),
        "dx": dict(common),
        "d_lora_a": dict(common),
        "d_lora_b": dict(common),
    }


_NUMERICS_RESULT_SHAPES = (
    (1, 64, 9216),
    (1, 64, 2560),
    (2560, 8),
    (8, 18432),
)


def _host_bf16_arguments() -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    forward_arguments = tuple(np.zeros((1,), dtype=bfloat16) for _ in range(6))
    cotangent = np.zeros((1,), dtype=bfloat16)
    return (*forward_arguments, cotangent), forward_arguments


def _numerics_result_tree(value: float = 1.0) -> tuple[np.ndarray, ...]:
    return tuple(
        np.full(shape, value, dtype=bfloat16) for shape in _NUMERICS_RESULT_SHAPES
    )


def _mock_host_input_manifest(
    forward_arguments: tuple[object, ...], step_arguments: tuple[object, ...]
) -> dict[str, object]:
    assert len(forward_arguments) == 6
    assert len(step_arguments) == 7
    assert all(left is right for left, right in zip(forward_arguments, step_arguments))
    assert all(type(argument) is np.ndarray for argument in step_arguments)
    assert all(argument.dtype == bfloat16 for argument in step_arguments)
    return {"test_host_bfloat16_inputs": True}


def _mock_benchmark_host_input_manifest(
    forward_arguments: tuple[object, ...], step_arguments: tuple[object, ...]
) -> dict[str, object]:
    assert len(forward_arguments) == 6
    assert len(step_arguments) == 7
    assert all(left is right for left, right in zip(forward_arguments, step_arguments))
    assert all(type(argument) is np.ndarray for argument in step_arguments)
    assert all(argument.dtype == bfloat16 for argument in step_arguments)
    return {
        name: {
            "shape": list(shape),
            "dtype": "bfloat16",
            "element_count": math.prod(shape),
            "bytes": 2 * math.prod(shape),
            "sha256": hashlib.sha256(name.encode()).hexdigest(),
        }
        for name, shape in probe._INPUT_SPECS
    }


class _FakeDevice:
    pass


class _FakeDeviceLeaf:
    def __init__(self, shape: tuple[int, ...], device: _FakeDevice) -> None:
        self.shape = shape
        self.dtype = np.dtype(bfloat16)
        self._device = device
        self.block_until_ready_calls = 0
        self.delete_calls = 0

    def block_until_ready(self) -> _FakeDeviceLeaf:
        self.block_until_ready_calls += 1
        return self

    def devices(self) -> set[_FakeDevice]:
        return {self._device}

    def delete(self) -> None:
        self.delete_calls += 1

    def is_deleted(self) -> bool:
        return self.delete_calls == 1


class _FakeJax:
    def __init__(
        self, device: _FakeDevice, staged_inputs: tuple[_FakeDeviceLeaf, ...]
    ) -> None:
        self.device = device
        self.staged_inputs = staged_inputs
        self.device_put_calls: list[tuple[tuple[object, ...], _FakeDevice]] = []
        self.device_get_calls: list[tuple[_FakeDeviceLeaf, ...]] = []
        self.clear_caches_calls = 0
        self.effects_barrier_calls = 0

    def device_put(
        self, arguments: tuple[object, ...], *, device: _FakeDevice
    ) -> tuple[_FakeDeviceLeaf, ...]:
        assert device is self.device
        self.device_put_calls.append((arguments, device))
        return self.staged_inputs

    def device_get(
        self, arguments: tuple[_FakeDeviceLeaf, ...]
    ) -> tuple[_FakeDeviceLeaf, ...]:
        self.device_get_calls.append(arguments)
        return arguments

    def clear_caches(self) -> None:
        self.clear_caches_calls += 1

    def effects_barrier(self) -> None:
        self.effects_barrier_calls += 1


class _FakeWatchdogProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid


def _install_benchmark_watchdog_mocks(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict[str, object]], list[tuple[str, int, bool | None]]]:
    watchdogs: list[dict[str, object]] = []
    events: list[tuple[str, int, bool | None]] = []

    def start_watchdog(_cgroup_kill_fd: int) -> dict[str, object]:
        identifier = len(watchdogs)
        state: dict[str, object] = {
            "identifier": identifier,
            "process": _FakeWatchdogProcess(10_000 + identifier),
            "arm_sent": False,
            "armed": False,
            "settled": False,
            "cgroup_wide_timeout_kill": True,
            "timeout_seconds": 5.0,
        }
        watchdogs.append(state)
        events.append(("started", identifier, None))
        return state

    def arm_watchdog(state: dict[str, object]) -> None:
        identifier = int(state["identifier"])
        armed_ns = time.monotonic_ns()
        state.update(
            {
                "arm_sent": True,
                "armed": True,
                "armed_monotonic_ns": armed_ns,
                "deadline_monotonic_ns": armed_ns + 5_000_000_000,
                "armed_ack_received_monotonic_ns": armed_ns + 1,
            }
        )
        events.append(("armed", identifier, None))

    def settle_watchdog(
        state: dict[str, object], *, invocation_completed: bool
    ) -> None:
        identifier = int(state["identifier"])
        assert state["settled"] is False
        state["settled"] = True
        events.append(("settled", identifier, invocation_completed))

    monkeypatch.setattr(probe, "_start_forward_once_watchdog", start_watchdog)
    monkeypatch.setattr(probe, "_arm_forward_once_watchdog", arm_watchdog)
    monkeypatch.setattr(probe, "_settle_forward_once_watchdog", settle_watchdog)
    return watchdogs, events


def _install_benchmark_device_mocks(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_FakeJax, _FakeDevice, tuple[_FakeDeviceLeaf, ...]]:
    device = _FakeDevice()
    staged_inputs = tuple(
        _FakeDeviceLeaf(shape, device) for _name, shape in probe._INPUT_SPECS
    )
    jax = _FakeJax(device, staged_inputs)

    def validate_roundtrip(
        roundtrip: object, host_manifest: dict[str, object]
    ) -> dict[str, object]:
        assert type(roundtrip) is tuple
        assert all(
            actual is expected
            for actual, expected in zip(roundtrip, staged_inputs, strict=True)
        )
        assert tuple(host_manifest) == tuple(name for name, _shape in probe._INPUT_SPECS)
        return {"all_hashes_match": True}

    def block_tree(value: object) -> object:
        if type(value) is tuple:
            return tuple(block_tree(leaf) for leaf in value)
        block_until_ready = getattr(value, "block_until_ready", None)
        return block_until_ready() if callable(block_until_ready) else value

    monkeypatch.setattr(probe, "_block_tree", block_tree)
    monkeypatch.setattr(
        probe, "_host_input_manifest", _mock_benchmark_host_input_manifest
    )
    monkeypatch.setattr(
        probe, "_validate_device_input_roundtrip", validate_roundtrip
    )
    return jax, device, staged_inputs


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
        "numerics_once": {
            "compiled_programs": [
                "reference_forward",
                "candidate_forward",
                "reference_forward_and_vjp",
                "candidate_forward_and_vjp",
            ],
            "program_order": [
                "reference_forward",
                "candidate_forward",
                "reference_forward_and_vjp",
                "candidate_forward_and_vjp",
            ],
            "executable_invocations_per_program": 1,
            "watchdog_seconds_per_program": 5.0,
            "requires_host_bfloat16_inputs": True,
            "requires_readonly_host_inputs": True,
            "requires_post_dispatch_input_rehash": True,
            "requires_exact_clean_compile_evidence": True,
            "requires_completed_compile_profile_summary": True,
            "requires_cgroup_wide_timeout_kill": True,
            "relative_l2_limit_exclusive": 0.03,
            "output_cosine_limit_inclusive": 0.9999,
            "gradient_cosine_similarity_report_only": True,
            "warmups": 0,
            "iterations": 0,
            "performance_qualification": False,
        },
        "benchmark_smoke": {
            "compiled_programs": [
                "reference_forward",
                "candidate_forward",
                "reference_forward_and_vjp",
                "candidate_forward_and_vjp",
            ],
            "warmup_supercycles": 2,
            "measured_supercycles": 4,
            "programs_per_supercycle": 4,
            "total_program_invocations": 24,
            "executable_invocations_per_program": 6,
            "watchdog_seconds_per_operation": 5.0,
            "independent_dispatch_watchdogs": 24,
            "independent_operation_watchdogs": 26,
            "guarded_device_input_setup": True,
            "guarded_device_input_teardown": True,
            "requires_prior_numerics_evidence": True,
            "requires_prior_smoke_profile_attestation": False,
            "raw_samples_only": True,
            "performance_qualification": False,
        },
        "benchmark": {
            "compiled_programs": [
                "reference_forward",
                "candidate_forward",
                "reference_forward_and_vjp",
                "candidate_forward_and_vjp",
            ],
            "warmup_supercycles": 8,
            "measured_supercycles": 32,
            "programs_per_supercycle": 4,
            "total_program_invocations": 160,
            "executable_invocations_per_program": 40,
            "watchdog_seconds_per_operation": 5.0,
            "independent_dispatch_watchdogs": 160,
            "independent_operation_watchdogs": 162,
            "guarded_device_input_setup": True,
            "guarded_device_input_teardown": True,
            "requires_prior_numerics_evidence": True,
            "requires_prior_smoke_profile_attestation": True,
            "raw_samples_only": True,
            "performance_qualification": False,
        },
        "execute_gates": {
            "relative_l2_limit_exclusive": 0.03,
            "output_cosine_limit_inclusive": 0.9999,
            "gradient_cosine_similarity_report_only": True,
            "minimum_forward_and_vjp_speedup": 1.10,
            "minimum_rematerialized_stage_speedup": 1.15,
            "deterministic_repeat_required": True,
        },
        "authorizes_default_model_enablement": False,
    }


def test_numerics_gate_uses_exclusive_three_percent_for_every_result() -> None:
    assert probe._numerics_gate_passed(_valid_errors())

    for name in ("forward_output", "output", "dx", "d_lora_a", "d_lora_b"):
        errors = _valid_errors()
        errors[name]["relative_l2"] = 0.03
        assert not probe._numerics_gate_passed(errors)

    for name in ("forward_output", "output"):
        output_cosine = _valid_errors()
        output_cosine[name]["cosine_similarity"] = 0.9998
        assert not probe._numerics_gate_passed(output_cosine)

    gradient_cosine_is_report_only = _valid_errors()
    for name in ("dx", "d_lora_a", "d_lora_b"):
        gradient_cosine_is_report_only[name]["cosine_similarity"] = -1.0
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


def test_parser_allows_zero_compile_samples_but_rejects_unguarded_execute(
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

    with pytest.raises(SystemExit):
        probe._parse_args(["--allow-gpu", "--execute", "--output", str(output)])


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


def test_parser_requires_exact_guarded_numerics_once_contract(tmp_path: Path) -> None:
    output = (tmp_path / "probe.jsonl").resolve()
    progress = (tmp_path / "progress.jsonl").resolve()
    evidence = (tmp_path / "compile.jsonl").resolve()
    profile_summary = (tmp_path / "telemetry.jsonl.summary.json").resolve()
    evidence.write_text("{}\n")
    profile_summary.write_text("{}\n")
    base = [
        "--allow-gpu",
        "--numerics-once",
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
    assert args.mode == "numerics_once"
    assert args.progress_output == progress
    assert args.compile_evidence == evidence
    assert args.compile_profile_summary == profile_summary
    assert (args.block_m, args.block_n, args.block_k) == (16, 32, 64)
    assert (args.warmups, args.iterations) == (0, 0)

    for mutation in (
        ["--block-m", "32"],
        ["--block-n", "16"],
        ["--block-k", "32"],
        ["--warmups", "1"],
        ["--iterations", "1"],
    ):
        with pytest.raises(SystemExit):
            probe._parse_args([*base, *mutation])

    for required_flag in (
        "--progress-output",
        "--compile-evidence",
        "--compile-profile-summary",
    ):
        missing = list(base)
        index = missing.index(required_flag)
        del missing[index : index + 2]
        with pytest.raises(SystemExit):
            probe._parse_args(missing)

    relative_progress = list(base)
    relative_progress[relative_progress.index(str(progress))] = "progress.jsonl"
    with pytest.raises(SystemExit):
        probe._parse_args(relative_progress)

    progress.write_text("occupied\n")
    with pytest.raises(SystemExit):
        probe._parse_args(base)


@pytest.mark.parametrize(
    ("mode_flag", "expected_mode", "warmups", "iterations"),
    (
        ("--benchmark-smoke", "benchmark_smoke", 2, 4),
        ("--benchmark", "benchmark", 8, 32),
    ),
)
def test_parser_requires_exact_guarded_benchmark_contract(
    tmp_path: Path,
    mode_flag: str,
    expected_mode: str,
    warmups: int,
    iterations: int,
) -> None:
    output = (tmp_path / f"{expected_mode}.jsonl").resolve()
    progress = (tmp_path / f"{expected_mode}-progress.jsonl").resolve()
    compile_evidence = (tmp_path / "compile.jsonl").resolve()
    compile_summary = (tmp_path / "compile-summary.json").resolve()
    numerics_evidence = (tmp_path / "numerics.jsonl").resolve()
    numerics_summary = (tmp_path / "numerics-summary.json").resolve()
    smoke_evidence = (tmp_path / "smoke.jsonl").resolve()
    smoke_summary = (tmp_path / "smoke-summary.json").resolve()
    for evidence in (
        compile_evidence,
        compile_summary,
        numerics_evidence,
        numerics_summary,
    ):
        evidence.write_text("{}\n")
    base = [
        "--allow-gpu",
        mode_flag,
        "--output",
        str(output),
        "--progress-output",
        str(progress),
        "--compile-evidence",
        str(compile_evidence),
        "--compile-profile-summary",
        str(compile_summary),
        "--numerics-evidence",
        str(numerics_evidence),
        "--numerics-profile-summary",
        str(numerics_summary),
    ]
    if expected_mode == "benchmark":
        smoke_evidence.write_text("{}\n")
        smoke_summary.write_text("{}\n")
        base.extend(
            (
                "--smoke-evidence",
                str(smoke_evidence),
                "--smoke-profile-summary",
                str(smoke_summary),
            )
        )
    base.extend(("--warmups", str(warmups), "--iterations", str(iterations)))
    args = probe._parse_args(base)
    assert args.mode == expected_mode
    assert (args.warmups, args.iterations) == (warmups, iterations)
    assert args.progress_output == progress
    assert args.compile_evidence == compile_evidence
    assert args.compile_profile_summary == compile_summary
    assert args.numerics_evidence == numerics_evidence
    assert args.numerics_profile_summary == numerics_summary
    assert (args.block_m, args.block_n, args.block_k) == (16, 32, 64)

    for required_flag in (
        "--progress-output",
        "--compile-evidence",
        "--compile-profile-summary",
        "--numerics-evidence",
        "--numerics-profile-summary",
    ):
        missing = list(base)
        index = missing.index(required_flag)
        del missing[index : index + 2]
        with pytest.raises(SystemExit):
            probe._parse_args(missing)

    for mutation in (
        ["--warmups", str(warmups + 1)],
        ["--iterations", str(iterations + 1)],
        ["--block-m", "32"],
        ["--block-n", "16"],
        ["--block-k", "32"],
    ):
        with pytest.raises(SystemExit):
            probe._parse_args([*base, *mutation])

    relative_numerics = list(base)
    relative_numerics[relative_numerics.index(str(numerics_evidence))] = (
        "numerics.jsonl"
    )
    with pytest.raises(SystemExit):
        probe._parse_args(relative_numerics)

    progress.write_text("occupied\n")
    with pytest.raises(SystemExit):
        probe._parse_args(base)


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


@pytest.mark.parametrize(
    ("mode", "scope_unit"),
    (
        ("benchmark_smoke", "skyrl-bf16-benchmark-smoke-4242-abc123.scope"),
        ("benchmark", "skyrl-bf16-benchmark-4242-abc123.scope"),
    ),
)
def test_benchmark_scope_requires_mode_specific_private_systemd_cgroup(
    tmp_path: Path, mode: str, scope_unit: str
) -> None:
    proc_root = tmp_path / "proc"
    cgroup_root = tmp_path / "cgroup"
    parent_pid = 4242
    scope = (
        f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/"
        f"app.slice/{scope_unit}"
    )
    (proc_root / "self").mkdir(parents=True)
    (proc_root / str(parent_pid)).mkdir(parents=True)
    (proc_root / "self" / "cgroup").write_text(f"0::{scope}\n")
    (proc_root / str(parent_pid) / "cgroup").write_text(f"0::{scope}\n")
    kill_path = cgroup_root / scope.removeprefix("/") / "cgroup.kill"
    kill_path.parent.mkdir(parents=True)
    kill_path.touch()

    manifest, descriptor = probe._require_guarded_scope(
        parent_pid,
        mode=mode,
        proc_root=proc_root,
        cgroup_root=cgroup_root,
    )
    try:
        assert manifest["validated"] is True
        assert manifest["scope_unit"] == scope_unit
        assert manifest["cgroup"] == scope
        assert manifest["timeout_kill_scope"] == "entire_systemd_cgroup"
        assert os.get_inheritable(descriptor) is False
    finally:
        os.close(descriptor)

    wrong_mode = "benchmark" if mode == "benchmark_smoke" else "benchmark_smoke"
    with pytest.raises(RuntimeError, match="private BF16 systemd scope"):
        probe._require_guarded_scope(
            parent_pid,
            mode=wrong_mode,
            proc_root=proc_root,
            cgroup_root=cgroup_root,
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
        "ml_dtypes": "test",
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
                "parent_command_python": str(Path(sys.executable).absolute()),
                "profiler_path": source["profiler"]["path"],
                "profiler_sha256": source["profiler"]["sha256"],
                "timeout_seconds": 600.0,
                "interval_seconds": 0.05,
                "baseline_seconds": 5.0,
                "sensor_grace_seconds": 15.0,
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


def test_numerics_evidence_is_exactly_passing_and_prior_gate_bound(
    tmp_path: Path,
) -> None:
    tmp_path.chmod(0o700)
    expected_contract = {"contract": "exact"}
    expected_geometry = {"geometry": "exact"}
    expected_device = {"device": "exact"}
    expected_source = {
        "kernel": {"path": "/repo/kernel.py", "sha256": "a" * 64},
        "probe": {"path": "/repo/probe.py", "sha256": "b" * 64},
        "profiler": {"path": "/repo/profile.py", "sha256": "c" * 64},
    }
    expected_git = {"commit": "d" * 40, "clean": True}
    expected_packages = {"jax": "test"}
    expected_compile_evidence = {"compile": "exact"}
    expected_compile_summary = {"compile_profile": "exact"}
    exact_once = dict.fromkeys(probe._PROGRAM_ORDER, 1)
    errors = _valid_errors()

    progress_records: list[dict[str, object]] = [
        {"event": "host_inputs_ready", "mode": "numerics_once"}
    ]
    for program in probe._PROGRAM_ORDER:
        progress_records.extend(
            (
                {
                    "event": "dispatch_started",
                    "mode": "numerics_once",
                    "program": program,
                },
                {
                    "event": "dispatch_completed",
                    "mode": "numerics_once",
                    "program": program,
                },
            )
        )
    progress_records.append(
        {
            "event": "numerics_completed",
            "mode": "numerics_once",
            "invocation_attempt_counts": exact_once,
            "invocation_completion_counts": exact_once,
            "passed": True,
            "errors": errors,
        }
    )
    progress = (tmp_path / "numerics-progress.jsonl").resolve()
    progress_raw = "".join(
        json.dumps(record, allow_nan=False) + "\n" for record in progress_records
    ).encode()
    progress.write_bytes(progress_raw)
    progress.chmod(0o600)
    progress_time = 1_700_000_000_000_000_000
    os.utime(progress, ns=(progress_time, progress_time))

    profiler_parent = {
        "validated": True,
        "limits": dict(probe._EXACT_PROFILE_LIMITS),
        "parent_command_python": str(Path(sys.executable).absolute()),
        "profiler_path": expected_source["profiler"]["path"],
        "profiler_sha256": expected_source["profiler"]["sha256"],
        "timeout_seconds": 600.0,
        "interval_seconds": 0.05,
        "baseline_seconds": 5.0,
        "sensor_grace_seconds": 15.0,
    }
    payload = {
        "schema_version": 1,
        "mode": "numerics_once",
        "passed": True,
        "contract": expected_contract,
        "geometry": expected_geometry,
        "device": expected_device,
        "source": {
            **expected_source,
            "git": expected_git,
            "packages": expected_packages,
        },
        "compilation": {
            program: {"lower_calls": 1, "compile_calls": 1}
            for program in probe._PROGRAM_ORDER
        },
        "invocation_contract": {
            "per_program_executable_invocations": exact_once,
            "per_program_executable_completions": exact_once,
            "total_executable_invocations": 4,
            "reference_executable_invocations": 2,
            "candidate_executable_invocations": 2,
        },
        "numerics": {
            "executed": True,
            "reference_compared": True,
            "passed": True,
            "errors": errors,
        },
        "numerics_once": {
            "compile_evidence": expected_compile_evidence,
            "compile_profile_summary": expected_compile_summary,
            "host_inputs_unchanged": True,
            "invocation_attempt_counts": exact_once,
            "invocation_completion_counts": exact_once,
            "progress": {
                "path": str(progress),
                "protocol": "durable_fsync_jsonl_v1",
                "record_count": 10,
                "bytes": len(progress_raw),
                "sha256": hashlib.sha256(progress_raw).hexdigest(),
                "mode": "0600",
                "directory_fsynced": True,
            },
            "watchdogs": [
                {
                    "program": program,
                    "dispatch_ordinal": ordinal,
                    "external_process": True,
                    "watchdog_pid": 20_000 + ordinal,
                    "cgroup_wide_timeout_kill": True,
                    "dispatch_completed": True,
                }
                for ordinal, program in enumerate(probe._PROGRAM_ORDER, start=1)
            ],
        },
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
            "profiler_parent": profiler_parent,
            "numerics_once_scope": {
                "validated": True,
                "scope_unit": "skyrl-bf16-numerics-123-abc.scope",
            },
        },
        "postflight": {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []},
    }
    evidence = (tmp_path / "numerics.jsonl").resolve()
    evidence.write_text(json.dumps(payload, allow_nan=False) + "\n")
    evidence.chmod(0o600)
    evidence_time = progress_time + 1_000_000
    os.utime(evidence, ns=(evidence_time, evidence_time))

    manifest = probe._validate_numerics_evidence(
        evidence,
        expected_contract=expected_contract,
        expected_geometry=expected_geometry,
        expected_source=expected_source,
        expected_git=expected_git,
        expected_device=expected_device,
        expected_packages=expected_packages,
        expected_compile_evidence=expected_compile_evidence,
        expected_compile_profile_summary=expected_compile_summary,
    )
    assert manifest["commit"] == "d" * 40
    assert manifest["program_order"] == list(probe._PROGRAM_ORDER)
    assert manifest["one_attempt_and_completion_per_program"] is True
    assert manifest["numerics_passed"] is True
    assert manifest["progress_record_count"] == 10
    assert manifest["progress_path"] == str(progress)

    payload["numerics"]["errors"]["dx"]["relative_l2"] = 0.03
    evidence.write_text(json.dumps(payload, allow_nan=False) + "\n")
    evidence.chmod(0o600)
    os.utime(evidence, ns=(evidence_time, evidence_time))
    with pytest.raises(RuntimeError, match="misses the exact numerical gate"):
        probe._validate_numerics_evidence(
            evidence,
            expected_contract=expected_contract,
            expected_geometry=expected_geometry,
            expected_source=expected_source,
            expected_git=expected_git,
            expected_device=expected_device,
            expected_packages=expected_packages,
            expected_compile_evidence=expected_compile_evidence,
            expected_compile_profile_summary=expected_compile_summary,
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
        "profile_binding": {
            "python_command": "/repo/.venv/bin/python",
            "profiler_path": "/repo/profile_rocm.py",
            "profiler_sha256": "f" * 64,
            "probe_path": "/repo/probe_bf16_rms_gate_up_lora_swiglu.py",
            "timeout_seconds": 600.0,
            "interval_seconds": 0.05,
            "baseline_seconds": 5.0,
            "sensor_grace_seconds": 15.0,
        },
    }
    telemetry_path = tmp_path / "telemetry.jsonl"
    telemetry_manifest = {
        "record_type": "manifest",
        "interval_seconds": 0.05,
        "baseline_seconds": 5.0,
        "duration_seconds": None,
        "timeout_seconds": 600.0,
        "safety_limits": {
            "max_junction_temp_c": 90.0,
            "max_gpu_power_watts": 400.0,
            "max_vram_bytes": 24.0 * 1024**3,
            "min_host_available_bytes": 0.0,
            "max_swap_bytes": 8.0 * 1024**3,
        },
        "sensor_grace_seconds": 15.0,
        "terminate_included_on_safety": False,
        "gpu": {
            "card": "card1",
            "vendor_id": "0x1002",
            "device_id": "0x744c",
        },
        "runtime": {
            "script_sha256": "f" * 64,
            "accelerator_environment": {
                "HIP_VISIBLE_DEVICES": "0",
                "JAX_PLATFORMS": "rocm",
                "XLA_FLAGS": "--xla_gpu_enable_command_buffer=",
            },
        },
        "command": [
            "/repo/.venv/bin/python",
            "/repo/probe_bf16_rms_gate_up_lora_swiglu.py",
            "--allow-gpu",
            "--compile-only",
            "--output",
            str(evidence),
            "--block-m",
            "16",
            "--block-n",
            "32",
            "--block-k",
            "64",
            "--warmups",
            "0",
            "--iterations",
            "0",
        ],
        "command_recorded": True,
        "passed_file_descriptor_count": 0,
    }
    telemetry_records = [telemetry_manifest] + [
        {"record_type": "sample", "ordinal": index} for index in range(31)
    ]
    telemetry_path.write_text(
        "".join(json.dumps(record) + "\n" for record in telemetry_records)
    )
    telemetry_path.chmod(0o600)
    telemetry_time = evidence_time + 500_000
    os.utime(telemetry_path, ns=(telemetry_time, telemetry_time))
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
    assert manifest["telemetry"]["record_count"] == 32
    assert manifest["telemetry"]["evidence_output_path"] == str(evidence)

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


def test_numerics_profile_summary_binds_exact_prior_gate_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    numerics_evidence = (tmp_path / "numerics.jsonl").resolve()
    numerics_progress = (tmp_path / "numerics-progress.jsonl").resolve()
    compile_evidence = (tmp_path / "compile.jsonl").resolve()
    compile_summary = (tmp_path / "compile-summary.json").resolve()
    numerics_summary = (tmp_path / "numerics-summary.json").resolve()
    evidence_manifest = {
        "profile_binding": {
            "python_command": "/repo/.venv/bin/python",
            "probe_path": "/repo/rocm/probe_bf16_rms_gate_up_lora_swiglu.py",
        },
        "progress_path": str(numerics_progress),
    }
    captured: dict[str, object] = {}

    def validate(path: Path, **arguments: object) -> dict[str, object]:
        captured["path"] = path
        captured.update(arguments)
        return {"validated": True}

    monkeypatch.setattr(probe, "_validate_compile_profile_summary", validate)
    result = probe._validate_numerics_profile_summary(
        numerics_summary,
        evidence_path=numerics_evidence,
        evidence_manifest=evidence_manifest,
        compile_evidence_path=compile_evidence,
        compile_profile_summary_path=compile_summary,
    )

    assert result == {"validated": True}
    assert captured == {
        "path": numerics_summary,
        "evidence_path": numerics_evidence,
        "evidence_manifest": evidence_manifest,
        "expected_child_command": [
            "/repo/.venv/bin/python",
            "/repo/rocm/probe_bf16_rms_gate_up_lora_swiglu.py",
            "--allow-gpu",
            "--numerics-once",
            "--output",
            str(numerics_evidence),
            "--progress-output",
            str(numerics_progress),
            "--compile-evidence",
            str(compile_evidence),
            "--compile-profile-summary",
            str(compile_summary),
            "--block-m",
            "16",
            "--block-n",
            "32",
            "--block-k",
            "64",
            "--warmups",
            "0",
            "--iterations",
            "0",
        ],
    }

    with pytest.raises(RuntimeError, match="profile/progress binding is missing"):
        probe._validate_numerics_profile_summary(
            numerics_summary,
            evidence_path=numerics_evidence,
            evidence_manifest={"profile_binding": {}},
            compile_evidence_path=compile_evidence,
            compile_profile_summary_path=compile_summary,
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


@pytest.mark.parametrize(
    ("mode", "warmups", "measurements"),
    (("benchmark_smoke", 2, 4), ("benchmark", 8, 32)),
)
def test_guarded_benchmark_schedule_has_exact_balanced_rotations(
    mode: str, warmups: int, measurements: int
) -> None:
    schedule = probe._benchmark_schedule(mode)
    assert len(schedule) == warmups + measurements
    assert [item["global_supercycle"] for item in schedule] == list(
        range(warmups + measurements)
    )
    assert [item["phase_supercycle"] for item in schedule[:warmups]] == list(
        range(warmups)
    )
    assert [item["phase_supercycle"] for item in schedule[warmups:]] == list(
        range(measurements)
    )
    assert [item["phase"] for item in schedule] == ["warmup"] * warmups + [
        "measurement"
    ] * measurements
    assert [item["order"] for item in schedule[:warmups]] == [
        probe._BENCHMARK_WARMUP_ORDERS[index % 2] for index in range(warmups)
    ]
    assert [item["order"] for item in schedule[warmups:]] == [
        probe._BENCHMARK_MEASUREMENT_ROTATION[index % 4]
        for index in range(measurements)
    ]
    assert all(
        len(order) == 4 and set(order) == set(probe._PROGRAM_ORDER)
        for order in (item["order"] for item in schedule)
    )
    flattened = [program for item in schedule for program in item["order"]]
    assert {program: flattened.count(program) for program in probe._PROGRAM_ORDER} == (
        dict.fromkeys(probe._PROGRAM_ORDER, warmups + measurements)
    )

    with pytest.raises(ValueError, match="unsupported guarded benchmark mode"):
        probe._benchmark_schedule("execute")


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


def test_numerics_once_plan_and_workload_run_exact_guarded_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    markers = {
        "reference_forward_and_vjp": object(),
        "candidate_forward_and_vjp": object(),
        "reference_forward": object(),
        "candidate_forward": object(),
    }
    step_arguments, forward_arguments = _host_bf16_arguments()
    plan = probe._compilation_plan(
        mode="numerics_once",
        reference_step=markers["reference_forward_and_vjp"],
        candidate_step=markers["candidate_forward_and_vjp"],
        reference_forward=markers["reference_forward"],
        candidate_forward=markers["candidate_forward"],
        step_arguments=step_arguments,
        forward_arguments=forward_arguments,
    )
    assert list(plan) == [
        "reference_forward",
        "candidate_forward",
        "reference_forward_and_vjp",
        "candidate_forward_and_vjp",
    ]
    assert plan == {
        "reference_forward": (markers["reference_forward"], forward_arguments),
        "candidate_forward": (markers["candidate_forward"], forward_arguments),
        "reference_forward_and_vjp": (
            markers["reference_forward_and_vjp"],
            step_arguments,
        ),
        "candidate_forward_and_vjp": (
            markers["candidate_forward_and_vjp"],
            step_arguments,
        ),
    }

    result_tree = _numerics_result_tree()
    calls: list[str] = []

    class Executable:
        def __init__(self, name: str, result: object):
            self.name = name
            self.result = result

        def __call__(self, *arguments):
            calls.append(self.name)
            expected_count = 7 if self.name.endswith("and_vjp") else 6
            assert len(arguments) == expected_count
            assert all(type(argument) is np.ndarray for argument in arguments)
            assert all(argument.dtype == bfloat16 for argument in arguments)
            return self.result

    executables = {
        "reference_forward": Executable("reference_forward", result_tree[0]),
        "candidate_forward": Executable("candidate_forward", result_tree[0]),
        "reference_forward_and_vjp": Executable(
            "reference_forward_and_vjp", result_tree
        ),
        "candidate_forward_and_vjp": Executable(
            "candidate_forward_and_vjp", result_tree
        ),
    }
    watchdogs: list[dict[str, object]] = []
    watchdog_events: list[tuple[str, int, bool | None]] = []

    def start_watchdog(_cgroup_kill_fd):
        identifier = len(watchdogs)
        state: dict[str, object] = {
            "identifier": identifier,
            "arm_sent": False,
            "armed": False,
            "settled": False,
            "cgroup_wide_timeout_kill": True,
            "timeout_seconds": 5.0,
        }
        watchdogs.append(state)
        watchdog_events.append(("started", identifier, None))
        return state

    def arm_watchdog(state):
        identifier = int(state["identifier"])
        armed_monotonic_ns = time.monotonic_ns()
        state.update(
            {
                "arm_sent": True,
                "armed": True,
                "armed_monotonic_ns": armed_monotonic_ns,
                "deadline_monotonic_ns": armed_monotonic_ns + 5_000_000_000,
                "armed_ack_received_monotonic_ns": armed_monotonic_ns + 1,
            }
        )
        watchdog_events.append(("armed", identifier, None))

    def settle_watchdog(state, *, invocation_completed):
        identifier = int(state["identifier"])
        assert state["settled"] is False
        state["settled"] = True
        watchdog_events.append(("settled", identifier, invocation_completed))

    monkeypatch.setattr(probe, "_start_forward_once_watchdog", start_watchdog)
    monkeypatch.setattr(probe, "_arm_forward_once_watchdog", arm_watchdog)
    monkeypatch.setattr(probe, "_settle_forward_once_watchdog", settle_watchdog)
    monkeypatch.setattr(probe, "_block_tree", lambda value: value)
    monkeypatch.setattr(probe, "_host_input_manifest", _mock_host_input_manifest)
    progress = tmp_path / "numerics-progress.jsonl"
    kill_path = tmp_path / "cgroup.kill"
    kill_path.touch()
    kill_fd = os.open(kill_path, os.O_WRONLY)
    try:
        result = probe._run_numerics_once_workload(
            executables=executables,
            step_arguments=step_arguments,
            forward_arguments=forward_arguments,
            progress_output=progress,
            binding={"test": "exact-guarded-numerics-order"},
            cgroup_kill_fd=kill_fd,
        )
    finally:
        os.close(kill_fd)

    expected_order = [
        "reference_forward",
        "candidate_forward",
        "reference_forward_and_vjp",
        "candidate_forward_and_vjp",
    ]
    assert calls == expected_order
    assert len({id(watchdog) for watchdog in watchdogs}) == 4
    assert watchdog_events == [
        (event, identifier, completed)
        for identifier in range(4)
        for event, completed in (
            ("started", None),
            ("armed", None),
            ("settled", True),
        )
    ]
    exact_once = dict.fromkeys(
        (
            "reference_forward",
            "candidate_forward",
            "reference_forward_and_vjp",
            "candidate_forward_and_vjp",
        ),
        1,
    )
    assert result["invocation_counts"] == exact_once
    assert result["invocation_completion_counts"] == exact_once
    assert result["host_inputs_unchanged"] is True
    assert set(result["errors"]) == set(_valid_errors())
    for manifest in result["errors"].values():
        assert manifest["relative_l2"] == 0.0
        assert manifest["cosine_similarity"] == pytest.approx(1.0, abs=1e-12)
        assert manifest["max_absolute"] == 0.0
        assert manifest["finite"] is True
    assert result["numerics_passed"] is True
    assert result["progress"]["record_count"] == 10
    assert result["progress"]["protocol"] == "durable_fsync_jsonl_v1"
    raw_progress = progress.read_bytes()
    assert result["progress"]["bytes"] == len(raw_progress)
    assert result["progress"]["sha256"] == hashlib.sha256(raw_progress).hexdigest()
    assert stat.S_IMODE(progress.stat().st_mode) == 0o600

    records = [json.loads(line) for line in raw_progress.splitlines()]
    assert [record["event"] for record in records] == [
        "host_inputs_ready",
        "dispatch_started",
        "dispatch_completed",
        "dispatch_started",
        "dispatch_completed",
        "dispatch_started",
        "dispatch_completed",
        "dispatch_started",
        "dispatch_completed",
        "numerics_completed",
    ]
    dispatch_records = [
        record
        for record in records
        if record["event"] in ("dispatch_started", "dispatch_completed")
    ]
    assert [record["program"] for record in dispatch_records] == [
        program for program in expected_order for _ in range(2)
    ]
    assert records[0]["host_inputs"] == {"test_host_bfloat16_inputs": True}
    assert records[-1]["passed"] is True
    assert records[-1]["errors"] == result["errors"]
    assert all(record["wall_time_ns"] > 0 for record in records)
    assert all(record["monotonic_time_ns"] > 0 for record in records)


@pytest.mark.parametrize("invalid_kind", ("shape", "dtype", "nonfinite"))
def test_numerics_once_invalid_candidate_forward_tree_never_completes_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    step_arguments, forward_arguments = _host_bf16_arguments()
    valid_tree = _numerics_result_tree()
    if invalid_kind == "shape":
        invalid = np.ones((1, 64, 9215), dtype=bfloat16)
    elif invalid_kind == "dtype":
        invalid = np.ones(_NUMERICS_RESULT_SHAPES[0], dtype=np.float32)
    else:
        invalid = np.full(_NUMERICS_RESULT_SHAPES[0], np.nan, dtype=bfloat16)

    executables = {
        "reference_forward": lambda *_arguments: valid_tree[0],
        "candidate_forward": lambda *_arguments: invalid,
        "reference_forward_and_vjp": lambda *_arguments: valid_tree,
        "candidate_forward_and_vjp": lambda *_arguments: valid_tree,
    }
    completions: list[bool] = []
    watchdog_identifier = 0

    def start_watchdog(_cgroup_kill_fd):
        nonlocal watchdog_identifier
        watchdog_identifier += 1
        return {
            "identifier": watchdog_identifier,
            "arm_sent": False,
            "armed": False,
            "settled": False,
            "cgroup_wide_timeout_kill": True,
            "timeout_seconds": 5.0,
        }

    def arm_watchdog(state):
        armed_monotonic_ns = time.monotonic_ns()
        state.update(
            {
                "arm_sent": True,
                "armed": True,
                "armed_monotonic_ns": armed_monotonic_ns,
                "deadline_monotonic_ns": armed_monotonic_ns + 5_000_000_000,
                "armed_ack_received_monotonic_ns": armed_monotonic_ns + 1,
            }
        )

    def settle_watchdog(state, *, invocation_completed):
        assert state["settled"] is False
        state["settled"] = True
        completions.append(invocation_completed)

    monkeypatch.setattr(probe, "_start_forward_once_watchdog", start_watchdog)
    monkeypatch.setattr(probe, "_arm_forward_once_watchdog", arm_watchdog)
    monkeypatch.setattr(probe, "_settle_forward_once_watchdog", settle_watchdog)
    monkeypatch.setattr(probe, "_block_tree", lambda value: value)
    monkeypatch.setattr(probe, "_host_input_manifest", _mock_host_input_manifest)
    progress = tmp_path / f"invalid-{invalid_kind}.jsonl"
    kill_path = tmp_path / "cgroup.kill"
    kill_path.touch()
    kill_fd = os.open(kill_path, os.O_WRONLY)
    expected_error = {
        "shape": "shape=",
        "dtype": "dtype=",
        "nonfinite": "nonfinite",
    }[invalid_kind]
    try:
        with pytest.raises(RuntimeError, match=expected_error):
            probe._run_numerics_once_workload(
                executables=executables,
                step_arguments=step_arguments,
                forward_arguments=forward_arguments,
                progress_output=progress,
                binding={"test": invalid_kind},
                cgroup_kill_fd=kill_fd,
            )
    finally:
        os.close(kill_fd)

    assert completions == [True, False]
    records = [json.loads(line) for line in progress.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "host_inputs_ready",
        "dispatch_started",
        "dispatch_completed",
        "dispatch_started",
    ]
    assert records[-1]["program"] == "candidate_forward"
    assert records[-1]["invocation_attempt_counts"]["candidate_forward"] == 1
    assert records[-1]["invocation_completion_counts"]["candidate_forward"] == 0


def test_numerics_once_executable_exception_never_signals_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    step_arguments, forward_arguments = _host_bf16_arguments()
    valid_tree = _numerics_result_tree()
    calls: list[str] = []

    def result(name: str, value: object):
        def executable(*_arguments):
            calls.append(name)
            if isinstance(value, BaseException):
                raise value
            return value

        return executable

    executables = {
        "reference_forward": result("reference_forward", valid_tree[0]),
        "candidate_forward": result("candidate_forward", valid_tree[0]),
        "reference_forward_and_vjp": result("reference_forward_and_vjp", valid_tree),
        "candidate_forward_and_vjp": result(
            "candidate_forward_and_vjp",
            RuntimeError("synthetic candidate VJP failure"),
        ),
    }
    completions: list[bool] = []
    watchdog_identifier = 0

    def start_watchdog(_cgroup_kill_fd):
        nonlocal watchdog_identifier
        watchdog_identifier += 1
        return {
            "identifier": watchdog_identifier,
            "arm_sent": False,
            "armed": False,
            "settled": False,
            "cgroup_wide_timeout_kill": True,
            "timeout_seconds": 5.0,
        }

    def arm_watchdog(state):
        armed_monotonic_ns = time.monotonic_ns()
        state.update(
            {
                "arm_sent": True,
                "armed": True,
                "armed_monotonic_ns": armed_monotonic_ns,
                "deadline_monotonic_ns": armed_monotonic_ns + 5_000_000_000,
                "armed_ack_received_monotonic_ns": armed_monotonic_ns + 1,
            }
        )

    def settle_watchdog(state, *, invocation_completed):
        assert state["settled"] is False
        state["settled"] = True
        completions.append(invocation_completed)

    monkeypatch.setattr(probe, "_start_forward_once_watchdog", start_watchdog)
    monkeypatch.setattr(probe, "_arm_forward_once_watchdog", arm_watchdog)
    monkeypatch.setattr(probe, "_settle_forward_once_watchdog", settle_watchdog)
    monkeypatch.setattr(probe, "_block_tree", lambda value: value)
    monkeypatch.setattr(probe, "_host_input_manifest", _mock_host_input_manifest)
    progress = tmp_path / "exception-progress.jsonl"
    kill_path = tmp_path / "cgroup.kill"
    kill_path.touch()
    kill_fd = os.open(kill_path, os.O_WRONLY)
    try:
        with pytest.raises(RuntimeError, match="synthetic candidate VJP failure"):
            probe._run_numerics_once_workload(
                executables=executables,
                step_arguments=step_arguments,
                forward_arguments=forward_arguments,
                progress_output=progress,
                binding={"test": "exception-is-not-completion"},
                cgroup_kill_fd=kill_fd,
            )
    finally:
        os.close(kill_fd)

    assert calls == [
        "reference_forward",
        "candidate_forward",
        "reference_forward_and_vjp",
        "candidate_forward_and_vjp",
    ]
    assert completions == [True, True, True, False]
    records = [json.loads(line) for line in progress.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "host_inputs_ready",
        "dispatch_started",
        "dispatch_completed",
        "dispatch_started",
        "dispatch_completed",
        "dispatch_started",
        "dispatch_completed",
        "dispatch_started",
    ]
    assert records[-1]["program"] == "candidate_forward_and_vjp"
    assert records[-1]["invocation_completion_counts"]["candidate_forward_and_vjp"] == 0


@pytest.mark.parametrize(
    ("mode", "warmups", "measurements", "expected_records"),
    (("benchmark_smoke", 2, 4, 54), ("benchmark", 8, 32, 326)),
)
def test_guarded_benchmark_runs_exact_schedule_and_durable_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    warmups: int,
    measurements: int,
    expected_records: int,
) -> None:
    jax, device, staged_inputs = _install_benchmark_device_mocks(monkeypatch)
    watchdogs, watchdog_events = _install_benchmark_watchdog_mocks(monkeypatch)
    step_arguments, forward_arguments = _host_bf16_arguments()
    calls: list[str] = []
    result_leaves: list[_FakeDeviceLeaf] = []

    def executable(program: str):
        def invoke(*arguments: object) -> object:
            calls.append(program)
            assert arguments == (
                staged_inputs
                if program.endswith("and_vjp")
                else staged_inputs[:-1]
            )
            specs = (
                probe._STEP_RESULT_SPECS
                if program.endswith("and_vjp")
                else probe._FORWARD_RESULT_SPECS
            )
            leaves = tuple(_FakeDeviceLeaf(shape, device) for _name, shape in specs)
            result_leaves.extend(leaves)
            return leaves if len(leaves) > 1 else leaves[0]

        return invoke

    executables = {program: executable(program) for program in probe._PROGRAM_ORDER}
    progress = (tmp_path / f"{mode}-progress.jsonl").resolve()
    kill_path = tmp_path / "cgroup.kill"
    kill_path.touch()
    kill_fd = os.open(kill_path, os.O_WRONLY)
    try:
        result = probe._run_guarded_benchmark_workload(
            mode=mode,
            jax=jax,
            device=device,
            executables=executables,
            host_step_arguments=step_arguments,
            host_forward_arguments=forward_arguments,
            progress_output=progress,
            binding={"test": f"guarded-{mode}"},
            cgroup_kill_fd=kill_fd,
        )
    finally:
        os.close(kill_fd)

    schedule = probe._benchmark_schedule(mode)
    expected_calls = [program for item in schedule for program in item["order"]]
    total_supercycles = warmups + measurements
    expected_dispatches = 4 * total_supercycles
    expected_watchdogs = expected_dispatches + 2
    expected_samples = 4 * measurements
    assert calls == expected_calls
    assert result["warmup_orders"] == [
        list(item["order"]) for item in schedule[:warmups]
    ]
    assert result["measurement_orders"] == [
        list(item["order"]) for item in schedule[warmups:]
    ]
    exact_counts = dict.fromkeys(probe._PROGRAM_ORDER, total_supercycles)
    assert result["invocation_counts"] == exact_counts
    assert result["invocation_completion_counts"] == exact_counts
    assert result["performance_qualification"] is False

    assert len(watchdogs) == expected_watchdogs
    assert len({id(watchdog) for watchdog in watchdogs}) == expected_watchdogs
    assert watchdog_events == [
        (event, identifier, completion)
        for identifier in range(expected_watchdogs)
        for event, completion in (
            ("started", None),
            ("armed", None),
            ("settled", True),
        )
    ]
    assert result["setup_watchdog"]["operation"] == "device_input_setup"
    assert result["setup_watchdog"]["watchdog_pid"] == 10_000
    assert result["setup_watchdog"]["operation_completed"] is True
    assert len(result["dispatch_watchdogs"]) == expected_dispatches
    assert [watchdog["watchdog_pid"] for watchdog in result["dispatch_watchdogs"]] == (
        list(range(10_001, 10_001 + expected_dispatches))
    )
    assert [watchdog["dispatch_ordinal"] for watchdog in result["dispatch_watchdogs"]] == (
        list(range(1, expected_dispatches + 1))
    )
    assert [watchdog["program"] for watchdog in result["dispatch_watchdogs"]] == (
        expected_calls
    )
    assert result["teardown_watchdog"] == {
        "operation": "device_input_teardown",
        "device_inputs_explicitly_deleted": True,
        "executable_references_cleared": True,
        "jax_caches_cleared": True,
        "all_device_inputs_ready_before_delete": True,
        "all_dispatch_results_blocked": True,
        "effects_barrier_completed": True,
        "external_process": True,
        "watchdog_pid": 10_001 + expected_dispatches,
        "timeout_action": "cgroup.kill_then_pidfd_SIGKILL_fallback",
        "cgroup_wide_timeout_kill": True,
        "timeout_seconds": 5.0,
        "armed_monotonic_ns": watchdogs[-1]["armed_monotonic_ns"],
        "deadline_monotonic_ns": watchdogs[-1]["deadline_monotonic_ns"],
        "armed_ack_received_monotonic_ns": watchdogs[-1][
            "armed_ack_received_monotonic_ns"
        ],
        "operation_completed": True,
    }

    expected_result_leaves = 10 * total_supercycles
    assert len(result_leaves) == expected_result_leaves
    assert len({id(leaf) for leaf in result_leaves}) == expected_result_leaves
    assert all(leaf.block_until_ready_calls == 1 for leaf in result_leaves)
    assert all(leaf.delete_calls == 1 and leaf.is_deleted() for leaf in result_leaves)
    assert all(leaf.block_until_ready_calls == 2 for leaf in staged_inputs)
    assert all(leaf.delete_calls == 1 and leaf.is_deleted() for leaf in staged_inputs)
    assert len(jax.device_put_calls) == 1
    assert jax.device_put_calls[0] == (step_arguments, device)
    assert jax.device_get_calls == [staged_inputs]
    assert jax.effects_barrier_calls == 1
    assert jax.clear_caches_calls == 1
    assert not hasattr(device, "synchronize_all_activity")
    assert executables == {}
    assert result["device_setup_counts"] == {"attempts": 1, "completions": 1}
    assert result["device_teardown_counts"] == {"attempts": 1, "completions": 1}

    raw_samples = result["raw_samples"]
    assert len(raw_samples) == expected_samples
    assert [sample["dispatch_ordinal"] for sample in raw_samples] == list(
        range(4 * warmups + 1, expected_dispatches + 1)
    )
    assert all(sample["phase"] == "measurement" for sample in raw_samples)
    assert all(
        type(sample["elapsed_seconds"]) is float
        and math.isfinite(sample["elapsed_seconds"])
        and sample["elapsed_seconds"] > 0
        for sample in raw_samples
    )
    assert set(result["raw_samples_by_program"]) == set(probe._PROGRAM_ORDER)
    assert all(
        len(result["raw_samples_by_program"][program]) == measurements
        for program in probe._PROGRAM_ORDER
    )
    assert all(
        sample["elapsed_seconds"]
        in result["raw_samples_by_program"][sample["program"]]
        for sample in raw_samples
    )

    raw_progress = progress.read_bytes()
    assert result["progress"] == {
        "path": str(progress),
        "protocol": "durable_fsync_jsonl_v1",
        "record_count": expected_records,
        "bytes": len(raw_progress),
        "sha256": hashlib.sha256(raw_progress).hexdigest(),
        "mode": "0600",
        "directory_fsynced": True,
    }
    assert stat.S_IMODE(progress.stat().st_mode) == 0o600
    records = [json.loads(line) for line in raw_progress.splitlines()]
    expected_events = [
        "host_inputs_ready",
        "device_input_setup_started",
        "device_input_setup_completed",
    ]
    for _program in expected_calls:
        expected_events.extend(("dispatch_started", "dispatch_completed"))
    expected_events.extend(
        (
            "benchmark_samples_completed",
            "device_input_teardown_started",
            "device_input_teardown_completed",
        )
    )
    assert len(records) == expected_records
    assert [record["event"] for record in records] == expected_events
    assert all(record["mode"] == mode for record in records)
    samples_record_index = 3 + 2 * expected_dispatches
    dispatch_records = records[3:samples_record_index]
    assert [record["program"] for record in dispatch_records] == [
        program for program in expected_calls for _ in range(2)
    ]
    assert [record["dispatch_ordinal"] for record in dispatch_records] == [
        ordinal for ordinal in range(1, expected_dispatches + 1) for _ in range(2)
    ]
    completed = dispatch_records[1::2]
    assert all(record["result_leaves_blocked_before_timer_stop"] is True for record in completed)
    assert all(record["result_references_released_before_completion"] is True for record in completed)
    assert [record["result_leaves_explicitly_deleted"] for record in completed] == [
        4 if program.endswith("and_vjp") else 1 for program in expected_calls
    ]
    assert records[samples_record_index]["raw_measurement_sample_count"] == (
        expected_samples
    )
    assert records[samples_record_index]["raw_samples"] == raw_samples
    assert records[-1]["explicitly_deleted_unique_input_leaves"] == 7
    assert records[-1]["all_device_inputs_ready_before_delete"] is True
    assert records[-1]["all_dispatch_results_blocked"] is True
    assert records[-1]["effects_barrier_completed"] is True
    assert records[-1]["invocation_attempt_counts"] == exact_counts
    assert records[-1]["invocation_completion_counts"] == exact_counts


def test_benchmark_smoke_dispatch_failure_never_completes_or_calls_later_programs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    jax, device, staged_inputs = _install_benchmark_device_mocks(monkeypatch)
    watchdogs, watchdog_events = _install_benchmark_watchdog_mocks(monkeypatch)
    step_arguments, forward_arguments = _host_bf16_arguments()
    calls: list[str] = []
    result_leaves: list[_FakeDeviceLeaf] = []

    def executable(program: str):
        def invoke(*_arguments: object) -> object:
            calls.append(program)
            if len(calls) == 2:
                raise RuntimeError("synthetic guarded benchmark dispatch failure")
            specs = (
                probe._STEP_RESULT_SPECS
                if program.endswith("and_vjp")
                else probe._FORWARD_RESULT_SPECS
            )
            leaves = tuple(_FakeDeviceLeaf(shape, device) for _name, shape in specs)
            result_leaves.extend(leaves)
            return leaves if len(leaves) > 1 else leaves[0]

        return invoke

    executables = {program: executable(program) for program in probe._PROGRAM_ORDER}
    progress = (tmp_path / "failed-benchmark-progress.jsonl").resolve()
    kill_path = tmp_path / "cgroup.kill"
    kill_path.touch()
    kill_fd = os.open(kill_path, os.O_WRONLY)
    try:
        with pytest.raises(
            RuntimeError, match="synthetic guarded benchmark dispatch failure"
        ):
            probe._run_guarded_benchmark_workload(
                mode="benchmark_smoke",
                jax=jax,
                device=device,
                executables=executables,
                host_step_arguments=step_arguments,
                host_forward_arguments=forward_arguments,
                progress_output=progress,
                binding={"test": "dispatch-failure-is-not-completion"},
                cgroup_kill_fd=kill_fd,
            )
    finally:
        os.close(kill_fd)

    expected_first_two = list(probe._BENCHMARK_WARMUP_ORDERS[0][:2])
    assert calls == expected_first_two
    assert len(watchdogs) == 3
    assert watchdog_events == [
        ("started", 0, None),
        ("armed", 0, None),
        ("settled", 0, True),
        ("started", 1, None),
        ("armed", 1, None),
        ("settled", 1, True),
        ("started", 2, None),
        ("armed", 2, None),
        ("settled", 2, False),
    ]
    assert len(result_leaves) == 1
    assert result_leaves[0].block_until_ready_calls == 1
    assert result_leaves[0].delete_calls == 1
    assert all(leaf.block_until_ready_calls == 1 for leaf in staged_inputs)
    assert all(leaf.delete_calls == 0 for leaf in staged_inputs)
    assert jax.effects_barrier_calls == 0
    assert jax.clear_caches_calls == 0
    assert len(executables) == 4

    records = [json.loads(line) for line in progress.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "host_inputs_ready",
        "device_input_setup_started",
        "device_input_setup_completed",
        "dispatch_started",
        "dispatch_completed",
        "dispatch_started",
    ]
    assert [record.get("program") for record in records[3:]] == [
        expected_first_two[0],
        expected_first_two[0],
        expected_first_two[1],
    ]
    assert records[-1]["invocation_attempt_counts"][expected_first_two[1]] == 1
    assert records[-1]["invocation_completion_counts"][expected_first_two[1]] == 0
    assert not any(
        record["event"]
        in {
            "benchmark_samples_completed",
            "device_input_teardown_started",
            "device_input_teardown_completed",
        }
        for record in records
    )


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
