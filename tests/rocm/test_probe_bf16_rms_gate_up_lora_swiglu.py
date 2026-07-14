from __future__ import annotations

import inspect
import math
import os
import stat
from pathlib import Path

import pytest

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
    assert "guarded_qwen35_rocm_process" in main_source
    assert "finally:" in main_source
    assert "postflight = safety_module.require_clean_amdgpu_boot()" in main_source

    workload_source = inspect.getsource(probe._run_compiled_workload)
    assert workload_source.index('if mode == "compile_only":') < workload_source.index(
        "def invoke("
    )
    assert "executables[key](*arguments)" in workload_source
