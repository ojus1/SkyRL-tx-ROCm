from __future__ import annotations

from pathlib import Path

import pytest

from rocm import probe_bf16_down_lora_residual as probe


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
        "d_residual": {**common, "bitwise_equal": True},
    }


def test_numerics_gate_enforces_distinct_output_and_gradient_contracts() -> None:
    assert probe._numerics_gate_passed(_valid_errors())

    output_l2 = _valid_errors()
    output_l2["output"]["relative_l2"] = 0.0051
    assert not probe._numerics_gate_passed(output_l2)

    output_cosine = _valid_errors()
    output_cosine["output"]["cosine_similarity"] = 0.9998
    assert not probe._numerics_gate_passed(output_cosine)

    gradient_l2 = _valid_errors()
    gradient_l2["d_lora_b"]["relative_l2"] = 0.0101
    assert not probe._numerics_gate_passed(gradient_l2)

    gradient_cosine = _valid_errors()
    gradient_cosine["dx"]["cosine_similarity"] = 0.998
    assert not probe._numerics_gate_passed(gradient_cosine)

    residual = _valid_errors()
    residual["d_residual"]["bitwise_equal"] = False
    assert not probe._numerics_gate_passed(residual)


def test_performance_gate_requires_speedups_and_every_sample_under_cap() -> None:
    arguments = {
        "forward_vjp_speedup": 1.03,
        "rematerialized_speedup": 1.05,
        "candidate_seconds": [0.001, 0.002],
        "candidate_forward_seconds": [0.001, 0.002],
    }
    assert probe._performance_gate_passed(**arguments)

    for mutation in (
        {"forward_vjp_speedup": 1.029},
        {"rematerialized_speedup": 1.049},
        {"candidate_seconds": [0.001, 0.101]},
        {"candidate_forward_seconds": [0.101, 0.001]},
    ):
        assert not probe._performance_gate_passed(**(arguments | mutation))


def test_parser_accepts_only_exact_production_geometry_and_sample_floor(
    tmp_path: Path,
) -> None:
    output = tmp_path / "probe.jsonl"
    args = probe._parse_args(["--allow-gpu", "--output", str(output)])
    assert (args.rows, args.in_features, args.out_features, args.rank) == (
        64,
        9216,
        2560,
        8,
    )
    assert args.warmups == 3
    assert args.iterations == 11

    for extra in (
        ["--rows", "63"],
        ["--in-features", "32"],
        ["--out-features", "64"],
        ["--rank", "4"],
        ["--warmups", "2"],
        ["--iterations", "10"],
    ):
        with pytest.raises(SystemExit):
            probe._parse_args(["--allow-gpu", "--output", str(output), *extra])


def test_environment_rejects_hidden_accelerator_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    with pytest.raises(RuntimeError, match="HSA_OVERRIDE_GFX_VERSION"):
        probe._configure_environment()


def test_environment_rejects_extra_xla_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=true")
    with pytest.raises(RuntimeError, match="rejects inherited XLA_FLAGS"):
        probe._configure_environment()
