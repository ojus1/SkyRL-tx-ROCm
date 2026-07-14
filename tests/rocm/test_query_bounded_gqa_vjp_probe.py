from __future__ import annotations

import ast
import gc
import hashlib
import importlib.util
import inspect
import io
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ml_dtypes
import numpy as np
import pytest

_REPO = Path(__file__).parents[2]
_PROBE_PATH = _REPO / "rocm" / "probe_query_bounded_gqa_vjp.py"
_SPEC = importlib.util.spec_from_file_location("vjp_probe_test", _PROBE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)

_CLEAN = {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}
_TARGET = "__gpu$xla.gpu.triton"


def _records(output: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in output.getvalue().splitlines()]


def _accelerator_modules() -> set[str]:
    return {
        name
        for name in sys.modules
        if name in {"jax", "jaxlib"} or name.startswith(("jax.", "jaxlib."))
    }


@pytest.fixture(scope="module")
def exact_case():
    return _PROBE._construct_host_case(np, ml_dtypes)


@pytest.fixture(scope="module")
def valid385_case():
    return _PROBE._construct_host_case(np, ml_dtypes, "valid385")


@pytest.fixture(scope="module")
def all_valid_t1024_case():
    return _PROBE._construct_host_case(np, ml_dtypes, "all_valid_t1024")


def test_default_is_abstract_refusal_without_accelerator_import():
    before = _accelerator_modules()
    output = io.StringIO()
    assert (
        _PROBE._execute(SimpleNamespace(platform="abstract", allow_gpu=False), output)
        == 0
    )
    assert _accelerator_modules() == before
    manifest, refused = _records(output)
    assert manifest["scope"] == "abstract_refusal"
    assert manifest["contract"] == {
        "operation": "query_bounded_gqa_t512_full_vjp_refusal",
        "exact_logical_internal_calls": ["forward_q0", "dq_q0", "dkdv_q0"],
        "gpu_work_authorized": False,
    }
    assert refused["jax_imported"] is False
    assert refused["counters"] == _PROBE._zero_counters()
    for name, path in _PROBE._source_files().items():
        assert manifest[name] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--platform", "rocm"), "requires the explicit --allow-gpu"),
        (("--platform", "rocm", "--allow-gpu"), "requires --output"),
        (("--allow-gpu",), "only valid with --platform rocm"),
        (("--sequence-length", "1024"), "unrecognized arguments"),
        (("--replay",), "unrecognized arguments"),
        (("--padding",), "unrecognized arguments"),
        (("--case", "valid384"), "invalid choice"),
        (("--case", "385"), "invalid choice"),
        (("--second-vjp",), "unrecognized arguments"),
        (("--compile-diagnostic",), "only valid with --platform rocm"),
        (
            ("--platform", "rocm", "--compile-diagnostic"),
            "requires the explicit --allow-gpu",
        ),
    ],
)
def test_parser_refuses_implicit_gpu_and_scope_broadening(arguments, message, capsys):
    with pytest.raises(SystemExit) as raised:
        _PROBE._parse_args(list(arguments))
    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_private_output_is_exclusive_mode_0600(tmp_path):
    path = tmp_path / "vjp.jsonl"
    assert _PROBE.main(["--output", str(path)]) == 0
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(SystemExit):
        _PROBE._parse_args(["--output", str(path)])


def test_compile_diagnostic_requires_full_rocm_acknowledgement_and_private_output(
    tmp_path,
):
    path = tmp_path / "compile-diagnostic.jsonl"
    args = _PROBE._parse_args(
        [
            "--platform",
            "rocm",
            "--allow-gpu",
            "--compile-diagnostic",
            "--output",
            str(path),
        ]
    )
    assert args.platform == "rocm"
    assert args.allow_gpu is True
    assert args.compile_diagnostic is True
    assert args.output == path
    contract = _PROBE._compile_diagnostic_contract()
    assert contract["checked_capability_creation_or_release_authorized"] is False
    assert contract["host_reference_construction_authorized"] is False
    assert contract["executable_invocation_authorized"] is False
    assert contract["always_stop_after_compile_and_postflight"] is True


def test_valid385_parser_and_compile_diagnostic_remain_explicit_and_closed(tmp_path):
    path = tmp_path / "valid385-compile.jsonl"
    args = _PROBE._parse_args(
        [
            "--platform",
            "rocm",
            "--allow-gpu",
            "--compile-diagnostic",
            "--case",
            "valid385",
            "--output",
            str(path),
        ]
    )
    assert args.case == "valid385"
    contract = _PROBE._compile_diagnostic_contract(args.case)
    assert contract["case"] == "valid385"
    assert contract["inputs"][3]["value"] == "ones_before_385_zeros_at_and_after_385"
    assert "bitwise_positive_zero" in contract["inputs"][4]["value"]
    assert contract["compile_gate"]["exact_argument_bytes"] == 10_487_808
    assert contract["compile_gate"]["exact_output_bytes"] == 10_485_792
    assert contract["checked_capability_creation_or_release_authorized"] is False
    assert contract["host_reference_construction_authorized"] is False
    assert contract["executable_invocation_authorized"] is False


def test_t1024_parser_is_compile_diagnostic_only_and_runtime_fails_closed(
    tmp_path, capsys
):
    diagnostic = tmp_path / "t1024-diagnostic.jsonl"
    args = _PROBE._parse_args(
        [
            "--platform",
            "rocm",
            "--allow-gpu",
            "--compile-diagnostic",
            "--case",
            "all_valid_t1024",
            "--output",
            str(diagnostic),
        ]
    )
    assert args.case == "all_valid_t1024"
    assert args.compile_diagnostic is True
    with pytest.raises(SystemExit) as raised:
        _PROBE._parse_args(
            [
                "--platform",
                "rocm",
                "--allow-gpu",
                "--case",
                "all_valid_t1024",
                "--output",
                str(tmp_path / "forbidden-runtime.jsonl"),
            ]
        )
    assert raised.value.code == 2
    assert "compile-diagnostic-only" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="runtime release is unavailable"):
        _PROBE._run_rocm(
            io.StringIO(),
            lambda: dict(_CLEAN),
            _PROBE._zero_counters(),
            environment={},
            case="all_valid_t1024",
        )


def test_exact_contract_fixes_shape_dispatch_memory_and_numerical_gates():
    contract = _PROBE._exact_contract()
    assert contract["operation"] == "query_bounded_gqa_t512_forward_and_full_vjp"
    assert contract["gpu_architecture"] == "gfx1100"
    assert contract["gpu_pci_device_id"] == "0x744c"
    assert [item["shape"] for item in contract["inputs"]] == [
        [1, 512, 16, 256],
        [1, 512, 4, 256],
        [1, 512, 4, 256],
        [1, 512],
        [1, 512, 16, 256],
    ]
    assert [item["shape"] for item in contract["outputs"]] == [
        [1, 512, 16, 256],
        [1, 512, 16, 256],
        [1, 512, 4, 256],
        [1, 512, 4, 256],
    ]
    assert contract["scale_exact_fraction"] == "3/32"
    assert contract["tiles"] == {
        "query_chunk_size": 512,
        "block_q": 64,
        "block_k": 64,
        "backward_block_q": 32,
        "backward_block_k": 32,
    }
    assert contract["compile_gate"]["exact_custom_calls"] == {
        "forward": 1,
        "dq": 1,
        "dkdv": 1,
    }
    assert contract["compile_gate"]["exact_total_custom_calls"] == 3
    assert contract["compile_gate"]["sole_target_is_exact_rocm_triton"] is True
    assert (
        contract["compile_gate"]["sole_target_sha256"]
        == hashlib.sha256(_TARGET.encode()).hexdigest()
    )
    assert (
        contract["compile_gate"]["all_calls_directly_owned_by_sole_entry_computation"]
        is True
    )
    assert contract["compile_gate"]["exact_argument_bytes"] == 10_487_808
    assert contract["compile_gate"]["exact_output_bytes"] == 10_485_792
    assert contract["compile_gate"]["exact_alias_bytes"] == 0
    execution = contract["execution_contract"]
    assert execution["logical_internal_dispatches"] == {
        "forward": 1,
        "dq": 1,
        "dkdv": 1,
        "total": 3,
    }
    assert execution["checked_executable_invocations"] == 1
    assert all(
        execution[name] == 0
        for name in (
            "warmup_invocations",
            "replay_invocations",
            "second_vjp_invocations",
            "gpu_reference_invocations",
            "device_error_reduction_invocations",
            "model_invocations",
        )
    )
    assert contract["numerical_gate_per_tensor"] == {
        "finite_required": True,
        "minimum_reference_l2_norm": 1e-8,
        "relative_l2_strictly_below": {
            "output": 0.01,
            "dq": 0.03,
            "dk": 0.03,
            "dv": 0.03,
        },
        "minimum_cosine": 0.9999,
        "maximum_absolute_error": 0.02,
    }


def test_valid385_contract_adds_only_host_case_and_validation_gates():
    default = _PROBE._exact_contract()
    padded = _PROBE._exact_contract("valid385")
    assert padded["case"] == "valid385"
    assert padded["valid_tokens"] == 385
    assert padded["padding_side"] == "right"
    assert padded["loss_mask_applied_to_host_dout_before_device_put"] is True
    assert padded["compile_gate"] == default["compile_gate"]
    assert padded["execution_contract"] == default["execution_contract"]
    assert padded["tiles"] == default["tiles"]
    assert padded["numerical_gate_per_tensor"] == default["numerical_gate_per_tensor"]
    assert padded["affected_forward_output_rows_numerically_gated"] == [385, 512]
    assert all(padded["padded_zero_gates"].values())


def test_t1024_contract_fixes_two_chunks_memory_and_withholds_runtime():
    contract = _PROBE._exact_contract("all_valid_t1024")
    assert contract["case"] == "all_valid_t1024"
    assert contract["sequence_length"] == 1024
    assert contract["query_chunks"] == 2
    assert contract["numerical_gate_per_tensor"] == _PROBE._numerical_gate_contract()
    assert [item["shape"] for item in contract["inputs"]] == [
        [1, 1024, 16, 256],
        [1, 1024, 4, 256],
        [1, 1024, 4, 256],
        [1, 1024],
        [1, 1024, 16, 256],
    ]
    gate = contract["compile_gate"]
    assert gate["exact_custom_calls"] == {"forward": 2, "dq": 2, "dkdv": 2}
    assert gate["exact_total_custom_calls"] == 6
    assert "exact_marker_query_start_pairs" not in gate
    assert gate["exact_abi_query_start_pairs"] == [
        {"kind": kind, "query_start": query_start, "query_size": 512}
        for kind in ("forward", "dq", "dkdv")
        for query_start in (0, 512)
    ]
    assert gate["human_marker_evidence_is_diagnostic_only"] is True
    assert gate["exact_argument_bytes"] == 20_975_616
    assert gate["exact_output_bytes"] == 20_971_552
    assert gate["exact_alias_bytes"] == 0
    assert gate["maximum_temporary_bytes"] is None
    assert gate["exact_temporary_bytes"] == 25_232_640
    assert gate["exact_optimized_hlo_zero_custom_call_entry_fusion_helpers"] == 9
    assert contract["output_tensor_leaf_bytes"] == 20_971_520
    assert contract["compiled_output_root_bytes"] == 32
    assert contract["host_input_bytes"] == 20_975_616
    assert contract["host_fp32_reference_bytes"] == 41_943_040
    assert contract["host_oracle_scratch_bytes"] == 323_072
    assert contract["internal_accumulator_memory"] == {
        "shape": [1, 1024, 4, 256],
        "dtype": "float32",
        "leaf_bytes": 4_194_304,
        "pair_bytes": 8_388_608,
    }
    diagnostic = _PROBE._compile_diagnostic_contract("all_valid_t1024")
    assert diagnostic["host_input_bytes"] == 20_975_616
    assert diagnostic["host_fp32_reference_bytes"] == 41_943_040
    assert diagnostic["host_oracle_scratch_bytes"] == 323_072
    assert (
        diagnostic["internal_accumulator_memory"]
        == contract["internal_accumulator_memory"]
    )
    assert contract["compile_diagnostic_only"] is True
    assert (
        contract["runtime_capability_release_authorized_in_this_source_revision"]
        is False
    )
    assert contract["physical_launch_count_claimed"] is False
    assert contract["optimized_hlo_fusion_helper_inventory_pin"] == 9
    assert contract["first_capture_exact_temporary_bytes_pin"] == 25_232_640
    assert (
        contract[
            "six_calls_are_bounded_attention_custom_calls_not_physical_launch_count"
        ]
        is True
    )
    assert all(contract["required_internal_accumulator_proof"].values())


def test_t1024_shape_and_external_memory_gates_are_exact():
    signature = _PROBE._shape_signature(
        SimpleNamespace(ShapeDtypeStruct=_FakeShape),
        SimpleNamespace(bfloat16="bf16", int32="i32"),
        "all_valid_t1024",
    )
    assert [item.shape for item in signature] == [
        (1, 1024, 16, 256),
        (1, 1024, 4, 256),
        (1, 1024, 4, 256),
        (1, 1024),
        (1, 1024, 16, 256),
    ]
    memory = {
        "available": True,
        "argument_size_in_bytes": 20_975_616,
        "output_size_in_bytes": 20_971_552,
        "alias_size_in_bytes": 0,
        "temp_size_in_bytes": 25_232_640,
    }
    assert _PROBE._compiled_memory_gate(memory, "all_valid_t1024")["passed"] is True
    for name, value in (
        ("argument_size_in_bytes", 20_975_615),
        ("output_size_in_bytes", 20_971_520),
        ("alias_size_in_bytes", 1),
        ("temp_size_in_bytes", 25_232_641),
    ):
        corrupt = dict(memory)
        corrupt[name] = value
        assert (
            _PROBE._compiled_memory_gate(corrupt, "all_valid_t1024")["passed"] is False
        )


def test_static_source_binding_and_kernel_api_are_exact(monkeypatch):
    proof = _PROBE._assert_static_source_bindings()
    assert proof["passed"] is True
    assert (
        proof["delegated_compile_probe_source_sha256"]
        == "bf01187101d20362072c96f70fbb80b2f8eed88fa55e60cbf479abba6db012a2"
    )
    assert (
        proof["delegated_nonzero_probe_source_sha256"]
        == "1758567bad19e261400d027c1aab51c28dffc621ce9ad6cd819f4a1575fff0f4"
    )
    path = _PROBE._source_files()["delegated_compile_probe_source_sha256"].resolve()
    original = _PROBE._file_sha256
    monkeypatch.setattr(
        _PROBE,
        "_file_sha256",
        lambda value: "0" * 64 if Path(value).resolve() == path else original(value),
    )
    with pytest.raises(RuntimeError, match="pin changed"):
        _PROBE._assert_static_source_bindings()


def test_gfx1100_drm_binding_requires_the_sole_exact_amd_pci_device(tmp_path):
    device = tmp_path / "card1" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n")
    (device / "device").write_text("0x744c\n")
    proof = _PROBE._assert_gfx1100_drm(tmp_path)
    assert proof == {
        "passed": True,
        "architecture": "gfx1100",
        "amd_pci_vendor_id": "0x1002",
        "amd_pci_device_id": "0x744c",
        "sole_amd_drm_card": "card1",
    }
    (device / "device").write_text("0x0000\n")
    with pytest.raises(RuntimeError, match="gfx1100"):
        _PROBE._assert_gfx1100_drm(tmp_path)


def test_exact_host_inputs_cotangent_hashes_norms_and_scratch(exact_case):
    inputs, manifests, expected, reference = exact_case
    q, k, v, key_mask, dout = inputs
    assert [item["name"] for item in manifests] == ["q", "k", "v", "key_mask", "dout"]
    assert {
        item["name"]: item["sha256"] for item in manifests
    } == _PROBE._EXPECTED_INPUT_SHA256
    assert {
        item["name"]: item["sha256"] for item in reference["outputs"]
    } == _PROBE._EXPECTED_REFERENCE_SHA256
    assert reference["reference_l2_norms"] == pytest.approx(
        _PROBE._EXPECTED_REFERENCE_NORMS, abs=1e-12
    )
    assert all(reference["calibration_pin_checks"].values())
    assert (
        reference["oracle"]["conservative_accounted_numpy_array_scratch_bytes"]
        == 323_072
    )
    assert reference["oracle"]["full_t_by_t_matrix_constructed"] is False
    assert reference["oracle"][
        "observed_maximum_absolute_valid_logit"
    ] == pytest.approx(1.4235591888427734, abs=1e-12)
    assert q.dtype == k.dtype == v.dtype == dout.dtype == ml_dtypes.bfloat16
    assert np.all(key_mask == 1)
    assert np.count_nonzero(dout) == dout.size
    assert not np.array_equal(q, dout)
    assert all(array.dtype == np.float32 for array in expected)


def test_valid385_host_inputs_oracle_hashes_zero_tails_and_sensitivity(valid385_case):
    inputs, manifests, expected, reference = valid385_case
    q, k, v, key_mask, dout = inputs
    assert {item["name"]: item["sha256"] for item in manifests} == (
        _PROBE._EXPECTED_VALID385_INPUT_SHA256
    )
    assert {item["name"]: item["sha256"] for item in reference["outputs"]} == (
        _PROBE._EXPECTED_VALID385_REFERENCE_SHA256
    )
    assert reference["reference_l2_norms"] == pytest.approx(
        _PROBE._EXPECTED_VALID385_REFERENCE_NORMS, abs=1e-12
    )
    assert all(reference["calibration_pin_checks"].values())
    assert reference["case"] == "valid385"
    assert reference["valid_tokens"] == 385
    assert np.all(key_mask[:, :385] == 1)
    assert np.all(key_mask[:, 385:] == 0)
    assert np.count_nonzero(dout[:, :385]) == dout[:, :385].size
    assert _PROBE._bitwise_positive_zero(np, dout[:, 385:])
    assert q.dtype == k.dtype == v.dtype == dout.dtype == ml_dtypes.bfloat16
    for value in expected[1:]:
        assert _PROBE._bitwise_positive_zero(np, value[:, 385:])
    sensitivity = reference["sensitivity_controls"]
    assert sensitivity["alternative_reference_sha256"] == (
        _PROBE._EXPECTED_VALID385_SENSITIVITY_SHA256
    )
    assert (
        sensitivity["ignored_key_mask"]["fails_affected_output_numerical_gate"] is True
    )
    assert all(
        sensitivity["ignored_key_mask"][
            "boundary_controls_fail_individual_output_numerical_gates"
        ].values()
    )
    assert all(
        sensitivity["ignored_key_mask"][
            "gradients_exactly_equal_to_correct_reference"
        ].values()
    )
    assert sensitivity["ignored_key_mask"]["backward_sensitivity_claimed"] is False
    assert sensitivity["ignored_loss_mask"]["fails_padded_dq_zero_gate"] is True
    assert all(
        sensitivity["ignored_loss_mask"][
            "full_gradient_numerical_gate_failures"
        ].values()
    )
    assert (
        sensitivity["ignored_loss_mask"]["output_exactly_equal_to_correct_reference"]
        is True
    )
    transform = reference["host_loss_mask_transformation"]
    assert (
        transform["raw_unmasked_dout_sha256"] == _PROBE._EXPECTED_INPUT_SHA256["dout"]
    )
    assert transform["raw_unmasked_dout_elements"] == 2_097_152
    assert transform["raw_unmasked_dout_nonzero_elements"] == 2_097_152
    assert transform["active_prefix_byte_equal_to_raw"] is True
    assert transform["padded_tail_bitwise_positive_zero"] is True
    assert transform["raw_values_emitted"] is False


def test_valid385_calibration_pin_mutation_fails_closed(monkeypatch):
    corrupted = dict(_PROBE._EXPECTED_VALID385_INPUT_SHA256)
    corrupted["key_mask"] = "0" * 64
    monkeypatch.setattr(_PROBE, "_EXPECTED_VALID385_INPUT_SHA256", corrupted)
    with pytest.raises(RuntimeError, match="calibration pin changed"):
        _PROBE._construct_host_case(np, ml_dtypes, "valid385")


def test_t1024_host_case_hashes_norms_oracle_and_chunk_sensitivity_controls(
    all_valid_t1024_case,
):
    inputs, manifests, expected, reference = all_valid_t1024_case
    q, k, v, key_mask, dout = inputs
    assert {item["name"]: item["sha256"] for item in manifests} == (
        _PROBE._EXPECTED_T1024_INPUT_SHA256
    )
    assert {item["name"]: item["sha256"] for item in reference["outputs"]} == (
        _PROBE._EXPECTED_T1024_REFERENCE_SHA256
    )
    assert reference["reference_l2_norms"] == pytest.approx(
        _PROBE._EXPECTED_T1024_REFERENCE_NORMS, abs=1e-12
    )
    assert all(reference["calibration_pin_checks"].values())
    assert reference["oracle"]["valid_tokens"] == 1024
    assert reference["oracle"]["right_padded_key_mask"] is False
    assert (
        reference["oracle"]["conservative_accounted_numpy_array_scratch_bytes"]
        == 323_072
    )
    assert reference["oracle"][
        "observed_maximum_absolute_valid_logit"
    ] == pytest.approx(1.580984115600586, abs=1e-12)
    assert reference["host_memory_accounting"] == {
        "input_bytes": 20_975_616,
        "fp32_reference_bytes": 41_943_040,
        "oracle_scratch_bytes": 323_072,
    }
    assert q.shape == dout.shape == (1, 1024, 16, 256)
    assert k.shape == v.shape == (1, 1024, 4, 256)
    assert q.dtype == k.dtype == v.dtype == dout.dtype == ml_dtypes.bfloat16
    assert key_mask.dtype == np.int32 and np.all(key_mask == 1)
    assert all(np.count_nonzero(value) == value.size for value in (q, k, v, dout))
    assert all(value.dtype == np.float32 for value in expected)
    reset = reference["reset_before_q512_sensitivity_control"]
    assert reset["alternative_reference_sha256"] == (
        _PROBE._EXPECTED_T1024_RESET_SENSITIVITY_SHA256
    )
    assert reset["per_key_half_metrics"] == (
        _PROBE._EXPECTED_T1024_RESET_SENSITIVITY_METRICS
    )
    assert all(reset["first_key_half_fails_numerical_gate"].values())
    assert all(reset["second_key_half_exactly_equal_by_causality"].values())
    assert reset["control_decisive"] is True
    assert reset["accelerator_used"] is False
    omitted = reference["omit_q512_sensitivity_control"]
    assert omitted["alternative_reference_sha256"] == (
        _PROBE._EXPECTED_T1024_OMIT_Q512_SENSITIVITY_SHA256
    )
    assert omitted["per_key_half_metrics"] == (
        _PROBE._EXPECTED_T1024_OMIT_Q512_SENSITIVITY_METRICS
    )
    assert all(omitted["all_key_halves_fail_numerical_gate"].values())
    assert omitted["control_decisive"] is True
    assert omitted["accelerator_used"] is False


def test_t1024_calibration_pin_mutation_fails_closed(monkeypatch):
    corrupted = dict(_PROBE._EXPECTED_T1024_REFERENCE_SHA256)
    corrupted["dv"] = "0" * 64
    monkeypatch.setattr(_PROBE, "_EXPECTED_T1024_REFERENCE_SHA256", corrupted)
    with pytest.raises(RuntimeError, match="calibration pin changed"):
        _PROBE._construct_host_case(np, ml_dtypes, "all_valid_t1024")


def test_t1024_host_memory_pin_mutation_fails_closed(monkeypatch):
    monkeypatch.setattr(_PROBE, "_T1024_EXPECTED_HOST_REFERENCE_BYTES", 41_943_039)
    with pytest.raises(RuntimeError, match="calibration pin changed"):
        _PROBE._construct_host_case(np, ml_dtypes, "all_valid_t1024")


@pytest.mark.parametrize(
    ("sequence", "query_heads", "kv_heads", "head_dim", "query_tile", "key_tile"),
    ((9, 4, 2, 8, 3, 4), (17, 6, 3, 7, 5, 6), (32, 8, 2, 5, 8, 7)),
)
def test_tiled_forward_vjp_matches_independent_dense_small_reference(
    sequence, query_heads, kv_heads, head_dim, query_tile, key_tile
):
    rng = np.random.Generator(np.random.PCG64(9800 + sequence))
    q = rng.uniform(-0.75, 0.75, (1, sequence, query_heads, head_dim)).astype(
        np.float32
    )
    k = rng.uniform(-0.75, 0.75, (1, sequence, kv_heads, head_dim)).astype(np.float32)
    v = rng.uniform(-0.75, 0.75, k.shape).astype(np.float32)
    dout = rng.uniform(-0.4, 0.4, q.shape).astype(np.float32)
    mask = np.ones((1, sequence), dtype=np.int32)
    tiled, metadata = _PROBE._tiled_causal_gqa_forward_vjp_oracle(
        np,
        q,
        k,
        v,
        mask,
        dout,
        scale=3 / 32,
        query_tile=query_tile,
        key_tile=key_tile,
    )
    dense = _PROBE._dense_causal_gqa_forward_vjp_reference(
        np, q, k, v, mask, dout, scale=3 / 32
    )
    for actual, expected in zip(tiled, dense, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=6e-7)
    assert metadata["full_t_by_t_matrix_constructed"] is False
    assert metadata["accelerator_used"] is False


def test_tiled_vjp_matches_independent_float64_directional_finite_differences():
    rng = np.random.Generator(np.random.PCG64(4421))
    q = rng.uniform(-0.5, 0.5, (1, 5, 4, 3)).astype(np.float32)
    k = rng.uniform(-0.5, 0.5, (1, 5, 2, 3)).astype(np.float32)
    v = rng.uniform(-0.5, 0.5, (1, 5, 2, 3)).astype(np.float32)
    dout = rng.uniform(-0.3, 0.3, q.shape).astype(np.float32)
    mask = np.ones((1, 5), np.int32)
    (_output, dq, dk, dv), _metadata = _PROBE._tiled_causal_gqa_forward_vjp_oracle(
        np, q, k, v, mask, dout, scale=3 / 32, query_tile=3, key_tile=2
    )

    def scalar_loss(q_item, k_item, v_item):
        q64 = np.asarray(q_item[0], np.float64)
        k64 = np.repeat(np.asarray(k_item[0], np.float64), 2, axis=1)
        v64 = np.repeat(np.asarray(v_item[0], np.float64), 2, axis=1)
        logits = np.einsum("thd,shd->hts", q64, k64) * (3 / 32)
        logits[:, np.triu(np.ones((5, 5), dtype=np.bool_), k=1)] = -np.inf
        logits -= np.max(logits, axis=2, keepdims=True)
        probability = np.exp(logits)
        probability /= np.sum(probability, axis=2, keepdims=True)
        value = np.einsum("hts,shd->thd", probability, v64)
        return float(np.sum(value * np.asarray(dout[0], np.float64)))

    epsilon = 1e-4
    for index, (primal, gradient) in enumerate(zip((q, k, v), (dq, dk, dv))):
        direction = rng.normal(size=primal.shape).astype(np.float32)
        plus = [q, k, v]
        minus = [q, k, v]
        plus[index] = primal + epsilon * direction
        minus[index] = primal - epsilon * direction
        finite_difference = (scalar_loss(*plus) - scalar_loss(*minus)) / (2 * epsilon)
        analytic = float(
            np.sum(np.asarray(gradient, np.float64) * np.asarray(direction, np.float64))
        )
        assert analytic == pytest.approx(finite_difference, rel=2e-3, abs=2e-4)


def test_padded_tiled_vjp_matches_independent_float64_directional_differences():
    rng = np.random.Generator(np.random.PCG64(8441))
    q = rng.uniform(-0.5, 0.5, (1, 6, 4, 3)).astype(np.float32)
    k = rng.uniform(-0.5, 0.5, (1, 6, 2, 3)).astype(np.float32)
    v = rng.uniform(-0.5, 0.5, (1, 6, 2, 3)).astype(np.float32)
    dout = rng.uniform(-0.3, 0.3, q.shape).astype(np.float32)
    dout[:, 4:] = 0
    mask = np.zeros((1, 6), np.int32)
    mask[:, :4] = 1
    (_output, dq, dk, dv), _metadata = _PROBE._tiled_causal_gqa_forward_vjp_oracle(
        np, q, k, v, mask, dout, scale=3 / 32, query_tile=3, key_tile=2
    )

    def scalar_loss(q_item, k_item, v_item):
        q64 = np.asarray(q_item[0], np.float64)
        k64 = np.repeat(np.asarray(k_item[0], np.float64), 2, axis=1)
        v64 = np.repeat(np.asarray(v_item[0], np.float64), 2, axis=1)
        logits = np.einsum("thd,shd->hts", q64, k64) * (3 / 32)
        invalid = np.triu(np.ones((6, 6), dtype=np.bool_), k=1)
        invalid[:, 4:] = True
        logits[:, invalid] = -np.inf
        logits -= np.max(logits, axis=2, keepdims=True)
        probability = np.exp(logits)
        probability /= np.sum(probability, axis=2, keepdims=True)
        value = np.einsum("hts,shd->thd", probability, v64)
        return float(np.sum(value * np.asarray(dout[0], np.float64)))

    epsilon = 1e-4
    for index, (primal, gradient) in enumerate(zip((q, k, v), (dq, dk, dv))):
        direction = rng.normal(size=primal.shape).astype(np.float32)
        plus = [q, k, v]
        minus = [q, k, v]
        plus[index] = primal + epsilon * direction
        minus[index] = primal - epsilon * direction
        finite_difference = (scalar_loss(*plus) - scalar_loss(*minus)) / (2 * epsilon)
        analytic = float(
            np.sum(np.asarray(gradient, np.float64) * np.asarray(direction, np.float64))
        )
        assert analytic == pytest.approx(finite_difference, rel=2e-3, abs=2e-4)


def test_padded_oracle_proves_key_mask_and_loss_mask_sensitivity_limits():
    rng = np.random.Generator(np.random.PCG64(8612))
    q = rng.uniform(-0.6, 0.6, (1, 7, 4, 3)).astype(np.float32)
    k = rng.uniform(-0.6, 0.6, (1, 7, 2, 3)).astype(np.float32)
    v = rng.uniform(-0.6, 0.6, k.shape).astype(np.float32)
    raw_dout = rng.uniform(-0.4, 0.4, q.shape).astype(np.float32)
    masked_dout = raw_dout.copy()
    masked_dout[:, 4:] = 0
    mask = np.zeros((1, 7), np.int32)
    mask[:, :4] = 1
    correct, _ = _PROBE._tiled_causal_gqa_forward_vjp_oracle(
        np, q, k, v, mask, masked_dout, scale=3 / 32, query_tile=3, key_tile=2
    )
    ignored_key, _ = _PROBE._tiled_causal_gqa_forward_vjp_oracle(
        np,
        q,
        k,
        v,
        np.ones_like(mask),
        masked_dout,
        scale=3 / 32,
        query_tile=3,
        key_tile=2,
    )
    assert not np.array_equal(ignored_key[0], correct[0])
    assert all(
        np.array_equal(wrong, expected)
        for wrong, expected in zip(ignored_key[1:], correct[1:], strict=True)
    )
    ignored_loss, _ = _PROBE._tiled_causal_gqa_forward_vjp_oracle(
        np,
        q,
        k,
        v,
        mask,
        raw_dout,
        scale=3 / 32,
        query_tile=3,
        key_tile=2,
        _require_loss_masked_padding=False,
    )
    assert np.array_equal(ignored_loss[0], correct[0])
    assert all(
        not np.array_equal(wrong, expected)
        for wrong, expected in zip(ignored_loss[1:], correct[1:], strict=True)
    )


def test_all_valid_two_chunk_oracle_accumulates_both_cotangent_halves():
    rng = np.random.Generator(np.random.PCG64(1024512))
    q = rng.uniform(-0.6, 0.6, (1, 8, 4, 3)).astype(np.float32)
    k = rng.uniform(-0.6, 0.6, (1, 8, 2, 3)).astype(np.float32)
    v = rng.uniform(-0.6, 0.6, k.shape).astype(np.float32)
    dout = rng.uniform(-0.4, 0.4, q.shape).astype(np.float32)
    mask = np.ones((1, 8), np.int32)
    full, _ = _PROBE._tiled_causal_gqa_forward_vjp_oracle(
        np, q, k, v, mask, dout, scale=3 / 32, query_tile=2, key_tile=3
    )
    first_dout = dout.copy()
    first_dout[:, 4:] = 0
    second_dout = dout.copy()
    second_dout[:, :4] = 0
    first, _ = _PROBE._tiled_causal_gqa_forward_vjp_oracle(
        np, q, k, v, mask, first_dout, scale=3 / 32, query_tile=2, key_tile=3
    )
    second, _ = _PROBE._tiled_causal_gqa_forward_vjp_oracle(
        np, q, k, v, mask, second_dout, scale=3 / 32, query_tile=2, key_tile=3
    )
    assert np.array_equal(first[0], full[0])
    assert np.array_equal(second[0], full[0])
    for full_gradient, first_gradient, second_gradient in zip(
        full[1:], first[1:], second[1:], strict=True
    ):
        np.testing.assert_allclose(
            first_gradient + second_gradient,
            full_gradient,
            rtol=3e-6,
            atol=6e-7,
        )


def test_exact_t512_tiled_oracle_matches_dense_reference(exact_case):
    inputs, _manifests, expected, _reference = exact_case
    dense = _PROBE._dense_causal_gqa_forward_vjp_reference(np, *inputs, scale=3 / 32)
    for actual, dense_item in zip(expected, dense, strict=True):
        np.testing.assert_allclose(actual, dense_item, rtol=3e-6, atol=6e-7)


def test_valid385_t512_tiled_oracle_matches_dense_reference(valid385_case):
    inputs, _manifests, expected, _reference = valid385_case
    dense = _PROBE._dense_causal_gqa_forward_vjp_reference(np, *inputs, scale=3 / 32)
    for actual, dense_item in zip(expected, dense, strict=True):
        np.testing.assert_allclose(actual, dense_item, rtol=3e-6, atol=8e-7)


def test_small_right_padded_loss_masked_tiled_oracle_matches_dense_reference():
    rng = np.random.Generator(np.random.PCG64(7744))
    q = rng.uniform(-0.7, 0.7, (1, 9, 4, 5)).astype(np.float32)
    k = rng.uniform(-0.7, 0.7, (1, 9, 2, 5)).astype(np.float32)
    v = rng.uniform(-0.7, 0.7, k.shape).astype(np.float32)
    dout = rng.uniform(-0.5, 0.5, q.shape).astype(np.float32)
    dout[:, 5:] = 0
    mask = np.zeros((1, 9), np.int32)
    mask[:, :5] = 1
    tiled, metadata = _PROBE._tiled_causal_gqa_forward_vjp_oracle(
        np, q, k, v, mask, dout, scale=3 / 32, query_tile=4, key_tile=3
    )
    dense = _PROBE._dense_causal_gqa_forward_vjp_reference(
        np, q, k, v, mask, dout, scale=3 / 32
    )
    for actual, expected in zip(tiled, dense, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=6e-7)
    assert metadata["valid_tokens"] == 5
    assert metadata["right_padded_key_mask"] is True
    for value in tiled[1:]:
        assert _PROBE._bitwise_positive_zero(np, value[:, 5:])


def test_oracle_rejects_invalid_padding_loss_mask_and_degenerate_shapes():
    q = np.ones((1, 8, 4, 3), np.float32)
    k = np.ones((1, 8, 2, 3), np.float32)
    v = np.ones_like(k)
    dout = np.ones_like(q)
    mask = np.ones((1, 8), np.int32)
    mask[:, -1] = 0
    with pytest.raises(RuntimeError, match="bitwise-positive-zero"):
        _PROBE._tiled_causal_gqa_forward_vjp_oracle(
            np, q, k, v, mask, dout, scale=3 / 32
        )
    with pytest.raises(RuntimeError, match="binary int32"):
        _PROBE._tiled_causal_gqa_forward_vjp_oracle(
            np, q, k, v, np.full_like(mask, 2), dout, scale=3 / 32
        )
    with pytest.raises(RuntimeError, match="binary int32"):
        _PROBE._tiled_causal_gqa_forward_vjp_oracle(
            np, q, k, v, np.ones_like(mask, dtype=np.bool_), dout, scale=3 / 32
        )
    nonprefix = np.ones_like(mask)
    nonprefix[:, 3] = 0
    with pytest.raises(RuntimeError, match="right-padded"):
        _PROBE._tiled_causal_gqa_forward_vjp_oracle(
            np, q, k, v, nonprefix, dout, scale=3 / 32
        )
    with pytest.raises(RuntimeError, match="nonempty"):
        _PROBE._tiled_causal_gqa_forward_vjp_oracle(
            np, q, k, v, np.zeros_like(mask), np.zeros_like(dout), scale=3 / 32
        )
    negative_zero_dout = dout.copy()
    negative_zero_dout[:, -1] = np.float32(-0.0)
    with pytest.raises(RuntimeError, match="bitwise-positive-zero"):
        _PROBE._tiled_causal_gqa_forward_vjp_oracle(
            np, q, k, v, mask, negative_zero_dout, scale=3 / 32
        )
    with pytest.raises(RuntimeError, match="shapes"):
        _PROBE._tiled_causal_gqa_forward_vjp_oracle(
            np,
            q,
            k[:, :, :, :2],
            v[:, :, :, :2],
            np.ones_like(mask),
            dout,
            scale=3 / 32,
        )


def _stablehlo_calls(
    *,
    target: str = _TARGET,
    metadata: dict[str, str] | None = None,
    markers: dict[str, str] | None = None,
    extra: str = "",
) -> str:
    metadata = metadata or dict.fromkeys(
        _PROBE._EXPECTED_MARKERS, "query_start=0 query_size=512"
    )
    markers = markers or _PROBE._EXPECTED_MARKERS
    definitions = [
        f'#loc{index} = loc("{markers[kind]} {metadata[kind]}")'
        for index, kind in enumerate(_PROBE._EXPECTED_MARKERS)
    ]
    calls = [
        f'    %{index} = stablehlo.custom_call @"{target}"() : () -> tensor<1xbf16> loc(#loc{index})'
        for index in range(3)
    ]
    return "\n".join(
        [
            *definitions,
            "module {",
            "  func.func public @main() attributes {probe = true} {",
            *calls,
            "    return",
            "  }",
            "}",
            extra,
        ]
    )


def _optimized_hlo_calls(
    *,
    target: str = _TARGET,
    metadata: dict[str, str] | None = None,
    markers: dict[str, str] | None = None,
    extra: str = "",
) -> str:
    metadata = metadata or dict.fromkeys(
        _PROBE._EXPECTED_MARKERS, "query_start=0 query_size=512"
    )
    markers = markers or _PROBE._EXPECTED_MARKERS
    calls = [
        (
            f'  %{index} = bf16[1] custom-call(), custom_call_target="{target}", '
            f'op_name="{markers[kind]} {metadata[kind]}"'
        )
        for index, kind in enumerate(_PROBE._EXPECTED_MARKERS)
    ]
    return "\n".join(
        ["ENTRY main (p0: bf16[1]{0}) -> bf16[1]{0} {", *calls, "}", extra]
    )


def _t1024_two_chunk_ir(dialect: str) -> str:
    specs = _PROBE._expected_call_specs("all_valid_t1024")
    if dialect == "stablehlo":
        query = "tensor<1x512x16x256xbf16>"
        key_value = "tensor<1x1024x4x256xbf16>"
        mask = "tensor<1x1024xi32>"
        lse = "tensor<1x16x512xf32>"
        accumulator = "tensor<1x1024x4x256xf32>"
        definitions = [
            f'#loc{index} = loc("{item["marker"]} query_start={item["query_start"]} query_size=512")'
            for index, item in enumerate(specs)
        ]
        parameters = (
            ("q0", query),
            ("q512", query),
            ("k", key_value),
            ("v", key_value),
            ("mask", mask),
            ("dout0", query),
            ("dout512", query),
            ("acc_dk", accumulator),
            ("acc_dv", accumulator),
        )
        arguments = ", ".join(f"%{name}: {shape}" for name, shape in parameters)
        aliases = (
            "#stablehlo.output_operand_alias<output_tuple_indices = [0], "
            "operand_index = 7, operand_tuple_indices = []>, "
            "#stablehlo.output_operand_alias<output_tuple_indices = [1], "
            "operand_index = 8, operand_tuple_indices = []>"
        )
        lines = [*definitions, "module {", f"  func.func public @main({arguments}) {{"]
        forward_types = f"({query}, {key_value}, {key_value}, {mask})"
        backward_types = (
            f"({query}, {key_value}, {key_value}, {mask}, {query}, {query}, {lse})"
        )
        for index, start in enumerate((0, 512)):
            lines.append(
                f'    %fw{start}:2 = stablehlo.custom_call @"{_TARGET}"'
                f"(%q{start}, %k, %v, %mask) : {forward_types} -> "
                f"({query}, {lse}) loc(#loc{index})"
            )
        for index, start in enumerate((0, 512), start=2):
            lines.append(
                f'    %dq{start} = stablehlo.custom_call @"{_TARGET}"'
                f"(%q{start}, %k, %v, %mask, %fw{start}#0, "
                f"%dout{start}, %fw{start}#1) : {backward_types} -> {query} "
                f"loc(#loc{index})"
            )
        dkdv_types = f"{backward_types[:-1]}, {accumulator}, {accumulator})"
        lines.append(
            f'    %dk0:2 = stablehlo.custom_call @"{_TARGET}"'
            f"(%q0, %k, %v, %mask, %fw0#0, %dout0, %fw0#1, "
            f"%acc_dk, %acc_dv) "
            f"{{output_operand_aliases = [{aliases}]}} : {dkdv_types} -> "
            f"({accumulator}, {accumulator}) loc(#loc4)"
        )
        lines.append(
            f'    %dk512:2 = stablehlo.custom_call @"{_TARGET}"'
            f"(%q512, %k, %v, %mask, %fw512#0, %dout512, %fw512#1, "
            f"%dk0#0, %dk0#1) "
            f"{{output_operand_aliases = [{aliases}]}} : {dkdv_types} -> "
            f"({accumulator}, {accumulator}) loc(#loc5)"
        )
        return "\n".join([*lines, "    return", "  }", "}"])
    if dialect == "optimized_hlo":
        query = "bf16[1,512,16,256]{3,2,1,0}"
        key_value = "bf16[1,1024,4,256]{3,2,1,0}"
        mask = "s32[1,1024]{1,0}"
        lse = "f32[1,16,512]{2,1,0}"
        accumulator = "f32[1,1024,4,256]"
        aliases = "{{0}: (7, {}), {1}: (8, {})}"
        lines = ["HloModule t1024"]
        for index in range(9):
            lines.extend(
                [
                    f"helper_{index} {{",
                    f"  ROOT %helper_root_{index} = () tuple()",
                    "}",
                ]
            )
        lines.extend(
            [
                "ENTRY main () -> () {",
                f"  %q0 = {query} parameter(0)",
                f"  %q512 = {query} parameter(1)",
                f"  %k = {key_value} parameter(2)",
                f"  %v = {key_value} parameter(3)",
                f"  %mask = {mask} parameter(4)",
                f"  %dout0 = {query} parameter(5)",
                f"  %dout512 = {query} parameter(6)",
                f"  %acc_dk = {accumulator} parameter(7)",
                f"  %acc_dv = {accumulator} parameter(8)",
            ]
        )
        for index, start in enumerate((0, 512)):
            lines.extend(
                [
                    f"  %fw{start} = ({query}, {lse}) custom-call(%q{start}, %k, %v, %mask), "
                    f'custom_call_target="{_TARGET}", op_name="{specs[index]["marker"]} '
                    f'query_start={start} query_size=512"',
                    f"  %fw{start}_out = {query} get-tuple-element(%fw{start}), index=0",
                    f"  %fw{start}_lse = {lse} get-tuple-element(%fw{start}), index=1",
                ]
            )
        for index, start in enumerate((0, 512), start=2):
            lines.append(
                f"  %dq{start} = {query} custom-call(%q{start}, %k, %v, %mask, "
                f"%fw{start}_out, %dout{start}, %fw{start}_lse), "
                f'custom_call_target="{_TARGET}", op_name="{specs[index]["marker"]} '
                f'query_start={start} query_size=512"'
            )
        q0_operands = "%q0, %k, %v, %mask, %fw0_out, %dout0, %fw0_lse, %acc_dk, %acc_dv"
        lines.append(
            f"  %dk0 = ({accumulator}, {accumulator}) custom-call({q0_operands}), "
            f'custom_call_target="{_TARGET}", output_to_operand_aliasing={aliases}, '
            f'op_name="{specs[4]["marker"]} query_start=0 query_size=512"'
        )
        lines.extend(
            (
                f"  %gte0 = {accumulator} get-tuple-element(%dk0), index=0",
                f"  %gte1 = {accumulator} get-tuple-element(%dk0), index=1",
            )
        )
        q512_operands = (
            "%q512, %k, %v, %mask, %fw512_out, %dout512, %fw512_lse, %gte0, %gte1"
        )
        lines.append(
            f"  %dk512 = ({accumulator}, {accumulator}) custom-call({q512_operands}), "
            f'custom_call_target="{_TARGET}", output_to_operand_aliasing={aliases}, '
            f'op_name="{specs[5]["marker"]} query_start=512 query_size=512"'
        )
        lines.extend(
            f"  %fusion_{index} = () fusion(), calls=%helper_{index}"
            for index in range(9)
        )
        return "\n".join([*lines, "  ROOT %root = () tuple()", "}"])
    raise ValueError("unsupported test dialect")


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
def test_t1024_strict_ir_accepts_six_calls_aliases_and_accumulator_chain(dialect):
    summary = _PROBE._strict_vjp_ir_summary(
        _t1024_two_chunk_ir(dialect), dialect, "all_valid_t1024"
    )
    assert summary["passed"] is True
    assert summary["custom_call_count"] == 6
    assert summary["marker_call_counts"] == {"forward": 2, "dq": 2, "dkdv": 2}
    assert all(item["count"] == 1 for item in summary["marker_query_start_counts"])
    assert summary["signature_call_counts"] == {
        "forward": 2,
        "dq": 2,
        "dkdv": 2,
    }
    assert all(item["count"] == 1 for item in summary["signature_query_start_counts"])
    assert all(call["signature_classification"]["passed"] for call in summary["calls"])
    assert all(
        call["signature_classification"]["raw_ir_emitted"] is False
        and call["signature_classification"]["raw_symbols_emitted"] is False
        for call in summary["calls"]
    )
    if dialect == "optimized_hlo":
        ownership = summary["entry_call_ownership"]
        assert ownership["allowed_independent_entry_fusion_helper_count"] == 9
        assert len(ownership["allowed_independent_entry_fusion_helpers"]) == 9
        assert ownership["forbidden_container_count"] == 0
        assert ownership["forbidden_containers"] == []
    dataflow = summary["two_chunk_accumulator_dataflow"]
    assert dataflow["passed"] is True
    assert all(dataflow["checks"].values())
    assert dataflow["q0_aliases"]["exact_internal_aliases_7_to_0_and_8_to_1"] is True
    assert dataflow["q512_aliases"]["exact_internal_aliases_7_to_0_and_8_to_1"] is True
    assert dataflow["accumulator_memory"] == {
        "shape": [1, 1024, 4, 256],
        "dtype": "float32",
        "leaf_bytes": 4_194_304,
        "pair_bytes": 8_388_608,
    }
    assert dataflow["raw_ir_emitted"] is False
    assert dataflow["raw_symbols_emitted"] is False


def test_t1024_stablehlo_accepts_exact_bare_target_with_balanced_operands():
    text = _t1024_two_chunk_ir("stablehlo").replace(f'@"{_TARGET}"', f"@{_TARGET}")
    summary = _PROBE._strict_vjp_ir_summary(text, "stablehlo", "all_valid_t1024")
    assert summary["passed"] is True
    assert all(
        call["signature_classification"]["parse_diagnostics"]["target_syntax_passed"]
        for call in summary["calls"]
    )
    assert [
        call["signature_classification"]["operand_count"] for call in summary["calls"]
    ] == [4, 4, 7, 7, 9, 9]


def _t1024_stablehlo_actual_layout_attribute_form() -> str:
    def layout(rank: int) -> str:
        minor_to_major = ", ".join(str(index) for index in reversed(range(rank)))
        return f"dense<[{minor_to_major}]> : tensor<{rank}xindex>"

    call_ranks = {
        "fw": ([4, 4, 4, 2], [4, 3]),
        "dq": ([4, 4, 4, 2, 4, 4, 3], [4]),
        "dk": ([4, 4, 4, 2, 4, 4, 3, 4, 4], [4, 4]),
    }
    lines = _t1024_two_chunk_ir("stablehlo").splitlines()
    for index, line in enumerate(lines):
        if "stablehlo.custom_call" not in line:
            continue
        lhs = line.split("=", 1)[0]
        kind = next(kind for kind in call_ranks if f"%{kind}" in lhs)
        operand_ranks, result_ranks = call_ranks[kind]
        layouts = (
            "operand_layouts = ["
            + ", ".join(layout(rank) for rank in operand_ranks)
            + "], result_layouts = ["
            + ", ".join(layout(rank) for rank in result_ranks)
            + "]"
        )
        if "{output_operand_aliases" in line:
            lines[index] = line.replace(
                "{output_operand_aliases",
                f"{{{layouts}, output_operand_aliases",
                1,
            )
        else:
            lines[index] = line.replace(") : (", f") {{{layouts}}} : (", 1)
    return "\n".join(lines)


def test_t1024_stablehlo_actual_layout_attributes_bind_only_final_signature():
    text = _t1024_stablehlo_actual_layout_attribute_form()
    summary = _PROBE._strict_vjp_ir_summary(text, "stablehlo", "all_valid_t1024")
    assert summary["passed"] is True
    classifications = [call["signature_classification"] for call in summary["calls"]]
    assert [item["operand_count"] for item in classifications] == [4, 4, 7, 7, 9, 9]
    assert [item["result_count"] for item in classifications] == [2, 2, 1, 1, 2, 2]
    assert all(None not in item["operand_shapes"] for item in classifications)
    assert all(None not in item["result_shapes"] for item in classifications)
    for classification in classifications:
        final_type = classification["parse_diagnostics"]["final_function_type"]
        assert final_type["top_level_colon_count_before_arrow"] == 1
        assert final_type["top_level_arrow_count"] == 1
        assert final_type["passed"] is True
        assert final_type["raw_type_text_emitted"] is False


def test_stablehlo_final_function_type_ignores_nested_attribute_type_spoofs():
    tail = (
        "{operand_layouts = [dense<[3, 2, 1, 0]> : tensor<4xi64>], "
        "nested = {spoof = (tensor<4xi64>) -> tensor<4xi64>}} : "
        "(tensor<1x512x16x256xbf16>, tensor<1x1024xi32>) -> "
        "tensor<1x512x16x256xbf16> loc(#loc0)"
    )
    parsed = _PROBE._stablehlo_final_function_type(tail)
    assert parsed["passed"] is True
    assert parsed["input_type_count"] == 2
    assert parsed["result_type_count"] == 1
    assert parsed["input_shapes"] == ["bf16[1,512,16,256]", "i32[1,1024]"]
    assert parsed["result_shapes"] == ["bf16[1,512,16,256]"]


@pytest.mark.parametrize(
    ("tail", "failed_check"),
    (
        (
            ": (tensor<1xbf16>) -> tensor<1xbf16> : (tensor<1xbf16>) -> tensor<1xbf16>",
            "exactly_one_top_level_function_arrow",
        ),
        (
            ": spoof : (tensor<1xbf16>) -> tensor<1xbf16>",
            "exactly_one_top_level_signature_colon_before_arrow",
        ),
        (
            ": (tensor<1xbf16>]) -> tensor<1xbf16>",
            "all_mixed_delimiters_balanced",
        ),
        (
            ": (tensor<1xbf16> -> tensor<1xbf16>",
            "all_mixed_delimiters_balanced",
        ),
        (
            ": (tensor<1xbf16>) -> tensor<1xbf16> tensor<4xi64>",
            "no_additional_arrow_or_tensor_type_after_result",
        ),
        (
            ": (tensor<1xbf16>) -> tensor<1xbf16> balanced_junk",
            "suffix_is_empty_or_one_exact_location",
        ),
        (
            ": (tensor<1xbf16>) -> tensor<1xbf16> loc(#loc0) trailing_junk",
            "suffix_is_empty_or_one_exact_location",
        ),
        (
            ": (tensor<1xbf16>) -> tensor<1xbf16> loc(#loc0) loc(#loc1)",
            "suffix_is_empty_or_one_exact_location",
        ),
        (
            ": (tensor<1xbf16>) -> tensor<1xbf16> loc(#loc0",
            "suffix_delimiters_balanced",
        ),
    ),
)
def test_stablehlo_final_function_type_adversaries_fail_closed(tail, failed_check):
    parsed = _PROBE._stablehlo_final_function_type(tail)
    assert parsed["passed"] is False
    assert parsed["checks"][failed_check] is False
    assert parsed["raw_type_text_emitted"] is False


def test_stablehlo_final_function_type_never_filters_unrecognized_types():
    parsed = _PROBE._stablehlo_final_function_type(
        ": (tensor<4xi64>, tensor<1xbf16>) -> tensor<1xbf16>"
    )
    assert parsed["passed"] is False
    assert parsed["input_type_count"] == 2
    assert parsed["input_shapes"] == [None, "bf16[1]"]


def _t1024_optimized_actual_annotation_form() -> str:
    text = _t1024_two_chunk_ir("optimized_hlo")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "%dq0 =" in line:
            lines[index] = line.replace("%dout0", "/*index=5*/ %dout0")
        elif "%dq512 =" in line:
            lines[index] = line.replace("%dout512", "/*index=5*/ %dout512")
        elif "%dk0 =" in line:
            lines[index] = line.replace("%acc_dk", "/*index=7*/ %acc_dk")
        elif "%dk512 =" in line:
            lines[index] = line.replace("%gte0", "/*index=7*/ %gte0")
    text = "\n".join(lines)
    for shape in (
        "bf16[1,512,16,256]",
        "bf16[1,1024,4,256]",
        "s32[1,1024]",
        "f32[1,16,512]",
        "f32[1,1024,4,256]",
    ):
        text = text.replace(f"{shape}{{", f"{shape} {{")
    return text


def test_t1024_optimized_actual_annotations_and_spaced_layouts_pass_exactly():
    summary = _PROBE._strict_vjp_ir_summary(
        _t1024_optimized_actual_annotation_form(),
        "optimized_hlo",
        "all_valid_t1024",
    )
    assert summary["passed"] is True
    assert summary["marker_counts_are_resolved_reference_diagnostics_only"] is True
    assert summary["signature_call_counts"] == {
        "forward": 2,
        "dq": 2,
        "dkdv": 2,
    }
    annotations = [
        call["signature_classification"]["parse_diagnostics"]["call_operand_parse"][
            "operand_annotation_count"
        ]
        for call in summary["calls"]
    ]
    assert annotations == [0, 0, 1, 1, 1, 1]
    for call in summary["calls"]:
        classification = call["signature_classification"]
        assert classification["passed"] is True
        assert (
            classification["parse_diagnostics"][
                "all_operands_resolve_to_entry_instructions"
            ]
            is True
        )
        assert (
            classification["parse_diagnostics"]["unresolved_operand_symbol_sha256"]
            == []
        )
    dataflow = summary["two_chunk_accumulator_dataflow"]
    assert dataflow["sanitized_dataflow"]["q512_operand_count"] == 9
    assert dataflow["sanitized_dataflow"]["operand_annotation_scan"] == {
        "annotation_count": 4,
        "malformed_count": 0,
        "passed": True,
    }


@pytest.mark.parametrize(
    "bad_annotation",
    (
        "/*index=7",
        "/*index=7 /*index=8*/",
        "/*not_an_index=7*/",
    ),
)
def test_t1024_optimized_malformed_operand_annotations_fail_closed(
    bad_annotation,
):
    text = _t1024_optimized_actual_annotation_form().replace(
        "/*index=7*/", bad_annotation, 1
    )
    summary = _PROBE._strict_vjp_ir_summary(text, "optimized_hlo", "all_valid_t1024")
    assert summary["passed"] is False
    assert any(
        call["signature_classification"]["parse_diagnostics"]["call_operand_parse"][
            "malformed_operand_annotation_count"
        ]
        > 0
        for call in summary["calls"]
    )


def test_t1024_optimized_annotations_cannot_hide_unresolved_or_junk_operands():
    text = _t1024_optimized_actual_annotation_form().replace(
        "/*index=7*/ %acc_dk", "/*index=7*/ %not_in_entry_graph", 1
    )
    summary = _PROBE._strict_vjp_ir_summary(text, "optimized_hlo", "all_valid_t1024")
    assert summary["passed"] is False
    dkdv = next(
        call
        for call in summary["calls"]
        if call["query_start"]["expected"] == 0
        and call["signature_classification"]["operand_count"] == 9
    )
    diagnostics = dkdv["signature_classification"]["parse_diagnostics"]
    assert diagnostics["all_operands_resolve_to_entry_instructions"] is False
    assert len(diagnostics["unresolved_operand_symbol_sha256"]) == 1

    junk = _t1024_optimized_actual_annotation_form().replace(
        "/*index=7*/ %acc_dk", "/*index=7*/ junk %acc_dk", 1
    )
    assert (
        _PROBE._strict_vjp_ir_summary(junk, "optimized_hlo", "all_valid_t1024")[
            "passed"
        ]
        is False
    )


def test_t1024_optimized_layout_stripping_rejects_residual_junk():
    text = _t1024_optimized_actual_annotation_form().replace(
        ") custom-call(%q0, %k, %v, %mask)",
        ") residual_junk custom-call(%q0, %k, %v, %mask)",
        1,
    )
    summary = _PROBE._strict_vjp_ir_summary(text, "optimized_hlo", "all_valid_t1024")
    assert summary["passed"] is False
    assert (
        summary["calls"][0]["signature_classification"]["parse_diagnostics"][
            "result_shape_parse"
        ]["checks"]["only_single_leaf_or_flat_tuple_shape_grammar_remains"]
        is False
    )


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
def test_t1024_signature_pairing_is_independent_of_custom_call_order(dialect):
    lines = _t1024_two_chunk_ir(dialect).splitlines()
    first = next(index for index, line in enumerate(lines) if "%dq0 =" in line)
    second = next(index for index, line in enumerate(lines) if "%dq512 =" in line)
    lines[first], lines[second] = lines[second], lines[first]
    summary = _PROBE._strict_vjp_ir_summary(
        "\n".join(lines), dialect, "all_valid_t1024"
    )
    assert summary["passed"] is True
    assert summary["calls"][2]["kind"] == "dq"
    assert summary["calls"][2]["query_start"]["expected"] == 512
    assert summary["calls"][3]["kind"] == "dq"
    assert summary["calls"][3]["query_start"]["expected"] == 0


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
@pytest.mark.parametrize("metadata_failure", ("missing", "ambiguous"))
def test_t1024_query_start_requires_exactly_one_of_two_canonical_parses(
    dialect, metadata_failure
):
    text = _t1024_two_chunk_ir(dialect)
    replacement = (
        "query_size=512"
        if metadata_failure == "missing"
        else "query_start=0 query_start=512 query_size=512"
    )
    text = text.replace("query_start=0 query_size=512", replacement, 1)
    summary = _PROBE._strict_vjp_ir_summary(text, dialect, "all_valid_t1024")
    assert summary["passed"] is False
    call = summary["calls"][0]
    assert call["query_start"]["passed"] is False
    assert len(call["query_start_candidate_parses"]) == 2
    assert not any(item["passed"] for item in call["query_start_candidate_parses"])


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
def test_t1024_expected_markers_are_diagnostic_only_and_cannot_spoof_abi(dialect):
    text = _t1024_two_chunk_ir(dialect).replace(
        "query_bounded_gqa_dq_q0", "query_bounded_gqa_forward_q0", 1
    )
    summary = _PROBE._strict_vjp_ir_summary(text, dialect, "all_valid_t1024")
    assert summary["passed"] is True
    assert summary["signature_call_counts"] == {
        "forward": 2,
        "dq": 2,
        "dkdv": 2,
    }
    dq0 = next(
        call
        for call in summary["calls"]
        if call["kind"] == "dq" and call["query_start"]["expected"] == 0
    )
    assert dq0["signature_classification"]["matching_kinds"] == ["dq"]

    if dialect == "stablehlo":
        spoofed_wrong_abi = text.replace(
            "-> tensor<1x512x16x256xbf16> loc(#loc2)",
            "-> tensor<1x1024x4x256xbf16> loc(#loc2)",
            1,
        )
    else:
        spoofed_wrong_abi = text.replace(
            "%dq0 = bf16[1,512,16,256]{3,2,1,0} custom-call",
            "%dq0 = bf16[1,1024,4,256]{3,2,1,0} custom-call",
            1,
        )
    assert (
        _PROBE._strict_vjp_ir_summary(spoofed_wrong_abi, dialect, "all_valid_t1024")[
            "passed"
        ]
        is False
    )


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
def test_t1024_missing_markers_pass_but_unexpected_lookalikes_fail(dialect):
    marker_free = _t1024_two_chunk_ir(dialect)
    for marker in (
        "query_bounded_gqa_forward_q0",
        "query_bounded_gqa_forward_q512",
        "query_bounded_gqa_dq_q0",
        "query_bounded_gqa_dq_q512",
        "query_bounded_gqa_dkdv_q0",
        "query_bounded_gqa_dkdv_q512",
    ):
        marker_free = marker_free.replace(marker, "source_name_not_preserved")
    summary = _PROBE._strict_vjp_ir_summary(marker_free, dialect, "all_valid_t1024")
    assert summary["passed"] is True
    assert summary["marker_call_counts"] == {"forward": 0, "dq": 0, "dkdv": 0}

    lookalike = marker_free.replace(
        "source_name_not_preserved", "query_bounded_gqa_forward_q0_spoof", 1
    )
    rejected = _PROBE._strict_vjp_ir_summary(lookalike, dialect, "all_valid_t1024")
    assert rejected["unexpected_bounded_marker_occurrences"] == 1
    assert rejected["passed"] is False


def test_t1024_optimized_helper_inventory_is_exactly_nine():
    lines = _t1024_two_chunk_ir("optimized_hlo").splitlines()
    helper = lines.index("helper_8 {")
    del lines[helper : helper + 3]
    lines.remove("  %fusion_8 = () fusion(), calls=%helper_8")
    summary = _PROBE._strict_vjp_ir_summary(
        "\n".join(lines), "optimized_hlo", "all_valid_t1024"
    )
    assert (
        summary["entry_call_ownership"]["allowed_independent_entry_fusion_helper_count"]
        == 8
    )
    assert (
        summary["checks"][
            "exact_nine_independent_zero_custom_call_entry_fusion_helpers"
        ]
        is False
    )
    assert summary["passed"] is False


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
def test_t1024_call_counts_cannot_spoof_missing_accumulator_chain(dialect):
    text = _t1024_two_chunk_ir(dialect)
    text = (
        text.replace("%dk0#0, %dk0#1", "%a7, %dk0#1")
        if dialect == "stablehlo"
        else text.replace("%gte0, %gte1", "%p7, %gte1")
    )
    summary = _PROBE._strict_vjp_ir_summary(text, dialect, "all_valid_t1024")
    assert summary["custom_call_count"] == 6
    assert summary["two_chunk_accumulator_dataflow"]["passed"] is False
    assert summary["passed"] is False


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
def test_t1024_quoted_alias_spoof_and_missing_real_alias_fail_closed(dialect):
    text = _t1024_two_chunk_ir(dialect)
    if dialect == "stablehlo":
        text = text.replace(
            "output_operand_aliases = [#stablehlo.output_operand_alias<",
            'backend_config = "output_operand_aliases = [#stablehlo.output_operand_alias<',
        ).replace(
            "operand_index = 8, operand_tuple_indices = []>]}",
            'operand_index = 8, operand_tuple_indices = []>]"}',
        )
    else:
        text = text.replace(
            "output_to_operand_aliasing={{0}: (7, {}), {1}: (8, {})}",
            'backend_config="output_to_operand_aliasing={{0}: (7, {}), {1}: (8, {})}"',
        )
    summary = _PROBE._strict_vjp_ir_summary(text, dialect, "all_valid_t1024")
    assert summary["two_chunk_accumulator_dataflow"]["passed"] is False
    assert summary["passed"] is False


def test_t1024_optimized_single_identity_copy_path_passes_but_duplicate_fails():
    text = _t1024_two_chunk_ir("optimized_hlo")
    copied = text.replace(
        "  %gte1 = f32[1,1024,4,256] get-tuple-element(%dk0), index=1",
        "  %gte1 = f32[1,1024,4,256] get-tuple-element(%dk0), index=1\n"
        "  %copied = f32[1,1024,4,256] copy(%gte0)",
    ).replace("%gte0, %gte1", "%copied, %gte1")
    copied_summary = _PROBE._strict_vjp_ir_summary(
        copied, "optimized_hlo", "all_valid_t1024"
    )
    assert copied_summary["passed"] is True
    dk_path = copied_summary["two_chunk_accumulator_dataflow"]["sanitized_dataflow"][
        "dk_accumulator_path"
    ]
    assert dk_path["path_opcodes"] == ["get-tuple-element", "copy"]
    assert dk_path["copy_count"] == 1
    assert dk_path["passed"] is True
    duplicate = text.replace(
        "%gte1 = f32[1,1024,4,256] get-tuple-element",
        "%gte0 = f32[1,1024,4,256] get-tuple-element",
    )
    assert (
        _PROBE._strict_vjp_ir_summary(duplicate, "optimized_hlo", "all_valid_t1024")[
            "passed"
        ]
        is False
    )


@pytest.mark.parametrize(
    "mutation",
    ("crosswire", "multi_copy", "convert", "concatenate", "branch"),
)
def test_t1024_optimized_accumulator_path_adversaries_fail_closed(mutation):
    text = _t1024_two_chunk_ir("optimized_hlo")
    anchor = "  %gte1 = f32[1,1024,4,256] get-tuple-element(%dk0), index=1"
    if mutation == "crosswire":
        text = text.replace("%gte0, %gte1", "%gte1, %gte0")
    elif mutation == "multi_copy":
        text = text.replace(
            anchor,
            f"{anchor}\n"
            "  %copy0 = f32[1,1024,4,256] copy(%gte0)\n"
            "  %copy1 = f32[1,1024,4,256] copy(%copy0)",
        ).replace("%gte0, %gte1", "%copy1, %gte1")
    elif mutation in {"convert", "concatenate"}:
        operands = "%gte0" if mutation == "convert" else "%gte0, %gte0"
        text = text.replace(
            anchor,
            f"{anchor}\n  %moved = f32[1,1024,4,256] {mutation}({operands})",
        ).replace("%gte0, %gte1", "%moved, %gte1")
    else:
        text = text.replace(
            anchor,
            f"{anchor}\n  %branch = f32[1,1024,4,256] copy(%gte0)",
        )
    summary = _PROBE._strict_vjp_ir_summary(text, "optimized_hlo", "all_valid_t1024")
    assert summary["two_chunk_accumulator_dataflow"]["passed"] is False
    assert summary["passed"] is False


@pytest.mark.parametrize("opcode", ("concatenate", "convert", "copy"))
def test_t1024_unrelated_data_movement_is_diagnostic_only(opcode):
    text = _t1024_two_chunk_ir("optimized_hlo")
    operands = "%p0, %p0" if opcode == "concatenate" else "%p0"
    text = text.replace(
        "  ROOT %root = () tuple()",
        f"  %unrelated = f32[1,1024,4,256] {opcode}({operands})\n"
        "  ROOT %root = () tuple()",
    )
    summary = _PROBE._strict_vjp_ir_summary(text, "optimized_hlo", "all_valid_t1024")
    assert summary["passed"] is True
    diagnostic = summary["two_chunk_accumulator_dataflow"][
        "whole_program_data_movement_opcode_counts_diagnostic_only"
    ]
    assert diagnostic[opcode] == 1


@pytest.mark.parametrize(
    ("opcode", "instruction"),
    (
        (
            "convert",
            "    %unrelated = stablehlo.convert %acc_dk : "
            "(tensor<1x1024x4x256xf32>) -> tensor<1x1024x4x256xbf16>",
        ),
        (
            "concatenate",
            "    %unrelated = stablehlo.concatenate %acc_dk, %acc_dk, dim = 0 : "
            "(tensor<1x1024x4x256xf32>, tensor<1x1024x4x256xf32>) -> "
            "tensor<2x1024x4x256xf32>",
        ),
    ),
)
def test_t1024_stablehlo_unrelated_data_movement_is_diagnostic_only(
    opcode, instruction
):
    text = _t1024_two_chunk_ir("stablehlo").replace(
        "    return\n  }\n}", f"{instruction}\n    return\n  }}\n}}"
    )
    summary = _PROBE._strict_vjp_ir_summary(text, "stablehlo", "all_valid_t1024")
    assert summary["passed"] is True
    diagnostic = summary["two_chunk_accumulator_dataflow"][
        "whole_program_data_movement_opcode_counts_diagnostic_only"
    ]
    assert diagnostic[opcode] == 1


def test_t1024_accumulator_byte_pin_mutation_fails_closed(monkeypatch):
    monkeypatch.setattr(
        _PROBE,
        "_T1024_EXPECTED_ACCUMULATOR_PAIR_BYTES",
        8_388_607,
    )
    summary = _PROBE._strict_vjp_ir_summary(
        _t1024_two_chunk_ir("optimized_hlo"),
        "optimized_hlo",
        "all_valid_t1024",
    )
    assert summary["two_chunk_accumulator_dataflow"]["passed"] is False
    assert summary["passed"] is False


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
def test_t1024_wrong_q512_metadata_or_accumulator_dtype_fails_closed(dialect):
    text = _t1024_two_chunk_ir(dialect).replace("query_start=512", "query_start=513", 1)
    assert (
        _PROBE._strict_vjp_ir_summary(text, dialect, "all_valid_t1024")["passed"]
        is False
    )
    wrong_dtype = _t1024_two_chunk_ir(dialect)
    if dialect == "stablehlo":
        wrong_dtype = wrong_dtype.replace(
            "-> (tensor<1x1024x4x256xf32>, tensor<1x1024x4x256xf32>)",
            "-> (tensor<1x1024x4x256xbf16>, tensor<1x1024x4x256xf32>)",
            1,
        )
    else:
        wrong_dtype = wrong_dtype.replace(
            "%gte0 = f32[1,1024,4,256] get-tuple-element",
            "%gte0 = bf16[1,1024,4,256] get-tuple-element",
            1,
        )
    assert (
        _PROBE._strict_vjp_ir_summary(wrong_dtype, dialect, "all_valid_t1024")["passed"]
        is False
    )


def _helper_owned_calls(dialect: str, *, main_invocations: int) -> str:
    if dialect == "stablehlo":
        definitions = [
            f'#loc{index} = loc("{marker} query_start=0 query_size=512")'
            for index, marker in enumerate(_PROBE._EXPECTED_MARKERS.values())
        ]
        calls = [
            f'    %{index} = stablehlo.custom_call @"{_TARGET}"() : () -> tensor<1xbf16> loc(#loc{index})'
            for index in range(3)
        ]
        main_calls = [
            "    func.call @helper() : () -> ()" for _ in range(main_invocations)
        ]
        return "\n".join(
            [
                *definitions,
                "module {",
                "  func.func private @helper() {",
                *calls,
                "    return",
                "  }",
                "  func.func public @main() {",
                *main_calls,
                "    return",
                "  }",
                "}",
            ]
        )
    calls = [
        (
            f'  %{index} = bf16[1] custom-call(), custom_call_target="{_TARGET}", '
            f'op_name="{marker} query_start=0 query_size=512"'
        )
        for index, marker in enumerate(_PROBE._EXPECTED_MARKERS.values())
    ]
    main_calls = [
        f"  %call{index} = () call(), to_apply=%helper"
        for index in range(main_invocations)
    ]
    return "\n".join(
        [
            "HloModule helper_spoof",
            "helper {",
            *calls,
            "  ROOT %helper_root = () tuple()",
            "}",
            "ENTRY main {",
            *main_calls,
            "  ROOT %main_root = () tuple()",
            "}",
        ]
    )


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
def test_strict_ir_accepts_exact_three_calls_metadata_and_target(dialect):
    text = _stablehlo_calls() if dialect == "stablehlo" else _optimized_hlo_calls()
    summary = _PROBE._strict_vjp_ir_summary(text, dialect)
    assert summary["passed"] is True
    assert summary["custom_call_count"] == 3
    assert summary["marker_call_counts"] == {"forward": 1, "dq": 1, "dkdv": 1}
    assert summary["entry_call_ownership"]["direct_entry_custom_call_count"] == 3
    assert all(call["passed"] for call in summary["calls"])


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
def test_raw_quote_first_lexing_ignores_escaped_quotes_braces_and_call_lookalikes(
    dialect,
):
    payload = (
        r"prefix \22 { custom-call( stablehlo.custom_call func.call ENTRY } suffix"
    )
    if dialect == "stablehlo":
        text = _stablehlo_calls().replace(
            "() : () -> tensor<1xbf16>",
            f'() {{backend_config = "{payload}"}} : () -> tensor<1xbf16>',
        )
    else:
        text = f'HloModule quoted_spoof, config="{payload}"\n' + _optimized_hlo_calls()
    summary = _PROBE._strict_vjp_ir_summary(text, dialect)
    assert summary["passed"] is True
    lexing = summary["quote_lexing_diagnostic"]
    assert lexing["syntax_was_lexed_before_local_escape_decoding"] is True
    assert lexing["raw_unquoted_custom_call_count"] == 3
    assert lexing["decoded_then_masked_custom_call_count_diagnostic_only"] != 3
    assert lexing["raw_mlir_hex_quote_escape_count"] > 0
    assert lexing["raw_quote_scan_passed"] is True
    serialized = json.dumps(summary)
    assert payload not in serialized
    assert "quoted_spoof" not in serialized


@pytest.mark.parametrize(
    "corrupt",
    (
        'HloModule broken, config="unterminated\n',
        'HloModule broken, config="bad \\q escape"\n',
    ),
)
def test_optimized_quote_scanner_fails_closed_on_unterminated_or_unknown_escape(
    corrupt,
):
    summary = _PROBE._strict_vjp_ir_summary(
        corrupt + _optimized_hlo_calls(), "optimized_hlo"
    )
    assert summary["passed"] is False
    ownership = summary["entry_call_ownership"]
    assert ownership["raw_quote_scan"]["passed"] is False
    assert ownership["checks"]["raw_quote_escapes_are_well_formed"] is False


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
@pytest.mark.parametrize("main_invocations", (0, 2))
def test_strict_ir_rejects_unreachable_or_repeated_helper_owned_calls(
    dialect, main_invocations
):
    summary = _PROBE._strict_vjp_ir_summary(
        _helper_owned_calls(dialect, main_invocations=main_invocations), dialect
    )
    assert summary["custom_call_count"] == 3
    assert summary["entry_call_ownership"]["direct_entry_custom_call_count"] == 0
    helper = next(
        item
        for item in summary["entry_call_ownership"]["computations"]
        if item["is_entry"] is False
    )
    assert helper["saturated_entry_multiplicity"] == (
        0 if main_invocations == 0 else ">1"
    )
    assert summary["passed"] is False


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
def test_helper_called_once_is_reported_once_but_runtime_direct_entry_gate_rejects(
    dialect,
):
    summary = _PROBE._strict_vjp_ir_summary(
        _helper_owned_calls(dialect, main_invocations=1), dialect
    )
    ownership = summary["entry_call_ownership"]
    helper = next(
        item for item in ownership["computations"] if item["is_entry"] is False
    )
    assert helper["reachable_from_entry"] is True
    assert helper["saturated_entry_multiplicity"] == 1
    assert ownership["direct_entry_custom_call_count"] == 0
    assert summary["passed"] is False
    serialized = json.dumps(summary)
    assert '"helper"' not in serialized
    assert _TARGET not in serialized


def _cyclic_helper_owned_calls(dialect: str) -> str:
    text = _helper_owned_calls(dialect, main_invocations=1)
    if dialect == "stablehlo":
        return text.replace(
            "  func.func private @helper() {",
            "  func.func private @helper() {\n    func.call @helper() : () -> ()",
            1,
        )
    return text.replace(
        "helper {",
        "helper {\n  %recursive = () call(), to_apply=%helper",
        1,
    )


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
def test_reachable_recursive_helper_is_cycle_unknown_and_rejected(dialect):
    summary = _PROBE._strict_vjp_ir_summary(
        _cyclic_helper_owned_calls(dialect), dialect
    )
    ownership = summary["entry_call_ownership"]
    assert ownership["entry_multiplicity"]["cycle_detected"] is True
    helper = next(
        item for item in ownership["computations"] if item["is_entry"] is False
    )
    assert helper["saturated_entry_multiplicity"] == "unknown"
    assert summary["passed"] is False


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
def test_direct_custom_call_with_called_computation_is_unknown_and_rejected(dialect):
    if dialect == "stablehlo":
        text = (
            _stablehlo_calls()
            .replace(
                "module {",
                "module {\n  func.func private @helper() { return }",
                1,
            )
            .replace(
                f'@"{_TARGET}"()',
                f'@"{_TARGET}"() {{called_computations = [@helper]}}',
                1,
            )
        )
    else:
        lines = _optimized_hlo_calls().splitlines()
        lines[1] += ", called_computations={%helper}"
        text = "\n".join(
            [
                "HloModule called_custom",
                "helper {",
                "  ROOT %helper_root = () tuple()",
                "}",
                *lines,
            ]
        )
    summary = _PROBE._strict_vjp_ir_summary(text, dialect)
    ownership = summary["entry_call_ownership"]
    assert (
        ownership["calls"][0]["has_nonempty_or_malformed_called_computations"] is True
    )
    assert ownership["forbidden_container_count"] > 0
    helper = next(
        item for item in ownership["computations"] if item["is_entry"] is False
    )
    assert helper["saturated_entry_multiplicity"] == "unknown"
    assert summary["passed"] is False


def _optimized_with_independent_entry_fusions(count: int) -> str:
    helpers: list[str] = ["HloModule independent_fusions"]
    for index in range(count):
        helpers.extend(
            [
                f"helper_{index} {{",
                f"  ROOT %helper_root_{index} = () tuple()",
                "}",
            ]
        )
    entry = _optimized_hlo_calls().splitlines()
    entry[-1:-1] = [
        f"  %fusion_{index} = () fusion(), calls=%helper_{index}"
        for index in range(count)
    ]
    return "\n".join([*helpers, *entry])


@pytest.mark.parametrize("fusion_count", (1, 3))
def test_independent_entry_fusions_with_zero_custom_call_helpers_are_allowed(
    fusion_count,
):
    summary = _PROBE._strict_vjp_ir_summary(
        _optimized_with_independent_entry_fusions(fusion_count), "optimized_hlo"
    )
    ownership = summary["entry_call_ownership"]
    assert summary["passed"] is True
    assert ownership["direct_entry_custom_call_count"] == 3
    assert all(
        item["custom_call_count"] == 0
        for item in ownership["computations"]
        if item["is_entry"] is False
    )
    assert ownership["allowed_independent_entry_fusion_helper_count"] == fusion_count
    assert (
        ownership["checks"]["no_container_can_own_or_duplicate_a_custom_call"] is True
    )


@pytest.mark.parametrize(
    "nested_reference",
    (
        "backend_config={calls=%helper_0}",
        "metadata={called_computations={%helper_0}}",
        "backend_config={condition=%helper_0}",
    ),
)
def test_nested_metadata_cannot_impersonate_required_top_level_fusion_edge(
    nested_reference,
):
    text = _optimized_with_independent_entry_fusions(1).replace(
        "calls=%helper_0", nested_reference, 1
    )
    summary = _PROBE._strict_vjp_ir_summary(text, "optimized_hlo")
    ownership = summary["entry_call_ownership"]
    assert ownership["allowed_independent_entry_fusion_helper_count"] == 0
    assert ownership["call_edges"] == []
    assert (
        ownership["checks"]["only_independent_direct_entry_fusion_helpers_are_present"]
        is False
    )
    assert summary["passed"] is False


@pytest.mark.parametrize(
    "reference_schema",
    (
        "condition=%helper_0",
        "to_apply=%helper_0",
        "called_computations=%helper_0",
        "branch_computations={%helper_0}",
        "calls=%helper_0, condition=[]",
    ),
)
def test_fusion_requires_exactly_one_singular_top_level_calls_attribute(
    reference_schema,
):
    text = _optimized_with_independent_entry_fusions(1).replace(
        "calls=%helper_0", reference_schema, 1
    )
    summary = _PROBE._strict_vjp_ir_summary(text, "optimized_hlo")
    ownership = summary["entry_call_ownership"]
    assert ownership["allowed_independent_entry_fusion_helper_count"] == 0
    assert (
        ownership["checks"]["only_independent_direct_entry_fusion_helpers_are_present"]
        is False
    )
    assert summary["passed"] is False


def test_fusion_helper_that_owns_custom_calls_remains_rejected():
    text = _helper_owned_calls("optimized_hlo", main_invocations=0).replace(
        "ENTRY main {",
        "ENTRY main {\n  %fusion = () fusion(), calls=%helper",
        1,
    )
    summary = _PROBE._strict_vjp_ir_summary(text, "optimized_hlo")
    ownership = summary["entry_call_ownership"]
    assert ownership["direct_entry_custom_call_count"] == 0
    assert (
        ownership["checks"]["all_nonentry_computations_have_zero_custom_calls"] is False
    )
    assert summary["passed"] is False


def test_cycle_inside_otherwise_zero_custom_call_fusion_helper_remains_rejected():
    text = _optimized_with_independent_entry_fusions(1).replace(
        "helper_0 {",
        "helper_0 {\n  %cycle = () fusion(), calls=%helper_0",
        1,
    )
    summary = _PROBE._strict_vjp_ir_summary(text, "optimized_hlo")
    ownership = summary["entry_call_ownership"]
    assert ownership["entry_multiplicity"]["cycle_detected"] is True
    assert (
        ownership["checks"]["no_container_can_own_or_duplicate_a_custom_call"] is False
    )
    assert summary["passed"] is False


@pytest.mark.parametrize(
    "replacement",
    (
        "%wrapped = () call(), to_apply=%helper_0",
        "%wrapped = () conditional(), branch_computations={%helper_0}",
        "%wrapped = () while(), condition=%helper_0, body=%helper_0",
        "%wrapped = () async-start(), calls=%helper_0",
        "%wrapped = () mystery(), to_apply=%helper_0",
        "%wrapped = () mystery(), calls=%helper_0",
        "%wrapped = () mystery(), called_computations={%helper_0}",
        "%wrapped = () mystery(), condition=%helper_0",
    ),
)
def test_zero_custom_call_helpers_under_nonfusion_wrappers_remain_rejected(
    replacement,
):
    text = _optimized_with_independent_entry_fusions(1).replace(
        "%fusion_0 = () fusion(), calls=%helper_0", replacement, 1
    )
    summary = _PROBE._strict_vjp_ir_summary(text, "optimized_hlo")
    ownership = summary["entry_call_ownership"]
    assert ownership["direct_entry_custom_call_count"] == 3
    assert all(
        item["custom_call_count"] == 0
        for item in ownership["computations"]
        if item["is_entry"] is False
    )
    assert (
        ownership["checks"]["only_independent_direct_entry_fusion_helpers_are_present"]
        is False
    )
    assert summary["passed"] is False


@pytest.mark.parametrize(
    "reference",
    (
        "calls=%helper_0",
        "called_computations={%helper_0}",
        "condition=%helper_0",
    ),
)
def test_unknown_helper_self_references_are_cycle_unknown_and_rejected(reference):
    text = _optimized_with_independent_entry_fusions(1).replace(
        "helper_0 {",
        f"helper_0 {{\n  %hidden = () mystery(), {reference}",
        1,
    )
    summary = _PROBE._strict_vjp_ir_summary(text, "optimized_hlo")
    ownership = summary["entry_call_ownership"]
    assert ownership["entry_multiplicity"]["cycle_detected"] is True
    assert ownership["unknown_callable_count"] > 0
    assert (
        ownership["checks"]["no_container_can_own_or_duplicate_a_custom_call"] is False
    )
    assert summary["passed"] is False


@pytest.mark.parametrize(
    "instruction",
    (
        "%wrapped = () conditional(), branch_computations={%helper}",
        "%wrapped = () while(), condition=%helper, body=%helper",
        "%wrapped = () async-start(), calls={%helper}",
        "%wrapped = () async-start(), calls=%helper",
        "%wrapped = () mystery(), to_apply=%helper",
    ),
)
def test_optimized_conditional_loop_async_and_unknown_wrappers_are_unknown_rejected(
    instruction,
):
    text = _helper_owned_calls("optimized_hlo", main_invocations=0).replace(
        "ENTRY main {", f"ENTRY main {{\n  {instruction}", 1
    )
    summary = _PROBE._strict_vjp_ir_summary(text, "optimized_hlo")
    ownership = summary["entry_call_ownership"]
    assert ownership["forbidden_container_count"] > 0
    helper = next(
        item for item in ownership["computations"] if item["is_entry"] is False
    )
    assert helper["saturated_entry_multiplicity"] == "unknown"
    assert summary["passed"] is False


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
@pytest.mark.parametrize("suffix", ("x", "garbage", "e2", ".0"))
def test_strict_ir_rejects_metadata_numeric_suffix_spoofs(dialect, suffix):
    metadata = dict.fromkeys(
        _PROBE._EXPECTED_MARKERS,
        f"query_start=0{suffix} query_size=512{suffix}",
    )
    text = (
        _stablehlo_calls(metadata=metadata)
        if dialect == "stablehlo"
        else _optimized_hlo_calls(metadata=metadata)
    )
    assert _PROBE._strict_vjp_ir_summary(text, dialect)["passed"] is False


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
@pytest.mark.parametrize(
    "failure", ("target", "marker", "extra", "while", "duplicate_metadata")
)
def test_strict_ir_rejects_target_marker_count_while_and_duplicate_metadata(
    dialect, failure
):
    target = "triton" if failure == "target" else _TARGET
    markers = dict(_PROBE._EXPECTED_MARKERS)
    metadata = dict.fromkeys(_PROBE._EXPECTED_MARKERS, "query_start=0 query_size=512")
    extra = ""
    if failure == "marker":
        markers["dq"] = "query_bounded_gqa_dq_q0_spoof"
    elif failure == "extra":
        extra = (
            f'  %9 = stablehlo.custom_call @"{_TARGET}"() : () -> tensor<1xbf16>'
            if dialect == "stablehlo"
            else f'  %9 = bf16[1] custom-call(), custom_call_target="{_TARGET}"'
        )
    elif failure == "while":
        extra = (
            "%loop = stablehlo.while() : () -> ()"
            if dialect == "stablehlo"
            else "%loop = bf16[1] while(%x)"
        )
    elif failure == "duplicate_metadata":
        metadata["forward"] += " query_start=0"
    text = (
        _stablehlo_calls(target=target, metadata=metadata, markers=markers, extra=extra)
        if dialect == "stablehlo"
        else _optimized_hlo_calls(
            target=target, metadata=metadata, markers=markers, extra=extra
        )
    )
    assert _PROBE._strict_vjp_ir_summary(text, dialect)["passed"] is False


def test_optimized_target_nested_only_in_backend_config_cannot_spoof_top_level_target():
    forged = _optimized_hlo_calls().replace(
        f'custom_call_target="{_TARGET}"',
        f'backend_config={{custom_call_target="{_TARGET}"}}',
    )
    summary = _PROBE._strict_vjp_ir_summary(forged, "optimized_hlo")
    assert summary["custom_call_count"] == 3
    assert all(call["target_count"] == 0 for call in summary["calls"])
    assert summary["passed"] is False


def test_structural_gate_requires_both_dialects():
    stable = _PROBE._strict_vjp_ir_summary(_stablehlo_calls(), "stablehlo")
    optimized = _PROBE._strict_vjp_ir_summary(_optimized_hlo_calls(), "optimized_hlo")
    assert _PROBE._structural_gate(stable, optimized)["passed"] is True
    assert _PROBE._structural_gate(stable)["passed"] is False


def test_memory_gate_requires_exact_promoted_signature_and_zero_alias():
    valid = {
        "available": True,
        "argument_size_in_bytes": 10_487_808,
        "output_size_in_bytes": 10_485_792,
        "alias_size_in_bytes": 0,
        "temp_size_in_bytes": 64 * 1024**2,
    }
    assert _PROBE._compiled_memory_gate(valid)["passed"] is True
    for name, value in (
        ("argument_size_in_bytes", 10_487_807),
        ("output_size_in_bytes", 10_485_791),
        ("alias_size_in_bytes", 1),
        ("temp_size_in_bytes", 64 * 1024**2 + 1),
    ):
        corrupt = dict(valid)
        corrupt[name] = value
        assert _PROBE._compiled_memory_gate(corrupt)["passed"] is False


class _FakeShape:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _CompiledFake:
    def __init__(self, state):
        self.state = state

    def as_text(self):
        return _optimized_hlo_calls()

    def memory_analysis(self):
        return SimpleNamespace(
            argument_size_in_bytes=10_487_808,
            output_size_in_bytes=10_485_792,
            alias_size_in_bytes=0,
            temp_size_in_bytes=2_130_688,
        )

    def __call__(self, *arguments):
        self.state["compiled_invocations"] += 1
        return (arguments[0], arguments[0], arguments[1], arguments[2])


class _LoweredFake:
    def __init__(self, state, compiled):
        self.state = state
        self.compiled = compiled

    def compiler_ir(self, *, dialect):
        assert dialect == "stablehlo"
        return _stablehlo_calls()

    def compile(self):
        self.state["compile_calls"] += 1
        return self.compiled


class _JitFake:
    def __init__(self, function, state, lowered):
        self.function = function
        self.state = state
        self.lowered = lowered

    def lower(self, *signature):
        self.state["lower_calls"] += 1
        self.state["signature"] = signature
        self.function(*signature)
        return self.lowered


class _JaxFake:
    ShapeDtypeStruct = _FakeShape

    def __init__(self, state, lowered):
        self.state = state
        self.lowered = lowered

    def jit(self, function):
        return _JitFake(function, self.state, self.lowered)

    def vjp(self, function, q, k, v):
        value = function(q, k, v)
        return value, lambda dout: (dout, k, v)


def test_compile_mock_uses_explicit_scale_three_calls_and_releases_without_invocation():
    state = {
        "lower_calls": 0,
        "compile_calls": 0,
        "compiled_invocations": 0,
        "api": [],
    }
    compiled = _CompiledFake(state)
    lowered = _LoweredFake(state, compiled)

    def api(*arguments, **keywords):
        state["api"].append((arguments, keywords))
        return arguments[0]

    counters = _PROBE._zero_counters()
    compiled_artifact, report = _PROBE._compile_vjp_artifact(
        _JaxFake(state, lowered),
        SimpleNamespace(bfloat16="bf16", int32="i32"),
        api,
        lambda: dict(_CLEAN),
        counters,
        io.StringIO(),
    )
    checked = _PROBE._release_checked_vjp(compiled_artifact, report, counters)
    assert checked.proof["passed"] is True
    assert state["lower_calls"] == state["compile_calls"] == 1
    assert state["compiled_invocations"] == 0
    assert [item.shape for item in state["signature"]] == [
        (1, 512, 16, 256),
        (1, 512, 4, 256),
        (1, 512, 4, 256),
        (1, 512),
        (1, 512, 16, 256),
    ]
    assert state["api"][0][1] == {
        "scale": 0.09375,
        "query_chunk_size": 512,
        "block_q": 64,
        "block_k": 64,
        "backward_block_q": 32,
        "backward_block_k": 32,
        "interpret": False,
    }
    assert report["release_gate"]["exact_logical_dispatches"] == {
        "forward": 1,
        "dq": 1,
        "dkdv": 1,
    }


def test_checked_capability_is_private_single_use_and_counts_one_executable():
    state = {"compiled_invocations": 0}
    counters = _PROBE._zero_counters()
    compiled = _CompiledFake(state)
    with pytest.raises(RuntimeError, match="without passed gates"):
        _PROBE._CheckedVjpExecutable(
            compiled, proof={"passed": True}, counters=counters, token=object()
        )
    checked = _PROBE._wrap_checked(compiled, {"passed": True}, counters)
    fake_jax = SimpleNamespace(block_until_ready=lambda value: value)
    arguments = ("q", "k", "v", "mask", "dout")
    result = checked.invoke(fake_jax, arguments, lambda: None)
    assert result == ("q", "q", "k", "v")
    assert state["compiled_invocations"] == 1
    assert counters["checked_executable_invocations"] == 1
    with pytest.raises(RuntimeError, match="already consumed"):
        checked.invoke(fake_jax, arguments, lambda: None)


def test_bf16_rounded_reference_passes_and_corrupted_gradient_fails(exact_case):
    _inputs, _manifests, expected, _reference = exact_case
    actual = tuple(item.astype(ml_dtypes.bfloat16) for item in expected)
    record = _PROBE._validate_candidate(
        np,
        actual,
        expected,
        0.01,
        _PROBE._completed_counters(),
        io.StringIO(),
    )
    assert record["gates"]["promotion_passed"] is True
    assert all(record["gates"]["per_tensor_numerical_passed"].values())
    corrupted = list(actual)
    corrupted[2] = np.full_like(corrupted[2], ml_dtypes.bfloat16(2.0))
    with pytest.raises(RuntimeError, match="numerical"):
        _PROBE._validate_candidate(
            np,
            tuple(corrupted),
            expected,
            0.01,
            _PROBE._completed_counters(),
            io.StringIO(),
        )


def _synthetic_numerical_metrics(relative_l2: float) -> dict[str, object]:
    return {
        "finite": True,
        "shape_dtype_nbytes_exact": True,
        "reference_l2_norm": 1.0,
        "relative_l2": relative_l2,
        "cosine_raw": 1.0,
        "cosine": 1.0,
        "max_abs": 0.0,
    }


def _patch_tensor_metrics(monkeypatch, relative_l2_values: list[float]) -> None:
    metrics = iter(_synthetic_numerical_metrics(value) for value in relative_l2_values)
    monkeypatch.setattr(
        _PROBE,
        "_tensor_metrics",
        lambda *_args, **_kwargs: next(metrics),
    )


def test_gradient_errors_between_one_and_three_percent_pass_full_vjp_gate(
    monkeypatch,
):
    _patch_tensor_metrics(monkeypatch, [0.005, 0.02, 0.02, 0.02])

    record = _PROBE._validate_candidate(
        np,
        (object(), object(), object(), object()),
        (object(), object(), object(), object()),
        0.01,
        _PROBE._completed_counters(),
        io.StringIO(),
    )

    assert record["gates"]["per_tensor_numerical_passed"] == {
        "output": True,
        "dq": True,
        "dk": True,
        "dv": True,
    }
    assert record["thresholds_per_tensor"]["relative_l2_strictly_below"] == {
        "output": 0.01,
        "dq": 0.03,
        "dk": 0.03,
        "dv": 0.03,
    }


def test_relative_l2_policy_lookup_is_exact_and_fails_closed() -> None:
    assert {
        name: _PROBE._relative_l2_limit(name) for name in ("output", "dq", "dk", "dv")
    } == {"output": 0.01, "dq": 0.03, "dk": 0.03, "dv": 0.03}
    with pytest.raises(RuntimeError, match="no exact relative-L2 policy"):
        _PROBE._relative_l2_limit("gradient")


def test_two_percent_output_error_still_fails_while_gradients_pass(monkeypatch):
    _patch_tensor_metrics(monkeypatch, [0.02, 0.005, 0.005, 0.005])
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="numerical"):
        _PROBE._validate_candidate(
            np,
            (object(), object(), object(), object()),
            (object(), object(), object(), object()),
            0.01,
            _PROBE._completed_counters(),
            output,
        )

    [record] = _records(output)
    assert record["gates"]["per_tensor_numerical_passed"] == {
        "output": False,
        "dq": True,
        "dk": True,
        "dv": True,
    }


def test_exact_three_percent_gradient_error_fails_strict_boundary(monkeypatch):
    _patch_tensor_metrics(monkeypatch, [0.005, 0.03, 0.03, 0.03])
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="numerical"):
        _PROBE._validate_candidate(
            np,
            (object(), object(), object(), object()),
            (object(), object(), object(), object()),
            0.01,
            _PROBE._completed_counters(),
            output,
        )

    [record] = _records(output)
    assert record["gates"]["per_tensor_numerical_passed"] == {
        "output": True,
        "dq": False,
        "dk": False,
        "dv": False,
    }


def _two_percent_bf16_perturbation(value):
    fp32 = np.asarray(value, np.float32)
    perturbation = np.clip(np.float32(0.02) * fp32, -0.014, 0.014)
    return (fp32 + perturbation).astype(ml_dtypes.bfloat16)


def test_real_array_two_percent_gradients_pass_but_output_fails(exact_case):
    _inputs, _manifests, expected, _reference = exact_case
    rounded = [item.astype(ml_dtypes.bfloat16) for item in expected]
    gradient_candidate = (
        rounded[0],
        *(_two_percent_bf16_perturbation(item) for item in expected[1:]),
    )

    passed = _PROBE._validate_candidate(
        np,
        gradient_candidate,
        expected,
        0.01,
        _PROBE._completed_counters(),
        io.StringIO(),
    )

    for name in ("dq", "dk", "dv"):
        metrics = passed["metrics"][name]
        assert 0.01 < metrics["relative_l2"] < 0.03
        assert metrics["cosine"] >= 0.9999
        assert metrics["max_abs"] <= 0.02
        assert passed["gates"]["per_tensor_numerical_passed"][name] is True

    output_candidate = (
        _two_percent_bf16_perturbation(expected[0]),
        rounded[1],
        rounded[2],
        rounded[3],
    )
    output = io.StringIO()
    with pytest.raises(RuntimeError, match="numerical"):
        _PROBE._validate_candidate(
            np,
            output_candidate,
            expected,
            0.01,
            _PROBE._completed_counters(),
            output,
        )
    [failed] = _records(output)
    output_metrics = failed["metrics"]["output"]
    assert 0.01 < output_metrics["relative_l2"] < 0.03
    assert output_metrics["cosine"] >= 0.9999
    assert output_metrics["max_abs"] <= 0.02
    assert failed["gates"]["per_tensor_numerical_passed"]["output"] is False
    assert all(
        failed["gates"]["per_tensor_numerical_passed"][name]
        for name in ("dq", "dk", "dv")
    )


def test_valid385_active_gradient_slices_use_three_percent_gate(
    monkeypatch, valid385_case
):
    _inputs, _manifests, expected, _reference = valid385_case
    actual = tuple(item.astype(ml_dtypes.bfloat16) for item in expected)
    _patch_tensor_metrics(
        monkeypatch,
        [
            0.005,
            0.02,
            0.02,
            0.02,
            0.005,
            0.005,
            0.005,
            0.005,
            0.02,
            0.02,
            0.02,
        ],
    )
    observed_limits = []
    original_numerical_gate = _PROBE._numerical_metrics_pass

    def record_limit(item, *, maximum_relative_l2):
        observed_limits.append(maximum_relative_l2)
        return original_numerical_gate(item, maximum_relative_l2=maximum_relative_l2)

    monkeypatch.setattr(_PROBE, "_numerical_metrics_pass", record_limit)

    record = _PROBE._validate_candidate(
        np,
        actual,
        expected,
        0.01,
        _PROBE._completed_counters(),
        io.StringIO(),
        "valid385",
    )

    assert record["padded_validation"]["affected_output_rows_numerical_passed"]
    assert all(
        record["padded_validation"]["boundary_output_rows_numerical_passed"].values()
    )
    assert record["padded_validation"]["active_gradient_numerical_passed"] == {
        "dq": True,
        "dk": True,
        "dv": True,
    }
    assert record["padded_validation"]["all_zero_tail_gates_passed"] is True
    assert observed_limits == [
        0.01,
        0.03,
        0.03,
        0.03,
        0.01,
        0.01,
        0.01,
        0.01,
        0.03,
        0.03,
        0.03,
    ]


def test_t1024_half_gates_route_output_and_gradient_limits(
    monkeypatch, all_valid_t1024_case
):
    _inputs, _manifests, expected, _reference = all_valid_t1024_case
    actual = tuple(item.astype(ml_dtypes.bfloat16) for item in expected)
    _patch_tensor_metrics(
        monkeypatch,
        [
            0.005,
            0.005,
            0.005,
            0.005,
            0.02,
            0.005,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
        ],
    )
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="numerical"):
        _PROBE._validate_candidate(
            np,
            actual,
            expected,
            0.01,
            _PROBE._completed_counters(),
            output,
            "all_valid_t1024",
        )

    [record] = _records(output)
    halves = record["t1024_half_validation"]["per_half_numerical_passed"]
    assert halves["output_query_rows_0_512"] is False
    assert halves["output_query_rows_512_1024"] is True
    assert all(
        passed for name, passed in halves.items() if not name.startswith("output_")
    )


def test_t1024_bf16_reference_passes_full_and_every_half_gate(
    all_valid_t1024_case,
):
    _inputs, _manifests, expected, _reference = all_valid_t1024_case
    actual = tuple(item.astype(ml_dtypes.bfloat16) for item in expected)
    record = _PROBE._validate_candidate(
        np,
        actual,
        expected,
        0.01,
        _PROBE._completed_counters(),
        io.StringIO(),
        "all_valid_t1024",
    )
    assert record["gates"]["all_tensor_numerical_passed"] is True
    assert record["gates"]["case_specific_t1024_half_validation_passed"] is True
    assert all(record["t1024_half_validation"]["per_half_numerical_passed"].values())
    assert record["gates"]["promotion_passed"] is True


def test_t1024_corrupted_query_half_fails_only_its_independent_half_gate(
    all_valid_t1024_case,
):
    _inputs, _manifests, expected, _reference = all_valid_t1024_case
    actual = [item.astype(ml_dtypes.bfloat16) for item in expected]
    actual[0][:, 512:] = ml_dtypes.bfloat16(2.0)
    output = io.StringIO()
    with pytest.raises(RuntimeError, match="numerical"):
        _PROBE._validate_candidate(
            np,
            tuple(actual),
            expected,
            0.01,
            _PROBE._completed_counters(),
            output,
            "all_valid_t1024",
        )
    [record] = _records(output)
    halves = record["t1024_half_validation"]["per_half_numerical_passed"]
    assert halves["output_query_rows_512_1024"] is False
    assert all(
        passed
        for name, passed in halves.items()
        if name != "output_query_rows_512_1024"
    )
    assert record["gates"]["case_specific_t1024_half_validation_passed"] is False


def test_valid385_bf16_reference_passes_all_numerical_and_zero_tail_gates(
    valid385_case,
):
    _inputs, _manifests, expected, _reference = valid385_case
    actual = tuple(item.astype(ml_dtypes.bfloat16) for item in expected)
    record = _PROBE._validate_candidate(
        np,
        actual,
        expected,
        0.01,
        _PROBE._completed_counters(),
        io.StringIO(),
        "valid385",
    )
    assert record["gates"]["promotion_passed"] is True
    assert record["gates"]["case_specific_padded_validation_passed"] is True
    assert record["padded_validation"]["affected_output_rows_numerical_passed"] is True
    assert record["padded_validation"]["all_zero_tail_gates_passed"] is True
    assert record["padded_validation"]["boundary_output_rows"] == [384, 385, 511]
    assert all(
        record["padded_validation"]["boundary_output_rows_numerical_passed"].values()
    )
    assert all(record["padded_validation"]["active_gradient_numerical_passed"].values())
    for item in record["padded_validation"]["exact_numeric_zero_tails"].values():
        assert item["actual"]["numeric_exact_zero"] is True
        assert item["reference"]["numeric_exact_zero"] is True
        assert item["reference"]["bitwise_positive_zero_diagnostic_only"] is True


def test_valid385_negative_zero_gradient_tail_is_accepted_as_numeric_zero(
    valid385_case,
):
    _inputs, _manifests, expected, _reference = valid385_case
    actual = [item.astype(ml_dtypes.bfloat16) for item in expected]
    actual[1][0, 385, 0, 0] = ml_dtypes.bfloat16(-0.0)
    record = _PROBE._validate_candidate(
        np,
        tuple(actual),
        expected,
        0.01,
        _PROBE._completed_counters(),
        io.StringIO(),
        "valid385",
    )
    dq = record["padded_validation"]["exact_numeric_zero_tails"]["dq"]
    assert dq["actual"]["numeric_exact_zero"] is True
    assert dq["actual"]["bitwise_positive_zero_diagnostic_only"] is False
    assert record["gates"]["promotion_passed"] is True


def test_valid385_nonzero_gradient_tail_fails_exact_numeric_zero_gate(valid385_case):
    _inputs, _manifests, expected, _reference = valid385_case
    actual = [item.astype(ml_dtypes.bfloat16) for item in expected]
    actual[1][0, 385, 0, 0] = ml_dtypes.bfloat16(2**-8)
    output = io.StringIO()
    with pytest.raises(RuntimeError, match="numerical"):
        _PROBE._validate_candidate(
            np,
            tuple(actual),
            expected,
            0.01,
            _PROBE._completed_counters(),
            output,
            "valid385",
        )
    [record] = _records(output)
    dq = record["padded_validation"]["exact_numeric_zero_tails"]["dq"]
    assert dq["actual"]["numeric_exact_zero"] is False
    assert dq["actual"]["count_nonzero"] == 1
    assert record["gates"]["case_specific_padded_validation_passed"] is False


def test_valid385_ignored_key_mask_fails_affected_forward_rows(valid385_case):
    inputs, _manifests, expected, _reference = valid385_case
    q, k, v, _mask, dout = inputs
    wrong, _ = _PROBE._tiled_causal_gqa_forward_vjp_oracle(
        np,
        q,
        k,
        v,
        np.ones((1, 512), np.int32),
        dout,
        scale=3 / 32,
    )
    actual = [item.astype(ml_dtypes.bfloat16) for item in expected]
    actual[0] = wrong[0].astype(ml_dtypes.bfloat16)
    output = io.StringIO()
    with pytest.raises(RuntimeError, match="numerical"):
        _PROBE._validate_candidate(
            np,
            tuple(actual),
            expected,
            0.01,
            _PROBE._completed_counters(),
            output,
            "valid385",
        )
    [record] = _records(output)
    assert record["padded_validation"]["affected_output_rows_numerical_passed"] is False
    assert record["padded_validation"]["all_zero_tail_gates_passed"] is True


def test_valid385_each_boundary_output_row_is_independently_gated(valid385_case):
    inputs, _manifests, expected, _reference = valid385_case
    q, k, v, _mask, dout = inputs
    alternatives = {}
    for valid_tokens, row in ((384, 384), (386, 385), (512, 511)):
        mask = np.zeros((1, 512), np.int32)
        mask[:, :valid_tokens] = 1
        alternative, _ = _PROBE._tiled_causal_gqa_forward_vjp_oracle(
            np,
            q,
            k,
            v,
            mask,
            dout,
            scale=3 / 32,
            _require_loss_masked_padding=valid_tokens != 384,
        )
        alternatives[row] = alternative[0]
    for row, wrong_output in alternatives.items():
        actual = [item.astype(ml_dtypes.bfloat16) for item in expected]
        actual[0][:, row] = wrong_output.astype(ml_dtypes.bfloat16)[:, row]
        output = io.StringIO()
        with pytest.raises(RuntimeError, match="numerical"):
            _PROBE._validate_candidate(
                np,
                tuple(actual),
                expected,
                0.01,
                _PROBE._completed_counters(),
                output,
                "valid385",
            )
        [record] = _records(output)
        boundary = record["padded_validation"]["boundary_output_rows_numerical_passed"]
        assert boundary[str(row)] is False
        assert all(value for name, value in boundary.items() if name != str(row))
        assert record["padded_validation"]["all_zero_tail_gates_passed"] is True


def test_valid385_ignored_loss_mask_fails_gradient_and_dq_tail_gates(
    valid385_case, exact_case
):
    inputs, _manifests, expected, _reference = valid385_case
    q, k, v, mask, _masked_dout = inputs
    unmasked_dout = exact_case[0][4]
    wrong, _ = _PROBE._tiled_causal_gqa_forward_vjp_oracle(
        np,
        q,
        k,
        v,
        mask,
        unmasked_dout,
        scale=3 / 32,
        _require_loss_masked_padding=False,
    )
    actual = tuple(item.astype(ml_dtypes.bfloat16) for item in wrong)
    output = io.StringIO()
    with pytest.raises(RuntimeError, match="numerical"):
        _PROBE._validate_candidate(
            np,
            actual,
            expected,
            0.01,
            _PROBE._completed_counters(),
            output,
            "valid385",
        )
    [record] = _records(output)
    assert record["gates"]["all_tensor_numerical_passed"] is False
    tails = record["padded_validation"]["exact_numeric_zero_tails"]
    assert tails["dq"]["actual"]["numeric_exact_zero"] is False
    assert tails["dk"]["actual"]["numeric_exact_zero"] is True
    assert tails["dv"]["actual"]["numeric_exact_zero"] is True


def test_nonfinite_candidate_emits_json_safe_failure_before_failing(exact_case):
    _inputs, _manifests, expected, _reference = exact_case
    actual = [item.astype(ml_dtypes.bfloat16) for item in expected]
    actual[1].reshape(-1)[0] = ml_dtypes.bfloat16(float("nan"))
    output = io.StringIO()
    with pytest.raises(RuntimeError, match="numerical"):
        _PROBE._validate_candidate(
            np,
            tuple(actual),
            expected,
            0.01,
            _PROBE._completed_counters(),
            output,
        )
    [record] = _records(output)
    assert record["record_type"] == "host_vjp_validation"
    assert record["gates"]["per_tensor_numerical_passed"]["dq"] is False
    assert record["metrics"]["dq"]["relative_l2"] is None
    assert record["metrics"]["dq"]["cosine"] is None


@pytest.mark.parametrize(
    "duration",
    (float("nan"), float("inf"), float("-inf")),
    ids=("nan", "posinf", "neginf"),
)
def test_nonfinite_candidate_duration_serializes_dispatch_then_fails_validation(
    duration, exact_case, monkeypatch
):
    _inputs, _manifests, expected, _reference = exact_case
    actual = tuple(item.astype(ml_dtypes.bfloat16) for item in expected)

    class Compiled:
        def __call__(self, *_arguments):
            return actual

    counters = _PROBE._zero_counters()
    executable = _PROBE._wrap_checked(Compiled(), {"passed": True}, counters)
    clock = iter((123.0, 0.0, duration))
    monkeypatch.setattr(_PROBE.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(
        _PROBE,
        "_journal_checkpoint",
        lambda _clean, _output, _stage, _counters: dict(_CLEAN),
    )
    output = io.StringIO()
    dispatched, measured = _PROBE._dispatch_candidate(
        SimpleNamespace(block_until_ready=lambda value: value),
        executable,
        ("q", "k", "v", "mask", "dout"),
        lambda: dict(_CLEAN),
        counters,
        output,
    )
    assert dispatched is actual
    assert not np.isfinite(measured)
    dispatch_records = _records(output)
    assert dispatch_records[-1]["record_type"] == "dispatch"
    assert dispatch_records[-1]["seconds"] is None

    with pytest.raises(RuntimeError, match="duration gates"):
        _PROBE._validate_candidate(np, dispatched, expected, measured, counters, output)
    records = _records(output)
    validation = records[-1]
    assert validation["record_type"] == "host_vjp_validation"
    assert validation["candidate_total_seconds"] is None
    assert validation["gates"]["safety_duration_passed"] is False
    assert validation["gates"]["promotion_duration_passed"] is False


def test_compile_diagnostic_destroys_unreleased_artifact_with_all_runtime_counters_zero(
    monkeypatch,
):
    state = {"invocations": 0, "destroyed": 0}

    class NeverCallable:
        def __call__(self, *_arguments):
            state["invocations"] += 1
            raise AssertionError("compile diagnostic invoked compiled executable")

        def __del__(self):
            state["destroyed"] += 1

    def compile_artifact(_jax, _jnp, _api, _clean, counters, _output):
        for name in (
            "lower_attempts",
            "lower_completions",
            "compile_attempts",
            "compile_completions",
        ):
            counters[name] += 1
        return NeverCallable(), {
            "structural_gate": {"passed": True},
            "compiled_memory_gate": {"passed": True},
            "release_gate": {"passed": True},
        }

    monkeypatch.setattr(_PROBE, "_compile_vjp_artifact", compile_artifact)
    for forbidden in (
        "_release_checked_vjp",
        "_construct_host_case",
        "_device_put_inputs",
        "_dispatch_candidate",
        "_device_get_candidate",
    ):
        monkeypatch.setattr(
            _PROBE,
            forbidden,
            lambda *_args, _name=forbidden, **_kwargs: (_ for _ in ()).throw(
                AssertionError(f"compile diagnostic reached {_name}")
            ),
        )
    counters = _PROBE._zero_counters()
    output = io.StringIO()
    result = _PROBE._run_compile_diagnostic(
        object(),
        object(),
        lambda *_args, **_kwargs: None,
        lambda: dict(_CLEAN),
        counters,
        output,
    )
    gc.collect()
    assert result == 0
    assert state == {"invocations": 0, "destroyed": 1}
    assert counters == _PROBE._compile_diagnostic_completed_counters()
    for name in (
        "input_device_put_attempts",
        "input_device_put_completions",
        "candidate_attempts",
        "candidate_completions",
        "device_get_attempts",
        "device_get_completions",
        "lowered_callable_invocations",
        "checked_executable_invocations",
        "checked_capability_release_attempts",
        "checked_capability_release_completions",
        "host_reference_construction_attempts",
        "host_reference_construction_completions",
    ):
        assert counters[name] == 0
    [record] = _records(output)
    assert record["record_type"] == "compile_diagnostic_completed"
    assert record["status"] == "structure_and_memory_passed_no_release"
    assert record["checked_capability_created_or_released"] is False
    assert record["host_reference_constructed"] is False
    assert record["host_or_device_inputs_constructed"] is False
    assert record["executable_invoked"] is False
    assert record["device_output_retrieved"] is False


def test_failed_compile_diagnostic_returns_nonzero_without_release_or_runtime(
    monkeypatch,
):
    state = {"destroyed": 0}

    class FailedArtifact:
        def __del__(self):
            state["destroyed"] += 1

    def compile_artifact(_jax, _jnp, _api, _clean, counters, _output):
        for name in (
            "lower_attempts",
            "lower_completions",
            "compile_attempts",
            "compile_completions",
        ):
            counters[name] += 1
        return FailedArtifact(), {
            "structural_gate": {"passed": False},
            "compiled_memory_gate": {"passed": True},
            "release_gate": {"passed": False},
        }

    monkeypatch.setattr(_PROBE, "_compile_vjp_artifact", compile_artifact)
    for forbidden in (
        "_release_checked_vjp",
        "_construct_host_case",
        "_device_put_inputs",
        "_dispatch_candidate",
        "_device_get_candidate",
    ):
        monkeypatch.setattr(
            _PROBE,
            forbidden,
            lambda *_args, _name=forbidden, **_kwargs: (_ for _ in ()).throw(
                AssertionError(f"failed compile diagnostic reached {_name}")
            ),
        )
    counters = _PROBE._zero_counters()
    output = io.StringIO()
    result = _PROBE._run_compile_diagnostic(
        object(),
        object(),
        lambda *_args, **_kwargs: None,
        lambda: dict(_CLEAN),
        counters,
        output,
    )
    gc.collect()
    assert result == 2
    assert state["destroyed"] == 1
    assert counters == _PROBE._compile_diagnostic_completed_counters()
    assert all(
        counters[name] == 0
        for name in (
            "input_device_put_attempts",
            "input_device_put_completions",
            "candidate_attempts",
            "candidate_completions",
            "device_get_attempts",
            "device_get_completions",
            "lowered_callable_invocations",
            "checked_executable_invocations",
            "checked_capability_release_attempts",
            "checked_capability_release_completions",
            "host_reference_construction_attempts",
            "host_reference_construction_completions",
        )
    )
    [record] = _records(output)
    assert record["status"] == "structure_or_memory_failed_no_release"
    assert record["checked_capability_created_or_released"] is False
    assert record["host_reference_constructed"] is False
    assert record["host_or_device_inputs_constructed"] is False
    assert record["executable_invoked"] is False
    assert record["device_output_retrieved"] is False


def test_execute_emits_explicit_failed_compile_terminal_after_postflight(monkeypatch):
    class Guard:
        def __enter__(self):
            return dict(_CLEAN)

        def __exit__(self, _error_type, _error, _traceback):
            return False

    monkeypatch.setattr(_PROBE, "_assert_fresh_accelerator_process", lambda: None)
    monkeypatch.setattr(
        _PROBE, "_assert_static_source_bindings", lambda: {"passed": True}
    )
    monkeypatch.setattr(
        _PROBE,
        "_load_safety_helpers",
        lambda: (lambda: Guard(), lambda: dict(_CLEAN)),
    )
    monkeypatch.setattr(
        _PROBE, "_safety_binding_manifest", lambda _helpers: {"passed": True}
    )
    monkeypatch.setattr(_PROBE, "_configure_rocm_environment", lambda: {})
    monkeypatch.setattr(_PROBE, "_environment_manifest", lambda _environment: {})
    monkeypatch.setattr(_PROBE, "_public_safety_preflight", lambda safety: dict(safety))
    monkeypatch.setattr(
        _PROBE,
        "_public_clean_safety",
        lambda safety, stage: {**safety, "stage": stage},
    )

    def failed_run(
        _output,
        _require_clean_boot,
        counters,
        *,
        environment,
        compile_diagnostic,
        case,
    ):
        assert environment == {}
        assert compile_diagnostic is True
        assert case == "all_valid_t1024"
        assert counters == _PROBE._zero_counters()
        return 2

    monkeypatch.setattr(_PROBE, "_run_rocm", failed_run)
    output = io.StringIO()
    result = _PROBE._execute(
        SimpleNamespace(
            platform="rocm",
            allow_gpu=True,
            compile_diagnostic=True,
            case="all_valid_t1024",
        ),
        output,
    )
    assert result == 2
    records = _records(output)
    assert records[-2]["record_type"] == "safety_postflight"
    assert records[-1]["record_type"] == "completed"
    assert records[-1]["status"] == "compile_diagnostic_failed_no_runtime_release"
    assert records[-1]["counters"] == _PROBE._zero_counters()


def test_t1024_compile_diagnostic_can_capture_evidence_but_never_release_runtime(
    monkeypatch,
):
    state = {"destroyed": 0}

    class DiagnosticArtifact:
        def __del__(self):
            state["destroyed"] += 1

    def compile_artifact(_jax, _jnp, _api, _clean, counters, _output, case):
        assert case == "all_valid_t1024"
        for name in (
            "lower_attempts",
            "lower_completions",
            "compile_attempts",
            "compile_completions",
        ):
            counters[name] += 1
        return DiagnosticArtifact(), {
            "structural_gate": {"passed": True},
            "compiled_memory_gate": {"passed": True},
            "release_gate": {
                "diagnostic_evidence_passed": True,
                "runtime_release_authorized_by_case": False,
                "passed": False,
            },
        }

    monkeypatch.setattr(_PROBE, "_compile_vjp_artifact", compile_artifact)
    for forbidden in (
        "_release_checked_vjp",
        "_construct_host_case",
        "_device_put_inputs",
        "_dispatch_candidate",
        "_device_get_candidate",
    ):
        monkeypatch.setattr(
            _PROBE,
            forbidden,
            lambda *_args, _name=forbidden, **_kwargs: (_ for _ in ()).throw(
                AssertionError(f"T1024 diagnostic reached {_name}")
            ),
        )
    counters = _PROBE._zero_counters()
    output = io.StringIO()
    result = _PROBE._run_compile_diagnostic(
        object(),
        object(),
        lambda *_args, **_kwargs: None,
        lambda: dict(_CLEAN),
        counters,
        output,
        "all_valid_t1024",
    )
    gc.collect()
    assert result == 0
    assert state["destroyed"] == 1
    assert counters == _PROBE._compile_diagnostic_completed_counters()
    [record] = _records(output)
    assert record["status"] == "t1024_diagnostic_evidence_passed_runtime_still_withheld"
    assert record["release_gate_observed_but_not_authorized"] == {
        "diagnostic_evidence_passed": True,
        "runtime_release_authorized_by_case": False,
        "passed": False,
    }
    assert record["checked_capability_created_or_released"] is False
    assert record["host_reference_constructed"] is False
    assert record["host_or_device_inputs_constructed"] is False
    assert record["executable_invoked"] is False
    assert record["device_output_retrieved"] is False


def test_fake_run_rocm_executes_one_put_one_checked_call_one_get_and_exact_counters(
    monkeypatch,
):
    state = {
        "compiled_invocations": 0,
        "device_put_calls": 0,
        "device_get_calls": 0,
        "journal_stages": [],
    }

    def compile_checked(_jax, _jnp, _api, _clean, counters, _output):
        for name in (
            "lower_attempts",
            "lower_completions",
            "compile_attempts",
            "compile_completions",
        ):
            counters[name] = 1
        return (
            _CompiledFake(state),
            {
                "release_gate": {"passed": True},
                "compiled_memory_gate": {"passed": True},
            },
        )

    def checkpoint(require_clean, output, stage, counters):
        assert require_clean() == _CLEAN
        state["journal_stages"].append(stage)
        _PROBE._emit(
            {
                "record_type": "journal_checkpoint",
                "stage": stage,
                "counters": dict(counters),
            },
            output,
        )
        return dict(_CLEAN)

    def validate(_np, actual, expected, seconds, counters, output, case):
        assert case == "all_valid"
        assert actual == ("q", "q", "k", "v")
        assert expected == (
            "expected-output",
            "expected-dq",
            "expected-dk",
            "expected-dv",
        )
        assert 0 <= seconds < 0.1
        assert counters == _PROBE._completed_counters()
        record = {"gates": {"promotion_passed": True}}
        _PROBE._emit({"record_type": "host_vjp_validation", **record}, output)
        return record

    class FakeJax:
        @staticmethod
        def block_until_ready(value):
            return value

        @staticmethod
        def device_put(value):
            state["device_put_calls"] += 1
            return value

        @staticmethod
        def device_get(value):
            state["device_get_calls"] += 1
            return value

    backend = SimpleNamespace(
        _backend_manifest=lambda *_args: {
            "platform_resolved": "gpu",
            "platform_family": "rocm",
            "visible_device_count": 1,
        }
    )
    monkeypatch.setattr(
        _PROBE,
        "_nonzero_probe",
        lambda: SimpleNamespace(_length_probe=lambda: backend),
    )
    monkeypatch.setattr(
        _PROBE, "_prove_command_buffers_disabled", lambda _env: {"passed": True}
    )
    monkeypatch.setattr(
        _PROBE,
        "_assert_gfx1100_drm",
        lambda: {"passed": True, "architecture": "gfx1100"},
    )
    monkeypatch.setattr(_PROBE, "_assert_kernel_binding", lambda _api: {"passed": True})
    monkeypatch.setattr(_PROBE, "_compile_vjp_artifact", compile_checked)
    monkeypatch.setattr(_PROBE, "_journal_checkpoint", checkpoint)
    monkeypatch.setattr(
        _PROBE,
        "_construct_host_case",
        lambda _np, _ml, case: (
            ("q", "k", "v", "mask", "dout"),
            [],
            ("expected-output", "expected-dq", "expected-dk", "expected-dv"),
            {"oracle": {"accelerator_used": False}},
        ),
    )
    monkeypatch.setattr(_PROBE, "_validate_candidate", validate)
    counters = _PROBE._zero_counters()
    output = io.StringIO()
    result = _PROBE._run_rocm(
        output,
        lambda: dict(_CLEAN),
        counters,
        environment={},
        _dependencies=(
            FakeJax,
            object(),
            object(),
            object(),
            object(),
            object(),
            lambda *_args, **_kwargs: None,
        ),
    )
    assert result == 0
    assert counters == _PROBE._completed_counters()
    assert state["compiled_invocations"] == 1
    assert state["device_put_calls"] == state["device_get_calls"] == 1
    assert state["journal_stages"] == [
        "after_backend_initialization_attempt",
        "after_host_reference_construction",
        "after_explicit_input_device_put_attempt",
        "after_candidate_dispatch_attempt",
        "after_candidate_device_get_attempt",
        "after_host_validation",
    ]
    records = _records(output)
    assert records[-1]["record_type"] == "runtime_passed"
    assert records[-1]["counters"] == _PROBE._completed_counters()
    backend_record = next(
        item for item in records if item["record_type"] == "backend_ready"
    )
    assert backend_record["architecture_binding"]["architecture"] == "gfx1100"


def test_ast_proves_lazy_import_one_compile_one_vjp_and_one_checked_execution():
    source = _PROBE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    roots = {
        alias.name.partition(".")[0]
        for node in module.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert {"jax", "jaxlib", "numpy", "ml_dtypes", "skyrl"}.isdisjoint(roots)
    compile_source = inspect.getsource(_PROBE._compile_vjp_artifact)
    assert compile_source.count("jax.vjp(") == 1
    assert compile_source.count(".lower(") == 1
    assert compile_source.count("lowered.compile(") == 1
    assert (
        inspect.getsource(_PROBE._dispatch_candidate).count("executable.invoke(") == 1
    )
    run_source = inspect.getsource(_PROBE._run_rocm)
    assert run_source.index("_prove_command_buffers_disabled") < run_source.index(
        "import jax"
    )
    assert run_source.index("_assert_gfx1100_drm") < run_source.index("import jax")
    assert run_source.count("_device_put_inputs(") == 1
    assert run_source.count("_device_get_candidate(") == 1
    assert "if counters != _completed_counters():" in run_source
    diagnostic_source = inspect.getsource(_PROBE._run_compile_diagnostic)
    for forbidden in (
        "_wrap_checked",
        "_CheckedVjpExecutable",
        "_release_checked_vjp",
        "_construct_host_case",
        "_device_put_inputs",
        "_dispatch_candidate",
        "_device_get_candidate",
    ):
        assert forbidden not in diagnostic_source
    execute_source = inspect.getsource(_PROBE._execute)
    assert execute_source.index("_load_safety_helpers") < execute_source.index(
        "_configure_rocm_environment"
    )


def test_default_subprocess_refuses_without_importing_jax():
    program = f"""
import contextlib, importlib.util, io, json, sys
before=set(sys.modules)
spec=importlib.util.spec_from_file_location('isolated_vjp_probe',{str(_PROBE_PATH)!r})
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
captured=io.StringIO()
with contextlib.redirect_stdout(captured):
    result=module.main([])
new=set(sys.modules)-before
bad=sorted(name for name in new if name in {{'jax','jaxlib'}} or name.startswith(('jax.','jaxlib.')))
print(json.dumps({{'result':result,'bad':bad,'records':[json.loads(line) for line in captured.getvalue().splitlines()]}}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(_REPO)
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["result"] == 0
    assert result["bad"] == []
    assert [item["record_type"] for item in result["records"]] == [
        "manifest",
        "refused",
    ]
