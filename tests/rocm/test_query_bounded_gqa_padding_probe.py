from __future__ import annotations

import ast
import contextlib
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
_PROBE_PATH = _REPO / "rocm" / "probe_query_bounded_gqa_padding.py"
_SPEC = importlib.util.spec_from_file_location("padding_probe_test", _PROBE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)

_CASES = {
    "valid768": 768,
    "valid769": 769,
    "valid831": 831,
    "valid832": 832,
    "valid833": 833,
    "valid1023": 1023,
}
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
def exact_cases():
    return {case: _PROBE._construct_host_inputs(np, ml_dtypes, case) for case in _CASES}


def test_default_is_refusal_without_accelerator_import():
    before = _accelerator_modules()
    output = io.StringIO()
    assert (
        _PROBE._execute(
            SimpleNamespace(platform="abstract", allow_gpu=False, case=None), output
        )
        == 0
    )
    assert _accelerator_modules() == before
    manifest, refused = _records(output)
    assert manifest["scope"] == "abstract_refusal"
    assert manifest["contract"] == {
        "operation": "query_bounded_gqa_right_padding_refusal",
        "allowed_cases": list(_CASES),
        "one_case_per_fresh_process": True,
        "multi_case_gpu_path_exists": False,
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
        (
            (
                "--platform",
                "rocm",
                "--allow-gpu",
                "--output",
                "/tmp/new-padding.jsonl",
            ),
            "requires exactly one --case",
        ),
        (("--allow-gpu",), "only valid with --platform rocm"),
        (("--case", "valid768"), "only valid with --platform rocm"),
        (("--case", "valid770"), "invalid choice"),
        (("--valid-length", "768"), "unrecognized arguments"),
        (("--cases", "valid768,valid769"), "unrecognized arguments"),
        (("--replay",), "unrecognized arguments"),
        (("--backward",), "unrecognized arguments"),
    ],
)
def test_parser_refuses_implicit_gpu_invalid_and_multi_case_scope(
    arguments, message, capsys
):
    with pytest.raises(SystemExit) as raised:
        _PROBE._parse_args(list(arguments))
    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_private_output_is_exclusive_mode_0600(tmp_path):
    path = tmp_path / "padding.jsonl"
    assert _PROBE.main(["--output", str(path)]) == 0
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(SystemExit):
        _PROBE._parse_args(["--output", str(path)])


@pytest.mark.parametrize(("case", "valid_length"), _CASES.items())
def test_exact_contract_case_geometry_semantics_and_one_shot(case, valid_length):
    contract = _PROBE._exact_contract(case)
    first_affected = valid_length - 768

    assert contract["case"] == case
    assert contract["valid_length"] == valid_length
    assert [item["shape"] for item in contract["inputs"]] == [
        [1, 256, 16, 256],
        [1, 1024, 4, 256],
        [1, 1024, 4, 256],
        [1, 1024],
    ]
    assert contract["query_start"] == 768
    assert contract["scale_exact_fraction"] == "3/32"
    assert contract["mask_semantics"] == {
        "keys_only": True,
        "query_rows_masked": False,
        "padded_query_rows_remain_defined": True,
        "right_padding_prefix_required": True,
    }
    assert contract["affected_query_offsets"] == [first_affected, 255]
    assert contract["affected_query_row_count"] == 256 - first_affected
    assert contract["case_policy"] == {
        "allowed_cases": list(_CASES),
        "one_case_per_fresh_process": True,
        "multi_case_gpu_path_exists": False,
    }
    assert contract["compile_gate"]["exact_marker"] == "query_bounded_gqa_forward_q768"
    assert contract["compile_gate"]["exact_argument_bytes"] == 6_295_552
    assert contract["compile_gate"]["exact_output_bytes"] == 2_097_152
    assert contract["compile_gate"]["exact_alias_bytes"] == 0
    plan = contract["dispatch_plan"]
    assert plan["lower_calls"] == plan["compile_calls"] == 1
    assert plan["input_tuple_device_put_calls"] == 1
    assert plan["checked_candidate_invocations"] == plan["device_get_calls"] == 1
    assert all(
        plan[name] == 0
        for name in (
            "warmup_invocations",
            "replay_invocations",
            "backward_invocations",
            "gpu_reference_invocations",
            "device_error_reduction_invocations",
            "model_invocations",
        )
    )


@pytest.mark.parametrize("invalid", ("valid0", "valid1024", "768", ""))
def test_contract_rejects_cases_outside_exact_enum(invalid):
    with pytest.raises(ValueError, match="outside"):
        _PROBE._exact_contract(invalid)


def test_static_binding_pins_promoted_nonzero_gate(monkeypatch):
    proof = _PROBE._assert_static_source_bindings()
    assert proof["passed"] is True
    assert proof["nonzero_scale_helper_resolved_file_matches_expected"] is True
    assert (
        proof["delegated_nonzero_scale_probe_source_sha256"]
        == "999e027d4cc35a8d59cc294020f8865036f8fb817a847ac38f96e36b597f74ac"
    )
    path = _PROBE._source_files()[
        "delegated_nonzero_scale_probe_source_sha256"
    ].resolve()
    original = _PROBE._file_sha256
    monkeypatch.setattr(
        _PROBE,
        "_file_sha256",
        lambda value: "0" * 64 if Path(value).resolve() == path else original(value),
    )
    with pytest.raises(RuntimeError, match="source SHA256"):
        _PROBE._assert_static_source_bindings()


@pytest.mark.parametrize(("case", "valid_length"), _CASES.items())
def test_exact_inputs_hashes_calibration_scratch_and_key_only_semantics(
    exact_cases, case, valid_length
):
    inputs, manifests, expected, expected_manifest, control = exact_cases[case]
    q, k, v, mask = inputs
    calibration = _PROBE._EXPECTED_CASE_CALIBRATION[case]
    first_affected = valid_length - 768

    assert q.shape == expected.shape == (1, 256, 16, 256)
    assert k.shape == v.shape == (1, 1024, 4, 256)
    assert mask.dtype == np.int32
    assert np.all(mask[:, :valid_length] == 1)
    assert np.all(mask[:, valid_length:] == 0)
    assert [item["sha256"] for item in manifests[:3]] == list(
        _PROBE._EXPECTED_QKV_SHA256
    )
    assert manifests[3]["sha256"] == calibration["mask_sha256"]
    assert expected_manifest["sha256"] == calibration["expected_sha256"]
    assert expected_manifest["oracle"]["valid_length"] == valid_length
    assert expected_manifest["oracle"]["key_only_mask_semantics"] is True
    assert expected_manifest["oracle"]["query_rows_masked"] is False
    assert (
        expected_manifest["oracle"]["conservative_accounted_numpy_array_scratch_bytes"]
        == 334_400
    )
    assert all(expected_manifest["calibration_pin_checks"].values())
    assert np.all(np.linalg.norm(expected[0], axis=(1, 2)) > 0)

    assert control["passed"] is True
    assert control["output"]["sha256"] == _PROBE._EXPECTED_WRONG_ALL_VALID_SHA256
    assert control["first_affected_row"]["query_offset"] == first_affected
    assert control["affected_rows_metrics"]["relative_l2"] == pytest.approx(
        calibration["affected_relative_l2"], abs=1e-12
    )
    assert control["first_affected_row"]["metrics"]["relative_l2"] == pytest.approx(
        calibration["first_affected_relative_l2"], abs=1e-12
    )
    assert control["affected_rows_metrics"]["relative_l2"] > 0.02
    assert control["first_affected_row"]["metrics"]["relative_l2"] > 0.02


def test_valid1023_control_is_localized_instead_of_aggregate_diluted(exact_cases):
    _inputs, _manifests, expected, _manifest, control = exact_cases["valid1023"]
    wrong = _PROBE._nonzero_probe()._streaming_causal_gqa_oracle  # provenance only
    assert callable(wrong)
    all_valid_output_hash = control["output"]["sha256"]
    assert all_valid_output_hash == _PROBE._EXPECTED_WRONG_ALL_VALID_SHA256
    assert control["affected_query_row_count"] == 1
    assert control["whole_output_metrics_informational_only"]["relative_l2"] < 0.002
    assert control["affected_rows_metrics"]["relative_l2"] > 0.03
    assert control["first_affected_row"]["metrics"]["relative_l2"] > 0.03
    assert control["first_affected_row"]["global_query_position"] == 1023
    assert expected.shape[1] == 256


@pytest.mark.parametrize("case", tuple(_CASES))
def test_streaming_exact_oracle_matches_independent_dense_reference(exact_cases, case):
    inputs, _manifests, expected, _manifest, _control = exact_cases[case]
    q, k, v, mask = inputs
    dense = _PROBE._dense_key_masked_causal_gqa_reference(
        np, q, k, v, mask, query_start=768, scale=3 / 32
    )
    np.testing.assert_allclose(expected, dense, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("valid_length", (1, 5, 8, 9, 17))
def test_streaming_small_oracle_matches_dense_for_prefix_masks(valid_length):
    rng = np.random.Generator(np.random.PCG64(7731 + valid_length))
    q = rng.uniform(-0.75, 0.75, size=(1, 9, 4, 8)).astype(np.float32)
    k = rng.uniform(-0.75, 0.75, size=(1, 17, 2, 8)).astype(np.float32)
    v = rng.uniform(-0.75, 0.75, size=(1, 17, 2, 8)).astype(np.float32)
    mask = np.zeros((1, 17), dtype=np.int32)
    mask[:, :valid_length] = 1
    streaming, metadata = _PROBE._streaming_key_masked_causal_gqa_oracle(
        np,
        q,
        k,
        v,
        mask,
        query_start=8,
        scale=3 / 32,
        query_tile=3,
        key_tile=5,
    )
    dense = _PROBE._dense_key_masked_causal_gqa_reference(
        np, q, k, v, mask, query_start=8, scale=3 / 32
    )
    np.testing.assert_allclose(streaming, dense, rtol=2e-6, atol=2e-7)
    assert metadata["valid_length"] == valid_length
    assert metadata["key_only_mask_semantics"] is True


@pytest.mark.parametrize(
    "mask",
    (
        np.zeros((1, 17), dtype=np.int32),
        np.asarray([[1, 0, 1] + [0] * 14], dtype=np.int32),
        np.asarray([[1, 2] + [0] * 15], dtype=np.int32),
    ),
)
def test_oracle_rejects_empty_nonprefix_and_nonbinary_masks(mask):
    q = np.ones((1, 2, 2, 3), dtype=np.float32)
    k = np.ones((1, 17, 1, 3), dtype=np.float32)
    v = np.ones((1, 17, 1, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        _PROBE._streaming_key_masked_causal_gqa_oracle(
            np, q, k, v, mask, query_start=8, scale=3 / 32
        )


@pytest.mark.parametrize(("case", "valid_length"), _CASES.items())
def test_transition_rows_bracket_first_mask_affected_query(case, valid_length):
    rows = _PROBE._transition_rows(valid_length)
    first = valid_length - 768
    by_role = {row["role"]: row for row in rows}
    assert by_role["first_affected"]["query_offset"] == first
    assert by_role["first_affected"]["global_query_position"] == valid_length
    assert by_role["first_affected"]["affected_by_padding"] is True
    if first > 0:
        assert by_role["last_unaffected"]["global_query_position"] == valid_length - 1
        assert by_role["last_unaffected"]["affected_by_padding"] is False
    if first < 255:
        assert by_role["second_affected"]["global_query_position"] == valid_length + 1


def test_row_metrics_identify_local_corruption_and_validation_records_transitions(
    exact_cases,
):
    _inputs, _manifests, expected, _manifest, _control = exact_cases["valid1023"]
    actual = expected.astype(ml_dtypes.bfloat16)
    output = io.StringIO()
    record = _PROBE._validate_candidate(
        np,
        actual,
        expected,
        0.01,
        _PROBE._completed_counters(),
        output,
        "valid1023",
    )
    assert record["gates"]["promotion_passed"] is True
    assert [item["role"] for item in record["transition_adjacent_row_metrics"]] == [
        "last_unaffected",
        "first_affected",
    ]

    corrupted = np.array(actual, copy=True)
    corrupted[:, 255] = ml_dtypes.bfloat16(2.0)
    summary = _PROBE._row_metrics(np, corrupted, expected)
    assert summary["maximum_relative_l2"]["query_offset"] == 255
    with pytest.raises(RuntimeError, match="row-local"):
        _PROBE._validate_candidate(
            np,
            corrupted,
            expected,
            0.01,
            _PROBE._completed_counters(),
            io.StringIO(),
            "valid1023",
        )


def _stablehlo(metadata: str) -> str:
    return "\n".join(
        [
            f'#loc0 = loc("query_bounded_gqa_forward_q768 {metadata}")',
            "module {",
            f'  %0 = stablehlo.custom_call @"{_TARGET}"() : () -> tensor<1xbf16> loc(#loc0)',
            "}",
        ]
    )


def _optimized_hlo(metadata: str) -> str:
    return "\n".join(
        [
            "ENTRY main {",
            (
                f'  ROOT %0 = bf16[1] custom-call(), custom_call_target="{_TARGET}", '
                f'op_name="query_bounded_gqa_forward_q768 {metadata}"'
            ),
            "}",
        ]
    )


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
@pytest.mark.parametrize(
    "metadata",
    (
        "query_start=768x query_size=256x",
        "query_start=768garbage query_size=256garbage",
        "query_start=768e2 query_size=256e2",
        "query_start=768.0 query_size=256.0",
        "prefix_query_start=768 prefix_query_size=256",
        "query_start=768 query_start=768 query_size=256",
        "query_start=768 query_size=256 query_size=256",
    ),
)
def test_pinned_raw_metadata_gate_rejects_suffix_prefix_and_duplicate_spoofs(
    dialect, metadata
):
    text = _stablehlo(metadata) if dialect == "stablehlo" else _optimized_hlo(metadata)
    summary = _PROBE._nonzero_probe()._strict_raw_query_metadata_summary(text, dialect)
    assert summary["passed"] is False


def test_ast_proves_lazy_import_single_case_and_delegated_one_shot_paths():
    source = _PROBE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    roots = {
        alias.name.partition(".")[0]
        for node in module.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert {"jax", "jaxlib", "numpy", "ml_dtypes", "skyrl"}.isdisjoint(roots)
    assert "for case in _CASE_ORDER" not in source
    assert "multi_case_gpu_path_exists" in source

    run_source = inspect.getsource(_PROBE._run_rocm)
    assert run_source.index("_prove_command_buffers_disabled") < run_source.index(
        "import jax"
    )
    assert run_source.count("nonzero._compile_checked_chunk(") == 1
    assert run_source.count("_device_put_inputs(") == 1
    assert run_source.count("_dispatch_candidate(") == 1
    assert run_source.count("_device_get_candidate(") == 1
    assert all(token not in source for token in ("jax.vjp(", "jax.grad("))

    execute_source = inspect.getsource(_PROBE._execute)
    assert execute_source.index("_load_safety_helpers") < execute_source.index(
        "_configure_rocm_environment"
    )


def test_execute_propagates_exactly_one_case_after_safety_binding(monkeypatch):
    events: list[str] = []

    @contextlib.contextmanager
    def guarded():
        events.append("guard")
        yield {"synthetic": True}

    def clean():
        return dict(_CLEAN)

    monkeypatch.setattr(_PROBE, "_assert_fresh_accelerator_process", lambda: None)
    monkeypatch.setattr(
        _PROBE, "_assert_static_source_bindings", lambda: {"passed": True}
    )
    monkeypatch.setattr(_PROBE, "_load_safety_helpers", lambda: (guarded, clean))
    monkeypatch.setattr(
        _PROBE,
        "_safety_binding_manifest",
        lambda helpers: events.append("safety_bound") or {"passed": True},
    )
    monkeypatch.setattr(
        _PROBE,
        "_configure_rocm_environment",
        lambda: events.append("environment") or {},
    )
    monkeypatch.setattr(_PROBE, "_environment_manifest", lambda environment: {})
    monkeypatch.setattr(_PROBE, "_public_safety_preflight", lambda raw: dict(_CLEAN))
    monkeypatch.setattr(_PROBE, "_public_clean_safety", lambda raw, stage: dict(_CLEAN))

    def run(output, clean_helper, counters, *, environment, case):
        assert clean_helper is clean
        assert case == "valid832"
        events.append(f"run:{case}")
        return 0

    monkeypatch.setattr(_PROBE, "_run_rocm", run)
    output = io.StringIO()
    assert (
        _PROBE._execute(
            SimpleNamespace(platform="rocm", allow_gpu=True, case="valid832"),
            output,
        )
        == 0
    )
    assert events.index("safety_bound") < events.index("environment")
    assert events.count("run:valid832") == 1


def test_default_subprocess_refuses_without_importing_jax():
    program = f"""
import contextlib, importlib.util, io, json, sys
before=set(sys.modules)
spec=importlib.util.spec_from_file_location('isolated_padding_probe',{str(_PROBE_PATH)!r})
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
    assert [record["record_type"] for record in result["records"]] == [
        "manifest",
        "refused",
    ]
