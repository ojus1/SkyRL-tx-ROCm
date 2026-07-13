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
_PROBE_PATH = _REPO / "rocm" / "probe_query_bounded_gqa_nonzero_scale.py"
_SPEC = importlib.util.spec_from_file_location("nonzero_scale_probe_test", _PROBE_PATH)
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
def exact_host_case():
    return _PROBE._construct_host_inputs(np, ml_dtypes)


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
        "operation": "query_bounded_gqa_nonzero_scale_sentinel_refusal",
        "sequence_length": 1024,
        "query_size": 256,
        "candidate_scale_exact_fraction": "3/32",
        "wrong_scale_exact_fraction": "1/16",
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
        (("--scale", "0.09375"), "unrecognized arguments"),
        (("--replay",), "unrecognized arguments"),
        (("--backward",), "unrecognized arguments"),
        (("--gpu-reference",), "unrecognized arguments"),
    ],
)
def test_parser_refuses_implicit_gpu_and_scope_broadening(arguments, message, capsys):
    with pytest.raises(SystemExit) as raised:
        _PROBE._parse_args(list(arguments))
    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_private_output_is_exclusive_mode_0600(tmp_path):
    path = tmp_path / "nonzero-scale.jsonl"
    assert _PROBE.main(["--output", str(path)]) == 0
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(SystemExit):
        _PROBE._parse_args(["--output", str(path)])


def test_exact_contract_is_immutable_t1024_c256_scale_sentinel():
    contract = _PROBE._exact_contract()

    assert [item["shape"] for item in contract["inputs"]] == [
        [1, 256, 16, 256],
        [1, 1024, 4, 256],
        [1, 1024, 4, 256],
        [1, 1024],
    ]
    assert contract["query_start"] == 768
    assert contract["scale"] == {
        "candidate": 0.09375,
        "candidate_exact_fraction": "3/32",
        "wrong_control": 0.0625,
        "wrong_control_exact_fraction": "1/16",
    }
    assert contract["integer_grid_denominator"] == 128
    assert contract["maximum_integer_magnitude"] == 96
    assert contract["compile_gate"] == {
        "independent_dialects": ["stablehlo", "optimized_hlo"],
        "exact_custom_call_count": 1,
        "exact_kernel_marker": "query_bounded_gqa_forward_q768",
        "exact_query_start_metadata_required_per_dialect": 768,
        "exact_query_size_metadata_required_per_dialect": 256,
        "absent_or_lookalike_metadata_rejected": True,
        "raw_integer_value_suffix_prefix_and_duplicate_ambiguity_rejected": True,
        "no_outer_while": True,
        "exact_argument_bytes": 6_295_552,
        "exact_output_bytes": 2_097_152,
        "exact_alias_bytes": 0,
        "maximum_temporary_bytes": 64 * 1024**2,
    }
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


def test_exact_host_grid_shapes_individual_draws_and_determinism(exact_host_case):
    inputs, manifests, expected, expected_manifest, _control = exact_host_case
    q, k, v, mask = inputs

    assert q.shape == expected.shape == (1, 256, 16, 256)
    assert k.shape == v.shape == (1, 1024, 4, 256)
    assert mask.shape == (1, 1024)
    assert all(str(value.dtype) == "bfloat16" for value in (q, k, v))
    assert mask.dtype == np.int32 and np.all(mask == 1)
    for value in (q, k, v):
        fp32 = np.asarray(value, dtype=np.float32)
        assert np.count_nonzero(value) == value.size
        assert float(np.max(np.abs(fp32))) == 96 / 128
        np.testing.assert_array_equal(fp32 * 128, np.rint(fp32 * 128))
        assert np.unique(value.reshape(-1)[:4096]).size > 128
    assert not np.array_equal(q[:, :, 0], q[:, :, 1])
    assert not np.array_equal(k[:, :, 0], k[:, :, 1])
    assert not np.array_equal(v[:, 1:], v[:, :-1])
    assert [item["name"] for item in manifests] == ["q_chunk", "k", "v", "key_mask"]
    assert [item["sha256"] for item in manifests] == [
        "16aa12a02e88387223f000513febba987c23490016e5a1c9fe32019a862afc5d",
        "85ba4ec243a74b9a2019d30c94c4a0edf62e2204a6d3148fe551113605134841",
        "60132fd8c7733d2f02381f90d59e5a3dc7d740c09f25e1833540d7c956771b8a",
        "b33dd739a3b1d1e659a638b318bdcfbaed8eb8cca224dbf0a76e9e1a81db57bc",
    ]
    assert expected.dtype == np.float32
    assert (
        expected_manifest["sha256"]
        == "623a578b3d6b4d96461fc2a9f2d7bf97ba4fc02e3c9791f449bae89d77193db5"
    )
    assert (
        expected_manifest["oracle"]["full_logits_or_probability_matrix_constructed"]
        is False
    )
    assert (
        expected_manifest["oracle"]["conservative_accounted_numpy_array_scratch_bytes"]
        == 332_288
    )

    again = _PROBE._construct_host_inputs(np, ml_dtypes)
    assert [item["sha256"] for item in manifests] == [
        item["sha256"] for item in again[1]
    ]
    assert expected_manifest["sha256"] == again[3]["sha256"]


def test_wrong_scale_control_has_calibrated_decisive_sensitivity(exact_host_case):
    control = exact_host_case[4]
    metrics = control["metrics_vs_scale_3_over_32"]

    assert control["passed"] is True
    assert control["scale_exact_fraction"] == "1/16"
    assert (
        control["output"]["sha256"]
        == "3d835dd7a97ad8a49d7dadc2028410215ae8a49eca7435b9e09e5f326945a64b"
    )
    assert metrics["finite"] is True
    assert metrics["relative_l2"] > 0.05
    assert metrics["relative_l2"] == pytest.approx(0.0984490283, abs=2e-8)
    assert metrics["cosine"] == pytest.approx(0.9953191390, abs=2e-9)
    assert metrics["max_abs"] == pytest.approx(0.00962096, abs=2e-8)
    assert exact_host_case[3]["oracle"][
        "observed_maximum_absolute_valid_logit"
    ] == pytest.approx(1.43171310425, abs=1e-10)


@pytest.mark.parametrize("scale", (3 / 32, 1 / 16, 1 / 8))
def test_streaming_oracle_matches_independent_dense_small_reference(scale):
    rng = np.random.Generator(np.random.PCG64(918273))
    q = rng.uniform(-0.75, 0.75, size=(1, 9, 4, 8)).astype(np.float32)
    k = rng.uniform(-0.75, 0.75, size=(1, 17, 2, 8)).astype(np.float32)
    v = rng.uniform(-0.75, 0.75, size=(1, 17, 2, 8)).astype(np.float32)
    streaming, metadata = _PROBE._streaming_causal_gqa_oracle(
        np,
        q,
        k,
        v,
        query_start=8,
        scale=scale,
        query_tile=3,
        key_tile=5,
    )
    dense = _PROBE._dense_small_reference(np, q, k, v, query_start=8, scale=scale)

    np.testing.assert_allclose(streaming, dense, rtol=2e-6, atol=2e-7)
    assert metadata["query_tile"] == 3
    assert metadata["key_tile"] == 5
    assert metadata["full_logits_or_probability_matrix_constructed"] is False
    assert metadata["conservative_accounted_numpy_array_scratch_bytes"] == 1279
    assert (
        metadata["scratch_bytes_are_conservative_accounting_not_measured_peak"] is True
    )
    assert (
        "simultaneously_live_bool_valid_mask" in metadata["scratch_accounting_includes"]
    )
    assert "numpy_blas_internal_workspace" in metadata["scratch_accounting_excludes"]


def test_dense_reference_refuses_production_size(exact_host_case):
    q, k, v, _mask = exact_host_case[0]
    with pytest.raises(RuntimeError, match="restricted to small CPU tests"):
        _PROBE._dense_small_reference(np, q, k, v, query_start=768, scale=3 / 32)


def _stablehlo(
    *,
    marker: str = "query_bounded_gqa_forward_q768",
    metadata: str | None = None,
) -> str:
    metadata = (
        f"{marker} query_start=768 query_size=256" if metadata is None else metadata
    )
    return "\n".join(
        [
            f'#loc0 = loc("{metadata}")',
            "module {",
            f'  %0 = stablehlo.custom_call @"{_TARGET}"() : () -> tensor<1xbf16> loc(#loc0)',
            "}",
        ]
    )


def _optimized_hlo(
    *,
    marker: str = "query_bounded_gqa_forward_q768",
    metadata: str | None = None,
) -> str:
    metadata = (
        f"{marker} query_start=768 query_size=256" if metadata is None else metadata
    )
    return "\n".join(
        [
            "ENTRY main {",
            (
                f'  ROOT %0 = bf16[1] custom-call(), custom_call_target="{_TARGET}", '
                f'op_name="{metadata}"'
            ),
            "}",
        ]
    )


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
@pytest.mark.parametrize(
    "metadata",
    (
        "query_bounded_gqa_forward_q768",
        "query_bounded_gqa_forward_q768 query_start=767 query_size=256",
        "query_bounded_gqa_forward_q768 query_start=768 query_size=255",
        "query_bounded_gqa_forward_q768 query_start_spoof=768 query_size_spoof=256",
        (
            "query_bounded_gqa_forward_q768 query_start=768 query_size=256 "
            "query_start_spoof=768 query_size_spoof=256"
        ),
    ),
)
def test_strict_local_metadata_gate_rejects_absent_incorrect_and_spoofed_per_dialect(
    dialect, metadata
):
    text = (
        _stablehlo(metadata=metadata)
        if dialect == "stablehlo"
        else _optimized_hlo(metadata=metadata)
    )
    summary = _PROBE._length_probe()._summarize_ir(text, dialect, 1024)
    other_dialect = "optimized_hlo" if dialect == "stablehlo" else "stablehlo"
    other_text = _optimized_hlo() if other_dialect == "optimized_hlo" else _stablehlo()
    other = _PROBE._length_probe()._summarize_ir(other_text, other_dialect, 1024)
    gate = _PROBE._strict_query_metadata_gate(summary, other)
    raw = _PROBE._strict_raw_query_metadata_summary(text, dialect)
    other_raw = _PROBE._strict_raw_query_metadata_summary(other_text, other_dialect)
    raw_gate = _PROBE._strict_raw_query_metadata_gate(raw, other_raw)

    assert gate["passed"] is False
    assert gate["per_dialect"][dialect]["passed"] is False
    assert raw["passed"] is False
    assert raw_gate["passed"] is False


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
@pytest.mark.parametrize("suffix", ("x", "garbage", "e2", ".0"))
def test_independent_raw_metadata_gate_rejects_numeric_suffix_spoofs_per_dialect(
    dialect, suffix
):
    metadata = (
        f"query_bounded_gqa_forward_q768 query_start=768{suffix} query_size=256{suffix}"
    )
    text = (
        _stablehlo(metadata=metadata)
        if dialect == "stablehlo"
        else _optimized_hlo(metadata=metadata)
    )
    delegated = _PROBE._length_probe()._summarize_ir(text, dialect, 1024)
    raw = _PROBE._strict_raw_query_metadata_summary(text, dialect)
    other_dialect = "optimized_hlo" if dialect == "stablehlo" else "stablehlo"
    other_text = _optimized_hlo() if dialect == "stablehlo" else _stablehlo()
    other_raw = _PROBE._strict_raw_query_metadata_summary(other_text, other_dialect)

    # This is the exact delegated-regex weakness the independent gate closes.
    assert delegated["passed"] is True
    assert raw["passed"] is False
    assert raw["query_start"]["noncanonical_value_occurrences"] == 1
    assert raw["query_size"]["noncanonical_value_occurrences"] == 1
    assert _PROBE._strict_raw_query_metadata_gate(raw, other_raw)["passed"] is False


@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
@pytest.mark.parametrize(
    "metadata",
    (
        "query_bounded_gqa_forward_q768 prefix_query_start=768 prefix_query_size=256",
        (
            "query_bounded_gqa_forward_q768 query_start=768 query_size=256 "
            "prefix_query_start=768 prefix_query_size=256"
        ),
        (
            "query_bounded_gqa_forward_q768 query_start=768 query_start=768 "
            "query_size=256"
        ),
        (
            "query_bounded_gqa_forward_q768 query_start=768 query_size=256 "
            "query_size=256"
        ),
    ),
)
def test_independent_raw_metadata_gate_rejects_prefix_lookalike_and_duplicates(
    dialect, metadata
):
    text = (
        _stablehlo(metadata=metadata)
        if dialect == "stablehlo"
        else _optimized_hlo(metadata=metadata)
    )
    assert _PROBE._strict_raw_query_metadata_summary(text, dialect)["passed"] is False


def test_strict_local_metadata_gate_requires_exact_combined_dialects():
    stable = _PROBE._length_probe()._summarize_ir(_stablehlo(), "stablehlo", 1024)
    optimized = _PROBE._length_probe()._summarize_ir(
        _optimized_hlo(), "optimized_hlo", 1024
    )
    assert _PROBE._strict_query_metadata_gate(stable, optimized)["passed"] is True
    stable_raw = _PROBE._strict_raw_query_metadata_summary(_stablehlo(), "stablehlo")
    optimized_raw = _PROBE._strict_raw_query_metadata_summary(
        _optimized_hlo(), "optimized_hlo"
    )
    assert (
        _PROBE._strict_raw_query_metadata_gate(stable_raw, optimized_raw)["passed"]
        is True
    )

    absent_optimized = _PROBE._length_probe()._summarize_ir(
        _optimized_hlo(metadata="query_bounded_gqa_forward_q768"),
        "optimized_hlo",
        1024,
    )
    combined = _PROBE._strict_query_metadata_gate(stable, absent_optimized)
    assert combined["passed"] is False
    assert combined["checks"]["stablehlo_exact_metadata_preserved"] is True
    assert combined["checks"]["optimized_hlo_exact_metadata_preserved"] is False
    absent_optimized_raw = _PROBE._strict_raw_query_metadata_summary(
        _optimized_hlo(metadata="query_bounded_gqa_forward_q768"),
        "optimized_hlo",
    )
    raw_combined = _PROBE._strict_raw_query_metadata_gate(
        stable_raw, absent_optimized_raw
    )
    assert raw_combined["passed"] is False


class _FakeShape:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _CompiledFake:
    def __init__(self, state):
        self.state = state

    def as_text(self):
        self.state["as_text"] += 1
        return _optimized_hlo()

    def memory_analysis(self):
        self.state["memory"] += 1
        return SimpleNamespace(
            argument_size_in_bytes=6_295_552,
            output_size_in_bytes=2_097_152,
            alias_size_in_bytes=0,
            temp_size_in_bytes=16_640,
        )

    def __call__(self, *arguments):
        self.state["invocations"] += 1
        return arguments[0]


class _LoweredFake:
    def __init__(self, state, compiled):
        self.state = state
        self.compiled = compiled

    def compiler_ir(self, *, dialect):
        assert dialect == "stablehlo"
        self.state["compiler_ir"] += 1
        return _stablehlo()

    def compile(self):
        self.state["compile"] += 1
        return self.compiled


class _JitFake:
    def __init__(self, function, state, lowered):
        self.function = function
        self.state = state
        self.lowered = lowered

    def lower(self, *signature):
        self.state["lower"] += 1
        self.state["signature"] = signature
        self.function(*signature)
        return self.lowered


class _JaxFake:
    ShapeDtypeStruct = _FakeShape

    def __init__(self, state, lowered):
        self.state = state
        self.lowered = lowered

    def jit(self, function):
        self.state["jit"] += 1
        return _JitFake(function, self.state, self.lowered)


def test_compile_mock_uses_exact_shapes_explicit_scale_and_zero_invocations():
    state = {
        "jit": 0,
        "lower": 0,
        "compiler_ir": 0,
        "compile": 0,
        "as_text": 0,
        "memory": 0,
        "invocations": 0,
        "api": [],
    }
    compiled = _CompiledFake(state)
    lowered = _LoweredFake(state, compiled)

    def api(*arguments, **keywords):
        state["api"].append((arguments, keywords))
        return "abstract"

    checked, report = _PROBE._compile_checked_chunk(
        _JaxFake(state, lowered),
        SimpleNamespace(bfloat16="bf16", int32="i32"),
        api,
        lambda: dict(_CLEAN),
        _PROBE._zero_counters(),
        io.StringIO(),
    )

    assert checked.proof["passed"] is True
    assert state["jit"] == state["lower"] == state["compile"] == 1
    assert state["invocations"] == 0
    assert [item.shape for item in state["signature"]] == [
        (1, 256, 16, 256),
        (1, 1024, 4, 256),
        (1, 1024, 4, 256),
        (1, 1024),
    ]
    assert state["api"][0][1] == {
        "query_start": 768,
        "scale": 0.09375,
        "block_q": 64,
        "block_k": 64,
        "interpret": False,
    }
    assert report["release_gate"]["explicit_scale_exact_fraction"] == "3/32"
    assert report["release_gate"]["strict_query_metadata_gate_passed"] is True
    assert report["release_gate"]["strict_raw_query_metadata_gate_passed"] is True
    assert report["strict_query_metadata_gate"]["passed"] is True
    assert report["strict_raw_query_metadata_gate"]["passed"] is True
    assert report["compiled_memory_gate"]["passed"] is True


@pytest.mark.parametrize(
    "failure", ("structural", "metadata", "raw_suffix_metadata", "memory")
)
def test_compile_failure_never_releases_or_invokes(failure):
    state = {
        name: 0
        for name in (
            "jit",
            "lower",
            "compiler_ir",
            "compile",
            "as_text",
            "memory",
            "invocations",
        )
    }
    state["api"] = []
    compiled = _CompiledFake(state)
    if failure == "structural":
        compiled.as_text = lambda: _optimized_hlo(marker="query_bounded_gqa_forward_q0")
    elif failure == "metadata":
        compiled.as_text = lambda: _optimized_hlo(
            metadata="query_bounded_gqa_forward_q768"
        )
    elif failure == "raw_suffix_metadata":
        compiled.as_text = lambda: _optimized_hlo(
            metadata=(
                "query_bounded_gqa_forward_q768 "
                "query_start=768garbage query_size=256garbage"
            )
        )
    else:
        compiled.memory_analysis = lambda: SimpleNamespace(
            argument_size_in_bytes=6_295_552,
            output_size_in_bytes=2_097_152,
            alias_size_in_bytes=1,
            temp_size_in_bytes=0,
        )
    lowered = _LoweredFake(state, compiled)

    with pytest.raises(RuntimeError, match="failed the structural"):
        _PROBE._compile_checked_chunk(
            _JaxFake(state, lowered),
            SimpleNamespace(bfloat16="bf16", int32="i32"),
            lambda *args, **kwargs: "abstract",
            lambda: dict(_CLEAN),
            _PROBE._zero_counters(),
            io.StringIO(),
        )
    assert state["invocations"] == 0


def test_static_source_binding_accepts_exact_pins_and_rejects_corruption(monkeypatch):
    proof = _PROBE._assert_static_source_bindings()
    assert proof["passed"] is True
    assert proof["length_helper_resolved_file_matches_expected"] is True
    assert (
        proof["delegated_length_probe_source_sha256"]
        == "64a38e3d381d8b75cae064a43c850c3d7d3d28e610631522cf52fba0a6483aa4"
    )
    assert (
        proof["delegated_chunk_compile_probe_source_sha256"]
        == "24eeed83e93da1133d2e1bc3d0065bc8369d13fa324d2157e37db8b9c4a4d12d"
    )
    assert (
        proof["query_bounded_gqa_kernel_source_sha256"]
        == "51e2fd91eb270f7b25ecdd117d7f06aa48a8e4af282a5a7e5e6b4c2a25dc52c9"
    )

    original = _PROBE._file_sha256
    target = _PROBE._source_files()["query_bounded_gqa_kernel_source_sha256"].resolve()
    monkeypatch.setattr(
        _PROBE,
        "_file_sha256",
        lambda path: "0" * 64 if Path(path).resolve() == target else original(path),
    )
    with pytest.raises(RuntimeError, match="pinned source SHA256 mismatch"):
        _PROBE._assert_static_source_bindings()


def test_safety_callables_resolve_to_the_pinned_helper(monkeypatch):
    helpers = _PROBE._load_safety_helpers()
    expected = _PROBE._source_files()["delegated_safety_helper_source_sha256"].resolve()
    assert all(
        Path(sys.modules[helper.__module__].__file__).resolve() == expected
        for helper in helpers
    )
    binding = _PROBE._safety_binding_manifest(helpers)
    assert binding["passed"] is True
    assert binding["callable_names"] == [
        "guarded_qwen35_rocm_process",
        "require_clean_amdgpu_boot",
    ]
    assert binding["retained_for_guard_and_journal_use"] is True

    length = _PROBE._length_probe()
    monkeypatch.setattr(
        length, "_load_safety_helpers", lambda: (lambda: None, lambda: None)
    )
    with pytest.raises(RuntimeError, match="safety helper"):
        _PROBE._load_safety_helpers()


def test_execute_binds_and_retains_safety_callables_before_environment(monkeypatch):
    events: list[str] = []

    @contextlib.contextmanager
    def guarded():
        events.append("guard_entered")
        yield {"synthetic": True}

    def require_clean():
        events.append("clean_checked")
        return dict(_CLEAN)

    monkeypatch.setattr(
        _PROBE, "_assert_fresh_accelerator_process", lambda: events.append("fresh")
    )
    monkeypatch.setattr(
        _PROBE,
        "_assert_static_source_bindings",
        lambda: events.append("source_bound") or {"passed": True},
    )
    monkeypatch.setattr(
        _PROBE,
        "_load_safety_helpers",
        lambda: events.append("safety_loaded") or (guarded, require_clean),
    )
    monkeypatch.setattr(
        _PROBE,
        "_safety_binding_manifest",
        lambda helpers: (
            events.append("safety_validated")
            or {
                "passed": True,
                "retained_for_guard_and_journal_use": helpers
                == (guarded, require_clean),
            }
        ),
    )
    monkeypatch.setattr(
        _PROBE,
        "_configure_rocm_environment",
        lambda: events.append("environment_mutated") or {},
    )
    monkeypatch.setattr(_PROBE, "_environment_manifest", lambda environment: {})
    monkeypatch.setattr(
        _PROBE,
        "_public_safety_preflight",
        lambda raw: events.append("safety_preflight") or dict(_CLEAN),
    )
    monkeypatch.setattr(
        _PROBE,
        "_public_clean_safety",
        lambda raw, stage: events.append(stage) or dict(_CLEAN),
    )

    def run_rocm(output, clean_callable, counters, *, environment):
        assert clean_callable is require_clean
        events.append("runtime")
        return 0

    monkeypatch.setattr(_PROBE, "_run_rocm", run_rocm)
    output = io.StringIO()
    assert (
        _PROBE._execute(SimpleNamespace(platform="rocm", allow_gpu=True), output) == 0
    )

    assert events.index("source_bound") < events.index("safety_loaded")
    assert events.index("safety_loaded") < events.index("safety_validated")
    assert events.index("safety_validated") < events.index("environment_mutated")
    assert events.index("environment_mutated") < events.index("guard_entered")
    records = _records(output)
    binding_index = next(
        index
        for index, record in enumerate(records)
        if record["record_type"] == "safety_callable_binding_proof"
    )
    environment_index = next(
        index
        for index, record in enumerate(records)
        if record["record_type"] == "environment"
    )
    assert binding_index < environment_index


def test_ast_proves_lazy_import_explicit_scale_and_exact_one_shot_source_paths():
    source = _PROBE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    roots = {
        alias.name.partition(".")[0]
        for node in module.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert {"jax", "jaxlib", "numpy", "ml_dtypes", "skyrl"}.isdisjoint(roots)

    compile_tree = ast.parse(inspect.getsource(_PROBE._compile_checked_chunk))
    calls = [node for node in ast.walk(compile_tree) if isinstance(node, ast.Call)]
    lowers = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute) and node.func.attr == "lower"
    ]
    compiles = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "compile"
        and ast.unparse(node.func.value) == "lowered"
    ]
    api_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "query_bounded_gqa_forward_chunk"
    ]
    assert len(lowers) == len(compiles) == len(api_calls) == 1
    keywords = {item.arg: ast.unparse(item.value) for item in api_calls[0].keywords}
    assert keywords["query_start"] == "_QUERY_START"
    assert keywords["scale"] == "_ATTENTION_SCALE"
    assert keywords["interpret"] == "False"

    run_source = inspect.getsource(_PROBE._run_rocm)
    assert run_source.index("_prove_command_buffers_disabled") < run_source.index(
        "import jax"
    )
    assert run_source.count("_device_put_inputs(") == 1
    assert run_source.count("_dispatch_candidate(") == 1
    assert run_source.count("_device_get_candidate(") == 1
    assert all(
        token not in source
        for token in ("jax.vjp(", "jax.grad(", "device_error_reduction" + "(")
    )
    execute_source = inspect.getsource(_PROBE._execute)
    assert execute_source.index("_load_safety_helpers") < execute_source.index(
        "_configure_rocm_environment"
    )


def test_default_subprocess_refuses_without_importing_jax():
    program = f"""
import contextlib, importlib.util, io, json, sys
before=set(sys.modules)
spec=importlib.util.spec_from_file_location('isolated_nonzero_scale_probe',{str(_PROBE_PATH)!r})
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
