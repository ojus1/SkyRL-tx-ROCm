from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import io
import json
import os
import stat
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ml_dtypes
import numpy as np
import pytest

_REPO = Path(__file__).parents[2]
_PROBE_PATH = _REPO / "rocm" / "probe_query_bounded_gqa_chunk_runtime.py"
_SPEC = importlib.util.spec_from_file_location(
    "probe_query_bounded_gqa_chunk_runtime_test", _PROBE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)

_TARGET = "__gpu$xla.gpu.triton"
_MARKER = "query_bounded_gqa_forward_q256"
_DISABLED_COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="
_CLEAN = {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}
_HEADLESS = {
    **_CLEAN,
    "amd_cards": ["card1"],
    "connected_amd_connectors": [],
    "kfd_path": "/dev/kfd",
    "kfd_accessible": True,
    "kfd_unowned": True,
}


def _accelerator_modules() -> set[str]:
    return {
        name
        for name in sys.modules
        if name in {"jax", "jaxlib", "skyrl.tx.kernels.query_bounded_gqa"}
        or name.startswith("jax.")
        or name.startswith("jaxlib.")
    }


def _records(output: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in output.getvalue().splitlines()]


def test_default_refuses_without_new_jax_import_and_hashes_all_sources():
    before = _accelerator_modules()
    output = io.StringIO()

    result = _PROBE._execute(
        SimpleNamespace(platform="abstract", allow_gpu=False), output
    )

    assert result == 0
    assert _accelerator_modules() == before
    manifest, refused = _records(output)
    assert manifest["scope"] == "abstract_refusal"
    assert manifest["compile_may_dispatch_gpu_work"] is False
    assert manifest["fresh_process_required"] is True
    assert manifest["raw_ir_emitted"] is False
    assert manifest["outer_profile_rocm_supervision_required"] is True
    assert (
        manifest["outer_profile_rocm_supervision_operational_not_internally_proven"]
        is True
    )
    assert (
        manifest["zero_qk_analytic_scope_does_not_validate_qk_scale_or_general_forward"]
        is True
    )
    assert manifest["counters"] == _PROBE._zero_counters()
    for name, path in _PROBE._source_files().items():
        assert manifest[name] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert refused["jax_imported"] is False


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--platform", "rocm"), "requires the explicit --allow-gpu"),
        (
            ("--platform", "rocm", "--allow-gpu"),
            "requires --output for a private JSONL artifact",
        ),
        (("--allow-gpu",), "only valid with --platform rocm"),
        (("--query-start", "0"), "unrecognized arguments"),
        (("--query-size", "512"), "unrecognized arguments"),
        (("--replay",), "unrecognized arguments"),
        (("--backward",), "unrecognized arguments"),
        (("--gpu-reference",), "unrecognized arguments"),
    ],
)
def test_parser_rejects_implicit_gpu_and_scope_broadening(arguments, message, capsys):
    with pytest.raises(SystemExit) as raised:
        _PROBE._parse_args(list(arguments))

    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_private_output_is_exclusive_mode_0600(tmp_path):
    path = tmp_path / "chunk-runtime.jsonl"

    assert _PROBE.main(["--output", str(path)]) == 0
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [
        json.loads(line)["record_type"] for line in path.read_text().splitlines()
    ] == [
        "manifest",
        "refused",
    ]
    with pytest.raises(SystemExit):
        _PROBE._parse_args(["--output", str(path)])


def test_contract_is_exact_last_chunk_single_candidate_only():
    contract = _PROBE._exact_contract()

    assert contract["operation"] == "query_bounded_gqa_forward_chunk_analytic_runtime"
    assert [item["shape"] for item in contract["inputs"]] == [
        [1, 256, 16, 256],
        [1, 512, 4, 256],
        [1, 512, 4, 256],
        [1, 512],
    ]
    assert contract["query_start"] == 256
    assert contract["query_stop"] == 512
    assert contract["global_query_positions"] == [256, 511]
    assert contract["tiles"] == {"block_q": 64, "block_k": 64}
    assert contract["compile_gate"] == {
        "promoted_parser_delegated": True,
        "dialects_independently_required": ["stablehlo", "optimized_hlo"],
        "exact_argument_bytes": 4_196_352,
        "exact_output_bytes": 2_097_152,
        "maximum_temporary_bytes": 64 * 1024**2,
    }
    assert contract["dispatch_plan"] == {
        "lower_calls": 1,
        "compile_calls": 1,
        "candidate_invocations": 1,
        "device_put_calls": 1,
        "device_get_calls": 1,
        "replay_invocations": 0,
        "backward_invocations": 0,
        "gpu_reference_invocations": 0,
        "device_error_reduction_invocations": 0,
        "model_invocations": 0,
    }
    assert contract["numerical_gate"]["candidate_seconds_strictly_below"] == 0.1
    assert (
        contract["numerical_gate"]["promotion_candidate_seconds_strictly_below"]
        == 0.075
    )
    assert contract["outer_supervision"] == {
        "profile_rocm_required": True,
        "operational_dependency": True,
        "internally_proven_by_child": False,
    }
    assert contract["analytic_scope"] == {
        "q_and_k_are_zero": True,
        "validates_global_causal_offset_and_kv_head_mapping": True,
        "validates_nonconstant_v_accumulation": True,
        "validates_nonzero_qk_logits": False,
        "validates_attention_scale": False,
        "validates_general_forward_inputs": False,
    }


def test_promoted_compile_contract_is_exact():
    proof = _PROBE._assert_promoted_compile_contract()

    assert {
        key: proof[key]
        for key in (
            "passed",
            "query_size",
            "sequence_length",
            "query_start",
            "block_q",
            "block_k",
            "argument_bytes",
            "output_bytes",
            "temp_bytes",
        )
    } == {
        "passed": True,
        "query_size": 256,
        "sequence_length": 512,
        "query_start": 256,
        "block_q": 64,
        "block_k": 64,
        "argument_bytes": 4_196_352,
        "output_bytes": 2_097_152,
        "temp_bytes": 64 * 1024**2,
    }
    assert proof["resolved_file_matches_expected"] is True
    assert proof["source_sha256"] == _PROBE._PROMOTED_COMPILE_SOURCE_SHA256
    assert proof["exact_marker"] == _MARKER
    assert proof["exact_target"] == _TARGET
    assert proof["exact_ir_check_names"] == sorted(_PROBE._EXPECTED_IR_CHECK_NAMES)


def test_promoted_compile_contract_binds_loaded_path_and_source_hash(monkeypatch):
    expected_file = _PROBE._source_files()[
        "promoted_chunk_compile_probe_source_sha256"
    ].resolve()
    loaded = _PROBE._compile_probe()

    assert Path(loaded.__file__).resolve() == expected_file
    assert _PROBE._file_sha256(expected_file) == _PROBE._PROMOTED_COMPILE_SOURCE_SHA256

    monkeypatch.setattr(_PROBE, "_file_sha256", lambda _path: "0" * 64)
    with pytest.raises(RuntimeError, match="source hash"):
        _PROBE._assert_promoted_compile_contract()

    monkeypatch.setattr(
        _PROBE,
        "_compile_probe",
        lambda: SimpleNamespace(__file__="/tmp/not-the-promoted-helper.py"),
    )
    with pytest.raises(RuntimeError, match="exact repository file"):
        _PROBE._assert_promoted_compile_contract()


def test_host_oracle_uses_global_offset_and_exact_query_to_kv_mapping():
    inputs, manifests, expected, oracle = _PROBE._construct_host_inputs(np, ml_dtypes)
    q, k, v, mask = inputs

    assert q.shape == (1, 256, 16, 256)
    assert k.shape == v.shape == (1, 512, 4, 256)
    assert mask.shape == (1, 512)
    assert str(q.dtype) == str(k.dtype) == str(v.dtype) == "bfloat16"
    assert str(mask.dtype) == "int32"
    assert np.count_nonzero(q) == 0
    assert np.count_nonzero(k) == 0
    assert np.all(mask == 1)
    assert not np.array_equal(v[:, :, 0], v[:, :, 1])
    assert not np.array_equal(v[:, 1:], v[:, :-1])
    assert not np.array_equal(v[:, :, :, 1:], v[:, :, :, :-1])
    assert float(np.min(np.asarray(v, dtype=np.float32))) <= -7.0
    assert float(np.max(np.asarray(v, dtype=np.float32))) >= 7.0
    assert not np.array_equal(
        np.diff(np.asarray(v[0, :, 0, 0], dtype=np.float32), n=2),
        np.zeros(510, dtype=np.float32),
    )
    assert expected.shape == q.shape
    v_fp32 = np.asarray(v, dtype=np.float32)
    for query_head in (0, 3, 4, 7, 8, 11, 12, 15):
        kv_head = query_head // 4
        np.testing.assert_array_equal(
            expected[0, 0, query_head],
            np.cumsum(v_fp32[0, :257, kv_head], axis=0, dtype=np.float32)[-1]
            / np.float32(257),
        )
        np.testing.assert_array_equal(
            expected[0, -1, query_head],
            np.cumsum(v_fp32[0, :512, kv_head], axis=0, dtype=np.float32)[-1]
            / np.float32(512),
        )
    assert not np.array_equal(expected[0, 0, 0], v_fp32[0, 0, 0])
    assert [item["name"] for item in manifests] == ["q_chunk", "k", "v", "key_mask"]
    assert oracle["shape"] == [1, 256, 16, 256]
    assert oracle["dtype"] == "float32"


def test_host_oracle_rejects_exclusive_prefix_dimension_and_head_corruption():
    inputs, _manifests, expected, _oracle = _PROBE._construct_host_inputs(np, ml_dtypes)
    v_fp32 = np.asarray(inputs[2], dtype=np.float32)
    exclusive_global_prefix = (
        np.cumsum(v_fp32, axis=1, dtype=np.float32)[
            :, _PROBE._QUERY_START - 1 : _PROBE._QUERY_STOP - 1
        ]
        / np.arange(
            _PROBE._QUERY_START,
            _PROBE._QUERY_STOP,
            dtype=np.float32,
        )[None, :, None, None]
    )
    query_to_kv = np.arange(_PROBE._QUERY_HEADS, dtype=np.int32) // _PROBE._GROUP_SIZE
    candidates = (
        exclusive_global_prefix[:, :, query_to_kv, :].astype(ml_dtypes.bfloat16),
        np.roll(expected, 1, axis=-1).astype(ml_dtypes.bfloat16),
        np.roll(expected, _PROBE._GROUP_SIZE, axis=2).astype(ml_dtypes.bfloat16),
    )

    for candidate in candidates:
        output = io.StringIO()
        with pytest.raises(RuntimeError, match="host numerical"):
            _PROBE._validate_candidate(
                np,
                candidate,
                expected,
                0.001,
                _PROBE._completed_counters(),
                output,
            )
        validation = _records(output)[-1]
        assert validation["record_type"] == "host_validation"
        assert validation["gates"]["numerical_passed"] is False
        assert validation["gates"]["promotion_passed"] is False


def _stablehlo(marker=_MARKER, target=_TARGET):
    return "\n".join(
        [
            f'#loc0 = loc("{marker} query_start=256 query_size=256")',
            "module {",
            f'  %0 = stablehlo.custom_call @"{target}"() : () -> tensor<1xbf16> loc(#loc0)',
            "}",
        ]
    )


def _optimized_hlo(marker=_MARKER, target=_TARGET):
    return "\n".join(
        [
            "ENTRY main {",
            f'  ROOT %0 = bf16[1] custom-call(), custom_call_target="{target}", '
            f'op_name="{marker} query_start=256 query_size=256"',
            "}",
        ]
    )


class _FakeShapeDtypeStruct:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _FakeCompiled:
    def __init__(self, state, *, optimized=None, memory=None, result=None):
        self.state = state
        self.optimized = _optimized_hlo() if optimized is None else optimized
        self.memory = (
            SimpleNamespace(
                argument_size_in_bytes=4_196_352,
                output_size_in_bytes=2_097_152,
                alias_size_in_bytes=0,
                temp_size_in_bytes=33_024,
            )
            if memory is None
            else memory
        )
        self.result = result

    def __call__(self, *arguments):
        self.state["compiled_invocations"] += 1
        self.state["compiled_arguments"] = arguments
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    def as_text(self):
        self.state["as_text_calls"] += 1
        return self.optimized

    def memory_analysis(self):
        self.state["memory_calls"] += 1
        return self.memory


class _FakeLowered:
    def __init__(self, state, compiled, *, stable=None, compile_error=None):
        self.state = state
        self.compiled = compiled
        self.stable = _stablehlo() if stable is None else stable
        self.compile_error = compile_error

    def compiler_ir(self, *, dialect):
        assert dialect == "stablehlo"
        self.state["compiler_ir_calls"] += 1
        return self.stable

    def compile(self):
        self.state["compile_calls"] += 1
        if self.compile_error is not None:
            raise self.compile_error
        return self.compiled


class _FakeJitted:
    def __init__(self, function, state, lowered, lower_error=None):
        self.function = function
        self.state = state
        self.lowered = lowered
        self.lower_error = lower_error

    def lower(self, *signature):
        self.state["lower_calls"] += 1
        self.state["signature"] = signature
        if self.lower_error is not None:
            raise self.lower_error
        self.function(*signature)
        return self.lowered


class _FakeJax:
    __version__ = "PRIVATE_JAX_VERSION"
    ShapeDtypeStruct = _FakeShapeDtypeStruct

    def __init__(self, state, lowered, *, lower_error=None):
        self.state = state
        self.lowered = lowered
        self.lower_error = lower_error
        self.device = SimpleNamespace(platform="gpu")

    def jit(self, function):
        self.state["jit_calls"] += 1
        return _FakeJitted(
            function, self.state, self.lowered, lower_error=self.lower_error
        )

    def default_backend(self):
        return "gpu"

    def devices(self):
        return [self.device]

    def device_put(self, value):
        self.state["device_put_calls"] += 1
        return value

    def block_until_ready(self, value):
        self.state["block_calls"] += 1
        return value

    def device_get(self, value):
        self.state["device_get_calls"] += 1
        if isinstance(value, _DeviceGetError):
            raise value.error
        return value


class _DeviceGetError:
    def __init__(self, error):
        self.error = error


def _state():
    return {
        "jit_calls": 0,
        "lower_calls": 0,
        "compiler_ir_calls": 0,
        "compile_calls": 0,
        "as_text_calls": 0,
        "memory_calls": 0,
        "compiled_invocations": 0,
        "device_put_calls": 0,
        "block_calls": 0,
        "device_get_calls": 0,
        "api_calls": [],
    }


def _api(state):
    def operation(*arguments, **keywords):
        state["api_calls"].append((arguments, keywords))
        return "abstract-output"

    return operation


def _clean():
    return dict(_CLEAN)


def test_compile_once_exact_api_creates_capability_without_invocation():
    state = _state()
    compiled = _FakeCompiled(state)
    lowered = _FakeLowered(state, compiled)
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    checked, report = _PROBE._compile_checked_chunk(
        _FakeJax(state, lowered),
        SimpleNamespace(bfloat16="bf16", int32="i32"),
        _api(state),
        _clean,
        counters,
        output,
    )

    assert type(checked) is _PROBE._CheckedChunkExecutable
    assert state["jit_calls"] == state["lower_calls"] == state["compile_calls"] == 1
    assert (
        state["compiler_ir_calls"]
        == state["as_text_calls"]
        == state["memory_calls"]
        == 1
    )
    assert state["compiled_invocations"] == 0
    assert [item.shape for item in state["signature"]] == [
        (1, 256, 16, 256),
        (1, 512, 4, 256),
        (1, 512, 4, 256),
        (1, 512),
    ]
    _, keywords = state["api_calls"][0]
    assert keywords == {
        "query_start": 256,
        "block_q": 64,
        "block_k": 64,
        "interpret": False,
    }
    assert report["release_gate"]["passed"] is True
    assert counters == {
        **_PROBE._zero_counters(),
        "lower_attempts": 1,
        "lower_completions": 1,
        "compile_attempts": 1,
        "compile_completions": 1,
    }
    assert [record["record_type"] for record in _records(output)] == [
        "stage",
        "lowered",
        "journal_checkpoint",
        "stage",
        "chunk_compiled",
        "journal_checkpoint",
        "compile_gate_passed",
    ]


def test_lower_exception_is_journaled_without_compile_or_invocation():
    state = _state()
    compiled = _FakeCompiled(state)
    lowered = _FakeLowered(state, compiled)
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="synthetic lower failure"):
        _PROBE._compile_checked_chunk(
            _FakeJax(
                state,
                lowered,
                lower_error=RuntimeError("synthetic lower failure"),
            ),
            SimpleNamespace(bfloat16="bf16", int32="i32"),
            _api(state),
            _clean,
            counters,
            output,
        )

    assert state["lower_calls"] == 1
    assert state["compile_calls"] == state["compiled_invocations"] == 0
    assert counters["lower_attempts"] == 1
    assert counters["lower_completions"] == 0
    records = _records(output)
    assert [record["record_type"] for record in records] == [
        "stage",
        "journal_checkpoint",
    ]
    assert records[-1]["stage"] == "after_chunk_lower_attempt"


def test_compile_exception_is_journaled_without_invocation():
    state = _state()
    compiled = _FakeCompiled(state)
    lowered = _FakeLowered(
        state, compiled, compile_error=RuntimeError("synthetic compile failure")
    )
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="synthetic compile failure"):
        _PROBE._compile_checked_chunk(
            _FakeJax(state, lowered),
            SimpleNamespace(bfloat16="bf16", int32="i32"),
            _api(state),
            _clean,
            counters,
            output,
        )

    assert state["lower_calls"] == state["compile_calls"] == 1
    assert state["compiled_invocations"] == 0
    assert counters["compile_attempts"] == 1
    assert counters["compile_completions"] == 0
    records = _records(output)
    assert [
        record["stage"]
        for record in records
        if record["record_type"] == "journal_checkpoint"
    ] == ["after_chunk_lower_attempt", "after_chunk_compile_attempt"]


@pytest.mark.parametrize(
    ("stable", "optimized", "memory"),
    [
        (_stablehlo(marker="query_bounded_gqa_forward_q0"), _optimized_hlo(), None),
        (_stablehlo(), _optimized_hlo(target="triton"), None),
        (
            _stablehlo(),
            _optimized_hlo(),
            SimpleNamespace(
                argument_size_in_bytes=4_196_351,
                output_size_in_bytes=2_097_152,
                temp_size_in_bytes=0,
            ),
        ),
    ],
)
def test_structural_or_exact_memory_failure_never_wraps_or_invokes(
    stable, optimized, memory
):
    state = _state()
    compiled = _FakeCompiled(state, optimized=optimized, memory=memory)
    lowered = _FakeLowered(state, compiled, stable=stable)

    with pytest.raises(RuntimeError, match="structural or exact memory gate"):
        _PROBE._compile_checked_chunk(
            _FakeJax(state, lowered),
            SimpleNamespace(bfloat16="bf16", int32="i32"),
            _api(state),
            _clean,
            _PROBE._zero_counters(),
            io.StringIO(),
        )

    assert state["compiled_invocations"] == 0


def test_checked_capability_is_private_and_one_shot():
    state = _state()
    compiled = _FakeCompiled(state, result="result")
    counters = _PROBE._zero_counters()
    proof = {"passed": True}

    with pytest.raises(RuntimeError, match="without both passed"):
        _PROBE._CheckedChunkExecutable(
            compiled,
            proof=proof,
            counters=counters,
            token=object(),
        )
    checked = _PROBE._wrap_checked(compiled, proof, counters)
    assert (
        checked.invoke(
            SimpleNamespace(block_until_ready=lambda value: value), (), lambda: None
        )
        == "result"
    )
    with pytest.raises(RuntimeError, match="already consumed"):
        checked.invoke(
            SimpleNamespace(block_until_ready=lambda value: value), (), lambda: None
        )
    assert state["compiled_invocations"] == 1
    assert counters["candidate_attempts"] == counters["candidate_completions"] == 1


def test_host_metrics_are_complete_and_validation_fails_closed():
    expected = np.ones((1, 256, 16, 256), dtype=np.float32)
    actual = expected.astype(ml_dtypes.bfloat16)
    metrics = _PROBE._host_metrics(np, actual, expected)

    assert set(metrics) == {
        "finite",
        "shape_dtype_nbytes_exact",
        "actual_shape",
        "actual_dtype",
        "reference_shape",
        "reference_dtype",
        "actual_nbytes",
        "reference_nbytes",
        "max_abs",
        "mean_abs",
        "relative_l2",
        "cosine",
        "actual_sha256",
        "reference_sha256",
    }
    bad = np.zeros((1, 256, 16, 256), dtype=ml_dtypes.bfloat16)
    reference = np.ones((1, 256, 16, 256), dtype=np.float32)
    with pytest.raises(RuntimeError, match="host numerical"):
        _PROBE._validate_candidate(
            np,
            bad,
            reference,
            0.001,
            _PROBE._completed_counters(),
            io.StringIO(),
        )


def test_safety_duration_between_75_and_100ms_is_not_promoted():
    expected = np.ones((1, 256, 16, 256), dtype=np.float32)
    actual = expected.astype(ml_dtypes.bfloat16)
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="promotion-duration"):
        _PROBE._validate_candidate(
            np,
            actual,
            expected,
            0.08,
            _PROBE._completed_counters(),
            output,
        )

    records = _records(output)
    assert [record["record_type"] for record in records] == ["host_validation"]
    validation = records[0]
    assert validation["status"] == "not_promoted"
    assert validation["gates"] == {
        "numerical_passed": True,
        "safety_duration_passed": True,
        "promotion_duration_passed": False,
        "promotion_passed": False,
    }
    assert validation["thresholds"] == {
        "finite_required": True,
        "relative_l2_strictly_below": 0.01,
        "minimum_cosine": 0.9999,
        "maximum_absolute_error": 0.02,
        "candidate_seconds_strictly_below": 0.1,
        "promotion_candidate_seconds_strictly_below": 0.075,
    }
    assert not any(record["record_type"] == "runtime_passed" for record in records)


def _valid_environment(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", _DISABLED_COMMAND_BUFFER_FLAG)
    return {"XLA_FLAGS_effective": _DISABLED_COMMAND_BUFFER_FLAG}


def test_runtime_does_not_emit_pass_for_80ms_promotion_failure(monkeypatch):
    state = _state()
    _inputs, _manifests, expected, _oracle = _PROBE._construct_host_inputs(
        np, ml_dtypes
    )
    compiled = _FakeCompiled(state, result=expected.astype(ml_dtypes.bfloat16))
    lowered = _FakeLowered(state, compiled)
    jax = _FakeJax(state, lowered)
    dependencies = (
        jax,
        SimpleNamespace(bfloat16="bf16", int32="i32"),
        SimpleNamespace(__version__="test"),
        SimpleNamespace(
            get_backend=lambda: SimpleNamespace(platform_version="ROCm test")
        ),
        np,
        ml_dtypes,
        _api(state),
    )
    original_dispatch = _PROBE._dispatch_candidate

    def promotion_slow_dispatch(*args, **kwargs):
        actual, _measured_seconds = original_dispatch(*args, **kwargs)
        return actual, 0.08

    monkeypatch.setattr(_PROBE, "_dispatch_candidate", promotion_slow_dispatch)
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="promotion-duration"):
        _PROBE._run_rocm(
            output,
            _clean,
            _PROBE._zero_counters(),
            environment=_valid_environment(monkeypatch),
            _dependencies=dependencies,
        )

    records = _records(output)
    validation = next(
        record for record in records if record["record_type"] == "host_validation"
    )
    assert validation["status"] == "not_promoted"
    assert validation["gates"]["safety_duration_passed"] is True
    assert validation["gates"]["promotion_duration_passed"] is False
    assert validation["gates"]["promotion_passed"] is False
    assert not any(record["record_type"] == "runtime_passed" for record in records)


def test_host_construction_exception_is_journaled_without_dispatch(monkeypatch):
    state = _state()
    compiled = _FakeCompiled(state)
    lowered = _FakeLowered(state, compiled)
    jax = _FakeJax(state, lowered)
    dependencies = (
        jax,
        SimpleNamespace(bfloat16="bf16", int32="i32"),
        SimpleNamespace(__version__="test"),
        SimpleNamespace(
            get_backend=lambda: SimpleNamespace(platform_version="ROCm test")
        ),
        np,
        ml_dtypes,
        _api(state),
    )
    counters = _PROBE._zero_counters()
    output = io.StringIO()
    monkeypatch.setattr(
        _PROBE,
        "_construct_host_inputs",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeError("synthetic host construction failure")
        ),
    )

    with pytest.raises(RuntimeError, match="synthetic host construction failure"):
        _PROBE._run_rocm(
            output,
            _clean,
            counters,
            environment=_valid_environment(monkeypatch),
            _dependencies=dependencies,
        )

    assert state["compiled_invocations"] == 0
    assert state["device_put_calls"] == state["device_get_calls"] == 0
    records = _records(output)
    assert any(
        record.get("stage") == "after_host_reference_construction" for record in records
    )
    assert not any(
        record["record_type"]
        in {
            "host_analytic_reference",
            "dispatch_started",
            "host_validation",
            "runtime_passed",
        }
        for record in records
    )


def test_mocked_runtime_exact_order_one_candidate_and_global_oracle(monkeypatch):
    state = _state()
    _inputs, _manifests, expected, _oracle = _PROBE._construct_host_inputs(
        np, ml_dtypes
    )
    actual = expected.astype(ml_dtypes.bfloat16)
    compiled = _FakeCompiled(state, result=actual)
    lowered = _FakeLowered(state, compiled)
    jax = _FakeJax(state, lowered)
    dependencies = (
        jax,
        SimpleNamespace(bfloat16="bf16", int32="i32"),
        SimpleNamespace(__version__="PRIVATE_JAXLIB_VERSION"),
        SimpleNamespace(
            get_backend=lambda: SimpleNamespace(platform_version="ROCm PRIVATE PATH")
        ),
        np,
        ml_dtypes,
        _api(state),
    )
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    result = _PROBE._run_rocm(
        output,
        _clean,
        counters,
        environment=_valid_environment(monkeypatch),
        _dependencies=dependencies,
    )

    assert result == 0
    assert state["compiled_invocations"] == 1
    assert state["device_put_calls"] == state["device_get_calls"] == 1
    assert counters == _PROBE._completed_counters()
    records = _records(output)
    assert [record["record_type"] for record in records] == [
        "command_buffer_environment_proof",
        "backend_ready",
        "journal_checkpoint",
        "stage",
        "lowered",
        "journal_checkpoint",
        "stage",
        "chunk_compiled",
        "journal_checkpoint",
        "compile_gate_passed",
        "host_analytic_reference",
        "journal_checkpoint",
        "journal_checkpoint",
        "dispatch_started",
        "journal_checkpoint",
        "dispatch",
        "journal_checkpoint",
        "host_validation",
        "journal_checkpoint",
        "runtime_passed",
    ]
    assert [
        record["stage"]
        for record in records
        if record["record_type"] == "journal_checkpoint"
    ] == list(_PROBE._JOURNAL_STAGES)
    started = next(
        record for record in records if record["record_type"] == "dispatch_started"
    )
    assert started["counters"]["candidate_attempts"] == 1
    assert started["counters"]["candidate_completions"] == 0
    validation = next(
        record for record in records if record["record_type"] == "host_validation"
    )
    assert validation["status"] == "passed"
    assert validation["global_query_positions"] == [256, 511]
    assert validation["gates"] == {
        "numerical_passed": True,
        "safety_duration_passed": True,
        "promotion_duration_passed": True,
        "promotion_passed": True,
    }
    artifact = output.getvalue()
    assert "PRIVATE_JAX_VERSION" not in artifact
    assert "PRIVATE_JAXLIB_VERSION" not in artifact
    assert "ROCm PRIVATE PATH" not in artifact


def test_candidate_failure_is_single_attempt_journaled_and_has_no_validation(
    monkeypatch,
):
    state = _state()
    compiled = _FakeCompiled(state, result=RuntimeError("synthetic candidate failure"))
    lowered = _FakeLowered(state, compiled)
    jax = _FakeJax(state, lowered)
    dependencies = (
        jax,
        SimpleNamespace(bfloat16="bf16", int32="i32"),
        SimpleNamespace(__version__="test"),
        SimpleNamespace(
            get_backend=lambda: SimpleNamespace(platform_version="ROCm test")
        ),
        np,
        ml_dtypes,
        _api(state),
    )
    output = io.StringIO()
    counters = _PROBE._zero_counters()

    with pytest.raises(RuntimeError, match="synthetic candidate failure"):
        _PROBE._run_rocm(
            output,
            _clean,
            counters,
            environment=_valid_environment(monkeypatch),
            _dependencies=dependencies,
        )

    assert state["compiled_invocations"] == 1
    assert counters["candidate_attempts"] == 1
    assert counters["candidate_completions"] == 0
    records = _records(output)
    assert any(
        record.get("stage") == "after_candidate_dispatch_attempt" for record in records
    )
    assert not any(
        record["record_type"] in {"dispatch", "host_validation", "runtime_passed"}
        for record in records
    )


def test_device_get_failure_is_journaled_and_has_no_host_validation(monkeypatch):
    state = _state()
    compiled = _FakeCompiled(
        state, result=_DeviceGetError(RuntimeError("synthetic device_get failure"))
    )
    lowered = _FakeLowered(state, compiled)
    jax = _FakeJax(state, lowered)
    dependencies = (
        jax,
        SimpleNamespace(bfloat16="bf16", int32="i32"),
        SimpleNamespace(__version__="test"),
        SimpleNamespace(
            get_backend=lambda: SimpleNamespace(platform_version="ROCm test")
        ),
        np,
        ml_dtypes,
        _api(state),
    )
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="synthetic device_get failure"):
        _PROBE._run_rocm(
            output,
            _clean,
            _PROBE._zero_counters(),
            environment=_valid_environment(monkeypatch),
            _dependencies=dependencies,
        )

    records = _records(output)
    assert any(
        record.get("stage") == "after_candidate_device_get_attempt"
        for record in records
    )
    assert not any(
        record["record_type"] in {"host_validation", "runtime_passed"}
        for record in records
    )


def test_command_buffer_failure_precedes_backend_and_compile(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_command_buffer=true")
    state = _state()
    compiled = _FakeCompiled(state)
    lowered = _FakeLowered(state, compiled)
    jax = _FakeJax(state, lowered)
    jax.default_backend = lambda: (_ for _ in ()).throw(AssertionError("backend used"))

    with pytest.raises(RuntimeError, match="does not match"):
        _PROBE._run_rocm(
            io.StringIO(),
            _clean,
            _PROBE._zero_counters(),
            environment={"XLA_FLAGS_effective": _DISABLED_COMMAND_BUFFER_FLAG},
            _dependencies=(
                jax,
                SimpleNamespace(bfloat16="bf16", int32="i32"),
                SimpleNamespace(__version__="test"),
                SimpleNamespace(get_backend=lambda: None),
                np,
                ml_dtypes,
                _api(state),
            ),
        )

    assert (
        state["lower_calls"]
        == state["compile_calls"]
        == state["compiled_invocations"]
        == 0
    )


def test_preflight_is_headless_kfd_and_terminal_error_is_digest_only(monkeypatch):
    assert _PROBE._public_safety_preflight(dict(_HEADLESS)) == _HEADLESS
    for mutation in (
        {"amd_cards": []},
        {"amd_cards": ["../card1"]},
        {"connected_amd_connectors": ["card1-DP-1"]},
        {"kfd_path": "/private/kfd"},
        {"kfd_accessible": False},
        {"kfd_unowned": False},
    ):
        with pytest.raises(RuntimeError, match="safety_preflight"):
            _PROBE._public_safety_preflight({**_HEADLESS, **mutation})

    secret = f"PRIVATE_RUNTIME_SECRET {_REPO}/cache"

    @contextmanager
    def guard():
        yield dict(_HEADLESS)

    monkeypatch.setattr(_PROBE, "_assert_fresh_accelerator_process", lambda: None)
    monkeypatch.setattr(_PROBE, "_configure_rocm_environment", lambda: {})
    monkeypatch.setattr(_PROBE, "_environment_manifest", lambda _environment: {})
    monkeypatch.setattr(_PROBE, "_load_safety_helpers", lambda: (guard, _clean))
    monkeypatch.setattr(
        _PROBE,
        "_run_rocm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    output = io.StringIO()

    assert (
        _PROBE._execute(SimpleNamespace(platform="rocm", allow_gpu=True), output) == 1
    )
    artifact = output.getvalue()
    records = _records(output)
    assert [record["record_type"] for record in records[-2:]] == [
        "safety_postflight",
        "error",
    ]
    error = records[-1]
    assert error["message_redacted"] is True
    assert error["message_sha256"] == hashlib.sha256(secret.encode()).hexdigest()
    assert secret not in artifact
    assert str(_REPO) not in artifact


def test_fresh_process_guard_precedes_environment(monkeypatch):
    state = {"environment": False}
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace())
    monkeypatch.setattr(
        _PROBE,
        "_configure_rocm_environment",
        lambda: state.__setitem__("environment", True),
    )
    output = io.StringIO()

    assert (
        _PROBE._execute(SimpleNamespace(platform="rocm", allow_gpu=True), output) == 1
    )
    assert state["environment"] is False
    assert _records(output)[-1]["stage"] == "fresh_process_preflight"


def test_ast_proves_no_eager_accelerator_import_and_exact_one_shot_paths():
    source = _PROBE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    imports = [
        node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    roots = {alias.name.partition(".")[0] for node in imports for alias in node.names}
    assert {"jax", "jaxlib", "numpy", "ml_dtypes", "skyrl"}.isdisjoint(roots)
    calls = [node for node in ast.walk(module) if isinstance(node, ast.Call)]
    attributes = [
        node.func.attr for node in calls if isinstance(node.func, ast.Attribute)
    ]
    executable_lowers = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "lower"
        and "jax.jit(forward_chunk)" in ast.unparse(node.func.value)
    ]
    assert len(executable_lowers) == 1
    assert attributes.count("compile") == 1
    assert attributes.count("device_put") == 1
    assert attributes.count("device_get") == 1
    assert attributes.count("invoke") == 1
    assert attributes.count("vjp") == 0
    assert attributes.count("grad") == 0
    run_source = inspect.getsource(_PROBE._run_rocm)
    assert run_source.index("_prove_command_buffers_disabled") < run_source.index(
        "import jax"
    )
    compile_source = inspect.getsource(_PROBE._compile_checked_chunk)
    assert compile_source.index("_structural_gate") < compile_source.index(
        "_wrap_checked"
    )
    assert compile_source.index("_compiled_memory_gate") < compile_source.index(
        "_wrap_checked"
    )
    dispatch_source = inspect.getsource(_PROBE._dispatch_candidate)
    assert dispatch_source.count("executable.invoke(") == 1


def test_default_subprocess_refuses_without_importing_jax():
    program = f"""
import contextlib, importlib.util, io, json, sys
before=set(sys.modules)
spec=importlib.util.spec_from_file_location('isolated_chunk_runtime',{str(_PROBE_PATH)!r})
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
