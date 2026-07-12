from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import io
import json
import math
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO = Path(__file__).parents[2]
_PROBE_PATH = _REPO / "rocm" / "probe_query_bounded_gqa_iid.py"
_SPEC = importlib.util.spec_from_file_location("probe_query_bounded_gqa_iid_test", _PROBE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)

_CLEAN_SAFETY = {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}
_HEADLESS_SAFETY = {
    **_CLEAN_SAFETY,
    "amd_cards": ["card1"],
    "connected_amd_connectors": [],
    "kfd_path": "/dev/kfd",
    "kfd_accessible": True,
    "kfd_unowned": True,
}
_DISABLED_COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="


def _jax_modules() -> set[str]:
    return {
        name
        for name in sys.modules
        if name in {"jax", "jaxlib"} or name.startswith("jax.") or name.startswith("jaxlib.")
    }


def test_default_execute_is_abstract_refusal_without_new_jax_import():
    before = _jax_modules()
    output = io.StringIO()

    result = _PROBE._execute(SimpleNamespace(platform="abstract", allow_gpu=False), output)

    assert result == 0
    assert _jax_modules() == before
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [record["record_type"] for record in records] == ["manifest", "refused"]
    manifest, refused = records
    assert manifest["scope"] == "abstract_refusal"
    assert manifest["compile_may_dispatch_gpu_work"] is False
    assert manifest["fresh_process_required"] is True
    assert manifest["counters"] == _PROBE._zero_counters()
    assert manifest["probe_source_sha256"] == hashlib.sha256(_PROBE_PATH.read_bytes()).hexdigest()
    assert (
        manifest["delegated_replay_probe_source_sha256"]
        == hashlib.sha256(Path(_PROBE._replay_probe().__file__).read_bytes()).hexdigest()
    )
    assert (
        manifest["delegated_runtime_probe_source_sha256"]
        == hashlib.sha256(Path(_PROBE._runtime_probe().__file__).read_bytes()).hexdigest()
    )
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
        (("--sequence-length", "1024"), "unrecognized arguments"),
        (("--repeats", "2"), "unrecognized arguments"),
        (("--replay",), "unrecognized arguments"),
        (("--backward",), "unrecognized arguments"),
        (("--padding",), "unrecognized arguments"),
    ],
)
def test_scope_broadening_or_unsafe_options_are_rejected(arguments, message, capsys):
    with pytest.raises(SystemExit) as raised:
        _PROBE._parse_args(list(arguments))

    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_output_is_private_and_exclusive_without_subprocess(tmp_path):
    output = tmp_path / "iid.jsonl"

    result = _PROBE.main(["--output", str(output)])

    assert result == 0
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["record_type"] for record in records] == ["manifest", "refused"]
    with pytest.raises(SystemExit):
        _PROBE._parse_args(["--output", str(output)])


def test_contract_is_exact_t512_fully_iid_single_forward_only():
    contract = _PROBE._exact_contract()

    assert contract["operation"] == ("query_bounded_gqa_forward_only_fully_iid_per_feature")
    assert contract["random_generator"] == "numpy.random.Generator(PCG64)"
    assert contract["iid_unit"] == "each individual Q/K/V feature"
    assert contract["zero_values_permitted"] is False
    assert contract["valid_length"] == 512
    assert contract["integer_grid_denominator"] == 128
    assert contract["qk_maximum_integer_magnitude"] == 48
    assert contract["v_maximum_integer_magnitude"] == 48
    assert contract["theoretical_maximum_absolute_qk_logit"] == pytest.approx(2.25)
    assert contract["dispatch_plan"] == {
        "checked_forward_invocations": 1,
        "replay_invocations": 0,
        "command_buffer_invocations": 0,
        "gpu_reference_invocations": 0,
        "device_error_reduction_invocations": 0,
        "backward_invocations": 0,
    }
    assert [entry["shape"] for entry in contract["inputs"]] == [
        [1, 512, 16, 256],
        [1, 512, 4, 256],
        [1, 512, 4, 256],
        [1, 512],
    ]
    assert contract["reference"] == {
        "location": "host_numpy_only",
        "dtype": "float32",
        "qk": "dense mapped-Hq-to-Hkv matmul",
        "softmax": "causal stable dense softmax",
        "av": "dense matmul",
    }
    assert contract["compiled_memory_gate"] == {
        "memory_analysis_required": True,
        "maximum_temporary_bytes": 64 * 1024**2,
        "maximum_argument_output_temporary_bytes": 128 * 1024**2,
    }
    assert "exact one-call" in contract["compiled_structural_gate"]


def _assert_grid(array, maximum):
    fp32 = np.asarray(array, dtype=np.float32)
    scaled = fp32 * _PROBE._GRID_DENOMINATOR
    assert np.count_nonzero(fp32) == fp32.size
    np.testing.assert_array_equal(scaled, np.rint(scaled))
    assert np.min(np.abs(scaled)) == 1
    assert np.max(np.abs(scaled)) == maximum
    assert np.all(np.isfinite(fp32))


def test_iid_per_feature_inputs_are_deterministic_nonzero_and_on_grid():
    first = _PROBE._construct_iid_inputs()
    second = _PROBE._construct_iid_inputs()
    q, k, v, mask = first

    assert q.shape == (1, 512, 16, 256)
    assert k.shape == v.shape == (1, 512, 4, 256)
    assert mask.shape == (1, 512)
    assert str(q.dtype) == str(k.dtype) == str(v.dtype) == "bfloat16"
    assert mask.dtype == np.int32
    assert np.all(mask == 1)
    _assert_grid(q, 48)
    _assert_grid(k, 48)
    _assert_grid(v, 48)
    for first_array, second_array in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_array, second_array)

    # Distinct features receive distinct PCG64 draws rather than broadcast
    # token/head scalars or shared direction vectors.
    q_fp32 = q.astype(np.float32)
    assert not np.array_equal(q_fp32[..., 0], q_fp32[..., 1])
    assert not np.array_equal(q_fp32[0, 0, 0], q_fp32[0, 1, 0])
    assert np.unique(q_fp32[0, 0, 0]).size > 64
    construction = inspect.getsource(_PROBE._iid_nonzero_grid)
    assert "size=shape" in construction
    assert "broadcast" not in construction


def test_dense_oracle_selected_row_matches_independent_scalar_reference(monkeypatch):
    monkeypatch.setattr(_PROBE, "_BATCH_SIZE", 1)
    monkeypatch.setattr(_PROBE, "_SEQUENCE_LENGTH", 7)
    monkeypatch.setattr(_PROBE, "_QUERY_HEADS", 4)
    monkeypatch.setattr(_PROBE, "_KV_HEADS", 2)
    monkeypatch.setattr(_PROBE, "_HEAD_DIM", 8)
    monkeypatch.setattr(_PROBE, "_GROUP_SIZE", 2)
    monkeypatch.setattr(_PROBE, "_QK_MAX_MAGNITUDE", 8)
    monkeypatch.setattr(_PROBE, "_V_MAX_MAGNITUDE", 12)
    q, k, v, mask = _PROBE._construct_iid_inputs()

    expected, oracle = _PROBE._dense_causal_gqa_oracle(q, k, v)

    assert expected.shape == (1, 7, 4, 8)
    assert expected.dtype == np.float32
    assert np.all(np.isfinite(expected))
    assert np.all(mask == 1)
    assert oracle["q_to_kv_mapping"] == [0, 0, 1, 1]
    assert oracle["accelerator_used"] is False

    query_position = 5
    query_head = 3
    kv_head = 1
    logits = []
    for key_position in range(query_position + 1):
        dot = 0.0
        for feature in range(8):
            dot += float(q[0, query_position, query_head, feature]) * float(k[0, key_position, kv_head, feature])
        logits.append(dot * (8**-0.5))
    row_max = max(logits)
    exponentials = [math.exp(value - row_max) for value in logits]
    denominator = sum(exponentials)
    probabilities = [value / denominator for value in exponentials]
    selected = np.asarray(
        [
            sum(
                probabilities[key_position] * float(v[0, key_position, kv_head, feature])
                for key_position in range(query_position + 1)
            )
            for feature in range(8)
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(expected[0, query_position, query_head], selected, rtol=2e-6, atol=2e-6)


def test_numpy_bf16_probability_path_predicts_finite_gate_margin(monkeypatch):
    monkeypatch.setattr(_PROBE, "_BATCH_SIZE", 1)
    monkeypatch.setattr(_PROBE, "_SEQUENCE_LENGTH", 8)
    monkeypatch.setattr(_PROBE, "_QUERY_HEADS", 4)
    monkeypatch.setattr(_PROBE, "_KV_HEADS", 2)
    monkeypatch.setattr(_PROBE, "_HEAD_DIM", 8)
    monkeypatch.setattr(_PROBE, "_GROUP_SIZE", 2)
    monkeypatch.setattr(_PROBE, "_QK_MAX_MAGNITUDE", 8)
    monkeypatch.setattr(_PROBE, "_V_MAX_MAGNITUDE", 12)
    q, k, v, _mask = _PROBE._construct_iid_inputs()
    expected, _oracle = _PROBE._dense_causal_gqa_oracle(q, k, v)

    emulated = _PROBE._emulate_bf16_probability_path(q, k, v, block_q=4, block_k=4)
    metrics = _PROBE._host_metrics(emulated, expected)

    assert emulated.shape == expected.shape
    assert str(emulated.dtype) == "bfloat16"
    assert np.all(np.isfinite(emulated.astype(np.float32)))
    assert metrics["finite"] is True
    assert math.isfinite(metrics["relative_l2"])
    assert math.isfinite(metrics["cosine"])
    assert math.isfinite(metrics["max_abs"])
    assert metrics["relative_l2"] < _PROBE._MAX_RELATIVE_L2
    assert metrics["cosine"] >= _PROBE._MIN_COSINE
    assert metrics["max_abs"] <= _PROBE._MAX_ABSOLUTE_ERROR


def test_host_manifest_labels_bf16_emulation_informational_only(monkeypatch):
    monkeypatch.setattr(_PROBE, "_BATCH_SIZE", 1)
    monkeypatch.setattr(_PROBE, "_SEQUENCE_LENGTH", 8)
    monkeypatch.setattr(_PROBE, "_QUERY_HEADS", 4)
    monkeypatch.setattr(_PROBE, "_KV_HEADS", 2)
    monkeypatch.setattr(_PROBE, "_HEAD_DIM", 8)
    monkeypatch.setattr(_PROBE, "_GROUP_SIZE", 2)
    monkeypatch.setattr(_PROBE, "_QK_MAX_MAGNITUDE", 8)
    monkeypatch.setattr(_PROBE, "_V_MAX_MAGNITUDE", 12)
    monkeypatch.setattr(
        _PROBE,
        "_emulate_bf16_probability_path",
        lambda q, _k, _v: q.copy(),
    )

    _inputs, _manifests, _expected, expected_manifest = _PROBE._construct_host_inputs()

    prediction = expected_manifest["bf16_probability_path_prediction"]
    assert prediction["authorization_effect"] == "informational_only"
    assert prediction["accelerator_used"] is False
    assert prediction["metrics_vs_full_fp32_reference"]["finite"] is True


class _FakeCompiled:
    def __init__(self, state, result):
        self.state = state
        self.result = result

    def __call__(self, *_arguments):
        self.state["compiled_invocations"] += 1
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result.copy()


def _checked(state, result, counters):
    return _PROBE._runtime_probe()._wrap_checked(
        _FakeCompiled(state, result),
        proof={"passed": True},
        counter_prefix="forward",
        counters=counters,
    )


class _RuntimeJax:
    __version__ = "test"

    @staticmethod
    def default_backend():
        return "gpu"

    @staticmethod
    def devices():
        return []

    @staticmethod
    def device_get(value):
        return value

    @staticmethod
    def device_put(value):
        return value

    @staticmethod
    def block_until_ready(value):
        return value


def _dependencies(jax=None):
    if jax is None:
        jax = _RuntimeJax()
    jnp = SimpleNamespace()
    jaxlib = SimpleNamespace(__version__="test")
    backend = SimpleNamespace(get_backend=lambda: SimpleNamespace(platform_version="ROCm test"))
    return jax, jnp, jaxlib, backend, object()


def _environment(monkeypatch, process_flags=None, returned_flags=None):
    if process_flags is None:
        process_flags = _DISABLED_COMMAND_BUFFER_FLAG
    if returned_flags is None:
        returned_flags = process_flags
    monkeypatch.setenv("XLA_FLAGS", process_flags)
    return {"XLA_FLAGS_effective": returned_flags}


def _patch_run(
    monkeypatch,
    result,
    expected,
    *,
    compile_error=None,
    device_put_error=None,
    journal_error_stage=None,
    corrupt_counter_stage=None,
):
    state = {"compile_calls": 0, "compiled_invocations": 0, "events": []}

    def compile_forward(_jax, _jnp, _operation, counters, _output):
        state["compile_calls"] += 1
        state["events"].append("compile")
        if compile_error is not None:
            raise compile_error
        return _checked(state, result, counters), {
            "structural_gate": {"passed": True},
            "compiled_memory_gate": {"passed": True},
        }

    def journal(_require, output, stage, counters):
        state["events"].append(stage)
        if stage == corrupt_counter_stage:
            counters["lowered_callable_invocations"] += 1
        if stage == journal_error_stage:
            raise RuntimeError(f"synthetic journal failure at {stage}")
        _PROBE._emit(
            {
                "record_type": "journal_checkpoint",
                "stage": stage,
                "safety": dict(_CLEAN_SAFETY),
                "counters": dict(counters),
            },
            output,
        )
        return dict(_CLEAN_SAFETY)

    def device_put(_jax, inputs):
        state["events"].append("device_put")
        if device_put_error is not None:
            raise device_put_error
        return inputs

    monkeypatch.setattr(_PROBE, "_compile_checked_forward", compile_forward)
    monkeypatch.setattr(
        _PROBE,
        "_construct_host_inputs",
        lambda: ((None, None, None, None), [], expected, {}),
    )
    monkeypatch.setattr(_PROBE, "_device_put_inputs", device_put)
    monkeypatch.setattr(_PROBE, "_allocator_snapshot", lambda _jax: [])
    monkeypatch.setattr(_PROBE, "_journal_checkpoint", journal)
    return state


def _run_patched(monkeypatch, counters, output, *, dependencies=None):
    return _PROBE._run_rocm(
        output,
        lambda: dict(_CLEAN_SAFETY),
        counters,
        environment=_environment(monkeypatch),
        _dependencies=_dependencies() if dependencies is None else dependencies,
    )


def test_mocked_happy_path_has_exact_order_one_compile_and_one_dispatch(monkeypatch):
    expected = np.asarray([1.0, -0.5], dtype=np.float32)
    state = _patch_run(monkeypatch, expected, expected)
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    result = _run_patched(monkeypatch, counters, output)

    assert result == 0
    assert state["compile_calls"] == 1
    assert state["compiled_invocations"] == 1
    assert counters == _PROBE._expected_completed_counters()
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [record["record_type"] for record in records] == [
        "command_buffer_environment_proof",
        "backend_ready",
        "journal_checkpoint",
        "journal_checkpoint",
        "host_fully_iid_dense_reference",
        "journal_checkpoint",
        "journal_checkpoint",
        "dispatch_started",
        "journal_checkpoint",
        "dispatch",
        "journal_checkpoint",
        "numerical_validation",
        "journal_checkpoint",
        "runtime_passed",
    ]
    journals = [record["stage"] for record in records if record["record_type"] == "journal_checkpoint"]
    assert journals == [
        "after_backend_initialization_attempt",
        "after_forward_compile_attempt",
        "after_host_reference_construction",
        "after_explicit_input_device_put",
        "after_fully_iid_candidate_dispatch",
        "after_fully_iid_candidate_device_get",
        "after_host_numerical_validation",
    ]
    started = next(record for record in records if record["record_type"] == "dispatch_started")
    assert started["counters"] == {
        "forward_attempts": 1,
        "forward_completions": 0,
        "lowered_callable_invocations": 0,
    }
    passed = records[-1]
    assert passed["replay_invocations"] == 0
    assert passed["backward_invocations"] == 0
    assert passed["gpu_reference_invocations"] == 0
    assert passed["device_error_reduction_invocations"] == 0
    assert passed["command_buffer_invocations"] == 0


@pytest.mark.parametrize(
    "compile_error",
    [
        RuntimeError("synthetic structural release failure"),
        RuntimeError("synthetic compile failure"),
    ],
)
def test_compile_or_release_failure_checkpoints_and_means_zero_dispatch(monkeypatch, compile_error):
    expected = np.asarray([1.0], dtype=np.float32)
    state = _patch_run(monkeypatch, expected, expected, compile_error=compile_error)
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="synthetic"):
        _run_patched(monkeypatch, counters, output)

    assert state["compile_calls"] == 1
    assert state["compiled_invocations"] == 0
    assert counters == _PROBE._zero_counters()
    assert "after_forward_compile_attempt" in state["events"]
    assert "dispatch_started" not in output.getvalue()
    assert "runtime_passed" not in output.getvalue()


def test_input_device_put_failure_checkpoints_and_means_zero_dispatch(monkeypatch):
    expected = np.asarray([1.0], dtype=np.float32)
    state = _patch_run(
        monkeypatch,
        expected,
        expected,
        device_put_error=RuntimeError("synthetic device_put failure"),
    )
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="synthetic device_put failure"):
        _run_patched(monkeypatch, counters, output)

    assert state["compiled_invocations"] == 0
    assert counters == _PROBE._zero_counters()
    assert "after_explicit_input_device_put" in state["events"]
    assert "dispatch_started" not in output.getvalue()


@pytest.mark.parametrize("failure_kind", ["compiled_call", "synchronization"])
def test_candidate_call_or_synchronization_failure_is_single_attempt_and_checkpoints(monkeypatch, failure_kind):
    expected = np.asarray([1.0], dtype=np.float32)
    result = RuntimeError("synthetic compiled candidate failure") if failure_kind == "compiled_call" else expected
    state = _patch_run(monkeypatch, result, expected)
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    class SynchronizationFails(_RuntimeJax):
        @staticmethod
        def block_until_ready(value):
            if failure_kind == "synchronization":
                raise RuntimeError("synthetic synchronization failure")
            return value

    with pytest.raises(RuntimeError, match="synthetic .*failure"):
        _run_patched(
            monkeypatch,
            counters,
            output,
            dependencies=_dependencies(SynchronizationFails()),
        )

    assert state["compiled_invocations"] == 1
    assert counters == {
        "forward_attempts": 1,
        "forward_completions": 0,
        "lowered_callable_invocations": 0,
    }
    assert "after_fully_iid_candidate_dispatch" in state["events"]
    assert "runtime_passed" not in output.getvalue()


def test_dispatch_journal_failure_yields_no_dispatch_or_success_record(monkeypatch):
    expected = np.asarray([1.0], dtype=np.float32)
    state = _patch_run(
        monkeypatch,
        expected,
        expected,
        journal_error_stage="after_fully_iid_candidate_dispatch",
    )
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="synthetic journal failure"):
        _run_patched(monkeypatch, counters, output)

    assert state["compiled_invocations"] == 1
    assert counters == _PROBE._expected_completed_counters()
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert not any(record["record_type"] == "dispatch" for record in records)
    assert not any(record["record_type"] == "runtime_passed" for record in records)


def test_device_get_failure_rechecks_journal_and_yields_no_validation(monkeypatch):
    expected = np.asarray([1.0], dtype=np.float32)
    state = _patch_run(monkeypatch, expected, expected)
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    class DeviceGetFails(_RuntimeJax):
        @staticmethod
        def device_get(_value):
            raise RuntimeError("synthetic device_get failure")

    with pytest.raises(RuntimeError, match="synthetic device_get failure"):
        _run_patched(
            monkeypatch,
            counters,
            output,
            dependencies=_dependencies(DeviceGetFails()),
        )

    assert state["compiled_invocations"] == 1
    assert counters == _PROBE._expected_completed_counters()
    assert "after_fully_iid_candidate_device_get" in state["events"]
    assert "numerical_validation" not in output.getvalue()
    assert "runtime_passed" not in output.getvalue()


def test_counter_corruption_after_dispatch_fails_closed(monkeypatch):
    expected = np.asarray([1.0], dtype=np.float32)
    state = _patch_run(
        monkeypatch,
        expected,
        expected,
        corrupt_counter_stage="after_fully_iid_candidate_dispatch",
    )
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="counter contract violated"):
        _run_patched(monkeypatch, counters, output)

    assert state["compiled_invocations"] == 1
    assert counters["lowered_callable_invocations"] == 1
    assert "runtime_passed" not in output.getvalue()


def test_out_of_order_dispatch_is_rejected_before_executable_invocation():
    state = {"compiled_invocations": 0}
    counters = _PROBE._zero_counters()
    executable = _checked(state, np.asarray([1.0], dtype=np.float32), counters)
    counters["forward_attempts"] = 1

    with pytest.raises(RuntimeError, match="out-of-order"):
        _PROBE._dispatch_single(
            _RuntimeJax(),
            executable,
            (),
            require_clean_boot=lambda: dict(_CLEAN_SAFETY),
            counters=counters,
            output=io.StringIO(),
        )

    assert state["compiled_invocations"] == 0


@pytest.mark.parametrize(
    ("actual", "seconds", "message"),
    [
        (np.asarray([0.0], dtype=np.float32), 0.001, "numerical or duration"),
        (np.asarray([1.0], dtype=np.float32), 0.1, "numerical or duration"),
        (np.asarray([1.0], dtype=np.float32), -0.001, "numerical or duration"),
    ],
)
def test_numerical_or_non_strict_duration_failure_emits_no_success(actual, seconds, message):
    output = io.StringIO()

    with pytest.raises(RuntimeError, match=message):
        _PROBE._validate_candidate(
            actual,
            np.asarray([1.0], dtype=np.float32),
            seconds,
            _PROBE._expected_completed_counters(),
            output,
        )

    validation = json.loads(output.getvalue())
    assert validation["record_type"] == "numerical_validation"
    assert validation["status"] == "failed"


def test_returned_and_process_flags_must_match_with_one_sole_empty_assignment(
    monkeypatch,
):
    unrelated = "--unrelated=true"
    valid = f"{unrelated} {_DISABLED_COMMAND_BUFFER_FLAG}"
    proof = _PROBE._prove_command_buffers_disabled(_environment(monkeypatch, process_flags=valid))

    assert proof["process_matches_returned"] is True
    assert proof["command_buffer_assignment_count"] == 1
    assert proof["sole_assignment_is_empty"] is True
    assert valid not in json.dumps(proof)

    with pytest.raises(RuntimeError, match="does not match the process"):
        _PROBE._prove_command_buffers_disabled(
            _environment(
                monkeypatch,
                process_flags=valid,
                returned_flags=_DISABLED_COMMAND_BUFFER_FLAG,
            )
        )
    with pytest.raises(RuntimeError, match="sole empty assignment"):
        _PROBE._prove_command_buffers_disabled(
            _environment(
                monkeypatch,
                process_flags=(f"{_DISABLED_COMMAND_BUFFER_FLAG} " "--xla_gpu_enable_command_buffer=true"),
            )
        )


def test_command_buffer_proof_failure_happens_before_backend(monkeypatch):
    environment = _environment(monkeypatch, process_flags="--xla_gpu_enable_command_buffer=false")
    dependencies = list(_dependencies())
    dependencies[0].default_backend = lambda: (_ for _ in ()).throw(AssertionError("backend must remain unused"))

    with pytest.raises(RuntimeError, match="sole empty assignment"):
        _PROBE._run_rocm(
            io.StringIO(),
            lambda: dict(_CLEAN_SAFETY),
            _PROBE._zero_counters(),
            environment=environment,
            _dependencies=tuple(dependencies),
        )


def test_delegated_compile_jsonl_error_text_is_redacted(monkeypatch):
    secret = f"PRIVATE_SECRET_COMPILE {str(_REPO)}/compiler-cache"
    runtime = _PROBE._runtime_probe()

    def delegated(_jax, _jnp, _operation, _counters, output):
        output.write(
            json.dumps(
                {
                    "record_type": "forward_compiled",
                    "compiled_memory": {"available": False, "error": secret},
                }
            )
            + "\n"
        )
        return object(), {}

    monkeypatch.setattr(runtime, "_compile_checked_forward", delegated)
    output = io.StringIO()

    _PROBE._compile_checked_forward(object(), object(), object(), _PROBE._zero_counters(), output)

    artifact = output.getvalue()
    assert secret not in artifact
    assert str(_REPO) not in artifact
    summary = json.loads(artifact)["compiled_memory"]["error"]
    encoded = secret.encode()
    assert summary == {
        "text_redacted": True,
        "utf8_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _configured_environment(secret_path=""):
    original = f"--xla_dump_to={secret_path}" if secret_path else ""
    effective = f"{original} {_DISABLED_COMMAND_BUFFER_FLAG}".strip()
    return {
        "JAX_PLATFORMS": "rocm",
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
        "JAX_ROCM_VISIBLE_DEVICES": "0",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_CLIENT_MEM_FRACTION": "0.75",
        "SKYRL_ROCM_PALLAS_ATTENTION": "1",
        "XLA_FLAGS_original": original,
        "XLA_FLAGS_effective": effective,
    }


def test_environment_artifact_digests_raw_xla_flags_and_paths(monkeypatch):
    secret_path = "/home/private-user/private-repo/compiler-dump"
    environment = _configured_environment(secret_path)
    monkeypatch.setenv("XLA_FLAGS", environment["XLA_FLAGS_effective"])

    manifest = _PROBE._environment_manifest(environment)
    proof = _PROBE._prove_command_buffers_disabled(environment)
    artifact = json.dumps({"environment": manifest, "proof": proof})

    assert secret_path not in artifact
    assert environment["XLA_FLAGS_original"] not in artifact
    assert environment["XLA_FLAGS_effective"] not in artifact
    assert manifest["raw_xla_flags_emitted"] is False
    assert proof["raw_xla_flags_emitted"] is False


def test_terminal_runtime_error_is_digest_only_and_postflight_precedes_it(
    monkeypatch,
):
    secret = f"PRIVATE_SECRET_RUNTIME {str(_REPO)}/private-runtime"
    environment = _configured_environment()

    @contextmanager
    def guard():
        yield dict(_HEADLESS_SAFETY)

    monkeypatch.setattr(_PROBE, "_assert_fresh_accelerator_process", lambda: None)
    monkeypatch.setattr(_PROBE, "_configure_rocm_environment", lambda: environment)
    monkeypatch.setattr(
        _PROBE,
        "_load_safety_helpers",
        lambda: (guard, lambda: dict(_CLEAN_SAFETY)),
    )
    monkeypatch.setattr(
        _PROBE,
        "_run_rocm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    output = io.StringIO()

    result = _PROBE._execute(SimpleNamespace(platform="rocm", allow_gpu=True), output)

    assert result == 1
    artifact = output.getvalue()
    records = [json.loads(line) for line in artifact.splitlines()]
    error = records[-1]
    encoded = secret.encode()
    assert [record["record_type"] for record in records[-2:]] == [
        "safety_postflight",
        "error",
    ]
    assert error["stage"] == "runtime"
    assert error["error_type"] == "RuntimeError"
    assert error["message_redacted"] is True
    assert error["message_utf8_bytes"] == len(encoded)
    assert error["message_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert "message" not in error
    assert secret not in artifact
    assert str(_REPO) not in artifact


def test_rocm_execute_refuses_preimported_jax_before_environment(monkeypatch):
    state = {"environment_configured": False}
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace())
    monkeypatch.setattr(
        _PROBE,
        "_configure_rocm_environment",
        lambda: state.__setitem__("environment_configured", True),
    )
    output = io.StringIO()

    result = _PROBE._execute(SimpleNamespace(platform="rocm", allow_gpu=True), output)

    assert result == 1
    assert state["environment_configured"] is False
    error = json.loads(output.getvalue().splitlines()[-1])
    assert error["record_type"] == "error"
    assert error["stage"] == "fresh_process_preflight"
    assert error["message_redacted"] is True
    assert "message" not in error


@pytest.mark.parametrize(
    "safety",
    [
        {"amdgpu_boot_clean": False, "fatal_amdgpu_events": []},
        {"amdgpu_boot_clean": True},
        {"amdgpu_boot_clean": True, "fatal_amdgpu_events": ["private fatal"]},
    ],
)
def test_dirty_or_malformed_safety_never_becomes_public_clean_proof(safety):
    with pytest.raises(RuntimeError, match="clean AMDGPU boot|fatal-event proof"):
        _PROBE._public_clean_safety(safety, "test_stage")


def test_preflight_validates_and_emits_controlled_headless_kfd_evidence():
    proof = _PROBE._public_safety_preflight(dict(_HEADLESS_SAFETY))

    assert proof == _HEADLESS_SAFETY


@pytest.mark.parametrize(
    "mutation",
    [
        {"amd_cards": []},
        {"amd_cards": ["/private/card1"]},
        {"amd_cards": ["card2", "card1"]},
        {"connected_amd_connectors": ["card1-HDMI-A-1"]},
        {"kfd_path": "/private/kfd"},
        {"kfd_accessible": False},
        {"kfd_unowned": False},
    ],
)
def test_malformed_headless_or_kfd_evidence_fails_closed(mutation):
    safety = {**_HEADLESS_SAFETY, **mutation}

    with pytest.raises(RuntimeError, match="safety_preflight"):
        _PROBE._public_safety_preflight(safety)


def test_source_has_no_early_jax_import_or_out_of_scope_accelerator_paths():
    source = _PROBE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    imports = [node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    roots = {alias.name.partition(".")[0] for node in imports for alias in node.names}
    assert "jax" not in roots
    assert "skyrl" not in roots
    assert "jax.random" not in source
    assert "jax.vjp" not in source
    assert "jax.grad" not in source
    run_source = inspect.getsource(_PROBE._run_rocm)
    assert run_source.index("_prove_command_buffers_disabled") < run_source.index("import jax")
    assert run_source.count("_dispatch_single(") == 1
    assert run_source.count("_device_get_checked(") == 1
    assert "_dispatch_checked_phase" not in run_source
