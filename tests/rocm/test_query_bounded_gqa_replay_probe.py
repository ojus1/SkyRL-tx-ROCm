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

import numpy as np
import pytest

_REPO = Path(__file__).parents[2]
_PROBE_PATH = _REPO / "rocm" / "probe_query_bounded_gqa_replay.py"
_SPEC = importlib.util.spec_from_file_location("probe_query_bounded_gqa_replay_test", _PROBE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)

_ACCELERATOR_ENVIRONMENT = (
    "GPU_DEVICE_ORDINAL",
    "HSA_OVERRIDE_GFX_VERSION",
    "HIP_VISIBLE_DEVICES",
    "JAX_PJRT_CLIENT_CREATE_OPTIONS",
    "JAX_PLATFORMS",
    "JAX_ROCM_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "SKYRL_ROCM_PALLAS_ATTENTION",
    "TF_FORCE_UNIFIED_MEMORY",
    "XLA_CLIENT_MEM_FRACTION",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in _ACCELERATOR_ENVIRONMENT:
        environment.pop(name, None)
    return environment


def _run(*arguments: str):
    return subprocess.run(
        [sys.executable, str(_PROBE_PATH), *arguments],
        cwd=_REPO,
        env=_clean_environment(),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def test_default_is_abstract_refusal_before_jax_import():
    result = _run()

    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert [record["record_type"] for record in records] == ["manifest", "refused"]
    manifest, refused = records
    assert manifest["scope"] == "abstract_refusal"
    assert manifest["compile_may_dispatch_gpu_work"] is False
    assert manifest["fresh_process_required"] is True
    assert manifest["counters"] == _PROBE._zero_counters()
    assert manifest["probe_source_sha256"] == hashlib.sha256(_PROBE_PATH.read_bytes()).hexdigest()
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
        (("--repeats", "3"), "unrecognized arguments"),
        (("--backward",), "unrecognized arguments"),
        (("--padding",), "unrecognized arguments"),
    ],
)
def test_unsafe_or_scope_broadening_options_are_rejected(arguments, message):
    result = _run(*arguments)

    assert result.returncode == 2
    assert result.stdout == ""
    assert message in result.stderr


def test_output_is_private_and_exclusive(tmp_path):
    output = tmp_path / "replay.jsonl"

    result = _run("--output", str(output))

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert [json.loads(line)["record_type"] for line in output.read_text().splitlines()] == ["manifest", "refused"]
    repeated = _run("--output", str(output))
    assert repeated.returncode == 2
    assert "refusing to overwrite existing output" in repeated.stderr


def test_contract_is_exact_nonzero_candidate_then_ordinary_replay():
    contract = _PROBE._exact_contract()

    assert contract["operation"] == "query_bounded_gqa_forward_candidate_and_replay"
    assert contract["seed"] == 20260713
    assert contract["randomness_used"] is True
    assert contract["valid_length"] == 512
    assert contract["dispatch_plan"] == {
        "checked_forward_invocations": 2,
        "candidate_invocations": 1,
        "ordinary_replay_invocations": 1,
        "command_buffer_replay_invocations": 0,
        "gpu_reference_invocations": 0,
        "device_error_reduction_invocations": 0,
        "backward_invocations": 0,
    }
    assert contract["compiled_memory_gate"] == {
        "memory_analysis_required": True,
        "maximum_temporary_bytes": 64 * 1024**2,
        "maximum_argument_output_temporary_bytes": 128 * 1024**2,
    }
    assert [entry["shape"] for entry in contract["inputs"]] == [
        [1, 512, 16, 256],
        [1, 512, 4, 256],
        [1, 512, 4, 256],
        [1, 512],
    ]


def test_factorized_random_inputs_are_deterministic_dense_nonzero_and_exact():
    first_inputs, first_manifests, expected, expected_manifest = _PROBE._construct_host_inputs()
    second_inputs, second_manifests, second_expected, second_expected_manifest = _PROBE._construct_host_inputs()
    q, k, v, mask = first_inputs

    assert q.shape == (1, 512, 16, 256)
    assert k.shape == v.shape == (1, 512, 4, 256)
    assert expected.shape == q.shape
    assert np.count_nonzero(q) == q.size
    assert np.count_nonzero(k) == k.size
    assert np.count_nonzero(v) == v.size
    assert np.all(mask == 1)
    assert np.unique(q.astype(np.float32)).size == 96
    assert expected_manifest["oracle"]["seed"] == 20260713
    assert all(factor != 0.0 for factor in expected_manifest["oracle"]["qk_direction_dot_factors_after_scale"])
    assert first_manifests == second_manifests
    assert expected_manifest == second_expected_manifest
    np.testing.assert_array_equal(expected, second_expected)
    for first, second in zip(first_inputs, second_inputs, strict=True):
        np.testing.assert_array_equal(first, second)

    # Independently check one nontrivial causal row using the materialized dense
    # BF16 arrays, not the factorization used by the probe's host oracle.
    query_position = 137
    query_head = 5
    kv_head = query_head // 4
    dense_q = q.astype(np.float32)[0, query_position, query_head]
    dense_k = k.astype(np.float32)[0, : query_position + 1, kv_head]
    logits = (dense_k @ dense_q) * (256**-0.5)
    probabilities = np.exp(logits - np.max(logits))
    probabilities /= np.sum(probabilities)
    dense_reference = probabilities @ v.astype(np.float32)[0, : query_position + 1, kv_head]
    np.testing.assert_allclose(
        expected[0, query_position, query_head],
        dense_reference,
        rtol=2e-6,
        atol=2e-6,
    )


_CLEAN_SAFETY = {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}


class _FakeCompiled:
    def __init__(self, state, results):
        self.state = state
        self.results = results

    def __call__(self, *_arguments):
        invocation = self.state["compiled_invocations"]
        self.state["compiled_invocations"] += 1
        result = self.results[invocation] if isinstance(self.results, list) else self.results
        if isinstance(result, BaseException):
            raise result
        return result.copy()


def _checked(state, results, counters):
    return _PROBE._runtime_probe()._wrap_checked(
        _FakeCompiled(state, results),
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


def _dependencies():
    jax = _RuntimeJax()
    jnp = SimpleNamespace()
    jaxlib = SimpleNamespace(__version__="test")
    backend = SimpleNamespace(get_backend=lambda: SimpleNamespace(platform_version="ROCm test"))
    return jax, jnp, jaxlib, backend, object()


def _environment(monkeypatch, process_flags=None, returned_flags=None):
    if process_flags is None:
        process_flags = _PROBE._DISABLED_COMMAND_BUFFER_FLAG
    if returned_flags is None:
        returned_flags = process_flags
    monkeypatch.setenv("XLA_FLAGS", process_flags)
    return {"XLA_FLAGS_effective": returned_flags}


def _valid_candidate_proofs(executable, counters, *, output=None):
    if output is None:
        output = io.StringIO()
    jax = _RuntimeJax()
    candidate, seconds, dispatch_proof = _PROBE._dispatch_checked_phase(
        jax,
        executable,
        (),
        phase="candidate",
        require_clean_boot=lambda: dict(_CLEAN_SAFETY),
        counters=counters,
        output=output,
    )
    candidate_host, device_get_proof = _PROBE._device_get_with_journal_proof(
        jax,
        candidate,
        phase="candidate",
        executable=executable,
        require_clean_boot=lambda: dict(_CLEAN_SAFETY),
        counters=counters,
        output=output,
    )
    _metrics, _gate, validation_proof = _PROBE._validate_phase(
        "candidate",
        candidate_host,
        candidate_host.copy(),
        seconds,
        counters,
        output,
        executable=executable,
    )
    assert validation_proof is not None
    return candidate_host, dispatch_proof, device_get_proof, validation_proof


def _complete_valid_candidate(executable, counters, *, output=None):
    candidate_host, dispatch_proof, device_get_proof, validation_proof = _valid_candidate_proofs(
        executable, counters, output=output
    )
    authorization = _PROBE._issue_replay_authorization(
        executable,
        validation_proof=validation_proof,
        dispatch_journal_proof=dispatch_proof,
        device_get_journal_proof=device_get_proof,
        counters=counters,
    )
    return candidate_host, authorization


def test_unvalidated_candidate_counters_cannot_launch_replay():
    state = {"compiled_invocations": 0}
    counters = _PROBE._zero_counters()
    executable = _checked(state, np.asarray([1.0], dtype=np.float32), counters)
    candidate, _seconds, _dispatch_proof = _PROBE._dispatch_checked_phase(
        _RuntimeJax(),
        executable,
        (),
        phase="candidate",
        require_clean_boot=lambda: dict(_CLEAN_SAFETY),
        counters=counters,
        output=io.StringIO(),
    )
    _PROBE._device_get_with_journal_proof(
        _RuntimeJax(),
        candidate,
        phase="candidate",
        executable=executable,
        require_clean_boot=lambda: dict(_CLEAN_SAFETY),
        counters=counters,
        output=io.StringIO(),
    )

    with pytest.raises(RuntimeError, match="opaque replay authorization"):
        _PROBE._dispatch_checked_phase(
            _RuntimeJax(),
            executable,
            (),
            phase="replay",
            require_clean_boot=lambda: dict(_CLEAN_SAFETY),
            counters=counters,
            output=io.StringIO(),
        )

    assert state["compiled_invocations"] == 1
    assert counters == _PROBE._expected_counters_after("candidate")


def test_bad_candidate_cannot_issue_authorization_or_launch_replay():
    state = {"compiled_invocations": 0}
    counters = _PROBE._zero_counters()
    executable = _checked(state, np.asarray([0.0], dtype=np.float32), counters)
    candidate, seconds, dispatch_proof = _PROBE._dispatch_checked_phase(
        _RuntimeJax(),
        executable,
        (),
        phase="candidate",
        require_clean_boot=lambda: dict(_CLEAN_SAFETY),
        counters=counters,
        output=io.StringIO(),
    )
    candidate_host, device_get_proof = _PROBE._device_get_with_journal_proof(
        _RuntimeJax(),
        candidate,
        phase="candidate",
        executable=executable,
        require_clean_boot=lambda: dict(_CLEAN_SAFETY),
        counters=counters,
        output=io.StringIO(),
    )
    with pytest.raises(RuntimeError, match="candidate numerical or duration"):
        _PROBE._validate_phase(
            "candidate",
            candidate_host,
            np.asarray([1.0], dtype=np.float32),
            seconds,
            counters,
            io.StringIO(),
            executable=executable,
        )
    with pytest.raises(RuntimeError, match="validation proof is required"):
        _PROBE._issue_replay_authorization(
            executable,
            validation_proof=None,
            dispatch_journal_proof=dispatch_proof,
            device_get_journal_proof=device_get_proof,
            counters=counters,
        )
    with pytest.raises(RuntimeError, match="opaque replay authorization"):
        _PROBE._dispatch_checked_phase(
            _RuntimeJax(),
            executable,
            (),
            phase="replay",
            require_clean_boot=lambda: dict(_CLEAN_SAFETY),
            counters=counters,
            output=io.StringIO(),
        )
    assert state["compiled_invocations"] == 1


def test_replay_authorization_is_bound_to_exact_executable_and_consumed_on_bad_attempt():
    first_state = {"compiled_invocations": 0}
    second_state = {"compiled_invocations": 0}
    counters = _PROBE._zero_counters()
    first = _checked(first_state, np.asarray([1.0], dtype=np.float32), counters)
    second = _checked(second_state, np.asarray([1.0], dtype=np.float32), counters)
    _candidate, authorization = _complete_valid_candidate(first, counters)

    with pytest.raises(RuntimeError, match="bound to a different"):
        _PROBE._dispatch_checked_phase(
            _RuntimeJax(),
            second,
            (),
            phase="replay",
            require_clean_boot=lambda: dict(_CLEAN_SAFETY),
            counters=counters,
            output=io.StringIO(),
            replay_authorization=authorization,
        )
    with pytest.raises(RuntimeError, match="already consumed"):
        _PROBE._dispatch_checked_phase(
            _RuntimeJax(),
            first,
            (),
            phase="replay",
            require_clean_boot=lambda: dict(_CLEAN_SAFETY),
            counters=counters,
            output=io.StringIO(),
            replay_authorization=authorization,
        )

    assert first_state["compiled_invocations"] == 1
    assert second_state["compiled_invocations"] == 0
    assert counters == _PROBE._expected_counters_after("candidate")


def test_replay_authorization_is_one_shot_after_success():
    state = {"compiled_invocations": 0}
    counters = _PROBE._zero_counters()
    executable = _checked(state, np.asarray([1.0], dtype=np.float32), counters)
    _candidate, authorization = _complete_valid_candidate(executable, counters)

    _replay, _seconds, _journal = _PROBE._dispatch_checked_phase(
        _RuntimeJax(),
        executable,
        (),
        phase="replay",
        require_clean_boot=lambda: dict(_CLEAN_SAFETY),
        counters=counters,
        output=io.StringIO(),
        replay_authorization=authorization,
    )
    with pytest.raises(RuntimeError, match="already consumed"):
        _PROBE._dispatch_checked_phase(
            _RuntimeJax(),
            executable,
            (),
            phase="replay",
            require_clean_boot=lambda: dict(_CLEAN_SAFETY),
            counters=counters,
            output=io.StringIO(),
            replay_authorization=authorization,
        )

    assert state["compiled_invocations"] == 2
    assert counters == _PROBE._expected_counters_after("replay")


def test_replay_attempt_with_corrupt_counters_consumes_authorization():
    state = {"compiled_invocations": 0}
    counters = _PROBE._zero_counters()
    executable = _checked(state, np.asarray([1.0], dtype=np.float32), counters)
    _candidate, authorization = _complete_valid_candidate(executable, counters)
    counters["forward_completions"] += 1

    with pytest.raises(RuntimeError, match="out-of-order replay"):
        _PROBE._dispatch_checked_phase(
            _RuntimeJax(),
            executable,
            (),
            phase="replay",
            require_clean_boot=lambda: dict(_CLEAN_SAFETY),
            counters=counters,
            output=io.StringIO(),
            replay_authorization=authorization,
        )
    counters["forward_completions"] -= 1
    with pytest.raises(RuntimeError, match="already consumed"):
        _PROBE._dispatch_checked_phase(
            _RuntimeJax(),
            executable,
            (),
            phase="replay",
            require_clean_boot=lambda: dict(_CLEAN_SAFETY),
            counters=counters,
            output=io.StringIO(),
            replay_authorization=authorization,
        )

    assert state["compiled_invocations"] == 1
    assert counters == _PROBE._expected_counters_after("candidate")


@pytest.mark.parametrize("failure_kind", ["compiled_call", "synchronization"])
def test_failed_replay_call_or_synchronization_consumes_authorization(failure_kind):
    state = {"compiled_invocations": 0}
    counters = _PROBE._zero_counters()
    value = np.asarray([1.0], dtype=np.float32)
    results = (
        [value, RuntimeError("synthetic compiled replay failure")]
        if failure_kind == "compiled_call"
        else [value, value]
    )
    executable = _checked(state, results, counters)
    _candidate, authorization = _complete_valid_candidate(executable, counters)

    class ReplayJax(_RuntimeJax):
        @staticmethod
        def block_until_ready(result):
            if failure_kind == "synchronization":
                raise RuntimeError("synthetic replay synchronization failure")
            return result

    with pytest.raises(RuntimeError, match="synthetic .*failure"):
        _PROBE._dispatch_checked_phase(
            ReplayJax(),
            executable,
            (),
            phase="replay",
            require_clean_boot=lambda: dict(_CLEAN_SAFETY),
            counters=counters,
            output=io.StringIO(),
            replay_authorization=authorization,
        )
    with pytest.raises(RuntimeError, match="already consumed"):
        _PROBE._dispatch_checked_phase(
            _RuntimeJax(),
            executable,
            (),
            phase="replay",
            require_clean_boot=lambda: dict(_CLEAN_SAFETY),
            counters=counters,
            output=io.StringIO(),
            replay_authorization=authorization,
        )

    assert state["compiled_invocations"] == 2
    assert counters["forward_attempts"] == 2
    assert counters["forward_completions"] == 1
    assert counters["replay_attempts"] == 1
    assert counters["replay_completions"] == 0


def test_replay_journal_failure_consumes_authorization(monkeypatch):
    state = {"compiled_invocations": 0}
    counters = _PROBE._zero_counters()
    executable = _checked(state, np.asarray([1.0], dtype=np.float32), counters)
    _candidate, authorization = _complete_valid_candidate(executable, counters)
    monkeypatch.setattr(
        _PROBE,
        "_journal_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic replay journal failure")),
    )

    with pytest.raises(RuntimeError, match="synthetic replay journal failure"):
        _PROBE._dispatch_checked_phase(
            _RuntimeJax(),
            executable,
            (),
            phase="replay",
            require_clean_boot=lambda: dict(_CLEAN_SAFETY),
            counters=counters,
            output=io.StringIO(),
            replay_authorization=authorization,
        )
    with pytest.raises(RuntimeError, match="already consumed"):
        _PROBE._dispatch_checked_phase(
            _RuntimeJax(),
            executable,
            (),
            phase="replay",
            require_clean_boot=lambda: dict(_CLEAN_SAFETY),
            counters=counters,
            output=io.StringIO(),
            replay_authorization=authorization,
        )

    assert state["compiled_invocations"] == 2
    assert counters == _PROBE._expected_counters_after("replay")


@pytest.mark.parametrize(
    "safety",
    [
        {"amdgpu_boot_clean": False, "fatal_amdgpu_events": []},
        {"amdgpu_boot_clean": True},
        {"amdgpu_boot_clean": True, "fatal_amdgpu_events": ["private fatal"]},
    ],
)
def test_dirty_or_malformed_journal_result_cannot_create_proof(safety):
    with pytest.raises(RuntimeError, match="clean AMDGPU boot|fatal-event proof"):
        _PROBE._clean_journal_proof(
            lambda: safety,
            io.StringIO(),
            "after_candidate_dispatch",
            _PROBE._zero_counters(),
            executable=object(),
        )


def test_swapped_or_malformed_candidate_journal_proofs_are_rejected():
    state = {"compiled_invocations": 0}
    counters = _PROBE._zero_counters()
    executable = _checked(state, np.asarray([1.0], dtype=np.float32), counters)
    _host, dispatch_proof, device_get_proof, validation_proof = _valid_candidate_proofs(executable, counters)

    with pytest.raises(RuntimeError, match="invalid journal proof"):
        _PROBE._issue_replay_authorization(
            executable,
            validation_proof=validation_proof,
            dispatch_journal_proof=device_get_proof,
            device_get_journal_proof=dispatch_proof,
            counters=counters,
        )
    with pytest.raises(RuntimeError, match="missing opaque journal proof"):
        _PROBE._issue_replay_authorization(
            executable,
            validation_proof=validation_proof,
            dispatch_journal_proof=object(),
            device_get_journal_proof=device_get_proof,
            counters=counters,
        )


def test_stale_candidate_journal_proofs_are_rejected():
    state = {"compiled_invocations": 0}
    counters = _PROBE._zero_counters()
    executable = _checked(state, np.asarray([1.0], dtype=np.float32), counters)
    stale_dispatch = _PROBE._clean_journal_proof(
        lambda: dict(_CLEAN_SAFETY),
        io.StringIO(),
        "after_candidate_dispatch",
        counters,
        executable=executable,
    )
    stale_device_get = _PROBE._clean_journal_proof(
        lambda: dict(_CLEAN_SAFETY),
        io.StringIO(),
        "after_candidate_device_get",
        counters,
        executable=executable,
    )
    _host, _dispatch, _device_get, validation_proof = _valid_candidate_proofs(executable, counters)

    with pytest.raises(RuntimeError, match="invalid journal proof"):
        _PROBE._issue_replay_authorization(
            executable,
            validation_proof=validation_proof,
            dispatch_journal_proof=stale_dispatch,
            device_get_journal_proof=stale_device_get,
            counters=counters,
        )


def test_failed_candidate_dispatch_checkpoints_and_never_releases_replay(monkeypatch):
    state = {"journal": 0, "invocations": 0}
    counters = _PROBE._zero_counters()

    class Failing:
        def __call__(self, *_arguments):
            state["invocations"] += 1
            raise RuntimeError("synthetic candidate failure")

    executable = _PROBE._runtime_probe()._wrap_checked(
        Failing(),
        proof={"passed": True},
        counter_prefix="forward",
        counters=counters,
    )
    monkeypatch.setattr(
        _PROBE,
        "_journal_checkpoint",
        lambda *_arguments: (state.__setitem__("journal", state["journal"] + 1) or dict(_CLEAN_SAFETY)),
    )

    with pytest.raises(RuntimeError, match="synthetic candidate failure"):
        _PROBE._dispatch_checked_phase(
            _RuntimeJax(),
            executable,
            (),
            phase="candidate",
            require_clean_boot=lambda: dict(_CLEAN_SAFETY),
            counters=counters,
            output=io.StringIO(),
        )

    assert state == {"journal": 1, "invocations": 1}
    assert counters["forward_attempts"] == 1
    assert counters["forward_completions"] == 0
    assert counters["candidate_attempts"] == 1
    assert counters["candidate_completions"] == 0
    assert counters["replay_attempts"] == counters["replay_completions"] == 0


def _patch_run(
    monkeypatch,
    results,
    expected,
    *,
    compile_error=None,
    journal_error_stage=None,
    corrupt_counter_stage=None,
):
    state = {"compile_calls": 0, "compiled_invocations": 0, "events": []}

    def compile_forward(_jax, _jnp, _operation, counters, _output):
        state["compile_calls"] += 1
        state["events"].append("compile_called")
        if compile_error is not None:
            raise compile_error
        return _checked(state, results, counters), {
            "structural_gate": {"passed": True},
            "compiled_memory_gate": {"passed": True},
        }

    def journal(_require, _output, stage, counters):
        state["events"].append(stage)
        if stage == corrupt_counter_stage:
            counters["forward_completions"] += 1
        if stage == journal_error_stage:
            raise RuntimeError(f"synthetic journal failure at {stage}")
        safety = dict(_CLEAN_SAFETY)
        _PROBE._emit(
            {
                "record_type": "journal_checkpoint",
                "timestamp": _PROBE._utc_now(),
                "stage": stage,
                "safety": safety,
                "counters": dict(counters),
            },
            _output,
        )
        return safety

    monkeypatch.setattr(_PROBE, "_compile_checked_forward", compile_forward)
    monkeypatch.setattr(
        _PROBE,
        "_construct_host_inputs",
        lambda: ((None, None, None, None), [], expected, {}),
    )
    monkeypatch.setattr(
        _PROBE,
        "_device_put_inputs",
        lambda _jax, inputs: state["events"].append("device_put") or inputs,
    )
    monkeypatch.setattr(_PROBE, "_allocator_snapshot", lambda _jax: [])
    monkeypatch.setattr(_PROBE, "_journal_checkpoint", journal)
    return state


def _run_patched(monkeypatch, state, counters, output, *, dependencies=None):
    del state  # State is asserted by callers; keep this helper's call signature explicit.
    return _PROBE._run_rocm(
        output,
        lambda: dict(_CLEAN_SAFETY),
        counters,
        environment=_environment(monkeypatch),
        _dependencies=_dependencies() if dependencies is None else dependencies,
    )


def test_runtime_happy_path_has_exact_mocked_record_order_and_one_compile(monkeypatch):
    expected = np.asarray([1.0, -0.5], dtype=np.float32)
    state = _patch_run(monkeypatch, expected, expected)
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    result = _run_patched(monkeypatch, state, counters, output)

    assert result == 0
    assert state["compile_calls"] == 1
    assert state["compiled_invocations"] == 2
    assert counters == _PROBE._expected_counters_after("replay")
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    record_types = [record["record_type"] for record in records]
    assert record_types == [
        "command_buffer_environment_proof",
        "backend_ready",
        "journal_checkpoint",
        "journal_checkpoint",
        "host_factorized_reference",
        "journal_checkpoint",
        "dispatch_started",
        "journal_checkpoint",
        "dispatch",
        "journal_checkpoint",
        "numerical_validation",
        "replay_release_gate",
        "dispatch_started",
        "journal_checkpoint",
        "dispatch",
        "journal_checkpoint",
        "numerical_validation",
        "replay_equality_validation",
        "runtime_passed",
    ]
    journals = [record for record in records if record["record_type"] == "journal_checkpoint"]
    assert [record["stage"] for record in journals] == [
        "after_backend_initialization",
        "after_forward_compile",
        "after_explicit_input_device_put",
        "after_candidate_dispatch",
        "after_candidate_device_get",
        "after_replay_dispatch",
        "after_replay_device_get",
    ]
    release_index = record_types.index("replay_release_gate")
    assert record_types.index("journal_checkpoint", 7) < release_index
    assert record_types.index("journal_checkpoint", 9) < release_index
    dispatch_starts = [record for record in records if record["record_type"] == "dispatch_started"]
    assert [record["label"] for record in dispatch_starts] == ["candidate", "replay"]
    assert dispatch_starts[0]["counters"] == {
        **_PROBE._zero_counters(),
        "forward_attempts": 1,
        "candidate_attempts": 1,
    }
    assert dispatch_starts[1]["counters"] == {
        **_PROBE._expected_counters_after("candidate"),
        "forward_attempts": 2,
        "replay_attempts": 1,
    }
    assert records[-1]["record_type"] == "runtime_passed"
    assert records[-1]["counters"] == _PROBE._expected_counters_after("replay")


def test_candidate_numerical_failure_prevents_replay(monkeypatch):
    state = _patch_run(
        monkeypatch,
        np.asarray([0.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
    )
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="candidate numerical or duration"):
        _run_patched(monkeypatch, state, counters, output)

    assert state["compiled_invocations"] == 1
    assert counters == _PROBE._expected_counters_after("candidate")
    assert counters["replay_attempts"] == counters["replay_completions"] == 0
    assert "replay_release_gate" not in output.getvalue()
    assert "runtime_passed" not in output.getvalue()


def test_candidate_journal_failure_prevents_authorization_and_replay(monkeypatch):
    expected = np.asarray([1.0], dtype=np.float32)
    state = _patch_run(
        monkeypatch,
        expected,
        expected,
        journal_error_stage="after_candidate_dispatch",
    )
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="after_candidate_dispatch"):
        _run_patched(monkeypatch, state, counters, output)

    assert state["compiled_invocations"] == 1
    assert counters["replay_attempts"] == counters["replay_completions"] == 0
    assert "replay_release_gate" not in output.getvalue()
    assert "runtime_passed" not in output.getvalue()


def test_candidate_counter_corruption_prevents_authorization_and_replay(monkeypatch):
    expected = np.asarray([1.0], dtype=np.float32)
    state = _patch_run(
        monkeypatch,
        expected,
        expected,
        corrupt_counter_stage="after_candidate_device_get",
    )
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="candidate invocation counter"):
        _run_patched(monkeypatch, state, counters, output)

    assert state["compiled_invocations"] == 1
    assert counters["replay_attempts"] == counters["replay_completions"] == 0
    assert "replay_release_gate" not in output.getvalue()
    assert "runtime_passed" not in output.getvalue()


@pytest.mark.parametrize(
    "compile_error",
    [
        RuntimeError("synthetic structural gate failure"),
        RuntimeError("synthetic compile failure"),
    ],
)
def test_structural_or_compile_failure_means_zero_dispatch(monkeypatch, compile_error):
    expected = np.asarray([1.0], dtype=np.float32)
    state = _patch_run(
        monkeypatch,
        expected,
        expected,
        compile_error=compile_error,
    )
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="synthetic"):
        _run_patched(monkeypatch, state, counters, output)

    assert state["compile_calls"] == 1
    assert state["compiled_invocations"] == 0
    assert counters == _PROBE._zero_counters()
    assert "dispatch_started" not in output.getvalue()
    assert "runtime_passed" not in output.getvalue()


def test_candidate_device_get_failure_rechecks_journal_and_prevents_replay(monkeypatch):
    expected = np.asarray([1.0], dtype=np.float32)
    state = _patch_run(monkeypatch, expected, expected)
    counters = _PROBE._zero_counters()
    dependencies = list(_dependencies())
    dependencies[0].device_get = lambda _value: (_ for _ in ()).throw(RuntimeError("synthetic device_get failure"))
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="synthetic device_get failure"):
        _run_patched(
            monkeypatch,
            state,
            counters,
            output,
            dependencies=tuple(dependencies),
        )

    assert state["compiled_invocations"] == 1
    assert state["events"].count("after_candidate_device_get") == 1
    assert counters == _PROBE._expected_counters_after("candidate")
    assert counters["replay_attempts"] == counters["replay_completions"] == 0
    assert "runtime_passed" not in output.getvalue()


def test_replay_device_get_failure_rechecks_journal_and_yields_no_success(monkeypatch):
    expected = np.asarray([1.0], dtype=np.float32)
    state = _patch_run(monkeypatch, expected, expected)
    counters = _PROBE._zero_counters()

    class ReplayDeviceGetFails(_RuntimeJax):
        def __init__(self):
            self.device_get_calls = 0

        def device_get(self, value):
            self.device_get_calls += 1
            if self.device_get_calls == 2:
                raise RuntimeError("synthetic replay device_get failure")
            return value

    dependencies = list(_dependencies())
    dependencies[0] = ReplayDeviceGetFails()
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="synthetic replay device_get failure"):
        _run_patched(
            monkeypatch,
            state,
            counters,
            output,
            dependencies=tuple(dependencies),
        )

    assert state["compiled_invocations"] == 2
    assert state["events"].count("after_replay_device_get") == 1
    assert counters == _PROBE._expected_counters_after("replay")
    assert "replay_equality_validation" not in output.getvalue()
    assert "runtime_passed" not in output.getvalue()


def test_replay_journal_failure_yields_no_success_record(monkeypatch):
    expected = np.asarray([1.0], dtype=np.float32)
    state = _patch_run(
        monkeypatch,
        expected,
        expected,
        journal_error_stage="after_replay_dispatch",
    )
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="after_replay_dispatch"):
        _run_patched(monkeypatch, state, counters, output)

    assert state["compiled_invocations"] == 2
    assert "replay_release_gate" in output.getvalue()
    assert "runtime_passed" not in output.getvalue()


def test_replay_numerical_failure_yields_no_success_record(monkeypatch):
    expected = np.asarray([1.0], dtype=np.float32)
    results = [expected, np.asarray([0.0], dtype=np.float32)]
    state = _patch_run(monkeypatch, results, expected)
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="replay numerical or duration"):
        _run_patched(monkeypatch, state, counters, output)

    assert state["compiled_invocations"] == 2
    assert "runtime_passed" not in output.getvalue()


def test_replay_equality_failure_yields_no_success_record(monkeypatch):
    expected = np.asarray([1.0], dtype=np.float32)
    results = [expected, np.asarray([1.0001], dtype=np.float32)]
    state = _patch_run(monkeypatch, results, expected)
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="not byte-identical"):
        _run_patched(monkeypatch, state, counters, output)

    assert state["compiled_invocations"] == 2
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    equality = next(record for record in records if record["record_type"] == "replay_equality_validation")
    assert equality["status"] == "failed"
    assert not any(record["record_type"] == "runtime_passed" for record in records)


@pytest.mark.parametrize(
    "flags",
    [
        "--xla_gpu_enable_command_buffer= --xla_gpu_enable_command_buffer=true",
        "--xla_gpu_enable_command_buffer=true --xla_gpu_enable_command_buffer=",
        "--xla_gpu_enable_command_buffer= --xla_gpu_enable_command_buffer=",
        "--xla_gpu_enable_command_buffer=false",
        "--xla_gpu_enable_command_buffer",
        "--noxla_gpu_enable_command_buffer",
    ],
)
def test_conflicting_command_buffer_assignments_fail_before_backend(monkeypatch, flags):
    environment = _environment(monkeypatch, process_flags=flags)
    dependencies = list(_dependencies())
    dependencies[0].default_backend = lambda: (_ for _ in ()).throw(AssertionError("backend was used"))

    with pytest.raises(RuntimeError, match="sole empty assignment"):
        _PROBE._run_rocm(
            io.StringIO(),
            lambda: dict(_CLEAN_SAFETY),
            _PROBE._zero_counters(),
            environment=environment,
            _dependencies=tuple(dependencies),
        )


def test_returned_xla_flags_mismatch_fails_before_backend(monkeypatch):
    environment = _environment(
        monkeypatch,
        process_flags="--xla_gpu_enable_command_buffer=",
        returned_flags="--unrelated=true --xla_gpu_enable_command_buffer=",
    )
    dependencies = list(_dependencies())
    dependencies[0].default_backend = lambda: (_ for _ in ()).throw(AssertionError("backend was used"))

    with pytest.raises(RuntimeError, match="does not match the process"):
        _PROBE._run_rocm(
            io.StringIO(),
            lambda: dict(_CLEAN_SAFETY),
            _PROBE._zero_counters(),
            environment=environment,
            _dependencies=tuple(dependencies),
        )


def test_shlex_proof_accepts_unrelated_flags_but_only_one_exact_disable(monkeypatch):
    secret_path = "/home/private-user/private-repo/compiler-dump"
    flags = f"--xla_dump_to={secret_path} --xla_gpu_enable_command_buffer= " "--lookalike_extra=false"
    proof = _PROBE._prove_command_buffers_disabled(_environment(monkeypatch, process_flags=flags))

    assert proof["process_matches_returned"] is True
    assert proof["token_count"] == 3
    assert proof["command_buffer_assignment_count"] == 1
    assert proof["sole_assignment_is_empty"] is True
    assert flags not in json.dumps(proof)
    assert secret_path not in json.dumps(proof)


def test_normal_environment_artifact_redacts_raw_xla_paths(monkeypatch):
    secret_path = "/home/private-user/private-repo/compiler-dump"
    effective_flags = f"--xla_dump_to={secret_path} --xla_gpu_enable_command_buffer="
    monkeypatch.setenv("XLA_FLAGS", effective_flags)
    configured = {
        "JAX_PLATFORMS": "rocm",
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
        "JAX_ROCM_VISIBLE_DEVICES": "0",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_CLIENT_MEM_FRACTION": "0.75",
        "SKYRL_ROCM_PALLAS_ATTENTION": "1",
        "XLA_FLAGS_original": f"--xla_dump_to={secret_path}",
        "XLA_FLAGS_effective": effective_flags,
    }

    @contextmanager
    def guard():
        yield dict(_CLEAN_SAFETY)

    monkeypatch.setattr(_PROBE, "_assert_fresh_accelerator_process", lambda: None)
    monkeypatch.setattr(_PROBE, "_configure_rocm_environment", lambda: configured)
    monkeypatch.setattr(
        _PROBE,
        "_load_safety_helpers",
        lambda: (guard, lambda: dict(_CLEAN_SAFETY)),
    )
    monkeypatch.setattr(_PROBE, "_run_rocm", lambda *_args, **_kwargs: 0)
    output = io.StringIO()

    result = _PROBE._execute(SimpleNamespace(platform="rocm", allow_gpu=True), output)

    assert result == 0
    artifact = output.getvalue()
    assert secret_path not in artifact
    assert effective_flags not in artifact
    assert str(_REPO) not in artifact
    environment_record = next(
        record for record in map(json.loads, artifact.splitlines()) if record["record_type"] == "environment"
    )
    manifest = environment_record["environment"]
    assert manifest["raw_xla_flags_emitted"] is False
    assert manifest["fixed_values_match_expected"] is True
    assert manifest["xla_flags_effective"]["sha256"] == hashlib.sha256(effective_flags.encode()).hexdigest()


def test_delegated_compile_error_text_is_redacted_before_jsonl_write():
    secret = f"PRIVATE_SECRET_SENTINEL_COMPILE {str(_REPO)}/compiler-cache"
    output = io.StringIO()
    writer = _PROBE._PrivateJsonlWriter(output)

    writer.write(
        json.dumps(
            {
                "record_type": "forward_compiled",
                "compiled_memory": {
                    "available": False,
                    "error": secret,
                },
            }
        )
        + "\n"
    )

    artifact = output.getvalue()
    assert secret not in artifact
    assert str(_REPO) not in artifact
    record = json.loads(artifact)
    summary = record["compiled_memory"]["error"]
    encoded = secret.encode("utf-8")
    assert summary == {
        "text_redacted": True,
        "utf8_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _configured_environment_for_execute():
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
        "XLA_FLAGS_original": "",
        "XLA_FLAGS_effective": _PROBE._DISABLED_COMMAND_BUFFER_FLAG,
    }


def _assert_redacted_terminal_error(
    output,
    *,
    message,
    error_type,
    stage,
):
    artifact = output.getvalue()
    records = [json.loads(line) for line in artifact.splitlines()]
    error = records[-1]
    encoded = message.encode("utf-8")
    assert error["record_type"] == "error"
    assert error["stage"] == stage
    assert error["status"] == "failed_closed"
    assert error["error_type"] == error_type
    assert error["message_redacted"] is True
    assert error["message_utf8_bytes"] == len(encoded)
    assert error["message_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert "message" not in error
    assert "PRIVATE_SECRET_SENTINEL" not in artifact
    assert "/home/" not in artifact
    assert str(_REPO) not in artifact
    return records


def test_environment_configuration_error_is_digest_only(monkeypatch):
    message = f"PRIVATE_SECRET_SENTINEL_ENV {str(_REPO)}/private-config"
    monkeypatch.setattr(_PROBE, "_assert_fresh_accelerator_process", lambda: None)
    monkeypatch.setattr(
        _PROBE,
        "_configure_rocm_environment",
        lambda: (_ for _ in ()).throw(ValueError(message)),
    )
    output = io.StringIO()

    result = _PROBE._execute(SimpleNamespace(platform="rocm", allow_gpu=True), output)

    assert result == 1
    _assert_redacted_terminal_error(
        output,
        message=message,
        error_type="ValueError",
        stage="bounded_environment",
    )


def test_runtime_error_is_digest_only_and_postflight_precedes_error(monkeypatch):
    message = f"PRIVATE_SECRET_SENTINEL_RUNTIME {str(_REPO)}/private-runtime"

    @contextmanager
    def guard():
        yield dict(_CLEAN_SAFETY)

    monkeypatch.setattr(_PROBE, "_assert_fresh_accelerator_process", lambda: None)
    monkeypatch.setattr(
        _PROBE,
        "_configure_rocm_environment",
        _configured_environment_for_execute,
    )
    monkeypatch.setattr(
        _PROBE,
        "_load_safety_helpers",
        lambda: (guard, lambda: dict(_CLEAN_SAFETY)),
    )
    monkeypatch.setattr(
        _PROBE,
        "_run_rocm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(message)),
    )
    output = io.StringIO()

    result = _PROBE._execute(SimpleNamespace(platform="rocm", allow_gpu=True), output)

    assert result == 1
    records = _assert_redacted_terminal_error(
        output,
        message=message,
        error_type="RuntimeError",
        stage="runtime",
    )
    assert [record["record_type"] for record in records[-2:]] == [
        "safety_postflight",
        "error",
    ]


def test_safety_postflight_error_is_digest_only(monkeypatch):
    message = f"PRIVATE_SECRET_SENTINEL_POSTFLIGHT {str(_REPO)}/private-postflight"

    @contextmanager
    def guard():
        yield dict(_CLEAN_SAFETY)

    monkeypatch.setattr(_PROBE, "_assert_fresh_accelerator_process", lambda: None)
    monkeypatch.setattr(
        _PROBE,
        "_configure_rocm_environment",
        _configured_environment_for_execute,
    )
    monkeypatch.setattr(
        _PROBE,
        "_load_safety_helpers",
        lambda: (
            guard,
            lambda: (_ for _ in ()).throw(OSError(message)),
        ),
    )
    monkeypatch.setattr(_PROBE, "_run_rocm", lambda *_args, **_kwargs: 0)
    output = io.StringIO()

    result = _PROBE._execute(SimpleNamespace(platform="rocm", allow_gpu=True), output)

    assert result == 1
    records = _assert_redacted_terminal_error(
        output,
        message=message,
        error_type="OSError",
        stage="safety_postflight",
    )
    assert not any(record["record_type"] == "completed" for record in records)


def test_rocm_execute_refuses_a_process_that_already_imported_jax(monkeypatch):
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
    ("candidate", "replay", "passed"),
    [
        (
            np.asarray([1.0, -2.0], dtype=np.float32),
            np.asarray([1.0, -2.0], dtype=np.float32),
            True,
        ),
        (
            np.asarray([1.0, -2.0], dtype=np.float32),
            np.asarray([1.0, -2.001], dtype=np.float32),
            False,
        ),
        (
            np.asarray([1.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float64),
            False,
        ),
    ],
)
def test_replay_equality_is_exact(candidate, replay, passed):
    assert _PROBE._replay_equality(candidate, replay)["passed"] is passed


def test_source_has_no_early_jax_import_backward_or_accelerator_rng():
    module = ast.parse(_PROBE_PATH.read_text(encoding="utf-8"))
    imports = [node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    roots = {alias.name.partition(".")[0] for node in imports for alias in node.names}
    assert "jax" not in roots
    assert "skyrl" not in roots

    full_source = _PROBE_PATH.read_text(encoding="utf-8")
    assert "jax.vjp" not in full_source
    assert "jax.grad" not in full_source
    assert "jax.random" not in full_source
    assert "_chunked_reference_attention" not in full_source
    run_source = inspect.getsource(_PROBE._run_rocm)
    assert run_source.index("_prove_command_buffers_disabled") < run_source.index("import jax")
