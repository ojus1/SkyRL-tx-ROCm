from __future__ import annotations

import argparse
import asyncio
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import tinker

_BENCH_PATH = Path(__file__).parents[2] / "rocm" / "bench_grpo_e2e.py"
_SPEC = importlib.util.spec_from_file_location("bench_grpo_e2e", _BENCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_BENCH = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BENCH
_SPEC.loader.exec_module(_BENCH)


def _sample_response() -> SimpleNamespace:
    return SimpleNamespace(
        sequences=[
            SimpleNamespace(
                tokens=list(range(100, 116)),
                logprobs=[-1.0 - index / 100 for index in range(16)],
                stop_reason="length",
            ),
            SimpleNamespace(
                tokens=list(range(200, 216)),
                logprobs=[-2.0 - index / 100 for index in range(16)],
                stop_reason=SimpleNamespace(value="length"),
            ),
        ]
    )


def test_canonical_sampling_specialization_records_full_generator_shape():
    specialization = _BENCH._sampling_specialization(
        prompt_tokens=49,
        max_new_tokens=16,
        group_size=2,
        sample_max_num_sequences=1,
        top_k=-1,
        top_p=1.0,
        prompt_logprobs=False,
        stop_token_width=0,
    )

    assert specialization == {
        "expanded_sample_count": 2,
        "effective_sample_microbatch_size": 1,
        "generator_call_count": 2,
        "generator_call_count_evidence": (
            "explicit_sequential_single_sample_requests"
        ),
        "prompt_tokens": 49,
        "prompt_bucket_tokens": 64,
        "total_kv_bucket_tokens": 96,
        "max_new_tokens": 16,
        "max_top_k": 0,
        "use_top_p": False,
        "prompt_logprobs": False,
        "stop_token_width": 0,
        "server_contract": (
            "effective_sample_microbatch_size must equal the JAX backend "
            "sample_max_num_sequences setting"
        ),
    }


def test_sample_extraction_accepts_exact_distinct_finite_group():
    sampled = _BENCH._extract_sampled_group(_sample_response())

    assert sampled.completions == (
        tuple(range(100, 116)),
        tuple(range(200, 216)),
    )
    assert sampled.logprobs[0] == pytest.approx(
        tuple(-1.0 - index / 100 for index in range(16))
    )
    assert sampled.stop_reasons == ("length", "length")


@pytest.mark.parametrize(
    ("mutate", "error", "message"),
    [
        (
            lambda response: response.sequences.pop(),
            ValueError,
            "wrong number of sequences",
        ),
        (
            lambda response: response.sequences[0].tokens.pop(),
            ValueError,
            "exactly max_new_tokens",
        ),
        (
            lambda response: setattr(
                response.sequences[1],
                "tokens",
                list(response.sequences[0].tokens),
            ),
            RuntimeError,
            "not distinct",
        ),
        (
            lambda response: response.sequences[1].logprobs.__setitem__(
                3, float("nan")
            ),
            FloatingPointError,
            "non-finite",
        ),
        (
            lambda response: response.sequences[1].logprobs.__setitem__(
                3, float("inf")
            ),
            FloatingPointError,
            "non-finite",
        ),
        (
            lambda response: response.sequences[1].logprobs.__setitem__(3, 0.1),
            ValueError,
            "positive log-probability",
        ),
        (
            lambda response: setattr(response.sequences[1], "stop_reason", "stop"),
            ValueError,
            "exact token limit",
        ),
    ],
)
def test_sample_extraction_rejects_invalid_sampler_evidence(mutate, error, message):
    response = _sample_response()
    mutate(response)

    with pytest.raises(error, match=message):
        _BENCH._extract_sampled_group(response)


def test_real_rollouts_have_exact_causal_shift_and_old_policy_alignment():
    prompt = tuple(range(10, 59))
    sampled = _BENCH.SampledGroup(
        # Deliberately put the lexicographically larger rollout first.
        completions=(tuple(range(200, 216)), tuple(range(100, 116))),
        logprobs=(
            tuple(-0.1 - index / 100 for index in range(16)),
            tuple(-1.1 - index / 100 for index in range(16)),
        ),
        stop_reasons=("length", "length"),
    )

    batch = _BENCH._build_real_rollouts(prompt, sampled)

    assert batch.context == 64
    assert batch.rewards == (1.0, 0.0)
    assert batch.advantages == (0.5, -0.5)
    assert len(batch.datums) == 2
    for datum, completion, old_action_logprobs, advantage in zip(
        batch.datums,
        sampled.completions,
        sampled.logprobs,
        batch.advantages,
        strict=True,
    ):
        full_sequence = [*prompt, *completion]
        assert datum.model_input.length == 64
        assert datum.model_input.chunks[0].tokens == full_sequence[:-1]
        assert datum.loss_fn_inputs["target_tokens"].tolist() == full_sequence[1:]
        assert datum.loss_fn_inputs["logprobs"].tolist() == pytest.approx(
            [0.0] * 48 + list(old_action_logprobs)
        )
        assert datum.loss_fn_inputs["advantages"].tolist() == (
            [0.0] * 48 + [advantage] * 16
        )
        assert "weights" not in datum.loss_fn_inputs


class _FakeFuture:
    def __init__(self, events, label, result, error=None):
        self.events = events
        self.label = label
        self.result = result
        self.error = error

    async def result_async(self):
        self.events.append(f"resolve:{self.label}")
        if self.error is not None:
            raise self.error
        return self.result


class _FakeSamplingClient:
    def __init__(self, events, *, fail_sample=False):
        self.events = events
        self.fail_sample = fail_sample
        self.sample_count = 0

    async def sample_async(self, **kwargs):
        self.events.append("sample")
        assert kwargs["prompt"].length == 49
        assert kwargs["num_samples"] == 1
        assert kwargs["include_prompt_logprobs"] is False
        assert kwargs["topk_prompt_logprobs"] == 0
        params = kwargs["sampling_params"]
        assert params.max_tokens == 16
        assert params.temperature == 1.0
        assert params.top_k == -1
        assert params.top_p == 1.0
        assert params.stop is None
        assert params.seed == 7 + self.sample_count
        if self.fail_sample:
            raise RuntimeError("synthetic sample failure")
        response = SimpleNamespace(
            sequences=[_sample_response().sequences[self.sample_count]]
        )
        self.sample_count += 1
        return response


class _FakeTrainingClient:
    def __init__(self, events, *, failures=frozenset()):
        self.events = events
        self.failures = failures
        self.snapshot_count = 0
        self.sampling_client = _FakeSamplingClient(
            events, fail_sample="sample" in failures
        )

    async def save_weights_and_get_sampling_client_async(self, *, retry_config):
        assert retry_config.enable_retry_logic is False
        self.snapshot_count += 1
        self.events.append(f"snapshot:{self.snapshot_count}")
        if self.snapshot_count == 2 and "post_snapshot" in self.failures:
            raise RuntimeError("synthetic post-snapshot failure")
        return self.sampling_client

    async def forward_backward_async(self, data, loss_fn):
        self.events.append("submit:forward_backward")
        assert loss_fn == "importance_sampling"
        assert len(data) == 2
        outputs = [
            {
                "logprobs": tinker.TensorData(
                    data=[-1.5] * datum.model_input.length,
                    dtype="float32",
                    shape=[datum.model_input.length],
                )
            }
            for datum in data
        ]
        error = (
            RuntimeError("synthetic learner failure")
            if "learner" in self.failures
            else None
        )
        return _FakeFuture(
            self.events,
            "forward_backward",
            SimpleNamespace(loss_fn_outputs=outputs),
            error,
        )

    async def optim_step_async(self, adam):
        self.events.append("submit:optim")
        assert adam.learning_rate == 1e-5
        if "optim_submit" in self.failures:
            raise RuntimeError("synthetic optimizer submission failure")
        metrics = {"skyrl.ai/grad_norm": 1.0}
        if "zero_grad_norm" in self.failures:
            metrics["skyrl.ai/grad_norm"] = 0.0
        elif "nonfinite_grad_norm" in self.failures:
            metrics["skyrl.ai/grad_norm"] = float("nan")
        elif "missing_grad_norm" in self.failures:
            metrics = {}
        error = (
            RuntimeError("synthetic optimizer result failure")
            if "optim_result" in self.failures
            else None
        )
        return _FakeFuture(
            self.events,
            "optim",
            SimpleNamespace(metrics=metrics),
            error,
        )

    def _guaranteed_model_id(self):
        return "model_real_sampler_gate"


class _FakeServiceClient:
    def __init__(self, events, training_client, failures, **kwargs):
        self.events = events
        self.training_client = training_client
        self.failures = failures
        assert kwargs["base_url"] == "http://127.0.0.1:8001"

    async def create_lora_training_client_async(self, **kwargs):
        self.events.append("adapter_create")
        assert kwargs["base_model"] == _BENCH.MODEL
        assert kwargs["rank"] == 8
        assert kwargs["seed"] == 7
        assert kwargs["train_mlp"] is True
        assert kwargs["train_attn"] is True
        assert kwargs["train_unembed"] is False
        if "create" in self.failures:
            raise RuntimeError("synthetic adapter creation failure")
        return self.training_client


class _FakeServerSeal:
    contract_sha256 = "a" * 64

    def verified_model_snapshot(self):
        return Path("/verified/models--Qwen--Qwen3.5-4B/snapshots") / (
            _BENCH.EXPECTED_MODEL_REVISION
        )

    def as_record(self):
        return {
            "record_type": "server_attestation",
            "schema_version": 1,
            "status": "passed",
            "attestation_sha256": self.contract_sha256,
        }


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        base_url="http://127.0.0.1:8001",
        server_pid=100,
        run_id="cpu-real-sampler-gate",
        seed=7,
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
    )


def _records(output: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in output.getvalue().splitlines()]


def _install_run_mocks(monkeypatch, *, failure=None):
    failures = (
        frozenset()
        if failure is None
        else frozenset((failure,))
        if isinstance(failure, str)
        else frozenset(failure)
    )
    events: list[str] = []
    training_client = _FakeTrainingClient(events, failures=failures)
    monkeypatch.setattr(
        _BENCH,
        "_build_prompt",
        lambda model_snapshot, tokenizer=None: tuple(range(10, 59)),
    )
    monkeypatch.setattr(
        _BENCH,
        "_client_source_revision",
        lambda: ("b" * 40, "c" * 40),
    )
    monkeypatch.setattr(
        _BENCH, "git_metadata", lambda path: {"path": str(path), "dirty": False}
    )
    monkeypatch.setattr(
        _BENCH,
        "package_version",
        lambda name: (
            _BENCH.EXPECTED_TINKER_VERSION if name == "tinker" else "test-version"
        ),
    )
    monkeypatch.setattr(_BENCH, "safe_accelerator_environment", lambda: {})

    server_seal = _FakeServerSeal()

    def fake_attest_local_server(
        *,
        server_pid,
        base_url,
        expected_git_head,
        expected_git_tree,
        expected_repo_root,
        require_startup_cache,
    ):
        events.append("server_attestation")
        assert server_pid == 100
        assert base_url == "http://127.0.0.1:8001"
        assert expected_git_head == "b" * 40
        assert expected_git_tree == "c" * 40
        assert expected_repo_root == _BENCH.REPO
        assert require_startup_cache is True
        if "attestation" in failures:
            raise RuntimeError("synthetic server attestation failure")
        return server_seal

    def fake_revalidate_local_server(seal):
        events.append("server_revalidation")
        assert seal is server_seal
        if "revalidation" in failures:
            raise RuntimeError("synthetic server revalidation failure")
        return {
            "record_type": "server_revalidation",
            "schema_version": 1,
            "status": "passed",
            "attestation_sha256": seal.contract_sha256,
        }

    monkeypatch.setattr(_BENCH, "attest_local_server", fake_attest_local_server)
    monkeypatch.setattr(_BENCH, "revalidate_local_server", fake_revalidate_local_server)
    monkeypatch.setattr(
        _BENCH.tinker,
        "ServiceClient",
        lambda **kwargs: _FakeServiceClient(
            events,
            training_client=training_client,
            failures=failures,
            **kwargs,
        ),
    )

    async def fake_unload(service_client, client):
        events.append("unload")
        assert service_client.training_client is client
        if "cleanup" in failures:
            raise RuntimeError("synthetic cleanup failure")
        return {"unloaded": True}

    monkeypatch.setattr(_BENCH, "unload_adapter", fake_unload)
    return events, training_client


def test_run_executes_real_sampler_update_snapshot_lifecycle_in_order(
    monkeypatch, capsys
):
    events, training_client = _install_run_mocks(monkeypatch)
    output = io.StringIO()

    asyncio.run(_BENCH._run(_args(), output))

    assert events == [
        "server_attestation",
        "adapter_create",
        "snapshot:1",
        "sample",
        "sample",
        "submit:forward_backward",
        "submit:optim",
        "resolve:forward_backward",
        "resolve:optim",
        "snapshot:2",
        "unload",
        "server_revalidation",
    ]
    assert training_client.snapshot_count == 2
    records = _records(output)
    assert [record["record_type"] for record in records] == [
        "manifest",
        "phase",
        "server_attestation",
        "phase",
        "phase",
        "phase",
        "phase",
        "phase",
        "sampled_group",
        "phase",
        "learner_result",
        "phase",
        "iteration",
        "cleanup",
        "phase",
        "server_revalidation",
        "summary",
    ]
    assert [
        record["phase"] for record in records if record["record_type"] == "phase"
    ] == [
        "server_attestation",
        "adapter_create",
        "initial_sampler_snapshot",
        "sample",
        "validate_sample",
        "grade_and_build",
        "learner_update",
        "post_update_sampler_snapshot",
        "server_revalidation",
    ]
    sampled_record = next(
        record for record in records if record["record_type"] == "sampled_group"
    )
    assert records[0]["client"]["tinker"] == _BENCH.EXPECTED_TINKER_VERSION
    assert records[0]["retry_safety"] == {
        "sdk_version": _BENCH.EXPECTED_TINKER_VERSION,
        "scope": "colocated_jax_ingress_future_enqueue",
        "duplicate_submission_creates_one_future": True,
        "backend_effect_exactly_once_across_engine_crash": False,
        "external_sampling_dispatch_covered": False,
    }
    assert records[0]["sampling"]["request_plan"] == {
        "execution": "sequential",
        "request_count": 2,
        "num_samples_per_request": 1,
        "seeds": [7, 8],
    }
    assert sampled_record["completion_tokens"] == [
        list(range(100, 116)),
        list(range(200, 216)),
    ]
    assert sampled_record["sampling_logprobs"][0] == pytest.approx(
        [-1.0 - index / 100 for index in range(16)]
    )
    cleanup_record = next(
        record for record in records if record["record_type"] == "cleanup"
    )
    assert cleanup_record["adapter_unloaded"] is True
    assert "model_id" not in cleanup_record
    assert "response" not in cleanup_record
    assert records[-1]["sampler_snapshots_completed"] == 2
    assert records[-1]["steady_state_throughput_available"] is False
    assert records[-1]["server_attestation_sha256"] == "a" * 64
    assert records[-1]["server_revalidated"] is True
    assert records[-1]["adapter_lifecycle_seconds"] > 0.0
    assert "cold_end_to_end_seconds" not in records[-1]
    assert "tokens_per_second" not in records[-1]
    assert json.loads(capsys.readouterr().out)["record_type"] == "summary"


def test_tokenizer_uses_exact_verified_snapshot_even_with_stale_main_ref(
    monkeypatch, tmp_path
):
    model_cache = tmp_path / _BENCH.EXPECTED_MODEL_CACHE_DIRECTORY
    snapshot = model_cache / "snapshots" / _BENCH.EXPECTED_MODEL_REVISION
    snapshot.mkdir(parents=True)
    refs = model_cache / "refs"
    refs.mkdir()
    (refs / "main").write_text("0" * 40, encoding="ascii")
    calls = []
    sentinel = object()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model, **kwargs):
            calls.append((model, kwargs))
            return sentinel

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )

    assert _BENCH._get_tokenizer(snapshot) is sentinel
    assert calls == [(str(snapshot), {"local_files_only": True})]


def test_tokenizer_rejects_non_attested_snapshot_revision(tmp_path):
    stale_snapshot = (
        tmp_path
        / _BENCH.EXPECTED_MODEL_CACHE_DIRECTORY
        / "snapshots"
        / ("0" * 40)
    )
    stale_snapshot.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="not canonical"):
        _BENCH._get_tokenizer(stale_snapshot)


@pytest.mark.parametrize(
    "stdout_error",
    [
        BrokenPipeError("synthetic broken pipe"),
        ValueError("I/O operation on closed file."),
    ],
)
def test_stdout_failure_after_summary_does_not_invalidate_durable_gate(
    monkeypatch, stdout_error
):
    _install_run_mocks(monkeypatch)
    output = io.StringIO()

    def fail_stdout(*args, **kwargs):
        del args, kwargs
        raise stdout_error

    monkeypatch.setattr("builtins.print", fail_stdout)

    asyncio.run(_BENCH._run(_args(), output))

    records = _records(output)
    assert records[-1]["record_type"] == "summary"
    assert sum(record["record_type"] == "summary" for record in records) == 1


def test_run_rejects_unattested_server_before_adapter_creation(monkeypatch):
    events, _ = _install_run_mocks(monkeypatch, failure="attestation")
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="synthetic server attestation failure"):
        asyncio.run(_BENCH._run(_args(), output))

    assert events == ["server_attestation"]
    records = _records(output)
    assert [record["record_type"] for record in records] == ["manifest", "phase"]
    assert records[-1]["phase"] == "server_attestation"
    assert records[-1]["status"] == "failed"


def test_run_rejects_dirty_or_unreadable_client_source_before_evidence(monkeypatch):
    events, _ = _install_run_mocks(monkeypatch)
    monkeypatch.setattr(
        _BENCH,
        "_client_source_revision",
        lambda: (_ for _ in ()).throw(RuntimeError("synthetic dirty source")),
    )
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="synthetic dirty source"):
        asyncio.run(_BENCH._run(_args(), output))

    assert events == []
    assert output.getvalue() == ""


def test_run_rejects_drifted_tinker_retry_contract_before_evidence(monkeypatch):
    events, _ = _install_run_mocks(monkeypatch)
    monkeypatch.setattr(_BENCH, "package_version", lambda name: "0.23.0")
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="require exactly tinker 0.22.4"):
        asyncio.run(_BENCH._run(_args(), output))

    assert events == []
    assert output.getvalue() == ""


def test_post_seal_attestation_phase_write_failure_still_revalidates(monkeypatch):
    events, _ = _install_run_mocks(monkeypatch)
    output = _FailOnWrite(fail_on_write=2)

    with pytest.raises(OSError, match="synthetic evidence write failure"):
        asyncio.run(_BENCH._run(_args(), output))

    assert events == ["server_attestation", "server_revalidation"]


def test_adapter_create_action_failure_revalidates_without_cleanup(monkeypatch):
    events, _ = _install_run_mocks(monkeypatch, failure="create")
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="synthetic adapter creation failure"):
        asyncio.run(_BENCH._run(_args(), output))

    assert events == [
        "server_attestation",
        "adapter_create",
        "server_revalidation",
    ]
    records = _records(output)
    assert not any(record["record_type"] == "cleanup" for record in records)
    assert not any(record["record_type"] == "summary" for record in records)


def test_inconsistent_create_result_is_cleaned_up_then_revalidated(monkeypatch):
    events, _ = _install_run_mocks(monkeypatch)
    original_timed_phase = _BENCH._timed_async_phase

    async def return_inconsistent_create_result(*args, **kwargs):
        result = await original_timed_phase(*args, **kwargs)
        if kwargs["phase"] == "adapter_create":
            return object()
        return result

    monkeypatch.setattr(
        _BENCH,
        "_timed_async_phase",
        return_inconsistent_create_result,
    )
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="inconsistent client"):
        asyncio.run(_BENCH._run(_args(), output))

    assert events == [
        "server_attestation",
        "adapter_create",
        "unload",
        "server_revalidation",
    ]
    cleanup = next(
        record for record in _records(output) if record["record_type"] == "cleanup"
    )
    assert cleanup["adapter_unloaded"] is True


def test_run_requires_same_server_after_cleanup_before_summary(monkeypatch):
    events, _ = _install_run_mocks(monkeypatch, failure="revalidation")
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="synthetic server revalidation failure"):
        asyncio.run(_BENCH._run(_args(), output))

    assert events[-2:] == ["unload", "server_revalidation"]
    records = _records(output)
    cleanup = next(record for record in records if record["record_type"] == "cleanup")
    assert cleanup["adapter_unloaded"] is True
    assert records[-1]["record_type"] == "phase"
    assert records[-1]["phase"] == "server_revalidation"
    assert records[-1]["status"] == "failed"
    assert not any(record["record_type"] == "summary" for record in records)


@pytest.mark.parametrize(
    ("failure", "error_type", "message", "forbidden_events"),
    [
        (
            "sample",
            RuntimeError,
            "synthetic sample failure",
            {"submit:forward_backward", "snapshot:2"},
        ),
        ("learner", RuntimeError, "synthetic learner failure", {"snapshot:2"}),
        (
            "optim_submit",
            RuntimeError,
            "synthetic optimizer submission failure",
            {"snapshot:2"},
        ),
        (
            "optim_result",
            RuntimeError,
            "synthetic optimizer result failure",
            {"snapshot:2"},
        ),
        ("zero_grad_norm", FloatingPointError, "finite and positive", {"snapshot:2"}),
        (
            "nonfinite_grad_norm",
            FloatingPointError,
            "finite and positive",
            {"snapshot:2"},
        ),
        ("missing_grad_norm", ValueError, "missing the numeric", {"snapshot:2"}),
        (
            "post_snapshot",
            RuntimeError,
            "synthetic post-snapshot failure",
            set(),
        ),
        ("cleanup", RuntimeError, "synthetic cleanup failure", set()),
    ],
)
def test_run_failure_paths_always_attempt_cleanup_and_never_emit_summary(
    monkeypatch, failure, error_type, message, forbidden_events
):
    events, _ = _install_run_mocks(monkeypatch, failure=failure)
    output = io.StringIO()

    with pytest.raises(error_type, match=message):
        asyncio.run(_BENCH._run(_args(), output))

    records = _records(output)
    assert "unload" in events
    assert events[-1] == "server_revalidation"
    assert forbidden_events.isdisjoint(events)
    assert not any(record["record_type"] == "summary" for record in records)
    cleanup_record = next(
        record for record in records if record["record_type"] == "cleanup"
    )
    if failure == "cleanup":
        assert cleanup_record["adapter_unloaded"] is False
        iteration = next(
            record for record in records if record["record_type"] == "iteration"
        )
        assert iteration["status"] == "completed"
    else:
        assert cleanup_record["adapter_unloaded"] is True
        iteration = next(
            record for record in records if record["record_type"] == "iteration"
        )
        assert iteration["status"] == "failed"
        assert iteration["error_type"] == error_type.__name__
    if failure == "learner":
        assert "resolve:optim" in events
        assert events.index("resolve:optim") < events.index("unload")
    if failure == "optim_submit":
        assert "resolve:forward_backward" in events
        assert events.index("resolve:forward_backward") < events.index("unload")


def test_both_learner_futures_are_settled_and_first_error_wins(monkeypatch):
    events, _ = _install_run_mocks(
        monkeypatch,
        failure={"learner", "optim_result"},
    )
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="synthetic learner failure") as raised:
        asyncio.run(_BENCH._run(_args(), output))

    assert raised.value.__notes__ == [
        "the queued optimizer future also failed while settling: RuntimeError"
    ]
    assert events.index("resolve:forward_backward") < events.index("resolve:optim")
    assert events.index("resolve:optim") < events.index("unload")
    assert events[-1] == "server_revalidation"


class _FailOnWrite(io.StringIO):
    def __init__(self, fail_on_write: int):
        super().__init__()
        self.fail_on_write = fail_on_write
        self.write_count = 0

    def write(self, value: str) -> int:
        self.write_count += 1
        if self.write_count == self.fail_on_write:
            raise OSError("synthetic evidence write failure")
        return super().write(value)


def test_async_phase_action_error_wins_over_failure_evidence_write():
    async def fail_action():
        raise ValueError("synthetic async action failure")

    with pytest.raises(ValueError, match="synthetic async action failure") as raised:
        asyncio.run(
            _BENCH._timed_async_phase(
                _FailOnWrite(fail_on_write=1),
                run_id="cpu-validation",
                phase="synthetic_async",
                action=fail_action,
                durations={},
            )
        )

    assert raised.value.__notes__ == [
        "phase failure evidence recording also failed: OSError"
    ]


def test_sync_phase_action_error_wins_over_failure_evidence_write():
    def fail_action():
        raise ValueError("synthetic sync action failure")

    with pytest.raises(ValueError, match="synthetic sync action failure") as raised:
        _BENCH._timed_sync_phase(
            _FailOnWrite(fail_on_write=1),
            run_id="cpu-validation",
            phase="synthetic_sync",
            action=fail_action,
            durations={},
        )

    assert raised.value.__notes__ == [
        "phase failure evidence recording also failed: OSError"
    ]


def test_adapter_is_unloaded_when_completed_create_phase_cannot_be_recorded(
    monkeypatch,
):
    events, _ = _install_run_mocks(monkeypatch)
    output = _FailOnWrite(fail_on_write=4)

    with pytest.raises(OSError, match="synthetic evidence write failure"):
        asyncio.run(_BENCH._run(_args(), output))

    assert events == [
        "server_attestation",
        "adapter_create",
        "unload",
        "server_revalidation",
    ]


def test_setup_write_error_wins_over_cleanup_and_revalidation_failures(monkeypatch):
    events, _ = _install_run_mocks(
        monkeypatch,
        failure={"cleanup", "revalidation"},
    )
    output = _FailOnWrite(fail_on_write=4)

    with pytest.raises(OSError, match="synthetic evidence write failure") as raised:
        asyncio.run(_BENCH._run(_args(), output))

    assert events == [
        "server_attestation",
        "adapter_create",
        "unload",
        "server_revalidation",
    ]
    assert raised.value.__notes__ == [
        "adapter cleanup also failed: RuntimeError",
        "server revalidation also failed: RuntimeError",
    ]


def test_iteration_error_wins_over_cleanup_and_revalidation_failures(monkeypatch):
    events, _ = _install_run_mocks(
        monkeypatch,
        failure={"sample", "cleanup", "revalidation"},
    )
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="synthetic sample failure") as raised:
        asyncio.run(_BENCH._run(_args(), output))

    assert events[-2:] == ["unload", "server_revalidation"]
    assert raised.value.__notes__ == [
        "adapter cleanup also failed: RuntimeError",
        "server revalidation also failed: RuntimeError",
    ]
    assert not any(record["record_type"] == "summary" for record in _records(output))


def test_cleanup_error_wins_over_revalidation_failure(monkeypatch):
    events, _ = _install_run_mocks(
        monkeypatch,
        failure={"cleanup", "revalidation"},
    )
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="synthetic cleanup failure") as raised:
        asyncio.run(_BENCH._run(_args(), output))

    assert events[-2:] == ["unload", "server_revalidation"]
    assert raised.value.__notes__ == ["server revalidation also failed: RuntimeError"]
    assert not any(record["record_type"] == "summary" for record in _records(output))


def test_adapter_is_unloaded_when_iteration_record_cannot_be_written(monkeypatch):
    events, _ = _install_run_mocks(monkeypatch)
    output = _FailOnWrite(fail_on_write=13)

    with pytest.raises(OSError, match="synthetic evidence write failure"):
        asyncio.run(_BENCH._run(_args(), output))

    assert "unload" in events
    assert events[-1] == "server_revalidation"
    records = _records(output)
    cleanup_record = next(
        record for record in records if record["record_type"] == "cleanup"
    )
    assert cleanup_record["adapter_unloaded"] is True
    assert not any(record["record_type"] == "summary" for record in records)


def test_revalidation_phase_write_failure_never_emits_summary(monkeypatch):
    events, _ = _install_run_mocks(monkeypatch)
    output = _FailOnWrite(fail_on_write=15)

    with pytest.raises(OSError, match="synthetic evidence write failure"):
        asyncio.run(_BENCH._run(_args(), output))

    assert events[-2:] == ["unload", "server_revalidation"]
    records = _records(output)
    assert records[-1]["record_type"] == "cleanup"
    assert not any(record["record_type"] == "summary" for record in records)


def test_revalidation_record_write_failure_never_emits_summary(monkeypatch):
    events, _ = _install_run_mocks(monkeypatch)
    output = _FailOnWrite(fail_on_write=16)

    with pytest.raises(OSError, match="synthetic evidence write failure"):
        asyncio.run(_BENCH._run(_args(), output))

    assert events[-2:] == ["unload", "server_revalidation"]
    records = _records(output)
    assert records[-1]["record_type"] == "phase"
    assert records[-1]["phase"] == "server_revalidation"
    assert records[-1]["status"] == "completed"
    assert not any(record["record_type"] == "summary" for record in records)


def _canonical_parser_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = _BENCH._build_parser()
    args = parser.parse_args(
        [
            "--one-iteration-gate",
            "--server-pid",
            "100",
            "--run-id",
            "cpu-validation",
            "--output",
            "unused.jsonl",
        ]
    )
    return parser, args


def test_argument_validation_accepts_largest_safe_group_base_seed():
    parser, args = _canonical_parser_args()
    args.seed = 2**31 - 2

    _BENCH._validate_args(parser, args)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("server_pid", 1),
        ("server_pid", 100.0),
        ("seed", 2**31 - 1),
        ("learning_rate", 0.0),
        ("learning_rate", 1e-50),
        ("adam_beta1", 1.0 - 1e-9),
        ("adam_beta2", 1.0 - 1e-9),
        ("adam_eps", 1e-50),
    ],
)
def test_argument_validation_rejects_unsafe_derived_seed_and_float32_values(
    field, value
):
    parser, args = _canonical_parser_args()
    setattr(args, field, value)

    with pytest.raises(SystemExit):
        _BENCH._validate_args(parser, args)


def test_main_never_reraises_a_raw_sdk_error_message(monkeypatch, tmp_path):
    secret = "Authorization: bearer-do-not-print"
    output_path = tmp_path / "redacted-error.jsonl"

    async def failed_run(args, output):
        del args, output
        raise RuntimeError(secret)

    monkeypatch.setattr(_BENCH, "_run", failed_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_BENCH_PATH),
            "--one-iteration-gate",
            "--server-pid",
            "100",
            "--run-id",
            "redacted-main",
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(SystemExit) as caught:
        _BENCH.main()

    assert secret not in str(caught.value)
    assert caught.value.__suppress_context__ is True
    written = output_path.read_text()
    assert secret not in written
    record = json.loads(written)
    assert record["error_type"] == "RuntimeError"
    assert record["error_message_redacted"] is True
