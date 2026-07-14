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

_BENCH_PATH = Path(__file__).parents[2] / "rocm" / "bench_grpo.py"
_SPEC = importlib.util.spec_from_file_location("bench_grpo", _BENCH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_BENCH = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BENCH
_SPEC.loader.exec_module(_BENCH)


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        # Stable, nonempty, and candidate-specific without depending on a model cache.
        return [11 + (ord(character) % 97) for character in text]


def test_fixed_rollouts_match_cookbook_shift_and_group_semantics():
    batch = _BENCH._build_fixed_rollouts(
        context=8,
        completion_tokens=3,
        group_size=2,
        sampling_logprob=-5.0,
        tokenizer=_FakeTokenizer(),
    )

    assert batch.rewards == (0.0, 1.0)
    assert batch.advantages == (-0.5, 0.5)
    assert len(batch.prompt_tokens) == 6
    assert len(batch.datums) == 2

    for datum, completion, scalar_advantage in zip(
        batch.datums, batch.completion_tokens, batch.advantages, strict=True
    ):
        model_tokens = datum.model_input.chunks[0].tokens
        target_tokens = datum.loss_fn_inputs["target_tokens"].tolist()
        old_logprobs = datum.loss_fn_inputs["logprobs"].tolist()
        advantages = datum.loss_fn_inputs["advantages"].tolist()

        full_sequence = [*batch.prompt_tokens, *completion]
        assert model_tokens == full_sequence[:-1]
        assert target_tokens == full_sequence[1:]
        assert old_logprobs == [0.0] * 5 + [-5.0] * 3
        assert advantages == [0.0] * 5 + [scalar_advantage] * 3
        assert "weights" not in datum.loss_fn_inputs


def test_one_token_completions_are_rollout_specific_and_do_not_cancel():
    batch = _BENCH._build_fixed_rollouts(
        context=4,
        completion_tokens=1,
        group_size=3,
        sampling_logprob=-5.0,
        tokenizer=_FakeTokenizer(),
    )

    assert len(set(batch.completion_tokens)) == 3
    assert [completion[0] for completion in batch.completion_tokens] == [59, 60, 61]

    maximum_group = _BENCH._build_fixed_rollouts(
        context=4,
        completion_tokens=1,
        group_size=16,
        sampling_logprob=-5.0,
        tokenizer=_FakeTokenizer(),
    )
    assert len(set(maximum_group.completion_tokens)) == 16


def test_local_api_defaults_missing_cookbook_weights_to_all_ones():
    batch = _BENCH._build_fixed_rollouts(
        context=8,
        completion_tokens=3,
        group_size=2,
        sampling_logprob=-5.0,
        tokenizer=_FakeTokenizer(),
    )
    datum = batch.datums[0]

    from skyrl.tinker import api

    api_datum = api.Datum(
        model_input={
            "chunks": [
                {"type": "encoded_text", "tokens": datum.model_input.chunks[0].tokens}
            ]
        },
        loss_fn_inputs={
            key: {"data": value.data} for key, value in datum.loss_fn_inputs.items()
        },
    )
    local_datum = api_datum.to_types()

    assert local_datum.loss_fn_inputs.weights.data == [1.0] * 8
    assert local_datum.loss_fn_inputs.advantages.data[-3:] == [-0.5] * 3


def test_manifest_argument_redaction_covers_compact_headers_credentials_and_urls():
    assert _BENCH.redacted_argv(
        [
            "--header",
            "Authorization: secret",
            "-HX-Api-Key:secret",
            "-usecret",
            "-U",
            "proxysecret",
            "--header=Cookie: secret",
            "TINKER_API_KEY=secret",
            "--endpoint=https://user:password@example.test/private/token?q=secret",
            "--ipv6=https://user:password@[2001:db8::1]:8443/private?q=secret",
        ]
    ) == [
        "--header",
        "<redacted>",
        "-H<redacted>",
        "-u<redacted>",
        "-U",
        "<redacted>",
        "--header=<redacted>",
        "TINKER_API_KEY=<redacted>",
        "--endpoint=https://example.test",
        "--ipv6=https://[2001:db8::1]:8443",
    ]


def test_accelerator_environment_records_device_selection_without_secrets(monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("GPU_DEVICE_ORDINAL", "0")
    monkeypatch.setenv("PJRT_DEVICE", "ROCM")
    monkeypatch.setenv("ROCR_SECRET_TOKEN", "do-not-record")

    environment = _BENCH.safe_accelerator_environment()

    assert environment["ROCR_VISIBLE_DEVICES"] == "0"
    assert environment["GPU_DEVICE_ORDINAL"] == "0"
    assert environment["PJRT_DEVICE"] == "ROCM"
    assert "ROCR_SECRET_TOKEN" not in environment


@pytest.mark.parametrize(
    "sampling_logprob", [-5.0001, float("-inf"), float("nan"), 0.1]
)
def test_fixed_rollouts_reject_unsafe_sampling_logprob(sampling_logprob):
    with pytest.raises(ValueError, match="sampling_logprob"):
        _BENCH._build_fixed_rollouts(
            context=8,
            completion_tokens=3,
            group_size=2,
            sampling_logprob=sampling_logprob,
            tokenizer=_FakeTokenizer(),
        )


def test_policy_metrics_use_completion_slice_even_for_zero_advantage_rollout():
    batch = _BENCH._build_fixed_rollouts(
        context=4,
        completion_tokens=2,
        group_size=3,
        sampling_logprob=-1.0,
        tokenizer=_FakeTokenizer(),
    )
    outputs = [
        {"logprobs": tinker.TensorData(data=[-2.0] * 4, dtype="float32", shape=[4])},
        {"logprobs": tinker.TensorData(data=[-1.5] * 4, dtype="float32", shape=[4])},
        {"logprobs": tinker.TensorData(data=[-1.0] * 4, dtype="float32", shape=[4])},
    ]

    metrics = _BENCH._policy_metrics(outputs, batch)

    assert metrics["action_target_logprob_mean"] == pytest.approx(-1.5)
    assert metrics["importance_ratio_min"] == pytest.approx(0.36787944117)
    assert metrics["importance_ratio_max"] == pytest.approx(1.0)
    assert metrics["policy_loss_mean"] == pytest.approx(-0.0526767132)


class _FakeFuture:
    def __init__(self, result):
        self._result = result

    async def result_async(self):
        return self._result


class _RaisingFuture:
    async def result_async(self):
        raise RuntimeError("synthetic learner failure")


class _FakeTrainingClient:
    def __init__(self):
        self.calls: list[str] = []

    async def forward_backward_async(self, data, loss_fn):
        self.calls.append(f"forward_backward:{loss_fn}:{len(data)}")
        outputs = [
            {
                "logprobs": tinker.TensorData(
                    data=[-2.0] * datum.model_input.length,
                    dtype="float32",
                    shape=[datum.model_input.length],
                )
            }
            for datum in data
        ]
        return _FakeFuture(SimpleNamespace(loss_fn_outputs=outputs))

    async def optim_step_async(self, adam):
        self.calls.append(f"optim:{adam.learning_rate}")
        return _FakeFuture(SimpleNamespace(metrics={"skyrl.ai/grad_norm": 1.0}))

    def _guaranteed_model_id(self):
        return "model_fixed"


class _FailingTrainingClient(_FakeTrainingClient):
    async def forward_backward_async(self, data, loss_fn):
        self.calls.append(f"forward_backward:{loss_fn}:{len(data)}")
        return _RaisingFuture()


class _FakeServiceClient:
    def __init__(self, training_client, **kwargs):
        del kwargs
        self.training_client = training_client

    async def create_lora_training_client_async(self, **kwargs):
        assert kwargs["base_model"] == _BENCH.MODEL
        assert kwargs["train_mlp"] is True
        assert kwargs["train_attn"] is True
        assert kwargs["train_unembed"] is False
        return self.training_client


def test_run_uses_importance_sampling_adam_and_cleanup_without_server(monkeypatch):
    batch = _BENCH._build_fixed_rollouts(
        context=4,
        completion_tokens=2,
        group_size=2,
        sampling_logprob=-1.0,
        tokenizer=_FakeTokenizer(),
    )
    training_client = _FakeTrainingClient()
    monkeypatch.setattr(_BENCH, "_build_fixed_rollouts", lambda *args, **kwargs: batch)
    monkeypatch.setattr(
        _BENCH.tinker,
        "ServiceClient",
        lambda **kwargs: _FakeServiceClient(training_client, **kwargs),
    )

    async def fake_unload(service_client, client):
        assert service_client.training_client is client
        return {"unloaded": True}

    monkeypatch.setattr(_BENCH, "unload_adapter", fake_unload)
    args = argparse.Namespace(
        context=4,
        completion_tokens=2,
        group_size=2,
        sampling_logprob=-1.0,
        base_url="http://127.0.0.1:8001",
        run_id="cpu-unit",
        warmup_steps=1,
        measured_steps=2,
        lora_rank=8,
        seed=0,
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
    )
    output = io.StringIO()

    asyncio.run(_BENCH._run(args, output))

    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [record["record_type"] for record in records] == [
        "manifest",
        "step",
        "step",
        "step",
        "cleanup",
        "summary",
    ]
    assert records[0]["sampling_performed"] is False
    assert records[0]["loss_fn"] == "importance_sampling"
    assert records[-2]["adapter_unloaded"] is True
    assert records[-1]["measured_steps"] == 2
    assert (
        training_client.calls
        == [
            "forward_backward:importance_sampling:2",
            "optim:1e-05",
        ]
        * 3
    )


def test_run_records_cleanup_failure_without_server(monkeypatch):
    batch = _BENCH._build_fixed_rollouts(
        context=4,
        completion_tokens=2,
        group_size=2,
        sampling_logprob=-1.0,
        tokenizer=_FakeTokenizer(),
    )
    training_client = _FakeTrainingClient()
    monkeypatch.setattr(_BENCH, "_build_fixed_rollouts", lambda *args, **kwargs: batch)
    monkeypatch.setattr(
        _BENCH.tinker,
        "ServiceClient",
        lambda **kwargs: _FakeServiceClient(training_client, **kwargs),
    )

    async def failed_unload(service_client, client):
        del service_client, client
        raise RuntimeError("synthetic unload failure")

    monkeypatch.setattr(_BENCH, "unload_adapter", failed_unload)
    args = argparse.Namespace(
        context=4,
        completion_tokens=2,
        group_size=2,
        sampling_logprob=-1.0,
        base_url="http://127.0.0.1:8001",
        run_id="cpu-cleanup-failure",
        warmup_steps=1,
        measured_steps=1,
        lora_rank=8,
        seed=0,
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
    )
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="synthetic unload failure"):
        asyncio.run(_BENCH._run(args, output))

    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert records[-1]["record_type"] == "cleanup"
    assert records[-1]["adapter_unloaded"] is False
    assert records[-1]["error_type"] == "RuntimeError"


def test_run_unloads_adapter_after_learner_failure_without_server(monkeypatch):
    batch = _BENCH._build_fixed_rollouts(
        context=4,
        completion_tokens=2,
        group_size=2,
        sampling_logprob=-1.0,
        tokenizer=_FakeTokenizer(),
    )
    training_client = _FailingTrainingClient()
    monkeypatch.setattr(_BENCH, "_build_fixed_rollouts", lambda *args, **kwargs: batch)
    monkeypatch.setattr(
        _BENCH.tinker,
        "ServiceClient",
        lambda **kwargs: _FakeServiceClient(training_client, **kwargs),
    )
    unload_calls: list[str] = []

    async def fake_unload(service_client, client):
        assert service_client.training_client is client
        unload_calls.append(client._guaranteed_model_id())
        return {"unloaded": True}

    monkeypatch.setattr(_BENCH, "unload_adapter", fake_unload)
    args = argparse.Namespace(
        context=4,
        completion_tokens=2,
        group_size=2,
        sampling_logprob=-1.0,
        base_url="http://127.0.0.1:8001",
        run_id="cpu-learner-failure",
        warmup_steps=1,
        measured_steps=1,
        lora_rank=8,
        seed=0,
        learning_rate=1e-5,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
    )
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="synthetic learner failure"):
        asyncio.run(_BENCH._run(args, output))

    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [record["record_type"] for record in records] == ["manifest", "cleanup"]
    assert records[-1]["adapter_unloaded"] is True
    assert unload_calls == ["model_fixed"]
    # Cookbook ordering intentionally enqueues Adam before resolving learner output.
    assert training_client.calls == [
        "forward_backward:importance_sampling:2",
        "optim:1e-05",
    ]
