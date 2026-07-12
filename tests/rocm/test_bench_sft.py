from __future__ import annotations

import argparse
import asyncio
import importlib.util
import io
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _load_bench_sft(monkeypatch: pytest.MonkeyPatch):
    """Load the standalone Cookbook client without installing Cookbook here."""

    cookbook = types.ModuleType("tinker_cookbook")
    supervised = types.ModuleType("tinker_cookbook.supervised")
    common = types.ModuleType("tinker_cookbook.supervised.common")
    tokenizer_utils = types.ModuleType("tinker_cookbook.tokenizer_utils")
    common.compute_mean_nll = lambda *_args, **_kwargs: 0.5
    common.datum_from_model_input_weights = lambda *_args, **_kwargs: None
    tokenizer_utils.get_tokenizer = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "tinker_cookbook", cookbook)
    monkeypatch.setitem(sys.modules, "tinker_cookbook.supervised", supervised)
    monkeypatch.setitem(sys.modules, "tinker_cookbook.supervised.common", common)
    monkeypatch.setitem(sys.modules, "tinker_cookbook.tokenizer_utils", tokenizer_utils)

    path = Path(__file__).resolve().parents[2] / "rocm" / "bench_sft.py"
    spec = importlib.util.spec_from_file_location("_rocm_bench_sft_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ImmediateFuture:
    def __init__(self, value: Any):
        self.value = value

    async def result_async(self) -> Any:
        return self.value


class _FakeTrainingClient:
    def __init__(self) -> None:
        self.forward_backward_calls = 0
        self.optim_step_calls = 0

    async def forward_backward_async(self, _data, *, loss_fn: str):
        assert loss_fn == "cross_entropy"
        self.forward_backward_calls += 1
        return _ImmediateFuture(SimpleNamespace(loss_fn_outputs=[{"logprobs": [0.0]}]))

    async def optim_step_async(self, _optimizer):
        self.optim_step_calls += 1
        return _ImmediateFuture(SimpleNamespace(metrics={"grad_norm": 1.0}))

    def _guaranteed_model_id(self) -> str:
        return "fake-model-id"


class _FakeServiceClient:
    def __init__(self, training_client: _FakeTrainingClient) -> None:
        self.training_client = training_client
        self.create_calls = 0

    async def create_lora_training_client_async(self, **_kwargs):
        self.create_calls += 1
        return self.training_client


def _args(*, one_update_gate: bool) -> argparse.Namespace:
    return argparse.Namespace(
        base_url="http://127.0.0.1:8001",
        context=64,
        warmup_steps=0 if one_update_gate else 1,
        measured_steps=1 if one_update_gate else 5,
        one_update_gate=one_update_gate,
        inter_step_delay_seconds=0.0,
        lora_rank=8,
        learning_rate=2e-4,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        seed=0,
        run_id="unit-test",
    )


def _run_with_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    one_update_gate: bool,
    warmup_steps: int | None = None,
    measured_steps: int | None = None,
    unload_error: BaseException | None = None,
    adam_error: BaseException | None = None,
):
    module = _load_bench_sft(monkeypatch)
    training_client = _FakeTrainingClient()
    service_client = _FakeServiceClient(training_client)
    datum = SimpleNamespace(loss_fn_inputs={"weights": [1.0]})

    monkeypatch.setattr(module, "_build_datum", lambda _context: datum)
    monkeypatch.setattr(module, "_model_revision", lambda _model: "test-revision")
    monkeypatch.setattr(module, "_git_metadata", lambda path: {"path": str(path)})
    monkeypatch.setattr(module, "_package_version", lambda _name: "test-version")
    monkeypatch.setattr(module, "compute_mean_nll", lambda *_args: 0.5)
    monkeypatch.setattr(
        module.tinker,
        "ServiceClient",
        lambda **_kwargs: service_client,
    )

    def fake_adam_params(**kwargs):
        if adam_error is not None:
            raise adam_error
        return kwargs

    monkeypatch.setattr(module.tinker, "AdamParams", fake_adam_params)

    async def fake_unload(_service_client, passed_training_client):
        assert passed_training_client is training_client
        if unload_error is not None:
            raise unload_error
        return {"unloaded": True}

    monkeypatch.setattr(module, "_unload_adapter", fake_unload)
    output = io.StringIO()
    args = _args(one_update_gate=one_update_gate)
    if warmup_steps is not None:
        args.warmup_steps = warmup_steps
    if measured_steps is not None:
        args.measured_steps = measured_steps
    run_error: BaseException | None = None
    try:
        asyncio.run(module._run(args, output))
    except BaseException as error:
        run_error = error
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    return training_client, service_client, records, run_error


def test_one_update_gate_executes_exactly_one_update_and_no_throughput(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, records, error = _run_with_fakes(monkeypatch, one_update_gate=True)

    assert error is None
    assert service.create_calls == 1
    assert client.forward_backward_calls == 1
    assert client.optim_step_calls == 1
    assert [record["record_type"] for record in records] == [
        "manifest",
        "step",
        "cleanup",
        "summary",
    ]
    assert records[0]["protocol"] == "one_update_gate"
    assert records[0]["updates_requested"] == 1
    assert records[0]["measured_steps"] == 0
    assert records[1]["phase"] == "one_update_gate"
    assert records[1]["cold_jit"] is True
    assert records[-1]["updates_completed"] == 1
    assert records[-1]["steady_state_throughput_available"] is False
    assert "useful_tokens_per_second_median" not in records[-1]


def test_steady_state_protocol_still_executes_one_cold_and_five_measured_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, records, error = _run_with_fakes(
        monkeypatch, one_update_gate=False
    )

    assert error is None
    assert service.create_calls == 1
    assert client.forward_backward_calls == 6
    assert client.optim_step_calls == 6
    steps = [record for record in records if record["record_type"] == "step"]
    assert [record["phase"] for record in steps] == ["cold_compile"] + ["measured"] * 5
    assert records[-1]["protocol"] == "steady_state"
    assert records[-1]["measured_steps"] == 5
    assert records[-1]["useful_tokens_per_second_median"] > 0


@pytest.mark.parametrize(
    ("warmup_steps", "measured_steps"),
    [(1, 0), (-5, 6), (1, 1)],
)
def test_malformed_programmatic_gate_fails_before_adapter_creation(
    monkeypatch: pytest.MonkeyPatch,
    warmup_steps: int,
    measured_steps: int,
) -> None:
    client, service, records, error = _run_with_fakes(
        monkeypatch,
        one_update_gate=True,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
    )

    assert isinstance(error, RuntimeError)
    assert "canonical" in str(error)
    assert service.create_calls == 0
    assert client.forward_backward_calls == 0
    assert client.optim_step_calls == 0
    assert records == []


def test_cleanup_failure_is_durably_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, records, error = _run_with_fakes(
        monkeypatch,
        one_update_gate=True,
        unload_error=RuntimeError("synthetic unload failure"),
    )

    assert isinstance(error, RuntimeError)
    assert str(error) == "synthetic unload failure"
    assert service.create_calls == 1
    assert client.forward_backward_calls == 1
    assert client.optim_step_calls == 1
    assert [record["record_type"] for record in records] == [
        "manifest",
        "step",
        "cleanup",
    ]
    assert records[-1]["adapter_unloaded"] is False
    assert records[-1]["error_type"] == "RuntimeError"
    assert records[-1]["error"] == "synthetic unload failure"


def test_optimizer_validation_failure_precedes_adapter_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, service, records, error = _run_with_fakes(
        monkeypatch,
        one_update_gate=True,
        adam_error=ValueError("synthetic Adam validation failure"),
    )

    assert isinstance(error, ValueError)
    assert service.create_calls == 0
    assert client.forward_backward_calls == 0
    assert client.optim_step_calls == 0
    assert records == []


def test_manifest_environment_includes_all_device_selection_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_bench_sft(monkeypatch)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("GPU_DEVICE_ORDINAL", "0")
    monkeypatch.setenv("PJRT_DEVICE_DEBUG", "1")
    monkeypatch.setenv("ROCR_SECRET_TOKEN", "redact-me")

    captured = module._safe_accelerator_environment()

    assert captured["ROCR_VISIBLE_DEVICES"] == "0"
    assert captured["GPU_DEVICE_ORDINAL"] == "0"
    assert captured["PJRT_DEVICE_DEBUG"] == "1"
    assert "ROCR_SECRET_TOKEN" not in captured
