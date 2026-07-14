from __future__ import annotations

import ast
import copy
import hashlib
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rocm import qwen35_cache_attestation as cache_attestation
from skyrl.tinker import api as api_module
from skyrl.tinker import engine as engine_module
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.engine import TinkerEngine

_KEY = "jit_forward_backward_and_accumulate-" + "a" * 64


class _Config:
    def __init__(self, values: dict[str, object]):
        self.values = values
        self.tensor_parallel_size = 1
        self.expert_parallel_size = 1
        self.fully_sharded_data_parallel_size = 1
        self.coordinator_address = None
        self.num_processes = None

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return dict(self.values)


def _engine_for_method(claim: dict[str, object]) -> TinkerEngine:
    engine = object.__new__(TinkerEngine)
    engine.runtime_source_attestation = {
        "status": "passed",
        "startup_cache_attestation": claim,
    }
    engine.config = SimpleNamespace(
        startup_launch_id="a" * 32,
        backend="jax",
    )
    engine.backend = SimpleNamespace(config=_Config({"enforce_eager": False}))
    engine.cache_evidence_status = "not_required"
    engine.cache_evidence = {}
    return engine


def test_engine_cache_attestation_opt_out_is_a_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine_for_method({"status": "not_required"})
    monkeypatch.setattr(
        engine_module,
        "run_engine_t64_cache_attestation",
        lambda *_args, **_kwargs: pytest.fail("opt-out invoked cache attestation"),
    )

    engine._attest_startup_cache_if_required(use_ray=False)

    assert engine.cache_evidence_status == "not_required"
    assert engine.cache_evidence == {}


def test_generic_engine_has_no_module_scope_rocm_dependency() -> None:
    tree = ast.parse(Path(engine_module.__file__).read_text(encoding="utf-8"))
    top_level_rocm_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").startswith("rocm")
            )
            or (
                isinstance(node, ast.Import)
                and any(alias.name.startswith("rocm") for alias in node.names)
            )
        )
    ]
    assert top_level_rocm_imports == []


def _api_config(
    model_path: str,
    *,
    abstract_model_load: bool = False,
    fused_mlp: bool = False,
) -> EngineConfig:
    return EngineConfig(
        base_model=model_path,
        backend="jax",
        backend_config={
            "max_lora_adapters": 2,
            "max_lora_rank": 8,
            "train_micro_batch_size": 1,
            "sample_max_num_sequences": 1,
            "gradient_checkpointing": True,
            "loss_chunk_size": 64,
            "qwen35_bf16_down_lora_residual": False,
            "qwen35_bf16_rms_gate_up_lora_swiglu_contiguous": fused_mlp,
            "abstract_model_load": abstract_model_load,
        },
    )


def test_api_required_cache_config_check_is_exact() -> None:
    source = {
        "status": "passed",
        "memory_mode": "growth",
        "startup_cache_attestation": {
            "status": "required-v1",
            "seed": {
                "model_path": "/models/pinned",
                "backend_config": {
                    "qwen35_bf16_rms_gate_up_lora_swiglu_contiguous": False
                },
            },
        },
    }

    api_module._validate_required_startup_cache_engine_config(
        source, _api_config("/models/pinned")
    )

    with pytest.raises(RuntimeError, match="exact JAX engine config"):
        api_module._validate_required_startup_cache_engine_config(
            source, _api_config("/models/other")
        )


def test_api_required_cache_config_binds_abstract_load_to_memory_mode() -> None:
    source = {
        "status": "passed",
        "memory_mode": "preallocate85",
        "startup_cache_attestation": {
            "status": "required-v1",
            "seed": {
                "model_path": "/models/pinned",
                "backend_config": {
                    "qwen35_bf16_rms_gate_up_lora_swiglu_contiguous": False
                },
            },
        },
    }

    api_module._validate_required_startup_cache_engine_config(
        source, _api_config("/models/pinned", abstract_model_load=True)
    )
    with pytest.raises(RuntimeError, match="exact JAX engine config"):
        api_module._validate_required_startup_cache_engine_config(
            source, _api_config("/models/pinned", abstract_model_load=False)
        )


def test_api_required_cache_config_binds_fused_mlp_policy_to_seed() -> None:
    source = {
        "status": "passed",
        "memory_mode": "growth",
        "startup_cache_attestation": {
            "status": "required-v1",
            "seed": {
                "model_path": "/models/pinned",
                "backend_config": {
                    "qwen35_bf16_rms_gate_up_lora_swiglu_contiguous": True
                },
            },
        },
    }

    api_module._validate_required_startup_cache_engine_config(
        source, _api_config("/models/pinned", fused_mlp=True)
    )
    with pytest.raises(RuntimeError, match="exact JAX engine config"):
        api_module._validate_required_startup_cache_engine_config(
            source, _api_config("/models/pinned", fused_mlp=False)
        )


@pytest.mark.parametrize("value", [None, 0, "false"])
def test_api_required_cache_config_requires_exact_fused_mlp_bool(value) -> None:
    backend_config = {}
    if value is not None:
        backend_config["qwen35_bf16_rms_gate_up_lora_swiglu_contiguous"] = value
    source = {
        "status": "passed",
        "memory_mode": "growth",
        "startup_cache_attestation": {
            "status": "required-v1",
            "seed": {
                "model_path": "/models/pinned",
                "backend_config": backend_config,
            },
        },
    }

    with pytest.raises(RuntimeError, match="exact bool"):
        api_module._validate_required_startup_cache_engine_config(
            source, _api_config("/models/pinned")
        )


def test_engine_cache_attestation_stores_only_helper_validated_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = {
        "status": "required-v1",
        "schema_name": cache_attestation.SCHEMA_NAME,
        "schema_version": 1,
    }
    engine = _engine_for_method(claim)
    evidence = {"kind": cache_attestation.RUNTIME_HIT_KIND}
    observed: list[tuple[object, object]] = []

    def attest(backend: object, claim_arg: object) -> dict[str, object]:
        observed.append((backend, claim_arg))
        return evidence

    monkeypatch.setattr(engine_module, "run_engine_t64_cache_attestation", attest)

    engine._attest_startup_cache_if_required(use_ray=False)

    assert observed == [(engine.backend, claim)]
    assert engine.cache_evidence_status == cache_attestation.RUNTIME_HIT_KIND
    assert engine.cache_evidence == evidence


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unsupervised", "supervised"),
        ("backend", "non-Ray JAX"),
        ("ray", "non-Ray JAX"),
        ("distributed", "distributed JAX"),
        ("multidevice", "one local JAX device"),
    ],
)
def test_engine_cache_attestation_rejects_unqualified_topologies(
    mutation: str, message: str
) -> None:
    engine = _engine_for_method({"status": "required-v1"})
    use_ray = False
    if mutation == "unsupervised":
        engine.config.startup_launch_id = None
    elif mutation == "backend":
        engine.config.backend = "fsdp"
    elif mutation == "ray":
        use_ray = True
    elif mutation == "distributed":
        engine.backend.config.coordinator_address = "127.0.0.1:1234"
    else:
        engine.backend.config.tensor_parallel_size = 2

    with pytest.raises(RuntimeError, match=message):
        engine._attest_startup_cache_if_required(use_ray=use_ray)


class _Monitoring:
    def __init__(self, order: list[str]):
        self.order = order
        self.event_listeners: list[Any] = []
        self.duration_listeners: list[Any] = []

    def register_event_listener(self, listener: Any) -> None:
        self.order.append("register_event")
        self.event_listeners.append(listener)

    def unregister_event_listener(self, listener: Any) -> None:
        self.order.append("unregister_event")
        self.event_listeners.remove(listener)

    def register_event_duration_secs_listener(self, listener: Any) -> None:
        self.order.append("register_duration")
        self.duration_listeners.append(listener)

    def unregister_event_duration_listener(self, listener: Any) -> None:
        self.order.append("unregister_duration")
        self.duration_listeners.remove(listener)

    def event(self, name: str) -> None:
        for listener in tuple(self.event_listeners):
            listener(name)

    def duration(self, name: str, value: float) -> None:
        for listener in tuple(self.duration_listeners):
            listener(name, value)


class _LossFnConfig:
    def __init__(self, **values: object):
        self.values = values


class _CompiledMustNotRun:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *_args: object) -> None:
        self.calls += 1
        raise AssertionError("compiled T64 executable must never be invoked")


class _Lowered:
    def __init__(
        self,
        monitoring: _Monitoring,
        atime_path: Path,
        compiled: _CompiledMustNotRun,
        order: list[str],
    ) -> None:
        self.monitoring = monitoring
        self.atime_path = atime_path
        self.compiled = compiled
        self.order = order
        self.compile_calls = 0

    def compile(self) -> _CompiledMustNotRun:
        self.order.append("compile")
        self.compile_calls += 1
        self.monitoring.event("/jax/compilation_cache/compile_requests_use_cache")
        self.monitoring.event("/jax/compilation_cache/cache_hits")
        self.monitoring.duration("/jax/compilation_cache/compile_time_saved_sec", 2.0)
        self.monitoring.duration(
            "/jax/compilation_cache/cache_retrieval_time_sec", 0.01
        )
        self.atime_path.write_bytes(time.time_ns().to_bytes(8, "little"))
        return self.compiled


class _ModelPass:
    def __init__(self, lowered: _Lowered, order: list[str]) -> None:
        self.lowered = lowered
        self.order = order
        self.lower_calls: list[tuple[object, ...]] = []

    def lower(self, *arguments: object) -> _Lowered:
        self.order.append("lower")
        self.lower_calls.append(arguments)
        return self.lowered


class _FakeJax:
    def __init__(self, monitoring: _Monitoring, order: list[str]) -> None:
        self.monitoring = monitoring
        self.order = order
        self.P = lambda *axes: axes

    def devices(self) -> list[object]:
        return [SimpleNamespace(platform="gpu")]

    def process_count(self) -> int:
        return 1

    def device_count(self) -> int:
        return 1

    def block_until_ready(self, value: object) -> object:
        self.order.append("block_until_ready")
        return value

    def NamedSharding(self, mesh: object, specification: object) -> tuple[object, ...]:
        return (mesh, specification)

    def ShapeDtypeStruct(
        self, dimensions: tuple[int, ...], dtype: object, *, sharding: object
    ) -> tuple[object, ...]:
        return (dimensions, dtype, sharding)

    @contextmanager
    def set_mesh(self, mesh: object):
        self.order.append("mesh_enter")
        yield mesh
        self.order.append("mesh_exit")


def test_runtime_helper_lowers_exact_t64_signature_and_never_invokes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    cache.chmod(0o700)
    executable = cache / f"{_KEY}-cache"
    atime = cache / f"{_KEY}-atime"
    executable.write_bytes(b"serialized T64 executable")
    atime.write_bytes((100).to_bytes(8, "little"))
    before = cache_attestation.snapshot_cache(cache)
    target = {"key": _KEY, **before.pairs[_KEY].executable.as_dict()}
    model_path = tmp_path / "model"
    model_path.mkdir()
    config_values = {"enforce_eager": False}
    config_sha = cache_attestation.canonical_json_sha256(
        config_values, domain="skyrl-qwen35-resolved-jax-backend-config-v1"
    )
    claim: dict[str, object] = {
        "status": "required-v1",
        "schema_name": cache_attestation.SCHEMA_NAME,
        "schema_version": 1,
        "prewarm_audit": {"sha256": "d" * 64},
        "prewarm_handoff": {"sha256": "e" * 64},
        "seed": {
            "model_path": str(model_path),
            "source_git_head": "a" * 40,
            "source_git_tree": "b" * 40,
            "cache_path": str(cache),
            "attention_backend": "pallas",
            "backend_config": config_values,
            "backend_config_sha256": config_sha,
            "target_cache_entry": target,
            "target_atime_transition": {
                "after_logical_atime_ns": 100,
                "after_sha256": hashlib.sha256(
                    (100).to_bytes(8, "little")
                ).hexdigest(),
            },
        },
    }
    monkeypatch.setattr(
        cache_attestation,
        "revalidate_startup_cache_claim",
        lambda claim_arg, **_kwargs: dict(claim_arg),
    )
    monkeypatch.setenv("XLA_FLAGS", cache_attestation.DISABLE_COMMAND_BUFFERS)
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(cache))
    monkeypatch.setenv("SKYRL_ROCM_PALLAS_ATTENTION", "1")
    order: list[str] = []
    monitoring = _Monitoring(order)
    compiled = _CompiledMustNotRun()
    lowered = _Lowered(monitoring, atime, compiled, order)
    model_pass = _ModelPass(lowered, order)
    backend = SimpleNamespace(
        config=_Config(config_values),
        base_model=str(model_path),
        mesh="mesh",
        accumulated_grads="grads",
        lora_params="lora",
        non_lora_params="base",
        _forward_backward_and_accumulate=model_pass,
    )
    fake_jax = _FakeJax(monitoring, order)
    fake_jnp = SimpleNamespace(int32="int32", float32="float32")

    evidence = cache_attestation.run_engine_t64_cache_attestation(
        backend,
        claim,
        jax_module=fake_jax,
        jnp_module=fake_jnp,
        loss_fn_config_class=_LossFnConfig,
    )

    assert evidence["kind"] == cache_attestation.RUNTIME_HIT_KIND
    assert evidence["normal_pjit_first_call_seeded"] is False
    assert compiled.calls == 0
    assert lowered.compile_calls == 1
    assert len(model_pass.lower_calls) == 1
    arguments = model_pass.lower_calls[0]
    assert arguments[:3] == ("grads", "lora", "base")
    assert len(arguments) == 12
    assert order.index("block_until_ready") < order.index("register_event")
    assert order.index("register_duration") < order.index("lower")
    assert order.index("compile") < order.index("unregister_duration")
    assert monitoring.event_listeners == []
    assert monitoring.duration_listeners == []

    tampered = []
    bad_duration = copy.deepcopy(evidence)
    bad_duration["monitoring"]["compile_time_saved_sec"] = [True]
    tampered.append(bad_duration)
    bad_snapshot_schema = copy.deepcopy(evidence)
    bad_snapshot_schema["snapshots"]["untrusted"] = "field"
    tampered.append(bad_snapshot_schema)
    bad_auxiliary_manifest = copy.deepcopy(evidence)
    bad_auxiliary_manifest["snapshots"][
        "auxiliary_manifest_after_sha256"
    ] = "0" * 64
    tampered.append(bad_auxiliary_manifest)
    bad_atime_hash = copy.deepcopy(evidence)
    bad_atime_hash["target_atime_transition"]["after_sha256"] = "0" * 64
    tampered.append(bad_atime_hash)
    bad_wall_bracket = copy.deepcopy(evidence)
    bad_wall_bracket["operation_wall_end_ns"] = (
        bad_wall_bracket["target_atime_transition"]["after_logical_atime_ns"] - 1
    )
    tampered.append(bad_wall_bracket)
    for candidate in tampered:
        with pytest.raises(cache_attestation.CacheAttestationError):
            cache_attestation.validate_runtime_cache_evidence(claim, candidate)


def test_runtime_helper_cleans_listeners_when_compile_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monitoring = _Monitoring([])
    capture = cache_attestation.PublicCacheMonitoringCapture(monitoring)

    with pytest.raises(RuntimeError, match="compiler failed"):
        with capture:
            raise RuntimeError("compiler failed")

    assert monitoring.event_listeners == []
    assert monitoring.duration_listeners == []
