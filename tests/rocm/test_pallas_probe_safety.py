from __future__ import annotations

import ast
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).parents[2]
_PROBE_PATH = _REPO / "rocm" / "probe_pallas_attention.py"
_SPEC = importlib.util.spec_from_file_location(
    "probe_pallas_attention_safety", _PROBE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)

_ACCELERATOR_ENVIRONMENT = (
    "GPU_DEVICE_ORDINAL",
    "HSA_OVERRIDE_GFX_VERSION",
    "HIP_VISIBLE_DEVICES",
    "JAX_PJRT_CLIENT_CREATE_OPTIONS",
    "JAX_PLATFORMS",
    "JAX_MOCK_GPU_TOPOLOGY",
    "JAX_ROCM_VISIBLE_DEVICES",
    "MOCK_NUM_GPU_PROCESSES",
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


def _clear_accelerator_environment(monkeypatch) -> None:
    for name in _ACCELERATOR_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)


def test_gpu_acknowledgement_is_required_before_environment_or_jax_setup():
    with pytest.raises(SystemExit) as caught:
        _PROBE._parse_args(["--sequence-length", "512"])

    assert caught.value.code == 2


def test_environment_is_bounded_and_command_buffers_are_forced_off(monkeypatch):
    _clear_accelerator_environment(monkeypatch)
    monkeypatch.setenv(
        "XLA_FLAGS",
        "--xla_gpu_enable_command_buffer=CUBLAS --xla_gpu_autotune_level=2",
    )

    effective = _PROBE._configure_environment()

    assert effective["JAX_PLATFORMS"] == "rocm"
    assert effective["ROCR_VISIBLE_DEVICES"] == "0"
    assert effective["HIP_VISIBLE_DEVICES"] == "0"
    assert effective["GPU_DEVICE_ORDINAL"] == "0"
    assert effective["JAX_ROCM_VISIBLE_DEVICES"] == "0"
    assert effective["XLA_PYTHON_CLIENT_ALLOCATOR"] == "bfc"
    assert effective["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert effective["XLA_CLIENT_MEM_FRACTION"] == "0.75"
    assert effective["SKYRL_ROCM_PALLAS_ATTENTION"] == "1"
    assert effective["XLA_FLAGS_effective"].split() == [
        "--xla_gpu_autotune_level=2",
        "--xla_gpu_enable_command_buffer=",
    ]


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("JAX_PLATFORMS", "cpu", "conflicts with required value 'rocm'"),
        ("ROCR_VISIBLE_DEVICES", "1", "conflicts with required value '0'"),
        ("JAX_ROCM_VISIBLE_DEVICES", "1", "conflicts with required value '0'"),
        (
            "XLA_PYTHON_CLIENT_ALLOCATOR",
            "platform",
            "conflicts with required value 'bfc'",
        ),
        (
            "XLA_CLIENT_MEM_FRACTION",
            "1.5",
            "conflicts with required value '0.75'",
        ),
        (
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "true",
            "conflicts with bounded growth allocation",
        ),
    ],
)
def test_environment_refuses_conflicting_inherited_settings(
    monkeypatch, name, value, message
):
    _clear_accelerator_environment(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=message):
        _PROBE._configure_environment()


@pytest.mark.parametrize(
    "name",
    [
        "HSA_OVERRIDE_GFX_VERSION",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB",
    ],
)
def test_environment_rejects_target_and_allocator_bypasses(monkeypatch, name):
    _clear_accelerator_environment(monkeypatch)
    monkeypatch.setenv(name, "1")

    with pytest.raises(RuntimeError, match=f"{name} must be unset"):
        _PROBE._configure_environment()


def test_environment_rejects_pjrt_post_plugin_override(monkeypatch):
    _clear_accelerator_environment(monkeypatch)
    monkeypatch.setenv(
        "JAX_PJRT_CLIENT_CREATE_OPTIONS",
        "preallocate:true;memory_fraction:1.5;visible_devices:1",
    )

    with pytest.raises(RuntimeError, match="can override"):
        _PROBE._configure_environment()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("JAX_MOCK_GPU_TOPOLOGY", "2x2x1", "physical single-GPU topology"),
        ("MOCK_NUM_GPU_PROCESSES", "2", "physical single-process topology"),
    ],
)
def test_environment_rejects_mock_topology_overrides(monkeypatch, name, value, message):
    _clear_accelerator_environment(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=message):
        _PROBE._configure_environment()


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_environment_rejects_truthy_unified_memory(monkeypatch, value):
    _clear_accelerator_environment(monkeypatch)
    monkeypatch.setenv("TF_FORCE_UNIFIED_MEMORY", value)

    with pytest.raises(RuntimeError, match="must be unset or false"):
        _PROBE._configure_environment()


def test_environment_removes_harmless_empty_or_false_overrides(monkeypatch):
    _clear_accelerator_environment(monkeypatch)
    monkeypatch.setenv("JAX_PJRT_CLIENT_CREATE_OPTIONS", "")
    monkeypatch.setenv("JAX_MOCK_GPU_TOPOLOGY", "")
    monkeypatch.setenv("MOCK_NUM_GPU_PROCESSES", "0")
    monkeypatch.setenv("TF_FORCE_UNIFIED_MEMORY", "false")

    _PROBE._configure_environment()

    assert "JAX_PJRT_CLIENT_CREATE_OPTIONS" not in __import__("os").environ
    assert "JAX_MOCK_GPU_TOPOLOGY" not in __import__("os").environ
    assert "MOCK_NUM_GPU_PROCESSES" not in __import__("os").environ
    assert "TF_FORCE_UNIFIED_MEMORY" not in __import__("os").environ


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--batch-size", "2"],
        ["--query-heads", "32"],
        ["--kv-heads", "8"],
        ["--head-dim", "128"],
    ],
)
def test_probe_rejects_non_qwen_shapes(extra_args):
    with pytest.raises(SystemExit) as caught:
        _PROBE._parse_args(["--allow-gpu", "--sequence-length", "512", *extra_args])

    assert caught.value.code == 2


def test_probe_caps_reference_and_repetition_work():
    for extra_args in (
        ["--sequence-length", "64"],
        ["--sequence-length", "512", "--dtype", "float16"],
        ["--sequence-length", "4096", "--reference"],
        ["--sequence-length", "512", "--warmups", "11"],
        ["--sequence-length", "512", "--repeats", "21"],
    ):
        with pytest.raises(SystemExit) as caught:
            _PROBE._parse_args(["--allow-gpu", *extra_args])
        assert caught.value.code == 2


def test_main_holds_shared_guard_around_run_without_importing_jax(monkeypatch):
    import rocm.amdgpu_safety as safety

    state = {"guard_active": False, "run_called": False, "postflight_calls": 0}
    preflight = {"amdgpu_boot_clean": True, "kfd_unowned": True}

    @contextmanager
    def fake_guard():
        state["guard_active"] = True
        try:
            yield preflight
        finally:
            state["guard_active"] = False

    def fake_run(
        args,
        effective_environment: dict[str, str | None],
        actual_preflight: dict[str, Any],
        require_clean_boot,
    ):
        assert state["guard_active"] is True
        assert args.sequence_length == 512
        assert effective_environment == {"configured": "yes"}
        assert actual_preflight is preflight
        assert require_clean_boot is safety.require_clean_amdgpu_boot
        state["run_called"] = True

    monkeypatch.setattr(safety, "guarded_qwen35_rocm_process", fake_guard)
    monkeypatch.setattr(
        safety,
        "require_clean_amdgpu_boot",
        lambda: (
            state.__setitem__("postflight_calls", state["postflight_calls"] + 1)
            or {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}
        ),
    )
    monkeypatch.setattr(_PROBE, "_configure_environment", lambda: {"configured": "yes"})
    monkeypatch.setattr(_PROBE, "_run", fake_run)

    _PROBE.main(["--allow-gpu", "--sequence-length", "512"])

    assert state == {
        "guard_active": False,
        "run_called": True,
        "postflight_calls": 1,
    }


def test_main_guarantees_postflight_when_run_raises(monkeypatch):
    import rocm.amdgpu_safety as safety

    state = {"postflight_calls": 0}

    @contextmanager
    def fake_guard():
        yield {"amdgpu_boot_clean": True, "kfd_unowned": True}

    def clean_boot():
        state["postflight_calls"] += 1
        return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}

    monkeypatch.setattr(safety, "guarded_qwen35_rocm_process", fake_guard)
    monkeypatch.setattr(safety, "require_clean_amdgpu_boot", clean_boot)
    monkeypatch.setattr(_PROBE, "_configure_environment", lambda: {})
    monkeypatch.setattr(
        _PROBE,
        "_run",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("synthetic GPU failure")),
    )

    with pytest.raises(RuntimeError, match="synthetic GPU failure"):
        _PROBE.main(["--allow-gpu", "--sequence-length", "512"])

    assert state["postflight_calls"] == 1


def test_probe_import_is_standard_library_only():
    module = ast.parse(_PROBE_PATH.read_text(encoding="utf-8"))
    top_level_imports = [
        node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported_roots = {
        alias.name.partition(".")[0]
        for node in top_level_imports
        for alias in node.names
    }

    assert "jax" not in imported_roots
    assert "skyrl" not in imported_roots
