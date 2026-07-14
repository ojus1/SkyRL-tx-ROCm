from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).parents[2]
_PROBE = _REPO / "rocm" / "probe_bfc_preallocation.py"
_LAUNCHER = _REPO / "rocm" / "start_qwen35.sh"
_SPEC = importlib.util.spec_from_file_location("probe_bfc_preallocation", _PROBE)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE_MODULE)
_ALLOCATOR_ENV = (
    "GPU_DEVICE_ORDINAL",
    "HIP_VISIBLE_DEVICES",
    "JAX_PLATFORMS",
    "ROCR_VISIBLE_DEVICES",
    "XLA_CLIENT_MEM_FRACTION",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in _ALLOCATOR_ENV:
        environment.pop(name, None)
    return environment


def _run_probe(*arguments: str, environment: dict[str, str] | None = None):
    return subprocess.run(
        [sys.executable, str(_PROBE), *arguments],
        cwd=_REPO,
        env=environment or _clean_environment(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_default_probe_is_cpu_only_and_reports_fixed_bfc_contract():
    environment = _clean_environment()
    environment["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer=CUBLAS"

    result = _run_probe("--settle-seconds", "0", environment=environment)

    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert [record["record_type"] for record in records] == [
        "manifest",
        "allocated",
        "settled",
    ]
    manifest, allocated, settled = records
    assert manifest["platform_requested"] == "abstract"
    assert manifest["platform_resolved"] == "cpu"
    assert manifest["allow_gpu"] is False
    assert manifest["scope"] == "cpu_configuration_guard"
    assert manifest["gpu_preflight"] is None
    assert manifest["command_buffers_disabled"] is True
    assert all("cpu" in device.lower() for device in manifest["devices"])
    assert manifest["environment"]["JAX_PLATFORMS"] == "cpu"
    assert manifest["environment"]["XLA_PYTHON_CLIENT_ALLOCATOR"] == "bfc"
    assert manifest["environment"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert manifest["environment"]["XLA_CLIENT_MEM_FRACTION"] == "0.85"
    assert (
        manifest["environment"]["XLA_FLAGS_effective"]
        == "--xla_gpu_enable_command_buffer="
    )
    assert allocated["array_bytes"] == 256
    assert "cpu" in allocated["array_device"].lower()
    assert settled["status"] == "passed"


def test_backend_validation_uses_public_gpu_name_and_rocm_version():
    fake_jax = SimpleNamespace(default_backend=lambda: "gpu")

    assert _PROBE_MODULE._validate_backend(
        fake_jax,
        lambda: SimpleNamespace(platform_version="ROCm 7.2.0"),
        "rocm",
    ) == ("gpu", "ROCm 7.2.0")


def test_backend_validation_rejects_cuda_for_rocm_request():
    fake_jax = SimpleNamespace(default_backend=lambda: "gpu")

    with pytest.raises(RuntimeError, match="does not identify as ROCm"):
        _PROBE_MODULE._validate_backend(
            fake_jax,
            lambda: SimpleNamespace(platform_version="CUDA 13.0"),
            "rocm",
        )


def _fake_character_stat(_path):
    return SimpleNamespace(st_mode=stat.S_IFCHR)


def _fake_unowned_fuser(arguments, **kwargs):
    assert arguments[-1].endswith("kfd")
    assert kwargs == {"capture_output": True, "text": True, "check": False}
    return SimpleNamespace(returncode=1, stdout="", stderr="")


def test_gpu_preflight_accepts_headless_amd_and_ignores_writeback(tmp_path):
    drm_root = tmp_path / "drm"
    (drm_root / "card1" / "device").mkdir(parents=True)
    (drm_root / "card1" / "device" / "vendor").write_text("0x1002\n")
    (drm_root / "card1-Writeback-1").mkdir()
    (drm_root / "card1-Writeback-1" / "status").write_text("connected\n")

    preflight = _PROBE_MODULE._gpu_preflight(
        drm_root=drm_root,
        kfd_path=tmp_path / "dev" / "kfd",
        stat_fn=_fake_character_stat,
        access_fn=lambda *_args: True,
        which_fn=lambda _name: "/usr/bin/fuser",
        run_fn=_fake_unowned_fuser,
    )

    assert preflight == {
        "amd_cards": ["card1"],
        "connected_amd_connectors": [],
        "kfd_path": str(tmp_path / "dev" / "kfd"),
        "kfd_accessible": True,
        "kfd_unowned": True,
    }


def test_gpu_preflight_rejects_active_amd_display_before_kfd(tmp_path):
    drm_root = tmp_path / "drm"
    (drm_root / "card1" / "device").mkdir(parents=True)
    (drm_root / "card1" / "device" / "vendor").write_text("0x1002\n")
    (drm_root / "card1-HDMI-A-1").mkdir()
    (drm_root / "card1-HDMI-A-1" / "status").write_text("connected\n")

    with pytest.raises(RuntimeError, match="AMD display connector is active"):
        _PROBE_MODULE._gpu_preflight(drm_root=drm_root)


def test_gpu_preflight_rejects_existing_kfd_owner(tmp_path):
    drm_root = tmp_path / "drm"
    (drm_root / "card1" / "device").mkdir(parents=True)
    (drm_root / "card1" / "device" / "vendor").write_text("0x1002\n")

    with pytest.raises(RuntimeError, match="already owned: 1234"):
        _PROBE_MODULE._gpu_preflight(
            drm_root=drm_root,
            kfd_path=tmp_path / "dev" / "kfd",
            stat_fn=_fake_character_stat,
            access_fn=lambda *_args: True,
            which_fn=lambda _name: "/usr/bin/fuser",
            run_fn=lambda *_args, **_kwargs: SimpleNamespace(
                returncode=0, stdout="1234", stderr=""
            ),
        )


def test_all_rocm_probes_share_one_exclusive_launch_lock(tmp_path):
    from rocm.probe_model_residency import _acquire_global_lock as residency_lock
    from rocm.probe_sft_compile import _acquire_global_lock as compile_lock

    lock_fd = _PROBE_MODULE._acquire_global_lock(runtime_dir=tmp_path)
    try:
        with pytest.raises(RuntimeError, match="global launch lock"):
            residency_lock(runtime_dir=tmp_path)
        with pytest.raises(RuntimeError, match="global launch lock"):
            compile_lock(runtime_dir=tmp_path)
    finally:
        os.close(lock_fd)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--platform", "rocm"), "requires the explicit --allow-gpu"),
        (("--allow-gpu",), "only valid with --platform rocm"),
        (("--fraction", "0.79"), "fraction must be finite and in"),
        (("--fraction", "nan"), "fraction must be finite and in"),
        (("--settle-seconds", "31"), "settle-seconds must be finite"),
    ],
)
def test_probe_rejects_unsafe_arguments_before_any_record(arguments, message):
    result = _run_probe(*arguments)

    assert result.returncode == 2
    assert result.stdout == ""
    assert message in result.stderr


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        (
            "XLA_PYTHON_CLIENT_ALLOCATOR",
            "platform",
            "conflicts with required BFC allocation",
        ),
        (
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "false",
            "conflicts with required fixed preallocation",
        ),
        ("XLA_CLIENT_MEM_FRACTION", "0.90", "conflicts with requested fraction"),
        (
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "0.85",
            "is deprecated and conflicts",
        ),
        ("JAX_PLATFORMS", "rocm", "conflicts with required value 'cpu'"),
    ],
)
def test_probe_fails_closed_on_conflicting_environment(name, value, message):
    environment = _clean_environment()
    environment[name] = value

    result = _run_probe("--settle-seconds", "0", environment=environment)

    assert result.returncode == 1
    assert result.stdout == ""
    assert message in result.stderr


def test_launcher_rejects_unknown_memory_mode_before_creating_run_root(tmp_path):
    run_root = tmp_path / "runs"
    environment = _clean_environment()
    environment.update(
        {
            "SKYRL_QWEN35_MEMORY_MODE": "unknown",
            "SKYRL_QWEN35_RUN_ROOT": str(run_root),
        }
    )

    result = subprocess.run(
        [str(_LAUNCHER), "cpu-invalid-mode"],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 2
    assert "must be growth or preallocate85" in result.stderr
    assert not run_root.exists()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("XLA_PYTHON_CLIENT_ALLOCATOR", "platform"),
        ("JAX_PLATFORMS", "cpu"),
    ],
)
def test_launcher_preallocate_mode_rejects_conflicting_environment_early(
    tmp_path, name, value
):
    run_root = tmp_path / "runs"
    environment = _clean_environment()
    environment.update(
        {
            "SKYRL_QWEN35_MEMORY_MODE": "preallocate85",
            "SKYRL_QWEN35_RUN_ROOT": str(run_root),
            name: value,
        }
    )

    result = subprocess.run(
        [str(_LAUNCHER), "cpu-conflict"],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 2
    assert "conflicts with SKYRL_QWEN35_MEMORY_MODE=preallocate85" in result.stderr
    assert not run_root.exists()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("ROCR_VISIBLE_DEVICES", "1", "must be unset or exactly 0"),
        ("SKYRL_ROCM_PALLAS_ATTENTION", "2", "must be exactly 0 or 1"),
    ],
)
def test_launcher_rejects_nonexact_attested_runtime_environment_early(
    tmp_path, name, value, message
):
    run_root = tmp_path / "runs"
    environment = _clean_environment()
    environment.update({"SKYRL_QWEN35_RUN_ROOT": str(run_root), name: value})

    result = subprocess.run(
        [str(_LAUNCHER), "cpu-invalid-runtime-environment"],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert not run_root.exists()
