from __future__ import annotations

import ast
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).parents[2]
_PROBE = _REPO / "rocm" / "probe_sft_compile.py"
_ACCELERATOR_ENV = (
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
    for name in _ACCELERATOR_ENV:
        environment.pop(name, None)
    return environment


def _run(*arguments: str, environment: dict[str, str] | None = None):
    return subprocess.run(
        [sys.executable, str(_PROBE), *arguments],
        cwd=_REPO,
        env=environment or _clean_environment(),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def test_default_is_cpu_forced_refusal_without_jax_import():
    environment = _clean_environment()
    environment["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer=CUBLAS"

    result = _run(environment=environment)

    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert [record["record_type"] for record in records] == ["manifest", "refused"]
    manifest, refused = records
    assert manifest["platform_requested"] == "abstract"
    assert manifest["allow_gpu"] is False
    assert manifest["scope"] == "cpu_guard_refusal"
    assert manifest["requested_context"] == 2048
    assert manifest["effective_context"] == 2048
    assert manifest["batch_size"] == 1
    assert manifest["fixed_preallocation_fraction"] == 0.85
    assert manifest["command_buffers_disabled"] is True
    assert manifest["environment"]["JAX_PLATFORMS"] == "cpu"
    assert manifest["environment"]["XLA_PYTHON_CLIENT_ALLOCATOR"] == "bfc"
    assert manifest["environment"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert manifest["environment"]["XLA_CLIENT_MEM_FRACTION"] == "0.85"
    assert (
        manifest["environment"]["XLA_FLAGS_effective"]
        == "--xla_gpu_enable_command_buffer="
    )
    assert manifest["normal_api_lifecycle_planned_before_lowering"] is False
    assert "autotuning/profiling kernels" in manifest["compile_dispatch_caveat"]
    assert refused["status"] == "cpu_guard_only"
    assert refused["jax_imported"] is False
    assert refused["model_pass_executable_invocations"] == 0


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--platform", "rocm"), "requires the explicit --allow-gpu"),
        (
            ("--platform", "rocm", "--allow-gpu"),
            "requires --output for a clean JSONL artifact",
        ),
        (("--allow-gpu",), "only valid with --platform rocm"),
        (("--allow-download",), "only valid with --platform rocm"),
        (("--context", "0"), "--context must be in"),
        (("--context", "2049"), "contexts above 2048 require"),
        (
            ("--context", "32769", "--allow-large-context"),
            "--context must be in",
        ),
    ],
)
def test_unsafe_options_are_rejected_before_any_record(arguments, message):
    result = _run(*arguments)

    assert result.returncode == 2
    assert result.stdout == ""
    assert message in result.stderr


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("JAX_PLATFORMS", "rocm", "conflicts with required value 'cpu'"),
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
        ("XLA_CLIENT_MEM_FRACTION", "0.90", "conflicts with required value '0.85'"),
        (
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "0.85",
            "is deprecated and conflicts",
        ),
    ],
)
def test_conflicting_environment_fails_before_any_record(name, value, message):
    environment = _clean_environment()
    environment[name] = value

    result = _run(environment=environment)

    assert result.returncode == 1
    assert result.stdout == ""
    assert message in result.stderr


def test_context_bucketing_is_reported_without_compilation():
    result = _run("--context", "1537")

    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout.splitlines()[0])
    assert manifest["requested_context"] == 1537
    assert manifest["effective_context"] == 2048


def test_output_is_private_exclusive_jsonl(tmp_path):
    output = tmp_path / "compile.jsonl"

    result = _run("--output", str(output))

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert [json.loads(line)["record_type"] for line in output.read_text().splitlines()] == [
        "manifest",
        "refused",
    ]

    repeated = _run("--output", str(output))
    assert repeated.returncode == 2
    assert "refusing to overwrite existing output" in repeated.stderr


def _fake_character_stat(_path):
    return SimpleNamespace(st_mode=stat.S_IFCHR)


def _headless_amd_drm(tmp_path: Path) -> Path:
    drm_root = tmp_path / "drm"
    (drm_root / "card1" / "device").mkdir(parents=True)
    (drm_root / "card1" / "device" / "vendor").write_text("0x1002\n")
    return drm_root


def test_hardware_preflight_accepts_only_headless_unowned_kfd(tmp_path):
    from rocm.probe_sft_compile import _hardware_preflight

    kfd = tmp_path / "dev" / "kfd"

    def unowned(arguments, **kwargs):
        assert arguments == ["fuser", str(kfd)]
        assert kwargs == {
            "capture_output": True,
            "text": True,
            "timeout": 5,
            "check": False,
        }
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    result = _hardware_preflight(
        drm_root=_headless_amd_drm(tmp_path),
        kfd=kfd,
        stat_fn=_fake_character_stat,
        access_fn=lambda *_args: True,
        run_fn=unowned,
    )

    assert result == {
        "amd_card_count": 1,
        "connected_amd_connectors": [],
        "kfd_path": str(kfd),
        "kfd_accessible": True,
        "kfd_unowned": True,
    }


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "message"),
    [
        (0, "1234", "", "already owned: 1234"),
        (0, "", "/dev/kfd: kernel", "already owned"),
        (2, "", "permission denied", "could not verify exclusive"),
        (1, "", "unexpected output", "could not verify exclusive"),
    ],
)
def test_hardware_preflight_fails_closed_on_fuser_outcomes(
    tmp_path, returncode, stdout, stderr, message
):
    from rocm.probe_sft_compile import _hardware_preflight

    with pytest.raises(RuntimeError, match=message):
        _hardware_preflight(
            drm_root=_headless_amd_drm(tmp_path),
            kfd=tmp_path / "dev" / "kfd",
            stat_fn=_fake_character_stat,
            access_fn=lambda *_args: True,
            run_fn=lambda *_args, **_kwargs: SimpleNamespace(
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            ),
        )


def test_validate_rocm_backend_uses_public_gpu_name_and_runtime_version():
    from rocm.probe_sft_compile import _validate_rocm_backend

    fake_jax = SimpleNamespace(default_backend=lambda: "gpu")
    assert _validate_rocm_backend(
        fake_jax, lambda: SimpleNamespace(platform_version="ROCm 7.2.0")
    ) == ("gpu", "ROCm 7.2.0")

    with pytest.raises(RuntimeError, match="does not identify as ROCm"):
        _validate_rocm_backend(
            fake_jax, lambda: SimpleNamespace(platform_version="CUDA 13.0")
        )


def test_module_has_no_top_level_jax_import_and_never_calls_compiled_executable():
    source = _PROBE.read_text(encoding="utf-8")
    module = ast.parse(source)
    top_level_imports = [
        node
        for node in module.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported_roots = {
        alias.name.partition(".")[0]
        for node in top_level_imports
        for alias in node.names
    }

    assert "jax" not in imported_roots
    assert "flax" not in imported_roots
    assert source.count("compiled = lowered.compile()") == 1
    assert "compiled(" not in source
    assert "model_pass(" not in source
