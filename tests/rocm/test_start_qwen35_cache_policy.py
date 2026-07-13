from __future__ import annotations

import importlib.util
import os
import pwd
import stat
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_LAUNCHER = _REPO / "rocm" / "start_qwen35.sh"
_CACHE_HELPER = _REPO / "rocm" / "prepare_jax_cache_dir.py"
_CACHE_NAMESPACE = (
    "jax0.10.2-jaxlib0.10.2-rocm-plugin0.10.2-pjrt0.10.2-"
    "rocm7.2.4-amdgpu6.16.13-gfx1100-v1"
)
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "start_qwen35_cache_policy_helper", _CACHE_HELPER
)
assert _HELPER_SPEC is not None and _HELPER_SPEC.loader is not None
_HELPER = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(_HELPER)


def _source() -> str:
    return _LAUNCHER.read_text(encoding="utf-8")


def _prepare_for_test(home: Path, maximum: int = 1024) -> Path:
    home_fd = os.open(home, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        return _HELPER._prepare_cache_in_open_home(home, home_fd, maximum)
    finally:
        os.close(home_fd)


def test_launcher_shell_syntax_is_valid() -> None:
    subprocess.run(["bash", "-n", str(_LAUNCHER)], check=True)


def test_executable_cache_is_private_versioned_and_bounded() -> None:
    source = _source()

    assert 'expected_jax_stack="0.10.2,0.10.2,0.10.2,0.10.2"' in source
    assert '("jax", "jaxlib", "jax-rocm7-plugin", "jax-rocm7-pjrt")' in source
    assert '"$(</opt/rocm/.info/version)" != "7.2.4"' in source
    assert '"$(</sys/module/amdgpu/version)" != "6.16.13"' in source
    assert source.index('expected_jax_stack="') < source.index(
        "python rocm/prepare_jax_cache_dir.py"
    )
    assert "refusing inherited JAX_COMPILATION_CACHE_DIR" in source
    assert "--max-autotune-bytes 4294967296" in source
    assert 'export JAX_COMPILATION_CACHE_DIR="$jax_cache_dir"' in source
    assert "export JAX_COMPILATION_CACHE_MAX_SIZE=17179869184" in source
    assert "serialized executable entries only" in source
    assert "textproto files are not part of this LRU" in source


def test_cache_populates_executables_and_only_qualified_autotuning() -> None:
    source = _source()

    assert "export JAX_ENABLE_COMPILATION_CACHE=true" in source
    assert "export JAX_RAISE_PERSISTENT_CACHE_ERRORS=true" in source
    assert "export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0" in source
    assert "export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1" in source
    assert (
        "export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="
        "xla_gpu_per_fusion_autotune_cache_dir"
    ) in source
    assert "xla_gpu_kernel_cache_file" not in source


def test_installed_jax_recognizes_exact_cache_environment() -> None:
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_COMPILATION_CACHE_DIR": "/tmp/skyrl-cache-policy-test",
        "JAX_ENABLE_COMPILATION_CACHE": "true",
        "JAX_RAISE_PERSISTENT_CACHE_ERRORS": "true",
        "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": (
            "xla_gpu_per_fusion_autotune_cache_dir"
        ),
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
        "JAX_COMPILATION_CACHE_MAX_SIZE": "17179869184",
    }
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from jax import config

values = config.values
assert values["jax_compilation_cache_dir"] == "/tmp/skyrl-cache-policy-test"
assert values["jax_enable_compilation_cache"] is True
assert values["jax_raise_persistent_cache_errors"] is True
assert values["jax_persistent_cache_enable_xla_caches"] == "xla_gpu_per_fusion_autotune_cache_dir"
assert values["jax_persistent_cache_min_compile_time_secs"] == 0.0
assert values["jax_persistent_cache_min_entry_size_bytes"] == -1
assert values["jax_compilation_cache_max_size"] == 17179869184
""",
        ],
        check=True,
        env=environment,
    )


def test_helper_resolves_home_from_account_database() -> None:
    home, home_fd = _HELPER._open_account_home()
    try:
        assert home == Path(pwd.getpwuid(os.getuid()).pw_dir)
    finally:
        os.close(home_fd)


def test_helper_cli_rejects_arbitrary_home_override(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(_CACHE_HELPER),
            "--home",
            str(tmp_path),
            "--max-autotune-bytes",
            "1024",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unrecognized arguments: --home" in result.stderr


def test_cache_helper_creates_private_fixed_namespace(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)

    result = _prepare_for_test(home)

    expected = home / ".cache" / "skyrl-jax-rocm-private-v1" / _CACHE_NAMESPACE
    assert result == expected
    for directory in (home / ".cache", expected.parent, expected):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700


def test_cache_helper_rejects_preexisting_writable_directory_without_repair(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    home_cache = home / ".cache"
    home_cache.mkdir(mode=0o700)
    unsafe = home_cache / "skyrl-jax-rocm-private-v1"
    unsafe.mkdir(mode=0o777)
    unsafe.chmod(0o777)
    (unsafe / "attacker-cache").write_bytes(b"untrusted")

    with pytest.raises(RuntimeError, match="mode is 0777"):
        _prepare_for_test(home)

    assert stat.S_IMODE(unsafe.stat().st_mode) == 0o777
    assert (unsafe / "attacker-cache").read_bytes() == b"untrusted"


def test_cache_helper_rejects_symlink_ancestor(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    (home / ".cache").symlink_to(target, target_is_directory=True)

    with pytest.raises(OSError):
        _prepare_for_test(home)

    assert not (target / "skyrl-jax-rocm-private-v1").exists()


def test_cache_helper_rejects_symlink_inside_namespace(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    namespace = _prepare_for_test(home)
    target = tmp_path / "attacker-cache"
    target.write_bytes(b"untrusted")
    (namespace / "planted-cache").symlink_to(target)

    with pytest.raises(RuntimeError, match="refusing symlink in trusted cache"):
        _prepare_for_test(home)


def test_cache_helper_separately_caps_autotune_tree_at_startup(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    namespace = _prepare_for_test(home, maximum=4)
    autotune = namespace / "xla_gpu_per_fusion_autotune_cache_dir"
    autotune.mkdir(mode=0o700)
    (autotune / "entry.textproto").write_bytes(b"12345")

    with pytest.raises(RuntimeError, match="above startup maximum 4"):
        _prepare_for_test(home, maximum=4)


def test_cache_setup_precedes_backend_start_and_graphs_remain_disabled() -> None:
    source = _source()

    cache_export = source.index('export JAX_COMPILATION_CACHE_DIR="$jax_cache_dir"')
    command_buffer_disable = source.index(
        "export XLA_FLAGS=--xla_gpu_enable_command_buffer="
    )
    backend_start = source.index("exec uv run --active --no-sync -m skyrl.tinker.api")
    assert cache_export < command_buffer_disable < backend_start
    assert source.count("export XLA_FLAGS=") == 1
