from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyrl.tx.kernels.quantized_lora import (
    GroupQuantizedWeight,
    quantized_lora_linear,
)

_REPO = Path(__file__).parents[2]
_PROBE_PATH = _REPO / "rocm" / "probe_w8a8_lora_forward.py"
_SPEC = importlib.util.spec_from_file_location(
    "probe_w8a8_lora_forward_test", _PROBE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)
_CONTROLLER_PATH = _REPO / "rocm" / "run_w8a8_lora_forward_gate.py"
_CONTROLLER_SPEC = importlib.util.spec_from_file_location(
    "w8a8_controller_for_probe_test", _CONTROLLER_PATH
)
assert _CONTROLLER_SPEC is not None and _CONTROLLER_SPEC.loader is not None
_CONTROLLER = importlib.util.module_from_spec(_CONTROLLER_SPEC)
_CONTROLLER_SPEC.loader.exec_module(_CONTROLLER)

_ACCELERATOR_ENVIRONMENT = (
    "AMDGCN_ENABLE_DUMP",
    "GPU_DEVICE_ORDINAL",
    "HSA_OVERRIDE_GFX_VERSION",
    "HIP_VISIBLE_DEVICES",
    "JAX_COMPILATION_CACHE_DIR",
    "JAX_ENABLE_COMPILATION_CACHE",
    "JAX_MOCK_GPU_TOPOLOGY",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
    "JAX_PJRT_CLIENT_CREATE_OPTIONS",
    "JAX_PLATFORMS",
    "JAX_RAISE_PERSISTENT_CACHE_ERRORS",
    "JAX_ROCM_VISIBLE_DEVICES",
    "MOCK_NUM_GPU_PROCESSES",
    "ROCR_VISIBLE_DEVICES",
    "TEST_UNDECLARED_OUTPUTS_DIR",
    "TF_FORCE_UNIFIED_MEMORY",
    "TF_XLA_HSACO_BITCODE_SIZE_THRESHOLD",
    "TF_XLA_HSACO_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "TRITON_DUMP_DIR",
    "TRITON_KERNEL_DUMP",
    "XLA_CLIENT_MEM_FRACTION",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)


@pytest.fixture(autouse=True)
def _force_cpu_default_device():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _clean_environment(monkeypatch) -> None:
    for name in _ACCELERATOR_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)


def _artifact_paths(tmp_path: Path) -> dict[str, Path]:
    tmp_path.chmod(0o700)
    return _PROBE._create_private_artifact_dir(tmp_path / "artifacts")


def _run(*arguments: str):
    environment = os.environ.copy()
    for name in _ACCELERATOR_ENVIRONMENT:
        environment.pop(name, None)
    return subprocess.run(
        [sys.executable, str(_PROBE_PATH), *arguments],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def test_default_is_abstract_refusal_before_jax_import() -> None:
    result = _run()

    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert [record["record_type"] for record in records] == ["manifest", "refused"]
    assert records[0]["platform_requested"] == "abstract"
    assert records[0]["jax_imported"] is False
    assert records[1]["jax_imported"] is False
    assert (
        records[0]["contract"]["dispatch_plan"]["compiled_executable_invocations"] == 0
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--platform", "rocm"), "requires the explicit --allow-gpu"),
        (("--platform", "rocm", "--allow-gpu"), "requires --output"),
        (
            ("--platform", "rocm", "--allow-gpu", "--output", "/tmp/out"),
            "requires --artifact-dir",
        ),
        (
            (
                "--platform",
                "rocm",
                "--allow-gpu",
                "--output",
                "/tmp/out",
                "--artifact-dir",
                "/tmp/artifacts",
            ),
            "requires --launcher-lock-fd",
        ),
        (("--allow-gpu",), "only valid with --platform rocm"),
        (("--backward",), "unrecognized arguments"),
        (("--repeats", "2"), "unrecognized arguments"),
        (("--warmups", "1"), "unrecognized arguments"),
        (("--phase", "execute"), "execute rung is disabled"),
    ],
)
def test_scope_broadening_or_unsafe_invocations_are_rejected(
    arguments: tuple[str, ...], message: str
) -> None:
    result = _run(*arguments)

    assert result.returncode == 2
    assert result.stdout == ""
    assert message in result.stderr


def test_abstract_private_output_is_exclusive_mode_0600(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    output = tmp_path / "abstract.jsonl"

    result = _run("--output", str(output))

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert [
        json.loads(line)["record_type"] for line in output.read_text().splitlines()
    ] == ["manifest", "refused"]
    repeated = _run("--output", str(output))
    assert repeated.returncode == 2
    assert "refusing to overwrite" in repeated.stderr


def test_contract_is_exact_tiny_forward_and_never_backward() -> None:
    compile_contract = _PROBE._exact_contract("compile")

    assert compile_contract["operation"] == "w8a8_group64_rank8_lora_forward_only"
    assert compile_contract["tiles"] == {
        "group_size": 64,
        "block_m": 16,
        "block_n": 16,
        "row_superblock": 16,
    }
    assert compile_contract["inputs"][0]["shape"] == [3, 64]
    assert compile_contract["output"]["shape"] == [3, 17]
    assert compile_contract["dispatch_plan"]["compiled_executable_invocations"] == 0
    assert compile_contract["execute_rung_enabled"] is False
    assert compile_contract["runtime_numerical_gate_evaluated"] is False
    assert compile_contract["dispatch_plan"]["backward_invocations"] == 0
    assert compile_contract["dispatch_plan"]["warmup_invocations"] == 0
    assert compile_contract["dispatch_plan"]["replay_invocations"] == 0
    with pytest.raises(RuntimeError, match="only the compile diagnostic contract"):
        _PROBE._exact_contract("execute")


def test_probe_top_level_is_standard_library_only() -> None:
    module = ast.parse(_PROBE_PATH.read_text(encoding="utf-8"))
    imports = [
        node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    roots = {alias.name.partition(".")[0] for node in imports for alias in node.names}

    assert not roots & {"jax", "jaxlib", "numpy", "ml_dtypes", "skyrl", "rocm"}


def test_probe_source_and_parent_bootstrap_hashes_are_current() -> None:
    manifest = _PROBE._assert_bound_sources()
    runtime_json = json.dumps(
        _CONTROLLER._EXPECTED_PROFILE_RUNTIME_SHA256,
        separators=(",", ":"),
        sort_keys=True,
    )

    assert set(_PROBE._EXPECTED_SOURCE_SHA256) <= set(manifest)
    assert (
        hashlib.sha256(_CONTROLLER._ISOLATED_PROFILE_BOOTSTRAP.encode()).hexdigest()
        == _PROBE._EXPECTED_PROFILE_BOOTSTRAP_SHA256
    )
    assert (
        hashlib.sha256(runtime_json.encode()).hexdigest()
        == _PROBE._EXPECTED_PROFILE_RUNTIME_JSON_SHA256
    )


def test_operational_import_binding_requires_isolated_no_site_python() -> None:
    with pytest.raises(RuntimeError, match="isolated Python contract"):
        _PROBE._establish_isolated_import_path()


def test_operational_import_binding_keeps_stdlib_then_repo_then_venv(
    tmp_path: Path,
) -> None:
    cache_root = (tmp_path / "python-cache").resolve()
    cache_root.mkdir(mode=0o700)
    code = (
        "import importlib.util,json,pathlib,sys;"
        "path=sys.argv[1];"
        "spec=importlib.util.spec_from_file_location('isolated_probe',path);"
        "module=importlib.util.module_from_spec(spec);"
        "spec.loader.exec_module(module);"
        "record=module._establish_isolated_import_path(pathlib.Path(sys.argv[2]));"
        "print(json.dumps({'record':record,'path':sys.path}))"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            f"pycache_prefix={cache_root}",
            "-c",
            code,
            str(_PROBE_PATH),
            str(tmp_path.resolve()),
        ],
        cwd=_REPO,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    expected_site = str(_REPO / ".venv" / "lib" / "python3.12" / "site-packages")
    assert payload["path"] == [
        "/usr/lib/python312.zip",
        "/usr/lib/python3.12",
        "/usr/lib/python3.12/lib-dynload",
        str(_REPO),
        expected_site,
    ]
    assert payload["record"]["site_packages"] == expected_site
    assert payload["record"]["repo_root"] == str(_REPO)
    assert payload["record"]["pycache_prefix"] == str(cache_root)
    assert payload["record"]["pycache_empty_before_bound_imports"] is True
    assert all(payload["record"]["checks"].values())


def test_environment_establishes_exact_sole_xla_flag_and_private_caches(
    monkeypatch, tmp_path: Path
) -> None:
    _clean_environment(monkeypatch)
    paths = _artifact_paths(tmp_path)

    effective = _PROBE._configure_environment(paths)

    assert effective["XLA_FLAGS"] == "--xla_gpu_enable_command_buffer="
    assert os.environ["XLA_FLAGS"] == "--xla_gpu_enable_command_buffer="
    assert effective["XLA_CLIENT_MEM_FRACTION"] == "0.075"
    assert effective["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert Path(effective["JAX_COMPILATION_CACHE_DIR"]) == paths["jax_cache"]
    assert Path(effective["TF_XLA_HSACO_CACHE_DIR"]) == paths["hsaco_cache"]
    assert Path(effective["TEST_UNDECLARED_OUTPUTS_DIR"]) == paths["compiler_dump"]
    assert effective["AMDGCN_ENABLE_DUMP"] == "1"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        (
            "XLA_FLAGS",
            "--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=2",
            "exact sole",
        ),
        ("HSA_OVERRIDE_GFX_VERSION", "11.0.0", "must be unset"),
        ("JAX_MOCK_GPU_TOPOLOGY", "2x1x1", "must be unset"),
        ("JAX_DISABLE_JIT", "1", "unexpected accelerator environment"),
        ("TF_FORCE_UNIFIED_MEMORY", "1", "must be unset"),
        ("TRITON_KERNEL_DUMP", "0", "must be unset"),
    ],
)
def test_environment_rejects_hidden_or_inherited_overrides(
    monkeypatch, tmp_path: Path, name: str, value: str, message: str
) -> None:
    _clean_environment(monkeypatch)
    monkeypatch.setenv(name, value)
    paths = _artifact_paths(tmp_path)

    with pytest.raises(RuntimeError, match=message):
        _PROBE._configure_environment(paths)


def test_host_case_is_deterministic_and_matches_portable_oracle() -> None:
    first_arguments, first_manifests, first_expected = _PROBE._construct_host_case()
    second_arguments, second_manifests, second_expected = _PROBE._construct_host_case()

    assert first_manifests == second_manifests
    for first, second in zip(first_arguments, second_arguments, strict=True):
        np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_expected, second_expected)

    x, codes, scales, lora_a, lora_b, scaling = first_arguments
    weight = GroupQuantizedWeight(codes, scales, 64, 64, 8, 64)
    actual = quantized_lora_linear(
        jnp.asarray(x),
        weight,
        jnp.asarray(lora_a),
        jnp.asarray(lora_b),
        jnp.asarray(scaling),
        activation_bits=8,
    )
    np.testing.assert_array_equal(np.asarray(actual), first_expected)
    assert np.any(codes < 0)
    assert np.any(codes > 0)


def test_host_case_detects_every_missing_lora_rank_and_output_column() -> None:
    arguments, _, expected = _PROBE._construct_host_case()
    x, codes, scales, lora_a, lora_b, scaling = arguments
    weight = GroupQuantizedWeight(codes, scales, 64, 64, 8, 64)
    expected_fp32 = np.asarray(expected, dtype=np.float32)
    denominator = np.linalg.norm(expected_fp32.ravel())

    def relative_with(candidate_b) -> float:
        actual = quantized_lora_linear(
            jnp.asarray(x),
            weight,
            jnp.asarray(lora_a),
            jnp.asarray(candidate_b),
            jnp.asarray(scaling),
            activation_bits=8,
        )
        difference = np.asarray(actual, dtype=np.float32) - expected_fp32
        return float(np.linalg.norm(difference.ravel()) / denominator)

    for rank in range(lora_b.shape[0]):
        missing = lora_b.copy()
        missing[rank, :] = 0
        assert relative_with(missing) > 0.01
    for column in range(lora_b.shape[1]):
        missing = lora_b.copy()
        missing[:, column] = 0
        assert relative_with(missing) > 0.01


def _stablehlo(target: str = "__gpu$xla.gpu.triton", *, backward=False) -> str:
    marker = _PROBE._EXPECTED_KERNEL_NAME
    if backward:
        marker += " w8a16_lora_input_vjp"
    return (
        "module {\n"
        f"  %0 = stablehlo.custom_call @{target}() "
        f'{{backend_config = "{marker}"}} : () -> tensor<1xf32>\n'
        "  return\n"
        "}\n"
    )


def _optimized_hlo(target: str = "__gpu$xla.gpu.triton", *, backward=False) -> str:
    marker = _PROBE._EXPECTED_KERNEL_NAME
    if backward:
        marker += " w8a16_lora_input_vjp"
    return (
        "ENTRY main {\n"
        f'  ROOT %0 = f32[1] custom-call(), custom_call_target="{target}", '
        f'op_name="{marker}"\n'
        "}\n"
    )


def test_structural_gate_accepts_only_the_named_forward_triton_call() -> None:
    stable = _PROBE._ir_summary(_stablehlo(), "stablehlo")
    optimized = _PROBE._ir_summary(_optimized_hlo(), "optimized_hlo")

    proof = _PROBE._structural_gate(stable, optimized)

    assert proof["passed"] is True


@pytest.mark.parametrize(
    ("stable_text", "optimized_text"),
    [
        (_stablehlo("lookalike_triton"), _optimized_hlo()),
        (_stablehlo(), _optimized_hlo("triton")),
        (_stablehlo(backward=True), _optimized_hlo()),
        (_stablehlo(), _optimized_hlo(backward=True)),
        (_stablehlo() + _stablehlo(), _optimized_hlo()),
    ],
)
def test_structural_gate_rejects_wrong_target_backward_or_extra_call(
    stable_text: str, optimized_text: str
) -> None:
    stable = _PROBE._ir_summary(stable_text, "stablehlo")
    optimized = _PROBE._ir_summary(optimized_text, "optimized_hlo")

    assert _PROBE._structural_gate(stable, optimized)["passed"] is False


@pytest.mark.parametrize(
    "text",
    [
        _stablehlo().replace(
            "  return\n",
            "  %1 = stablehlo.custom_call @hipGraphLaunch() : () -> tensor<1xf32>\n"
            "  return\n",
        ),
        _stablehlo().replace(
            _PROBE._EXPECTED_KERNEL_NAME,
            _PROBE._EXPECTED_KERNEL_NAME + "_lookalike",
        ),
        _stablehlo().replace(
            "  return\n", "  %1 = stablehlo.while () : () -> ()\n  return\n"
        ),
        _stablehlo().replace(
            _PROBE._EXPECTED_KERNEL_NAME,
            _PROBE._EXPECTED_KERNEL_NAME + " hipGraphLaunch",
        ),
    ],
)
def test_stablehlo_precompile_gate_rejects_graph_lookalike_extra_call_or_loop(
    text: str,
) -> None:
    summary = _PROBE._ir_summary(text, "stablehlo")

    assert _PROBE._stablehlo_gate(summary)["passed"] is False


def test_optimized_parser_counts_whitespace_before_call_parenthesis() -> None:
    text = _optimized_hlo().replace("custom-call()", "custom-call   ()")
    summary = _PROBE._ir_summary(text, "optimized_hlo")

    assert summary["raw_custom_call_opcode_count"] == 1
    assert summary["custom_call_count"] == 1
    assert summary["custom_call_parser_consistent"] is True


@pytest.mark.parametrize(
    ("memory", "passed"),
    [
        (
            {
                "available": True,
                "argument_size_in_bytes": 1024,
                "output_size_in_bytes": 1024,
                "temp_size_in_bytes": 1024,
            },
            True,
        ),
        ({"available": False}, False),
        (
            {
                "available": True,
                "argument_size_in_bytes": 1,
                "output_size_in_bytes": 1,
                "temp_size_in_bytes": 257 * 1024**2,
            },
            False,
        ),
    ],
)
def test_compiled_memory_gate_fails_closed(
    memory: dict[str, object], passed: bool
) -> None:
    assert _PROBE._memory_gate(memory)["passed"] is passed


def test_rocm_backend_helper_rejects_execute_before_importing_jax(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "jax", None)

    with pytest.raises(RuntimeError, match="only the compile diagnostic backend"):
        _PROBE._run_rocm(
            argparse.Namespace(phase="execute"),
            None,
            tmp_path,
            tmp_path,
            lambda: {},
        )


def test_hardware_launch_gate_requires_power_and_temperature_margin(
    tmp_path: Path,
) -> None:
    device = tmp_path / "device"
    hwmon = device / "hwmon" / "hwmon0"
    hwmon.mkdir(parents=True)
    (hwmon / "power1_cap").write_text("300000000\n")
    (hwmon / "temp2_label").write_text("junction\n")
    (hwmon / "temp2_input").write_text("84000\n")

    result = _PROBE._read_hardware_limits(device)

    assert result["power_cap_uw"] == 300_000_000
    assert result["junction_temperature_millic"] == 84_000
    (hwmon / "power1_cap").write_text("315000001\n")
    with pytest.raises(RuntimeError, match="exceeds"):
        _PROBE._read_hardware_limits(device)
    (hwmon / "power1_cap").write_text("300000000\n")
    (hwmon / "temp2_input").write_text("85001\n")
    with pytest.raises(RuntimeError, match="<=85 C"):
        _PROBE._read_hardware_limits(device)


def test_controller_supervision_binds_scope_parent_and_exact_chain(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    self_root = proc_root / "self"
    parent_root = proc_root / str(os.getppid())
    self_root.mkdir(parents=True)
    parent_root.mkdir(parents=True)
    (self_root / "cgroup").write_text(
        "0::/user.slice/user-1000.slice/user@1000.service/app.slice/"
        "skyrl-w8a8-compile-123-abcdef.scope\n"
    )
    (parent_root / "exe").symlink_to((_REPO / ".venv" / "bin" / "python").resolve())
    run_dir = tmp_path.resolve()
    lock_fd = 41
    command = _CONTROLLER._profile_command(
        phase="compile", run_dir=run_dir, card="card1", lock_fd=lock_fd
    )
    (parent_root / "cmdline").write_bytes(
        b"\0".join(value.encode() for value in command) + b"\0"
    )

    result = _PROBE._validate_controller_supervision(
        run_dir, lock_fd, proc_root=proc_root
    )

    assert result["validated"] is True
    assert result["scope"] == "skyrl-w8a8-compile-123-abcdef.scope"
    (self_root / "cgroup").write_text("0::/user.slice/direct.scope\n")
    with pytest.raises(RuntimeError, match="private W8 compile scope"):
        _PROBE._validate_controller_supervision(run_dir, lock_fd, proc_root=proc_root)


def test_artifact_inventory_hashes_regular_files_and_rejects_symlinks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    root.chmod(0o700)
    (root / "kernel.hsaco").write_bytes(b"code-object")
    (root / "kernel.hsaco").chmod(0o600)

    inventory = _PROBE._artifact_inventory(root)

    assert inventory["file_count"] == 1
    assert inventory["total_bytes"] == len(b"code-object")
    assert inventory["files"][0]["path"] == "kernel.hsaco"
    (root / "bad-link").symlink_to(root / "kernel.hsaco")
    with pytest.raises(RuntimeError, match="symlink"):
        _PROBE._artifact_inventory(root)


def test_source_contains_no_graph_capture_or_replay_api() -> None:
    source = _PROBE_PATH.read_text(encoding="utf-8")
    forbidden = (
        "hipGraph",
        "cudaGraph",
        "capture_begin",
        "capture_end",
        "stream_capture",
        "command_buffer(",
    )
    assert not any(token in source for token in forbidden)
