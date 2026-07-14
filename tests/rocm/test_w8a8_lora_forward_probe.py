from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import types
from collections.abc import Callable
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
    runtime_contract = _PROBE._exact_contract("execute")

    assert compile_contract["operation"] == "w8a8_group64_rank8_lora_forward_only"
    assert compile_contract["tiles"] == {
        "group_size": 64,
        "block_m": 16,
        "block_n": 16,
        "row_superblock": 16,
        "logical_out_features": 17,
        "physical_out_features": 32,
        "full_output_tiles_only": True,
    }
    assert compile_contract["inputs"][0]["shape"] == [3, 64]
    assert compile_contract["inputs"][1]["shape"] == [64, 32]
    assert compile_contract["inputs"][2]["shape"] == [1, 32]
    assert compile_contract["inputs"][4]["shape"] == [8, 32]
    assert compile_contract["output"]["shape"] == [3, 17]
    assert compile_contract["dispatch_plan"]["compiled_executable_invocations"] == 0
    assert compile_contract["execute_rung_enabled"] is False
    assert compile_contract["runtime_numerical_gate_evaluated"] is False
    assert compile_contract["dispatch_plan"]["backward_invocations"] == 0
    assert compile_contract["dispatch_plan"]["warmup_invocations"] == 0
    assert compile_contract["dispatch_plan"]["replay_invocations"] == 0
    assert runtime_contract["execute_rung_enabled"] is True
    assert runtime_contract["runtime_numerical_gate_evaluated"] is True
    runtime_dispatch = runtime_contract["dispatch_plan"]
    assert runtime_dispatch["host_oracle_attempts"] == 1
    assert runtime_dispatch["host_sensitivity_comparisons"] == 35
    assert runtime_dispatch["isa_qualification_attempts"] == 1
    assert runtime_dispatch["tuple_device_put_attempts"] == 1
    assert runtime_dispatch["device_put_leaves"] == 6
    assert runtime_dispatch["input_readiness_invocations"] == 1
    assert runtime_dispatch["compiled_executable_invocations"] == 1
    assert runtime_dispatch["compiled_executable_completions"] == 1
    assert runtime_dispatch["output_readiness_invocations"] == 1
    assert runtime_dispatch["device_get_attempts"] == 1
    assert runtime_dispatch["device_get_leaves"] == 1
    assert runtime_dispatch["backward_invocations"] == 0
    assert runtime_dispatch["warmup_invocations"] == 0
    assert runtime_dispatch["replay_invocations"] == 0
    assert runtime_dispatch["gpu_reference_invocations"] == 0
    assert runtime_dispatch["device_error_reduction_invocations"] == 0
    assert runtime_dispatch["model_invocations"] == 0
    with pytest.raises(RuntimeError, match="unsupported W8 qualification phase"):
        _PROBE._exact_contract("benchmark")


def test_abstract_execute_contract_still_refuses_before_jax_import() -> None:
    result = _run("--phase", "execute")

    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert [record["record_type"] for record in records] == ["manifest", "refused"]
    assert records[0]["contract"]["phase"] == "execute"
    assert (
        records[0]["contract"]["dispatch_plan"]["compiled_executable_invocations"] == 1
    )
    assert all(record.get("jax_imported") is False for record in records)


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
    np.testing.assert_array_equal(np.asarray(actual)[:, :17], first_expected)
    np.testing.assert_array_equal(codes[:, 17:], 0)
    np.testing.assert_array_equal(scales[:, 17:], 0)
    np.testing.assert_array_equal(lora_b[:, 17:], 0)
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
        difference = np.asarray(actual, dtype=np.float32)[:, :17] - expected_fp32
        return float(np.linalg.norm(difference.ravel()) / denominator)

    for rank in range(lora_b.shape[0]):
        missing = lora_b.copy()
        missing[rank, :] = 0
        assert relative_with(missing) > 0.01
    for column in range(17):
        missing = lora_b.copy()
        missing[:, column] = 0
        assert relative_with(missing) > 0.01


def test_host_oracle_sensitivity_has_exact_35_strict_comparisons() -> None:
    arguments, _, expected = _PROBE._construct_host_case()

    sensitivity = _PROBE._host_oracle_sensitivity(arguments, expected)

    assert sensitivity["passed"] is True
    assert sensitivity["comparison_count"] == 35
    assert sensitivity["groups"] == {
        "global": 7,
        "omitted_rows": 3,
        "omitted_columns": 17,
        "omitted_ranks": 8,
    }
    assert len({item["label"] for item in sensitivity["comparisons"]}) == 35
    assert all(
        item["passed"] is True and item["relative_l2_error"] > 0.03
        for item in sensitivity["comparisons"]
    )


def test_runtime_numerical_gate_is_host_only_strict_and_all_51() -> None:
    _, _, expected = _PROBE._construct_host_case()

    exact = _PROBE._runtime_numerical_validation(expected.copy(), expected, 0.999)

    assert exact["passed"] is True
    assert exact["element_count"] == 51
    assert exact["finite_actual_count"] == 51
    assert exact["finite_expected_count"] == 51
    assert exact["bitwise_equal_diagnostic"] is True
    assert exact["actual"]["sha256"] == exact["expected"]["sha256"]
    assert exact["relative_l2_error"] == 0.0
    assert type(exact["relative_l2_error"]) is float
    assert exact["cosine_similarity"] == 1.0
    assert type(exact["cosine_similarity"]) is float
    assert exact["maximum_absolute_error"] == 0.0
    assert type(exact["maximum_absolute_error"]) is float

    wrong = expected.copy()
    wrong[0, 0] = np.asarray(wrong[0, 0], dtype=np.float32) + 1.0
    wrong_result = _PROBE._runtime_numerical_validation(wrong, expected, 0.1)
    timeout_result = _PROBE._runtime_numerical_validation(expected, expected, 1.0)
    dtype_result = _PROBE._runtime_numerical_validation(
        np.asarray(expected, dtype=np.float32), expected, 0.1
    )

    assert wrong_result["passed"] is False
    assert timeout_result["checks"]["dispatch_finite_below_one_second"] is False
    assert dtype_result["checks"]["actual_dtype_bfloat16"] is False


def test_runtime_numerical_gate_rejects_every_material_boundary_mutation() -> None:
    _, _, expected = _PROBE._construct_host_case()
    relative_l2 = np.array(expected, copy=True)
    relative_l2[...] = (
        np.asarray(relative_l2, dtype=np.float32) + np.float32(0.5)
    ).astype(expected.dtype)
    low_cosine = np.asarray(-np.asarray(expected, dtype=np.float32)).astype(
        expected.dtype
    )
    nonfinite = np.array(expected, copy=True)
    nonfinite[0, 0] = np.asarray(np.nan, dtype=expected.dtype)
    excessive_absolute = np.array(expected, copy=True)
    excessive_absolute[0, 0] = np.asarray(
        np.asarray(excessive_absolute[0, 0], dtype=np.float32) + 1.0,
        dtype=expected.dtype,
    )
    wrong_shape = np.asarray(expected).reshape((1, 3, 17))
    wrong_dtype_and_nbytes = np.asarray(expected, dtype=np.float32)
    noncontiguous = np.asfortranarray(expected)
    assert noncontiguous.flags.c_contiguous is False

    cases = (
        (
            "relative_l2",
            relative_l2,
            expected,
            0.1,
            ("relative_l2_below_one_percent",),
        ),
        (
            "cosine",
            low_cosine,
            expected,
            0.1,
            ("cosine_at_least_0_9999",),
        ),
        (
            "nonfinite",
            nonfinite,
            expected,
            0.1,
            ("all_51_actual_elements_finite",),
        ),
        (
            "maximum_absolute",
            excessive_absolute,
            expected,
            0.1,
            ("maximum_absolute_error_at_most_0_25",),
        ),
        (
            "shape",
            wrong_shape,
            expected,
            0.1,
            ("actual_shape_exact",),
        ),
        (
            "dtype_and_nbytes",
            wrong_dtype_and_nbytes,
            expected,
            0.1,
            ("actual_dtype_bfloat16", "actual_nbytes_exact"),
        ),
        (
            "contiguity",
            noncontiguous,
            expected,
            0.1,
            ("actual_c_contiguous",),
        ),
        (
            "dispatch_nonfinite",
            expected,
            expected,
            float("nan"),
            ("dispatch_finite_below_one_second",),
        ),
        (
            "dispatch_negative",
            expected,
            expected,
            -0.001,
            ("dispatch_finite_below_one_second",),
        ),
    )

    for label, actual, reference, dispatch_seconds, failed_checks in cases:
        result = _PROBE._runtime_numerical_validation(
            actual, reference, dispatch_seconds
        )
        assert result["passed"] is False, label
        assert all(not bool(result["checks"][name]) for name in failed_checks), label


def test_one_shot_dispatch_capability_is_consumed_before_call_and_never_retries() -> (
    None
):
    capability = _PROBE._OneShotDispatchCapability()

    assert capability.snapshot() == {
        "consumed": False,
        "compiled_executable_attempts": 0,
        "compiled_executable_completions": 0,
    }
    capability.consume()
    assert capability.snapshot()["compiled_executable_attempts"] == 1
    with pytest.raises(RuntimeError, match="already consumed"):
        capability.consume()
    capability.complete()
    assert capability.snapshot()["compiled_executable_completions"] == 1
    with pytest.raises(RuntimeError, match="already consumed"):
        capability.consume()
    with pytest.raises(RuntimeError, match="completion state is invalid"):
        capability.complete()


def _stablehlo(target: str = "__gpu$xla.gpu.triton", *, backward=False) -> str:
    name = _PROBE._EXPECTED_KERNEL_NAME
    marker = name
    if backward:
        marker += " w8a16_lora_input_vjp"
    return (
        "module {\n"
        "  func.func public @main() -> tensor<1xf32> {\n"
        f"    %0 = stablehlo.custom_call @{target}() "
        f'{{mhlo.backend_config = {{ir = "payload\\00{marker}\\00", '
        f'name = "{name}"}}}} : () -> tensor<1xf32>\n'
        "    return %0 : tensor<1xf32>\n"
        "  }\n"
        "}\n"
    )


def _optimized_hlo(target: str = "__gpu$xla.gpu.triton", *, backward=False) -> str:
    name = _PROBE._EXPECTED_KERNEL_NAME
    marker = name
    if backward:
        marker += " w8a16_lora_input_vjp"
    return (
        "ENTRY %main () -> f32[1] {\n"
        f'  ROOT %0 = f32[1] custom-call(), custom_call_target="{target}", '
        f'metadata={{op_name="jit(candidate)/{marker}/pallas_call"}}, '
        f'backend_config={{ir="payload\\00{marker}\\00", name="{name}"}}\n'
        "}\n"
    )


def test_structural_gate_accepts_only_the_named_forward_triton_call() -> None:
    stable = _PROBE._ir_summary(_stablehlo(), "stablehlo")
    optimized = _PROBE._ir_summary(_optimized_hlo(), "optimized_hlo")

    proof = _PROBE._structural_gate(stable, optimized)

    assert proof["passed"] is True
    assert stable["expected_kernel_marker_count"] == 1
    assert optimized["expected_kernel_marker_count"] == 2
    assert optimized["custom_call_count"] == 1


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
            "    return %0 : tensor<1xf32>\n",
            "    %1 = stablehlo.custom_call @hipGraphLaunch() : () -> tensor<1xf32>\n"
            "    return %0 : tensor<1xf32>\n",
        ),
        _stablehlo().replace(
            _PROBE._EXPECTED_KERNEL_NAME,
            _PROBE._EXPECTED_KERNEL_NAME + "_lookalike",
        ),
        _stablehlo().replace(
            "    return %0 : tensor<1xf32>\n",
            "    %1 = stablehlo.while () : () -> ()\n    return %0 : tensor<1xf32>\n",
        ),
        _stablehlo().replace(
            f"payload\\00{_PROBE._EXPECTED_KERNEL_NAME}\\00",
            f"payload\\00{_PROBE._EXPECTED_KERNEL_NAME} hipGraphLaunch\\00",
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
    "replacement",
    [
        'kernel="wrong"',
        'name="wrong"',
        f'debug={{name="{_PROBE._EXPECTED_KERNEL_NAME}"}}',
        f'name="{_PROBE._EXPECTED_KERNEL_NAME}", name="{_PROBE._EXPECTED_KERNEL_NAME}"',
        f'/* name="{_PROBE._EXPECTED_KERNEL_NAME}" */',
        f'fake-name="{_PROBE._EXPECTED_KERNEL_NAME}"',
    ],
)
def test_optimized_gate_rejects_backend_name_decoys(replacement: str) -> None:
    exact = f'name="{_PROBE._EXPECTED_KERNEL_NAME}"'
    text = _optimized_hlo().replace(exact, replacement)

    summary = _PROBE._ir_summary(text, "optimized_hlo")

    assert (
        _PROBE._structural_gate(_PROBE._ir_summary(_stablehlo(), "stablehlo"), summary)[
            "passed"
        ]
        is False
    )


@pytest.mark.parametrize(
    "replacement",
    [
        'kernel = "wrong"',
        'name = "wrong"',
        f'debug = {{name = "{_PROBE._EXPECTED_KERNEL_NAME}"}}',
        f'name = "{_PROBE._EXPECTED_KERNEL_NAME}", '
        f'name = "{_PROBE._EXPECTED_KERNEL_NAME}"',
        f'/* name = "{_PROBE._EXPECTED_KERNEL_NAME}" */',
        f'fake-name = "{_PROBE._EXPECTED_KERNEL_NAME}"',
    ],
)
def test_stable_gate_rejects_backend_name_decoys(replacement: str) -> None:
    exact = f'name = "{_PROBE._EXPECTED_KERNEL_NAME}"'
    text = _stablehlo().replace(exact, replacement)

    assert (
        _PROBE._stablehlo_gate(_PROBE._ir_summary(text, "stablehlo"))["passed"] is False
    )


def test_optimized_gate_rejects_nested_or_duplicate_target_decoys() -> None:
    exact = 'custom_call_target="__gpu$xla.gpu.triton"'
    cases = [
        'custom_call_target="wrong", '
        'metadata={custom_call_target="__gpu$xla.gpu.triton"}',
        f"{exact}, {exact}",
    ]

    stable = _PROBE._ir_summary(_stablehlo(), "stablehlo")
    for replacement in cases:
        optimized = _PROBE._ir_summary(
            _optimized_hlo().replace(exact, replacement), "optimized_hlo"
        )
        assert _PROBE._structural_gate(stable, optimized)["passed"] is False


def test_probe_gates_reject_backend_map_scope_decoys() -> None:
    marker = _PROBE._EXPECTED_KERNEL_NAME
    stable = _stablehlo()
    optimized = _optimized_hlo()
    stable_map = (
        f'mhlo.backend_config = {{ir = "payload\\00{marker}\\00", name = "{marker}"}}'
    )
    optimized_map = f'backend_config={{ir="payload\\00{marker}\\00", name="{marker}"}}'
    nested_stable = stable.replace(stable_map, f"metadata = {{{stable_map}}}")
    nested_optimized = optimized.replace(
        optimized_map, f"metadata_decoy={{{optimized_map}}}"
    )

    assert (
        _PROBE._stablehlo_gate(_PROBE._ir_summary(nested_stable, "stablehlo"))["passed"]
        is False
    )
    assert (
        _PROBE._structural_gate(
            _PROBE._ir_summary(stable, "stablehlo"),
            _PROBE._ir_summary(nested_optimized, "optimized_hlo"),
        )["passed"]
        is False
    )


def test_probe_gates_reject_dead_call_and_control_dependency() -> None:
    stable = _stablehlo()
    optimized = _optimized_hlo()
    dead_stable = stable.replace(
        "    return %0 : tensor<1xf32>\n",
        "    %1 = stablehlo.constant dense<0.0> : tensor<1xf32>\n"
        "    return %1 : tensor<1xf32>\n",
    )
    dead_optimized = optimized.replace(
        "  ROOT %0 = f32[1] custom-call()",
        "  %0 = f32[1] custom-call()",
    )
    dead_optimized = (
        dead_optimized.removesuffix("}\n")
        + "  ROOT %1 = f32[1] constant(0), control-predecessors={%0}\n}\n"
    )

    assert (
        _PROBE._stablehlo_gate(_PROBE._ir_summary(dead_stable, "stablehlo"))["passed"]
        is False
    )
    assert (
        _PROBE._structural_gate(
            _PROBE._ir_summary(stable, "stablehlo"),
            _PROBE._ir_summary(dead_optimized, "optimized_hlo"),
        )["passed"]
        is False
    )


def test_probe_gates_reject_call_owned_only_by_dead_helper() -> None:
    stable = _stablehlo()
    optimized = _optimized_hlo()
    stable_call = next(
        line for line in stable.splitlines() if "stablehlo.custom_call" in line
    )
    stable_without_live_call = stable.replace(
        stable_call, "    %0 = stablehlo.constant dense<0.0> : tensor<1xf32>"
    )
    dead_stable = (
        stable_without_live_call.removesuffix("}\n")
        + "  func.func private @dead() -> tensor<1xf32> {\n"
        + stable_call
        + "\n    return %0 : tensor<1xf32>\n  }\n}\n"
    )
    optimized_call = next(
        line for line in optimized.splitlines() if "custom-call" in line
    )
    optimized_without_live_call = optimized.replace(
        optimized_call, "  ROOT %0 = f32[1] constant(0)"
    )
    dead_optimized = (
        "%dead {\n"
        + optimized_call.replace("  ROOT ", "  ")
        + "\n  ROOT %dead_root = f32[1] copy(%0)\n}\n\n"
        + optimized_without_live_call
    )

    stable_summary = _PROBE._ir_summary(dead_stable, "stablehlo")
    optimized_summary = _PROBE._ir_summary(dead_optimized, "optimized_hlo")

    assert stable_summary["sole_custom_call_owned_by_entry"] is False
    assert _PROBE._stablehlo_gate(stable_summary)["passed"] is False
    assert optimized_summary["sole_custom_call_owned_by_entry"] is False
    assert (
        _PROBE._structural_gate(
            _PROBE._ir_summary(stable, "stablehlo"), optimized_summary
        )["passed"]
        is False
    )


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
    (hwmon / "power1_cap").write_text("400000000\n")
    assert _PROBE._read_hardware_limits(device)["power_cap_uw"] == 400_000_000
    (hwmon / "power1_cap").write_text("400000001\n")
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

    (self_root / "cgroup").write_text(
        "0::/user.slice/user-1000.slice/user@1000.service/app.slice/"
        "skyrl-w8a8-runtime-123-fedcba.scope\n"
    )
    runtime_command = _CONTROLLER._profile_command(
        phase="execute", run_dir=run_dir, card="card1", lock_fd=lock_fd
    )
    (parent_root / "cmdline").write_bytes(
        b"\0".join(value.encode() for value in runtime_command) + b"\0"
    )
    runtime = _PROBE._validate_controller_supervision(
        run_dir, lock_fd, "execute", proc_root=proc_root
    )
    assert runtime["scope"] == "skyrl-w8a8-runtime-123-fedcba.scope"


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


def _fresh_cache_artifacts(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, object], str]:
    root = tmp_path / "artifacts"
    cache_dir = root / "jax-cache"
    cache_dir.mkdir(parents=True, mode=0o700)
    root.chmod(0o700)
    cache_dir.chmod(0o700)
    cache = cache_dir / f"jit_candidate-{'a' * 64}-cache"
    cache.write_bytes(b"fresh-cache")
    cache.chmod(0o600)
    inventory = _PROBE._artifact_inventory(root)
    cache_sha = hashlib.sha256(b"fresh-cache").hexdigest()
    return root, cache, inventory, cache_sha


def _exact_fresh_isa_evidence(cache: Path, cache_sha: str) -> dict[str, object]:
    return {
        "status": "passed_offline_isa_verification",
        "offline_only": True,
        "device_access_performed": False,
        "jax_modules_imported_by_verifier": False,
        "cache": {
            "path": str(cache),
            "sha256": cache_sha,
            "expected_sha256_matched": True,
        },
        "elf_inventory": {
            "elf_count": 7,
            "nested_elf_count": 6,
            "unique_exact_symbol_candidate_count": 1,
            "ordered_nested_contract_matched": True,
        },
        "thunk_inventory": {
            "executable_record_sha256_is_path_dependent": True,
            "caller_bound_autotune_cache_path": str(
                cache.parent / "xla_gpu_per_fusion_autotune_cache_dir"
            ),
            "caller_bound_autotune_cache_path_occurrences": 1,
            "caller_bound_autotune_cache_path_normalized": True,
            "normalized_executable_record_bytes": 52909,
            "normalized_executable_record_sha256": (
                _PROBE._EXPECTED_NORMALIZED_EXECUTABLE_RECORD_SHA256
            ),
            "thunk_count": 6,
            "all_thunks_are_exact_custom_kernels": True,
            "sequential_wrapper_present": False,
            "device_to_device_copy_thunk_present": False,
            "ordered_thunks": [
                {
                    "kernel": kernel,
                    "grid": grid,
                    "threads": threads,
                    "shared_memory_bytes": shared,
                }
                for kernel, grid, threads, shared in _PROBE._EXPECTED_ORDERED_THUNK_LAUNCHES
            ],
        },
        "candidate": {
            "bytes": _PROBE._EXPECTED_NESTED_ELF_BYTES,
            "sha256": _PROBE._EXPECTED_NESTED_ELF_SHA256,
            "expected_sha256_matched": True,
            "written_elf": None,
        },
        "isa": {
            "symbol": _PROBE._EXPECTED_KERNEL_NAME,
            "amdgpu_target": _PROBE._EXPECTED_NESTED_ELF_TARGET,
            "instruction": "v_wmma_i32_16x16x16_iu8",
            "static_instruction_count": 4,
            "signed_neg_lo": [1, 1, 0],
            "resources": {
                "sgpr_count": 34,
                "vgpr_count": 105,
                "sgpr_spill_count": 0,
                "vgpr_spill_count": 0,
                "private_segment_fixed_size": 0,
            },
            "control_flow": copy.deepcopy(_PROBE._EXPECTED_CONTROL_FLOW),
            "tail_store": copy.deepcopy(_PROBE._EXPECTED_TAIL_STORE),
        },
    }


def _set_nested(
    mapping: dict[str, object], path: tuple[str, ...], value: object
) -> None:
    target = mapping
    for key in path[:-1]:
        child = target[key]
        assert isinstance(child, dict)
        target = child
    target[path[-1]] = value


def test_fresh_isa_helper_binds_inventory_digest_and_exact_native_contract(
    monkeypatch, tmp_path: Path
) -> None:
    root, cache, inventory, cache_sha = _fresh_cache_artifacts(tmp_path)
    observed = {}

    def inspect_cache(path, **kwargs):
        observed.update({"path": path, **kwargs})
        return _exact_fresh_isa_evidence(cache, cache_sha)

    import rocm.inspect_w8a8_lora_isa as inspector

    monkeypatch.setattr(inspector, "inspect_cache", inspect_cache)
    result = _PROBE._qualify_fresh_nested_elf(root, inventory)

    assert result["passed"] is True
    assert observed["path"] == cache
    assert observed["expected_cache_sha256"] == cache_sha
    assert observed["expected_elf_sha256"] == _PROBE._EXPECTED_NESTED_ELF_SHA256


@pytest.mark.parametrize("candidate_count", [0, 2])
def test_fresh_isa_helper_rejects_zero_or_two_cache_candidates_without_inspection(
    monkeypatch, tmp_path: Path, candidate_count: int
) -> None:
    root, _cache, inventory, _cache_sha = _fresh_cache_artifacts(tmp_path)
    candidate = next(
        item
        for item in inventory["files"]
        if item["path"].startswith("jax-cache/jit_candidate-")
    )
    inventory = copy.deepcopy(inventory)
    inventory["files"] = [
        item
        for item in inventory["files"]
        if not item["path"].startswith("jax-cache/jit_candidate-")
    ]
    if candidate_count == 2:
        first = copy.deepcopy(candidate)
        second = copy.deepcopy(candidate)
        first["path"] = f"jax-cache/jit_candidate-{'b' * 64}-cache"
        second["path"] = f"jax-cache/jit_candidate-{'c' * 64}-cache"
        inventory["files"].extend([first, second])

    import rocm.inspect_w8a8_lora_isa as inspector

    def unexpected_inspection(*_args, **_kwargs):
        raise AssertionError("inspector must not run for an ambiguous cache inventory")

    monkeypatch.setattr(inspector, "inspect_cache", unexpected_inspection)
    with pytest.raises(
        RuntimeError,
        match=rf"expected one fresh candidate cache artifact, observed {candidate_count}",
    ):
        _PROBE._qualify_fresh_nested_elf(root, inventory)


@pytest.mark.parametrize(
    ("path", "value", "failed_check"),
    [
        (("cache", "sha256"), "0" * 64, "caller_bound_fresh_cache"),
        (("elf_inventory", "elf_count"), 5, "one_unique_exact_symbol_candidate"),
        (
            ("elf_inventory", "ordered_nested_contract_matched"),
            False,
            "one_unique_exact_symbol_candidate",
        ),
        (
            ("elf_inventory", "unique_exact_symbol_candidate_count"),
            2,
            "one_unique_exact_symbol_candidate",
        ),
        (("thunk_inventory", "thunk_count"), 5, "six_ordered_custom_kernel_thunks"),
        (
            ("thunk_inventory", "normalized_executable_record_bytes"),
            52_908,
            "six_ordered_custom_kernel_thunks",
        ),
        (
            ("thunk_inventory", "normalized_executable_record_sha256"),
            "0" * 64,
            "six_ordered_custom_kernel_thunks",
        ),
        (
            ("thunk_inventory", "caller_bound_autotune_cache_path"),
            "/tmp/wrong/xla_gpu_per_fusion_autotune_cache_dir",
            "six_ordered_custom_kernel_thunks",
        ),
        (
            ("thunk_inventory", "caller_bound_autotune_cache_path_occurrences"),
            2,
            "six_ordered_custom_kernel_thunks",
        ),
        (
            ("thunk_inventory", "caller_bound_autotune_cache_path_normalized"),
            False,
            "six_ordered_custom_kernel_thunks",
        ),
        (
            ("thunk_inventory", "executable_record_sha256_is_path_dependent"),
            False,
            "six_ordered_custom_kernel_thunks",
        ),
        (
            ("thunk_inventory", "device_to_device_copy_thunk_present"),
            True,
            "six_ordered_custom_kernel_thunks",
        ),
        (("candidate", "bytes"), 7159, "candidate_bytes_exact"),
        (("candidate", "sha256"), "0" * 64, "candidate_sha256_exact"),
        (
            ("candidate", "expected_sha256_matched"),
            False,
            "candidate_sha256_exact",
        ),
        (("isa", "symbol"), "lookalike", "symbol_exact"),
        (("isa", "amdgpu_target"), "amdgcn--gfx1101", "target_exact"),
        (("isa", "static_instruction_count"), 3, "four_static_signed_iu8_wmma"),
        (("isa", "instruction"), "v_mfma_i32", "four_static_signed_iu8_wmma"),
        (("isa", "signed_neg_lo"), [0, 0, 0], "four_static_signed_iu8_wmma"),
        (("isa", "resources", "sgpr_count"), 35, "registers_exact"),
        (("isa", "resources", "vgpr_count"), 63, "registers_exact"),
        (
            ("isa", "control_flow", "barrier_count"),
            8,
            "full_tile_control_flow_exact",
        ),
        (
            ("isa", "tail_store", "standalone_immediate_17_or_0x11_count"),
            1,
            "full_tile_control_flow_exact",
        ),
        (
            ("isa", "resources", "sgpr_spill_count"),
            1,
            "zero_spills_and_private_segment",
        ),
        (
            ("isa", "resources", "vgpr_spill_count"),
            1,
            "zero_spills_and_private_segment",
        ),
        (
            ("isa", "resources", "private_segment_fixed_size"),
            4,
            "zero_spills_and_private_segment",
        ),
    ],
)
def test_fresh_isa_helper_fails_closed_on_nested_evidence_mutation(
    monkeypatch,
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    failed_check: str,
) -> None:
    root, cache, inventory, cache_sha = _fresh_cache_artifacts(tmp_path)
    evidence = _exact_fresh_isa_evidence(cache, cache_sha)
    _set_nested(evidence, path, value)

    import rocm.inspect_w8a8_lora_isa as inspector

    monkeypatch.setattr(inspector, "inspect_cache", lambda *_args, **_kwargs: evidence)
    with pytest.raises(RuntimeError, match=failed_check):
        _PROBE._qualify_fresh_nested_elf(root, inventory)


def _jsonl_records(output: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in output.getvalue().splitlines()]


class _FakeRuntimeOutput:
    def __init__(self, value: object, operations: list[str]) -> None:
        self.value = value
        self._operations = operations

    def block_until_ready(self):
        self._operations.append("output_block_until_ready")
        return self


class _FakeRuntimeCompiled:
    def __init__(
        self,
        output: io.StringIO,
        expected: object,
        operations: list[str],
        *,
        dispatch_failure: bool,
    ) -> None:
        self._output = output
        self._expected = expected
        self._operations = operations
        self._dispatch_failure = dispatch_failure

    def __call__(self, *arguments):
        assert len(arguments) == 6
        records = _jsonl_records(self._output)
        assert records[-1]["record_type"] == "dispatch_started"
        assert records[-1]["one_shot_capability"] == {
            "consumed": True,
            "compiled_executable_attempts": 1,
            "compiled_executable_completions": 0,
        }
        self._operations.append("compiled_call_after_capability_consumed")
        if self._dispatch_failure:
            raise RuntimeError("synthetic dispatch failure")
        return _FakeRuntimeOutput(np.array(self._expected, copy=True), self._operations)

    def as_text(self) -> str:
        self._operations.append("compiled_as_text")
        return _optimized_hlo()

    def memory_analysis(self):
        self._operations.append("compiled_memory_analysis")
        return types.SimpleNamespace(
            argument_size_in_bytes=4096,
            output_size_in_bytes=102,
            alias_size_in_bytes=0,
            temp_size_in_bytes=4096,
            generated_code_size_in_bytes=8440,
        )


class _FakeRuntimeLowered:
    def __init__(self, compiled: _FakeRuntimeCompiled, operations: list[str]) -> None:
        self._compiled = compiled
        self._operations = operations

    def compiler_ir(self, *, dialect: str):
        assert dialect == "stablehlo"
        self._operations.append("lowered_compiler_ir")
        return _stablehlo()

    def compile(self) -> _FakeRuntimeCompiled:
        self._operations.append("lowered_compile")
        return self._compiled


class _FakeRuntimeJitted:
    def __init__(self, lowered: _FakeRuntimeLowered, operations: list[str]) -> None:
        self._lowered = lowered
        self._operations = operations

    def lower(self, *signature) -> _FakeRuntimeLowered:
        assert len(signature) == 6
        self._operations.append("jitted_lower")
        return self._lowered


def _install_fake_jax_runtime(
    monkeypatch,
    output: io.StringIO,
    expected: object,
    operations: list[str],
    *,
    dispatch_failure: bool,
) -> None:
    device = types.SimpleNamespace(platform="gpu", id=0)
    compiled = _FakeRuntimeCompiled(
        output,
        expected,
        operations,
        dispatch_failure=dispatch_failure,
    )
    lowered = _FakeRuntimeLowered(compiled, operations)

    fake_jax = types.ModuleType("jax")
    fake_jax.__path__ = []
    fake_jax.__version__ = "0.10.2"
    fake_jnp = types.ModuleType("jax.numpy")
    fake_jaxlib = types.ModuleType("jaxlib")
    fake_jaxlib.__version__ = "0.10.2"
    fake_extend = types.ModuleType("jax.extend")
    fake_extend.__path__ = []
    fake_backend = types.ModuleType("jax.extend.backend")

    def shape_dtype_struct(shape, dtype):
        operations.append("shape_dtype_struct")
        return types.SimpleNamespace(shape=shape, dtype=dtype)

    def default_backend() -> str:
        operations.append("default_backend")
        return "gpu"

    def get_backend():
        operations.append("get_backend")
        return types.SimpleNamespace(platform_version="ROCm 7.2.4 synthetic")

    def devices():
        operations.append("devices")
        return [device]

    def jit(function):
        assert callable(function)
        operations.append("jit")
        return _FakeRuntimeJitted(lowered, operations)

    def device_put(arguments, *, device: object):
        assert device is not None
        assert len(arguments) == 6
        operations.append("tuple_device_put")
        return tuple(arguments)

    def block_until_ready(arguments):
        assert isinstance(arguments, tuple) and len(arguments) == 6
        operations.append("input_block_until_ready")
        return arguments

    def device_get(candidate_output):
        assert isinstance(candidate_output, _FakeRuntimeOutput)
        operations.append("device_get")
        return candidate_output.value

    fake_jax.ShapeDtypeStruct = shape_dtype_struct
    fake_jax.default_backend = default_backend
    fake_jax.devices = devices
    fake_jax.jit = jit
    fake_jax.device_put = device_put
    fake_jax.block_until_ready = block_until_ready
    fake_jax.device_get = device_get
    fake_jax.numpy = fake_jnp
    fake_backend.get_backend = get_backend
    fake_extend.backend = fake_backend
    fake_jax.extend = fake_extend

    fake_quantized = types.ModuleType("skyrl.tx.kernels.quantized_lora")
    fake_quantized.__file__ = str(
        _REPO / "skyrl" / "tx" / "kernels" / "quantized_lora.py"
    )
    fake_w8a8 = types.ModuleType("skyrl.tx.kernels.rocm.w8a8_lora")
    fake_w8a8.__file__ = str(
        _REPO / "skyrl" / "tx" / "kernels" / "rocm" / "w8a8_lora.py"
    )

    import skyrl.tx.kernels as kernels_package
    import skyrl.tx.kernels.rocm as rocm_package

    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_jnp)
    monkeypatch.setitem(sys.modules, "jaxlib", fake_jaxlib)
    monkeypatch.setitem(sys.modules, "jax.extend", fake_extend)
    monkeypatch.setitem(sys.modules, "jax.extend.backend", fake_backend)
    monkeypatch.setitem(sys.modules, "skyrl.tx.kernels.quantized_lora", fake_quantized)
    monkeypatch.setitem(sys.modules, "skyrl.tx.kernels.rocm.w8a8_lora", fake_w8a8)
    monkeypatch.setattr(
        kernels_package, "quantized_lora", fake_quantized, raising=False
    )
    monkeypatch.setattr(rocm_package, "w8a8_lora", fake_w8a8, raising=False)


def _prepare_fake_rocm_execute(
    monkeypatch,
    tmp_path: Path,
    *,
    dispatch_failure: bool,
) -> tuple[io.StringIO, list[str], Path, Callable[[], dict[str, bool]]]:
    output = io.StringIO()
    operations: list[str] = []
    _, _, expected = _PROBE._construct_host_case()
    _install_fake_jax_runtime(
        monkeypatch,
        output,
        expected,
        operations,
        dispatch_failure=dispatch_failure,
    )
    paths = _artifact_paths(tmp_path)
    artifacts = paths["jax_cache"].parent
    cache = paths["jax_cache"] / f"jit_candidate-{'d' * 64}-cache"
    cache.write_bytes(b"synthetic-fresh-cache")
    cache.chmod(0o600)

    def module_origins(_modules):
        operations.append("module_origins")
        return {"synthetic": "strictly mocked"}

    def hardware_limits(_device_root):
        operations.append("hardware_limits")
        return {
            "power_cap_uw": 400_000_000,
            "junction_temperature_millic": 40_000,
        }

    def fresh_qualification(_artifact_root, inventory):
        assert inventory["file_count"] == 1
        operations.append("fresh_isa_qualification")
        return {"checks": {"synthetic_exact": True}, "passed": True}

    original_validation = _PROBE._runtime_numerical_validation

    def numerical_validation(actual, reference, dispatch_seconds):
        operations.append("runtime_numerical_validation")
        return original_validation(actual, reference, dispatch_seconds)

    def source_postflight():
        operations.append("source_postflight")
        return {"synthetic": "source"}

    def git_postflight():
        operations.append("git_postflight")
        return {"worktree_clean": True}

    def stack_postflight():
        operations.append("stack_postflight")
        return {"synthetic": "stack"}

    def require_clean_boot():
        operations.append("require_clean_boot")
        return {"clean": True}

    monkeypatch.setattr(_PROBE, "_module_origin_manifest", module_origins)
    monkeypatch.setattr(_PROBE, "_read_hardware_limits", hardware_limits)
    monkeypatch.setattr(_PROBE, "_qualify_fresh_nested_elf", fresh_qualification)
    monkeypatch.setattr(_PROBE, "_runtime_numerical_validation", numerical_validation)
    monkeypatch.setattr(_PROBE, "_assert_bound_sources", source_postflight)
    monkeypatch.setattr(_PROBE, "_git_manifest", git_postflight)
    monkeypatch.setattr(_PROBE, "_stack_manifest", stack_postflight)
    monkeypatch.setenv("XLA_FLAGS", _PROBE._COMMAND_BUFFER_FLAG)
    return output, operations, artifacts, require_clean_boot


_SUCCESS_RUNTIME_OPERATIONS = [
    *("shape_dtype_struct" for _ in range(6)),
    "module_origins",
    "default_backend",
    "get_backend",
    "devices",
    "hardware_limits",
    "require_clean_boot",
    "hardware_limits",
    "jit",
    "jitted_lower",
    "require_clean_boot",
    "lowered_compiler_ir",
    "hardware_limits",
    "lowered_compile",
    "require_clean_boot",
    "hardware_limits",
    "compiled_as_text",
    "compiled_memory_analysis",
    "fresh_isa_qualification",
    "require_clean_boot",
    "tuple_device_put",
    "input_block_until_ready",
    "require_clean_boot",
    "hardware_limits",
    "require_clean_boot",
    "compiled_call_after_capability_consumed",
    "output_block_until_ready",
    "require_clean_boot",
    "device_get",
    "require_clean_boot",
    "runtime_numerical_validation",
    "source_postflight",
    "git_postflight",
    "stack_postflight",
    "require_clean_boot",
]


def test_run_rocm_execute_success_is_exactly_one_shot_with_fake_jax(
    monkeypatch, tmp_path: Path
) -> None:
    output, operations, artifacts, require_clean_boot = _prepare_fake_rocm_execute(
        monkeypatch, tmp_path, dispatch_failure=False
    )

    result = _PROBE._run_rocm(
        argparse.Namespace(phase="execute"),
        output,
        artifacts,
        tmp_path / "synthetic-device",
        require_clean_boot,
    )

    assert result == 0
    assert operations == _SUCCESS_RUNTIME_OPERATIONS
    records = _jsonl_records(output)
    assert [record["record_type"] for record in records] == [
        "host_oracle",
        "backend_ready",
        "journal_checkpoint",
        "lowered",
        "compiled_unreleased",
        "fresh_isa_qualification",
        "executable_released",
        "input_device_put",
        "journal_checkpoint",
        "dispatch_preflight",
        "dispatch_started",
        "journal_checkpoint",
        "dispatch",
        "device_get",
        "journal_checkpoint",
        "numerical_validation",
        "completed",
    ]
    by_type = {record["record_type"]: record for record in records}
    assert by_type["dispatch_preflight"]["one_shot_capability"] == {
        "consumed": False,
        "compiled_executable_attempts": 0,
        "compiled_executable_completions": 0,
    }
    assert by_type["dispatch_started"]["one_shot_capability"] == {
        "consumed": True,
        "compiled_executable_attempts": 1,
        "compiled_executable_completions": 0,
    }
    post_attempt = next(
        record
        for record in records
        if record["record_type"] == "journal_checkpoint"
        and record.get("stage") == "after_candidate_dispatch_attempt"
    )
    assert post_attempt["compiled_executable_attempts"] == 1
    assert post_attempt["compiled_executable_completions"] == 0
    assert by_type["dispatch"]["one_shot_capability"] == {
        "consumed": True,
        "compiled_executable_attempts": 1,
        "compiled_executable_completions": 1,
    }
    assert by_type["dispatch"]["output_readiness_invocations"] == 1
    assert by_type["device_get"]["device_get_attempts"] == 1
    assert by_type["device_get"]["device_get_completions"] == 1
    assert operations.count("compiled_call_after_capability_consumed") == 1
    assert operations.count("output_block_until_ready") == 1
    assert operations.count("device_get") == 1


def test_run_rocm_dispatch_failure_consumes_once_and_never_copies_back(
    monkeypatch, tmp_path: Path
) -> None:
    output, operations, artifacts, require_clean_boot = _prepare_fake_rocm_execute(
        monkeypatch, tmp_path, dispatch_failure=True
    )

    with pytest.raises(RuntimeError, match="synthetic dispatch failure"):
        _PROBE._run_rocm(
            argparse.Namespace(phase="execute"),
            output,
            artifacts,
            tmp_path / "synthetic-device",
            require_clean_boot,
        )

    failure_prefix_length = _SUCCESS_RUNTIME_OPERATIONS.index(
        "output_block_until_ready"
    )
    assert operations == _SUCCESS_RUNTIME_OPERATIONS[:failure_prefix_length] + [
        "require_clean_boot"
    ]
    records = _jsonl_records(output)
    assert [record["record_type"] for record in records] == [
        "host_oracle",
        "backend_ready",
        "journal_checkpoint",
        "lowered",
        "compiled_unreleased",
        "fresh_isa_qualification",
        "executable_released",
        "input_device_put",
        "journal_checkpoint",
        "dispatch_preflight",
        "dispatch_started",
        "journal_checkpoint",
    ]
    started = records[-2]
    post_attempt = records[-1]
    assert started["one_shot_capability"] == {
        "consumed": True,
        "compiled_executable_attempts": 1,
        "compiled_executable_completions": 0,
    }
    assert post_attempt["stage"] == "after_candidate_dispatch_attempt"
    assert post_attempt["compiled_executable_attempts"] == 1
    assert post_attempt["compiled_executable_completions"] == 0
    assert operations.count("compiled_call_after_capability_consumed") == 1
    assert "output_block_until_ready" not in operations
    assert "device_get" not in operations
    assert "runtime_numerical_validation" not in operations


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


def test_execute_source_contains_one_transfer_dispatch_and_copyback_site() -> None:
    source = _PROBE_PATH.read_text(encoding="utf-8")

    assert source.count("jax.device_put(host_arguments, device=devices[0])") == 1
    assert source.count("jax.block_until_ready(device_arguments)") == 1
    assert source.count("candidate_output = compiled(*device_arguments)") == 1
    assert source.count("candidate_output.block_until_ready()") == 1
    assert source.count("host_actual = jax.device_get(candidate_output)") == 1
