from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).parents[2]
_PROBE_PATH = _REPO / "rocm" / "probe_gdn_composed_s512_ffi.py"
_PROFILE_PATH = _REPO / "rocm" / "profile_rocm.py"
_SUPERVISOR_PATH = _REPO / "rocm" / "process_supervision.py"
_SPEC = importlib.util.spec_from_file_location("gdn_composed_s512_probe_test", _PROBE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)
_EXECUTE_HELPER, _PREPARE_HELPER = _PROBE._load_helpers()


def _layouts(*values: str) -> str:
    return "[" + ", ".join(f"dense<[{value}]>" for value in values) + "]"


def _stablehlo(
    *,
    execute_query: str = "%arg0",
    execute_u: str = "%prep#0",
    before_execute: str = "",
    extra: str = "",
    returned: tuple[str, str] = ("%execute#0", "%execute#1"),
) -> str:
    prepare_operand_layouts = _layouts("3,2,1,0", "3,2,1,0", "2,1,0", "2,1,0")
    prepare_result_layouts = _layouts("3,2,1,0", "3,2,1,0", "2,1,0")
    execute_operand_layouts = _layouts(
        "3,2,1,0", "3,2,1,0", "3,2,1,0", "3,2,1,0", "2,1,0", "3,2,1,0"
    )
    execute_result_layouts = _layouts("3,2,1,0", "3,2,1,0")
    return "\n".join(
        [
            "module {",
            "  func.func @main(%arg0: tensor<1x512x16x128xf32>, "
            "%arg1: tensor<1x512x16x128xf32>, "
            "%arg2: tensor<1x512x32x128xf32>, "
            "%arg3: tensor<1x512x32xf32>, %arg4: tensor<1x512x32xf32>, "
            "%arg5: tensor<1x32x128x128xf32>) -> "
            "(tensor<1x512x32x128xbf16>, tensor<1x32x128x128xf32>) {",
            f'    %prep:3 = stablehlo.custom_call @"{_PROBE._PREPARE_TARGET}"'
            "(%arg1, %arg2, %arg3, %arg4) {",
            f"      operand_layouts = {prepare_operand_layouts},",
            f"      result_layouts = {prepare_result_layouts}",
            "    } : (tensor<1x512x16x128xf32>, tensor<1x512x32x128xf32>, "
            "tensor<1x512x32xf32>, tensor<1x512x32xf32>) -> "
            "tuple<tensor<1x512x32x128xf32>, tensor<1x512x32x128xf32>, "
            "tensor<1x512x32xf32>>",
            before_execute,
            f'    %execute:2 = stablehlo.custom_call @"{_PROBE._EXECUTE_TARGET}"'
            f"({execute_query}, %arg1, {execute_u}, %prep#1, %prep#2, %arg5) {{",
            f"      operand_layouts = {execute_operand_layouts},",
            f"      result_layouts = {execute_result_layouts}",
            "    } : (tensor<1x512x16x128xf32>, tensor<1x512x16x128xf32>, "
            "tensor<1x512x32x128xf32>, tensor<1x512x32x128xf32>, "
            "tensor<1x512x32xf32>, tensor<1x32x128x128xf32>) -> "
            "tuple<tensor<1x512x32x128xbf16>, tensor<1x32x128x128xf32>>",
            extra,
            f"    return {returned[0]}, {returned[1]} : "
            "tensor<1x512x32x128xbf16>, tensor<1x32x128x128xf32>",
            "  }",
            "}",
        ]
    )


def _optimized(
    *,
    execute_query: str = "%query",
    execute_u: str = "%u",
    extra: str = "",
    execute_is_root: bool = True,
    unrelated_root: str = "",
) -> str:
    return "\n".join(
        [
            "HloModule composed",
            "ENTRY %main {",
            "  %query = f32[1,512,16,128]{3,2,1,0} parameter(0)",
            "  %key = f32[1,512,16,128]{3,2,1,0} parameter(1)",
            "  %value = f32[1,512,32,128]{3,2,1,0} parameter(2)",
            "  %g = f32[1,512,32]{2,1,0} parameter(3)",
            "  %beta = f32[1,512,32]{2,1,0} parameter(4)",
            "  %state = f32[1,32,128,128]{3,2,1,0} parameter(5)",
            "  %prepare = (f32[1,512,32,128]{3,2,1,0}, "
            "f32[1,512,32,128]{3,2,1,0}, f32[1,512,32]{2,1,0}) "
            "custom-call(%key, %value, %g, %beta), "
            f'custom_call_target="{_PROBE._PREPARE_TARGET}"',
            "  %u = f32[1,512,32,128]{3,2,1,0} get-tuple-element(%prepare), index=0",
            "  %w = f32[1,512,32,128]{3,2,1,0} get-tuple-element(%prepare), index=1",
            "  %gamma = f32[1,512,32]{2,1,0} get-tuple-element(%prepare), index=2",
            extra,
            f"  {'ROOT ' if execute_is_root else ''}%execute = "
            "(bf16[1,512,32,128]{3,2,1,0}, "
            "f32[1,32,128,128]{3,2,1,0}) custom-call("
            f"{execute_query}, %key, {execute_u}, %w, %gamma, %state), "
            f'custom_call_target="{_PROBE._EXECUTE_TARGET}"',
            unrelated_root,
            "}",
        ]
    )


def _profile_argv(telemetry: Path, current: list[str], power: str) -> list[str]:
    return [
        sys.executable,
        str(_PROFILE_PATH),
        "--timeout",
        "300",
        "--interval",
        "0.1",
        "--sensor-grace-seconds",
        "60",
        "--max-junction-temp-c",
        "90",
        "--max-gpu-power-watts",
        power,
        "--max-vram-gib",
        "24",
        "--min-host-available-gib",
        "0",
        "--max-swap-gib",
        "8",
        "--card",
        "card1",
        "--output",
        str(telemetry),
        "--",
        *current,
    ]


def test_default_mode_is_import_light_and_refuses_gpu() -> None:
    tree = ast.parse(_PROBE_PATH.read_text(encoding="utf-8"))
    top_imports = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }
    assert top_imports.isdisjoint({"jax", "jaxlib", "numpy", "skyrl"})
    result = subprocess.run(
        [sys.executable, str(_PROBE_PATH)],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert [item["record_type"] for item in records] == ["manifest", "refused"]
    assert records[1]["status"] == "no_gpu_abstract_manifest_only"
    assert records[1]["jax_imported"] is False
    assert records[1]["numpy_imported"] is False


def test_contract_is_exact_two_call_one_shot_with_current_resource_envelope() -> None:
    contract = _PROBE._exact_contract()
    assert contract["targets_in_dataflow_order"] == [
        _PROBE._PREPARE_TARGET,
        _PROBE._EXECUTE_TARGET,
    ]
    assert contract["compile_gate"] == {
        "exact_custom_calls_per_dialect": 2,
        "prepare_results_feed_execute_positions": [2, 3, 4],
        "while_calls": 0,
        "alias_bytes": 0,
        "argument_bytes": 19_005_440,
        "logical_intermediate_bytes": 16_842_752,
        "logical_output_bytes": 6_291_456,
        "compiler_output_bytes": 6_291_472,
        "temporary_bytes_range": [16_842_752, 64 * 1024**2],
        "combined_bytes_maximum": 96 * 1024**2,
    }
    assert contract["outer_supervision"] == {
        "maximum_junction_temperature_c": 90.0,
        "maximum_gpu_power_watts": 400.0,
        "maximum_gpu_vram_gib": 24.0,
    }
    assert contract["one_shot"] is True
    assert [contract[name] for name in ("warmup", "replay", "graph", "backward", "model")] == [0] * 5


def test_runtime_and_compile_diagnostic_terminal_counters_are_separate_and_exact() -> None:
    zero = _PROBE._zero_counters()
    runtime = _PROBE._completed_runtime_counters()
    diagnostic = _PROBE._completed_compile_diagnostic_counters()
    assert runtime.keys() == diagnostic.keys() == zero.keys()
    assert _PROBE._assert_terminal_counters(
        runtime, compile_diagnostic=False
    ) == {
        "passed": True,
        "mode": "runtime",
        "exact_full_counter_map": True,
        "forbidden_invocation_counts_zero": True,
    }
    assert _PROBE._assert_terminal_counters(
        diagnostic, compile_diagnostic=True
    )["passed"] is True
    assert diagnostic["prepare_oracle_attempts"] == 0
    assert diagnostic["checked_executable_attempts"] == 0
    assert runtime["prepare_oracle_attempts"] == 1
    assert runtime["checked_executable_attempts"] == 1
    forbidden = (
        "warmup_invocations",
        "replay_invocations",
        "graph_invocations",
        "gpu_reference_invocations",
        "gpu_reduction_invocations",
        "backward_invocations",
        "model_invocations",
    )
    assert all(runtime[name] == diagnostic[name] == 0 for name in forbidden)
    corrupted = dict(runtime)
    corrupted["warmup_invocations"] = 1
    with pytest.raises(RuntimeError, match="counter contract"):
        _PROBE._assert_terminal_counters(corrupted, compile_diagnostic=False)


def test_runtime_wrapper_imports_bind_canonical_packages_loader_and_sources() -> None:
    prepare, execute, proof = _PROBE._import_bound_wrappers()
    assert proof["passed"] is True
    assert proof["all_runtime_modules_exact"] is True
    assert all(proof["relationships"].values())
    assert set(proof["final_source_sha256"]) == set(
        _PROBE._BOUND_RUNTIME_MODULES.values()
    )
    assert prepare.__file__ is not None
    assert execute.__file__ is not None
    assert Path(prepare.__file__).resolve() == _PROBE._source_files()[
        "prepare_wrapper"
    ].resolve()
    assert Path(execute.__file__).resolve() == _PROBE._source_files()[
        "execute_wrapper"
    ].resolve()
    for source_name, digest in proof["final_source_sha256"].items():
        assert digest == _PROBE._EXPECTED_SOURCE_SHA256[source_name]


def test_profile_supervision_sources_are_exact_and_watcher_is_outer_only() -> None:
    files = _PROBE._source_files()
    assert files["profile_rocm"].resolve() == _PROFILE_PATH.resolve()
    assert files["process_supervision"].resolve() == _SUPERVISOR_PATH.resolve()
    assert "watch_rocm_safety" not in files
    assert "watch_rocm_safety" not in _PROBE._EXPECTED_SOURCE_SHA256
    assert _PROBE._EXPECTED_SOURCE_SHA256["profile_rocm"] == (
        "a991401c0f54921456685fbbc47a12d50616ae0921238044837908ce49ff4551"
    )
    assert _PROBE._EXPECTED_SOURCE_SHA256["process_supervision"] == (
        "05f9d84313db2dc6e2262e53e65f42f044ade2cb596d5310c289427e9cd7e3cf"
    )

    bound = _PROBE._assert_bound_sources()
    assert bound["process_supervision"] == _PROBE._EXPECTED_SOURCE_SHA256[
        "process_supervision"
    ]
    proof = _PROBE._assert_profile_supervision_sources()
    assert proof == {
        "passed": True,
        "all_profile_supervision_sources_exact": True,
        "final_source_sha256": {
            name: _PROBE._EXPECTED_SOURCE_SHA256[name]
            for name in _PROBE._PROFILE_SUPERVISION_SOURCES
        },
    }


@pytest.mark.parametrize("dependency_state", ["missing", "mutated"])
def test_bound_supervisor_dependency_rejects_missing_or_mutated_source(
    dependency_state: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    replacement = tmp_path / "process_supervision.py"
    if dependency_state == "mutated":
        replacement.write_bytes(_SUPERVISOR_PATH.read_bytes() + b"\n# mutation\n")
    files = dict(_PROBE._source_files())
    files["process_supervision"] = replacement
    monkeypatch.setattr(_PROBE, "_source_files", lambda: files)

    expected = "unavailable" if dependency_state == "missing" else "hash mismatch"
    with pytest.raises(RuntimeError, match=f"dependency source {expected}"):
        _PROBE._assert_bound_sources()
    with pytest.raises(RuntimeError, match=f"dependency source {expected}"):
        _PROBE._assert_profile_supervision_sources()


def test_loaded_wrapper_identity_rejects_fabricated_file_origin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepare, _execute, _proof = _PROBE._import_bound_wrappers()
    fabricated = tmp_path / "gdn_prepare_ffi.py"
    fabricated.write_text("# shadow\n", encoding="utf-8")
    monkeypatch.setattr(prepare, "__file__", str(fabricated))
    with pytest.raises(RuntimeError, match="not the committed source"):
        _PROBE._exact_loaded_source(
            "skyrl.tx.kernels.rocm.gdn_prepare_ffi", "prepare_wrapper"
        )


def test_runtime_wrapper_resolution_rejects_pythonpath_shadow(tmp_path: Path) -> None:
    shadow = tmp_path / "skyrl"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("SHADOW = True\n", encoding="utf-8")
    script = "\n".join(
        [
            "import importlib.util",
            "from pathlib import Path",
            f"path = Path({str(_PROBE_PATH)!r})",
            "spec = importlib.util.spec_from_file_location('composed_shadow_gate', path)",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            "try:",
            "    module._exact_source_spec('skyrl', 'package_skyrl')",
            "except RuntimeError:",
            "    print('rejected')",
            "else:",
            "    raise SystemExit('shadow unexpectedly accepted')",
        ]
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "rejected"


def test_profile_verifier_accepts_400_watts_and_rejects_above(tmp_path: Path) -> None:
    telemetry = (tmp_path / "telemetry.jsonl").resolve()
    telemetry.write_bytes(b"{}\n")
    telemetry.chmod(0o600)
    current = [sys.executable, str(_PROBE_PATH), "--platform", "rocm"]
    proof = _PROBE._validate_profile_argv(
        _EXECUTE_HELPER,
        _profile_argv(telemetry, current, "400"),
        current,
        _REPO,
        _PROFILE_PATH.resolve(),
    )
    assert proof["passed"] is True
    assert proof["resource_values"]["power"] == 400.0
    assert proof["maximum_gpu_power_watts"] == 400.0
    assert proof["legacy_verifier_power_normalized"] is True
    with pytest.raises(RuntimeError, match="GPU power contract"):
        _PROBE._validate_profile_argv(
            _EXECUTE_HELPER,
            _profile_argv(telemetry, current, "400.01"),
            current,
            _REPO,
            _PROFILE_PATH.resolve(),
        )


def test_stablehlo_requires_exact_two_call_dataflow() -> None:
    summary = _PROBE._composed_ir_summary(
        _stablehlo(), "stablehlo", _PREPARE_HELPER, _EXECUTE_HELPER
    )
    assert summary["passed"] is True
    assert summary["custom_call_count"] == 2
    assert summary["dataflow"]["prepare_origins_by_execute_operand"] == [
        [],
        [],
        [0],
        [1],
        [2],
        [],
    ]
    swapped = _PROBE._composed_ir_summary(
        _stablehlo(execute_u="%prep#1"),
        "stablehlo",
        _PREPARE_HELPER,
        _EXECUTE_HELPER,
    )
    assert swapped["passed"] is False
    assert swapped["dataflow"]["passed"] is False
    looped = _PROBE._composed_ir_summary(
        _stablehlo(extra="    stablehlo.while"),
        "stablehlo",
        _PREPARE_HELPER,
        _EXECUTE_HELPER,
    )
    assert looped["passed"] is False
    assert looped["checks"]["no_while"] is False


@pytest.mark.parametrize(
    ("text", "failed_check"),
    [
        (
            _stablehlo(
                execute_u="%double_u",
                before_execute=(
                    "    %double_u = stablehlo.add %prep#0, %prep#0 : "
                    "tensor<1x512x32x128xf32>"
                ),
            ),
            "direct_prepare_results_in_exact_positions",
        ),
        (
            _stablehlo(
                execute_query="%negated_query",
                before_execute=(
                    "    %negated_query = stablehlo.negate %arg0 : "
                    "tensor<1x512x16x128xf32>"
                ),
            ),
            "execute_external_operands_are_exact_entry_arguments",
        ),
    ],
)
def test_stablehlo_rejects_transformed_execute_operands(
    text: str, failed_check: str
) -> None:
    summary = _PROBE._composed_ir_summary(
        text, "stablehlo", _PREPARE_HELPER, _EXECUTE_HELPER
    )
    assert summary["passed"] is False
    assert summary["dataflow"]["checks"][failed_check] is False


def test_stablehlo_requires_execute_results_as_exact_function_return() -> None:
    text = _stablehlo(
        before_execute=(
            "    %unrelated_output = stablehlo.convert %arg2 : "
            "(tensor<1x512x32x128xf32>) -> tensor<1x512x32x128xbf16>"
        ),
        returned=("%unrelated_output", "%arg5"),
    )
    summary = _PROBE._composed_ir_summary(
        text, "stablehlo", _PREPARE_HELPER, _EXECUTE_HELPER
    )
    assert summary["passed"] is False
    assert (
        summary["result_proof"]["checks"][
            "execute_results_are_exact_function_return"
        ]
        is False
    )


def test_optimized_hlo_requires_gte_indices_and_entry_parameters() -> None:
    summary = _PROBE._composed_ir_summary(
        _optimized(), "optimized_hlo", _PREPARE_HELPER, _EXECUTE_HELPER
    )
    assert summary["passed"] is True
    assert summary["dataflow"]["prepare_origins_by_execute_operand"] == [
        [],
        [],
        [0],
        [1],
        [2],
        [],
    ]
    wrong_gte = _PROBE._composed_ir_summary(
        _optimized(execute_u="%w"),
        "optimized_hlo",
        _PREPARE_HELPER,
        _EXECUTE_HELPER,
    )
    assert wrong_gte["passed"] is False
    assert wrong_gte["dataflow"]["passed"] is False
    wrong_parameter = _PROBE._composed_ir_summary(
        _optimized().replace("%key, %value, %g, %beta", "%query, %value, %g, %beta"),
        "optimized_hlo",
        _PREPARE_HELPER,
        _EXECUTE_HELPER,
    )
    assert wrong_parameter["passed"] is False
    assert wrong_parameter["dataflow"]["checks"]["prepare_consumes_exact_entry_parameters"] is False


@pytest.mark.parametrize(
    ("text", "failed_check"),
    [
        (
            _optimized(
                execute_u="%double_u",
                extra=(
                    "  %double_u = f32[1,512,32,128]{3,2,1,0} "
                    "add(%u, %u)"
                ),
            ),
            "no_transformed_prepare_or_execute_operands",
        ),
        (
            _optimized(
                execute_query="%negated_query",
                extra=(
                    "  %negated_query = f32[1,512,16,128]{3,2,1,0} "
                    "negate(%query)"
                ),
            ),
            "external_execute_operands_are_exact_entry_parameters",
        ),
    ],
)
def test_optimized_hlo_rejects_transformed_execute_operands(
    text: str, failed_check: str
) -> None:
    summary = _PROBE._composed_ir_summary(
        text, "optimized_hlo", _PREPARE_HELPER, _EXECUTE_HELPER
    )
    assert summary["passed"] is False
    assert summary["dataflow"]["checks"][failed_check] is False


def test_optimized_hlo_requires_execute_custom_call_as_exact_root() -> None:
    unrelated_root = "\n".join(
        [
            "  %unrelated_output = bf16[1,512,32,128]{3,2,1,0} convert(%value)",
            "  ROOT %unrelated = (bf16[1,512,32,128]{3,2,1,0}, "
            "f32[1,32,128,128]{3,2,1,0}) tuple(%unrelated_output, %state)",
        ]
    )
    summary = _PROBE._composed_ir_summary(
        _optimized(execute_is_root=False, unrelated_root=unrelated_root),
        "optimized_hlo",
        _PREPARE_HELPER,
        _EXECUTE_HELPER,
    )
    assert summary["passed"] is False
    assert (
        summary["result_proof"]["checks"][
            "execute_custom_call_is_root_instruction"
        ]
        is False
    )
    assert (
        summary["result_proof"]["checks"]["execute_result_is_exact_entry_root"]
        is False
    )


def test_memory_gate_requires_visible_intermediates_no_alias_and_exact_io() -> None:
    exact = {
        "argument_size_in_bytes": _PROBE._ARGUMENT_BYTES,
        "output_size_in_bytes": _PROBE._COMPILER_OUTPUT_BYTES,
        "alias_size_in_bytes": 0,
        "temp_size_in_bytes": _PROBE._INTERMEDIATE_BYTES,
    }
    assert _PROBE._memory_gate(exact)["passed"] is True
    for name, value in (
        ("argument_size_in_bytes", _PROBE._ARGUMENT_BYTES - 1),
        ("output_size_in_bytes", _PROBE._COMPILER_OUTPUT_BYTES - 1),
        ("alias_size_in_bytes", 1),
        ("temp_size_in_bytes", _PROBE._INTERMEDIATE_BYTES - 1),
        ("temp_size_in_bytes", _PROBE._MAX_TEMP_BYTES + 1),
    ):
        changed = dict(exact)
        changed[name] = value
        assert _PROBE._memory_gate(changed)["passed"] is False


def test_checked_executable_is_single_use() -> None:
    counters = _PROBE._zero_counters()

    class Compiled:
        calls = 0

        def __call__(self, *_values: Any) -> tuple[str, str]:
            self.calls += 1
            return "output", "state"

    class Jax:
        @staticmethod
        def block_until_ready(value: Any) -> Any:
            return value

    compiled = Compiled()
    executable = _PROBE._CheckedExecutable(
        compiled, {"passed": True}, counters, _PROBE._CHECKED_TOKEN
    )
    assert executable.invoke(Jax(), tuple(range(6))) == ("output", "state")
    with pytest.raises(RuntimeError, match="consumed"):
        executable.invoke(Jax(), tuple(range(6)))
    assert compiled.calls == 1
    assert counters["checked_executable_attempts"] == 1
    assert counters["checked_executable_completions"] == 1


def test_post_validation_host_rehash_rejects_mutated_primal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primals = tuple(bytearray([index]) for index in range(6))

    class Helper:
        @staticmethod
        def _array_sha256(value: bytearray) -> str:
            return hashlib.sha256(value).hexdigest()

    expected = {
        name: Helper._array_sha256(value)
        for name, value in zip(_PROBE._INPUT_NAMES, primals, strict=True)
    }
    monkeypatch.setattr(_PROBE, "_EXPECTED_INPUT_SHA256", expected)
    report = {"input_sha256": dict(expected)}
    proof = _PROBE._assert_host_primals_unchanged(primals, report, Helper)
    assert proof["passed"] is True
    primals[2][0] ^= 1
    with pytest.raises(RuntimeError, match="host input boundary changed"):
        _PROBE._assert_host_primals_unchanged(primals, report, Helper)


def test_host_oracle_exact_hashes_require_attested_single_thread_blas() -> None:
    script = "\n".join(
        [
            "import importlib.util, json",
            "from pathlib import Path",
            "import numpy as np",
            f"path = Path({str(_PROBE_PATH)!r})",
            "spec = importlib.util.spec_from_file_location('composed_oracle_child', path)",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            "execute, _prepare = module._load_helpers()",
            "prepare_oracle = module._load_exact_module(module._source_files()['prepare_oracle'], module._EXPECTED_SOURCE_SHA256['prepare_oracle'], 'prepare_oracle_child')",
            "execute_oracle = module._load_exact_module(module._source_files()['execute_oracle'], module._EXPECTED_SOURCE_SHA256['execute_oracle'], 'execute_oracle_child')",
            "_primals, _reference, report = module._construct_host_reference(np, execute, prepare_oracle.gdn_prepare_s512_numpy, execute_oracle.gdn_execute_s512_numpy, module._zero_counters())",
            "print(json.dumps(report['checks'], sort_keys=True))",
        ]
    )
    environment = dict(os.environ)
    environment.update(
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    checks = json.loads(result.stdout)
    assert checks["input_hashes_exact"] is True
    assert checks["prepared_hashes_exact"] is True
    assert checks["reference_hashes_exact"] is True
    assert checks["reference_tuple_hash_exact"] is True


def test_output_file_is_private_and_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "result.jsonl"
    with _PROBE._open_output(path) as output:
        output.write("{}\n")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        _PROBE._open_output(path)
