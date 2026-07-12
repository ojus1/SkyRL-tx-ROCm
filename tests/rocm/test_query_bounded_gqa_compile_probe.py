from __future__ import annotations

import ast
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_REPO = Path(__file__).parents[2]
_PROBE_PATH = _REPO / "rocm" / "probe_query_bounded_gqa_compile.py"
_SPEC = importlib.util.spec_from_file_location(
    "probe_query_bounded_gqa_compile_test", _PROBE_PATH
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
    "JAX_ROCM_VISIBLE_DEVICES",
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


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in _ACCELERATOR_ENVIRONMENT:
        environment.pop(name, None)
    return environment


def _run(*arguments: str):
    return subprocess.run(
        [sys.executable, str(_PROBE_PATH), *arguments],
        cwd=_REPO,
        env=_clean_environment(),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def test_default_is_no_gpu_abstract_refusal_without_jax_import():
    result = _run()

    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert [record["record_type"] for record in records] == ["manifest", "refused"]
    manifest, refused = records
    assert manifest["platform_requested"] == "abstract"
    assert manifest["allow_gpu"] is False
    assert manifest["scope"] == "abstract_refusal"
    assert manifest["compile_may_dispatch_gpu_work"] is False
    assert "autotuning/profiling kernels" in manifest["compile_dispatch_caveat"]
    assert manifest["compiled_executable_invocations"] == 0
    assert refused == {
        "compiled_executable_invocations": 0,
        "jax_imported": False,
        "reason": "pass --platform rocm --allow-gpu explicitly to lower and compile",
        "record_type": "refused",
        "status": "no_gpu_abstract_manifest_only",
        "timestamp": refused["timestamp"],
    }


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--platform", "rocm"), "requires the explicit --allow-gpu"),
        (
            ("--platform", "rocm", "--allow-gpu"),
            "requires --output for a private JSONL artifact",
        ),
        (("--allow-gpu",), "only valid with --platform rocm"),
        (("--sequence-length", "1024"), "unrecognized arguments"),
        (("--query-heads", "32"), "unrecognized arguments"),
    ],
)
def test_unsafe_or_non_exact_options_are_rejected_before_any_record(arguments, message):
    result = _run(*arguments)

    assert result.returncode == 2
    assert result.stdout == ""
    assert message in result.stderr


def test_output_is_private_exclusive_jsonl(tmp_path):
    output = tmp_path / "gqa-compile.jsonl"

    result = _run("--output", str(output))

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert [
        json.loads(line)["record_type"] for line in output.read_text().splitlines()
    ] == ["manifest", "refused"]

    repeated = _run("--output", str(output))
    assert repeated.returncode == 2
    assert "refusing to overwrite existing output" in repeated.stderr


def test_manifest_fixes_exact_qwen_forward_and_vjp_contract():
    contract = _PROBE._exact_contract()

    assert contract["operation"] == "query_bounded_gqa_forward_and_vjp"
    assert contract["inputs"] == [
        {"name": "q", "shape": [1, 512, 16, 256], "dtype": "bfloat16"},
        {"name": "k", "shape": [1, 512, 4, 256], "dtype": "bfloat16"},
        {"name": "v", "shape": [1, 512, 4, 256], "dtype": "bfloat16"},
        {"name": "key_mask", "shape": [1, 512], "dtype": "int32"},
        {"name": "dout", "shape": [1, 512, 16, 256], "dtype": "bfloat16"},
    ]
    assert contract["outputs"] == [
        {"name": "output", "shape": [1, 512, 16, 256], "dtype": "bfloat16"},
        {"name": "dq", "shape": [1, 512, 16, 256], "dtype": "bfloat16"},
        {"name": "dk", "shape": [1, 512, 4, 256], "dtype": "bfloat16"},
        {"name": "dv", "shape": [1, 512, 4, 256], "dtype": "bfloat16"},
    ]
    assert contract["tiles"] == {
        "query_chunk_size": 512,
        "block_q": 64,
        "block_k": 64,
        "backward_block_q": 32,
        "backward_block_k": 32,
    }
    assert contract["expected_pallas_calls"] == {
        "forward": 1,
        "dq": 1,
        "dkdv": 1,
        "total": 3,
    }


def test_help_identifies_rocm_compilation_as_gpu_work():
    result = _run("--help")

    assert result.returncode == 0
    assert "GPU work" in result.stdout
    assert "autotuning/profiling kernels" in result.stdout


def test_rocm_environment_reuses_hardened_pallas_configurator(monkeypatch):
    import rocm.probe_pallas_attention as hardened_probe

    expected = {"bounded": "shared"}
    monkeypatch.setattr(hardened_probe, "_configure_environment", lambda: expected)

    assert _PROBE._configure_rocm_environment() is expected


def test_shared_guard_wraps_compile_and_postflight(monkeypatch):
    state = {
        "guard_active": False,
        "compile_calls": 0,
        "postflight_calls": 0,
    }

    @contextmanager
    def guarded_process():
        state["guard_active"] = True
        try:
            yield {"amdgpu_boot_clean": True, "kfd_unowned": True}
        finally:
            state["guard_active"] = False

    def postflight():
        assert state["guard_active"] is True
        state["postflight_calls"] += 1
        return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}

    def compile_rocm(output):
        assert state["guard_active"] is True
        state["compile_calls"] += 1
        return {"pallas_count_gate": {"passed": True}}

    monkeypatch.setattr(
        _PROBE, "_configure_rocm_environment", lambda: {"bounded": "yes"}
    )
    monkeypatch.setattr(
        _PROBE, "_load_safety_helpers", lambda: (guarded_process, postflight)
    )
    monkeypatch.setattr(_PROBE, "_compile_rocm", compile_rocm)
    output = io.StringIO()

    result = _PROBE._execute(SimpleNamespace(platform="rocm", allow_gpu=True), output)

    assert result == 0
    assert state == {
        "guard_active": False,
        "compile_calls": 1,
        "postflight_calls": 1,
    }
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert records[0]["scope"] == "rocm_compile_only_gpu_work"
    assert records[0]["compile_may_dispatch_gpu_work"] is True
    assert "autotuning/profiling kernels" in records[0]["compile_dispatch_caveat"]
    assert [record["record_type"] for record in records] == [
        "manifest",
        "environment",
        "safety_preflight",
        "safety_postflight",
        "completed",
    ]


def test_postflight_is_mandatory_when_compilation_raises(monkeypatch):
    state = {"postflight_calls": 0}

    @contextmanager
    def guarded_process():
        yield {"amdgpu_boot_clean": True, "kfd_unowned": True}

    def postflight():
        state["postflight_calls"] += 1
        return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}

    monkeypatch.setattr(_PROBE, "_configure_rocm_environment", lambda: {})
    monkeypatch.setattr(
        _PROBE, "_load_safety_helpers", lambda: (guarded_process, postflight)
    )
    monkeypatch.setattr(
        _PROBE,
        "_compile_rocm",
        lambda _output: (_ for _ in ()).throw(
            RuntimeError("synthetic compile failure")
        ),
    )
    output = io.StringIO()

    result = _PROBE._execute(SimpleNamespace(platform="rocm", allow_gpu=True), output)

    assert result == 1
    assert state["postflight_calls"] == 1
    final = json.loads(output.getvalue().splitlines()[-1])
    assert final["record_type"] == "error"
    assert final["stage"] == "lower_and_compile"
    assert final["message"] == "synthetic compile failure"


def test_postflight_failure_is_attributed_to_postflight(monkeypatch):
    @contextmanager
    def guarded_process():
        yield {"amdgpu_boot_clean": True, "kfd_unowned": True}

    monkeypatch.setattr(_PROBE, "_configure_rocm_environment", lambda: {})
    monkeypatch.setattr(
        _PROBE,
        "_load_safety_helpers",
        lambda: (
            guarded_process,
            lambda: (_ for _ in ()).throw(RuntimeError("synthetic journal failure")),
        ),
    )
    monkeypatch.setattr(
        _PROBE,
        "_compile_rocm",
        lambda _output: {"pallas_count_gate": {"passed": True}},
    )
    output = io.StringIO()

    result = _PROBE._execute(SimpleNamespace(platform="rocm", allow_gpu=True), output)

    assert result == 1
    final = json.loads(output.getvalue().splitlines()[-1])
    assert final["record_type"] == "error"
    assert final["stage"] == "safety_postflight"
    assert final["message"] == "synthetic journal failure"


class _FakeShapeDtypeStruct:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _FakeCompiled:
    def __init__(self, state: dict[str, Any]):
        self._state = state

    def __call__(self, *_args, **_kwargs):
        self._state["compiled_callable_invocations"] += 1
        raise AssertionError("the compile-only probe invoked the executable")

    def as_text(self):
        return "\n".join(
            f'%{index} = custom-call(), custom_call_target="triton", op_name="{name}"'
            for index, name in enumerate(_PROBE._EXPECTED_PALLAS_CALLS.values())
        )

    def memory_analysis(self):
        return SimpleNamespace(
            argument_size_in_bytes=10,
            output_size_in_bytes=20,
            alias_size_in_bytes=5,
            temp_size_in_bytes=30,
        )

    def cost_analysis(self):
        return {"flops": 123.0}


class _FakeLowered:
    def __init__(self, state: dict[str, Any]):
        self._state = state

    def compiler_ir(self, *, dialect):
        assert dialect == "stablehlo"
        return "\n".join(
            f'%{index} = stablehlo.custom_call @triton {{op_name = "{name}"}}'
            for index, name in enumerate(_PROBE._EXPECTED_PALLAS_CALLS.values())
        )

    def compile(self):
        self._state["compile_calls"] += 1
        return _FakeCompiled(self._state)


class _FakeJitted:
    def __init__(self, state: dict[str, Any]):
        self._state = state

    def lower(self, *signature):
        self._state["signature"] = signature
        return _FakeLowered(self._state)


class _FakeJax:
    ShapeDtypeStruct = _FakeShapeDtypeStruct

    def __init__(self, state: dict[str, Any]):
        self._state = state

    def jit(self, function):
        self._state["jitted_function"] = function
        return _FakeJitted(self._state)


def test_compiled_callable_cannot_be_invoked_or_escape_metadata_extraction():
    state: dict[str, Any] = {
        "compile_calls": 0,
        "compiled_callable_invocations": 0,
    }
    fake_jnp = SimpleNamespace(bfloat16="bf16", int32="i32")
    output = io.StringIO()

    report = _PROBE._lower_and_compile_exact(
        _FakeJax(state), fake_jnp, object(), output
    )

    assert state["compile_calls"] == 1
    assert state["compiled_callable_invocations"] == 0
    assert [item.shape for item in state["signature"]] == [
        (1, 512, 16, 256),
        (1, 512, 4, 256),
        (1, 512, 4, 256),
        (1, 512),
        (1, 512, 16, 256),
    ]
    assert [item.dtype for item in state["signature"]] == [
        "bf16",
        "bf16",
        "bf16",
        "i32",
        "bf16",
    ]
    assert report["pallas_count_gate"]["passed"] is True
    assert report["compiled_memory"] == {
        "available": True,
        "argument_size_in_bytes": 10,
        "output_size_in_bytes": 20,
        "alias_size_in_bytes": 5,
        "temp_size_in_bytes": 30,
    }
    assert "compiled" not in report
    assert [
        json.loads(line)["record_type"] for line in output.getvalue().splitlines()
    ] == [
        "stage",
        "lowered",
        "stage",
        "compiled",
    ]


def _stablehlo_calls(
    names: list[str], *, multiline_first: bool = False, target: str = "triton"
) -> str:
    definitions = [f'#loc{index} = loc("{name}")' for index, name in enumerate(names)]
    calls = []
    for index, _name in enumerate(names):
        if index == 0 and multiline_first:
            calls.extend(
                [
                    f"  %{index} = stablehlo.custom_call @{target}(",
                    "    %arg0",
                    f") : (tensor<1xf32>) -> tensor<1xf32> loc(#loc{index})",
                ]
            )
        else:
            calls.append(
                f"  %{index} = stablehlo.custom_call @{target}() "
                f": () -> tensor<1xf32> loc(#loc{index})"
            )
    return "\n".join([*definitions, "module {", *calls, "}"])


def _optimized_hlo_calls(names: list[str], *, target: str = "triton") -> str:
    calls = [
        f'  %{index} = f32[1] custom-call(), custom_call_target="{target}", '
        f'op_name="{name}"'
        for index, name in enumerate(names)
    ]
    return "\n".join(["ENTRY main {", *calls, "}"])


def _expected_names() -> list[str]:
    return list(_PROBE._EXPECTED_PALLAS_CALLS.values())


def test_multiline_stablehlo_calls_resolve_referenced_location_metadata():
    stablehlo = _PROBE._ir_summary(
        _stablehlo_calls(_expected_names(), multiline_first=True), "stablehlo"
    )
    optimized = _PROBE._ir_summary(
        _optimized_hlo_calls(_expected_names()), "optimized_hlo"
    )

    assert stablehlo["custom_call_count"] == 3
    assert stablehlo["pallas_custom_call_count"] == 3
    assert stablehlo["pallas_name_call_counts"] == {
        "forward": 1,
        "dq": 1,
        "dkdv": 1,
    }
    assert all(
        call["has_exactly_one_expected_marker"] for call in stablehlo["pallas_calls"]
    )
    assert _PROBE._pallas_count_gate(stablehlo, optimized)["passed"] is True


@pytest.mark.parametrize("dialect", ["stablehlo", "optimized_hlo"])
@pytest.mark.parametrize(
    ("names", "failed_check_suffix"),
    [
        (
            lambda expected: [expected[0], expected[0], expected[2]],
            "each_expected_marker_once",
        ),
        (
            lambda expected: [expected[0], expected[1], "unrelated_kernel_name"],
            "one_expected_marker_per_pallas_call",
        ),
    ],
)
def test_duplicate_or_missing_call_names_fail_closed(
    dialect, names, failed_check_suffix
):
    expected = _expected_names()
    stable_names = names(expected) if dialect == "stablehlo" else expected
    optimized_names = names(expected) if dialect == "optimized_hlo" else expected
    stablehlo = _PROBE._ir_summary(_stablehlo_calls(stable_names), "stablehlo")
    optimized = _PROBE._ir_summary(
        _optimized_hlo_calls(optimized_names), "optimized_hlo"
    )

    gate = _PROBE._pallas_count_gate(stablehlo, optimized)

    assert gate["passed"] is False
    assert gate["checks"][f"{dialect}_{failed_check_suffix}"] is False


@pytest.mark.parametrize("dialect", ["stablehlo", "optimized_hlo"])
def test_unreferenced_marker_outside_custom_call_does_not_satisfy_gate(dialect):
    expected = _expected_names()
    incomplete_names = [expected[0], expected[1], "unrelated_kernel_name"]
    stable_text = _stablehlo_calls(
        incomplete_names if dialect == "stablehlo" else expected
    )
    optimized_text = _optimized_hlo_calls(
        incomplete_names if dialect == "optimized_hlo" else expected
    )
    if dialect == "stablehlo":
        stable_text += f'\n#loc999 = loc("{expected[2]}")'
    else:
        optimized_text += f'\n%outside = f32[1] add(%x, %y), op_name="{expected[2]}"'
    stablehlo = _PROBE._ir_summary(stable_text, "stablehlo")
    optimized = _PROBE._ir_summary(optimized_text, "optimized_hlo")

    gate = _PROBE._pallas_count_gate(stablehlo, optimized)

    affected = stablehlo if dialect == "stablehlo" else optimized
    assert affected["pallas_name_call_counts"]["dkdv"] == 0
    assert gate["passed"] is False


@pytest.mark.parametrize("dialect", ["stablehlo", "optimized_hlo"])
def test_suffixed_expected_names_are_not_exact_markers(dialect):
    expected = _expected_names()
    suffixed = [f"{name}_not_exact" for name in expected]
    stablehlo = _PROBE._ir_summary(
        _stablehlo_calls(suffixed if dialect == "stablehlo" else expected),
        "stablehlo",
    )
    optimized = _PROBE._ir_summary(
        _optimized_hlo_calls(suffixed if dialect == "optimized_hlo" else expected),
        "optimized_hlo",
    )

    gate = _PROBE._pallas_count_gate(stablehlo, optimized)

    affected = stablehlo if dialect == "stablehlo" else optimized
    assert affected["pallas_name_call_counts"] == {
        "forward": 0,
        "dq": 0,
        "dkdv": 0,
    }
    assert gate["passed"] is False
    assert gate["checks"][f"{dialect}_each_expected_marker_once"] is False


@pytest.mark.parametrize("dialect", ["stablehlo", "optimized_hlo"])
@pytest.mark.parametrize("lookalike_target", ["nottriton", "pallasian_kernel"])
def test_words_containing_pallas_or_triton_are_not_custom_call_targets(
    dialect, lookalike_target
):
    expected = _expected_names()
    stablehlo = _PROBE._ir_summary(
        _stablehlo_calls(
            expected,
            target=lookalike_target if dialect == "stablehlo" else "triton",
        ),
        "stablehlo",
    )
    optimized = _PROBE._ir_summary(
        _optimized_hlo_calls(
            expected,
            target=lookalike_target if dialect == "optimized_hlo" else "triton",
        ),
        "optimized_hlo",
    )

    gate = _PROBE._pallas_count_gate(stablehlo, optimized)

    affected = stablehlo if dialect == "stablehlo" else optimized
    assert affected["custom_call_count"] == 3
    assert affected["pallas_custom_call_count"] == 0
    assert gate["passed"] is False
    assert gate["checks"][f"{dialect}_three_pallas_custom_calls"] is False


def test_gate_requires_both_ir_dialects_exactly_once():
    stablehlo = _PROBE._ir_summary(_stablehlo_calls(_expected_names()), "stablehlo")

    gate = _PROBE._pallas_count_gate(stablehlo)

    assert gate["passed"] is False
    assert gate["checks"]["exactly_stablehlo_and_optimized_hlo"] is False


@pytest.mark.parametrize("dialect", ["stablehlo", "optimized_hlo"])
def test_outer_while_fails_closed(dialect):
    expected = _expected_names()
    stable_text = _stablehlo_calls(expected)
    optimized_text = _optimized_hlo_calls(expected)
    if dialect == "stablehlo":
        stable_text += "\n%loop = stablehlo.while(%arg0) : tensor<1xf32>"
    else:
        optimized_text += "\n%loop = f32[1] while(%arg0)"
    stablehlo = _PROBE._ir_summary(stable_text, "stablehlo")
    optimized = _PROBE._ir_summary(optimized_text, "optimized_hlo")

    gate = _PROBE._pallas_count_gate(stablehlo, optimized)

    assert gate["passed"] is False
    assert gate["checks"][f"{dialect}_no_while"] is False


def test_probe_imports_only_standard_library_before_explicit_rocm_path():
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
