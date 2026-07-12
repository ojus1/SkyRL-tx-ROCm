from __future__ import annotations

import ast
import importlib.util
import inspect
import io
import json
import os
import stat
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO = Path(__file__).parents[2]
_PROBE_PATH = _REPO / "rocm" / "probe_query_bounded_gqa_runtime.py"
_DOC_PATH = _REPO / "rocm" / "QUERY_BOUNDED_GQA.md"
_SPEC = importlib.util.spec_from_file_location(
    "probe_query_bounded_gqa_runtime_test", _PROBE_PATH
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


def test_default_is_abstract_refusal_before_jax_import():
    result = _run()

    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert [record["record_type"] for record in records] == ["manifest", "refused"]
    manifest, refused = records
    assert manifest["platform_requested"] == "abstract"
    assert manifest["compile_may_dispatch_gpu_work"] is False
    assert manifest["prior_compile_artifact_used"] is False
    assert manifest["model_dispatcher_connected"] is False
    assert manifest["counters"] == _PROBE._zero_counters()
    assert refused["jax_imported"] is False


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
        (("--repeats", "2"), "unrecognized arguments"),
        (("--backward",), "unrecognized arguments"),
    ],
)
def test_unsafe_or_scope_broadening_options_are_rejected(arguments, message):
    result = _run(*arguments)

    assert result.returncode == 2
    assert result.stdout == ""
    assert message in result.stderr


def test_output_is_private_and_exclusive(tmp_path):
    output = tmp_path / "runtime.jsonl"

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


def test_runtime_reuses_hardened_bounded_environment(monkeypatch):
    import rocm.probe_query_bounded_gqa_compile as compile_probe

    expected = {"bounded": "shared"}
    monkeypatch.setattr(compile_probe, "_configure_rocm_environment", lambda: expected)

    assert _PROBE._configure_rocm_environment() is expected


def test_contract_is_exact_single_forward_host_oracle():
    contract = _PROBE._exact_contract()

    assert contract["operation"] == "query_bounded_gqa_forward_only"
    assert contract["inputs"] == [
        {"name": "q", "shape": [1, 512, 16, 256], "dtype": "bfloat16"},
        {"name": "k", "shape": [1, 512, 4, 256], "dtype": "bfloat16"},
        {"name": "v", "shape": [1, 512, 4, 256], "dtype": "bfloat16"},
        {
            "name": "key_mask",
            "shape": [1, 512],
            "dtype": "int32",
            "value": "all_ones",
        },
    ]
    assert contract["randomness_used"] is False
    assert contract["seed"] == 0
    assert contract["dispatch_plan"] == {
        "checked_forward_invocations": 1,
        "gpu_reference_invocations": 0,
        "device_error_reduction_invocations": 0,
        "backward_invocations": 0,
        "replay_invocations": 0,
    }


def test_host_analytic_inputs_are_deterministic_and_check_gqa_mapping():
    first_inputs, first_manifests, first_expected, first_expected_manifest = (
        _PROBE._construct_host_inputs()
    )
    second_inputs, second_manifests, second_expected, second_expected_manifest = (
        _PROBE._construct_host_inputs()
    )
    q, k, v, mask = first_inputs

    assert np.count_nonzero(q) == 0
    assert np.count_nonzero(k) == 0
    assert np.all(mask == 1)
    assert float(np.min(v.astype(np.float32))) >= -1.0
    assert float(np.max(v.astype(np.float32))) <= 1.0
    np.testing.assert_array_equal(first_expected, second_expected)
    assert first_manifests == second_manifests
    assert first_expected_manifest == second_expected_manifest
    # Query heads in each group of four map to the same KV head, while the
    # adjacent group receives a distinct analytic value.
    np.testing.assert_array_equal(first_expected[0, :, 0], first_expected[0, :, 3])
    assert not np.array_equal(first_expected[0, :, 3], first_expected[0, :, 4])
    # With zero Q/K, the final causal row is the cumulative mean of all V rows.
    np.testing.assert_allclose(
        first_expected[0, -1, 0],
        np.mean(v.astype(np.float32)[0, :, 0], axis=0),
        rtol=1e-6,
        atol=1e-6,
    )
    for first, second in zip(first_inputs, second_inputs, strict=True):
        np.testing.assert_array_equal(first, second)


def _stablehlo(
    name: str,
    *,
    target: str = "__gpu$xla.gpu.triton",
    while_: bool = False,
    extra_custom_call: bool = False,
) -> str:
    lines = [
        f'#loc0 = loc("{name}")',
        "module {",
        f"  %0 = stablehlo.custom_call @{target}() : () -> tensor<1xf32> loc(#loc0)",
    ]
    if extra_custom_call:
        lines.append(
            "  %1 = stablehlo.custom_call @unrelated_runtime_call() "
            ": () -> tensor<1xf32>"
        )
    lines.append("}")
    text = "\n".join(lines)
    if while_:
        text += "\n%loop = stablehlo.while(%arg0) : tensor<1xf32>"
    return text


def _optimized_hlo(
    name: str,
    *,
    target: str = "__gpu$xla.gpu.triton",
    while_: bool = False,
    extra_custom_call: bool = False,
) -> str:
    lines = [
        "ENTRY main {",
        f'  %0 = f32[1] custom-call(), custom_call_target="{target}", op_name="{name}"',
    ]
    if extra_custom_call:
        lines.append(
            '  %1 = f32[1] custom-call(), custom_call_target="unrelated_runtime_call"'
        )
    lines.append("}")
    text = "\n".join(lines)
    if while_:
        text += "\n%loop = f32[1] while(%arg0)"
    return text


def _summaries(
    stable_name: str = "query_bounded_gqa_forward_q0",
    optimized_name: str = "query_bounded_gqa_forward_q0",
    *,
    stable_target: str = "__gpu$xla.gpu.triton",
    optimized_target: str = "__gpu$xla.gpu.triton",
    stable_while: bool = False,
    optimized_while: bool = False,
    stable_extra_custom_call: bool = False,
    optimized_extra_custom_call: bool = False,
):
    return (
        _PROBE._summarize_ir(
            _stablehlo(
                stable_name,
                target=stable_target,
                while_=stable_while,
                extra_custom_call=stable_extra_custom_call,
            ),
            "stablehlo",
        ),
        _PROBE._summarize_ir(
            _optimized_hlo(
                optimized_name,
                target=optimized_target,
                while_=optimized_while,
                extra_custom_call=optimized_extra_custom_call,
            ),
            "optimized_hlo",
        ),
    )


def test_forward_ir_gate_accepts_exact_one_forward_only():
    stablehlo, optimized = _summaries()

    gate = _PROBE._forward_ir_gate(stablehlo, optimized)

    assert gate["passed"] is True
    for summary in (stablehlo, optimized):
        assert summary["custom_call_count"] == 1
        assert summary["pallas_calls"][0]["custom_call_targets"] == [
            "__gpu$xla.gpu.triton"
        ]


@pytest.mark.parametrize("dialect", ["stablehlo", "optimized_hlo"])
@pytest.mark.parametrize(
    "bad_name",
    [
        "query_bounded_gqa_forward_q0_not_exact",
        "query_bounded_gqa_dq_q0",
        "query_bounded_gqa_dkdv_q0",
    ],
)
def test_forward_ir_gate_rejects_wrong_or_lookalike_exact_names(dialect, bad_name):
    kwargs = {f"{dialect.removesuffix('_hlo')}_name": bad_name}
    if dialect == "stablehlo":
        stablehlo, optimized = _summaries(stable_name=bad_name)
    else:
        stablehlo, optimized = _summaries(optimized_name=bad_name)

    gate = _PROBE._forward_ir_gate(stablehlo, optimized)

    assert kwargs  # keep the parametrized dialect visible in failure output
    assert gate["passed"] is False


@pytest.mark.parametrize("dialect", ["stablehlo", "optimized_hlo"])
@pytest.mark.parametrize("lookalike", ["nottriton", "pallasian_kernel"])
def test_forward_ir_gate_rejects_pallas_triton_target_substrings(dialect, lookalike):
    if dialect == "stablehlo":
        stablehlo, optimized = _summaries(stable_target=lookalike)
    else:
        stablehlo, optimized = _summaries(optimized_target=lookalike)

    gate = _PROBE._forward_ir_gate(stablehlo, optimized)

    assert gate["passed"] is False


@pytest.mark.parametrize("dialect", ["stablehlo", "optimized_hlo"])
@pytest.mark.parametrize(
    "known_but_nonexact_target",
    ["triton", "triton_kernel_call", "xla.gpu.triton"],
)
def test_forward_ir_gate_rejects_known_but_nonexact_targets(
    dialect, known_but_nonexact_target
):
    if dialect == "stablehlo":
        stablehlo, optimized = _summaries(stable_target=known_but_nonexact_target)
    else:
        stablehlo, optimized = _summaries(optimized_target=known_but_nonexact_target)

    gate = _PROBE._forward_ir_gate(stablehlo, optimized)

    assert gate["passed"] is False
    assert gate["checks"][f"{dialect}_sole_exact_rocm_triton_target"] is False


@pytest.mark.parametrize("dialect", ["stablehlo", "optimized_hlo"])
def test_forward_ir_gate_rejects_an_extra_unrelated_custom_call(dialect):
    stablehlo, optimized = _summaries(
        stable_extra_custom_call=dialect == "stablehlo",
        optimized_extra_custom_call=dialect == "optimized_hlo",
    )

    gate = _PROBE._forward_ir_gate(stablehlo, optimized)

    affected = stablehlo if dialect == "stablehlo" else optimized
    assert affected["custom_call_count"] == 2
    assert affected["pallas_custom_call_count"] == 1
    assert gate["passed"] is False
    assert gate["checks"][f"{dialect}_exactly_one_custom_call_total"] is False


@pytest.mark.parametrize("dialect", ["stablehlo", "optimized_hlo"])
def test_forward_ir_gate_rejects_outer_while(dialect):
    stablehlo, optimized = _summaries(
        stable_while=dialect == "stablehlo",
        optimized_while=dialect == "optimized_hlo",
    )

    assert _PROBE._forward_ir_gate(stablehlo, optimized)["passed"] is False


@pytest.mark.parametrize(
    ("memory", "passed"),
    [
        (
            {
                "available": True,
                "argument_size_in_bytes": 10 * 1024**2,
                "output_size_in_bytes": 4 * 1024**2,
                "temp_size_in_bytes": 63 * 1024**2,
            },
            True,
        ),
        ({"available": False}, False),
        (
            {
                "available": True,
                "argument_size_in_bytes": 1,
                "output_size_in_bytes": 1,
                "temp_size_in_bytes": 64 * 1024**2 + 1,
            },
            False,
        ),
        (
            {
                "available": True,
                "argument_size_in_bytes": 64 * 1024**2,
                "output_size_in_bytes": 64 * 1024**2,
                "temp_size_in_bytes": 1,
            },
            False,
        ),
    ],
)
def test_compiled_memory_gate_is_required_and_bounded(memory, passed):
    assert _PROBE._compiled_memory_gate(memory)["passed"] is passed


class _FakeShape:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _FakeCompiled:
    def __init__(self, state, optimized_name="query_bounded_gqa_forward_q0"):
        self.state = state
        self.optimized_name = optimized_name

    def __call__(self, *_arguments):
        self.state["compiled_invocations"] += 1
        return "result"

    def as_text(self):
        return _optimized_hlo(self.optimized_name)

    def memory_analysis(self):
        return SimpleNamespace(
            argument_size_in_bytes=10 * 1024**2,
            output_size_in_bytes=4 * 1024**2,
            alias_size_in_bytes=0,
            temp_size_in_bytes=2 * 1024**2,
        )


class _FakeLowered:
    def __init__(self, state, stable_name, optimized_name):
        self.state = state
        self.stable_name = stable_name
        self.optimized_name = optimized_name

    def compiler_ir(self, *, dialect):
        assert dialect == "stablehlo"
        return _stablehlo(self.stable_name)

    def compile(self):
        self.state["compile_calls"] += 1
        return _FakeCompiled(self.state, self.optimized_name)


class _FakeJitted:
    def __init__(self, state, stable_name, optimized_name):
        self.state = state
        self.stable_name = stable_name
        self.optimized_name = optimized_name

    def lower(self, *signature):
        self.state["signature"] = signature
        return _FakeLowered(self.state, self.stable_name, self.optimized_name)


class _FakeCompileJax:
    ShapeDtypeStruct = _FakeShape

    def __init__(
        self,
        state,
        stable_name="query_bounded_gqa_forward_q0",
        optimized_name="query_bounded_gqa_forward_q0",
    ):
        self.state = state
        self.stable_name = stable_name
        self.optimized_name = optimized_name

    def jit(self, function):
        self.state["jitted_function"] = function
        return _FakeJitted(self.state, self.stable_name, self.optimized_name)


def test_compile_gate_precedes_and_controls_first_executable_invocation():
    state = {"compile_calls": 0, "compiled_invocations": 0}
    counters = _PROBE._zero_counters()
    jnp = SimpleNamespace(bfloat16="bf16", int32="i32")
    output = io.StringIO()

    checked, report = _PROBE._compile_checked_forward(
        _FakeCompileJax(state), jnp, object(), counters, output
    )

    assert report["structural_gate"]["passed"] is True
    assert report["compiled_memory_gate"]["passed"] is True
    assert state["compiled_invocations"] == 0
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert records[-1]["stage"] == "forward_executable_released_after_gate"
    blocking_jax = SimpleNamespace(block_until_ready=lambda value: value)
    dispatch_output = io.StringIO()
    result, _seconds = _PROBE._dispatch_checked(
        blocking_jax,
        checked,
        (None, None, None, None),
        label="candidate",
        require_clean_boot=lambda: {"amdgpu_boot_clean": True},
        counters=counters,
        output=dispatch_output,
    )
    assert result == "result"
    assert state["compiled_invocations"] == 1
    assert counters["forward_attempts"] == 1
    assert counters["forward_completions"] == 1
    dispatch_records = [
        json.loads(line) for line in dispatch_output.getvalue().splitlines()
    ]
    assert [record["record_type"] for record in dispatch_records] == [
        "dispatch_started",
        "journal_checkpoint",
        "dispatch",
    ]
    assert dispatch_records[0]["counters"]["forward_attempts"] == 1
    assert dispatch_records[0]["counters"]["forward_completions"] == 0
    assert dispatch_records[-1]["counters"]["forward_completions"] == 1


def test_failed_structure_never_exposes_or_invokes_compiled_callable():
    state = {"compile_calls": 0, "compiled_invocations": 0}
    counters = _PROBE._zero_counters()
    jnp = SimpleNamespace(bfloat16="bf16", int32="i32")

    with pytest.raises(RuntimeError, match="structural or compiled-memory"):
        _PROBE._compile_checked_forward(
            _FakeCompileJax(
                state,
                stable_name="query_bounded_gqa_forward_q0_not_exact",
            ),
            jnp,
            object(),
            counters,
            io.StringIO(),
        )

    assert state["compiled_invocations"] == 0
    assert counters == _PROBE._zero_counters()


def test_failed_compiled_memory_gate_never_exposes_callable(monkeypatch):
    state = {"compile_calls": 0, "compiled_invocations": 0}
    counters = _PROBE._zero_counters()
    jnp = SimpleNamespace(bfloat16="bf16", int32="i32")
    monkeypatch.setattr(
        _PROBE,
        "_compiled_memory",
        lambda _compiled: {
            "available": True,
            "argument_size_in_bytes": 1,
            "output_size_in_bytes": 1,
            "temp_size_in_bytes": 64 * 1024**2 + 1,
        },
    )

    with pytest.raises(RuntimeError, match="structural or compiled-memory"):
        _PROBE._compile_checked_forward(
            _FakeCompileJax(state),
            jnp,
            object(),
            counters,
            io.StringIO(),
        )

    assert state["compiled_invocations"] == 0
    assert counters == _PROBE._zero_counters()


def test_checked_executable_rejects_missing_gate_or_private_token():
    with pytest.raises(RuntimeError, match="without a passed gate"):
        _PROBE._CheckedExecutable(
            object(),
            proof={"passed": False},
            counter_prefix="forward",
            counters=_PROBE._zero_counters(),
            token=object(),
        )


def test_dispatch_checks_journal_even_when_candidate_raises():
    counters = _PROBE._zero_counters()
    state = {"journal": 0}

    class Failing:
        def __call__(self, *_arguments):
            raise RuntimeError("synthetic dispatch failure")

    checked = _PROBE._wrap_checked(
        Failing(),
        proof={"passed": True},
        counter_prefix="forward",
        counters=counters,
    )
    jax = SimpleNamespace(block_until_ready=lambda value: value)

    output = io.StringIO()
    with pytest.raises(RuntimeError, match="synthetic dispatch failure"):
        _PROBE._dispatch_checked(
            jax,
            checked,
            (),
            label="candidate",
            require_clean_boot=lambda: (
                state.__setitem__("journal", state["journal"] + 1)
                or {"amdgpu_boot_clean": True}
            ),
            counters=counters,
            output=output,
        )

    assert counters["forward_attempts"] == 1
    assert counters["forward_completions"] == 0
    assert state["journal"] == 1
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [record["record_type"] for record in records] == [
        "dispatch_started",
        "journal_checkpoint",
    ]
    assert records[0]["counters"]["forward_attempts"] == 1
    assert records[0]["counters"]["forward_completions"] == 0
    assert records[1]["counters"]["forward_completions"] == 0


class _RuntimeJax:
    __version__ = "test"

    @staticmethod
    def default_backend():
        return "gpu"

    @staticmethod
    def devices():
        return []

    @staticmethod
    def device_get(value):
        return value

    @staticmethod
    def block_until_ready(value):
        return value


def _run_dependencies():
    jax = _RuntimeJax()
    jnp = SimpleNamespace()
    jaxlib = SimpleNamespace(__version__="test")
    backend = SimpleNamespace(
        get_backend=lambda: SimpleNamespace(platform_version="ROCm test")
    )
    return jax, jnp, jaxlib, backend, object()


def _patch_runtime_happy_path(monkeypatch, metrics, *, dispatch_seconds=0.01):
    events = []

    class Candidate:
        def __call__(self, *_arguments):
            events.append("candidate_invoked")
            return np.zeros((1,), dtype=np.float32)

    def compile_forward(_jax, _jnp, _operation, counters, _output):
        events.append("structural_and_memory_gate_passed")
        checked = _PROBE._wrap_checked(
            Candidate(),
            proof={"passed": True},
            counter_prefix="forward",
            counters=counters,
        )
        return checked, {
            "structural_gate": {"passed": True},
            "compiled_memory_gate": {"passed": True},
        }

    monkeypatch.setattr(_PROBE, "_compile_checked_forward", compile_forward)
    monkeypatch.setattr(
        _PROBE,
        "_construct_host_inputs",
        lambda: ((None, None, None, None), [], np.zeros((1,), dtype=np.float32), {}),
    )
    monkeypatch.setattr(
        _PROBE,
        "_device_put_inputs",
        lambda _jax, inputs: events.append("device_put") or inputs,
    )
    monkeypatch.setattr(_PROBE, "_allocator_snapshot", lambda _jax: [])
    monkeypatch.setattr(_PROBE, "_host_metrics", lambda *_arguments: metrics)
    monkeypatch.setattr(
        _PROBE,
        "_journal_checkpoint",
        lambda _require, _output, stage, _counters: (
            events.append(stage) or {"amdgpu_boot_clean": True}
        ),
    )
    monkeypatch.setattr(
        _PROBE,
        "_dispatch_checked",
        lambda _jax, executable, arguments, **_kwargs: (
            executable.invoke(_jax, *arguments, on_started=lambda: None),
            dispatch_seconds,
        ),
    )
    return events


def test_runtime_gate_invokes_candidate_exactly_once_after_gate(monkeypatch):
    metrics = {
        "finite": True,
        "max_abs": 0.001,
        "mean_abs": 0.0001,
        "relative_l2": 0.001,
        "cosine": 0.99999,
    }
    events = _patch_runtime_happy_path(monkeypatch, metrics)
    counters = _PROBE._zero_counters()

    result = _PROBE._run_rocm(
        io.StringIO(),
        lambda: {"amdgpu_boot_clean": True},
        counters,
        _dependencies=_run_dependencies(),
    )

    assert result == 0
    assert events.index("structural_and_memory_gate_passed") < events.index(
        "candidate_invoked"
    )
    assert events.index("after_backend_initialization") < events.index(
        "structural_and_memory_gate_passed"
    )
    assert events.index("after_forward_compile") < events.index("device_put")
    assert events.index("after_forward_compile") < events.index("candidate_invoked")
    assert events.count("candidate_invoked") == 1
    assert counters == {
        "forward_attempts": 1,
        "forward_completions": 1,
        "lowered_callable_invocations": 0,
    }


def test_backend_journal_failure_prevents_compile_input_and_candidate(monkeypatch):
    events = []
    monkeypatch.setattr(_PROBE, "_allocator_snapshot", lambda _jax: [])

    def fail_backend_checkpoint(_require, _output, stage, _counters):
        events.append(stage)
        if stage == "after_backend_initialization":
            raise RuntimeError("synthetic backend quarantine")
        return {"amdgpu_boot_clean": True}

    monkeypatch.setattr(_PROBE, "_journal_checkpoint", fail_backend_checkpoint)
    monkeypatch.setattr(
        _PROBE,
        "_compile_checked_forward",
        lambda *_arguments: events.append("compile_called"),
    )
    monkeypatch.setattr(
        _PROBE,
        "_construct_host_inputs",
        lambda: events.append("inputs_constructed"),
    )
    counters = _PROBE._zero_counters()

    with pytest.raises(RuntimeError, match="synthetic backend quarantine"):
        _PROBE._run_rocm(
            io.StringIO(),
            lambda: {"amdgpu_boot_clean": True},
            counters,
            _dependencies=_run_dependencies(),
        )

    assert events == ["after_backend_initialization"]
    assert counters == _PROBE._zero_counters()


@pytest.mark.parametrize(
    "metrics",
    [
        {
            "finite": False,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "relative_l2": 0.0,
            "cosine": 1.0,
        },
        {
            "finite": True,
            "max_abs": 0.021,
            "mean_abs": 0.001,
            "relative_l2": 0.001,
            "cosine": 1.0,
        },
        {
            "finite": True,
            "max_abs": 0.001,
            "mean_abs": 0.001,
            "relative_l2": 0.01,
            "cosine": 1.0,
        },
        {
            "finite": True,
            "max_abs": 0.001,
            "mean_abs": 0.001,
            "relative_l2": 0.001,
            "cosine": 0.9998,
        },
    ],
)
def test_numerical_threshold_failure_is_fatal_after_one_candidate(monkeypatch, metrics):
    _patch_runtime_happy_path(monkeypatch, metrics)
    counters = _PROBE._zero_counters()

    with pytest.raises(RuntimeError, match="single-forward gate failed"):
        _PROBE._run_rocm(
            io.StringIO(),
            lambda: {"amdgpu_boot_clean": True},
            counters,
            _dependencies=_run_dependencies(),
        )

    assert counters["forward_attempts"] == 1
    assert counters["forward_completions"] == 1


def test_dispatch_at_100ms_is_fatal_after_one_candidate(monkeypatch):
    metrics = {
        "finite": True,
        "max_abs": 0.001,
        "mean_abs": 0.0001,
        "relative_l2": 0.001,
        "cosine": 0.99999,
    }
    _patch_runtime_happy_path(monkeypatch, metrics, dispatch_seconds=0.1)
    counters = _PROBE._zero_counters()

    with pytest.raises(RuntimeError, match="single-forward gate failed"):
        _PROBE._run_rocm(
            io.StringIO(),
            lambda: {"amdgpu_boot_clean": True},
            counters,
            _dependencies=_run_dependencies(),
        )

    assert counters["forward_attempts"] == 1
    assert counters["forward_completions"] == 1


def test_outer_guard_postflight_runs_when_runtime_raises(monkeypatch):
    state = {"postflight": 0}

    @contextmanager
    def guarded_process():
        yield {"amdgpu_boot_clean": True, "kfd_unowned": True}

    def postflight():
        state["postflight"] += 1
        return {"amdgpu_boot_clean": True}

    monkeypatch.setattr(_PROBE, "_configure_rocm_environment", lambda: {})
    monkeypatch.setattr(
        _PROBE, "_load_safety_helpers", lambda: (guarded_process, postflight)
    )
    monkeypatch.setattr(
        _PROBE,
        "_run_rocm",
        lambda *_arguments: (_ for _ in ()).throw(RuntimeError("synthetic runtime")),
    )
    output = io.StringIO()

    result = _PROBE._execute(SimpleNamespace(platform="rocm", allow_gpu=True), output)

    assert result == 1
    assert state["postflight"] == 1
    error = json.loads(output.getvalue().splitlines()[-1])
    assert error["stage"] == "runtime"
    assert error["message"] == "synthetic runtime"


def test_conflicting_environment_fails_before_guard(monkeypatch):
    state = {"guard_loaded": False}
    monkeypatch.setattr(
        _PROBE,
        "_configure_rocm_environment",
        lambda: (_ for _ in ()).throw(RuntimeError("unsafe inherited allocator")),
    )
    monkeypatch.setattr(
        _PROBE,
        "_load_safety_helpers",
        lambda: state.__setitem__("guard_loaded", True),
    )

    result = _PROBE._execute(
        SimpleNamespace(platform="rocm", allow_gpu=True), io.StringIO()
    )

    assert result == 1
    assert state["guard_loaded"] is False


def test_source_has_no_early_jax_import_backward_or_pre_gate_call():
    module = ast.parse(_PROBE_PATH.read_text(encoding="utf-8"))
    imports = [
        node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    roots = {alias.name.partition(".")[0] for node in imports for alias in node.names}
    assert "jax" not in roots
    assert "skyrl" not in roots

    compile_source = inspect.getsource(_PROBE._compile_checked_forward)
    assert compile_source.index('if not release_gate["passed"]') < compile_source.index(
        "checked = _wrap_checked"
    )
    full_source = _PROBE_PATH.read_text(encoding="utf-8")
    assert "jax.vjp" not in full_source
    assert "jax.grad" not in full_source
    assert "_chunked_reference_attention" not in full_source


def test_documented_tiny_probe_uses_expected_telemetry_caps():
    text = _DOC_PATH.read_text(encoding="utf-8")

    assert "--max-vram-gib 4" in text  # Existing compile-only gate.
    assert "--max-vram-gib 2" in text  # New single-execution gate.
    assert "--max-junction-temp-c 70" in text
    assert "--max-gpu-power-watts 315" in text
