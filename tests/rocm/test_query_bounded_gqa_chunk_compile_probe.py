from __future__ import annotations

import ast
import hashlib
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
from typing import Any

import pytest

_REPO = Path(__file__).parents[2]
_PROBE_PATH = _REPO / "rocm" / "probe_query_bounded_gqa_chunk_compile.py"
_SPEC = importlib.util.spec_from_file_location("probe_query_bounded_gqa_chunk_compile_test", _PROBE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)

_TARGET = "__gpu$xla.gpu.triton"
_MARKER = "query_bounded_gqa_forward_q256"
_DISABLED_COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="
_CLEAN_SAFETY = {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}
_HEADLESS_SAFETY = {
    **_CLEAN_SAFETY,
    "amd_cards": ["card1"],
    "connected_amd_connectors": [],
    "kfd_path": "/dev/kfd",
    "kfd_accessible": True,
    "kfd_unowned": True,
}
_ACCELERATOR_ENVIRONMENT = {
    "JAX_PLATFORMS",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "JAX_ROCM_VISIBLE_DEVICES",
    "XLA_FLAGS",
}


def _jax_modules() -> set[str]:
    return {
        name
        for name in sys.modules
        if name in {"jax", "jaxlib"} or name.startswith("jax.") or name.startswith("jaxlib.")
    }


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in _ACCELERATOR_ENVIRONMENT:
        environment.pop(name, None)
    environment["PYTHONPATH"] = str(_REPO)
    return subprocess.run(
        [sys.executable, str(_PROBE_PATH), *arguments],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )


def test_default_execute_refuses_without_importing_jax_and_hashes_all_sources():
    before = _jax_modules()
    output = io.StringIO()

    result = _PROBE._execute(SimpleNamespace(platform="abstract", allow_gpu=False), output)

    assert result == 0
    assert _jax_modules() == before
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [record["record_type"] for record in records] == [
        "manifest",
        "refused",
    ]
    manifest, refused = records
    assert manifest["scope"] == "abstract_refusal"
    assert manifest["compile_may_dispatch_gpu_work"] is False
    assert manifest["fresh_process_required"] is True
    assert manifest["raw_ir_emitted"] is False
    assert manifest["counters"] == _PROBE._zero_counters()
    for name, path in _PROBE._source_files().items():
        assert manifest[name] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert refused["jax_imported"] is False
    assert refused["counters"] == _PROBE._zero_counters()


def test_default_subprocess_is_refusal_only_and_stdout_contains_no_jax_output():
    result = _run()

    assert result.returncode == 0
    assert result.stderr == ""
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert [record["record_type"] for record in records] == [
        "manifest",
        "refused",
    ]
    assert records[-1]["jax_imported"] is False


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--platform", "rocm"), "requires the explicit --allow-gpu"),
        (
            ("--platform", "rocm", "--allow-gpu"),
            "requires --output for a private JSONL artifact",
        ),
        (("--allow-gpu",), "only valid with --platform rocm"),
        (("--query-size", "512"), "unrecognized arguments"),
        (("--sequence-length", "1024"), "unrecognized arguments"),
        (("--query-start", "0"), "unrecognized arguments"),
        (("--interpret",), "unrecognized arguments"),
        (("--execute",), "unrecognized arguments"),
        (("--replay",), "unrecognized arguments"),
        (("--backward",), "unrecognized arguments"),
    ],
)
def test_unsafe_or_scope_broadening_arguments_are_rejected(arguments, message, capsys):
    with pytest.raises(SystemExit) as raised:
        _PROBE._parse_args(list(arguments))

    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_output_is_exclusive_mode_0600_and_not_overwritten(tmp_path):
    output = tmp_path / "chunk-compile.jsonl"

    result = _PROBE.main(["--output", str(output)])

    assert result == 0
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert [json.loads(line)["record_type"] for line in output.read_text().splitlines()] == ["manifest", "refused"]
    with pytest.raises(SystemExit):
        _PROBE._parse_args(["--output", str(output)])


def test_subprocess_output_is_private_and_exclusive(tmp_path):
    output = tmp_path / "chunk-compile-subprocess.jsonl"

    result = _run("--output", str(output))

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    repeated = _run("--output", str(output))
    assert repeated.returncode == 2
    assert "refusing to overwrite existing output" in repeated.stderr
    assert str(output) not in repeated.stderr


def test_contract_fixes_exact_c256_t512_last_chunk_and_memory_bytes():
    contract = _PROBE._exact_contract()

    assert contract["operation"] == ("query_bounded_gqa_forward_chunk_compile_only")
    assert contract["api"] == "query_bounded_gqa_forward_chunk"
    assert contract["inputs"] == [
        {
            "name": "q_chunk",
            "shape": [1, 256, 16, 256],
            "dtype": "bfloat16",
        },
        {"name": "k", "shape": [1, 512, 4, 256], "dtype": "bfloat16"},
        {"name": "v", "shape": [1, 512, 4, 256], "dtype": "bfloat16"},
        {"name": "key_mask", "shape": [1, 512], "dtype": "int32"},
    ]
    assert contract["output"] == {
        "shape": [1, 256, 16, 256],
        "dtype": "bfloat16",
    }
    assert contract["query_start"] == 256
    assert contract["query_stop"] == 512
    assert contract["sequence_length"] == 512
    assert contract["tiles"] == {"block_q": 64, "block_k": 64}
    assert contract["interpret"] is False
    assert contract["compile_plan"] == {
        "lower_calls": 1,
        "compile_calls": 1,
        "compiled_executable_invocations": 0,
        "lowered_callable_invocations": 0,
        "executable_returned": False,
    }
    assert contract["compiled_memory_gate"] == {
        "memory_analysis_required": True,
        "exact_argument_bytes": 4_196_352,
        "exact_output_bytes": 2_097_152,
        "maximum_temporary_bytes": 64 * 1024**2,
    }


def _stablehlo(
    *,
    marker: str = _MARKER,
    target: str = _TARGET,
    metadata: str | None = "query_start=256 query_size=256",
    extra_call: bool = False,
    outside: str = "",
) -> str:
    location = marker if metadata is None else f"{marker} {metadata}"
    lines = [
        f'#loc0 = loc("{location}")',
        "module {",
        f'  %0 = stablehlo.custom_call @"{target}"() ' ": () -> tensor<1xbf16> loc(#loc0)",
    ]
    if extra_call:
        lines.append('  %1 = stablehlo.custom_call @"not_pallas"() ' ": () -> tensor<1xbf16>")
    lines.append("}")
    if outside:
        lines.append(outside)
    return "\n".join(lines)


def _optimized_hlo(
    *,
    marker: str = _MARKER,
    target: str = _TARGET,
    metadata: str | None = "query_start=256 query_size=256",
    extra_call: bool = False,
    outside: str = "",
) -> str:
    operation_name = marker if metadata is None else f"{marker} {metadata}"
    lines = [
        "ENTRY main {",
        f'  ROOT %0 = bf16[1] custom-call(), custom_call_target="{target}", ' f'op_name="{operation_name}"',
    ]
    if extra_call:
        lines.append('  %1 = bf16[1] custom-call(), custom_call_target="not_pallas", ' 'op_name="unrelated"')
    lines.append("}")
    if outside:
        lines.append(outside)
    return "\n".join(lines)


@pytest.mark.parametrize(
    ("dialect", "builder"),
    [("stablehlo", _stablehlo), ("optimized_hlo", _optimized_hlo)],
)
def test_exact_ir_summary_passes_without_emitting_raw_ir(dialect, builder):
    raw = builder()

    summary = _PROBE._ir_summary(raw, dialect)

    assert summary["passed"] is True
    assert summary["custom_call_count"] == 1
    assert summary["pallas_custom_call_count"] == 1
    assert summary["unexpected_query_bounded_token_occurrences"] == 0
    assert summary["metadata"]["preserved"] is True
    assert summary["metadata"]["query_start"]["values_match_expected"] is True
    assert summary["metadata"]["query_size"]["values_match_expected"] is True
    artifact = json.dumps(summary)
    assert raw not in artifact
    assert "raw_ir_emitted" in summary and summary["raw_ir_emitted"] is False


@pytest.mark.parametrize(
    ("dialect", "builder"),
    [("stablehlo", _stablehlo), ("optimized_hlo", _optimized_hlo)],
)
def test_absent_query_metadata_is_accepted_only_as_not_preserved(dialect, builder):
    summary = _PROBE._ir_summary(builder(metadata=None), dialect)

    assert summary["metadata"]["preserved"] is False
    assert summary["checks"]["preserved_query_metadata_is_exact"] is True
    assert summary["passed"] is True


@pytest.mark.parametrize(
    ("dialect", "builder"),
    [("stablehlo", _stablehlo), ("optimized_hlo", _optimized_hlo)],
)
def test_mlir_hex_escaped_exact_query_metadata_is_decoded(dialect, builder):
    metadata = r"\22query_start\22:\22256\22,\22query_size\22:\22256\22"
    summary = _PROBE._ir_summary(builder(metadata=metadata), dialect)

    assert summary["metadata"]["query_start"]["exact_single_occurrence"] is True
    assert summary["metadata"]["query_size"]["exact_single_occurrence"] is True
    assert summary["checks"]["preserved_query_metadata_is_exact"] is True
    assert summary["passed"] is True


@pytest.mark.parametrize(
    ("dialect", "builder"),
    [("stablehlo", _stablehlo), ("optimized_hlo", _optimized_hlo)],
)
def test_mlir_hex_escaped_kernel_marker_is_classified_after_decode(dialect, builder):
    exact = _PROBE._ir_summary(builder(marker=r"query_bounded_gqa_forward_q\32\35\36"), dialect)
    wrong = _PROBE._ir_summary(builder(marker=r"query_bounded_gqa_forward_q\30"), dialect)

    assert exact["checks"]["exact_full_forward_q256_marker_in_sole_call"] is True
    assert exact["passed"] is True
    assert wrong["unexpected_query_bounded_token_occurrences"] > 0
    assert wrong["passed"] is False


@pytest.mark.parametrize(
    ("dialect", "builder"),
    [("stablehlo", _stablehlo), ("optimized_hlo", _optimized_hlo)],
)
@pytest.mark.parametrize(
    ("metadata", "marker"),
    [
        ("query_start=0 query_size=256", _MARKER),
        ("query_start=256 query_size=512", _MARKER),
        ("query_start=256", _MARKER),
        ("query_size=256", _MARKER),
        ('query_start="private" query_size=256', _MARKER),
        ("foo.query_start=0 foo.query_size=512", _MARKER),
        ("query_start=256 query_start=256 query_size=256", _MARKER),
        ("query_start=256 query_size=256 query_size=256", _MARKER),
    ],
)
def test_wrong_partial_or_unparseable_preserved_metadata_fails_closed(dialect, builder, metadata, marker):
    raw = builder(marker=marker, metadata=metadata)

    summary = _PROBE._ir_summary(raw, dialect)

    assert summary["metadata"]["preserved"] is True
    assert summary["checks"]["preserved_query_metadata_is_exact"] is False
    assert summary["passed"] is False
    assert "private" not in json.dumps(summary)


@pytest.mark.parametrize(
    ("dialect", "builder"),
    [("stablehlo", _stablehlo), ("optimized_hlo", _optimized_hlo)],
)
@pytest.mark.parametrize(
    "marker",
    [
        "query_bounded_gqa_forward_q0",
        "query_bounded_gqa_forward_q512",
        "query_bounded_gqa_dq_q256",
        "query_bounded_gqa_dkdv_q256",
        "query_bounded_gqa_forward_q256_suffix",
        "prefix_query_bounded_gqa_forward_q256",
        "query-bounded-gqa-forward-q256",
        "query.bounded.gqa.forward.q256",
        "query_bounded_gqa_forward_q25",
        "unrelated_kernel",
    ],
)
def test_q0_other_forward_backward_and_lookalike_markers_fail_closed(dialect, builder, marker):
    summary = _PROBE._ir_summary(builder(marker=marker), dialect)

    assert summary["passed"] is False
    assert summary["checks"]["exact_full_forward_q256_marker_in_sole_call"] is False


@pytest.mark.parametrize(
    ("dialect", "builder"),
    [("stablehlo", _stablehlo), ("optimized_hlo", _optimized_hlo)],
)
def test_expected_marker_outside_call_cannot_satisfy_the_call_gate(dialect, builder):
    outside = f'%outside = add(), op_name="{_MARKER}"'
    summary = _PROBE._ir_summary(builder(marker="unrelated_kernel", outside=outside), dialect)

    assert summary["expected_marker_occurrences"] == 1
    assert summary["checks"]["exact_full_forward_q256_marker_in_sole_call"] is False
    assert summary["passed"] is False


@pytest.mark.parametrize(
    ("dialect", "builder"),
    [("stablehlo", _stablehlo), ("optimized_hlo", _optimized_hlo)],
)
def test_duplicate_exact_marker_metadata_outside_the_sole_call_is_benign(dialect, builder):
    outside = f'%outside = add(), op_name="{_MARKER}"'
    summary = _PROBE._ir_summary(builder(outside=outside), dialect)

    assert summary["expected_marker_occurrences"] == 2
    assert summary["checks"]["exact_full_forward_q256_marker_in_sole_call"] is True
    assert summary["checks"]["exactly_one_custom_call_total"] is True
    assert summary["passed"] is True


@pytest.mark.parametrize(
    ("dialect", "builder"),
    [("stablehlo", _stablehlo), ("optimized_hlo", _optimized_hlo)],
)
def test_non_kernel_public_api_name_is_not_misclassified_as_call_marker(dialect, builder):
    summary = _PROBE._ir_summary(builder(outside="query_bounded_gqa_forward_chunk"), dialect)

    assert summary["unexpected_query_bounded_token_occurrences"] == 0
    assert summary["passed"] is True


@pytest.mark.parametrize(
    ("dialect", "builder"),
    [("stablehlo", _stablehlo), ("optimized_hlo", _optimized_hlo)],
)
@pytest.mark.parametrize("target", ["triton", "xla.gpu.triton", "nottriton"])
def test_nonexact_triton_target_fails_even_when_it_is_pallas_like(dialect, builder, target):
    summary = _PROBE._ir_summary(builder(target=target), dialect)

    assert summary["checks"]["sole_exact_rocm_triton_target"] is False
    assert summary["passed"] is False


@pytest.mark.parametrize(
    ("dialect", "builder"),
    [("stablehlo", _stablehlo), ("optimized_hlo", _optimized_hlo)],
)
def test_an_extra_nonpallas_custom_call_still_fails_total_call_gate(dialect, builder):
    summary = _PROBE._ir_summary(builder(extra_call=True), dialect)

    assert summary["custom_call_count"] == 2
    assert summary["pallas_custom_call_count"] == 1
    assert summary["checks"]["exactly_one_custom_call_total"] is False
    assert summary["passed"] is False


@pytest.mark.parametrize(
    ("dialect", "builder", "outer_while"),
    [
        ("stablehlo", _stablehlo, "%loop = stablehlo.while(%arg0)"),
        ("optimized_hlo", _optimized_hlo, "%loop = bf16[1] while (%arg0)"),
    ],
)
def test_outer_while_fails_closed(dialect, builder, outer_while):
    summary = _PROBE._ir_summary(builder(outside=outer_while), dialect)

    assert summary["while_count"] == 1
    assert summary["checks"]["no_outer_while"] is False
    assert summary["passed"] is False


def test_structural_gate_requires_both_independent_dialects_exactly_once():
    stable = _PROBE._ir_summary(_stablehlo(), "stablehlo")
    optimized = _PROBE._ir_summary(_optimized_hlo(), "optimized_hlo")

    assert _PROBE._structural_gate(stable, optimized)["passed"] is True
    assert _PROBE._structural_gate(stable)["passed"] is False
    assert _PROBE._structural_gate(stable, stable)["passed"] is False
    broken = dict(optimized)
    broken["passed"] = False
    assert _PROBE._structural_gate(stable, broken)["passed"] is False
    forged = dict(optimized)
    forged["checks"] = {}
    forged["passed"] = True
    assert _PROBE._structural_gate(stable, forged)["passed"] is False


def _memory(
    *,
    argument: Any = 4_196_352,
    output: Any = 2_097_152,
    temporary: Any = 33_024,
    available: bool = True,
) -> dict[str, Any]:
    return {
        "available": available,
        "argument_size_in_bytes": argument,
        "output_size_in_bytes": output,
        "temp_size_in_bytes": temporary,
    }


def test_exact_memory_gate_reports_argument_output_temporary_combined():
    gate = _PROBE._compiled_memory_gate(_memory())

    assert gate["passed"] is True
    assert gate["argument_output_temporary_bytes"] == (4_196_352 + 2_097_152 + 33_024)
    assert all(gate["checks"].values())


@pytest.mark.parametrize(
    ("memory", "failed_check"),
    [
        (_memory(available=False), "memory_analysis_available"),
        (
            {
                "available": True,
                "output_size_in_bytes": 2_097_152,
                "temp_size_in_bytes": 0,
            },
            "memory_analysis_available",
        ),
        (_memory(argument=4_196_351), "argument_bytes_exactly_4196352"),
        (_memory(argument=4_196_353), "argument_bytes_exactly_4196352"),
        (_memory(output=2_097_151), "output_bytes_exactly_2097152"),
        (_memory(output=2_097_153), "output_bytes_exactly_2097152"),
        (
            _memory(temporary=64 * 1024**2 + 1),
            "temporary_bytes_at_most_64_mib",
        ),
        (_memory(argument=True), "memory_analysis_available"),
        (_memory(output=2_097_152.0), "memory_analysis_available"),
        (_memory(temporary=-1), "memory_analysis_available"),
    ],
)
def test_missing_drifted_oversized_or_noninteger_memory_fails_closed(memory, failed_check):
    gate = _PROBE._compiled_memory_gate(memory)

    assert gate["passed"] is False
    assert gate["checks"][failed_check] is False


class _FakeShapeDtypeStruct:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _FakeCompiled:
    def __init__(
        self,
        state: dict[str, Any],
        *,
        optimized_text: str | None = None,
        memory: Any | None = None,
    ) -> None:
        self._state = state
        self._optimized_text = _optimized_hlo() if optimized_text is None else optimized_text
        self._memory = (
            SimpleNamespace(
                argument_size_in_bytes=4_196_352,
                output_size_in_bytes=2_097_152,
                alias_size_in_bytes=0,
                temp_size_in_bytes=33_024,
            )
            if memory is None
            else memory
        )

    def __call__(self, *_args, **_kwargs):
        self._state["compiled_executable_invocations"] += 1
        raise AssertionError("compile-only gate invoked its executable")

    def as_text(self):
        self._state["as_text_calls"] += 1
        return self._optimized_text

    def memory_analysis(self):
        self._state["memory_analysis_calls"] += 1
        return self._memory


class _FakeLowered:
    def __init__(
        self,
        state: dict[str, Any],
        *,
        stable_text: str | None = None,
        compiled: _FakeCompiled | None = None,
        compile_error: BaseException | None = None,
    ) -> None:
        self._state = state
        self._stable_text = _stablehlo() if stable_text is None else stable_text
        self._compiled = _FakeCompiled(state) if compiled is None else compiled
        self._compile_error = compile_error

    def compiler_ir(self, *, dialect):
        assert dialect == "stablehlo"
        self._state["compiler_ir_calls"] += 1
        return self._stable_text

    def compile(self):
        self._state["compile_calls"] += 1
        if self._compile_error is not None:
            raise self._compile_error
        return self._compiled


class _FakeJitted:
    def __init__(
        self,
        function,
        state: dict[str, Any],
        lowered: _FakeLowered,
        *,
        lower_error: BaseException | None = None,
    ) -> None:
        self._function = function
        self._state = state
        self._lowered = lowered
        self._lower_error = lower_error

    def lower(self, *signature):
        self._state["lower_calls"] += 1
        self._state["signature"] = signature
        if self._lower_error is not None:
            raise self._lower_error
        self._function(*signature)
        return self._lowered


class _FakeJax:
    ShapeDtypeStruct = _FakeShapeDtypeStruct

    def __init__(
        self,
        state: dict[str, Any],
        lowered: _FakeLowered,
        *,
        lower_error: BaseException | None = None,
    ) -> None:
        self._state = state
        self._lowered = lowered
        self._lower_error = lower_error

    def jit(self, function):
        self._state["jit_calls"] += 1
        return _FakeJitted(
            function,
            self._state,
            self._lowered,
            lower_error=self._lower_error,
        )


def _fake_state() -> dict[str, Any]:
    return {
        "jit_calls": 0,
        "lower_calls": 0,
        "compiler_ir_calls": 0,
        "compile_calls": 0,
        "as_text_calls": 0,
        "memory_analysis_calls": 0,
        "compiled_executable_invocations": 0,
        "api_calls": [],
        "journals": [],
    }


def _fake_api(state):
    def api(*arguments, **keywords):
        state["api_calls"].append((arguments, keywords))
        return "abstract-output"

    return api


def _fake_journal(state):
    def require():
        return dict(_CLEAN_SAFETY)

    return require


def test_lower_compile_once_uses_exact_api_and_never_invokes_or_returns_executable():
    state = _fake_state()
    compiled = _FakeCompiled(state)
    lowered = _FakeLowered(state, compiled=compiled)
    jax = _FakeJax(state, lowered)
    jnp = SimpleNamespace(bfloat16="bf16", int32="i32")
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    report = _PROBE._lower_and_compile_exact(
        jax,
        jnp,
        _fake_api(state),
        _fake_journal(state),
        counters,
        output,
    )

    assert state["jit_calls"] == 1
    assert state["lower_calls"] == 1
    assert state["compiler_ir_calls"] == 1
    assert state["compile_calls"] == 1
    assert state["as_text_calls"] == 1
    assert state["memory_analysis_calls"] == 1
    assert state["compiled_executable_invocations"] == 0
    assert [item.shape for item in state["signature"]] == [
        (1, 256, 16, 256),
        (1, 512, 4, 256),
        (1, 512, 4, 256),
        (1, 512),
    ]
    assert [item.dtype for item in state["signature"]] == [
        "bf16",
        "bf16",
        "bf16",
        "i32",
    ]
    assert len(state["api_calls"]) == 1
    arguments, keywords = state["api_calls"][0]
    assert arguments == state["signature"]
    assert keywords == {
        "query_start": 256,
        "block_q": 64,
        "block_k": 64,
        "interpret": False,
    }
    assert report["release_gate"]["passed"] is True
    assert report["executable_returned"] is False
    assert report["counters"] == _PROBE._zero_counters()
    assert "compiled" not in report
    assert all(value is not compiled for value in report.values())
    assert counters == _PROBE._zero_counters()
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [record["record_type"] for record in records] == [
        "stage",
        "lowered",
        "journal_checkpoint",
        "stage",
        "chunk_compiled",
        "journal_checkpoint",
    ]
    assert [record["stage"] for record in records if record["record_type"] == "journal_checkpoint"] == [
        "after_chunk_lower_attempt",
        "after_chunk_compile_attempt",
    ]


@pytest.mark.parametrize(
    ("stable_text", "optimized_text", "memory"),
    [
        (_stablehlo(marker="query_bounded_gqa_forward_q0"), _optimized_hlo(), None),
        (_stablehlo(), _optimized_hlo(target="triton"), None),
        (
            _stablehlo(),
            _optimized_hlo(),
            SimpleNamespace(
                argument_size_in_bytes=4_196_351,
                output_size_in_bytes=2_097_152,
                temp_size_in_bytes=0,
            ),
        ),
    ],
)
def test_structural_or_memory_violation_never_invokes_executable(stable_text, optimized_text, memory):
    state = _fake_state()
    compiled = _FakeCompiled(
        state,
        optimized_text=optimized_text,
        memory=memory,
    )
    lowered = _FakeLowered(state, stable_text=stable_text, compiled=compiled)

    with pytest.raises(RuntimeError, match="structural or exact memory gate"):
        _PROBE._lower_and_compile_exact(
            _FakeJax(state, lowered),
            SimpleNamespace(bfloat16="bf16", int32="i32"),
            _fake_api(state),
            _fake_journal(state),
            _PROBE._zero_counters(),
            io.StringIO(),
        )

    assert state["compile_calls"] == 1
    assert state["compiled_executable_invocations"] == 0


def test_lower_failure_still_checkpoints_and_never_compiles():
    state = _fake_state()
    lowered = _FakeLowered(state)
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="synthetic lower failure"):
        _PROBE._lower_and_compile_exact(
            _FakeJax(
                state,
                lowered,
                lower_error=RuntimeError("synthetic lower failure"),
            ),
            SimpleNamespace(bfloat16="bf16", int32="i32"),
            _fake_api(state),
            _fake_journal(state),
            _PROBE._zero_counters(),
            output,
        )

    assert state["lower_calls"] == 1
    assert state["compile_calls"] == 0
    assert state["compiled_executable_invocations"] == 0
    journals = [
        json.loads(line)["stage"]
        for line in output.getvalue().splitlines()
        if json.loads(line)["record_type"] == "journal_checkpoint"
    ]
    assert journals == ["after_chunk_lower_attempt"]


def test_compile_failure_still_checkpoints_and_never_invokes():
    state = _fake_state()
    lowered = _FakeLowered(state, compile_error=RuntimeError("synthetic compile failure"))
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="synthetic compile failure"):
        _PROBE._lower_and_compile_exact(
            _FakeJax(state, lowered),
            SimpleNamespace(bfloat16="bf16", int32="i32"),
            _fake_api(state),
            _fake_journal(state),
            _PROBE._zero_counters(),
            output,
        )

    assert state["compile_calls"] == 1
    assert state["compiled_executable_invocations"] == 0
    journals = [
        json.loads(line)["stage"]
        for line in output.getvalue().splitlines()
        if json.loads(line)["record_type"] == "journal_checkpoint"
    ]
    assert journals == [
        "after_chunk_lower_attempt",
        "after_chunk_compile_attempt",
    ]


class _BackendJax(_FakeJax):
    __version__ = "private-jax-version"

    def default_backend(self):
        return "gpu"

    def devices(self):
        return ["PRIVATE_DEVICE_DESCRIPTION"]


def test_backend_manifest_requires_exactly_one_visible_device():
    backend = SimpleNamespace(get_backend=lambda: SimpleNamespace(platform_version="ROCm PRIVATE PLATFORM PATH"))
    for devices in ([], ["device0", "device1"]):
        jax = SimpleNamespace(
            __version__="private-jax-version",
            default_backend=lambda: "gpu",
            devices=lambda devices=devices: devices,
        )
        with pytest.raises(RuntimeError, match="exactly one visible accelerator"):
            _PROBE._backend_manifest(jax, SimpleNamespace(__version__="private-jaxlib-version"), backend)


def _dependencies(state, lowered):
    jax = _BackendJax(state, lowered)
    jnp = SimpleNamespace(bfloat16="bf16", int32="i32")
    jaxlib = SimpleNamespace(__version__="private-jaxlib-version")
    backend = SimpleNamespace(get_backend=lambda: SimpleNamespace(platform_version="ROCm PRIVATE PLATFORM PATH"))
    return jax, jnp, jaxlib, backend, _fake_api(state)


def _environment(monkeypatch, *, flags=_DISABLED_COMMAND_BUFFER_FLAG):
    monkeypatch.setenv("XLA_FLAGS", flags)
    return {"XLA_FLAGS_effective": flags}


def test_mocked_run_has_exact_order_zero_invocations_and_report_only(
    monkeypatch,
):
    state = _fake_state()
    lowered = _FakeLowered(state)
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    report = _PROBE._run_rocm(
        output,
        lambda: dict(_CLEAN_SAFETY),
        counters,
        environment=_environment(monkeypatch),
        _dependencies=_dependencies(state, lowered),
    )

    assert report["release_gate"]["passed"] is True
    assert report["executable_returned"] is False
    assert state["compiled_executable_invocations"] == 0
    assert counters == _PROBE._zero_counters()
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [record["record_type"] for record in records] == [
        "command_buffer_environment_proof",
        "journal_checkpoint",
        "backend_ready",
        "journal_checkpoint",
        "stage",
        "lowered",
        "journal_checkpoint",
        "stage",
        "chunk_compiled",
        "journal_checkpoint",
        "compile_only_passed",
    ]
    journals = [record["stage"] for record in records if record["record_type"] == "journal_checkpoint"]
    assert journals == list(_PROBE._JOURNAL_STAGES)
    assert all(record["counters"] == _PROBE._zero_counters() for record in records)
    artifact = output.getvalue()
    assert "PRIVATE_DEVICE_DESCRIPTION" not in artifact
    assert "private-jax-version" not in artifact
    assert "private-jaxlib-version" not in artifact
    assert "ROCm PRIVATE PLATFORM PATH" not in artifact


@pytest.mark.parametrize(
    "flags",
    [
        "--xla_gpu_enable_command_buffer=false",
        "--xla_gpu_enable_command_buffer",
        "--noxla_gpu_enable_command_buffer",
        ("--xla_gpu_enable_command_buffer= " "--xla_gpu_enable_command_buffer=true"),
        "--xla_gpu_enable_command_buffer= --xla_gpu_enable_command_buffer=",
    ],
)
def test_invalid_command_buffer_proof_fails_before_backend(monkeypatch, flags):
    state = _fake_state()
    lowered = _FakeLowered(state)
    dependencies = list(_dependencies(state, lowered))
    dependencies[0].default_backend = lambda: (_ for _ in ()).throw(AssertionError("backend must remain unused"))

    with pytest.raises(RuntimeError, match="sole empty assignment"):
        _PROBE._run_rocm(
            io.StringIO(),
            lambda: dict(_CLEAN_SAFETY),
            _PROBE._zero_counters(),
            environment=_environment(monkeypatch, flags=flags),
            _dependencies=tuple(dependencies),
        )

    assert state["lower_calls"] == 0
    assert state["compile_calls"] == 0
    assert state["compiled_executable_invocations"] == 0


def _configured_environment(secret_path=""):
    original = f"--xla_dump_to={secret_path}" if secret_path else ""
    effective = f"{original} {_DISABLED_COMMAND_BUFFER_FLAG}".strip()
    return {
        "JAX_PLATFORMS": "rocm",
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
        "JAX_ROCM_VISIBLE_DEVICES": "0",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_CLIENT_MEM_FRACTION": "0.75",
        "SKYRL_ROCM_PALLAS_ATTENTION": "1",
        "XLA_FLAGS_original": original,
        "XLA_FLAGS_effective": effective,
    }


def test_environment_and_command_buffer_records_digest_private_xla_paths(
    monkeypatch,
):
    secret = "/home/private-user/private-project/compiler-dump"
    environment = _configured_environment(secret)
    monkeypatch.setenv("XLA_FLAGS", environment["XLA_FLAGS_effective"])

    artifact = json.dumps(
        {
            "environment": _PROBE._environment_manifest(environment),
            "proof": _PROBE._prove_command_buffers_disabled(environment),
        }
    )

    assert secret not in artifact
    assert environment["XLA_FLAGS_original"] not in artifact
    assert environment["XLA_FLAGS_effective"] not in artifact


@pytest.mark.parametrize(
    "safety",
    [
        {"amdgpu_boot_clean": False, "fatal_amdgpu_events": []},
        {"amdgpu_boot_clean": True},
        {"amdgpu_boot_clean": True, "fatal_amdgpu_events": ["private fatal"]},
    ],
)
def test_dirty_or_malformed_journal_result_never_becomes_public(safety):
    with pytest.raises(RuntimeError, match="clean AMDGPU boot|fatal-event proof"):
        _PROBE._public_clean_safety(safety, "test_stage")


def test_preflight_emits_only_controlled_headless_and_kfd_evidence():
    assert _PROBE._public_safety_preflight(dict(_HEADLESS_SAFETY)) == (_HEADLESS_SAFETY)


@pytest.mark.parametrize(
    "mutation",
    [
        {"amd_cards": []},
        {"amd_cards": ["/private/card1"]},
        {"amd_cards": ["card2", "card1"]},
        {"connected_amd_connectors": ["card1-HDMI-A-1"]},
        {"kfd_path": "/private/kfd"},
        {"kfd_accessible": False},
        {"kfd_unowned": False},
    ],
)
def test_malformed_headless_or_kfd_evidence_fails_closed(mutation):
    with pytest.raises(RuntimeError, match="safety_preflight"):
        _PROBE._public_safety_preflight({**_HEADLESS_SAFETY, **mutation})


def test_unknown_or_out_of_order_journal_stage_is_rejected():
    with pytest.raises(RuntimeError, match="undeclared"):
        _PROBE._journal_checkpoint(
            lambda: dict(_CLEAN_SAFETY),
            io.StringIO(),
            "after_execution",
            _PROBE._zero_counters(),
        )


def test_corrupted_invocation_counter_prevents_journal_and_success():
    counters = _PROBE._zero_counters()
    counters["compiled_executable_invocations"] = 1

    with pytest.raises(RuntimeError, match="zero-invocation"):
        _PROBE._journal_checkpoint(
            lambda: dict(_CLEAN_SAFETY),
            io.StringIO(),
            "after_chunk_compile_attempt",
            counters,
        )


def test_terminal_runtime_error_is_digest_only_and_postflight_precedes_it(
    monkeypatch,
):
    secret = f"PRIVATE_RUNTIME_SECRET {_REPO}/compiler-cache"
    environment = _configured_environment()

    @contextmanager
    def guard():
        yield dict(_HEADLESS_SAFETY)

    monkeypatch.setattr(_PROBE, "_assert_fresh_accelerator_process", lambda: None)
    monkeypatch.setattr(_PROBE, "_configure_rocm_environment", lambda: environment)
    monkeypatch.setattr(
        _PROBE,
        "_load_safety_helpers",
        lambda: (guard, lambda: dict(_CLEAN_SAFETY)),
    )
    monkeypatch.setattr(
        _PROBE,
        "_run_rocm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    output = io.StringIO()

    result = _PROBE._execute(SimpleNamespace(platform="rocm", allow_gpu=True), output)

    assert result == 1
    artifact = output.getvalue()
    records = [json.loads(line) for line in artifact.splitlines()]
    error = records[-1]
    encoded = secret.encode()
    assert [record["record_type"] for record in records[-2:]] == [
        "safety_postflight",
        "error",
    ]
    assert error["stage"] == "compile_only_runtime"
    assert error["message_redacted"] is True
    assert error["message_utf8_bytes"] == len(encoded)
    assert error["message_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert "message" not in error
    assert secret not in artifact
    assert str(_REPO) not in artifact


def test_rocm_refuses_preimported_jax_before_environment(monkeypatch):
    state = {"environment_configured": False}
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace())
    monkeypatch.setattr(
        _PROBE,
        "_configure_rocm_environment",
        lambda: state.__setitem__("environment_configured", True),
    )
    output = io.StringIO()

    result = _PROBE._execute(SimpleNamespace(platform="rocm", allow_gpu=True), output)

    assert result == 1
    assert state["environment_configured"] is False
    error = json.loads(output.getvalue().splitlines()[-1])
    assert error["stage"] == "fresh_process_preflight"
    assert error["message_redacted"] is True
    assert "message" not in error


def test_source_has_no_early_jax_import_or_execution_path():
    source = _PROBE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    imports = [node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    roots = {alias.name.partition(".")[0] for node in imports for alias in node.names}
    assert "jax" not in roots
    assert "skyrl" not in roots
    assert "jax.random" not in source
    assert "jax.vjp" not in source
    assert "jax.grad" not in source
    assert "jax.device_put" not in source
    assert "jax.device_get" not in source
    assert "block_until_ready" not in source
    run_source = inspect.getsource(_PROBE._run_rocm)
    assert run_source.index("_prove_command_buffers_disabled") < run_source.index("import jax")
    assert run_source.index('"before_backend_initialization"') < run_source.index("import jax")


def test_ast_proves_exact_api_constants_one_lower_one_compile_and_no_call():
    module = ast.parse(_PROBE_PATH.read_text(encoding="utf-8"))
    function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_lower_and_compile_exact"
    )
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    lower_calls = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "lower"]
    compile_calls = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "compile"]
    executable_calls = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "compiled"]
    api_calls = [
        node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "query_bounded_gqa_forward_chunk"
    ]

    assert len(lower_calls) == 1
    assert len(compile_calls) == 1
    assert executable_calls == []
    assert len(api_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in api_calls[0].keywords}
    assert set(keywords) == {
        "query_start",
        "block_q",
        "block_k",
        "interpret",
    }
    assert isinstance(keywords["query_start"], ast.Name)
    assert keywords["query_start"].id == "_QUERY_START"
    assert isinstance(keywords["block_q"], ast.Name)
    assert keywords["block_q"].id == "_BLOCK_Q"
    assert isinstance(keywords["block_k"], ast.Name)
    assert keywords["block_k"].id == "_BLOCK_K"
    assert isinstance(keywords["interpret"], ast.Constant)
    assert keywords["interpret"].value is False
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    assert len(returns) == 2
    assert any(isinstance(node.value, ast.Name) and node.value.id == "report" for node in returns)
    assert not any(
        isinstance(node.value, ast.Name) and node.value.id in {"compiled", "lowered", "lowered_callable"}
        for node in returns
    )
