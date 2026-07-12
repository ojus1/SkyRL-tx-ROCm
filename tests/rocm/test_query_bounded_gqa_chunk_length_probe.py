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
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ml_dtypes
import numpy as np
import pytest

_REPO = Path(__file__).parents[2]
_PROBE_PATH = _REPO / "rocm" / "probe_query_bounded_gqa_chunk_length.py"
_SPEC = importlib.util.spec_from_file_location(
    "query_chunk_length_probe_test", _PROBE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)

_LENGTHS = (1024, 2048, 4096, 8192, 16384, 24576, 32768)
_TARGET = "__gpu$xla.gpu.triton"
_MARKER_PREFIX = "query_bounded_gqa_forward_q"


def _marker(length: int) -> str:
    return f"{_MARKER_PREFIX}{length - 256}"


_CLEAN = {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}


def _records(output: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in output.getvalue().splitlines()]


def _accelerator_modules() -> set[str]:
    return {
        name
        for name in sys.modules
        if name in {"jax", "jaxlib"} or name.startswith(("jax.", "jaxlib."))
    }


def test_default_is_abstract_refusal_without_accelerator_import():
    before = _accelerator_modules()
    output = io.StringIO()

    assert (
        _PROBE._execute(
            SimpleNamespace(platform="abstract", allow_gpu=False, sequence_length=None),
            output,
        )
        == 0
    )

    assert _accelerator_modules() == before
    manifest, refused = _records(output)
    assert manifest["scope"] == "abstract_refusal"
    assert manifest["contract"]["allowed_sequence_lengths"] == list(_LENGTHS)
    assert manifest["contract"]["gpu_work_authorized"] is False
    assert manifest["sequential_promotion_required"] is True
    assert manifest["this_probe_alone_cannot_authorize_skipping_smaller_rung"] is True
    assert manifest["counters"] == _PROBE._zero_counters()
    assert refused["jax_imported"] is False
    for name, path in _PROBE._source_files().items():
        assert manifest[name] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--platform", "rocm"), "requires the explicit --allow-gpu"),
        (("--platform", "rocm", "--allow-gpu"), "requires --output"),
        (
            ("--platform", "rocm", "--allow-gpu", "--output", "/tmp/new-length.jsonl"),
            "requires --sequence-length",
        ),
        (("--allow-gpu",), "only valid with --platform rocm"),
        (("--sequence-length", "1024"), "only valid with --platform rocm"),
        (
            (
                "--platform",
                "rocm",
                "--allow-gpu",
                "--sequence-length",
                "512",
                "--output",
                "/tmp/new-length.jsonl",
            ),
            "invalid choice",
        ),
        (("--query-start", "0"), "unrecognized arguments"),
        (("--replay",), "unrecognized arguments"),
        (("--backward",), "unrecognized arguments"),
        (("--gpu-reference",), "unrecognized arguments"),
    ],
)
def test_parser_refuses_implicit_gpu_invalid_lengths_and_scope_broadening(
    arguments, message, capsys
):
    with pytest.raises(SystemExit) as raised:
        _PROBE._parse_args(list(arguments))
    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_private_output_is_exclusive_mode_0600(tmp_path):
    path = tmp_path / "length.jsonl"
    assert _PROBE.main(["--output", str(path)]) == 0
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(SystemExit):
        _PROBE._parse_args(["--output", str(path)])


@pytest.mark.parametrize("length", _LENGTHS)
def test_exact_contract_and_memory_formula_for_each_rung(length):
    contract = _PROBE._exact_contract(length)
    expected_arguments = 2_097_152 + length * 4100

    assert contract["sequence_length"] == length
    assert [item["shape"] for item in contract["inputs"]] == [
        [1, 256, 16, 256],
        [1, length, 4, 256],
        [1, length, 4, 256],
        [1, length],
    ]
    assert contract["query_start"] == length - 256
    assert contract["query_stop"] == length
    assert contract["global_query_positions"] == [length - 256, length - 1]
    assert contract["tiles"] == {"block_q": 64, "block_k": 64}
    assert contract["compile_gate"]["exact_argument_bytes"] == expected_arguments
    assert contract["compile_gate"]["exact_output_bytes"] == 2_097_152
    assert contract["compile_gate"]["maximum_temporary_bytes"] == 64 * 1024**2
    assert contract["compile_gate"]["exact_kernel_marker"] == _marker(length)
    assert contract["dispatch_plan"]["candidate_invocations"] == 1
    assert contract["dispatch_plan"]["replay_invocations"] == 0
    assert contract["ladder_policy"]["one_length_per_fresh_process"] is True
    assert contract["ladder_policy"]["sequential_promotion_required"] is True
    assert (
        contract["ladder_policy"]["this_probe_cannot_authorize_skipping_smaller_rungs"]
        is True
    )
    assert _PROBE._expected_argument_bytes(length) == expected_arguments


@pytest.mark.parametrize("length", (512, 1000, 65536))
def test_exact_contract_rejects_non_rungs(length):
    with pytest.raises(ValueError, match="outside"):
        _PROBE._exact_contract(length)


def _stablehlo(
    length: int,
    *,
    marker: str | None = None,
    target: str = _TARGET,
    metadata: str | None = None,
):
    query_start = length - 256
    marker = _marker(length) if marker is None else marker
    metadata = (
        f"{marker} query_start={query_start} query_size=256"
        if metadata is None
        else metadata
    )
    return "\n".join(
        [
            f'#loc0 = loc("{metadata}")',
            "module {",
            f'  %0 = stablehlo.custom_call @"{target}"() : () -> tensor<1xbf16> loc(#loc0)',
            "}",
        ]
    )


def _optimized_hlo(
    length: int,
    *,
    marker: str | None = None,
    target: str = _TARGET,
    metadata: str | None = None,
):
    query_start = length - 256
    marker = _marker(length) if marker is None else marker
    metadata = (
        f"{marker} query_start={query_start} query_size=256"
        if metadata is None
        else metadata
    )
    return "\n".join(
        [
            "ENTRY main {",
            f'  ROOT %0 = bf16[1] custom-call(), custom_call_target="{target}", op_name="{metadata}"',
            "}",
        ]
    )


@pytest.mark.parametrize("length", _LENGTHS)
@pytest.mark.parametrize("dialect", ("stablehlo", "optimized_hlo"))
def test_ir_parser_accepts_exact_single_call_for_each_length(length, dialect):
    text = _stablehlo(length) if dialect == "stablehlo" else _optimized_hlo(length)
    summary = _PROBE._summarize_ir(text, dialect, length)

    assert summary["passed"] is True
    assert summary["custom_call_count"] == 1
    assert summary["pallas_custom_call_count"] == 1
    assert summary["while_count"] == 0
    assert summary["expected_marker"] == _marker(length)
    assert summary["expected_marker_occurrences"] >= 1
    assert summary["metadata"]["query_start"]["expected"] == length - 256
    assert summary["metadata"]["query_size"]["expected"] == 256
    assert all(summary["checks"].values())


@pytest.mark.parametrize("length", _LENGTHS)
@pytest.mark.parametrize(
    "corruptor",
    (
        lambda length: _stablehlo(
            length,
            metadata=f"{_marker(length)} query_start={length - 257} query_size=256",
        ),
        lambda length: _stablehlo(
            length,
            metadata=f"{_marker(length)} query_start={length - 256} query_size=255",
        ),
        lambda length: _stablehlo(length, marker="query_bounded_gqa_forward_q0"),
        lambda length: _stablehlo(length, marker="query_bounded_gqa_forward_q256"),
        lambda length: _stablehlo(
            length,
            marker=_marker(2048 if length == 1024 else 1024),
        ),
        lambda length: _stablehlo(length, marker="query_bounded_gqa_dq_q256"),
        lambda length: _stablehlo(length, target="triton"),
        lambda length: (
            _stablehlo(length)
            + '\n  %1 = stablehlo.custom_call @"triton"() : () -> tensor<1xbf16>'
        ),
        lambda length: _stablehlo(length) + "\n  %1 = stablehlo.while() : () -> ()",
    ),
)
def test_ir_parser_rejects_wrong_metadata_target_marker_extra_call_and_while(
    length, corruptor
):
    assert (
        _PROBE._summarize_ir(corruptor(length), "stablehlo", length)["passed"] is False
    )


@pytest.mark.parametrize("length", _LENGTHS)
def test_ir_metadata_may_be_cleanly_absent_but_not_partial(length):
    absent = _stablehlo(length, metadata=_marker(length))
    partial = _stablehlo(
        length, metadata=f"{_marker(length)} query_start={length - 256}"
    )
    assert _PROBE._summarize_ir(absent, "stablehlo", length)["passed"] is True
    assert _PROBE._summarize_ir(partial, "stablehlo", length)["passed"] is False


@pytest.mark.parametrize("length", _LENGTHS)
def test_ir_metadata_allows_repeated_exact_canonical_values_but_rejects_mixed(length):
    query_start = length - 256
    exact = _stablehlo(
        length,
        metadata=(
            f"{_marker(length)} query_start={query_start} query_size=256 "
            f"query_start={query_start} query_size=256"
        ),
    )
    mixed = _stablehlo(
        length,
        metadata=(
            f"{_marker(length)} query_start={query_start} query_size=256 "
            f"query_start={query_start - 1} query_size=256"
        ),
    )
    assert _PROBE._summarize_ir(exact, "stablehlo", length)["passed"] is True
    assert _PROBE._summarize_ir(mixed, "stablehlo", length)["passed"] is False


@pytest.mark.parametrize("length", _LENGTHS)
def test_compiled_memory_gate_exact_formula_zero_alias_and_limits(length):
    expected = 2_097_152 + length * 4100
    valid = {
        "available": True,
        "argument_size_in_bytes": expected,
        "output_size_in_bytes": 2_097_152,
        "alias_size_in_bytes": 0,
        "temp_size_in_bytes": 64 * 1024**2,
    }
    assert _PROBE._compiled_memory_gate(valid, length)["passed"] is True

    for key, value in (
        ("argument_size_in_bytes", expected - 1),
        ("output_size_in_bytes", 2_097_151),
        ("alias_size_in_bytes", 1),
        ("temp_size_in_bytes", 64 * 1024**2 + 1),
    ):
        corrupt = dict(valid)
        corrupt[key] = value
        assert _PROBE._compiled_memory_gate(corrupt, length)["passed"] is False


@pytest.mark.parametrize("length", (1024, 4096))
def test_host_oracle_last_chunk_global_inclusive_prefix_and_gqa_mapping(length):
    inputs, manifests, expected, oracle = _PROBE._construct_host_inputs(
        np, ml_dtypes, length
    )
    q, k, v, mask = inputs

    assert q.shape == expected.shape == (1, 256, 16, 256)
    assert k.shape == v.shape == (1, length, 4, 256)
    assert mask.shape == (1, length)
    assert np.count_nonzero(q) == np.count_nonzero(k) == 0
    assert np.all(mask == 1)
    assert not np.array_equal(v[:, 1:], v[:, :-1])
    assert not np.array_equal(v[:, :, 0], v[:, :, 1])
    assert not np.array_equal(v[:, :, :, 1:], v[:, :, :, :-1])
    v_fp32 = np.asarray(v, dtype=np.float32)
    for query_head in (0, 3, 4, 7, 12, 15):
        kv_head = query_head // 4
        for chunk_index, global_query in ((0, length - 256), (255, length - 1)):
            np.testing.assert_array_equal(
                expected[0, chunk_index, query_head],
                np.cumsum(
                    v_fp32[0, : global_query + 1, kv_head],
                    axis=0,
                    dtype=np.float32,
                )[-1]
                / np.float32(global_query + 1),
            )
    assert [item["name"] for item in manifests] == ["q_chunk", "k", "v", "key_mask"]
    assert oracle["shape"] == [1, 256, 16, 256]


def test_oracle_validation_rejects_exclusive_prefix_head_and_dimension_corruption():
    length = 1024
    inputs, _manifests, expected, _oracle = _PROBE._construct_host_inputs(
        np, ml_dtypes, length
    )
    v_fp32 = np.asarray(inputs[2], dtype=np.float32)
    query_start = length - 256
    exclusive = (
        np.cumsum(v_fp32, axis=1, dtype=np.float32)[:, query_start - 1 : length - 1]
        / np.arange(query_start, length, dtype=np.float32)[None, :, None, None]
    )
    query_to_kv = np.arange(16, dtype=np.int32) // 4
    corruptions = (
        exclusive[:, :, query_to_kv, :].astype(ml_dtypes.bfloat16),
        np.roll(expected, 4, axis=2).astype(ml_dtypes.bfloat16),
        np.roll(expected, 1, axis=-1).astype(ml_dtypes.bfloat16),
    )
    for candidate in corruptions:
        with pytest.raises(RuntimeError, match="host numerical"):
            _PROBE._validate_candidate(
                np,
                candidate,
                expected,
                0.001,
                _PROBE._completed_counters(),
                io.StringIO(),
                length,
            )


class _FakeCompiled:
    def __init__(self, result: Any = "result", *, error: BaseException | None = None):
        self.result = result
        self.error = error
        self.calls = 0

    def __call__(self, *arguments):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def test_checked_capability_is_private_one_shot_and_consumed_on_failure():
    counters = _PROBE._zero_counters()
    proof = {"passed": True}
    compiled = _FakeCompiled()
    with pytest.raises(RuntimeError, match="without both passed"):
        _PROBE._CheckedChunkExecutable(
            compiled, proof=proof, counters=counters, token=object()
        )

    checked = _PROBE._wrap_checked(compiled, proof, counters)
    fake_jax = SimpleNamespace(block_until_ready=lambda value: value)
    assert checked.invoke(fake_jax, (), lambda: None) == "result"
    with pytest.raises(RuntimeError, match="already consumed"):
        checked.invoke(fake_jax, (), lambda: None)
    assert compiled.calls == 1

    failing = _PROBE._wrap_checked(
        _FakeCompiled(error=RuntimeError("synthetic candidate failure")),
        proof,
        _PROBE._zero_counters(),
    )
    with pytest.raises(RuntimeError, match="synthetic candidate failure"):
        failing.invoke(fake_jax, (), lambda: None)
    with pytest.raises(RuntimeError, match="already consumed"):
        failing.invoke(fake_jax, (), lambda: None)


class _FakeShape:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _FakeMemoryStats:
    def __init__(self, length: int):
        self.argument_size_in_bytes = 2_097_152 + length * 4100
        self.output_size_in_bytes = 2_097_152
        self.alias_size_in_bytes = 0
        self.temp_size_in_bytes = 33_024


class _CompileFake:
    def __init__(self, length: int, state: dict[str, Any]):
        self.length = length
        self.state = state

    def as_text(self):
        self.state["as_text"] += 1
        return _optimized_hlo(self.length)

    def memory_analysis(self):
        self.state["memory"] += 1
        return _FakeMemoryStats(self.length)

    def __call__(self, *arguments):
        self.state["invocations"] += 1
        return "candidate"


class _LoweredFake:
    def __init__(self, length: int, state: dict[str, Any], compiled: Any):
        self.length = length
        self.state = state
        self.compiled = compiled

    def compiler_ir(self, *, dialect):
        assert dialect == "stablehlo"
        self.state["compiler_ir"] += 1
        return _stablehlo(self.length)

    def compile(self):
        self.state["compile"] += 1
        return self.compiled


class _JitFake:
    def __init__(self, function, state, lowered):
        self.function = function
        self.state = state
        self.lowered = lowered

    def lower(self, *signature):
        self.state["lower"] += 1
        self.state["signature"] = signature
        self.function(*signature)
        return self.lowered


class _JaxFake:
    ShapeDtypeStruct = _FakeShape

    def __init__(self, state, lowered):
        self.state = state
        self.lowered = lowered

    def jit(self, function):
        self.state["jit"] += 1
        return _JitFake(function, self.state, self.lowered)


@pytest.mark.parametrize("length", _LENGTHS)
def test_compile_gate_exact_shapes_offset_and_zero_invocations(length):
    state = {
        "jit": 0,
        "lower": 0,
        "compiler_ir": 0,
        "compile": 0,
        "as_text": 0,
        "memory": 0,
        "invocations": 0,
        "api": [],
    }
    compiled = _CompileFake(length, state)
    lowered = _LoweredFake(length, state, compiled)

    def api(*arguments, **keywords):
        state["api"].append((arguments, keywords))
        return "abstract"

    checked, report = _PROBE._compile_checked_chunk(
        _JaxFake(state, lowered),
        SimpleNamespace(bfloat16="bf16", int32="i32"),
        api,
        lambda: dict(_CLEAN),
        _PROBE._zero_counters(),
        io.StringIO(),
        length,
    )

    assert type(checked) is _PROBE._CheckedChunkExecutable
    assert state["jit"] == state["lower"] == state["compile"] == 1
    assert state["invocations"] == 0
    assert [item.shape for item in state["signature"]] == [
        (1, 256, 16, 256),
        (1, length, 4, 256),
        (1, length, 4, 256),
        (1, length),
    ]
    assert state["api"][0][1] == {
        "query_start": length - 256,
        "block_q": 64,
        "block_k": 64,
        "interpret": False,
    }
    assert report["release_gate"]["passed"] is True


@pytest.mark.parametrize("failure", ("structural", "memory"))
def test_compile_failure_never_releases_or_invokes(failure):
    length = 1024
    state = {
        "jit": 0,
        "lower": 0,
        "compiler_ir": 0,
        "compile": 0,
        "as_text": 0,
        "memory": 0,
        "invocations": 0,
        "api": [],
    }
    compiled = _CompileFake(length, state)
    if failure == "structural":
        compiled.as_text = lambda: _optimized_hlo(length, target="triton")
    else:
        compiled.memory_analysis = lambda: SimpleNamespace(
            argument_size_in_bytes=2_097_152 + length * 4100,
            output_size_in_bytes=2_097_152,
            alias_size_in_bytes=1,
            temp_size_in_bytes=0,
        )
    lowered = _LoweredFake(length, state, compiled)

    with pytest.raises(RuntimeError, match="structural or exact memory gate"):
        _PROBE._compile_checked_chunk(
            _JaxFake(state, lowered),
            SimpleNamespace(bfloat16="bf16", int32="i32"),
            lambda *args, **kwargs: "abstract",
            lambda: dict(_CLEAN),
            _PROBE._zero_counters(),
            io.StringIO(),
            length,
        )
    assert state["invocations"] == 0


@pytest.mark.parametrize("stage", ("lower", "compile"))
def test_lower_and_compile_exceptions_are_journaled_without_invocation(stage):
    length = 1024
    state = {
        "jit": 0,
        "lower": 0,
        "compiler_ir": 0,
        "compile": 0,
        "as_text": 0,
        "memory": 0,
        "invocations": 0,
        "api": [],
    }
    compiled = _CompileFake(length, state)
    lowered = _LoweredFake(length, state, compiled)
    jax = _JaxFake(state, lowered)
    if stage == "lower":

        class FailingJit:
            def lower(self, *signature):
                state["lower"] += 1
                raise RuntimeError("synthetic lower failure")

        jax.jit = lambda function: FailingJit()
    else:

        def fail_compile():
            state["compile"] += 1
            raise RuntimeError("synthetic compile failure")

        lowered.compile = fail_compile
    output = io.StringIO()

    with pytest.raises(RuntimeError, match=f"synthetic {stage} failure"):
        _PROBE._compile_checked_chunk(
            jax,
            SimpleNamespace(bfloat16="bf16", int32="i32"),
            lambda *args, **kwargs: "abstract",
            lambda: dict(_CLEAN),
            _PROBE._zero_counters(),
            output,
            length,
        )

    assert state["invocations"] == 0
    checkpoints = [
        record["stage"]
        for record in _records(output)
        if record["record_type"] == "journal_checkpoint"
    ]
    assert checkpoints == (
        ["after_chunk_lower_attempt"]
        if stage == "lower"
        else ["after_chunk_lower_attempt", "after_chunk_compile_attempt"]
    )


def test_static_source_binding_accepts_exact_audited_sources():
    proof = _PROBE._assert_static_source_bindings()

    assert proof == {
        "passed": True,
        "compile_helper_resolved_file_matches_expected": True,
        "compile_helper_source_sha256": (
            "24eeed83e93da1133d2e1bc3d0065bc8369d13fa324d2157e37db8b9c4a4d12d"
        ),
        "query_kernel_source_sha256": (
            "51e2fd91eb270f7b25ecdd117d7f06aa48a8e4af282a5a7e5e6b4c2a25dc52c9"
        ),
    }


def test_static_source_binding_rejects_wrong_compile_helper_path(monkeypatch):
    monkeypatch.setattr(
        _PROBE,
        "_compile_probe",
        lambda: SimpleNamespace(__file__="/tmp/not-the-audited-helper.py"),
    )

    with pytest.raises(RuntimeError, match="exact repository source"):
        _PROBE._assert_static_source_bindings()


@pytest.mark.parametrize("target", ("helper", "kernel"))
def test_static_source_binding_rejects_wrong_source_hash(monkeypatch, target):
    original = _PROBE._file_sha256
    helper_path = _PROBE._source_files()[
        "delegated_chunk_compile_probe_source_sha256"
    ].resolve()
    kernel_path = _PROBE._source_files()[
        "query_bounded_gqa_kernel_source_sha256"
    ].resolve()

    def corrupted(path):
        resolved = Path(path).resolve()
        if (target == "helper" and resolved == helper_path) or (
            target == "kernel" and resolved == kernel_path
        ):
            return "0" * 64
        return original(path)

    monkeypatch.setattr(_PROBE, "_file_sha256", corrupted)
    with pytest.raises(RuntimeError, match="source SHA256"):
        _PROBE._assert_static_source_bindings()


def test_kernel_binding_rejects_wrong_path_and_hash_and_accepts_real_api(monkeypatch):
    from skyrl.tx.kernels.query_bounded_gqa import query_bounded_gqa_forward_chunk

    proof = _PROBE._assert_kernel_binding(query_bounded_gqa_forward_chunk, 24576)
    assert proof["passed"] is True
    assert proof["exact_marker"] == "query_bounded_gqa_forward_q24320"
    assert (
        proof["source_sha256"]
        == hashlib.sha256(
            _PROBE._source_files()[
                "query_bounded_gqa_kernel_source_sha256"
            ].read_bytes()
        ).hexdigest()
    )

    with pytest.raises(RuntimeError, match="exact repository kernel source"):
        _PROBE._assert_kernel_binding(lambda: None, 24576)

    monkeypatch.setattr(_PROBE, "_file_sha256", lambda _path: "0" * 64)
    with pytest.raises(RuntimeError, match="source SHA256"):
        _PROBE._assert_kernel_binding(query_bounded_gqa_forward_chunk, 24576)


def test_ast_proves_lazy_accelerator_import_and_exact_one_shot_paths():
    source = _PROBE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    roots = {
        alias.name.partition(".")[0]
        for node in module.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert {"jax", "jaxlib", "numpy", "ml_dtypes", "skyrl"}.isdisjoint(roots)

    calls = [node for node in ast.walk(module) if isinstance(node, ast.Call)]
    attributes = [
        node.func.attr for node in calls if isinstance(node.func, ast.Attribute)
    ]
    executable_lowers = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "lower"
        and "jax.jit(forward_chunk)" in ast.unparse(node.func.value)
    ]
    executable_compiles = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "compile"
        and ast.unparse(node.func.value) == "lowered"
    ]
    assert len(executable_lowers) == 1
    assert len(executable_compiles) == 1
    assert attributes.count("device_put") == 1
    assert attributes.count("device_get") == 1
    assert attributes.count("invoke") == 1
    assert attributes.count("vjp") == 0
    assert attributes.count("grad") == 0
    assert "query_bounded_gqa_forward(" not in source

    run_source = inspect.getsource(_PROBE._run_rocm)
    assert run_source.index("_prove_command_buffers_disabled") < run_source.index(
        "import jax"
    )
    compile_source = inspect.getsource(_PROBE._compile_checked_chunk)
    assert compile_source.index("_structural_gate") < compile_source.index(
        "_wrap_checked"
    )
    assert compile_source.index("_compiled_memory_gate") < compile_source.index(
        "_wrap_checked"
    )
    assert (
        inspect.getsource(_PROBE._dispatch_candidate).count("executable.invoke(") == 1
    )


def test_default_subprocess_refuses_without_importing_jax():
    program = f"""
import contextlib, importlib.util, io, json, sys
before=set(sys.modules)
spec=importlib.util.spec_from_file_location('isolated_length_probe',{str(_PROBE_PATH)!r})
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
captured=io.StringIO()
with contextlib.redirect_stdout(captured):
    result=module.main([])
new=set(sys.modules)-before
bad=sorted(name for name in new if name in {{'jax','jaxlib'}} or name.startswith(('jax.','jaxlib.')))
print(json.dumps({{'result':result,'bad':bad,'records':[json.loads(line) for line in captured.getvalue().splitlines()]}}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(_REPO)
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["result"] == 0
    assert result["bad"] == []
    assert [item["record_type"] for item in result["records"]] == [
        "manifest",
        "refused",
    ]
