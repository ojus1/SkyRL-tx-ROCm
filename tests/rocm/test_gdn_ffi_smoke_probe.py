from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ml_dtypes
import numpy as np
import pytest

_REPO = Path(__file__).parents[2]
_PROBE_PATH = _REPO / "rocm" / "probe_gdn_ffi_smoke.py"
_SPEC = importlib.util.spec_from_file_location("probe_gdn_ffi_smoke_test", _PROBE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)

_TARGET = "skyrl_gdn_ffi_smoke_bf16_copy_v1"
_BASENAME = "libskyrl_gdn_ffi_smoke_gfx1100.so"
_CLEAN = {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}
_HEADLESS = {
    **_CLEAN,
    "amd_cards": ["card1"],
    "connected_amd_connectors": [],
    "kfd_path": "/dev/kfd",
    "kfd_accessible": True,
    "kfd_unowned": True,
}


def _records(output: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in output.getvalue().splitlines()]


def _accelerator_modules() -> set[str]:
    return {
        name
        for name in sys.modules
        if name in {"jax", "jaxlib", "skyrl.tx.kernels.rocm"}
        or name.startswith("jax.")
        or name.startswith("jaxlib.")
        or name.startswith("skyrl.tx.kernels.rocm.")
    }


def _abstract_args() -> SimpleNamespace:
    return SimpleNamespace(
        platform="abstract",
        allow_gpu=False,
        case=None,
        library=None,
        library_sha256=None,
        output=None,
    )


def _make_library(
    tmp_path: Path, payload: bytes = b"private-test-library"
) -> tuple[Path, str]:
    path = (tmp_path / _BASENAME).resolve()
    path.write_bytes(payload)
    path.chmod(0o600)
    return path, hashlib.sha256(payload).hexdigest()


def test_default_refusal_imports_no_accelerator_or_rocm_package_and_hashes_sources():
    before = _accelerator_modules()
    output = io.StringIO()

    assert _PROBE._execute(_abstract_args(), output) == 0

    assert _accelerator_modules() == before
    manifest, refused = _records(output)
    assert manifest["scope"] == "abstract_refusal"
    assert manifest["compile_may_dispatch_gpu_work"] is False
    assert manifest["outer_profile_rocm_supervision_required"] is True
    assert manifest["raw_library_path_emitted"] is False
    assert manifest["raw_ir_emitted"] is False
    assert manifest["counters"] == _PROBE._zero_counters()
    for name, path in _PROBE._source_files().items():
        assert manifest[name] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert refused["jax_imported"] is False
    assert refused["skyrl_rocm_package_imported"] is False
    assert refused["shared_library_loaded"] is False


@pytest.mark.parametrize(
    "arguments",
    [
        ["--platform", "rocm"],
        ["--platform", "rocm", "--allow-gpu"],
        ["--platform", "rocm", "--allow-gpu", "--case", "copy8"],
        ["--allow-gpu"],
        ["--case", "copy8"],
        ["--library", "/tmp/library.so"],
        ["--library-sha256", "0" * 64],
        ["--output", "/tmp/abstract.jsonl"],
        ["--platform", "rocm", "--allow-gpu", "--case", "wrong"],
        ["--platform", "rocm", "--allow-gpu", "--case", "copy8", "--backward"],
        ["--platform", "rocm", "--allow-gpu", "--case", "copy8", "--replay"],
    ],
)
def test_parser_rejects_incomplete_gpu_acknowledgement_and_scope_broadening(arguments):
    with pytest.raises(SystemExit) as raised:
        _PROBE._parse_args(arguments)
    assert raised.value.code == 2


@pytest.mark.parametrize(
    "digest",
    ["", "0" * 63, "0" * 65, "g" * 64, "A" * 64, "0x" + "0" * 62],
)
def test_library_sha_argument_requires_exact_canonical_64_hex(digest):
    with pytest.raises((SystemExit, ValueError)):
        _PROBE._parse_args(
            [
                "--platform",
                "rocm",
                "--allow-gpu",
                "--case",
                "copy8",
                "--library",
                "/tmp/" + _BASENAME,
                "--library-sha256",
                digest,
                "--output",
                "/tmp/out.jsonl",
            ]
        )


def test_complete_rocm_parser_contract(tmp_path):
    library, digest = _make_library(tmp_path)
    output = tmp_path / "private.jsonl"
    args = _PROBE._parse_args(
        [
            "--platform",
            "rocm",
            "--allow-gpu",
            "--case",
            "copy8",
            "--library",
            str(library),
            "--library-sha256",
            digest,
            "--output",
            str(output),
        ]
    )
    assert args.platform == "rocm"
    assert args.allow_gpu is True
    assert args.case == "copy8"
    assert args.library == library
    assert args.library_sha256 == digest
    assert args.output == output


def test_private_output_is_exclusive_mode_0600(tmp_path):
    path = tmp_path / "nested" / "result.jsonl"
    with _PROBE._open_exclusive_output(path) as output:
        output.write("{}\n")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        _PROBE._open_exclusive_output(path)
    link = tmp_path / "link.jsonl"
    link.symlink_to(path)
    with pytest.raises(FileExistsError):
        _PROBE._open_exclusive_output(link)


def test_contract_is_exact_one_shot_copy_and_outer_profile_limits():
    contract = _PROBE._exact_contract()
    assert contract["operation"] == "gdn_typed_ffi_distinct_output_copy_prerequisite"
    assert contract["case"] == "copy8"
    assert contract["target"] == _TARGET
    assert contract["input"] == {
        "shape": [1, 1024, 32, 128],
        "dtype": "bfloat16",
        "nbytes": 8 * 1024**2,
        "value": "deterministic_nonconstant_finite_host_grid",
    }
    assert contract["output"]["distinct_buffer_required"] is True
    assert contract["dispatch_plan"] == {
        "batched_device_put_calls": 1,
        "lower_calls": 1,
        "compile_calls": 1,
        "ffi_custom_calls_per_dialect": 1,
        "candidate_invocations": 1,
        "candidate_synchronizations": 1,
        "device_get_calls": 1,
        "replay_invocations": 0,
        "backward_invocations": 0,
        "model_invocations": 0,
    }
    assert contract["compiled_memory_gate"] == {
        "exact_argument_bytes": 8 * 1024**2,
        "exact_output_bytes": 8 * 1024**2,
        "exact_alias_bytes": 0,
        "maximum_temporary_bytes": 64 * 1024**2,
    }
    assert contract["duration_gate"] == {
        "candidate_seconds_strictly_below": 0.1,
        "promotion_seconds_strictly_below": 0.075,
    }
    assert contract["library_load_gate"] == {
        "source_path_is_dlopen_target": False,
        "exact_lowercase_sha256_required": True,
        "one_pass_hashing_copy_to_memfd": True,
        "private_snapshot_mode": 0o600,
        "required_snapshot_seals_mask": 0x000F,
        "snapshot_fd_retained_for_process_lifetime": True,
        "cdll_retained_for_process_lifetime": True,
    }
    assert not any(contract["scope_exclusions"].values())
    outer = contract["outer_supervision"]
    assert outer["timeout_seconds"] == 90
    assert outer["sensor_grace_seconds"] == 15
    assert outer["maximum_vram_bytes"] == 2 * 1024**3
    assert outer["maximum_junction_temperature_c"] == 70
    assert outer["maximum_gpu_power_watts"] == 200
    assert outer["minimum_host_available_bytes"] == 8 * 1024**3
    assert outer["maximum_swap_bytes"] == 0


def test_bound_source_hashes_are_exact_and_fail_closed(monkeypatch):
    proof = _PROBE._assert_bound_sources()
    assert proof == {
        "passed": True,
        "committed_sources_exact": True,
        "python_sha256": _PROBE._EXPECTED_WRAPPER_SHA256,
        "hip_sha256": _PROBE._EXPECTED_HIP_SHA256,
        "build_sha256": _PROBE._EXPECTED_BUILD_SHA256,
    }
    real_hash = _PROBE._file_sha256
    wrapper = _PROBE._source_files()["gdn_ffi_python_source_sha256"]
    monkeypatch.setattr(
        _PROBE,
        "_file_sha256",
        lambda path: "0" * 64 if path == wrapper else real_hash(path),
    )
    with pytest.raises(RuntimeError, match="source hash"):
        _PROBE._assert_bound_sources()


def test_library_path_hash_identity_and_permissions(tmp_path):
    library, digest = _make_library(tmp_path)
    manifest = _PROBE._validate_library_path(library, digest)
    assert manifest["validated"] is True
    assert manifest["canonical"] is True
    assert manifest["basename"] == _BASENAME
    assert manifest["sha256"] == digest
    assert manifest["raw_path_emitted"] is False
    assert (
        _PROBE._assert_same_library(library, manifest)["identity"]
        == manifest["identity"]
    )

    with pytest.raises(ValueError, match="absolute"):
        _PROBE._validate_library_path(Path(_BASENAME), digest)
    wrong = (tmp_path / "wrong.so").resolve()
    wrong.write_bytes(b"private-test-library")
    wrong.chmod(0o600)
    with pytest.raises(ValueError, match="exact audited"):
        _PROBE._validate_library_path(wrong, digest)
    with pytest.raises(RuntimeError, match="SHA-256"):
        _PROBE._validate_library_path(library, "0" * 64)


def test_library_path_rejects_symlink_and_writable_file(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    library, digest = _make_library(real_dir)
    link_dir = tmp_path / "link"
    link_dir.symlink_to(real_dir, target_is_directory=True)
    linked = link_dir / _BASENAME
    with pytest.raises(ValueError, match="canonical"):
        _PROBE._validate_library_path(linked.absolute(), digest)

    file_link = tmp_path / _BASENAME
    file_link.symlink_to(library)
    with pytest.raises(ValueError, match="regular"):
        _PROBE._validate_library_path(file_link.absolute(), digest)

    library.chmod(0o620)
    with pytest.raises(ValueError, match="group- or world-writable"):
        _PROBE._validate_library_path(library, digest)


def test_library_postcheck_detects_identity_change(monkeypatch, tmp_path):
    library, digest = _make_library(tmp_path, b"first")
    manifest = _PROBE._validate_library_path(library, digest)
    changed = {**manifest, "identity": (9, 8, 7, 6)}
    monkeypatch.setattr(_PROBE, "_validate_library_path", lambda *_args: changed)
    with pytest.raises(RuntimeError, match="identity changed"):
        _PROBE._assert_same_library(library, manifest)


def test_environment_requires_sole_command_buffer_disable(monkeypatch):
    names = {
        "JAX_PLATFORMS",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "JAX_ROCM_VISIBLE_DEVICES",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_CLIENT_MEM_FRACTION",
        "XLA_FLAGS",
        "HSA_OVERRIDE_GFX_VERSION",
        "XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB",
        "JAX_PJRT_CLIENT_CREATE_OPTIONS",
        "JAX_MOCK_GPU_TOPOLOGY",
        "TF_FORCE_UNIFIED_MEMORY",
        "MOCK_NUM_GPU_PROCESSES",
    }
    for name in names:
        monkeypatch.delenv(name, raising=False)
    environment = _PROBE._configure_rocm_environment()
    assert environment["XLA_FLAGS_effective"] == "--xla_gpu_enable_command_buffer="
    assert (
        _PROBE._environment_manifest(environment)["fixed_values_match_expected"] is True
    )
    assert _PROBE._prove_command_buffers_disabled(environment)["sole_xla_flag"] is True

    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_command_buffer= --other=1")
    with pytest.raises(RuntimeError, match="solely"):
        _PROBE._configure_rocm_environment()


def test_command_buffer_proof_rejects_environment_or_process_mutation(monkeypatch):
    environment = {"XLA_FLAGS_effective": "--xla_gpu_enable_command_buffer="}
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_command_buffer=true")
    with pytest.raises(RuntimeError, match="not proven"):
        _PROBE._prove_command_buffers_disabled(environment)


def _stablehlo(*, target: str = _TARGET, extra: str = "", alias: str = "") -> str:
    return "\n".join(
        [
            "module {",
            f'  %0 = stablehlo.custom_call @"{target}"(%arg0) {alias} : (tensor<1x1024x32x128xbf16>) -> tensor<1x1024x32x128xbf16>',
            extra,
            "  return %0 : tensor<1x1024x32x128xbf16>",
            "}",
        ]
    )


def _optimized(*, target: str = _TARGET, extra: str = "", alias: str = "") -> str:
    return "\n".join(
        [
            "ENTRY main {",
            f'  ROOT %copy = bf16[1,1024,32,128] custom-call(%arg0), custom_call_target="{target}"{alias}',
            extra,
            "}",
        ]
    )


def test_independent_ir_summaries_accept_exact_distinct_output_call():
    stable = _PROBE._ir_summary(_stablehlo(), "stablehlo")
    optimized = _PROBE._ir_summary(_optimized(), "optimized_hlo")
    assert stable["passed"] is True
    assert optimized["passed"] is True
    assert stable["calls"][0]["targets"] == [_TARGET]
    assert optimized["calls"][0]["targets"] == [_TARGET]
    assert _PROBE._structural_gate(stable, optimized)["passed"] is True
    assert stable["raw_ir_emitted"] is False
    assert optimized["raw_ir_emitted"] is False


def test_ir_summaries_allow_explicitly_empty_alias_collections():
    stable = _PROBE._ir_summary(
        _stablehlo(alias="{output_operand_aliases = []}"), "stablehlo"
    )
    optimized = _PROBE._ir_summary(
        _optimized(alias=", output_to_operand_aliasing={}"), "optimized_hlo"
    )
    assert stable["calls"][0]["alias_metadata_tokens_seen"] == [
        "output_operand_aliases"
    ]
    assert optimized["calls"][0]["alias_metadata_tokens_seen"] == [
        "output_to_operand_aliasing"
    ]
    assert stable["passed"] is True
    assert optimized["passed"] is True


@pytest.mark.parametrize(
    ("stable", "optimized"),
    [
        (_stablehlo(target="wrong"), _optimized()),
        (_stablehlo(), _optimized(target="wrong")),
        (
            _stablehlo(
                extra='  %1 = stablehlo.custom_call @"other"() : () -> tensor<i32>'
            ),
            _optimized(),
        ),
        (
            _stablehlo(),
            _optimized(
                extra='  %other = s32[] custom-call(), custom_call_target="other"'
            ),
        ),
        (_stablehlo(extra="  %1 = stablehlo.while(%0) : tensor<i32>"), _optimized()),
        (
            _stablehlo(),
            _optimized(
                extra="  %loop = s32[] while(%state), condition=cond, body=body"
            ),
        ),
        (
            _stablehlo(
                alias="{output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>]}"
            ),
            _optimized(),
        ),
        (
            "\n".join(
                [
                    "#alias = #stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>",
                    "module {",
                    f'  %0 = stablehlo.custom_call @"{_TARGET}"(%arg0) {{output_operand_aliases = [#alias]}} : (tensor<1x1024x32x128xbf16>) -> tensor<1x1024x32x128xbf16>',
                    "  return %0 : tensor<1x1024x32x128xbf16>",
                    "}",
                ]
            ),
            _optimized(),
        ),
        (_stablehlo(), _optimized(alias=", output_to_operand_aliasing={{}: (0, {})}")),
    ],
)
def test_ir_gate_rejects_other_calls_loops_targets_and_aliases(stable, optimized):
    summaries = (
        _PROBE._ir_summary(stable, "stablehlo"),
        _PROBE._ir_summary(optimized, "optimized_hlo"),
    )
    assert _PROBE._structural_gate(*summaries)["passed"] is False


def test_ir_parser_rejects_unparsed_textual_custom_call():
    generic = (
        'module { %0 = "stablehlo.custom_call"() {call_target_name = "'
        + _TARGET
        + '"} : () -> tensor<i32> }'
    )
    summary = _PROBE._ir_summary(generic, "stablehlo")
    assert summary["passed"] is False
    assert summary["custom_call_count"] == 0


def test_structural_gate_requires_exact_two_dialects():
    stable = _PROBE._ir_summary(_stablehlo(), "stablehlo")
    optimized = _PROBE._ir_summary(_optimized(), "optimized_hlo")
    assert _PROBE._structural_gate(stable)["passed"] is False
    assert _PROBE._structural_gate(stable, stable)["passed"] is False
    assert _PROBE._structural_gate(stable, optimized, optimized)["passed"] is False


def test_compiled_memory_gate_requires_exact_sizes_zero_alias_and_bounded_temp():
    exact = {
        "available": True,
        "argument_size_in_bytes": 8 * 1024**2,
        "output_size_in_bytes": 8 * 1024**2,
        "alias_size_in_bytes": 0,
        "temp_size_in_bytes": 64 * 1024**2,
    }
    assert _PROBE._compiled_memory_gate(exact)["passed"] is True
    for key, value in (
        ("argument_size_in_bytes", 8 * 1024**2 + 1),
        ("output_size_in_bytes", 8 * 1024**2 - 1),
        ("alias_size_in_bytes", 1),
        ("temp_size_in_bytes", 64 * 1024**2 + 1),
    ):
        corrupt = {**exact, key: value}
        assert _PROBE._compiled_memory_gate(corrupt)["passed"] is False
    missing = dict(exact)
    del missing["alias_size_in_bytes"]
    assert _PROBE._compiled_memory_gate(missing)["passed"] is False


class _MemoryStats:
    argument_size_in_bytes = 8 * 1024**2
    output_size_in_bytes = 8 * 1024**2
    alias_size_in_bytes = 0
    temp_size_in_bytes = 4096
    generated_code_size_in_bytes = 2048


class _FakeCompiled:
    def __init__(self, state: dict[str, Any], optimized: str | None = None):
        self.state = state
        self.optimized = _optimized() if optimized is None else optimized

    def __call__(self, value):
        self.state["compiled_invocations"] += 1
        return value

    def as_text(self):
        self.state["as_text_calls"] += 1
        return self.optimized

    def memory_analysis(self):
        self.state["memory_calls"] += 1
        return _MemoryStats()


class _FakeLowered:
    def __init__(
        self,
        state: dict[str, Any],
        compiled: Any,
        stable: str | None = None,
        compile_error: BaseException | None = None,
    ):
        self.state = state
        self.compiled = compiled
        self.stable = _stablehlo() if stable is None else stable
        self.compile_error = compile_error

    def compiler_ir(self, *, dialect):
        assert dialect == "stablehlo"
        self.state["compiler_ir_calls"] += 1
        return self.stable

    def compile(self):
        self.state["compile_calls"] += 1
        if self.compile_error is not None:
            raise self.compile_error
        return self.compiled


class _FakeJitted:
    def __init__(
        self,
        function: Any,
        state: dict[str, Any],
        lowered: Any,
        lower_error: BaseException | None = None,
    ):
        self.function = function
        self.state = state
        self.lowered = lowered
        self.lower_error = lower_error

    def lower(self, signature):
        self.state["lower_calls"] += 1
        self.state["signature"] = signature
        if self.lower_error is not None:
            raise self.lower_error
        self.function(signature)
        return self.lowered


class _FakeShapeDtypeStruct:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


class _FakeJax:
    ShapeDtypeStruct = _FakeShapeDtypeStruct

    def __init__(
        self,
        state: dict[str, Any],
        lowered: Any,
        lower_error: BaseException | None = None,
    ):
        self.state = state
        self.lowered = lowered
        self.lower_error = lower_error

    def jit(self, function):
        self.state["jit_calls"] += 1
        return _FakeJitted(function, self.state, self.lowered, self.lower_error)

    def block_until_ready(self, value):
        self.state["block_calls"] += 1
        return value

    def device_put(self, value):
        self.state["device_put_calls"] += 1
        return value

    def device_get(self, value):
        self.state["device_get_calls"] += 1
        return value


def _fake_state() -> dict[str, int]:
    return {
        "compiled_invocations": 0,
        "as_text_calls": 0,
        "memory_calls": 0,
        "compiler_ir_calls": 0,
        "compile_calls": 0,
        "lower_calls": 0,
        "jit_calls": 0,
        "block_calls": 0,
        "device_put_calls": 0,
        "device_get_calls": 0,
        "wrapper_calls": 0,
    }


def _fake_registration(path: Path, digest: str, size: int) -> SimpleNamespace:
    return SimpleNamespace(
        library_path=path,
        library_sha256=digest,
        snapshot_sha256=digest,
        snapshot_size_bytes=size,
        snapshot_mode=0o600,
        snapshot_seals=0x000F,
        sealed_snapshot=True,
        snapshot_fd_retained=True,
    )


def test_compile_checked_lowers_and_compiles_once_then_releases_private_capability(
    tmp_path,
):
    state = _fake_state()
    compiled = _FakeCompiled(state)
    lowered = _FakeLowered(state, compiled)
    jax = _FakeJax(state, lowered)
    counters = _PROBE._zero_counters()
    output = io.StringIO()
    journals: list[str] = []

    digest = "1" * 64
    library_size = 37

    def wrapper(value, *, enabled, library_path, library_sha256):
        state["wrapper_calls"] += 1
        assert enabled is True
        assert library_path == tmp_path
        assert library_sha256 == digest
        return value

    def register(path, *, enabled, library_sha256):
        assert path == tmp_path
        assert enabled is True
        assert library_sha256 == digest
        return _fake_registration(path, digest, library_size)

    def clean():
        return dict(_CLEAN)

    original = _PROBE._journal_checkpoint

    def journal(require_clean_boot, stream, stage, observed_counters):
        journals.append(stage)
        return original(require_clean_boot, stream, stage, observed_counters)

    _PROBE._journal_checkpoint = journal
    try:
        checked, report = _PROBE._compile_checked(
            jax,
            SimpleNamespace(bfloat16="bfloat16"),
            wrapper,
            register,
            tmp_path,
            digest,
            library_size,
            clean,
            counters,
            output,
        )
    finally:
        _PROBE._journal_checkpoint = original

    assert isinstance(checked, _PROBE._CheckedExecutable)
    assert report["release_gate"]["passed"] is True
    assert report["sealed_registration"]["passed"] is True
    assert report["sealed_registration"]["snapshot_sha256"] == digest
    assert state["jit_calls"] == 1
    assert state["lower_calls"] == 1
    assert state["compile_calls"] == 1
    assert state["wrapper_calls"] == 1
    assert state["compiled_invocations"] == 0
    assert counters["lower_attempts"] == counters["lower_completions"] == 1
    assert counters["compile_attempts"] == counters["compile_completions"] == 1
    assert counters["ffi_python_trace_calls"] == 1
    assert journals == ["after_ffi_lower_attempt", "after_ffi_compile_attempt"]


@pytest.mark.parametrize("phase", ["lower", "compile"])
def test_compile_failure_always_journals_risky_attempt(phase):
    state = _fake_state()
    compiled = _FakeCompiled(state)
    lowered = _FakeLowered(
        state,
        compiled,
        compile_error=RuntimeError("private compile failure")
        if phase == "compile"
        else None,
    )
    jax = _FakeJax(
        state,
        lowered,
        lower_error=RuntimeError("private lower failure") if phase == "lower" else None,
    )
    counters = _PROBE._zero_counters()
    output = io.StringIO()
    digest = "2" * 64
    library_size = 41
    with pytest.raises(RuntimeError, match="private"):
        _PROBE._compile_checked(
            jax,
            SimpleNamespace(bfloat16="bfloat16"),
            lambda value, **_kwargs: value,
            lambda path, **_kwargs: _fake_registration(path, digest, library_size),
            Path("/private/library"),
            digest,
            library_size,
            lambda: dict(_CLEAN),
            counters,
            output,
        )
    stages = [
        record["stage"]
        for record in _records(output)
        if record["record_type"] == "journal_checkpoint"
    ]
    assert "after_ffi_lower_attempt" in stages
    assert ("after_ffi_compile_attempt" in stages) is (phase == "compile")


def test_capability_rejects_unchecked_construction_and_is_one_shot():
    state = _fake_state()
    compiled = _FakeCompiled(state)
    counters = _PROBE._zero_counters()
    with pytest.raises(RuntimeError, match="without both passed"):
        _PROBE._CheckedExecutable(
            compiled,
            proof={"passed": True},
            counters=counters,
            token=object(),
        )
    with pytest.raises(RuntimeError, match="without both passed"):
        _PROBE._wrap_checked(compiled, {"passed": False}, counters)

    checked = _PROBE._wrap_checked(compiled, {"passed": True}, counters)
    jax = _FakeJax(state, None)
    assert checked.invoke(jax, "value", lambda: None) == "value"
    assert counters["candidate_attempts"] == 1
    assert counters["candidate_completions"] == 1
    assert counters["candidate_synchronizations"] == 1
    assert state["compiled_invocations"] == 1
    with pytest.raises(RuntimeError, match="already consumed"):
        checked.invoke(jax, "value", lambda: None)
    assert state["compiled_invocations"] == 1


def test_host_input_is_exact_deterministic_nonconstant_finite_bitwise_oracle():
    first, first_manifest = _PROBE._construct_host_input(np, ml_dtypes)
    second, second_manifest = _PROBE._construct_host_input(np, ml_dtypes)
    assert first.shape == (1, 1024, 32, 128)
    assert str(first.dtype) == "bfloat16"
    assert first.nbytes == 8 * 1024**2
    assert np.all(np.isfinite(np.asarray(first, dtype=np.float32)))
    assert np.unique(first.reshape(-1)[:8192]).size > 100
    assert first.tobytes() == second.tobytes()
    assert first_manifest == second_manifest
    assert first_manifest["sha256"] == hashlib.sha256(first.tobytes()).hexdigest()


def test_complete_host_oracle_accepts_only_all_bits_and_strict_durations():
    expected, _ = _PROBE._construct_host_input(np, ml_dtypes)
    counters = _PROBE._completed_counters()
    passed_output = io.StringIO()
    passed = _PROBE._validate_output(
        np, expected, expected.copy(), 0.001, counters, passed_output
    )
    assert passed["metrics"]["complete_bitwise_equal"] is True
    assert passed["metrics"]["complete_byte_count_compared"] == 8 * 1024**2
    assert passed["gates"]["promotion_passed"] is True

    corrupted = expected.copy()
    bits = corrupted.view(np.uint16).reshape(-1)
    bits[-1] ^= np.uint16(1)
    for actual, seconds in ((corrupted, 0.001), (expected, 0.075), (expected, 0.1)):
        output = io.StringIO()
        with pytest.raises(RuntimeError, match="exact oracle or duration"):
            _PROBE._validate_output(np, expected, actual, seconds, counters, output)
        assert _records(output)[0]["status"] == "failed"


def test_device_put_is_one_batched_singleton_and_device_get_is_one_call():
    state = _fake_state()
    jax = _FakeJax(state, None)
    counters = _PROBE._zero_counters()
    value = object()
    assert _PROBE._device_put(jax, value, counters) is value
    assert state["device_put_calls"] == 1
    assert counters["input_device_put_attempts"] == 1
    assert counters["input_device_put_completions"] == 1
    output = io.StringIO()
    assert (
        _PROBE._device_get(jax, value, lambda: dict(_CLEAN), counters, output) is value
    )
    assert state["device_get_calls"] == 1
    assert counters["device_get_attempts"] == 1
    assert counters["device_get_completions"] == 1
    assert _records(output)[0]["stage"] == "after_candidate_device_get_attempt"


def test_dispatch_failure_still_journals_and_consumes_capability():
    class FailingCompiled:
        def __call__(self, _value):
            raise RuntimeError("private candidate failure")

    state = _fake_state()
    jax = _FakeJax(state, None)
    counters = _PROBE._zero_counters()
    checked = _PROBE._wrap_checked(FailingCompiled(), {"passed": True}, counters)
    output = io.StringIO()
    with pytest.raises(RuntimeError, match="private candidate"):
        _PROBE._dispatch(
            jax,
            checked,
            object(),
            lambda: dict(_CLEAN),
            counters,
            output,
        )
    assert counters["candidate_attempts"] == 1
    assert counters["candidate_completions"] == 0
    assert _records(output)[-1]["stage"] == "after_candidate_dispatch_attempt"
    with pytest.raises(RuntimeError, match="already consumed"):
        checked.invoke(jax, object(), lambda: None)


def test_safety_preflight_requires_clean_headless_accessible_unowned_gpu():
    assert _PROBE._public_safety_preflight(dict(_HEADLESS))["kfd_unowned"] is True
    for mutation in (
        {"amdgpu_boot_clean": False},
        {"fatal_amdgpu_events": ["private"]},
        {"connected_amd_connectors": ["card1-DP-1"]},
        {"kfd_path": "/private/kfd"},
        {"kfd_accessible": False},
        {"kfd_unowned": False},
        {"amd_cards": []},
    ):
        with pytest.raises(RuntimeError):
            _PROBE._public_safety_preflight({**_HEADLESS, **mutation})


def test_execute_guarantees_postflight_and_redacts_runtime_error(monkeypatch, tmp_path):
    library, digest = _make_library(tmp_path)
    args = SimpleNamespace(
        platform="rocm",
        allow_gpu=True,
        case="copy8",
        library=library,
        library_sha256=digest,
        output=tmp_path / "result.jsonl",
    )
    postflight_calls = 0

    def clean():
        nonlocal postflight_calls
        postflight_calls += 1
        return dict(_CLEAN)

    @contextmanager
    def guarded():
        yield dict(_HEADLESS)

    monkeypatch.setattr(_PROBE, "_assert_fresh_accelerator_process", lambda: None)
    monkeypatch.setattr(_PROBE, "_assert_bound_sources", lambda: {"passed": True})
    monkeypatch.setattr(
        _PROBE,
        "_configure_rocm_environment",
        lambda: {"XLA_FLAGS_effective": _PROBE._COMMAND_BUFFER_FLAG},
    )
    monkeypatch.setattr(
        _PROBE, "_environment_manifest", lambda _environment: {"passed": True}
    )
    monkeypatch.setattr(_PROBE, "_load_safety_helpers", lambda: (guarded, clean))
    monkeypatch.setattr(
        _PROBE,
        "_run_rocm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("PRIVATE SECRET")),
    )
    output = io.StringIO()
    assert _PROBE._execute(args, output) == 1
    records = _records(output)
    assert postflight_calls == 1
    assert any(record["record_type"] == "safety_postflight" for record in records)
    error = records[-1]
    assert error["record_type"] == "error"
    assert error["message_redacted"] is True
    assert "PRIVATE SECRET" not in output.getvalue()


def test_backend_manifest_requires_exact_single_rocm_gpu():
    jaxlib = SimpleNamespace(__version__="private")

    class Backend:
        def __init__(self, version):
            self.platform_version = version

    good = SimpleNamespace(
        __version__="private",
        default_backend=lambda: "gpu",
        devices=lambda: [object()],
    )
    manifest = _PROBE._backend_manifest(
        good, jaxlib, SimpleNamespace(get_backend=lambda: Backend("ROCm 7.2"))
    )
    assert manifest["platform_family"] == "rocm"
    for backend, version, devices in (
        ("cpu", "ROCm 7.2", [object()]),
        ("gpu", "CUDA 13", [object()]),
        ("gpu", "ROCm 7.2", [object(), object()]),
    ):
        bad = SimpleNamespace(
            __version__="private",
            default_backend=lambda backend=backend: backend,
            devices=lambda devices=devices: devices,
        )
        with pytest.raises(RuntimeError, match="exactly one ROCm GPU"):
            _PROBE._backend_manifest(
                bad,
                jaxlib,
                SimpleNamespace(get_backend=lambda version=version: Backend(version)),
            )


def test_sealed_registration_manifest_proves_exact_hash_seals_mode_and_retention(
    tmp_path,
):
    path = (tmp_path / _BASENAME).resolve()
    digest = "3" * 64
    registration = _fake_registration(path, digest, 1234)
    manifest = _PROBE._sealed_registration_manifest(
        registration,
        library_path=path,
        library_sha256=digest,
        library_size_bytes=1234,
    )
    assert manifest["passed"] is True
    assert manifest["library_sha256"] == digest
    assert manifest["snapshot_sha256"] == digest
    assert manifest["snapshot_mode"] == 0o600
    assert manifest["snapshot_seals"] & 0x000F == 0x000F
    assert manifest["dlopen_source"] == "retained_sealed_memfd_snapshot_only"
    assert manifest["raw_library_path_emitted"] is False

    for field, value in (
        ("library_sha256", "0" * 64),
        ("snapshot_sha256", "0" * 64),
        ("snapshot_size_bytes", 1235),
        ("snapshot_mode", 0o644),
        ("snapshot_seals", 0x000E),
        ("sealed_snapshot", False),
        ("snapshot_fd_retained", False),
    ):
        corrupt = _fake_registration(path, digest, 1234)
        setattr(corrupt, field, value)
        with pytest.raises(RuntimeError, match="sealed snapshot identity"):
            _PROBE._sealed_registration_manifest(
                corrupt,
                library_path=path,
                library_sha256=digest,
                library_size_bytes=1234,
            )


def test_source_ast_keeps_abstract_branch_before_accelerator_helpers_and_no_broad_paths():
    source = _PROBE_PATH.read_text()
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module or "" for node in tree.body if isinstance(node, ast.ImportFrom)
    )
    assert "jax" not in imports
    assert "jaxlib" not in imports
    assert not any(name.startswith("skyrl") for name in imports)
    assert "jax.grad" not in source
    assert "custom_vjp" not in source
    assert "library_sha256=library_sha256" in source
    assert "retained_sealed_memfd_snapshot_only" in source
    assert "required_snapshot_seals_mask" in source
    assert 'replay_invocations": 0' in source
    assert 'backward_invocations": 0' in source
    assert 'model_invocations": 0' in source
    assert 'performance_claim_authorized": False' in source
    execute = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_execute"
    )
    abstract_return_index = next(
        index
        for index, node in enumerate(execute.body)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and any(
            isinstance(child, ast.Constant) and child.value == "abstract"
            for child in ast.walk(node.test)
        )
    )
    accelerator_call_indices = [
        index
        for index, node in enumerate(execute.body)
        if any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id
            in {"_assert_fresh_accelerator_process", "_load_safety_helpers"}
            for child in ast.walk(node)
        )
    ]
    assert accelerator_call_indices
    assert all(index > abstract_return_index for index in accelerator_call_indices)
