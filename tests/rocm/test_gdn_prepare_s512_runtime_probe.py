from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

_REPO = Path(__file__).parents[2]
_PROBE_PATH = _REPO / "rocm" / "probe_gdn_prepare_s512_runtime.py"
_SPEC = importlib.util.spec_from_file_location(
    "probe_gdn_prepare_s512_runtime_test", _PROBE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)

_TARGET = "skyrl_gdn_prepare_s512_f32_v1"
_BASENAME = "libskyrl_gdn_prepare_s512_gfx1100.so"
_CLEAN = {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}


def _records(output: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in output.getvalue().splitlines()]


def _abstract_args() -> SimpleNamespace:
    return SimpleNamespace(
        platform="abstract",
        allow_gpu=False,
        case=None,
        library=None,
        library_sha256=None,
        output=None,
    )


def _make_library(tmp_path: Path, payload: bytes = b"runtime-test") -> tuple[Path, str]:
    path = (tmp_path / _BASENAME).resolve()
    path.write_bytes(payload)
    path.chmod(0o600)
    return path, hashlib.sha256(payload).hexdigest()


def _stablehlo(*, target: str = _TARGET, malformed: bool = False) -> str:
    key_shape = "1x512x32x128" if malformed else "1x512x16x128"
    layouts = (
        "operand_layouts = [dense<[3,2,1,0]> : tensor<4xindex>, "
        "dense<[3,2,1,0]> : tensor<4xindex>, "
        "dense<[2,1,0]> : tensor<3xindex>, "
        "dense<[2,1,0]> : tensor<3xindex>], "
        "result_layouts = [dense<[3,2,1,0]> : tensor<4xindex>, "
        "dense<[3,2,1,0]> : tensor<4xindex>, "
        "dense<[2,1,0]> : tensor<3xindex>]"
    )
    return "\n".join(
        [
            "module {",
            f'  %0:3 = stablehlo.custom_call @"{target}"(%arg0, %arg1, %arg2, %arg3) '
            f"{{{layouts}}} : (tensor<{key_shape}xf32>, tensor<1x512x32x128xf32>, "
            "tensor<1x512x32xf32>, tensor<1x512x32xf32>) -> "
            "tuple<tensor<1x512x32x128xf32>, tensor<1x512x32x128xf32>, "
            "tensor<1x512x32xf32>>",
            "  return %0 : tuple<tensor<1x512x32x128xf32>, "
            "tensor<1x512x32x128xf32>, tensor<1x512x32xf32>>",
            "}",
        ]
    )


def _optimized(*, target: str = _TARGET) -> str:
    return "\n".join(
        [
            "ENTRY main {",
            "  %key = f32[1,512,16,128]{3,2,1,0} parameter(0)",
            "  %value = f32[1,512,32,128]{3,2,1,0} parameter(1)",
            "  %g = f32[1,512,32]{2,1,0} parameter(2)",
            "  %beta = f32[1,512,32]{2,1,0} parameter(3)",
            "  ROOT %prepare = (f32[1,512,32,128]{3,2,1,0}, "
            "f32[1,512,32,128]{3,2,1,0}, f32[1,512,32]{2,1,0}) "
            "custom-call(%key, %value, %g, %beta), "
            f'custom_call_target="{target}"',
            "}",
        ]
    )


class _FakeCompiled:
    def __init__(
        self,
        oracle: Any,
        *,
        optimized: str | None = None,
        temporary_bytes: int = 0,
    ) -> None:
        self.oracle = oracle
        self.optimized = optimized or _optimized()
        self.temporary_bytes = temporary_bytes
        self.invocations = 0

    def as_text(self) -> str:
        return self.optimized

    def memory_analysis(self) -> SimpleNamespace:
        return SimpleNamespace(
            argument_size_in_bytes=12_713_984,
            output_size_in_bytes=16_842_776,
            alias_size_in_bytes=0,
            temp_size_in_bytes=self.temporary_bytes,
            generated_code_size_in_bytes=4096,
        )

    def __call__(self, *inputs: Any) -> tuple[Any, Any, Any]:
        self.invocations += 1
        return self.oracle(*inputs)


class _FakeLowered:
    def __init__(
        self,
        compiled: _FakeCompiled,
        *,
        stable: str | None = None,
    ) -> None:
        self.compiled = compiled
        self.stable = stable or _stablehlo()
        self.compile_calls = 0

    def compiler_ir(self, *, dialect: str) -> str:
        assert dialect == "stablehlo"
        return self.stable

    def compile(self) -> _FakeCompiled:
        self.compile_calls += 1
        return self.compiled


class _FakeJit:
    def __init__(self, function: Any, lowered: _FakeLowered) -> None:
        self.function = function
        self.lowered = lowered

    def lower(self, *signatures: Any) -> _FakeLowered:
        self.function(*signatures)
        return self.lowered


class _FakeJax:
    __version__ = "test-jax"

    def __init__(self, lowered: _FakeLowered) -> None:
        self.lowered = lowered
        self.device_put_calls = 0
        self.device_get_calls = 0
        self.block_calls = 0

    @staticmethod
    def ShapeDtypeStruct(shape: tuple[int, ...], dtype: Any) -> SimpleNamespace:
        return SimpleNamespace(shape=shape, dtype=dtype)

    def jit(self, function: Any) -> _FakeJit:
        return _FakeJit(function, self.lowered)

    def device_put(self, value: Any) -> Any:
        self.device_put_calls += 1
        return value

    def device_get(self, value: Any) -> Any:
        self.device_get_calls += 1
        return value

    def block_until_ready(self, value: Any) -> Any:
        self.block_calls += 1
        return value

    @staticmethod
    def default_backend() -> str:
        return "gpu"

    @staticmethod
    def devices() -> list[SimpleNamespace]:
        return [SimpleNamespace()]


def _registration(path: Path, digest: str, size: int) -> SimpleNamespace:
    return SimpleNamespace(
        library_path=path,
        library_sha256=digest,
        snapshot_sha256=digest,
        snapshot_size_bytes=size,
        snapshot_mode=0o600,
        snapshot_seals=0xF,
        sealed_snapshot=True,
        snapshot_fd_retained=True,
        target_name=_TARGET,
        platform="ROCM",
        registration_api_version=1,
        custom_call_api_version=4,
    )


@pytest.fixture(scope="module")
def host_reference() -> tuple[Any, Any, Any, Any]:
    oracle_module = _PROBE._load_oracle_module()
    host, host_report = _PROBE._construct_host_case(np)
    counters = _PROBE._zero_counters()
    reference, reference_report, equations = _PROBE._construct_reference(
        np, oracle_module.gdn_prepare_s512_numpy, host, counters
    )
    return host, host_report, reference, reference_report, equations


def test_default_subprocess_refuses_without_numpy_jax_skyrl_or_library_import():
    environment = os.environ.copy()
    for name in (
        "JAX_PLATFORMS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        environment.pop(name, None)
    snippet = (
        "import runpy,sys; "
        f"sys.argv=[{str(_PROBE_PATH)!r}]; "
        f"runpy.run_path({str(_PROBE_PATH)!r},run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", snippet],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    manifest, refused = [json.loads(line) for line in result.stdout.splitlines()]
    assert manifest["scope"] == "abstract_refusal"
    assert manifest["counters"] == _PROBE._zero_counters()
    assert refused["jax_imported"] is False
    assert refused["numpy_imported"] is False
    assert refused["skyrl_rocm_package_imported"] is False
    assert refused["shared_library_loaded"] is False


def test_abstract_execute_reports_every_source_hash_without_accelerator_work():
    output = io.StringIO()
    assert _PROBE._execute(_abstract_args(), output) == 0
    manifest, refused = _records(output)
    for name, path in _PROBE._source_files().items():
        assert manifest[name] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert refused["counters"] == _PROBE._zero_counters()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--platform", "rocm"],
        ["--platform", "rocm", "--allow-gpu"],
        ["--platform", "rocm", "--allow-gpu", "--case", _PROBE._CASE],
        ["--allow-gpu"],
        ["--case", _PROBE._CASE],
        ["--library", "/tmp/" + _BASENAME],
        ["--library-sha256", "0" * 64],
        ["--output", "/tmp/runtime.jsonl"],
        ["--platform", "rocm", "--allow-gpu", "--case", "wrong"],
        ["--platform", "rocm", "--allow-gpu", "--backward"],
        ["--platform", "rocm", "--allow-gpu", "--replay"],
        ["--platform", "rocm", "--allow-gpu", "--warmup"],
    ],
)
def test_parser_rejects_incomplete_acknowledgement_and_scope_broadening(arguments):
    with pytest.raises(SystemExit) as raised:
        _PROBE._parse_args(arguments)
    assert raised.value.code == 2


@pytest.mark.parametrize("digest", ["", "0" * 63, "0" * 65, "A" * 64, "g" * 64])
def test_parser_requires_canonical_lowercase_library_digest(digest):
    with pytest.raises(SystemExit):
        _PROBE._parse_args(
            [
                "--platform",
                "rocm",
                "--allow-gpu",
                "--case",
                _PROBE._CASE,
                "--library",
                "/tmp/" + _BASENAME,
                "--library-sha256",
                digest,
                "--output",
                "/tmp/runtime.jsonl",
            ]
        )


def test_contract_pins_compile_memory_invocation_duration_and_outer_caps():
    contract = _PROBE._exact_contract()
    assert contract["compile_gate"] == {
        "fresh_register_lower_compile": True,
        "stablehlo_precompile_gate": True,
        "optimized_hlo_gate": True,
        "exact_custom_calls": 1,
        "while_calls": 0,
        "alias_bytes": 0,
        "argument_bytes": 12_713_984,
        "logical_output_bytes": 16_842_752,
        "tuple_pointer_bytes": 24,
        "compiler_output_bytes": 16_842_776,
        "temporary_bytes_expected": 0,
        "temporary_bytes_hard_maximum": 64 * 1024**2,
        "combined_bytes_hard_maximum": 96 * 1024**2,
    }
    assert contract["invocation_contract"]["checked_executable_invocations"] == 1
    assert contract["invocation_contract"]["warmups"] == 0
    assert contract["invocation_contract"]["replays"] == 0
    assert contract["duration_gate"] == {
        "promotion_seconds_strictly_below": 0.25,
        "hard_seconds_strictly_below": 2.0,
    }
    assert contract["outer_supervision"]["maximum_gpu_power_watts"] == 315
    assert contract["outer_supervision"]["maximum_junction_temperature_c"] == 90
    assert contract["outer_supervision"]["maximum_swap_bytes"] == 0


def test_bound_sources_cover_every_executable_dependency():
    result = _PROBE._assert_bound_sources()
    assert result["passed"] is True
    assert result["all_executable_dependencies_exact"] is True
    assert result["compile_helper"] == _PROBE._EXPECTED_COMPILE_HELPER_SHA256
    assert result["oracle"] == _PROBE._EXPECTED_ORACLE_SHA256
    assert result["wrapper"] == _PROBE._EXPECTED_WRAPPER_SHA256
    assert result["hip"] == _PROBE._EXPECTED_HIP_SHA256
    assert result["sealed_loader"] == _PROBE._EXPECTED_SEALED_LOADER_SHA256
    assert result["safety"] == _PROBE._EXPECTED_SAFETY_SHA256


def test_exact_file_loader_rejects_hash_mismatch(tmp_path):
    changed = tmp_path / "changed.py"
    changed.write_text("VALUE = 1\n")
    with pytest.raises(RuntimeError, match="changed source"):
        _PROBE._load_exact_file_module(changed, "0" * 64, "changed")


def test_numeric_environment_requires_all_three_thread_caps(monkeypatch):
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    assert _PROBE._validate_host_numeric_environment() == {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(RuntimeError, match="thread caps exactly 1"):
        _PROBE._validate_host_numeric_environment()


def test_splitmix_and_host_case_reproduce_all_published_hashes(host_reference):
    host, report, _reference, _reference_report, _equations = host_reference
    assert report["individual_sha256"] == _PROBE._EXPECTED_INPUT_SHA256
    assert report["framed_tuple_sha256"] == _PROBE._EXPECTED_INPUT_TUPLE_SHA256
    assert report["checks"]["individual_hashes_exact"] is True
    assert report["checks"]["framed_tuple_hash_exact"] is True
    assert report["invariants"]["key_norm_min"] >= 0.9999998
    assert report["invariants"]["key_norm_max"] <= 1.0000002
    assert host[0].shape == (1, 512, 16, 128)
    assert host[1].shape == (1, 512, 32, 128)


def test_explicit_tuple_framing_is_order_name_shape_and_payload_sensitive():
    first = np.arange(12, dtype=np.float32).reshape(3, 4)
    second = np.arange(5, dtype=np.float32)
    baseline = _PROBE._framed_tuple_sha256((("first", first), ("second", second)))
    assert baseline != _PROBE._framed_tuple_sha256(
        (("second", second), ("first", first))
    )
    assert baseline != _PROBE._framed_tuple_sha256(
        (("renamed", first), ("second", second))
    )
    assert baseline != _PROBE._framed_tuple_sha256(
        (("first", first.reshape(2, 6)), ("second", second))
    )
    changed = first.copy()
    changed[0, 0] = 1
    assert baseline != _PROBE._framed_tuple_sha256(
        (("first", changed), ("second", second))
    )


def test_dense_reference_reproduces_hashes_and_all_sensitivity_calibrations(
    host_reference,
):
    _host, _host_report, _reference, report, _equations = host_reference
    sensitivity = report["sensitivities"]
    assert report["individual_sha256"] == _PROBE._EXPECTED_REFERENCE_SHA256
    assert report["framed_tuple_sha256"] == _PROBE._EXPECTED_REFERENCE_TUPLE_SHA256
    assert sensitivity["condition_min"] == pytest.approx(1.3251028, abs=2e-6)
    assert sensitivity["condition_max"] == pytest.approx(1.4126751, abs=2e-6)
    assert sensitivity["identity_u_relative_l2"] == pytest.approx(0.1070796, abs=2e-7)
    assert sensitivity["identity_w_relative_l2"] == pytest.approx(0.1234961, abs=2e-7)
    assert sensitivity["missing_decay_strict_lower_relative_l2"] == pytest.approx(
        0.1361953, abs=2e-7
    )
    assert sensitivity["wrong_global_gamma_relative_l2"] == pytest.approx(
        1.3953155, abs=2e-6
    )
    assert sensitivity["reference_u_equation_residual"]["relative_l2"] < 3e-8
    assert sensitivity["reference_w_equation_residual"]["relative_l2"] < 3e-8
    assert all(report["checks"].values())


def test_reference_constructor_fails_before_release_on_wrong_oracle(host_reference):
    host = host_reference[0]

    def identity_oracle(_key, value, _g, beta):
        return value.copy(), value.copy(), beta.copy()

    with pytest.raises(RuntimeError, match="reference failed"):
        _PROBE._construct_reference(np, identity_oracle, host, _PROBE._zero_counters())


def test_numerical_validation_accepts_exact_outputs_and_reports_raw_cosine(
    host_reference,
):
    _host, _host_report, reference, _reference_report, equations = host_reference
    result = _PROBE._validate_actual(np, reference, reference, equations, 0.249)
    assert result["passed"] is True
    assert result["promotion_passed"] is True
    assert result["classification"] == "promotable"
    assert (
        result["actual_framed_tuple_sha256"] == _PROBE._EXPECTED_REFERENCE_TUPLE_SHA256
    )
    for metrics in result["metrics"].values():
        assert metrics["relative_l2"] == 0
        assert metrics["cosine"] == 1
        assert metrics["cosine_raw"] == pytest.approx(1, abs=3e-15)


def test_numerical_validation_classifies_completed_slow_result_unpromotable(
    host_reference,
):
    _host, _host_report, reference, _reference_report, equations = host_reference
    result = _PROBE._validate_actual(np, reference, reference, equations, 0.25)
    assert result["passed"] is True
    assert result["promotion_passed"] is False
    assert result["classification"] == "completed_unpromotable"


@pytest.mark.parametrize("seconds", [2.0, float("inf"), float("nan"), -0.1])
def test_numerical_validation_rejects_hard_or_invalid_duration(host_reference, seconds):
    _host, _host_report, reference, _reference_report, equations = host_reference
    with pytest.raises(RuntimeError, match="hard-duration"):
        _PROBE._validate_actual(np, reference, reference, equations, seconds)


@pytest.mark.parametrize("output_index", [0, 1, 2])
def test_numerical_validation_rejects_each_corrupted_output(
    host_reference, output_index
):
    _host, _host_report, reference, _reference_report, equations = host_reference
    corrupt = [item.copy() for item in reference]
    corrupt[output_index].reshape(-1)[-1] += np.float32(0.01)
    with pytest.raises(RuntimeError, match="numerical"):
        _PROBE._validate_actual(np, reference, tuple(corrupt), equations, 0.1)


def test_numerical_validation_rejects_wrong_shape_and_dtype(host_reference):
    _host, _host_report, reference, _reference_report, equations = host_reference
    wrong_shape = (reference[0][:, :-1], reference[1], reference[2])
    with pytest.raises((RuntimeError, ValueError)):
        _PROBE._validate_actual(np, reference, wrong_shape, equations, 0.1)
    wrong_dtype = (reference[0].astype(np.float64), reference[1], reference[2])
    with pytest.raises(RuntimeError, match="numerical"):
        _PROBE._validate_actual(np, reference, wrong_dtype, equations, 0.1)


def test_checked_capability_requires_both_proofs_and_is_one_shot(host_reference):
    host, _host_report, reference, _reference_report, _equations = host_reference
    counters = _PROBE._zero_counters()

    def compiled(*_inputs):
        return reference

    with pytest.raises(RuntimeError, match="unchecked"):
        _PROBE._release_checked(compiled, {"passed": False}, {"passed": True}, counters)
    executable = _PROBE._release_checked(
        compiled, {"passed": True}, {"passed": True}, counters
    )
    jax = SimpleNamespace(block_until_ready=lambda value: value)
    assert executable.invoke(jax, host, lambda: None) is reference
    assert counters["checked_executable_attempts"] == 1
    assert counters["checked_executable_completions"] == 1
    assert counters["block_until_ready_calls"] == 1
    with pytest.raises(RuntimeError, match="already consumed"):
        executable.invoke(jax, host, lambda: None)


def test_checked_completion_is_not_incremented_when_readiness_fails(host_reference):
    host, _host_report, reference, _reference_report, _equations = host_reference
    counters = _PROBE._zero_counters()
    executable = _PROBE._release_checked(
        lambda *_inputs: reference,
        {"passed": True},
        {"passed": True},
        counters,
    )

    def fail_ready(_value):
        raise RuntimeError("readiness failed")

    with pytest.raises(RuntimeError, match="readiness failed"):
        executable.invoke(
            SimpleNamespace(block_until_ready=fail_ready), host, lambda: None
        )
    assert counters["checked_executable_attempts"] == 1
    assert counters["checked_executable_completions"] == 0
    assert counters["block_until_ready_calls"] == 0


def test_tuple_transfer_helpers_count_one_call_and_exact_leaf_counts(host_reference):
    host, _host_report, reference, _reference_report, _equations = host_reference
    jax = SimpleNamespace(
        device_put=lambda value: value,
        block_until_ready=lambda value: value,
        device_get=lambda value: value,
    )
    counters = _PROBE._zero_counters()
    placed = _PROBE._tuple_device_put(jax, host, counters)
    received = _PROBE._tuple_device_get(jax, reference, counters)
    assert placed is host
    assert received is reference
    assert counters["tuple_device_put_completions"] == 1
    assert counters["device_put_leaves"] == 4
    assert counters["block_until_ready_calls"] == 1
    assert counters["tuple_device_get_completions"] == 1
    assert counters["device_get_leaves"] == 3


def test_compile_gate_reuses_hardened_parser_and_never_invokes_executable(
    tmp_path, host_reference
):
    path, digest = _make_library(tmp_path)
    host = host_reference[0]
    oracle = _PROBE._load_oracle_module().gdn_prepare_s512_numpy
    compiled = _FakeCompiled(oracle)
    lowered = _FakeLowered(compiled)
    jax = _FakeJax(lowered)
    helper = _PROBE._load_compile_helper()
    counters = _PROBE._zero_counters()
    observed, report = _PROBE._compile_unreleased(
        jax,
        SimpleNamespace(float32=np.float32),
        lambda _k, value, g, _beta, **_kwargs: (value, value, g),
        lambda *_args, **_kwargs: _registration(path, digest, path.stat().st_size),
        helper,
        path,
        digest,
        path.stat().st_size,
        lambda: dict(_CLEAN),
        counters,
        io.StringIO(),
    )
    assert observed is compiled
    assert report["release_gate"]["passed"] is True
    assert report["temporary_memory_expectation"]["observed_temporary_bytes"] == 0
    assert compiled.invocations == 0
    assert lowered.compile_calls == 1
    assert counters["ffi_python_trace_calls"] == 1
    assert counters["shape_dtype_structs"] == 4
    assert host[0].shape == _PROBE._KEY_SHAPE


def test_failed_stablehlo_gate_never_compiles_or_releases(tmp_path):
    path, digest = _make_library(tmp_path)
    oracle = _PROBE._load_oracle_module().gdn_prepare_s512_numpy
    compiled = _FakeCompiled(oracle)
    lowered = _FakeLowered(compiled, stable=_stablehlo(malformed=True))
    counters = _PROBE._zero_counters()
    with pytest.raises(RuntimeError, match="StableHLO failed"):
        _PROBE._compile_unreleased(
            _FakeJax(lowered),
            SimpleNamespace(float32=np.float32),
            lambda _k, value, g, _beta, **_kwargs: (value, value, g),
            lambda *_args, **_kwargs: _registration(path, digest, path.stat().st_size),
            _PROBE._load_compile_helper(),
            path,
            digest,
            path.stat().st_size,
            lambda: dict(_CLEAN),
            counters,
            io.StringIO(),
        )
    assert lowered.compile_calls == 0
    assert compiled.invocations == 0
    assert counters["compile_attempts"] == 0


def test_nonzero_temporary_memory_fails_expected_zero_release_gate(tmp_path):
    path, digest = _make_library(tmp_path)
    oracle = _PROBE._load_oracle_module().gdn_prepare_s512_numpy
    compiled = _FakeCompiled(oracle, temporary_bytes=1)
    lowered = _FakeLowered(compiled)
    with pytest.raises(RuntimeError, match="release gate"):
        _PROBE._compile_unreleased(
            _FakeJax(lowered),
            SimpleNamespace(float32=np.float32),
            lambda _k, value, g, _beta, **_kwargs: (value, value, g),
            lambda *_args, **_kwargs: _registration(path, digest, path.stat().st_size),
            _PROBE._load_compile_helper(),
            path,
            digest,
            path.stat().st_size,
            lambda: dict(_CLEAN),
            _PROBE._zero_counters(),
            io.StringIO(),
        )
    assert compiled.invocations == 0


def test_mock_end_to_end_has_exact_counters_one_invocation_and_postcheck(
    tmp_path, monkeypatch
):
    path, digest = _make_library(tmp_path)
    oracle = _PROBE._load_oracle_module().gdn_prepare_s512_numpy
    compiled = _FakeCompiled(oracle)
    lowered = _FakeLowered(compiled)
    jax = _FakeJax(lowered)
    helper = _PROBE._load_compile_helper()
    backend = SimpleNamespace(
        get_backend=lambda: SimpleNamespace(platform_version="ROCm test")
    )
    dependencies = (
        jax,
        SimpleNamespace(float32=np.float32),
        SimpleNamespace(__version__="test-jaxlib"),
        backend,
        np,
        lambda _k, value, g, _beta, **_kwargs: (value, value, g),
        lambda *_args, **_kwargs: _registration(path, digest, path.stat().st_size),
        oracle,
    )
    counters = _PROBE._zero_counters()
    output = io.StringIO()
    monkeypatch.setenv("XLA_FLAGS", _PROBE._COMMAND_BUFFER_FLAG)
    args = SimpleNamespace(library=path, library_sha256=digest)
    manifest = helper._validate_library_path(path, digest)
    assert (
        _PROBE._run_rocm(
            args,
            output,
            lambda: dict(_CLEAN),
            counters,
            environment={"XLA_FLAGS_effective": _PROBE._COMMAND_BUFFER_FLAG},
            library_manifest=manifest,
            helper=helper,
            _dependencies=dependencies,
        )
        == 0
    )
    assert counters == _PROBE._completed_counters()
    assert compiled.invocations == 1
    assert jax.device_put_calls == 1
    assert jax.device_get_calls == 1
    assert jax.block_calls == 2
    records = _records(output)
    assert (
        sum(item["record_type"] == "checked_executable_released" for item in records)
        == 1
    )
    assert sum(item["record_type"] == "dispatch_started" for item in records) == 1
    assert records[-1]["record_type"] == "runtime_passed"
    journals = [
        item["stage"] for item in records if item["record_type"] == "journal_checkpoint"
    ]
    assert journals[-1] == "after_library_postcheck"


def test_library_postcheck_and_journal_are_unconditional_on_early_failure():
    events: list[str] = []

    def fail_command_buffer(_environment):
        events.append("early_failure")
        raise RuntimeError("command buffer proof failed")

    def postcheck(path, manifest):
        events.append("postcheck")
        assert path == Path("/private/library.so")
        assert manifest["sha256"] == "a" * 64
        return {"validated": True}

    helper = SimpleNamespace(
        _prove_command_buffers_disabled=fail_command_buffer,
        _assert_same_library=postcheck,
    )
    output = io.StringIO()
    with pytest.raises(RuntimeError, match="command buffer proof failed"):
        _PROBE._run_rocm(
            SimpleNamespace(
                library=Path("/private/library.so"), library_sha256="a" * 64
            ),
            output,
            lambda: dict(_CLEAN),
            _PROBE._zero_counters(),
            environment={},
            library_manifest={"sha256": "a" * 64},
            helper=helper,
        )
    assert events == ["early_failure", "postcheck"]
    assert [
        record["stage"]
        for record in _records(output)
        if record["record_type"] == "journal_checkpoint"
    ] == ["after_library_postcheck"]


def test_ast_orders_binding_before_environment_and_host_gate_before_device_put():
    tree = ast.parse(_PROBE_PATH.read_text())
    execute = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_execute"
    )
    run = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_rocm_body"
    )
    source = ast.unparse(execute)
    assert source.index("_assert_bound_sources()") < source.index(
        "_configure_rocm_environment()"
    )
    assert source.index("_assert_bound_sources()") < source.index(
        "_load_safety_helpers()"
    )
    run_source = ast.unparse(run)
    assert run_source.index("_construct_host_case(np)") < run_source.index(
        "_tuple_device_put"
    )
    assert run_source.index("_construct_reference") < run_source.index(
        "_release_checked"
    )
    assert run_source.index("_release_checked") < run_source.index("_tuple_device_put")


def test_ast_has_only_one_compiled_invocation_site_and_no_forbidden_gpu_paths():
    tree = ast.parse(_PROBE_PATH.read_text())
    checked = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_CheckedExecutable"
    )
    invoke = next(
        node
        for node in checked.body
        if isinstance(node, ast.FunctionDef) and node.name == "invoke"
    )
    compiled_calls = [
        node
        for node in ast.walk(invoke)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_compiled"
    ]
    assert len(compiled_calls) == 1
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert {"grad", "vjp", "value_and_grad", "pmap"}.isdisjoint(names)


def test_journal_and_safety_helpers_fail_closed():
    with pytest.raises(RuntimeError, match="undeclared"):
        _PROBE._journal_checkpoint(lambda: dict(_CLEAN), io.StringIO(), "wrong", {})
    with pytest.raises(RuntimeError, match="connector idle"):
        _PROBE._public_safety_preflight(
            {
                **_CLEAN,
                "amd_cards": ["card1"],
                "connected_amd_connectors": ["card1-DP-1"],
                "kfd_path": "/dev/kfd",
                "kfd_accessible": True,
                "kfd_unowned": True,
            }
        )


def test_open_exclusive_output_is_private_and_refuses_overwrite(tmp_path):
    path = tmp_path / "runtime.jsonl"
    with _PROBE._open_exclusive_output(path) as output:
        output.write("{}\n")
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        _PROBE._open_exclusive_output(path)
