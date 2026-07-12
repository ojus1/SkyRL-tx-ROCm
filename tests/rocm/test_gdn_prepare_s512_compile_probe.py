from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_REPO = Path(__file__).parents[2]
_PROBE_PATH = _REPO / "rocm" / "probe_gdn_prepare_s512_compile.py"
_SPEC = importlib.util.spec_from_file_location(
    "probe_gdn_prepare_s512_compile_test", _PROBE_PATH
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


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
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
    ):
        environment.pop(name, None)
    return environment


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_PROBE_PATH), *arguments],
        cwd=_REPO,
        env=_clean_environment(),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def _make_library(
    tmp_path: Path, payload: bytes = b"private-gdn-prepare"
) -> tuple[Path, str]:
    path = (tmp_path / _BASENAME).resolve()
    path.write_bytes(payload)
    path.chmod(0o600)
    return path, hashlib.sha256(payload).hexdigest()


def _stablehlo(*, target: str = _TARGET, extra: str = "", alias: str = "") -> str:
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
            f"{{{layouts}{alias}}} : (tensor<1x512x16x128xf32>, tensor<1x512x32x128xf32>, tensor<1x512x32xf32>, tensor<1x512x32xf32>) -> tuple<tensor<1x512x32x128xf32>, tensor<1x512x32x128xf32>, tensor<1x512x32xf32>>",
            extra,
            "  return %0 : tuple<tensor<1x512x32x128xf32>, tensor<1x512x32x128xf32>, tensor<1x512x32xf32>>",
            "}",
        ]
    )


def _optimized(*, target: str = _TARGET, extra: str = "", alias: str = "") -> str:
    return "\n".join(
        [
            "ENTRY main {",
            "  ROOT %prepare = (f32[1,512,32,128]{3,2,1,0}, f32[1,512,32,128]{3,2,1,0}, f32[1,512,32]{2,1,0}) "
            "custom-call(f32[1,512,16,128]{3,2,1,0} %key, f32[1,512,32,128]{3,2,1,0} %value, "
            "f32[1,512,32]{2,1,0} %g, f32[1,512,32]{2,1,0} %beta), "
            f'custom_call_target="{target}"{alias}',
            extra,
            "}",
        ]
    )


def test_default_refusal_has_no_accelerator_import_or_library_load():
    result = _run()

    assert result.returncode == 0, result.stderr
    manifest, refused = [json.loads(line) for line in result.stdout.splitlines()]
    assert manifest["scope"] == "abstract_refusal"
    assert manifest["compile_may_dispatch_gpu_work"] is False
    assert manifest["counters"] == _PROBE._zero_counters()
    assert manifest["raw_library_path_emitted"] is False
    assert manifest["raw_ir_emitted"] is False
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
        ["--platform", "rocm", "--allow-gpu", "--case", "s512-compile"],
        ["--allow-gpu"],
        ["--case", "s512-compile"],
        ["--library", "/tmp/" + _BASENAME],
        ["--library-sha256", "0" * 64],
        ["--output", "/tmp/abstract.jsonl"],
        ["--platform", "rocm", "--allow-gpu", "--case", "wrong"],
        ["--platform", "rocm", "--allow-gpu", "--backward"],
        ["--platform", "rocm", "--allow-gpu", "--execute"],
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
                "s512-compile",
                "--library",
                "/tmp/" + _BASENAME,
                "--library-sha256",
                digest,
                "--output",
                "/tmp/private.jsonl",
            ]
        )


def test_complete_rocm_parser_contract_and_fresh_output(tmp_path):
    library, digest = _make_library(tmp_path)
    output = tmp_path / "result.jsonl"
    args = _PROBE._parse_args(
        [
            "--platform",
            "rocm",
            "--allow-gpu",
            "--case",
            "s512-compile",
            "--library",
            str(library),
            "--library-sha256",
            digest,
            "--output",
            str(output),
        ]
    )
    assert args.library == library
    assert args.library_sha256 == digest
    output.write_text("occupied")
    with pytest.raises(SystemExit):
        _PROBE._parse_args(
            [
                "--platform",
                "rocm",
                "--allow-gpu",
                "--case",
                "s512-compile",
                "--library",
                str(library),
                "--library-sha256",
                digest,
                "--output",
                str(output),
            ]
        )


def test_output_open_is_exclusive_private_and_symlink_safe(tmp_path):
    output = tmp_path / "nested" / "result.jsonl"
    with _PROBE._open_exclusive_output(output) as stream:
        stream.write("{}\n")
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        _PROBE._open_exclusive_output(output)
    link = tmp_path / "link.jsonl"
    link.symlink_to(output)
    with pytest.raises(FileExistsError):
        _PROBE._open_exclusive_output(link)


def test_contract_fixes_exact_abi_compile_counts_memory_and_scope():
    contract = _PROBE._exact_contract()
    assert contract["operation"] == "gdn_prepare_s512_typed_ffi_compile_only"
    assert contract["target"] == _TARGET
    assert contract["inputs"] == [
        {"name": "key", "shape": [1, 512, 16, 128], "dtype": "float32"},
        {"name": "value", "shape": [1, 512, 32, 128], "dtype": "float32"},
        {"name": "g", "shape": [1, 512, 32], "dtype": "float32"},
        {"name": "beta", "shape": [1, 512, 32], "dtype": "float32"},
    ]
    assert contract["dispatch_plan"] == {
        "shape_dtype_structs": 4,
        "registration_attempts": 1,
        "lower_calls": 1,
        "compile_calls": 1,
        "ffi_custom_calls_per_dialect": 1,
        "constructed_user_arrays": 0,
        "lowered_callable_invocations": 0,
        "compiled_executable_invocations": 0,
        "device_put_calls": 0,
        "device_get_calls": 0,
        "synchronizations": 0,
    }
    assert contract["compiled_memory_gate"] == {
        "exact_argument_bytes": 12_713_984,
        "exact_output_bytes": 16_842_752,
        "exact_alias_bytes": 0,
        "maximum_temporary_bytes": 64 * 1024**2,
        "maximum_argument_output_temporary_bytes": 96 * 1024**2,
    }
    assert not any(contract["scope_exclusions"].values())


def test_bound_committed_sources_are_exact_and_checked_before_environment(monkeypatch):
    proof = _PROBE._assert_bound_sources()
    assert proof == {
        "passed": True,
        "committed_sources_exact": True,
        "wrapper": _PROBE._EXPECTED_WRAPPER_SHA256,
        "hip": _PROBE._EXPECTED_HIP_SHA256,
        "safety": _PROBE._EXPECTED_SAFETY_SHA256,
        "sealed_loader": _PROBE._EXPECTED_SEALED_LOADER_SHA256,
        "package_skyrl": _PROBE._EXPECTED_PACKAGE_SHA256["skyrl"],
        "package_tx": _PROBE._EXPECTED_PACKAGE_SHA256["tx"],
        "package_kernels": _PROBE._EXPECTED_PACKAGE_SHA256["kernels"],
        "package_rocm": _PROBE._EXPECTED_PACKAGE_SHA256["rocm"],
    }
    real_hash = _PROBE._file_sha256
    wrapper = _PROBE._source_files()["gdn_prepare_wrapper_source_sha256"]
    monkeypatch.setattr(
        _PROBE,
        "_file_sha256",
        lambda path: "0" * 64 if path == wrapper else real_hash(path),
    )
    with pytest.raises(RuntimeError, match="source hash"):
        _PROBE._assert_bound_sources()

    tree = ast.parse(_PROBE_PATH.read_text())
    execute = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_execute"
    )
    calls = [
        node.func.id
        for node in ast.walk(execute)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.index("_assert_bound_sources") < calls.index(
        "_configure_rocm_environment"
    )


def test_library_validation_binds_path_hash_identity_and_permissions(tmp_path):
    library, digest = _make_library(tmp_path)
    manifest = _PROBE._validate_library_path(library, digest)
    assert manifest["canonical"] is True
    assert manifest["sha256"] == digest
    assert manifest["raw_path_emitted"] is False
    assert (
        _PROBE._assert_same_library(library, manifest)["identity"]
        == manifest["identity"]
    )

    with pytest.raises(ValueError, match="absolute"):
        _PROBE._validate_library_path(Path(_BASENAME), digest)
    wrong = (tmp_path / "wrong.so").resolve()
    wrong.write_bytes(b"wrong")
    wrong.chmod(0o600)
    with pytest.raises(ValueError, match="basename"):
        _PROBE._validate_library_path(wrong, hashlib.sha256(b"wrong").hexdigest())
    with pytest.raises(RuntimeError, match="SHA-256"):
        _PROBE._validate_library_path(library, "0" * 64)
    library.chmod(0o620)
    with pytest.raises(ValueError, match="group- or world-writable"):
        _PROBE._validate_library_path(library, digest)


def test_library_validation_rejects_symlinks(tmp_path):
    target = tmp_path / "target.so"
    target.write_bytes(b"target")
    target.chmod(0o600)
    link = tmp_path / _BASENAME
    link.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        _PROBE._validate_library_path(
            link.absolute(), hashlib.sha256(b"target").hexdigest()
        )


def test_environment_requires_sole_command_buffer_disable(monkeypatch):
    names = (
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
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    environment = _PROBE._configure_rocm_environment()
    assert _PROBE._prove_command_buffers_disabled(environment)["sole_xla_flag"] is True

    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_command_buffer= --other=true")
    with pytest.raises(RuntimeError, match="solely"):
        _PROBE._configure_rocm_environment()


def test_independent_ir_summaries_prove_target_no_alias_loop_and_physical_layouts():
    stable = _PROBE._ir_summary(_stablehlo(), "stablehlo")
    optimized = _PROBE._ir_summary(_optimized(), "optimized_hlo")
    assert stable["passed"] is True
    assert optimized["passed"] is True
    assert stable["calls"][0]["targets"] == [_TARGET]
    for summary in (stable, optimized):
        layouts = summary["calls"][0]["layouts"]
        assert layouts["expected_r4"] == [3, 2, 1, 0]
        assert layouts["expected_r3"] == [2, 1, 0]
        assert layouts["observed_r4_count"] == 4
        assert layouts["observed_r3_count"] == 3
        assert layouts["passed"] is True
    assert _PROBE._structural_gate(stable, optimized)["passed"] is True


@pytest.mark.parametrize(
    ("stable", "optimized"),
    [
        (_stablehlo(target="lookalike_" + _TARGET), _optimized()),
        (_stablehlo(), _optimized(target=_TARGET + "_lookalike")),
        (
            _stablehlo(
                extra='  %1 = stablehlo.custom_call @"other"() : () -> tensor<i32>'
            ),
            _optimized(),
        ),
        (
            _stablehlo(),
            _optimized(extra='  %x = s32[] custom-call(), custom_call_target="other"'),
        ),
        (
            _stablehlo(),
            _optimized(extra='  %x = s32[] custom-call (), custom_call_target="other"'),
        ),
        (_stablehlo(extra="  %1 = stablehlo.while(%0) : tensor<i32>"), _optimized()),
        (
            _stablehlo(),
            _optimized(extra="  %x = s32[] while(%state), condition=c, body=b"),
        ),
        (
            _stablehlo(),
            _optimized(extra="  %x = s32[] while (%state), condition=c, body=b"),
        ),
        (_stablehlo(alias=", output_operand_aliases = [#alias]"), _optimized()),
        (_stablehlo(), _optimized(alias=", output_to_operand_aliasing={{}: (0, {})}")),
        (
            _stablehlo(),
            _optimized().replace(
                "ENTRY main {",
                "HloModule m, input_output_alias={{}: (0, {}, may-alias)}\nENTRY main {",
            ),
        ),
        (_stablehlo().replace("dense<[3,2,1,0]>", "dense<[0,1,2,3]>", 1), _optimized()),
        (_stablehlo(), _optimized().replace("{2,1,0}", "{0,1,2}", 1)),
    ],
)
def test_ir_gate_rejects_lookalikes_extra_calls_loops_aliases_and_layout_drift(
    stable, optimized
):
    summaries = (
        _PROBE._ir_summary(stable, "stablehlo"),
        _PROBE._ir_summary(optimized, "optimized_hlo"),
    )
    assert _PROBE._structural_gate(*summaries)["passed"] is False


def test_ir_parser_rejects_unparsed_textual_custom_call_and_missing_dialect():
    generic = (
        'module { %0 = "stablehlo.custom_call"() {call_target_name = "'
        + _TARGET
        + '"} : () -> tensor<i32> }'
    )
    stable = _PROBE._ir_summary(generic, "stablehlo")
    assert stable["passed"] is False
    optimized = _PROBE._ir_summary(_optimized(), "optimized_hlo")
    assert _PROBE._structural_gate(stable)["passed"] is False
    assert _PROBE._structural_gate(optimized, optimized)["passed"] is False


def test_ir_alias_parser_allows_explicitly_empty_module_and_call_collections():
    stable = _PROBE._ir_summary(
        _stablehlo(alias=", output_operand_aliases = []"), "stablehlo"
    )
    optimized = _PROBE._ir_summary(
        _optimized(alias=", output_to_operand_aliasing={}").replace(
            "ENTRY main {", "HloModule m, input_output_alias={}\nENTRY main {"
        ),
        "optimized_hlo",
    )
    assert stable["nonempty_alias_metadata"] == []
    assert optimized["nonempty_alias_metadata"] == []
    assert _PROBE._structural_gate(stable, optimized)["passed"] is True


def test_stablehlo_layout_labels_inside_backend_config_cannot_spoof_attributes():
    quoted_spoof = (
        "operand_layouts = [dense<[3,2,1,0]>, dense<[3,2,1,0]>, "
        "dense<[2,1,0]>, dense<[2,1,0]>], result_layouts = "
        "[dense<[3,2,1,0]>, dense<[3,2,1,0]>, dense<[2,1,0]>]"
    )
    stable = re.sub(
        r"\{operand_layouts\s*=.*?\}\s*:",
        '{backend_config = "' + quoted_spoof + '"} :',
        _stablehlo(),
        count=1,
        flags=re.DOTALL,
    )
    summary = _PROBE._ir_summary(stable, "stablehlo")
    assert summary["calls"][0]["layouts"]["observed_r4_count"] == 0
    assert summary["calls"][0]["layouts"]["observed_r3_count"] == 0
    assert summary["checks"]["physical_row_major_layouts_exact"] is False
    assert summary["passed"] is False


def test_stablehlo_layouts_cannot_come_from_neighboring_top_level_attributes():
    spoof_inputs = (
        "[dense<[3,2,1,0]>, dense<[3,2,1,0]>, dense<[2,1,0]>, dense<[2,1,0]>]"
    )
    spoof_outputs = "[dense<[3,2,1,0]>, dense<[3,2,1,0]>, dense<[2,1,0]>]"
    replacement = (
        "{operand_layouts = [], spoof_inputs = "
        + spoof_inputs
        + ", result_layouts = [], spoof_outputs = "
        + spoof_outputs
        + "} :"
    )
    stable = re.sub(
        r"\{operand_layouts\s*=.*?\}\s*:",
        replacement,
        _stablehlo(),
        count=1,
        flags=re.DOTALL,
    )
    summary = _PROBE._ir_summary(stable, "stablehlo")
    assert summary["calls"][0]["layouts"]["observed_r4_count"] == 0
    assert summary["calls"][0]["layouts"]["observed_r3_count"] == 0
    assert summary["checks"]["physical_row_major_layouts_exact"] is False
    assert summary["passed"] is False


@pytest.mark.parametrize(
    "stable",
    [
        _stablehlo().replace("operand_layouts = [", "operand_layouts = [#junk, ", 1),
        _stablehlo().replace("tensor<4xindex>", "tensor<3xindex>", 1),
    ],
)
def test_stablehlo_layout_values_must_be_fully_consumed_and_typed(stable):
    summary = _PROBE._ir_summary(stable, "stablehlo")
    layouts = summary["calls"][0]["layouts"]
    assert layouts["checks"]["layout_attribute_values_fully_parsed"] is False
    assert summary["checks"]["physical_row_major_layouts_exact"] is False
    assert summary["passed"] is False


@pytest.mark.parametrize(
    ("dialect", "text"),
    [
        (
            "stablehlo",
            _stablehlo().replace(
                "tensor<1x512x16x128xf32>", "tensor<1x512x32x128xf32>", 1
            ),
        ),
        (
            "optimized_hlo",
            _optimized().replace("f32[1,512,16,128]", "f32[1,512,32,128]", 1),
        ),
        (
            "stablehlo",
            _stablehlo()
            .replace("dense<[3,2,1,0]>", "dense<[SPOOF]>", 1)
            .replace("dense<[2,1,0]>", "dense<[3,2,1,0]>", 1)
            .replace("dense<[SPOOF]>", "dense<[2,1,0]>", 1),
        ),
        (
            "optimized_hlo",
            _optimized()
            .replace("{3,2,1,0}", "{9}", 1)
            .replace("{2,1,0}", "{3,2,1,0}", 1)
            .replace("{9}", "{2,1,0}", 1),
        ),
        (
            "stablehlo",
            _stablehlo().replace(
                "%arg0, %arg1, %arg2, %arg3",
                "%arg0, %arg1, %arg2, %arg3, %arg4",
                1,
            ),
        ),
        (
            "optimized_hlo",
            _optimized().replace(
                "f32[1,512,32]{2,1,0} %beta)",
                "f32[1,512,32]{2,1,0} %beta, s32[] %extra)",
                1,
            ),
        ),
    ],
)
def test_signature_layout_gate_rejects_spoofs_with_seven_layout_tokens(dialect, text):
    summary = _PROBE._ir_summary(text, dialect)
    assert summary["passed"] is False
    assert summary["checks"]["physical_row_major_layouts_exact"] is False


def test_compiled_memory_gate_requires_exact_sizes_and_both_caps():
    exact = {
        "available": True,
        "argument_size_in_bytes": 12_713_984,
        "output_size_in_bytes": 16_842_752,
        "alias_size_in_bytes": 0,
        "temp_size_in_bytes": 64 * 1024**2,
    }
    assert _PROBE._compiled_memory_gate(exact)["passed"] is True
    exact["temp_size_in_bytes"] = 70_000_000
    assert _PROBE._compiled_memory_gate(exact)["passed"] is False
    exact["temp_size_in_bytes"] = 60_000_000
    assert _PROBE._compiled_memory_gate(exact)["passed"] is True
    for key, value in (
        ("argument_size_in_bytes", 12_713_985),
        ("output_size_in_bytes", 16_842_751),
        ("alias_size_in_bytes", 1),
    ):
        assert _PROBE._compiled_memory_gate({**exact, key: value})["passed"] is False
    missing = dict(exact)
    del missing["alias_size_in_bytes"]
    assert _PROBE._compiled_memory_gate(missing)["passed"] is False


def _registration(path: Path, digest: str, size: int) -> SimpleNamespace:
    return SimpleNamespace(
        library_path=path,
        library_sha256=digest,
        snapshot_sha256=digest,
        snapshot_size_bytes=size,
        snapshot_mode=0o600,
        snapshot_seals=0x000F,
        sealed_snapshot=True,
        snapshot_fd_retained=True,
        target_name=_TARGET,
        platform="ROCM",
        registration_api_version=1,
        custom_call_api_version=4,
    )


def test_registration_manifest_requires_complete_sealed_identity(tmp_path):
    path, digest = _make_library(tmp_path)
    registration = _registration(path, digest, path.stat().st_size)
    assert (
        _PROBE._sealed_registration_manifest(
            registration,
            library_path=path,
            library_sha256=digest,
            library_size_bytes=path.stat().st_size,
        )["passed"]
        is True
    )
    registration.snapshot_seals = 0x0007
    with pytest.raises(RuntimeError, match="sealed identity"):
        _PROBE._sealed_registration_manifest(
            registration,
            library_path=path,
            library_sha256=digest,
            library_size_bytes=path.stat().st_size,
        )


class _Memory:
    argument_size_in_bytes = 12_713_984
    output_size_in_bytes = 16_842_752
    alias_size_in_bytes = 0
    temp_size_in_bytes = 4096
    generated_code_size_in_bytes = 2048


class _FakeCompiled:
    def __init__(self, state: dict[str, Any], optimized: str | None = None):
        self.state = state
        self.optimized = _optimized() if optimized is None else optimized

    def __call__(self, *_args):
        self.state["compiled_invocations"] += 1
        raise AssertionError("compile-only probe invoked executable")

    def as_text(self):
        self.state["as_text_calls"] += 1
        return self.optimized

    def memory_analysis(self):
        self.state["memory_calls"] += 1
        return _Memory()


class _FakeLowered:
    def __init__(self, state: dict[str, Any], compiled: Any, stable: str | None = None):
        self.state = state
        self.compiled = compiled
        self.stable = _stablehlo() if stable is None else stable

    def compiler_ir(self, *, dialect):
        assert dialect == "stablehlo"
        self.state["compiler_ir_calls"] += 1
        return self.stable

    def compile(self):
        self.state["compile_calls"] += 1
        return self.compiled


class _FakeJitted:
    def __init__(self, function: Any, state: dict[str, Any], lowered: Any):
        self.function = function
        self.state = state
        self.lowered = lowered

    def lower(self, *signatures):
        self.state["lower_calls"] += 1
        self.state["signatures"] = signatures
        self.function(*signatures)
        return self.lowered


class _FakeJax:
    __version__ = "fake-jax"

    def __init__(self, state: dict[str, Any], lowered: Any):
        self.state = state
        self.lowered = lowered

    def ShapeDtypeStruct(self, shape, dtype):
        self.state["shape_specs"].append((tuple(shape), dtype))
        return SimpleNamespace(shape=tuple(shape), dtype=dtype)

    def jit(self, function):
        self.state["jit_calls"] += 1
        return _FakeJitted(function, self.state, self.lowered)

    def default_backend(self):
        return "gpu"

    def devices(self):
        return [SimpleNamespace()]


def _fake_state() -> dict[str, Any]:
    return {
        "shape_specs": [],
        "jit_calls": 0,
        "lower_calls": 0,
        "compile_calls": 0,
        "compiler_ir_calls": 0,
        "as_text_calls": 0,
        "memory_calls": 0,
        "wrapper_calls": 0,
        "registration_calls": 0,
        "compiled_invocations": 0,
    }


def test_compile_exact_registers_then_lowers_and_compiles_once_without_arrays_or_invocation(
    tmp_path,
):
    path, digest = _make_library(tmp_path)
    state = _fake_state()
    compiled = _FakeCompiled(state)
    lowered = _FakeLowered(state, compiled)
    jax = _FakeJax(state, lowered)
    counters = _PROBE._zero_counters()
    output = io.StringIO()

    def register(library_path, *, library_sha256, enabled):
        state["registration_calls"] += 1
        assert state["lower_calls"] == 0
        assert (library_path, library_sha256, enabled) == (path, digest, True)
        return _registration(path, digest, path.stat().st_size)

    def wrapper(key, value, g, beta, **kwargs):
        state["wrapper_calls"] += 1
        assert kwargs == {
            "enabled": True,
            "library_path": path,
            "library_sha256": digest,
        }
        return value, value, g

    report = _PROBE._compile_exact(
        jax,
        SimpleNamespace(float32="f32"),
        wrapper,
        register,
        path,
        digest,
        path.stat().st_size,
        lambda: dict(_CLEAN),
        counters,
        output,
    )

    assert report["release_gate"]["passed"] is True
    assert state["registration_calls"] == 1
    assert state["wrapper_calls"] == 1
    assert state["jit_calls"] == state["lower_calls"] == state["compile_calls"] == 1
    assert state["compiled_invocations"] == 0
    assert counters["shape_dtype_structs"] == 4
    assert counters["constructed_user_arrays"] == 0
    assert counters["compiled_executable_invocations"] == 0
    stages = [
        record["stage"]
        for record in _records(output)
        if record["record_type"] == "journal_checkpoint"
    ]
    assert stages == [
        "after_ffi_registration_attempt",
        "after_ffi_lower_attempt",
        "after_ffi_compile_attempt",
    ]


def test_failed_ir_gate_never_invokes_compiled_object(tmp_path):
    path, digest = _make_library(tmp_path)
    state = _fake_state()
    compiled = _FakeCompiled(state, optimized=_optimized(target="wrong"))
    jax = _FakeJax(state, _FakeLowered(state, compiled))
    with pytest.raises(RuntimeError, match="structural or memory"):
        _PROBE._compile_exact(
            jax,
            SimpleNamespace(float32="f32"),
            lambda _k, v, g, _b, **_kwargs: (v, v, g),
            lambda *_args, **_kwargs: _registration(path, digest, path.stat().st_size),
            path,
            digest,
            path.stat().st_size,
            lambda: dict(_CLEAN),
            _PROBE._zero_counters(),
            io.StringIO(),
        )
    assert state["compiled_invocations"] == 0


def test_failed_stablehlo_precompile_gate_never_calls_compile(tmp_path):
    path, digest = _make_library(tmp_path)
    state = _fake_state()
    compiled = _FakeCompiled(state)
    malformed_stable = _stablehlo().replace(
        "tensor<1x512x16x128xf32>", "tensor<1x512x32x128xf32>", 1
    )
    jax = _FakeJax(state, _FakeLowered(state, compiled, stable=malformed_stable))
    counters = _PROBE._zero_counters()
    with pytest.raises(RuntimeError, match="StableHLO failed the precompile"):
        _PROBE._compile_exact(
            jax,
            SimpleNamespace(float32="f32"),
            lambda _k, v, g, _b, **_kwargs: (v, v, g),
            lambda *_args, **_kwargs: _registration(path, digest, path.stat().st_size),
            path,
            digest,
            path.stat().st_size,
            lambda: dict(_CLEAN),
            counters,
            io.StringIO(),
        )
    assert state["compile_calls"] == 0
    assert state["compiled_invocations"] == 0
    assert counters["compile_attempts"] == 0


def test_ast_compile_path_contains_no_array_placement_sync_or_callable_invocation():
    tree = ast.parse(_PROBE_PATH.read_text())
    compile_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_compile_exact"
    )
    attributes = {
        node.attr
        for node in ast.walk(compile_function)
        if isinstance(node, ast.Attribute)
    }
    assert {
        "device_put",
        "device_get",
        "block_until_ready",
        "array",
        "zeros",
        "ones",
    }.isdisjoint(attributes)
    calls = [node for node in ast.walk(compile_function) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(call.func, ast.Name) and call.func.id == "compiled" for call in calls
    )
    assert not any(
        isinstance(call.func, ast.Name) and call.func.id == "lowered" for call in calls
    )


def test_backend_manifest_requires_exactly_one_rocm_gpu():
    state = _fake_state()
    jax = _FakeJax(state, None)
    backend = SimpleNamespace(
        get_backend=lambda: SimpleNamespace(platform_version="ROCm 7.2")
    )
    result = _PROBE._backend_manifest(
        jax, SimpleNamespace(__version__="fake-jaxlib"), backend
    )
    assert result["platform_family"] == "rocm"
    jax.devices = lambda: [SimpleNamespace(), SimpleNamespace()]
    with pytest.raises(RuntimeError, match="exactly one ROCm GPU"):
        _PROBE._backend_manifest(jax, SimpleNamespace(__version__="fake"), backend)


def test_journal_checkpoint_rejects_undeclared_stage():
    with pytest.raises(RuntimeError, match="undeclared"):
        _PROBE._journal_checkpoint(lambda: dict(_CLEAN), io.StringIO(), "surprise", {})
