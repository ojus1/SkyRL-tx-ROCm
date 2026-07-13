from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

_REPO = Path(__file__).parents[2]
_PROBE_PATH = _REPO / "rocm" / "probe_gdn_execute_s512_ffi.py"
_PROFILE_PATH = _REPO / "rocm" / "profile_rocm.py"
_SPEC = importlib.util.spec_from_file_location(
    "probe_gdn_execute_s512_ffi_test", _PROBE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)

_TARGET = "skyrl_gdn_execute_s512_f32_bf16_v1"
_BASENAME = "libskyrl_gdn_execute_s512_gfx1100.so"
_CLEAN = {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}


def _records(output: io.StringIO) -> list[dict[str, Any]]:
    return [json.loads(line) for line in output.getvalue().splitlines()]


def _abstract_args() -> SimpleNamespace:
    return SimpleNamespace(
        platform="abstract",
        allow_gpu=False,
        compile_diagnostic=False,
        case=None,
        library=None,
        library_sha256=None,
        output=None,
    )


def _make_library(
    tmp_path: Path, payload: bytes = b"inert-not-a-shared-library"
) -> tuple[Path, str]:
    path = (tmp_path / _BASENAME).resolve()
    path.write_bytes(payload)
    path.chmod(0o600)
    return path, hashlib.sha256(payload).hexdigest()


def _stablehlo(
    *,
    target: str = _TARGET,
    input_shape: str = "1x512x16x128",
    alias: str = "",
    extra: str = "",
) -> str:
    operand_layouts = (
        "[dense<[3,2,1,0]> : tensor<4xindex>, "
        "dense<[3,2,1,0]> : tensor<4xindex>, "
        "dense<[3,2,1,0]> : tensor<4xindex>, "
        "dense<[3,2,1,0]> : tensor<4xindex>, "
        "dense<[2,1,0]> : tensor<3xindex>, "
        "dense<[3,2,1,0]> : tensor<4xindex>]"
    )
    result_layouts = (
        "[dense<[3,2,1,0]> : tensor<4xindex>, dense<[3,2,1,0]> : tensor<4xindex>]"
    )
    return "\n".join(
        [
            "module {",
            "  func.func @main(%arg0: tensor<1x512x16x128xf32>, "
            "%arg1: tensor<1x512x16x128xf32>, "
            "%arg2: tensor<1x512x32x128xf32>, "
            "%arg3: tensor<1x512x32x128xf32>, "
            "%arg4: tensor<1x512x32xf32>, "
            "%arg5: tensor<1x32x128x128xf32>) {",
            f'    %0:2 = stablehlo.custom_call @"{target}"(%arg0, %arg1, %arg2, '
            "%arg3, %arg4, %arg5) {",
            f"      operand_layouts = {operand_layouts},",
            f"      result_layouts = {result_layouts}{alias}",
            f"    }} : (tensor<{input_shape}xf32>, tensor<1x512x16x128xf32>, "
            "tensor<1x512x32x128xf32>, tensor<1x512x32x128xf32>, "
            "tensor<1x512x32xf32>, tensor<1x32x128x128xf32>) -> "
            "tuple<tensor<1x512x32x128xbf16>, tensor<1x32x128x128xf32>>",
            extra,
            "    return %0 : tuple<tensor<1x512x32x128xbf16>, "
            "tensor<1x32x128x128xf32>>",
            "  }",
            "}",
        ]
    )


def _optimized(
    *,
    target: str = _TARGET,
    query_dtype: str = "f32",
    alias: str = "",
    extra: str = "",
) -> str:
    return "\n".join(
        [
            "ENTRY %main {",
            f"  %query = {query_dtype}[1,512,16,128]{{3,2,1,0}} parameter(0)",
            "  %key = f32[1,512,16,128]{3,2,1,0} parameter(1)",
            "  %u = f32[1,512,32,128]{3,2,1,0} parameter(2)",
            "  %w = f32[1,512,32,128]{3,2,1,0} parameter(3)",
            "  %gamma = f32[1,512,32]{2,1,0} parameter(4)",
            "  %state = f32[1,32,128,128]{3,2,1,0} parameter(5)",
            "  ROOT %execute = (bf16[1,512,32,128]{3,2,1,0}, "
            "f32[1,32,128,128]{3,2,1,0}) custom-call(%query, %key, %u, %w, "
            f'%gamma, %state), custom_call_target="{target}"{alias}',
            extra,
            "}",
        ]
    )


def _profile_argv(
    telemetry: Path, current_argv: list[str], *, profile_path: Path = _PROFILE_PATH
) -> list[str]:
    return [
        sys.executable,
        str(profile_path),
        "--timeout",
        "300",
        "--interval",
        "0.1",
        "--sensor-grace-seconds",
        "60",
        "--max-junction-temp-c",
        "90",
        "--max-gpu-power-watts",
        "315",
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
        *current_argv,
    ]


class _FakeCompiled:
    def __init__(self, *, optimized: str | None = None) -> None:
        self.optimized = optimized or _optimized()
        self.invocations = 0

    def as_text(self) -> str:
        return self.optimized

    @staticmethod
    def memory_analysis() -> SimpleNamespace:
        return SimpleNamespace(
            argument_size_in_bytes=27_328_512,
            output_size_in_bytes=6_291_472,
            alias_size_in_bytes=0,
            temp_size_in_bytes=0,
            generated_code_size_in_bytes=4096,
        )

    def __call__(self, *_inputs: Any) -> tuple[Any, Any]:
        self.invocations += 1
        raise AssertionError("compile diagnostic must not invoke the executable")


class _FakeLowered:
    def __init__(self, compiled: _FakeCompiled, *, stable: str | None = None) -> None:
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
    __version__ = "0.10.2"

    def __init__(self, lowered: _FakeLowered) -> None:
        self.lowered = lowered

    @staticmethod
    def ShapeDtypeStruct(shape: tuple[int, ...], dtype: Any) -> SimpleNamespace:
        return SimpleNamespace(shape=shape, dtype=dtype)

    def jit(self, function: Any) -> _FakeJit:
        return _FakeJit(function, self.lowered)

    @staticmethod
    def default_backend() -> str:
        return "gpu"

    @staticmethod
    def devices() -> list[SimpleNamespace]:
        return [SimpleNamespace(id=0, device_kind="fake gfx1100")]


def _registration(path: Path, digest: str, size: int) -> SimpleNamespace:
    return SimpleNamespace(
        library_path=path,
        library_sha256=digest,
        snapshot_sha256=digest,
        snapshot_size_bytes=size,
        snapshot_mode=0o600,
        snapshot_seals=0xF,
        target_name=_TARGET,
        platform="ROCM",
        registration_api_version=1,
        custom_call_api_version=4,
        sealed_snapshot=True,
        snapshot_fd_retained=True,
    )


@pytest.fixture(scope="module")
def host_reference() -> tuple[
    tuple[Any, ...], tuple[Any, Any], dict[str, Any], dict[str, Any]
]:
    prepare, execute = _PROBE._load_oracles()
    counters = _PROBE._zero_counters()
    boundary, reference, input_report, reference_report = (
        _PROBE._construct_host_reference(
            np,
            prepare.gdn_prepare_s512_numpy,
            execute.gdn_execute_s512_numpy,
            counters,
        )
    )
    assert counters == {
        **_PROBE._zero_counters(),
        "host_boundary_arrays": 6,
        "prepare_oracle_attempts": 1,
        "prepare_oracle_completions": 1,
        "execute_oracle_attempts": 1,
        "execute_oracle_completions": 1,
        "sensitivity_oracle_invocations": 11,
    }
    return boundary, reference, input_report, reference_report


def test_probe_source_has_only_standard_library_top_level_imports() -> None:
    source = _PROBE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source)
    imports = [
        node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported_roots = {
        alias.name.partition(".")[0] for node in imports for alias in node.names
    }

    assert imported_roots.isdisjoint({"jax", "jaxlib", "numpy", "skyrl", "ctypes"})
    assert "import numpy as np_module" in source
    assert "from skyrl.tx.kernels.rocm.gdn_execute_ffi import" in source
    assert source.index('if args.platform == "abstract":') < source.index(
        "_assert_fresh_accelerator_process()"
    )


def test_default_subprocess_refuses_without_accelerator_import_or_library_map() -> None:
    environment = os.environ.copy()
    for name in (
        "JAX_PLATFORMS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        environment.pop(name, None)
    snippet = "\n".join(
        [
            "import json,runpy,sys",
            "code=0",
            "try:",
            f"    runpy.run_path({str(_PROBE_PATH)!r},run_name='__main__')",
            "except SystemExit as error:",
            "    code=int(error.code or 0)",
            "modules=set(sys.modules)",
            "maps=open('/proc/self/maps',encoding='utf-8').read()",
            "print(json.dumps({",
            "    'subprocess_exit':code,",
            "    'jax_loaded':any(x=='jax' or x.startswith('jax.') for x in modules),",
            "    'numpy_loaded':any(x=='numpy' or x.startswith('numpy.') for x in modules),",
            "    'skyrl_loaded':any(x=='skyrl' or x.startswith('skyrl.') for x in modules),",
            f"    'candidate_library_mapped':{_BASENAME!r} in maps,",
            "},sort_keys=True))",
        ]
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
    manifest, refused, independent = [
        json.loads(line) for line in result.stdout.splitlines()
    ]
    assert manifest["scope"] == "abstract_refusal"
    assert manifest["counters"] == _PROBE._zero_counters()
    assert refused["status"] == "no_gpu_abstract_manifest_only"
    assert refused["jax_imported"] is False
    assert refused["numpy_imported"] is False
    assert refused["skyrl_rocm_package_imported"] is False
    assert refused["shared_library_loaded"] is False
    assert independent == {
        "candidate_library_mapped": False,
        "jax_loaded": False,
        "numpy_loaded": False,
        "skyrl_loaded": False,
        "subprocess_exit": 0,
    }


def test_abstract_manifest_binds_all_sources_and_refuses_every_counter() -> None:
    output = io.StringIO()
    assert _PROBE._execute(_abstract_args(), output) == 0
    manifest, refused = _records(output)

    for name, path in _PROBE._source_files().items():
        assert manifest[name] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert refused["counters"] == _PROBE._zero_counters()
    assert manifest["raw_library_path_emitted"] is False
    assert manifest["raw_ir_emitted"] is False
    assert manifest["raw_tensors_emitted"] is False


@pytest.mark.parametrize(
    "arguments",
    [
        ["--platform", "rocm"],
        ["--platform", "rocm", "--allow-gpu"],
        ["--platform", "rocm", "--allow-gpu", "--case", _PROBE._CASE],
        ["--allow-gpu"],
        ["--compile-diagnostic"],
        ["--case", _PROBE._CASE],
        ["--library", "/tmp/" + _BASENAME],
        ["--library-sha256", "0" * 64],
        ["--output", "/tmp/execute.jsonl"],
        ["--platform", "rocm", "--allow-gpu", "--case", "wrong"],
        ["--platform", "rocm", "--allow-gpu", "--warmup"],
        ["--platform", "rocm", "--allow-gpu", "--replay"],
        ["--platform", "rocm", "--allow-gpu", "--backward"],
    ],
)
def test_parser_rejects_incomplete_acknowledgement_and_scope_broadening(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        _PROBE._parse_args(arguments)
    assert raised.value.code == 2


@pytest.mark.parametrize("digest", ["", "0" * 63, "0" * 65, "A" * 64, "g" * 64])
def test_parser_requires_exact_lowercase_library_digest(digest: str) -> None:
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
                "/tmp/execute.jsonl",
            ]
        )


@pytest.mark.parametrize("compile_diagnostic", [False, True])
def test_complete_parser_contract_requires_fresh_canonical_output(
    tmp_path: Path, compile_diagnostic: bool
) -> None:
    library, _digest = _make_library(tmp_path)
    output = tmp_path / "result.jsonl"
    argv = [
        "--platform",
        "rocm",
        "--allow-gpu",
        "--case",
        _PROBE._CASE,
        "--library",
        str(library),
        "--library-sha256",
        _PROBE._EXPECTED_LIBRARY_SHA256,
        "--output",
        str(output),
    ]
    if compile_diagnostic:
        argv.append("--compile-diagnostic")
    args = _PROBE._parse_args(argv)
    assert args.compile_diagnostic is compile_diagnostic
    assert args.output == output

    output.write_text("occupied", encoding="utf-8")
    with pytest.raises(SystemExit):
        _PROBE._parse_args(argv)


def test_contract_pins_exact_abi_counts_memory_thresholds_and_resource_caps() -> None:
    contract = _PROBE._exact_contract()
    assert (
        contract["operation"] == "gdn_execute_s512_typed_ffi_one_shot_numerical_runtime"
    )
    assert contract["target"] == _TARGET
    assert [item["shape"] for item in contract["inputs"]] == [
        [1, 512, 16, 128],
        [1, 512, 16, 128],
        [1, 512, 32, 128],
        [1, 512, 32, 128],
        [1, 512, 32],
        [1, 32, 128, 128],
    ]
    assert contract["compile_gate"]["argument_bytes"] == 27_328_512
    assert contract["compile_gate"]["logical_output_bytes"] == 6_291_456
    assert contract["compile_gate"]["compiler_output_bytes"] == 6_291_472
    assert contract["invocation_contract"]["checked_executable_invocations"] == 1
    for name in (
        "warmups",
        "replays",
        "graphs",
        "gpu_references",
        "gpu_reductions",
        "backward",
        "model",
    ):
        assert contract["invocation_contract"][name] == 0
    outer = contract["outer_supervision"]
    assert outer["maximum_vram_gib"] == 24
    assert outer["maximum_junction_temperature_c"] == 90
    assert outer["maximum_gpu_power_watts"] == 315
    assert outer["maximum_swap_gib"] == 8
    assert outer["sensor_grace_seconds_maximum"] == 60


def test_counter_contracts_are_exact_and_compile_diagnostic_has_no_runtime_work() -> (
    None
):
    zero = _PROBE._zero_counters()
    diagnostic = _PROBE._completed_compile_diagnostic_counters()
    assert diagnostic.keys() == zero.keys()
    assert {name for name, value in diagnostic.items() if value} == {
        "backend_initialization_attempts",
        "backend_initialization_completions",
        "registration_attempts",
        "registration_completions",
        "shape_dtype_structs",
        "ffi_python_trace_calls",
        "lower_attempts",
        "lower_completions",
        "compile_attempts",
        "compile_completions",
    }
    assert diagnostic["shape_dtype_structs"] == 6
    for name in (
        "host_boundary_arrays",
        "prepare_oracle_attempts",
        "prepare_oracle_completions",
        "execute_oracle_attempts",
        "execute_oracle_completions",
        "tuple_device_put_attempts",
        "tuple_device_put_completions",
        "device_put_leaves",
        "input_readiness_barriers",
        "checked_executable_attempts",
        "checked_executable_completions",
        "output_readiness_barriers",
        "tuple_device_get_attempts",
        "tuple_device_get_completions",
        "device_get_leaves",
        "lowered_callable_invocations",
        "warmup_invocations",
        "replay_invocations",
        "graph_invocations",
        "gpu_reference_invocations",
        "gpu_reduction_invocations",
        "backward_invocations",
        "model_invocations",
    ):
        assert diagnostic[name] == 0


def test_bound_sources_cover_every_runtime_dependency() -> None:
    bound = _PROBE._assert_bound_sources()
    assert bound == {
        "passed": True,
        "all_executable_dependencies_exact": True,
        **_PROBE._EXPECTED_SOURCE_SHA256,
    }


def test_exact_file_loader_is_hash_bound(tmp_path: Path) -> None:
    source = tmp_path / "inert_oracle.py"
    source.write_text("VALUE = 17\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    loaded = _PROBE._load_exact_file_module(source, digest, "inert_oracle")
    assert loaded.VALUE == 17
    with pytest.raises(RuntimeError, match="refusing changed"):
        _PROBE._load_exact_file_module(source, "0" * 64, "inert_oracle")


def test_private_canonical_library_validation_uses_inert_bytes_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    library, digest = _make_library(tmp_path)
    monkeypatch.setattr(_PROBE, "_EXPECTED_LIBRARY_SHA256", digest)
    monkeypatch.setattr(_PROBE, "_EXPECTED_LIBRARY_SIZE_BYTES", library.stat().st_size)
    manifest = _PROBE._validate_library_path(library, digest)
    assert manifest["validated"] is True
    assert manifest["canonical"] is True
    assert manifest["mode"] == 0o600
    assert manifest["sha256"] == digest
    assert manifest["size_bytes"] == library.stat().st_size
    assert (
        _PROBE._assert_same_library(library, manifest)["identity"]
        == manifest["identity"]
    )


def test_library_validation_rejects_digest_mode_basename_and_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    library, digest = _make_library(tmp_path)
    monkeypatch.setattr(_PROBE, "_EXPECTED_LIBRARY_SHA256", digest)
    monkeypatch.setattr(_PROBE, "_EXPECTED_LIBRARY_SIZE_BYTES", library.stat().st_size)
    with pytest.raises(RuntimeError, match="SHA-256"):
        _PROBE._validate_library_path(library, "0" * 64)

    library.chmod(0o640)
    with pytest.raises(ValueError, match="0600"):
        _PROBE._validate_library_path(library, digest)
    library.chmod(0o600)

    wrong = tmp_path / "wrong.so"
    wrong.write_bytes(b"inert")
    wrong.chmod(0o600)
    with pytest.raises(ValueError, match="basename"):
        _PROBE._validate_library_path(
            wrong.resolve(), hashlib.sha256(b"inert").hexdigest()
        )

    target = tmp_path / "target"
    target.write_bytes(b"symlink payload")
    target.chmod(0o600)
    link = tmp_path / _BASENAME
    library.unlink()
    link.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        _PROBE._validate_library_path(
            link.absolute(), hashlib.sha256(b"symlink payload").hexdigest()
        )


def test_output_open_is_exclusive_private_and_symlink_safe(tmp_path: Path) -> None:
    output = tmp_path / "result.jsonl"
    with _PROBE._open_exclusive_output(output) as stream:
        stream.write("{}\n")
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        _PROBE._open_exclusive_output(output)

    link = tmp_path / "link.jsonl"
    link.symlink_to(output)
    with pytest.raises(FileExistsError):
        _PROBE._open_exclusive_output(link)


def test_profile_argv_validator_accepts_exact_private_supervision(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text("{}\n", encoding="utf-8")
    telemetry.chmod(0o600)
    current = [sys.executable, str(_PROBE_PATH), "--platform", "rocm"]
    proof = _PROBE._validate_profile_argv(
        _profile_argv(telemetry, current), current, _REPO, _PROFILE_PATH.resolve()
    )
    assert proof["passed"] is True
    assert proof["direct_parent"] is True
    assert proof["profile_source_exact"] is True
    assert proof["telemetry_mode"] == 0o600
    assert all(proof["resource_checks"].values())
    assert proof["raw_argv_emitted"] is False
    assert proof["raw_paths_emitted"] is False


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--timeout", "300.001"),
        ("--interval", "0.101"),
        ("--sensor-grace-seconds", "60.1"),
        ("--max-junction-temp-c", "90.1"),
        ("--max-gpu-power-watts", "315.1"),
        ("--max-vram-gib", "24.1"),
        ("--min-host-available-gib", "-0.1"),
        ("--max-swap-gib", "8.1"),
        ("--card", "card0"),
    ],
)
def test_profile_argv_validator_rejects_each_relaxed_bound(
    tmp_path: Path, flag: str, value: str
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text("{}\n", encoding="utf-8")
    telemetry.chmod(0o600)
    current = [sys.executable, str(_PROBE_PATH), "--platform", "rocm"]
    parent = _profile_argv(telemetry, current)
    parent[parent.index(flag) + 1] = value
    with pytest.raises(RuntimeError, match="resource contract"):
        _PROBE._validate_profile_argv(parent, current, _REPO, _PROFILE_PATH.resolve())


def test_profile_argv_rejects_attach_duration_duplicate_flag_and_public_telemetry(
    tmp_path: Path,
) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text("{}\n", encoding="utf-8")
    telemetry.chmod(0o600)
    current = [sys.executable, str(_PROBE_PATH), "--platform", "rocm"]

    with_duration = _profile_argv(telemetry, current)
    with_duration[with_duration.index("--") : with_duration.index("--")] = [
        "--duration",
        "1",
    ]
    with pytest.raises(RuntimeError, match="resource contract"):
        _PROBE._validate_profile_argv(
            with_duration, current, _REPO, _PROFILE_PATH.resolve()
        )

    duplicate = _profile_argv(telemetry, current)
    duplicate[duplicate.index("--") : duplicate.index("--")] = ["--timeout", "1"]
    with pytest.raises(RuntimeError, match="exactly once"):
        _PROBE._validate_profile_argv(
            duplicate, current, _REPO, _PROFILE_PATH.resolve()
        )

    telemetry.chmod(0o640)
    with pytest.raises(RuntimeError, match="not private"):
        _PROBE._validate_profile_argv(
            _profile_argv(telemetry, current), current, _REPO, _PROFILE_PATH.resolve()
        )


def test_boot_seal_detects_a_current_boot_change(tmp_path: Path) -> None:
    boot_path = tmp_path / "boot_id"
    first = "01234567-89ab-cdef-0123-456789abcdef"
    second = "fedcba98-7654-3210-fedc-ba9876543210"
    boot_path.write_text(first + "\n", encoding="ascii")
    seal = _PROBE._BootSeal(boot_path)
    assert seal.check() == {
        "same_current_boot": True,
        "boot_id_sha256": hashlib.sha256(first.encode()).hexdigest(),
        "raw_boot_id_emitted": False,
    }
    boot_path.write_text(second + "\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="changed"):
        seal.check()


def test_host_prepare_execute_oracles_reproduce_exact_hashes_norms_and_contract(
    host_reference: tuple[
        tuple[Any, ...], tuple[Any, Any], dict[str, Any], dict[str, Any]
    ],
) -> None:
    boundary, reference, inputs, outputs = host_reference
    assert inputs["passed"] is True
    assert inputs["individual_sha256"] == _PROBE._EXPECTED_INPUT_SHA256
    assert inputs["framed_tuple_sha256"] == _PROBE._EXPECTED_INPUT_TUPLE_SHA256
    assert inputs["norms_f64"] == _PROBE._EXPECTED_INPUT_NORMS
    assert all(inputs["checks"].values())
    assert outputs["passed"] is True
    assert outputs["individual_sha256"] == _PROBE._EXPECTED_REFERENCE_SHA256
    assert outputs["framed_tuple_sha256"] == _PROBE._EXPECTED_REFERENCE_TUPLE_SHA256
    assert outputs["norms_f64"] == _PROBE._EXPECTED_REFERENCE_NORMS
    assert all(outputs["checks"].values())
    assert tuple(item.shape for item in boundary) == _PROBE._INPUT_SHAPES
    assert sum(item.nbytes for item in boundary) == 27_328_512
    assert tuple(item.shape for item in reference) == (
        _PROBE._PREPARED_SHAPE,
        _PROBE._STATE_SHAPE,
    )
    assert sum(item.nbytes for item in reference) == 6_291_456


def test_tuple_framing_is_domain_name_order_shape_and_payload_sensitive() -> None:
    first = np.arange(12, dtype=np.float32).reshape(3, 4)
    second = np.arange(5, dtype=np.float32)
    items = (("first", first), ("second", second))
    baseline = _PROBE._framed_tuple_sha256(b"domain-a\x00", items)
    assert baseline != _PROBE._framed_tuple_sha256(b"domain-b\x00", items)
    assert baseline != _PROBE._framed_tuple_sha256(
        b"domain-a\x00", (("second", second), ("first", first))
    )
    assert baseline != _PROBE._framed_tuple_sha256(
        b"domain-a\x00", (("renamed", first), ("second", second))
    )
    assert baseline != _PROBE._framed_tuple_sha256(
        b"domain-a\x00", (("first", first.reshape(2, 6)), ("second", second))
    )
    changed = first.copy()
    changed[0, 0] = 1
    assert baseline != _PROBE._framed_tuple_sha256(
        b"domain-a\x00", (("first", changed), ("second", second))
    )


def test_stablehlo_exact_typed_ffi_and_layout_passes() -> None:
    summary = _PROBE._ir_summary(_stablehlo(), "stablehlo")
    assert summary["passed"] is True
    assert summary["custom_call_count"] == 1
    assert summary["target_call_count"] == 1
    assert all(summary["checks"].values())
    assert all(summary["typed_abi"]["checks"].values())


def test_stablehlo_nested_region_cannot_own_the_sole_call() -> None:
    lines = _stablehlo().splitlines()
    call_index = next(
        index for index, line in enumerate(lines) if "stablehlo.custom_call" in line
    )
    return_index = next(
        index for index, line in enumerate(lines) if line.lstrip().startswith("return")
    )
    for index in range(call_index, return_index):
        if lines[index]:
            lines[index] = "  " + lines[index]
    lines.insert(call_index, "    {")
    lines.insert(return_index + 1, "    }")
    summary = _PROBE._ir_summary("\n".join(lines), "stablehlo")
    assert summary["passed"] is False
    assert (
        summary["entry_ownership"]["checks"][
            "custom_call_is_directly_owned_by_function"
        ]
        is False
    )


@pytest.mark.parametrize(
    ("dialect", "real_ir", "comment_spoof"),
    (
        (
            "stablehlo",
            _stablehlo(),
            "%quoted = stablehlo.custom_call @skyrl_gdn_execute_s512_f32_bf16_v1()",
        ),
        (
            "optimized_hlo",
            _optimized(),
            "ROOT %quoted = f32[1]{0} custom-call()",
        ),
    ),
)
def test_ir_call_discovery_ignores_multiline_block_comment_spoofs(
    dialect: str, real_ir: str, comment_spoof: str
) -> None:
    text = f"/* before\n{comment_spoof}\nafter */\n{real_ir}"
    summary = _PROBE._ir_summary(text, dialect)
    assert summary["passed"] is True
    assert summary["custom_call_count"] == 1


@pytest.mark.parametrize(
    ("dialect", "text"),
    (
        (
            "stablehlo",
            _stablehlo(target="wrong_target").replace(
                "    operand_layouts =",
                "    // stablehlo.custom_call "
                "@skyrl_gdn_execute_s512_f32_bf16_v1\n"
                "    operand_layouts =",
                1,
            ),
        ),
        (
            "optimized_hlo",
            _optimized(
                target="wrong_target", extra='  // custom_call_target="' + _TARGET + '"'
            ),
        ),
    ),
)
def test_target_identity_cannot_be_supplied_by_a_comment(
    dialect: str, text: str
) -> None:
    assert _PROBE._ir_summary(text, dialect)["passed"] is False


@pytest.mark.parametrize(
    ("text", "message"),
    (
        ('quoted_payload = "unterminated\n', "multiline quoted string"),
        ("/* unterminated\n", "unterminated block comment"),
        ("*/ orphan\n", "orphan block-comment close"),
        ("/* outer /* nested */\n", "nested block comment"),
    ),
)
def test_ir_lexer_rejects_malformed_quotes_and_comments(
    text: str, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _PROBE._ir_summary(text + _stablehlo(), "stablehlo")


@pytest.mark.parametrize(
    "text",
    [
        _stablehlo(input_shape="1x511x16x128"),
        _stablehlo(target="wrong_target"),
        _stablehlo(alias=", output_to_operand_aliasing = [0]"),
        _stablehlo(extra="  %loop = stablehlo.while(%arg0)"),
        _stablehlo().replace(
            "dense<[2,1,0]> : tensor<3xindex>",
            "dense<[1,2,0]> : tensor<3xindex>",
            1,
        ),
        _stablehlo(extra=_stablehlo().splitlines()[2]),
        _stablehlo().replace("%arg0, %arg1", "%copied, %arg1", 1),
    ],
)
def test_stablehlo_gate_rejects_wrong_abi_alias_loop_layout_or_call_count(
    text: str,
) -> None:
    assert _PROBE._ir_summary(text, "stablehlo")["passed"] is False


def test_optimized_hlo_exact_typed_ffi_and_layout_passes() -> None:
    summary = _PROBE._ir_summary(_optimized(), "optimized_hlo")
    assert summary["passed"] is True
    assert summary["custom_call_count"] == 1
    assert summary["target_call_count"] == 1
    assert all(summary["checks"].values())
    assert all(summary["typed_abi"]["checks"].values())


def test_optimized_hlo_nested_region_cannot_own_the_sole_call() -> None:
    lines = _optimized().splitlines()
    call_index = next(
        index for index, line in enumerate(lines) if "custom-call(" in line
    )
    lines[call_index] = "  " + lines[call_index]
    lines.insert(call_index, "  {")
    lines.insert(call_index + 2, "  }")
    summary = _PROBE._ir_summary("\n".join(lines), "optimized_hlo")
    assert summary["passed"] is False
    assert (
        summary["entry_ownership"]["checks"]["custom_call_is_directly_owned_by_entry"]
        is False
    )


@pytest.mark.parametrize(
    "text",
    [
        _optimized(query_dtype="bf16"),
        _optimized(target="wrong_target"),
        _optimized(alias=", output_to_operand_aliasing={[0]}"),
        _optimized(extra="  %loop = f32[1]{0} while(%query)"),
        _optimized(extra=_optimized().splitlines()[-3]),
        _optimized()
        + "\n%helper {\n"
        + "  %query = f32[1,512,16,128]{3,2,1,0} parameter(0)\n"
        + "}",
    ],
)
def test_optimized_hlo_gate_rejects_wrong_abi_alias_loop_or_call_count(
    text: str,
) -> None:
    assert _PROBE._ir_summary(text, "optimized_hlo")["passed"] is False


def test_compiled_memory_gate_accepts_exact_boundary_and_rejects_relaxation() -> None:
    exact = {
        "argument_size_in_bytes": 27_328_512,
        "output_size_in_bytes": 6_291_472,
        "alias_size_in_bytes": 0,
        "temp_size_in_bytes": 64 * 1024**2,
        "generated_code_size_in_bytes": 4096,
    }
    passed = _PROBE._compiled_memory_gate(exact)
    assert passed["passed"] is True
    assert passed["combined_argument_output_temporary_bytes"] == 100_728_848

    for change in (
        {"argument_size_in_bytes": 27_328_511},
        {"output_size_in_bytes": 6_291_456},
        {"alias_size_in_bytes": 1},
        {"temp_size_in_bytes": 64 * 1024**2 + 1},
        {"temp_size_in_bytes": None},
    ):
        assert _PROBE._compiled_memory_gate({**exact, **change})["passed"] is False


def test_checked_capability_requires_both_proofs_and_refuses_second_invocation() -> (
    None
):
    counters = _PROBE._zero_counters()

    def compiled(*_inputs: Any) -> tuple[str, str]:
        return "output", "state"

    with pytest.raises(RuntimeError, match="unchecked"):
        _PROBE._release_checked(compiled, {"passed": False}, {"passed": True}, counters)
    executable = _PROBE._release_checked(
        compiled, {"passed": True}, {"passed": True}, counters
    )
    jax = SimpleNamespace(block_until_ready=lambda value: value)
    with pytest.raises(RuntimeError, match="six inputs"):
        executable.invoke(jax, (1, 2))
    assert executable.invoke(jax, (1, 2, 3, 4, 5, 6)) == ("output", "state")
    assert counters["checked_executable_attempts"] == 1
    assert counters["checked_executable_completions"] == 1
    assert counters["output_readiness_barriers"] == 1
    with pytest.raises(RuntimeError, match="already consumed"):
        executable.invoke(jax, (1, 2, 3, 4, 5, 6))


def test_numerical_validation_accepts_exact_reference_and_duration_classes(
    host_reference: tuple[
        tuple[Any, ...], tuple[Any, Any], dict[str, Any], dict[str, Any]
    ],
) -> None:
    reference = host_reference[1]
    fast = _PROBE._validate_actual(np, reference, reference, 0.249)
    assert fast["passed"] is True
    assert fast["promotion_passed"] is True
    assert fast["classification"] == "promotable"
    assert fast["actual_framed_tuple_sha256"] == _PROBE._EXPECTED_REFERENCE_TUPLE_SHA256
    for metric in fast["metrics"].values():
        assert metric["relative_l2"] == 0
        assert metric["cosine"] == 1
        assert metric["cosine_raw"] == pytest.approx(1, abs=3e-15)

    slow = _PROBE._validate_actual(np, reference, reference, 0.25)
    assert slow["passed"] is True
    assert slow["promotion_passed"] is False
    assert slow["classification"] == "completed_unpromotable"


@pytest.mark.parametrize("output_index", [0, 1])
def test_numerical_validation_rejects_each_perturbed_output(
    host_reference: tuple[
        tuple[Any, ...], tuple[Any, Any], dict[str, Any], dict[str, Any]
    ],
    output_index: int,
) -> None:
    reference = host_reference[1]
    corrupt = [item.copy() for item in reference]
    corrupt[output_index].reshape(-1)[-1] += np.float32(0.01)
    with pytest.raises(RuntimeError, match="numerical"):
        _PROBE._validate_actual(np, reference, tuple(corrupt), 0.1)


@pytest.mark.parametrize("seconds", [2.0, float("inf"), float("nan"), -0.1])
def test_numerical_validation_rejects_invalid_or_hard_duration(
    host_reference: tuple[
        tuple[Any, ...], tuple[Any, Any], dict[str, Any], dict[str, Any]
    ],
    seconds: float,
) -> None:
    reference = host_reference[1]
    with pytest.raises(RuntimeError, match="hard-duration"):
        _PROBE._validate_actual(np, reference, reference, seconds)


def test_compile_diagnostic_fake_backend_cannot_construct_or_invoke_runtime_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    library, digest = _make_library(tmp_path)
    monkeypatch.setattr(_PROBE, "_EXPECTED_LIBRARY_SHA256", digest)
    monkeypatch.setattr(_PROBE, "_EXPECTED_LIBRARY_SIZE_BYTES", library.stat().st_size)
    boot_path = tmp_path / "boot_id"
    boot_path.write_text("01234567-89ab-cdef-0123-456789abcdef\n", encoding="ascii")
    compiled = _FakeCompiled()
    lowered = _FakeLowered(compiled)
    jax = _FakeJax(lowered)
    jaxlib = SimpleNamespace(__version__="0.10.2")
    jax_backend = SimpleNamespace(
        get_backend=lambda: SimpleNamespace(platform_version="ROCm 7.2.4")
    )

    def registration(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return _registration(library, digest, library.stat().st_size)

    dependencies = (
        jax,
        SimpleNamespace(float32=np.float32),
        jaxlib,
        jax_backend,
        lambda *_inputs, **_kwargs: ("abstract-output", "abstract-state"),
        registration,
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("compile diagnostic crossed into host/device runtime work")

    for name in (
        "_load_oracles",
        "_construct_host_reference",
        "_tuple_device_put",
        "_dispatch",
        "_tuple_device_get",
        "_validate_actual",
    ):
        monkeypatch.setattr(_PROBE, name, forbidden)
    monkeypatch.setenv("XLA_FLAGS", _PROBE._COMMAND_BUFFER_FLAG)
    counters = _PROBE._zero_counters()
    output = io.StringIO()
    args = SimpleNamespace(
        compile_diagnostic=True,
        library=library,
        library_sha256=digest,
    )
    result = _PROBE._run_rocm_body(
        args,
        output,
        lambda: dict(_CLEAN),
        _PROBE._BootSeal(boot_path),
        counters,
        environment={"XLA_FLAGS_effective": _PROBE._COMMAND_BUFFER_FLAG},
        library_manifest=_PROBE._validate_library_path(library, digest),
        _dependencies=dependencies,
    )

    assert result == 0
    assert compiled.invocations == 0
    assert lowered.compile_calls == 1
    assert counters == _PROBE._completed_compile_diagnostic_counters()
    passed = next(
        record
        for record in _records(output)
        if record["record_type"] == "compile_diagnostic_gates_passed_pending_postcheck"
    )
    assert passed["host_inputs_constructed"] == 0
    assert passed["oracle_invocations"] == 0
    assert passed["device_transfers"] == 0
    assert passed["executable_invocations"] == 0
    assert passed["counters"] == _PROBE._completed_compile_diagnostic_counters()
