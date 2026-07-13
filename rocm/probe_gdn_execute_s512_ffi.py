#!/usr/bin/env python3
"""Fail-closed one-shot runtime gate for exact S512 GDN execute typed FFI.

The default ``abstract`` mode emits a refusal manifest without importing JAX,
NumPy, SkyRL's ROCm package, or a shared library.  The explicit ROCm mode is a
single-use numerical qualifier.  It requires direct ``profile_rocm.py``
supervision, a clean current boot, the global Qwen3.5 ROCm launch lock, the
exact gfx1100/0x744c ROCm stack, and a canonical private shared object whose
SHA-256 is supplied explicitly.

The separate ``--compile-diagnostic`` rung registers, lowers, compiles,
inspects, destroys, and stops without constructing host inputs or invoking an
executable.  The numerical rung repeats that compile gate, constructs the
exact deterministic FP32 boundary on the CPU, uses the committed prepare and
execute oracles before device placement, and invokes the checked executable
exactly once.  It blocks the two-result output tuple once and compares every
BF16 output and FP32 state element on the host.  Neither rung authorizes a
warmup, replay, command buffer, graph, model, VJP, device reference, or
benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import stat
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, ContextManager, TextIO

_CASE = "s512-execute-one-shot"
_TARGET = "skyrl_gdn_execute_s512_f32_bf16_v1"
_LIBRARY_BASENAME = "libskyrl_gdn_execute_s512_gfx1100.so"
_EXPECTED_LIBRARY_SHA256 = (
    "435487abce7299bf9a835840f195cb95f0a644804fc2c41843ba9f5621ebd53b"
)
_EXPECTED_LIBRARY_SIZE_BYTES = 80_128

_QUERY_SHAPE = (1, 512, 16, 128)
_PREPARED_SHAPE = (1, 512, 32, 128)
_GAMMA_SHAPE = (1, 512, 32)
_STATE_SHAPE = (1, 32, 128, 128)
_INPUT_SHAPES = (
    _QUERY_SHAPE,
    _QUERY_SHAPE,
    _PREPARED_SHAPE,
    _PREPARED_SHAPE,
    _GAMMA_SHAPE,
    _STATE_SHAPE,
)
_INPUT_NAMES = (
    "query",
    "key",
    "prepared_u",
    "prepared_w",
    "gamma",
    "initial_state",
)
_OUTPUT_NAMES = ("output", "final_state")

_ARGUMENT_BYTES = 27_328_512
_LOGICAL_OUTPUT_BYTES = 6_291_456
_TUPLE_POINTER_BYTES = 16
_COMPILER_OUTPUT_BYTES = 6_291_472
_MAX_TEMP_BYTES = 64 * 1024**2
_MAX_COMBINED_BYTES = 128 * 1024**2

_LOWER_HARD_SECONDS = 30.0
_COMPILE_HARD_SECONDS = 180.0
_RUNTIME_PROMOTION_SECONDS = 0.250
_RUNTIME_HARD_SECONDS = 1.0
_COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="
_REQUIRED_SNAPSHOT_SEALS = 0x000F
_CHECKED_CAPABILITY_TOKEN = object()

_EXPECTED_ROCM_VERSION = "7.2.4"
_EXPECTED_AMDGPU_VERSION = "6.16.13"
_EXPECTED_GFX_TARGET_VERSION = 110000
_EXPECTED_PCI_VENDOR_ID = "0x1002"
_EXPECTED_PCI_DEVICE_ID = "0x744c"
_EXPECTED_KFD_VENDOR_ID = 4098
_EXPECTED_KFD_DEVICE_ID = 29772
_EXPECTED_STACK_VERSIONS = {
    "jax": "0.10.2",
    "jaxlib": "0.10.2",
    "jax-rocm7-plugin": "0.10.2",
    "jax-rocm7-pjrt": "0.10.2",
    "numpy": "2.3.5",
    "ml_dtypes": "0.5.4",
}

_EXPECTED_SOURCE_SHA256 = {
    "execute_oracle": "cca11be2212603d74d52ce936910283805c1fa5e73c6bfbc5c79a98e529949c2",
    "prepare_oracle": "019da59345f04ed2bfea553d797deb1a1c15a34c11147780616d4b4a44195aad",
    "wrapper": "a588a1c17960172f07f6e504aa5dbaf5d8f9be91e3edf7775de0e632ab52851b",
    "hip": "f858c129cb781a9c9d4e87679973b454c7cf852b640f58ba13ad53e003059f36",
    "build": "25a1590df7f5ed89be983bb6c5a8649791eb4916becddd9d6f75919ab534821d",
    "sealed_loader": "66b868b7909a2279d5ddca0e1582f8563e8097723e970d09fce733aef2ba425a",
    "safety": "8b9441e0147b35b000fe8340ea7b7c702372320c569ced71410b92b558953e91",
    "profile_rocm": "ed230758101a2a540b3a09e7f84ac92256d2bb41c70dbc399b9466fe0b979684",
    "package_skyrl": "667d4a15b970b851e20d17510224670c14646cfb6d5a1e388ca6b9cc6da8bf41",
    "package_tx": "a7abb3e76d66df1f4472bb7a02b032ef31b959ca937fd351637b4e9b4a8fa95a",
    "package_kernels": "40abe638c7726fe5680b7c88321042016a0f695d86acfbef52337421e7257c1a",
    "package_rocm": "6d12a789cf1108538a04fbacd0b38a15dbcb8255cd0ca0fadf5a76c4191a4cfd",
}

_EXPECTED_INPUT_SHA256 = {
    "query": "558299f9f5a6c8ac7ef81187ce3b75c0cb4bccee176d19fde6f36c9193f19d14",
    "key": "8f54cb823b5d2f6311473dddb410f369bb1022afd300f309c5a0396793777dbe",
    "prepared_u": "53eb9a96b97593b3af7677fd350b9016fd5ab37df1dff08016fb928ccfaf829f",
    "prepared_w": "15811df05a34c1e8e8f09305eecb0e62639b9152a46728ecc334ca2caad85e0b",
    "gamma": "1a8c8a7d83cd03f2ec848646da7835d6bea59c2696c97edd64b523c08ded4dca",
    "initial_state": "b8b5b6cfc479df1fbc0dbe9ed743136613da386535b6e0a3034d62abf20d97f2",
}
_EXPECTED_REFERENCE_SHA256 = {
    "output": "383b59c47d65417dff22c60214c9acb017542cc7d92622a93fde6ca9f682f14b",
    "final_state": "1f008feffeb448fea3378cba54d994c4a0c1640fba6f168d898defea0f425cd1",
}
_EXPECTED_INPUT_NORMS = {
    "query": 7.999999864336225,
    "key": 90.50966798190066,
    "prepared_u": 0.6628828405579906,
    "prepared_w": 3.876961308461615,
    "gamma": 123.04377685419234,
    "initial_state": 1.2541158672624613,
}
_EXPECTED_REFERENCE_NORMS = {
    "output": 0.17290999555612815,
    "final_state": 0.7637801972452101,
}
_SENSITIVITY_MINIMA = {
    "wrong_hv_to_key_mapping": {"output": 1.0, "final_state": 0.80},
    "zero_initial_state": {"output": 0.80, "final_state": 0.70},
    "unit_gamma": {"output": 0.30, "final_state": 0.65},
    "reset_state_at_each_chunk": {"output": 0.45, "final_state": 0.80},
}
_INPUT_FRAMING_DOMAIN = b"skyrl-gdn-execute-s512-boundary-v1\x00"
_REFERENCE_FRAMING_DOMAIN = b"skyrl-gdn-execute-s512-reference-v1\x00"
_EXPECTED_INPUT_TUPLE_SHA256 = (
    "4d460fd9fb092d8a36146f5d2ff4b7e2fb3834c666b2a8ed21eed74c6c60e5ec"
)
_EXPECTED_REFERENCE_TUPLE_SHA256 = (
    "195c82753d98113f0b1cf18b922bd959949e9039146cf2fe6e0880a5c037392f"
)

_JOURNAL_STAGES = frozenset(
    {
        "before_host_oracle",
        "after_host_oracle_attempt",
        "before_backend_initialization",
        "after_backend_initialization_attempt",
        "after_ffi_registration_attempt",
        "after_ffi_lower_attempt",
        "after_ffi_compile_attempt",
        "after_input_device_put_attempt",
        "after_candidate_dispatch_attempt",
        "after_candidate_device_get_attempt",
        "after_host_validation_attempt",
        "after_library_postcheck",
    }
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _emit(record: dict[str, Any], output: TextIO) -> None:
    output.write(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    output.flush()


def _redacted_error(error: BaseException) -> dict[str, Any]:
    encoded = str(error).encode("utf-8", errors="replace")
    return {
        "message_redacted": True,
        "message_utf8_bytes": len(encoded),
        "message_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _source_files() -> dict[str, Path]:
    repo = Path(__file__).resolve().parent.parent
    rocm_package = repo / "skyrl" / "tx" / "kernels" / "rocm"
    return {
        "probe_source_sha256": Path(__file__),
        "execute_oracle_source_sha256": rocm_package / "gdn_execute_oracle.py",
        "prepare_oracle_source_sha256": rocm_package / "gdn_prepare_oracle.py",
        "wrapper_source_sha256": rocm_package / "gdn_execute_ffi.py",
        "hip_source_sha256": rocm_package / "ffi" / "gdn_execute_s512.hip",
        "build_source_sha256": rocm_package
        / "ffi"
        / "build_gdn_execute_s512_gfx1100.sh",
        "sealed_loader_source_sha256": rocm_package / "gdn_ffi_smoke.py",
        "safety_source_sha256": repo / "rocm" / "amdgpu_safety.py",
        "profile_rocm_source_sha256": repo / "rocm" / "profile_rocm.py",
        "package_skyrl_source_sha256": repo / "skyrl" / "__init__.py",
        "package_tx_source_sha256": repo / "skyrl" / "tx" / "__init__.py",
        "package_kernels_source_sha256": repo
        / "skyrl"
        / "tx"
        / "kernels"
        / "__init__.py",
        "package_rocm_source_sha256": rocm_package / "__init__.py",
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {name: _file_sha256(path) for name, path in _source_files().items()}


def _assert_bound_sources() -> dict[str, Any]:
    files = _source_files()
    observed = {
        "execute_oracle": _file_sha256(files["execute_oracle_source_sha256"]),
        "prepare_oracle": _file_sha256(files["prepare_oracle_source_sha256"]),
        "wrapper": _file_sha256(files["wrapper_source_sha256"]),
        "hip": _file_sha256(files["hip_source_sha256"]),
        "build": _file_sha256(files["build_source_sha256"]),
        "sealed_loader": _file_sha256(files["sealed_loader_source_sha256"]),
        "safety": _file_sha256(files["safety_source_sha256"]),
        "profile_rocm": _file_sha256(files["profile_rocm_source_sha256"]),
        "package_skyrl": _file_sha256(files["package_skyrl_source_sha256"]),
        "package_tx": _file_sha256(files["package_tx_source_sha256"]),
        "package_kernels": _file_sha256(files["package_kernels_source_sha256"]),
        "package_rocm": _file_sha256(files["package_rocm_source_sha256"]),
    }
    if observed != _EXPECTED_SOURCE_SHA256:
        raise RuntimeError("GDN execute runtime dependency source hash mismatch")
    return {
        "passed": True,
        "all_executable_dependencies_exact": True,
        **observed,
    }


def _sha256_argument(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise argparse.ArgumentTypeError(
            "must be exactly 64 lowercase hexadecimal digits"
        )
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("abstract", "rocm"), default="abstract")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--compile-diagnostic", action="store_true")
    parser.add_argument("--case", choices=(_CASE,))
    parser.add_argument("--library", type=Path)
    parser.add_argument("--library-sha256", type=_sha256_argument)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    guarded = (args.case, args.library, args.library_sha256, args.output)
    if args.platform == "abstract":
        if (
            args.allow_gpu
            or args.compile_diagnostic
            or any(value is not None for value in guarded)
        ):
            parser.error("guarded case/library/hash/output require --platform rocm")
        return args
    if not args.allow_gpu:
        parser.error("--platform rocm requires --allow-gpu")
    if args.case != _CASE:
        parser.error(f"--platform rocm requires --case {_CASE}")
    if args.library is None:
        parser.error("--platform rocm requires --library")
    if args.library_sha256 is None:
        parser.error("--platform rocm requires --library-sha256")
    if args.library_sha256 != _EXPECTED_LIBRARY_SHA256:
        parser.error("--library-sha256 must bind the exact qualified native artifact")
    if args.output is None:
        parser.error("--platform rocm requires --output")
    if not args.output.is_absolute():
        parser.error("--output must be absolute")
    try:
        parent = args.output.parent.resolve(strict=True)
    except OSError:
        parser.error("--output parent must already exist and resolve canonically")
    if args.output != parent / args.output.name:
        parser.error("--output must be canonical and must not contain symlinks or '..'")
    if args.output.exists() or args.output.is_symlink():
        parser.error("refusing to overwrite existing or symbolic-link output")
    return args


def _outer_profile_contract() -> dict[str, Any]:
    return {
        "profile_rocm_required": True,
        "direct_parent_and_source_hash_proven_by_child": True,
        "timeout_seconds_maximum": 300.0,
        "sampling_interval_seconds_maximum": 0.10,
        "sensor_grace_seconds_maximum": 60.0,
        "maximum_vram_gib": 24.0,
        "maximum_junction_temperature_c": 90.0,
        "maximum_gpu_power_watts": 315.0,
        "minimum_host_available_gib_minimum": 0.0,
        "maximum_swap_gib": 8.0,
        "card": "card1",
    }


def _exact_contract() -> dict[str, Any]:
    return {
        "operation": "gdn_execute_s512_typed_ffi_one_shot_numerical_runtime",
        "case": _CASE,
        "target": _TARGET,
        "native_artifact_contract": {
            "sha256": _EXPECTED_LIBRARY_SHA256,
            "size_bytes": _EXPECTED_LIBRARY_SIZE_BYTES,
            "mode": 0o600,
            "architecture": "gfx1100",
            "exact_exported_handler_symbol": _TARGET,
            "offline_build_metadata_is_claimed_outside_probe": True,
            "child_runs_native_inspection_tools": False,
        },
        "modes": {
            "compile_diagnostic": (
                "register/lower/compile/inspect/destroy with zero host or runtime work"
            ),
            "numerical_runtime": (
                "repeat compile gate then exactly one checked executable invocation"
            ),
        },
        "inputs": [
            {"name": name, "shape": list(shape), "dtype": "float32"}
            for name, shape in zip(_INPUT_NAMES, _INPUT_SHAPES, strict=True)
        ],
        "outputs": [
            {"name": "output", "shape": list(_PREPARED_SHAPE), "dtype": "bfloat16"},
            {"name": "final_state", "shape": list(_STATE_SHAPE), "dtype": "float32"},
        ],
        "host_oracle": {
            "prepare_oracle_calls": 1,
            "execute_oracle_calls": 1,
            "execute_sensitivity_oracle_calls": 11,
            "computed_before_device_placement": True,
            "full_reference_arrays_compared": True,
            "input_bytes": _ARGUMENT_BYTES,
            "reference_bytes": _LOGICAL_OUTPUT_BYTES,
        },
        "compile_gate": {
            "jit_lower_calls": 1,
            "compile_calls": 1,
            "stablehlo_gate": True,
            "optimized_hlo_gate": True,
            "exact_typed_ffi_custom_calls_per_dialect": 1,
            "while_calls": 0,
            "alias_bytes": 0,
            "argument_bytes": _ARGUMENT_BYTES,
            "logical_output_bytes": _LOGICAL_OUTPUT_BYTES,
            "tuple_pointer_bytes": _TUPLE_POINTER_BYTES,
            "compiler_output_bytes": _COMPILER_OUTPUT_BYTES,
            "temporary_bytes_maximum": _MAX_TEMP_BYTES,
            "combined_bytes_maximum": _MAX_COMBINED_BYTES,
            "lower_seconds_strictly_below": _LOWER_HARD_SECONDS,
            "compile_seconds_strictly_below": _COMPILE_HARD_SECONDS,
        },
        "invocation_contract": {
            "tuple_device_puts": 1,
            "input_leaves": 6,
            "input_readiness_barriers": 1,
            "checked_executable_invocations": 1,
            "output_readiness_barriers": 1,
            "tuple_device_gets": 1,
            "output_leaves": 2,
            "lowered_callable_invocations": 0,
            "warmups": 0,
            "replays": 0,
            "graphs": 0,
            "gpu_references": 0,
            "gpu_reductions": 0,
            "backward": 0,
            "model": 0,
        },
        "numerical_thresholds": {
            "output_relative_l2_lte": 5e-3,
            "output_cosine_gte": 0.9999,
            "output_max_absolute_lte": 5e-3,
            "state_relative_l2_lte": 2e-4,
            "state_cosine_gte": 0.999999,
            "state_max_absolute_lte": 2e-4,
        },
        "duration_gate": {
            "promotion_seconds_strictly_below": _RUNTIME_PROMOTION_SECONDS,
            "hard_seconds_strictly_below": _RUNTIME_HARD_SECONDS,
        },
        "scope_exclusions": {
            "warmup": False,
            "replay": False,
            "command_buffer": False,
            "graph": False,
            "vjp": False,
            "model": False,
            "training": False,
            "performance_benchmark": False,
        },
        "outer_supervision": _outer_profile_contract(),
    }


def _zero_counters() -> dict[str, int]:
    return {
        "host_boundary_arrays": 0,
        "prepare_oracle_attempts": 0,
        "prepare_oracle_completions": 0,
        "execute_oracle_attempts": 0,
        "execute_oracle_completions": 0,
        "sensitivity_oracle_invocations": 0,
        "backend_initialization_attempts": 0,
        "backend_initialization_completions": 0,
        "registration_attempts": 0,
        "registration_completions": 0,
        "shape_dtype_structs": 0,
        "ffi_python_trace_calls": 0,
        "lower_attempts": 0,
        "lower_completions": 0,
        "compile_attempts": 0,
        "compile_completions": 0,
        "checked_capability_release_attempts": 0,
        "checked_capability_release_completions": 0,
        "tuple_device_put_attempts": 0,
        "tuple_device_put_completions": 0,
        "device_put_leaves": 0,
        "input_readiness_barriers": 0,
        "checked_executable_attempts": 0,
        "checked_executable_completions": 0,
        "output_readiness_barriers": 0,
        "tuple_device_get_attempts": 0,
        "tuple_device_get_completions": 0,
        "device_get_leaves": 0,
        "lowered_callable_invocations": 0,
        "warmup_invocations": 0,
        "replay_invocations": 0,
        "graph_invocations": 0,
        "gpu_reference_invocations": 0,
        "gpu_reduction_invocations": 0,
        "backward_invocations": 0,
        "model_invocations": 0,
    }


def _completed_counters() -> dict[str, int]:
    result = _zero_counters()
    result.update(
        {
            "host_boundary_arrays": 6,
            "prepare_oracle_attempts": 1,
            "prepare_oracle_completions": 1,
            "execute_oracle_attempts": 1,
            "execute_oracle_completions": 1,
            "sensitivity_oracle_invocations": 11,
            "backend_initialization_attempts": 1,
            "backend_initialization_completions": 1,
            "registration_attempts": 1,
            "registration_completions": 1,
            "shape_dtype_structs": 6,
            "ffi_python_trace_calls": 1,
            "lower_attempts": 1,
            "lower_completions": 1,
            "compile_attempts": 1,
            "compile_completions": 1,
            "checked_capability_release_attempts": 1,
            "checked_capability_release_completions": 1,
            "tuple_device_put_attempts": 1,
            "tuple_device_put_completions": 1,
            "device_put_leaves": 6,
            "input_readiness_barriers": 1,
            "checked_executable_attempts": 1,
            "checked_executable_completions": 1,
            "output_readiness_barriers": 1,
            "tuple_device_get_attempts": 1,
            "tuple_device_get_completions": 1,
            "device_get_leaves": 2,
        }
    )
    return result


def _completed_compile_diagnostic_counters() -> dict[str, int]:
    result = _zero_counters()
    result.update(
        {
            "backend_initialization_attempts": 1,
            "backend_initialization_completions": 1,
            "registration_attempts": 1,
            "registration_completions": 1,
            "shape_dtype_structs": 6,
            "ffi_python_trace_calls": 1,
            "lower_attempts": 1,
            "lower_completions": 1,
            "compile_attempts": 1,
            "compile_completions": 1,
        }
    )
    return result


def _assert_fresh_accelerator_process() -> None:
    imported = sorted(
        name
        for name in sys.modules
        if name in {"jax", "jaxlib", "numpy", "skyrl.tx.kernels.rocm"}
        or name.startswith("jax.")
        or name.startswith("jaxlib.")
        or name.startswith("numpy.")
        or name.startswith("skyrl.tx.kernels.rocm.")
    )
    if imported:
        raise RuntimeError("GDN execute runtime gate requires a fresh process")


def _load_exact_file_module(path: Path, expected_sha256: str, name: str) -> ModuleType:
    if _file_sha256(path) != expected_sha256:
        raise RuntimeError(f"refusing changed {name} source")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not create exact {name} module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not isinstance(getattr(module, "__file__", None), str):
        raise RuntimeError(f"exact {name} module has no source identity")
    if Path(module.__file__).resolve() != path.resolve():
        raise RuntimeError(f"loaded {name} from an unexpected path")
    if _file_sha256(path) != expected_sha256:
        raise RuntimeError(f"{name} source changed while loading")
    return module


def _load_oracles() -> tuple[ModuleType, ModuleType]:
    files = _source_files()
    prepare = _load_exact_file_module(
        files["prepare_oracle_source_sha256"],
        _EXPECTED_SOURCE_SHA256["prepare_oracle"],
        "_skyrl_exact_gdn_prepare_oracle_for_execute_gate",
    )
    execute = _load_exact_file_module(
        files["execute_oracle_source_sha256"],
        _EXPECTED_SOURCE_SHA256["execute_oracle"],
        "_skyrl_exact_gdn_execute_oracle_gate",
    )
    return prepare, execute


def _validate_host_numeric_environment() -> dict[str, str]:
    expected = {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    if any(os.environ.get(name) != value for name, value in expected.items()):
        raise RuntimeError("GDN execute oracle requires all BLAS thread caps exactly 1")
    return expected


def _validate_exact_or_unset(name: str, expected: str) -> None:
    observed = os.environ.get(name)
    if observed is not None and observed != expected:
        raise RuntimeError(f"{name} conflicts with exact GDN execute environment")


def _configure_rocm_environment() -> dict[str, str]:
    fixed = {
        "JAX_PLATFORMS": "rocm",
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
        "JAX_ROCM_VISIBLE_DEVICES": "0",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_CLIENT_MEM_FRACTION": "0.10",
    }
    for name, expected in fixed.items():
        observed = os.environ.get(name)
        if name == "XLA_PYTHON_CLIENT_PREALLOCATE" and observed is not None:
            if observed.strip().lower() not in {"0", "false", "no", "off"}:
                raise RuntimeError("preallocation conflicts with bounded execute gate")
            continue
        _validate_exact_or_unset(name, expected)
    inherited = os.environ.get("XLA_FLAGS")
    if inherited is not None:
        try:
            tokens = shlex.split(inherited, posix=True)
        except ValueError as error:
            raise RuntimeError("invalid inherited XLA_FLAGS quoting") from error
        if tokens != [_COMMAND_BUFFER_FLAG]:
            raise RuntimeError(
                "XLA_FLAGS must contain solely the disabled command-buffer flag"
            )
    for name in (
        "HSA_OVERRIDE_GFX_VERSION",
        "XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB",
        "JAX_PJRT_CLIENT_CREATE_OPTIONS",
        "JAX_MOCK_GPU_TOPOLOGY",
        "TF_FORCE_UNIFIED_MEMORY",
    ):
        if os.environ.get(name) not in {None, ""}:
            raise RuntimeError(f"{name} must be unset for physical execute gate")
        os.environ.pop(name, None)
    if os.environ.get("MOCK_NUM_GPU_PROCESSES", "").strip() not in {"", "0"}:
        raise RuntimeError("MOCK_NUM_GPU_PROCESSES must be unset or zero")
    os.environ.pop("MOCK_NUM_GPU_PROCESSES", None)
    os.environ.update(fixed)
    os.environ["XLA_FLAGS"] = _COMMAND_BUFFER_FLAG
    return {**fixed, "XLA_FLAGS_effective": _COMMAND_BUFFER_FLAG}


def _prove_command_buffers_disabled(environment: dict[str, str]) -> dict[str, Any]:
    effective = environment.get("XLA_FLAGS_effective")
    if effective != _COMMAND_BUFFER_FLAG or os.environ.get("XLA_FLAGS") != effective:
        raise RuntimeError("command buffers are not proven disabled")
    if shlex.split(effective, posix=True) != [_COMMAND_BUFFER_FLAG]:
        raise RuntimeError("effective XLA_FLAGS contains an unauthorized token")
    return {
        "passed": True,
        "command_buffers_disabled": True,
        "sole_xla_flag": True,
        "effective_xla_flags_sha256": hashlib.sha256(effective.encode()).hexdigest(),
        "raw_xla_flags_emitted": False,
    }


def _validate_library_path(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_absolute():
        raise ValueError("GDN execute library path must be absolute")
    if path.name != _LIBRARY_BASENAME:
        raise ValueError("GDN execute library must use the exact audited basename")
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError("GDN execute library cannot be inspected") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("GDN execute library must be a real regular file")
    if before.st_uid != os.getuid():
        raise ValueError("GDN execute library must be owned by the current user")
    if stat.S_IMODE(before.st_mode) != 0o600:
        raise ValueError("GDN execute library mode must be exactly 0600")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError("GDN execute library cannot be resolved") from error
    if resolved != path:
        raise ValueError("GDN execute library path must be canonical and symlink-free")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise RuntimeError("GDN execute library changed while being opened")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    after = path.lstat()
    identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
    )
    if identity != (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
    ):
        raise RuntimeError("GDN execute library changed while being hashed")
    observed_sha256 = digest.hexdigest()
    if expected_sha256 != _EXPECTED_LIBRARY_SHA256:
        raise RuntimeError("GDN execute CLI SHA-256 is not the qualified artifact hash")
    if observed_sha256 != expected_sha256:
        raise RuntimeError("GDN execute library SHA-256 does not match")
    if int(after.st_size) != _EXPECTED_LIBRARY_SIZE_BYTES:
        raise RuntimeError(
            "GDN execute library size is not the qualified artifact size"
        )
    return {
        "validated": True,
        "canonical": True,
        "symlink_free": True,
        "owner_exact": True,
        "mode": 0o600,
        "basename": _LIBRARY_BASENAME,
        "size_bytes": int(after.st_size),
        "sha256": observed_sha256,
        "identity": identity,
        "raw_path_emitted": False,
    }


def _assert_same_library(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    current = _validate_library_path(path, str(manifest["sha256"]))
    if tuple(current["identity"]) != tuple(manifest["identity"]):
        raise RuntimeError("GDN execute library identity changed after registration")
    return current


def _read_proc_argv(pid: int) -> list[str]:
    try:
        payload = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError as error:
        raise RuntimeError("could not inspect profile_rocm parent command") from error
    if not payload or not payload.endswith(b"\x00"):
        raise RuntimeError(
            "profile_rocm parent command line is unavailable or malformed"
        )
    try:
        values = [part.decode("utf-8") for part in payload[:-1].split(b"\x00")]
    except UnicodeDecodeError as error:
        raise RuntimeError("profile_rocm parent command line is not UTF-8") from error
    if not values or any(not value for value in values):
        raise RuntimeError(
            "profile_rocm parent command line contains an empty argument"
        )
    return values


def _profile_flag(prefix: list[str], name: str) -> str:
    matches: list[str] = []
    for index, value in enumerate(prefix):
        if value == name:
            if index + 1 >= len(prefix):
                raise RuntimeError(f"profile supervisor flag {name} has no value")
            matches.append(prefix[index + 1])
        elif value.startswith(name + "="):
            matches.append(value.split("=", 1)[1])
    if len(matches) != 1 or not matches[0]:
        raise RuntimeError(f"profile supervisor must set {name} exactly once")
    return matches[0]


def _finite_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise RuntimeError(f"profile supervisor {name} is not numeric") from error
    if not math.isfinite(parsed):
        raise RuntimeError(f"profile supervisor {name} is not finite")
    return parsed


def _resolve_argument_path(value: str, cwd: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    try:
        return candidate.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(
            "profile supervisor contains an unresolvable path"
        ) from error


def _validate_profile_argv(
    parent_argv: list[str],
    current_argv: list[str],
    parent_cwd: Path,
    profile_path: Path,
) -> dict[str, Any]:
    delimiters = [index for index, value in enumerate(parent_argv) if value == "--"]
    if len(delimiters) != 1:
        raise RuntimeError("profile supervisor requires exactly one command delimiter")
    delimiter = delimiters[0]
    prefix = parent_argv[:delimiter]
    child = parent_argv[delimiter + 1 :]
    if len(child) != len(current_argv) or len(child) < 2:
        raise RuntimeError("profile supervisor child argv does not match this process")
    profile_matches = [
        index
        for index, value in enumerate(prefix)
        if value.endswith("profile_rocm.py")
        and _resolve_argument_path(value, parent_cwd) == profile_path
    ]
    if len(profile_matches) != 1:
        raise RuntimeError("direct parent is not the exact profile_rocm source")
    if _resolve_argument_path(child[0], parent_cwd) != Path(current_argv[0]).resolve():
        raise RuntimeError("profile supervisor child interpreter identity mismatch")
    if _resolve_argument_path(child[1], parent_cwd) != Path(current_argv[1]).resolve():
        raise RuntimeError("profile supervisor child probe identity mismatch")
    if child[2:] != current_argv[2:]:
        raise RuntimeError(
            "profile supervisor child arguments differ from this process"
        )

    values = {
        "timeout": _finite_float(_profile_flag(prefix, "--timeout"), "timeout"),
        "interval": _finite_float(_profile_flag(prefix, "--interval"), "interval"),
        "sensor_grace": _finite_float(
            _profile_flag(prefix, "--sensor-grace-seconds"), "sensor grace"
        ),
        "temperature": _finite_float(
            _profile_flag(prefix, "--max-junction-temp-c"), "temperature"
        ),
        "power": _finite_float(_profile_flag(prefix, "--max-gpu-power-watts"), "power"),
        "vram": _finite_float(_profile_flag(prefix, "--max-vram-gib"), "VRAM"),
        "host": _finite_float(
            _profile_flag(prefix, "--min-host-available-gib"), "host memory"
        ),
        "swap": _finite_float(_profile_flag(prefix, "--max-swap-gib"), "swap"),
    }
    checks = {
        "timeout": 0 < values["timeout"] <= 300.0,
        "interval": 0 < values["interval"] <= 0.10,
        # Runtime constructs and validates the complete CPU oracle before it
        # initializes ROCm.  A headless runtime-suspended card therefore has
        # no readable thermal sensor during that intentional CPU-only phase.
        "sensor_grace": 0 <= values["sensor_grace"] <= 60.0,
        "temperature": 0 < values["temperature"] <= 90.0,
        "power": 0 < values["power"] <= 315.0,
        "vram": 0 < values["vram"] <= 24.0,
        "host": values["host"] >= 0,
        "swap": 0 <= values["swap"] <= 8.0,
        "card": _profile_flag(prefix, "--card") == "card1",
        "not_attach_duration": "--duration" not in prefix
        and not any(value.startswith("--duration=") for value in prefix),
    }
    if not all(checks.values()):
        raise RuntimeError("profile supervisor resource contract is not bounded")
    telemetry = Path(_profile_flag(prefix, "--output"))
    if not telemetry.is_absolute():
        telemetry = parent_cwd / telemetry
    try:
        telemetry = telemetry.resolve(strict=True)
        info = telemetry.stat()
    except OSError as error:
        raise RuntimeError(
            "profile supervisor telemetry artifact is unavailable"
        ) from error
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        raise RuntimeError("profile supervisor telemetry artifact is not private")
    return {
        "passed": True,
        "direct_parent": True,
        "profile_source_sha256": _file_sha256(profile_path),
        "profile_source_exact": _file_sha256(profile_path)
        == _EXPECTED_SOURCE_SHA256["profile_rocm"],
        "child_argv_sha256": hashlib.sha256(
            b"\x00".join(value.encode() for value in child)
        ).hexdigest(),
        "telemetry_sha256_at_preflight": _file_sha256(telemetry),
        "telemetry_mode": 0o600,
        "resource_values": values,
        "resource_checks": checks,
        "raw_argv_emitted": False,
        "raw_paths_emitted": False,
    }


def _prove_profile_supervision() -> dict[str, Any]:
    parent_pid = os.getppid()
    parent_argv = _read_proc_argv(parent_pid)
    try:
        parent_cwd = Path(f"/proc/{parent_pid}/cwd").resolve(strict=True)
    except OSError as error:
        raise RuntimeError("could not resolve profile_rocm parent cwd") from error
    proof = _validate_profile_argv(
        parent_argv,
        [sys.executable, *sys.argv],
        parent_cwd,
        _source_files()["profile_rocm_source_sha256"].resolve(),
    )
    if proof["profile_source_exact"] is not True:
        raise RuntimeError("profile_rocm supervisor source hash mismatch")
    return proof


def _read_exact_text(path: Path, expected: str, label: str) -> str:
    try:
        observed = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise RuntimeError(f"could not read exact {label}") from error
    if observed != expected:
        raise RuntimeError(f"exact {label} mismatch")
    return observed


def _parse_kfd_properties(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError("could not read KFD topology properties") from error
    for line in lines:
        pieces = line.split()
        if len(pieces) == 2 and re.fullmatch(r"[a-z_]+", pieces[0]):
            try:
                result[pieces[0]] = int(pieces[1])
            except ValueError:
                continue
    return result


def _hardware_stack_preflight() -> dict[str, Any]:
    from importlib import metadata

    rocm = _read_exact_text(
        Path("/opt/rocm/.info/version"), _EXPECTED_ROCM_VERSION, "ROCm version"
    )
    amdgpu = _read_exact_text(
        Path("/sys/module/amdgpu/version"),
        _EXPECTED_AMDGPU_VERSION,
        "AMDGPU module version",
    )
    vendor = _read_exact_text(
        Path("/sys/class/drm/card1/device/vendor"),
        _EXPECTED_PCI_VENDOR_ID,
        "PCI vendor ID",
    )
    device = _read_exact_text(
        Path("/sys/class/drm/card1/device/device"),
        _EXPECTED_PCI_DEVICE_ID,
        "PCI device ID",
    )
    matching_nodes = []
    for path in sorted(Path("/sys/class/kfd/kfd/topology/nodes").glob("*/properties")):
        properties = _parse_kfd_properties(path)
        if (
            properties.get("vendor_id") == _EXPECTED_KFD_VENDOR_ID
            and properties.get("device_id") == _EXPECTED_KFD_DEVICE_ID
            and properties.get("gfx_target_version") == _EXPECTED_GFX_TARGET_VERSION
        ):
            matching_nodes.append(path.parent.name)
    if len(matching_nodes) != 1:
        raise RuntimeError("expected exactly one gfx1100/0x744c KFD topology node")
    versions = {}
    for name, expected in _EXPECTED_STACK_VERSIONS.items():
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"required ROCm stack package {name} is missing"
            ) from error
        if versions[name] != expected:
            raise RuntimeError(f"exact ROCm stack package {name} version mismatch")
    return {
        "passed": True,
        "gfx": "gfx1100",
        "gfx_target_version": _EXPECTED_GFX_TARGET_VERSION,
        "pci_vendor_id": vendor,
        "pci_device_id": device,
        "kfd_node_count": 1,
        "rocm_version": rocm,
        "amdgpu_version": amdgpu,
        "package_versions": versions,
    }


class _BootSeal:
    __slots__ = ("_path", "_boot_id")

    def __init__(self, path: Path = Path("/proc/sys/kernel/random/boot_id")) -> None:
        self._path = path
        self._boot_id = self._read()

    def _read(self) -> str:
        try:
            value = self._path.read_text(encoding="ascii").strip()
        except OSError as error:
            raise RuntimeError("could not read current Linux boot identity") from error
        if re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value) is None:
            raise RuntimeError("current Linux boot identity is malformed")
        return value

    def check(self) -> dict[str, Any]:
        if self._read() != self._boot_id:
            raise RuntimeError("Linux boot identity changed during execute gate")
        return {
            "same_current_boot": True,
            "boot_id_sha256": hashlib.sha256(self._boot_id.encode()).hexdigest(),
            "raw_boot_id_emitted": False,
        }


def _load_safety_helpers() -> tuple[
    Callable[[], ContextManager[dict[str, Any]]], Callable[[], dict[str, Any]]
]:
    try:
        from rocm.amdgpu_safety import (
            guarded_qwen35_rocm_process,
            require_clean_amdgpu_boot,
        )
    except ModuleNotFoundError:
        from amdgpu_safety import (  # type: ignore[no-redef]
            guarded_qwen35_rocm_process,
            require_clean_amdgpu_boot,
        )
    return guarded_qwen35_rocm_process, require_clean_amdgpu_boot


def _public_clean_safety(safety: Any, stage: str) -> dict[str, Any]:
    if not isinstance(safety, dict) or safety.get("amdgpu_boot_clean") is not True:
        raise RuntimeError(f"{stage} did not prove a clean AMDGPU boot")
    if safety.get("fatal_amdgpu_events") != []:
        raise RuntimeError(f"{stage} returned invalid fatal-event evidence")
    return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}


def _public_safety_preflight(safety: Any) -> dict[str, Any]:
    public = _public_clean_safety(safety, "safety_preflight")
    cards = safety.get("amd_cards")
    if cards != ["card1"]:
        raise RuntimeError("safety preflight did not prove exact AMD card1 ownership")
    if safety.get("connected_amd_connectors") != []:
        raise RuntimeError("safety preflight did not prove every AMD connector idle")
    if safety.get("kfd_path") != "/dev/kfd":
        raise RuntimeError("safety preflight did not prove exact KFD path")
    if (
        safety.get("kfd_accessible") is not True
        or safety.get("kfd_unowned") is not True
    ):
        raise RuntimeError("safety preflight did not prove accessible unowned KFD")
    return {
        **public,
        "amd_cards": ["card1"],
        "connected_amd_connectors": [],
        "kfd_path": "/dev/kfd",
        "kfd_accessible": True,
        "kfd_unowned": True,
    }


def _journal_checkpoint(
    require_clean_boot: Callable[[], dict[str, Any]],
    boot: _BootSeal,
    output: TextIO,
    stage: str,
    counters: dict[str, int],
) -> None:
    if stage not in _JOURNAL_STAGES:
        raise RuntimeError("refusing an undeclared GDN execute journal stage")
    _emit(
        {
            "record_type": "journal_checkpoint",
            "timestamp": _utc_now(),
            "stage": stage,
            "safety": _public_clean_safety(require_clean_boot(), stage),
            "boot": boot.check(),
            "counters": dict(counters),
        },
        output,
    )


def _array_sha256(array: Any) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _array_norm(np: Any, array: Any) -> float:
    value = np.asarray(array, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(value))


def _framed_tuple_sha256(domain: bytes, items: tuple[tuple[str, Any], ...]) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    for name, array in items:
        metadata = (
            name
            + "\x00"
            + str(array.dtype.str)
            + "\x00"
            + ",".join(str(int(dimension)) for dimension in array.shape)
        ).encode("utf-8")
        payload = array.tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _splitmix64_f32(np: Any, shape: tuple[int, ...], seed: int) -> Any:
    indices = np.arange(math.prod(shape), dtype=np.uint64)
    z = (indices ^ np.uint64(seed)) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    values = (z >> np.uint64(40)).astype(np.uint32).astype(np.float32) * np.float32(
        2.0 / 16_777_216.0
    ) - np.float32(1.0)
    return values.reshape(shape)


def _normalize_rows(np: Any, raw: Any, scale: float) -> Any:
    norm = np.sqrt(np.sum(raw * raw, axis=-1, dtype=np.float32) + np.float32(1e-6))
    return np.ascontiguousarray(
        raw / norm[..., None] * np.float32(scale), dtype=np.float32
    )


def _norms_exact(observed: dict[str, float], expected: dict[str, float]) -> bool:
    return observed.keys() == expected.keys() and all(
        math.isclose(observed[name], expected[name], rel_tol=1e-12, abs_tol=1e-12)
        for name in expected
    )


def _relative_l2(np: Any, reference: Any, alternative: Any) -> float:
    reference64 = np.asarray(reference, dtype=np.float64)
    alternative64 = np.asarray(alternative, dtype=np.float64)
    denominator = float(np.linalg.norm(reference64.reshape(-1)))
    if not math.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("GDN sensitivity denominator is not finite and nonzero")
    return float(
        np.linalg.norm((alternative64 - reference64).reshape(-1)) / denominator
    )


def _host_sensitivity_report(
    np: Any,
    execute_oracle: Callable[..., Any],
    boundary: tuple[Any, ...],
    reference: tuple[Any, Any],
    counters: dict[str, int],
) -> dict[str, Any]:
    query, key, prepared_u, prepared_w, gamma, initial_state = boundary

    def run(*inputs: Any) -> tuple[Any, Any]:
        counters["sensitivity_oracle_invocations"] += 1
        result = execute_oracle(*inputs, output_bfloat16=True)
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("GDN sensitivity oracle returned invalid outputs")
        return result

    variants: dict[str, tuple[Any, Any]] = {
        # Rolling both Q and K head axes makes the fixed hv//2 read the adjacent
        # query/key head, exactly exercising the value-to-key-head ownership rule.
        "wrong_hv_to_key_mapping": run(
            np.ascontiguousarray(np.roll(query, 1, axis=2)),
            np.ascontiguousarray(np.roll(key, 1, axis=2)),
            prepared_u,
            prepared_w,
            gamma,
            initial_state,
        ),
        "zero_initial_state": run(
            query,
            key,
            prepared_u,
            prepared_w,
            gamma,
            np.zeros_like(initial_state),
        ),
        "unit_gamma": run(
            query,
            key,
            prepared_u,
            prepared_w,
            np.ones_like(gamma),
            initial_state,
        ),
    }
    reset_output = np.empty_like(reference[0])
    reset_final_state = None
    for chunk_index in range(8):
        token_slice = slice(chunk_index * 64, (chunk_index + 1) * 64)
        isolated_query = np.zeros_like(query)
        isolated_key = np.zeros_like(key)
        isolated_u = np.zeros_like(prepared_u)
        isolated_w = np.zeros_like(prepared_w)
        isolated_gamma = np.ones_like(gamma)
        isolated_query[:, token_slice] = query[:, token_slice]
        isolated_key[:, token_slice] = key[:, token_slice]
        isolated_u[:, token_slice] = prepared_u[:, token_slice]
        isolated_w[:, token_slice] = prepared_w[:, token_slice]
        isolated_gamma[:, token_slice] = gamma[:, token_slice]
        isolated_output, isolated_state = run(
            isolated_query,
            isolated_key,
            isolated_u,
            isolated_w,
            isolated_gamma,
            initial_state,
        )
        reset_output[:, token_slice] = isolated_output[:, token_slice]
        if chunk_index == 7:
            reset_final_state = isolated_state
    if reset_final_state is None:
        raise RuntimeError("reset-at-each-chunk sensitivity was not constructed")
    variants["reset_state_at_each_chunk"] = (reset_output, reset_final_state)

    metrics = {
        name: {
            "output_relative_l2": _relative_l2(np, reference[0], outputs[0]),
            "final_state_relative_l2": _relative_l2(np, reference[1], outputs[1]),
            "output_sha256": _array_sha256(outputs[0]),
            "final_state_sha256": _array_sha256(outputs[1]),
        }
        for name, outputs in variants.items()
    }
    checks = {}
    for name, item in metrics.items():
        minimum = _SENSITIVITY_MINIMA[name]
        checks[f"{name}_output_material"] = (
            item["output_relative_l2"] >= minimum["output"]
        )
        checks[f"{name}_state_material"] = (
            item["final_state_relative_l2"] >= minimum["final_state"]
        )
        checks[f"{name}_output_hash_differs"] = (
            item["output_sha256"] != _EXPECTED_REFERENCE_SHA256["output"]
        )
        checks[f"{name}_state_hash_differs"] = (
            item["final_state_sha256"] != _EXPECTED_REFERENCE_SHA256["final_state"]
        )
    checks["exact_sensitivity_oracle_calls"] = (
        counters["sensitivity_oracle_invocations"] == 11
    )
    if not all(checks.values()):
        raise RuntimeError("GDN execute host case failed semantic sensitivity gates")
    return {
        "passed": True,
        "minimum_required_relative_l2": _SENSITIVITY_MINIMA,
        "metrics": metrics,
        "checks": checks,
        "raw_tensors_emitted": False,
    }


def _construct_host_reference(
    np: Any,
    prepare_oracle: Callable[..., Any],
    execute_oracle: Callable[..., Any],
    counters: dict[str, int],
) -> tuple[tuple[Any, ...], tuple[Any, Any], dict[str, Any], dict[str, Any]]:
    query = _normalize_rows(
        np,
        _splitmix64_f32(np, _QUERY_SHAPE, 0x243F6A8885A308D3),
        1.0 / math.sqrt(128),
    )
    key = _normalize_rows(
        np,
        _splitmix64_f32(np, _QUERY_SHAPE, 0x13198A2E03707344),
        1.0,
    )
    value = np.ascontiguousarray(
        _splitmix64_f32(np, _PREPARED_SHAPE, 0xA4093822299F31D0) * np.float32(0.025),
        dtype=np.float32,
    )
    token = np.arange(512, dtype=np.int32)[:, None]
    head = np.arange(32, dtype=np.int32)[None, :]
    chunk = token // 64
    g = np.ascontiguousarray(
        -(
            np.float32(0.0008)
            + np.float32(0.0001) * (token % 7)
            + np.float32(0.00005) * (head % 5)
            + np.float32(0.000025) * ((chunk + head) % 4)
        )[None, ...],
        dtype=np.float32,
    )
    beta = np.ascontiguousarray(
        (
            np.float32(0.008)
            + np.float32(0.042)
            * (
                ((3 * token + 5 * head + 7 * chunk) % 17).astype(np.float32)
                / np.float32(16)
            )
        )[None, ...],
        dtype=np.float32,
    )
    initial_state = np.ascontiguousarray(
        _splitmix64_f32(np, _STATE_SHAPE, 0x082EFA98EC4E6C89) * np.float32(0.003),
        dtype=np.float32,
    )

    counters["prepare_oracle_attempts"] += 1
    prepare_started = time.perf_counter()
    prepared_u, prepared_w, gamma = prepare_oracle(key, value, g, beta)
    prepare_seconds = time.perf_counter() - prepare_started
    counters["prepare_oracle_completions"] += 1
    boundary = (query, key, prepared_u, prepared_w, gamma, initial_state)
    counters["host_boundary_arrays"] += len(boundary)
    hashes = {
        name: _array_sha256(array)
        for name, array in zip(_INPUT_NAMES, boundary, strict=True)
    }
    norms = {
        name: _array_norm(np, array)
        for name, array in zip(_INPUT_NAMES, boundary, strict=True)
    }
    framed = _framed_tuple_sha256(
        _INPUT_FRAMING_DOMAIN,
        tuple(zip(_INPUT_NAMES, boundary, strict=True)),
    )
    query_row_norm = np.sqrt(np.sum(query * query, axis=-1, dtype=np.float32))
    key_row_norm = np.sqrt(np.sum(key * key, axis=-1, dtype=np.float32))
    input_checks = {
        "six_arrays": len(boundary) == 6,
        "shapes_exact": tuple(item.shape for item in boundary) == _INPUT_SHAPES,
        "dtypes_exact": all(item.dtype == np.dtype(np.float32) for item in boundary),
        "c_contiguous": all(bool(item.flags.c_contiguous) for item in boundary),
        "all_finite": all(bool(np.all(np.isfinite(item))) for item in boundary),
        "bytes_exact": sum(int(item.nbytes) for item in boundary) == _ARGUMENT_BYTES,
        "pairwise_disjoint": all(
            not np.shares_memory(boundary[left], boundary[right])
            for left in range(len(boundary))
            for right in range(left + 1, len(boundary))
        ),
        "individual_hashes_exact": hashes == _EXPECTED_INPUT_SHA256,
        "framed_tuple_hash_exact": framed == _EXPECTED_INPUT_TUPLE_SHA256,
        "norms_exact": _norms_exact(norms, _EXPECTED_INPUT_NORMS),
        "query_norm_exact_scale": float(
            np.max(np.abs(query_row_norm - np.float32(1.0 / math.sqrt(128))))
        )
        <= 2e-8,
        "key_norm_unit": float(np.max(np.abs(key_row_norm - np.float32(1.0)))) <= 2e-7,
        "gamma_strictly_positive": bool(np.all(gamma > 0)),
        "gamma_stable_minimum": float(np.min(gamma)) >= 0.90,
        "gamma_below_or_equal_one": float(np.max(gamma)) <= 1.0,
        "initial_state_nonzero_signed": float(np.min(initial_state)) < -0.0029
        and float(np.max(initial_state)) > 0.0029,
    }
    if not all(input_checks.values()):
        raise RuntimeError("deterministic GDN execute boundary failed identity gates")
    before_reference = dict(hashes)

    counters["execute_oracle_attempts"] += 1
    reference_started = time.perf_counter()
    reference = execute_oracle(*boundary, output_bfloat16=True)
    reference_seconds = time.perf_counter() - reference_started
    counters["execute_oracle_completions"] += 1
    if not isinstance(reference, tuple) or len(reference) != 2:
        raise RuntimeError("GDN execute oracle did not return exactly two arrays")
    reference_hashes = {
        name: _array_sha256(array)
        for name, array in zip(_OUTPUT_NAMES, reference, strict=True)
    }
    reference_norms = {
        name: _array_norm(np, array)
        for name, array in zip(_OUTPUT_NAMES, reference, strict=True)
    }
    reference_framed = _framed_tuple_sha256(
        _REFERENCE_FRAMING_DOMAIN,
        tuple(zip(_OUTPUT_NAMES, reference, strict=True)),
    )
    boundary_unchanged = before_reference == {
        name: _array_sha256(array)
        for name, array in zip(_INPUT_NAMES, boundary, strict=True)
    }
    reference_checks = {
        "two_arrays": len(reference) == 2,
        "shapes_exact": tuple(item.shape for item in reference)
        == (_PREPARED_SHAPE, _STATE_SHAPE),
        "dtypes_exact": str(reference[0].dtype) == "bfloat16"
        and reference[1].dtype == np.dtype(np.float32),
        "c_contiguous": all(bool(item.flags.c_contiguous) for item in reference),
        "all_finite": all(bool(np.all(np.isfinite(item))) for item in reference),
        "bytes_exact": sum(int(item.nbytes) for item in reference)
        == _LOGICAL_OUTPUT_BYTES,
        "individual_hashes_exact": reference_hashes == _EXPECTED_REFERENCE_SHA256,
        "framed_tuple_hash_exact": reference_framed == _EXPECTED_REFERENCE_TUPLE_SHA256,
        "norms_exact": _norms_exact(reference_norms, _EXPECTED_REFERENCE_NORMS),
        "boundary_unmodified": boundary_unchanged,
        "output_nonzero_signed": float(np.min(reference[0])) < 0
        and float(np.max(reference[0])) > 0,
        "state_nonzero_signed": float(np.min(reference[1])) < 0
        and float(np.max(reference[1])) > 0,
    }
    if not all(reference_checks.values()):
        raise RuntimeError("committed GDN execute reference failed identity gates")
    sensitivity = _host_sensitivity_report(
        np, execute_oracle, boundary, reference, counters
    )
    input_report = {
        "passed": True,
        "construction": "splitmix64_plus_committed_prepare_oracle_v1",
        "prepare_oracle_seconds": prepare_seconds,
        "individual_sha256": hashes,
        "framed_tuple_sha256": framed,
        "norms_f64": norms,
        "ranges": {
            "gamma": [float(np.min(gamma)), float(np.max(gamma))],
            "initial_state": [
                float(np.min(initial_state)),
                float(np.max(initial_state)),
            ],
        },
        "checks": input_checks,
        "raw_tensors_emitted": False,
    }
    reference_report = {
        "passed": True,
        "execute_oracle_seconds": reference_seconds,
        "individual_sha256": reference_hashes,
        "framed_tuple_sha256": reference_framed,
        "norms_f64": reference_norms,
        "ranges": {
            "output": [float(np.min(reference[0])), float(np.max(reference[0]))],
            "final_state": [
                float(np.min(reference[1])),
                float(np.max(reference[1])),
            ],
        },
        "checks": reference_checks,
        "sensitivities": sensitivity,
        "raw_tensors_emitted": False,
    }
    return boundary, reference, input_report, reference_report


def _mask_quoted_strings(text: str) -> str:
    masked = list(text)
    state = "plain"
    escaped = False
    index = 0
    while index < len(text):
        character = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if state == "quoted":
            if character != "\n":
                masked[index] = " "
            else:
                raise RuntimeError("compiler IR contains a multiline quoted string")
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                masked[index] = '"'
                state = "plain"
        elif state == "line_comment":
            if character == "\n":
                state = "plain"
            else:
                masked[index] = " "
        elif state == "block_comment":
            if character != "\n":
                masked[index] = " "
            if character == "/" and following == "*":
                raise RuntimeError("compiler IR contains a nested block comment")
            if character == "*" and following == "/":
                masked[index + 1] = " "
                state = "plain"
                index += 1
        elif character == '"':
            state = "quoted"
            masked[index] = '"'
        elif character == "/" and following == "/":
            state = "line_comment"
            masked[index] = masked[index + 1] = " "
            index += 1
        elif character == "/" and following == "*":
            state = "block_comment"
            masked[index] = masked[index + 1] = " "
            index += 1
        elif character == "*" and following == "/":
            raise RuntimeError("compiler IR contains an orphan block-comment close")
        index += 1
    if state == "quoted":
        raise RuntimeError("compiler IR contains an unterminated quoted string")
    if state == "block_comment":
        raise RuntimeError("compiler IR contains an unterminated block comment")
    return "".join(masked)


def _stablehlo_entry_ownership(masked: str, call_position: int) -> dict[str, Any]:
    functions = list(re.finditer(r"\bfunc\.func\b", masked))
    depth = 0
    malformed = False
    if len(functions) == 1 and call_position > functions[0].end():
        for character in masked[functions[0].end() : call_position]:
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth < 0:
                    malformed = True
                    break
    checks = {
        "exactly_one_function": len(functions) == 1,
        "custom_call_is_inside_function": depth > 0 and not malformed,
        "custom_call_is_directly_owned_by_function": depth == 1 and not malformed,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "function_count": len(functions),
        "brace_depth_at_custom_call": depth,
    }


def _optimized_entry_scope(
    text: str, masked: str, call_position: int | None
) -> tuple[str, dict[str, Any]]:
    header = re.compile(
        r"^(?P<indent>\s*)(?P<entry>ENTRY\s+)?"
        r"(?P<name>%?[A-Za-z_][A-Za-z0-9_.-]*)\b(?P<tail>.*)\{\s*$",
        re.MULTILINE,
    )
    computations: list[tuple[re.Match[str], int, int]] = []
    for match in header.finditer(masked):
        if "=" in match.group("tail"):
            continue
        opening = masked.rfind("{", match.start(), match.end())
        depth = 0
        closing = None
        for index in range(opening, len(masked)):
            if masked[index] == "{":
                depth += 1
            elif masked[index] == "}":
                depth -= 1
                if depth == 0:
                    closing = index
                    break
                if depth < 0:
                    break
        if closing is not None:
            computations.append((match, opening, closing))
    entries = [item for item in computations if item[0].group("entry") is not None]
    scope = text[entries[0][0].start() : entries[0][2] + 1] if len(entries) == 1 else ""
    call_depth = 0
    call_inside_entry = False
    if len(entries) == 1 and call_position is not None:
        _match, opening, closing = entries[0]
        call_inside_entry = opening < call_position < closing
        if call_inside_entry:
            for character in masked[opening:call_position]:
                if character == "{":
                    call_depth += 1
                elif character == "}":
                    call_depth -= 1
    checks = {
        "exactly_one_computation": len(computations) == 1,
        "exactly_one_entry_computation": len(entries) == 1,
        "sole_computation_is_entry": len(computations) == 1 and len(entries) == 1,
        "custom_call_is_inside_entry": call_inside_entry,
        "custom_call_is_directly_owned_by_entry": call_depth == 1,
    }
    public = {
        "passed": all(checks.values()),
        "checks": checks,
        "computation_count": len(computations),
        "entry_count": len(entries),
        "brace_depth_at_custom_call": call_depth,
        "raw_computation_emitted": False,
    }
    return scope, public


def _custom_call_blocks(text: str, dialect: str) -> list[str]:
    raw_lines = text.splitlines()
    masked_lines = _mask_quoted_strings(text).splitlines()
    if len(raw_lines) != len(masked_lines):
        raise RuntimeError("IR quote masking changed the line count")
    if dialect == "stablehlo":
        start = re.compile(r"^(?P<indent>\s*)%[^=]+?=\s*stablehlo\.custom_call\b")
        boundary = re.compile(
            r"^\s*(?:%[^=]+?=|#[A-Za-z_][\w.-]*\s*=|(?:stablehlo\.|func\.)?return\b|}\s*$)"
        )
    elif dialect == "optimized_hlo":
        start = re.compile(
            r"^(?P<indent>\s*)(?:ROOT\s+)?[^=]+?=\s*.*\bcustom-call\s*\("
        )
        boundary = re.compile(r"^\s*(?:(?:ROOT\s+)?[^=]+?=|}\s*$)")
    else:
        raise ValueError("unsupported IR dialect")
    blocks: list[str] = []
    index = 0
    while index < len(masked_lines):
        match = start.search(masked_lines[index])
        if match is None:
            index += 1
            continue
        base_indent = len(match.group("indent").expandtabs())
        block_lines = [raw_lines[index]]
        index += 1
        while index < len(masked_lines):
            candidate = masked_lines[index]
            candidate_indent = len(candidate) - len(candidate.lstrip(" \t"))
            if (
                candidate.strip()
                and candidate_indent <= base_indent
                and boundary.match(candidate)
            ):
                break
            block_lines.append(raw_lines[index])
            index += 1
        blocks.append("\n".join(block_lines))
    return blocks


def _unquoted_attribute_values(block: str, name: str) -> list[str]:
    structural = _mask_quoted_strings(block)
    values: list[str] = []
    for match in re.finditer(rf"\b{re.escape(name)}\b\s*=\s*\"", structural):
        opening = structural.find('"', match.start(), match.end())
        closing = structural.find('"', opening + 1)
        if opening < 0 or closing < 0:
            raise RuntimeError("compiler IR target attribute is malformed")
        value = block[opening + 1 : closing]
        if re.fullmatch(r"[A-Za-z0-9_.$-]+", value) is None:
            raise RuntimeError("compiler IR target attribute is not canonical")
        values.append(value)
    return values


def _custom_call_targets(block: str, dialect: str) -> list[str]:
    structural = _mask_quoted_strings(block)
    if dialect == "optimized_hlo":
        return _unquoted_attribute_values(block, "custom_call_target")
    if dialect != "stablehlo":
        raise ValueError("unsupported IR dialect")
    targets: list[str] = []
    pattern = re.compile(r"\bstablehlo\.custom_call\s+@\s*")
    for match in pattern.finditer(structural):
        position = match.end()
        if position < len(structural) and structural[position] == '"':
            closing = structural.find('"', position + 1)
            if closing < 0:
                raise RuntimeError("StableHLO custom-call target is malformed")
            value = block[position + 1 : closing]
        else:
            bare = re.match(r"[A-Za-z0-9_.$-]+", structural[position:])
            value = bare.group(0) if bare is not None else ""
        if re.fullmatch(r"[A-Za-z0-9_.$-]+", value) is None:
            raise RuntimeError("StableHLO custom-call target is not canonical")
        targets.append(value)
    return targets


def _integer_sequence(value: str) -> tuple[int, ...] | None:
    pieces = [piece.strip() for piece in value.split(",")]
    if not pieces or any(re.fullmatch(r"[0-9]+", piece) is None for piece in pieces):
        return None
    return tuple(int(piece) for piece in pieces)


def _balanced_square_value(text: str, after_assignment: int) -> tuple[str, int] | None:
    opening = after_assignment
    while opening < len(text) and text[opening].isspace():
        opening += 1
    if opening >= len(text) or text[opening] != "[":
        return None
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "[":
            depth += 1
        elif text[index] == "]":
            depth -= 1
            if depth == 0:
                return text[opening : index + 1], index + 1
            if depth < 0:
                return None
    return None


def _dense_layout_list(value: str) -> tuple[tuple[int, ...], ...] | None:
    if len(value) < 2 or value[0] != "[" or value[-1] != "]":
        return None
    inner = value[1:-1]
    if not inner.strip():
        return ()
    position = 0
    layouts: list[tuple[int, ...]] = []
    entry = re.compile(
        r"\s*dense<\[\s*(?P<values>[0-9,\s]+?)\s*\]>\s*"
        r"(?::\s*tensor<\s*(?P<annotation>[0-9]+)\s*x\s*"
        r"(?:index|i64)\s*>)?\s*"
    )
    while position < len(inner):
        match = entry.match(inner, position)
        if match is None:
            return None
        layout = _integer_sequence(match.group("values"))
        if layout is None:
            return None
        annotation = match.group("annotation")
        if annotation is not None and int(annotation) != len(layout):
            return None
        layouts.append(layout)
        position = match.end()
        if position == len(inner):
            break
        if inner[position] != ",":
            return None
        position += 1
        if not inner[position:].strip():
            return None
    return tuple(layouts)


def _layout_attribute(block: str, name: str) -> tuple[tuple[int, ...], ...] | None:
    matches = list(re.finditer(rf"\b{re.escape(name)}\b\s*=", block))
    if len(matches) != 1:
        return None
    value = _balanced_square_value(block, matches[0].end())
    return _dense_layout_list(value[0]) if value is not None else None


def _stablehlo_proof(block: str) -> dict[str, Any]:
    block = _mask_quoted_strings(block)
    tensor = r"tensor<[^>]+>"
    signature_pattern = re.compile(
        rf":\s*\((?P<inputs>{tensor}(?:\s*,\s*{tensor})*)\)\s*->\s*"
        rf"(?:(?:tuple<|\()(?P<outputs>{tensor}(?:\s*,\s*{tensor})*)(?:>|\))|"
        rf"(?P<single>{tensor}))",
        re.DOTALL,
    )
    signatures = list(signature_pattern.finditer(block))
    signature = signatures[-1] if signatures else None
    inputs = (
        tuple(re.findall(tensor, signature.group("inputs")))
        if signature is not None
        else ()
    )
    if signature is None:
        outputs: tuple[str, ...] = ()
    elif signature.group("outputs") is not None:
        outputs = tuple(re.findall(tensor, signature.group("outputs")))
    else:
        outputs = (str(signature.group("single")),)
    operand_match = re.search(
        r"\bstablehlo\.custom_call\s+@(?:\"[^\"]+\"|[A-Za-z0-9_.$-]+)"
        r"\((?P<operands>[^)]*)\)",
        block,
        re.DOTALL,
    )
    operand_names = (
        re.findall(r"%[A-Za-z0-9_.$-]+", operand_match.group("operands"))
        if operand_match is not None
        else []
    )
    input_layouts = _layout_attribute(block, "operand_layouts")
    output_layouts = _layout_attribute(block, "result_layouts")
    expected_inputs = (
        "tensor<1x512x16x128xf32>",
        "tensor<1x512x16x128xf32>",
        "tensor<1x512x32x128xf32>",
        "tensor<1x512x32x128xf32>",
        "tensor<1x512x32xf32>",
        "tensor<1x32x128x128xf32>",
    )
    expected_outputs = (
        "tensor<1x512x32x128xbf16>",
        "tensor<1x32x128x128xf32>",
    )
    expected_input_layouts = (
        (3, 2, 1, 0),
        (3, 2, 1, 0),
        (3, 2, 1, 0),
        (3, 2, 1, 0),
        (2, 1, 0),
        (3, 2, 1, 0),
    )
    expected_output_layouts = ((3, 2, 1, 0), (3, 2, 1, 0))
    checks = {
        "one_typed_signature": len(signatures) == 1,
        "six_distinct_call_operands": len(operand_names) == 6
        and len(set(operand_names)) == 6,
        "direct_entry_arguments_in_order": operand_names
        == [f"%arg{index}" for index in range(6)],
        "input_types_exact": inputs == expected_inputs,
        "output_types_exact": outputs == expected_outputs,
        "input_layouts_exact": input_layouts == expected_input_layouts,
        "output_layouts_exact": output_layouts == expected_output_layouts,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "input_types": list(inputs),
        "output_types": list(outputs),
        "input_layouts": [list(item) for item in (input_layouts or ())],
        "output_layouts": [list(item) for item in (output_layouts or ())],
    }


def _optimized_hlo_proof(block: str, full_text: str) -> dict[str, Any]:
    block = _mask_quoted_strings(block)
    instruction = re.search(
        r"=\s*(?P<outputs>\([^)]*\))\s*custom-call\s*\("
        r"(?P<inputs>[^)]*)\)\s*,",
        block,
        re.DOTALL,
    )
    if instruction is None:
        operand_names: list[str] = []
        output_text = ""
    else:
        operand_names = [
            part.strip() for part in instruction.group("inputs").split(",")
        ]
        output_text = instruction.group("outputs")
    operand_names_exact = len(operand_names) == 6 and all(
        re.fullmatch(r"%[A-Za-z_][A-Za-z0-9_.$-]*", item) is not None
        for item in operand_names
    )
    masked = _mask_quoted_strings(full_text)
    definition_pattern = re.compile(
        r"^\s*(?:ROOT\s+)?(?P<name>%[A-Za-z_][A-Za-z0-9_.$-]*)\s*=\s*"
        r"(?P<dtype>[A-Za-z0-9_]+)\[\s*(?P<shape>[0-9,\s]+)\s*\]"
        r"\{\s*(?P<layout>[0-9,\s]+)\s*\}\s+parameter\(\s*(?P<index>[0-9]+)\s*\)",
        re.MULTILINE,
    )
    expected_inputs = (
        ("f32", (1, 512, 16, 128), (3, 2, 1, 0)),
        ("f32", (1, 512, 16, 128), (3, 2, 1, 0)),
        ("f32", (1, 512, 32, 128), (3, 2, 1, 0)),
        ("f32", (1, 512, 32, 128), (3, 2, 1, 0)),
        ("f32", (1, 512, 32), (2, 1, 0)),
        ("f32", (1, 32, 128, 128), (3, 2, 1, 0)),
    )
    observed_inputs = []
    unique_definitions = True
    if operand_names_exact:
        for index, name in enumerate(operand_names):
            matches = [
                match
                for match in definition_pattern.finditer(masked)
                if match.group("name") == name
            ]
            if len(matches) != 1 or int(matches[0].group("index")) != index:
                unique_definitions = False
                continue
            match = matches[0]
            observed_inputs.append(
                (
                    match.group("dtype"),
                    _integer_sequence(match.group("shape")),
                    _integer_sequence(match.group("layout")),
                )
            )
    shape_pattern = re.compile(
        r"(?P<dtype>[A-Za-z0-9_]+)\[\s*(?P<shape>[0-9,\s]+)\s*\]"
        r"\{\s*(?P<layout>[0-9,\s]+)\s*\}"
    )
    observed_outputs = tuple(
        (
            match.group("dtype"),
            _integer_sequence(match.group("shape")),
            _integer_sequence(match.group("layout")),
        )
        for match in shape_pattern.finditer(output_text)
    )
    expected_outputs = (
        ("bf16", (1, 512, 32, 128), (3, 2, 1, 0)),
        ("f32", (1, 32, 128, 128), (3, 2, 1, 0)),
    )
    checks = {
        "six_operand_references": operand_names_exact,
        "unique_parameter_definitions_in_order": unique_definitions
        and len(observed_inputs) == 6,
        "input_types_and_layouts_exact": tuple(observed_inputs) == expected_inputs,
        "output_types_and_layouts_exact": observed_outputs == expected_outputs,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "operand_reference_count": len(operand_names),
        "observed_inputs": [
            [dtype, list(shape or ()), list(layout or ())]
            for dtype, shape, layout in observed_inputs
        ],
        "observed_outputs": [
            [dtype, list(shape or ()), list(layout or ())]
            for dtype, shape, layout in observed_outputs
        ],
    }


def _has_nonempty_aliasing(text: str) -> bool:
    masked = _mask_quoted_strings(text)
    for match in re.finditer(
        r"\b(?:output_to_operand_aliasing|input_output_aliases?)\b\s*=\s*"
        r"(?P<value>\[[^\]]*\]|\{[^}]*\})",
        masked,
        re.DOTALL,
    ):
        value = match.group("value")
        if value not in {"[]", "{}"}:
            return True
    return bool(re.search(r"\binput_output_alias\b", masked))


def _ir_summary(text: str, dialect: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("compiler returned empty IR")
    masked = _mask_quoted_strings(text)
    opcode_pattern = (
        r"\bstablehlo\.custom_call\b"
        if dialect == "stablehlo"
        else r"\bcustom-call\s*\("
        if dialect == "optimized_hlo"
        else None
    )
    if opcode_pattern is None:
        raise ValueError("unsupported IR dialect")
    opcode_positions = [match.start() for match in re.finditer(opcode_pattern, masked)]
    blocks = _custom_call_blocks(text, dialect)
    targets_per_block = [_custom_call_targets(block, dialect) for block in blocks]
    target_blocks = [
        block
        for block, targets in zip(blocks, targets_per_block, strict=True)
        if targets == [_TARGET]
    ]
    other_targets = sorted(
        target
        for targets in targets_per_block
        for target in targets
        if target != _TARGET
    )
    optimized_scope = ""
    ownership = (
        _stablehlo_entry_ownership(masked, opcode_positions[0])
        if dialect == "stablehlo" and len(opcode_positions) == 1
        else None
    )
    if dialect == "optimized_hlo":
        optimized_scope, ownership = _optimized_entry_scope(
            text, masked, opcode_positions[0] if len(opcode_positions) == 1 else None
        )
    if ownership is None:
        ownership = {"passed": False, "checks": {"one_opcode": False}}
    proof = (
        _stablehlo_proof(target_blocks[0])
        if dialect == "stablehlo" and len(target_blocks) == 1
        else _optimized_hlo_proof(target_blocks[0], optimized_scope)
        if dialect == "optimized_hlo" and len(target_blocks) == 1
        else {"passed": False, "checks": {"one_target_block": False}}
    )
    checks = {
        "exactly_one_unquoted_custom_call_opcode": len(opcode_positions) == 1,
        "block_parser_matches_unquoted_opcode_count": len(blocks)
        == len(opcode_positions),
        "exactly_one_custom_call_total": len(blocks) == 1,
        "exactly_one_exact_target_call": len(target_blocks) == 1,
        "exactly_one_target_attribute_per_call": all(
            len(targets) == 1 for targets in targets_per_block
        ),
        "no_other_custom_call_targets": other_targets == [],
        "no_while_opcode": re.search(r"\bwhile\s*\(", masked) is None,
        "no_input_output_aliasing": not _has_nonempty_aliasing(text),
        "direct_entry_ownership": ownership["passed"],
        "typed_abi_and_layout": proof["passed"],
    }
    return {
        "dialect": dialect,
        "passed": all(checks.values()),
        "checks": checks,
        "custom_call_count": len(blocks),
        "target_call_count": len(target_blocks),
        "other_target_count": len(other_targets),
        "entry_ownership": ownership,
        "typed_abi": proof,
        "ir_utf8_bytes": len(text.encode("utf-8")),
        "ir_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "raw_ir_emitted": False,
    }


def _integer_stat(value: Any) -> int | None:
    if value is None or type(value) is bool:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def _compiled_memory(compiled: Any) -> dict[str, Any]:
    analysis = compiled.memory_analysis()
    result = {
        "argument_size_in_bytes": _integer_stat(
            getattr(analysis, "argument_size_in_bytes", None)
        ),
        "output_size_in_bytes": _integer_stat(
            getattr(analysis, "output_size_in_bytes", None)
        ),
        "alias_size_in_bytes": _integer_stat(
            getattr(analysis, "alias_size_in_bytes", None)
        ),
        "temp_size_in_bytes": _integer_stat(
            getattr(analysis, "temp_size_in_bytes", None)
        ),
        "generated_code_size_in_bytes": _integer_stat(
            getattr(analysis, "generated_code_size_in_bytes", None)
        ),
    }
    return result


def _compiled_memory_gate(memory: dict[str, Any]) -> dict[str, Any]:
    argument = memory.get("argument_size_in_bytes")
    output = memory.get("output_size_in_bytes")
    alias = memory.get("alias_size_in_bytes")
    temporary = memory.get("temp_size_in_bytes")
    combined = (
        argument + output + temporary
        if all(isinstance(item, int) for item in (argument, output, temporary))
        else None
    )
    checks = {
        "argument_bytes_exact": argument == _ARGUMENT_BYTES,
        "output_bytes_exact": output == _COMPILER_OUTPUT_BYTES,
        "alias_bytes_zero": alias == 0,
        "temporary_bytes_bounded": isinstance(temporary, int)
        and temporary <= _MAX_TEMP_BYTES,
        "combined_bytes_bounded": isinstance(combined, int)
        and combined <= _MAX_COMBINED_BYTES,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "logical_output_payload_bytes": _LOGICAL_OUTPUT_BYTES,
        "tuple_pointer_bytes": _TUPLE_POINTER_BYTES,
        "combined_argument_output_temporary_bytes": combined,
    }


def _sealed_registration_manifest(
    registration: Any,
    *,
    library_path: Path,
    library_sha256: str,
    library_size_bytes: int,
) -> dict[str, Any]:
    required = {
        "library_path": library_path,
        "library_sha256": library_sha256,
        "snapshot_sha256": library_sha256,
        "snapshot_size_bytes": library_size_bytes,
        "snapshot_mode": 0o600,
        "target_name": _TARGET,
        "platform": "ROCM",
        "registration_api_version": 1,
        "custom_call_api_version": 4,
        "sealed_snapshot": True,
        "snapshot_fd_retained": True,
    }
    observed = {name: getattr(registration, name, None) for name in required}
    seals = _integer_stat(getattr(registration, "snapshot_seals", None))
    checks = {
        "fields_exact": observed == required,
        "required_seals": isinstance(seals, int)
        and seals & _REQUIRED_SNAPSHOT_SEALS == _REQUIRED_SNAPSHOT_SEALS,
    }
    if not all(checks.values()):
        raise RuntimeError("GDN execute sealed registration identity mismatch")
    return {
        "passed": True,
        "target": _TARGET,
        "platform": "ROCM",
        "library_sha256": library_sha256,
        "snapshot_sha256": library_sha256,
        "snapshot_size_bytes": library_size_bytes,
        "snapshot_mode": 0o600,
        "snapshot_seals": seals,
        "snapshot_fd_retained": True,
        "raw_library_path_emitted": False,
        "dlopen_source": "retained_sealed_memfd_snapshot_only",
    }


def _backend_manifest(jax: Any, jaxlib: Any, jax_backend: Any) -> dict[str, Any]:
    resolved = jax.default_backend()
    platform_version = str(jax_backend.get_backend().platform_version)
    devices = jax.devices()
    if (
        resolved != "gpu"
        or "rocm" not in platform_version.lower()
        or len(devices) != 1
        or getattr(devices[0], "id", None) != 0
    ):
        raise RuntimeError("JAX did not resolve exact ROCm device ordinal zero")
    if str(jax.__version__) != _EXPECTED_STACK_VERSIONS["jax"]:
        raise RuntimeError("runtime JAX version differs from exact stack preflight")
    if str(jaxlib.__version__) != _EXPECTED_STACK_VERSIONS["jaxlib"]:
        raise RuntimeError("runtime jaxlib version differs from exact stack preflight")
    return {
        "passed": True,
        "platform_resolved": "gpu",
        "platform_family": "rocm",
        "visible_device_count": 1,
        "device_ordinal": 0,
        "jax_version": str(jax.__version__),
        "jaxlib_version": str(jaxlib.__version__),
        "platform_version_sha256": hashlib.sha256(
            platform_version.encode()
        ).hexdigest(),
        "device_kind_sha256": hashlib.sha256(
            str(getattr(devices[0], "device_kind", "")).encode()
        ).hexdigest(),
        "raw_device_descriptions_emitted": False,
    }


def _compile_unreleased(
    jax: Any,
    jnp: Any,
    gdn_execute_s512: Callable[..., Any],
    register_gdn_execute_s512: Callable[..., Any],
    library_path: Path,
    library_sha256: str,
    library_size_bytes: int,
    require_clean_boot: Callable[[], dict[str, Any]],
    boot: _BootSeal,
    counters: dict[str, int],
    output: TextIO,
) -> tuple[Any, dict[str, Any]]:
    counters["registration_attempts"] += 1
    try:
        registration = register_gdn_execute_s512(
            library_path, library_sha256=library_sha256, enabled=True
        )
        sealed = _sealed_registration_manifest(
            registration,
            library_path=library_path,
            library_sha256=library_sha256,
            library_size_bytes=library_size_bytes,
        )
        counters["registration_completions"] += 1
        _emit(
            {
                "record_type": "ffi_registered",
                "timestamp": _utc_now(),
                "sealed_registration": sealed,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot,
            boot,
            output,
            "after_ffi_registration_attempt",
            counters,
        )

    signatures = tuple(
        jax.ShapeDtypeStruct(shape, jnp.float32) for shape in _INPUT_SHAPES
    )
    counters["shape_dtype_structs"] += len(signatures)

    def execute(*inputs: Any) -> tuple[Any, Any]:
        counters["ffi_python_trace_calls"] += 1
        return gdn_execute_s512(
            *inputs,
            enabled=True,
            library_path=library_path,
            library_sha256=library_sha256,
        )

    counters["lower_attempts"] += 1
    lowered = None
    try:
        started = time.perf_counter()
        lowered = jax.jit(execute).lower(*signatures)
        lower_seconds = time.perf_counter() - started
        counters["lower_completions"] += 1
        if counters["ffi_python_trace_calls"] != 1:
            raise RuntimeError("lowering did not trace exactly one execute FFI call")
        stable_text = str(lowered.compiler_ir(dialect="stablehlo"))
        stablehlo = _ir_summary(stable_text, "stablehlo")
        del stable_text
        precompile_gate = {
            "stablehlo_passed": stablehlo["passed"],
            "lower_duration_passed": math.isfinite(lower_seconds)
            and 0 <= lower_seconds < _LOWER_HARD_SECONDS,
        }
        precompile_gate["passed"] = all(precompile_gate.values())
        _emit(
            {
                "record_type": "lowered",
                "timestamp": _utc_now(),
                "lower_seconds": lower_seconds,
                "stablehlo": stablehlo,
                "precompile_gate": precompile_gate,
                "counters": dict(counters),
            },
            output,
        )
        if not precompile_gate["passed"]:
            raise RuntimeError("GDN execute StableHLO or lower-duration gate failed")
    finally:
        _journal_checkpoint(
            require_clean_boot, boot, output, "after_ffi_lower_attempt", counters
        )

    counters["compile_attempts"] += 1
    compiled = None
    released = False
    try:
        started = time.perf_counter()
        compiled = lowered.compile()
        compile_seconds = time.perf_counter() - started
        counters["compile_completions"] += 1
        optimized_text = compiled.as_text()
        optimized_hlo = _ir_summary(optimized_text, "optimized_hlo")
        del optimized_text
        memory = _compiled_memory(compiled)
        memory_gate = _compiled_memory_gate(memory)
        release_gate = {
            "stablehlo_passed": stablehlo["passed"],
            "optimized_hlo_passed": optimized_hlo["passed"],
            "compiled_memory_passed": memory_gate["passed"],
            "compile_duration_passed": math.isfinite(compile_seconds)
            and 0 <= compile_seconds < _COMPILE_HARD_SECONDS,
        }
        release_gate["passed"] = all(release_gate.values())
        report = {
            "record_type": "ffi_compiled_unreleased",
            "timestamp": _utc_now(),
            "lower_seconds": lower_seconds,
            "compile_seconds": compile_seconds,
            "stablehlo": stablehlo,
            "optimized_hlo": optimized_hlo,
            "compiled_memory": memory,
            "compiled_memory_gate": memory_gate,
            "release_gate": release_gate,
            "executable_capability_released": False,
            "counters": dict(counters),
        }
        _emit(report, output)
        if not release_gate["passed"]:
            raise RuntimeError("GDN execute compile release gate failed")
        released = True
        return compiled, report
    finally:
        if compiled is not None and not released:
            del compiled
        if lowered is not None:
            del lowered
        _journal_checkpoint(
            require_clean_boot, boot, output, "after_ffi_compile_attempt", counters
        )


class _CheckedExecutable:
    __slots__ = ("_compiled", "_consumed", "_counters", "proof")

    def __init__(
        self,
        compiled: Any,
        *,
        proof: dict[str, Any],
        counters: dict[str, int],
        token: object,
    ) -> None:
        if token is not _CHECKED_CAPABILITY_TOKEN or proof.get("passed") is not True:
            raise RuntimeError("refusing unchecked GDN execute executable")
        self._compiled = compiled
        self._consumed = False
        self._counters = counters
        self.proof = proof

    def invoke(self, jax: Any, inputs: tuple[Any, ...]) -> tuple[Any, Any]:
        if self._consumed:
            raise RuntimeError("GDN execute one-shot capability was already consumed")
        if not isinstance(inputs, tuple) or len(inputs) != 6:
            raise RuntimeError("checked GDN execute requires exactly six inputs")
        self._consumed = True
        self._counters["checked_executable_attempts"] += 1
        result = self._compiled(*inputs)
        result = jax.block_until_ready(result)
        self._counters["output_readiness_barriers"] += 1
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("checked GDN execute did not return two output leaves")
        self._counters["checked_executable_completions"] += 1
        return result


def _release_checked(
    compiled: Any,
    compile_gate: dict[str, Any],
    host_gate: dict[str, Any],
    counters: dict[str, int],
) -> _CheckedExecutable:
    proof = {
        "compile_release_gate_passed": compile_gate.get("passed") is True,
        "host_oracle_release_gate_passed": host_gate.get("passed") is True,
    }
    proof["passed"] = all(proof.values())
    counters["checked_capability_release_attempts"] += 1
    executable = _CheckedExecutable(
        compiled,
        proof=proof,
        counters=counters,
        token=_CHECKED_CAPABILITY_TOKEN,
    )
    counters["checked_capability_release_completions"] += 1
    return executable


def _tuple_device_put(
    jax: Any, host: tuple[Any, ...], counters: dict[str, int]
) -> tuple[Any, ...]:
    counters["tuple_device_put_attempts"] += 1
    placed = jax.device_put(host)
    placed = jax.block_until_ready(placed)
    counters["input_readiness_barriers"] += 1
    if not isinstance(placed, tuple) or len(placed) != 6:
        raise RuntimeError("tuple device_put did not preserve six input leaves")
    counters["tuple_device_put_completions"] += 1
    counters["device_put_leaves"] += 6
    return placed


def _dispatch(
    jax: Any,
    executable: _CheckedExecutable,
    inputs: tuple[Any, ...],
    require_clean_boot: Callable[[], dict[str, Any]],
    boot: _BootSeal,
    counters: dict[str, int],
    output: TextIO,
) -> tuple[tuple[Any, Any], float]:
    _emit(
        {
            "record_type": "dispatch_started",
            "timestamp": _utc_now(),
            "label": "single_s512_execute_candidate",
            "counters": dict(counters),
        },
        output,
    )
    started = time.perf_counter()
    try:
        result = executable.invoke(jax, inputs)
    finally:
        seconds = time.perf_counter() - started
        _journal_checkpoint(
            require_clean_boot,
            boot,
            output,
            "after_candidate_dispatch_attempt",
            counters,
        )
    _emit(
        {
            "record_type": "dispatch",
            "timestamp": _utc_now(),
            "label": "single_s512_execute_candidate",
            "seconds_including_output_readiness": seconds,
            "counters": dict(counters),
        },
        output,
    )
    return result, seconds


def _tuple_device_get(
    jax: Any, device: tuple[Any, Any], counters: dict[str, int]
) -> tuple[Any, Any]:
    counters["tuple_device_get_attempts"] += 1
    host = jax.device_get(device)
    if not isinstance(host, tuple) or len(host) != 2:
        raise RuntimeError("tuple device_get did not preserve two output leaves")
    counters["tuple_device_get_completions"] += 1
    counters["device_get_leaves"] += 2
    return host


def _numerical_metrics(np: Any, expected: Any, actual: Any) -> dict[str, Any]:
    expected_array = np.asarray(expected)
    actual_array = np.asarray(actual)
    expected64 = np.asarray(expected_array, dtype=np.float64)
    actual64 = np.asarray(actual_array, dtype=np.float64)
    difference = actual64 - expected64
    expected_flat = expected64.reshape(-1)
    actual_flat = actual64.reshape(-1)
    difference_flat = difference.reshape(-1)
    expected_norm = float(np.linalg.norm(expected_flat))
    actual_norm = float(np.linalg.norm(actual_flat))
    if min(expected_norm, actual_norm) <= 0 or not all(
        math.isfinite(item) for item in (expected_norm, actual_norm)
    ):
        raise RuntimeError("GDN execute output norms are not finite and nonzero")
    cosine_raw = float(
        np.dot(expected_flat, actual_flat) / (expected_norm * actual_norm)
    )
    return {
        "relative_l2": float(np.linalg.norm(difference_flat) / expected_norm),
        "cosine": float(np.clip(cosine_raw, -1.0, 1.0)),
        "cosine_raw": cosine_raw,
        "max_absolute": float(np.max(np.abs(difference_flat))),
        "finite_expected": bool(np.all(np.isfinite(expected_array))),
        "finite_actual": bool(np.all(np.isfinite(actual_array))),
        "finite_difference": bool(np.all(np.isfinite(difference))),
        "reference_norm": expected_norm,
        "actual_norm": actual_norm,
        "reference_sha256": _array_sha256(expected_array),
        "actual_sha256": _array_sha256(actual_array),
        "actual_shape": list(actual_array.shape),
        "actual_dtype": str(actual_array.dtype),
        "actual_nbytes": int(actual_array.nbytes),
    }


def _validate_actual(
    np: Any,
    reference: tuple[Any, Any],
    actual: tuple[Any, Any],
    seconds: float,
) -> dict[str, Any]:
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise RuntimeError("host candidate must contain exactly two arrays")
    actual_arrays = tuple(np.asarray(item) for item in actual)
    shape_dtype_bytes_exact = (
        actual_arrays[0].shape == _PREPARED_SHAPE
        and actual_arrays[0].dtype == reference[0].dtype
        and int(actual_arrays[0].nbytes) == 4_194_304
        and actual_arrays[1].shape == _STATE_SHAPE
        and actual_arrays[1].dtype == np.dtype(np.float32)
        and int(actual_arrays[1].nbytes) == 2_097_152
    )
    output_buffers_c_contiguous = all(
        bool(item.flags.c_contiguous) for item in actual_arrays
    )
    output_buffers_non_overlapping = not np.shares_memory(
        actual_arrays[0], actual_arrays[1]
    )
    metrics = {
        name: _numerical_metrics(np, expected, observed)
        for name, expected, observed in zip(
            _OUTPUT_NAMES, reference, actual_arrays, strict=True
        )
    }
    thresholds = {
        "output": {
            "relative_l2_lte": 5e-3,
            "cosine_gte": 0.9999,
            "max_abs_lte": 5e-3,
        },
        "final_state": {
            "relative_l2_lte": 2e-4,
            "cosine_gte": 0.999999,
            "max_abs_lte": 2e-4,
        },
    }
    numerical_checks: dict[str, bool] = {}
    for name, threshold in thresholds.items():
        item = metrics[name]
        numerical_checks[f"{name}_finite"] = (
            item["finite_expected"]
            and item["finite_actual"]
            and item["finite_difference"]
        )
        numerical_checks[f"{name}_relative_l2"] = (
            item["relative_l2"] <= threshold["relative_l2_lte"]
        )
        numerical_checks[f"{name}_cosine"] = item["cosine"] >= threshold["cosine_gte"]
        numerical_checks[f"{name}_max_absolute"] = (
            item["max_absolute"] <= threshold["max_abs_lte"]
        )
    duration_hard = math.isfinite(seconds) and 0 <= seconds < _RUNTIME_HARD_SECONDS
    duration_promotion = (
        math.isfinite(seconds) and 0 <= seconds < _RUNTIME_PROMOTION_SECONDS
    )
    passed = (
        shape_dtype_bytes_exact
        and output_buffers_c_contiguous
        and output_buffers_non_overlapping
        and all(numerical_checks.values())
        and duration_hard
    )
    promotion = passed and duration_promotion
    result = {
        "passed": passed,
        "promotion_passed": promotion,
        "classification": (
            "promotable"
            if promotion
            else "completed_unpromotable"
            if passed
            else "failed"
        ),
        "candidate_seconds_including_output_readiness": seconds,
        "shape_dtype_nbytes_exact": shape_dtype_bytes_exact,
        "output_buffers_c_contiguous": output_buffers_c_contiguous,
        "output_buffers_non_overlapping": output_buffers_non_overlapping,
        "metrics": metrics,
        "numerical_checks": numerical_checks,
        "duration_checks": {
            "hard_below_1_second": duration_hard,
            "promotion_below_250_milliseconds": duration_promotion,
        },
        "thresholds": thresholds,
        "actual_framed_tuple_sha256": _framed_tuple_sha256(
            _REFERENCE_FRAMING_DOMAIN,
            tuple(zip(_OUTPUT_NAMES, actual_arrays, strict=True)),
        ),
        "reference_framed_tuple_sha256": _framed_tuple_sha256(
            _REFERENCE_FRAMING_DOMAIN,
            tuple(zip(_OUTPUT_NAMES, reference, strict=True)),
        ),
        "raw_tensors_emitted": False,
    }
    if not passed:
        raise RuntimeError(
            "GDN execute candidate failed numerical or hard-duration gate"
        )
    return result


def _initialize_backend(
    require_clean_boot: Callable[[], dict[str, Any]],
    boot: _BootSeal,
    counters: dict[str, int],
    output: TextIO,
    *,
    _dependencies: tuple[Any, Any, Any, Any, Callable[..., Any], Callable[..., Any]]
    | None = None,
) -> tuple[Any, Any, Callable[..., Any], Callable[..., Any], dict[str, Any]]:
    _journal_checkpoint(
        require_clean_boot,
        boot,
        output,
        "before_backend_initialization",
        counters,
    )
    counters["backend_initialization_attempts"] += 1
    try:
        if _dependencies is None:
            import jax
            import jax.numpy as jnp
            import jaxlib
            from jax.extend import backend as jax_backend

            from skyrl.tx.kernels.rocm.gdn_execute_ffi import (
                gdn_execute_s512,
                register_gdn_execute_s512,
            )
        else:
            (
                jax,
                jnp,
                jaxlib,
                jax_backend,
                gdn_execute_s512,
                register_gdn_execute_s512,
            ) = _dependencies
        wrapper_path = _source_files()["wrapper_source_sha256"].resolve()
        wrapper_module = sys.modules.get("skyrl.tx.kernels.rocm.gdn_execute_ffi")
        if _dependencies is None and (
            wrapper_module is None
            or not isinstance(getattr(wrapper_module, "__file__", None), str)
            or Path(wrapper_module.__file__).resolve() != wrapper_path
        ):
            raise RuntimeError("loaded execute wrapper is not exact bound source")
        backend = _backend_manifest(jax, jaxlib, jax_backend)
        counters["backend_initialization_completions"] += 1
        _emit(
            {
                "record_type": "backend_ready",
                "timestamp": _utc_now(),
                "backend": backend,
                "counters": dict(counters),
            },
            output,
        )
        return jax, jnp, gdn_execute_s512, register_gdn_execute_s512, backend
    finally:
        _journal_checkpoint(
            require_clean_boot,
            boot,
            output,
            "after_backend_initialization_attempt",
            counters,
        )


def _run_rocm_body(
    args: argparse.Namespace,
    output: TextIO,
    require_clean_boot: Callable[[], dict[str, Any]],
    boot: _BootSeal,
    counters: dict[str, int],
    *,
    environment: dict[str, str],
    library_manifest: dict[str, Any],
    _dependencies: tuple[Any, Any, Any, Any, Callable[..., Any], Callable[..., Any]]
    | None = None,
) -> int:
    _emit(
        {
            "record_type": "command_buffer_environment_proof",
            "timestamp": _utc_now(),
            "proof": _prove_command_buffers_disabled(environment),
            "counters": dict(counters),
        },
        output,
    )
    boundary: tuple[Any, ...] | None = None
    reference: tuple[Any, Any] | None = None
    host_input_report: dict[str, Any] | None = None
    reference_report: dict[str, Any] | None = None
    np = None
    if not args.compile_diagnostic:
        _journal_checkpoint(
            require_clean_boot, boot, output, "before_host_oracle", counters
        )
        try:
            prepare_module, execute_module = _load_oracles()
            import numpy as np_module

            np = np_module
            boundary, reference, host_input_report, reference_report = (
                _construct_host_reference(
                    np,
                    prepare_module.gdn_prepare_s512_numpy,
                    execute_module.gdn_execute_s512_numpy,
                    counters,
                )
            )
            _emit(
                {
                    "record_type": "host_oracle",
                    "timestamp": _utc_now(),
                    "inputs": host_input_report,
                    "reference": reference_report,
                    "device_transfer_started": False,
                    "counters": dict(counters),
                },
                output,
            )
        finally:
            _journal_checkpoint(
                require_clean_boot,
                boot,
                output,
                "after_host_oracle_attempt",
                counters,
            )

    compiled = None
    final_library = None
    try:
        (
            jax,
            jnp,
            gdn_execute_s512,
            register_gdn_execute_s512,
            _backend,
        ) = _initialize_backend(
            require_clean_boot,
            boot,
            counters,
            output,
            _dependencies=_dependencies,
        )
        compiled, compile_report = _compile_unreleased(
            jax,
            jnp,
            gdn_execute_s512,
            register_gdn_execute_s512,
            args.library,
            args.library_sha256,
            int(library_manifest["size_bytes"]),
            require_clean_boot,
            boot,
            counters,
            output,
        )
        if args.compile_diagnostic:
            del compiled
            compiled = None
            if counters != _completed_compile_diagnostic_counters():
                raise RuntimeError("compile diagnostic counter contract was not exact")
            _emit(
                {
                    "record_type": "compile_diagnostic_gates_passed_pending_postcheck",
                    "timestamp": _utc_now(),
                    "status": "passed_exact_s512_execute_compile_only",
                    "release_gate": compile_report["release_gate"],
                    "host_inputs_constructed": 0,
                    "oracle_invocations": 0,
                    "device_transfers": 0,
                    "executable_invocations": 0,
                    "counters": dict(counters),
                },
                output,
            )
        else:
            if any(
                item is None
                for item in (
                    boundary,
                    reference,
                    host_input_report,
                    reference_report,
                    np,
                )
            ):
                raise RuntimeError("runtime host oracle was not completed")
            host_gate = {
                "input_identity_gate_passed": host_input_report["passed"],
                "execute_reference_gate_passed": reference_report["passed"],
            }
            host_gate["passed"] = all(host_gate.values())
            executable = _release_checked(
                compiled, compile_report["release_gate"], host_gate, counters
            )
            _emit(
                {
                    "record_type": "checked_executable_released",
                    "timestamp": _utc_now(),
                    "proof": executable.proof,
                    "one_shot": True,
                    "counters": dict(counters),
                },
                output,
            )
            try:
                device_inputs = _tuple_device_put(jax, boundary, counters)
            finally:
                _journal_checkpoint(
                    require_clean_boot,
                    boot,
                    output,
                    "after_input_device_put_attempt",
                    counters,
                )
            device_outputs, seconds = _dispatch(
                jax,
                executable,
                device_inputs,
                require_clean_boot,
                boot,
                counters,
                output,
            )
            try:
                actual = _tuple_device_get(jax, device_outputs, counters)
            finally:
                _journal_checkpoint(
                    require_clean_boot,
                    boot,
                    output,
                    "after_candidate_device_get_attempt",
                    counters,
                )
            try:
                validation = _validate_actual(np, reference, actual, seconds)
                host_hashes_after_device_run = {
                    name: _array_sha256(array)
                    for name, array in zip(_INPUT_NAMES, boundary, strict=True)
                }
                host_inputs_unchanged = (
                    host_hashes_after_device_run
                    == host_input_report["individual_sha256"]
                    == _EXPECTED_INPUT_SHA256
                )
                validation["host_inputs_unchanged_after_device_run"] = (
                    host_inputs_unchanged
                )
                if not host_inputs_unchanged:
                    raise RuntimeError(
                        "host input boundary changed during device qualification"
                    )
                _emit(
                    {
                        "record_type": "host_validation",
                        "timestamp": _utc_now(),
                        "validation": validation,
                        "counters": dict(counters),
                    },
                    output,
                )
            finally:
                _journal_checkpoint(
                    require_clean_boot,
                    boot,
                    output,
                    "after_host_validation_attempt",
                    counters,
                )
            if counters != _completed_counters():
                raise RuntimeError("runtime one-shot counter contract was not exact")
            _emit(
                {
                    "record_type": "runtime_gates_passed_pending_postcheck",
                    "timestamp": _utc_now(),
                    "status": (
                        "passed_exact_s512_execute_promotable"
                        if validation["promotion_passed"]
                        else "passed_exact_s512_execute_unpromotable"
                    ),
                    "compile_release_gate": compile_report["release_gate"],
                    "host_oracle_gate": host_gate,
                    "validation": validation,
                    "counters": dict(counters),
                    "warmup_validated": False,
                    "replay_validated": False,
                    "vjp_validated": False,
                    "model_validated": False,
                    "performance_benchmark_authorized": False,
                },
                output,
            )
        return 0
    finally:
        if compiled is not None:
            del compiled
        try:
            final_library = _assert_same_library(args.library, library_manifest)
        finally:
            _journal_checkpoint(
                require_clean_boot,
                boot,
                output,
                "after_library_postcheck",
                counters,
            )
        if final_library is not None:
            _emit(
                {
                    "record_type": "library_postcheck",
                    "timestamp": _utc_now(),
                    "library": {
                        key: value
                        for key, value in final_library.items()
                        if key != "identity"
                    },
                    "counters": dict(counters),
                },
                output,
            )


def _execute(args: argparse.Namespace, output: TextIO) -> int:
    counters = _zero_counters()
    compile_diagnostic = bool(getattr(args, "compile_diagnostic", False))
    _emit(
        {
            "record_type": "manifest",
            "timestamp": _utc_now(),
            "platform_requested": args.platform,
            "allow_gpu": args.allow_gpu,
            "compile_diagnostic": compile_diagnostic,
            "case": args.case,
            "scope": (
                "abstract_refusal"
                if args.platform == "abstract"
                else "guarded_s512_execute_compile_diagnostic"
                if compile_diagnostic
                else "guarded_s512_execute_one_shot_runtime"
            ),
            "contract": _exact_contract(),
            "fresh_process_required": True,
            "outer_profile_rocm_supervision_required_and_proven": (
                args.platform == "rocm"
            ),
            "raw_library_path_emitted": False,
            "raw_ir_emitted": False,
            "raw_tensors_emitted": False,
            "counters": dict(counters),
            **_source_hashes(),
        },
        output,
    )
    if args.platform == "abstract":
        _emit(
            {
                "record_type": "refused",
                "timestamp": _utc_now(),
                "status": "no_gpu_abstract_manifest_only",
                "reason": (
                    "use a fresh direct profile_rocm.py child with --platform rocm, "
                    f"--allow-gpu, --case {_CASE}, exact library/hash/output, and "
                    "optionally --compile-diagnostic for the mandatory first rung"
                ),
                "jax_imported": False,
                "numpy_imported": False,
                "skyrl_rocm_package_imported": False,
                "shared_library_loaded": False,
                "counters": dict(counters),
            },
            output,
        )
        return 0

    stage = "fresh_process_preflight"
    try:
        _assert_fresh_accelerator_process()
        stage = "bound_sources"
        bound_sources = _assert_bound_sources()
        stage = "profile_supervision"
        profile_proof = _prove_profile_supervision()
        stage = "library_preflight"
        library = _validate_library_path(args.library, args.library_sha256)
        _emit(
            {
                "record_type": "prerequisite_proof",
                "timestamp": _utc_now(),
                "bound_sources": bound_sources,
                "profile_supervision": profile_proof,
                "library": {
                    key: value for key, value in library.items() if key != "identity"
                },
                "counters": dict(counters),
            },
            output,
        )
        stage = "bounded_environment"
        numeric_environment = (
            None if compile_diagnostic else _validate_host_numeric_environment()
        )
        environment = _configure_rocm_environment()
        _emit(
            {
                "record_type": "environment",
                "timestamp": _utc_now(),
                "host_numeric_environment": numeric_environment,
                "command_buffers": _prove_command_buffers_disabled(environment),
                "counters": dict(counters),
            },
            output,
        )
        boot = _BootSeal()
        guarded_process, require_clean_boot = _load_safety_helpers()
        stage = "safety_preflight"
        with guarded_process() as raw_safety:
            try:
                _emit(
                    {
                        "record_type": "safety_preflight",
                        "timestamp": _utc_now(),
                        "safety": _public_safety_preflight(raw_safety),
                        "boot": boot.check(),
                        "hardware_stack": _hardware_stack_preflight(),
                        "counters": dict(counters),
                    },
                    output,
                )
                stage = "compile_diagnostic" if compile_diagnostic else "runtime"
                result = _run_rocm_body(
                    args,
                    output,
                    require_clean_boot,
                    boot,
                    counters,
                    environment=environment,
                    library_manifest=library,
                )
            finally:
                try:
                    postflight = _public_clean_safety(
                        require_clean_boot(), "safety_postflight"
                    )
                    current_boot = boot.check()
                except Exception:
                    stage = "safety_postflight"
                    raise
                _emit(
                    {
                        "record_type": "safety_postflight",
                        "timestamp": _utc_now(),
                        "stage": "same_current_boot_rechecked",
                        "safety": postflight,
                        "boot": current_boot,
                        "counters": dict(counters),
                    },
                    output,
                )
        _emit(
            {
                "record_type": "completed",
                "timestamp": _utc_now(),
                "status": "passed",
                "mode": "compile_diagnostic" if compile_diagnostic else "runtime",
                "counters": dict(counters),
            },
            output,
        )
        return result
    except Exception as error:
        _emit(
            {
                "record_type": "error",
                "timestamp": _utc_now(),
                "stage": stage,
                "status": "failed_closed",
                "error_type": type(error).__name__,
                **_redacted_error(error),
                "counters": dict(counters),
            },
            output,
        )
        return 1


def _open_exclusive_output(path: Path) -> TextIO:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise RuntimeError("exclusive GDN execute output is not private mode 0600")
        return os.fdopen(descriptor, "w", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.output is None:
        return _execute(args, sys.stdout)
    with _open_exclusive_output(args.output) as output:
        return _execute(args, output)


if __name__ == "__main__":
    sys.exit(main())
