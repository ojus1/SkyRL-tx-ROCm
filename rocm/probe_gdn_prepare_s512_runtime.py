#!/usr/bin/env python3
"""Fail-closed one-shot numerical runtime gate for exact S512 GDN prepare.

The default ``abstract`` path emits a refusal manifest without importing JAX,
SkyRL's ROCm package, NumPy, or a shared library.  The explicit ROCm path binds
every executable dependency first, repeats the promoted compile-only gate in a
fresh process, constructs and validates one deterministic dense host oracle,
and only then releases a private one-shot executable capability.

The sole authorized invocation is ``s512-dense-all-valid``.  It validates the
prepare-only U/W/within-chunk-gamma boundary; it is not a recurrence, model,
VJP, training, replay, warmup, or performance benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, TextIO

_CASE = "s512-dense-all-valid"
_TARGET = "skyrl_gdn_prepare_s512_f32_v1"
_LIBRARY_BASENAME = "libskyrl_gdn_prepare_s512_gfx1100.so"
_KEY_SHAPE = (1, 512, 16, 128)
_VALUE_SHAPE = (1, 512, 32, 128)
_GATE_SHAPE = (1, 512, 32)
_ARGUMENT_BYTES = 12_713_984
_LOGICAL_OUTPUT_BYTES = 16_842_752
_TUPLE_POINTER_BYTES = 24
_COMPILER_OUTPUT_BYTES = 16_842_776
_MAX_TEMP_BYTES = 64 * 1024**2
_MAX_TOTAL_BYTES = 96 * 1024**2
_PROMOTION_SECONDS = 0.250
_HARD_SECONDS = 2.0
_COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="
_REQUIRED_SNAPSHOT_SEALS = 0x000F
_CHECKED_CAPABILITY_TOKEN = object()

_EXPECTED_COMPILE_HELPER_SHA256 = (
    "cacf782aa58a9735b44492ba6df7725e3ac777c7e57170c174c1563ff2f3c2b5"
)
_EXPECTED_ORACLE_SHA256 = (
    "019da59345f04ed2bfea553d797deb1a1c15a34c11147780616d4b4a44195aad"
)
_EXPECTED_WRAPPER_SHA256 = (
    "39204094b1d9e1e8caddcc833cc02edb9dab2b7e32bbee75c5462fd771a6052f"
)
_EXPECTED_HIP_SHA256 = (
    "8deaf58f5bf68e936472c434c985a52bea8ea1e26b75983af9ec8ae9c1e80f45"
)
_EXPECTED_SAFETY_SHA256 = (
    "8b9441e0147b35b000fe8340ea7b7c702372320c569ced71410b92b558953e91"
)
_EXPECTED_SEALED_LOADER_SHA256 = (
    "66b868b7909a2279d5ddca0e1582f8563e8097723e970d09fce733aef2ba425a"
)
_EXPECTED_PACKAGE_SHA256 = {
    "skyrl": "667d4a15b970b851e20d17510224670c14646cfb6d5a1e388ca6b9cc6da8bf41",
    "tx": "a7abb3e76d66df1f4472bb7a02b032ef31b959ca937fd351637b4e9b4a8fa95a",
    "kernels": "40abe638c7726fe5680b7c88321042016a0f695d86acfbef52337421e7257c1a",
    "rocm": "6d12a789cf1108538a04fbacd0b38a15dbcb8255cd0ca0fadf5a76c4191a4cfd",
}
_EXPECTED_INPUT_SHA256 = {
    "key": "2e4575756c918f44128c348a53816e246162fdfcb957155d910236a9651e9089",
    "value": "26bb0760af33a92fe5412ab2263fe28608473d11c4daab0fdaa310589aec9203",
    "g": "7b92e2bc06b11acf7733e1d8dd35301a855176f436928d9acaee9cd268c44ffd",
    "beta": "03b58170cf95fe0eec3990d4f73e7adae0238eef75e7e7d3f2bf230d360986d8",
}
_EXPECTED_REFERENCE_SHA256 = {
    "u": "b5e87a4ba0efa1db90843dc7e1b1569dc0d452ae1c13766ae041d2d1003cdbcb",
    "w": "425dc1f6399c23d3f79e349d55878a644d435a57915ee07df6363b95e5ad2d7e",
    "gamma": "6ff083b3478602b3a17253d241f84dec1dd3f5701ce630cb2c67d1f3cf956371",
}

# An explicit, locally reproducible framing replaces the unpublished framing
# behind the design note's combined hashes.  Individual published hashes above
# remain the hard semantic identity.  These tuple hashes are filled by tests
# from the documented v1 framing and are independently pinned as well.
_FRAMING_DOMAIN = b"skyrl-gdn-prepare-s512-array-tuple-v1\x00"
_EXPECTED_INPUT_TUPLE_SHA256 = (
    "9ee8387f7139fb6f43d0e161653d8d84f21a5167a584ab3c4a4b83591a2c9926"
)
_EXPECTED_REFERENCE_TUPLE_SHA256 = (
    "2dc7066b43a0562703a16c6cddfa3c8066da18acbf6fbcfc5a246ab14f99130b"
)

_JOURNAL_STAGES = frozenset(
    {
        "before_backend_initialization",
        "after_backend_initialization_attempt",
        "after_ffi_registration_attempt",
        "after_ffi_lower_attempt",
        "after_ffi_compile_attempt",
        "after_host_oracle_attempt",
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


def _redacted_message(error: BaseException) -> dict[str, Any]:
    encoded = str(error).encode("utf-8", errors="replace")
    return {
        "message_redacted": True,
        "message_utf8_bytes": len(encoded),
        "message_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _source_files() -> dict[str, Path]:
    repo = Path(__file__).resolve().parent.parent
    return {
        "runtime_probe_source_sha256": Path(__file__),
        "compile_helper_source_sha256": repo
        / "rocm"
        / "probe_gdn_prepare_s512_compile.py",
        "oracle_source_sha256": repo
        / "skyrl"
        / "tx"
        / "kernels"
        / "rocm"
        / "gdn_prepare_oracle.py",
        "wrapper_source_sha256": repo
        / "skyrl"
        / "tx"
        / "kernels"
        / "rocm"
        / "gdn_prepare_ffi.py",
        "hip_source_sha256": repo
        / "skyrl"
        / "tx"
        / "kernels"
        / "rocm"
        / "ffi"
        / "gdn_prepare_s512.hip",
        "sealed_loader_source_sha256": repo
        / "skyrl"
        / "tx"
        / "kernels"
        / "rocm"
        / "gdn_ffi_smoke.py",
        "safety_source_sha256": repo / "rocm" / "amdgpu_safety.py",
        "skyrl_package_source_sha256": repo / "skyrl" / "__init__.py",
        "tx_package_source_sha256": repo / "skyrl" / "tx" / "__init__.py",
        "kernels_package_source_sha256": repo
        / "skyrl"
        / "tx"
        / "kernels"
        / "__init__.py",
        "rocm_package_source_sha256": repo
        / "skyrl"
        / "tx"
        / "kernels"
        / "rocm"
        / "__init__.py",
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {name: _file_sha256(path) for name, path in _source_files().items()}


def _assert_bound_sources() -> dict[str, Any]:
    files = _source_files()
    observed = {
        "compile_helper": _file_sha256(files["compile_helper_source_sha256"]),
        "oracle": _file_sha256(files["oracle_source_sha256"]),
        "wrapper": _file_sha256(files["wrapper_source_sha256"]),
        "hip": _file_sha256(files["hip_source_sha256"]),
        "sealed_loader": _file_sha256(files["sealed_loader_source_sha256"]),
        "safety": _file_sha256(files["safety_source_sha256"]),
        "package_skyrl": _file_sha256(files["skyrl_package_source_sha256"]),
        "package_tx": _file_sha256(files["tx_package_source_sha256"]),
        "package_kernels": _file_sha256(files["kernels_package_source_sha256"]),
        "package_rocm": _file_sha256(files["rocm_package_source_sha256"]),
    }
    expected = {
        "compile_helper": _EXPECTED_COMPILE_HELPER_SHA256,
        "oracle": _EXPECTED_ORACLE_SHA256,
        "wrapper": _EXPECTED_WRAPPER_SHA256,
        "hip": _EXPECTED_HIP_SHA256,
        "sealed_loader": _EXPECTED_SEALED_LOADER_SHA256,
        "safety": _EXPECTED_SAFETY_SHA256,
        "package_skyrl": _EXPECTED_PACKAGE_SHA256["skyrl"],
        "package_tx": _EXPECTED_PACKAGE_SHA256["tx"],
        "package_kernels": _EXPECTED_PACKAGE_SHA256["kernels"],
        "package_rocm": _EXPECTED_PACKAGE_SHA256["rocm"],
    }
    if observed != expected:
        raise RuntimeError("GDN runtime dependency source hash mismatch")
    runtime_sha = _file_sha256(files["runtime_probe_source_sha256"])
    return {
        "passed": True,
        "all_executable_dependencies_exact": True,
        "runtime_probe_sha256": runtime_sha,
        **observed,
    }


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


def _load_compile_helper() -> ModuleType:
    return _load_exact_file_module(
        _source_files()["compile_helper_source_sha256"],
        _EXPECTED_COMPILE_HELPER_SHA256,
        "_skyrl_exact_gdn_prepare_compile_gate",
    )


def _load_oracle_module() -> ModuleType:
    return _load_exact_file_module(
        _source_files()["oracle_source_sha256"],
        _EXPECTED_ORACLE_SHA256,
        "_skyrl_exact_gdn_prepare_oracle",
    )


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
    parser.add_argument("--case", choices=(_CASE,))
    parser.add_argument("--library", type=Path)
    parser.add_argument("--library-sha256", type=_sha256_argument)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    guarded = (args.case, args.library, args.library_sha256, args.output)
    if args.platform == "abstract":
        if args.allow_gpu or any(value is not None for value in guarded):
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
    if args.output is None:
        parser.error("--platform rocm requires --output")
    if args.output.exists() or args.output.is_symlink():
        parser.error("refusing to overwrite existing or symbolic-link output")
    return args


def _outer_profile_contract() -> dict[str, Any]:
    return {
        "profile_rocm_required": True,
        "timeout_seconds": 120,
        "sensor_grace_seconds": 15,
        "maximum_vram_bytes": 2 * 1024**3,
        "maximum_junction_temperature_c": 90,
        "maximum_gpu_power_watts": 315,
        "minimum_host_available_bytes": 8 * 1024**3,
        "maximum_swap_bytes": 0,
        "sampling_interval_seconds": 0.05,
    }


def _exact_contract() -> dict[str, Any]:
    return {
        "operation": "gdn_prepare_s512_typed_ffi_one_shot_numerical_runtime",
        "case": _CASE,
        "target": _TARGET,
        "inputs": [
            {"name": "key", "shape": list(_KEY_SHAPE), "dtype": "float32"},
            {"name": "value", "shape": list(_VALUE_SHAPE), "dtype": "float32"},
            {"name": "g", "shape": list(_GATE_SHAPE), "dtype": "float32"},
            {"name": "beta", "shape": list(_GATE_SHAPE), "dtype": "float32"},
        ],
        "outputs": [
            {"name": "u", "shape": list(_VALUE_SHAPE), "dtype": "float32"},
            {"name": "w", "shape": list(_VALUE_SHAPE), "dtype": "float32"},
            {"name": "gamma", "shape": list(_GATE_SHAPE), "dtype": "float32"},
        ],
        "compile_gate": {
            "fresh_register_lower_compile": True,
            "stablehlo_precompile_gate": True,
            "optimized_hlo_gate": True,
            "exact_custom_calls": 1,
            "while_calls": 0,
            "alias_bytes": 0,
            "argument_bytes": _ARGUMENT_BYTES,
            "logical_output_bytes": _LOGICAL_OUTPUT_BYTES,
            "tuple_pointer_bytes": _TUPLE_POINTER_BYTES,
            "compiler_output_bytes": _COMPILER_OUTPUT_BYTES,
            "temporary_bytes_expected": 0,
            "temporary_bytes_hard_maximum": _MAX_TEMP_BYTES,
            "combined_bytes_hard_maximum": _MAX_TOTAL_BYTES,
        },
        "invocation_contract": {
            "tuple_device_puts": 1,
            "input_leaves": 4,
            "input_readiness_barriers": 1,
            "checked_executable_invocations": 1,
            "output_readiness_barriers": 1,
            "tuple_device_gets": 1,
            "output_leaves": 3,
            "lowered_callable_invocations": 0,
            "warmups": 0,
            "replays": 0,
            "gpu_references": 0,
            "gpu_reductions": 0,
            "backward": 0,
            "model": 0,
        },
        "duration_gate": {
            "promotion_seconds_strictly_below": _PROMOTION_SECONDS,
            "hard_seconds_strictly_below": _HARD_SECONDS,
        },
        "framing": {
            "algorithm": "domain || repeated(be64(metadata_len)||metadata||be64(payload_len)||payload)",
            "domain_ascii_nul_terminated": "skyrl-gdn-prepare-s512-array-tuple-v1",
            "metadata": "utf8(name) || NUL || dtype.str || NUL || comma-separated-shape",
        },
        "scope_exclusions": {
            "recurrence": False,
            "vjp": False,
            "model": False,
            "training": False,
            "replay": False,
            "performance_benchmark": False,
        },
        "outer_supervision": _outer_profile_contract(),
    }


def _zero_counters() -> dict[str, int]:
    return {
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
        "host_candidate_arrays": 0,
        "dense_oracle_attempts": 0,
        "dense_oracle_completions": 0,
        "tuple_device_put_attempts": 0,
        "tuple_device_put_completions": 0,
        "device_put_leaves": 0,
        "checked_executable_attempts": 0,
        "checked_executable_completions": 0,
        "tuple_device_get_attempts": 0,
        "tuple_device_get_completions": 0,
        "device_get_leaves": 0,
        "block_until_ready_calls": 0,
        "lowered_callable_invocations": 0,
        "warmup_invocations": 0,
        "replay_invocations": 0,
        "gpu_reference_invocations": 0,
        "gpu_reduction_invocations": 0,
        "backward_invocations": 0,
        "model_invocations": 0,
    }


def _completed_counters() -> dict[str, int]:
    result = _zero_counters()
    result.update(
        {
            "backend_initialization_attempts": 1,
            "backend_initialization_completions": 1,
            "registration_attempts": 1,
            "registration_completions": 1,
            "shape_dtype_structs": 4,
            "ffi_python_trace_calls": 1,
            "lower_attempts": 1,
            "lower_completions": 1,
            "compile_attempts": 1,
            "compile_completions": 1,
            "host_candidate_arrays": 4,
            "dense_oracle_attempts": 1,
            "dense_oracle_completions": 1,
            "tuple_device_put_attempts": 1,
            "tuple_device_put_completions": 1,
            "device_put_leaves": 4,
            "checked_executable_attempts": 1,
            "checked_executable_completions": 1,
            "tuple_device_get_attempts": 1,
            "tuple_device_get_completions": 1,
            "device_get_leaves": 3,
            "block_until_ready_calls": 2,
        }
    )
    return result


def _assert_fresh_accelerator_process() -> None:
    imported = sorted(
        name
        for name in sys.modules
        if name in {"jax", "jaxlib", "skyrl.tx.kernels.rocm"}
        or name.startswith("jax.")
        or name.startswith("jaxlib.")
        or name.startswith("skyrl.tx.kernels.rocm.")
    )
    if imported:
        raise RuntimeError("GDN runtime gate requires a fresh accelerator process")


def _validate_host_numeric_environment() -> dict[str, str]:
    expected = {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    if any(os.environ.get(name) != value for name, value in expected.items()):
        raise RuntimeError("GDN runtime oracle requires all BLAS thread caps exactly 1")
    return expected


def _public_clean_safety(safety: Any, stage: str) -> dict[str, Any]:
    if not isinstance(safety, dict) or safety.get("amdgpu_boot_clean") is not True:
        raise RuntimeError(f"{stage} did not prove a clean AMDGPU boot")
    if safety.get("fatal_amdgpu_events") != []:
        raise RuntimeError(f"{stage} returned invalid fatal-event evidence")
    return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}


def _public_safety_preflight(safety: Any) -> dict[str, Any]:
    public = _public_clean_safety(safety, "safety_preflight")
    cards = safety.get("amd_cards")
    if (
        not isinstance(cards, list)
        or not cards
        or cards != sorted(set(cards))
        or not all(
            isinstance(card, str) and re.fullmatch(r"card[0-9]+", card)
            for card in cards
        )
    ):
        raise RuntimeError("safety preflight returned invalid AMD card evidence")
    if safety.get("connected_amd_connectors") != []:
        raise RuntimeError("safety preflight did not prove every AMD connector idle")
    if safety.get("kfd_path") != "/dev/kfd":
        raise RuntimeError("safety preflight did not prove the exact KFD path")
    if (
        safety.get("kfd_accessible") is not True
        or safety.get("kfd_unowned") is not True
    ):
        raise RuntimeError("safety preflight did not prove accessible unowned KFD")
    return {
        **public,
        "amd_cards": list(cards),
        "connected_amd_connectors": [],
        "kfd_path": "/dev/kfd",
        "kfd_accessible": True,
        "kfd_unowned": True,
    }


def _journal_checkpoint(
    require_clean_boot: Callable[[], dict[str, Any]],
    output: TextIO,
    stage: str,
    counters: dict[str, int],
) -> None:
    if stage not in _JOURNAL_STAGES:
        raise RuntimeError("refusing an undeclared GDN runtime journal stage")
    _emit(
        {
            "record_type": "journal_checkpoint",
            "timestamp": _utc_now(),
            "stage": stage,
            "safety": _public_clean_safety(require_clean_boot(), stage),
            "counters": dict(counters),
        },
        output,
    )


def _array_sha256(array: Any) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _framed_tuple_sha256(items: tuple[tuple[str, Any], ...]) -> str:
    """Hash a named array tuple with an explicit domain and unambiguous lengths."""
    digest = hashlib.sha256()
    digest.update(_FRAMING_DOMAIN)
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
    """Return the exact approved row-major SplitMix64-derived FP32 grid."""
    indices = np.arange(math.prod(shape), dtype=np.uint64)
    z = (indices ^ np.uint64(seed)) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    values = (z >> np.uint64(40)).astype(np.uint32).astype(np.float32) * np.float32(
        2.0 / 16_777_216.0
    ) - np.float32(1.0)
    return values.reshape(shape)


def _construct_host_case(np: Any) -> tuple[tuple[Any, Any, Any, Any], dict[str, Any]]:
    raw_key = _splitmix64_f32(np, _KEY_SHAPE, 0x243F6A8885A308D3)
    norm = np.sqrt(
        np.sum(raw_key * raw_key, axis=-1, dtype=np.float32) + np.float32(1e-6)
    )
    key = np.ascontiguousarray(raw_key / norm[..., None], dtype=np.float32)
    value = np.ascontiguousarray(
        _splitmix64_f32(np, _VALUE_SHAPE, 0x13198A2E03707344) * np.float32(0.625),
        dtype=np.float32,
    )
    token = np.arange(512, dtype=np.int32)[:, None]
    head = np.arange(32, dtype=np.int32)[None, :]
    chunk = token // 64
    g = np.ascontiguousarray(
        -(
            np.float32(0.004)
            + np.float32(0.00035) * (token % 7)
            + np.float32(0.00025) * (head % 5)
            + np.float32(0.00015) * ((chunk + head) % 4)
        )[None, ...],
        dtype=np.float32,
    )
    beta = np.ascontiguousarray(
        (
            np.float32(0.12)
            + np.float32(0.24)
            * (
                ((3 * token + 5 * head + 7 * chunk) % 17).astype(np.float32)
                / np.float32(16)
            )
        )[None, ...],
        dtype=np.float32,
    )
    arrays = (key, value, g, beta)
    names = ("key", "value", "g", "beta")
    hashes = {name: _array_sha256(array) for name, array in zip(names, arrays)}
    shapes_exact = tuple(array.shape for array in arrays) == (
        _KEY_SHAPE,
        _VALUE_SHAPE,
        _GATE_SHAPE,
        _GATE_SHAPE,
    )
    dtypes_exact = all(array.dtype == np.dtype(np.float32) for array in arrays)
    contiguous = all(bool(array.flags.c_contiguous) for array in arrays)
    finite = all(bool(np.all(np.isfinite(array))) for array in arrays)
    key_norm = np.sqrt(np.sum(key * key, axis=-1, dtype=np.float32))
    invariants = {
        "shapes_exact": shapes_exact,
        "dtypes_exact": dtypes_exact,
        "c_contiguous": contiguous,
        "all_finite": finite,
        "key_norm_min": float(np.min(key_norm)),
        "key_norm_max": float(np.max(key_norm)),
        "value_min": float(np.min(value)),
        "value_max": float(np.max(value)),
        "g_min": float(np.min(g)),
        "g_max": float(np.max(g)),
        "beta_min": float(np.min(beta)),
        "beta_max": float(np.max(beta)),
    }
    checks = {
        "shapes_exact": shapes_exact,
        "dtypes_exact": dtypes_exact,
        "c_contiguous": contiguous,
        "all_finite": finite,
        "individual_hashes_exact": hashes == _EXPECTED_INPUT_SHA256,
        "key_rows_unit_norm": float(np.max(np.abs(key_norm - np.float32(1.0)))) <= 2e-7,
        "value_signed_nonzero": float(np.min(value)) < -0.62
        and float(np.max(value)) > 0.62,
        "g_strictly_negative": bool(np.all(g < 0)),
        "beta_strictly_positive": bool(np.all(beta > 0)),
    }
    framed = _framed_tuple_sha256(tuple(zip(names, arrays)))
    if _EXPECTED_INPUT_TUPLE_SHA256:
        checks["framed_tuple_hash_exact"] = framed == _EXPECTED_INPUT_TUPLE_SHA256
    if not all(checks.values()):
        raise RuntimeError(
            "deterministic GDN runtime host inputs failed identity gates"
        )
    return arrays, {
        "passed": True,
        "construction": "approved_splitmix64_v1_no_randomness",
        "individual_sha256": hashes,
        "framed_tuple_sha256": framed,
        "framing_domain_sha256": hashlib.sha256(_FRAMING_DOMAIN).hexdigest(),
        "invariants": invariants,
        "checks": checks,
    }


def _relative_l2(np: Any, expected: Any, actual: Any) -> float:
    expected64 = np.asarray(expected, dtype=np.float64)
    difference64 = np.asarray(actual, dtype=np.float64) - expected64
    denominator = float(np.linalg.norm(expected64.reshape(-1)))
    if not math.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("relative-L2 denominator is not finite and nonzero")
    return float(np.linalg.norm(difference64.reshape(-1)) / denominator)


def _chunk_equations(np: Any, host: tuple[Any, Any, Any, Any]) -> dict[str, Any]:
    key, value, g, beta = host
    key_chunks = key.reshape(8, 64, 16, 128).transpose(0, 2, 1, 3)
    value_chunks = value.reshape(8, 64, 16, 2, 128).transpose(0, 2, 3, 1, 4)
    g_chunks = g.reshape(8, 64, 16, 2).transpose(0, 2, 3, 1)
    beta_chunks = beta.reshape(8, 64, 16, 2).transpose(0, 2, 3, 1)
    gram = np.einsum(
        "chid,chjd->chij",
        key_chunks,
        key_chunks,
        dtype=np.float32,
        optimize=False,
    )
    prefix = np.cumsum(g_chunks, axis=-1, dtype=np.float32)
    gamma_chunks = np.exp(prefix).astype(np.float32, copy=False)
    decay = np.exp(prefix[..., :, None] - prefix[..., None, :]).astype(
        np.float32, copy=False
    )
    decay = np.tril(decay)
    strict_lower = np.tril(
        beta_chunks[..., :, None] * gram[:, :, None, :, :] * decay,
        k=-1,
    ).astype(np.float32, copy=False)
    no_decay_lower = np.tril(
        beta_chunks[..., :, None] * gram[:, :, None, :, :], k=-1
    ).astype(np.float32, copy=False)
    unit_lower = strict_lower + np.eye(64, dtype=np.float32)
    rhs_u = beta_chunks[..., None] * value_chunks
    rhs_w = (
        beta_chunks[..., None] * gamma_chunks[..., None] * key_chunks[:, :, None, :, :]
    )
    return {
        "key_chunks": key_chunks,
        "strict_lower": strict_lower,
        "no_decay_lower": no_decay_lower,
        "unit_lower": unit_lower,
        "rhs_u": rhs_u,
        "rhs_w": rhs_w,
        "gamma_chunks": gamma_chunks,
    }


def _chunk_outputs(value: Any, gate: Any) -> tuple[Any, Any]:
    value_chunks = value.reshape(8, 64, 16, 2, 128).transpose(0, 2, 3, 1, 4)
    gate_chunks = gate.reshape(8, 64, 16, 2).transpose(0, 2, 3, 1)
    return value_chunks, gate_chunks


def _equation_residual(
    np: Any, unit_lower: Any, solution: Any, rhs: Any
) -> dict[str, float]:
    residual = np.matmul(unit_lower, solution) - rhs
    rhs64 = np.asarray(rhs, dtype=np.float64)
    residual64 = np.asarray(residual, dtype=np.float64)
    denominator = float(np.linalg.norm(rhs64.reshape(-1)))
    if denominator <= 0 or not math.isfinite(denominator):
        raise RuntimeError("equation residual denominator is invalid")
    return {
        "relative_l2": float(np.linalg.norm(residual64.reshape(-1)) / denominator),
        "max_absolute": float(np.max(np.abs(residual64))),
    }


def _construct_reference(
    np: Any,
    oracle: Callable[..., Any],
    host: tuple[Any, Any, Any, Any],
    counters: dict[str, int],
) -> tuple[tuple[Any, Any, Any], dict[str, Any], dict[str, Any]]:
    counters["dense_oracle_attempts"] += 1
    started = time.perf_counter()
    reference = oracle(*host)
    oracle_seconds = time.perf_counter() - started
    if not isinstance(reference, tuple) or len(reference) != 3:
        raise RuntimeError("dense GDN oracle did not return exactly three arrays")
    counters["dense_oracle_completions"] += 1
    u, w, gamma = reference
    names = ("u", "w", "gamma")
    hashes = {name: _array_sha256(array) for name, array in zip(names, reference)}
    shapes_exact = tuple(array.shape for array in reference) == (
        _VALUE_SHAPE,
        _VALUE_SHAPE,
        _GATE_SHAPE,
    )
    dtypes_exact = all(array.dtype == np.dtype(np.float32) for array in reference)
    contiguous = all(bool(array.flags.c_contiguous) for array in reference)
    finite = all(bool(np.all(np.isfinite(array))) for array in reference)
    equations = _chunk_equations(np, host)
    u_chunks, gamma_chunks = _chunk_outputs(u, gamma)
    w_chunks, _ = _chunk_outputs(w, gamma)
    strict_lower_norms = np.linalg.norm(
        np.asarray(equations["strict_lower"], dtype=np.float64), axis=(-2, -1)
    )
    conditions = np.linalg.cond(equations["unit_lower"])
    identity_u = _relative_l2(np, equations["rhs_u"], u_chunks)
    identity_w = _relative_l2(np, equations["rhs_w"], w_chunks)
    no_decay_difference = _relative_l2(
        np, equations["no_decay_lower"], equations["strict_lower"]
    )
    global_gamma = np.exp(np.cumsum(host[2][0], axis=0, dtype=np.float32)).astype(
        np.float32, copy=False
    )[None, ...]
    wrong_global_difference = _relative_l2(np, global_gamma, gamma)
    chunk_u_norms = np.linalg.norm(
        np.asarray(u_chunks, dtype=np.float64), axis=(-2, -1)
    )
    chunk_w_norms = np.linalg.norm(
        np.asarray(w_chunks, dtype=np.float64), axis=(-2, -1)
    )
    pair_u_difference = np.linalg.norm(
        np.asarray(u_chunks[:, :, 0] - u_chunks[:, :, 1], dtype=np.float64),
        axis=(-2, -1),
    )
    pair_w_difference = np.linalg.norm(
        np.asarray(w_chunks[:, :, 0] - w_chunks[:, :, 1], dtype=np.float64),
        axis=(-2, -1),
    )
    gamma_step = np.diff(gamma_chunks, axis=-1)
    gamma_tokens = gamma.reshape(8, 64, 32)
    gamma_resets = gamma_tokens[1:, 0, :] - gamma_tokens[:-1, -1, :]
    residual_u = _equation_residual(
        np, equations["unit_lower"], u_chunks, equations["rhs_u"]
    )
    residual_w = _equation_residual(
        np, equations["unit_lower"], w_chunks, equations["rhs_w"]
    )
    framed = _framed_tuple_sha256(tuple(zip(names, reference)))
    checks = {
        "shapes_exact": shapes_exact,
        "dtypes_exact": dtypes_exact,
        "c_contiguous": contiguous,
        "all_finite": finite,
        "individual_hashes_exact": hashes == _EXPECTED_REFERENCE_SHA256,
        "all_strict_lower_systems_nonzero": bool(np.all(strict_lower_norms > 0)),
        "maximum_condition_at_most_1_5": float(np.max(conditions)) <= 1.5,
        "identity_u_difference_at_least_0_08": identity_u >= 0.08,
        "identity_w_difference_at_least_0_08": identity_w >= 0.08,
        "missing_decay_difference_at_least_0_10": no_decay_difference >= 0.10,
        "wrong_global_gamma_difference_at_least_1": wrong_global_difference >= 1.0,
        "every_chunk_u_nonzero": bool(np.all(chunk_u_norms > 0)),
        "every_chunk_w_nonzero": bool(np.all(chunk_w_norms > 0)),
        "every_paired_u_output_distinct": bool(np.all(pair_u_difference > 0)),
        "every_paired_w_output_distinct": bool(np.all(pair_w_difference > 0)),
        "gamma_strictly_decreases_inside_chunks": bool(np.all(gamma_step < 0)),
        "gamma_resets_at_chunk_boundaries": bool(np.all(gamma_resets > 0)),
        "reference_u_residual_at_most_1e_6": residual_u["relative_l2"] <= 1e-6,
        "reference_w_residual_at_most_1e_6": residual_w["relative_l2"] <= 1e-6,
    }
    if _EXPECTED_REFERENCE_TUPLE_SHA256:
        checks["framed_tuple_hash_exact"] = framed == _EXPECTED_REFERENCE_TUPLE_SHA256
    if not all(checks.values()):
        raise RuntimeError("dense GDN reference failed sensitivity or identity gates")
    report = {
        "passed": True,
        "oracle_seconds": oracle_seconds,
        "individual_sha256": hashes,
        "framed_tuple_sha256": framed,
        "ranges": {
            "u": [float(np.min(u)), float(np.max(u))],
            "w": [float(np.min(w)), float(np.max(w))],
            "gamma": [float(np.min(gamma)), float(np.max(gamma))],
        },
        "sensitivities": {
            "strict_lower_norm_min": float(np.min(strict_lower_norms)),
            "condition_min": float(np.min(conditions)),
            "condition_max": float(np.max(conditions)),
            "identity_u_relative_l2": identity_u,
            "identity_w_relative_l2": identity_w,
            "missing_decay_strict_lower_relative_l2": no_decay_difference,
            "wrong_global_gamma_relative_l2": wrong_global_difference,
            "minimum_paired_u_difference_l2": float(np.min(pair_u_difference)),
            "minimum_paired_w_difference_l2": float(np.min(pair_w_difference)),
            "reference_u_equation_residual": residual_u,
            "reference_w_equation_residual": residual_w,
        },
        "checks": checks,
    }
    return reference, report, equations


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
            raise RuntimeError("refusing an unchecked GDN runtime executable")
        self._compiled = compiled
        self._consumed = False
        self._counters = counters
        self.proof = proof

    def invoke(
        self,
        jax: Any,
        inputs: tuple[Any, Any, Any, Any],
        on_started: Callable[[], None],
    ) -> tuple[Any, Any, Any]:
        if self._consumed:
            raise RuntimeError("GDN runtime one-shot capability was already consumed")
        if not isinstance(inputs, tuple) or len(inputs) != 4:
            raise RuntimeError("checked GDN runtime requires exactly four input leaves")
        self._consumed = True
        self._counters["checked_executable_attempts"] += 1
        on_started()
        result = self._compiled(*inputs)
        result = jax.block_until_ready(result)
        self._counters["block_until_ready_calls"] += 1
        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError(
                "checked GDN executable did not return a three-leaf tuple"
            )
        self._counters["checked_executable_completions"] += 1
        return result


def _release_checked(
    compiled: Any,
    compile_proof: dict[str, Any],
    host_proof: dict[str, Any],
    counters: dict[str, int],
) -> _CheckedExecutable:
    proof = {
        "compile_release_gate_passed": compile_proof.get("passed") is True,
        "host_oracle_release_gate_passed": host_proof.get("passed") is True,
    }
    proof["passed"] = all(proof.values())
    return _CheckedExecutable(
        compiled,
        proof=proof,
        counters=counters,
        token=_CHECKED_CAPABILITY_TOKEN,
    )


def _compile_unreleased(
    jax: Any,
    jnp: Any,
    gdn_prepare_s512: Callable[..., Any],
    register_gdn_prepare_s512: Callable[..., Any],
    helper: ModuleType,
    library_path: Path,
    library_sha256: str,
    library_size_bytes: int,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    output: TextIO,
) -> tuple[Any, dict[str, Any]]:
    counters["registration_attempts"] += 1
    try:
        registration = register_gdn_prepare_s512(
            library_path, library_sha256=library_sha256, enabled=True
        )
        sealed = helper._sealed_registration_manifest(
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
            require_clean_boot, output, "after_ffi_registration_attempt", counters
        )

    signatures = (
        jax.ShapeDtypeStruct(_KEY_SHAPE, jnp.float32),
        jax.ShapeDtypeStruct(_VALUE_SHAPE, jnp.float32),
        jax.ShapeDtypeStruct(_GATE_SHAPE, jnp.float32),
        jax.ShapeDtypeStruct(_GATE_SHAPE, jnp.float32),
    )
    counters["shape_dtype_structs"] += len(signatures)

    def prepare(key: Any, value: Any, g: Any, beta: Any) -> tuple[Any, Any, Any]:
        counters["ffi_python_trace_calls"] += 1
        return gdn_prepare_s512(
            key,
            value,
            g,
            beta,
            enabled=True,
            library_path=library_path,
            library_sha256=library_sha256,
        )

    counters["lower_attempts"] += 1
    lowered = None
    try:
        started = time.perf_counter()
        lowered = jax.jit(prepare).lower(*signatures)
        lower_seconds = time.perf_counter() - started
        counters["lower_completions"] += 1
        if counters["ffi_python_trace_calls"] != 1:
            raise RuntimeError("lowering did not trace exactly one GDN FFI call")
        stable_text = str(lowered.compiler_ir(dialect="stablehlo"))
        stablehlo = helper._ir_summary(stable_text, "stablehlo")
        del stable_text
        precompile = {
            "stablehlo_structural_signature_layout_gate_passed": stablehlo["passed"],
            "passed": stablehlo["passed"],
        }
        _emit(
            {
                "record_type": "lowered",
                "timestamp": _utc_now(),
                "lower_seconds": lower_seconds,
                "stablehlo": stablehlo,
                "stablehlo_precompile_gate": precompile,
                "sealed_registration": sealed,
                "counters": dict(counters),
            },
            output,
        )
        if not precompile["passed"]:
            raise RuntimeError("StableHLO failed before GDN compilation")
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_ffi_lower_attempt", counters
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
        optimized_hlo = helper._ir_summary(optimized_text, "optimized_hlo")
        del optimized_text
        memory = helper._compiled_memory(compiled)
        structural = helper._structural_gate(stablehlo, optimized_hlo)
        memory_gate = helper._compiled_memory_gate(memory)
        temp = helper._integer_stat(memory.get("temp_size_in_bytes"))
        temp_expectation = {
            "expected_temporary_bytes": 0,
            "observed_temporary_bytes": temp,
            "passed": temp == 0,
        }
        release_gate = {
            "structural_gate_passed": structural["passed"],
            "compiled_memory_gate_passed": memory_gate["passed"],
            "temporary_bytes_expected_zero": temp_expectation["passed"],
        }
        release_gate["passed"] = all(release_gate.values())
        report = {
            "record_type": "ffi_compiled_unreleased",
            "timestamp": _utc_now(),
            "compile_seconds": compile_seconds,
            "stablehlo": stablehlo,
            "optimized_hlo": optimized_hlo,
            "structural_gate": structural,
            "compiled_memory": memory,
            "compiled_memory_gate": memory_gate,
            "temporary_memory_expectation": temp_expectation,
            "sealed_registration": sealed,
            "release_gate": release_gate,
            "executable_capability_released": False,
            "counters": dict(counters),
        }
        _emit(report, output)
        if not release_gate["passed"]:
            raise RuntimeError(
                "GDN executable failed structural or memory release gate"
            )
        released = True
        return compiled, report
    finally:
        if compiled is not None and not released:
            del compiled
        if lowered is not None:
            del lowered
        _journal_checkpoint(
            require_clean_boot, output, "after_ffi_compile_attempt", counters
        )


def _tuple_device_put(
    jax: Any,
    host: tuple[Any, Any, Any, Any],
    counters: dict[str, int],
) -> tuple[Any, Any, Any, Any]:
    counters["tuple_device_put_attempts"] += 1
    placed = jax.device_put(host)
    placed = jax.block_until_ready(placed)
    counters["block_until_ready_calls"] += 1
    if not isinstance(placed, tuple) or len(placed) != 4:
        raise RuntimeError("tuple device_put did not preserve four input leaves")
    counters["tuple_device_put_completions"] += 1
    counters["device_put_leaves"] += 4
    return placed


def _dispatch(
    jax: Any,
    executable: _CheckedExecutable,
    inputs: tuple[Any, Any, Any, Any],
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    output: TextIO,
) -> tuple[tuple[Any, Any, Any], float]:
    started: float | None = None

    def on_started() -> None:
        nonlocal started
        _emit(
            {
                "record_type": "dispatch_started",
                "timestamp": _utc_now(),
                "label": "single_s512_dense_all_valid_candidate",
                "counters": dict(counters),
            },
            output,
        )
        started = time.perf_counter()

    fallback = time.perf_counter()
    try:
        result = executable.invoke(jax, inputs, on_started)
    finally:
        seconds = time.perf_counter() - (started if started is not None else fallback)
        _journal_checkpoint(
            require_clean_boot, output, "after_candidate_dispatch_attempt", counters
        )
    _emit(
        {
            "record_type": "dispatch",
            "timestamp": _utc_now(),
            "label": "single_s512_dense_all_valid_candidate",
            "seconds_including_output_readiness": seconds,
            "counters": dict(counters),
        },
        output,
    )
    return result, seconds


def _tuple_device_get(
    jax: Any, device: tuple[Any, Any, Any], counters: dict[str, int]
) -> tuple[Any, Any, Any]:
    counters["tuple_device_get_attempts"] += 1
    host = jax.device_get(device)
    if not isinstance(host, tuple) or len(host) != 3:
        raise RuntimeError("tuple device_get did not preserve three output leaves")
    counters["tuple_device_get_completions"] += 1
    counters["device_get_leaves"] += 3
    return host


def _numerical_metrics(np: Any, expected: Any, actual: Any) -> dict[str, Any]:
    expected_array = np.asarray(expected)
    actual_array = np.asarray(actual)
    expected64 = np.asarray(expected_array, dtype=np.float64)
    actual64 = np.asarray(actual_array, dtype=np.float64)
    difference64 = actual64 - expected64
    expected_flat = expected64.reshape(-1)
    actual_flat = actual64.reshape(-1)
    difference_flat = difference64.reshape(-1)
    expected_norm = float(np.linalg.norm(expected_flat))
    actual_norm = float(np.linalg.norm(actual_flat))
    if min(expected_norm, actual_norm) <= 0 or not all(
        math.isfinite(item) for item in (expected_norm, actual_norm)
    ):
        raise RuntimeError("GDN output cosine norms are not finite and nonzero")
    cosine_raw = float(
        np.dot(expected_flat, actual_flat) / (expected_norm * actual_norm)
    )
    cosine = float(np.clip(cosine_raw, -1.0, 1.0))
    return {
        "relative_l2": float(np.linalg.norm(difference_flat) / expected_norm),
        "cosine_raw": cosine_raw,
        "cosine": cosine,
        "max_absolute": float(np.max(np.abs(difference_flat))),
        "finite_expected": bool(np.all(np.isfinite(expected_array))),
        "finite_actual": bool(np.all(np.isfinite(actual_array))),
        "finite_difference": bool(np.all(np.isfinite(difference64))),
        "expected_norm": expected_norm,
        "actual_norm": actual_norm,
        "actual_sha256": _array_sha256(actual_array),
        "reference_sha256": _array_sha256(expected_array),
        "actual_shape": list(actual_array.shape),
        "actual_dtype": str(actual_array.dtype),
        "actual_nbytes": int(actual_array.nbytes),
    }


def _validate_actual(
    np: Any,
    reference: tuple[Any, Any, Any],
    actual: tuple[Any, Any, Any],
    equations: dict[str, Any],
    seconds: float,
) -> dict[str, Any]:
    if not isinstance(actual, tuple) or len(actual) != 3:
        raise RuntimeError("host candidate must contain exactly three output arrays")
    actual_arrays = tuple(np.asarray(item) for item in actual)
    expected_shapes = (_VALUE_SHAPE, _VALUE_SHAPE, _GATE_SHAPE)
    expected_nbytes = (8 * 1024**2, 8 * 1024**2, 64 * 1024)
    shape_dtype_bytes_exact = all(
        item.shape == shape
        and item.dtype == np.dtype(np.float32)
        and int(item.nbytes) == nbytes
        for item, shape, nbytes in zip(actual_arrays, expected_shapes, expected_nbytes)
    )
    metrics = {
        name: _numerical_metrics(np, expected, observed)
        for name, expected, observed in zip(
            ("u", "w", "gamma"), reference, actual_arrays
        )
    }
    u_chunks, _ = _chunk_outputs(actual_arrays[0], actual_arrays[2])
    w_chunks, _ = _chunk_outputs(actual_arrays[1], actual_arrays[2])
    residuals = {
        "u": _equation_residual(
            np, equations["unit_lower"], u_chunks, equations["rhs_u"]
        ),
        "w": _equation_residual(
            np, equations["unit_lower"], w_chunks, equations["rhs_w"]
        ),
    }
    output_thresholds = {
        "u": {"relative_l2_lt": 2e-4, "cosine_gte": 0.99999995, "max_abs_lte": 5e-5},
        "w": {"relative_l2_lt": 2e-4, "cosine_gte": 0.99999995, "max_abs_lte": 5e-5},
        "gamma": {
            "relative_l2_lt": 2e-6,
            "cosine_gte": 0.999999999,
            "max_abs_lte": 2e-6,
        },
    }
    output_checks: dict[str, bool] = {}
    for name, threshold in output_thresholds.items():
        item = metrics[name]
        output_checks[f"{name}_finite"] = (
            item["finite_expected"]
            and item["finite_actual"]
            and item["finite_difference"]
        )
        output_checks[f"{name}_relative_l2"] = (
            item["relative_l2"] < threshold["relative_l2_lt"]
        )
        output_checks[f"{name}_cosine"] = item["cosine"] >= threshold["cosine_gte"]
        output_checks[f"{name}_max_absolute"] = (
            item["max_absolute"] <= threshold["max_abs_lte"]
        )
    residual_checks = {
        "actual_u_residual_relative_l2": residuals["u"]["relative_l2"] < 3e-4,
        "actual_w_residual_relative_l2": residuals["w"]["relative_l2"] < 3e-4,
        "actual_u_residual_max_absolute": residuals["u"]["max_absolute"] <= 1e-4,
        "actual_w_residual_max_absolute": residuals["w"]["max_absolute"] <= 1e-4,
    }
    duration_hard = math.isfinite(seconds) and 0 <= seconds < _HARD_SECONDS
    duration_promotion = math.isfinite(seconds) and 0 <= seconds < _PROMOTION_SECONDS
    correctness = (
        shape_dtype_bytes_exact
        and all(output_checks.values())
        and all(residual_checks.values())
    )
    passed = correctness and duration_hard
    promotion = passed and duration_promotion
    names = ("u", "w", "gamma")
    actual_tuple_hash = _framed_tuple_sha256(tuple(zip(names, actual_arrays)))
    reference_tuple_hash = _framed_tuple_sha256(tuple(zip(names, reference)))
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
        "metrics": metrics,
        "equation_residuals": residuals,
        "actual_framed_tuple_sha256": actual_tuple_hash,
        "reference_framed_tuple_sha256": reference_tuple_hash,
        "output_checks": output_checks,
        "residual_checks": residual_checks,
        "duration_checks": {
            "hard_below_2_seconds": duration_hard,
            "promotion_below_250_milliseconds": duration_promotion,
        },
        "thresholds": {
            "outputs": output_thresholds,
            "equation_residual_relative_l2_lt": 3e-4,
            "equation_residual_max_absolute_lte": 1e-4,
            "promotion_seconds_lt": _PROMOTION_SECONDS,
            "hard_seconds_lt": _HARD_SECONDS,
        },
    }
    if not passed:
        raise RuntimeError(
            "GDN runtime candidate failed numerical or hard-duration gate"
        )
    return result


def _run_rocm_body(
    args: argparse.Namespace,
    output: TextIO,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    *,
    environment: dict[str, str],
    library_manifest: dict[str, Any],
    helper: ModuleType,
    _postcheck_state: dict[str, bool],
    _dependencies: tuple[
        Any,
        Any,
        Any,
        Any,
        Any,
        Callable[..., Any],
        Callable[..., Any],
        Callable[..., Any],
    ]
    | None = None,
) -> int:
    command_buffer = helper._prove_command_buffers_disabled(environment)
    _emit(
        {
            "record_type": "command_buffer_environment_proof",
            "timestamp": _utc_now(),
            "proof": command_buffer,
            "counters": dict(counters),
        },
        output,
    )
    _journal_checkpoint(
        require_clean_boot, output, "before_backend_initialization", counters
    )
    counters["backend_initialization_attempts"] += 1
    try:
        if _dependencies is None:
            oracle_module = _load_oracle_module()
            import jax
            import jax.numpy as jnp
            import jaxlib
            import numpy as np
            from jax.extend import backend as jax_backend

            from skyrl.tx.kernels.rocm.gdn_prepare_ffi import (
                gdn_prepare_s512,
                register_gdn_prepare_s512,
            )

            oracle = oracle_module.gdn_prepare_s512_numpy
        else:
            (
                jax,
                jnp,
                jaxlib,
                jax_backend,
                np,
                gdn_prepare_s512,
                register_gdn_prepare_s512,
                oracle,
            ) = _dependencies
        wrapper_path = _source_files()["wrapper_source_sha256"].resolve()
        wrapper_module = sys.modules.get("skyrl.tx.kernels.rocm.gdn_prepare_ffi")
        if _dependencies is None and (
            wrapper_module is None
            or not isinstance(getattr(wrapper_module, "__file__", None), str)
            or Path(wrapper_module.__file__).resolve() != wrapper_path
        ):
            raise RuntimeError("loaded GDN wrapper is not the exact committed file")
        backend = helper._backend_manifest(jax, jaxlib, jax_backend)
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
    finally:
        _journal_checkpoint(
            require_clean_boot,
            output,
            "after_backend_initialization_attempt",
            counters,
        )

    operation: dict[str, Any] | None = None
    final_library: dict[str, Any] | None = None
    compiled = None
    try:
        compiled, compile_report = _compile_unreleased(
            jax,
            jnp,
            gdn_prepare_s512,
            register_gdn_prepare_s512,
            helper,
            args.library,
            args.library_sha256,
            int(library_manifest["size_bytes"]),
            require_clean_boot,
            counters,
            output,
        )
        try:
            host, host_report = _construct_host_case(np)
            counters["host_candidate_arrays"] += 4
            reference, reference_report, equations = _construct_reference(
                np, oracle, host, counters
            )
            host_proof = {
                "input_identity_gate_passed": host_report["passed"],
                "dense_reference_gate_passed": reference_report["passed"],
                "passed": host_report["passed"] and reference_report["passed"],
            }
            _emit(
                {
                    "record_type": "host_oracle",
                    "timestamp": _utc_now(),
                    "inputs": host_report,
                    "reference": reference_report,
                    "release_gate": host_proof,
                    "device_transfer_started": False,
                    "counters": dict(counters),
                },
                output,
            )
        finally:
            _journal_checkpoint(
                require_clean_boot, output, "after_host_oracle_attempt", counters
            )
        executable = _release_checked(
            compiled, compile_report["release_gate"], host_proof, counters
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
            device_inputs = _tuple_device_put(jax, host, counters)
        finally:
            _journal_checkpoint(
                require_clean_boot, output, "after_input_device_put_attempt", counters
            )
        device_outputs, seconds = _dispatch(
            jax,
            executable,
            device_inputs,
            require_clean_boot,
            counters,
            output,
        )
        try:
            actual = _tuple_device_get(jax, device_outputs, counters)
        finally:
            _journal_checkpoint(
                require_clean_boot,
                output,
                "after_candidate_device_get_attempt",
                counters,
            )
        try:
            validation = _validate_actual(np, reference, actual, equations, seconds)
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
                output,
                "after_host_validation_attempt",
                counters,
            )
        operation = {
            "compile_report": compile_report,
            "host_report": host_report,
            "reference_report": reference_report,
            "validation": validation,
        }
    finally:
        if compiled is not None:
            del compiled
        try:
            final_library = helper._assert_same_library(args.library, library_manifest)
        finally:
            try:
                _journal_checkpoint(
                    require_clean_boot, output, "after_library_postcheck", counters
                )
            finally:
                _postcheck_state["completed"] = True

    if operation is None or final_library is None:
        raise RuntimeError("GDN runtime operation did not complete")
    if counters != _completed_counters():
        raise RuntimeError("GDN runtime one-shot counter contract was not exact")
    validation = operation["validation"]
    _emit(
        {
            "record_type": "runtime_passed",
            "timestamp": _utc_now(),
            "status": (
                "passed_exact_s512_runtime_promotable"
                if validation["promotion_passed"]
                else "passed_exact_s512_runtime_unpromotable"
            ),
            "compile_release_gate": operation["compile_report"]["release_gate"],
            "host_oracle_gate": {
                "inputs": operation["host_report"]["passed"],
                "reference": operation["reference_report"]["passed"],
            },
            "validation": validation,
            "library": {
                key: value for key, value in final_library.items() if key != "identity"
            },
            "counters": dict(counters),
            "recurrence_validated": False,
            "vjp_validated": False,
            "model_path_validated": False,
            "training_validated": False,
            "performance_benchmark_authorized": False,
        },
        output,
    )
    return 0


def _run_rocm(
    args: argparse.Namespace,
    output: TextIO,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    *,
    environment: dict[str, str],
    library_manifest: dict[str, Any],
    helper: ModuleType,
    _dependencies: tuple[
        Any,
        Any,
        Any,
        Any,
        Any,
        Callable[..., Any],
        Callable[..., Any],
        Callable[..., Any],
    ]
    | None = None,
) -> int:
    """Run with exactly one library postcheck, including every failure path."""
    postcheck = {"completed": False}
    try:
        return _run_rocm_body(
            args,
            output,
            require_clean_boot,
            counters,
            environment=environment,
            library_manifest=library_manifest,
            helper=helper,
            _postcheck_state=postcheck,
            _dependencies=_dependencies,
        )
    finally:
        if not postcheck["completed"]:
            try:
                helper._assert_same_library(args.library, library_manifest)
            finally:
                try:
                    _journal_checkpoint(
                        require_clean_boot,
                        output,
                        "after_library_postcheck",
                        counters,
                    )
                finally:
                    postcheck["completed"] = True


def _execute(args: argparse.Namespace, output: TextIO) -> int:
    counters = _zero_counters()
    _emit(
        {
            "record_type": "manifest",
            "timestamp": _utc_now(),
            "platform_requested": args.platform,
            "allow_gpu": args.allow_gpu,
            "case": args.case,
            "scope": (
                "abstract_refusal"
                if args.platform == "abstract"
                else "guarded_s512_dense_all_valid_runtime"
            ),
            "contract": _exact_contract(),
            "fresh_process_required": True,
            "outer_profile_rocm_supervision_required": True,
            "raw_library_path_emitted": False,
            "raw_ir_emitted": False,
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
                    "use a fresh profile_rocm.py child with --platform rocm, "
                    f"--allow-gpu, --case {_CASE}, --library, --library-sha256, and --output"
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
        helper = _load_compile_helper()
        stage = "library_preflight"
        library = helper._validate_library_path(args.library, args.library_sha256)
        _emit(
            {
                "record_type": "prerequisite_proof",
                "timestamp": _utc_now(),
                "bound_sources": bound_sources,
                "library": {
                    key: value for key, value in library.items() if key != "identity"
                },
                "counters": dict(counters),
            },
            output,
        )
        stage = "bounded_environment"
        numeric_environment = _validate_host_numeric_environment()
        environment = helper._configure_rocm_environment()
        _emit(
            {
                "record_type": "environment",
                "timestamp": _utc_now(),
                "host_numeric_environment": numeric_environment,
                "command_buffers": helper._prove_command_buffers_disabled(environment),
                "raw_unrelated_environment_emitted": False,
                "counters": dict(counters),
            },
            output,
        )
        guarded_process, require_clean_boot = helper._load_safety_helpers()
        stage = "safety_preflight"
        with guarded_process() as raw_safety:
            _emit(
                {
                    "record_type": "safety_preflight",
                    "timestamp": _utc_now(),
                    "safety": _public_safety_preflight(raw_safety),
                    "counters": dict(counters),
                },
                output,
            )
            stage = "runtime"
            try:
                result = _run_rocm(
                    args,
                    output,
                    require_clean_boot,
                    counters,
                    environment=environment,
                    library_manifest=library,
                    helper=helper,
                )
            finally:
                try:
                    postflight = _public_clean_safety(
                        require_clean_boot(), "safety_postflight"
                    )
                except Exception:
                    stage = "safety_postflight"
                    raise
                _emit(
                    {
                        "record_type": "safety_postflight",
                        "timestamp": _utc_now(),
                        "safety": postflight,
                        "counters": dict(counters),
                    },
                    output,
                )
        _emit(
            {
                "record_type": "completed",
                "timestamp": _utc_now(),
                "status": "passed",
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
                **_redacted_message(error),
                "counters": dict(counters),
            },
            output,
        )
        return 1


def _open_exclusive_output(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
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
