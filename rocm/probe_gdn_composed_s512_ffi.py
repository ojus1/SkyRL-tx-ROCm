#!/usr/bin/env python3
"""Fail-closed one-shot gate for composed exact-S512 GDN prepare and execute.

The default ``abstract`` mode emits a refusal without importing NumPy, JAX, a
SkyRL ROCm module, or either native library.  The explicit ROCm mode requires
direct ``profile_rocm.py`` supervision and the same clean-boot, headless,
exclusive-device, exact-stack, graph-free environment as the separately
qualified prepare and execute probes.

This probe does not add a new production wrapper.  It traces the two existing
sealed typed-FFI wrappers in one ``jax.jit`` function and requires exactly this
dataflow::

    (U, W, gamma) = prepare(K, V, g, beta)
    (O, S8) = execute(Q, K, U, W, gamma, S0)

Both StableHLO and optimized HLO must contain exactly the two exact custom-call
targets, in that order, with the three prepare results feeding execute operand
positions 2, 3, and 4.  Other custom calls, loops, aliases, inline host work,
warmup, replay, graph capture, backward, and model execution are forbidden.
The runtime mode transfers only the six transformed primals and consumes one
checked executable capability exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import math
import os
import re
import stat
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, TextIO

_CASE = "s512-composed-one-shot"
_PREPARE_TARGET = "skyrl_gdn_prepare_s512_f32_v1"
_EXECUTE_TARGET = "skyrl_gdn_execute_s512_f32_bf16_v1"
_PREPARE_BASENAME = "libskyrl_gdn_prepare_s512_gfx1100.so"
_EXECUTE_BASENAME = "libskyrl_gdn_execute_s512_gfx1100.so"
_PREPARE_SHA256 = "56f667eee1eddac6b881feba18fc9a3315bc2b22e8fdfe08effa36e32e6315ef"
_EXECUTE_SHA256 = "435487abce7299bf9a835840f195cb95f0a644804fc2c41843ba9f5621ebd53b"
_PREPARE_SIZE = 80_232
_EXECUTE_SIZE = 80_128

_QUERY_SHAPE = (1, 512, 16, 128)
_VALUE_SHAPE = (1, 512, 32, 128)
_GATE_SHAPE = (1, 512, 32)
_STATE_SHAPE = (1, 32, 128, 128)
_INPUT_NAMES = ("query", "key", "value", "g", "beta", "initial_state")
_INPUT_SHAPES = (
    _QUERY_SHAPE,
    _QUERY_SHAPE,
    _VALUE_SHAPE,
    _GATE_SHAPE,
    _GATE_SHAPE,
    _STATE_SHAPE,
)
_OUTPUT_NAMES = ("output", "final_state")
_ARGUMENT_BYTES = 19_005_440
_INTERMEDIATE_BYTES = 16_842_752
_LOGICAL_OUTPUT_BYTES = 6_291_456
_COMPILER_OUTPUT_BYTES = _LOGICAL_OUTPUT_BYTES + 16
_MAX_TEMP_BYTES = 64 * 1024**2
_MAX_COMBINED_BYTES = 96 * 1024**2
_MAX_GPU_POWER_WATTS = 400.0
_LOWER_HARD_SECONDS = 30.0
_COMPILE_HARD_SECONDS = 180.0
_RUNTIME_PROMOTION_SECONDS = 0.250
_RUNTIME_HARD_SECONDS = 1.0
_CHECKED_TOKEN = object()

_EXPECTED_INPUT_SHA256 = {
    "query": "558299f9f5a6c8ac7ef81187ce3b75c0cb4bccee176d19fde6f36c9193f19d14",
    "key": "8f54cb823b5d2f6311473dddb410f369bb1022afd300f309c5a0396793777dbe",
    "value": "08ccd639083cf31b4453b4bdf290211d331e5926c2c318929380c5017e3d740a",
    "g": "0fd8275826e63c86d719993444cf790c2f6eb087e9ec69c180bbfd49def88c44",
    "beta": "22f4d27f242617749eaa4d74e4395e7daaa9fc7852350bcdef3dcea7143044af",
    "initial_state": "b8b5b6cfc479df1fbc0dbe9ed743136613da386535b6e0a3034d62abf20d97f2",
}
_EXPECTED_PREPARED_SHA256 = {
    "prepared_u": "53eb9a96b97593b3af7677fd350b9016fd5ab37df1dff08016fb928ccfaf829f",
    "prepared_w": "15811df05a34c1e8e8f09305eecb0e62639b9152a46728ecc334ca2caad85e0b",
    "gamma": "1a8c8a7d83cd03f2ec848646da7835d6bea59c2696c97edd64b523c08ded4dca",
}
_EXPECTED_REFERENCE_SHA256 = {
    "output": "383b59c47d65417dff22c60214c9acb017542cc7d92622a93fde6ca9f682f14b",
    "final_state": "1f008feffeb448fea3378cba54d994c4a0c1640fba6f168d898defea0f425cd1",
}
_EXPECTED_INPUT_TUPLE_SHA256 = "d128ebfee806fbd66d6b9fd0891b2c9a7f8fe12e882ec36ce0319ba803bc2a7f"
_EXPECTED_REFERENCE_TUPLE_SHA256 = "a53d30c7862b9cfff1e7ca5c64797d05380261687f54b8598227beabcbaf3cd3"
_INPUT_DOMAIN = b"skyrl-gdn-composed-s512-input-v1\x00"
_REFERENCE_DOMAIN = b"skyrl-gdn-composed-s512-reference-v1\x00"

_EXPECTED_SOURCE_SHA256 = {
    "execute_probe_helper": "8fec271483be3652e4ec43e5cd6e909ae57268d69206c1be45f29100abbd1fae",
    "prepare_probe_helper": "cacf782aa58a9735b44492ba6df7725e3ac777c7e57170c174c1563ff2f3c2b5",
    "prepare_wrapper": "39204094b1d9e1e8caddcc833cc02edb9dab2b7e32bbee75c5462fd771a6052f",
    "execute_wrapper": "a588a1c17960172f07f6e504aa5dbaf5d8f9be91e3edf7775de0e632ab52851b",
    "prepare_hip": "8deaf58f5bf68e936472c434c985a52bea8ea1e26b75983af9ec8ae9c1e80f45",
    "execute_hip": "f858c129cb781a9c9d4e87679973b454c7cf852b640f58ba13ad53e003059f36",
    "prepare_build": "7982783e4689aa93d103f0a15a4ed240da506bdd16ed8215a6c1c7b39acb40b7",
    "execute_build": "25a1590df7f5ed89be983bb6c5a8649791eb4916becddd9d6f75919ab534821d",
    "prepare_oracle": "019da59345f04ed2bfea553d797deb1a1c15a34c11147780616d4b4a44195aad",
    "execute_oracle": "cca11be2212603d74d52ce936910283805c1fa5e73c6bfbc5c79a98e529949c2",
    "sealed_loader": "66b868b7909a2279d5ddca0e1582f8563e8097723e970d09fce733aef2ba425a",
    "safety": "7ad79b9b9b54089add72dff65ea18505a794c51f0c4bafe231fbd3b745f23ba6",
    "profile_rocm": "3dde5139a95ba9698b298a638af07125dcf4fb7a1a91a55c2b5ea4762f89db8f",
    "package_skyrl": "667d4a15b970b851e20d17510224670c14646cfb6d5a1e388ca6b9cc6da8bf41",
    "package_tx": "a7abb3e76d66df1f4472bb7a02b032ef31b959ca937fd351637b4e9b4a8fa95a",
    "package_kernels": "40abe638c7726fe5680b7c88321042016a0f695d86acfbef52337421e7257c1a",
    "package_rocm": "6d12a789cf1108538a04fbacd0b38a15dbcb8255cd0ca0fadf5a76c4191a4cfd",
}

_BOUND_RUNTIME_MODULES = {
    "skyrl": "package_skyrl",
    "skyrl.tx": "package_tx",
    "skyrl.tx.kernels": "package_kernels",
    "skyrl.tx.kernels.rocm": "package_rocm",
    "skyrl.tx.kernels.rocm.gdn_ffi_smoke": "sealed_loader",
    "skyrl.tx.kernels.rocm.gdn_prepare_ffi": "prepare_wrapper",
    "skyrl.tx.kernels.rocm.gdn_execute_ffi": "execute_wrapper",
}

_JOURNAL_STAGES = frozenset(
    {
        "before_host_oracle",
        "after_host_oracle_attempt",
        "before_backend_initialization",
        "after_backend_initialization_attempt",
        "after_prepare_registration_attempt",
        "after_execute_registration_attempt",
        "after_lower_attempt",
        "after_compile_attempt",
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
    output.write(json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n")
    output.flush()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_files() -> dict[str, Path]:
    repo = Path(__file__).resolve().parent.parent
    kernels = repo / "skyrl" / "tx" / "kernels"
    rocm = kernels / "rocm"
    ffi = rocm / "ffi"
    return {
        "probe": Path(__file__),
        "execute_probe_helper": repo / "rocm" / "probe_gdn_execute_s512_ffi.py",
        "prepare_probe_helper": repo / "rocm" / "probe_gdn_prepare_s512_compile.py",
        "prepare_wrapper": rocm / "gdn_prepare_ffi.py",
        "execute_wrapper": rocm / "gdn_execute_ffi.py",
        "prepare_hip": ffi / "gdn_prepare_s512.hip",
        "execute_hip": ffi / "gdn_execute_s512.hip",
        "prepare_build": ffi / "build_gdn_prepare_s512_gfx1100.sh",
        "execute_build": ffi / "build_gdn_execute_s512_gfx1100.sh",
        "prepare_oracle": rocm / "gdn_prepare_oracle.py",
        "execute_oracle": rocm / "gdn_execute_oracle.py",
        "sealed_loader": rocm / "gdn_ffi_smoke.py",
        "safety": repo / "rocm" / "amdgpu_safety.py",
        "profile_rocm": repo / "rocm" / "profile_rocm.py",
        "package_skyrl": repo / "skyrl" / "__init__.py",
        "package_tx": repo / "skyrl" / "tx" / "__init__.py",
        "package_kernels": kernels / "__init__.py",
        "package_rocm": rocm / "__init__.py",
    }


def _source_hashes() -> dict[str, str]:
    return {f"{name}_source_sha256": _file_sha256(path) for name, path in _source_files().items()}


def _assert_bound_sources() -> dict[str, Any]:
    files = _source_files()
    observed = {name: _file_sha256(files[name]) for name in _EXPECTED_SOURCE_SHA256}
    if observed != _EXPECTED_SOURCE_SHA256:
        raise RuntimeError("composed GDN dependency source hash mismatch")
    return {"passed": True, "all_executable_dependencies_exact": True, **observed}


def _exact_source_spec(module_name: str, source_name: str) -> dict[str, Any]:
    """Prove normal import resolution points at one committed source file."""

    expected_path = _source_files()[source_name].resolve(strict=True)
    expected_sha256 = _EXPECTED_SOURCE_SHA256[source_name]
    if _file_sha256(expected_path) != expected_sha256:
        raise RuntimeError(f"refusing changed {source_name} source")
    spec = importlib.util.find_spec(module_name)
    if spec is None or not isinstance(spec.origin, str) or spec.loader is None:
        raise RuntimeError(f"could not resolve exact {module_name} source")
    if type(spec.loader) is not importlib.machinery.SourceFileLoader:
        raise RuntimeError(f"{module_name} does not use the exact source-file loader")
    origin = Path(spec.origin).resolve(strict=True)
    loader_path = Path(spec.loader.get_filename(module_name)).resolve(strict=True)
    if origin != expected_path or loader_path != expected_path:
        raise RuntimeError(f"{module_name} resolves outside the committed repository")
    if spec.submodule_search_locations is not None:
        locations = [Path(item).resolve(strict=True) for item in spec.submodule_search_locations]
        if locations != [expected_path.parent]:
            raise RuntimeError(f"{module_name} has an unexpected package search path")
    return {
        "module_name_sha256": hashlib.sha256(module_name.encode()).hexdigest(),
        "source_sha256": expected_sha256,
        "origin_exact": True,
        "loader_exact": True,
        "package_search_path_exact": spec.submodule_search_locations is None
        or locations == [expected_path.parent],
        "raw_path_emitted": False,
    }


def _exact_loaded_source(module_name: str, source_name: str) -> dict[str, Any]:
    """Bind an imported module to its expected source and rehash it post-import."""

    expected_path = _source_files()[source_name].resolve(strict=True)
    expected_sha256 = _EXPECTED_SOURCE_SHA256[source_name]
    module = sys.modules.get(module_name)
    if module is None or not isinstance(getattr(module, "__file__", None), str):
        raise RuntimeError(f"exact {module_name} module was not loaded")
    spec = getattr(module, "__spec__", None)
    loader = getattr(module, "__loader__", None)
    if (
        spec is None
        or not isinstance(getattr(spec, "origin", None), str)
        or loader is None
        or type(loader) is not importlib.machinery.SourceFileLoader
    ):
        raise RuntimeError(f"loaded {module_name} has no exact source identity")
    paths = (
        Path(module.__file__).resolve(strict=True),
        Path(spec.origin).resolve(strict=True),
        Path(loader.get_filename(module_name)).resolve(strict=True),
    )
    if any(path != expected_path for path in paths):
        raise RuntimeError(f"loaded {module_name} is not the committed source")
    raw_package_path = getattr(module, "__path__", None)
    expected_package = module_name if raw_package_path is not None else module_name.rpartition(".")[0]
    if getattr(module, "__package__", None) != expected_package:
        raise RuntimeError(f"loaded {module_name} has an unexpected package identity")
    if raw_package_path is not None:
        package_paths = [Path(item).resolve(strict=True) for item in raw_package_path]
        if package_paths != [expected_path.parent]:
            raise RuntimeError(f"loaded {module_name} has an unexpected package path")
    if _file_sha256(expected_path) != expected_sha256:
        raise RuntimeError(f"loaded {module_name} source changed during import")
    return {
        "module_name_sha256": hashlib.sha256(module_name.encode()).hexdigest(),
        "source_sha256": expected_sha256,
        "file_origin_loader_exact": True,
        "package_identity_exact": True,
        "package_search_path_exact": raw_package_path is None
        or package_paths == [expected_path.parent],
        "post_import_rehash_exact": True,
        "raw_path_emitted": False,
    }


def _import_bound_wrappers() -> tuple[ModuleType, ModuleType, dict[str, Any]]:
    """Import the wrappers through the canonical package chain and bind them."""

    resolved: dict[str, dict[str, Any]] = {}
    loaded: dict[str, dict[str, Any]] = {}
    modules: dict[str, ModuleType] = {}
    for module_name, source_name in _BOUND_RUNTIME_MODULES.items():
        resolved[module_name] = _exact_source_spec(module_name, source_name)
        modules[module_name] = importlib.import_module(module_name)
        loaded[module_name] = _exact_loaded_source(module_name, source_name)

    loader = modules["skyrl.tx.kernels.rocm.gdn_ffi_smoke"]
    prepare = modules["skyrl.tx.kernels.rocm.gdn_prepare_ffi"]
    execute = modules["skyrl.tx.kernels.rocm.gdn_execute_ffi"]
    relationships = {
        "prepare_uses_exact_sealed_loader": getattr(prepare, "_sealed_loader", None)
        is loader,
        "execute_uses_exact_sealed_loader": getattr(execute, "_sealed_loader", None)
        is loader,
        "prepare_target_exact": getattr(prepare, "GDN_PREPARE_S512_TARGET", None)
        == _PREPARE_TARGET,
        "execute_target_exact": getattr(execute, "GDN_EXECUTE_S512_TARGET", None)
        == _EXECUTE_TARGET,
        "prepare_entrypoints_exact": all(
            callable(getattr(prepare, name, None))
            and getattr(prepare, name).__module__ == prepare.__name__
            for name in ("gdn_prepare_s512", "register_gdn_prepare_s512")
        ),
        "execute_entrypoints_exact": all(
            callable(getattr(execute, name, None))
            and getattr(execute, name).__module__ == execute.__name__
            for name in ("gdn_execute_s512", "register_gdn_execute_s512")
        ),
    }
    if not all(relationships.values()):
        raise RuntimeError("loaded GDN wrapper dependency graph is not exact")
    # Rehash every executable package/wrapper dependency once more after both
    # wrappers and their common loader have finished importing.
    final_rehash = {
        source_name: _file_sha256(_source_files()[source_name])
        for source_name in _BOUND_RUNTIME_MODULES.values()
    }
    expected = {
        source_name: _EXPECTED_SOURCE_SHA256[source_name]
        for source_name in _BOUND_RUNTIME_MODULES.values()
    }
    if final_rehash != expected:
        raise RuntimeError("GDN wrapper source changed after dependency import")
    proof = {
        "passed": True,
        "resolved_before_import": resolved,
        "loaded_after_import": loaded,
        "relationships": relationships,
        "final_source_sha256": final_rehash,
        "all_runtime_modules_exact": True,
        "raw_paths_emitted": False,
    }
    return prepare, execute, proof


def _load_exact_module(path: Path, expected: str, name: str) -> ModuleType:
    if _file_sha256(path) != expected:
        raise RuntimeError(f"refusing changed {name} source")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not create {name} module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(str(module.__file__)).resolve() != path.resolve() or _file_sha256(path) != expected:
        raise RuntimeError(f"{name} source identity changed while loading")
    return module


def _load_helpers() -> tuple[ModuleType, ModuleType]:
    files = _source_files()
    execute = _load_exact_module(
        files["execute_probe_helper"], _EXPECTED_SOURCE_SHA256["execute_probe_helper"], "_skyrl_gdn_execute_probe_helper"
    )
    prepare = _load_exact_module(
        files["prepare_probe_helper"], _EXPECTED_SOURCE_SHA256["prepare_probe_helper"], "_skyrl_gdn_prepare_probe_helper"
    )
    return execute, prepare


def _sha256_arg(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise argparse.ArgumentTypeError("must be exactly 64 lowercase hexadecimal digits")
    return value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("abstract", "rocm"), default="abstract")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--compile-diagnostic", action="store_true")
    parser.add_argument("--case", choices=(_CASE,))
    parser.add_argument("--prepare-library", type=Path)
    parser.add_argument("--prepare-library-sha256", type=_sha256_arg)
    parser.add_argument("--execute-library", type=Path)
    parser.add_argument("--execute-library-sha256", type=_sha256_arg)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    guarded = (
        args.case,
        args.prepare_library,
        args.prepare_library_sha256,
        args.execute_library,
        args.execute_library_sha256,
        args.output,
    )
    if args.platform == "abstract":
        if args.allow_gpu or args.compile_diagnostic or any(item is not None for item in guarded):
            parser.error("guarded options require --platform rocm")
        return args
    if not args.allow_gpu or args.case != _CASE:
        parser.error(f"--platform rocm requires --allow-gpu --case {_CASE}")
    if args.prepare_library is None or args.prepare_library_sha256 != _PREPARE_SHA256:
        parser.error("ROCm mode requires the exact prepare library and SHA-256")
    if args.execute_library is None or args.execute_library_sha256 != _EXECUTE_SHA256:
        parser.error("ROCm mode requires the exact execute library and SHA-256")
    if args.output is None or not args.output.is_absolute():
        parser.error("ROCm mode requires an absolute --output")
    try:
        parent = args.output.parent.resolve(strict=True)
    except OSError:
        parser.error("--output parent must already exist and resolve canonically")
    if args.output != parent / args.output.name or args.output.exists() or args.output.is_symlink():
        parser.error("--output must be fresh, canonical, and symlink-free")
    return args


def _exact_contract() -> dict[str, Any]:
    return {
        "operation": "gdn_composed_prepare_execute_s512_one_shot",
        "case": _CASE,
        "targets_in_dataflow_order": [_PREPARE_TARGET, _EXECUTE_TARGET],
        "inputs": [
            {"name": name, "shape": list(shape), "dtype": "float32"}
            for name, shape in zip(_INPUT_NAMES, _INPUT_SHAPES, strict=True)
        ],
        "outputs": [
            {"name": "output", "shape": list(_VALUE_SHAPE), "dtype": "bfloat16"},
            {"name": "final_state", "shape": list(_STATE_SHAPE), "dtype": "float32"},
        ],
        "compile_gate": {
            "exact_custom_calls_per_dialect": 2,
            "prepare_results_feed_execute_positions": [2, 3, 4],
            "while_calls": 0,
            "alias_bytes": 0,
            "argument_bytes": _ARGUMENT_BYTES,
            "logical_intermediate_bytes": _INTERMEDIATE_BYTES,
            "logical_output_bytes": _LOGICAL_OUTPUT_BYTES,
            "compiler_output_bytes": _COMPILER_OUTPUT_BYTES,
            "temporary_bytes_range": [_INTERMEDIATE_BYTES, _MAX_TEMP_BYTES],
            "combined_bytes_maximum": _MAX_COMBINED_BYTES,
        },
        "numerical_thresholds": {
            "output_relative_l2_lte": 5e-3,
            "output_cosine_gte": 0.9999,
            "output_max_absolute_lte": 5e-3,
            "state_relative_l2_lte": 2e-4,
            "state_cosine_gte": 0.999999,
            "state_max_absolute_lte": 2e-4,
        },
        "one_shot": True,
        "outer_supervision": {
            "maximum_junction_temperature_c": 90.0,
            "maximum_gpu_power_watts": _MAX_GPU_POWER_WATTS,
            "maximum_gpu_vram_gib": 24.0,
        },
        "warmup": 0,
        "replay": 0,
        "graph": 0,
        "backward": 0,
        "model": 0,
    }


def _zero_counters() -> dict[str, int]:
    names = (
        "prepare_oracle_attempts", "prepare_oracle_completions", "execute_oracle_attempts",
        "execute_oracle_completions", "backend_initialization_attempts",
        "backend_initialization_completions", "prepare_registration_attempts",
        "prepare_registration_completions", "execute_registration_attempts",
        "execute_registration_completions", "shape_dtype_structs", "prepare_trace_calls",
        "execute_trace_calls", "lower_attempts", "lower_completions", "compile_attempts",
        "compile_completions", "capability_release_attempts", "capability_release_completions",
        "tuple_device_put_attempts", "tuple_device_put_completions", "device_put_leaves",
        "input_readiness_barriers", "checked_executable_attempts",
        "checked_executable_completions", "output_readiness_barriers",
        "tuple_device_get_attempts", "tuple_device_get_completions", "device_get_leaves",
        "warmup_invocations", "replay_invocations", "graph_invocations",
        "gpu_reference_invocations", "gpu_reduction_invocations", "backward_invocations",
        "model_invocations",
    )
    return {name: 0 for name in names}


def _completed_runtime_counters() -> dict[str, int]:
    counters = _zero_counters()
    counters.update(
        {
            "prepare_oracle_attempts": 1,
            "prepare_oracle_completions": 1,
            "execute_oracle_attempts": 1,
            "execute_oracle_completions": 1,
            "backend_initialization_attempts": 1,
            "backend_initialization_completions": 1,
            "prepare_registration_attempts": 1,
            "prepare_registration_completions": 1,
            "execute_registration_attempts": 1,
            "execute_registration_completions": 1,
            "shape_dtype_structs": 6,
            "prepare_trace_calls": 1,
            "execute_trace_calls": 1,
            "lower_attempts": 1,
            "lower_completions": 1,
            "compile_attempts": 1,
            "compile_completions": 1,
            "capability_release_attempts": 1,
            "capability_release_completions": 1,
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
    return counters


def _completed_compile_diagnostic_counters() -> dict[str, int]:
    counters = _zero_counters()
    counters.update(
        {
            "backend_initialization_attempts": 1,
            "backend_initialization_completions": 1,
            "prepare_registration_attempts": 1,
            "prepare_registration_completions": 1,
            "execute_registration_attempts": 1,
            "execute_registration_completions": 1,
            "shape_dtype_structs": 6,
            "prepare_trace_calls": 1,
            "execute_trace_calls": 1,
            "lower_attempts": 1,
            "lower_completions": 1,
            "compile_attempts": 1,
            "compile_completions": 1,
        }
    )
    return counters


def _assert_terminal_counters(
    counters: dict[str, int], *, compile_diagnostic: bool
) -> dict[str, Any]:
    expected = (
        _completed_compile_diagnostic_counters()
        if compile_diagnostic
        else _completed_runtime_counters()
    )
    if counters != expected:
        mode = "compile diagnostic" if compile_diagnostic else "runtime one-shot"
        raise RuntimeError(f"{mode} counter contract was not exact")
    forbidden = (
        "warmup_invocations",
        "replay_invocations",
        "graph_invocations",
        "gpu_reference_invocations",
        "gpu_reduction_invocations",
        "backward_invocations",
        "model_invocations",
    )
    if any(counters[name] != 0 for name in forbidden):
        raise RuntimeError("forbidden composed invocation counter was nonzero")
    return {
        "passed": True,
        "mode": "compile_diagnostic" if compile_diagnostic else "runtime",
        "exact_full_counter_map": True,
        "forbidden_invocation_counts_zero": True,
    }


def _assert_fresh_process() -> None:
    imported = sorted(
        name for name in sys.modules
        if name in {"jax", "jaxlib", "numpy", "skyrl"}
        or name.startswith(("jax.", "jaxlib.", "numpy.", "skyrl."))
    )
    if imported:
        raise RuntimeError("composed GDN gate requires a fresh accelerator process")


def _validate_library(path: Path, digest: str, basename: str, size: int) -> dict[str, Any]:
    if not path.is_absolute() or path.name != basename:
        raise ValueError("GDN library path/basename is not exact")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise ValueError("GDN library must be a real regular file")
    if before.st_uid != os.getuid() or stat.S_IMODE(before.st_mode) != 0o600:
        raise ValueError("GDN library must be user-owned mode 0600")
    if path.resolve(strict=True) != path:
        raise ValueError("GDN library must be canonical and symlink-free")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    checksum = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (before.st_dev, before.st_ino, before.st_size):
            raise RuntimeError("GDN library changed while opening")
        while chunk := os.read(descriptor, 1024 * 1024):
            checksum.update(chunk)
    finally:
        os.close(descriptor)
    after = path.lstat()
    identity = (int(after.st_dev), int(after.st_ino), int(after.st_size), int(after.st_mtime_ns))
    if identity != (int(before.st_dev), int(before.st_ino), int(before.st_size), int(before.st_mtime_ns)):
        raise RuntimeError("GDN library changed while hashing")
    if checksum.hexdigest() != digest or int(after.st_size) != size:
        raise RuntimeError("GDN library digest or size mismatch")
    return {
        "passed": True, "basename": basename, "sha256": digest, "size_bytes": size,
        "mode": 0o600, "identity": identity, "raw_path_emitted": False,
    }


def _assert_same_library(path: Path, prior: dict[str, Any]) -> dict[str, Any]:
    current = _validate_library(path, str(prior["sha256"]), str(prior["basename"]), int(prior["size_bytes"]))
    if tuple(current["identity"]) != tuple(prior["identity"]):
        raise RuntimeError("GDN library identity changed after registration")
    return current


def _validate_profile_argv(
    helper: ModuleType,
    parent_argv: list[str],
    current_argv: list[str],
    parent_cwd: Path,
    profile_path: Path,
) -> dict[str, Any]:
    """Use the sealed structural verifier while accepting the current 400 W cap.

    The separately qualified execute helper predates the user-approved 400 W
    envelope and embeds a 315 W ceiling.  Its structural checks are reused on
    a private argv copy whose sole power value is normalized to that legacy
    ceiling; the real, finite power value is independently parsed, bounded,
    and restored in the public proof.  The supervised child argv is never
    changed.
    """

    delimiters = [index for index, value in enumerate(parent_argv) if value == "--"]
    if len(delimiters) != 1:
        raise RuntimeError("profile supervisor requires exactly one command delimiter")
    delimiter = delimiters[0]
    prefix = parent_argv[:delimiter]
    power = helper._finite_float(
        helper._profile_flag(prefix, "--max-gpu-power-watts"), "power"
    )
    if not 0 < power <= _MAX_GPU_POWER_WATTS:
        raise RuntimeError("profile supervisor GPU power contract is not bounded")

    normalized = list(parent_argv)
    locations: list[tuple[str, int]] = []
    for index, value in enumerate(prefix):
        if value == "--max-gpu-power-watts":
            if index + 1 >= delimiter:
                raise RuntimeError("profile supervisor power flag has no value")
            locations.append(("separate", index + 1))
        elif value.startswith("--max-gpu-power-watts="):
            locations.append(("joined", index))
    if len(locations) != 1:
        raise RuntimeError("profile supervisor must set GPU power exactly once")
    style, index = locations[0]
    normalized[index] = (
        "315.0" if style == "separate" else "--max-gpu-power-watts=315.0"
    )
    proof = helper._validate_profile_argv(
        normalized, current_argv, parent_cwd, profile_path
    )
    values = dict(proof["resource_values"])
    checks = dict(proof["resource_checks"])
    values["power"] = power
    checks["power"] = True
    proof["resource_values"] = values
    proof["resource_checks"] = checks
    proof["maximum_gpu_power_watts"] = _MAX_GPU_POWER_WATTS
    proof["legacy_verifier_power_normalized"] = True
    proof["passed"] = all(checks.values())
    if proof["passed"] is not True:
        raise RuntimeError("profile supervisor resource contract is not bounded")
    return proof


def _stable_operands(block: str) -> list[str]:
    match = re.search(r"\bstablehlo\.custom_call\s+@(?:\"[^\"]+\"|[A-Za-z0-9_.$-]+)\(([^)]*)\)", block, re.DOTALL)
    return re.findall(r"%[A-Za-z0-9_.$#-]+", match.group(1)) if match else []


def _optimized_operands(block: str) -> list[str]:
    match = re.search(r"=\s*\([^)]*\)\s*custom-call\s*\(([^)]*)\)\s*,", block, re.DOTALL)
    if match is None:
        return []
    values = [part.strip() for part in match.group(1).split(",")]
    return values if all(re.fullmatch(r"%[A-Za-z_][A-Za-z0-9_.$-]*", item) for item in values) else []


def _optimized_direct_origin(
    text: str, prepare_name: str, operand: str
) -> tuple[set[int], set[int], bool]:
    """Classify only direct parameters and direct prepare GTEs.

    Arithmetic, copies, conversions, bitcasts, and every other transform are
    intentionally rejected.  Merely walking their dependencies would prove
    ancestry, not value identity (for example ``add(%u, %u)``).
    """

    definitions: dict[str, list[str]] = {}
    for line in text.splitlines():
        match = re.match(
            r"^\s*(?:ROOT\s+)?(?P<name>%[A-Za-z_][A-Za-z0-9_.$-]*)"
            r"\s*=\s*(?P<rhs>.*)$",
            line,
        )
        if match:
            definitions.setdefault(match.group("name"), []).append(match.group("rhs"))
    matches = definitions.get(operand, [])
    if len(matches) != 1:
        raise RuntimeError("optimized HLO dependency has a non-unique definition")
    rhs = matches[0]
    typed_prefix = (
        r"[A-Za-z][A-Za-z0-9_]*\[\s*[0-9,\s]+\s*\]"
        r"\{\s*[0-9,\s]+\s*\}\s+"
    )
    parameter = re.fullmatch(
        typed_prefix + r"parameter\(\s*([0-9]+)\s*\)(?:\s*,.*)?", rhs
    )
    if parameter is not None:
        return set(), {int(parameter.group(1))}, True
    prepare_result = re.fullmatch(
        typed_prefix
        + rf"get-tuple-element\(\s*{re.escape(prepare_name)}\s*\)"
        + r"\s*,\s*index\s*=\s*([0-9]+)(?:\s*,.*)?",
        rhs,
    )
    if prepare_result is not None:
        return {int(prepare_result.group(1))}, set(), True
    return set(), set(), False


def _execute_result_proof(
    text: str,
    dialect: str,
    execute_block: str,
    execute_helper: ModuleType,
) -> dict[str, Any]:
    masked = execute_helper._mask_quoted_strings(text)
    if dialect == "stablehlo":
        assignment = re.search(
            r"^\s*(%[A-Za-z0-9_.$-]+):2\s*=", execute_block, re.MULTILINE
        )
        execute_name = assignment.group(1) if assignment else ""
        returns = list(
            re.finditer(
                r"^\s*(?:(?:stablehlo|func)\.)?return\b(?P<values>[^:\n}]*)",
                masked,
                re.MULTILINE,
            )
        )
        returned = (
            re.findall(r"%[A-Za-z0-9_.$#-]+", returns[0].group("values"))
            if len(returns) == 1
            else []
        )
        expected = [f"{execute_name}#0", f"{execute_name}#1"]
        function_outputs = re.findall(
            r"tensor<[^>]+>",
            re.search(
                r"\bfunc\.func\b.*?\)\s*->\s*\((?P<outputs>.*?)\)\s*\{",
                masked,
                re.DOTALL,
            ).group("outputs"),
        ) if re.search(
            r"\bfunc\.func\b.*?\)\s*->\s*\((?P<outputs>.*?)\)\s*\{",
            masked,
            re.DOTALL,
        ) else []
        checks = {
            "execute_has_two_named_results": bool(execute_name),
            "exactly_one_function_return": len(returns) == 1,
            "execute_results_are_exact_function_return": returned == expected,
            "function_result_types_exact": function_outputs
            == [
                "tensor<1x512x32x128xbf16>",
                "tensor<1x32x128x128xf32>",
            ],
        }
    else:
        assignment = re.search(
            r"^\s*(?P<root>ROOT\s+)?(?P<name>%[A-Za-z_][A-Za-z0-9_.$-]*)"
            r"\s*=",
            execute_block,
            re.MULTILINE,
        )
        execute_name = assignment.group("name") if assignment else ""
        roots = re.findall(
            r"^\s*ROOT\s+(%[A-Za-z_][A-Za-z0-9_.$-]*)\s*=",
            masked,
            re.MULTILINE,
        )
        checks = {
            "execute_result_named": bool(execute_name),
            "execute_custom_call_is_root_instruction": bool(
                assignment and assignment.group("root")
            ),
            "exactly_one_entry_root": len(roots) == 1,
            "execute_result_is_exact_entry_root": roots == [execute_name],
        }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "execute_result_name_sha256": hashlib.sha256(execute_name.encode()).hexdigest()
        if execute_name
        else None,
        "raw_names_emitted": False,
    }


def _dataflow_proof(text: str, dialect: str, prepare_block: str, execute_block: str) -> dict[str, Any]:
    if dialect == "stablehlo":
        assignment = re.search(r"^\s*(%[A-Za-z0-9_.$-]+):3\s*=", prepare_block)
        prepare_name = assignment.group(1) if assignment else ""
        prepare_operands = _stable_operands(prepare_block)
        operands = _stable_operands(execute_block)
        expected = [f"{prepare_name}#{index}" for index in range(3)]
        origins = [
            [position - 2]
            if position in (2, 3, 4) and value == expected[position - 2]
            else []
            for position, value in enumerate(operands)
        ]
        checks = {
            "prepare_tuple_result_named": bool(prepare_name),
            "prepare_consumes_exact_entry_arguments": prepare_operands
            == ["%arg1", "%arg2", "%arg3", "%arg4"],
            "execute_has_six_operands": len(operands) == 6,
            "direct_prepare_results_in_exact_positions": len(operands) == 6 and operands[2:5] == expected,
            "execute_external_operands_are_exact_entry_arguments": len(operands) == 6
            and [operands[0], operands[1], operands[5]]
            == ["%arg0", "%arg1", "%arg5"],
            "other_execute_operands_do_not_reference_prepare": len(operands) == 6
            and all(not item.startswith(prepare_name + "#") for item in (operands[0], operands[1], operands[5])),
        }
    else:
        prep_match = re.search(r"^\s*(?:ROOT\s+)?(?P<name>%[A-Za-z_][A-Za-z0-9_.$-]*)\s*=", prepare_block)
        prepare_name = prep_match.group("name") if prep_match else ""
        prepare_operands = _optimized_operands(prepare_block)
        operands = _optimized_operands(execute_block)
        execute_origins = (
            [_optimized_direct_origin(text, prepare_name, item) for item in operands]
            if prepare_name and len(operands) == 6
            else []
        )
        prepare_origins = (
            [
                _optimized_direct_origin(text, prepare_name, item)
                for item in prepare_operands
            ]
            if prepare_name and len(prepare_operands) == 4
            else []
        )
        execute_prepare = [item[0] for item in execute_origins]
        execute_parameters = [item[1] for item in execute_origins]
        execute_direct = [item[2] for item in execute_origins]
        prepare_parameters = [item[1] for item in prepare_origins]
        prepare_direct = [item[2] for item in prepare_origins]
        checks = {
            "prepare_tuple_result_named": bool(prepare_name),
            "prepare_has_four_operands": len(prepare_operands) == 4,
            "prepare_consumes_exact_entry_parameters": prepare_direct
            == [True, True, True, True]
            and prepare_parameters == [{1}, {2}, {3}, {4}],
            "execute_has_six_operands": len(operands) == 6,
            "prepare_results_reach_exact_execute_positions": execute_direct
            == [True, True, True, True, True, True]
            and len(execute_prepare) == 6
            and execute_prepare[2:5] == [{0}, {1}, {2}],
            "prepared_execute_operands_do_not_depend_on_parameters": len(execute_parameters) == 6
            and execute_parameters[2:5] == [set(), set(), set()],
            "external_execute_operands_are_exact_entry_parameters": execute_direct
            == [True, True, True, True, True, True]
            and len(execute_parameters) == 6
            and [execute_parameters[0], execute_parameters[1], execute_parameters[5]]
            == [{0}, {1}, {5}],
            "other_execute_operands_do_not_depend_on_prepare": len(execute_prepare) == 6
            and execute_prepare[0] == execute_prepare[1] == execute_prepare[5] == set(),
            "no_transformed_prepare_or_execute_operands": prepare_direct
            == [True, True, True, True]
            and execute_direct == [True, True, True, True, True, True],
        }
        origins = [sorted(item) for item in execute_prepare]
    return {
        "passed": all(checks.values()), "checks": checks,
        "prepare_result_name_sha256": hashlib.sha256(prepare_name.encode()).hexdigest() if prepare_name else None,
        "execute_operand_count": len(operands), "prepare_origins_by_execute_operand": origins,
        "raw_names_emitted": False,
    }


def _optimized_execute_abi(block: str, text: str, prepare_helper: ModuleType) -> dict[str, Any]:
    operands = _optimized_operands(block)
    definitions = prepare_helper._optimized_hlo_instruction_definitions(text)
    by_name: dict[str, list[dict[str, Any]]] = {}
    for item in definitions:
        if item["top_level_instruction"]:
            by_name.setdefault(item["name"], []).append(item)
    observed = [
        (
            by_name[name][0].get("dtype"),
            by_name[name][0].get("shape"),
            by_name[name][0].get("layout"),
        )
        if len(by_name.get(name, [])) == 1
        else (None, None, None)
        for name in operands
    ]
    expected = (
        ("f32", _QUERY_SHAPE, (3, 2, 1, 0)), ("f32", _QUERY_SHAPE, (3, 2, 1, 0)),
        ("f32", _VALUE_SHAPE, (3, 2, 1, 0)), ("f32", _VALUE_SHAPE, (3, 2, 1, 0)),
        ("f32", _GATE_SHAPE, (2, 1, 0)), ("f32", _STATE_SHAPE, (3, 2, 1, 0)),
    )
    output_match = re.search(r"=\s*(?P<outputs>\([^)]*\))\s*custom-call", block, re.DOTALL)
    shape = re.compile(
        r"(?P<dtype>[A-Za-z0-9_]+)\[\s*(?P<dims>[0-9,\s]+)\s*\]"
        r"\{\s*(?P<layout>[0-9,\s]+)\s*\}"
    )
    outputs = tuple(
        (
            m.group("dtype"),
            tuple(int(x.strip()) for x in m.group("dims").split(",")),
            tuple(int(x.strip()) for x in m.group("layout").split(",")),
        )
        for m in shape.finditer(output_match.group("outputs") if output_match else "")
    )
    expected_outputs = (("bf16", _VALUE_SHAPE, (3, 2, 1, 0)), ("f32", _STATE_SHAPE, (3, 2, 1, 0)))
    checks = {
        "six_unique_typed_operands_exact": len(operands) == 6
        and len(set(operands)) == 6
        and tuple(observed) == expected,
        "two_typed_results_exact": outputs == expected_outputs,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _composed_ir_summary(text: str, dialect: str, prepare_helper: ModuleType, execute_helper: ModuleType) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("compiler returned empty IR")
    blocks = prepare_helper._custom_call_blocks(text, dialect)
    metadata = prepare_helper._metadata_definitions(text)
    calls = []
    for block in blocks:
        resolved = prepare_helper._resolved_block_metadata(block, metadata)
        targets = sorted(prepare_helper._custom_call_targets(resolved, dialect))
        calls.append((block, resolved, targets))
    prepare_calls = [item for item in calls if item[2] == [_PREPARE_TARGET]]
    execute_calls = [item for item in calls if item[2] == [_EXECUTE_TARGET]]
    prepare_block = prepare_calls[0][0] if len(prepare_calls) == 1 else ""
    execute_block = execute_calls[0][0] if len(execute_calls) == 1 else ""
    prepare_layout = prepare_helper._layout_proof(prepare_block, dialect, text) if prepare_block else {"passed": False}
    if dialect == "stablehlo" and execute_block:
        raw_execute = execute_helper._stablehlo_proof(execute_block)
        execute_checks = dict(raw_execute["checks"])
        execute_checks.pop("direct_entry_arguments_in_order", None)
        execute_operands = _stable_operands(execute_block)
        # The isolated execute helper predates tuple-result composition and
        # tokenizes ``%prepare#0`` as ``%prepare``.  Rebind only its distinct
        # operand check using the composed parser, which retains tuple indices.
        execute_checks["six_distinct_call_operands"] = (
            len(execute_operands) == 6 and len(set(execute_operands)) == 6
        )
        execute_layout = {"passed": all(execute_checks.values()), "checks": execute_checks}
    elif execute_block:
        execute_layout = _optimized_execute_abi(execute_block, text, prepare_helper)
    else:
        execute_layout = {"passed": False}
    dataflow = _dataflow_proof(text, dialect, prepare_block, execute_block) if prepare_block and execute_block else {"passed": False}
    result_proof = (
        _execute_result_proof(text, dialect, execute_block, execute_helper)
        if execute_block
        else {"passed": False}
    )
    masked = execute_helper._mask_quoted_strings(text)
    opcode = r"\bstablehlo\.custom_call\b" if dialect == "stablehlo" else r"\bcustom-call\s*\("
    positions = [match.start() for match in re.finditer(opcode, masked)]
    ownership = []
    for position in positions:
        if dialect == "stablehlo":
            ownership.append(execute_helper._stablehlo_entry_ownership(masked, position))
        else:
            _scope, proof = execute_helper._optimized_entry_scope(text, masked, position)
            ownership.append(proof)
    while_count = len(re.findall(r"\bstablehlo\.while\b" if dialect == "stablehlo" else r"\bwhile\s*\(", masked))
    alias = prepare_helper._nonempty_alias_metadata(text)
    checks = {
        "parser_count_matches_opcode_count": len(blocks) == len(positions),
        "exactly_two_custom_calls": len(blocks) == len(positions) == 2,
        "one_exact_prepare_target": len(prepare_calls) == 1,
        "one_exact_execute_target": len(execute_calls) == 1,
        "no_other_or_lookalike_targets": sorted(item[2] for item in calls) == [[_EXECUTE_TARGET], [_PREPARE_TARGET]],
        "prepare_precedes_execute": bool(prepare_block and execute_block and text.find(prepare_block) < text.find(execute_block)),
        "both_directly_entry_owned": len(ownership) == 2 and all(item.get("passed") is True for item in ownership),
        "prepare_abi_layout_exact": prepare_layout.get("passed") is True,
        "execute_abi_layout_exact": execute_layout.get("passed") is True,
        "prepare_execute_dataflow_exact": dataflow.get("passed") is True,
        "execute_results_are_exact_entry_outputs": result_proof.get("passed") is True,
        "no_while": while_count == 0,
        "no_alias_metadata": alias == [],
    }
    return {
        "dialect": dialect, "passed": all(checks.values()), "checks": checks,
        "custom_call_count": len(blocks), "while_count": while_count,
        "prepare_layout": prepare_layout, "execute_layout": execute_layout,
        "dataflow": dataflow, "result_proof": result_proof,
        "nonempty_alias_metadata": alias,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_utf8_bytes": len(text.encode()), "raw_ir_emitted": False,
    }


def _compiled_memory(compiled: Any, execute_helper: ModuleType) -> dict[str, Any]:
    return execute_helper._compiled_memory(compiled)


def _memory_gate(memory: dict[str, Any]) -> dict[str, Any]:
    argument = memory.get("argument_size_in_bytes")
    output = memory.get("output_size_in_bytes")
    alias = memory.get("alias_size_in_bytes")
    temporary = memory.get("temp_size_in_bytes")
    combined = argument + output + temporary if all(type(item) is int for item in (argument, output, temporary)) else None
    checks = {
        "argument_bytes_exact": argument == _ARGUMENT_BYTES,
        "output_bytes_exact": output == _COMPILER_OUTPUT_BYTES,
        "alias_bytes_zero": alias == 0,
        "intermediate_is_compiler_visible": type(temporary) is int and temporary >= _INTERMEDIATE_BYTES,
        "temporary_bytes_bounded": type(temporary) is int and temporary <= _MAX_TEMP_BYTES,
        "combined_bytes_bounded": type(combined) is int and combined <= _MAX_COMBINED_BYTES,
    }
    return {"passed": all(checks.values()), "checks": checks, "combined_bytes": combined}


def _construct_host_reference(np: Any, execute_helper: ModuleType, prepare_oracle: Callable[..., Any], execute_oracle: Callable[..., Any], counters: dict[str, int]) -> tuple[tuple[Any, ...], tuple[Any, Any], dict[str, Any]]:
    query = execute_helper._normalize_rows(np, execute_helper._splitmix64_f32(np, _QUERY_SHAPE, 0x243F6A8885A308D3), 1.0 / math.sqrt(128))
    key = execute_helper._normalize_rows(np, execute_helper._splitmix64_f32(np, _QUERY_SHAPE, 0x13198A2E03707344), 1.0)
    value = np.ascontiguousarray(execute_helper._splitmix64_f32(np, _VALUE_SHAPE, 0xA4093822299F31D0) * np.float32(0.025), dtype=np.float32)
    token = np.arange(512, dtype=np.int32)[:, None]
    head = np.arange(32, dtype=np.int32)[None, :]
    chunk = token // 64
    g = np.ascontiguousarray(-(np.float32(0.0008) + np.float32(0.0001) * (token % 7) + np.float32(0.00005) * (head % 5) + np.float32(0.000025) * ((chunk + head) % 4))[None, ...], dtype=np.float32)
    beta = np.ascontiguousarray((np.float32(0.008) + np.float32(0.042) * (((3 * token + 5 * head + 7 * chunk) % 17).astype(np.float32) / np.float32(16)))[None, ...], dtype=np.float32)
    state = np.ascontiguousarray(execute_helper._splitmix64_f32(np, _STATE_SHAPE, 0x082EFA98EC4E6C89) * np.float32(0.003), dtype=np.float32)
    primals = (query, key, value, g, beta, state)
    pristine_input_hashes = {
        name: execute_helper._array_sha256(item)
        for name, item in zip(_INPUT_NAMES, primals, strict=True)
    }
    counters["prepare_oracle_attempts"] += 1
    prepared = prepare_oracle(key, value, g, beta)
    counters["prepare_oracle_completions"] += 1
    counters["execute_oracle_attempts"] += 1
    reference = execute_oracle(query, key, *prepared, state, output_bfloat16=True)
    counters["execute_oracle_completions"] += 1
    input_hashes = {name: execute_helper._array_sha256(item) for name, item in zip(_INPUT_NAMES, primals, strict=True)}
    prepared_hashes = {name: execute_helper._array_sha256(item) for name, item in zip(("prepared_u", "prepared_w", "gamma"), prepared, strict=True)}
    reference_hashes = {name: execute_helper._array_sha256(item) for name, item in zip(_OUTPUT_NAMES, reference, strict=True)}
    checks = {
        "six_disjoint_fp32_c_inputs": len(primals) == 6
        and all(item.dtype == np.dtype(np.float32) and item.flags.c_contiguous for item in primals)
        and all(not np.shares_memory(primals[i], primals[j]) for i in range(6) for j in range(i + 1, 6)),
        "input_shapes_exact": tuple(item.shape for item in primals) == _INPUT_SHAPES,
        "input_bytes_exact": sum(item.nbytes for item in primals) == _ARGUMENT_BYTES,
        "oracles_did_not_mutate_inputs": input_hashes == pristine_input_hashes,
        "input_hashes_exact": input_hashes == _EXPECTED_INPUT_SHA256,
        "input_tuple_hash_exact": execute_helper._framed_tuple_sha256(_INPUT_DOMAIN, tuple(zip(_INPUT_NAMES, primals, strict=True))) == _EXPECTED_INPUT_TUPLE_SHA256,
        "prepared_shapes_dtypes_and_bytes_exact": tuple(item.shape for item in prepared)
        == (_VALUE_SHAPE, _VALUE_SHAPE, _GATE_SHAPE)
        and all(item.dtype == np.dtype(np.float32) and item.flags.c_contiguous for item in prepared)
        and sum(item.nbytes for item in prepared) == _INTERMEDIATE_BYTES,
        "prepared_hashes_exact": prepared_hashes == _EXPECTED_PREPARED_SHA256,
        "reference_shapes_dtypes_and_bytes_exact": tuple(item.shape for item in reference)
        == (_VALUE_SHAPE, _STATE_SHAPE)
        and str(reference[0].dtype) == "bfloat16"
        and reference[1].dtype == np.dtype(np.float32)
        and all(item.flags.c_contiguous for item in reference)
        and sum(item.nbytes for item in reference) == _LOGICAL_OUTPUT_BYTES,
        "reference_hashes_exact": reference_hashes == _EXPECTED_REFERENCE_SHA256,
        "reference_tuple_hash_exact": execute_helper._framed_tuple_sha256(_REFERENCE_DOMAIN, tuple(zip(_OUTPUT_NAMES, reference, strict=True))) == _EXPECTED_REFERENCE_TUPLE_SHA256,
        "all_finite": all(np.all(np.isfinite(item)) for item in (*primals, *prepared, *reference)),
    }
    if not all(checks.values()):
        raise RuntimeError("composed host oracle identity gate failed")
    return primals, reference, {"passed": True, "checks": checks, "input_sha256": input_hashes, "prepared_sha256": prepared_hashes, "reference_sha256": reference_hashes, "raw_tensors_emitted": False}


def _assert_host_primals_unchanged(
    primals: tuple[Any, ...],
    host_report: dict[str, Any],
    execute_helper: ModuleType,
) -> dict[str, Any]:
    if len(primals) != len(_INPUT_NAMES):
        raise RuntimeError("composed host input boundary no longer has six leaves")
    observed = {
        name: execute_helper._array_sha256(item)
        for name, item in zip(_INPUT_NAMES, primals, strict=True)
    }
    passed = (
        observed == host_report.get("input_sha256") == _EXPECTED_INPUT_SHA256
    )
    if not passed:
        raise RuntimeError(
            "host input boundary changed during composed device qualification"
        )
    return {
        "passed": True,
        "six_host_primals_rehashed_after_validation": True,
        "individual_hashes_match_host_oracle_and_exact_contract": True,
        "raw_tensors_emitted": False,
    }


class _CheckedExecutable:
    __slots__ = ("_compiled", "_consumed", "_counters", "proof")

    def __init__(self, compiled: Any, proof: dict[str, Any], counters: dict[str, int], token: object) -> None:
        if token is not _CHECKED_TOKEN or proof.get("passed") is not True:
            raise RuntimeError("refusing unchecked composed executable")
        self._compiled, self._consumed, self._counters, self.proof = compiled, False, counters, proof

    def invoke(self, jax: Any, values: tuple[Any, ...]) -> tuple[Any, Any]:
        if self._consumed or len(values) != 6:
            raise RuntimeError("composed executable capability is consumed or malformed")
        self._consumed = True
        self._counters["checked_executable_attempts"] += 1
        result = jax.block_until_ready(self._compiled(*values))
        self._counters["output_readiness_barriers"] += 1
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("composed executable did not return two leaves")
        self._counters["checked_executable_completions"] += 1
        return result


def _journal(require_clean_boot: Callable[[], dict[str, Any]], boot: Any, helper: ModuleType, output: TextIO, stage: str, counters: dict[str, int]) -> None:
    if stage not in _JOURNAL_STAGES:
        raise RuntimeError("refusing undeclared composed GDN journal stage")
    _emit({"record_type": "journal_checkpoint", "timestamp": _utc_now(), "stage": stage, "safety": helper._public_clean_safety(require_clean_boot(), stage), "boot": boot.check(), "counters": dict(counters)}, output)


def _compile_unreleased(jax: Any, jnp: Any, prepare_call: Callable[..., Any], execute_call: Callable[..., Any], register_prepare: Callable[..., Any], register_execute: Callable[..., Any], prepare_path: Path, prepare_sha: str, execute_path: Path, execute_sha: str, prepare_manifest: dict[str, Any], execute_manifest: dict[str, Any], prepare_helper: ModuleType, execute_helper: ModuleType, require_clean_boot: Callable[[], dict[str, Any]], boot: Any, counters: dict[str, int], output: TextIO) -> tuple[Any, dict[str, Any]]:
    counters["prepare_registration_attempts"] += 1
    try:
        registered_prepare = register_prepare(prepare_path, library_sha256=prepare_sha, enabled=True)
        prepare_seal = prepare_helper._sealed_registration_manifest(registered_prepare, library_path=prepare_path, library_sha256=prepare_sha, library_size_bytes=int(prepare_manifest["size_bytes"]))
        counters["prepare_registration_completions"] += 1
    finally:
        _journal(require_clean_boot, boot, execute_helper, output, "after_prepare_registration_attempt", counters)
    counters["execute_registration_attempts"] += 1
    try:
        registered_execute = register_execute(execute_path, library_sha256=execute_sha, enabled=True)
        execute_seal = execute_helper._sealed_registration_manifest(registered_execute, library_path=execute_path, library_sha256=execute_sha, library_size_bytes=int(execute_manifest["size_bytes"]))
        counters["execute_registration_completions"] += 1
    finally:
        _journal(require_clean_boot, boot, execute_helper, output, "after_execute_registration_attempt", counters)
    _emit({"record_type": "ffi_registered", "timestamp": _utc_now(), "prepare": prepare_seal, "execute": execute_seal, "counters": dict(counters)}, output)
    signatures = tuple(jax.ShapeDtypeStruct(shape, jnp.float32) for shape in _INPUT_SHAPES)
    counters["shape_dtype_structs"] += 6

    def composed(query: Any, key: Any, value: Any, g: Any, beta: Any, state: Any) -> tuple[Any, Any]:
        counters["prepare_trace_calls"] += 1
        prepared_u, prepared_w, gamma = prepare_call(key, value, g, beta, enabled=True, library_path=prepare_path, library_sha256=prepare_sha)
        counters["execute_trace_calls"] += 1
        return execute_call(query, key, prepared_u, prepared_w, gamma, state, enabled=True, library_path=execute_path, library_sha256=execute_sha)

    counters["lower_attempts"] += 1
    lowered = None
    try:
        started = time.perf_counter()
        lowered = jax.jit(composed).lower(*signatures)
        lower_seconds = time.perf_counter() - started
        counters["lower_completions"] += 1
        if counters["prepare_trace_calls"] != 1 or counters["execute_trace_calls"] != 1:
            raise RuntimeError("lowering did not trace each FFI call exactly once")
        stable = _composed_ir_summary(str(lowered.compiler_ir(dialect="stablehlo")), "stablehlo", prepare_helper, execute_helper)
        if not stable["passed"] or not math.isfinite(lower_seconds) or not 0 <= lower_seconds < _LOWER_HARD_SECONDS:
            raise RuntimeError("composed StableHLO or lower-duration gate failed")
        _emit({"record_type": "lowered", "timestamp": _utc_now(), "seconds": lower_seconds, "stablehlo": stable, "counters": dict(counters)}, output)
    finally:
        _journal(require_clean_boot, boot, execute_helper, output, "after_lower_attempt", counters)
    counters["compile_attempts"] += 1
    compiled = None
    released = False
    try:
        started = time.perf_counter()
        compiled = lowered.compile()
        compile_seconds = time.perf_counter() - started
        counters["compile_completions"] += 1
        optimized = _composed_ir_summary(compiled.as_text(), "optimized_hlo", prepare_helper, execute_helper)
        memory = _compiled_memory(compiled, execute_helper)
        memory_gate = _memory_gate(memory)
        gate = {
            "stablehlo": stable["passed"], "optimized_hlo": optimized["passed"],
            "memory": memory_gate["passed"], "duration": math.isfinite(compile_seconds) and 0 <= compile_seconds < _COMPILE_HARD_SECONDS,
        }
        gate["passed"] = all(gate.values())
        report = {"record_type": "ffi_compiled_unreleased", "timestamp": _utc_now(), "compile_seconds": compile_seconds, "stablehlo": stable, "optimized_hlo": optimized, "compiled_memory": memory, "memory_gate": memory_gate, "release_gate": gate, "counters": dict(counters)}
        _emit(report, output)
        if not gate["passed"]:
            raise RuntimeError("composed compile release gate failed")
        released = True
        return compiled, report
    finally:
        if compiled is not None and not released:
            del compiled
        if lowered is not None:
            del lowered
        _journal(require_clean_boot, boot, execute_helper, output, "after_compile_attempt", counters)


def _run(args: argparse.Namespace, output: TextIO, execute_helper: ModuleType, prepare_helper: ModuleType, prepare_manifest: dict[str, Any], execute_manifest: dict[str, Any], require_clean_boot: Callable[[], dict[str, Any]], boot: Any, counters: dict[str, int]) -> int:
    primals = reference = host_report = None
    if not args.compile_diagnostic:
        _journal(require_clean_boot, boot, execute_helper, output, "before_host_oracle", counters)
        try:
            prepare_oracle = _load_exact_module(_source_files()["prepare_oracle"], _EXPECTED_SOURCE_SHA256["prepare_oracle"], "_skyrl_composed_prepare_oracle")
            execute_oracle = _load_exact_module(_source_files()["execute_oracle"], _EXPECTED_SOURCE_SHA256["execute_oracle"], "_skyrl_composed_execute_oracle")
            import numpy as np

            primals, reference, host_report = _construct_host_reference(np, execute_helper, prepare_oracle.gdn_prepare_s512_numpy, execute_oracle.gdn_execute_s512_numpy, counters)
            _emit({"record_type": "host_oracle", "timestamp": _utc_now(), "report": host_report, "device_transfer_started": False, "counters": dict(counters)}, output)
        finally:
            _journal(require_clean_boot, boot, execute_helper, output, "after_host_oracle_attempt", counters)
    _journal(require_clean_boot, boot, execute_helper, output, "before_backend_initialization", counters)
    counters["backend_initialization_attempts"] += 1
    try:
        prepare_wrapper, execute_wrapper, loaded_sources = _import_bound_wrappers()
        import jax
        import jax.numpy as jnp
        import jaxlib
        from jax.extend import backend as jax_backend

        gdn_prepare_s512 = prepare_wrapper.gdn_prepare_s512
        register_gdn_prepare_s512 = prepare_wrapper.register_gdn_prepare_s512
        gdn_execute_s512 = execute_wrapper.gdn_execute_s512
        register_gdn_execute_s512 = execute_wrapper.register_gdn_execute_s512

        backend = execute_helper._backend_manifest(jax, jaxlib, jax_backend)
        counters["backend_initialization_completions"] += 1
    finally:
        _journal(require_clean_boot, boot, execute_helper, output, "after_backend_initialization_attempt", counters)
    _emit({"record_type": "backend_ready", "timestamp": _utc_now(), "backend": backend, "loaded_sources": loaded_sources, "counters": dict(counters)}, output)
    compiled = None
    try:
        compiled, compile_report = _compile_unreleased(jax, jnp, gdn_prepare_s512, gdn_execute_s512, register_gdn_prepare_s512, register_gdn_execute_s512, args.prepare_library, args.prepare_library_sha256, args.execute_library, args.execute_library_sha256, prepare_manifest, execute_manifest, prepare_helper, execute_helper, require_clean_boot, boot, counters, output)
        if args.compile_diagnostic:
            del compiled
            compiled = None
            terminal_counters = _assert_terminal_counters(
                counters, compile_diagnostic=True
            )
            _emit({"record_type": "compile_diagnostic_passed", "timestamp": _utc_now(), "status": "passed_composed_s512_compile_only", "terminal_counter_gate": terminal_counters, "counters": dict(counters)}, output)
            return 0
        if primals is None or reference is None or host_report is None:
            raise RuntimeError("host oracle did not complete")
        counters["capability_release_attempts"] += 1
        proof = {"compile_release_gate": compile_report["release_gate"]["passed"], "host_oracle_gate": host_report["passed"]}
        proof["passed"] = all(proof.values())
        executable = _CheckedExecutable(compiled, proof, counters, _CHECKED_TOKEN)
        counters["capability_release_completions"] += 1
        counters["tuple_device_put_attempts"] += 1
        try:
            device_inputs = jax.block_until_ready(jax.device_put(primals))
            counters["input_readiness_barriers"] += 1
            if not isinstance(device_inputs, tuple) or len(device_inputs) != 6:
                raise RuntimeError("device_put did not preserve six leaves")
            counters["tuple_device_put_completions"] += 1
            counters["device_put_leaves"] += 6
        finally:
            _journal(require_clean_boot, boot, execute_helper, output, "after_input_device_put_attempt", counters)
        started = time.perf_counter()
        try:
            device_outputs = executable.invoke(jax, device_inputs)
        finally:
            seconds = time.perf_counter() - started
            _journal(require_clean_boot, boot, execute_helper, output, "after_candidate_dispatch_attempt", counters)
        counters["tuple_device_get_attempts"] += 1
        try:
            actual = jax.device_get(device_outputs)
            if not isinstance(actual, tuple) or len(actual) != 2:
                raise RuntimeError("device_get did not preserve two leaves")
            counters["tuple_device_get_completions"] += 1
            counters["device_get_leaves"] += 2
        finally:
            _journal(require_clean_boot, boot, execute_helper, output, "after_candidate_device_get_attempt", counters)
        try:
            validation = execute_helper._validate_actual(__import__("numpy"), reference, actual, seconds)
            host_immutability = _assert_host_primals_unchanged(
                primals, host_report, execute_helper
            )
            validation["host_inputs_unchanged_after_device_run"] = True
            validation["host_input_immutability_proof"] = host_immutability
            _emit({"record_type": "host_validation", "timestamp": _utc_now(), "validation": validation, "counters": dict(counters)}, output)
        finally:
            _journal(require_clean_boot, boot, execute_helper, output, "after_host_validation_attempt", counters)
        terminal_counters = _assert_terminal_counters(
            counters, compile_diagnostic=False
        )
        _emit({"record_type": "runtime_passed", "timestamp": _utc_now(), "status": "passed_composed_s512_promotable" if validation["promotion_passed"] else "passed_composed_s512_unpromotable", "validation": validation, "terminal_counter_gate": terminal_counters, "vjp_validated": False, "model_validated": False, "counters": dict(counters)}, output)
        return 0
    finally:
        if compiled is not None:
            del compiled


def _execute(args: argparse.Namespace, output: TextIO) -> int:
    counters = _zero_counters()
    _emit({"record_type": "manifest", "timestamp": _utc_now(), "platform_requested": args.platform, "allow_gpu": args.allow_gpu, "compile_diagnostic": bool(args.compile_diagnostic), "case": args.case, "contract": _exact_contract(), "fresh_process_required": True, "raw_paths_emitted": False, "raw_ir_emitted": False, "raw_tensors_emitted": False, "counters": dict(counters), **_source_hashes()}, output)
    if args.platform == "abstract":
        _emit({"record_type": "refused", "timestamp": _utc_now(), "status": "no_gpu_abstract_manifest_only", "jax_imported": False, "numpy_imported": False, "skyrl_rocm_package_imported": False, "shared_libraries_loaded": False, "counters": dict(counters)}, output)
        return 0
    stage = "fresh_process"
    try:
        _assert_fresh_process()
        stage = "bound_sources"
        bound = _assert_bound_sources()
        execute_helper, prepare_helper = _load_helpers()
        stage = "profile_supervision"
        parent_pid = os.getppid()
        parent_argv = execute_helper._read_proc_argv(parent_pid)
        parent_cwd = Path(f"/proc/{parent_pid}/cwd").resolve(strict=True)
        profile = _validate_profile_argv(
            execute_helper,
            parent_argv,
            [sys.executable, *sys.argv],
            parent_cwd,
            _source_files()["profile_rocm"].resolve(),
        )
        if _file_sha256(_source_files()["profile_rocm"]) != _EXPECTED_SOURCE_SHA256["profile_rocm"]:
            raise RuntimeError("profile_rocm source mismatch")
        profile["profile_source_exact"] = True
        stage = "library_preflight"
        prepare_manifest = _validate_library(args.prepare_library, args.prepare_library_sha256, _PREPARE_BASENAME, _PREPARE_SIZE)
        execute_manifest = _validate_library(args.execute_library, args.execute_library_sha256, _EXECUTE_BASENAME, _EXECUTE_SIZE)
        _emit({"record_type": "prerequisite_proof", "timestamp": _utc_now(), "bound_sources": bound, "profile_supervision": profile, "prepare_library": {k: v for k, v in prepare_manifest.items() if k != "identity"}, "execute_library": {k: v for k, v in execute_manifest.items() if k != "identity"}, "counters": dict(counters)}, output)
        stage = "environment"
        if not args.compile_diagnostic:
            execute_helper._validate_host_numeric_environment()
        environment = execute_helper._configure_rocm_environment()
        execute_helper._prove_command_buffers_disabled(environment)
        boot = execute_helper._BootSeal()
        guarded_process, require_clean_boot = execute_helper._load_safety_helpers()
        stage = "safety_preflight"
        with guarded_process() as raw_safety:
            _emit({"record_type": "safety_preflight", "timestamp": _utc_now(), "safety": execute_helper._public_safety_preflight(raw_safety), "boot": boot.check(), "hardware_stack": execute_helper._hardware_stack_preflight(), "counters": dict(counters)}, output)
            try:
                result = _run(args, output, execute_helper, prepare_helper, prepare_manifest, execute_manifest, require_clean_boot, boot, counters)
            finally:
                try:
                    stage = "safety_postflight"
                    _emit({"record_type": "safety_postflight", "timestamp": _utc_now(), "safety": execute_helper._public_clean_safety(require_clean_boot(), "safety_postflight"), "boot": boot.check(), "counters": dict(counters)}, output)
                finally:
                    stage = "library_postcheck"
                    postcheck_errors: list[BaseException] = []
                    try:
                        _assert_same_library(args.prepare_library, prepare_manifest)
                    except BaseException as error:
                        postcheck_errors.append(error)
                    try:
                        _assert_same_library(args.execute_library, execute_manifest)
                    except BaseException as error:
                        postcheck_errors.append(error)
                    try:
                        _journal(require_clean_boot, boot, execute_helper, output, "after_library_postcheck", counters)
                    finally:
                        if postcheck_errors:
                            raise RuntimeError("one or more GDN library postchecks failed") from postcheck_errors[0]
        _emit({"record_type": "completed", "timestamp": _utc_now(), "status": "passed", "mode": "compile_diagnostic" if args.compile_diagnostic else "runtime", "counters": dict(counters)}, output)
        return result
    except Exception as error:
        encoded = str(error).encode("utf-8", errors="replace")
        _emit({"record_type": "error", "timestamp": _utc_now(), "stage": stage, "status": "failed_closed", "error_type": type(error).__name__, "message_redacted": True, "message_utf8_bytes": len(encoded), "message_sha256": hashlib.sha256(encoded).hexdigest(), "counters": dict(counters)}, output)
        return 1


def _open_output(path: Path) -> TextIO:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise RuntimeError("composed probe output is not private mode 0600")
        return os.fdopen(descriptor, "w", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.output is None:
        return _execute(args, sys.stdout)
    with _open_output(args.output) as output:
        return _execute(args, output)


if __name__ == "__main__":
    sys.exit(main())
