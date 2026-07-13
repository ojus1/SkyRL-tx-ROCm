#!/usr/bin/env python3
"""Guarded exact-T512 full arbitrary-cotangent GQA VJP gate.

The default ``abstract`` mode emits a refusal manifest without importing JAX.
ROCm requires ``--platform rocm --allow-gpu --output`` in a fresh process under
``profile_rocm.py``.  The exact BF16 B1/T512/Hq16/Hkv4/D256 forward and custom
VJP are lowered and compiled once with explicit scale 3/32.  A private
capability is released only after StableHLO and optimized HLO independently
prove exactly one direct entry-level q0 forward, q0 dQ, and q0 dK/dV logical
ROCm Triton call, canonical query metadata, no extra custom call, callable
container, or outer while, and exact compiled-memory bounds.  The capability
is consumed by one invocation only.

Q/K/V and the output cotangent are independent deterministic nonzero BF16
PCG64 grids.  Validation uses an independent host NumPy FP32 causal-GQA
forward/backward oracle.  It streams over 32-query/32-key tiles and recomputes
probabilities, never retaining a complete T-by-T matrix.  Output, dQ, dK, and
dV are gated independently.  There is no warmup, replay, second invocation,
GPU reference/reduction, model call, padding case, or backward-of-backward.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, ContextManager, TextIO

_BATCH_SIZE = 1
_SEQUENCE_LENGTH = 512
_QUERY_HEADS = 16
_KV_HEADS = 4
_HEAD_DIM = 256
_GROUP_SIZE = _QUERY_HEADS // _KV_HEADS
_QUERY_CHUNK_SIZE = 512
_BLOCK_Q = 64
_BLOCK_K = 64
_BACKWARD_BLOCK_Q = 32
_BACKWARD_BLOCK_K = 32
_ORACLE_QUERY_TILE = 32
_ORACLE_KEY_TILE = 32
_ATTENTION_SCALE = 3 / 32
_INPUT_SEED = 20260713
_COTANGENT_SEED = 20260714
_GRID_DENOMINATOR = 128
_QKV_MAXIMUM_INTEGER_MAGNITUDE = 96
_COTANGENT_MAXIMUM_INTEGER_MAGNITUDE = 48
_EXPECTED_ARGUMENT_BYTES = 10_487_808
_EXPECTED_OUTPUT_BYTES = 10_485_792
_EXPECTED_ALIAS_BYTES = 0
_MAX_TEMP_BYTES = 64 * 1024**2
_MAX_RELATIVE_L2 = 0.01
_MIN_COSINE = 0.9999
_MAX_ABSOLUTE_ERROR = 0.02
_MIN_REFERENCE_NORM = 1e-8
_MAX_CANDIDATE_SECONDS = 0.1
_MAX_PROMOTION_CANDIDATE_SECONDS = 0.075
_EXACT_TARGET = "__gpu$xla.gpu.triton"
_EXPECTED_MARKERS = {
    "forward": "query_bounded_gqa_forward_q0",
    "dq": "query_bounded_gqa_dq_q0",
    "dkdv": "query_bounded_gqa_dkdv_q0",
}
_EXPECTED_COMPILE_PROBE_SOURCE_SHA256 = (
    "bf01187101d20362072c96f70fbb80b2f8eed88fa55e60cbf479abba6db012a2"
)
_EXPECTED_NONZERO_PROBE_SOURCE_SHA256 = (
    "999e027d4cc35a8d59cc294020f8865036f8fb817a847ac38f96e36b597f74ac"
)
_EXPECTED_KERNEL_SOURCE_SHA256 = (
    "51e2fd91eb270f7b25ecdd117d7f06aa48a8e4af282a5a7e5e6b4c2a25dc52c9"
)
_EXPECTED_INPUT_SHA256 = {
    "q": "ff889e094bfb7ce5e55446f1fff3956a747eed7986e9c30d170f7c4433b9bda7",
    "k": "e5d2258530fcb373b981167ba116b43306bb424f565bbf2da7f5c55810369978",
    "v": "26f13a090d34805fb6fc1ea002c65e7d7cdb9952b8a582cd132246fe54420699",
    "key_mask": "6323b30c3d5f9b893f1133983aa3761cef653959de5a6e4f8e798c358bd226e1",
    "dout": "23b69593d1dde3c5a9bea2d5679fe78891e5d96d083e31b5fc4fa1460d3c0622",
}
_EXPECTED_REFERENCE_SHA256 = {
    "output": "273c1e5694bc21b03798ccd73b8c8be669218d41c82e1b23236c2e3a4bea506f",
    "dq": "936167f7fcc92a5bc8251ca4c332f80fae74ab36c98ca552d68e16da651702d5",
    "dk": "04e64911be655c3f1bacf2d3d091be301df06d2a98d534267e4ebde3b74d24e6",
    "dv": "9c4f5d999e7548e769dad32950b9177a939b114f06436b75ab4acb38c9e73e83",
}
_EXPECTED_REFERENCE_NORMS = {
    "output": 75.4267547716931,
    "dq": 9.212431174660864,
    "dk": 9.257514343887854,
    "dv": 38.029129972931884,
}
_EXPECTED_ORACLE_SCRATCH_BYTES = 323_072
_EXPECTED_MAXIMUM_ABSOLUTE_VALID_LOGIT = 1.4235591888427734
_EXPECTED_AMD_PCI_DEVICE_ID = "0x744c"
_EXPECTED_GPU_ARCHITECTURE = "gfx1100"
_IR_NAME_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.$-]*")
_CHECKED_CAPABILITY_TOKEN = object()
_JOURNAL_STAGES = frozenset(
    {
        "after_backend_initialization_attempt",
        "after_vjp_lower_attempt",
        "after_vjp_compile_attempt",
        "after_host_reference_construction",
        "after_explicit_input_device_put_attempt",
        "after_candidate_dispatch_attempt",
        "after_candidate_device_get_attempt",
        "after_host_validation",
    }
)
_COMPILE_GPU_WORK_CAVEAT = (
    "lowered.compile may dispatch bounded GPU autotuning/profiling work; the "
    "compiled VJP executable remains inaccessible until exact structural and "
    "compiled-memory gates pass"
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _emit(record: dict[str, Any], output: TextIO) -> None:
    output.write(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    output.flush()


def _redacted_message_summary(error: BaseException) -> dict[str, Any]:
    encoded = str(error).encode("utf-8", errors="replace")
    return {
        "message_redacted": True,
        "message_utf8_bytes": len(encoded),
        "message_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _json_safe_duration(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _source_files() -> dict[str, Path]:
    repo = _repo_root()
    return {
        "probe_source_sha256": Path(__file__),
        "delegated_compile_probe_source_sha256": repo
        / "rocm"
        / "probe_query_bounded_gqa_compile.py",
        "delegated_nonzero_probe_source_sha256": repo
        / "rocm"
        / "probe_query_bounded_gqa_nonzero_scale.py",
        "query_bounded_gqa_kernel_source_sha256": repo
        / "skyrl"
        / "tx"
        / "kernels"
        / "query_bounded_gqa.py",
        "delegated_safety_helper_source_sha256": repo / "rocm" / "amdgpu_safety.py",
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {name: _file_sha256(path) for name, path in _source_files().items()}


def _compile_probe() -> Any:
    try:
        from rocm import probe_query_bounded_gqa_compile
    except ModuleNotFoundError:
        import probe_query_bounded_gqa_compile  # type: ignore[no-redef]

    return probe_query_bounded_gqa_compile


def _nonzero_probe() -> Any:
    try:
        from rocm import probe_query_bounded_gqa_nonzero_scale
    except ModuleNotFoundError:
        import probe_query_bounded_gqa_nonzero_scale  # type: ignore[no-redef]

    return probe_query_bounded_gqa_nonzero_scale


def _assert_static_source_bindings() -> dict[str, Any]:
    files = _source_files()
    compile_probe = _compile_probe()
    nonzero = _nonzero_probe()
    expected_compile = files["delegated_compile_probe_source_sha256"].resolve()
    expected_nonzero = files["delegated_nonzero_probe_source_sha256"].resolve()
    if (
        not isinstance(getattr(compile_probe, "__file__", None), str)
        or Path(compile_probe.__file__).resolve() != expected_compile
    ):
        raise RuntimeError(
            "compile helper did not resolve to the exact repository source"
        )
    if (
        not isinstance(getattr(nonzero, "__file__", None), str)
        or Path(nonzero.__file__).resolve() != expected_nonzero
    ):
        raise RuntimeError(
            "nonzero helper did not resolve to the exact repository source"
        )
    hashes = {
        "delegated_compile_probe_source_sha256": _file_sha256(expected_compile),
        "delegated_nonzero_probe_source_sha256": _file_sha256(expected_nonzero),
        "query_bounded_gqa_kernel_source_sha256": _file_sha256(
            files["query_bounded_gqa_kernel_source_sha256"].resolve()
        ),
    }
    expected = {
        "delegated_compile_probe_source_sha256": _EXPECTED_COMPILE_PROBE_SOURCE_SHA256,
        "delegated_nonzero_probe_source_sha256": _EXPECTED_NONZERO_PROBE_SOURCE_SHA256,
        "query_bounded_gqa_kernel_source_sha256": _EXPECTED_KERNEL_SOURCE_SHA256,
    }
    if hashes != expected:
        raise RuntimeError("VJP delegated source SHA256 pin changed")
    delegated = nonzero._assert_static_source_bindings()
    if delegated.get("passed") is not True:
        raise RuntimeError("delegated promoted source binding did not pass")
    return {
        "passed": True,
        "compile_helper_resolved_file_matches_expected": True,
        "nonzero_helper_resolved_file_matches_expected": True,
        "delegated_source_binding": delegated,
        **hashes,
    }


def _assert_kernel_binding(api: Any) -> dict[str, Any]:
    expected_file = _source_files()["query_bounded_gqa_kernel_source_sha256"].resolve()
    loaded_file = inspect.getsourcefile(api)
    if loaded_file is None or Path(loaded_file).resolve() != expected_file:
        raise RuntimeError("VJP API did not bind to the exact repository kernel source")
    if _file_sha256(expected_file) != _EXPECTED_KERNEL_SOURCE_SHA256:
        raise RuntimeError("loaded VJP kernel source SHA256 changed")
    signature = inspect.signature(api)
    parameters = signature.parameters
    exact = (
        list(parameters)[:4] == ["q", "k", "v", "key_mask"]
        and parameters["scale"].default is None
        and parameters["query_chunk_size"].default == _QUERY_CHUNK_SIZE
        and parameters["block_q"].default == _BLOCK_Q
        and parameters["block_k"].default == _BLOCK_K
        and parameters["backward_block_q"].default == _BACKWARD_BLOCK_Q
        and parameters["backward_block_k"].default == _BACKWARD_BLOCK_K
        and parameters["interpret"].default is False
    )
    if not exact:
        raise RuntimeError("VJP API signature or defaults changed")
    return {
        "passed": True,
        "resolved_file_matches_expected": True,
        "source_sha256": _EXPECTED_KERNEL_SOURCE_SHA256,
        "signature_semantics_exact": True,
    }


def _assert_gfx1100_drm(
    drm_root: Path = Path("/sys/class/drm"),
) -> dict[str, Any]:
    """Bind the sole visible AMD card to this RX 7900 XTX PCI identity."""
    amd_cards: list[tuple[str, str]] = []
    for card in sorted(drm_root.glob("card[0-9]*")):
        try:
            vendor = (card / "device" / "vendor").read_text().strip().lower()
            device = (card / "device" / "device").read_text().strip().lower()
        except OSError:
            continue
        if vendor == "0x1002":
            amd_cards.append((card.name, device))
    if len(amd_cards) != 1 or amd_cards[0][1] != _EXPECTED_AMD_PCI_DEVICE_ID:
        raise RuntimeError(
            "VJP gate requires the sole AMD DRM card to be the pinned gfx1100 device"
        )
    return {
        "passed": True,
        "architecture": _EXPECTED_GPU_ARCHITECTURE,
        "amd_pci_vendor_id": "0x1002",
        "amd_pci_device_id": _EXPECTED_AMD_PCI_DEVICE_ID,
        "sole_amd_drm_card": amd_cards[0][0],
    }


def _zero_counters() -> dict[str, int]:
    return {
        "lower_attempts": 0,
        "lower_completions": 0,
        "compile_attempts": 0,
        "compile_completions": 0,
        "input_device_put_attempts": 0,
        "input_device_put_completions": 0,
        "candidate_attempts": 0,
        "candidate_completions": 0,
        "device_get_attempts": 0,
        "device_get_completions": 0,
        "lowered_callable_invocations": 0,
        "checked_executable_invocations": 0,
    }


def _completed_counters() -> dict[str, int]:
    result = _zero_counters()
    for name in (
        "lower_attempts",
        "lower_completions",
        "compile_attempts",
        "compile_completions",
        "input_device_put_attempts",
        "input_device_put_completions",
        "candidate_attempts",
        "candidate_completions",
        "device_get_attempts",
        "device_get_completions",
        "checked_executable_invocations",
    ):
        result[name] = 1
    return result


def _exact_contract() -> dict[str, Any]:
    q_shape = [_BATCH_SIZE, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM]
    kv_shape = [_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM]
    return {
        "model_family": "Qwen/Qwen3.5-4B",
        "operation": "query_bounded_gqa_t512_forward_and_full_vjp",
        "gpu_architecture": _EXPECTED_GPU_ARCHITECTURE,
        "gpu_pci_device_id": _EXPECTED_AMD_PCI_DEVICE_ID,
        "inputs": [
            {"name": "q", "shape": q_shape, "dtype": "bfloat16"},
            {"name": "k", "shape": kv_shape, "dtype": "bfloat16"},
            {"name": "v", "shape": kv_shape, "dtype": "bfloat16"},
            {
                "name": "key_mask",
                "shape": [_BATCH_SIZE, _SEQUENCE_LENGTH],
                "dtype": "int32",
                "value": "all_ones",
            },
            {
                "name": "dout",
                "shape": q_shape,
                "dtype": "bfloat16",
                "value": "independent_host_pcg64_nonzero_signed_grid",
            },
        ],
        "outputs": [
            {"name": "output", "shape": q_shape, "dtype": "bfloat16"},
            {"name": "dq", "shape": q_shape, "dtype": "bfloat16"},
            {"name": "dk", "shape": kv_shape, "dtype": "bfloat16"},
            {"name": "dv", "shape": kv_shape, "dtype": "bfloat16"},
        ],
        "scale": _ATTENTION_SCALE,
        "scale_exact_fraction": "3/32",
        "tiles": {
            "query_chunk_size": _QUERY_CHUNK_SIZE,
            "block_q": _BLOCK_Q,
            "block_k": _BLOCK_K,
            "backward_block_q": _BACKWARD_BLOCK_Q,
            "backward_block_k": _BACKWARD_BLOCK_K,
        },
        "compile_gate": {
            "required_dialects": ["stablehlo", "optimized_hlo"],
            "exact_custom_calls": dict.fromkeys(_EXPECTED_MARKERS, 1),
            "exact_total_custom_calls": 3,
            "sole_target": _EXACT_TARGET,
            "exact_query_start_metadata_per_call": 0,
            "exact_query_size_metadata_per_call": 512,
            "all_calls_directly_owned_by_sole_entry_computation": True,
            "no_callable_or_control_flow_container": True,
            "no_outer_while": True,
            "exact_argument_bytes": _EXPECTED_ARGUMENT_BYTES,
            "exact_output_bytes": _EXPECTED_OUTPUT_BYTES,
            "exact_alias_bytes": _EXPECTED_ALIAS_BYTES,
            "maximum_temporary_bytes": _MAX_TEMP_BYTES,
        },
        "execution_contract": {
            "lower_calls": 1,
            "compile_calls": 1,
            "input_tuple_device_put_calls": 1,
            "checked_executable_invocations": 1,
            "device_get_calls": 1,
            "logical_internal_dispatches": {
                "forward": 1,
                "dq": 1,
                "dkdv": 1,
                "total": 3,
            },
            "warmup_invocations": 0,
            "replay_invocations": 0,
            "second_vjp_invocations": 0,
            "gpu_reference_invocations": 0,
            "device_error_reduction_invocations": 0,
            "model_invocations": 0,
        },
        "reference": {
            "location": "host_numpy_only",
            "dtype": "float32",
            "algorithm": "three-pass tiled causal softmax forward and analytic VJP",
            "query_tile": _ORACLE_QUERY_TILE,
            "key_tile": _ORACLE_KEY_TILE,
            "full_t_by_t_matrix_constructed": False,
            "equations": {
                "delta": "row_sum(output * dout)",
                "dscores": "probability * (dout @ V.T - delta) * scale",
                "dq": "dscores @ K",
                "dk": "dscores.T @ Q grouped over query heads",
                "dv": "probability.T @ dout grouped over query heads",
            },
        },
        "numerical_gate_per_tensor": {
            "finite_required": True,
            "minimum_reference_l2_norm": _MIN_REFERENCE_NORM,
            "relative_l2_strictly_below": _MAX_RELATIVE_L2,
            "minimum_cosine": _MIN_COSINE,
            "maximum_absolute_error": _MAX_ABSOLUTE_ERROR,
        },
        "candidate_total_seconds_strictly_below": _MAX_CANDIDATE_SECONDS,
        "promotion_total_seconds_strictly_below": (_MAX_PROMOTION_CANDIDATE_SECONDS),
    }


def _abstract_contract() -> dict[str, Any]:
    return {
        "operation": "query_bounded_gqa_t512_full_vjp_refusal",
        "exact_logical_internal_calls": ["forward_q0", "dq_q0", "dkdv_q0"],
        "gpu_work_authorized": False,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=("abstract", "rocm"),
        default="abstract",
        help="refusal-only by default; guarded ROCm work requires rocm plus --allow-gpu",
    )
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help="acknowledge one guarded compile and one checked full-VJP invocation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="exclusive mode-0600 JSONL artifact (required for ROCm)",
    )
    args = parser.parse_args(argv)
    if args.platform == "rocm" and not args.allow_gpu:
        parser.error(
            "--platform rocm requires the explicit --allow-gpu acknowledgement"
        )
    if args.platform == "abstract" and args.allow_gpu:
        parser.error("--allow-gpu is only valid with --platform rocm")
    if args.platform == "rocm" and args.output is None:
        parser.error("--platform rocm requires --output for a private JSONL artifact")
    if args.output is not None and args.output.exists():
        parser.error("refusing to overwrite existing output")
    return args


def _configure_rocm_environment() -> dict[str, str | None]:
    return _nonzero_probe()._configure_rocm_environment()


def _environment_manifest(environment: dict[str, str | None]) -> dict[str, Any]:
    return _nonzero_probe()._environment_manifest(environment)


def _prove_command_buffers_disabled(
    environment: dict[str, str | None],
) -> dict[str, Any]:
    return _nonzero_probe()._prove_command_buffers_disabled(environment)


def _load_safety_helpers() -> tuple[
    Callable[[], ContextManager[dict[str, Any]]],
    Callable[[], dict[str, Any]],
]:
    return _nonzero_probe()._load_safety_helpers()


def _safety_binding_manifest(helpers: tuple[Any, Any]) -> dict[str, Any]:
    return _nonzero_probe()._safety_binding_manifest(helpers)


def _open_exclusive_output(path: Path) -> TextIO:
    return _nonzero_probe()._open_exclusive_output(path)


def _assert_fresh_accelerator_process() -> None:
    _nonzero_probe()._assert_fresh_accelerator_process()


def _public_clean_safety(safety: Any, stage: str) -> dict[str, Any]:
    return _nonzero_probe()._public_clean_safety(safety, stage)


def _public_safety_preflight(safety: Any) -> dict[str, Any]:
    return _nonzero_probe()._public_safety_preflight(safety)


def _journal_checkpoint(
    require_clean_boot: Callable[[], dict[str, Any]],
    output: TextIO,
    stage: str,
    counters: dict[str, int],
) -> dict[str, Any]:
    if stage not in _JOURNAL_STAGES:
        raise RuntimeError("refusing an undeclared VJP journal stage")
    safety = _public_clean_safety(require_clean_boot(), stage)
    _emit(
        {
            "record_type": "journal_checkpoint",
            "timestamp": _utc_now(),
            "stage": stage,
            "safety": safety,
            "counters": dict(counters),
        },
        output,
    )
    return safety


def _decode_ir(text: str) -> str:
    return _nonzero_probe()._decode_local_mlir_hex_escapes(text)


def _mask_ir_quoted_content(text: str) -> str:
    """Mask quoted payloads while retaining offsets, braces, and newlines."""
    result = list(text)
    quoted = False
    escaped = False
    for index, character in enumerate(text):
        if quoted:
            if character != "\n":
                result[index] = " "
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
            result[index] = " "
    return "".join(result)


def _matching_closing_brace(masked_text: str, opening: int) -> int | None:
    if opening < 0 or opening >= len(masked_text) or masked_text[opening] != "{":
        raise ValueError("entry-region opening must identify a brace")
    depth = 0
    for index in range(opening, len(masked_text)):
        character = masked_text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index
            if depth < 0:
                return None
    return None


def _entry_call_ownership(text: str, dialect: str) -> dict[str, Any]:
    """Require every logical custom call directly in the sole entry body."""
    decoded = _decode_ir(text)
    masked = _mask_ir_quoted_content(decoded)
    if dialect == "stablehlo":
        entry_pattern = re.compile(r"(?m)^[^\n]*\bfunc\.func\s+public\s+@main\b[^\n]*$")
        call_pattern = re.compile(r"\bstablehlo\.custom_call\b")
        forbidden_pattern = re.compile(
            r"\b(?:func\.call|stablehlo\.(?:case|if|map|while))\b"
        )
    elif dialect == "optimized_hlo":
        entry_pattern = re.compile(r"(?m)^\s*ENTRY\b[^\n]*$")
        call_pattern = re.compile(r"\bcustom-call\(")
        forbidden_pattern = re.compile(
            r"(?<!custom-)\b(?:async-start|call|conditional|map|while)\s*\("
        )
    else:
        raise ValueError("unsupported VJP IR dialect")

    entries = list(entry_pattern.finditer(masked))
    calls = list(call_pattern.finditer(masked))
    forbidden = list(forbidden_pattern.finditer(masked))
    opening: int | None = None
    closing: int | None = None
    if len(entries) == 1:
        opening = masked.rfind("{", entries[0].start(), entries[0].end())
        if opening >= 0:
            closing = _matching_closing_brace(masked, opening)
        else:
            opening = None

    direct: list[bool] = []
    if opening is not None and closing is not None:
        for call in calls:
            owned = opening < call.start() < closing
            relative = masked[opening + 1 : call.start()]
            depth = relative.count("{") - relative.count("}")
            direct.append(owned and depth == 0)
    else:
        direct = [False] * len(calls)

    checks = {
        "exactly_one_entry_computation": len(entries) == 1,
        "entry_region_is_balanced": opening is not None and closing is not None,
        "all_custom_calls_directly_owned_by_entry": len(direct) == 3 and all(direct),
        "no_callable_or_control_flow_container": len(forbidden) == 0,
    }
    return {
        "entry_count": len(entries),
        "custom_call_count": len(calls),
        "direct_entry_custom_call_count": sum(direct),
        "forbidden_container_count": len(forbidden),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _is_bounded_marker_like(token: str) -> bool:
    normalized = re.sub(r"[-.$]+", "_", token.lower())
    return any(
        stem in normalized
        for stem in (
            "query_bounded_gqa_forward_q",
            "query_bounded_gqa_dq_q",
            "query_bounded_gqa_dkdv_q",
        )
    )


def _strict_vjp_ir_summary(text: str, dialect: str) -> dict[str, Any]:
    compile_probe = _compile_probe()
    definitions = compile_probe._metadata_definitions(text)
    raw_blocks = compile_probe._custom_call_blocks(text, dialect)
    blocks = [
        compile_probe._resolved_block_metadata(block, definitions)
        for block in raw_blocks
    ]
    decoded_text = _decode_ir(text)
    ownership = _entry_call_ownership(text, dialect)
    all_tokens = _IR_NAME_TOKEN_PATTERN.findall(decoded_text)
    bounded_tokens = [token for token in all_tokens if _is_bounded_marker_like(token)]
    unexpected = [
        token for token in bounded_tokens if token not in _EXPECTED_MARKERS.values()
    ]
    if dialect == "stablehlo":
        textual_count = len(re.findall(r"\bstablehlo\.custom_call\b", text))
        while_count = len(re.findall(r"\bstablehlo\.while\b", text))
    elif dialect == "optimized_hlo":
        textual_count = len(re.findall(r"\bcustom-call\(", text))
        while_count = len(re.findall(r"\bwhile\s*\(", text))
    else:
        raise ValueError("unsupported VJP IR dialect")

    calls: list[dict[str, Any]] = []
    marker_counts = dict.fromkeys(_EXPECTED_MARKERS, 0)
    all_call_checks: list[bool] = []
    for index, block in enumerate(blocks):
        targets = compile_probe._custom_call_targets(block, dialect)
        tokens = _IR_NAME_TOKEN_PATTERN.findall(_decode_ir(block))
        kinds = [kind for kind, marker in _EXPECTED_MARKERS.items() if marker in tokens]
        for kind in kinds:
            marker_counts[kind] += 1
        query_start = _nonzero_probe()._canonical_raw_metadata_field(
            _decode_ir(block), "query_start", 0
        )
        query_size = _nonzero_probe()._canonical_raw_metadata_field(
            _decode_ir(block), "query_size", _SEQUENCE_LENGTH
        )
        checks = {
            "sole_exact_target": targets == {_EXACT_TARGET},
            "exactly_one_expected_marker": len(kinds) == 1,
            "query_start_is_exact_canonical_zero": query_start["passed"],
            "query_size_is_exact_canonical_512": query_size["passed"],
        }
        all_call_checks.append(all(checks.values()))
        calls.append(
            {
                "index": index,
                "kind": kinds[0] if len(kinds) == 1 else None,
                "targets": sorted(targets),
                "query_start": query_start,
                "query_size": query_size,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    checks = {
        "parser_matches_textual_custom_call_count": len(raw_blocks) == textual_count,
        "exactly_three_custom_calls_total": len(raw_blocks) == textual_count == 3,
        "all_three_calls_pass_target_marker_and_metadata": len(all_call_checks) == 3
        and all(all_call_checks),
        "each_expected_marker_occurs_in_exactly_one_call": all(
            marker_counts[kind] == 1 for kind in _EXPECTED_MARKERS
        ),
        "no_unexpected_or_lookalike_bounded_markers": not unexpected,
        "all_calls_directly_owned_by_sole_entry_without_callable_container": (
            ownership["passed"]
        ),
        "no_outer_while": while_count == 0,
    }
    return {
        "dialect": dialect,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "characters": len(text),
        "lines": len(text.splitlines()),
        "custom_call_count": len(raw_blocks),
        "custom_call_token_count": textual_count,
        "marker_call_counts": marker_counts,
        "unexpected_bounded_marker_occurrences": len(unexpected),
        "while_count": while_count,
        "calls": calls,
        "entry_call_ownership": ownership,
        "checks": checks,
        "raw_ir_emitted": False,
        "passed": all(checks.values()),
    }


def _structural_gate(*summaries: dict[str, Any]) -> dict[str, Any]:
    dialects = [str(summary.get("dialect")) for summary in summaries]
    exact = len(summaries) == 2 and sorted(dialects) == [
        "optimized_hlo",
        "stablehlo",
    ]
    per = {str(summary.get("dialect")): summary for summary in summaries}
    checks = {
        "exactly_one_summary_per_required_dialect": exact,
        "stablehlo_passed": exact and per.get("stablehlo", {}).get("passed") is True,
        "optimized_hlo_passed": exact
        and per.get("optimized_hlo", {}).get("passed") is True,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _compiled_memory_gate(memory: dict[str, Any]) -> dict[str, Any]:
    values = {
        name: memory.get(name)
        for name in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "alias_size_in_bytes",
            "temp_size_in_bytes",
        )
    }
    available = memory.get("available") is True and all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in values.values()
    )
    checks = {
        "memory_analysis_available": available,
        "argument_bytes_exact": values["argument_size_in_bytes"]
        == _EXPECTED_ARGUMENT_BYTES,
        "output_bytes_exact": values["output_size_in_bytes"] == _EXPECTED_OUTPUT_BYTES,
        "alias_bytes_exact": values["alias_size_in_bytes"] == _EXPECTED_ALIAS_BYTES,
        "temporary_bytes_at_most_64_mib": available
        and values["temp_size_in_bytes"] <= _MAX_TEMP_BYTES,
    }
    return {
        "expected_argument_bytes": _EXPECTED_ARGUMENT_BYTES,
        "expected_output_bytes": _EXPECTED_OUTPUT_BYTES,
        "expected_alias_bytes": _EXPECTED_ALIAS_BYTES,
        "maximum_temporary_bytes": _MAX_TEMP_BYTES,
        "checks": checks,
        "passed": all(checks.values()),
    }


class _CheckedVjpExecutable:
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
            raise RuntimeError("refusing to expose VJP executable without passed gates")
        self._compiled = compiled
        self._consumed = False
        self._counters = counters
        self.proof = proof

    def invoke(
        self, jax: Any, arguments: tuple[Any, ...], on_started: Callable[[], None]
    ) -> Any:
        if self._consumed:
            raise RuntimeError("checked VJP capability was already consumed")
        self._consumed = True
        self._counters["candidate_attempts"] += 1
        self._counters["checked_executable_invocations"] += 1
        on_started()
        result = self._compiled(*arguments)
        result = jax.block_until_ready(result)
        self._counters["candidate_completions"] += 1
        return result


def _wrap_checked(
    compiled: Any, proof: dict[str, Any], counters: dict[str, int]
) -> _CheckedVjpExecutable:
    return _CheckedVjpExecutable(
        compiled,
        proof=proof,
        counters=counters,
        token=_CHECKED_CAPABILITY_TOKEN,
    )


def _shape_signature(jax: Any, jnp: Any) -> tuple[Any, ...]:
    q_shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM)
    kv_shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM)
    return (
        jax.ShapeDtypeStruct(q_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(kv_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(kv_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct((_BATCH_SIZE, _SEQUENCE_LENGTH), jnp.int32),
        jax.ShapeDtypeStruct(q_shape, jnp.bfloat16),
    )


def _compile_checked_vjp(
    jax: Any,
    jnp: Any,
    query_bounded_gqa: Any,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    output: TextIO,
) -> tuple[_CheckedVjpExecutable, dict[str, Any]]:
    def forward_and_vjp(
        q_arg: Any, k_arg: Any, v_arg: Any, mask_arg: Any, dout_arg: Any
    ) -> tuple[Any, Any, Any, Any]:
        value, pullback = jax.vjp(
            lambda q_item, k_item, v_item: query_bounded_gqa(
                q_item,
                k_item,
                v_item,
                mask_arg,
                scale=_ATTENTION_SCALE,
                query_chunk_size=_QUERY_CHUNK_SIZE,
                block_q=_BLOCK_Q,
                block_k=_BLOCK_K,
                backward_block_q=_BACKWARD_BLOCK_Q,
                backward_block_k=_BACKWARD_BLOCK_K,
                interpret=False,
            ),
            q_arg,
            k_arg,
            v_arg,
        )
        dq, dk, dv = pullback(dout_arg)
        return value, dq, dk, dv

    _emit(
        {
            "record_type": "stage",
            "stage": "vjp_lower_started",
            "timestamp": _utc_now(),
            "counters": dict(counters),
        },
        output,
    )
    counters["lower_attempts"] += 1
    lower_start = time.perf_counter()
    try:
        lowered = jax.jit(forward_and_vjp).lower(*_shape_signature(jax, jnp))
        lower_seconds = time.perf_counter() - lower_start
        counters["lower_completions"] += 1
        stablehlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
        stablehlo = _strict_vjp_ir_summary(stablehlo_text, "stablehlo")
        del stablehlo_text
        _emit(
            {
                "record_type": "lowered",
                "stage": "vjp_lower_complete_metadata_only",
                "timestamp": _utc_now(),
                "lower_seconds": _json_safe_duration(lower_seconds),
                "stablehlo": stablehlo,
                "raw_ir_emitted": False,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_vjp_lower_attempt", counters
        )

    _emit(
        {
            "record_type": "stage",
            "stage": "vjp_compile_started",
            "timestamp": _utc_now(),
            "counters": dict(counters),
        },
        output,
    )
    counters["compile_attempts"] += 1
    compiled = None
    release = False
    compile_start = time.perf_counter()
    try:
        compiled = lowered.compile()
        compile_seconds = time.perf_counter() - compile_start
        counters["compile_completions"] += 1
        optimized_hlo_text = compiled.as_text()
        optimized_hlo = _strict_vjp_ir_summary(optimized_hlo_text, "optimized_hlo")
        del optimized_hlo_text
        memory = _compile_probe()._compiled_memory(compiled)
        structural = _structural_gate(stablehlo, optimized_hlo)
        memory_gate = _compiled_memory_gate(memory)
        proof = {
            "structural_gate_passed": structural["passed"],
            "compiled_memory_gate_passed": memory_gate["passed"],
            "exact_logical_dispatches": dict.fromkeys(_EXPECTED_MARKERS, 1),
            "explicit_scale_exact_fraction": "3/32",
            "passed": structural["passed"] and memory_gate["passed"],
        }
        report = {
            "record_type": "vjp_compiled",
            "stage": "vjp_compile_release_gate",
            "timestamp": _utc_now(),
            "lower_seconds": _json_safe_duration(lower_seconds),
            "compile_seconds": _json_safe_duration(compile_seconds),
            "stablehlo": stablehlo,
            "optimized_hlo": optimized_hlo,
            "compiled_memory": memory,
            "structural_gate": structural,
            "compiled_memory_gate": memory_gate,
            "release_gate": proof,
            "raw_ir_emitted": False,
            "counters": dict(counters),
        }
        _emit(report, output)
        if not proof["passed"]:
            raise RuntimeError(
                "VJP executable failed structural or memory release gate"
            )
        checked = _wrap_checked(compiled, proof, counters)
        release = True
    finally:
        if compiled is not None and not release:
            del compiled
        _journal_checkpoint(
            require_clean_boot, output, "after_vjp_compile_attempt", counters
        )
    del lowered
    return checked, report


def _iid_nonzero_grid(
    np: Any,
    ml_dtypes: Any,
    rng: Any,
    shape: tuple[int, ...],
    maximum_magnitude: int,
) -> Any:
    magnitudes = rng.integers(1, maximum_magnitude + 1, size=shape, dtype=np.int16)
    signs = 2 * rng.integers(0, 2, size=shape, dtype=np.int16) - 1
    result = (
        (magnitudes * signs).astype(np.float32) / np.float32(_GRID_DENOMINATOR)
    ).astype(ml_dtypes.bfloat16)
    if tuple(result.shape) != shape or int(np.count_nonzero(result)) != result.size:
        raise RuntimeError("nonzero BF16 grid construction violated its contract")
    return result


def _validate_oracle_inputs(
    np: Any, q: Any, k: Any, v: Any, key_mask: Any, dout: Any
) -> tuple[int, int, int, int]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or dout.ndim != 4:
        raise RuntimeError("VJP oracle requires rank-four Q/K/V/dout")
    batch, sequence, query_heads, head_dim = q.shape
    if (
        batch != 1
        or k.shape[0] != batch
        or v.shape != k.shape
        or k.shape[1] != sequence
        or k.shape[3] != head_dim
        or dout.shape != q.shape
        or query_heads % k.shape[2]
        or sequence <= 0
        or head_dim <= 0
    ):
        raise RuntimeError("VJP oracle arrays violate grouped self-attention shapes")
    if (
        not isinstance(key_mask, np.ndarray)
        or key_mask.shape != (batch, sequence)
        or key_mask.dtype != np.dtype(np.int32)
        or not bool(np.all(key_mask == 1))
    ):
        raise RuntimeError("VJP gate requires an exact all-valid int32-ones key mask")
    return sequence, query_heads, k.shape[2], head_dim


def _tiled_causal_gqa_forward_vjp_oracle(
    np: Any,
    q: Any,
    k: Any,
    v: Any,
    key_mask: Any,
    dout: Any,
    *,
    scale: float,
    query_tile: int = _ORACLE_QUERY_TILE,
    key_tile: int = _ORACLE_KEY_TILE,
) -> tuple[tuple[Any, Any, Any, Any], dict[str, Any]]:
    sequence, query_heads, kv_heads, head_dim = _validate_oracle_inputs(
        np, q, k, v, key_mask, dout
    )
    if query_tile <= 0 or key_tile <= 0 or not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("VJP oracle tile sizes and scale must be positive")
    group_size = query_heads // kv_heads
    output = np.empty(q.shape, dtype=np.float32)
    dq = np.zeros(q.shape, dtype=np.float32)
    dk = np.zeros(k.shape, dtype=np.float32)
    dv = np.zeros(v.shape, dtype=np.float32)
    maximum_absolute_valid_logit = 0.0

    for query_head in range(query_heads):
        kv_head = query_head // group_size
        for query_offset in range(0, sequence, query_tile):
            query_stop = min(query_offset + query_tile, sequence)
            rows = query_stop - query_offset
            query_positions = np.arange(query_offset, query_stop, dtype=np.int32)
            q_block = np.asarray(q[0, query_offset:query_stop, query_head], np.float32)
            dout_block = np.asarray(
                dout[0, query_offset:query_stop, query_head], np.float32
            )

            row_max = np.full((rows,), -np.inf, dtype=np.float32)
            row_sum = np.zeros((rows,), dtype=np.float32)
            for key_offset in range(0, query_stop, key_tile):
                key_stop = min(key_offset + key_tile, query_stop)
                key_positions = np.arange(key_offset, key_stop, dtype=np.int32)
                k_block = np.asarray(k[0, key_offset:key_stop, kv_head], np.float32)
                logits = (q_block @ k_block.T).astype(np.float32, copy=False)
                logits *= np.float32(scale)
                valid = key_positions[None, :] <= query_positions[:, None]
                valid_logits = logits[valid]
                if valid_logits.size:
                    maximum_absolute_valid_logit = max(
                        maximum_absolute_valid_logit,
                        float(np.max(np.abs(valid_logits))),
                    )
                logits = np.where(valid, logits, np.float32(-np.inf))
                block_max = np.max(logits, axis=1)
                next_max = np.maximum(row_max, block_max)
                correction = np.exp(row_max - next_max).astype(np.float32, copy=False)
                probabilities = np.exp(logits - next_max[:, None]).astype(
                    np.float32, copy=False
                )
                row_sum = correction * row_sum + np.sum(
                    probabilities, axis=1, dtype=np.float32
                )
                row_max = next_max

            output_block = np.zeros((rows, head_dim), dtype=np.float32)
            for key_offset in range(0, query_stop, key_tile):
                key_stop = min(key_offset + key_tile, query_stop)
                key_positions = np.arange(key_offset, key_stop, dtype=np.int32)
                k_block = np.asarray(k[0, key_offset:key_stop, kv_head], np.float32)
                v_block = np.asarray(v[0, key_offset:key_stop, kv_head], np.float32)
                logits = (q_block @ k_block.T).astype(np.float32, copy=False)
                logits *= np.float32(scale)
                valid = key_positions[None, :] <= query_positions[:, None]
                probabilities = np.exp(
                    np.where(valid, logits - row_max[:, None], np.float32(-np.inf))
                ).astype(np.float32, copy=False)
                probabilities /= row_sum[:, None]
                output_block += (probabilities @ v_block).astype(np.float32, copy=False)
                dv[0, key_offset:key_stop, kv_head] += (
                    probabilities.T @ dout_block
                ).astype(np.float32, copy=False)
            output[0, query_offset:query_stop, query_head] = output_block
            delta = np.sum(output_block * dout_block, axis=1, dtype=np.float32)

            dq_block = np.zeros((rows, head_dim), dtype=np.float32)
            for key_offset in range(0, query_stop, key_tile):
                key_stop = min(key_offset + key_tile, query_stop)
                key_positions = np.arange(key_offset, key_stop, dtype=np.int32)
                k_block = np.asarray(k[0, key_offset:key_stop, kv_head], np.float32)
                v_block = np.asarray(v[0, key_offset:key_stop, kv_head], np.float32)
                logits = (q_block @ k_block.T).astype(np.float32, copy=False)
                logits *= np.float32(scale)
                valid = key_positions[None, :] <= query_positions[:, None]
                probabilities = np.exp(
                    np.where(valid, logits - row_max[:, None], np.float32(-np.inf))
                ).astype(np.float32, copy=False)
                probabilities /= row_sum[:, None]
                dprobabilities = (dout_block @ v_block.T).astype(np.float32, copy=False)
                dscores = probabilities * (dprobabilities - delta[:, None])
                dscores *= np.float32(scale)
                dq_block += (dscores @ k_block).astype(np.float32, copy=False)
                dk[0, key_offset:key_stop, kv_head] += (dscores.T @ q_block).astype(
                    np.float32, copy=False
                )
            dq[0, query_offset:query_stop, query_head] = dq_block

    if not all(bool(np.all(np.isfinite(item))) for item in (output, dq, dk, dv)):
        raise RuntimeError("tiled FP32 VJP oracle produced non-finite tensors")
    conservative_scratch_bytes = (
        4
        * (
            6 * query_tile * key_tile
            + 5 * query_tile * head_dim
            + 4 * key_tile * head_dim
            + 12 * query_tile
            + 4 * key_tile
        )
        + query_tile * key_tile
        + 4 * (2 * query_tile + 2 * key_tile)
    )
    return (output, dq, dk, dv), {
        "implementation": "independent_host_fp32_three_pass_tiled_causal_gqa_forward_vjp",
        "query_tile": query_tile,
        "key_tile": key_tile,
        "full_t_by_t_matrix_constructed": False,
        "q_k_v_dout_converted_to_fp32_by_active_tile_only": True,
        "global_fp32_outputs": ["output", "dq", "dk", "dv"],
        "conservative_accounted_numpy_array_scratch_bytes": conservative_scratch_bytes,
        "scratch_accounting_excludes": [
            "required_input_output_and_gradient_arrays",
            "python_and_numpy_object_overhead",
            "numpy_blas_internal_workspace",
            "allocator_fragmentation",
        ],
        "scratch_bytes_are_conservative_accounting_not_measured_peak": True,
        "observed_maximum_absolute_valid_logit": maximum_absolute_valid_logit,
        "scale": float(scale),
        "accelerator_used": False,
    }


def _dense_causal_gqa_forward_vjp_reference(
    np: Any, q: Any, k: Any, v: Any, key_mask: Any, dout: Any, *, scale: float
) -> tuple[Any, Any, Any, Any]:
    sequence, query_heads, kv_heads, head_dim = _validate_oracle_inputs(
        np, q, k, v, key_mask, dout
    )
    if sequence > _SEQUENCE_LENGTH:
        raise RuntimeError("dense reference is restricted to bounded CPU tests")
    group_size = query_heads // kv_heads
    q_fp32 = np.asarray(q, np.float32)[0]
    k_fp32 = np.asarray(k, np.float32)[0]
    v_fp32 = np.asarray(v, np.float32)[0]
    dout_fp32 = np.asarray(dout, np.float32)[0]
    output = np.empty(q.shape, np.float32)
    dq = np.zeros(q.shape, np.float32)
    dk = np.zeros(k.shape, np.float32)
    dv = np.zeros(v.shape, np.float32)
    invalid = np.triu(np.ones((sequence, sequence), dtype=np.bool_), k=1)
    for query_head in range(query_heads):
        kv_head = query_head // group_size
        logits = (q_fp32[:, query_head] @ k_fp32[:, kv_head].T).astype(
            np.float32, copy=False
        )
        logits *= np.float32(scale)
        logits[invalid] = -np.inf
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits).astype(np.float32, copy=False)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True, dtype=np.float32)
        output_head = (probabilities @ v_fp32[:, kv_head]).astype(
            np.float32, copy=False
        )
        output[0, :, query_head] = output_head
        delta = np.sum(output_head * dout_fp32[:, query_head], axis=1, dtype=np.float32)
        dprobabilities = (dout_fp32[:, query_head] @ v_fp32[:, kv_head].T).astype(
            np.float32, copy=False
        )
        dscores = probabilities * (dprobabilities - delta[:, None])
        dscores *= np.float32(scale)
        dq[0, :, query_head] = dscores @ k_fp32[:, kv_head]
        dk[0, :, kv_head] += dscores.T @ q_fp32[:, query_head]
        dv[0, :, kv_head] += probabilities.T @ dout_fp32[:, query_head]
    return output, dq, dk, dv


def _array_manifest(name: str, value: Any) -> dict[str, Any]:
    return _nonzero_probe()._array_manifest(name, value)


def _construct_host_case(
    np: Any, ml_dtypes: Any
) -> tuple[tuple[Any, ...], list[dict[str, Any]], tuple[Any, ...], dict[str, Any]]:
    q_shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM)
    kv_shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM)
    input_rng = np.random.Generator(np.random.PCG64(_INPUT_SEED))
    q = _iid_nonzero_grid(
        np, ml_dtypes, input_rng, q_shape, _QKV_MAXIMUM_INTEGER_MAGNITUDE
    )
    k = _iid_nonzero_grid(
        np, ml_dtypes, input_rng, kv_shape, _QKV_MAXIMUM_INTEGER_MAGNITUDE
    )
    v = _iid_nonzero_grid(
        np, ml_dtypes, input_rng, kv_shape, _QKV_MAXIMUM_INTEGER_MAGNITUDE
    )
    key_mask = np.ones((_BATCH_SIZE, _SEQUENCE_LENGTH), dtype=np.int32)
    dout_rng = np.random.Generator(np.random.PCG64(_COTANGENT_SEED))
    dout = _iid_nonzero_grid(
        np,
        ml_dtypes,
        dout_rng,
        q_shape,
        _COTANGENT_MAXIMUM_INTEGER_MAGNITUDE,
    )
    inputs = (q, k, v, key_mask, dout)
    expected, oracle = _tiled_causal_gqa_forward_vjp_oracle(
        np, q, k, v, key_mask, dout, scale=_ATTENTION_SCALE
    )
    input_manifests = [
        _array_manifest(name, value)
        for name, value in zip(("q", "k", "v", "key_mask", "dout"), inputs, strict=True)
    ]
    expected_manifests = [
        _array_manifest(name, value)
        for name, value in zip(("output", "dq", "dk", "dv"), expected, strict=True)
    ]
    norms = {
        name: float(np.linalg.norm(np.asarray(value, np.float64).ravel()))
        for name, value in zip(("output", "dq", "dk", "dv"), expected, strict=True)
    }
    if not all(
        math.isfinite(norm) and norm > _MIN_REFERENCE_NORM for norm in norms.values()
    ):
        raise RuntimeError("VJP host reference contains a degenerate tensor norm")
    checks = {
        "input_hashes_pinned": {
            item["name"]: item["sha256"] for item in input_manifests
        }
        == _EXPECTED_INPUT_SHA256,
        "reference_hashes_pinned": {
            item["name"]: item["sha256"] for item in expected_manifests
        }
        == _EXPECTED_REFERENCE_SHA256,
        "reference_norms_pinned": all(
            math.isclose(norms[name], expected_norm, rel_tol=0.0, abs_tol=1e-12)
            for name, expected_norm in _EXPECTED_REFERENCE_NORMS.items()
        ),
        "scratch_accounting_pinned": oracle[
            "conservative_accounted_numpy_array_scratch_bytes"
        ]
        == _EXPECTED_ORACLE_SCRATCH_BYTES,
        "maximum_valid_logit_pinned": math.isclose(
            oracle["observed_maximum_absolute_valid_logit"],
            _EXPECTED_MAXIMUM_ABSOLUTE_VALID_LOGIT,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    }
    if not all(checks.values()):
        raise RuntimeError("VJP host input or oracle calibration pin changed")
    return (
        inputs,
        input_manifests,
        expected,
        {
            "outputs": expected_manifests,
            "reference_l2_norms": norms,
            "oracle": oracle,
            "calibration_pin_checks": checks,
        },
    )


def _tensor_metrics(
    np: Any,
    actual_host: Any,
    expected_host: Any,
    *,
    expected_shape: tuple[int, ...],
    expected_actual_nbytes: int,
) -> dict[str, Any]:
    actual_raw = np.asarray(actual_host)
    expected_raw = np.asarray(expected_host)
    actual = actual_raw.astype(np.float32)
    expected = expected_raw.astype(np.float32)
    if tuple(actual.shape) != expected_shape or tuple(expected.shape) != expected_shape:
        raise RuntimeError("candidate or VJP reference returned the wrong shape")
    actual64 = actual.ravel().astype(np.float64)
    expected64 = expected.ravel().astype(np.float64)
    difference64 = actual64 - expected64
    actual_norm = float(np.linalg.norm(actual64))
    expected_norm = float(np.linalg.norm(expected64))
    denominator = max(expected_norm, float(np.finfo(np.float64).tiny))
    cosine_denominator = max(
        actual_norm * expected_norm, float(np.finfo(np.float64).tiny)
    )
    cosine_raw = float(np.vdot(actual64, expected64) / cosine_denominator)
    return {
        "finite": bool(
            np.all(np.isfinite(actual))
            and np.all(np.isfinite(expected))
            and np.all(np.isfinite(difference64))
        ),
        "shape_dtype_nbytes_exact": (
            str(actual_raw.dtype) == "bfloat16"
            and str(expected_raw.dtype) == "float32"
            and int(actual_raw.nbytes) == expected_actual_nbytes
            and int(expected_raw.nbytes) == 2 * expected_actual_nbytes
        ),
        "actual_shape": list(actual_raw.shape),
        "actual_dtype": str(actual_raw.dtype),
        "reference_dtype": str(expected_raw.dtype),
        "actual_nbytes": int(actual_raw.nbytes),
        "reference_nbytes": int(expected_raw.nbytes),
        "reference_l2_norm": expected_norm,
        "max_abs": float(np.max(np.abs(difference64))),
        "mean_abs": float(np.mean(np.abs(difference64))),
        "relative_l2": float(np.linalg.norm(difference64) / denominator),
        "cosine_raw": cosine_raw,
        "cosine": float(np.clip(cosine_raw, -1.0, 1.0)),
        "actual_sha256": hashlib.sha256(actual_raw.tobytes(order="C")).hexdigest(),
        "reference_sha256": hashlib.sha256(expected_raw.tobytes(order="C")).hexdigest(),
    }


def _json_safe_metrics(metrics: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            key: None
            if isinstance(value, float) and not math.isfinite(value)
            else value
            for key, value in values.items()
        }
        for name, values in metrics.items()
    }


def _validate_candidate(
    np: Any,
    actual_host: Any,
    expected_host: tuple[Any, ...],
    seconds: float,
    counters: dict[str, int],
    output: TextIO,
) -> dict[str, Any]:
    if not isinstance(actual_host, tuple) or len(actual_host) != 4:
        raise RuntimeError("checked VJP executable did not return an exact four-tuple")
    shapes = (
        (_BATCH_SIZE, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM),
        (_BATCH_SIZE, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM),
        (_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM),
        (_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM),
    )
    nbytes = (4_194_304, 4_194_304, 1_048_576, 1_048_576)
    metrics = {
        name: _tensor_metrics(
            np,
            actual,
            expected,
            expected_shape=shape,
            expected_actual_nbytes=size,
        )
        for name, actual, expected, shape, size in zip(
            ("output", "dq", "dk", "dv"),
            actual_host,
            expected_host,
            shapes,
            nbytes,
            strict=True,
        )
    }
    per_tensor = {
        name: (
            item["finite"]
            and item["shape_dtype_nbytes_exact"]
            and item["reference_l2_norm"] > _MIN_REFERENCE_NORM
            and math.isfinite(item["relative_l2"])
            and item["relative_l2"] < _MAX_RELATIVE_L2
            and math.isfinite(item["cosine_raw"])
            and item["cosine"] >= _MIN_COSINE
            and math.isfinite(item["max_abs"])
            and item["max_abs"] <= _MAX_ABSOLUTE_ERROR
        )
        for name, item in metrics.items()
    }
    safety_duration = math.isfinite(seconds) and 0 <= seconds < _MAX_CANDIDATE_SECONDS
    promotion_duration = (
        math.isfinite(seconds) and 0 <= seconds < _MAX_PROMOTION_CANDIDATE_SECONDS
    )
    passed = all(per_tensor.values()) and safety_duration and promotion_duration
    record = {
        "record_type": "host_vjp_validation",
        "timestamp": _utc_now(),
        "status": (
            "passed"
            if passed
            else "not_promoted"
            if all(per_tensor.values()) and safety_duration
            else "failed"
        ),
        "metrics": _json_safe_metrics(metrics),
        "thresholds_per_tensor": {
            "finite_required": True,
            "minimum_reference_l2_norm": _MIN_REFERENCE_NORM,
            "relative_l2_strictly_below": _MAX_RELATIVE_L2,
            "minimum_cosine": _MIN_COSINE,
            "maximum_absolute_error": _MAX_ABSOLUTE_ERROR,
        },
        "duration_thresholds": {
            "candidate_total_seconds_strictly_below": _MAX_CANDIDATE_SECONDS,
            "promotion_total_seconds_strictly_below": (
                _MAX_PROMOTION_CANDIDATE_SECONDS
            ),
        },
        "gates": {
            "per_tensor_numerical_passed": per_tensor,
            "all_tensor_numerical_passed": all(per_tensor.values()),
            "safety_duration_passed": safety_duration,
            "promotion_duration_passed": promotion_duration,
            "promotion_passed": passed,
        },
        "candidate_total_seconds": _json_safe_duration(seconds),
        "counters": dict(counters),
    }
    _emit(record, output)
    if not passed:
        raise RuntimeError("full VJP candidate failed numerical or duration gates")
    return record


def _device_put_inputs(
    jax: Any, host_inputs: tuple[Any, ...], counters: dict[str, int]
) -> tuple[Any, ...]:
    counters["input_device_put_attempts"] += 1
    placed = jax.device_put(host_inputs)
    placed = jax.block_until_ready(placed)
    counters["input_device_put_completions"] += 1
    if not isinstance(placed, tuple) or len(placed) != 5:
        raise RuntimeError("explicit VJP device_put did not preserve the five-tuple")
    return placed


def _dispatch_candidate(
    jax: Any,
    executable: _CheckedVjpExecutable,
    arguments: tuple[Any, ...],
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    output: TextIO,
) -> tuple[Any, float]:
    start: float | None = None

    def on_started() -> None:
        nonlocal start
        _emit(
            {
                "record_type": "dispatch_started",
                "timestamp": _utc_now(),
                "label": "single_checked_t512_forward_and_full_vjp",
                "expected_internal_custom_calls": dict.fromkeys(_EXPECTED_MARKERS, 1),
                "counters": dict(counters),
            },
            output,
        )
        start = time.perf_counter()

    attempt_start = time.perf_counter()
    try:
        result = executable.invoke(jax, arguments, on_started)
    finally:
        seconds = time.perf_counter() - (start if start is not None else attempt_start)
        _journal_checkpoint(
            require_clean_boot, output, "after_candidate_dispatch_attempt", counters
        )
    _emit(
        {
            "record_type": "dispatch",
            "timestamp": _utc_now(),
            "label": "single_checked_t512_forward_and_full_vjp",
            "seconds": _json_safe_duration(seconds),
            "expected_internal_custom_call_count": 3,
            "counters": dict(counters),
        },
        output,
    )
    return result, seconds


def _device_get_candidate(
    jax: Any,
    value: Any,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    output: TextIO,
) -> Any:
    counters["device_get_attempts"] += 1
    try:
        host = jax.device_get(value)
        counters["device_get_completions"] += 1
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_candidate_device_get_attempt", counters
        )
    return host


def _run_rocm(
    output: TextIO,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    *,
    environment: dict[str, str | None],
    _dependencies: tuple[Any, Any, Any, Any, Any, Any, Any] | None = None,
) -> int:
    proof = _prove_command_buffers_disabled(environment)
    architecture_binding = _assert_gfx1100_drm()
    _emit(
        {
            "record_type": "command_buffer_environment_proof",
            "timestamp": _utc_now(),
            "status": "passed",
            "proof": proof,
            "counters": dict(counters),
        },
        output,
    )
    try:
        if _dependencies is None:
            import jax
            import jax.numpy as jnp
            import jaxlib
            import ml_dtypes
            import numpy as np
            from jax.extend import backend as jax_backend

            from skyrl.tx.kernels.query_bounded_gqa import query_bounded_gqa
        else:
            jax, jnp, jaxlib, jax_backend, np, ml_dtypes, query_bounded_gqa = (
                _dependencies
            )
        backend = (
            _nonzero_probe()._length_probe()._backend_manifest(jax, jaxlib, jax_backend)
        )
        kernel_binding = _assert_kernel_binding(query_bounded_gqa)
        _emit(
            {
                "record_type": "backend_ready",
                "timestamp": _utc_now(),
                "backend": backend,
                "architecture_binding": architecture_binding,
                "kernel_binding": kernel_binding,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_backend_initialization_attempt", counters
        )

    executable, compile_report = _compile_checked_vjp(
        jax, jnp, query_bounded_gqa, require_clean_boot, counters, output
    )
    try:
        host_inputs, input_manifests, expected_host, reference_manifest = (
            _construct_host_case(np, ml_dtypes)
        )
        _emit(
            {
                "record_type": "host_t512_full_vjp_reference",
                "timestamp": _utc_now(),
                "construction": {
                    "q_k_v": "independent nonzero BF16 host PCG64 signed grids",
                    "dout": "separate-seed independent nonzero BF16 host PCG64 signed grid",
                    "key_mask": "all int32 ones",
                    "scale_exact_fraction": "3/32",
                    "oracle": "independent host FP32 three-pass tiled causal forward/VJP",
                    "accelerator_rng_used": False,
                },
                "inputs": input_manifests,
                "reference": reference_manifest,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_host_reference_construction", counters
        )

    try:
        inputs = _device_put_inputs(jax, host_inputs, counters)
    finally:
        _journal_checkpoint(
            require_clean_boot,
            output,
            "after_explicit_input_device_put_attempt",
            counters,
        )
    actual, seconds = _dispatch_candidate(
        jax, executable, inputs, require_clean_boot, counters, output
    )
    actual_host = _device_get_candidate(
        jax, actual, require_clean_boot, counters, output
    )
    try:
        validation = _validate_candidate(
            np, actual_host, expected_host, seconds, counters, output
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_host_validation", counters
        )
    if counters != _completed_counters():
        raise RuntimeError("full VJP runtime counter contract was not exact")
    _emit(
        {
            "record_type": "runtime_passed",
            "timestamp": _utc_now(),
            "status": "passed_exact_t512_full_arbitrary_cotangent_vjp",
            "scale_exact_fraction": "3/32",
            "compile_release_gate": compile_report["release_gate"],
            "compiled_memory_gate": compile_report["compiled_memory_gate"],
            "host_validation": validation,
            "counters": dict(counters),
            "logical_internal_custom_calls": dict.fromkeys(_EXPECTED_MARKERS, 1),
            "warmup_invocations": 0,
            "replay_invocations": 0,
            "second_vjp_invocations": 0,
            "gpu_reference_invocations": 0,
            "device_error_reduction_invocations": 0,
            "model_dispatcher_connected": False,
        },
        output,
    )
    return 0


def _execute(args: argparse.Namespace, output: TextIO) -> int:
    counters = _zero_counters()
    _emit(
        {
            "record_type": "manifest",
            "timestamp": _utc_now(),
            "platform_requested": args.platform,
            "allow_gpu": args.allow_gpu,
            "scope": (
                "abstract_refusal"
                if args.platform == "abstract"
                else "guarded_exact_t512_full_vjp"
            ),
            "contract": (
                _abstract_contract()
                if args.platform == "abstract"
                else _exact_contract()
            ),
            "compile_may_dispatch_gpu_work": args.platform == "rocm",
            "compile_dispatch_caveat": _COMPILE_GPU_WORK_CAVEAT,
            "fresh_process_required": True,
            "prior_compile_artifact_used": False,
            "raw_ir_emitted": False,
            "outer_profile_rocm_supervision_required": True,
            "outer_profile_rocm_supervision_operational_not_internally_proven": True,
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
                    "pass --platform rocm --allow-gpu --output explicitly under "
                    "profile_rocm.py in a fresh process"
                ),
                "jax_imported": False,
                "counters": dict(counters),
            },
            output,
        )
        return 0

    stage = "fresh_process_preflight"
    try:
        _assert_fresh_accelerator_process()
        stage = "source_binding_preflight"
        source_binding = _assert_static_source_bindings()
        _emit(
            {
                "record_type": "source_binding_proof",
                "timestamp": _utc_now(),
                "stage": "pinned_sources_bound_before_environment_or_jax",
                "proof": source_binding,
                "counters": dict(counters),
            },
            output,
        )
        stage = "safety_callable_binding_preflight"
        guarded_process, require_clean_boot = _load_safety_helpers()
        safety_binding = _safety_binding_manifest((guarded_process, require_clean_boot))
        if safety_binding["passed"] is not True:
            raise RuntimeError("retained safety callable binding proof did not pass")
        _emit(
            {
                "record_type": "safety_callable_binding_proof",
                "timestamp": _utc_now(),
                "stage": "validated_and_retained_before_environment_mutation",
                "proof": safety_binding,
                "counters": dict(counters),
            },
            output,
        )
        stage = "bounded_environment"
        environment = _configure_rocm_environment()
        _emit(
            {
                "record_type": "environment",
                "timestamp": _utc_now(),
                "stage": "bounded_rocm_environment_configured",
                "environment": _environment_manifest(environment),
                "counters": dict(counters),
            },
            output,
        )
        stage = "safety_preflight"
        with guarded_process() as raw_safety:
            safety = _public_safety_preflight(raw_safety)
            _emit(
                {
                    "record_type": "safety_preflight",
                    "timestamp": _utc_now(),
                    "stage": "guard_acquired",
                    "safety": safety,
                    "counters": dict(counters),
                },
                output,
            )
            stage = "runtime"
            try:
                result = _run_rocm(
                    output,
                    require_clean_boot,
                    counters,
                    environment=environment,
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
                        "stage": "current_boot_rechecked",
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
                **_redacted_message_summary(error),
                "counters": dict(counters),
            },
            output,
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.output is None:
        return _execute(args, sys.stdout)
    with _open_exclusive_output(args.output) as output:
        return _execute(args, output)


if __name__ == "__main__":
    sys.exit(main())
