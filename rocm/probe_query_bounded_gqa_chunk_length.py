#!/usr/bin/env python3
"""Guarded C256 GQA last-chunk length-ladder probe.

The default ``abstract`` mode emits a refusal manifest without importing JAX.
ROCm requires ``--platform rocm --allow-gpu --output`` in a fresh process under
``profile_rocm.py``.  The exact BF16 C256 last-query-chunk operation is lowered
and compiled once.  This probe independently parses both IR dialects and
applies its length-derived argument, exact 2,097,152-byte output, zero-alias,
and 64-MiB temporary-memory gates.  Only a private
one-shot capability created after those gates may invoke the executable.

Exactly one of T={1024,2048,4096,8192,16384,24576,32768} is admitted per fresh
process.  The single candidate uses zero Q/K, deterministic high-contrast non-affine V
that varies across position, KV head, and head dimension, plus an all-valid
mask.  An independent host NumPy FP32 oracle computes global causal prefix
means for the final 256 global query positions.  There is no replay, backward path,
accelerator reference, device-side reduction, model dispatcher, or second
executable invocation.
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
_QUERY_SIZE = 256
_ALLOWED_SEQUENCE_LENGTHS = (1024, 2048, 4096, 8192, 16384, 24576, 32768)
_QUERY_HEADS = 16
_KV_HEADS = 4
_HEAD_DIM = 256
_GROUP_SIZE = _QUERY_HEADS // _KV_HEADS
_BLOCK_Q = 64
_BLOCK_K = 64
_EXPECTED_OUTPUT_BYTES = 2_097_152
_MAX_TEMP_BYTES = 64 * 1024**2
_MAX_RELATIVE_L2 = 0.01
_MIN_COSINE = 0.9999
_MAX_ABSOLUTE_ERROR = 0.02
_MAX_DISPATCH_SECONDS = 0.1
_MAX_PROMOTION_DISPATCH_SECONDS = 0.075
_MARKER_PREFIX = "query_bounded_gqa_forward_q"
_EXACT_ROCM_TRITON_TARGET = "__gpu$xla.gpu.triton"
_EXPECTED_COMPILE_HELPER_SOURCE_SHA256 = (
    "24eeed83e93da1133d2e1bc3d0065bc8369d13fa324d2157e37db8b9c4a4d12d"
)
_EXPECTED_KERNEL_SOURCE_SHA256 = (
    "51e2fd91eb270f7b25ecdd117d7f06aa48a8e4af282a5a7e5e6b4c2a25dc52c9"
)
_IR_NAME_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.$-]*")
_PALLAS_TRITON_TARGETS = frozenset(
    {
        "pallas",
        "pallas_call",
        "triton",
        "triton_kernel_call",
        "xla.gpu.triton",
        _EXACT_ROCM_TRITON_TARGET,
    }
)
_EXPECTED_IR_CHECK_NAMES = frozenset(
    {
        "parser_count_matches_textual_custom_call_count",
        "exactly_one_custom_call_total",
        "exactly_one_pallas_call",
        "sole_exact_rocm_triton_target",
        "exact_full_forward_query_start_marker_in_sole_call",
        "no_q0_other_forward_dq_dkdv_or_lookalike_query_bounded_tokens",
        "no_outer_while",
        "preserved_query_metadata_is_exact",
    }
)
_CHECKED_CAPABILITY_TOKEN = object()
_JOURNAL_STAGES = (
    "after_backend_initialization_attempt",
    "after_chunk_lower_attempt",
    "after_chunk_compile_attempt",
    "after_host_reference_construction",
    "after_explicit_input_device_put_attempt",
    "after_candidate_dispatch_attempt",
    "after_candidate_device_get_attempt",
    "after_host_validation",
)
_COMPILE_GPU_WORK_CAVEAT = (
    "lowered.compile may dispatch bounded GPU autotuning/profiling work; the "
    "compiled executable remains inaccessible until both parameterized structural "
    "and exact compiled-memory gates pass"
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _emit(record: dict[str, Any], output: TextIO) -> None:
    output.write(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    output.flush()


def _redacted_text_summary(text: str) -> dict[str, Any]:
    encoded = text.encode("utf-8", errors="replace")
    return {
        "text_redacted": True,
        "utf8_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _redacted_message_summary(error: BaseException) -> dict[str, Any]:
    summary = _redacted_text_summary(str(error))
    return {
        "message_redacted": True,
        "message_utf8_bytes": summary["utf8_bytes"],
        "message_sha256": summary["sha256"],
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
    }


def _completed_counters() -> dict[str, int]:
    return {
        "lower_attempts": 1,
        "lower_completions": 1,
        "compile_attempts": 1,
        "compile_completions": 1,
        "input_device_put_attempts": 1,
        "input_device_put_completions": 1,
        "candidate_attempts": 1,
        "candidate_completions": 1,
        "device_get_attempts": 1,
        "device_get_completions": 1,
        "lowered_callable_invocations": 0,
    }


def _source_files() -> dict[str, Path]:
    repo = Path(__file__).resolve().parent.parent
    return {
        "probe_source_sha256": Path(__file__),
        "delegated_chunk_compile_probe_source_sha256": repo
        / "rocm"
        / "probe_query_bounded_gqa_chunk_compile.py",
        "delegated_replay_probe_source_sha256": repo
        / "rocm"
        / "probe_query_bounded_gqa_replay.py",
        "delegated_runtime_probe_source_sha256": repo
        / "rocm"
        / "probe_query_bounded_gqa_runtime.py",
        "delegated_compile_probe_source_sha256": repo
        / "rocm"
        / "probe_query_bounded_gqa_compile.py",
        "delegated_environment_probe_source_sha256": repo
        / "rocm"
        / "probe_pallas_attention.py",
        "delegated_safety_helper_source_sha256": repo / "rocm" / "amdgpu_safety.py",
        "query_bounded_gqa_kernel_source_sha256": repo
        / "skyrl"
        / "tx"
        / "kernels"
        / "query_bounded_gqa.py",
    }


def _source_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in _source_files().items()
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_argument_bytes(sequence_length: int) -> int:
    if sequence_length not in _ALLOWED_SEQUENCE_LENGTHS:
        raise ValueError("sequence length is outside the exact admitted ladder")
    return _EXPECTED_OUTPUT_BYTES + sequence_length * 4100


def _expected_marker(sequence_length: int) -> str:
    if sequence_length not in _ALLOWED_SEQUENCE_LENGTHS:
        raise ValueError("sequence length is outside the exact admitted ladder")
    return f"{_MARKER_PREFIX}{sequence_length - _QUERY_SIZE}"


def _exact_contract(sequence_length: int) -> dict[str, Any]:
    expected_argument_bytes = _expected_argument_bytes(sequence_length)
    expected_marker = _expected_marker(sequence_length)
    query_start = sequence_length - _QUERY_SIZE
    query_stop = sequence_length
    q_shape = [_BATCH_SIZE, _QUERY_SIZE, _QUERY_HEADS, _HEAD_DIM]
    kv_shape = [_BATCH_SIZE, sequence_length, _KV_HEADS, _HEAD_DIM]
    return {
        "model_family": "Qwen/Qwen3.5-4B",
        "operation": "query_bounded_gqa_forward_chunk_length_ladder",
        "inputs": [
            {
                "name": "q_chunk",
                "shape": q_shape,
                "dtype": "bfloat16",
                "value": "all_zeros",
            },
            {"name": "k", "shape": kv_shape, "dtype": "bfloat16", "value": "all_zeros"},
            {
                "name": "v",
                "shape": kv_shape,
                "dtype": "bfloat16",
                "value": "deterministic_high_contrast_nonlinear_varying_position_kv_head_and_head_dimension",
            },
            {
                "name": "key_mask",
                "shape": [_BATCH_SIZE, sequence_length],
                "dtype": "int32",
                "value": "all_ones",
            },
        ],
        "output": {"shape": q_shape, "dtype": "bfloat16"},
        "sequence_length": sequence_length,
        "query_start": query_start,
        "query_stop": query_stop,
        "global_query_positions": [query_start, query_stop - 1],
        "scale": _HEAD_DIM**-0.5,
        "tiles": {"block_q": _BLOCK_Q, "block_k": _BLOCK_K},
        "interpret": False,
        "compile_gate": {
            "parameterized_independent_parser": True,
            "dialects_independently_required": ["stablehlo", "optimized_hlo"],
            "exact_argument_bytes": expected_argument_bytes,
            "exact_output_bytes": _EXPECTED_OUTPUT_BYTES,
            "maximum_temporary_bytes": _MAX_TEMP_BYTES,
            "exact_kernel_marker": expected_marker,
        },
        "dispatch_plan": {
            "lower_calls": 1,
            "compile_calls": 1,
            "candidate_invocations": 1,
            "device_put_calls": 1,
            "device_get_calls": 1,
            "replay_invocations": 0,
            "backward_invocations": 0,
            "gpu_reference_invocations": 0,
            "device_error_reduction_invocations": 0,
            "model_invocations": 0,
        },
        "reference": {
            "location": "host_numpy_only",
            "algorithm": "FP32 global causal prefix mean of BF16 V",
            "v_construction": (
                "deterministic high-contrast non-affine BF16 values varying "
                "across position, KV head, and head dimension"
            ),
            "query_slice": [query_start, query_stop],
            "query_to_kv_head": "query_head // 4",
        },
        "numerical_gate": {
            "finite_required": True,
            "relative_l2_strictly_below": _MAX_RELATIVE_L2,
            "minimum_cosine": _MIN_COSINE,
            "maximum_absolute_error": _MAX_ABSOLUTE_ERROR,
            "candidate_seconds_strictly_below": _MAX_DISPATCH_SECONDS,
            "promotion_candidate_seconds_strictly_below": _MAX_PROMOTION_DISPATCH_SECONDS,
        },
        "outer_supervision": {
            "profile_rocm_required": True,
            "operational_dependency": True,
            "internally_proven_by_child": False,
        },
        "analytic_scope": {
            "q_and_k_are_zero": True,
            "validates_global_causal_offset_and_kv_head_mapping": True,
            "validates_nonconstant_v_accumulation": True,
            "validates_nonzero_qk_logits": False,
            "validates_attention_scale": False,
            "validates_general_forward_inputs": False,
        },
        "ladder_policy": {
            "allowed_sequence_lengths": list(_ALLOWED_SEQUENCE_LENGTHS),
            "one_length_per_fresh_process": True,
            "sequential_promotion_required": True,
            "this_probe_cannot_authorize_skipping_smaller_rungs": True,
        },
    }


def _abstract_contract() -> dict[str, Any]:
    return {
        "operation": "query_bounded_gqa_forward_chunk_length_ladder_refusal",
        "query_size": _QUERY_SIZE,
        "allowed_sequence_lengths": list(_ALLOWED_SEQUENCE_LENGTHS),
        "one_length_per_fresh_process": True,
        "sequential_promotion_required": True,
        "this_probe_cannot_authorize_skipping_smaller_rungs": True,
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
        help="acknowledge one guarded compile and one checked candidate invocation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="exclusive mode-0600 JSONL artifact (required for ROCm)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        choices=_ALLOWED_SEQUENCE_LENGTHS,
        help="exactly one admitted ladder rung; required only for guarded ROCm",
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
    if args.platform == "rocm" and args.sequence_length is None:
        parser.error("--platform rocm requires --sequence-length")
    if args.platform == "abstract" and args.sequence_length is not None:
        parser.error("--sequence-length is only valid with --platform rocm")
    if args.output is not None and args.output.exists():
        parser.error("refusing to overwrite existing output")
    return args


def _compile_probe() -> Any:
    try:
        from rocm import probe_query_bounded_gqa_chunk_compile
    except ModuleNotFoundError:
        import probe_query_bounded_gqa_chunk_compile  # type: ignore[no-redef]

    return probe_query_bounded_gqa_chunk_compile


def _assert_static_source_bindings() -> dict[str, Any]:
    helper = _compile_probe()
    helper_path = _source_files()[
        "delegated_chunk_compile_probe_source_sha256"
    ].resolve()
    raw_helper_path = getattr(helper, "__file__", None)
    if (
        not isinstance(raw_helper_path, str)
        or Path(raw_helper_path).resolve() != helper_path
    ):
        raise RuntimeError(
            "compile helper did not resolve to the exact repository source"
        )
    helper_sha256 = _file_sha256(helper_path)
    if helper_sha256 != _EXPECTED_COMPILE_HELPER_SOURCE_SHA256:
        raise RuntimeError(
            "compile helper source SHA256 differs from the audited source"
        )

    kernel_path = _source_files()["query_bounded_gqa_kernel_source_sha256"].resolve()
    kernel_sha256 = _file_sha256(kernel_path)
    if kernel_sha256 != _EXPECTED_KERNEL_SOURCE_SHA256:
        raise RuntimeError("query kernel source SHA256 differs from the audited source")
    return {
        "passed": True,
        "compile_helper_resolved_file_matches_expected": True,
        "compile_helper_source_sha256": helper_sha256,
        "query_kernel_source_sha256": kernel_sha256,
    }


def _configure_rocm_environment() -> dict[str, str | None]:
    return _compile_probe()._configure_rocm_environment()


def _environment_manifest(environment: dict[str, str | None]) -> dict[str, Any]:
    return _compile_probe()._environment_manifest(environment)


def _prove_command_buffers_disabled(
    environment: dict[str, str | None],
) -> dict[str, Any]:
    return _compile_probe()._prove_command_buffers_disabled(environment)


def _load_safety_helpers() -> tuple[
    Callable[[], ContextManager[dict[str, Any]]],
    Callable[[], dict[str, Any]],
]:
    return _compile_probe()._load_safety_helpers()


def _open_exclusive_output(path: Path) -> TextIO:
    return _compile_probe()._open_exclusive_output(path)


def _metadata_definitions(text: str) -> dict[str, str]:
    definitions: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*(#[A-Za-z_][\w.-]*)\s*=\s*(.*)$", line)
        if match is not None:
            definitions[match.group(1)] = match.group(2)
    return definitions


def _resolved_block_metadata(block: str, definitions: dict[str, str]) -> str:
    pieces = [block]
    pending = list(re.findall(r"#[A-Za-z_][\w.-]*", block))
    visited: set[str] = set()
    while pending:
        reference = pending.pop()
        if reference in visited or reference not in definitions:
            continue
        visited.add(reference)
        definition = definitions[reference]
        pieces.append(f"{reference} = {definition}")
        pending.extend(re.findall(r"#[A-Za-z_][\w.-]*", definition))
    return "\n".join(pieces)


def _custom_call_blocks(text: str, dialect: str) -> list[str]:
    lines = text.splitlines()
    if dialect == "stablehlo":
        start_pattern = re.compile(
            r"^(?P<indent>\s*)%[^=]+?=\s*stablehlo\.custom_call\b"
        )
        boundary_pattern = re.compile(
            r"^\s*(?:%[^=]+?=|#[A-Za-z_][\w.-]*\s*=|(?:stablehlo\.|func\.)?return\b|}\s*$)"
        )
    elif dialect == "optimized_hlo":
        start_pattern = re.compile(
            r"^(?P<indent>\s*)(?:ROOT\s+)?[^=]+?=\s*.*\bcustom-call\("
        )
        boundary_pattern = re.compile(r"^\s*(?:(?:ROOT\s+)?[^=]+?=|}\s*$)")
    else:
        raise ValueError("unsupported IR dialect")
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        start = start_pattern.search(lines[index])
        if start is None:
            index += 1
            continue
        base_indent = len(start.group("indent").expandtabs())
        block_lines = [lines[index]]
        index += 1
        while index < len(lines):
            candidate = lines[index]
            candidate_indent = len(candidate) - len(candidate.lstrip(" \t"))
            if (
                candidate.strip()
                and candidate_indent <= base_indent
                and boundary_pattern.match(candidate)
            ):
                break
            block_lines.append(candidate)
            index += 1
        blocks.append("\n".join(block_lines))
    return blocks


def _custom_call_targets(block: str, dialect: str) -> set[str]:
    targets = set(
        re.findall(r'(?:call_target_name|custom_call_target)\s*=\s*"([^"]+)"', block)
    )
    if dialect == "stablehlo":
        for match in re.finditer(
            r'\bstablehlo\.custom_call\s+@(?:"([^"]+)"|([A-Za-z0-9_.$-]+))', block
        ):
            targets.add(match.group(1) or match.group(2))
    elif dialect != "optimized_hlo":
        raise ValueError("unsupported IR dialect")
    return targets


def _decode_mlir_hex_escapes(text: str) -> str:
    return re.sub(
        r"\\([0-9A-Fa-f]{2})", lambda match: chr(int(match.group(1), 16)), text
    ).replace(r"\"", '"')


def _metadata_field(block: str, name: str, expected: int) -> dict[str, Any]:
    normalized = _decode_mlir_hex_escapes(block)
    tokens = _IR_NAME_TOKEN_PATTERN.findall(normalized)
    token_occurrences = sum(token == name for token in tokens)
    lookalikes = [token for token in tokens if token != name and name in token]
    pattern = re.compile(
        rf'(?<![A-Za-z0-9_.$-])"?{re.escape(name)}"?(?![A-Za-z0-9_.$-])\s*(?:=|:)\s*"?(-?[0-9]+)"?'
    )
    parsed = [int(value) for value in pattern.findall(normalized)]
    return {
        "expected": expected,
        "token_occurrences": token_occurrences,
        "lookalike_token_occurrences": len(lookalikes),
        "parsed_occurrences": len(parsed),
        "all_occurrences_parsed": token_occurrences == len(parsed),
        "values_match_expected": bool(parsed)
        and all(value == expected for value in parsed),
        "exact_if_present": token_occurrences >= 1
        and token_occurrences == len(parsed)
        and all(value == expected for value in parsed)
        and not lookalikes,
        "exact_single_occurrence": token_occurrences == 1
        and len(parsed) == 1
        and parsed[0] == expected
        and not lookalikes,
        "cleanly_absent": token_occurrences == 0 and not parsed and not lookalikes,
    }


def _is_query_bounded_kernel_marker_like(token: str) -> bool:
    normalized = re.sub(r"[-.$]+", "_", token.lower())
    return any(
        stem in normalized
        for stem in (
            "query_bounded_gqa_forward_q",
            "query_bounded_gqa_dq_q",
            "query_bounded_gqa_dkdv_q",
        )
    )


def _summarize_ir(text: str, dialect: str, sequence_length: int) -> dict[str, Any]:
    query_start = sequence_length - _QUERY_SIZE
    expected_marker = _expected_marker(sequence_length)
    if dialect == "stablehlo":
        custom_call_token_count = len(re.findall(r"\bstablehlo\.custom_call\b", text))
        while_count = len(re.findall(r"\bstablehlo\.while\b", text))
    elif dialect == "optimized_hlo":
        custom_call_token_count = len(re.findall(r"\bcustom-call\(", text))
        while_count = len(re.findall(r"\bwhile\s*\(", text))
    else:
        raise ValueError("unsupported IR dialect")
    definitions = _metadata_definitions(text)
    raw_blocks = _custom_call_blocks(text, dialect)
    blocks = [_resolved_block_metadata(block, definitions) for block in raw_blocks]
    targets = [_custom_call_targets(block, dialect) for block in blocks]
    pallas_indices = [
        i for i, item in enumerate(targets) if item & _PALLAS_TRITON_TARGETS
    ]
    sole_block = blocks[0] if len(blocks) == 1 else ""
    sole_targets = targets[0] if len(targets) == 1 else set()
    sole_tokens = _IR_NAME_TOKEN_PATTERN.findall(_decode_mlir_hex_escapes(sole_block))
    all_tokens = _IR_NAME_TOKEN_PATTERN.findall(_decode_mlir_hex_escapes(text))
    bounded = [
        token for token in all_tokens if _is_query_bounded_kernel_marker_like(token)
    ]
    unexpected = [token for token in bounded if token != expected_marker]
    start_metadata = _metadata_field(sole_block, "query_start", query_start)
    size_metadata = _metadata_field(sole_block, "query_size", _QUERY_SIZE)
    metadata_absent = (
        start_metadata["cleanly_absent"] and size_metadata["cleanly_absent"]
    )
    metadata_exact = metadata_absent or (
        start_metadata["exact_if_present"] and size_metadata["exact_if_present"]
    )
    checks = {
        "parser_count_matches_textual_custom_call_count": len(raw_blocks)
        == custom_call_token_count,
        "exactly_one_custom_call_total": len(raw_blocks) == 1
        and custom_call_token_count == 1,
        "exactly_one_pallas_call": len(pallas_indices) == 1,
        "sole_exact_rocm_triton_target": sole_targets == {_EXACT_ROCM_TRITON_TARGET},
        "exact_full_forward_query_start_marker_in_sole_call": expected_marker
        in sole_tokens,
        "no_q0_other_forward_dq_dkdv_or_lookalike_query_bounded_tokens": not unexpected,
        "no_outer_while": while_count == 0,
        "preserved_query_metadata_is_exact": metadata_exact,
    }
    return {
        "dialect": dialect,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "characters": len(text),
        "lines": len(text.splitlines()),
        "raw_ir_emitted": False,
        "custom_call_count": len(raw_blocks),
        "custom_call_token_count": custom_call_token_count,
        "pallas_custom_call_count": len(pallas_indices),
        "expected_marker": expected_marker,
        "expected_marker_occurrences": sum(
            token == expected_marker for token in all_tokens
        ),
        "unexpected_query_bounded_token_occurrences": len(unexpected),
        "while_count": while_count,
        "metadata": {
            "preserved": not metadata_absent,
            "query_start": start_metadata,
            "query_size": size_metadata,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _structural_gate(*summaries: dict[str, Any]) -> dict[str, Any]:
    dialects = [str(summary.get("dialect")) for summary in summaries]
    exact_dialects = sorted(dialects) == ["optimized_hlo", "stablehlo"]
    per_dialect: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        dialect = str(summary.get("dialect"))
        raw_checks = summary.get("checks")
        checks = dict(raw_checks) if isinstance(raw_checks, dict) else {}
        passed = (
            set(checks) == _EXPECTED_IR_CHECK_NAMES
            and all(value is True for value in checks.values())
            and summary.get("passed") is True
        )
        per_dialect[dialect] = {"checks": checks, "passed": passed}
    passed = (
        exact_dialects
        and per_dialect.get("stablehlo", {}).get("passed") is True
        and per_dialect.get("optimized_hlo", {}).get("passed") is True
    )
    return {
        "checks": {
            "exactly_one_independent_summary_per_required_dialect": exact_dialects,
            "stablehlo_passed": exact_dialects
            and per_dialect.get("stablehlo", {}).get("passed") is True,
            "optimized_hlo_passed": exact_dialects
            and per_dialect.get("optimized_hlo", {}).get("passed") is True,
        },
        "per_dialect": per_dialect,
        "passed": passed,
    }


def _compiled_memory(compiled: Any) -> dict[str, Any]:
    return _compile_probe()._compiled_memory(compiled)


def _compiled_memory_gate(
    memory: dict[str, Any], sequence_length: int
) -> dict[str, Any]:
    expected_argument = _expected_argument_bytes(sequence_length)
    required = ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes")
    values = {name: memory.get(name) for name in required}
    available = memory.get("available") is True and all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in values.values()
    )
    argument = values["argument_size_in_bytes"] if available else None
    result = values["output_size_in_bytes"] if available else None
    temporary = values["temp_size_in_bytes"] if available else None
    alias = memory.get("alias_size_in_bytes")
    alias_exact = isinstance(alias, int) and not isinstance(alias, bool) and alias == 0
    checks = {
        "memory_analysis_available": available,
        "argument_bytes_exact_formula": argument == expected_argument,
        "output_bytes_exactly_2097152": result == _EXPECTED_OUTPUT_BYTES,
        "temporary_bytes_at_most_64_mib": temporary is not None
        and temporary <= _MAX_TEMP_BYTES,
        "alias_bytes_exactly_zero": alias_exact,
    }
    combined = argument + result + temporary if available else None
    return {
        "expected_argument_bytes": expected_argument,
        "expected_output_bytes": _EXPECTED_OUTPUT_BYTES,
        "expected_alias_bytes": 0,
        "maximum_temporary_bytes": _MAX_TEMP_BYTES,
        "argument_output_temporary_bytes": combined,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _assert_kernel_binding(api: Any, sequence_length: int) -> dict[str, Any]:
    expected_file = _source_files()["query_bounded_gqa_kernel_source_sha256"].resolve()
    loaded_file = inspect.getsourcefile(api)
    if loaded_file is None or Path(loaded_file).resolve() != expected_file:
        raise RuntimeError(
            "query chunk API did not bind to the exact repository kernel source"
        )
    source_sha256 = _file_sha256(expected_file)
    if source_sha256 != _EXPECTED_KERNEL_SOURCE_SHA256:
        raise RuntimeError(
            "loaded query chunk API source SHA256 differs from the audited source"
        )
    signature = inspect.signature(api)
    required = {
        "q_chunk",
        "k",
        "v",
        "key_mask",
        "query_start",
        "scale",
        "block_q",
        "block_k",
        "interpret",
    }
    parameters = signature.parameters
    if not required.issubset(parameters):
        raise RuntimeError(
            "query chunk API signature no longer preserves the audited semantics"
        )
    exact_semantics = (
        list(parameters)[:4] == ["q_chunk", "k", "v", "key_mask"]
        and parameters["query_start"].kind is inspect.Parameter.KEYWORD_ONLY
        and parameters["query_start"].default is inspect.Parameter.empty
        and parameters["scale"].default is None
        and parameters["block_q"].default == _BLOCK_Q
        and parameters["block_k"].default == _BLOCK_K
        and parameters["interpret"].default is False
    )
    if not exact_semantics:
        raise RuntimeError("query chunk API defaults or keyword-only semantics changed")
    return {
        "passed": True,
        "resolved_file_matches_expected": True,
        "source_sha256": source_sha256,
        "required_parameters": sorted(required),
        "signature_semantics_exact": True,
        "exact_marker": _expected_marker(sequence_length),
        "exact_target": _EXACT_ROCM_TRITON_TARGET,
    }


def _assert_fresh_accelerator_process() -> None:
    imported = sorted(
        name
        for name in sys.modules
        if name in {"jax", "jaxlib"}
        or name.startswith("jax.")
        or name.startswith("jaxlib.")
        or name == "skyrl.tx.kernels.query_bounded_gqa"
    )
    if imported:
        raise RuntimeError(
            "ROCm chunk runtime gate requires a fresh process before JAX/kernel import"
        )


def _public_clean_safety(safety: Any, stage: str) -> dict[str, Any]:
    if not isinstance(safety, dict) or safety.get("amdgpu_boot_clean") is not True:
        raise RuntimeError(f"{stage} did not prove a clean AMDGPU boot")
    if safety.get("fatal_amdgpu_events") != []:
        raise RuntimeError(f"{stage} returned invalid fatal-event evidence")
    return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}


def _public_safety_preflight(safety: Any) -> dict[str, Any]:
    public = _public_clean_safety(safety, "safety_preflight")
    amd_cards = safety.get("amd_cards")
    if (
        not isinstance(amd_cards, list)
        or not amd_cards
        or not all(
            isinstance(card, str) and re.fullmatch(r"card[0-9]+", card)
            for card in amd_cards
        )
        or amd_cards != sorted(set(amd_cards))
    ):
        raise RuntimeError("safety_preflight returned invalid AMD DRM card evidence")
    if safety.get("connected_amd_connectors") != []:
        raise RuntimeError("safety_preflight did not prove every AMD connector idle")
    if safety.get("kfd_path") != "/dev/kfd":
        raise RuntimeError("safety_preflight did not prove the exact KFD device")
    if (
        safety.get("kfd_accessible") is not True
        or safety.get("kfd_unowned") is not True
    ):
        raise RuntimeError("safety_preflight did not prove accessible unowned KFD")
    return {
        **public,
        "amd_cards": list(amd_cards),
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
) -> dict[str, Any]:
    if stage not in _JOURNAL_STAGES:
        raise RuntimeError("refusing an undeclared chunk-runtime journal stage")
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


class _CheckedChunkExecutable:
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
            raise RuntimeError(
                "refusing to expose a chunk executable without both passed compile gates"
            )
        self._compiled = compiled
        self._consumed = False
        self._counters = counters
        self.proof = proof

    def invoke(
        self, jax: Any, arguments: tuple[Any, ...], on_started: Callable[[], None]
    ) -> Any:
        if self._consumed:
            raise RuntimeError("checked chunk capability was already consumed")
        self._consumed = True
        self._counters["candidate_attempts"] += 1
        on_started()
        result = self._compiled(*arguments)
        result = jax.block_until_ready(result)
        self._counters["candidate_completions"] += 1
        return result


def _wrap_checked(
    compiled: Any, proof: dict[str, Any], counters: dict[str, int]
) -> _CheckedChunkExecutable:
    return _CheckedChunkExecutable(
        compiled,
        proof=proof,
        counters=counters,
        token=_CHECKED_CAPABILITY_TOKEN,
    )


def _shape_signature(
    jax: Any, jnp: Any, sequence_length: int
) -> tuple[Any, Any, Any, Any]:
    return (
        jax.ShapeDtypeStruct(
            (_BATCH_SIZE, _QUERY_SIZE, _QUERY_HEADS, _HEAD_DIM), jnp.bfloat16
        ),
        jax.ShapeDtypeStruct(
            (_BATCH_SIZE, sequence_length, _KV_HEADS, _HEAD_DIM), jnp.bfloat16
        ),
        jax.ShapeDtypeStruct(
            (_BATCH_SIZE, sequence_length, _KV_HEADS, _HEAD_DIM), jnp.bfloat16
        ),
        jax.ShapeDtypeStruct((_BATCH_SIZE, sequence_length), jnp.int32),
    )


def _compile_checked_chunk(
    jax: Any,
    jnp: Any,
    query_bounded_gqa_forward_chunk: Any,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    output: TextIO,
    sequence_length: int,
) -> tuple[_CheckedChunkExecutable, dict[str, Any]]:
    query_start = sequence_length - _QUERY_SIZE

    def forward_chunk(q_arg: Any, k_arg: Any, v_arg: Any, mask_arg: Any) -> Any:
        return query_bounded_gqa_forward_chunk(
            q_arg,
            k_arg,
            v_arg,
            mask_arg,
            query_start=query_start,
            block_q=_BLOCK_Q,
            block_k=_BLOCK_K,
            interpret=False,
        )

    _emit(
        {
            "record_type": "stage",
            "stage": "chunk_lower_started",
            "timestamp": _utc_now(),
            "counters": dict(counters),
        },
        output,
    )
    counters["lower_attempts"] += 1
    lower_start = time.perf_counter()
    try:
        lowered = jax.jit(forward_chunk).lower(
            *_shape_signature(jax, jnp, sequence_length)
        )
        lower_seconds = time.perf_counter() - lower_start
        counters["lower_completions"] += 1
        stablehlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
        stablehlo = _summarize_ir(stablehlo_text, "stablehlo", sequence_length)
        del stablehlo_text
        _emit(
            {
                "record_type": "lowered",
                "stage": "chunk_lower_complete_metadata_only",
                "timestamp": _utc_now(),
                "lower_seconds": lower_seconds,
                "stablehlo": stablehlo,
                "raw_ir_emitted": False,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_chunk_lower_attempt", counters
        )

    _emit(
        {
            "record_type": "stage",
            "stage": "chunk_compile_started",
            "timestamp": _utc_now(),
            "counters": dict(counters),
        },
        output,
    )
    counters["compile_attempts"] += 1
    compile_start = time.perf_counter()
    compiled = None
    release = False
    try:
        compiled = lowered.compile()
        compile_seconds = time.perf_counter() - compile_start
        counters["compile_completions"] += 1
        optimized_hlo_text = compiled.as_text()
        optimized_hlo = _summarize_ir(
            optimized_hlo_text, "optimized_hlo", sequence_length
        )
        del optimized_hlo_text
        memory = _compiled_memory(compiled)
        structural = _structural_gate(stablehlo, optimized_hlo)
        memory_gate = _compiled_memory_gate(memory, sequence_length)
        proof = {
            "structural_gate_passed": structural["passed"],
            "compiled_memory_gate_passed": memory_gate["passed"],
            "passed": structural["passed"] and memory_gate["passed"],
        }
        report = {
            "record_type": "chunk_compiled",
            "stage": "chunk_compile_gate",
            "exact_kernel_marker": _expected_marker(sequence_length),
            "timestamp": _utc_now(),
            "lower_seconds": lower_seconds,
            "compile_seconds": compile_seconds,
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
                "chunk executable failed the parameterized structural or exact memory gate"
            )
        checked = _wrap_checked(compiled, proof, counters)
        release = True
    finally:
        if compiled is not None and not release:
            del compiled
        _journal_checkpoint(
            require_clean_boot, output, "after_chunk_compile_attempt", counters
        )
    del lowered
    _emit(
        {
            "record_type": "compile_gate_passed",
            "timestamp": _utc_now(),
            "status": "checked_one_shot_capability_created",
            "release_gate": report["release_gate"],
            "counters": dict(counters),
        },
        output,
    )
    return checked, report


def _array_manifest(name: str, value: Any) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "nbytes": int(value.nbytes),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _construct_host_inputs(
    np: Any, ml_dtypes: Any, sequence_length: int
) -> tuple[tuple[Any, ...], list[dict[str, Any]], Any, dict[str, Any]]:
    query_start = sequence_length - _QUERY_SIZE
    q = np.zeros(
        (_BATCH_SIZE, _QUERY_SIZE, _QUERY_HEADS, _HEAD_DIM), dtype=ml_dtypes.bfloat16
    )
    k = np.zeros(
        (_BATCH_SIZE, sequence_length, _KV_HEADS, _HEAD_DIM), dtype=ml_dtypes.bfloat16
    )
    positions = np.arange(sequence_length, dtype=np.int32)[:, None, None]
    heads = np.arange(_KV_HEADS, dtype=np.int32)[None, :, None]
    features = np.arange(_HEAD_DIM, dtype=np.int32)[None, None, :]
    checker = np.where(
        ((positions + 3 * heads + features) % 2) == 0,
        np.float32(-8.0),
        np.float32(8.0),
    )
    nonlinear_grid = (
        (
            positions * 37
            + heads * 101
            + features * 29
            + (positions % 13) * (features % 17) * 7
            + (positions % 11) * (heads + 1) * 19
        )
        % 257
        - 128
    ).astype(np.float32) / np.float32(64.0)
    head_bias = (
        heads.astype(np.float32) - np.float32((_KV_HEADS - 1) / 2)
    ) * np.float32(0.25)
    v = (checker + nonlinear_grid + head_bias)[None].astype(ml_dtypes.bfloat16)
    key_mask = np.ones((_BATCH_SIZE, sequence_length), dtype=np.int32)
    v_fp32 = np.asarray(v, dtype=np.float32)
    if (
        bool(np.any(q != 0))
        or bool(np.any(k != 0))
        or not bool(np.all(key_mask == 1))
        or not all(
            bool(np.any(v[:, :, head] != v[:, :, 0])) for head in range(1, _KV_HEADS)
        )
        or not bool(np.any(v[:, 1:] != v[:, :-1]))
        or not bool(np.any(v[:, :, :, 1:] != v[:, :, :, :-1]))
        or float(np.min(v_fp32)) > -7.0
        or float(np.max(v_fp32)) < 7.0
    ):
        raise RuntimeError(
            "analytic host construction violated the zero-QK, all-valid, or "
            "high-contrast position/head/dimension-varying V contract"
        )

    prefix_counts = np.arange(1, sequence_length + 1, dtype=np.float32)[
        None, :, None, None
    ]
    global_prefix_means = np.cumsum(v_fp32, axis=1, dtype=np.float32) / prefix_counts
    query_to_kv = np.arange(_QUERY_HEADS, dtype=np.int32) // _GROUP_SIZE
    expected = global_prefix_means[:, query_start:sequence_length, query_to_kv, :]
    if tuple(expected.shape) != tuple(q.shape):
        raise RuntimeError(
            "global-offset analytic oracle produced the wrong chunk shape"
        )
    inputs = (q, k, v, key_mask)
    input_manifests = [
        _array_manifest(name, value)
        for name, value in zip(("q_chunk", "k", "v", "key_mask"), inputs, strict=True)
    ]
    oracle_manifest = _array_manifest("global_prefix_mean_reference", expected)
    return inputs, input_manifests, expected, oracle_manifest


def _device_put_inputs(
    jax: Any, host_inputs: tuple[Any, ...], counters: dict[str, int]
) -> tuple[Any, ...]:
    counters["input_device_put_attempts"] += 1
    placed = jax.device_put(host_inputs)
    placed = jax.block_until_ready(placed)
    counters["input_device_put_completions"] += 1
    if not isinstance(placed, tuple) or len(placed) != 4:
        raise RuntimeError("explicit input device_put did not preserve the exact tuple")
    return placed


def _dispatch_candidate(
    jax: Any,
    executable: _CheckedChunkExecutable,
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
                "label": "single_last_chunk_candidate",
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
            "label": "single_last_chunk_candidate",
            "seconds": seconds,
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


def _host_metrics(np: Any, actual_host: Any, expected_host: Any) -> dict[str, Any]:
    actual_raw = np.asarray(actual_host)
    expected_raw = np.asarray(expected_host)
    actual = actual_raw.astype(np.float32)
    expected = expected_raw.astype(np.float32)
    exact_shape = (_BATCH_SIZE, _QUERY_SIZE, _QUERY_HEADS, _HEAD_DIM)
    if tuple(actual.shape) != exact_shape or tuple(expected.shape) != exact_shape:
        raise RuntimeError(
            "candidate or analytic reference returned the wrong output shape"
        )
    shape_dtype_exact = (
        str(actual_raw.dtype) == "bfloat16"
        and str(expected_raw.dtype) == "float32"
        and int(actual_raw.nbytes) == _EXPECTED_OUTPUT_BYTES
        and int(expected_raw.nbytes) == 2 * _EXPECTED_OUTPUT_BYTES
    )
    difference = actual - expected
    actual_norm = float(np.linalg.norm(actual.ravel()))
    expected_norm = float(np.linalg.norm(expected.ravel()))
    denominator = max(expected_norm, float(np.finfo(np.float32).tiny))
    cosine_denominator = max(
        actual_norm * expected_norm, float(np.finfo(np.float32).tiny)
    )
    return {
        "finite": bool(
            np.all(np.isfinite(actual))
            and np.all(np.isfinite(expected))
            and np.all(np.isfinite(difference))
        ),
        "shape_dtype_nbytes_exact": shape_dtype_exact,
        "actual_shape": list(actual_raw.shape),
        "actual_dtype": str(actual_raw.dtype),
        "reference_shape": list(expected_raw.shape),
        "reference_dtype": str(expected_raw.dtype),
        "actual_nbytes": int(actual_raw.nbytes),
        "reference_nbytes": int(expected_raw.nbytes),
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference.ravel()) / denominator),
        "cosine": float(np.vdot(actual.ravel(), expected.ravel()) / cosine_denominator),
        "actual_sha256": hashlib.sha256(actual_raw.tobytes(order="C")).hexdigest(),
        "reference_sha256": hashlib.sha256(expected_raw.tobytes(order="C")).hexdigest(),
    }


def _validate_candidate(
    np: Any,
    actual_host: Any,
    expected_host: Any,
    seconds: float,
    counters: dict[str, int],
    output: TextIO,
    sequence_length: int,
) -> dict[str, Any]:
    metrics = _host_metrics(np, actual_host, expected_host)
    numerical = (
        metrics["finite"]
        and metrics["shape_dtype_nbytes_exact"]
        and math.isfinite(metrics["relative_l2"])
        and metrics["relative_l2"] < _MAX_RELATIVE_L2
        and math.isfinite(metrics["cosine"])
        and metrics["cosine"] >= _MIN_COSINE
        and math.isfinite(metrics["max_abs"])
        and metrics["max_abs"] <= _MAX_ABSOLUTE_ERROR
    )
    safety_duration = (
        math.isfinite(seconds) and seconds >= 0 and seconds < _MAX_DISPATCH_SECONDS
    )
    promotion_duration = (
        math.isfinite(seconds)
        and seconds >= 0
        and seconds < _MAX_PROMOTION_DISPATCH_SECONDS
    )
    passed = numerical and safety_duration and promotion_duration
    record = {
        "record_type": "host_validation",
        "timestamp": _utc_now(),
        "status": (
            "passed"
            if passed
            else "not_promoted"
            if numerical and safety_duration
            else "failed"
        ),
        "global_query_positions": [sequence_length - _QUERY_SIZE, sequence_length - 1],
        "query_to_kv_head": "query_head // 4",
        "metrics": metrics,
        "thresholds": {
            "finite_required": True,
            "relative_l2_strictly_below": _MAX_RELATIVE_L2,
            "minimum_cosine": _MIN_COSINE,
            "maximum_absolute_error": _MAX_ABSOLUTE_ERROR,
            "candidate_seconds_strictly_below": _MAX_DISPATCH_SECONDS,
            "promotion_candidate_seconds_strictly_below": (
                _MAX_PROMOTION_DISPATCH_SECONDS
            ),
        },
        "gates": {
            "numerical_passed": numerical,
            "safety_duration_passed": safety_duration,
            "promotion_duration_passed": promotion_duration,
            "promotion_passed": passed,
        },
        "candidate_seconds": seconds,
        "counters": dict(counters),
    }
    _emit(record, output)
    if not passed:
        raise RuntimeError(
            "last-chunk analytic candidate failed the host numerical, safety-duration, "
            "or promotion-duration gate"
        )
    return record


def _backend_manifest(jax: Any, jaxlib: Any, jax_backend: Any) -> dict[str, Any]:
    resolved = jax.default_backend()
    platform_version = str(jax_backend.get_backend().platform_version)
    devices = jax.devices()
    if resolved != "gpu" or "rocm" not in platform_version.lower() or len(devices) != 1:
        raise RuntimeError(
            "requested ROCm but JAX did not resolve exactly one ROCm GPU"
        )
    return {
        "platform_resolved": "gpu",
        "platform_family": "rocm",
        "visible_device_count": 1,
        "jax_version": _redacted_text_summary(str(jax.__version__)),
        "jaxlib_version": _redacted_text_summary(str(jaxlib.__version__)),
        "platform_version": _redacted_text_summary(platform_version),
        "raw_device_descriptions_emitted": False,
    }


def _run_rocm(
    output: TextIO,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    *,
    environment: dict[str, str | None],
    sequence_length: int,
    _dependencies: tuple[Any, Any, Any, Any, Any, Any, Any] | None = None,
) -> int:
    proof = _prove_command_buffers_disabled(environment)
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

            from skyrl.tx.kernels.query_bounded_gqa import (
                query_bounded_gqa_forward_chunk,
            )
        else:
            (
                jax,
                jnp,
                jaxlib,
                jax_backend,
                np,
                ml_dtypes,
                query_bounded_gqa_forward_chunk,
            ) = _dependencies
        backend = _backend_manifest(jax, jaxlib, jax_backend)
        kernel_binding = _assert_kernel_binding(
            query_bounded_gqa_forward_chunk, sequence_length
        )
        _emit(
            {
                "record_type": "backend_ready",
                "timestamp": _utc_now(),
                "backend": backend,
                "kernel_binding": kernel_binding,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_backend_initialization_attempt", counters
        )

    executable, compile_report = _compile_checked_chunk(
        jax,
        jnp,
        query_bounded_gqa_forward_chunk,
        require_clean_boot,
        counters,
        output,
        sequence_length,
    )
    try:
        host_inputs, input_manifests, expected_host, oracle_manifest = (
            _construct_host_inputs(np, ml_dtypes, sequence_length)
        )
        _emit(
            {
                "record_type": "host_analytic_reference",
                "timestamp": _utc_now(),
                "construction": {
                    "q_chunk": "all BF16 zeros",
                    "k": "all BF16 zeros",
                    "v": (
                        "deterministic high-contrast non-affine BF16 values varying "
                        "across position, KV head, and head dimension"
                    ),
                    "key_mask": "all int32 ones",
                    "oracle": (
                        "independent host FP32 global prefix means sliced at "
                        f"positions {sequence_length - _QUERY_SIZE}:{sequence_length}"
                    ),
                    "randomness_used": False,
                },
                "inputs": input_manifests,
                "oracle": oracle_manifest,
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
        jax,
        executable,
        inputs,
        require_clean_boot,
        counters,
        output,
    )
    actual_host = _device_get_candidate(
        jax,
        actual,
        require_clean_boot,
        counters,
        output,
    )
    try:
        validation = _validate_candidate(
            np, actual_host, expected_host, seconds, counters, output, sequence_length
        )
    finally:
        _journal_checkpoint(
            require_clean_boot, output, "after_host_validation", counters
        )
    if counters != _completed_counters():
        raise RuntimeError(
            "last-chunk runtime counter contract was not completed exactly"
        )
    _emit(
        {
            "record_type": "runtime_passed",
            "timestamp": _utc_now(),
            "status": f"passed_exact_c256_t{sequence_length}_last_chunk_analytic_candidate",
            "sequence_length": sequence_length,
            "query_start": sequence_length - _QUERY_SIZE,
            "exact_kernel_marker": _expected_marker(sequence_length),
            "compile_release_gate": compile_report["release_gate"],
            "compiled_memory_gate": compile_report["compiled_memory_gate"],
            "host_validation": {
                "metrics": validation["metrics"],
                "gates": validation["gates"],
                "candidate_seconds": seconds,
            },
            "counters": dict(counters),
            "replay_invocations": 0,
            "backward_invocations": 0,
            "gpu_reference_invocations": 0,
            "device_error_reduction_invocations": 0,
            "model_dispatcher_connected": False,
            "sequential_promotion_required": True,
            "this_probe_cannot_authorize_skipping_smaller_rungs": True,
        },
        output,
    )
    return 0


def _execute(args: argparse.Namespace, output: TextIO) -> int:
    counters = _zero_counters()
    sequence_length = getattr(args, "sequence_length", None)
    _emit(
        {
            "record_type": "manifest",
            "timestamp": _utc_now(),
            "platform_requested": args.platform,
            "allow_gpu": args.allow_gpu,
            "scope": (
                "abstract_refusal"
                if args.platform == "abstract"
                else f"guarded_exact_c256_t{sequence_length}_last_chunk_length_ladder"
            ),
            "contract": (
                _abstract_contract()
                if args.platform == "abstract"
                else _exact_contract(sequence_length)
            ),
            "compile_may_dispatch_gpu_work": args.platform == "rocm",
            "compile_dispatch_caveat": _COMPILE_GPU_WORK_CAVEAT,
            "fresh_process_required": True,
            "prior_compile_artifact_used": False,
            "raw_ir_emitted": False,
            "outer_profile_rocm_supervision_required": True,
            "outer_profile_rocm_supervision_operational_not_internally_proven": True,
            "zero_qk_analytic_scope_does_not_validate_qk_scale_or_general_forward": True,
            "sequential_promotion_required": True,
            "this_probe_alone_cannot_authorize_skipping_smaller_rung": True,
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
                    "profile_rocm.py in a fresh process with one --sequence-length rung"
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
                "stage": "audited_sources_bound_before_environment_or_jax",
                "proof": source_binding,
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
        guarded_process, require_clean_boot = _load_safety_helpers()
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
                    sequence_length=sequence_length,
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
