#!/usr/bin/env python3
"""Fail-closed compile-only gate for the exact C256/T512 GQA last chunk.

The default ``abstract`` mode emits a refusal manifest without importing JAX.
ROCm lowering requires ``--platform rocm --allow-gpu --output`` in a fresh
process under the guarded profiler.  The only lowered operation is
``query_bounded_gqa_forward_chunk`` with a BF16 ``[1,256,16,256]`` query chunk,
BF16 ``[1,512,4,256]`` K/V, an int32 ``[1,512]`` key mask, and
``query_start=256``.  Lowering and compilation each happen once.  Compilation
may dispatch GPU autotuning/profiling work, but the compiled executable is
never invoked, exposed, or returned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import operator
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, ContextManager, TextIO

_BATCH_SIZE = 1
_QUERY_SIZE = 256
_SEQUENCE_LENGTH = 512
_QUERY_HEADS = 16
_KV_HEADS = 4
_HEAD_DIM = 256
_QUERY_START = 256
_BLOCK_Q = 64
_BLOCK_K = 64
_DTYPE = "bfloat16"
_MASK_DTYPE = "int32"
_EXPECTED_MARKER = "query_bounded_gqa_forward_q256"
_EXACT_ROCM_TRITON_TARGET = "__gpu$xla.gpu.triton"
_EXPECTED_ARGUMENT_BYTES = 4_196_352
_EXPECTED_OUTPUT_BYTES = 2_097_152
_MAX_TEMP_BYTES = 64 * 1024**2
_IR_NAME_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.$-]*")
_IR_CHECK_NAMES = frozenset(
    {
        "parser_count_matches_textual_custom_call_count",
        "exactly_one_custom_call_total",
        "exactly_one_pallas_call",
        "sole_exact_rocm_triton_target",
        "exact_full_forward_q256_marker_in_sole_call",
        "no_q0_other_forward_dq_dkdv_or_lookalike_query_bounded_tokens",
        "all_query_bounded_marker_occurrences_belong_to_sole_call",
        "no_outer_while",
        "preserved_query_metadata_is_exact",
    }
)
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
_JOURNAL_STAGES = (
    "before_backend_initialization",
    "after_backend_initialization_attempt",
    "after_chunk_lower_attempt",
    "after_chunk_compile_attempt",
)
_COMPILE_GPU_WORK_CAVEAT = (
    "lowered.compile may dispatch bounded GPU autotuning/profiling kernels and "
    "allocate compiler-managed buffers; no returned executable invocation is "
    "authorized by this compile-only gate"
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _emit(record: dict[str, Any], output: TextIO) -> None:
    output.write(json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n")
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
        "forward_attempts": 0,
        "forward_completions": 0,
        "lowered_callable_invocations": 0,
        "compiled_executable_invocations": 0,
    }


def _assert_zero_counters(counters: dict[str, int]) -> None:
    expected = _zero_counters()
    if counters != expected:
        raise RuntimeError("compile-only zero-invocation counter contract was violated")


def _source_files() -> dict[str, Path]:
    repo = Path(__file__).resolve().parent.parent
    return {
        "probe_source_sha256": Path(__file__),
        "delegated_replay_probe_source_sha256": repo / "rocm" / "probe_query_bounded_gqa_replay.py",
        "delegated_runtime_probe_source_sha256": repo / "rocm" / "probe_query_bounded_gqa_runtime.py",
        "delegated_compile_probe_source_sha256": repo / "rocm" / "probe_query_bounded_gqa_compile.py",
        "delegated_environment_probe_source_sha256": repo / "rocm" / "probe_pallas_attention.py",
        "delegated_safety_helper_source_sha256": repo / "rocm" / "amdgpu_safety.py",
        "query_bounded_gqa_kernel_source_sha256": repo / "skyrl" / "tx" / "kernels" / "query_bounded_gqa.py",
    }


def _source_hashes() -> dict[str, str]:
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in _source_files().items()}


def _exact_contract() -> dict[str, Any]:
    q_shape = [_BATCH_SIZE, _QUERY_SIZE, _QUERY_HEADS, _HEAD_DIM]
    kv_shape = [_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM]
    return {
        "model_family": "Qwen/Qwen3.5-4B",
        "operation": "query_bounded_gqa_forward_chunk_compile_only",
        "inputs": [
            {"name": "q_chunk", "shape": q_shape, "dtype": _DTYPE},
            {"name": "k", "shape": kv_shape, "dtype": _DTYPE},
            {"name": "v", "shape": kv_shape, "dtype": _DTYPE},
            {
                "name": "key_mask",
                "shape": [_BATCH_SIZE, _SEQUENCE_LENGTH],
                "dtype": _MASK_DTYPE,
            },
        ],
        "output": {"shape": q_shape, "dtype": _DTYPE},
        "api": "query_bounded_gqa_forward_chunk",
        "query_start": _QUERY_START,
        "query_stop": _QUERY_START + _QUERY_SIZE,
        "sequence_length": _SEQUENCE_LENGTH,
        "scale": _HEAD_DIM**-0.5,
        "tiles": {"block_q": _BLOCK_Q, "block_k": _BLOCK_K},
        "interpret": False,
        "compile_plan": {
            "lower_calls": 1,
            "compile_calls": 1,
            "compiled_executable_invocations": 0,
            "lowered_callable_invocations": 0,
            "executable_returned": False,
        },
        "compiled_ir_gate": {
            "dialects": ["stablehlo", "optimized_hlo"],
            "custom_calls_per_dialect": 1,
            "pallas_calls_per_dialect": 1,
            "sole_target": _EXACT_ROCM_TRITON_TARGET,
            "exact_marker": _EXPECTED_MARKER,
            "outer_while_calls": 0,
            "preserved_metadata": {
                "query_start": _QUERY_START,
                "query_size": _QUERY_SIZE,
            },
        },
        "compiled_memory_gate": {
            "memory_analysis_required": True,
            "exact_argument_bytes": _EXPECTED_ARGUMENT_BYTES,
            "exact_output_bytes": _EXPECTED_OUTPUT_BYTES,
            "maximum_temporary_bytes": _MAX_TEMP_BYTES,
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=("abstract", "rocm"),
        default="abstract",
        help=(
            "refusal-only by default; guarded ROCm lowering/compilation is GPU "
            "work and requires rocm plus --allow-gpu"
        ),
    )
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help="acknowledge that compile() may dispatch bounded GPU profiling work",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="exclusive mode-0600 JSONL artifact (required for ROCm)",
    )
    args = parser.parse_args(argv)
    if args.platform == "rocm" and not args.allow_gpu:
        parser.error("--platform rocm requires the explicit --allow-gpu acknowledgement")
    if args.platform == "abstract" and args.allow_gpu:
        parser.error("--allow-gpu is only valid with --platform rocm")
    if args.platform == "rocm" and args.output is None:
        parser.error("--platform rocm requires --output for a private JSONL artifact")
    if args.output is not None and args.output.exists():
        parser.error("refusing to overwrite existing output")
    return args


def _replay_probe() -> Any:
    try:
        from rocm import probe_query_bounded_gqa_replay
    except ModuleNotFoundError:
        import probe_query_bounded_gqa_replay  # type: ignore[no-redef]

    return probe_query_bounded_gqa_replay


def _configure_rocm_environment() -> dict[str, str | None]:
    return _replay_probe()._configure_rocm_environment()


def _environment_manifest(environment: dict[str, str | None]) -> dict[str, Any]:
    return _replay_probe()._environment_manifest(environment)


def _prove_command_buffers_disabled(
    environment: dict[str, str | None],
) -> dict[str, Any]:
    return _replay_probe()._prove_command_buffers_disabled(environment)


def _load_safety_helpers() -> tuple[
    Callable[[], ContextManager[dict[str, Any]]],
    Callable[[], dict[str, Any]],
]:
    return _replay_probe()._load_safety_helpers()


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
        raise RuntimeError("ROCm chunk compile gate requires a fresh process before JAX/kernel import")


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


def _public_clean_safety(safety: Any, stage: str) -> dict[str, Any]:
    if not isinstance(safety, dict) or safety.get("amdgpu_boot_clean") is not True:
        raise RuntimeError(f"{stage} did not prove a clean AMDGPU boot")
    if safety.get("fatal_amdgpu_events") != []:
        raise RuntimeError(f"{stage} returned an invalid fatal-event proof")
    return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}


def _public_safety_preflight(safety: Any) -> dict[str, Any]:
    public = _public_clean_safety(safety, "safety_preflight")
    amd_cards = safety.get("amd_cards")
    if (
        not isinstance(amd_cards, list)
        or not amd_cards
        or not all(isinstance(card, str) and re.fullmatch(r"card[0-9]+", card) for card in amd_cards)
        or amd_cards != sorted(set(amd_cards))
    ):
        raise RuntimeError("safety_preflight returned invalid AMD DRM card evidence")
    if safety.get("connected_amd_connectors") != []:
        raise RuntimeError("safety_preflight did not prove every AMD connector idle")
    if safety.get("kfd_path") != "/dev/kfd":
        raise RuntimeError("safety_preflight did not prove the exact KFD device")
    if safety.get("kfd_accessible") is not True:
        raise RuntimeError("safety_preflight did not prove KFD accessibility")
    if safety.get("kfd_unowned") is not True:
        raise RuntimeError("safety_preflight did not prove KFD was unowned")
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
        raise RuntimeError("refusing an undeclared compile-gate journal stage")
    _assert_zero_counters(counters)
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
        start_pattern = re.compile(r"^(?P<indent>\s*)%[^=]+?=\s*stablehlo\.custom_call\b")
        boundary_pattern = re.compile(
            r"^\s*(?:%[^=]+?=|#[A-Za-z_][\w.-]*\s*=|" r"(?:stablehlo\.|func\.)?return\b|}\s*$)"
        )
    elif dialect == "optimized_hlo":
        start_pattern = re.compile(r"^(?P<indent>\s*)(?:ROOT\s+)?[^=]+?=\s*.*\bcustom-call\(")
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
            if candidate.strip() and candidate_indent <= base_indent and boundary_pattern.match(candidate):
                break
            block_lines.append(candidate)
            index += 1
        blocks.append("\n".join(block_lines))
    return blocks


def _custom_call_targets(block: str, dialect: str) -> set[str]:
    targets = set(
        re.findall(
            r'(?:call_target_name|custom_call_target)\s*=\s*"([^"]+)"',
            block,
        )
    )
    if dialect == "stablehlo":
        for match in re.finditer(
            r'\bstablehlo\.custom_call\s+@(?:"([^"]+)"|([A-Za-z0-9_.$-]+))',
            block,
        ):
            targets.add(match.group(1) or match.group(2))
    elif dialect != "optimized_hlo":
        raise ValueError("unsupported IR dialect")
    return targets


def _metadata_field(block: str, name: str, expected: int) -> dict[str, Any]:
    normalized = block.replace(r"\"", '"')
    tokens = _IR_NAME_TOKEN_PATTERN.findall(normalized)
    token_occurrences = sum(token == name for token in tokens)
    lookalike_tokens = [token for token in tokens if token != name and name in token]
    pattern = re.compile(rf"(?<![A-Za-z0-9_.$-]){re.escape(name)}(?![A-Za-z0-9_.$-])" r"\s*(?:=|:)\s*\"?(-?[0-9]+)\"?")
    parsed = [int(value) for value in pattern.findall(normalized)]
    return {
        "expected": expected,
        "token_occurrences": token_occurrences,
        "lookalike_token_occurrences": len(lookalike_tokens),
        "parsed_occurrences": len(parsed),
        "all_occurrences_parsed": token_occurrences == len(parsed),
        "values_match_expected": bool(parsed) and all(value == expected for value in parsed),
        "exact_single_occurrence": (
            token_occurrences == 1 and len(parsed) == 1 and parsed[0] == expected and not lookalike_tokens
        ),
        "cleanly_absent": token_occurrences == 0 and not parsed and not lookalike_tokens,
    }


def _ir_summary(text: str, dialect: str) -> dict[str, Any]:
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
    resolved_blocks = [_resolved_block_metadata(block, definitions) for block in raw_blocks]
    targets = [_custom_call_targets(block, dialect) for block in resolved_blocks]
    pallas_indices = [index for index, block_targets in enumerate(targets) if block_targets & _PALLAS_TRITON_TARGETS]
    sole_block = resolved_blocks[0] if len(resolved_blocks) == 1 else ""
    sole_targets = targets[0] if len(targets) == 1 else set()
    sole_tokens = _IR_NAME_TOKEN_PATTERN.findall(sole_block)
    all_tokens = _IR_NAME_TOKEN_PATTERN.findall(text)
    query_bounded_tokens = [token for token in all_tokens if "query_bounded" in re.sub(r"[-.$]+", "_", token.lower())]
    sole_query_bounded_tokens = [
        token for token in sole_tokens if "query_bounded" in re.sub(r"[-.$]+", "_", token.lower())
    ]
    unexpected_query_bounded = [token for token in query_bounded_tokens if token != _EXPECTED_MARKER]
    query_start = _metadata_field(sole_block, "query_start", _QUERY_START)
    query_size = _metadata_field(sole_block, "query_size", _QUERY_SIZE)
    metadata_absent = query_start["cleanly_absent"] and query_size["cleanly_absent"]
    metadata_exact_if_preserved = metadata_absent or (
        query_start["exact_single_occurrence"] and query_size["exact_single_occurrence"]
    )
    checks = {
        "parser_count_matches_textual_custom_call_count": len(raw_blocks) == custom_call_token_count,
        "exactly_one_custom_call_total": len(raw_blocks) == 1 and custom_call_token_count == 1,
        "exactly_one_pallas_call": len(pallas_indices) == 1,
        "sole_exact_rocm_triton_target": sole_targets == {_EXACT_ROCM_TRITON_TARGET},
        "exact_full_forward_q256_marker_in_sole_call": _EXPECTED_MARKER in sole_tokens,
        "no_q0_other_forward_dq_dkdv_or_lookalike_query_bounded_tokens": not unexpected_query_bounded,
        "all_query_bounded_marker_occurrences_belong_to_sole_call": sorted(query_bounded_tokens)
        == sorted(sole_query_bounded_tokens),
        "no_outer_while": while_count == 0,
        "preserved_query_metadata_is_exact": metadata_exact_if_preserved,
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
        "expected_marker_occurrences": sum(token == _EXPECTED_MARKER for token in all_tokens),
        "unexpected_query_bounded_token_occurrences": len(unexpected_query_bounded),
        "while_count": while_count,
        "metadata": {
            "preserved": not metadata_absent,
            "query_start": query_start,
            "query_size": query_size,
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
            set(checks) == _IR_CHECK_NAMES
            and all(value is True for value in checks.values())
            and summary.get("passed") is True
        )
        per_dialect[dialect] = {"checks": checks, "passed": passed}
    return {
        "checks": {
            "exactly_one_independent_summary_per_required_dialect": exact_dialects,
            "stablehlo_passed": exact_dialects and per_dialect.get("stablehlo", {}).get("passed") is True,
            "optimized_hlo_passed": exact_dialects and per_dialect.get("optimized_hlo", {}).get("passed") is True,
        },
        "per_dialect": per_dialect,
        "passed": exact_dialects
        and per_dialect.get("stablehlo", {}).get("passed") is True
        and per_dialect.get("optimized_hlo", {}).get("passed") is True,
    }


def _integer_stat(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = operator.index(value)
    except TypeError:
        return None
    return int(result) if result >= 0 else None


def _compiled_memory(compiled: Any) -> dict[str, Any]:
    stats = compiled.memory_analysis()
    if stats is None:
        return {"available": False}
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "generated_code_size_in_bytes",
    )
    values: dict[str, int] = {}
    for name in names:
        if not hasattr(stats, name):
            continue
        parsed = _integer_stat(getattr(stats, name))
        if parsed is not None:
            values[name] = parsed
    return {"available": True, **values}


def _compiled_memory_gate(memory: dict[str, Any]) -> dict[str, Any]:
    required = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
    )
    available = memory.get("available") is True and all(
        _integer_stat(memory.get(name)) is not None for name in required
    )
    argument = _integer_stat(memory.get("argument_size_in_bytes")) if available else None
    output = _integer_stat(memory.get("output_size_in_bytes")) if available else None
    temporary = _integer_stat(memory.get("temp_size_in_bytes")) if available else None
    combined = (
        argument + output + temporary if argument is not None and output is not None and temporary is not None else None
    )
    checks = {
        "memory_analysis_available": available,
        "argument_bytes_exactly_4196352": argument == _EXPECTED_ARGUMENT_BYTES,
        "output_bytes_exactly_2097152": output == _EXPECTED_OUTPUT_BYTES,
        "temporary_bytes_at_most_64_mib": temporary is not None and temporary <= _MAX_TEMP_BYTES,
    }
    return {
        "expected_argument_bytes": _EXPECTED_ARGUMENT_BYTES,
        "expected_output_bytes": _EXPECTED_OUTPUT_BYTES,
        "maximum_temporary_bytes": _MAX_TEMP_BYTES,
        "argument_output_temporary_bytes": combined,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _shape_signature(jax: Any, jnp: Any) -> tuple[Any, Any, Any, Any]:
    return (
        jax.ShapeDtypeStruct((_BATCH_SIZE, _QUERY_SIZE, _QUERY_HEADS, _HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct((_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM), jnp.bfloat16),
        jax.ShapeDtypeStruct(
            (_BATCH_SIZE, _SEQUENCE_LENGTH),
            jnp.int32,
        ),
    )


def _lower_and_compile_exact(
    jax: Any,
    jnp: Any,
    query_bounded_gqa_forward_chunk: Any,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    output: TextIO,
) -> dict[str, Any]:
    """Return only a privacy-safe report; never expose the executable."""

    def forward_chunk(q_arg: Any, k_arg: Any, v_arg: Any, mask_arg: Any) -> Any:
        return query_bounded_gqa_forward_chunk(
            q_arg,
            k_arg,
            v_arg,
            mask_arg,
            query_start=_QUERY_START,
            block_q=_BLOCK_Q,
            block_k=_BLOCK_K,
            interpret=False,
        )

    _assert_zero_counters(counters)
    _emit(
        {
            "record_type": "stage",
            "stage": "chunk_lower_started",
            "timestamp": _utc_now(),
            "counters": dict(counters),
        },
        output,
    )
    lower_start = time.perf_counter()
    try:
        lowered_callable = jax.jit(forward_chunk)
        lowered = lowered_callable.lower(*_shape_signature(jax, jnp))
        lower_seconds = time.perf_counter() - lower_start
        stablehlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
        stablehlo = _ir_summary(stablehlo_text, "stablehlo")
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
            require_clean_boot,
            output,
            "after_chunk_lower_attempt",
            counters,
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
    compile_start = time.perf_counter()
    compiled = None
    try:
        compiled = lowered.compile()
        compile_seconds = time.perf_counter() - compile_start
        optimized_hlo_text = compiled.as_text()
        optimized_hlo = _ir_summary(optimized_hlo_text, "optimized_hlo")
        del optimized_hlo_text
        memory = _compiled_memory(compiled)
        structural_gate = _structural_gate(stablehlo, optimized_hlo)
        memory_gate = _compiled_memory_gate(memory)
        release_gate = {
            "structural_gate_passed": structural_gate["passed"],
            "compiled_memory_gate_passed": memory_gate["passed"],
            "passed": structural_gate["passed"] and memory_gate["passed"],
        }
        report = {
            "record_type": "chunk_compiled",
            "stage": "chunk_compile_complete_metadata_only",
            "timestamp": _utc_now(),
            "lower_seconds": lower_seconds,
            "compile_seconds": compile_seconds,
            "stablehlo": stablehlo,
            "optimized_hlo": optimized_hlo,
            "compiled_memory": memory,
            "structural_gate": structural_gate,
            "compiled_memory_gate": memory_gate,
            "release_gate": release_gate,
            "raw_ir_emitted": False,
            "executable_returned": False,
            "counters": dict(counters),
        }
        _emit(report, output)
        if not release_gate["passed"]:
            raise RuntimeError("chunk compile failed the independent structural or exact memory gate")
    finally:
        if compiled is not None:
            del compiled
        _journal_checkpoint(
            require_clean_boot,
            output,
            "after_chunk_compile_attempt",
            counters,
        )
    del lowered
    del lowered_callable
    _assert_zero_counters(counters)
    return report


def _backend_manifest(jax: Any, jaxlib: Any, jax_backend: Any) -> dict[str, Any]:
    resolved_backend = jax.default_backend()
    platform_version = str(jax_backend.get_backend().platform_version)
    if resolved_backend != "gpu" or "rocm" not in platform_version.lower():
        raise RuntimeError("requested ROCm but JAX did not resolve the ROCm GPU backend")
    devices = jax.devices()
    if len(devices) != 1:
        raise RuntimeError("ROCm compile gate requires exactly one visible accelerator device")
    return {
        "platform_resolved": "gpu",
        "platform_family": "rocm",
        "device_count": len(devices),
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
    _dependencies: tuple[Any, Any, Any, Any, Any] | None = None,
) -> dict[str, Any]:
    command_buffer_proof = _prove_command_buffers_disabled(environment)
    _emit(
        {
            "record_type": "command_buffer_environment_proof",
            "timestamp": _utc_now(),
            "status": "passed",
            "proof": command_buffer_proof,
            "counters": dict(counters),
        },
        output,
    )
    _journal_checkpoint(
        require_clean_boot,
        output,
        "before_backend_initialization",
        counters,
    )
    try:
        if _dependencies is None:
            import jax
            import jax.numpy as jnp
            import jaxlib
            from jax.extend import backend as jax_backend

            from skyrl.tx.kernels.query_bounded_gqa import (
                query_bounded_gqa_forward_chunk,
            )
        else:
            jax, jnp, jaxlib, jax_backend, query_bounded_gqa_forward_chunk = _dependencies

        backend = _backend_manifest(jax, jaxlib, jax_backend)
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

    report = _lower_and_compile_exact(
        jax,
        jnp,
        query_bounded_gqa_forward_chunk,
        require_clean_boot,
        counters,
        output,
    )
    _assert_zero_counters(counters)
    _emit(
        {
            "record_type": "compile_only_passed",
            "timestamp": _utc_now(),
            "status": "passed_exact_c256_t512_last_query_chunk_compile_only",
            "release_gate": report["release_gate"],
            "executable_returned": False,
            "counters": dict(counters),
        },
        output,
    )
    return report


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
                else "guarded_exact_c256_t512_last_query_chunk_compile_only"
            ),
            "contract": _exact_contract(),
            "compile_may_dispatch_gpu_work": args.platform == "rocm",
            "compile_dispatch_caveat": _COMPILE_GPU_WORK_CAVEAT,
            "fresh_process_required": True,
            "prior_compile_artifact_used": False,
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
                    "pass --platform rocm --allow-gpu --output explicitly under " "profile_rocm.py in a fresh process"
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
        with guarded_process() as safety_preflight:
            _emit(
                {
                    "record_type": "safety_preflight",
                    "timestamp": _utc_now(),
                    "stage": "guard_acquired",
                    "safety": _public_safety_preflight(safety_preflight),
                    "counters": dict(counters),
                },
                output,
            )
            stage = "compile_only_runtime"
            try:
                _run_rocm(
                    output,
                    require_clean_boot,
                    counters,
                    environment=environment,
                )
            finally:
                try:
                    safety_postflight = _public_clean_safety(require_clean_boot(), "safety_postflight")
                except Exception:
                    stage = "safety_postflight"
                    raise
                _emit(
                    {
                        "record_type": "safety_postflight",
                        "timestamp": _utc_now(),
                        "stage": "current_boot_rechecked",
                        "safety": safety_postflight,
                        "counters": dict(counters),
                    },
                    output,
                )
        _assert_zero_counters(counters)
        _emit(
            {
                "record_type": "completed",
                "timestamp": _utc_now(),
                "status": "passed_compile_only",
                "executable_returned": False,
                "counters": dict(counters),
            },
            output,
        )
        return 0
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
