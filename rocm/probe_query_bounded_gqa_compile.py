#!/usr/bin/env python3
"""Fail-closed compile-only gate for query-bounded Qwen3.5-4B GQA.

The default ``abstract`` mode writes a refusal manifest without importing JAX.
Real ROCm lowering requires both ``--platform rocm`` and ``--allow-gpu``.  That
path lowers and compiles the exact BF16 ``B=1, T=512, Hq=16, Hkv=4, D=256``
forward plus arbitrary-cotangent VJP, extracts compiler metadata, and discards
the executable.  It never invokes the lowered or compiled callable, but
``compile()`` may dispatch bounded GPU autotuning/profiling kernels and must be
treated as GPU work under the guarded telemetry wrapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
_DTYPE = "bfloat16"
_MASK_DTYPE = "int32"
_QUERY_CHUNK_SIZE = 512
_BLOCK_Q = 64
_BLOCK_K = 64
_BACKWARD_BLOCK_Q = 32
_BACKWARD_BLOCK_K = 32
_EXPECTED_PALLAS_CALLS = {
    "forward": "query_bounded_gqa_forward_q0",
    "dq": "query_bounded_gqa_dq_q0",
    "dkdv": "query_bounded_gqa_dkdv_q0",
}
_PALLAS_TRITON_CUSTOM_CALL_TARGETS = frozenset(
    {
        "pallas",
        "pallas_call",
        "triton",
        "triton_kernel_call",
        "xla.gpu.triton",
        "__gpu$xla.gpu.triton",
    }
)
_IR_NAME_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.$-]*")
_COMPILE_GPU_WORK_CAVEAT = (
    "lowered.compile may dispatch bounded GPU autotuning/profiling kernels and "
    "allocate compiler-managed buffers; treat this compile-only path as GPU work "
    "even though the returned callable is never invoked"
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _emit(record: dict[str, Any], output: TextIO) -> None:
    output.write(
        json.dumps(
            record,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    output.flush()


def _exact_contract() -> dict[str, Any]:
    q_shape = [_BATCH_SIZE, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM]
    kv_shape = [_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM]
    return {
        "model_family": "Qwen/Qwen3.5-4B",
        "operation": "query_bounded_gqa_forward_and_vjp",
        "inputs": [
            {"name": "q", "shape": q_shape, "dtype": _DTYPE},
            {"name": "k", "shape": kv_shape, "dtype": _DTYPE},
            {"name": "v", "shape": kv_shape, "dtype": _DTYPE},
            {
                "name": "key_mask",
                "shape": [_BATCH_SIZE, _SEQUENCE_LENGTH],
                "dtype": _MASK_DTYPE,
            },
            {"name": "dout", "shape": q_shape, "dtype": _DTYPE},
        ],
        "outputs": [
            {"name": "output", "shape": q_shape, "dtype": _DTYPE},
            {"name": "dq", "shape": q_shape, "dtype": _DTYPE},
            {"name": "dk", "shape": kv_shape, "dtype": _DTYPE},
            {"name": "dv", "shape": kv_shape, "dtype": _DTYPE},
        ],
        "scale": _HEAD_DIM**-0.5,
        "tiles": {
            "query_chunk_size": _QUERY_CHUNK_SIZE,
            "block_q": _BLOCK_Q,
            "block_k": _BLOCK_K,
            "backward_block_q": _BACKWARD_BLOCK_Q,
            "backward_block_k": _BACKWARD_BLOCK_K,
        },
        "expected_pallas_calls": {
            "forward": 1,
            "dq": 1,
            "dkdv": 1,
            "total": len(_EXPECTED_PALLAS_CALLS),
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=("abstract", "rocm"),
        default="abstract",
        help=(
            "no-GPU refusal by default; ROCm lowering/compilation is GPU work "
            "and requires rocm plus --allow-gpu"
        ),
    )
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help=(
            "acknowledge that ROCm compile() may dispatch bounded GPU "
            "autotuning/profiling kernels"
        ),
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
        parser.error(f"refusing to overwrite existing output: {args.output}")
    return args


def _configure_rocm_environment() -> dict[str, str | None]:
    """Reuse the hardened Pallas probe's bounded ROCm environment contract."""
    try:
        from rocm.probe_pallas_attention import _configure_environment
    except ModuleNotFoundError:
        from probe_pallas_attention import (
            _configure_environment,  # type: ignore[no-redef]
        )

    return _configure_environment()


def _load_safety_helpers() -> tuple[
    Callable[[], ContextManager[dict[str, Any]]],
    Callable[[], dict[str, Any]],
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


def _json_scalar(value: Any) -> bool | int | float | str | None:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):
        try:
            result = value.item()
        except (TypeError, ValueError):
            return str(value)
        if isinstance(result, float) and not math.isfinite(result):
            return str(result)
        if result is None or isinstance(result, (bool, int, float, str)):
            return result
    return str(value)


def _compiled_memory(compiled: Any) -> dict[str, Any]:
    try:
        stats = compiled.memory_analysis()
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    if stats is None:
        return {"available": False}
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_temp_size_in_bytes",
        "generated_code_size_in_bytes",
    )
    return {
        "available": True,
        **{
            name: int(getattr(stats, name))
            for name in names
            if hasattr(stats, name) and getattr(stats, name) is not None
        },
    }


def _cost_analysis(compiled: Any) -> dict[str, Any]:
    try:
        raw = compiled.cost_analysis()
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    if raw is None:
        return {"available": False}
    if isinstance(raw, list):
        raw = (
            raw[0]
            if len(raw) == 1
            else {f"module_{index}": value for index, value in enumerate(raw)}
        )
    if not isinstance(raw, dict):
        return {"available": True, "value": str(raw)}
    return {
        "available": True,
        "metrics": {
            str(key): _json_scalar(value) for key, value in sorted(raw.items())
        },
    }


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
            r"^\s*(?:%[^=]+?=|#[A-Za-z_][\w.-]*\s*=|"
            r"(?:stablehlo\.|func\.)?return\b|}\s*$)"
        )
    elif dialect == "optimized_hlo":
        start_pattern = re.compile(
            r"^(?P<indent>\s*)(?:ROOT\s+)?[^=]+?=\s*.*\bcustom-call\("
        )
        boundary_pattern = re.compile(r"^\s*(?:(?:ROOT\s+)?[^=]+?=|}\s*$)")
    else:
        raise ValueError(f"unsupported IR dialect: {dialect}")

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
        re.findall(
            r"(?:call_target_name|custom_call_target)\s*=\s*\"([^\"]+)\"",
            block,
        )
    )
    if dialect == "stablehlo":
        for match in re.finditer(
            r"\bstablehlo\.custom_call\s+@(?:\"([^\"]+)\"|([A-Za-z0-9_.$-]+))",
            block,
        ):
            targets.add(match.group(1) or match.group(2))
    elif dialect != "optimized_hlo":
        raise ValueError(f"unsupported IR dialect: {dialect}")
    return targets


def _exact_ir_name_tokens(text: str) -> set[str]:
    return set(_IR_NAME_TOKEN_PATTERN.findall(text))


def _ir_summary(text: str, dialect: str) -> dict[str, Any]:
    if dialect == "stablehlo":
        while_pattern = re.compile(r"\bstablehlo\.while\b")
    elif dialect == "optimized_hlo":
        while_pattern = re.compile(r"\bwhile\(")
    else:
        raise ValueError(f"unsupported IR dialect: {dialect}")

    definitions = _metadata_definitions(text)
    blocks = _custom_call_blocks(text, dialect)
    pallas_blocks: list[tuple[str, set[str]]] = []
    for block in blocks:
        targets = _custom_call_targets(block, dialect)
        if targets & _PALLAS_TRITON_CUSTOM_CALL_TARGETS:
            pallas_blocks.append(
                (_resolved_block_metadata(block, definitions), targets)
            )
    name_call_counts = {
        kind: sum(
            marker in _exact_ir_name_tokens(block) for block, _targets in pallas_blocks
        )
        for kind, marker in _EXPECTED_PALLAS_CALLS.items()
    }
    calls = []
    for index, (block, targets) in enumerate(pallas_blocks):
        tokens = _exact_ir_name_tokens(block)
        markers = [
            kind for kind, marker in _EXPECTED_PALLAS_CALLS.items() if marker in tokens
        ]
        calls.append(
            {
                "pallas_call_index": index,
                "custom_call_targets": sorted(targets),
                "expected_markers": markers,
                "has_exactly_one_expected_marker": len(markers) == 1,
            }
        )
    return {
        "dialect": dialect,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "characters": len(text),
        "lines": len(text.splitlines()),
        "custom_call_count": len(blocks),
        "pallas_custom_call_count": len(pallas_blocks),
        "pallas_named_logical_call_count": sum(
            count == 1 for count in name_call_counts.values()
        ),
        "pallas_name_call_counts": name_call_counts,
        "pallas_calls": calls,
        "while_count": len(while_pattern.findall(text)),
    }


def _pallas_count_gate(*summaries: dict[str, Any]) -> dict[str, Any]:
    expected = len(_EXPECTED_PALLAS_CALLS)
    dialects = [str(summary["dialect"]) for summary in summaries]
    checks: dict[str, bool] = {
        "exactly_stablehlo_and_optimized_hlo": sorted(dialects)
        == ["optimized_hlo", "stablehlo"]
    }
    for summary in summaries:
        dialect = str(summary["dialect"])
        targeted = int(summary["pallas_custom_call_count"])
        custom_calls = int(summary["custom_call_count"])
        checks[f"{dialect}_three_pallas_custom_calls"] = targeted == expected
        checks[f"{dialect}_has_at_least_three_custom_calls"] = custom_calls >= expected
        checks[f"{dialect}_each_expected_marker_once"] = all(
            int(summary["pallas_name_call_counts"][kind]) == 1
            for kind in _EXPECTED_PALLAS_CALLS
        )
        checks[f"{dialect}_one_expected_marker_per_pallas_call"] = all(
            bool(call["has_exactly_one_expected_marker"])
            for call in summary["pallas_calls"]
        )
        checks[f"{dialect}_no_while"] = int(summary["while_count"]) == 0
    return {
        "expected_total": expected,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _lower_and_compile_exact(
    jax: Any, jnp: Any, query_bounded_gqa: Any, output: TextIO
) -> dict[str, Any]:
    """Compile the fixed signature and return metadata without leaking a callable."""
    q_shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM)
    kv_shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM)
    q = jax.ShapeDtypeStruct(q_shape, jnp.bfloat16)
    k = jax.ShapeDtypeStruct(kv_shape, jnp.bfloat16)
    v = jax.ShapeDtypeStruct(kv_shape, jnp.bfloat16)
    key_mask = jax.ShapeDtypeStruct((_BATCH_SIZE, _SEQUENCE_LENGTH), jnp.int32)
    dout = jax.ShapeDtypeStruct(q_shape, jnp.bfloat16)

    def forward_and_vjp(
        q_arg: Any, k_arg: Any, v_arg: Any, mask_arg: Any, dout_arg: Any
    ):
        value, pullback = jax.vjp(
            lambda q_item, k_item, v_item: query_bounded_gqa(
                q_item,
                k_item,
                v_item,
                mask_arg,
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

    lowered_callable = jax.jit(forward_and_vjp)
    _emit(
        {
            "record_type": "stage",
            "stage": "lower_started",
            "timestamp": _utc_now(),
            "compiled_executable_invocations": 0,
        },
        output,
    )
    lower_start = time.perf_counter()
    lowered = lowered_callable.lower(q, k, v, key_mask, dout)
    lower_seconds = time.perf_counter() - lower_start
    stablehlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
    stablehlo = _ir_summary(stablehlo_text, "stablehlo")
    del stablehlo_text
    _emit(
        {
            "record_type": "lowered",
            "stage": "lower_complete",
            "timestamp": _utc_now(),
            "lower_seconds": lower_seconds,
            "stablehlo": stablehlo,
            "compiled_executable_invocations": 0,
        },
        output,
    )

    _emit(
        {
            "record_type": "stage",
            "stage": "compile_started",
            "timestamp": _utc_now(),
            "compiled_executable_invocations": 0,
        },
        output,
    )
    compile_start = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - compile_start
    optimized_hlo_text = compiled.as_text()
    optimized_hlo = _ir_summary(optimized_hlo_text, "optimized_hlo")
    del optimized_hlo_text
    compiled_memory = _compiled_memory(compiled)
    cost_analysis = _cost_analysis(compiled)
    count_gate = _pallas_count_gate(stablehlo, optimized_hlo)
    # The executable is deliberately not returned.  Only non-executing
    # compiler metadata methods above are permitted before it is discarded.
    del compiled
    report = {
        "record_type": "compiled",
        "stage": "compile_complete_metadata_only",
        "timestamp": _utc_now(),
        "compile_seconds": compile_seconds,
        "stablehlo": stablehlo,
        "optimized_hlo": optimized_hlo,
        "pallas_count_gate": count_gate,
        "compiled_memory": compiled_memory,
        "cost_analysis": cost_analysis,
        "compiled_executable_invocations": 0,
        "lowered_callable_invocations": 0,
    }
    _emit(report, output)
    return report


def _compile_rocm(output: TextIO) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import jaxlib
    from jax.extend import backend as jax_backend

    from skyrl.tx.kernels.query_bounded_gqa import query_bounded_gqa

    resolved_backend = jax.default_backend()
    platform_version = str(jax_backend.get_backend().platform_version)
    if resolved_backend != "gpu" or "rocm" not in platform_version.lower():
        raise RuntimeError(
            "requested ROCm but JAX resolved "
            f"{resolved_backend!r} with platform version {platform_version!r}"
        )
    _emit(
        {
            "record_type": "backend_ready",
            "stage": "rocm_backend_initialized",
            "timestamp": _utc_now(),
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "platform_resolved": resolved_backend,
            "platform_version": platform_version,
            "devices": [str(device) for device in jax.devices()],
            "compiled_executable_invocations": 0,
        },
        output,
    )
    return _lower_and_compile_exact(jax, jnp, query_bounded_gqa, output)


def _execute(args: argparse.Namespace, output: TextIO) -> int:
    _emit(
        {
            "record_type": "manifest",
            "timestamp": _utc_now(),
            "platform_requested": args.platform,
            "allow_gpu": args.allow_gpu,
            "scope": "abstract_refusal"
            if args.platform == "abstract"
            else "rocm_compile_only_gpu_work",
            "contract": _exact_contract(),
            "no_user_input_arrays_constructed": True,
            "compile_may_dispatch_gpu_work": args.platform == "rocm",
            "compile_dispatch_caveat": _COMPILE_GPU_WORK_CAVEAT,
            "lowered_callable_invocations": 0,
            "compiled_executable_invocations": 0,
            "execution_policy": (
                "lower and compile from ShapeDtypeStruct inputs; extract metadata; "
                "never invoke the lowered or compiled callable; compilation itself "
                "is guarded GPU work"
            ),
        },
        output,
    )
    if args.platform == "abstract":
        _emit(
            {
                "record_type": "refused",
                "timestamp": _utc_now(),
                "status": "no_gpu_abstract_manifest_only",
                "reason": "pass --platform rocm --allow-gpu explicitly to lower and compile",
                "jax_imported": False,
                "compiled_executable_invocations": 0,
            },
            output,
        )
        return 0

    stage = "bounded_environment"
    try:
        effective_environment = _configure_rocm_environment()
        _emit(
            {
                "record_type": "environment",
                "stage": "bounded_rocm_environment_configured",
                "timestamp": _utc_now(),
                "environment": effective_environment,
                "compiled_executable_invocations": 0,
            },
            output,
        )
        guarded_process, require_clean_boot = _load_safety_helpers()
        stage = "safety_preflight"
        with guarded_process() as safety_preflight:
            _emit(
                {
                    "record_type": "safety_preflight",
                    "stage": "guard_acquired",
                    "timestamp": _utc_now(),
                    "safety": safety_preflight,
                    "compiled_executable_invocations": 0,
                },
                output,
            )
            stage = "lower_and_compile"
            try:
                report = _compile_rocm(output)
            finally:
                try:
                    safety_postflight = require_clean_boot()
                except Exception:
                    # Preserve a compile failure's stage when the postflight
                    # succeeds, but attribute an actual postflight failure to
                    # the safety check even if it masks an earlier exception.
                    stage = "safety_postflight"
                    raise
                _emit(
                    {
                        "record_type": "safety_postflight",
                        "stage": "current_boot_rechecked",
                        "timestamp": _utc_now(),
                        "safety": safety_postflight,
                        "compiled_executable_invocations": 0,
                    },
                    output,
                )
        if not report["pallas_count_gate"]["passed"]:
            raise RuntimeError(
                "compiled IR failed the exact three-call/name/no-while Pallas gate"
            )
        _emit(
            {
                "record_type": "completed",
                "timestamp": _utc_now(),
                "status": "passed_compile_only",
                "compiled_executable_invocations": 0,
                "lowered_callable_invocations": 0,
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
                "message": str(error),
                "compiled_executable_invocations": 0,
                "lowered_callable_invocations": 0,
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
