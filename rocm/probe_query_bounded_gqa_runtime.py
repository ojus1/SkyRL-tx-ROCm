#!/usr/bin/env python3
"""Guarded exact-T512 forward runtime gate for query-bounded Qwen3.5 GQA.

The default ``abstract`` mode emits a refusal manifest without importing JAX.
ROCm requires ``--platform rocm --allow-gpu --output`` and must be run through
``profile_rocm.py``. Compilation may dispatch bounded GPU profiling kernels.
The immutable BF16 B1/T512/Hq16/Hkv4/D256 forward is lowered and compiled from
abstract inputs, and both IR dialects must pass the exact one-call structural
gate before the compiled executable is wrapped as callable. The checked
forward then runs exactly once against a host-only analytic causal-GQA oracle.
The full-output comparison runs in host NumPy after an explicit device
transfer, so it cannot hide another GPU dispatch. No GPU reference, RNG,
replay, backward path, padding case, or model dispatcher is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
_SEED = 0
_QUERY_CHUNK_SIZE = 512
_BLOCK_Q = 64
_BLOCK_K = 64
_BACKWARD_BLOCK_Q = 32
_BACKWARD_BLOCK_K = 32
_MAX_RELATIVE_L2 = 0.01
_MIN_COSINE = 0.9999
_MAX_ABSOLUTE_ERROR = 0.02
_MAX_DISPATCH_SECONDS = 0.1
_MAX_COMPILED_TEMP_BYTES = 64 * 1024**2
_MAX_COMPILED_TOTAL_BYTES = 128 * 1024**2
_EXACT_ROCM_TRITON_TARGET = "__gpu$xla.gpu.triton"
_CHECKED_EXECUTABLE_TOKEN = object()
_COMPILE_GPU_WORK_CAVEAT = (
    "lowered.compile may dispatch bounded GPU autotuning/profiling kernels and "
    "allocate compiler-managed buffers; the returned forward executable remains "
    "unavailable until fresh StableHLO and optimized-HLO gates pass"
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


def _zero_counters() -> dict[str, int]:
    return {
        "forward_attempts": 0,
        "forward_completions": 0,
        "lowered_callable_invocations": 0,
    }


def _exact_contract() -> dict[str, Any]:
    q_shape = [_BATCH_SIZE, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM]
    kv_shape = [_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM]
    return {
        "model_family": "Qwen/Qwen3.5-4B",
        "operation": "query_bounded_gqa_forward_only",
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
        ],
        "output": {"shape": q_shape, "dtype": "bfloat16"},
        "seed": _SEED,
        "randomness_used": False,
        "valid_length": _SEQUENCE_LENGTH,
        "scale": _HEAD_DIM**-0.5,
        "tiles": {
            "query_chunk_size": _QUERY_CHUNK_SIZE,
            "block_q": _BLOCK_Q,
            "block_k": _BLOCK_K,
            "backward_block_q": _BACKWARD_BLOCK_Q,
            "backward_block_k": _BACKWARD_BLOCK_K,
        },
        "dispatch_plan": {
            "checked_forward_invocations": 1,
            "gpu_reference_invocations": 0,
            "device_error_reduction_invocations": 0,
            "backward_invocations": 0,
            "replay_invocations": 0,
        },
        "numerical_gate": {
            "all_outputs_finite": True,
            "relative_l2_strictly_below": _MAX_RELATIVE_L2,
            "minimum_cosine": _MIN_COSINE,
            "maximum_absolute_error": _MAX_ABSOLUTE_ERROR,
        },
        "compiled_memory_gate": {
            "memory_analysis_required": True,
            "maximum_temporary_bytes": _MAX_COMPILED_TEMP_BYTES,
            "maximum_argument_output_temporary_bytes": _MAX_COMPILED_TOTAL_BYTES,
        },
        "maximum_candidate_dispatch_seconds": _MAX_DISPATCH_SECONDS,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=("abstract", "rocm"),
        default="abstract",
        help=(
            "no-GPU refusal by default; guarded ROCm compilation and execution "
            "require rocm plus --allow-gpu"
        ),
    )
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help=(
            "acknowledge bounded ROCm compilation and exactly one structurally "
            "and memory-checked candidate invocation"
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


def _load_compile_probe() -> Any:
    try:
        from rocm import probe_query_bounded_gqa_compile
    except ModuleNotFoundError:
        import probe_query_bounded_gqa_compile  # type: ignore[no-redef]

    return probe_query_bounded_gqa_compile


def _configure_rocm_environment() -> dict[str, str | None]:
    return _load_compile_probe()._configure_rocm_environment()


def _load_safety_helpers() -> tuple[
    Callable[[], ContextManager[dict[str, Any]]],
    Callable[[], dict[str, Any]],
]:
    return _load_compile_probe()._load_safety_helpers()


def _open_exclusive_output(path: Path) -> TextIO:
    return _load_compile_probe()._open_exclusive_output(path)


def _summarize_ir(text: str, dialect: str) -> dict[str, Any]:
    return _load_compile_probe()._ir_summary(text, dialect)


def _compiled_memory(compiled: Any) -> dict[str, Any]:
    return _load_compile_probe()._compiled_memory(compiled)


def _forward_ir_gate(*summaries: dict[str, Any]) -> dict[str, Any]:
    dialects = [str(summary["dialect"]) for summary in summaries]
    checks: dict[str, bool] = {
        "exactly_stablehlo_and_optimized_hlo": sorted(dialects)
        == ["optimized_hlo", "stablehlo"]
    }
    for summary in summaries:
        dialect = str(summary["dialect"])
        name_counts = summary["pallas_name_call_counts"]
        calls = summary["pallas_calls"]
        checks[f"{dialect}_exactly_one_custom_call_total"] = (
            int(summary["custom_call_count"]) == 1
        )
        checks[f"{dialect}_exactly_one_pallas_call"] = (
            int(summary["pallas_custom_call_count"]) == 1
        )
        checks[f"{dialect}_exactly_one_forward_marker"] = (
            int(name_counts["forward"]) == 1
        )
        checks[f"{dialect}_no_dq_or_dkdv_marker"] = (
            int(name_counts["dq"]) == 0 and int(name_counts["dkdv"]) == 0
        )
        checks[f"{dialect}_forward_is_only_call_marker"] = len(calls) == 1 and calls[0][
            "expected_markers"
        ] == ["forward"]
        checks[f"{dialect}_sole_exact_rocm_triton_target"] = len(calls) == 1 and set(
            calls[0]["custom_call_targets"]
        ) == {_EXACT_ROCM_TRITON_TARGET}
        checks[f"{dialect}_no_outer_while"] = int(summary["while_count"]) == 0
    return {"checks": checks, "passed": all(checks.values())}


def _compiled_memory_gate(memory: dict[str, Any]) -> dict[str, Any]:
    required = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
    )
    values_available = memory.get("available") is True and all(
        isinstance(memory.get(name), int) and int(memory[name]) >= 0
        for name in required
    )
    total = sum(int(memory[name]) for name in required) if values_available else None
    checks = {
        "memory_analysis_available": values_available,
        "temporary_at_most_64_mib": values_available
        and int(memory["temp_size_in_bytes"]) <= _MAX_COMPILED_TEMP_BYTES,
        "argument_output_temporary_at_most_128_mib": values_available
        and total is not None
        and total <= _MAX_COMPILED_TOTAL_BYTES,
    }
    return {
        "checks": checks,
        "argument_output_temporary_bytes": total,
        "passed": all(checks.values()),
    }


class _CheckedExecutable:
    __slots__ = ("_compiled", "_counter_prefix", "_counters", "proof")

    def __init__(
        self,
        compiled: Any,
        *,
        proof: dict[str, Any],
        counter_prefix: str,
        counters: dict[str, int],
        token: object,
    ) -> None:
        if token is not _CHECKED_EXECUTABLE_TOKEN or not proof.get("passed", False):
            raise RuntimeError("refusing to expose an executable without a passed gate")
        self._compiled = compiled
        self._counter_prefix = counter_prefix
        self._counters = counters
        self.proof = proof

    def invoke(
        self,
        jax: Any,
        *arguments: Any,
        on_started: Callable[[], None],
    ) -> Any:
        attempt = f"{self._counter_prefix}_attempts"
        completion = f"{self._counter_prefix}_completions"
        self._counters[attempt] += 1
        on_started()
        result = self._compiled(*arguments)
        result = jax.block_until_ready(result)
        self._counters[completion] += 1
        return result


def _wrap_checked(
    compiled: Any,
    *,
    proof: dict[str, Any],
    counter_prefix: str,
    counters: dict[str, int],
) -> _CheckedExecutable:
    return _CheckedExecutable(
        compiled,
        proof=proof,
        counter_prefix=counter_prefix,
        counters=counters,
        token=_CHECKED_EXECUTABLE_TOKEN,
    )


def _shape_signature(jax: Any, jnp: Any) -> tuple[Any, Any, Any, Any]:
    q_shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM)
    kv_shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM)
    return (
        jax.ShapeDtypeStruct(q_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(kv_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(kv_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct((_BATCH_SIZE, _SEQUENCE_LENGTH), jnp.int32),
    )


def _compile_checked_forward(
    jax: Any,
    jnp: Any,
    query_bounded_gqa: Any,
    counters: dict[str, int],
    output: TextIO,
) -> tuple[_CheckedExecutable, dict[str, Any]]:
    def forward(q_arg: Any, k_arg: Any, v_arg: Any, mask_arg: Any) -> Any:
        return query_bounded_gqa(
            q_arg,
            k_arg,
            v_arg,
            mask_arg,
            query_chunk_size=_QUERY_CHUNK_SIZE,
            block_q=_BLOCK_Q,
            block_k=_BLOCK_K,
            backward_block_q=_BACKWARD_BLOCK_Q,
            backward_block_k=_BACKWARD_BLOCK_K,
            interpret=False,
        )

    _emit(
        {
            "record_type": "stage",
            "stage": "forward_lower_started",
            "timestamp": _utc_now(),
            "counters": dict(counters),
        },
        output,
    )
    lower_start = time.perf_counter()
    lowered = jax.jit(forward).lower(*_shape_signature(jax, jnp))
    lower_seconds = time.perf_counter() - lower_start
    stablehlo = _summarize_ir(
        str(lowered.compiler_ir(dialect="stablehlo")), "stablehlo"
    )
    _emit(
        {
            "record_type": "lowered",
            "stage": "forward_lower_complete",
            "timestamp": _utc_now(),
            "lower_seconds": lower_seconds,
            "stablehlo": stablehlo,
            "counters": dict(counters),
        },
        output,
    )

    _emit(
        {
            "record_type": "stage",
            "stage": "forward_compile_started",
            "timestamp": _utc_now(),
            "counters": dict(counters),
        },
        output,
    )
    compile_start = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - compile_start
    optimized_hlo = _summarize_ir(compiled.as_text(), "optimized_hlo")
    memory = _compiled_memory(compiled)
    structural_gate = _forward_ir_gate(stablehlo, optimized_hlo)
    memory_gate = _compiled_memory_gate(memory)
    release_gate = {
        "passed": structural_gate["passed"] and memory_gate["passed"],
        "structural_gate": structural_gate,
        "compiled_memory_gate": memory_gate,
    }
    report = {
        "record_type": "forward_compiled",
        "stage": "forward_compile_structural_gate",
        "timestamp": _utc_now(),
        "lower_seconds": lower_seconds,
        "compile_seconds": compile_seconds,
        "stablehlo": stablehlo,
        "optimized_hlo": optimized_hlo,
        "compiled_memory": memory,
        "structural_gate": structural_gate,
        "compiled_memory_gate": memory_gate,
        "release_gate": release_gate,
        "counters": dict(counters),
    }
    _emit(report, output)
    if not release_gate["passed"]:
        del compiled
        raise RuntimeError(
            "forward executable failed the structural or compiled-memory release gate"
        )
    checked = _wrap_checked(
        compiled,
        proof=release_gate,
        counter_prefix="forward",
        counters=counters,
    )
    _emit(
        {
            "record_type": "structural_gate",
            "stage": "forward_executable_released_after_gate",
            "timestamp": _utc_now(),
            "status": "passed",
            "counters": dict(counters),
        },
        output,
    )
    return checked, report


def _journal_checkpoint(
    require_clean_boot: Callable[[], dict[str, Any]],
    output: TextIO,
    stage: str,
    counters: dict[str, int],
) -> dict[str, Any]:
    safety = require_clean_boot()
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


def _dispatch_checked(
    jax: Any,
    executable: _CheckedExecutable,
    arguments: tuple[Any, ...],
    *,
    label: str,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    output: TextIO,
) -> tuple[Any, float]:
    attempt_start = time.perf_counter()
    dispatch_start: float | None = None

    def emit_started() -> None:
        nonlocal dispatch_start
        _emit(
            {
                "record_type": "dispatch_started",
                "timestamp": _utc_now(),
                "label": label,
                "counters": dict(counters),
            },
            output,
        )
        # Measure only the checked executable and synchronization, not the
        # required pre-dispatch JSONL flush above.
        dispatch_start = time.perf_counter()

    try:
        result = executable.invoke(jax, *arguments, on_started=emit_started)
    finally:
        dispatch_seconds = time.perf_counter() - (
            dispatch_start if dispatch_start is not None else attempt_start
        )
        _journal_checkpoint(
            require_clean_boot,
            output,
            f"after_{label}_dispatch",
            counters,
        )
    _emit(
        {
            "record_type": "dispatch",
            "timestamp": _utc_now(),
            "label": label,
            "seconds": dispatch_seconds,
            "counters": dict(counters),
        },
        output,
    )
    return result, dispatch_seconds


def _array_manifest(name: str, value: Any) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "nbytes": int(value.nbytes),
        "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


def _construct_host_inputs() -> tuple[
    tuple[Any, Any, Any, Any], list[dict[str, Any]], Any, dict[str, Any]
]:
    import ml_dtypes
    import numpy as np

    q_shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM)
    kv_shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM)
    q = np.zeros(q_shape, dtype=ml_dtypes.bfloat16)
    k = np.zeros(kv_shape, dtype=ml_dtypes.bfloat16)
    positions = np.linspace(-0.9, 0.9, _SEQUENCE_LENGTH, dtype=np.float32)
    kv_offsets = np.asarray((-0.075, -0.025, 0.025, 0.075), dtype=np.float32)
    values = positions[:, None] + kv_offsets[None, :]
    v = (
        np.broadcast_to(values[None, :, :, None], kv_shape)
        .copy()
        .astype(ml_dtypes.bfloat16)
    )
    key_mask = np.ones((_BATCH_SIZE, _SEQUENCE_LENGTH), dtype=np.int32)
    host_inputs = (
        q,
        k,
        v,
        key_mask,
    )
    manifests = [
        _array_manifest(name, value)
        for name, value in zip(("q", "k", "v", "key_mask"), host_inputs, strict=True)
    ]
    v_fp32 = np.asarray(v, dtype=np.float32)
    prefix_counts = np.arange(1, _SEQUENCE_LENGTH + 1, dtype=np.float32)[
        None, :, None, None
    ]
    cumulative_means = np.cumsum(v_fp32, axis=1, dtype=np.float32) / prefix_counts
    query_to_kv = np.arange(_QUERY_HEADS, dtype=np.int32) // (_QUERY_HEADS // _KV_HEADS)
    expected = cumulative_means[:, :, query_to_kv, :]
    return host_inputs, manifests, expected, _array_manifest("expected", expected)


def _device_put_inputs(jax: Any, host_inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    return jax.block_until_ready(jax.device_put(host_inputs))


def _allocator_snapshot(jax: Any) -> list[dict[str, Any]]:
    snapshots = []
    for device in jax.devices():
        raw = device.memory_stats()
        snapshots.append(
            {
                "device": str(device),
                "memory_stats": None
                if raw is None
                else {str(key): str(value) for key, value in sorted(raw.items())},
            }
        )
    return snapshots


def _host_metrics(actual_host: Any, expected_host: Any) -> dict[str, Any]:
    import numpy as np

    actual_raw = np.asarray(actual_host)
    expected_raw = np.asarray(expected_host)
    actual = actual_raw.astype(np.float32)
    expected = expected_raw.astype(np.float32)
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
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference.ravel()) / denominator),
        "cosine": float(np.vdot(actual.ravel(), expected.ravel()) / cosine_denominator),
        "actual_sha256": hashlib.sha256(actual_raw.tobytes(order="C")).hexdigest(),
        "reference_sha256": hashlib.sha256(expected_raw.tobytes(order="C")).hexdigest(),
    }


def _run_rocm(
    output: TextIO,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    *,
    _dependencies: tuple[Any, Any, Any, Any, Any] | None = None,
) -> int:
    if _dependencies is None:
        import jax
        import jax.numpy as jnp
        import jaxlib
        from jax.extend import backend as jax_backend

        from skyrl.tx.kernels.query_bounded_gqa import query_bounded_gqa
    else:
        jax, jnp, jaxlib, jax_backend, query_bounded_gqa = _dependencies

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
            "timestamp": _utc_now(),
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "platform_resolved": resolved_backend,
            "platform_version": platform_version,
            "devices": [str(device) for device in jax.devices()],
            "allocator": _allocator_snapshot(jax),
            "counters": dict(counters),
        },
        output,
    )
    _journal_checkpoint(
        require_clean_boot, output, "after_backend_initialization", counters
    )

    forward, compile_report = _compile_checked_forward(
        jax, jnp, query_bounded_gqa, counters, output
    )
    _journal_checkpoint(require_clean_boot, output, "after_forward_compile", counters)
    host_inputs, input_manifests, expected_host, expected_manifest = (
        _construct_host_inputs()
    )
    _emit(
        {
            "record_type": "host_analytic_reference",
            "timestamp": _utc_now(),
            "construction": {
                "q": "all BF16 zeros",
                "k": "all BF16 zeros",
                "v": (
                    "BF16 broadcast of linspace(-0.9,0.9,T) plus KV-head "
                    "offsets [-0.075,-0.025,0.025,0.075]"
                ),
                "key_mask": "all int32 ones",
                "expected": "host FP32 cumulative mean of BF16 V mapped by qh//4",
                "randomness_used": False,
                "seed_manifest_value": _SEED,
            },
            "inputs": input_manifests,
            "expected": expected_manifest,
            "counters": dict(counters),
        },
        output,
    )
    inputs = _device_put_inputs(jax, host_inputs)
    _journal_checkpoint(
        require_clean_boot, output, "after_explicit_input_device_put", counters
    )
    allocator_before = _allocator_snapshot(jax)

    actual, dispatch_seconds = _dispatch_checked(
        jax,
        forward,
        inputs,
        label="single_forward_candidate",
        require_clean_boot=require_clean_boot,
        counters=counters,
        output=output,
    )
    actual_host = jax.device_get(actual)
    _journal_checkpoint(
        require_clean_boot, output, "after_candidate_device_get", counters
    )
    metrics = _host_metrics(actual_host, expected_host)
    numerical_passed = (
        metrics["finite"]
        and math.isfinite(metrics["relative_l2"])
        and metrics["relative_l2"] < _MAX_RELATIVE_L2
        and math.isfinite(metrics["cosine"])
        and metrics["cosine"] >= _MIN_COSINE
        and math.isfinite(metrics["max_abs"])
        and metrics["max_abs"] <= _MAX_ABSOLUTE_ERROR
    )
    duration_passed = dispatch_seconds < _MAX_DISPATCH_SECONDS
    passed = numerical_passed and duration_passed
    _emit(
        {
            "record_type": "numerical_validation",
            "timestamp": _utc_now(),
            "status": "passed" if passed else "failed",
            "thresholds": {
                "finite_required": True,
                "relative_l2_strictly_below": _MAX_RELATIVE_L2,
                "minimum_cosine": _MIN_COSINE,
                "maximum_absolute_error": _MAX_ABSOLUTE_ERROR,
                "dispatch_seconds_strictly_below": _MAX_DISPATCH_SECONDS,
            },
            "metrics": metrics,
            "gates": {
                "numerical_passed": numerical_passed,
                "duration_passed": duration_passed,
            },
            "single_candidate_dispatch_seconds": dispatch_seconds,
            "counters": dict(counters),
        },
        output,
    )
    if not passed:
        raise RuntimeError(
            "single-forward gate failed: require duration <100 ms, finite output, "
            "relative L2 <1%, cosine >=0.9999, and max absolute error <=0.02"
        )
    expected_counters = {
        "forward_attempts": 1,
        "forward_completions": 1,
        "lowered_callable_invocations": 0,
    }
    if counters != expected_counters:
        raise RuntimeError(
            f"invocation counter contract violated: {counters!r} != {expected_counters!r}"
        )
    _emit(
        {
            "record_type": "runtime_passed",
            "timestamp": _utc_now(),
            "status": "passed_exact_t512_forward_only",
            "compile_structural_gate": compile_report["structural_gate"],
            "compiled_memory_gate": compile_report["compiled_memory_gate"],
            "allocator_before_execution": allocator_before,
            "allocator_after_validation": _allocator_snapshot(jax),
            "counters": dict(counters),
            "backward_invocations": 0,
            "replay_invocations": 0,
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
            "scope": "abstract_refusal"
            if args.platform == "abstract"
            else "guarded_exact_t512_forward_runtime",
            "contract": _exact_contract(),
            "compile_may_dispatch_gpu_work": args.platform == "rocm",
            "compile_dispatch_caveat": _COMPILE_GPU_WORK_CAVEAT,
            "prior_compile_artifact_used": False,
            "model_dispatcher_connected": False,
            "counters": dict(counters),
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
                    "profile_rocm.py"
                ),
                "jax_imported": False,
                "counters": dict(counters),
            },
            output,
        )
        return 0

    stage = "bounded_environment"
    try:
        environment = _configure_rocm_environment()
        _emit(
            {
                "record_type": "environment",
                "timestamp": _utc_now(),
                "stage": "bounded_rocm_environment_configured",
                "environment": environment,
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
                    "safety": safety_preflight,
                    "counters": dict(counters),
                },
                output,
            )
            stage = "runtime"
            try:
                result = _run_rocm(output, require_clean_boot, counters)
            finally:
                try:
                    safety_postflight = require_clean_boot()
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
                "message": str(error),
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
