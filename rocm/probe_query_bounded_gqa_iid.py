#!/usr/bin/env python3
"""Guarded exact-T512 fully-IID forward gate for query-bounded Qwen3.5 GQA.

The default ``abstract`` mode emits a refusal manifest without importing JAX.
ROCm requires ``--platform rocm --allow-gpu --output`` in a fresh process under
``profile_rocm.py``.  The exact BF16 B1/T512/Hq16/Hkv4/D256 forward is lowered
and compiled once, and the executable is released only after the promoted
probe's exact one-call IR and compiled-memory gates pass.  It is invoked once
on deterministic host-PCG64 inputs whose every Q/K/V feature is independently
sampled, nonzero, and drawn from a bounded power-of-two integer grid.

Validation uses an independent host-only FP32 causal-GQA implementation with
dense QK matmuls, stable softmax, and dense AV matmuls.  There is no replay,
backward path, padding case, accelerator RNG, GPU reference, device-side error
reduction, command-buffer execution, or model dispatcher.
"""

from __future__ import annotations

import argparse
import hashlib
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
_SEED = 20260713
_GRID_DENOMINATOR = 128
_QK_MAX_MAGNITUDE = 48
_V_MAX_MAGNITUDE = _QK_MAX_MAGNITUDE
_MAX_RELATIVE_L2 = 0.01
_MIN_COSINE = 0.9999
_MAX_ABSOLUTE_ERROR = 0.02
_MAX_DISPATCH_SECONDS = 0.1
_MAX_COMPILED_TEMP_BYTES = 64 * 1024**2
_MAX_COMPILED_TOTAL_BYTES = 128 * 1024**2


def _replay_probe() -> Any:
    try:
        from rocm import probe_query_bounded_gqa_replay
    except ModuleNotFoundError:
        import probe_query_bounded_gqa_replay  # type: ignore[no-redef]

    return probe_query_bounded_gqa_replay


def _runtime_probe() -> Any:
    return _replay_probe()._runtime_probe()


def _source_hashes() -> dict[str, str]:
    replay_file = Path(_replay_probe().__file__)
    runtime_file = Path(_runtime_probe().__file__)
    return {
        "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "delegated_replay_probe_source_sha256": hashlib.sha256(replay_file.read_bytes()).hexdigest(),
        "delegated_runtime_probe_source_sha256": hashlib.sha256(runtime_file.read_bytes()).hexdigest(),
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _emit(record: dict[str, Any], output: TextIO) -> None:
    output.write(json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n")
    output.flush()


def _zero_counters() -> dict[str, int]:
    return {
        "forward_attempts": 0,
        "forward_completions": 0,
        "lowered_callable_invocations": 0,
    }


def _expected_completed_counters() -> dict[str, int]:
    return {
        "forward_attempts": 1,
        "forward_completions": 1,
        "lowered_callable_invocations": 0,
    }


def _exact_contract() -> dict[str, Any]:
    q_shape = [_BATCH_SIZE, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM]
    kv_shape = [_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM]
    maximum_logit = _HEAD_DIM * (_QK_MAX_MAGNITUDE / _GRID_DENOMINATOR) ** 2 * (_HEAD_DIM**-0.5)
    return {
        "model_family": "Qwen/Qwen3.5-4B",
        "operation": "query_bounded_gqa_forward_only_fully_iid_per_feature",
        "inputs": [
            {
                "name": "q",
                "shape": q_shape,
                "dtype": "bfloat16",
                "value": "host_pcg64_iid_nonzero_signed_integer_grid",
            },
            {
                "name": "k",
                "shape": kv_shape,
                "dtype": "bfloat16",
                "value": "host_pcg64_iid_nonzero_signed_integer_grid",
            },
            {
                "name": "v",
                "shape": kv_shape,
                "dtype": "bfloat16",
                "value": "host_pcg64_iid_nonzero_signed_integer_grid",
            },
            {
                "name": "key_mask",
                "shape": [_BATCH_SIZE, _SEQUENCE_LENGTH],
                "dtype": "int32",
                "value": "all_ones",
            },
        ],
        "output": {"shape": q_shape, "dtype": "bfloat16"},
        "seed": _SEED,
        "random_generator": "numpy.random.Generator(PCG64)",
        "randomness_location": "host_numpy_only_before_device_put",
        "iid_unit": "each individual Q/K/V feature",
        "zero_values_permitted": False,
        "integer_grid_denominator": _GRID_DENOMINATOR,
        "qk_maximum_integer_magnitude": _QK_MAX_MAGNITUDE,
        "v_maximum_integer_magnitude": _V_MAX_MAGNITUDE,
        "theoretical_maximum_absolute_qk_logit": maximum_logit,
        "valid_length": _SEQUENCE_LENGTH,
        "scale": _HEAD_DIM**-0.5,
        "tiles": {
            "query_chunk_size": 512,
            "block_q": 64,
            "block_k": 64,
            "backward_block_q": 32,
            "backward_block_k": 32,
        },
        "dispatch_plan": {
            "checked_forward_invocations": 1,
            "replay_invocations": 0,
            "command_buffer_invocations": 0,
            "gpu_reference_invocations": 0,
            "device_error_reduction_invocations": 0,
            "backward_invocations": 0,
        },
        "reference": {
            "location": "host_numpy_only",
            "dtype": "float32",
            "qk": "dense mapped-Hq-to-Hkv matmul",
            "softmax": "causal stable dense softmax",
            "av": "dense matmul",
        },
        "numerical_gate": {
            "all_outputs_finite": True,
            "relative_l2_strictly_below": _MAX_RELATIVE_L2,
            "minimum_cosine": _MIN_COSINE,
            "maximum_absolute_error": _MAX_ABSOLUTE_ERROR,
            "dispatch_seconds_strictly_below": _MAX_DISPATCH_SECONDS,
        },
        "compiled_memory_gate": {
            "memory_analysis_required": True,
            "maximum_temporary_bytes": _MAX_COMPILED_TEMP_BYTES,
            "maximum_argument_output_temporary_bytes": _MAX_COMPILED_TOTAL_BYTES,
        },
        "compiled_structural_gate": ("delegated exact one-call forward-only IR release gate"),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=("abstract", "rocm"),
        default="abstract",
        help=(
            "no-GPU refusal by default; guarded ROCm compilation and exactly "
            "one checked invocation require rocm plus --allow-gpu"
        ),
    )
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help=("acknowledge bounded ROCm compilation and exactly one fully-IID " "candidate invocation"),
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
        parser.error(f"refusing to overwrite existing output: {args.output}")
    return args


def _configure_rocm_environment() -> dict[str, str | None]:
    return _replay_probe()._configure_rocm_environment()


def _environment_manifest(environment: dict[str, str | None]) -> dict[str, Any]:
    return _replay_probe()._environment_manifest(environment)


def _prove_command_buffers_disabled(
    environment: dict[str, str | None],
) -> dict[str, Any]:
    return _replay_probe()._prove_command_buffers_disabled(environment)


def _assert_fresh_accelerator_process() -> None:
    _replay_probe()._assert_fresh_accelerator_process()


def _load_safety_helpers() -> tuple[
    Callable[[], ContextManager[dict[str, Any]]],
    Callable[[], dict[str, Any]],
]:
    return _replay_probe()._load_safety_helpers()


def _open_exclusive_output(path: Path) -> TextIO:
    return _replay_probe()._open_exclusive_output(path)


def _compile_checked_forward(
    jax: Any,
    jnp: Any,
    query_bounded_gqa: Any,
    counters: dict[str, int],
    output: TextIO,
) -> tuple[Any, dict[str, Any]]:
    return _replay_probe()._compile_checked_forward(jax, jnp, query_bounded_gqa, counters, output)


def _allocator_snapshot(jax: Any) -> list[dict[str, Any]]:
    return _replay_probe()._allocator_snapshot(jax)


def _array_manifest(name: str, value: Any) -> dict[str, Any]:
    return _replay_probe()._array_manifest(name, value)


def _host_metrics(actual_host: Any, expected_host: Any) -> dict[str, Any]:
    return _replay_probe()._host_metrics(actual_host, expected_host)


def _redacted_message_summary(error: BaseException) -> dict[str, Any]:
    return _replay_probe()._redacted_message_summary(error)


def _journal_checkpoint(
    require_clean_boot: Callable[[], dict[str, Any]],
    output: TextIO,
    stage: str,
    counters: dict[str, int],
) -> dict[str, Any]:
    return _replay_probe()._journal_checkpoint(require_clean_boot, output, stage, counters)


def _public_clean_safety(safety: Any, stage: str) -> dict[str, Any]:
    if not isinstance(safety, dict) or safety.get("amdgpu_boot_clean") is not True:
        raise RuntimeError(f"{stage} did not prove a clean AMDGPU boot")
    if safety.get("fatal_amdgpu_events") != []:
        raise RuntimeError(f"{stage} returned an invalid fatal-event proof")
    return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}


def _public_safety_preflight(safety: Any) -> dict[str, Any]:
    """Validate and emit only controlled boot, headless, and KFD evidence."""
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


def _iid_nonzero_grid(
    rng: Any,
    shape: tuple[int, ...],
    maximum_magnitude: int,
) -> Any:
    import ml_dtypes
    import numpy as np

    magnitudes = rng.integers(1, maximum_magnitude + 1, size=shape, dtype=np.int16)
    signs = 2 * rng.integers(0, 2, size=shape, dtype=np.int16) - 1
    values = (magnitudes * signs).astype(np.float32) / float(_GRID_DENOMINATOR)
    result = values.astype(ml_dtypes.bfloat16)
    if result.shape != shape or np.count_nonzero(result) != result.size:
        raise RuntimeError("IID integer-grid construction produced a zero or bad shape")
    return result


def _construct_iid_inputs() -> tuple[Any, Any, Any, Any]:
    import numpy as np

    q_shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM)
    kv_shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM)
    rng = np.random.Generator(np.random.PCG64(_SEED))
    q = _iid_nonzero_grid(rng, q_shape, _QK_MAX_MAGNITUDE)
    k = _iid_nonzero_grid(rng, kv_shape, _QK_MAX_MAGNITUDE)
    v = _iid_nonzero_grid(rng, kv_shape, _V_MAX_MAGNITUDE)
    key_mask = np.ones((_BATCH_SIZE, _SEQUENCE_LENGTH), dtype=np.int32)
    return q, k, v, key_mask


def _dense_causal_gqa_oracle(q: Any, k: Any, v: Any) -> tuple[Any, dict[str, Any]]:
    """Compute the complete mapped GQA result through independent FP32 matmuls."""
    import numpy as np

    expected_q_shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM)
    expected_kv_shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM)
    if q.shape != expected_q_shape or k.shape != expected_kv_shape or v.shape != expected_kv_shape:
        raise RuntimeError("host oracle received arrays outside the immutable contract")
    if _BATCH_SIZE != 1 or _QUERY_HEADS != _KV_HEADS * _GROUP_SIZE:
        raise RuntimeError("host oracle requires the exact B1 grouped-query mapping")

    q_fp32 = np.asarray(q, dtype=np.float32)[0]
    k_fp32 = np.asarray(k, dtype=np.float32)[0]
    v_fp32 = np.asarray(v, dtype=np.float32)[0]
    expected = np.empty(expected_q_shape, dtype=np.float32)
    causal_invalid = np.triu(np.ones((_SEQUENCE_LENGTH, _SEQUENCE_LENGTH), dtype=np.bool_), k=1)
    q_to_kv = np.arange(_QUERY_HEADS, dtype=np.int32) // _GROUP_SIZE
    observed_max_abs_logit = 0.0
    for query_head, kv_head in enumerate(q_to_kv):
        logits = (q_fp32[:, query_head, :] @ k_fp32[:, kv_head, :].T).astype(np.float32, copy=False)
        logits *= np.float32(_HEAD_DIM**-0.5)
        observed_max_abs_logit = max(observed_max_abs_logit, float(np.max(np.abs(logits))))
        logits[causal_invalid] = -np.inf
        row_max = np.max(logits, axis=-1, keepdims=True)
        probabilities = np.exp(logits - row_max).astype(np.float32, copy=False)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True, dtype=np.float32)
        expected[0, :, query_head, :] = (probabilities @ v_fp32[:, kv_head, :]).astype(np.float32, copy=False)
    if not np.all(np.isfinite(expected)):
        raise RuntimeError("host FP32 dense oracle produced a non-finite output")
    return expected, {
        "implementation": "independent_host_dense_causal_gqa",
        "q_to_kv_mapping": [int(value) for value in q_to_kv],
        "qk_operation": "float32 dense matrix multiplication",
        "softmax_operation": "float32 subtract-max exp normalize",
        "av_operation": "float32 dense matrix multiplication",
        "observed_maximum_absolute_unmasked_logit": observed_max_abs_logit,
        "accelerator_used": False,
    }


def _emulate_bf16_probability_path(
    q: Any,
    k: Any,
    v: Any,
    *,
    block_q: int = 64,
    block_k: int = 64,
) -> Any:
    """Predict the kernel's online FP32 softmax and BF16 probability/V dot."""
    import ml_dtypes
    import numpy as np

    expected_q_shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM)
    expected_kv_shape = (_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM)
    if q.shape != expected_q_shape or k.shape != expected_kv_shape or v.shape != expected_kv_shape:
        raise RuntimeError("BF16 path emulator received arrays outside the contract")
    if (
        _BATCH_SIZE != 1
        or _QUERY_HEADS != _KV_HEADS * _GROUP_SIZE
        or block_q <= 0
        or block_k <= 0
        or _SEQUENCE_LENGTH % block_q
        or _SEQUENCE_LENGTH % block_k
    ):
        raise RuntimeError("BF16 path emulator requires exact divisible B1 GQA tiles")

    q_fp32 = np.asarray(q, dtype=np.float32)[0]
    k_fp32 = np.asarray(k, dtype=np.float32)[0]
    v_fp32 = np.asarray(v, dtype=np.float32)[0]
    emulated = np.empty(expected_q_shape, dtype=ml_dtypes.bfloat16)
    log2_scale = np.float32((_HEAD_DIM**-0.5) * math.log2(math.e))
    for query_head in range(_QUERY_HEADS):
        kv_head = query_head // _GROUP_SIZE
        for query_start in range(0, _SEQUENCE_LENGTH, block_q):
            query_stop = query_start + block_q
            query_positions = np.arange(query_start, query_stop, dtype=np.int32)
            row_max = np.full((block_q,), -np.inf, dtype=np.float32)
            row_sum = np.zeros((block_q,), dtype=np.float32)
            accumulator = np.zeros((block_q, _HEAD_DIM), dtype=np.float32)
            key_block_limit = (query_stop + block_k - 1) // block_k
            for key_block in range(key_block_limit):
                key_start = key_block * block_k
                key_stop = key_start + block_k
                key_positions = np.arange(key_start, key_stop, dtype=np.int32)
                logits = (
                    q_fp32[query_start:query_stop, query_head, :] @ k_fp32[key_start:key_stop, kv_head, :].T
                ).astype(np.float32, copy=False)
                logits *= log2_scale
                logits = np.where(
                    key_positions[None, :] <= query_positions[:, None],
                    logits,
                    -np.inf,
                )
                current_max = np.max(logits, axis=-1)
                next_max = np.maximum(row_max, current_max)
                correction = np.zeros_like(next_max)
                prior_finite = np.isfinite(row_max)
                correction[prior_finite] = np.exp2(row_max[prior_finite] - next_max[prior_finite]).astype(
                    np.float32, copy=False
                )
                probabilities = np.exp2(logits - next_max[:, None]).astype(np.float32, copy=False)
                row_sum = correction * row_sum + np.sum(probabilities, axis=-1, dtype=np.float32)
                bf16_probabilities = probabilities.astype(ml_dtypes.bfloat16).astype(np.float32)
                accumulator = correction[:, None] * accumulator + (
                    bf16_probabilities @ v_fp32[key_start:key_stop, kv_head, :]
                ).astype(np.float32, copy=False)
                row_max = next_max
            emulated[0, query_start:query_stop, query_head, :] = (accumulator / row_sum[:, None]).astype(
                ml_dtypes.bfloat16
            )
    return emulated


def _construct_host_inputs() -> tuple[tuple[Any, Any, Any, Any], list[dict[str, Any]], Any, dict[str, Any]]:
    host_inputs = _construct_iid_inputs()
    q, k, v, _key_mask = host_inputs
    expected, oracle = _dense_causal_gqa_oracle(q, k, v)
    emulated = _emulate_bf16_probability_path(q, k, v)
    emulated_metrics = _host_metrics(emulated, expected)
    manifests = [
        _array_manifest(name, value) for name, value in zip(("q", "k", "v", "key_mask"), host_inputs, strict=True)
    ]
    return (
        host_inputs,
        manifests,
        expected,
        {
            **_array_manifest("expected", expected),
            "oracle": oracle,
            "bf16_probability_path_prediction": {
                "scope": (
                    "NumPy emulation of online FP32 softmax with BF16 "
                    "probability and V inputs to FP32 AV accumulation"
                ),
                "output": _array_manifest("emulated_bf16_output", emulated),
                "metrics_vs_full_fp32_reference": emulated_metrics,
                "authorization_effect": "informational_only",
                "accelerator_used": False,
            },
        },
    )


def _device_put_inputs(jax: Any, host_inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    return jax.block_until_ready(jax.device_put(host_inputs))


def _dispatch_single(
    jax: Any,
    executable: Any,
    arguments: tuple[Any, ...],
    *,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    output: TextIO,
) -> tuple[Any, float]:
    expected_before = _zero_counters()
    if counters != expected_before:
        raise RuntimeError(f"refusing out-of-order candidate dispatch: {counters!r} != " f"{expected_before!r}")
    attempt_start = time.perf_counter()
    dispatch_start: float | None = None

    def emit_started() -> None:
        nonlocal dispatch_start
        _emit(
            {
                "record_type": "dispatch_started",
                "timestamp": _utc_now(),
                "label": "fully_iid_candidate",
                "counters": dict(counters),
            },
            output,
        )
        dispatch_start = time.perf_counter()

    try:
        result = executable.invoke(jax, *arguments, on_started=emit_started)
    finally:
        dispatch_seconds = time.perf_counter() - (dispatch_start if dispatch_start is not None else attempt_start)
        _journal_checkpoint(
            require_clean_boot,
            output,
            "after_fully_iid_candidate_dispatch",
            counters,
        )
    _emit(
        {
            "record_type": "dispatch",
            "timestamp": _utc_now(),
            "label": "fully_iid_candidate",
            "seconds": dispatch_seconds,
            "counters": dict(counters),
        },
        output,
    )
    return result, dispatch_seconds


def _device_get_checked(
    jax: Any,
    value: Any,
    *,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    output: TextIO,
) -> Any:
    try:
        return jax.device_get(value)
    finally:
        _journal_checkpoint(
            require_clean_boot,
            output,
            "after_fully_iid_candidate_device_get",
            counters,
        )


def _validate_candidate(
    actual_host: Any,
    expected_host: Any,
    dispatch_seconds: float,
    counters: dict[str, int],
    output: TextIO,
) -> tuple[dict[str, Any], dict[str, bool]]:
    expected_counters = _expected_completed_counters()
    if counters != expected_counters:
        raise RuntimeError(f"candidate invocation counter contract violated: {counters!r} != " f"{expected_counters!r}")
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
    duration_passed = math.isfinite(dispatch_seconds) and 0.0 <= dispatch_seconds < _MAX_DISPATCH_SECONDS
    gates = {
        "numerical_passed": bool(numerical_passed),
        "duration_passed": bool(duration_passed),
        "passed": bool(numerical_passed and duration_passed),
    }
    _emit(
        {
            "record_type": "numerical_validation",
            "timestamp": _utc_now(),
            "status": "passed" if gates["passed"] else "failed",
            "thresholds": {
                "finite_required": True,
                "relative_l2_strictly_below": _MAX_RELATIVE_L2,
                "minimum_cosine": _MIN_COSINE,
                "maximum_absolute_error": _MAX_ABSOLUTE_ERROR,
                "dispatch_seconds_strictly_below": _MAX_DISPATCH_SECONDS,
            },
            "metrics": metrics,
            "gates": gates,
            "dispatch_seconds": dispatch_seconds,
            "counters": dict(counters),
        },
        output,
    )
    if not gates["passed"]:
        raise RuntimeError("fully-IID candidate numerical or duration gate failed")
    return metrics, gates


def _run_rocm(
    output: TextIO,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    *,
    environment: dict[str, str | None],
    _dependencies: tuple[Any, Any, Any, Any, Any] | None = None,
) -> int:
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
    if _dependencies is None:
        import jax
        import jax.numpy as jnp
        import jaxlib
        from jax.extend import backend as jax_backend

        from skyrl.tx.kernels.query_bounded_gqa import query_bounded_gqa
    else:
        jax, jnp, jaxlib, jax_backend, query_bounded_gqa = _dependencies

    try:
        resolved_backend = jax.default_backend()
        platform_version = str(jax_backend.get_backend().platform_version)
        if resolved_backend != "gpu" or "rocm" not in platform_version.lower():
            raise RuntimeError(
                "requested ROCm but JAX resolved " f"{resolved_backend!r} with platform version {platform_version!r}"
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
    finally:
        _journal_checkpoint(
            require_clean_boot,
            output,
            "after_backend_initialization_attempt",
            counters,
        )

    try:
        executable, compile_report = _compile_checked_forward(jax, jnp, query_bounded_gqa, counters, output)
    finally:
        _journal_checkpoint(
            require_clean_boot,
            output,
            "after_forward_compile_attempt",
            counters,
        )

    try:
        host_inputs, input_manifests, expected_host, expected_manifest = _construct_host_inputs()
        _emit(
            {
                "record_type": "host_fully_iid_dense_reference",
                "timestamp": _utc_now(),
                "construction": {
                    "q_k_v": (
                        "every BF16 feature independently sampled by host PCG64 "
                        "from a bounded nonzero signed integer grid"
                    ),
                    "key_mask": "all int32 ones",
                    "expected": "full host FP32 dense causal GQA oracle",
                    "accelerator_rng_used": False,
                    "seed": _SEED,
                },
                "inputs": input_manifests,
                "expected": expected_manifest,
                "counters": dict(counters),
            },
            output,
        )
    finally:
        _journal_checkpoint(
            require_clean_boot,
            output,
            "after_host_reference_construction",
            counters,
        )

    try:
        inputs = _device_put_inputs(jax, host_inputs)
    finally:
        _journal_checkpoint(
            require_clean_boot,
            output,
            "after_explicit_input_device_put",
            counters,
        )
    allocator_before = _allocator_snapshot(jax)

    actual, dispatch_seconds = _dispatch_single(
        jax,
        executable,
        inputs,
        require_clean_boot=require_clean_boot,
        counters=counters,
        output=output,
    )
    actual_host = _device_get_checked(
        jax,
        actual,
        require_clean_boot=require_clean_boot,
        counters=counters,
        output=output,
    )
    metrics, gates = _validate_candidate(actual_host, expected_host, dispatch_seconds, counters, output)
    _journal_checkpoint(
        require_clean_boot,
        output,
        "after_host_numerical_validation",
        counters,
    )
    _emit(
        {
            "record_type": "runtime_passed",
            "timestamp": _utc_now(),
            "status": "passed_exact_t512_fully_iid_forward_only",
            "compile_structural_gate": compile_report["structural_gate"],
            "compiled_memory_gate": compile_report["compiled_memory_gate"],
            "candidate": {
                "metrics": metrics,
                "gates": gates,
                "dispatch_seconds": dispatch_seconds,
            },
            "allocator_before_execution": allocator_before,
            "allocator_after_validation": _allocator_snapshot(jax),
            "counters": dict(counters),
            "replay_invocations": 0,
            "backward_invocations": 0,
            "gpu_reference_invocations": 0,
            "device_error_reduction_invocations": 0,
            "command_buffer_invocations": 0,
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
                "abstract_refusal" if args.platform == "abstract" else "guarded_exact_t512_fully_iid_forward_only"
            ),
            "contract": _exact_contract(),
            "compile_may_dispatch_gpu_work": args.platform == "rocm",
            "fresh_process_required": True,
            "prior_compile_artifact_used": False,
            "model_dispatcher_connected": False,
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
