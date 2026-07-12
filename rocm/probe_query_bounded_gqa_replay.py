#!/usr/bin/env python3
"""Guarded exact-T512 nonzero-input replay gate for query-bounded GQA.

The default ``abstract`` mode emits a refusal manifest without importing JAX.
ROCm requires ``--platform rocm --allow-gpu --output`` and must be run through
``profile_rocm.py``.  One exact BF16 B1/T512/Hq16/Hkv4/D256 forward executable
is lowered and compiled, then released only after the same strict IR and memory
gates as the promoted single-forward probe.  It is invoked once on seeded,
dense, nonzero Q/K/V.  Only after that result passes a host-only causal-GQA
oracle and a clean-journal checkpoint is the same executable invoked once more
as an ordinary host-driven replay.  Command buffers remain disabled.

There is no GPU reference, device-side error reduction, RNG on the accelerator,
backward path, padding case, model dispatcher, or executable recompilation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
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
_MAX_RELATIVE_L2 = 0.01
_MIN_COSINE = 0.9999
_MAX_ABSOLUTE_ERROR = 0.02
_MAX_DISPATCH_SECONDS = 0.1
_MAX_COMPILED_TEMP_BYTES = 64 * 1024**2
_MAX_COMPILED_TOTAL_BYTES = 128 * 1024**2
_DISABLED_COMMAND_BUFFER_FLAG = "--xla_gpu_enable_command_buffer="
_COMMAND_BUFFER_FLAG_NAME = _DISABLED_COMMAND_BUFFER_FLAG.removesuffix("=")
_NEGATED_COMMAND_BUFFER_FLAG_NAME = "--no" + _COMMAND_BUFFER_FLAG_NAME.removeprefix("--")
_PHASES = ("candidate", "replay")
_JOURNAL_PROOF_TOKEN = object()
_VALIDATION_PROOF_TOKEN = object()
_REPLAY_AUTHORIZATION_TOKEN = object()


def _runtime_probe() -> Any:
    try:
        from rocm import probe_query_bounded_gqa_runtime
    except ModuleNotFoundError:
        import probe_query_bounded_gqa_runtime  # type: ignore[no-redef]

    return probe_query_bounded_gqa_runtime


def _source_hashes() -> dict[str, str]:
    runtime_file = Path(_runtime_probe().__file__)
    return {
        "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "delegated_runtime_probe_source_sha256": hashlib.sha256(runtime_file.read_bytes()).hexdigest(),
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _emit(record: dict[str, Any], output: TextIO) -> None:
    output.write(json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n")
    output.flush()


def _redacted_text_summary(message: str) -> dict[str, Any]:
    message_bytes = message.encode("utf-8", errors="replace")
    return {
        "text_redacted": True,
        "utf8_bytes": len(message_bytes),
        "sha256": hashlib.sha256(message_bytes).hexdigest(),
    }


def _redacted_message_summary(error: BaseException) -> dict[str, Any]:
    summary = _redacted_text_summary(str(error))
    return {
        "message_redacted": summary["text_redacted"],
        "message_utf8_bytes": summary["utf8_bytes"],
        "message_sha256": summary["sha256"],
    }


def _redact_nested_exception_text(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (
                _redacted_text_summary(child)
                if key in {"error", "message"} and isinstance(child, str)
                else _redact_nested_exception_text(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redact_nested_exception_text(child) for child in value]
    return value


class _PrivateJsonlWriter:
    """Redact delegated JSONL exception strings before writing the artifact."""

    __slots__ = ("_output",)

    def __init__(self, output: TextIO) -> None:
        self._output = output

    def write(self, text: str) -> int:
        if not text.endswith("\n") or text.count("\n") != 1:
            raise RuntimeError("delegated probe emitted a non-atomic JSONL record")
        record = json.loads(text)
        _emit(_redact_nested_exception_text(record), self._output)
        return len(text)

    def flush(self) -> None:
        self._output.flush()


def _zero_counters() -> dict[str, int]:
    return {
        "forward_attempts": 0,
        "forward_completions": 0,
        "candidate_attempts": 0,
        "candidate_completions": 0,
        "replay_attempts": 0,
        "replay_completions": 0,
        "lowered_callable_invocations": 0,
    }


class _JournalProof:
    __slots__ = ("_counters", "_executable", "_stage")

    def __init__(
        self,
        *,
        stage: str,
        counters: dict[str, int],
        executable: Any,
        token: object,
    ) -> None:
        if token is not _JOURNAL_PROOF_TOKEN:
            raise RuntimeError("journal proofs are issued only by a clean checkpoint")
        self._stage = stage
        self._counters = tuple(sorted(counters.items()))
        self._executable = executable


class _CandidateValidationProof:
    __slots__ = ("_authorization_issued", "_counters", "_executable")

    def __init__(
        self,
        *,
        executable: Any,
        counters: dict[str, int],
        token: object,
    ) -> None:
        if token is not _VALIDATION_PROOF_TOKEN:
            raise RuntimeError("candidate validation proofs are issued only by a passed gate")
        self._executable = executable
        self._counters = tuple(sorted(counters.items()))
        self._authorization_issued = False


class _ReplayAuthorization:
    __slots__ = ("_consumed", "_executable")

    def __init__(self, *, executable: Any, token: object) -> None:
        if token is not _REPLAY_AUTHORIZATION_TOKEN:
            raise RuntimeError("replay authorization is issued only after candidate validation")
        self._executable = executable
        self._consumed = False

    def consume_for(self, executable: Any) -> None:
        if self._consumed:
            raise RuntimeError("replay authorization was already consumed")
        # Consume before checking identity: the first attempted replay is the
        # only attempt authorized, including a caller presenting the wrong
        # executable or corrupted counters.
        self._consumed = True
        if executable is not self._executable:
            raise RuntimeError("replay authorization is bound to a different checked executable")


def _expected_counters_after(phase: str) -> dict[str, int]:
    if phase not in _PHASES:
        raise ValueError(f"unsupported dispatch phase: {phase!r}")
    candidate = int(phase in _PHASES)
    replay = int(phase == "replay")
    return {
        "forward_attempts": candidate + replay,
        "forward_completions": candidate + replay,
        "candidate_attempts": candidate,
        "candidate_completions": candidate,
        "replay_attempts": replay,
        "replay_completions": replay,
        "lowered_callable_invocations": 0,
    }


def _exact_contract() -> dict[str, Any]:
    q_shape = [_BATCH_SIZE, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM]
    kv_shape = [_BATCH_SIZE, _SEQUENCE_LENGTH, _KV_HEADS, _HEAD_DIM]
    return {
        "model_family": "Qwen/Qwen3.5-4B",
        "operation": "query_bounded_gqa_forward_candidate_and_replay",
        "inputs": [
            {
                "name": "q",
                "shape": q_shape,
                "dtype": "bfloat16",
                "value": "seeded_dense_nonzero_factorized_random",
            },
            {
                "name": "k",
                "shape": kv_shape,
                "dtype": "bfloat16",
                "value": "seeded_dense_nonzero_factorized_random",
            },
            {
                "name": "v",
                "shape": kv_shape,
                "dtype": "bfloat16",
                "value": "seeded_dense_nonzero_factorized_random",
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
        "randomness_used": True,
        "randomness_location": "host_numpy_only_before_device_put",
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
            "checked_forward_invocations": 2,
            "candidate_invocations": 1,
            "ordinary_replay_invocations": 1,
            "command_buffer_replay_invocations": 0,
            "gpu_reference_invocations": 0,
            "device_error_reduction_invocations": 0,
            "backward_invocations": 0,
        },
        "replay_release_condition": (
            "candidate host numerical gate, duration gate, exact counters, and "
            "current-boot journal checkpoint all pass"
        ),
        "numerical_gate_per_invocation": {
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
        "replay_equality_gate": "host output arrays are exactly equal byte-for-byte",
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=("abstract", "rocm"),
        default="abstract",
        help=(
            "no-GPU refusal by default; guarded ROCm compilation and exactly "
            "two checked invocations require rocm plus --allow-gpu"
        ),
    )
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help=(
            "acknowledge bounded ROCm compilation, one candidate invocation, "
            "and one conditionally released ordinary replay"
        ),
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
    return _runtime_probe()._configure_rocm_environment()


def _xla_flags_summary(value: str | None) -> dict[str, Any]:
    raw = "" if value is None else value
    try:
        token_count: int | None = len(shlex.split(raw, posix=True))
        quoting_valid = True
    except ValueError:
        token_count = None
        quoting_valid = False
    return {
        "present": value is not None,
        "utf8_bytes": len(raw.encode("utf-8")),
        "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "shlex_quoting_valid": quoting_valid,
        "token_count": token_count,
    }


def _environment_manifest(environment: dict[str, str | None]) -> dict[str, Any]:
    expected_fixed = {
        "JAX_PLATFORMS": "rocm",
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
        "JAX_ROCM_VISIBLE_DEVICES": "0",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_CLIENT_MEM_FRACTION": "0.75",
        "SKYRL_ROCM_PALLAS_ATTENTION": "1",
    }
    fixed = {
        name: value if environment.get(name) == value else "<unexpected>" for name, value in expected_fixed.items()
    }
    return {
        "fixed_values": fixed,
        "fixed_values_match_expected": all(environment.get(name) == value for name, value in expected_fixed.items()),
        "xla_flags_original": _xla_flags_summary(environment.get("XLA_FLAGS_original")),
        "xla_flags_effective": _xla_flags_summary(environment.get("XLA_FLAGS_effective")),
        "raw_xla_flags_emitted": False,
    }


def _command_buffer_assignments(tokens: list[str]) -> list[str]:
    assignments = []
    for token in tokens:
        if token == _COMMAND_BUFFER_FLAG_NAME or token.startswith(f"{_COMMAND_BUFFER_FLAG_NAME}="):
            assignments.append(token)
        elif token == _NEGATED_COMMAND_BUFFER_FLAG_NAME or token.startswith(f"{_NEGATED_COMMAND_BUFFER_FLAG_NAME}="):
            assignments.append(token)
    return assignments


def _prove_command_buffers_disabled(
    environment: dict[str, str | None],
) -> dict[str, Any]:
    returned_flags = environment.get("XLA_FLAGS_effective")
    process_flags = os.environ.get("XLA_FLAGS")
    if not isinstance(returned_flags, str) or process_flags is None:
        raise RuntimeError("command-buffer proof requires returned and process XLA_FLAGS")
    if returned_flags != process_flags:
        raise RuntimeError("returned XLA_FLAGS_effective does not match the process environment")
    try:
        tokens = shlex.split(process_flags, posix=True)
    except ValueError as error:
        raise RuntimeError("invalid process XLA_FLAGS quoting") from error
    assignments = _command_buffer_assignments(tokens)
    if assignments != [_DISABLED_COMMAND_BUFFER_FLAG]:
        raise RuntimeError("command buffers are not proven disabled by exactly one sole empty " "assignment")
    return {
        "process_matches_returned": True,
        "shlex_quoting_valid": True,
        "token_count": len(tokens),
        "command_buffer_assignment_count": len(assignments),
        "sole_assignment_is_empty": True,
        "xla_flags_sha256": hashlib.sha256(process_flags.encode("utf-8")).hexdigest(),
        "raw_xla_flags_emitted": False,
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
        preview = ", ".join(imported[:5])
        raise RuntimeError(
            "ROCm replay gate requires a fresh process before JAX/kernel import; " f"already imported: {preview}"
        )


def _load_safety_helpers() -> tuple[
    Callable[[], ContextManager[dict[str, Any]]],
    Callable[[], dict[str, Any]],
]:
    return _runtime_probe()._load_safety_helpers()


def _open_exclusive_output(path: Path) -> TextIO:
    return _runtime_probe()._open_exclusive_output(path)


def _compile_checked_forward(
    jax: Any,
    jnp: Any,
    query_bounded_gqa: Any,
    counters: dict[str, int],
    output: TextIO,
) -> tuple[Any, dict[str, Any]]:
    return _runtime_probe()._compile_checked_forward(
        jax,
        jnp,
        query_bounded_gqa,
        counters,
        _PrivateJsonlWriter(output),
    )


def _allocator_snapshot(jax: Any) -> list[dict[str, Any]]:
    return _runtime_probe()._allocator_snapshot(jax)


def _array_manifest(name: str, value: Any) -> dict[str, Any]:
    return _runtime_probe()._array_manifest(name, value)


def _host_metrics(actual_host: Any, expected_host: Any) -> dict[str, Any]:
    return _runtime_probe()._host_metrics(actual_host, expected_host)


def _journal_checkpoint(
    require_clean_boot: Callable[[], dict[str, Any]],
    output: TextIO,
    stage: str,
    counters: dict[str, int],
) -> dict[str, Any]:
    safety = require_clean_boot()
    if not isinstance(safety, dict) or safety.get("amdgpu_boot_clean") is not True:
        raise RuntimeError(f"{stage} did not prove a clean AMDGPU boot")
    if safety.get("fatal_amdgpu_events") != []:
        raise RuntimeError(f"{stage} returned an invalid fatal-event proof")
    public_safety = {
        "amdgpu_boot_clean": True,
        "fatal_amdgpu_events": [],
    }
    _emit(
        {
            "record_type": "journal_checkpoint",
            "timestamp": _utc_now(),
            "stage": stage,
            "safety": public_safety,
            "counters": dict(counters),
        },
        output,
    )
    return public_safety


def _clean_journal_proof(
    require_clean_boot: Callable[[], dict[str, Any]],
    output: TextIO,
    stage: str,
    counters: dict[str, int],
    *,
    executable: Any,
) -> _JournalProof:
    safety = _journal_checkpoint(require_clean_boot, output, stage, counters)
    if not isinstance(safety, dict) or safety.get("amdgpu_boot_clean") is not True:
        raise RuntimeError(f"{stage} did not prove a clean AMDGPU boot")
    if safety.get("fatal_amdgpu_events") != []:
        raise RuntimeError(f"{stage} returned an invalid fatal-event proof")
    return _JournalProof(
        stage=stage,
        counters=counters,
        executable=executable,
        token=_JOURNAL_PROOF_TOKEN,
    )


def _issue_replay_authorization(
    executable: Any,
    *,
    validation_proof: _CandidateValidationProof,
    dispatch_journal_proof: _JournalProof,
    device_get_journal_proof: _JournalProof,
    counters: dict[str, int],
) -> _ReplayAuthorization:
    expected = _expected_counters_after("candidate")
    expected_snapshot = tuple(sorted(expected.items()))
    if type(validation_proof) is not _CandidateValidationProof:
        raise RuntimeError("candidate validation proof is required for replay")
    if validation_proof._authorization_issued:
        raise RuntimeError("candidate validation proof already issued authorization")
    if validation_proof._executable is not executable:
        raise RuntimeError("candidate validation proof is bound to another executable")
    if validation_proof._counters != expected_snapshot or counters != expected:
        raise RuntimeError("candidate counters changed before replay authorization")
    required_journals = (
        (dispatch_journal_proof, "after_candidate_dispatch"),
        (device_get_journal_proof, "after_candidate_device_get"),
    )
    for proof, expected_stage in required_journals:
        if type(proof) is not _JournalProof:
            raise RuntimeError(f"missing opaque journal proof for {expected_stage}")
        if (
            proof._stage != expected_stage
            or proof._counters != expected_snapshot
            or proof._executable is not executable
        ):
            raise RuntimeError(f"invalid journal proof for {expected_stage}")
    validation_proof._authorization_issued = True
    return _ReplayAuthorization(
        executable=executable,
        token=_REPLAY_AUTHORIZATION_TOKEN,
    )


def _random_nonzero_scalars(rng: Any, shape: tuple[int, ...], maximum: int) -> Any:
    import ml_dtypes
    import numpy as np

    magnitude = rng.integers(1, maximum + 1, size=shape, dtype=np.int16)
    sign = rng.choice(np.asarray((-1, 1), dtype=np.int16), size=shape)
    return ((magnitude * sign).astype(np.float32) / 128.0).astype(ml_dtypes.bfloat16)


def _construct_host_inputs() -> tuple[tuple[Any, Any, Any, Any], list[dict[str, Any]], Any, dict[str, Any]]:
    """Build dense random-looking BF16 inputs with an independent cheap oracle.

    Each token/head vector is a nonzero random BF16 scalar times a seeded +/-1
    feature direction.  The actual kernel still reads and multiplies every one
    of the 256 feature values.  The exact factorization lets the CPU oracle
    reduce QK to scalar outer products and AV to scalar matrix-vector products,
    avoiding a second accelerator implementation or a billion-operation host
    reference.
    """
    import ml_dtypes
    import numpy as np

    rng = np.random.default_rng(_SEED)
    k_directions = rng.choice(np.asarray((-1, 1), dtype=np.int8), size=(_KV_HEADS, _HEAD_DIM))
    q_directions = np.empty((_QUERY_HEADS, _HEAD_DIM), dtype=np.int8)
    for query_head in range(_QUERY_HEADS):
        kv_head = query_head // _GROUP_SIZE
        direction = k_directions[kv_head].copy()
        # Controlled nonzero correlations exercise distinct logits for all four
        # Q heads in a GQA group while retaining seeded random feature patterns.
        flip_count = 48 + 8 * (query_head % _GROUP_SIZE)
        flip_indices = rng.choice(_HEAD_DIM, size=flip_count, replace=False)
        direction[flip_indices] *= -1
        q_directions[query_head] = direction
    v_directions = rng.choice(np.asarray((-1, 1), dtype=np.int8), size=(_KV_HEADS, _HEAD_DIM))

    q_scalars = _random_nonzero_scalars(rng, (_SEQUENCE_LENGTH, _QUERY_HEADS), maximum=48)
    k_scalars = _random_nonzero_scalars(rng, (_SEQUENCE_LENGTH, _KV_HEADS), maximum=48)
    v_scalars = _random_nonzero_scalars(rng, (_SEQUENCE_LENGTH, _KV_HEADS), maximum=96)
    q = (q_scalars[None, :, :, None] * q_directions[None, None, :, :]).astype(ml_dtypes.bfloat16)
    k = (k_scalars[None, :, :, None] * k_directions[None, None, :, :]).astype(ml_dtypes.bfloat16)
    v = (v_scalars[None, :, :, None] * v_directions[None, None, :, :]).astype(ml_dtypes.bfloat16)
    key_mask = np.ones((_BATCH_SIZE, _SEQUENCE_LENGTH), dtype=np.int32)
    host_inputs = (q, k, v, key_mask)
    manifests = [
        _array_manifest(name, value) for name, value in zip(("q", "k", "v", "key_mask"), host_inputs, strict=True)
    ]

    q_scalar_fp32 = np.asarray(q_scalars, dtype=np.float32)
    k_scalar_fp32 = np.asarray(k_scalars, dtype=np.float32)
    v_scalar_fp32 = np.asarray(v_scalars, dtype=np.float32)
    expected = np.empty(
        (_BATCH_SIZE, _SEQUENCE_LENGTH, _QUERY_HEADS, _HEAD_DIM),
        dtype=np.float32,
    )
    causal_invalid = np.triu(np.ones((_SEQUENCE_LENGTH, _SEQUENCE_LENGTH), dtype=np.bool_), k=1)
    q_to_kv = np.arange(_QUERY_HEADS, dtype=np.int32) // _GROUP_SIZE
    mapped_k_directions = k_directions[q_to_kv]
    dot_factors = np.sum(
        q_directions.astype(np.int32) * mapped_k_directions.astype(np.int32),
        axis=-1,
        dtype=np.int32,
    ).astype(np.float32) * (_HEAD_DIM**-0.5)
    if np.any(dot_factors == 0.0):
        raise RuntimeError("constructed Q/K direction correlation must be nonzero")
    for query_head, kv_head in enumerate(q_to_kv):
        logits = q_scalar_fp32[:, query_head, None] * k_scalar_fp32[None, :, kv_head] * dot_factors[query_head]
        logits[causal_invalid] = -np.inf
        row_max = np.max(logits, axis=-1, keepdims=True)
        probabilities = np.exp(logits - row_max).astype(np.float32)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        weighted_scalar = probabilities @ v_scalar_fp32[:, kv_head]
        expected[0, :, query_head, :] = weighted_scalar[:, None] * v_directions[kv_head][None, :]
    expected_manifest = _array_manifest("expected", expected)
    oracle = {
        "factorization": "token_head_scalar_times_seeded_int8_sign_direction",
        "qk_direction_dot_factors_after_scale": [float(value) for value in dot_factors],
        "softmax_accumulator_dtype": "float32",
        "random_generator": "numpy.default_rng(PCG64)",
        "seed": _SEED,
    }
    return host_inputs, manifests, expected, {**expected_manifest, "oracle": oracle}


def _device_put_inputs(jax: Any, host_inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    return jax.block_until_ready(jax.device_put(host_inputs))


def _dispatch_checked_phase(
    jax: Any,
    executable: Any,
    arguments: tuple[Any, ...],
    *,
    phase: str,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    output: TextIO,
    replay_authorization: _ReplayAuthorization | None = None,
) -> tuple[Any, float, _JournalProof]:
    if phase not in _PHASES:
        raise ValueError(f"unsupported dispatch phase: {phase!r}")
    if phase == "candidate" and replay_authorization is not None:
        raise RuntimeError("candidate dispatch cannot accept replay authorization")
    if phase == "replay":
        if type(replay_authorization) is not _ReplayAuthorization:
            raise RuntimeError("opaque replay authorization is required")
        _ReplayAuthorization.consume_for(replay_authorization, executable)
    expected_before = _zero_counters() if phase == "candidate" else _expected_counters_after("candidate")
    if counters != expected_before:
        raise RuntimeError(f"refusing out-of-order {phase} dispatch: {counters!r} != " f"{expected_before!r}")

    counters[f"{phase}_attempts"] += 1
    attempt_start = time.perf_counter()
    dispatch_start: float | None = None

    def emit_started() -> None:
        nonlocal dispatch_start
        _emit(
            {
                "record_type": "dispatch_started",
                "timestamp": _utc_now(),
                "label": phase,
                "counters": dict(counters),
            },
            output,
        )
        dispatch_start = time.perf_counter()

    try:
        result = executable.invoke(jax, *arguments, on_started=emit_started)
        counters[f"{phase}_completions"] += 1
    finally:
        dispatch_seconds = time.perf_counter() - (dispatch_start if dispatch_start is not None else attempt_start)
        journal_proof = _clean_journal_proof(
            require_clean_boot,
            output,
            f"after_{phase}_dispatch",
            counters,
            executable=executable,
        )
    _emit(
        {
            "record_type": "dispatch",
            "timestamp": _utc_now(),
            "label": phase,
            "seconds": dispatch_seconds,
            "counters": dict(counters),
        },
        output,
    )
    return result, dispatch_seconds, journal_proof


def _device_get_with_journal_proof(
    jax: Any,
    value: Any,
    *,
    phase: str,
    executable: Any,
    require_clean_boot: Callable[[], dict[str, Any]],
    counters: dict[str, int],
    output: TextIO,
) -> tuple[Any, _JournalProof]:
    if phase not in _PHASES:
        raise ValueError(f"unsupported device-get phase: {phase!r}")
    try:
        host_value = jax.device_get(value)
    finally:
        journal_proof = _clean_journal_proof(
            require_clean_boot,
            output,
            f"after_{phase}_device_get",
            counters,
            executable=executable,
        )
    return host_value, journal_proof


def _numerical_gate(metrics: dict[str, Any], dispatch_seconds: float) -> dict[str, Any]:
    numerical_passed = (
        metrics["finite"]
        and math.isfinite(metrics["relative_l2"])
        and metrics["relative_l2"] < _MAX_RELATIVE_L2
        and math.isfinite(metrics["cosine"])
        and metrics["cosine"] >= _MIN_COSINE
        and math.isfinite(metrics["max_abs"])
        and metrics["max_abs"] <= _MAX_ABSOLUTE_ERROR
    )
    duration_passed = math.isfinite(dispatch_seconds) and (dispatch_seconds < _MAX_DISPATCH_SECONDS)
    return {
        "numerical_passed": numerical_passed,
        "duration_passed": duration_passed,
        "passed": numerical_passed and duration_passed,
    }


def _validate_phase(
    phase: str,
    actual_host: Any,
    expected_host: Any,
    dispatch_seconds: float,
    counters: dict[str, int],
    output: TextIO,
    *,
    executable: Any,
) -> tuple[dict[str, Any], dict[str, Any], _CandidateValidationProof | None]:
    if phase not in _PHASES:
        raise ValueError(f"unsupported validation phase: {phase!r}")
    metrics = _host_metrics(actual_host, expected_host)
    gate = _numerical_gate(metrics, dispatch_seconds)
    _emit(
        {
            "record_type": "numerical_validation",
            "timestamp": _utc_now(),
            "phase": phase,
            "status": "passed" if gate["passed"] else "failed",
            "thresholds": {
                "finite_required": True,
                "relative_l2_strictly_below": _MAX_RELATIVE_L2,
                "minimum_cosine": _MIN_COSINE,
                "maximum_absolute_error": _MAX_ABSOLUTE_ERROR,
                "dispatch_seconds_strictly_below": _MAX_DISPATCH_SECONDS,
            },
            "metrics": metrics,
            "gates": gate,
            "dispatch_seconds": dispatch_seconds,
            "counters": dict(counters),
        },
        output,
    )
    if not gate["passed"]:
        raise RuntimeError(f"{phase} numerical or duration gate failed")
    expected_counters = _expected_counters_after(phase)
    if counters != expected_counters:
        raise RuntimeError(f"{phase} invocation counter contract violated: {counters!r} != " f"{expected_counters!r}")
    validation_proof = (
        _CandidateValidationProof(
            executable=executable,
            counters=counters,
            token=_VALIDATION_PROOF_TOKEN,
        )
        if phase == "candidate"
        else None
    )
    return metrics, gate, validation_proof


def _replay_equality(candidate_host: Any, replay_host: Any) -> dict[str, Any]:
    import numpy as np

    candidate = np.asarray(candidate_host)
    replay = np.asarray(replay_host)
    same_shape = candidate.shape == replay.shape
    same_dtype = candidate.dtype == replay.dtype
    byte_equal = same_shape and same_dtype and bool(np.array_equal(candidate, replay))
    candidate_sha256 = hashlib.sha256(candidate.tobytes(order="C")).hexdigest()
    replay_sha256 = hashlib.sha256(replay.tobytes(order="C")).hexdigest()
    return {
        "same_shape": same_shape,
        "same_dtype": same_dtype,
        "byte_equal": byte_equal,
        "candidate_sha256": candidate_sha256,
        "replay_sha256": replay_sha256,
        "passed": byte_equal and candidate_sha256 == replay_sha256,
    }


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
    _journal_checkpoint(require_clean_boot, output, "after_backend_initialization", counters)

    executable, compile_report = _compile_checked_forward(jax, jnp, query_bounded_gqa, counters, output)
    _journal_checkpoint(require_clean_boot, output, "after_forward_compile", counters)
    host_inputs, input_manifests, expected_host, expected_manifest = _construct_host_inputs()
    _emit(
        {
            "record_type": "host_factorized_reference",
            "timestamp": _utc_now(),
            "construction": {
                "q_k_v": ("dense BF16 seeded nonzero token/head scalars times seeded " "+/-1 feature directions"),
                "key_mask": "all int32 ones",
                "expected": "host FP32 factorized causal GQA softmax and AV",
                "accelerator_rng_used": False,
                "seed": _SEED,
            },
            "inputs": input_manifests,
            "expected": expected_manifest,
            "counters": dict(counters),
        },
        output,
    )
    try:
        inputs = _device_put_inputs(jax, host_inputs)
    finally:
        _journal_checkpoint(require_clean_boot, output, "after_explicit_input_device_put", counters)
    allocator_before = _allocator_snapshot(jax)

    candidate, candidate_seconds, candidate_dispatch_journal_proof = _dispatch_checked_phase(
        jax,
        executable,
        inputs,
        phase="candidate",
        require_clean_boot=require_clean_boot,
        counters=counters,
        output=output,
    )
    candidate_host, candidate_device_get_journal_proof = _device_get_with_journal_proof(
        jax,
        candidate,
        phase="candidate",
        executable=executable,
        require_clean_boot=require_clean_boot,
        counters=counters,
        output=output,
    )
    candidate_metrics, candidate_gate, candidate_validation_proof = _validate_phase(
        "candidate",
        candidate_host,
        expected_host,
        candidate_seconds,
        counters,
        output,
        executable=executable,
    )
    if candidate_validation_proof is None:
        raise RuntimeError("candidate validation did not issue an opaque proof")
    replay_authorization = _issue_replay_authorization(
        executable,
        validation_proof=candidate_validation_proof,
        dispatch_journal_proof=candidate_dispatch_journal_proof,
        device_get_journal_proof=candidate_device_get_journal_proof,
        counters=counters,
    )
    _emit(
        {
            "record_type": "replay_release_gate",
            "timestamp": _utc_now(),
            "status": "released",
            "reason": "candidate numerical, duration, journal, and counter gates passed",
            "command_buffers_enabled": False,
            "same_compiled_executable": True,
            "counters": dict(counters),
        },
        output,
    )

    replay, replay_seconds, _replay_dispatch_journal_proof = _dispatch_checked_phase(
        jax,
        executable,
        inputs,
        phase="replay",
        require_clean_boot=require_clean_boot,
        counters=counters,
        output=output,
        replay_authorization=replay_authorization,
    )
    replay_host, _replay_device_get_journal_proof = _device_get_with_journal_proof(
        jax,
        replay,
        phase="replay",
        executable=executable,
        require_clean_boot=require_clean_boot,
        counters=counters,
        output=output,
    )
    replay_metrics, replay_gate, replay_validation_proof = _validate_phase(
        "replay",
        replay_host,
        expected_host,
        replay_seconds,
        counters,
        output,
        executable=executable,
    )
    if replay_validation_proof is not None:
        raise RuntimeError("replay validation unexpectedly issued authorization")
    equality = _replay_equality(candidate_host, replay_host)
    _emit(
        {
            "record_type": "replay_equality_validation",
            "timestamp": _utc_now(),
            "status": "passed" if equality["passed"] else "failed",
            "equality": equality,
            "counters": dict(counters),
        },
        output,
    )
    if not equality["passed"]:
        raise RuntimeError("candidate and replay outputs are not byte-identical")

    _emit(
        {
            "record_type": "runtime_passed",
            "timestamp": _utc_now(),
            "status": "passed_exact_t512_nonzero_candidate_and_replay",
            "compile_structural_gate": compile_report["structural_gate"],
            "compiled_memory_gate": compile_report["compiled_memory_gate"],
            "candidate": {
                "metrics": candidate_metrics,
                "gate": candidate_gate,
                "dispatch_seconds": candidate_seconds,
            },
            "replay": {
                "metrics": replay_metrics,
                "gate": replay_gate,
                "dispatch_seconds": replay_seconds,
                "equality": equality,
            },
            "allocator_before_execution": allocator_before,
            "allocator_after_validation": _allocator_snapshot(jax),
            "counters": dict(counters),
            "backward_invocations": 0,
            "gpu_reference_invocations": 0,
            "device_error_reduction_invocations": 0,
            "command_buffer_replay_invocations": 0,
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
                "abstract_refusal" if args.platform == "abstract" else "guarded_exact_t512_nonzero_forward_replay"
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
                    "safety": safety_preflight,
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
