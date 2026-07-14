#!/usr/bin/env python3
"""One-iteration real-sampler GRPO systems gate for local Qwen3.5-4B.

Unlike :mod:`bench_grpo`, this gate includes the sampler lifecycle.  It creates
one rank-8 LoRA adapter, exports its initial sampler view, obtains two real
16-token completions, builds the exact causal importance-sampling datums from
their returned log-probabilities, performs one forward/backward plus Adam
update, exports the post-update sampler view, and unloads the adapter.

The geometry is deliberately fixed: a 49-token prompt plus 16 generated tokens
produces a shifted learner context of 64.  The current Qwen launcher pins
``sample_max_num_sequences=1``, so the two samples execute as two B1 generator
calls.  The full generator specialization signature is recorded explicitly;
changing that server setting without changing this gate invalidates the run.

This is a cold lifecycle and safety gate, not steady-state throughput or RL
quality evidence.  It emits phase timings and the actual sampled token IDs and
old-policy log-probabilities, but never derives a tokens/s claim from one cold
iteration.  Before adapter creation and again after cleanup, it binds the
loopback endpoint to the hardened source-snapshot launcher/API/READY-engine
process tree, pinned model blobs, exact backend/Pallas/cache environment, live
launch lock, and graph-free XLA policy.  Start the server with a nonempty
``SKYRL_QWEN35_PREWARM_BUCKETS`` list containing 64 and
``SKYRL_QWEN35_ENGINE_T64_CACHE_ATTEST=1``; the gate independently revalidates
the resulting strict AOT cache-hit artifacts.  The unqualified normal launcher
and cache-opt-out modes are intentionally rejected as benchmark provenance.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import math
import os
import re
import statistics
import struct
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import tinker
from tinker.lib.retry_handler import RetryConfig

ROCM_DIR = Path(__file__).resolve().parent
if str(ROCM_DIR) not in sys.path:
    sys.path.insert(0, str(ROCM_DIR))

from bench_support import (  # noqa: E402
    MODEL,
    REPO,
    WORKSPACE,
    git_metadata,
    json_dumps,
    json_safe,
    package_version,
    redact_url,
    redacted_argv,
    round_up_seq_len,
    safe_accelerator_environment,
    unload_adapter,
)
from local_server_attestation import (  # noqa: E402
    EXPECTED_MODEL_CACHE_DIRECTORY,
    EXPECTED_MODEL_REVISION,
    attest_local_server,
    revalidate_local_server,
)

SCHEMA_VERSION = 1
PROMPT_TOKENS = 49
MAX_NEW_TOKENS = 16
GROUP_SIZE = 2
LORA_RANK = 8
SAMPLE_MAX_NUM_SEQUENCES = 1
LEARNER_CONTEXT = PROMPT_TOKENS + MAX_NEW_TOKENS - 1
EXPECTED_TINKER_VERSION = "0.22.4"

T = TypeVar("T")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _emit(output, record: dict[str, Any]) -> None:
    output.write(json_dumps(record, separators=(",", ":")) + "\n")
    output.flush()


def _error_evidence(error: BaseException) -> dict[str, Any]:
    encoded = str(error).encode("utf-8", errors="replace")
    return {
        "error_type": type(error).__name__,
        "error_message_redacted": True,
        "error_message_utf8_bytes": len(encoded),
        "error_message_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _client_source_revision() -> tuple[str, str]:
    environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/nonexistent",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "XDG_CONFIG_HOME": "/nonexistent",
    }
    prefix = (
        "/usr/bin/git",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-C",
        str(REPO),
    )

    def run(*arguments: str) -> bytes:
        try:
            result = subprocess.run(
                [*prefix, *arguments],
                capture_output=True,
                check=False,
                env=environment,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(
                f"cannot inspect benchmark Git source: {type(error).__name__}"
            ) from error
        if result.returncode != 0 or result.stderr:
            raise RuntimeError("benchmark Git source inspection failed")
        return result.stdout

    top_level = run("rev-parse", "--show-toplevel")
    head = run("rev-parse", "--verify", "HEAD^{commit}")
    tree = run("rev-parse", "--verify", "HEAD^{tree}")
    status = run(
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    try:
        top_level_text = top_level.decode("utf-8").removesuffix("\n")
        head_text = head.decode("ascii").removesuffix("\n")
        tree_text = tree.decode("ascii").removesuffix("\n")
    except UnicodeError as error:
        raise RuntimeError("benchmark Git source output is not canonical") from error
    if (
        top_level_text != str(REPO)
        or top_level != f"{REPO}\n".encode()
        or head != f"{head_text}\n".encode("ascii")
        or tree != f"{tree_text}\n".encode("ascii")
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", head_text) is None
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", tree_text) is None
        or len(head_text) != len(tree_text)
        or status != b""
    ):
        raise RuntimeError(
            "benchmark requires one clean canonical Git HEAD/tree checkout"
        )
    return head_text, tree_text


def _get_tokenizer(model_snapshot: Path) -> Any:
    """Load the tokenizer from the exact snapshot verified by server attestation."""
    snapshot = Path(model_snapshot)
    expected_suffix = (
        EXPECTED_MODEL_CACHE_DIRECTORY,
        "snapshots",
        EXPECTED_MODEL_REVISION,
    )
    try:
        resolved = snapshot.resolve(strict=True)
    except OSError as error:
        raise RuntimeError("verified tokenizer snapshot is unavailable") from error
    if (
        not snapshot.is_absolute()
        or resolved != snapshot
        or tuple(snapshot.parts[-3:]) != expected_suffix
        or not snapshot.is_dir()
    ):
        raise RuntimeError("verified tokenizer snapshot path is not canonical")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)


def _repeat_tokens(seed: list[int], length: int) -> tuple[int, ...]:
    if length <= 0:
        raise ValueError("prompt token length must be positive")
    if not seed:
        raise ValueError("tokenizer returned no prompt tokens")
    return tuple(int(seed[index % len(seed)]) for index in range(length))


def _build_prompt(
    model_snapshot: Path, tokenizer: Any | None = None
) -> tuple[int, ...]:
    tokenizer = tokenizer or _get_tokenizer(model_snapshot)
    seed = tokenizer.encode(" real sampler GRPO systems gate", add_special_tokens=False)
    return _repeat_tokens(seed, PROMPT_TOKENS)


async def _save_exact_sampler_snapshot(training_client: Any) -> Any:
    """Submit once logically; ingress retries retain one server sequence ID."""
    return await training_client.save_weights_and_get_sampling_client_async(
        retry_config=RetryConfig(enable_retry_logic=False)
    )


def _sampling_specialization(
    *,
    prompt_tokens: int,
    max_new_tokens: int,
    group_size: int,
    sample_max_num_sequences: int,
    top_k: int,
    top_p: float,
    prompt_logprobs: bool,
    stop_token_width: int,
) -> dict[str, Any]:
    """Describe every known shape/static generator specialization dimension."""
    integer_values = (
        prompt_tokens,
        max_new_tokens,
        group_size,
        sample_max_num_sequences,
        stop_token_width,
    )
    if any(type(value) is not int for value in integer_values):
        raise TypeError("sampling specialization integer fields must be exact ints")
    if min(prompt_tokens, max_new_tokens, group_size, sample_max_num_sequences) <= 0:
        raise ValueError("sampling specialization dimensions must be positive")
    if stop_token_width < 0:
        raise ValueError("stop_token_width must be nonnegative")
    if type(prompt_logprobs) is not bool:
        raise TypeError("prompt_logprobs must be an exact bool")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and in (0, 1]")

    prompt_bucket = round_up_seq_len(prompt_tokens)
    total_kv_bucket = round_up_seq_len(prompt_bucket + max_new_tokens)
    return {
        "expanded_sample_count": group_size,
        "effective_sample_microbatch_size": sample_max_num_sequences,
        "generator_call_count": math.ceil(group_size / sample_max_num_sequences),
        "generator_call_count_evidence": (
            "explicit_sequential_single_sample_requests"
        ),
        "prompt_tokens": prompt_tokens,
        "prompt_bucket_tokens": prompt_bucket,
        "total_kv_bucket_tokens": total_kv_bucket,
        "max_new_tokens": max_new_tokens,
        "max_top_k": top_k if top_k > 0 else 0,
        "use_top_p": top_p < 1.0,
        "prompt_logprobs": prompt_logprobs,
        "stop_token_width": stop_token_width,
        "server_contract": (
            "effective_sample_microbatch_size must equal the JAX backend "
            "sample_max_num_sequences setting"
        ),
    }


@dataclass(frozen=True, slots=True)
class SampledGroup:
    completions: tuple[tuple[int, ...], ...]
    logprobs: tuple[tuple[float, ...], ...]
    stop_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SampleResponseGroup:
    sequences: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class RealRolloutBatch:
    datums: tuple[tinker.Datum, ...]
    prompt_tokens: tuple[int, ...]
    completion_tokens: tuple[tuple[int, ...], ...]
    sampling_logprobs: tuple[tuple[float, ...], ...]
    rewards: tuple[float, ...]
    advantages: tuple[float, ...]
    context: int


@dataclass(frozen=True, slots=True)
class _Finalization:
    cleanup_seconds: float | None
    adapter_lifecycle_seconds: float | None
    cleanup_error: BaseException | None
    revalidation_error: BaseException | None
    revalidation_durations: dict[str, float]


def _stop_reason_text(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).lower()


def _extract_sampled_group(
    response: Any,
    *,
    group_size: int = GROUP_SIZE,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> SampledGroup:
    sequences = getattr(response, "sequences", None)
    if not isinstance(sequences, (list, tuple)) or len(sequences) != group_size:
        raise ValueError("sampler returned the wrong number of sequences")

    completions: list[tuple[int, ...]] = []
    logprob_groups: list[tuple[float, ...]] = []
    stop_reasons: list[str] = []
    for sequence in sequences:
        raw_tokens = getattr(sequence, "tokens", None)
        raw_logprobs = getattr(sequence, "logprobs", None)
        if not isinstance(raw_tokens, list) or len(raw_tokens) != max_new_tokens:
            raise ValueError("sampler must return exactly max_new_tokens tokens")
        if not isinstance(raw_logprobs, list) or len(raw_logprobs) != max_new_tokens:
            raise ValueError("sampler must return one log-probability per token")
        if any(
            type(token) is bool or not isinstance(token, int) or token < 0
            for token in raw_tokens
        ):
            raise TypeError("sampled token IDs must be nonnegative exact integers")
        converted_logprobs = tuple(float(value) for value in raw_logprobs)
        if not all(math.isfinite(value) for value in converted_logprobs):
            raise FloatingPointError("sampler returned a non-finite log-probability")
        if any(value > 0.0 for value in converted_logprobs):
            raise ValueError("sampler returned a positive log-probability")
        stop_reason = _stop_reason_text(getattr(sequence, "stop_reason", ""))
        if stop_reason != "length":
            raise ValueError("sampler did not terminate at the exact token limit")
        completions.append(tuple(raw_tokens))
        logprob_groups.append(converted_logprobs)
        stop_reasons.append(stop_reason)

    if len(set(completions)) != group_size:
        raise RuntimeError("real sampler completions are not distinct")
    return SampledGroup(
        completions=tuple(completions),
        logprobs=tuple(logprob_groups),
        stop_reasons=tuple(stop_reasons),
    )


def _rank_rewards(completions: tuple[tuple[int, ...], ...]) -> tuple[float, ...]:
    if len(completions) < 2 or len(set(completions)) != len(completions):
        raise ValueError("reward ranking requires at least two distinct completions")
    ordered_indices = sorted(range(len(completions)), key=completions.__getitem__)
    denominator = len(completions) - 1
    rewards = [0.0] * len(completions)
    for rank, index in enumerate(ordered_indices):
        rewards[index] = rank / denominator
    return tuple(rewards)


def _build_real_rollouts(
    prompt_tokens: tuple[int, ...],
    sampled: SampledGroup,
) -> RealRolloutBatch:
    if len(prompt_tokens) != PROMPT_TOKENS:
        raise ValueError(f"prompt must contain exactly {PROMPT_TOKENS} tokens")
    rewards = _rank_rewards(sampled.completions)
    mean_reward = math.fsum(rewards) / len(rewards)
    advantages = tuple(reward - mean_reward for reward in rewards)
    if not math.isclose(math.fsum(advantages), 0.0, abs_tol=1e-12):
        raise AssertionError("group-centered advantages do not sum to zero")

    datums: list[tinker.Datum] = []
    prompt_target_count = len(prompt_tokens) - 1
    for completion, old_action_logprobs, advantage in zip(
        sampled.completions,
        sampled.logprobs,
        advantages,
        strict=True,
    ):
        full_sequence = (*prompt_tokens, *completion)
        model_tokens = list(full_sequence[:-1])
        target_tokens = list(full_sequence[1:])
        old_logprobs = [0.0] * prompt_target_count + list(old_action_logprobs)
        token_advantages = [0.0] * prompt_target_count + [advantage] * len(completion)
        if not (
            len(model_tokens)
            == len(target_tokens)
            == len(old_logprobs)
            == len(token_advantages)
            == LEARNER_CONTEXT
        ):
            raise AssertionError("sampled rollout causal alignment is inconsistent")
        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(model_tokens),
            loss_fn_inputs={
                "target_tokens": target_tokens,
                "logprobs": old_logprobs,
                "advantages": token_advantages,
            },
        )
        if "weights" in datum.loss_fn_inputs:
            raise AssertionError("real GRPO datums must omit explicit weights")
        datums.append(datum)

    return RealRolloutBatch(
        datums=tuple(datums),
        prompt_tokens=prompt_tokens,
        completion_tokens=sampled.completions,
        sampling_logprobs=sampled.logprobs,
        rewards=rewards,
        advantages=advantages,
        context=LEARNER_CONTEXT,
    )


def _tensor_values(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        result = value.tolist()
    elif hasattr(value, "data"):
        result = value.data
    else:
        result = value
    if not isinstance(result, list) or any(isinstance(item, list) for item in result):
        raise TypeError("expected a one-dimensional tensor-like value")
    return [float(item) for item in result]


def _learner_metrics(
    outputs: list[dict[str, Any]], batch: RealRolloutBatch
) -> dict[str, float]:
    if len(outputs) != len(batch.datums):
        raise ValueError("learner returned the wrong number of rollout outputs")
    action_logprobs: list[float] = []
    ratios: list[float] = []
    for output, datum in zip(outputs, batch.datums, strict=True):
        target_logprobs = _tensor_values(output["logprobs"])
        old_logprobs = _tensor_values(datum.loss_fn_inputs["logprobs"])
        if len(target_logprobs) != LEARNER_CONTEXT:
            raise ValueError("learner returned the wrong token dimension")
        for target, old in zip(
            target_logprobs[-MAX_NEW_TOKENS:],
            old_logprobs[-MAX_NEW_TOKENS:],
            strict=True,
        ):
            if not math.isfinite(target):
                raise FloatingPointError(
                    "learner returned a non-finite log-probability"
                )
            if target > 0.0:
                raise ValueError("learner returned a positive log-probability")
            ratio = math.exp(target - old)
            if not math.isfinite(ratio):
                raise FloatingPointError("learner importance ratio is non-finite")
            action_logprobs.append(target)
            ratios.append(ratio)
    return {
        "action_target_logprob_mean": statistics.fmean(action_logprobs),
        "importance_ratio_mean": statistics.fmean(ratios),
        "importance_ratio_min": min(ratios),
        "importance_ratio_max": max(ratios),
    }


def _validated_optimizer_metrics(value: Any) -> dict[str, Any]:
    metrics = json_safe(value)
    if not isinstance(metrics, dict):
        raise TypeError("optimizer metrics must be a mapping")
    grad_norm = metrics.get("skyrl.ai/grad_norm")
    if isinstance(grad_norm, bool) or not isinstance(grad_norm, (int, float)):
        raise ValueError("optimizer metrics are missing the numeric gradient norm")
    grad_norm = float(grad_norm)
    if not math.isfinite(grad_norm) or grad_norm <= 0.0:
        raise FloatingPointError("optimizer gradient norm must be finite and positive")
    metrics["skyrl.ai/grad_norm"] = grad_norm
    return metrics


async def _timed_async_phase(
    output,
    *,
    run_id: str,
    phase: str,
    action: Callable[[], Awaitable[T]],
    durations: dict[str, float],
) -> T:
    wall_start_ns = time.time_ns()
    monotonic_start_ns = time.perf_counter_ns()
    try:
        result = await action()
    except BaseException as error:
        monotonic_end_ns = time.perf_counter_ns()
        try:
            _emit(
                output,
                {
                    "record_type": "phase",
                    "run_id": run_id,
                    "iteration": 0,
                    "phase": phase,
                    "status": "failed",
                    "wall_start_ns": wall_start_ns,
                    "wall_end_ns": time.time_ns(),
                    "monotonic_start_ns": monotonic_start_ns,
                    "monotonic_end_ns": monotonic_end_ns,
                    "duration_seconds": (monotonic_end_ns - monotonic_start_ns) / 1e9,
                    **_error_evidence(error),
                },
            )
        except BaseException as record_error:
            error.add_note(
                "phase failure evidence recording also failed: "
                f"{type(record_error).__name__}"
            )
        raise
    monotonic_end_ns = time.perf_counter_ns()
    duration = (monotonic_end_ns - monotonic_start_ns) / 1e9
    durations[phase] = duration
    _emit(
        output,
        {
            "record_type": "phase",
            "run_id": run_id,
            "iteration": 0,
            "phase": phase,
            "status": "completed",
            "wall_start_ns": wall_start_ns,
            "wall_end_ns": time.time_ns(),
            "monotonic_start_ns": monotonic_start_ns,
            "monotonic_end_ns": monotonic_end_ns,
            "duration_seconds": duration,
        },
    )
    return result


def _timed_sync_phase(
    output,
    *,
    run_id: str,
    phase: str,
    action: Callable[[], T],
    durations: dict[str, float],
) -> T:
    wall_start_ns = time.time_ns()
    monotonic_start_ns = time.perf_counter_ns()
    try:
        result = action()
    except BaseException as error:
        monotonic_end_ns = time.perf_counter_ns()
        try:
            _emit(
                output,
                {
                    "record_type": "phase",
                    "run_id": run_id,
                    "iteration": 0,
                    "phase": phase,
                    "status": "failed",
                    "wall_start_ns": wall_start_ns,
                    "wall_end_ns": time.time_ns(),
                    "monotonic_start_ns": monotonic_start_ns,
                    "monotonic_end_ns": monotonic_end_ns,
                    "duration_seconds": (monotonic_end_ns - monotonic_start_ns) / 1e9,
                    **_error_evidence(error),
                },
            )
        except BaseException as record_error:
            error.add_note(
                "phase failure evidence recording also failed: "
                f"{type(record_error).__name__}"
            )
        raise
    monotonic_end_ns = time.perf_counter_ns()
    duration = (monotonic_end_ns - monotonic_start_ns) / 1e9
    durations[phase] = duration
    _emit(
        output,
        {
            "record_type": "phase",
            "run_id": run_id,
            "iteration": 0,
            "phase": phase,
            "status": "completed",
            "wall_start_ns": wall_start_ns,
            "wall_end_ns": time.time_ns(),
            "monotonic_start_ns": monotonic_start_ns,
            "monotonic_end_ns": monotonic_end_ns,
            "duration_seconds": duration,
        },
    )
    return result


async def _finalize_after_seal(
    output,
    *,
    run_id: str,
    server_seal: Any,
    service_client: Any | None,
    training_client: Any | None,
    adapter_lifecycle_start_ns: int | None,
) -> _Finalization:
    """Clean up any created adapter, then revalidate the exact server seal."""
    cleanup_seconds: float | None = None
    adapter_lifecycle_seconds: float | None = None
    cleanup_error: BaseException | None = None

    if training_client is not None:
        try:
            cleanup_wall_start_ns = time.time_ns()
            cleanup_start_ns = time.perf_counter_ns()
            adapter_unloaded = False
            try:
                if service_client is None:
                    raise RuntimeError("adapter exists without its service client")
                await unload_adapter(service_client, training_client)
                adapter_unloaded = True
            except BaseException as caught:
                cleanup_error = caught
            cleanup_end_ns = time.perf_counter_ns()
            cleanup_seconds = (cleanup_end_ns - cleanup_start_ns) / 1e9
            if adapter_lifecycle_start_ns is not None:
                adapter_lifecycle_seconds = (
                    cleanup_end_ns - adapter_lifecycle_start_ns
                ) / 1e9
            cleanup_record: dict[str, Any] = {
                "record_type": "cleanup",
                "run_id": run_id,
                "wall_start_ns": cleanup_wall_start_ns,
                "wall_end_ns": time.time_ns(),
                "duration_seconds": cleanup_seconds,
                "adapter_unloaded": adapter_unloaded,
            }
            if cleanup_error is not None:
                cleanup_record.update(_error_evidence(cleanup_error))
            try:
                _emit(output, cleanup_record)
            except BaseException as cleanup_record_error:
                if cleanup_error is None:
                    cleanup_error = cleanup_record_error
                else:
                    cleanup_error.add_note(
                        "cleanup evidence recording also failed: "
                        f"{type(cleanup_record_error).__name__}"
                    )
        except BaseException as cleanup_finalization_error:
            if cleanup_error is None:
                cleanup_error = cleanup_finalization_error
            elif cleanup_finalization_error is not cleanup_error:
                cleanup_error.add_note(
                    "cleanup finalization also failed: "
                    f"{type(cleanup_finalization_error).__name__}"
                )

    revalidation_error: BaseException | None = None
    revalidation_durations: dict[str, float] = {}
    try:
        revalidation_record = _timed_sync_phase(
            output,
            run_id=run_id,
            phase="server_revalidation",
            action=lambda: revalidate_local_server(server_seal),
            durations=revalidation_durations,
        )
        _emit(output, revalidation_record)
    except BaseException as caught:
        revalidation_error = caught

    return _Finalization(
        cleanup_seconds=cleanup_seconds,
        adapter_lifecycle_seconds=adapter_lifecycle_seconds,
        cleanup_error=cleanup_error,
        revalidation_error=revalidation_error,
        revalidation_durations=revalidation_durations,
    )


def _raise_preferred_error(
    primary_error: BaseException | None,
    finalization: _Finalization,
) -> None:
    """Raise primary, cleanup, then revalidation errors in precedence order."""
    if primary_error is not None:
        if finalization.cleanup_error is not None:
            primary_error.add_note(
                "adapter cleanup also failed: "
                f"{type(finalization.cleanup_error).__name__}"
            )
        if finalization.revalidation_error is not None:
            primary_error.add_note(
                "server revalidation also failed: "
                f"{type(finalization.revalidation_error).__name__}"
            )
        raise primary_error
    if finalization.cleanup_error is not None:
        if finalization.revalidation_error is not None:
            finalization.cleanup_error.add_note(
                "server revalidation also failed: "
                f"{type(finalization.revalidation_error).__name__}"
            )
        raise finalization.cleanup_error
    if finalization.revalidation_error is not None:
        raise finalization.revalidation_error


async def _run(args: argparse.Namespace, output) -> None:
    tinker_version = package_version("tinker")
    if tinker_version != EXPECTED_TINKER_VERSION:
        raise RuntimeError(
            "real-sampler GRPO retry semantics require exactly tinker "
            f"{EXPECTED_TINKER_VERSION}"
        )
    expected_git_head, expected_git_tree = _client_source_revision()
    completion_seeds = tuple(args.seed + index for index in range(GROUP_SIZE))
    specialization = _sampling_specialization(
        prompt_tokens=PROMPT_TOKENS,
        max_new_tokens=MAX_NEW_TOKENS,
        group_size=GROUP_SIZE,
        sample_max_num_sequences=SAMPLE_MAX_NUM_SEQUENCES,
        top_k=-1,
        top_p=1.0,
        prompt_logprobs=False,
        stop_token_width=0,
    )
    manifest = {
        "record_type": "manifest",
        "schema_version": SCHEMA_VERSION,
        "suite": "real_sampler_grpo_one_iteration",
        "protocol": "one_iteration_gate",
        "warning": (
            "Cold real-sampler systems gate; not steady-state throughput, reward "
            "quality, convergence, or policy-improvement evidence."
        ),
        "run_id": args.run_id,
        "wall_time": _utc_now(),
        "wall_time_ns": time.time_ns(),
        "argv": redacted_argv(sys.argv),
        "model": MODEL,
        "model_revision": EXPECTED_MODEL_REVISION,
        "base_url": redact_url(args.base_url),
        "server_pid": args.server_pid,
        "server_attestation_required": True,
        "server_source_git_head": expected_git_head,
        "server_source_git_tree": expected_git_tree,
        "startup_cache_attestation_required": True,
        "prompt_tokens": PROMPT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "learner_context": LEARNER_CONTEXT,
        "effective_padded_learner_context": round_up_seq_len(LEARNER_CONTEXT),
        "group_size": GROUP_SIZE,
        "lora_rank": LORA_RANK,
        "lora_targets": {
            "attention": True,
            "mlp": True,
            "unembedding": False,
        },
        "seed": args.seed,
        "sampling": {
            "temperature": 1.0,
            "top_k": -1,
            "top_p": 1.0,
            "stop": None,
            "whole_operation_retries": False,
            "submit_retries_require_server_sequence_dedup": True,
            "request_plan": {
                "execution": "sequential",
                "request_count": GROUP_SIZE,
                "num_samples_per_request": 1,
                "seeds": completion_seeds,
            },
            "specialization": specialization,
        },
        "retry_safety": {
            "sdk_version": EXPECTED_TINKER_VERSION,
            "scope": "colocated_jax_ingress_future_enqueue",
            "duplicate_submission_creates_one_future": True,
            "backend_effect_exactly_once_across_engine_crash": False,
            "external_sampling_dispatch_covered": False,
        },
        "reward_rule": "lexicographic completion rank scaled to [0,1], then group mean centered",
        "optimizer": {
            "name": "adam",
            "learning_rate": args.learning_rate,
            "beta1": args.adam_beta1,
            "beta2": args.adam_beta2,
            "eps": args.adam_eps,
        },
        "client": {
            "python": sys.version,
            "tinker": tinker_version,
            "tinker_cookbook": package_version("tinker-cookbook"),
            "benchmark_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "client_environment": safe_accelerator_environment(),
        "repositories": {
            "skyrl": git_metadata(REPO),
            "tinker_cookbook": git_metadata(WORKSPACE / "tinker-cookbook"),
        },
    }
    _emit(output, manifest)

    setup_durations: dict[str, float] = {}
    server_seal: Any | None = None

    def acquire_server_seal() -> Any:
        nonlocal server_seal
        server_seal = attest_local_server(
            server_pid=args.server_pid,
            base_url=args.base_url,
            expected_git_head=expected_git_head,
            expected_git_tree=expected_git_tree,
            expected_repo_root=REPO,
            require_startup_cache=True,
        )
        return server_seal

    try:
        returned_server_seal = _timed_sync_phase(
            output,
            run_id=args.run_id,
            phase="server_attestation",
            action=acquire_server_seal,
            durations=setup_durations,
        )
    except BaseException as attestation_error:
        if server_seal is None:
            raise
        finalization = await _finalize_after_seal(
            output,
            run_id=args.run_id,
            server_seal=server_seal,
            service_client=None,
            training_client=None,
            adapter_lifecycle_start_ns=None,
        )
        _raise_preferred_error(attestation_error, finalization)
        raise AssertionError("unreachable")
    if server_seal is None or returned_server_seal is not server_seal:
        attestation_consistency_error = RuntimeError(
            "server attestation returned an inconsistent seal"
        )
        if server_seal is None:
            raise attestation_consistency_error
        finalization = await _finalize_after_seal(
            output,
            run_id=args.run_id,
            server_seal=server_seal,
            service_client=None,
            training_client=None,
            adapter_lifecycle_start_ns=None,
        )
        _raise_preferred_error(attestation_consistency_error, finalization)
        raise AssertionError("unreachable")

    service_client: Any | None = None
    training_client: Any | None = None
    adapter_lifecycle_start_ns: int | None = None

    async def create_adapter():
        nonlocal training_client
        training_client = await service_client.create_lora_training_client_async(
            base_model=MODEL,
            rank=LORA_RANK,
            seed=args.seed,
            train_mlp=True,
            train_attn=True,
            train_unembed=False,
            user_metadata={
                "suite": "real_sampler_grpo_one_iteration",
                "run_id": args.run_id,
            },
        )
        return training_client

    try:
        prompt_tokens = _build_prompt(server_seal.verified_model_snapshot())
        _emit(output, server_seal.as_record())
        adam = tinker.AdamParams(
            learning_rate=args.learning_rate,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            eps=args.adam_eps,
        )
        service_client = tinker.ServiceClient(
            base_url=args.base_url,
            user_metadata={
                "recipe_name": "qwen35_rocm_real_sampler_grpo_gate",
                "run_id": args.run_id,
            },
        )
        adapter_lifecycle_start_ns = time.perf_counter_ns()
        returned_training_client = await _timed_async_phase(
            output,
            run_id=args.run_id,
            phase="adapter_create",
            action=create_adapter,
            durations=setup_durations,
        )
        if training_client is None or returned_training_client is not training_client:
            raise RuntimeError("adapter creation returned an inconsistent client")
    except BaseException as setup_error:
        finalization = await _finalize_after_seal(
            output,
            run_id=args.run_id,
            server_seal=server_seal,
            service_client=service_client,
            training_client=training_client,
            adapter_lifecycle_start_ns=adapter_lifecycle_start_ns,
        )
        _raise_preferred_error(setup_error, finalization)
        raise AssertionError("unreachable")

    primary_error: BaseException | None = None
    phase_durations: dict[str, float] = {}
    iteration_wall_start_ns = time.time_ns()
    iteration_start_ns = time.perf_counter_ns()
    learner_metrics: dict[str, float] | None = None
    sampled: SampledGroup | None = None
    try:
        sampling_client = await _timed_async_phase(
            output,
            run_id=args.run_id,
            phase="initial_sampler_snapshot",
            action=lambda: _save_exact_sampler_snapshot(training_client),
            durations=phase_durations,
        )
        async def sample_group():
            sequences: list[Any] = []
            for seed in completion_seeds:
                response = await sampling_client.sample_async(
                    prompt=tinker.ModelInput.from_ints(list(prompt_tokens)),
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=MAX_NEW_TOKENS,
                        seed=seed,
                        stop=None,
                        temperature=1.0,
                        top_k=-1,
                        top_p=1.0,
                    ),
                    include_prompt_logprobs=False,
                    topk_prompt_logprobs=0,
                )
                returned = getattr(response, "sequences", None)
                if not isinstance(returned, (list, tuple)) or len(returned) != 1:
                    raise ValueError(
                        "single-sample request returned the wrong sequence count"
                    )
                sequences.append(returned[0])
            return _SampleResponseGroup(sequences=tuple(sequences))

        response = await _timed_async_phase(
            output,
            run_id=args.run_id,
            phase="sample",
            action=sample_group,
            durations=phase_durations,
        )
        sampled = _timed_sync_phase(
            output,
            run_id=args.run_id,
            phase="validate_sample",
            action=lambda: _extract_sampled_group(response),
            durations=phase_durations,
        )
        batch = _timed_sync_phase(
            output,
            run_id=args.run_id,
            phase="grade_and_build",
            action=lambda: _build_real_rollouts(prompt_tokens, sampled),
            durations=phase_durations,
        )
        _emit(
            output,
            {
                "record_type": "sampled_group",
                "run_id": args.run_id,
                "iteration": 0,
                "prompt_tokens": list(prompt_tokens),
                "completion_tokens": [list(item) for item in batch.completion_tokens],
                "sampling_logprobs": [list(item) for item in batch.sampling_logprobs],
                "stop_reasons": list(sampled.stop_reasons),
                "rewards": list(batch.rewards),
                "advantages": list(batch.advantages),
                "learner_context": batch.context,
            },
        )

        async def learner_update() -> tuple[Any, Any]:
            fwd_bwd_future = await training_client.forward_backward_async(
                list(batch.datums),
                loss_fn="importance_sampling",
            )
            try:
                optim_future = await training_client.optim_step_async(adam)
            except BaseException as submission_error:
                try:
                    await fwd_bwd_future.result_async()
                except BaseException as settle_error:
                    submission_error.add_note(
                        "the already-submitted learner future also failed while settling: "
                        f"{type(settle_error).__name__}"
                    )
                raise

            fwd_bwd_result: Any | None = None
            optim_result: Any | None = None
            fwd_bwd_error: BaseException | None = None
            optim_error: BaseException | None = None
            try:
                fwd_bwd_result = await fwd_bwd_future.result_async()
            except BaseException as caught:
                fwd_bwd_error = caught
            try:
                optim_result = await optim_future.result_async()
            except BaseException as caught:
                optim_error = caught

            if fwd_bwd_error is not None:
                if optim_error is not None:
                    fwd_bwd_error.add_note(
                        "the queued optimizer future also failed while settling: "
                        f"{type(optim_error).__name__}"
                    )
                raise fwd_bwd_error
            if optim_error is not None:
                raise optim_error
            if fwd_bwd_result is None or optim_result is None:
                raise RuntimeError("learner futures returned no result")
            return fwd_bwd_result, optim_result

        fwd_bwd_result, optim_result = await _timed_async_phase(
            output,
            run_id=args.run_id,
            phase="learner_update",
            action=learner_update,
            durations=phase_durations,
        )
        learner_metrics = _learner_metrics(fwd_bwd_result.loss_fn_outputs, batch)
        optimizer_metrics = _validated_optimizer_metrics(optim_result.metrics)
        _emit(
            output,
            {
                "record_type": "learner_result",
                "run_id": args.run_id,
                "iteration": 0,
                **learner_metrics,
                "optimizer_metrics": optimizer_metrics,
            },
        )
        await _timed_async_phase(
            output,
            run_id=args.run_id,
            phase="post_update_sampler_snapshot",
            action=lambda: _save_exact_sampler_snapshot(training_client),
            durations=phase_durations,
        )
    except BaseException as caught:
        primary_error = caught

    iteration_seconds: float | None = None
    try:
        iteration_end_ns = time.perf_counter_ns()
        iteration_seconds = (iteration_end_ns - iteration_start_ns) / 1e9
        if not math.isfinite(iteration_seconds) or iteration_seconds <= 0.0:
            raise RuntimeError("cold policy iteration duration is invalid")
        iteration_record: dict[str, Any] = {
            "record_type": "iteration",
            "run_id": args.run_id,
            "iteration": 0,
            "phase": "cold_policy_iteration",
            "status": "completed" if primary_error is None else "failed",
            "wall_start_ns": iteration_wall_start_ns,
            "wall_end_ns": time.time_ns(),
            "duration_seconds": iteration_seconds,
            "phase_seconds": phase_durations,
        }
        if primary_error is not None:
            iteration_record.update(_error_evidence(primary_error))
        elif learner_metrics is not None:
            iteration_record["cold_learner_fraction"] = (
                phase_durations["learner_update"] / iteration_seconds
            )
        _emit(output, iteration_record)
    except BaseException as iteration_record_error:
        if primary_error is None:
            primary_error = iteration_record_error
        else:
            primary_error.add_note(
                "iteration evidence recording also failed: "
                f"{type(iteration_record_error).__name__}"
            )

    finalization = await _finalize_after_seal(
        output,
        run_id=args.run_id,
        server_seal=server_seal,
        service_client=service_client,
        training_client=training_client,
        adapter_lifecycle_start_ns=adapter_lifecycle_start_ns,
    )
    _raise_preferred_error(primary_error, finalization)
    if iteration_seconds is None:
        raise RuntimeError("cold policy iteration produced no duration")
    cleanup_seconds = finalization.cleanup_seconds
    adapter_lifecycle_seconds = finalization.adapter_lifecycle_seconds
    if cleanup_seconds is None or adapter_lifecycle_seconds is None:
        raise RuntimeError("completed adapter lifecycle has no cleanup timing")
    if not math.isfinite(adapter_lifecycle_seconds) or adapter_lifecycle_seconds <= 0.0:
        raise RuntimeError("cold adapter lifecycle duration is invalid")

    summary = {
        "record_type": "summary",
        "run_id": args.run_id,
        "wall_time": _utc_now(),
        "protocol": "one_iteration_gate",
        "iterations_completed": 1,
        "updates_completed": 1,
        "sampler_snapshots_completed": 2,
        "adapter_unloaded": True,
        "cold_iteration_only": True,
        "steady_state_throughput_available": False,
        "server_attestation_sha256": server_seal.contract_sha256,
        "server_attestation_seconds": setup_durations["server_attestation"],
        "server_revalidation_seconds": finalization.revalidation_durations[
            "server_revalidation"
        ],
        "server_revalidated": True,
        "adapter_create_seconds": setup_durations["adapter_create"],
        "cold_policy_iteration_seconds": iteration_seconds,
        "cleanup_seconds": cleanup_seconds,
        "adapter_lifecycle_seconds": adapter_lifecycle_seconds,
        "cold_learner_fraction": phase_durations["learner_update"] / iteration_seconds,
        "sampling_specialization": specialization,
    }
    _emit(output, summary)
    _print_summary(summary)


def _print_summary(summary: dict[str, Any]) -> None:
    """Keep the durable JSONL result authoritative when stdout is unavailable."""
    try:
        print(json_dumps(summary, indent=2, sort_keys=True))
    except (BrokenPipeError, OSError, ValueError):
        return


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--server-pid",
        type=int,
        required=True,
        help="outer uv PID returned by placing start_qwen35.sh in the background",
    )
    parser.add_argument("--one-iteration-gate", action="store_true")
    parser.add_argument("--prompt-tokens", type=int, default=PROMPT_TOKENS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument(
        "--sample-max-num-sequences",
        type=int,
        default=SAMPLE_MAX_NUM_SEQUENCES,
        help="must exactly match the server JAX backend setting",
    )
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    canonical = (
        ("--prompt-tokens", args.prompt_tokens, PROMPT_TOKENS),
        ("--max-new-tokens", args.max_new_tokens, MAX_NEW_TOKENS),
        ("--group-size", args.group_size, GROUP_SIZE),
        (
            "--sample-max-num-sequences",
            args.sample_max_num_sequences,
            SAMPLE_MAX_NUM_SEQUENCES,
        ),
        ("--lora-rank", args.lora_rank, LORA_RANK),
    )
    if not args.one_iteration_gate:
        parser.error(
            "--one-iteration-gate is required; no steady-state protocol exists yet"
        )
    for name, value, expected in canonical:
        if value != expected:
            parser.error(f"{name} must remain exactly {expected} for this gate")
    if type(args.server_pid) is not int or args.server_pid <= 1:
        parser.error("--server-pid must be an integer greater than 1")
    maximum_base_seed = 2**31 - GROUP_SIZE
    if type(args.seed) is not int or not 0 <= args.seed <= maximum_base_seed:
        parser.error(
            f"--seed must be an integer in [0, {maximum_base_seed}] so every "
            "derived group seed fits signed int32"
        )
    optimizer_values = (
        args.learning_rate,
        args.adam_beta1,
        args.adam_beta2,
        args.adam_eps,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in optimizer_values
    ):
        parser.error("Adam parameters must be numeric")
    if (
        not math.isfinite(args.learning_rate)
        or not 0 < args.learning_rate <= 1e-2
        or not math.isfinite(args.adam_beta1)
        or not math.isfinite(args.adam_beta2)
        or not math.isfinite(args.adam_eps)
        or not 0 <= args.adam_beta1 < 1
        or not 0 <= args.adam_beta2 < 1
        or not 0 < args.adam_eps <= 1e-2
    ):
        parser.error("invalid Adam parameters")
    float32_values = tuple(
        struct.unpack("!f", struct.pack("!f", float(value)))[0]
        for value in optimizer_values
    )
    float32_learning_rate, float32_beta1, float32_beta2, float32_eps = float32_values
    if (
        float32_learning_rate <= 0.0
        or float32_eps <= 0.0
        or not 0 <= float32_beta1 < 1
        or not 0 <= float32_beta2 < 1
    ):
        parser.error("Adam parameters are invalid after float32 conversion")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id) is None:
        parser.error(
            "--run-id must start with a letter or digit and contain only letters, "
            "digits, dot, underscore, or dash"
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        output = os.fdopen(descriptor, "w", encoding="utf-8")
    except FileExistsError:
        parser.error(f"refusing to resume or overwrite existing output: {args.output}")

    with output:
        try:
            asyncio.run(_run(args, output))
        except BaseException as error:
            try:
                _emit(
                    output,
                    {
                        "record_type": "error",
                        "run_id": args.run_id,
                        "wall_time": _utc_now(),
                        **_error_evidence(error),
                    },
                )
            finally:
                raise SystemExit(
                    "GRPO gate failed; the original error was redacted and hashed "
                    "in the output record"
                ) from None


if __name__ == "__main__":
    main()
