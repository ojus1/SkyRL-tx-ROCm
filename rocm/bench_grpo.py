"""Fixed-rollout GRPO learner benchmark for the local SkyRL Tinker API.

This benchmark deliberately does not sample.  It constructs one deterministic
group of synthetic rollouts, applies the Cookbook's group-mean advantage rule,
and repeatedly executes the same ``importance_sampling`` forward/backward plus
Adam update.  It therefore isolates learner throughput and stability from
generation, grading, checkpoint export, and sampler synchronization.

The data contract matches ``tinker_cookbook.rl.data_processing`` after its
action ``mask`` is removed before the built-in loss call: prompt advantages and
sampling log-probabilities are zero, completion tokens receive one scalar
group-relative advantage, and no explicit ``weights`` field is sent.  SkyRL's
local API consequently supplies all-one loss weights, so each sequence loss is
normalized by its full shifted context length, just like the Cookbook path.

This is synthetic systems evidence, not an RL quality benchmark.  In
particular, the fixed sampling log-probabilities did not come from a sampler.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import math
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tinker

# ``rocm`` is a standalone tools directory rather than an installed package.
# Add only that directory so this file works both as a script and in unit tests.
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
    model_revision,
    package_version,
    redact_url,
    redacted_argv,
    round_up_seq_len,
    safe_accelerator_environment,
    unload_adapter,
)

SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _get_tokenizer() -> Any:
    """Load only the already-cached tokenizer; a benchmark must not fetch data."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, local_files_only=True)


def _repeat_tokens(seed: list[int], length: int) -> list[int]:
    if length < 0:
        raise ValueError("token length must be nonnegative")
    if length == 0:
        return []
    if not seed:
        raise ValueError("tokenizer returned no benchmark tokens")
    return [seed[index % len(seed)] for index in range(length)]


@dataclass(frozen=True)
class FixedRolloutBatch:
    """The exact, reusable learner inputs for one synthetic GRPO group."""

    datums: tuple[tinker.Datum, ...]
    prompt_tokens: tuple[int, ...]
    completion_tokens: tuple[tuple[int, ...], ...]
    rewards: tuple[float, ...]
    advantages: tuple[float, ...]
    sampling_logprob: float
    context: int

    @property
    def group_size(self) -> int:
        return len(self.datums)

    @property
    def completion_length(self) -> int:
        return len(self.completion_tokens[0])


def _build_fixed_rollouts(
    context: int,
    completion_tokens: int,
    group_size: int,
    sampling_logprob: float,
    *,
    tokenizer: Any | None = None,
) -> FixedRolloutBatch:
    """Build one non-degenerate, group-centered batch without sampling.

    ``context`` is the shifted learner length.  The unshifted sequence is one
    token longer, so the common prompt has
    ``context - completion_tokens + 1`` tokens.  Each datum is the usual causal
    pair ``full_sequence[:-1]`` -> ``full_sequence[1:]``.
    """
    if context < 2:
        raise ValueError("context must be at least 2")
    if not 1 <= completion_tokens <= context:
        raise ValueError("completion_tokens must be in [1, context]")
    if not 2 <= group_size <= 16:
        raise ValueError("group_size must be in [2, 16]")
    if not math.isfinite(sampling_logprob) or not -5.0 <= sampling_logprob <= 0.0:
        raise ValueError("sampling_logprob must be finite and in [-5, 0]")

    tokenizer = tokenizer or _get_tokenizer()
    prompt_length = context - completion_tokens + 1
    prompt_seed = tokenizer.encode(" fixed rollout prompt", add_special_tokens=False)
    prompt = _repeat_tokens(prompt_seed, prompt_length)

    rewards = [index / (group_size - 1) for index in range(group_size)]
    mean_reward = math.fsum(rewards) / group_size
    advantages = [reward - mean_reward for reward in rewards]

    datums: list[tinker.Datum] = []
    completions: list[tuple[int, ...]] = []
    prompt_target_count = prompt_length - 1
    for rollout_index, advantage in enumerate(advantages):
        # Put rollout identity first.  Qwen tokenizes the former
        # ``fixed candidate N`` text with a shared three-token prefix, which
        # made completion lengths 1-3 identical and cancelled the group gradient.
        # A single hexadecimal character is unique across the accepted
        # 0..15 range.  Decimal labels 10..15 share a leading ``1`` token and
        # would collide again when completion_tokens=1.
        rollout_label = format(rollout_index, "x")
        completion_seed = tokenizer.encode(
            rollout_label, add_special_tokens=False
        ) + tokenizer.encode(" fixed candidate", add_special_tokens=False)
        completion = _repeat_tokens(completion_seed, completion_tokens)
        full_sequence = prompt + completion
        model_tokens = full_sequence[:-1]
        targets = full_sequence[1:]
        old_logprobs = [0.0] * prompt_target_count + [
            sampling_logprob
        ] * completion_tokens
        token_advantages = [0.0] * prompt_target_count + [advantage] * completion_tokens

        assert len(model_tokens) == len(targets) == context
        assert len(old_logprobs) == len(token_advantages) == context
        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(model_tokens),
            loss_fn_inputs={
                "target_tokens": targets,
                "logprobs": old_logprobs,
                "advantages": token_advantages,
            },
        )
        # Do not add weights: Cookbook removes its action mask before calling
        # importance_sampling, and the local API defaults missing weights to 1.
        assert "weights" not in datum.loss_fn_inputs
        datums.append(datum)
        completions.append(tuple(completion))

    if len(set(completions)) != group_size:
        raise AssertionError("fixed rollout completions are not unique")
    if not math.isclose(math.fsum(advantages), 0.0, abs_tol=1e-12):
        raise AssertionError("group-relative advantages do not sum to zero")
    if not any(value < 0 for value in advantages) or not any(
        value > 0 for value in advantages
    ):
        raise AssertionError("fixed rollout group is degenerate")

    return FixedRolloutBatch(
        datums=tuple(datums),
        prompt_tokens=tuple(prompt),
        completion_tokens=tuple(completions),
        rewards=tuple(rewards),
        advantages=tuple(advantages),
        sampling_logprob=sampling_logprob,
        context=context,
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


def _policy_metrics(
    loss_fn_outputs: list[dict[str, Any]], batch: FixedRolloutBatch
) -> dict[str, float]:
    """Summarize the exact learner log-probabilities on action tokens."""
    if len(loss_fn_outputs) != batch.group_size:
        raise ValueError("learner returned the wrong number of rollout outputs")

    action_logprobs: list[float] = []
    ratios: list[float] = []
    per_sequence_losses: list[float] = []
    completion_length = batch.completion_length

    for output, datum in zip(loss_fn_outputs, batch.datums, strict=True):
        target_logprobs = _tensor_values(output["logprobs"])
        old_logprobs = _tensor_values(datum.loss_fn_inputs["logprobs"])
        token_advantages = _tensor_values(datum.loss_fn_inputs["advantages"])
        if (
            len(target_logprobs) != batch.context
            or len(old_logprobs) != batch.context
            or len(token_advantages) != batch.context
        ):
            raise ValueError("learner returned the wrong token dimension")

        sequence_loss = 0.0
        for target_logprob, old_logprob, token_advantage in zip(
            target_logprobs, old_logprobs, token_advantages, strict=True
        ):
            if not math.isfinite(target_logprob):
                raise FloatingPointError("non-finite learner target log-probability")
            ratio = math.exp(target_logprob - old_logprob)
            sequence_loss += -ratio * token_advantage
        per_sequence_losses.append(sequence_loss / batch.context)

        for target_logprob, old_logprob in zip(
            target_logprobs[-completion_length:],
            old_logprobs[-completion_length:],
            strict=True,
        ):
            action_logprobs.append(target_logprob)
            ratios.append(math.exp(target_logprob - old_logprob))

    values = (*action_logprobs, *ratios, *per_sequence_losses)
    if not values or not all(math.isfinite(value) for value in values):
        raise FloatingPointError("non-finite fixed-rollout policy metrics")
    return {
        "policy_loss_mean": statistics.fmean(per_sequence_losses),
        "action_target_logprob_mean": statistics.fmean(action_logprobs),
        "importance_ratio_mean": statistics.fmean(ratios),
        "importance_ratio_min": min(ratios),
        "importance_ratio_max": max(ratios),
    }


def _summary(
    durations: list[float],
    context: int,
    padded_context: int,
    group_size: int,
    completion_tokens: int,
) -> dict[str, float | int]:
    ordered = sorted(durations)
    median = statistics.median(ordered)
    mad = statistics.median(abs(value - median) for value in ordered)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "measured_steps": len(ordered),
        "step_seconds_median": median,
        "step_seconds_p95": p95,
        "step_seconds_mad": mad,
        "learner_tokens_per_second_median": group_size * context / median,
        "padded_learner_tokens_per_second_median": group_size * padded_context / median,
        "action_tokens_per_second_median": group_size * completion_tokens / median,
    }


async def _run(args: argparse.Namespace, output) -> None:
    batch = _build_fixed_rollouts(
        args.context,
        args.completion_tokens,
        args.group_size,
        args.sampling_logprob,
    )
    padded_context = round_up_seq_len(args.context)
    manifest = {
        "record_type": "manifest",
        "schema_version": SCHEMA_VERSION,
        "suite": "fixed_rollout_grpo_learner",
        "warning": (
            "Synthetic fixed-rollout learner benchmark; sampling logprobs are synthetic and "
            "results are not RL quality evidence."
        ),
        "run_id": args.run_id,
        "wall_time": _utc_now(),
        "wall_time_ns": time.time_ns(),
        "argv": redacted_argv(sys.argv),
        "model": MODEL,
        "model_revision": model_revision(MODEL),
        "base_url": redact_url(args.base_url),
        "requested_context": args.context,
        "effective_padded_context": padded_context,
        "batch_size": args.group_size,
        "group_size": args.group_size,
        "prompt_tokens": len(batch.prompt_tokens),
        "completion_tokens_per_rollout": args.completion_tokens,
        "learner_tokens_per_step": args.group_size * args.context,
        "action_tokens_per_step": args.group_size * args.completion_tokens,
        "rewards": list(batch.rewards),
        "advantages": list(batch.advantages),
        "sampling_logprob": args.sampling_logprob,
        "loss_fn": "importance_sampling",
        "loss_normalization": "implicit all-one weights; per-sequence full shifted context",
        "sampling_performed": False,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "lora_rank": args.lora_rank,
        "seed": args.seed,
        "optimizer": {
            "name": "adam",
            "learning_rate": args.learning_rate,
            "beta1": args.adam_beta1,
            "beta2": args.adam_beta2,
            "eps": args.adam_eps,
        },
        "client": {
            "python": sys.version,
            "tinker": package_version("tinker"),
            "tinker_cookbook": package_version("tinker-cookbook"),
            "benchmark_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "environment": safe_accelerator_environment(),
        "repositories": {
            "skyrl": git_metadata(REPO),
            "tinker_cookbook": git_metadata(WORKSPACE / "tinker-cookbook"),
        },
    }
    output.write(json_dumps(manifest, separators=(",", ":")) + "\n")
    output.flush()

    service_client = tinker.ServiceClient(
        base_url=args.base_url,
        user_metadata={
            "recipe_name": "qwen35_rocm_fixed_rollout_grpo_benchmark",
            "run_id": args.run_id,
        },
    )
    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL,
        rank=args.lora_rank,
        seed=args.seed,
        train_mlp=True,
        train_attn=True,
        train_unembed=False,
        user_metadata={"suite": "fixed_rollout_grpo_learner", "run_id": args.run_id},
    )
    adam = tinker.AdamParams(
        learning_rate=args.learning_rate,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        eps=args.adam_eps,
    )

    durations: list[float] = []
    total_steps = args.warmup_steps + args.measured_steps
    primary_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        for step in range(total_steps):
            if step == 0:
                phase = "cold_compile"
            elif step < args.warmup_steps:
                phase = "warmup"
            else:
                phase = "measured"

            wall_start_ns = time.time_ns()
            monotonic_start_ns = time.perf_counter_ns()
            fwd_bwd_future = await training_client.forward_backward_async(
                list(batch.datums), loss_fn="importance_sampling"
            )
            optim_future = await training_client.optim_step_async(adam)
            enqueue_end_ns = time.perf_counter_ns()
            fwd_bwd_result = await fwd_bwd_future.result_async()
            optim_result = await optim_future.result_async()
            monotonic_end_ns = time.perf_counter_ns()
            wall_end_ns = time.time_ns()

            duration = (monotonic_end_ns - monotonic_start_ns) / 1e9
            enqueue_seconds = (enqueue_end_ns - monotonic_start_ns) / 1e9
            policy_metrics = _policy_metrics(fwd_bwd_result.loss_fn_outputs, batch)
            numeric_values = (duration, enqueue_seconds, *policy_metrics.values())
            if (
                not all(math.isfinite(value) for value in numeric_values)
                or duration <= 0
            ):
                raise FloatingPointError(
                    f"non-finite or non-positive step metrics: {numeric_values}"
                )

            record = {
                "record_type": "step",
                "run_id": args.run_id,
                "step": step,
                "phase": phase,
                "cold_jit": step == 0,
                "requested_context": args.context,
                "effective_padded_context": padded_context,
                "batch_size": args.group_size,
                "group_size": args.group_size,
                "completion_tokens_per_rollout": args.completion_tokens,
                "wall_start_ns": wall_start_ns,
                "wall_end_ns": wall_end_ns,
                "step_seconds": duration,
                "enqueue_seconds": enqueue_seconds,
                "resolve_seconds": duration - enqueue_seconds,
                "learner_tokens_per_second": args.group_size * args.context / duration,
                "padded_learner_tokens_per_second": (
                    args.group_size * padded_context / duration
                ),
                "action_tokens_per_second": (
                    args.group_size * args.completion_tokens / duration
                ),
                **policy_metrics,
                "optimizer_metrics": json_safe(optim_result.metrics),
            }
            output.write(json_dumps(record, separators=(",", ":")) + "\n")
            output.flush()
            if phase == "measured":
                durations.append(duration)
    except BaseException as caught:
        primary_error = caught

    cleanup_start_ns = time.time_ns()
    try:
        unload_result = await unload_adapter(service_client, training_client)
        output.write(
            json_dumps(
                {
                    "record_type": "cleanup",
                    "run_id": args.run_id,
                    "wall_start_ns": cleanup_start_ns,
                    "wall_end_ns": time.time_ns(),
                    "model_id": str(training_client._guaranteed_model_id()),
                    "adapter_unloaded": True,
                    "response": json_safe(unload_result),
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        output.flush()
    except BaseException as caught:
        cleanup_error = caught
        output.write(
            json_dumps(
                {
                    "record_type": "cleanup",
                    "run_id": args.run_id,
                    "wall_start_ns": cleanup_start_ns,
                    "wall_end_ns": time.time_ns(),
                    "model_id": str(training_client._guaranteed_model_id()),
                    "adapter_unloaded": False,
                    "error_type": type(caught).__name__,
                    "error": str(caught),
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        output.flush()

    if primary_error is not None:
        if cleanup_error is not None:
            primary_error.add_note(f"adapter cleanup also failed: {cleanup_error}")
        raise primary_error
    if cleanup_error is not None:
        raise cleanup_error

    final = {
        "record_type": "summary",
        "run_id": args.run_id,
        "wall_time": _utc_now(),
        "requested_context": args.context,
        "effective_padded_context": padded_context,
        "group_size": args.group_size,
        "completion_tokens_per_rollout": args.completion_tokens,
        **_summary(
            durations,
            args.context,
            padded_context,
            args.group_size,
            args.completion_tokens,
        ),
    }
    output.write(json_dumps(final, separators=(",", ":")) + "\n")
    output.flush()
    print(json_dumps(final, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--completion-tokens", type=int, required=True)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--sampling-logprob", type=float, default=-5.0)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measured-steps", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 2 <= args.context <= 16384:
        parser.error(
            "--context must be in [2, 16384] until watchdog-bounded 32K attention is implemented"
        )
    if not 1 <= args.completion_tokens <= args.context:
        parser.error("--completion-tokens must be in [1, context]")
    if not 2 <= args.group_size <= 16:
        parser.error("--group-size must be in [2, 16]")
    if (
        not math.isfinite(args.sampling_logprob)
        or not -5.0 <= args.sampling_logprob <= 0.0
    ):
        parser.error("--sampling-logprob must be finite and in [-5, 0]")
    if args.warmup_steps < 1:
        parser.error("--warmup-steps must be at least 1 so cold JIT is never measured")
    if args.measured_steps < 5:
        parser.error("--measured-steps must be at least 5")
    if (
        not 1 <= args.lora_rank <= 128
        or not math.isfinite(args.learning_rate)
        or not 0 <= args.learning_rate <= 1e-2
    ):
        parser.error("--lora-rank must be in [1, 128] and --learning-rate in [0, 1e-2]")
    if (
        not math.isfinite(args.adam_beta1)
        or not math.isfinite(args.adam_beta2)
        or not math.isfinite(args.adam_eps)
        or not 0 <= args.adam_beta1 < 1
        or not 0 <= args.adam_beta2 < 1
        or args.adam_eps <= 0
    ):
        parser.error("invalid Adam parameters")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id) is None:
        parser.error(
            "--run-id must start with a letter or digit and then contain only letters, "
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
            output.write(
                json_dumps(
                    {
                        "record_type": "error",
                        "run_id": args.run_id,
                        "wall_time": _utc_now(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            output.flush()
            raise


if __name__ == "__main__":
    main()
