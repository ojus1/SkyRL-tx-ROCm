"""One-context synthetic SFT benchmark for the local SkyRL Tinker API.

This intentionally bypasses Cookbook's lifecycle/checkpoint loop while keeping
the same Tinker ``forward_backward`` + Adam ``optim_step`` operations. Timings
include the local HTTP/database/future path, so this is an end-to-end training
step benchmark rather than a pure kernel benchmark. It is not a rendered-chat
quality benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import tinker
import torch
from tinker_cookbook.supervised.common import (
    compute_mean_nll,
    datum_from_model_input_weights,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL = "Qwen/Qwen3.5-4B"
SCHEMA_VERSION = 1
REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
_SENSITIVE_WORDS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH", "COOKIE")
_SENSITIVE_ARGUMENT_FLAGS = ("--header", "--proxy-header", "--user", "--proxy-user", "-u")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_dumps(value: Any, **kwargs) -> str:
    return json.dumps(value, allow_nan=False, **kwargs)


def _redact_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        if ":" in hostname:
            hostname = f"[{hostname}]"
        if parsed.port is not None:
            hostname = f"{hostname}:{parsed.port}"
        # Paths often contain webhook tokens or object-store keys.
        return urlunsplit((parsed.scheme, hostname, "", "", ""))
    except ValueError:
        return "<redacted-url>"


def _redacted_argv(values: list[str]) -> list[str]:
    result: list[str] = []
    redact_next = False
    for value in values:
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        lowered = value.lower()
        sensitive_flag = next(
            (
                flag
                for flag in _SENSITIVE_ARGUMENT_FLAGS
                if lowered == flag or lowered.startswith(f"{flag}=")
            ),
            None,
        )
        if sensitive_flag is not None:
            if "=" in value:
                result.append(f"{value.split('=', 1)[0]}=<redacted>")
            else:
                result.append(value)
                redact_next = True
            continue
        if value.startswith(("-H", "-u", "-U")) and len(value) > 2:
            result.append(f"{value[:2]}<redacted>")
            continue

        key: str | None = None
        candidate = value
        equals_index = value.find("=")
        scheme_index = value.find("://")
        if equals_index >= 0 and (scheme_index < 0 or equals_index < scheme_index):
            key, candidate = value.split("=", 1)
        if key is not None and any(word in key.upper() for word in _SENSITIVE_WORDS):
            result.append(f"{key}=<redacted>")
        elif "://" in candidate:
            sanitized = _redact_url(candidate)
            result.append(f"{key}={sanitized}" if key is not None else sanitized)
        else:
            result.append(value)
        if value == "-H":
            redact_next = True
        elif value.startswith("-") and any(word in value.upper() for word in _SENSITIVE_WORDS):
            redact_next = "=" not in value
    return result


def _round_up_seq_len(seq_len: int) -> int:
    """Mirror ``skyrl.tx.utils.models.round_up_seq_len`` without importing JAX."""
    if seq_len <= 32:
        return 32
    msb_pos = seq_len.bit_length() - 1
    mask = (1 << msb_pos) | (1 << (msb_pos - 1))
    rounded = seq_len & mask
    if rounded < seq_len:
        rounded += 1 << (msb_pos - 1)
    return rounded


def _run_text(*args: str, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_metadata(path: Path) -> dict[str, Any]:
    commit = _run_text("git", "rev-parse", "HEAD", cwd=path)
    status = _run_text("git", "status", "--porcelain=v1", cwd=path)
    diff = _run_text("git", "diff", "--binary", "HEAD", cwd=path)
    return {
        "path": str(path),
        "commit": commit,
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
        "tracked_diff_sha256": hashlib.sha256((diff or "").encode()).hexdigest(),
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _model_revision(model: str) -> str | None:
    default_hf_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface"
    hf_home = Path(os.environ.get("HF_HOME", default_hf_home))
    hub_cache = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub"))
    reference = hub_cache / f"models--{model.replace('/', '--')}" / "refs/main"
    try:
        return reference.read_text().strip()
    except (FileNotFoundError, PermissionError):
        return None


def _safe_accelerator_environment() -> dict[str, str]:
    prefixes = ("JAX_", "XLA_", "HSA_", "HIP_", "ROCM_", "AMD_")
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(prefixes) and not any(word in key.upper() for word in _SENSITIVE_WORDS)
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    return str(value)


def _build_datum(context: int) -> tinker.Datum:
    tokenizer = get_tokenizer(MODEL)
    seed_tokens = tokenizer.encode(" benchmark context", add_special_tokens=False)
    if not seed_tokens:
        raise RuntimeError("Tokenizer returned no benchmark tokens")
    tokens = [seed_tokens[index % len(seed_tokens)] for index in range(context + 1)]
    full_input = tinker.ModelInput.from_ints(tokens)
    weights = torch.ones(context + 1, dtype=torch.float32)
    datum = datum_from_model_input_weights(full_input, weights, reduction="mean")

    targets = datum.loss_fn_inputs["target_tokens"]
    shifted_weights = datum.loss_fn_inputs["weights"]
    assert full_input.length == context + 1
    assert datum.model_input.length == context
    assert targets.shape == [context]
    assert shifted_weights.shape == [context]
    assert math.isclose(sum(shifted_weights.data), 1.0, rel_tol=1e-5, abs_tol=1e-5)
    return datum


def _summary(durations: list[float], context: int, padded_context: int) -> dict[str, float | int]:
    ordered = sorted(durations)
    median = statistics.median(ordered)
    mad = statistics.median(abs(value - median) for value in ordered)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "measured_steps": len(ordered),
        "step_seconds_median": median,
        "step_seconds_p95": p95,
        "step_seconds_mad": mad,
        "useful_tokens_per_second_median": context / median,
        "padded_tokens_per_second_median": padded_context / median,
    }


async def _unload_adapter(service_client: tinker.ServiceClient, training_client) -> Any:
    """Unload the benchmark adapter using the SDK's generated local endpoint."""
    from tinker.lib.api_future_impl import _APIFuture
    from tinker.lib.client_connection_pool_type import ClientConnectionPoolType

    model_id = training_client._guaranteed_model_id()
    request_start = time.time()

    async def unload_on_sdk_loop() -> Any:
        # InternalClientHolder owns a dedicated event loop. Directly borrowing
        # its AsyncTinker client from asyncio.run() violates the SDK's loop
        # affinity, so schedule both request creation and future polling there.
        with service_client.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            future = await client.models.unload(
                request=tinker.types.UnloadModelRequest(model_id=model_id)
            )
        return await _APIFuture(
            tinker.types.UnloadModelResponse,
            service_client.holder,
            future,
            request_start_time=request_start,
            request_type="UnloadModel",
        ).result_async()

    return await service_client.holder.run_coroutine_threadsafe(unload_on_sdk_loop())


async def _run(args: argparse.Namespace, output) -> None:
    datum = _build_datum(args.context)
    padded_context = _round_up_seq_len(args.context)
    manifest = {
        "record_type": "manifest",
        "schema_version": SCHEMA_VERSION,
        "suite": "synthetic_nll",
        "warning": "End-to-end local Tinker step benchmark; not rendered-chat SFT quality evidence.",
        "run_id": args.run_id,
        "wall_time": _utc_now(),
        "wall_time_ns": time.time_ns(),
        "argv": _redacted_argv(sys.argv),
        "model": MODEL,
        "model_revision": _model_revision(MODEL),
        "base_url": _redact_url(args.base_url),
        "requested_context": args.context,
        "effective_padded_context": padded_context,
        "batch_size": 1,
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
            "platform": platform.platform(),
            "tinker": _package_version("tinker"),
            "tinker_cookbook": _package_version("tinker-cookbook"),
            "torch": _package_version("torch"),
            "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "environment": _safe_accelerator_environment(),
        "repositories": {
            "skyrl": _git_metadata(REPO),
            "tinker_cookbook": _git_metadata(WORKSPACE / "tinker-cookbook"),
        },
    }
    output.write(_json_dumps(manifest, separators=(",", ":")) + "\n")
    output.flush()

    service_client = tinker.ServiceClient(
        base_url=args.base_url,
        user_metadata={"recipe_name": "qwen35_rocm_synthetic_nll_benchmark", "run_id": args.run_id},
    )
    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL,
        rank=args.lora_rank,
        seed=args.seed,
        user_metadata={"suite": "synthetic_nll", "run_id": args.run_id},
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
            fwd_bwd_future = await training_client.forward_backward_async([datum], loss_fn="cross_entropy")
            optim_future = await training_client.optim_step_async(adam)
            enqueue_end_ns = time.perf_counter_ns()
            fwd_bwd_result = await fwd_bwd_future.result_async()
            optim_result = await optim_future.result_async()
            monotonic_end_ns = time.perf_counter_ns()
            wall_end_ns = time.time_ns()

            duration = (monotonic_end_ns - monotonic_start_ns) / 1e9
            enqueue_seconds = (enqueue_end_ns - monotonic_start_ns) / 1e9
            logprobs = [item["logprobs"] for item in fwd_bwd_result.loss_fn_outputs]
            mean_nll = compute_mean_nll(logprobs, [datum.loss_fn_inputs["weights"]])
            numeric_values = (duration, enqueue_seconds, mean_nll)
            if not all(math.isfinite(value) for value in numeric_values) or duration <= 0:
                raise FloatingPointError(f"non-finite or non-positive step metrics: {numeric_values}")
            record = {
                "record_type": "step",
                "run_id": args.run_id,
                "step": step,
                "phase": phase,
                "cold_jit": step == 0,
                "requested_context": args.context,
                "effective_padded_context": padded_context,
                "batch_size": 1,
                "wall_start_ns": wall_start_ns,
                "wall_end_ns": wall_end_ns,
                "step_seconds": duration,
                "enqueue_seconds": enqueue_seconds,
                "resolve_seconds": duration - enqueue_seconds,
                "train_mean_nll": mean_nll,
                "useful_tokens_per_second": args.context / duration,
                "padded_tokens_per_second": padded_context / duration,
                "optimizer_metrics": _json_safe(optim_result.metrics),
            }
            output.write(_json_dumps(record, separators=(",", ":")) + "\n")
            output.flush()
            if phase == "measured":
                durations.append(duration)
    except BaseException as caught:
        primary_error = caught

    cleanup_start_ns = time.time_ns()
    try:
        unload_result = await _unload_adapter(service_client, training_client)
        cleanup_record = {
            "record_type": "cleanup",
            "run_id": args.run_id,
            "wall_start_ns": cleanup_start_ns,
            "wall_end_ns": time.time_ns(),
            "model_id": str(training_client._guaranteed_model_id()),
            "adapter_unloaded": True,
            "response": _json_safe(unload_result),
        }
        output.write(_json_dumps(cleanup_record, separators=(",", ":")) + "\n")
        output.flush()
    except BaseException as caught:
        cleanup_error = caught

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
        **_summary(durations, args.context, padded_context),
    }
    output.write(_json_dumps(final, separators=(",", ":")) + "\n")
    output.flush()
    print(_json_dumps(final, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measured-steps", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not 2 <= args.context <= 16384:
        parser.error(
            "--context must be in [2, 16384] until watchdog-bounded 32K attention is implemented"
        )
    if args.warmup_steps < 1:
        parser.error("--warmup-steps must be at least 1 so cold JIT is never measured as steady state")
    if args.measured_steps < 5:
        parser.error("--measured-steps must be at least 5")
    if args.lora_rank <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate < 0:
        parser.error("--lora-rank must be positive and --learning-rate must be nonnegative")
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
                _json_dumps(
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
