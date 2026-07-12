"""Isolated correctness and latency probe for opt-in ROCm Pallas attention."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections.abc import Callable
from typing import Any

MAX_VALIDATED_SEQUENCE_LENGTH = 16_384


def _block_until_ready(jax, value):
    return jax.block_until_ready(value)


def _timed(jax, function: Callable, *args):
    start = time.perf_counter()
    result = function(*args)
    _block_until_ready(jax, result)
    return result, time.perf_counter() - start


def _timing_summary(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    median = statistics.median(ordered)
    return {
        "raw_seconds": values,
        "median_seconds": median,
        "p95_seconds": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        "mad_seconds": statistics.median(abs(value - median) for value in ordered),
    }


def _error_metrics(jnp, actual, expected) -> dict[str, float]:
    actual = actual.astype(jnp.float32)
    expected = expected.astype(jnp.float32)
    difference = actual - expected
    actual_norm = jnp.linalg.norm(actual.ravel())
    expected_norm = jnp.linalg.norm(expected.ravel())
    denominator = jnp.maximum(expected_norm, jnp.finfo(jnp.float32).tiny)
    cosine_denominator = jnp.maximum(actual_norm * expected_norm, jnp.finfo(jnp.float32).tiny)
    return {
        "max_abs": float(jnp.max(jnp.abs(difference))),
        "mean_abs": float(jnp.mean(jnp.abs(difference))),
        "relative_l2": float(jnp.linalg.norm(difference.ravel()) / denominator),
        "cosine": float(jnp.vdot(actual.ravel(), expected.ravel()) / cosine_denominator),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--valid-length", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--query-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, choices=(64, 128, 256), default=256)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()

    valid_length = args.valid_length if args.valid_length is not None else args.sequence_length
    if args.sequence_length <= 0 or args.sequence_length % 64:
        parser.error("--sequence-length must be a positive multiple of 64")
    if args.sequence_length > MAX_VALIDATED_SEQUENCE_LENGTH:
        parser.error(
            f"--sequence-length must not exceed {MAX_VALIDATED_SEQUENCE_LENGTH}; "
            "the monolithic 32K backward kernel caused a gfx1100 ring timeout"
        )
    if not 1 <= valid_length <= args.sequence_length:
        parser.error("--valid-length must be in [1, sequence-length]")
    if args.batch_size <= 0 or args.query_heads <= 0 or args.kv_heads <= 0 or args.head_dim <= 0:
        parser.error("batch/head dimensions must be positive")
    if args.query_heads % args.kv_heads:
        parser.error("--query-heads must be divisible by --kv-heads")
    if args.warmups < 1 or args.repeats < 1:
        parser.error("--warmups and --repeats must be positive")
    score_bytes = args.batch_size * args.query_heads * args.sequence_length**2 * 4
    if args.reference and score_bytes > 2 * 1024**3:
        parser.error("generic reference score matrix would exceed 2 GiB; omit --reference")

    # The selector reads this at JIT trace time. Set it before importing SkyRL.
    os.environ["SKYRL_ROCM_PALLAS_ATTENTION"] = "1"

    import jax
    import jax.numpy as jnp
    from jax.extend import backend as jax_backend

    from skyrl.tx.layers.attention import dot_product_attention

    platform_version = jax_backend.get_backend().platform_version
    if jax.default_backend() != "gpu" or "rocm" not in platform_version.lower():
        raise RuntimeError(f"This probe requires a ROCm JAX GPU, got {jax.default_backend()}: {platform_version}")

    dtype = {"bfloat16": jnp.bfloat16, "float16": jnp.float16}[args.dtype]
    q_shape = (args.batch_size, args.sequence_length, args.query_heads, args.head_dim)
    kv_shape = (args.batch_size, args.sequence_length, args.kv_heads, args.head_dim)
    q_key, k_key, v_key = jax.random.split(jax.random.key(args.seed), 3)
    q = jax.random.normal(q_key, q_shape, dtype=dtype)
    k = jax.random.normal(k_key, kv_shape, dtype=dtype)
    v = jax.random.normal(v_key, kv_shape, dtype=dtype)
    mask = (jnp.arange(args.sequence_length)[None, :] < valid_length).astype(jnp.int32)
    mask = jnp.broadcast_to(mask, (args.batch_size, args.sequence_length))
    scale = args.head_dim**-0.5
    valid_query_mask = mask[:, :, None, None].astype(jnp.float32)

    @jax.jit
    def pallas_forward(q_arg, k_arg, v_arg):
        return dot_product_attention(q_arg, k_arg, v_arg, mask, True, args.head_dim)

    def loss(function, q_arg, k_arg, v_arg):
        output = function(q_arg, k_arg, v_arg).astype(jnp.float32)
        return jnp.sum(output * output * valid_query_mask)

    pallas_backward = jax.jit(jax.grad(lambda *items: loss(pallas_forward, *items), argnums=(0, 1, 2)))

    memory_before = jax.devices()[0].memory_stats()
    pallas_output, forward_compile_seconds = _timed(jax, pallas_forward, q, k, v)
    forward_timings = []
    for _ in range(args.warmups - 1):
        _timed(jax, pallas_forward, q, k, v)
    for _ in range(args.repeats):
        _, elapsed = _timed(jax, pallas_forward, q, k, v)
        forward_timings.append(elapsed)

    result: dict[str, Any] = {
        "backend": jax.default_backend(),
        "platform_version": platform_version,
        "device": str(jax.devices()[0]),
        "config": {
            **vars(args),
            "valid_length": valid_length,
            "score_bytes_if_generic": score_bytes,
        },
        "forward": {
            "compile_and_first_seconds": forward_compile_seconds,
            "finite": bool(jnp.all(jnp.isfinite(pallas_output))),
            **_timing_summary(forward_timings),
        },
        "memory_before": memory_before,
    }

    pallas_gradients = None
    if args.backward:
        pallas_gradients, backward_compile_seconds = _timed(jax, pallas_backward, q, k, v)
        backward_timings = []
        for _ in range(args.warmups - 1):
            _timed(jax, pallas_backward, q, k, v)
        for _ in range(args.repeats):
            _, elapsed = _timed(jax, pallas_backward, q, k, v)
            backward_timings.append(elapsed)
        result["backward"] = {
            "compile_and_first_seconds": backward_compile_seconds,
            "finite": all(bool(jnp.all(jnp.isfinite(gradient))) for gradient in pallas_gradients),
            **_timing_summary(backward_timings),
        }

    if args.reference:
        @jax.jit
        def generic_forward(q_arg, k_arg, v_arg):
            return jax.nn.dot_product_attention(
                q_arg,
                k_arg,
                v_arg,
                scale=scale,
                mask=mask[:, None, None, :].astype(bool),
                is_causal=True,
            )

        try:
            generic_output, generic_compile_seconds = _timed(jax, generic_forward, q, k, v)
            result["reference"] = {
                "status": "passed",
                "compile_and_first_seconds": generic_compile_seconds,
                "valid_output_error": _error_metrics(
                    jnp,
                    pallas_output * valid_query_mask,
                    generic_output * valid_query_mask,
                ),
            }
            if args.backward:
                generic_backward = jax.jit(
                    jax.grad(lambda *items: loss(generic_forward, *items), argnums=(0, 1, 2))
                )
                generic_gradients, generic_backward_compile_seconds = _timed(jax, generic_backward, q, k, v)
                result["reference"]["backward_compile_and_first_seconds"] = generic_backward_compile_seconds
                result["reference"]["gradient_error"] = {
                    name: _error_metrics(jnp, actual, expected)
                    for name, actual, expected in zip(
                        ("dq", "dk", "dv"), pallas_gradients, generic_gradients, strict=True
                    )
                }
        except BaseException as error:
            # A failed quadratic reference is itself useful evidence. Preserve
            # the already-completed Pallas result instead of losing the run.
            result["reference"] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }

    result["memory_after"] = jax.devices()[0].memory_stats()
    print(json.dumps(result, indent=2, sort_keys=True, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
