"""Isolated correctness and latency probe for opt-in ROCm Pallas attention."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import statistics
import time
from collections.abc import Callable
from typing import Any

MAX_VALIDATED_SEQUENCE_LENGTH = 16_384
_DISABLE_COMMAND_BUFFERS = "--xla_gpu_enable_command_buffer="


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
    cosine_denominator = jnp.maximum(
        actual_norm * expected_norm, jnp.finfo(jnp.float32).tiny
    )
    return {
        "max_abs": float(jnp.max(jnp.abs(difference))),
        "mean_abs": float(jnp.mean(jnp.abs(difference))),
        "relative_l2": float(jnp.linalg.norm(difference.ravel()) / denominator),
        "cosine": float(
            jnp.vdot(actual.ravel(), expected.ravel()) / cosine_denominator
        ),
    }


def _chunked_reference_attention(
    jax, jnp, q, k, v, mask, scale, *, query_block_size=16
):
    """Evaluate causal GQA without materializing the full attention matrix.

    JAX's generic GQA lowering can fuse an entire query tile and request more
    shared memory than gfx1100 provides, even for a 512-token correctness
    check. Mapping fixed-size query blocks keeps both the reference forward and
    its automatically differentiated backward independently bounded.
    """
    batch_size, sequence_length, query_heads, head_dim = q.shape
    if sequence_length % query_block_size:
        raise ValueError(
            "sequence length must be divisible by the reference query block size"
        )

    repeats = query_heads // k.shape[2]
    k_repeated = jnp.repeat(k, repeats, axis=2).astype(jnp.float32)
    v_repeated = jnp.repeat(v, repeats, axis=2).astype(jnp.float32)
    query_blocks = q.reshape(
        batch_size,
        sequence_length // query_block_size,
        query_block_size,
        query_heads,
        head_dim,
    ).transpose(1, 0, 2, 3, 4)
    query_positions = jnp.arange(sequence_length, dtype=jnp.int32).reshape(
        sequence_length // query_block_size, query_block_size
    )
    key_positions = jnp.arange(sequence_length, dtype=jnp.int32)
    valid_keys = mask.astype(bool)[:, None, None, :]

    def attend_query_block(items):
        q_block, q_positions = items
        logits = jnp.einsum(
            "bqhd,bkhd->bhqk",
            q_block.astype(jnp.float32),
            k_repeated,
            preferred_element_type=jnp.float32,
        )
        logits *= scale
        causal = key_positions[None, None, None, :] <= q_positions[None, None, :, None]
        logits = jnp.where(valid_keys & causal, logits, jnp.finfo(jnp.float32).min)
        probabilities = jax.nn.softmax(logits, axis=-1)
        output = jnp.einsum(
            "bhqk,bkhd->bqhd",
            probabilities,
            v_repeated,
            preferred_element_type=jnp.float32,
        )
        return output.astype(q.dtype)

    output_blocks = jax.lax.map(attend_query_block, (query_blocks, query_positions))
    return output_blocks.transpose(1, 0, 2, 3, 4).reshape(q.shape)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help="required acknowledgement because this probe always executes ROCm kernels",
    )
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--valid-length", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--query-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, choices=(64, 128, 256), default=256)
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args(argv)

    valid_length = (
        args.valid_length if args.valid_length is not None else args.sequence_length
    )
    if not args.allow_gpu:
        parser.error(
            "this ROCm probe requires the explicit --allow-gpu acknowledgement"
        )
    if args.sequence_length < 512 or args.sequence_length % 64:
        parser.error("--sequence-length must be a multiple of 64 in [512, 16384]")
    if args.sequence_length > MAX_VALIDATED_SEQUENCE_LENGTH:
        parser.error(
            f"--sequence-length must not exceed {MAX_VALIDATED_SEQUENCE_LENGTH}; "
            "the monolithic 32K backward kernel caused a gfx1100 ring timeout"
        )
    if not 1 <= valid_length <= args.sequence_length:
        parser.error("--valid-length must be in [1, sequence-length]")
    if (args.batch_size, args.query_heads, args.kv_heads, args.head_dim) != (
        1,
        16,
        4,
        256,
    ):
        parser.error(
            "this bounded probe is restricted to the exact Qwen3.5-4B shape: "
            "batch=1, query-heads=16, kv-heads=4, head-dim=256"
        )
    if not 1 <= args.warmups <= 10 or not 1 <= args.repeats <= 20:
        parser.error("--warmups must be in [1, 10] and --repeats in [1, 20]")
    if args.reference and args.sequence_length > 2048:
        parser.error(
            "--reference is capped at 2048 tokens because its mapped quadratic "
            "work has not passed the longer watchdog-safety ladder"
        )
    return args


def _validate_exact_or_unset(name: str, expected: str) -> None:
    value = os.environ.get(name)
    if value is not None and value != expected:
        raise RuntimeError(
            f"{name}={value!r} conflicts with required value {expected!r}"
        )


def _require_unset(name: str) -> None:
    if name in os.environ:
        raise RuntimeError(f"{name} must be unset for this bounded ROCm probe")


def _force_command_buffers_disabled(original: str) -> str:
    try:
        tokens = shlex.split(original)
    except ValueError as error:
        raise RuntimeError(f"invalid XLA_FLAGS quoting: {error}") from error
    if any(any(character.isspace() for character in token) for token in tokens):
        raise RuntimeError("XLA_FLAGS entries containing whitespace are unsupported")
    flag_name = _DISABLE_COMMAND_BUFFERS.partition("=")[0]
    tokens = [
        token
        for token in tokens
        if token != flag_name and not token.startswith(f"{flag_name}=")
    ]
    return " ".join((*tokens, _DISABLE_COMMAND_BUFFERS))


def _configure_environment() -> dict[str, str | None]:
    _validate_exact_or_unset("JAX_PLATFORMS", "rocm")
    for name in (
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "JAX_ROCM_VISIBLE_DEVICES",
    ):
        _validate_exact_or_unset(name, "0")
    _validate_exact_or_unset("XLA_PYTHON_CLIENT_ALLOCATOR", "bfc")
    _validate_exact_or_unset("XLA_CLIENT_MEM_FRACTION", "0.75")
    inherited_preallocation = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
    if inherited_preallocation is not None and inherited_preallocation.lower() not in {
        "0",
        "false",
        "no",
        "off",
    }:
        raise RuntimeError(
            "XLA_PYTHON_CLIENT_PREALLOCATE="
            f"{inherited_preallocation!r} conflicts with bounded growth allocation"
        )
    for name in (
        "HSA_OVERRIDE_GFX_VERSION",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB",
    ):
        _require_unset(name)
    pjrt_options = os.environ.get("JAX_PJRT_CLIENT_CREATE_OPTIONS", "")
    if pjrt_options:
        raise RuntimeError(
            "JAX_PJRT_CLIENT_CREATE_OPTIONS must be unset because it can override "
            "the bounded allocator and device contract"
        )
    os.environ.pop("JAX_PJRT_CLIENT_CREATE_OPTIONS", None)
    mock_topology = os.environ.get("JAX_MOCK_GPU_TOPOLOGY", "")
    if mock_topology:
        raise RuntimeError(
            "JAX_MOCK_GPU_TOPOLOGY must be unset because this probe requires the "
            "physical single-GPU topology"
        )
    os.environ.pop("JAX_MOCK_GPU_TOPOLOGY", None)
    mock_processes = os.environ.get("MOCK_NUM_GPU_PROCESSES", "")
    if mock_processes.strip() not in {"", "0"}:
        raise RuntimeError(
            "MOCK_NUM_GPU_PROCESSES must be unset or zero because this probe "
            "requires the physical single-process topology"
        )
    os.environ.pop("MOCK_NUM_GPU_PROCESSES", None)
    unified_memory = os.environ.get("TF_FORCE_UNIFIED_MEMORY")
    if unified_memory is not None and unified_memory.strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }:
        raise RuntimeError(
            "TF_FORCE_UNIFIED_MEMORY must be unset or false for this bounded ROCm probe"
        )
    os.environ.pop("TF_FORCE_UNIFIED_MEMORY", None)

    original_xla_flags = os.environ.get("XLA_FLAGS", "")
    os.environ.update(
        {
            "JAX_PLATFORMS": "rocm",
            "ROCR_VISIBLE_DEVICES": "0",
            "HIP_VISIBLE_DEVICES": "0",
            "GPU_DEVICE_ORDINAL": "0",
            "JAX_ROCM_VISIBLE_DEVICES": "0",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_CLIENT_MEM_FRACTION": "0.75",
            "XLA_FLAGS": _force_command_buffers_disabled(original_xla_flags),
            "SKYRL_ROCM_PALLAS_ATTENTION": "1",
        }
    )
    return {
        "JAX_PLATFORMS": os.environ["JAX_PLATFORMS"],
        "ROCR_VISIBLE_DEVICES": os.environ["ROCR_VISIBLE_DEVICES"],
        "HIP_VISIBLE_DEVICES": os.environ["HIP_VISIBLE_DEVICES"],
        "GPU_DEVICE_ORDINAL": os.environ["GPU_DEVICE_ORDINAL"],
        "JAX_ROCM_VISIBLE_DEVICES": os.environ["JAX_ROCM_VISIBLE_DEVICES"],
        "XLA_PYTHON_CLIENT_ALLOCATOR": os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"],
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"],
        "XLA_CLIENT_MEM_FRACTION": os.environ["XLA_CLIENT_MEM_FRACTION"],
        "XLA_FLAGS_original": original_xla_flags,
        "XLA_FLAGS_effective": os.environ["XLA_FLAGS"],
        "SKYRL_ROCM_PALLAS_ATTENTION": os.environ["SKYRL_ROCM_PALLAS_ATTENTION"],
    }


def _run(
    args: argparse.Namespace,
    effective_environment: dict[str, str | None],
    safety_preflight: dict[str, Any],
    require_clean_boot: Callable[[], dict[str, Any]],
) -> None:
    valid_length = (
        args.valid_length if args.valid_length is not None else args.sequence_length
    )
    score_bytes = args.batch_size * args.query_heads * args.sequence_length**2 * 4
    reference_block_score_bytes = (
        args.batch_size * args.query_heads * 16 * args.sequence_length * 4
    )

    import jax
    import jax.numpy as jnp
    from jax.extend import backend as jax_backend

    from skyrl.tx.layers.attention import dot_product_attention

    platform_version = jax_backend.get_backend().platform_version
    if jax.default_backend() != "gpu" or "rocm" not in platform_version.lower():
        raise RuntimeError(
            f"This probe requires a ROCm JAX GPU, got {jax.default_backend()}: {platform_version}"
        )

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

    pallas_backward = jax.jit(
        jax.grad(lambda *items: loss(pallas_forward, *items), argnums=(0, 1, 2))
    )

    memory_before = jax.devices()[0].memory_stats()
    pallas_output, forward_compile_seconds = _timed(jax, pallas_forward, q, k, v)
    require_clean_boot()
    forward_timings = []
    for _ in range(args.warmups - 1):
        _timed(jax, pallas_forward, q, k, v)
        require_clean_boot()
    for _ in range(args.repeats):
        _, elapsed = _timed(jax, pallas_forward, q, k, v)
        require_clean_boot()
        forward_timings.append(elapsed)

    result: dict[str, Any] = {
        "backend": jax.default_backend(),
        "platform_version": platform_version,
        "device": str(jax.devices()[0]),
        "config": {
            **vars(args),
            "valid_length": valid_length,
            "logical_score_bytes_if_unchunked": score_bytes,
            "reference_block_score_bytes": reference_block_score_bytes,
        },
        "forward": {
            "compile_and_first_seconds": forward_compile_seconds,
            "finite": bool(jnp.all(jnp.isfinite(pallas_output))),
            **_timing_summary(forward_timings),
        },
        "safety": {
            "environment": effective_environment,
            "preflight": safety_preflight,
        },
        "memory_before": memory_before,
    }

    pallas_gradients = None
    if args.backward:
        pallas_gradients, backward_compile_seconds = _timed(
            jax, pallas_backward, q, k, v
        )
        require_clean_boot()
        backward_timings = []
        for _ in range(args.warmups - 1):
            _timed(jax, pallas_backward, q, k, v)
            require_clean_boot()
        for _ in range(args.repeats):
            _, elapsed = _timed(jax, pallas_backward, q, k, v)
            require_clean_boot()
            backward_timings.append(elapsed)
        result["backward"] = {
            "compile_and_first_seconds": backward_compile_seconds,
            "finite": all(
                bool(jnp.all(jnp.isfinite(gradient))) for gradient in pallas_gradients
            ),
            **_timing_summary(backward_timings),
        }

    if args.reference:

        @jax.jit
        def generic_forward(q_arg, k_arg, v_arg):
            return _chunked_reference_attention(
                jax,
                jnp,
                q_arg,
                k_arg,
                v_arg,
                mask,
                scale,
            )

        try:
            generic_output, generic_compile_seconds = _timed(
                jax, generic_forward, q, k, v
            )
            require_clean_boot()
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
                    jax.grad(
                        lambda *items: loss(generic_forward, *items), argnums=(0, 1, 2)
                    )
                )
                generic_gradients, generic_backward_compile_seconds = _timed(
                    jax, generic_backward, q, k, v
                )
                require_clean_boot()
                result["reference"]["backward_compile_and_first_seconds"] = (
                    generic_backward_compile_seconds
                )
                result["reference"]["gradient_error"] = {
                    name: _error_metrics(jnp, actual, expected)
                    for name, actual, expected in zip(
                        ("dq", "dk", "dv"),
                        pallas_gradients,
                        generic_gradients,
                        strict=True,
                    )
                }
        except Exception as error:
            # Fail closed before any further backend query if the reference
            # surfaced or coincided with a fatal asynchronous driver event.
            require_clean_boot()
            # A failed quadratic reference is itself useful evidence. Preserve
            # the already-completed Pallas result instead of losing the run.
            result["reference"] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }

    result["safety"]["postflight_before_memory_query"] = require_clean_boot()
    result["memory_after"] = jax.devices()[0].memory_stats()
    result["safety"]["postflight"] = require_clean_boot()
    print(json.dumps(result, indent=2, sort_keys=True, default=str, allow_nan=False))


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    effective_environment = _configure_environment()
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

    with guarded_qwen35_rocm_process() as safety_preflight:
        try:
            _run(
                args,
                effective_environment,
                safety_preflight,
                require_clean_amdgpu_boot,
            )
        finally:
            # This also covers import, compilation, and execution exceptions
            # that occur before _run can build its normal JSON result.
            require_clean_amdgpu_boot()


if __name__ == "__main__":
    main()
