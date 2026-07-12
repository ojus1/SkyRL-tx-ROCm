#!/usr/bin/env python3
"""CPU-only lowering probe for the unwired Qwen3.5 GDN superblock oracle.

The probe compiles but never executes exact-shape forward and scalar-VJP
programs.  CPU ``CompiledMemoryStats`` and StableHLO structure are useful for
checking bounded scaling and reverse residuals; they are not ROCm performance
or allocator measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Callable, Sequence
from typing import Any

# Refuse inherited GPU selection before importing JAX.  This probe exists to
# inspect semantics/lowering without waking the display or compute GPU.
if os.environ.get("JAX_PLATFORMS") not in (None, "cpu"):
    raise RuntimeError("probe_gdn_superblock.py requires JAX_PLATFORMS=cpu")
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import jaxlib

from skyrl.tx.kernels.qwen3_5_gdn_superblock import (
    Qwen35GDNSuperblockConfig,
    qwen35_gdn_superblock_logical_buffers,
    qwen35_gdn_superblocks,
)
from skyrl.tx.models.qwen3_5 import chunk_gated_delta_rule


def _shape_bytes(shape: Sequence[int], dtype: jnp.dtype) -> int:
    return math.prod(shape) * jnp.dtype(dtype).itemsize


def _memory_stats(compiled: Any) -> dict[str, int]:
    stats = compiled.memory_analysis()
    return {
        name: int(getattr(stats, name))
        for name in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "alias_size_in_bytes",
            "temp_size_in_bytes",
        )
    }


def _lower(
    function: Callable[..., Any],
    arguments: tuple[jax.ShapeDtypeStruct, ...],
) -> dict[str, Any]:
    lowered = jax.jit(function).lower(*arguments)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    return {
        "stablehlo": {
            "characters": len(stablehlo),
            "lines": len(stablehlo.splitlines()),
            "dot_general": stablehlo.count("stablehlo.dot_general"),
            "while": stablehlo.count("stablehlo.while"),
            "optimization_barrier": stablehlo.count("optimization_barrier"),
        },
        "compiled_memory": _memory_stats(lowered.compile()),
    }


def _saved_residual_summary(
    loss: Callable[..., jax.Array],
    arguments: tuple[jax.ShapeDtypeStruct, ...],
) -> dict[str, Any]:
    # JAX 0.10.2 exposes the human-readable printer publicly but not the list
    # it prints.  Keep this isolated introspection optional so a future JAX
    # change cannot affect the oracle itself.
    try:
        from jax._src.ad_checkpoint import saved_residuals
    except ImportError:
        return {"available": False}

    residuals = saved_residuals(loss, *arguments)
    entries = []
    for aval, source in residuals:
        entries.append(
            {
                "shape": list(aval.shape),
                "dtype": str(aval.dtype),
                "bytes": _shape_bytes(aval.shape, aval.dtype),
                "source": source,
            }
        )
    return {
        "available": True,
        "count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument(
        "--chunks-per-superblock",
        type=int,
        choices=(8, 16),
        default=16,
    )
    parser.add_argument(
        "--allow-large-compile",
        action="store_true",
        help=(
            "allow sequence lengths above 4096; this remains CPU-only but exact "
            "current-rule VJP compilation can consume substantial host RAM"
        ),
    )
    args = parser.parse_args()
    if args.sequence_length <= 0:
        parser.error("--sequence-length must be positive")
    if args.sequence_length > 4096 and not args.allow_large_compile:
        parser.error(
            "lengths above 4096 require --allow-large-compile because exact "
            "current-rule VJP compilation can consume substantial host RAM"
        )

    batch = 1
    key_heads = 16
    value_heads = 32
    key_dim = value_dim = 128
    config = Qwen35GDNSuperblockConfig(
        chunk_size=64,
        chunks_per_superblock=args.chunks_per_superblock,
    )

    grouped_arguments = (
        jax.ShapeDtypeStruct(
            (batch, args.sequence_length, key_heads, key_dim), jnp.bfloat16
        ),
        jax.ShapeDtypeStruct(
            (batch, args.sequence_length, key_heads, key_dim), jnp.bfloat16
        ),
        jax.ShapeDtypeStruct(
            (batch, args.sequence_length, value_heads, value_dim), jnp.bfloat16
        ),
        jax.ShapeDtypeStruct((batch, args.sequence_length, value_heads), jnp.float32),
        jax.ShapeDtypeStruct((batch, args.sequence_length, value_heads), jnp.bfloat16),
        jax.ShapeDtypeStruct((batch, value_heads, key_dim, value_dim), jnp.float32),
    )
    repeated_arguments = (
        jax.ShapeDtypeStruct(
            (batch, args.sequence_length, value_heads, key_dim), jnp.bfloat16
        ),
        jax.ShapeDtypeStruct(
            (batch, args.sequence_length, value_heads, key_dim), jnp.bfloat16
        ),
        *grouped_arguments[2:],
    )

    def grouped_forward(
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        g: jax.Array,
        beta: jax.Array,
        state: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return qwen35_gdn_superblocks(
            query,
            key,
            value,
            g,
            beta,
            initial_state=state,
            config=config,
        )

    def current_forward(
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        g: jax.Array,
        beta: jax.Array,
        state: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=64,
            initial_state=state,
        )

    def scalar_loss(
        function: Callable[..., tuple[jax.Array, jax.Array]],
    ) -> Callable[..., jax.Array]:
        def loss(*values: jax.Array) -> jax.Array:
            output, state = function(*values)
            return jnp.sum(output.astype(jnp.float32)) + jnp.sum(state)

        return loss

    grouped_loss = scalar_loss(grouped_forward)
    current_loss = scalar_loss(current_forward)
    grouped_vjp = jax.grad(grouped_loss, argnums=tuple(range(6)))
    current_vjp = jax.grad(current_loss, argnums=tuple(range(6)))

    qk_repeat_bytes = 2 * _shape_bytes(
        (
            batch,
            args.sequence_length,
            value_heads - key_heads,
            key_dim,
        ),
        jnp.bfloat16,
    )
    report = {
        "jax_platform": jax.default_backend(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "executes_compiled_programs": False,
        "sequence_length": args.sequence_length,
        "config": {
            "chunk_size": config.chunk_size,
            "chunks_per_superblock": config.chunks_per_superblock,
            "tokens_per_superblock": config.tokens_per_superblock,
        },
        "qwen35_geometry": {
            "batch": batch,
            "key_heads": key_heads,
            "value_heads": value_heads,
            "key_head_dim": key_dim,
            "value_head_dim": value_dim,
        },
        "avoided_repeated_qk_argument_bytes": qk_repeat_bytes,
        "one_superblock_logical_buffers": qwen35_gdn_superblock_logical_buffers(
            batch_size=batch,
            num_value_heads=value_heads,
            key_head_dim=key_dim,
            value_head_dim=value_dim,
            config=config,
        ),
        "grouped_superblock": {
            "forward": _lower(grouped_forward, grouped_arguments),
            "vjp": _lower(grouped_vjp, grouped_arguments),
            "saved_residuals": _saved_residual_summary(grouped_loss, grouped_arguments),
        },
        "current_repeated_chunk_rule": {
            "forward": _lower(current_forward, repeated_arguments),
            "vjp": _lower(current_vjp, repeated_arguments),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
