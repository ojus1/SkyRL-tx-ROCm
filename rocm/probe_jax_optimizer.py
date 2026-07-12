"""Replay probe for SkyRL's exact Qwen3.5-4B LoRA Adam update shape.

The probe intentionally does not load the base model or run a forward pass.  It
builds the same 402-leaf, two-adapter, rank-8 LoRA parameter tree used by the
local Qwen3.5-4B configuration and repeatedly executes SkyRL's optimizer update
pattern.  This isolates repeated executable/command-buffer behavior with less
than one GiB of accelerator memory.

Safety defaults:

* CPU is the default and is forced before JAX is imported.
* GPU use requires both ``--platform gpu`` and ``--allow-gpu``.
* At least three executions are required so first-record/subsequent-replay
  behavior is observable.
* ``--command-buffer-mode disable`` applies the documented empty-set XLA flag
  before JAX is imported.

Output is JSON Lines.  No GPU run is performed merely by importing this file.
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
from typing import Any, TextIO

# Derived with nnx.eval_shape from Qwen/Qwen3.5-4B revision
# 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a and the current SkyRL Qwen3.5
# module layout.  The leading dimension is max_lora_adapters=2 and the final
# low-rank dimension is max_lora_rank=8.
_SHAPE_GROUPS: tuple[tuple[str, tuple[int, ...], int], ...] = (
    ("lora_A", (2, 2560, 8), 136),
    ("lora_B", (2, 8, 2560), 65),
    ("lora_B", (2, 8, 32), 48),
    ("lora_B", (2, 8, 18432), 32),
    ("lora_A", (2, 4096, 8), 32),
    ("lora_A", (2, 9216, 8), 32),
    ("lora_B", (2, 8, 4096), 24),
    ("lora_B", (2, 8, 8192), 24),
    ("lora_B", (2, 8, 10240), 8),
    ("lora_A", (2, 248320, 8), 1),
)
_EXPECTED_LEAVES = 402
_EXPECTED_ELEMENTS = 34_512_896
_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _append_xla_flag(original: str, flag: str) -> str:
    return " ".join(part for part in (original.strip(), flag) if part)


def _command_buffer_flag(mode: str) -> str | None:
    if mode == "disable":
        return "--xla_gpu_enable_command_buffer="
    if mode == "library-only":
        return "--xla_gpu_enable_command_buffer=CUBLAS,CUBLASLT"
    if mode == "no-fusions":
        return "--xla_gpu_enable_command_buffer=-dynamic_slice_fusion,-fusion"
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help="required acknowledgement when --platform gpu is selected",
    )
    parser.add_argument(
        "--command-buffer-mode",
        choices=("inherit", "disable", "library-only", "no-fusions"),
        default="disable",
        help=(
            "inherit XLA defaults; disable all command buffers; explicitly allow "
            "only BLAS calls; or remove ordinary/dynamic-slice fusions from defaults"
        ),
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-value", type=float, default=2**-10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.platform == "gpu" and not args.allow_gpu:
        parser.error("--platform gpu requires the explicit --allow-gpu acknowledgement")
    if args.platform == "cpu" and args.allow_gpu:
        parser.error("--allow-gpu is only valid together with --platform gpu")
    if args.steps < 3:
        parser.error("--steps must be at least 3 to exercise command-buffer replay")
    if args.steps > 1000:
        parser.error("--steps must not exceed 1000")
    numeric = (
        args.learning_rate,
        args.beta1,
        args.beta2,
        args.eps,
        args.weight_decay,
        args.gradient_value,
    )
    if not all(math.isfinite(value) for value in numeric):
        parser.error("optimizer and gradient values must be finite")
    if args.learning_rate < 0 or not 0 <= args.beta1 < 1 or not 0 <= args.beta2 < 1:
        parser.error("learning rate must be nonnegative and beta values must be in [0, 1)")
    if args.eps <= 0 or args.gradient_value == 0:
        parser.error("epsilon must be positive and gradient value must be nonzero")
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    return args


def _host_tree_stats(tree: Any, jax: Any, np: Any) -> dict[str, Any]:
    digest = hashlib.sha256()
    finite = True
    weighted_sum = 0.0
    element_count = 0
    leaves = jax.tree.leaves(jax.device_get(tree))
    for index, leaf in enumerate(leaves, start=1):
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
        numeric = array.astype(np.float64, copy=False)
        finite = finite and bool(np.isfinite(numeric).all())
        weighted_sum += index * float(numeric.sum(dtype=np.float64))
        element_count += int(array.size)
    return {
        "checksum_sha256": digest.hexdigest(),
        "finite": finite,
        "leaf_count": len(leaves),
        "element_count": element_count,
        "weighted_sum_float64": weighted_sum,
    }


def _run(args: argparse.Namespace, output: TextIO) -> None:
    # Everything above this point is standard-library-only.  Select the platform
    # and command-buffer flags before importing JAX or any SkyRL module.
    original_jax_platforms = os.environ.get("JAX_PLATFORMS")
    original_xla_flags = os.environ.get("XLA_FLAGS", "")
    # JAX's public device platform is named ``gpu``, but an explicit
    # ``JAX_PLATFORMS=gpu`` expands to multiple optional plugins in 0.10.2 and
    # fails when oneAPI is absent. Select the installed ROCm plugin directly.
    os.environ["JAX_PLATFORMS"] = "rocm" if args.platform == "gpu" else "cpu"
    command_buffer_flag = _command_buffer_flag(args.command_buffer_mode)
    if command_buffer_flag is not None:
        os.environ["XLA_FLAGS"] = _append_xla_flag(original_xla_flags, command_buffer_flag)
    effective_xla_flags = os.environ.get("XLA_FLAGS", "")
    try:
        effective_xla_flag_tokens = shlex.split(effective_xla_flags)
    except ValueError as error:
        raise ValueError(f"invalid XLA_FLAGS quoting: {error}") from error

    import jax
    import jax.numpy as jnp
    import ml_dtypes
    import numpy as np
    import optax
    from flax import nnx

    from skyrl.backends.jax import AccumulatedGradients

    resolved_backend = jax.default_backend()
    if resolved_backend != args.platform:
        raise RuntimeError(f"requested platform {args.platform!r}, resolved {resolved_backend!r}")

    class Leaf(nnx.Module):
        def __init__(self, name: str, shape: tuple[int, ...], ordinal: int):
            host = np.zeros(shape, dtype=ml_dtypes.bfloat16)
            if name == "lora_A":
                # Real LoRA A is nonzero while B starts at zero.  A deterministic
                # low-amplitude sentinel is sufficient for optimizer isolation.
                host[1].fill((ordinal % 7 + 1) * 2**-12)
            setattr(self, name, nnx.Param(jax.device_put(host)))

    class ExactLoraTree(nnx.Module):
        def __init__(self):
            leaves = []
            ordinal = 0
            for name, shape, count in _SHAPE_GROUPS:
                for _ in range(count):
                    leaves.append(Leaf(name, shape, ordinal))
                    ordinal += 1
            self.leaves = nnx.List(leaves)

    build_start = time.perf_counter()
    model = ExactLoraTree()
    _, lora_params, _ = nnx.split(model, nnx.Param, ...)
    param_leaves = jax.tree.leaves(lora_params)
    element_count = sum(math.prod(leaf.shape) for leaf in param_leaves)
    if len(param_leaves) != _EXPECTED_LEAVES or element_count != _EXPECTED_ELEMENTS:
        raise AssertionError(
            f"shape inventory mismatch: leaves={len(param_leaves)}, elements={element_count}"
        )

    optimizer = nnx.Optimizer(
        model,
        optax.inject_hyperparams(optax.adamw)(learning_rate=0.0),
        wrt=nnx.Param,
    )
    hp = optimizer.opt_state.hyperparams
    for name, value in (
        ("learning_rate", args.learning_rate),
        ("b1", args.beta1),
        ("b2", args.beta2),
        ("eps", args.eps),
        ("weight_decay", args.weight_decay),
    ):
        hp[name][...] = value

    def synthetic_gradient(param: Any) -> Any:
        host = np.zeros(param.shape, dtype=ml_dtypes.bfloat16)
        host[1].fill(args.gradient_value)
        return jax.device_put(host)

    grad_sum = jax.tree.map(synthetic_gradient, lora_params)
    template_accumulated = AccumulatedGradients(
        grad_sum=grad_sum,
        counts=jax.device_put(np.asarray([0, 1], dtype=np.int32)),
    )
    build_seconds = time.perf_counter() - build_start

    manifest = {
        "record_type": "manifest",
        "timestamp": _utc_now(),
        "argv": sys.argv,
        "platform_requested": args.platform,
        "platform_resolved": resolved_backend,
        "devices": [str(device) for device in jax.devices()],
        "allow_gpu": args.allow_gpu,
        "steps": args.steps,
        "model": "Qwen/Qwen3.5-4B",
        "model_revision": _MODEL_REVISION,
        "max_lora_adapters": 2,
        "max_lora_rank": 8,
        "adapter_index": 1,
        "shape_groups": [
            {"name": name, "shape": shape, "count": count}
            for name, shape, count in _SHAPE_GROUPS
        ],
        "parameter_leaves": len(param_leaves),
        "parameter_elements": element_count,
        "parameter_bytes_bfloat16": element_count * 2,
        "optimizer": {
            "name": "optax.inject_hyperparams(optax.adamw)",
            "learning_rate": args.learning_rate,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "eps": args.eps,
            "weight_decay": args.weight_decay,
            "gradient_value": args.gradient_value,
        },
        "environment": {
            "JAX_PLATFORMS_original": original_jax_platforms,
            "JAX_PLATFORMS_effective": os.environ["JAX_PLATFORMS"],
            "XLA_FLAGS_original": original_xla_flags,
            "XLA_FLAGS_effective": effective_xla_flags,
            "command_buffer_mode": args.command_buffer_mode,
            "command_buffer_flag_appended": command_buffer_flag,
            "XLA_FLAGS_tokens_effective": effective_xla_flag_tokens,
        },
        "versions": {
            "jax": jax.__version__,
            "optax": optax.__version__,
        },
        "build_seconds": build_seconds,
    }
    output.write(_json_dumps(manifest) + "\n")
    output.flush()

    def sentinel_diagnostics(tree: Any, opt_state: Any, adapter_index: Any) -> tuple[Any, Any, Any]:
        param_arrays = [
            leaf
            for leaf in jax.tree.leaves(tree)
            if hasattr(leaf, "ndim") and leaf.ndim > 0 and leaf.shape[0] == 2
        ]
        state_arrays = [
            leaf
            for leaf in jax.tree.leaves(opt_state)
            if hasattr(leaf, "ndim") and leaf.ndim > 0 and leaf.shape[0] == 2
        ]

        def sample(arrays: list[Any]) -> Any:
            if not arrays:
                return jnp.zeros((0,), dtype=jnp.float32)
            positions = sorted({round(i * (len(arrays) - 1) / 7) for i in range(8)})
            return jnp.stack(
                [arrays[position][adapter_index].reshape(-1)[0].astype(jnp.float32) for position in positions]
            )

        param_samples = sample(param_arrays)
        state_samples = sample(state_arrays)
        param_weights = jnp.arange(1, param_samples.size + 1, dtype=jnp.float32)
        state_weights = jnp.arange(1, state_samples.size + 1, dtype=jnp.float32)
        checksum = jnp.sum(param_samples * param_weights) + jnp.sum(state_samples * state_weights)
        finite = jnp.all(jnp.isfinite(param_samples)) & jnp.all(jnp.isfinite(state_samples))
        return checksum, finite, jnp.int32(param_samples.size + state_samples.size)

    def compute_grads_and_update(
        accumulated_grads: AccumulatedGradients,
        params: nnx.State,
        opt: nnx.Optimizer,
        adapter_index: Any,
    ) -> tuple[AccumulatedGradients, dict[str, Any]]:
        mean_grads = accumulated_grads.get_mean(adapter_index)
        grad_norm = optax.global_norm(mean_grads)
        opt.update(params, mean_grads)
        checksum, finite, sentinel_count = sentinel_diagnostics(params, opt.opt_state, adapter_index)
        return accumulated_grads.reset_adapter(adapter_index), {
            "grad_norm": grad_norm.astype(jnp.float32),
            "learning_rate": opt.opt_state.hyperparams["learning_rate"].astype(jnp.float32),
            "optimizer_step": opt.step[...],
            "sentinel_checksum": checksum,
            "sentinel_count": sentinel_count,
            "sentinels_finite": finite,
        }

    compiled_update = nnx.jit(compute_grads_and_update)
    adapter_index = jax.device_put(np.asarray(1, dtype=np.int32))
    lower_start = time.perf_counter()
    lowered = compiled_update.lower(template_accumulated, lora_params, optimizer, adapter_index)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    lower_seconds = time.perf_counter() - lower_start
    entry_line = next(line for line in stablehlo.splitlines() if "func.func public @main" in line)
    lowering_record = {
        "record_type": "lowering",
        "timestamp": _utc_now(),
        "lower_seconds": lower_seconds,
        "stablehlo_sha256": hashlib.sha256(stablehlo.encode()).hexdigest(),
        "stablehlo_chars": len(stablehlo),
        "stablehlo_lines": len(stablehlo.splitlines()),
        "entry_line_chars": len(entry_line),
        "entry_tensor_arguments": entry_line.count("%arg"),
        "entry_tensor_results": entry_line.split("->", 1)[1].count("tensor<"),
        "operation_counts": {
            operation: stablehlo.count(f"stablehlo.{operation}")
            for operation in (
                "dynamic_slice",
                "power",
                "reduce",
                "scatter",
                "select",
                "sqrt",
            )
        },
    }
    output.write(_json_dumps(lowering_record) + "\n")
    output.flush()

    previous_checksum: float | None = None
    all_step_checks_passed = True
    for step in range(1, args.steps + 1):
        start = time.perf_counter()
        _, diagnostics = compiled_update(template_accumulated, lora_params, optimizer, adapter_index)
        diagnostics = jax.device_get(diagnostics)
        elapsed = time.perf_counter() - start
        checksum = float(diagnostics["sentinel_checksum"])
        step_record = {
            "record_type": "step",
            "timestamp": _utc_now(),
            "step": step,
            "elapsed_seconds": elapsed,
            "cold_compile": step == 1,
            "expected_execution_phase": (
                "ordinary_dispatch"
                if args.command_buffer_mode == "disable"
                else ("command_buffer_record" if step == 1 else "command_buffer_replay")
            ),
            "grad_norm": float(diagnostics["grad_norm"]),
            "learning_rate": float(diagnostics["learning_rate"]),
            "optimizer_step": int(diagnostics["optimizer_step"]),
            "sentinel_checksum": checksum,
            "sentinel_count": int(diagnostics["sentinel_count"]),
            "sentinels_finite": bool(diagnostics["sentinels_finite"]),
            "checksum_changed": previous_checksum is None or checksum != previous_checksum,
        }
        step_ok = (
            step_record["sentinels_finite"]
            and math.isfinite(step_record["grad_norm"])
            and math.isfinite(checksum)
            and step_record["optimizer_step"] == step
            and step_record["checksum_changed"]
        )
        step_record["checks_passed"] = step_ok
        all_step_checks_passed = all_step_checks_passed and step_ok
        output.write(_json_dumps(step_record) + "\n")
        output.flush()
        previous_checksum = checksum

    params_host, opt_state_host = jax.device_get((lora_params, optimizer.opt_state))
    params_stats = _host_tree_stats(params_host, jax, np)
    opt_state_stats = _host_tree_stats(opt_state_host, jax, np)
    completed = all_step_checks_passed and params_stats["finite"] and opt_state_stats["finite"]
    final = {
        "record_type": "summary",
        "timestamp": _utc_now(),
        "status": "passed" if completed else "failed",
        "steps": args.steps,
        "all_step_checks_passed": all_step_checks_passed,
        "parameters": params_stats,
        "optimizer_state": opt_state_stats,
    }
    output.write(_json_dumps(final) + "\n")
    output.flush()
    if not completed:
        raise RuntimeError("optimizer replay probe completed with failed finite/checksum checks")


def main() -> int:
    # Parse and validate all safety controls before importing JAX in _run().
    args = _parse_args()
    output: TextIO
    if args.output is None:
        output = sys.stdout
        close_output = False
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        output = os.fdopen(descriptor, "w", encoding="utf-8")
        close_output = True
    try:
        _run(args, output)
        return 0
    except BaseException as error:
        output.write(
            _json_dumps(
                {
                    "record_type": "error",
                    "timestamp": _utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            + "\n"
        )
        output.flush()
        raise
    finally:
        if close_output:
            output.close()


if __name__ == "__main__":
    raise SystemExit(main())
