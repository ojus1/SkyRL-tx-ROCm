"""Guarded gfx1100 gate for the BF16 RMS + gate/up LoRA + SwiGLU stage.

The compile-only mode proves that the exact production forward and
forward-plus-VJP programs compile without invoking either executable.  The
execute mode is a separate, explicit qualification rung that checks numerical
agreement, determinism, and isolated-stage speed.  Neither mode authorizes
default model enablement; that still requires a separate end-to-end gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_DISABLE_COMMAND_BUFFERS = "--xla_gpu_enable_command_buffer="
_EXACT_GEOMETRY = (1, 64, 64, 2560, 18432, 9216, 8)
_EXACT_EPSILON = 1e-6
_SUPPORTED_BLOCK_SIZES = frozenset((16, 32, 64, 128))
_MIN_WARMUPS = 3
_MIN_ITERATIONS = 11
_RELATIVE_L2_LIMIT = 0.03
_OUTPUT_COSINE_LIMIT = 0.9999
_MIN_FORWARD_VJP_SPEEDUP = 1.10
_MIN_REMATERIALIZED_STAGE_SPEEDUP = 1.15
_EXACT_PROFILE_LIMITS = {
    "--max-junction-temp-c": 90.0,
    "--max-gpu-power-watts": 400.0,
    "--max-vram-gib": 24.0,
    "--min-host-available-gib": 0.0,
    "--max-swap-gib": 8.0,
}


def _exact_contract() -> dict[str, Any]:
    return {
        "case": "qwen35_b1_t64_bf16_rms_gate_up_lora_swiglu",
        "geometry": {
            "batch_size": 1,
            "sequence_length": 64,
            "rows": 64,
            "in_features": 2560,
            "physical_gate_up_features": 18432,
            "product_features": 9216,
            "rank": 8,
            "dtype": "bfloat16",
            "eps": _EXACT_EPSILON,
        },
        "initial_tiles": {"block_m": 16, "pair_block_n": 32, "block_k": 64},
        "target": {
            "drm_card": "card1",
            "pci_id": "1002:744c",
            "architecture": "gfx1100",
            "device_kind": "Radeon RX 7900 XTX",
        },
        "profile_limits": dict(_EXACT_PROFILE_LIMITS),
        "capture": {
            "xla_flags": _DISABLE_COMMAND_BUFFERS,
            "command_buffers_enabled": False,
            "graph_capture_enabled": False,
        },
        "compile_only": {
            "reference_executable_invocations": 0,
            "candidate_executable_invocations": 0,
        },
        "execute_gates": {
            "relative_l2_limit_exclusive": _RELATIVE_L2_LIMIT,
            "output_cosine_limit_inclusive": _OUTPUT_COSINE_LIMIT,
            "minimum_forward_and_vjp_speedup": _MIN_FORWARD_VJP_SPEEDUP,
            "minimum_rematerialized_stage_speedup": _MIN_REMATERIALIZED_STAGE_SPEEDUP,
            "deterministic_repeat_required": True,
        },
        "authorizes_default_model_enablement": False,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-gpu", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--compile-only", dest="mode", action="store_const", const="compile_only"
    )
    mode.add_argument("--execute", dest="mode", action="store_const", const="execute")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--in-features", type=int, default=2560)
    parser.add_argument("--physical-features", type=int, default=18432)
    parser.add_argument("--product-features", type=int, default=9216)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--eps", type=float, default=_EXACT_EPSILON)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=32)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=_MIN_WARMUPS)
    parser.add_argument("--iterations", type=int, default=_MIN_ITERATIONS)
    args = parser.parse_args(argv)

    if not args.allow_gpu:
        parser.error("this probe requires the explicit --allow-gpu acknowledgement")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if not args.output.is_absolute():
        parser.error("--output must be an absolute path")

    geometry = (
        args.batch_size,
        args.sequence_length,
        args.rows,
        args.in_features,
        args.physical_features,
        args.product_features,
        args.rank,
    )
    if geometry != _EXACT_GEOMETRY:
        parser.error(
            "the production gate requires exact B/T/M/K/physical-N/product-N/rank="
            f"{_EXACT_GEOMETRY}, got {geometry}"
        )
    if args.batch_size * args.sequence_length != args.rows:
        parser.error("--batch-size * --sequence-length must equal --rows")
    if args.physical_features != 2 * args.product_features:
        parser.error("--physical-features must equal 2 * --product-features")
    if not math.isfinite(args.eps) or args.eps != _EXACT_EPSILON:
        parser.error(f"--eps must be exactly {_EXACT_EPSILON}")

    for name, value, maximum in (
        ("block-m", args.block_m, args.rows),
        ("block-n", args.block_n, 128),
        ("block-k", args.block_k, 64),
    ):
        if value not in _SUPPORTED_BLOCK_SIZES or value > maximum:
            parser.error(f"--{name} is not a supported bounded tile")
    if args.rows % args.block_m:
        parser.error("--rows must be divisible by --block-m")
    if args.product_features % args.block_n:
        parser.error("--product-features must be divisible by --block-n")
    if args.in_features % args.block_k:
        parser.error("--in-features must be divisible by --block-k")
    if args.mode == "execute" and args.warmups < _MIN_WARMUPS:
        parser.error(f"execute mode requires --warmups of at least {_MIN_WARMUPS}")
    if args.mode == "execute" and args.iterations < _MIN_ITERATIONS:
        parser.error(
            f"execute mode requires --iterations of at least {_MIN_ITERATIONS}"
        )
    if args.warmups < 0 or args.iterations < 0:
        parser.error("--warmups and --iterations cannot be negative")
    return args


def _disable_command_buffers() -> tuple[str, str]:
    original = os.environ.get("XLA_FLAGS", "")
    try:
        tokens = shlex.split(original)
    except ValueError as error:
        raise RuntimeError(f"invalid XLA_FLAGS quoting: {error}") from error
    if tokens not in ([], [_DISABLE_COMMAND_BUFFERS]):
        raise RuntimeError(
            "the exact benchmark rejects inherited XLA_FLAGS other than the "
            "command-buffer disable token"
        )
    os.environ["XLA_FLAGS"] = _DISABLE_COMMAND_BUFFERS
    return original, _DISABLE_COMMAND_BUFFERS


def _configure_environment() -> dict[str, Any]:
    expected = {
        "JAX_PLATFORMS": "rocm",
        "ROCR_VISIBLE_DEVICES": "0",
        "HIP_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
    }
    allowed_names = {*expected, "XLA_FLAGS"}
    accelerator_prefixes = (
        "JAX_",
        "XLA_",
        "HSA_",
        "HIP_",
        "ROCM_",
        "ROCR_",
        "AMD_",
        "GPU_",
        "CUDA_",
        "TF_XLA_",
        "TRITON_",
        "PJRT_",
        "ROCBLAS_",
        "HIPBLASLT_",
        "TENSILE_",
    )
    inherited_accelerator_environment = {
        name: value
        for name, value in os.environ.items()
        if name.startswith(accelerator_prefixes)
    }
    unexpected = sorted(set(inherited_accelerator_environment) - allowed_names)
    if unexpected:
        raise RuntimeError(
            "the exact benchmark rejects inherited accelerator environment: "
            + ", ".join(unexpected)
        )
    for name in ("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH"):
        if os.environ.get(name):
            raise RuntimeError(f"the exact benchmark rejects inherited {name}")
    for name, value in expected.items():
        inherited = os.environ.get(name)
        if inherited is not None and inherited != value:
            raise RuntimeError(
                f"{name}={inherited!r} conflicts with required {value!r}"
            )
        os.environ[name] = value
    original, effective = _disable_command_buffers()
    return {
        **expected,
        "inherited": inherited_accelerator_environment,
        "XLA_FLAGS_original": original,
        "XLA_FLAGS_effective": effective,
        "command_buffers_enabled": False,
        "graph_capture_enabled": False,
    }


def _resolve_command_path(value: str, cwd: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else cwd / path).resolve(strict=True)


def _absolute_command_path(value: str, cwd: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else cwd / path).absolute()


def _single_flag_value(command: list[str], name: str) -> str:
    positions = [index for index, value in enumerate(command) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise RuntimeError(f"profile_rocm parent must contain one {name}")
    return command[positions[0] + 1]


def _require_profile_parent(repo: Path) -> dict[str, Any]:
    parent_pid = os.getppid()
    parent_root = Path("/proc") / str(parent_pid)
    parent_cwd = (parent_root / "cwd").resolve(strict=True)
    raw_command = (parent_root / "cmdline").read_bytes()
    if not raw_command or not raw_command.endswith(b"\0"):
        raise RuntimeError("profile_rocm parent command line is unavailable")
    try:
        command = [part.decode("utf-8") for part in raw_command[:-1].split(b"\0")]
    except UnicodeDecodeError as error:
        raise RuntimeError("profile_rocm parent command line is not UTF-8") from error
    if len(command) < 4 or command.count("--") != 1:
        raise RuntimeError("profile_rocm parent command line is malformed")
    separator = command.index("--")
    profiler_command = command[:separator]
    child_command = command[separator + 1 :]
    expected_python = (repo / ".venv" / "bin" / "python").resolve(strict=True)
    expected_python_command = (repo / ".venv" / "bin" / "python").absolute()
    expected_profiler = (repo / "rocm" / "profile_rocm.py").resolve(strict=True)
    expected_probe = Path(__file__).resolve(strict=True)
    if (parent_root / "exe").resolve(strict=True) != expected_python:
        raise RuntimeError("probe parent is not the exact project Python runtime")
    if (
        _absolute_command_path(profiler_command[0], parent_cwd)
        != expected_python_command
    ):
        raise RuntimeError("profile_rocm was not launched through the project venv")
    if (
        len(profiler_command) < 2
        or _resolve_command_path(profiler_command[1], parent_cwd) != expected_profiler
    ):
        raise RuntimeError("direct parent is not the exact profile_rocm source")
    if len(child_command) < 2:
        raise RuntimeError("profile_rocm child command is incomplete")
    if (
        _absolute_command_path(child_command[0], parent_cwd) != expected_python_command
        or _resolve_command_path(child_command[1], parent_cwd) != expected_probe
    ):
        raise RuntimeError("profile_rocm does not directly supervise this exact probe")
    if "--record-command" not in profiler_command:
        raise RuntimeError("profile_rocm must record its redacted command provenance")

    observed_limits: dict[str, float] = {}
    for name, expected in _EXACT_PROFILE_LIMITS.items():
        actual = float(_single_flag_value(profiler_command, name))
        if not math.isfinite(actual) or actual != expected:
            raise RuntimeError(
                f"profile_rocm {name} must be exactly {expected}, got {actual}"
            )
        observed_limits[name] = actual
    if _single_flag_value(profiler_command, "--card") != "card1":
        raise RuntimeError("profile_rocm must monitor exact AMD DRM device card1")
    timeout = float(_single_flag_value(profiler_command, "--timeout"))
    interval = float(_single_flag_value(profiler_command, "--interval"))
    baseline_seconds = float(_single_flag_value(profiler_command, "--baseline-seconds"))
    sensor_grace_seconds = float(
        _single_flag_value(profiler_command, "--sensor-grace-seconds")
    )
    if not 0 < timeout <= 1800:
        raise RuntimeError("profile_rocm timeout must be in (0, 1800]")
    if not 0 < interval <= 0.25:
        raise RuntimeError("profile_rocm interval must be in (0, 0.25]")
    if baseline_seconds < 2:
        raise RuntimeError("profile_rocm baseline must be at least 2 seconds")
    if not 0 <= sensor_grace_seconds <= 60:
        raise RuntimeError("profile_rocm sensor grace must be in [0, 60]")

    return {
        "validated": True,
        "parent_pid": parent_pid,
        "parent_executable": str(expected_python),
        "parent_command_python": str(expected_python_command),
        "parent_command_sha256": hashlib.sha256(raw_command).hexdigest(),
        "profiler_path": str(expected_profiler),
        "profiler_sha256": hashlib.sha256(expected_profiler.read_bytes()).hexdigest(),
        "limits": observed_limits,
        "timeout_seconds": timeout,
        "interval_seconds": interval,
        "baseline_seconds": baseline_seconds,
        "sensor_grace_seconds": sensor_grace_seconds,
    }


def _require_exact_card_identity(
    card_root: Path = Path("/sys/class/drm/card1/device"),
) -> dict[str, str]:
    resolved = card_root.resolve(strict=True)
    vendor = (card_root / "vendor").read_text().strip().lower()
    device = (card_root / "device").read_text().strip().lower()
    driver = (card_root / "driver").resolve(strict=True).name
    if vendor != "0x1002" or device != "0x744c" or driver != "amdgpu":
        raise RuntimeError(
            "exact card1 identity must be AMD 1002:744c/gfx1100 using amdgpu, got "
            f"vendor={vendor}, device={device}, driver={driver}"
        )
    return {
        "drm_card": "card1",
        "pci_vendor": vendor,
        "pci_device": device,
        "driver": driver,
        "architecture": "gfx1100",
        "pci_bdf": resolved.name,
        "sysfs_device": str(resolved),
    }


def _block_tree(tree: Any) -> Any:
    import jax

    return jax.tree.map(
        lambda value: (
            value.block_until_ready() if hasattr(value, "block_until_ready") else value
        ),
        tree,
    )


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    actual_f64 = actual.astype(np.float64)
    expected_f64 = expected.astype(np.float64)
    denominator = max(float(np.linalg.norm(expected_f64)), 1e-12)
    return float(np.linalg.norm(actual_f64 - expected_f64) / denominator)


def _cosine_similarity(actual: np.ndarray, expected: np.ndarray) -> float:
    actual_f64 = actual.astype(np.float64).reshape(-1)
    expected_f64 = expected.astype(np.float64).reshape(-1)
    denominator = float(np.linalg.norm(actual_f64) * np.linalg.norm(expected_f64))
    if denominator == 0:
        return 1.0 if np.array_equal(actual_f64, expected_f64) else 0.0
    cosine = float(np.dot(actual_f64, expected_f64) / denominator)
    return min(1.0, max(-1.0, cosine))


def _error_manifest(actual_tree: Any, expected_tree: Any) -> dict[str, Any]:
    import jax

    names = ("output", "dx", "d_lora_a", "d_lora_b")
    actual_leaves = jax.tree.leaves(actual_tree)
    expected_leaves = jax.tree.leaves(expected_tree)
    if len(actual_leaves) != len(names) or len(expected_leaves) != len(names):
        raise RuntimeError("unexpected benchmark result tree")
    errors: dict[str, Any] = {}
    for name, actual, expected in zip(
        names, actual_leaves, expected_leaves, strict=True
    ):
        actual_host = np.asarray(actual, dtype=np.float32)
        expected_host = np.asarray(expected, dtype=np.float32)
        errors[name] = {
            "relative_l2": _relative_l2(actual_host, expected_host),
            "cosine_similarity": _cosine_similarity(actual_host, expected_host),
            "max_absolute": float(np.max(np.abs(actual_host - expected_host))),
            "finite": bool(
                np.all(np.isfinite(actual_host)) and np.all(np.isfinite(expected_host))
            ),
        }
    return errors


def _numerics_gate_passed(errors: dict[str, Any]) -> bool:
    return bool(
        all(manifest["finite"] for manifest in errors.values())
        and all(
            manifest["relative_l2"] < _RELATIVE_L2_LIMIT for manifest in errors.values()
        )
        and errors["output"]["cosine_similarity"] >= _OUTPUT_COSINE_LIMIT
    )


def _performance_gate_passed(
    *,
    forward_vjp_speedup: float,
    rematerialized_speedup: float,
    candidate_seconds: list[float],
    candidate_forward_seconds: list[float],
) -> bool:
    samples = [*candidate_seconds, *candidate_forward_seconds]
    return bool(
        math.isfinite(forward_vjp_speedup)
        and math.isfinite(rematerialized_speedup)
        and forward_vjp_speedup >= _MIN_FORWARD_VJP_SPEEDUP
        and rematerialized_speedup >= _MIN_REMATERIALIZED_STAGE_SPEEDUP
        and samples
        and all(math.isfinite(sample) and sample > 0 for sample in samples)
    )


def _private_write(path: Path, payload: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(payload, allow_nan=False, sort_keys=True))
        output.write("\n")


def _package_versions() -> dict[str, str | None]:
    names = ("jax", "jaxlib", "jax-rocm7-pjrt", "jax-rocm7-plugin", "numpy")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git_manifest(repo: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.rstrip("\n")

    status = run("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": status.splitlines(),
        "clean": not status,
    }


def _compile_callable(
    jax: Any, callable_: Any, arguments: tuple[Any, ...]
) -> tuple[Any, float, float]:
    lower_started = time.perf_counter()
    lowered = jax.jit(callable_).lower(*arguments)
    lower_seconds = time.perf_counter() - lower_started
    compile_started = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - compile_started
    return executable, lower_seconds, compile_seconds


def _alternating_order(iteration: int) -> tuple[str, str]:
    return (
        ("reference", "candidate") if iteration % 2 == 0 else ("candidate", "reference")
    )


def _run_compiled_workload(
    *,
    mode: str,
    warmups: int,
    iterations: int,
    executables: dict[str, Any],
    step_arguments: tuple[Any, ...] | None,
    forward_arguments: tuple[Any, ...] | None,
) -> dict[str, Any]:
    counts = {
        "reference_forward": 0,
        "candidate_forward": 0,
        "reference_forward_and_vjp": 0,
        "candidate_forward_and_vjp": 0,
    }
    if mode == "compile_only":
        return {
            "invocation_counts": counts,
            "compile_only_zero_candidate_reference_executable_invocations": True,
            "warmup_orders": [],
            "measurement_orders": [],
        }
    if mode != "execute":
        raise ValueError(f"unsupported workload mode: {mode}")
    if step_arguments is None or forward_arguments is None:
        raise RuntimeError("execute mode requires concrete device arguments")

    def invoke(key: str, arguments: tuple[Any, ...]) -> Any:
        counts[key] += 1
        return _block_tree(executables[key](*arguments))

    warmup_orders: list[dict[str, Any]] = []
    for iteration in range(warmups):
        step_order = _alternating_order(iteration)
        forward_order = _alternating_order(iteration + 1)
        warmup_orders.append({"forward_and_vjp": step_order, "forward": forward_order})
        for name in step_order:
            invoke(f"{name}_forward_and_vjp", step_arguments)
        for name in forward_order:
            invoke(f"{name}_forward", forward_arguments)

    def measure_pair(suffix: str, arguments: tuple[Any, ...], offset: int):
        samples = {"reference": [], "candidate": []}
        orders: list[tuple[str, str]] = []
        for iteration in range(iterations):
            order = _alternating_order(iteration + offset)
            orders.append(order)
            for name in order:
                started = time.perf_counter()
                invoke(f"{name}_{suffix}", arguments)
                samples[name].append(time.perf_counter() - started)
        return samples, orders

    step_samples, step_orders = measure_pair("forward_and_vjp", step_arguments, warmups)
    forward_samples, forward_orders = measure_pair(
        "forward", forward_arguments, warmups + 1
    )
    expected = invoke("reference_forward_and_vjp", step_arguments)
    actual = invoke("candidate_forward_and_vjp", step_arguments)
    actual_repeat = invoke("candidate_forward_and_vjp", step_arguments)
    return {
        "invocation_counts": counts,
        "compile_only_zero_candidate_reference_executable_invocations": False,
        "warmup_orders": warmup_orders,
        "measurement_orders": {
            "forward_and_vjp": step_orders,
            "forward": forward_orders,
        },
        "step_samples": step_samples,
        "forward_samples": forward_samples,
        "expected": expected,
        "actual": actual,
        "actual_repeat": actual_repeat,
    }


def _run(args: argparse.Namespace, preflight: dict[str, Any]) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from jax.extend import backend as jax_backend

    kernel_module = importlib.import_module(
        "skyrl.tx.kernels.rocm.bf16_rms_gate_up_lora_swiglu"
    )
    candidate_operation = kernel_module.bf16_rms_gate_up_lora_swiglu

    platform_version = jax_backend.get_backend().platform_version
    if jax.default_backend() != "gpu" or "rocm" not in platform_version.lower():
        raise RuntimeError(
            f"expected a ROCm JAX GPU, got {jax.default_backend()!r}: {platform_version}"
        )
    devices = jax.devices()
    if len(devices) != 1:
        raise RuntimeError(f"expected exactly one visible ROCm device, got {devices}")
    if devices[0].device_kind != "Radeon RX 7900 XTX":
        raise RuntimeError(
            "the exact gate requires Radeon RX 7900 XTX/gfx1100, got "
            f"{devices[0].device_kind!r}"
        )

    def reference_forward(x, rms_delta, weight, a, b, scale):
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
        normalized = (
            x_f32
            * jax.lax.rsqrt(variance + args.eps)
            * (1.0 + rms_delta.astype(jnp.float32))
        ).astype(jnp.bfloat16)
        flat_normalized = normalized.reshape((-1, normalized.shape[-1]))
        z = (flat_normalized @ a).astype(jnp.bfloat16)
        base = (flat_normalized @ weight).astype(jnp.bfloat16)
        low_rank = (z @ b).astype(jnp.bfloat16)
        projection = (base + (low_rank * scale).astype(jnp.bfloat16)).astype(
            jnp.bfloat16
        )
        gate = projection[:, 0::2]
        up = projection[:, 1::2]
        product = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
        return product.reshape((*x.shape[:-1], args.product_features))

    def candidate_forward(x, rms_delta, weight, a, b, scale):
        return candidate_operation(
            x,
            rms_delta,
            weight,
            a,
            b,
            scale,
            enabled=True,
            eps=args.eps,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
        )

    def make_step(forward):
        def step(x, rms_delta, weight, a, b, scale, cotangent):
            def objective(x_arg, a_arg, b_arg):
                output = forward(x_arg, rms_delta, weight, a_arg, b_arg, scale)
                loss = jnp.sum(
                    output.astype(jnp.float32) * cotangent.astype(jnp.float32)
                )
                return loss, output

            (_, output), gradients = jax.value_and_grad(
                objective, argnums=(0, 1, 2), has_aux=True
            )(x, a, b)
            return (output, *gradients)

        return step

    reference_step = make_step(reference_forward)
    candidate_step = make_step(candidate_forward)
    signature_arguments = (
        jax.ShapeDtypeStruct(
            (args.batch_size, args.sequence_length, args.in_features), jnp.bfloat16
        ),
        jax.ShapeDtypeStruct((args.in_features,), jnp.bfloat16),
        jax.ShapeDtypeStruct((args.in_features, args.physical_features), jnp.bfloat16),
        jax.ShapeDtypeStruct((args.in_features, args.rank), jnp.bfloat16),
        jax.ShapeDtypeStruct((args.rank, args.physical_features), jnp.bfloat16),
        jax.ShapeDtypeStruct((), jnp.bfloat16),
        jax.ShapeDtypeStruct(
            (args.batch_size, args.sequence_length, args.product_features), jnp.bfloat16
        ),
    )

    concrete_arguments: tuple[Any, ...] | None = None
    if args.mode == "execute":
        rng = np.random.default_rng(20260714)

        def device_bf16(shape: tuple[int, ...], scale: float = 1.0):
            host = rng.standard_normal(shape, dtype=np.float32) * scale
            return jax.device_put(jnp.asarray(host, dtype=jnp.bfloat16), devices[0])

        concrete_arguments = (
            device_bf16(
                (args.batch_size, args.sequence_length, args.in_features), 0.02
            ),
            device_bf16((args.in_features,), 0.01),
            device_bf16(
                (args.in_features, args.physical_features),
                1.0 / math.sqrt(args.in_features),
            ),
            device_bf16((args.in_features, args.rank), 0.01),
            device_bf16((args.rank, args.physical_features), 0.01),
            jax.device_put(jnp.asarray(2.0, dtype=jnp.bfloat16), devices[0]),
            device_bf16(
                (args.batch_size, args.sequence_length, args.product_features), 0.02
            ),
        )
    compile_arguments = (
        concrete_arguments if concrete_arguments is not None else signature_arguments
    )
    compile_forward_arguments = compile_arguments[:-1]

    callables = {
        "reference_forward_and_vjp": (reference_step, compile_arguments),
        "candidate_forward_and_vjp": (candidate_step, compile_arguments),
        "reference_forward": (reference_forward, compile_forward_arguments),
        "candidate_forward": (candidate_forward, compile_forward_arguments),
    }
    executables: dict[str, Any] = {}
    compilation: dict[str, Any] = {}
    for name, (callable_, callable_arguments) in callables.items():
        executable, lower_seconds, compile_seconds = _compile_callable(
            jax, callable_, callable_arguments
        )
        executables[name] = executable
        compilation[name] = {
            "lower_calls": 1,
            "compile_calls": 1,
            "lower_seconds": lower_seconds,
            "compile_seconds": compile_seconds,
        }

    workload = _run_compiled_workload(
        mode=args.mode,
        warmups=args.warmups,
        iterations=args.iterations,
        executables=executables,
        step_arguments=concrete_arguments,
        forward_arguments=(
            concrete_arguments[:-1] if concrete_arguments is not None else None
        ),
    )
    numerics: dict[str, Any]
    determinism: dict[str, Any]
    measurement: dict[str, Any]
    performance_gate: dict[str, Any]
    recommendation = False
    if args.mode == "compile_only":
        numerics = {"executed": False, "passed": None}
        determinism = {"executed": False, "passed": None}
        measurement = {
            "executed": False,
            "warmups": 0,
            "iterations": 0,
            "requested_warmups_ignored": args.warmups,
            "requested_iterations_ignored": args.iterations,
        }
        performance_gate = {
            "executed": False,
            "passed": None,
            "minimum_forward_and_vjp_speedup": _MIN_FORWARD_VJP_SPEEDUP,
            "minimum_rematerialized_stage_speedup": _MIN_REMATERIALIZED_STAGE_SPEEDUP,
        }
        passed = all(
            manifest["lower_calls"] == 1 and manifest["compile_calls"] == 1
            for manifest in compilation.values()
        ) and not any(workload["invocation_counts"].values())
    else:
        errors = _error_manifest(workload["actual"], workload["expected"])
        numerics_passed = _numerics_gate_passed(errors)
        actual_leaves = jax.tree.leaves(workload["actual"])
        repeat_leaves = jax.tree.leaves(workload["actual_repeat"])
        deterministic = all(
            np.array_equal(np.asarray(left), np.asarray(right))
            for left, right in zip(actual_leaves, repeat_leaves, strict=True)
        )
        step_samples = workload["step_samples"]
        forward_samples = workload["forward_samples"]
        reference_step_median = statistics.median(step_samples["reference"])
        candidate_step_median = statistics.median(step_samples["candidate"])
        reference_forward_median = statistics.median(forward_samples["reference"])
        candidate_forward_median = statistics.median(forward_samples["candidate"])
        forward_vjp_speedup = reference_step_median / candidate_step_median
        forward_speedup = reference_forward_median / candidate_forward_median
        rematerialized_reference = reference_step_median + reference_forward_median
        rematerialized_candidate = candidate_step_median + candidate_forward_median
        rematerialized_speedup = rematerialized_reference / rematerialized_candidate
        performance_passed = _performance_gate_passed(
            forward_vjp_speedup=forward_vjp_speedup,
            rematerialized_speedup=rematerialized_speedup,
            candidate_seconds=step_samples["candidate"],
            candidate_forward_seconds=forward_samples["candidate"],
        )
        numerics = {
            "executed": True,
            "passed": numerics_passed,
            "relative_l2_limit_exclusive": _RELATIVE_L2_LIMIT,
            "output_cosine_limit_inclusive": _OUTPUT_COSINE_LIMIT,
            "errors": errors,
        }
        determinism = {
            "executed": True,
            "passed": deterministic,
            "repeated_candidate_result_tree_bitwise_equal": deterministic,
        }
        measurement = {
            "executed": True,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "alternating_warmup_orders": workload["warmup_orders"],
            "alternating_measurement_orders": workload["measurement_orders"],
            "forward_and_vjp": {
                "reference_seconds": step_samples["reference"],
                "candidate_seconds": step_samples["candidate"],
                "reference_median_seconds": reference_step_median,
                "candidate_median_seconds": candidate_step_median,
                "speedup": forward_vjp_speedup,
            },
            "forward": {
                "reference_seconds": forward_samples["reference"],
                "candidate_seconds": forward_samples["candidate"],
                "reference_median_seconds": reference_forward_median,
                "candidate_median_seconds": candidate_forward_median,
                "speedup": forward_speedup,
            },
            "rematerialized_stage_estimate": {
                "reference_seconds": rematerialized_reference,
                "candidate_seconds": rematerialized_candidate,
                "speedup": rematerialized_speedup,
            },
        }
        performance_gate = {
            "executed": True,
            "passed": performance_passed,
            "minimum_forward_and_vjp_speedup": _MIN_FORWARD_VJP_SPEEDUP,
            "minimum_rematerialized_stage_speedup": _MIN_REMATERIALIZED_STAGE_SPEEDUP,
        }
        recommendation = bool(numerics_passed and deterministic and performance_passed)
        passed = recommendation

    probe_path = Path(__file__).resolve(strict=True)
    repo = probe_path.parent.parent
    source_path = Path(kernel_module.__file__).resolve(strict=True)
    expected_source_path = (
        repo / "skyrl" / "tx" / "kernels" / "rocm" / "bf16_rms_gate_up_lora_swiglu.py"
    ).resolve(strict=True)
    if source_path != expected_source_path:
        raise RuntimeError(
            f"imported kernel source {source_path} is not exact repo source {expected_source_path}"
        )
    profiler_path = (repo / "rocm" / "profile_rocm.py").resolve(strict=True)
    invocation_counts = workload["invocation_counts"]
    reference_invocations = (
        invocation_counts["reference_forward"]
        + invocation_counts["reference_forward_and_vjp"]
    )
    candidate_invocations = (
        invocation_counts["candidate_forward"]
        + invocation_counts["candidate_forward_and_vjp"]
    )
    return {
        "schema_version": 1,
        "mode": args.mode,
        "qualification_scope": "isolated_stage_only",
        "authorizes_default_model_enablement": False,
        "recommend_for_opt_in_model_integration": recommendation,
        "passed": bool(passed),
        "contract": _exact_contract(),
        "preflight": preflight,
        "device": {
            "backend": jax.default_backend(),
            "device_kind": devices[0].device_kind,
            "architecture": "gfx1100",
            "platform_version": platform_version,
        },
        "geometry": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "rows": args.rows,
            "in_features": args.in_features,
            "physical_gate_up_features": args.physical_features,
            "product_features": args.product_features,
            "rank": args.rank,
            "dtype": "bfloat16",
            "eps": args.eps,
            "block_m": args.block_m,
            "pair_block_n": args.block_n,
            "block_k": args.block_k,
        },
        "invocation_contract": {
            "per_program_executable_invocations": invocation_counts,
            "reference_executable_invocations": reference_invocations,
            "candidate_executable_invocations": candidate_invocations,
            "total_executable_invocations": reference_invocations
            + candidate_invocations,
            "compile_only_zero_candidate_reference_executable_invocations": workload[
                "compile_only_zero_candidate_reference_executable_invocations"
            ],
        },
        "compilation": compilation,
        "measurement": measurement,
        "performance_gate": performance_gate,
        "numerics": numerics,
        "determinism": determinism,
        "source": {
            "kernel": {
                "path": str(source_path),
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            },
            "probe": {
                "path": str(probe_path),
                "sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
            },
            "profiler": {
                "path": str(profiler_path),
                "sha256": hashlib.sha256(profiler_path.read_bytes()).hexdigest(),
            },
            "git": _git_manifest(repo),
            "packages": _package_versions(),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    preloaded = sorted(
        name
        for name in sys.modules
        if name == "jax"
        or name.startswith("jax.")
        or name == "jaxlib"
        or name.startswith("jaxlib.")
        or name == "flax"
        or name.startswith("flax.")
        or name.startswith("skyrl.tx.kernels")
    )
    if preloaded:
        raise RuntimeError(
            "the exact ROCm benchmark requires a fresh process before JAX/kernel "
            f"import; already loaded: {preloaded[:8]}"
        )
    repo = Path(__file__).resolve(strict=True).parent.parent
    expected_prefix = (repo / ".venv").resolve(strict=True)
    if Path(sys.prefix).resolve(strict=True) != expected_prefix:
        raise RuntimeError(
            f"exact benchmark requires project venv {expected_prefix}, got {sys.prefix}"
        )
    expected_executable = (repo / ".venv" / "bin" / "python").absolute()
    if Path(sys.executable).absolute() != expected_executable:
        raise RuntimeError(
            f"exact benchmark requires {expected_executable}, got {sys.executable}"
        )
    profiler_parent = _require_profile_parent(repo)
    card_identity = _require_exact_card_identity()
    environment = _configure_environment()
    try:
        safety_module = importlib.import_module("rocm.amdgpu_safety")
    except ModuleNotFoundError:
        safety_module = importlib.import_module("amdgpu_safety")
    safety_path = Path(safety_module.__file__).resolve(strict=True)
    expected_safety_path = (repo / "rocm" / "amdgpu_safety.py").resolve(strict=True)
    if safety_path != expected_safety_path:
        raise RuntimeError(
            f"imported safety source {safety_path} is not exact repo source {expected_safety_path}"
        )

    with safety_module.guarded_qwen35_rocm_process() as hardware_preflight:
        try:
            payload = _run(
                args,
                {
                    "environment": environment,
                    "hardware": hardware_preflight,
                    "profiler_parent": profiler_parent,
                    "card_identity": card_identity,
                    "safety_source": {
                        "path": str(safety_path),
                        "sha256": hashlib.sha256(safety_path.read_bytes()).hexdigest(),
                    },
                },
            )
        finally:
            postflight = safety_module.require_clean_amdgpu_boot()
        payload["postflight"] = postflight
    _private_write(args.output, payload)
    print(json.dumps(payload, allow_nan=False, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
