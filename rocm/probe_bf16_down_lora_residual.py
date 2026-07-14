"""Guarded isolated-stage gate for the BF16 down+LoRA+residual candidate.

A pass qualifies this one operation for opt-in model integration.  It does not
authorize default enablement; that still requires the repository's separate
end-to-end throughput or peak-memory promotion gate.
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
_EXACT_GEOMETRY = (64, 9216, 2560, 8)
_MIN_WARMUPS = 3
_MIN_ITERATIONS = 11
_OUTPUT_RELATIVE_L2_LIMIT = 0.005
_OUTPUT_COSINE_LIMIT = 0.9999
_GRADIENT_RELATIVE_L2_LIMIT = 0.01
_GRADIENT_COSINE_LIMIT = 0.999
_MIN_FORWARD_VJP_SPEEDUP = 1.03
_MIN_REMATERIALIZED_STAGE_SPEEDUP = 1.05
_MAX_DISPATCH_SECONDS = 0.1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--in-features", type=int, default=9216)
    parser.add_argument("--out-features", type=int, default=2560)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--row-superblock", type=int, default=256)
    parser.add_argument("--pallas-input-vjp", action="store_true")
    parser.add_argument("--warmups", type=int, default=_MIN_WARMUPS)
    parser.add_argument("--iterations", type=int, default=_MIN_ITERATIONS)
    args = parser.parse_args(argv)
    if not args.allow_gpu:
        parser.error("this probe requires the explicit --allow-gpu acknowledgement")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if not args.output.is_absolute():
        parser.error("--output must be an absolute path")
    geometry = (args.rows, args.in_features, args.out_features, args.rank)
    if geometry != _EXACT_GEOMETRY:
        parser.error(
            "the production gate requires exact "
            f"M/K/N/rank={_EXACT_GEOMETRY}, got {geometry}"
        )
    for name in ("rows", "in_features", "out_features", "warmups", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmups < _MIN_WARMUPS:
        parser.error(f"--warmups must be at least {_MIN_WARMUPS}")
    if args.iterations < _MIN_ITERATIONS:
        parser.error(f"--iterations must be at least {_MIN_ITERATIONS}")
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
    effective = _DISABLE_COMMAND_BUFFERS
    os.environ["XLA_FLAGS"] = effective
    return original, effective


def _configure_environment() -> dict[str, str]:
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

    exact_limits = {
        "--max-junction-temp-c": 90.0,
        "--max-gpu-power-watts": 400.0,
        "--max-vram-gib": 24.0,
        "--min-host-available-gib": 0.0,
        "--max-swap-gib": 8.0,
    }
    observed_limits: dict[str, float] = {}
    for name, expected in exact_limits.items():
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
    if not 0 < timeout <= 900:
        raise RuntimeError("profile_rocm timeout must be in (0, 900]")
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
            "exact card1 identity must be AMD 1002:744c using amdgpu, got "
            f"vendor={vendor}, device={device}, driver={driver}"
        )
    return {
        "drm_card": "card1",
        "pci_vendor": vendor,
        "pci_device": device,
        "driver": driver,
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


def _timed(callable_, arguments: tuple[Any, ...]) -> float:
    started = time.perf_counter()
    _block_tree(callable_(*arguments))
    return time.perf_counter() - started


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

    names = ("output", "dx", "d_lora_a", "d_lora_b", "d_residual")
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
    errors["d_residual"]["bitwise_equal"] = bool(
        np.array_equal(np.asarray(actual_leaves[-1]), np.asarray(expected_leaves[-1]))
    )
    return errors


def _numerics_gate_passed(errors: dict[str, Any]) -> bool:
    finite = all(manifest["finite"] for manifest in errors.values())
    output_error = errors["output"]
    gradient_errors = [errors[name] for name in ("dx", "d_lora_a", "d_lora_b")]
    return bool(
        finite
        and output_error["relative_l2"] <= _OUTPUT_RELATIVE_L2_LIMIT
        and output_error["cosine_similarity"] >= _OUTPUT_COSINE_LIMIT
        and all(
            manifest["relative_l2"] <= _GRADIENT_RELATIVE_L2_LIMIT
            and manifest["cosine_similarity"] >= _GRADIENT_COSINE_LIMIT
            for manifest in gradient_errors
        )
        and errors["d_residual"]["bitwise_equal"]
    )


def _performance_gate_passed(
    *,
    forward_vjp_speedup: float,
    rematerialized_speedup: float,
    candidate_seconds: list[float],
    candidate_forward_seconds: list[float],
) -> bool:
    return bool(
        forward_vjp_speedup >= _MIN_FORWARD_VJP_SPEEDUP
        and rematerialized_speedup >= _MIN_REMATERIALIZED_STAGE_SPEEDUP
        and max(candidate_seconds) < _MAX_DISPATCH_SECONDS
        and max(candidate_forward_seconds) < _MAX_DISPATCH_SECONDS
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


def _run(args: argparse.Namespace, preflight: dict[str, Any]) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from jax.extend import backend as jax_backend

    kernel_module = importlib.import_module("skyrl.tx.kernels.rocm.bf16_lora_residual")
    bf16_lora_residual = kernel_module.bf16_lora_residual

    platform_version = jax_backend.get_backend().platform_version
    if jax.default_backend() != "gpu" or "rocm" not in platform_version.lower():
        raise RuntimeError(
            f"expected a ROCm JAX GPU, got {jax.default_backend()!r}: "
            f"{platform_version}"
        )
    devices = jax.devices()
    if len(devices) != 1:
        raise RuntimeError(f"expected exactly one visible ROCm device, got {devices}")
    if devices[0].device_kind != "Radeon RX 7900 XTX":
        raise RuntimeError(
            "the exact gate requires Radeon RX 7900 XTX/gfx1100, got "
            f"{devices[0].device_kind!r}"
        )

    rng = np.random.default_rng(20260714)

    def device_bf16(shape: tuple[int, ...], scale: float = 1.0):
        host = (rng.standard_normal(shape, dtype=np.float32) * scale).astype(np.float32)
        return jax.device_put(jnp.asarray(host, dtype=jnp.bfloat16), devices[0])

    x = device_bf16((args.rows, args.in_features), 0.02)
    frozen_weight = device_bf16(
        (args.in_features, args.out_features), 1.0 / math.sqrt(args.in_features)
    )
    lora_a = device_bf16((args.in_features, args.rank), 0.01)
    lora_b = device_bf16((args.rank, args.out_features), 0.01)
    residual = device_bf16((args.rows, args.out_features), 0.02)
    cotangent = device_bf16((args.rows, args.out_features), 0.02)
    scaling = jax.device_put(jnp.asarray(2.0, dtype=jnp.bfloat16), devices[0])
    arguments = (x, frozen_weight, lora_a, lora_b, scaling, residual, cotangent)
    forward_arguments = arguments[:-1]

    def baseline_forward(x, weight, a, b, scale, residual):
        return x @ weight + ((x @ a) @ b) * scale + residual

    def candidate_forward(x, weight, a, b, scale, residual):
        return bf16_lora_residual(
            x,
            weight,
            a,
            b,
            scale,
            residual,
            enabled=True,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            row_superblock=args.row_superblock,
            pallas_input_vjp=args.pallas_input_vjp,
        )

    def baseline_step(x, weight, a, b, scale, residual, cotangent):
        def objective(x_arg, a_arg, b_arg, residual_arg):
            output = x_arg @ weight + ((x_arg @ a_arg) @ b_arg) * scale
            output = output + residual_arg
            loss = jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32))
            return loss, output

        (_, output), gradients = jax.value_and_grad(
            objective, argnums=(0, 1, 2, 3), has_aux=True
        )(x, a, b, residual)
        return (output, *gradients)

    def candidate_step(x, weight, a, b, scale, residual, cotangent):
        def objective(x_arg, a_arg, b_arg, residual_arg):
            output = bf16_lora_residual(
                x_arg,
                weight,
                a_arg,
                b_arg,
                scale,
                residual_arg,
                enabled=True,
                block_m=args.block_m,
                block_n=args.block_n,
                block_k=args.block_k,
                row_superblock=args.row_superblock,
                pallas_input_vjp=args.pallas_input_vjp,
            )
            loss = jnp.sum(output.astype(jnp.float32) * cotangent.astype(jnp.float32))
            return loss, output

        (_, output), gradients = jax.value_and_grad(
            objective, argnums=(0, 1, 2, 3), has_aux=True
        )(x, a, b, residual)
        return (output, *gradients)

    def compile_callable(callable_, callable_arguments):
        lower_started = time.perf_counter()
        lowered = jax.jit(callable_).lower(*callable_arguments)
        lower_seconds = time.perf_counter() - lower_started
        compile_started = time.perf_counter()
        executable = lowered.compile()
        compile_seconds = time.perf_counter() - compile_started
        return executable, lower_seconds, compile_seconds

    baseline, baseline_lower_seconds, baseline_compile_seconds = compile_callable(
        baseline_step, arguments
    )
    candidate, candidate_lower_seconds, candidate_compile_seconds = compile_callable(
        candidate_step, arguments
    )
    (
        baseline_forward_executable,
        baseline_forward_lower_seconds,
        baseline_forward_compile_seconds,
    ) = compile_callable(baseline_forward, forward_arguments)
    (
        candidate_forward_executable,
        candidate_forward_lower_seconds,
        candidate_forward_compile_seconds,
    ) = compile_callable(candidate_forward, forward_arguments)

    for _ in range(args.warmups):
        _block_tree(baseline(*arguments))
        _block_tree(candidate(*arguments))
        _block_tree(baseline_forward_executable(*forward_arguments))
        _block_tree(candidate_forward_executable(*forward_arguments))

    def measure_pair(baseline_executable, candidate_executable, callable_arguments):
        baseline_samples: list[float] = []
        candidate_samples: list[float] = []
        for iteration in range(args.iterations):
            order = (
                (
                    (baseline_executable, baseline_samples),
                    (candidate_executable, candidate_samples),
                )
                if iteration % 2 == 0
                else (
                    (candidate_executable, candidate_samples),
                    (baseline_executable, baseline_samples),
                )
            )
            for executable, samples in order:
                samples.append(_timed(executable, callable_arguments))
        return baseline_samples, candidate_samples

    baseline_seconds, candidate_seconds = measure_pair(baseline, candidate, arguments)
    baseline_forward_seconds, candidate_forward_seconds = measure_pair(
        baseline_forward_executable,
        candidate_forward_executable,
        forward_arguments,
    )

    expected = _block_tree(baseline(*arguments))
    actual = _block_tree(candidate(*arguments))
    actual_repeat = _block_tree(candidate(*arguments))
    errors = _error_manifest(actual, expected)
    deterministic = all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(
            jax.tree.leaves(actual), jax.tree.leaves(actual_repeat), strict=True
        )
    )
    max_relative_l2 = max(manifest["relative_l2"] for manifest in errors.values())
    baseline_median = statistics.median(baseline_seconds)
    candidate_median = statistics.median(candidate_seconds)
    baseline_forward_median = statistics.median(baseline_forward_seconds)
    candidate_forward_median = statistics.median(candidate_forward_seconds)
    forward_vjp_speedup = baseline_median / candidate_median
    rematerialized_baseline = baseline_median + baseline_forward_median
    rematerialized_candidate = candidate_median + candidate_forward_median
    rematerialized_speedup = rematerialized_baseline / rematerialized_candidate
    numerics_passed = _numerics_gate_passed(errors)
    performance_passed = _performance_gate_passed(
        forward_vjp_speedup=forward_vjp_speedup,
        rematerialized_speedup=rematerialized_speedup,
        candidate_seconds=candidate_seconds,
        candidate_forward_seconds=candidate_forward_seconds,
    )
    probe_path = Path(__file__).resolve(strict=True)
    repo = probe_path.parent.parent
    source_path = Path(kernel_module.__file__).resolve(strict=True)
    expected_source_path = (
        repo / "skyrl" / "tx" / "kernels" / "rocm" / "bf16_lora_residual.py"
    ).resolve(strict=True)
    if source_path != expected_source_path:
        raise RuntimeError(
            f"imported kernel source {source_path} is not exact repo source "
            f"{expected_source_path}"
        )
    profiler_path = (repo / "rocm" / "profile_rocm.py").resolve(strict=True)

    return {
        "schema_version": 3,
        "qualification_scope": "isolated_stage_only",
        "authorizes_default_model_enablement": False,
        "passed": bool(numerics_passed and performance_passed and deterministic),
        "preflight": preflight,
        "device": {
            "backend": jax.default_backend(),
            "device_kind": devices[0].device_kind,
            "platform_version": platform_version,
        },
        "geometry": {
            "rows": args.rows,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "rank": args.rank,
            "block_m": args.block_m,
            "block_n": args.block_n,
            "block_k": args.block_k,
            "row_superblock": args.row_superblock,
            "pallas_input_vjp": args.pallas_input_vjp,
        },
        "measurement": {
            "warmups": args.warmups,
            "iterations": args.iterations,
            "forward_and_vjp": {
                "baseline_seconds": baseline_seconds,
                "candidate_seconds": candidate_seconds,
                "baseline_median_seconds": baseline_median,
                "candidate_median_seconds": candidate_median,
                "speedup": forward_vjp_speedup,
            },
            "forward": {
                "baseline_seconds": baseline_forward_seconds,
                "candidate_seconds": candidate_forward_seconds,
                "baseline_median_seconds": baseline_forward_median,
                "candidate_median_seconds": candidate_forward_median,
                "speedup": baseline_forward_median / candidate_forward_median,
            },
            "rematerialized_stage_estimate": {
                "baseline_seconds": rematerialized_baseline,
                "candidate_seconds": rematerialized_candidate,
                "speedup": rematerialized_speedup,
            },
        },
        "performance_gate": {
            "passed": performance_passed,
            "minimum_forward_and_vjp_speedup": _MIN_FORWARD_VJP_SPEEDUP,
            "minimum_rematerialized_stage_speedup": _MIN_REMATERIALIZED_STAGE_SPEEDUP,
            "maximum_candidate_dispatch_seconds": _MAX_DISPATCH_SECONDS,
        },
        "compilation": {
            "baseline_lower_seconds": baseline_lower_seconds,
            "baseline_compile_seconds": baseline_compile_seconds,
            "candidate_lower_seconds": candidate_lower_seconds,
            "candidate_compile_seconds": candidate_compile_seconds,
            "baseline_forward_lower_seconds": baseline_forward_lower_seconds,
            "baseline_forward_compile_seconds": baseline_forward_compile_seconds,
            "candidate_forward_lower_seconds": candidate_forward_lower_seconds,
            "candidate_forward_compile_seconds": candidate_forward_compile_seconds,
        },
        "numerics": {
            "passed": numerics_passed,
            "output_relative_l2_limit": _OUTPUT_RELATIVE_L2_LIMIT,
            "output_cosine_limit": _OUTPUT_COSINE_LIMIT,
            "gradient_relative_l2_limit": _GRADIENT_RELATIVE_L2_LIMIT,
            "gradient_cosine_limit": _GRADIENT_COSINE_LIMIT,
            "maximum_relative_l2": max_relative_l2,
            "errors": errors,
        },
        "determinism": {
            "passed": deterministic,
            "repeated_candidate_outputs_bitwise_equal": deterministic,
        },
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
            f"imported safety source {safety_path} is not exact repo source "
            f"{expected_safety_path}"
        )

    with safety_module.guarded_qwen35_rocm_process() as hardware_preflight:
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
        payload["postflight"] = safety_module.require_clean_amdgpu_boot()
    _private_write(args.output, payload)
    print(json.dumps(payload, allow_nan=False, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
