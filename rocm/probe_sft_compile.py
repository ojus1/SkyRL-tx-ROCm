#!/usr/bin/env python3
"""Guarded compile-only probe for exact Qwen3.5-4B LoRA SFT.

The default path is a CPU-forced refusal record.  A real ROCm backend load and
compile requires ``--platform rocm --allow-gpu``.  The ROCm path constructs the
normal ``JaxBackendImpl`` state, creates the rank-8 adapter (including its Adam
state), lowers the exact batch-one forward/backward/accumulate JIT, and calls
``compile()``.  It never calls the compiled model-pass executable.

"Compile-only" is deliberately narrow: checkpoint device transfers, BFC
preallocation, LoRA initialization, and optimizer/gradient-state setup are
normal backend setup and may dispatch small accelerator operations.  No SFT
forward, backward, accumulation, or optimizer step is executed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

_MODEL = "Qwen/Qwen3.5-4B"
_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_MODEL_ID = "compile_probe_adapter"
_DISABLE_COMMAND_BUFFERS = "--xla_gpu_enable_command_buffer="
_MEMORY_FRACTION = "0.85"
_MAX_CONTEXT = 32_768
_DEFAULT_CONTEXT = 2_048
_PALLAS_REQUIRED_CONTEXT = 512
_PALLAS_MAX_CONTEXT = 16_384


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _emit(record: dict[str, Any], output: TextIO) -> None:
    output.write(_json_dumps(record) + "\n")
    output.flush()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=("abstract", "rocm"),
        default="abstract",
        help="CPU-only refusal by default; real lowering requires rocm and --allow-gpu",
    )
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help="required acknowledgement for ROCm model loading and compilation",
    )
    parser.add_argument("--context", type=int, default=_DEFAULT_CONTEXT)
    parser.add_argument(
        "--allow-large-context",
        action="store_true",
        help="required above context 2048 because compilation can consume substantial RAM/VRAM",
    )
    parser.add_argument(
        "--construction",
        choices=("abstract-load", "eager"),
        default="abstract-load",
        help="JaxBackendImpl model construction route used before lowering",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("xla", "pallas"),
        default="xla",
        help="explicit attention route; effective ROCm contexts >=512 require pallas",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow fetching the pinned model revision when it is absent locally",
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
    if args.platform == "abstract" and args.allow_download:
        parser.error("--allow-download is only valid with --platform rocm")
    if args.platform == "abstract" and args.attention_backend != "xla":
        parser.error("--attention-backend pallas is only valid with --platform rocm")
    if args.platform == "rocm" and args.output is None:
        parser.error("--platform rocm requires --output for a clean JSONL artifact")
    if args.context <= 0 or args.context > _MAX_CONTEXT:
        parser.error(f"--context must be in [1, {_MAX_CONTEXT}]")
    if args.context > _DEFAULT_CONTEXT and not args.allow_large_context:
        parser.error("contexts above 2048 require --allow-large-context")
    effective_context = _round_up_seq_len(args.context)
    if args.platform == "rocm":
        if effective_context >= _PALLAS_REQUIRED_CONTEXT and args.attention_backend != "pallas":
            parser.error(
                "ROCm effective contexts >=512 require --attention-backend pallas; "
                "the quadratic XLA fallback is refused"
            )
        if args.attention_backend == "pallas" and not (
            _PALLAS_REQUIRED_CONTEXT <= effective_context <= _PALLAS_MAX_CONTEXT
            and effective_context % 64 == 0
        ):
            parser.error(
                "--attention-backend pallas requires a 64-aligned effective context "
                "in [512, 16384]"
            )
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    return args


def _round_up_seq_len(seq_len: int) -> int:
    """Mirror the backend's static sequence-bucketing rule without importing JAX."""
    if seq_len <= 32:
        return 32
    msb_pos = seq_len.bit_length() - 1
    mask = (1 << msb_pos) | (1 << (msb_pos - 1))
    result = seq_len & mask
    if result < seq_len:
        result += 1 << (msb_pos - 1)
    return result


def _validate_exact_or_unset(name: str, expected: str) -> None:
    value = os.environ.get(name)
    if value is not None and value != expected:
        raise RuntimeError(f"{name}={value!r} conflicts with required value {expected!r}")


def _validate_true_or_unset(name: str) -> None:
    value = os.environ.get(name)
    if value is not None and value.lower() not in {"1", "true"}:
        raise RuntimeError(f"{name}={value!r} conflicts with required fixed preallocation")


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


def _configure_environment(args: argparse.Namespace) -> dict[str, str | None]:
    requested_platform = "cpu" if args.platform == "abstract" else "rocm"
    _validate_exact_or_unset("JAX_PLATFORMS", requested_platform)

    allocator = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR")
    if allocator is not None and allocator.lower() != "bfc":
        raise RuntimeError(
            f"XLA_PYTHON_CLIENT_ALLOCATOR={allocator!r} conflicts with required BFC allocation"
        )
    _validate_true_or_unset("XLA_PYTHON_CLIENT_PREALLOCATE")
    _validate_exact_or_unset("XLA_CLIENT_MEM_FRACTION", _MEMORY_FRACTION)
    if "XLA_PYTHON_CLIENT_MEM_FRACTION" in os.environ:
        raise RuntimeError(
            "XLA_PYTHON_CLIENT_MEM_FRACTION is deprecated and conflicts with "
            "the fixed XLA_CLIENT_MEM_FRACTION=0.85 contract"
        )

    if args.platform == "rocm":
        for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
            _validate_exact_or_unset(name, "0")
            os.environ[name] = "0"
        pallas_value = "1" if args.attention_backend == "pallas" else "0"
        _validate_exact_or_unset("SKYRL_ROCM_PALLAS_ATTENTION", pallas_value)
        os.environ["SKYRL_ROCM_PALLAS_ATTENTION"] = pallas_value

    original_xla_flags = os.environ.get("XLA_FLAGS", "")
    os.environ["JAX_PLATFORMS"] = requested_platform
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "bfc"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    os.environ["XLA_CLIENT_MEM_FRACTION"] = _MEMORY_FRACTION
    os.environ["XLA_FLAGS"] = _force_command_buffers_disabled(original_xla_flags)

    return {
        "JAX_PLATFORMS": os.environ["JAX_PLATFORMS"],
        "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "GPU_DEVICE_ORDINAL": os.environ.get("GPU_DEVICE_ORDINAL"),
        "SKYRL_ROCM_PALLAS_ATTENTION": os.environ.get(
            "SKYRL_ROCM_PALLAS_ATTENTION"
        ),
        "XLA_PYTHON_CLIENT_ALLOCATOR": os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"],
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"],
        "XLA_CLIENT_MEM_FRACTION": os.environ["XLA_CLIENT_MEM_FRACTION"],
        "XLA_FLAGS_original": original_xla_flags,
        "XLA_FLAGS_effective": os.environ["XLA_FLAGS"],
    }


def _acquire_global_lock() -> int:
    lock_parent = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
    lock_dir = lock_parent / f"skyrl-qwen35-rocm-{os.getuid()}"
    if lock_dir.is_symlink():
        raise RuntimeError(f"refusing symlinked launch-lock directory: {lock_dir}")
    try:
        lock_dir.mkdir(mode=0o700)
    except FileExistsError:
        pass
    info = lock_dir.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError(f"refusing unsafe launch-lock directory: {lock_dir}")
    lock_dir.chmod(0o700)
    lock_fd = os.open(lock_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(lock_fd)
        raise RuntimeError("another Qwen3.5 ROCm process holds the global launch lock") from error
    return lock_fd


def _hardware_preflight(
    drm_root: Path = Path("/sys/class/drm"),
    kfd: Path = Path("/dev/kfd"),
    *,
    stat_fn: Any = os.stat,
    access_fn: Any = os.access,
    run_fn: Any = subprocess.run,
) -> dict[str, Any]:
    """Require a headless and exclusively owned AMD compute device."""
    try:
        kfd_stat = stat_fn(kfd)
    except OSError as error:
        raise RuntimeError(f"{kfd} is missing or inaccessible: {error}") from error
    if not stat.S_ISCHR(kfd_stat.st_mode) or not access_fn(kfd, os.R_OK | os.W_OK):
        raise RuntimeError(f"{kfd} must be an accessible character device")

    amd_cards = 0
    connected: list[str] = []
    for card_path in sorted(drm_root.glob("card[0-9]*")):
        if not card_path.name.removeprefix("card").isdigit():
            continue
        vendor_path = card_path / "device" / "vendor"
        try:
            vendor = vendor_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if vendor != "0x1002":
            continue
        amd_cards += 1
        for status_path in sorted(card_path.parent.glob(f"{card_path.name}-*/status")):
            if "Writeback" in str(status_path.parent):
                continue
            try:
                status_text = status_path.read_text(encoding="utf-8").strip()
            except OSError as error:
                raise RuntimeError(
                    f"cannot verify AMD connector state at {status_path}: {error}"
                ) from error
            if status_text == "connected":
                connected.append(str(status_path.parent))
    if amd_cards == 0:
        raise RuntimeError("no AMD DRM card was found")
    if connected:
        raise RuntimeError(
            "physical AMD display connector(s) are active; move the display to the iGPU: "
            + ", ".join(connected)
        )

    try:
        result = run_fn(
            ["fuser", str(kfd)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError("fuser is required for the /dev/kfd ownership check") from error
    ownership_output = " ".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    if result.returncode == 0:
        raise RuntimeError(
            f"{kfd} is already owned: "
            f"{ownership_output or 'owner reported without a PID'}"
        )
    if result.returncode != 1 or ownership_output:
        raise RuntimeError(
            f"could not verify exclusive {kfd} ownership with fuser: "
            f"{ownership_output or f'return code {result.returncode}'}"
        )
    return {
        "amd_card_count": amd_cards,
        "connected_amd_connectors": connected,
        "kfd_path": str(kfd),
        "kfd_accessible": True,
        "kfd_unowned": True,
    }


def _validate_rocm_backend(jax: Any, backend_getter: Any) -> tuple[str, str]:
    """Verify the public GPU backend is backed by the ROCm PJRT plugin."""
    resolved_backend = jax.default_backend()
    platform_version = str(backend_getter().platform_version)
    if resolved_backend != "gpu":
        raise RuntimeError(
            f"requested ROCm, expected public backend 'gpu', got {resolved_backend!r}"
        )
    if "rocm" not in platform_version.lower():
        raise RuntimeError(
            "requested ROCm, but the resolved GPU backend does not identify as "
            f"ROCm: {platform_version!r}"
        )
    return resolved_backend, platform_version


def _json_scalar(value: Any) -> bool | int | float | str | None:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):
        try:
            result = value.item()
        except (TypeError, ValueError):
            return str(value)
        if isinstance(result, float) and not math.isfinite(result):
            return str(result)
        if isinstance(result, (bool, int, float, str)) or result is None:
            return result
    return str(value)


def _allocator_snapshot(jax: Any) -> list[dict[str, Any]]:
    snapshots = []
    for device in jax.devices():
        raw = device.memory_stats()
        stats = None
        if raw is not None:
            stats = {str(key): _json_scalar(value) for key, value in sorted(raw.items())}
        snapshots.append({"device": str(device), "memory_stats": stats})
    return snapshots


def _tree_summary(jax: Any, tree: Any) -> dict[str, Any]:
    by_dtype: dict[str, int] = {}
    total = 0
    count = 0
    for leaf in jax.tree.leaves(tree):
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            continue
        byte_count = math.prod(leaf.shape) * leaf.dtype.itemsize
        dtype = str(leaf.dtype)
        by_dtype[dtype] = by_dtype.get(dtype, 0) + byte_count
        total += byte_count
        count += 1
    return {"array_leaves": count, "logical_bytes": total, "bytes_by_dtype": by_dtype}


def _ir_summary(text: str, dialect: str) -> dict[str, Any]:
    if dialect == "stablehlo":
        counts = {
            name: text.count(f"stablehlo.{name}")
            for name in (
                "custom_call",
                "dot_general",
                "dynamic_slice",
                "gather",
                "reduce",
                "reduce_window",
                "scatter",
                "while",
            )
        }
        counts["optimization_barrier"] = text.count("optimization_barrier")
    else:
        counts = {
            "custom_call": text.count("custom-call("),
            "dot": text.count(" dot("),
            "fusion": text.count(" fusion("),
            "while": text.count(" while("),
            "entry_computation": text.count("ENTRY "),
        }
    return {
        "dialect": dialect,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "characters": len(text),
        "lines": len(text.splitlines()),
        "operation_counts": counts,
    }


def _compiled_memory(compiled: Any) -> dict[str, Any]:
    try:
        stats = compiled.memory_analysis()
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    if stats is None:
        return {"available": False}
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_temp_size_in_bytes",
        "generated_code_size_in_bytes",
    )
    result = {
        name: int(getattr(stats, name))
        for name in names
        if hasattr(stats, name) and getattr(stats, name) is not None
    }
    return {"available": True, **result}


def _cost_analysis(compiled: Any) -> dict[str, Any]:
    try:
        raw = compiled.cost_analysis()
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    if raw is None:
        return {"available": False}
    if isinstance(raw, list):
        raw = raw[0] if len(raw) == 1 else {f"module_{i}": value for i, value in enumerate(raw)}
    if not isinstance(raw, dict):
        return {"available": True, "value": str(raw)}
    return {
        "available": True,
        "metrics": {str(key): _json_scalar(value) for key, value in sorted(raw.items())},
    }


def _run_rocm(
    args: argparse.Namespace, hardware: dict[str, Any], output: TextIO
) -> int:
    # All imports that can initialize JAX occur after the environment and
    # hardware guards above.
    import jax
    import jax.numpy as jnp
    import jaxlib
    from flax import nnx
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from jax.extend import backend as jax_backend

    from skyrl.backends.jax import JaxBackendConfig, JaxBackendImpl
    from skyrl.tinker import types
    from skyrl.tinker.loss_fns import LossFnConfig

    resolved_backend, platform_version = _validate_rocm_backend(
        jax, jax_backend.get_backend
    )

    try:
        checkpoint_path = snapshot_download(
            _MODEL,
            revision=_MODEL_REVISION,
            allow_patterns=("*.safetensors", "*.json", "*.txt", "*.jinja"),
            local_files_only=not args.allow_download,
        )
    except LocalEntryNotFoundError as error:
        raise RuntimeError(
            "the pinned model revision is not fully cached; rerun with --allow-download"
        ) from error
    resolved_revision = Path(checkpoint_path).resolve().name
    if resolved_revision != _MODEL_REVISION:
        raise RuntimeError(
            f"resolved model revision {resolved_revision!r}, expected {_MODEL_REVISION!r}"
        )

    config_kwargs = {
        "max_lora_adapters": 2,
        "max_lora_rank": 8,
        "train_micro_batch_size": 1,
        "sample_max_num_sequences": 1,
        "gradient_checkpointing": True,
        "loss_chunk_size": 64,
        "abstract_model_load": args.construction == "abstract-load",
    }
    if "abstract_model_load" not in JaxBackendConfig.model_fields:
        raise RuntimeError(
            "this SkyRL revision lacks JaxBackendConfig.abstract_model_load; "
            "the requested direct construction route is unavailable"
        )

    setup_start = time.perf_counter()
    backend = JaxBackendImpl(
        checkpoint_path,
        JaxBackendConfig(**config_kwargs),
        process_id=0,
    )
    backend.create_model(
        _MODEL_ID,
        types.LoraConfig(rank=8, alpha=32.0, seed=0),
    )
    adapter_index = backend.models[_MODEL_ID].adapter_index
    if adapter_index != 1:
        raise RuntimeError(f"expected active adapter index 1, got {adapter_index}")
    optimizer_state = nnx.state(backend.optimizers[_MODEL_ID])
    jax.block_until_ready(
        (
            backend.lora_params,
            backend.non_lora_params,
            backend.accumulated_grads,
            optimizer_state,
        )
    )
    setup_seconds = time.perf_counter() - setup_start

    effective_context = _round_up_seq_len(args.context)
    from skyrl.tx.utils.models import round_up_seq_len

    backend_effective_context = round_up_seq_len(args.context)
    if effective_context != backend_effective_context:
        raise RuntimeError(
            f"local context bucket {effective_context} != backend bucket {backend_effective_context}"
        )

    batch_2d = jax.NamedSharding(backend.mesh, jax.P("fsdp", None))
    batch_1d = jax.NamedSharding(backend.mesh, jax.P("fsdp"))

    def shape(name: str, dimensions: tuple[int, ...], dtype: Any, sharding: Any):
        return name, jax.ShapeDtypeStruct(dimensions, dtype, sharding=sharding)

    named_inputs = (
        shape("input_ids", (1, effective_context), jnp.int32, batch_2d),
        shape("attention_mask", (1, effective_context), jnp.int32, batch_2d),
        shape("adapter_indices", (1,), jnp.int32, batch_1d),
        shape("target_ids", (1, effective_context), jnp.int32, batch_2d),
        shape("loss_mask", (1, effective_context), jnp.float32, batch_2d),
        shape("loss_fn_types", (1,), jnp.int32, batch_1d),
        shape("sampling_logprobs", (1, effective_context), jnp.float32, batch_2d),
        shape("advantages", (1, effective_context), jnp.float32, batch_2d),
    )
    loss_fn_config = LossFnConfig(
        clip_low_threshold=jax.ShapeDtypeStruct((1,), jnp.float32, sharding=batch_1d),
        clip_high_threshold=jax.ShapeDtypeStruct((1,), jnp.float32, sharding=batch_1d),
    )
    input_signature = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sharding": str(value.sharding),
        }
        for name, value in named_inputs
    ]
    input_signature.extend(
        {
            "name": f"loss_fn_config.{name}",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sharding": str(value.sharding),
        }
        for name, value in (
            ("clip_low_threshold", loss_fn_config.clip_low_threshold),
            ("clip_high_threshold", loss_fn_config.clip_high_threshold),
        )
    )

    _emit(
        {
            "record_type": "backend_ready",
            "timestamp": _utc_now(),
            "stage": "backend_setup_complete",
            "platform_resolved": resolved_backend,
            "platform_version": platform_version,
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "devices": [str(device) for device in jax.devices()],
            "hardware_preflight": hardware,
            "checkpoint_path": checkpoint_path,
            "resolved_revision": resolved_revision,
            "construction": args.construction,
            "attention_backend": args.attention_backend,
            "backend_config": config_kwargs,
            "active_adapter": {
                "index": adapter_index,
                "rank": 8,
                "alpha": 32.0,
                "seed": 0,
            },
            "setup_seconds": setup_seconds,
            "setup_dispatch_caveat": (
                "checkpoint device_put, fixed BFC preallocation, LoRA initialization, "
                "and optimizer/gradient setup may dispatch; no model pass ran"
            ),
            "logical_state": {
                "non_lora_params": _tree_summary(jax, backend.non_lora_params),
                "lora_params": _tree_summary(jax, backend.lora_params),
                "accumulated_grads": _tree_summary(jax, backend.accumulated_grads),
                "optimizer": _tree_summary(jax, optimizer_state),
            },
            "allocator": _allocator_snapshot(jax),
            "model_pass_executable_invocations": 0,
        },
        output,
    )

    model_pass = backend._forward_backward_and_accumulate
    if not hasattr(model_pass, "lower"):
        raise RuntimeError(
            "JaxBackendImpl._forward_backward_and_accumulate does not expose lower(); "
            "compile-only probing is unavailable"
        )

    lower_start = time.perf_counter()
    with jax.set_mesh(backend.mesh):
        lowered = model_pass.lower(
            backend.accumulated_grads,
            backend.lora_params,
            backend.non_lora_params,
            *(value for _, value in named_inputs),
            loss_fn_config,
        )
    lower_seconds = time.perf_counter() - lower_start
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    stablehlo_summary = _ir_summary(stablehlo, "stablehlo")
    del stablehlo
    _emit(
        {
            "record_type": "lowered",
            "timestamp": _utc_now(),
            "requested_context": args.context,
            "effective_context": effective_context,
            "batch_size": 1,
            "loss": "cross_entropy (dynamic LOSS_TYPES input has shape [1])",
            "input_signature": input_signature,
            "lower_seconds": lower_seconds,
            "stablehlo": stablehlo_summary,
            "allocator": _allocator_snapshot(jax),
            "model_pass_executable_invocations": 0,
        },
        output,
    )

    compile_start = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - compile_start
    optimized_hlo = compiled.as_text()
    optimized_hlo_summary = _ir_summary(optimized_hlo, "optimized_hlo")
    del optimized_hlo
    report = {
        "record_type": "compiled",
        "timestamp": _utc_now(),
        "requested_context": args.context,
        "effective_context": effective_context,
        "batch_size": 1,
        "compile_seconds": compile_seconds,
        "compiled_memory": _compiled_memory(compiled),
        "cost_analysis": _cost_analysis(compiled),
        "optimized_hlo": optimized_hlo_summary,
        "allocator": _allocator_snapshot(jax),
        "model_pass_executable_invocations": 0,
        "optimizer_step_invocations": 0,
        "status": "passed",
    }
    _emit(report, output)
    return 0


def _execute(
    args: argparse.Namespace,
    effective_environment: dict[str, str | None],
    output: TextIO,
) -> int:
    effective_context = _round_up_seq_len(args.context)
    _emit(
        {
            "record_type": "manifest",
            "timestamp": _utc_now(),
            "platform_requested": args.platform,
            "allow_gpu": args.allow_gpu,
            "model": _MODEL,
            "model_revision": _MODEL_REVISION,
            "requested_context": args.context,
            "effective_context": effective_context,
            "batch_size": 1,
            "construction": args.construction,
            "attention_backend": args.attention_backend,
            "fixed_preallocation_fraction": float(_MEMORY_FRACTION),
            "command_buffers_disabled": _DISABLE_COMMAND_BUFFERS
            in shlex.split(os.environ["XLA_FLAGS"]),
            "environment": effective_environment,
            "scope": (
                "cpu_guard_refusal"
                if args.platform == "abstract"
                else "backend_setup_and_model_pass_compile_only"
            ),
            "normal_api_lifecycle_planned_before_lowering": args.platform == "rocm",
            "compile_dispatch_caveat": (
                "lowered.compile may execute bounded GPU autotuning/profiling kernels "
                "and allocate representative buffers; the compiled model-pass callable "
                "itself is never invoked"
            ),
            "model_pass_executable_invocations": 0,
        },
        output,
    )
    if args.platform == "abstract":
        _emit(
            {
                "record_type": "refused",
                "timestamp": _utc_now(),
                "status": "cpu_guard_only",
                "reason": (
                    "exact Qwen3.5 ROCm lowering is not attempted by default; "
                    "pass --platform rocm --allow-gpu explicitly"
                ),
                "jax_imported": False,
                "model_pass_executable_invocations": 0,
            },
            output,
        )
        return 0

    launch_lock = None
    stage = "hardware_preflight"
    try:
        launch_lock = _acquire_global_lock()
        hardware = _hardware_preflight()
        stage = "rocm_backend"
        return _run_rocm(args, hardware, output)
    except Exception as error:
        _emit(
            {
                "record_type": "error",
                "timestamp": _utc_now(),
                "stage": stage,
                "error_type": type(error).__name__,
                "message": str(error),
                "model_pass_executable_invocations": 0,
                "status": "failed",
            },
            output,
        )
        return 1
    finally:
        if launch_lock is not None:
            os.close(launch_lock)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    effective_environment = _configure_environment(args)
    if args.output is None:
        return _execute(args, effective_environment, sys.stdout)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        return _execute(args, effective_environment, output)


if __name__ == "__main__":
    sys.exit(main())
