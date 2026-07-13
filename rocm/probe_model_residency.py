"""Measure logical and live-buffer residency of the exact Qwen3.5-4B model state.

The normal SkyRL loader first initializes the model and then replaces its base
weights from safetensors.  This probe separates those stages so allocator cache
from genuinely live duplicate arrays can be distinguished.

Safety policy:

* Abstract CPU shape evaluation is the default and never loads model weights.
* ROCm execution requires both ``--platform rocm`` and ``--allow-gpu``.
* Allocator growth is the default; ``preallocate85`` requires explicit BFC/0.85 settings.
* GPU command buffers are disabled before JAX is imported.
* The probe performs initialization/loading only: it does not compile or run a
  model forward/backward pass.

The ``backend_ready`` record retains the module plus its split NNX states to
measure aliasing.  It does not include accumulated gradients or per-adapter
optimizer state, so it is not a complete training-backend residency total.
Likewise, ``abstract-load`` changes construction only; compare unique live
buffers and allocator-pool statistics before interpreting it as a saving.

The abstract default writes JSON Lines to stdout.  ROCm requires an exclusive
mode-0600 ``--output`` artifact.  Wrap every GPU invocation with
``rocm/profile_rocm.py`` to record physical VRAM and safety telemetry.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO

_MODEL = "Qwen/Qwen3.5-4B"
_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_DISABLE_COMMAND_BUFFERS = "--xla_gpu_enable_command_buffer="


def _acquire_global_lock(runtime_dir: Path | None = None) -> int:
    """Serialize all guarded Qwen3.5 ROCm processes before KFD preflight."""
    try:
        from rocm.amdgpu_safety import acquire_qwen35_rocm_launch_lock
    except ModuleNotFoundError as error:  # Direct execution from the rocm directory.
        if error.name != "rocm":
            raise
        from amdgpu_safety import acquire_qwen35_rocm_launch_lock

    return acquire_qwen35_rocm_launch_lock(runtime_dir=runtime_dir)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=("abstract", "rocm"), default="abstract")
    parser.add_argument(
        "--construction",
        choices=("eager", "abstract-load"),
        default="eager",
        help="ROCm model-construction route; ignored by abstract shape evaluation",
    )
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help="required acknowledgement when --platform rocm is selected",
    )
    parser.add_argument(
        "--allocator-mode",
        choices=("growth", "preallocate85"),
        default="growth",
        help=(
            "ROCm allocator policy: grow on demand, or require the default BFC "
            "allocator with a fixed 0.85 preallocation"
        ),
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=5.0,
        help="bounded idle interval before the final live-buffer snapshot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="exclusive mode-0600 JSONL artifact (required for ROCm)",
    )
    args = parser.parse_args(argv)
    if args.platform == "rocm" and not args.allow_gpu:
        parser.error("--platform rocm requires --allow-gpu")
    if args.platform == "abstract" and args.allow_gpu:
        parser.error("--allow-gpu is only valid with --platform rocm")
    if args.platform == "abstract" and args.construction != "eager":
        parser.error("--construction is only valid with --platform rocm")
    if args.platform == "abstract" and args.allocator_mode != "growth":
        parser.error("--allocator-mode is only valid with --platform rocm")
    if not math.isfinite(args.settle_seconds) or not 0 <= args.settle_seconds <= 30:
        parser.error("--settle-seconds must be finite and in [0, 30]")
    if args.platform == "rocm" and args.output is None:
        parser.error("--platform rocm requires --output for a clean JSONL artifact")
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    return args


def _force_xla_flag(original: str, flag: str) -> str:
    """Put one XLA flag last, removing conflicting spellings of that flag."""
    flag_name = flag.partition("=")[0]
    parts = [
        part
        for part in original.split()
        if part != flag_name and not part.startswith(f"{flag_name}=")
    ]
    return " ".join((*parts, flag))


def _configure_environment(args: argparse.Namespace) -> None:
    os.environ["JAX_PLATFORMS"] = "cpu" if args.platform == "abstract" else "rocm"
    if args.platform == "rocm":
        for name in (
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
        ):
            visible_devices = os.environ.get(name)
            if visible_devices not in (None, "0"):
                raise RuntimeError(
                    f"the residency probe requires {name}=0; got {visible_devices!r}"
                )
            os.environ[name] = "0"
        allocator_mode = getattr(args, "allocator_mode", "growth")
        preallocate = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
        memory_mode = os.environ.get("SKYRL_QWEN35_MEMORY_MODE")
        if "XLA_PYTHON_CLIENT_MEM_FRACTION" in os.environ:
            raise RuntimeError(
                "XLA_PYTHON_CLIENT_MEM_FRACTION is deprecated; use "
                "XLA_CLIENT_MEM_FRACTION"
            )
        if allocator_mode == "growth":
            if memory_mode == "preallocate85" or (
                preallocate is not None and preallocate.lower() in {"1", "true", "yes", "on"}
            ):
                raise RuntimeError(
                    "external fixed preallocation conflicts with --allocator-mode growth; "
                    "select --allocator-mode preallocate85 explicitly"
                )
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        elif allocator_mode == "preallocate85":
            allocator = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR")
            fraction = os.environ.get("XLA_CLIENT_MEM_FRACTION")
            if allocator not in (None, "bfc"):
                raise RuntimeError(
                    "--allocator-mode preallocate85 requires the BFC allocator; "
                    f"got XLA_PYTHON_CLIENT_ALLOCATOR={allocator!r}"
                )
            if memory_mode not in (None, "preallocate85"):
                raise RuntimeError(
                    "--allocator-mode preallocate85 conflicts with "
                    f"SKYRL_QWEN35_MEMORY_MODE={memory_mode!r}"
                )
            if preallocate is not None and preallocate.lower() not in {"1", "true", "yes", "on"}:
                raise RuntimeError(
                    "--allocator-mode preallocate85 conflicts with "
                    f"XLA_PYTHON_CLIENT_PREALLOCATE={preallocate!r}"
                )
            if fraction is not None:
                try:
                    exact_fraction = float(fraction)
                except ValueError as exc:
                    raise RuntimeError(
                        "XLA_CLIENT_MEM_FRACTION must be exactly 0.85 for "
                        "--allocator-mode preallocate85"
                    ) from exc
                if exact_fraction != 0.85:
                    raise RuntimeError(
                        "XLA_CLIENT_MEM_FRACTION must be exactly 0.85 for "
                        f"--allocator-mode preallocate85; got {fraction!r}"
                    )
            os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "bfc"
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
            os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.85"
        else:
            raise RuntimeError(f"unsupported allocator mode {allocator_mode!r}")
        os.environ["XLA_FLAGS"] = _force_xla_flag(
            os.environ.get("XLA_FLAGS", ""), _DISABLE_COMMAND_BUFFERS
        )


def _gpu_preflight(
    drm_root: Path = Path("/sys/class/drm"),
    kfd_path: Path = Path("/dev/kfd"),
    *,
    stat_fn: Any = os.stat,
    access_fn: Any = os.access,
    which_fn: Any = shutil.which,
    run_fn: Any = subprocess.run,
) -> dict[str, Any]:
    """Require a headless, exclusively owned AMD compute device."""
    amd_cards: list[str] = []
    connected_connectors: list[str] = []
    for card_path in sorted(drm_root.glob("card[0-9]*")):
        if re.fullmatch(r"card[0-9]+", card_path.name) is None:
            continue
        try:
            vendor = (card_path / "device" / "vendor").read_text().strip()
        except OSError:
            continue
        if vendor != "0x1002":
            continue
        amd_cards.append(card_path.name)
        for status_path in sorted(drm_root.glob(f"{card_path.name}-*/status")):
            if "Writeback" in status_path.parent.name:
                continue
            try:
                status = status_path.read_text().strip()
            except OSError as error:
                raise RuntimeError(
                    f"cannot verify AMD connector state at {status_path}: {error}"
                ) from error
            if status == "connected":
                connected_connectors.append(status_path.parent.name)

    if not amd_cards:
        raise RuntimeError("no AMD DRM card was found")
    if connected_connectors:
        connectors = ", ".join(connected_connectors)
        raise RuntimeError(
            "refusing ROCm residency measurement while an AMD display connector "
            f"is active: {connectors}"
        )

    try:
        kfd_stat = stat_fn(kfd_path)
    except OSError as error:
        raise RuntimeError(f"{kfd_path} is missing or inaccessible: {error}") from error
    if not stat.S_ISCHR(kfd_stat.st_mode) or not access_fn(
        kfd_path, os.R_OK | os.W_OK
    ):
        raise RuntimeError(f"{kfd_path} must be an accessible character device")

    fuser = which_fn("fuser")
    if fuser is None:
        raise RuntimeError("fuser is required to verify exclusive /dev/kfd ownership")
    ownership = run_fn(
        [fuser, str(kfd_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    ownership_output = " ".join(
        part.strip() for part in (ownership.stdout, ownership.stderr) if part.strip()
    )
    if ownership.returncode == 0:
        raise RuntimeError(
            f"refusing ROCm residency measurement while {kfd_path} is already "
            f"owned: {ownership_output or 'owner reported without a PID'}"
        )
    if ownership.returncode != 1 or ownership_output:
        raise RuntimeError(
            f"could not verify exclusive {kfd_path} ownership with fuser: "
            f"{ownership_output or f'return code {ownership.returncode}'}"
        )

    return {
        "amd_cards": amd_cards,
        "connected_amd_connectors": connected_connectors,
        "kfd_path": str(kfd_path),
        "kfd_accessible": True,
        "kfd_unowned": True,
    }


def _path_string(path: tuple[Any, ...]) -> str:
    return ".".join(str(getattr(key, "key", getattr(key, "name", key))) for key in path)


def _state_snapshot(nnx, model) -> dict[str, Any]:
    by_dtype: defaultdict[str, int] = defaultdict(int)
    by_kind: defaultdict[str, int] = defaultdict(int)
    leaves = 0
    total = 0
    for path, variable in nnx.to_flat_state(nnx.state(model)):
        value = variable.get_raw_value()
        elements = math.prod(value.shape)
        byte_count = elements * value.dtype.itemsize
        path_text = _path_string(path)
        if any(
            name in path_text
            for name in ("lora_A", "lora_B", "lora_scaling", "lora_ranks")
        ):
            kind = "lora"
        elif "rng" in path_text:
            kind = "rng"
        else:
            kind = "base"
        leaves += 1
        total += byte_count
        by_dtype[str(value.dtype)] += byte_count
        by_kind[kind] += byte_count
    return {
        "leaf_count": leaves,
        "logical_bytes": total,
        "bytes_by_dtype": dict(sorted(by_dtype.items())),
        "bytes_by_kind": dict(sorted(by_kind.items())),
    }


def _block_model(jax, nnx, model) -> None:
    values = [
        variable.get_raw_value() for _, variable in nnx.to_flat_state(nnx.state(model))
    ]
    jax.block_until_ready(values)


def _live_snapshot(jax) -> dict[str, Any]:
    arrays = jax.live_arrays()
    logical_bytes = sum(array.nbytes for array in arrays)
    unique_buffers: dict[tuple[str, int], int] = {}
    pointer_failures = 0
    for array in arrays:
        try:
            key = (str(array.device), int(array.unsafe_buffer_pointer()))
        except (AttributeError, RuntimeError, ValueError):
            pointer_failures += 1
            continue
        unique_buffers[key] = max(unique_buffers.get(key, 0), array.nbytes)
    return {
        "array_count": len(arrays),
        "logical_bytes_with_aliases": logical_bytes,
        "unique_buffer_count": len(unique_buffers),
        "unique_buffer_bytes_lower_bound": sum(unique_buffers.values()),
        "pointer_failures": pointer_failures,
    }


def _allocator_snapshot(jax) -> list[dict[str, Any]]:
    snapshots = []
    for device in jax.devices():
        raw_stats = device.memory_stats()
        stats = None
        if raw_stats is not None:
            stats = {
                str(key): value
                for key, value in sorted(raw_stats.items())
                if isinstance(value, (bool, int, float, str)) or value is None
            }
        snapshots.append({"device": str(device), "memory_stats": stats})
    return snapshots


def _emit(record: dict[str, Any], output: TextIO) -> None:
    output.write(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    output.flush()


def _run(
    args: argparse.Namespace,
    gpu_preflight: dict[str, Any] | None,
    output: TextIO,
) -> int:
    # Imports below this line may initialize JAX, so environment policy must be
    # fixed first.
    import jax
    from flax import nnx
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    from skyrl.tinker import types
    from skyrl.tx.layers.lora import init_lora_adapter
    from skyrl.tx.models.configs import Qwen3Config
    from skyrl.tx.utils.models import (
        bind_abstract_mesh_shardings,
        get_dtype,
        get_model_class,
        load_qwen3_5_safetensors_abstract,
        load_safetensors,
    )

    checkpoint_path = snapshot_download(
        _MODEL,
        revision=_MODEL_REVISION,
        allow_patterns=("*.safetensors", "*.json", "*.txt", "*.jinja"),
        local_files_only=True,
    )
    revision = Path(checkpoint_path).resolve().name
    if revision != _MODEL_REVISION:
        raise RuntimeError(
            f"resolved model revision {revision!r}, expected {_MODEL_REVISION!r}"
        )

    base_config = AutoConfig.from_pretrained(checkpoint_path, local_files_only=True)
    model_config = Qwen3Config(
        base_config,
        max_lora_adapters=2,
        max_lora_rank=8,
        shard_attention_heads=True,
        loss_chunk_size=64,
        gradient_checkpointing=True,
        mhc_expansion_rate=1,
    )
    model_class = get_model_class(model_config)
    mesh = jax.make_mesh(
        (1, 1, 1),
        ("fsdp", "ep", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * 3,
    )

    _emit(
        {
            "record_type": "manifest",
            "platform": args.platform,
            "construction": args.construction
            if args.platform == "rocm"
            else "abstract",
            "allocator_mode": args.allocator_mode,
            "jax_version": jax.__version__,
            "model": _MODEL,
            "expected_revision": _MODEL_REVISION,
            "resolved_revision": revision,
            "command_buffers_disabled": _DISABLE_COMMAND_BUFFERS
            in os.environ.get("XLA_FLAGS", "").split(),
            "gpu_preflight": gpu_preflight,
            "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
            "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
            "gpu_device_ordinal": os.environ.get("GPU_DEVICE_ORDINAL"),
            "xla_python_client_allocator": os.environ.get(
                "XLA_PYTHON_CLIENT_ALLOCATOR", "default"
            ),
            "xla_client_memory_fraction": os.environ.get(
                "XLA_CLIENT_MEM_FRACTION"
            ),
            "xla_python_client_preallocate": os.environ.get(
                "XLA_PYTHON_CLIENT_PREALLOCATE"
            ),
            "skyrl_qwen35_memory_mode": os.environ.get(
                "SKYRL_QWEN35_MEMORY_MODE"
            ),
        },
        output,
    )

    with jax.set_mesh(mesh), nnx.use_eager_sharding(True):
        if args.platform == "abstract":
            model = nnx.eval_shape(
                lambda: model_class(
                    model_config,
                    dtype=get_dtype(model_config.get_config().dtype),
                    rngs=nnx.Rngs(0),
                )
            )
            _emit(
                {"record_type": "abstract_state", **_state_snapshot(nnx, model)},
                output,
            )
            return 0

        def model_factory():
            return model_class(
                model_config,
                dtype=get_dtype(model_config.get_config().dtype),
                rngs=nnx.Rngs(0),
            )

        if args.construction == "abstract-load":
            model = nnx.eval_shape(model_factory)
            bind_abstract_mesh_shardings(model, mesh)
        else:
            model = model_factory()
            _block_model(jax, nnx, model)
        _emit(
            {
                "record_type": "initialized",
                "state": _state_snapshot(nnx, model),
                "live": _live_snapshot(jax),
                "allocator": _allocator_snapshot(jax),
            },
            output,
        )

        if args.construction == "abstract-load":
            model = load_qwen3_5_safetensors_abstract(
                checkpoint_path,
                model_config,
                model_class,
                dtype=get_dtype(model_config.get_config().dtype),
                mesh=mesh,
            )
        else:
            load_safetensors(checkpoint_path, model_config, model)
        _block_model(jax, nnx, model)
        gc.collect()
        loaded_live = _live_snapshot(jax)
        _emit(
            {
                "record_type": "loaded",
                "state": _state_snapshot(nnx, model),
                "live": loaded_live,
                "allocator": _allocator_snapshot(jax),
            },
            output,
        )

        graphdef, lora_params, non_lora_params = nnx.split(
            model, model.is_lora_param, ...
        )
        init_lora_adapter(
            model,
            adapter_index=0,
            lora_config=types.LoraConfig(rank=1, alpha=1.0, seed=0),
        )
        _block_model(jax, nnx, model)
        gc.collect()
        split_live = _live_snapshot(jax)
        _emit(
            {
                "record_type": "backend_ready",
                "state": _state_snapshot(nnx, model),
                "live": split_live,
                "allocator": _allocator_snapshot(jax),
                "split_objects_retained": bool(
                    graphdef is not None and lora_params and non_lora_params
                ),
                "backend_auxiliary_state_included": False,
                "split_unique_buffer_count_delta": split_live[
                    "unique_buffer_count"
                ]
                - loaded_live["unique_buffer_count"],
                "split_unique_buffer_bytes_lower_bound_delta": split_live[
                    "unique_buffer_bytes_lower_bound"
                ]
                - loaded_live["unique_buffer_bytes_lower_bound"],
            },
            output,
        )

        if args.settle_seconds:
            time.sleep(args.settle_seconds)
        gc.collect()
        _emit(
            {
                "record_type": "settled",
                "live": _live_snapshot(jax),
                "allocator": _allocator_snapshot(jax),
            },
            output,
        )

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_environment(args)
    launch_lock = _acquire_global_lock() if args.platform == "rocm" else None
    try:
        if args.platform == "rocm":
            try:
                from rocm.amdgpu_safety import require_clean_amdgpu_boot
            except ModuleNotFoundError:
                from amdgpu_safety import require_clean_amdgpu_boot

            try:
                boot_preflight = require_clean_amdgpu_boot()
            except RuntimeError as error:
                print(str(error), file=sys.stderr)
                return 2
            gpu_preflight = {**boot_preflight, **_gpu_preflight()}
        else:
            gpu_preflight = None
        if args.output is None:
            return _run(args, gpu_preflight, sys.stdout)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            return _run(args, gpu_preflight, output)
    finally:
        if launch_lock is not None:
            os.close(launch_lock)


if __name__ == "__main__":
    sys.exit(main())
