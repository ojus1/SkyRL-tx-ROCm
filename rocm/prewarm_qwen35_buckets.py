#!/usr/bin/env python3
"""Opt-in compile-only prewarm for fixed Qwen3.5 LoRA training buckets.

The default invocation is a CPU-only plan: it parses and reports buckets without
importing JAX or opening an accelerator.  Actual ROCm compilation requires both
``--execute-rocm`` and ``--allow-gpu`` and is intended to be launched by
``start_qwen35.sh`` after that launcher establishes its private, versioned JAX
cache and disables every XLA GPU command buffer.

The execution path constructs the launcher's exact backend and rank-8 LoRA
state, then calls ``lower()`` and ``lowered.compile()`` once per fixed sequence
bucket.  A separate, default-off option lowers and compiles the exact
sequence-independent Adam update once, after every requested training bucket.
It never invokes a compiled callable, runs an optimizer step, exports or
serializes an executable, or uses a graph API.  ROCm compilation is still
allowed to initialize the device, allocate representative buffers, and run XLA
autotuning/profiling kernels; "compile-only" does not mean zero device work.
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
import sys
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, TextIO

try:
    from rocm import probe_sft_compile as compile_probe
    from rocm.prepare_jax_cache_dir import prepare_cache
except ModuleNotFoundError:
    import probe_sft_compile as compile_probe
    from prepare_jax_cache_dir import prepare_cache


_MODEL = "Qwen/Qwen3.5-4B"
_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_MODEL_ID = "startup_prewarm_adapter"
_DEFAULT_BUCKETS = (64, 256)
_MAX_BUCKET = 2_048
_MAX_BUCKET_COUNT = 8
_AUTOTUNE_STARTUP_MAX_BYTES = 4 * 1024**3
_DISABLE_COMMAND_BUFFERS = "--xla_gpu_enable_command_buffer="
_EXPECTED_STACK = {
    "jax": "0.10.2",
    "jaxlib": "0.10.2",
    "jax-rocm7-plugin": "0.10.2",
    "jax-rocm7-pjrt": "0.10.2",
}
_CACHE_ENVIRONMENT = {
    "JAX_ENABLE_COMPILATION_CACHE": "true",
    "JAX_ENABLE_PGLE": "false",
    "JAX_COMPILATION_CACHE_EXPECT_PGLE": "false",
    "JAX_RAISE_PERSISTENT_CACHE_ERRORS": "true",
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": ("xla_gpu_per_fusion_autotune_cache_dir"),
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
    "JAX_COMPILATION_CACHE_MAX_SIZE": "17179869184",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _require_clean_amdgpu_boot() -> dict[str, Any]:
    try:
        from rocm.amdgpu_safety import require_clean_amdgpu_boot
    except ModuleNotFoundError:
        from amdgpu_safety import require_clean_amdgpu_boot

    return require_clean_amdgpu_boot()


def _emit(record: dict[str, Any], output: TextIO) -> None:
    output.write(
        json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    output.flush()


def _emit_backend_ready(
    record: dict[str, Any], hardware: dict[str, Any], output: TextIO
) -> None:
    _emit(
        {
            **record,
            "record_type": "backend_ready",
            "hardware_preflight": hardware,
        },
        output,
    )


def _parse_buckets(value: str) -> tuple[int, ...]:
    pieces = value.split(",")
    if not pieces or any(not piece.strip() for piece in pieces):
        raise argparse.ArgumentTypeError(
            "buckets must be a comma-separated integer list"
        )
    try:
        buckets = tuple(int(piece.strip()) for piece in pieces)
    except ValueError as error:
        raise argparse.ArgumentTypeError("every bucket must be an integer") from error
    if len(buckets) > _MAX_BUCKET_COUNT:
        raise argparse.ArgumentTypeError(
            f"at most {_MAX_BUCKET_COUNT} representative buckets are allowed"
        )
    if tuple(sorted(set(buckets))) != buckets:
        raise argparse.ArgumentTypeError(
            "buckets must be unique and strictly increasing"
        )
    for bucket in buckets:
        if not 32 <= bucket <= _MAX_BUCKET:
            raise argparse.ArgumentTypeError(f"buckets must be in [32, {_MAX_BUCKET}]")
        if compile_probe._round_up_seq_len(bucket) != bucket:
            raise argparse.ArgumentTypeError(
                f"{bucket} is not a canonical SkyRL static sequence bucket"
            )
    return buckets


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--buckets",
        type=_parse_buckets,
        default=_DEFAULT_BUCKETS,
        help="strictly increasing canonical buckets (default CPU plan: 64,256)",
    )
    parser.add_argument(
        "--execute-rocm",
        action="store_true",
        help="perform ROCm backend setup and compile-only cache population",
    )
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help="second acknowledgement required with --execute-rocm",
    )
    parser.add_argument(
        "--compile-optimizer",
        action="store_true",
        help=(
            "after all selected train buckets, lower().compile() the exact "
            "sequence-independent Adam update once without invoking it"
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="already-cached pinned Qwen3.5 revision (required for execution)",
    )
    parser.add_argument(
        "--construction",
        choices=("eager", "abstract-load"),
        default="eager",
        help="must match the server backend construction route",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("xla", "pallas"),
        default="xla",
        help="must match SKYRL_ROCM_PALLAS_ATTENTION",
    )
    parser.add_argument(
        "--launcher-lock-fd",
        type=int,
        help="inherited start_qwen35.sh global-lock descriptor",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="exclusive private JSONL audit artifact (required for execution)",
    )
    args = parser.parse_args(argv)

    if args.execute_rocm != args.allow_gpu:
        parser.error("ROCm compilation requires both --execute-rocm and --allow-gpu")
    if not args.execute_rocm and any(
        value is not None
        for value in (args.model_path, args.launcher_lock_fd, args.output)
    ):
        parser.error(
            "model, lock, and output options are only valid with --execute-rocm"
        )
    if args.execute_rocm and args.model_path is None:
        parser.error("--execute-rocm requires --model-path")
    if args.execute_rocm and args.output is None:
        parser.error("--execute-rocm requires --output")
    if args.output is not None and not args.output.is_absolute():
        parser.error("--output must be absolute")
    if args.attention_backend == "xla" and any(
        bucket >= 512 for bucket in args.buckets
    ):
        parser.error(
            "ROCm buckets >=512 require --attention-backend pallas; refusing the "
            "quadratic XLA fallback"
        )
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    return args


def _validate_stack_and_model(model_path: Path) -> Path:
    observed = {package: metadata.version(package) for package in _EXPECTED_STACK}
    if observed != _EXPECTED_STACK:
        raise RuntimeError(
            f"installed JAX stack {observed!r} does not match {_EXPECTED_STACK!r}"
        )
    if Path("/opt/rocm/.info/version").read_text(encoding="utf-8").strip() != "7.2.4":
        raise RuntimeError("ROCm must be exactly 7.2.4")
    if (
        Path("/sys/module/amdgpu/version").read_text(encoding="utf-8").strip()
        != "6.16.13"
    ):
        raise RuntimeError("AMDGPU must be exactly 6.16.13")

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        expected = Path(
            snapshot_download(
                _MODEL,
                revision=_MODEL_REVISION,
                allow_patterns=("*.safetensors", "*.json", "*.txt", "*.jinja"),
                local_files_only=True,
            )
        ).resolve(strict=True)
    except LocalEntryNotFoundError as error:
        raise RuntimeError(
            f"pinned {_MODEL} revision {_MODEL_REVISION} is not fully cached"
        ) from error
    resolved = model_path.resolve(strict=True)
    if (
        not resolved.is_dir()
        or resolved.name != _MODEL_REVISION
        or resolved != expected
    ):
        raise RuntimeError(
            "model path does not equal huggingface_hub's local snapshot for pinned "
            f"{_MODEL} revision {_MODEL_REVISION}"
        )
    return resolved


def _validate_sysfs_amd_gpu(
    drm_root: Path = Path("/sys/class/drm"),
) -> dict[str, Any]:
    amd_devices: list[dict[str, str]] = []
    for card in sorted(drm_root.glob("card[0-9]*")):
        if not card.name.removeprefix("card").isdigit():
            continue
        try:
            vendor = (card / "device" / "vendor").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if vendor != "0x1002":
            continue
        try:
            device_id = (card / "device" / "device").read_text(encoding="utf-8").strip()
            pci_device = (card / "device").resolve(strict=True).name
        except OSError as error:
            raise RuntimeError(f"cannot identify AMD GPU at {card}: {error}") from error
        amd_devices.append(
            {
                "drm_card": card.name,
                "vendor_id": vendor,
                "device_id": device_id,
                "pci_device": pci_device,
            }
        )
    if len(amd_devices) != 1 or amd_devices[0]["device_id"] != "0x744c":
        raise RuntimeError(
            "startup prewarm requires exactly one AMD DRM GPU with PCI device "
            f"0x744c; observed {amd_devices!r}"
        )
    return {"sysfs_amd_gpu_count": 1, "sysfs_amd_gpu": amd_devices[0]}


def _validate_visible_jax_device(jax: Any) -> dict[str, Any]:
    devices = list(jax.devices())
    if len(devices) != 1:
        raise RuntimeError(
            f"startup prewarm requires exactly one visible JAX device, got {devices!r}"
        )
    device = devices[0]
    platform = str(getattr(device, "platform", ""))
    if platform != "gpu":
        raise RuntimeError(f"expected JAX GPU device, got platform {platform!r}")
    raw_kind = getattr(device, "device_kind", None)
    device_kind = None if raw_kind is None else str(raw_kind).strip()
    if device_kind:
        normalized = device_kind.lower().replace(" ", "")
        if "gfx1100" not in normalized and "rx7900xtx" not in normalized:
            raise RuntimeError(
                "visible JAX device kind does not identify gfx1100/RX 7900 XTX: "
                f"{device_kind!r}"
            )
        kind_validation = "matched_gfx1100_or_rx7900xtx"
    else:
        kind_validation = "not_exposed_sysfs_0x744c_binding_only"
    return {
        "jax_visible_device_count": 1,
        "jax_visible_device": str(device),
        "jax_visible_device_platform": platform,
        "jax_visible_device_kind": device_kind,
        "jax_visible_device_kind_validation": kind_validation,
    }


def _validate_environment(attention_backend: str, construction: str) -> Path:
    required = {
        "JAX_PLATFORMS": "rocm",
        "ROCR_VISIBLE_DEVICES": "0",
        "SKYRL_ROCM_PALLAS_ATTENTION": ("1" if attention_backend == "pallas" else "0"),
        **_CACHE_ENVIRONMENT,
    }
    if construction == "abstract-load":
        required.update(
            {
                "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
                "XLA_CLIENT_MEM_FRACTION": "0.85",
                "HIP_VISIBLE_DEVICES": "0",
                "GPU_DEVICE_ORDINAL": "0",
            }
        )
    else:
        required["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    mismatches = {
        name: {"expected": expected, "observed": os.environ.get(name)}
        for name, expected in required.items()
        if os.environ.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"launcher environment mismatch: {mismatches!r}")
    try:
        xla_tokens = shlex.split(os.environ.get("XLA_FLAGS", ""))
    except ValueError as error:
        raise RuntimeError(f"invalid XLA_FLAGS quoting: {error}") from error
    if xla_tokens != [_DISABLE_COMMAND_BUFFERS]:
        raise RuntimeError(
            "XLA_FLAGS must contain only the exact empty command-buffer disable"
        )

    expected_cache = prepare_cache(_AUTOTUNE_STARTUP_MAX_BYTES)
    configured_cache = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if configured_cache != str(expected_cache):
        raise RuntimeError(
            "JAX_COMPILATION_CACHE_DIR is not the validated private stack namespace"
        )
    return expected_cache


def _revalidate_cache_after_compile(cache_path: Path) -> None:
    validated_cache = prepare_cache(_AUTOTUNE_STARTUP_MAX_BYTES)
    if validated_cache != cache_path:
        raise RuntimeError("private cache namespace changed during startup prewarm")


def _executable_cache_snapshot(cache_path: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    with os.scandir(cache_path) as entries:
        for entry in entries:
            if not entry.name.endswith("-cache"):
                continue
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise RuntimeError(
                    f"non-regular executable cache entry appeared: {entry.path}"
                )
            metadata = entry.stat(follow_symlinks=False)
            snapshot[entry.name] = (metadata.st_size, metadata.st_mtime_ns)
    return snapshot


def _cache_evidence(
    events: dict[str, int],
    durations: dict[str, list[float]],
    monitoring_issues: list[str],
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    added = sorted(after.keys() - before.keys())
    removed = sorted(before.keys() - after.keys())
    changed = sorted(
        name for name in before.keys() & after.keys() if before[name] != after[name]
    )
    manifest = "\n".join(
        f"{name}\0{size}\0{mtime_ns}"
        for name, (size, mtime_ns) in sorted(after.items())
    )
    hits = events.get("/jax/compilation_cache/cache_hits", 0)
    misses = events.get("/jax/compilation_cache/cache_misses", 0)
    requests = events.get("/jax/compilation_cache/compile_requests_use_cache", 0)
    saved = durations.get("/jax/compilation_cache/compile_time_saved_sec", [])
    retrieval = durations.get("/jax/compilation_cache/cache_retrieval_time_sec", [])
    has_exact_hit_durations = len(saved) == 1 and len(retrieval) == 1
    has_any_durations = bool(saved or retrieval)
    if monitoring_issues:
        classification = "malformed_public_monitoring_evidence"
    elif removed:
        classification = "ambiguous_top_level_cache_removal"
    elif requests == 1 and hits == 1 and misses == 0 and (added or changed):
        classification = "ambiguous_hit_with_top_level_cache_mutation"
    elif requests == 1 and hits == 1 and misses == 0 and has_exact_hit_durations:
        classification = "strict_public_monitoring_hit"
    elif requests == 1 and hits == 1 and misses == 0:
        classification = "hit_without_exact_public_duration_evidence"
    elif has_any_durations:
        classification = "duration_events_without_single_cache_hit"
    elif (
        requests == 1 and hits == 0 and misses == 1 and len(added) == 1 and not changed
    ):
        classification = "strict_public_monitoring_miss_with_single_cache_add"
    elif requests == 1 and hits == 0 and misses == 1 and len(added) > 1:
        classification = "ambiguous_miss_with_multiple_added_cache_entries"
    elif requests == 1 and hits == 0 and misses == 1 and changed:
        classification = "ambiguous_miss_with_changed_cache_entry"
    elif misses > 0 and (added or changed):
        classification = "non_strict_miss_events_with_cache_change"
    elif misses > 0:
        classification = "miss_event_without_top_level_cache_change"
    elif hits or misses or requests:
        classification = "mixed_or_incomplete_public_monitoring_events"
    else:
        classification = "no_public_hit_or_miss_event_observed"
    return {
        "classification": classification,
        "public_monitoring_events": {
            "compile_requests_use_cache": requests,
            "cache_hits": hits,
            "cache_misses": misses,
        },
        "public_monitoring_duration_events": {
            "compile_time_saved_sec": saved,
            "cache_retrieval_time_sec": retrieval,
        },
        "public_monitoring_schema_issues": monitoring_issues,
        "top_level_executable_cache": {
            "entries_before": len(before),
            "entries_after": len(after),
            "bytes_before": sum(size for size, _mtime in before.values()),
            "bytes_after": sum(size for size, _mtime in after.values()),
            "added_entries": added,
            "changed_entries": changed,
            "removed_entries": removed,
            "post_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        },
        "evidence_limit": (
            "JAX 0.10.2 public monitoring reports process-level hit/miss and "
            "duration events without a module or cache key; directory deltas are "
            "observable but do not prove which key a compile request used"
        ),
    }


def _run_compile_attempt(
    *,
    target: str,
    identity: dict[str, Any],
    record_prefix: str,
    cache_path: Path,
    jax: Any,
    compile_fn: Callable[[], tuple[Any, float, float]],
    boot_validator: Callable[[], dict[str, Any]],
    output: TextIO,
) -> tuple[Any, float, float, dict[str, Any]]:
    before = _executable_cache_snapshot(cache_path)
    events: dict[str, int] = {}
    durations: dict[str, list[float]] = {}
    monitoring_issues: list[str] = []
    event_names = {
        "/jax/compilation_cache/cache_hits",
        "/jax/compilation_cache/cache_misses",
        "/jax/compilation_cache/compile_requests_use_cache",
    }
    duration_names = {
        "/jax/compilation_cache/compile_time_saved_sec",
        "/jax/compilation_cache/cache_retrieval_time_sec",
    }

    def listener(event: str, **metadata: str | int) -> None:
        if event not in event_names:
            return
        if metadata:
            monitoring_issues.append(f"unexpected event metadata for {event}")
        events[event] = events.get(event, 0) + 1

    def duration_listener(
        event: str, duration_secs: float, **metadata: str | int
    ) -> None:
        if event not in duration_names:
            return
        if metadata:
            monitoring_issues.append(f"unexpected duration metadata for {event}")
        if (
            isinstance(duration_secs, bool)
            or not isinstance(duration_secs, (int, float))
            or not math.isfinite(duration_secs)
            or duration_secs < 0
        ):
            monitoring_issues.append(f"invalid numeric duration for {event}")
            return
        durations.setdefault(event, []).append(float(duration_secs))

    result: tuple[Any, float, float] | None = None
    compile_error: BaseException | None = None
    listener_registered = False
    duration_listener_registered = False
    try:
        jax.monitoring.register_event_listener(listener)
        listener_registered = True
        jax.monitoring.register_event_duration_secs_listener(duration_listener)
        duration_listener_registered = True
        result = compile_fn()
    except BaseException as error:
        compile_error = error
    finally:
        if duration_listener_registered:
            try:
                jax.monitoring.unregister_event_duration_listener(duration_listener)
            except BaseException as error:
                if compile_error is None:
                    compile_error = error
        if listener_registered:
            try:
                jax.monitoring.unregister_event_listener(listener)
            except BaseException as error:
                if compile_error is None:
                    compile_error = error

    cache_error: BaseException | None = None
    after: dict[str, tuple[int, int]] = {}
    try:
        _revalidate_cache_after_compile(cache_path)
        after = _executable_cache_snapshot(cache_path)
    except BaseException as error:
        cache_error = error

    try:
        postflight = boot_validator()
        if (
            postflight.get("amdgpu_boot_clean") is not True
            or postflight.get("fatal_amdgpu_events") != []
        ):
            raise RuntimeError(
                f"{target} AMDGPU postflight returned invalid manifest {postflight!r}"
            )
    except BaseException as error:
        raise RuntimeError(
            f"{target} AMDGPU postflight failed: {type(error).__name__}: {error}"
        ) from error

    _emit(
        {
            "record_type": f"{record_prefix}_postflight",
            "timestamp": _utc_now(),
            "compile_target": target,
            **identity,
            "status": "clean",
            "compile_succeeded": compile_error is None,
            "cache_revalidated": cache_error is None,
            **postflight,
            "model_pass_executable_invocations": 0,
            "optimizer_step_invocations": 0,
        },
        output,
    )
    if cache_error is not None:
        raise RuntimeError(
            f"{target} trusted cache revalidation failed: "
            f"{type(cache_error).__name__}: {cache_error}"
        ) from cache_error
    if compile_error is not None:
        raise RuntimeError(
            f"{target} compile attempt failed after clean postflight: "
            f"{type(compile_error).__name__}: {compile_error}"
        ) from compile_error
    if result is None:
        raise RuntimeError(f"{target} compile attempt returned no result")
    evidence = _cache_evidence(events, durations, monitoring_issues, before, after)
    accepted = evidence["classification"] in {
        "strict_public_monitoring_hit",
        "strict_public_monitoring_miss_with_single_cache_add",
    }
    _emit(
        {
            "record_type": f"{record_prefix}_cache_evidence",
            "timestamp": _utc_now(),
            "compile_target": target,
            **identity,
            "status": "accepted" if accepted else "rejected",
            "evidence": evidence,
            "model_pass_executable_invocations": 0,
            "optimizer_step_invocations": 0,
        },
        output,
    )
    if not accepted:
        raise RuntimeError(
            f"{target} persistent-cache evidence is not promotable: "
            f"{evidence['classification']}"
        )
    return (*result, evidence)


def _run_bucket_compile_attempt(
    *,
    bucket: int,
    cache_path: Path,
    jax: Any,
    compile_fn: Callable[[], tuple[Any, float, float]],
    boot_validator: Callable[[], dict[str, Any]],
    output: TextIO,
) -> tuple[Any, float, float, dict[str, Any]]:
    return _run_compile_attempt(
        target="train_bucket_forward_backward_accumulate",
        identity={"bucket": bucket},
        record_prefix="bucket",
        cache_path=cache_path,
        jax=jax,
        compile_fn=compile_fn,
        boot_validator=boot_validator,
        output=output,
    )


def _run_optimizer_compile_attempt(
    *,
    cache_path: Path,
    jax: Any,
    compile_fn: Callable[[], tuple[Any, float, float]],
    boot_validator: Callable[[], dict[str, Any]],
    output: TextIO,
) -> tuple[Any, float, float, dict[str, Any]]:
    return _run_compile_attempt(
        target="sequence_independent_compute_grads_and_update",
        identity={},
        record_prefix="optimizer",
        cache_path=cache_path,
        jax=jax,
        compile_fn=compile_fn,
        boot_validator=boot_validator,
        output=output,
    )


def _validate_inherited_lock(lock_fd: int) -> int:
    if lock_fd < 0:
        raise RuntimeError("launcher lock descriptor must be nonnegative")
    lock_parent = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
    lock_dir = lock_parent / f"skyrl-qwen35-rocm-{os.getuid()}"
    path_metadata = lock_dir.lstat()
    descriptor_metadata = os.fstat(lock_fd)
    if (
        not stat.S_ISDIR(path_metadata.st_mode)
        or stat.S_ISLNK(path_metadata.st_mode)
        or path_metadata.st_uid != os.getuid()
        or stat.S_IMODE(path_metadata.st_mode) != 0o700
        or not stat.S_ISDIR(descriptor_metadata.st_mode)
        or (path_metadata.st_dev, path_metadata.st_ino)
        != (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
    ):
        raise RuntimeError("inherited descriptor is not the private global launch lock")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(
            "another Qwen3.5 ROCm process owns the global lock"
        ) from error
    return lock_fd


def _open_private_output(path: Path) -> int:
    if not path.is_absolute():
        raise RuntimeError("ROCm audit output must be an absolute path")
    parent = path.parent
    if not parent.exists():
        raise RuntimeError("ROCm audit output parent must already exist")
    if parent != Path(os.path.realpath(parent)):
        raise RuntimeError("ROCm audit output parent must not contain a symlink")
    parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_fd = os.open(parent, parent_flags)
    try:
        parent_metadata = os.fstat(parent_fd)
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or parent_metadata.st_uid != os.getuid()
            or stat.S_IMODE(parent_metadata.st_mode) != 0o700
        ):
            raise RuntimeError(
                "ROCm audit output parent must be a real mode-0700 directory "
                "owned by the current user"
            )
        descriptor = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
        ):
            os.close(descriptor)
            os.unlink(path.name, dir_fd=parent_fd)
            raise RuntimeError("ROCm audit output is not a private regular file")
        return descriptor
    finally:
        os.close(parent_fd)


def _shape_signature(jax: Any, jnp: Any, backend: Any, bucket: int) -> tuple[Any, ...]:
    from skyrl.tinker.loss_fns import LossFnConfig

    batch_2d = jax.NamedSharding(backend.mesh, jax.P("fsdp", None))
    batch_1d = jax.NamedSharding(backend.mesh, jax.P("fsdp"))

    def shape(dimensions: tuple[int, ...], dtype: Any, sharding: Any) -> Any:
        return jax.ShapeDtypeStruct(dimensions, dtype, sharding=sharding)

    return (
        shape((1, bucket), jnp.int32, batch_2d),
        shape((1, bucket), jnp.int32, batch_2d),
        shape((1,), jnp.int32, batch_1d),
        shape((1, bucket), jnp.int32, batch_2d),
        shape((1, bucket), jnp.float32, batch_2d),
        shape((1,), jnp.int32, batch_1d),
        shape((1, bucket), jnp.float32, batch_2d),
        shape((1, bucket), jnp.float32, batch_2d),
        LossFnConfig(
            clip_low_threshold=shape((1,), jnp.float32, batch_1d),
            clip_high_threshold=shape((1,), jnp.float32, batch_1d),
        ),
    )


def _lower_and_compile(
    jax: Any,
    backend: Any,
    jitted_function: Any,
    arguments: tuple[Any, ...],
) -> tuple[Any, float, float]:
    lower_start = time.perf_counter()
    mesh_context = (
        jax.set_mesh(backend.mesh) if hasattr(jax, "set_mesh") else nullcontext()
    )
    with mesh_context:
        lowered = jitted_function.lower(*arguments)
    lower_seconds = time.perf_counter() - lower_start
    compile_start = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - compile_start
    return compiled, lower_seconds, compile_seconds


def _lower_and_compile_bucket(
    jax: Any,
    model_pass: Any,
    backend: Any,
    signature: tuple[Any, ...],
) -> tuple[Any, float, float]:
    return _lower_and_compile(
        jax,
        backend,
        model_pass,
        (
            backend.accumulated_grads,
            backend.lora_params,
            backend.non_lora_params,
            *signature,
        ),
    )


def _lower_and_compile_optimizer(
    jax: Any,
    optimizer_pass: Any,
    backend: Any,
    optimizer: Any,
    adapter_index: Any,
) -> tuple[Any, float, float]:
    return _lower_and_compile(
        jax,
        backend,
        optimizer_pass,
        (
            backend.accumulated_grads,
            backend.lora_params,
            optimizer,
            adapter_index,
        ),
    )


def _run_rocm(
    args: argparse.Namespace,
    model_path: Path,
    cache_path: Path,
    hardware: dict[str, Any],
    output: TextIO,
    boot_validator: Callable[[], dict[str, Any]],
) -> int:
    # Imports that can initialize JAX stay behind all opt-ins and preflight checks.
    import jax
    import jax.numpy as jnp
    import jaxlib
    from flax import nnx
    from jax.extend import backend as jax_backend

    from skyrl.backends.jax import JaxBackendConfig, JaxBackendImpl
    from skyrl.tinker import types

    resolved_backend, platform_version = compile_probe._validate_rocm_backend(
        jax, jax_backend.get_backend
    )
    hardware = {**hardware, **_validate_visible_jax_device(jax)}
    config_values = {
        "max_lora_adapters": 2,
        "max_lora_rank": 8,
        "train_micro_batch_size": 1,
        "sample_max_num_sequences": 1,
        "gradient_checkpointing": True,
        "loss_chunk_size": 64,
        "abstract_model_load": args.construction == "abstract-load",
    }
    if "abstract_model_load" not in JaxBackendConfig.model_fields:
        raise RuntimeError("SkyRL backend lacks the required abstract_model_load field")

    setup_start = time.perf_counter()
    backend = JaxBackendImpl(
        str(model_path), JaxBackendConfig(**config_values), process_id=0
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
            backend.accumulated_grads,
            backend.lora_params,
            backend.non_lora_params,
            optimizer_state,
        )
    )
    setup_seconds = time.perf_counter() - setup_start
    _emit_backend_ready(
        {
            "timestamp": _utc_now(),
            "model": _MODEL,
            "model_revision": _MODEL_REVISION,
            "platform_resolved": resolved_backend,
            "platform_version": platform_version,
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "cache_path": str(cache_path),
            "backend_config": config_values,
            "adapter_index": adapter_index,
            "setup_seconds": setup_seconds,
            "setup_dispatch_caveat": (
                "backend/model construction, pinned-weight loading, LoRA parameter "
                "and Adam-state initialization, array placement, and explicit "
                "block_until_ready synchronization may perform ordinary setup array "
                "work; no training pass or optimizer update executable ran"
            ),
            "optimizer_compile_requested": args.compile_optimizer,
            "train_bucket_lower_calls": 0,
            "train_bucket_compile_calls": 0,
            "optimizer_lower_calls": 0,
            "optimizer_compile_calls": 0,
            "model_pass_executable_invocations": 0,
            "optimizer_step_invocations": 0,
        },
        hardware,
        output,
    )

    model_pass = backend._forward_backward_and_accumulate
    if not hasattr(model_pass, "lower"):
        raise RuntimeError("backend model pass does not expose lower()")
    for bucket in args.buckets:
        signature = _shape_signature(jax, jnp, backend, bucket)
        compiled, lower_seconds, compile_seconds, cache_evidence = (
            _run_bucket_compile_attempt(
                bucket=bucket,
                cache_path=cache_path,
                jax=jax,
                compile_fn=lambda signature=signature: _lower_and_compile_bucket(
                    jax, model_pass, backend, signature
                ),
                boot_validator=boot_validator,
                output=output,
            )
        )
        memory = compile_probe._compiled_memory(compiled)
        del compiled
        _emit(
            {
                "record_type": "bucket_compiled",
                "timestamp": _utc_now(),
                "compile_target": "train_bucket_forward_backward_accumulate",
                "bucket": bucket,
                "batch_size": 1,
                "attention_backend": args.attention_backend,
                "lower_seconds": lower_seconds,
                "compile_seconds": compile_seconds,
                "compiled_memory": memory,
                "persistent_cache_evidence": cache_evidence,
                "train_bucket_lower_calls": 1,
                "train_bucket_compile_calls": 1,
                "optimizer_lower_calls": 0,
                "optimizer_compile_calls": 0,
                "model_pass_executable_invocations": 0,
                "optimizer_step_invocations": 0,
                "status": "passed",
            },
            output,
        )

    if args.compile_optimizer:
        optimizer_pass = backend._compute_grads_and_update
        if not hasattr(optimizer_pass, "lower"):
            raise RuntimeError("backend Adam update does not expose lower()")
        optimizer = backend.optimizers[_MODEL_ID]
        optimizer_adapter_index = jnp.int32(adapter_index)
        compiled, lower_seconds, compile_seconds, cache_evidence = (
            _run_optimizer_compile_attempt(
                cache_path=cache_path,
                jax=jax,
                compile_fn=lambda: _lower_and_compile_optimizer(
                    jax,
                    optimizer_pass,
                    backend,
                    optimizer,
                    optimizer_adapter_index,
                ),
                boot_validator=boot_validator,
                output=output,
            )
        )
        memory = compile_probe._compiled_memory(compiled)
        del compiled
        _emit(
            {
                "record_type": "optimizer_compiled",
                "timestamp": _utc_now(),
                "compile_target": "sequence_independent_compute_grads_and_update",
                "optimizer": "Adam",
                "sequence_independent": True,
                "lower_seconds": lower_seconds,
                "compile_seconds": compile_seconds,
                "compiled_memory": memory,
                "persistent_cache_evidence": cache_evidence,
                "train_bucket_lower_calls": 0,
                "train_bucket_compile_calls": 0,
                "optimizer_lower_calls": 1,
                "optimizer_compile_calls": 1,
                "model_pass_executable_invocations": 0,
                "optimizer_step_invocations": 0,
                "optimizer_state_mutations_through_step": 0,
                "status": "passed",
            },
            output,
        )
    return 0


def _execute(args: argparse.Namespace, output: TextIO) -> int:
    _emit(
        {
            "record_type": "manifest",
            "timestamp": _utc_now(),
            "mode": "rocm_compile_only" if args.execute_rocm else "cpu_plan_only",
            "buckets": list(args.buckets),
            "batch_size": 1,
            "attention_backend": args.attention_backend,
            "optimizer_compile_requested": args.compile_optimizer,
            "train_bucket_lower_calls_planned": len(args.buckets),
            "train_bucket_compile_calls_planned": len(args.buckets),
            "optimizer_lower_calls_planned": 1 if args.compile_optimizer else 0,
            "optimizer_compile_calls_planned": 1 if args.compile_optimizer else 0,
            "command_buffers_required_disabled": _DISABLE_COMMAND_BUFFERS,
            "compiled_callable_invocations": 0,
            "optimizer_step_invocations": 0,
            "executable_export_used": False,
            "graph_api_used": False,
        },
        output,
    )
    if not args.execute_rocm:
        _emit(
            {
                "record_type": "plan",
                "timestamp": _utc_now(),
                "status": "cpu_plan_only",
                "jax_imported": False,
                "gpu_accessed": False,
                "optimizer_compile_planned": args.compile_optimizer,
                "note": (
                    "CPU compilation cannot populate the ROCm executable cache; "
                    "execution requires both explicit ROCm acknowledgements"
                ),
            },
            output,
        )
        return 0

    lock_fd = None
    stage = "static_validation"
    failure_stage = stage
    primary_error: BaseException | None = None
    try:
        try:
            model_path = _validate_stack_and_model(args.model_path)
            stage = "global_lock"
            lock_fd = (
                compile_probe._acquire_global_lock()
                if args.launcher_lock_fd is None
                else _validate_inherited_lock(args.launcher_lock_fd)
            )
            stage = "hardware_preflight"
            hardware = {
                **_require_clean_amdgpu_boot(),
                **_validate_sysfs_amd_gpu(),
                **compile_probe._hardware_preflight(),
            }
            stage = "trusted_cache_validation"
            cache_path = _validate_environment(
                args.attention_backend, args.construction
            )
            stage = "rocm_compile_only"
            if (
                _run_rocm(
                    args,
                    model_path,
                    cache_path,
                    hardware,
                    output,
                    _require_clean_amdgpu_boot,
                )
                != 0
            ):
                raise RuntimeError(
                    "compile-only bucket stage returned a nonzero status"
                )
        except BaseException as error:
            failure_stage = stage
            primary_error = error

        postflight_error: BaseException | None = None
        postflight: dict[str, Any] | None = None
        try:
            postflight = _require_clean_amdgpu_boot()
            if (
                postflight.get("amdgpu_boot_clean") is not True
                or postflight.get("fatal_amdgpu_events") != []
            ):
                raise RuntimeError(
                    "AMDGPU postflight returned an invalid clean manifest: "
                    f"{postflight!r}"
                )
        except BaseException as error:
            postflight_error = error

        if postflight_error is not None:
            primary_context = (
                ""
                if primary_error is None
                else (
                    f"; primary {failure_stage} failure was "
                    f"{type(primary_error).__name__}: {primary_error}"
                )
            )
            _emit(
                {
                    "record_type": "error",
                    "timestamp": _utc_now(),
                    "stage": "hardware_postflight",
                    "error_type": type(postflight_error).__name__,
                    "message": f"{postflight_error}{primary_context}",
                    "compiled_callable_invocations": 0,
                    "optimizer_step_invocations": 0,
                    "status": "failed",
                },
                output,
            )
            return 1

        assert postflight is not None
        _emit(
            {
                "record_type": "hardware_postflight",
                "timestamp": _utc_now(),
                "status": "clean",
                "operation_succeeded": primary_error is None,
                **postflight,
                "model_pass_executable_invocations": 0,
                "optimizer_step_invocations": 0,
            },
            output,
        )
        if primary_error is not None:
            _emit(
                {
                    "record_type": "error",
                    "timestamp": _utc_now(),
                    "stage": failure_stage,
                    "error_type": type(primary_error).__name__,
                    "message": str(primary_error),
                    "compiled_callable_invocations": 0,
                    "optimizer_step_invocations": 0,
                    "status": "failed",
                },
                output,
            )
            return 1

        _emit(
            {
                "record_type": "complete",
                "timestamp": _utc_now(),
                "buckets": list(args.buckets),
                "optimizer_compiled": args.compile_optimizer,
                "train_bucket_lower_calls": len(args.buckets),
                "train_bucket_compile_calls": len(args.buckets),
                "optimizer_lower_calls": 1 if args.compile_optimizer else 0,
                "optimizer_compile_calls": 1 if args.compile_optimizer else 0,
                "cache_revalidated_after_each_compile": True,
                "amdgpu_postflight_clean": True,
                "status": "passed",
                "model_pass_executable_invocations": 0,
                "optimizer_step_invocations": 0,
            },
            output,
        )
        return 0
    finally:
        if lock_fd is not None:
            os.close(lock_fd)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.output is None:
        return _execute(args, sys.stdout)
    try:
        descriptor = _open_private_output(args.output)
    except (OSError, RuntimeError) as error:
        print(f"cannot create private ROCm audit output: {error}", file=sys.stderr)
        return 2
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        return _execute(args, output)


if __name__ == "__main__":
    raise SystemExit(main())
