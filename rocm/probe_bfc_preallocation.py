"""Guarded fixed-BFC preallocation smoke probe for the ROCm capacity experiment.

The default ``abstract`` platform forces JAX to CPU and exercises only the
configuration and JSON-reporting path.  A ROCm allocation requires both
``--platform rocm`` and ``--allow-gpu``.  Even then the probe performs only one
256-byte host-to-device transfer; fixed BFC preallocation is the intended and
dominant device-memory operation.

The allocator environment is configured before JAX is imported.  Conflicting
inherited settings fail closed instead of being silently overridden.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, TextIO

_DEFAULT_FRACTION = Decimal("0.85")
_MIN_FRACTION = Decimal("0.80")
_MAX_FRACTION = Decimal("0.90")
_TINY_ALLOCATION_BYTES = 256
_DISABLE_COMMAND_BUFFERS = "--xla_gpu_enable_command_buffer="


def _acquire_global_lock() -> int:
    """Serialize all guarded Qwen3.5 ROCm processes before KFD preflight."""
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
        raise RuntimeError(
            "another Qwen3.5 ROCm process holds the global launch lock"
        ) from error
    return lock_fd


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _fraction(value: str) -> Decimal:
    try:
        fraction = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("fraction must be a decimal number") from error
    if not fraction.is_finite() or not _MIN_FRACTION <= fraction <= _MAX_FRACTION:
        raise argparse.ArgumentTypeError(
            f"fraction must be finite and in [{_MIN_FRACTION}, {_MAX_FRACTION}]"
        )
    return fraction


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform",
        choices=("abstract", "rocm"),
        default="abstract",
        help="CPU-only guard path by default; ROCm requires --allow-gpu",
    )
    parser.add_argument(
        "--allow-gpu",
        action="store_true",
        help="required acknowledgement when --platform rocm is selected",
    )
    parser.add_argument(
        "--fraction",
        type=_fraction,
        default=_DEFAULT_FRACTION,
        help=f"fixed BFC fraction in [{_MIN_FRACTION}, {_MAX_FRACTION}]",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=5.0,
        help="bounded idle interval after the tiny allocation",
    )
    parser.add_argument("--output", type=Path, help="optional private JSONL output")
    args = parser.parse_args(argv)

    if args.platform == "rocm" and not args.allow_gpu:
        parser.error(
            "--platform rocm requires the explicit --allow-gpu acknowledgement"
        )
    if args.platform == "abstract" and args.allow_gpu:
        parser.error("--allow-gpu is only valid with --platform rocm")
    if not math.isfinite(args.settle_seconds) or not 0 <= args.settle_seconds <= 30:
        parser.error("--settle-seconds must be finite and in [0, 30]")
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    return args


def _validate_exact_or_unset(name: str, expected: str) -> None:
    value = os.environ.get(name)
    if value is not None and value != expected:
        raise RuntimeError(
            f"{name}={value!r} conflicts with required value {expected!r}"
        )


def _validate_true_or_unset(name: str) -> None:
    value = os.environ.get(name)
    if value is not None and value.lower() not in {"1", "true"}:
        raise RuntimeError(
            f"{name}={value!r} conflicts with required fixed preallocation"
        )


def _validate_fraction_or_unset(name: str, expected: Decimal) -> None:
    value = os.environ.get(name)
    if value is None:
        return
    try:
        actual = Decimal(value)
    except InvalidOperation as error:
        raise RuntimeError(
            f"{name}={value!r} is not a valid decimal fraction"
        ) from error
    if not actual.is_finite() or actual != expected:
        raise RuntimeError(
            f"{name}={value!r} conflicts with requested fraction {format(expected, 'f')!r}"
        )


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
            "XLA_PYTHON_CLIENT_ALLOCATOR="
            f"{allocator!r} conflicts with required BFC allocation"
        )
    _validate_true_or_unset("XLA_PYTHON_CLIENT_PREALLOCATE")
    _validate_fraction_or_unset("XLA_CLIENT_MEM_FRACTION", args.fraction)
    if "XLA_PYTHON_CLIENT_MEM_FRACTION" in os.environ:
        raise RuntimeError(
            "XLA_PYTHON_CLIENT_MEM_FRACTION is deprecated and conflicts with "
            "the required XLA_CLIENT_MEM_FRACTION setting"
        )

    if args.platform == "rocm":
        for name in (
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
        ):
            _validate_exact_or_unset(name, "0")
            os.environ[name] = "0"

    original_xla_flags = os.environ.get("XLA_FLAGS", "")
    os.environ["JAX_PLATFORMS"] = requested_platform
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "bfc"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    os.environ["XLA_CLIENT_MEM_FRACTION"] = format(args.fraction, "f")
    os.environ["XLA_FLAGS"] = _force_command_buffers_disabled(original_xla_flags)

    return {
        "JAX_PLATFORMS": os.environ["JAX_PLATFORMS"],
        "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "GPU_DEVICE_ORDINAL": os.environ.get("GPU_DEVICE_ORDINAL"),
        "XLA_PYTHON_CLIENT_ALLOCATOR": os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"],
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"],
        "XLA_CLIENT_MEM_FRACTION": os.environ["XLA_CLIENT_MEM_FRACTION"],
        "XLA_FLAGS_original": original_xla_flags,
        "XLA_FLAGS_effective": os.environ["XLA_FLAGS"],
    }


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
            "refusing ROCm allocation while an AMD display connector is active: "
            f"{connectors}"
        )

    try:
        kfd_stat = stat_fn(kfd_path)
    except OSError as error:
        raise RuntimeError(f"{kfd_path} is missing or inaccessible: {error}") from error
    if not stat.S_ISCHR(kfd_stat.st_mode) or not access_fn(kfd_path, os.R_OK | os.W_OK):
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
            f"refusing ROCm allocation while {kfd_path} is already owned: "
            f"{ownership_output or 'owner reported without a PID'}"
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


def _allocator_snapshot(jax: Any) -> list[dict[str, Any]]:
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
    output.write(_json_dumps(record) + "\n")
    output.flush()


def _validate_backend(
    jax: Any, backend_getter: Any, platform: str
) -> tuple[str, str]:
    """Return the resolved JAX backend and verify the requested ROCm runtime.

    The ROCm PJRT plugin is selected with ``JAX_PLATFORMS=rocm``, but JAX 0.10.2
    deliberately exposes its default backend as ``gpu``.  The backend version,
    rather than the public platform name, distinguishes ROCm from CUDA.
    """
    resolved_backend = jax.default_backend()
    platform_version = str(backend_getter().platform_version)
    expected_backend = "cpu" if platform == "abstract" else "gpu"
    if resolved_backend != expected_backend:
        raise RuntimeError(
            f"requested {platform!r}, expected backend {expected_backend!r}, "
            f"resolved {resolved_backend!r}"
        )
    if platform == "rocm" and "rocm" not in platform_version.lower():
        raise RuntimeError(
            "requested ROCm, but the resolved GPU backend does not identify as "
            f"ROCm: {platform_version!r}"
        )
    return resolved_backend, platform_version


def _run(
    args: argparse.Namespace,
    effective_environment: dict[str, str | None],
    gpu_preflight: dict[str, Any] | None,
    output: TextIO,
) -> int:
    # Environment and safety policy are fixed before importing JAX.
    import jax
    import numpy as np
    from jax.extend import backend as jax_backend

    resolved_backend, platform_version = _validate_backend(
        jax, jax_backend.get_backend, args.platform
    )

    command_buffer_tokens = shlex.split(os.environ["XLA_FLAGS"])
    _emit(
        {
            "record_type": "manifest",
            "timestamp": _utc_now(),
            "platform_requested": args.platform,
            "platform_resolved": resolved_backend,
            "platform_version": platform_version,
            "allow_gpu": args.allow_gpu,
            "fraction": float(args.fraction),
            "fraction_decimal": format(args.fraction, "f"),
            "settle_seconds": args.settle_seconds,
            "tiny_allocation_bytes": _TINY_ALLOCATION_BYTES,
            "command_buffers_disabled": _DISABLE_COMMAND_BUFFERS
            in command_buffer_tokens,
            "environment": effective_environment,
            "gpu_preflight": gpu_preflight,
            "devices": [str(device) for device in jax.devices()],
            "jax_version": jax.__version__,
            "scope": (
                "cpu_configuration_guard"
                if args.platform == "abstract"
                else "allocation_only"
            ),
        },
        output,
    )

    start = time.perf_counter()
    host = np.zeros((_TINY_ALLOCATION_BYTES,), dtype=np.uint8)
    tiny = jax.device_put(host, jax.devices()[0])
    tiny.block_until_ready()
    allocation_seconds = time.perf_counter() - start
    _emit(
        {
            "record_type": "allocated",
            "timestamp": _utc_now(),
            "allocation_seconds": allocation_seconds,
            "array_bytes": tiny.nbytes,
            "array_device": str(tiny.device),
            "allocator": _allocator_snapshot(jax),
        },
        output,
    )

    if args.settle_seconds:
        time.sleep(args.settle_seconds)
    _emit(
        {
            "record_type": "settled",
            "timestamp": _utc_now(),
            "settle_seconds": args.settle_seconds,
            "array_bytes": tiny.nbytes,
            "allocator": _allocator_snapshot(jax),
            "status": "passed",
        },
        output,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    effective_environment = _configure_environment(args)
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
            return _run(args, effective_environment, gpu_preflight, sys.stdout)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            return _run(args, effective_environment, gpu_preflight, output)
    finally:
        if launch_lock is not None:
            os.close(launch_lock)


if __name__ == "__main__":
    sys.exit(main())
