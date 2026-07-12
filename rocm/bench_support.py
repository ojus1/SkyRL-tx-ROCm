"""Shared, dependency-light support for standalone ROCm learner benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import tinker

MODEL = "Qwen/Qwen3.5-4B"
REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
_SENSITIVE_WORDS = (
    "TOKEN",
    "KEY",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "AUTH",
    "COOKIE",
)
_SENSITIVE_ARGUMENT_FLAGS = (
    "--header",
    "--proxy-header",
    "--user",
    "--proxy-user",
    "-u",
)


def json_dumps(value: Any, **kwargs) -> str:
    return json.dumps(value, allow_nan=False, **kwargs)


def redact_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        if ":" in hostname:
            hostname = f"[{hostname}]"
        if parsed.port is not None:
            hostname = f"{hostname}:{parsed.port}"
        return urlunsplit((parsed.scheme, hostname, "", "", ""))
    except ValueError:
        return "<redacted-url>"


def redacted_argv(values: list[str]) -> list[str]:
    result: list[str] = []
    redact_next = False
    for value in values:
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        lowered = value.lower()
        sensitive_flag = next(
            (
                flag
                for flag in _SENSITIVE_ARGUMENT_FLAGS
                if lowered == flag or lowered.startswith(f"{flag}=")
            ),
            None,
        )
        if sensitive_flag is not None:
            if "=" in value:
                result.append(f"{value.split('=', 1)[0]}=<redacted>")
            else:
                result.append(value)
                redact_next = True
            continue
        if value.startswith(("-H", "-u", "-U")) and len(value) > 2:
            result.append(f"{value[:2]}<redacted>")
            continue
        key: str | None = None
        candidate = value
        equals_index = value.find("=")
        scheme_index = value.find("://")
        if equals_index >= 0 and (scheme_index < 0 or equals_index < scheme_index):
            key, candidate = value.split("=", 1)
        if key is not None and any(word in key.upper() for word in _SENSITIVE_WORDS):
            result.append(f"{key}=<redacted>")
        elif "://" in candidate:
            sanitized = redact_url(candidate)
            result.append(f"{key}={sanitized}" if key is not None else sanitized)
        else:
            result.append(value)
        if value == "-H":
            redact_next = True
        elif value.startswith("-") and any(
            word in value.upper() for word in _SENSITIVE_WORDS
        ):
            redact_next = "=" not in value
    return result


def round_up_seq_len(seq_len: int) -> int:
    """Mirror ``skyrl.tx.utils.models.round_up_seq_len`` without importing JAX."""
    if seq_len <= 32:
        return 32
    msb_pos = seq_len.bit_length() - 1
    mask = (1 << msb_pos) | (1 << (msb_pos - 1))
    rounded = seq_len & mask
    if rounded < seq_len:
        rounded += 1 << (msb_pos - 1)
    return rounded


def _run_text(*args: str, cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            args, cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_metadata(path: Path) -> dict[str, Any]:
    commit = _run_text("git", "rev-parse", "HEAD", cwd=path)
    status = _run_text("git", "status", "--porcelain=v1", cwd=path)
    diff = _run_text("git", "diff", "--binary", "HEAD", cwd=path)
    return {
        "path": str(path),
        "commit": commit,
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
        "tracked_diff_sha256": hashlib.sha256((diff or "").encode()).hexdigest(),
    }


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def model_revision(model: str) -> str | None:
    default_hf_home = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface"
    )
    hf_home = Path(os.environ.get("HF_HOME", default_hf_home))
    hub_cache = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub"))
    reference = hub_cache / f"models--{model.replace('/', '--')}" / "refs/main"
    try:
        return reference.read_text().strip()
    except (FileNotFoundError, PermissionError):
        return None


def safe_accelerator_environment() -> dict[str, str]:
    prefixes = ("JAX_", "XLA_", "HSA_", "HIP_", "ROCM_", "ROCR_", "PJRT_", "AMD_")
    exact_names = {"GPU_DEVICE_ORDINAL"}
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if (key.startswith(prefixes) or key in exact_names)
        and not any(word in key.upper() for word in _SENSITIVE_WORDS)
    }


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump())
    return str(value)


async def unload_adapter(service_client: tinker.ServiceClient, training_client) -> Any:
    """Unload an adapter on the SDK-owned loop rather than borrowing its client."""
    from tinker.lib.api_future_impl import _APIFuture
    from tinker.lib.client_connection_pool_type import ClientConnectionPoolType

    model_id = training_client._guaranteed_model_id()
    request_start = time.time()

    async def unload_on_sdk_loop() -> Any:
        with service_client.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            future = await client.models.unload(
                request=tinker.types.UnloadModelRequest(model_id=model_id)
            )
        return await _APIFuture(
            tinker.types.UnloadModelResponse,
            service_client.holder,
            future,
            request_start_time=request_start,
            request_type="UnloadModel",
        ).result_async()

    return await service_client.holder.run_coroutine_threadsafe(unload_on_sdk_loop())
