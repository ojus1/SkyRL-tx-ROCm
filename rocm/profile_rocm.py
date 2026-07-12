"""Low-overhead ROCm, host, and named-process telemetry recorder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import psutil

_SENSITIVE_WORDS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH", "COOKIE")
_SENSITIVE_ARGUMENT_FLAGS = ("--header", "--proxy-header", "--user", "--proxy-user", "-u")
_GIB = 1024**3
try:
    from rocm.amdgpu_safety import AMDGPU_FATAL_PATTERN as _AMDGPU_FATAL_PATTERN
except ModuleNotFoundError:
    from amdgpu_safety import AMDGPU_FATAL_PATTERN as _AMDGPU_FATAL_PATTERN


def _redact_argument(value: str) -> str:
    lowered = value.strip().lower()
    if lowered.startswith(("authorization:", "proxy-authorization:", "cookie:", "bearer ")):
        return "<redacted>"
    key: str | None = None
    candidate = value
    equals_index = value.find("=")
    scheme_index = value.find("://")
    if equals_index >= 0 and (scheme_index < 0 or equals_index < scheme_index):
        key, candidate = value.split("=", 1)
        if any(word in key.upper() for word in _SENSITIVE_WORDS):
            return f"{key}=<redacted>"

    if "://" in candidate:
        try:
            parsed = urlsplit(candidate)
            hostname = parsed.hostname or ""
            if ":" in hostname:
                hostname = f"[{hostname}]"
            if parsed.port is not None:
                hostname = f"{hostname}:{parsed.port}"
            # URL paths routinely contain webhook tokens and object-store keys.
            # Retain only the origin; query, fragment, userinfo, and path are
            # all unnecessary for benchmark provenance.
            candidate = urlunsplit((parsed.scheme, hostname, "", "", ""))
        except ValueError:
            candidate = "<redacted-url>"
    return f"{key}={candidate}" if key is not None else candidate


def _redact_argv(values: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for value in values:
        if redact_next:
            redacted.append("<redacted>")
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
                redacted.append(f"{value.split('=', 1)[0]}=<redacted>")
            else:
                redacted.append(value)
                redact_next = True
            continue
        # curl accepts a short-option argument without intervening whitespace.
        if value.startswith(("-H", "-u", "-U")) and len(value) > 2:
            redacted.append(f"{value[:2]}<redacted>")
            continue
        redacted.append(_redact_argument(value))
        if value == "-H":
            redact_next = True
        elif value.startswith("-") and any(word in value.upper() for word in _SENSITIVE_WORDS):
            redact_next = "=" not in value
    return redacted


def _safe_accelerator_environment(environ: Mapping[str, str]) -> dict[str, str]:
    prefixes = ("JAX_", "XLA_", "HSA_", "HIP_", "ROCM_", "AMD_")
    return {
        key: value
        for key, value in sorted(environ.items())
        if key.startswith(prefixes) and not any(word in key.upper() for word in _SENSITIVE_WORDS)
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _json_dumps(value: Any, **kwargs) -> str:
    return json.dumps(value, allow_nan=False, **kwargs)


def _write_private_text(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(value)


def _read_number(path: Path, scale: float = 1.0) -> float | None:
    try:
        value = float(path.read_text().strip()) / scale
        return value if math.isfinite(value) else None
    except (OSError, ValueError):
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _kernel_driver_errors_since(epoch_seconds: float) -> list[str] | None:
    """Read new fatal AMDGPU kernel messages; return None if journal access fails."""
    try:
        result = subprocess.run(
            [
                "journalctl",
                "--dmesg",
                "--since",
                f"@{epoch_seconds:.6f}",
                "--no-pager",
                "--output=short-iso",
            ],
            check=False,
            capture_output=True,
            text=True,
            # Journal inspection must not stall the thermal/memory sampler.
            timeout=1.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return [line for line in result.stdout.splitlines() if _AMDGPU_FATAL_PATTERN.search(line)][-100:]


def _find_gpu(card_name: str | None) -> tuple[Path, Path | None, dict[str, Any]]:
    if card_name is not None and re.fullmatch(r"card\d+", card_name) is None:
        raise RuntimeError(f"Invalid DRM card name: {card_name!r}")
    candidates = (
        [Path("/sys/class/drm") / card_name]
        if card_name
        else sorted(Path("/sys/class/drm").glob("card*"))
    )
    for card in candidates:
        if not re.fullmatch(r"card\d+", card.name):
            continue
        device = card / "device"
        vendor_id = _read_text(device / "vendor")
        if (
            vendor_id is None
            or vendor_id.lower() != "0x1002"
            or not (device / "gpu_busy_percent").exists()
        ):
            continue
        hwmons = sorted((device / "hwmon").glob("hwmon*"))
        resolved_device = device.resolve()
        identity = {
            "card": card.name,
            "card_path": str(card),
            "pci_bdf": resolved_device.name,
            "vendor_id": vendor_id,
            "device_id": _read_text(device / "device"),
            "subsystem_vendor_id": _read_text(device / "subsystem_vendor"),
            "subsystem_device_id": _read_text(device / "subsystem_device"),
            "hwmon": str(hwmons[0]) if hwmons else None,
            "hwmon_name": _read_text(hwmons[0] / "name") if hwmons else None,
        }
        return device, hwmons[0] if hwmons else None, identity
    selected = f" {card_name!r}" if card_name else ""
    raise RuntimeError(f"No AMD DRM device{selected} with gpu_busy_percent was found")


def _find_labeled_sensor(hwmon: Path, kind: str, label: str, suffix: str) -> Path | None:
    for label_path in sorted(hwmon.glob(f"{kind}[0-9]*_label")):
        if _read_text(label_path) == label:
            return label_path.with_name(label_path.name.removesuffix("_label") + suffix)
    return None


def _read_smaps_rollup(pid: int) -> tuple[int | None, int | None]:
    """Return (PSS, USS) in bytes from Linux smaps_rollup."""
    try:
        fields: dict[str, int] = {}
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            match = re.search(r"(\d+)\s+kB", value)
            if match:
                fields[key] = int(match.group(1)) * 1024
        pss = fields.get("Pss")
        uss_fields = ("Private_Clean", "Private_Dirty", "Private_Hugetlb")
        uss = sum(fields.get(key, 0) for key in uss_fields)
        return pss, uss
    except OSError:
        return None, None


def _empty_process_metrics(pid: int, **flags: bool) -> dict[str, float | int | str | bool | None]:
    return {
        "root_pid": pid,
        "process_count": 0,
        "rss_bytes": 0,
        "pss_bytes": None,
        "uss_bytes": None,
        "cpu_seconds": 0.0,
        "thread_count": 0,
        "read_bytes": 0,
        "write_bytes": 0,
        **flags,
    }


def _process_metrics(
    pid: int,
    expected_create_time: float | None = None,
) -> dict[str, float | int | str | bool | None]:
    try:
        root = psutil.Process(pid)
        if expected_create_time is not None and root.create_time() != expected_create_time:
            return _empty_process_metrics(pid, identity_mismatch=True)
        if root.status() == psutil.STATUS_ZOMBIE:
            return _empty_process_metrics(pid, exited=True)
        processes = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
        return _empty_process_metrics(pid)

    totals: dict[str, float | int] = {
        "process_count": 0,
        "rss_bytes": 0,
        "cpu_seconds": 0.0,
        "thread_count": 0,
        "read_bytes": 0,
        "write_bytes": 0,
    }
    pss_total = 0
    uss_total = 0
    full_memory_available = True
    for process in processes:
        try:
            memory = process.memory_info()
            cpu = process.cpu_times()
            io = process.io_counters()
            pss, uss = _read_smaps_rollup(process.pid)
            totals["process_count"] += 1
            totals["rss_bytes"] += memory.rss
            if pss is None or uss is None:
                full_memory_available = False
            else:
                pss_total += pss
                uss_total += uss
            totals["cpu_seconds"] += cpu.user + cpu.system
            totals["thread_count"] += process.num_threads()
            totals["read_bytes"] += io.read_bytes
            totals["write_bytes"] += io.write_bytes
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            continue
    return {
        "root_pid": pid,
        **totals,
        "pss_bytes": pss_total if full_memory_available else None,
        "uss_bytes": uss_total if full_memory_available else None,
    }


def _process_manifest(
    pid: int,
    *,
    record_command: bool,
    expected_create_time: float | None = None,
) -> dict[str, Any]:
    try:
        process = psutil.Process(pid)
        if expected_create_time is not None and process.create_time() != expected_create_time:
            return {"pid": pid, "unavailable": True, "identity_mismatch": True}
        if process.status() == psutil.STATUS_ZOMBIE:
            return {"pid": pid, "unavailable": True, "exited": True}
        cmdline = _redact_argv(process.cmdline()) if record_command else ["<arguments omitted>"]
        raw_environ = process.environ()
        return {
            "pid": pid,
            "executable": process.exe(),
            "create_time": process.create_time(),
            "command": cmdline,
            "accelerator_environment": _safe_accelerator_environment(raw_environ),
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
        return {"pid": pid, "unavailable": True}


def _sample(
    device: Path,
    hwmon: Path | None,
    targets: dict[str, int],
    target_create_times: Mapping[str, float],
    start: float,
    phase: str,
) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    sample: dict[str, Any] = {
        "record_type": "sample",
        "timestamp": datetime.now(UTC).isoformat(),
        "wall_time_ns": time.time_ns(),
        "elapsed_seconds": time.monotonic() - start,
        "phase": phase,
        "gpu_busy_percent": _read_number(device / "gpu_busy_percent"),
        "vram_used_bytes": _read_number(device / "mem_info_vram_used"),
        "vram_total_bytes": _read_number(device / "mem_info_vram_total"),
        "gtt_used_bytes": _read_number(device / "mem_info_gtt_used"),
        "gtt_total_bytes": _read_number(device / "mem_info_gtt_total"),
        "host_memory_used_bytes": memory.used,
        "host_memory_available_bytes": memory.available,
        "host_memory_percent": memory.percent,
        "host_swap_used_bytes": swap.used,
        "host_cpu_percent": psutil.cpu_percent(interval=None),
        "host_load_1m": os.getloadavg()[0],
        "processes": {
            label: _process_metrics(pid, target_create_times.get(label))
            for label, pid in targets.items()
        },
    }
    if hwmon is not None:
        edge = _find_labeled_sensor(hwmon, "temp", "edge", "_input") or hwmon / "temp1_input"
        junction = _find_labeled_sensor(hwmon, "temp", "junction", "_input") or hwmon / "temp2_input"
        memory_temp = _find_labeled_sensor(hwmon, "temp", "mem", "_input") or hwmon / "temp3_input"
        core_clock = _find_labeled_sensor(hwmon, "freq", "sclk", "_input") or hwmon / "freq1_input"
        memory_clock = _find_labeled_sensor(hwmon, "freq", "mclk", "_input") or hwmon / "freq2_input"
        sample.update(
            {
                "gpu_power_watts": _read_number(hwmon / "power1_average", 1_000_000),
                "gpu_edge_temp_c": _read_number(edge, 1_000),
                "gpu_junction_temp_c": _read_number(junction, 1_000),
                "gpu_memory_temp_c": _read_number(memory_temp, 1_000),
                "gpu_core_clock_mhz": _read_number(core_clock, 1_000_000),
                "gpu_memory_clock_mhz": _read_number(memory_clock, 1_000_000),
            }
        )
    return sample


def _safety_violation(
    sample: Mapping[str, Any],
    limits: Mapping[str, float | None],
    *,
    require_available: bool = False,
) -> dict[str, Any] | None:
    """Return the first configured resource-limit violation in priority order."""
    checks = (
        ("gpu_junction_temp_c", limits.get("max_junction_temp_c"), "maximum", lambda value, limit: value > limit),
        ("vram_used_bytes", limits.get("max_vram_bytes"), "maximum", lambda value, limit: value > limit),
        (
            "host_memory_available_bytes",
            limits.get("min_host_available_bytes"),
            "minimum",
            lambda value, limit: value < limit,
        ),
        ("host_swap_used_bytes", limits.get("max_swap_bytes"), "maximum", lambda value, limit: value > limit),
    )
    for metric, limit, kind, breached in checks:
        value = sample.get(metric)
        if limit is not None and value is None and require_available:
            return {
                "metric": metric,
                "value": None,
                "limit": float(limit),
                "limit_kind": kind,
                "unavailable": True,
            }
        if limit is not None and value is not None and breached(float(value), float(limit)):
            return {
                "metric": metric,
                "value": float(value),
                "limit": float(limit),
                "limit_kind": kind,
            }
    return None


def _nearest_rank_p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _metric_summary(measured: list[float], baseline: list[float]) -> dict[str, float]:
    result: dict[str, float] = {}
    if baseline:
        result["baseline_mean"] = statistics.fmean(baseline)
    if measured:
        result.update(
            {
                "measured_mean": statistics.fmean(measured),
                "measured_min": min(measured),
                "measured_max": max(measured),
                "measured_p95": _nearest_rank_p95(measured),
            }
        )
        if baseline:
            result["measured_max_minus_baseline"] = max(measured) - statistics.fmean(baseline)
    return result


def _summarize(
    samples: list[dict[str, Any]],
    *,
    returncode: int | None,
    status: str,
    received_signal: int | None,
    targets: dict[str, int],
    safety_violation: dict[str, Any] | None = None,
    kernel_driver_errors: list[str] | None = None,
) -> dict[str, Any]:
    baseline_samples = [sample for sample in samples if sample["phase"] == "baseline"]
    measured_samples = [sample for sample in samples if sample["phase"] == "measured"]
    summary: dict[str, Any] = {
        "record_type": "summary",
        "status": status,
        "samples": len(samples),
        "baseline_samples": len(baseline_samples),
        "measured_samples": len(measured_samples),
        "elapsed_seconds": samples[-1]["elapsed_seconds"] if samples else 0.0,
        "returncode": returncode,
        "received_signal": received_signal,
        "metrics": {},
        "processes": {},
    }
    if safety_violation is not None:
        summary["safety_violation"] = safety_violation
    summary["kernel_log_available"] = kernel_driver_errors is not None
    if kernel_driver_errors:
        summary["kernel_driver_errors"] = kernel_driver_errors
    scalar_keys = (
        "gpu_busy_percent",
        "vram_used_bytes",
        "gtt_used_bytes",
        "gpu_power_watts",
        "gpu_junction_temp_c",
        "host_memory_used_bytes",
        "host_swap_used_bytes",
    )
    for key in scalar_keys:
        baseline = [float(sample[key]) for sample in baseline_samples if sample.get(key) is not None]
        measured = [float(sample[key]) for sample in measured_samples if sample.get(key) is not None]
        values = _metric_summary(measured, baseline)
        if values:
            summary["metrics"][key] = values

    process_keys = ("rss_bytes", "pss_bytes", "uss_bytes", "thread_count")
    for label, pid in targets.items():
        process_summary: dict[str, Any] = {"pid": pid, "metrics": {}}
        for key in process_keys:
            baseline = [
                float(sample["processes"][label][key])
                for sample in baseline_samples
                if label in sample["processes"]
                and sample["processes"][label]["process_count"]
                and sample["processes"][label][key] is not None
            ]
            measured = [
                float(sample["processes"][label][key])
                for sample in measured_samples
                if label in sample["processes"]
                and sample["processes"][label]["process_count"]
                and sample["processes"][label][key] is not None
            ]
            values = _metric_summary(measured, baseline)
            if values:
                process_summary["metrics"][key] = values
        cpu_samples = [
            (sample["elapsed_seconds"], float(sample["processes"][label]["cpu_seconds"]))
            for sample in measured_samples
            if label in sample["processes"] and sample["processes"][label]["process_count"]
        ]
        if len(cpu_samples) >= 2 and cpu_samples[-1][0] > cpu_samples[0][0]:
            process_summary["average_cpu_cores"] = (cpu_samples[-1][1] - cpu_samples[0][1]) / (
                cpu_samples[-1][0] - cpu_samples[0][0]
            )
        alive_samples = [
            sample["processes"][label]
            for sample in measured_samples
            if label in sample["processes"] and sample["processes"][label]["process_count"]
        ]
        baseline_alive = [
            sample["processes"][label]
            for sample in baseline_samples
            if label in sample["processes"] and sample["processes"][label]["process_count"]
        ]
        counter_start = baseline_alive[-1] if baseline_alive else (alive_samples[0] if alive_samples else None)
        if counter_start is not None and alive_samples:
            process_summary["counter_deltas"] = {
                key: float(alive_samples[-1][key]) - float(counter_start[key])
                for key in ("read_bytes", "write_bytes")
            }
        summary["processes"][label] = process_summary
    return summary


def _parse_pid_spec(value: str) -> tuple[str, int]:
    if "=" in value:
        label, raw_pid = value.split("=", 1)
    else:
        label, raw_pid = f"pid_{value}", value
    if not label or not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise argparse.ArgumentTypeError("PID label must use only letters, digits, dot, underscore, or dash")
    try:
        pid = int(raw_pid)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid PID: {raw_pid!r}") from error
    if pid <= 0:
        raise argparse.ArgumentTypeError("PID must be positive")
    return label, pid


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _pid_matches_create_time(pid: int, expected_create_time: float) -> bool:
    try:
        process = psutil.Process(pid)
        return process.create_time() == expected_create_time and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
        return False


def _process_descendants(pid: int) -> list[psutil.Process]:
    """Snapshot descendants so children that leave the process group are not leaked."""
    try:
        return psutil.Process(pid).children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
        return []


def _live_processes(processes: list[psutil.Process]) -> list[psutil.Process]:
    live: list[psutil.Process] = []
    for process in processes:
        try:
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                live.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            continue
    return live


def _terminate(process: subprocess.Popen, grace_seconds: float) -> int:
    """Reap the leader and terminate its process group plus escaped descendants."""
    leader_returncode = process.poll()
    descendants = _process_descendants(process.pid)
    if _process_group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    # A worker may create a new session/process group. Signal every descendant
    # captured before terminating the leader so such workers cannot outlive a
    # timeout or safety stop.
    for descendant in reversed(descendants):
        try:
            descendant.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        process.poll()  # Reap the leader so it does not remain as a zombie.
        if not _process_group_exists(process.pid) and not _live_processes(descendants):
            break
        time.sleep(0.05)

    if _process_group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    survivors = _live_processes(descendants)
    for descendant in survivors:
        try:
            descendant.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    if survivors:
        psutil.wait_procs(survivors, timeout=min(grace_seconds, 1.0))

    if leader_returncode is None:
        try:
            return process.wait(timeout=min(grace_seconds, 1.0))
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait()
    # Ensure the already-exited leader is reaped while preserving its status.
    process.wait()
    return leader_returncode


def _terminate_process_trees(
    pids: list[int],
    grace_seconds: float,
    expected_create_times: Mapping[int, float] | None = None,
) -> tuple[list[int], list[int]]:
    """Terminate explicitly opted-in PID trees without assuming process groups."""
    processes: dict[int, psutil.Process] = {}
    roots: dict[int, psutil.Process] = {}
    protected_ids = {os.getpid(), os.getppid()}
    try:
        protected_ids.update(process.pid for process in psutil.Process().parents())
    except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
        pass
    for pid in pids:
        try:
            root = psutil.Process(pid)
            expected_create_time = (
                expected_create_times.get(pid)
                if expected_create_times is not None
                else None
            )
            if expected_create_time is not None and root.create_time() != expected_create_time:
                continue
            if root.pid in protected_ids:
                continue
            roots[root.pid] = root
            processes[root.pid] = root
            try:
                descendants = root.children(recursive=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
                descendants = []
            for candidate in descendants:
                if candidate.pid not in protected_ids:
                    processes[candidate.pid] = candidate
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            continue
    root_ids = set(roots)
    # Stop roots from launching more workers, then terminate the descendant
    # snapshot deepest-first. This also lets a server begin graceful teardown.
    descendants = [process for pid, process in processes.items() if pid not in root_ids]
    ordered = [*roots.values(), *reversed(descendants)]
    for candidate in ordered:
        try:
            candidate.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(ordered, timeout=grace_seconds)
    for candidate in alive:
        try:
            candidate.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, survivors = psutil.wait_procs(alive, timeout=grace_seconds)
    survivor_ids = sorted(process.pid for process in survivors)
    survivor_id_set = set(survivor_ids)
    terminated_ids = sorted(pid for pid in processes if pid not in survivor_id_set)
    return terminated_ids, survivor_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--card", help="DRM card name, for example card1 (default: auto-detect)")
    parser.add_argument(
        "--include-pid",
        action="append",
        default=[],
        type=_parse_pid_spec,
        metavar="[LABEL=]PID",
        help="record this process tree separately; repeat for server and client",
    )
    parser.add_argument("--baseline-seconds", type=float, default=0.0)
    parser.add_argument("--duration", type=float, help="measurement duration when no command is supplied")
    parser.add_argument("--timeout", type=float, help="terminate a wrapped command after this many measured seconds")
    parser.add_argument("--terminate-grace-seconds", type=float, default=5.0)
    parser.add_argument(
        "--terminate-included-on-safety",
        action="store_true",
        help="also terminate every --include-pid tree after a safety limit or driver error",
    )
    parser.add_argument(
        "--sensor-grace-seconds",
        type=float,
        default=5.0,
        help="allow runtime-suspended GPU sensors this long to become readable",
    )
    parser.add_argument("--max-junction-temp-c", type=float)
    parser.add_argument("--max-vram-gib", type=float)
    parser.add_argument("--min-host-available-gib", type=float)
    parser.add_argument("--max-swap-gib", type=float)
    parser.add_argument(
        "--record-command",
        action="store_true",
        help="record redacted command arguments; omit by default to avoid credential leakage",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if (
        not math.isfinite(args.interval)
        or not math.isfinite(args.baseline_seconds)
        or not math.isfinite(args.terminate_grace_seconds)
        or not math.isfinite(args.sensor_grace_seconds)
        or args.interval <= 0
        or args.baseline_seconds < 0
        or args.terminate_grace_seconds <= 0
        or args.sensor_grace_seconds < 0
    ):
        parser.error(
            "interval and terminate grace must be positive; baseline and sensor grace must be nonnegative"
        )
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be positive")
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("--timeout must be positive")
    resource_limits = (
        ("--max-junction-temp-c", args.max_junction_temp_c),
        ("--max-vram-gib", args.max_vram_gib),
        ("--min-host-available-gib", args.min_host_available_gib),
        ("--max-swap-gib", args.max_swap_gib),
    )
    for name, value in resource_limits:
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"{name} must be finite and nonnegative")
    if args.command and args.duration is not None:
        parser.error("--duration is for attach-only mode; use --timeout with a command")
    if not args.command and args.timeout is not None:
        parser.error("--timeout requires a command")
    if not args.command and args.duration is None and not args.include_pid:
        parser.error("provide a command, --duration, or at least one --include-pid")

    explicit_targets = dict(args.include_pid)
    if len(explicit_targets) != len(args.include_pid):
        parser.error("duplicate --include-pid labels are not allowed")
    if "command" in explicit_targets:
        parser.error("the --include-pid label 'command' is reserved for the wrapped command")
    if args.terminate_included_on_safety and not explicit_targets:
        parser.error("--terminate-included-on-safety requires at least one --include-pid")

    explicit_pid_create_times: dict[int, float] = {}
    for label, pid in explicit_targets.items():
        if pid in explicit_pid_create_times:
            continue
        try:
            process_identity = psutil.Process(pid)
            if process_identity.status() == psutil.STATUS_ZOMBIE:
                parser.error(f"--include-pid target {label}={pid} has already exited")
            explicit_pid_create_times[pid] = process_identity.create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            parser.error(f"--include-pid target {label}={pid} does not exist or is inaccessible")

    device, hwmon, gpu_identity = _find_gpu(args.card)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    if args.output.exists() or summary_path.exists():
        parser.error("refusing to overwrite an existing telemetry or summary file")

    process: subprocess.Popen | None = None
    samples: list[dict[str, Any]] = []
    targets = dict(explicit_targets)
    target_create_times = {
        label: explicit_pid_create_times[pid]
        for label, pid in explicit_targets.items()
    }
    returncode: int | None = None
    status = "running"
    received_signal: int | None = None
    safety_violation: dict[str, Any] | None = None
    kernel_driver_errors: list[str] | None = None
    terminated_explicit_pids: list[int] = []
    surviving_explicit_pids: list[int] = []
    explicit_termination_attempted = False
    stop_requested = False
    kernel_log_start = time.time()
    start = time.monotonic()
    measured_start: float | None = None
    safety_limits = {
        "max_junction_temp_c": args.max_junction_temp_c,
        "max_vram_bytes": args.max_vram_gib * _GIB if args.max_vram_gib is not None else None,
        "min_host_available_bytes": (
            args.min_host_available_gib * _GIB if args.min_host_available_gib is not None else None
        ),
        "max_swap_bytes": args.max_swap_gib * _GIB if args.max_swap_gib is not None else None,
    }

    def handle_signal(signum, _frame):
        nonlocal received_signal, stop_requested
        received_signal = signum
        stop_requested = True
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    error: BaseException | None = None
    output_descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(output_descriptor, "w", encoding="utf-8") as output:
        manifest = {
            "record_type": "manifest",
            "timestamp": datetime.now(UTC).isoformat(),
            "wall_time_ns": time.time_ns(),
            "interval_seconds": args.interval,
            "baseline_seconds": args.baseline_seconds,
            "duration_seconds": args.duration,
            "timeout_seconds": args.timeout,
            "safety_limits": safety_limits,
            "sensor_grace_seconds": args.sensor_grace_seconds,
            "terminate_included_on_safety": args.terminate_included_on_safety,
            "gpu": gpu_identity,
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "rocm": _read_text(Path("/opt/rocm/.info/version")),
                "jax": _package_version("jax"),
                "jaxlib": _package_version("jaxlib"),
                "jax_rocm_plugin": _package_version("jax-rocm7-plugin"),
                "jax_rocm_pjrt": _package_version("jax-rocm7-pjrt"),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "accelerator_environment": _safe_accelerator_environment(os.environ),
            },
            "explicit_processes": {
                label: _process_manifest(
                    pid,
                    record_command=args.record_command,
                    expected_create_time=target_create_times[label],
                )
                for label, pid in explicit_targets.items()
            },
            "command": (
                _redact_argv(args.command)
                if args.record_command
                else ([_redact_argument(args.command[0]), "<arguments omitted>"] if args.command else [])
            ),
            "command_recorded": args.record_command,
        }
        output.write(_json_dumps(manifest, separators=(",", ":")) + "\n")
        output.flush()

        def record(phase: str) -> dict[str, Any] | None:
            sample = _sample(device, hwmon, targets, target_create_times, start, phase)
            samples.append(sample)
            output.write(_json_dumps(sample, separators=(",", ":")) + "\n")
            output.flush()
            sensors_must_be_available = (
                phase == "measured"
                and measured_start is not None
                and time.monotonic() - measured_start >= args.sensor_grace_seconds
            )
            return _safety_violation(
                sample,
                safety_limits,
                require_available=sensors_must_be_available,
            )

        try:
            next_sample = time.monotonic()
            baseline_end = next_sample + args.baseline_seconds
            while time.monotonic() < baseline_end and not stop_requested:
                now = time.monotonic()
                if now >= next_sample:
                    safety_violation = record("baseline")
                    if safety_violation is not None:
                        status = "safety_limit"
                        stop_requested = True
                        break
                    next_sample = now + args.interval
                time.sleep(max(0.01, min(args.interval / 4, next_sample - time.monotonic())))

            if not stop_requested:
                safety_violation = record("preflight")
                if safety_violation is not None:
                    status = "safety_limit"
                    stop_requested = True

            if args.command and not stop_requested:
                process = subprocess.Popen(args.command, start_new_session=True)
                targets["command"] = process.pid

            measured_start = time.monotonic()
            next_sample = measured_start
            next_kernel_check = measured_start
            while not stop_requested:
                now = time.monotonic()
                if now >= next_sample:
                    safety_violation = record("measured")
                    if safety_violation is not None:
                        status = "safety_limit"
                        stop_requested = True
                        break
                    # Skip missed ticks rather than biasing means with catch-up samples.
                    next_sample = now + args.interval

                if now >= next_kernel_check:
                    kernel_driver_errors = _kernel_driver_errors_since(kernel_log_start)
                    if kernel_driver_errors:
                        status = "driver_error"
                        stop_requested = True
                        break
                    next_kernel_check = now + 1.0

                if process is not None and process.poll() is not None:
                    status = "completed" if process.returncode == 0 else "command_failed"
                    returncode = process.returncode
                    break
                if args.timeout is not None and now - measured_start >= args.timeout:
                    status = "timeout"
                    stop_requested = True
                    break
                if process is None and args.duration is not None and now - measured_start >= args.duration:
                    status = "completed"
                    break
                if process is None and args.duration is None and explicit_targets:
                    if not any(
                        _pid_matches_create_time(pid, explicit_pid_create_times[pid])
                        for pid in explicit_targets.values()
                    ):
                        status = "targets_exited"
                        break
                time.sleep(max(0.01, min(args.interval / 4, next_sample - time.monotonic())))
        except BaseException as caught:
            error = caught
            status = "error"
        finally:
            if process is not None:
                returncode = _terminate(process, args.terminate_grace_seconds)
            if received_signal is not None and status not in {
                "safety_limit",
                "driver_error",
                "timeout",
                "error",
            }:
                status = "signal"
            # Do not delay termination of an explicitly opted-in server tree
            # behind a journal query after an already-known guardrail stop.
            if status in {"safety_limit", "driver_error"} and args.terminate_included_on_safety:
                explicit_termination_attempted = True
                terminated_explicit_pids, surviving_explicit_pids = _terminate_process_trees(
                    list(explicit_targets.values()),
                    args.terminate_grace_seconds,
                    explicit_pid_create_times,
                )
            final_kernel_errors = _kernel_driver_errors_since(kernel_log_start)
            if final_kernel_errors is not None:
                combined_errors = [*(kernel_driver_errors or []), *final_kernel_errors]
                kernel_driver_errors = list(dict.fromkeys(combined_errors))[-100:]
            # Preserve the first, higher-priority explicit safety stop. A
            # driver error outranks timeout/signal/command status, but not an
            # already-recorded resource-limit violation.
            if kernel_driver_errors and status not in {"safety_limit", "error"}:
                status = "driver_error"
            if (
                status in {"safety_limit", "driver_error"}
                and args.terminate_included_on_safety
                and not explicit_termination_attempted
            ):
                explicit_termination_attempted = True
                terminated_explicit_pids, surviving_explicit_pids = _terminate_process_trees(
                    list(explicit_targets.values()),
                    args.terminate_grace_seconds,
                    explicit_pid_create_times,
                )
            summary = _summarize(
                samples,
                returncode=returncode,
                status=status,
                received_signal=received_signal,
                targets=targets,
                safety_violation=safety_violation,
                kernel_driver_errors=kernel_driver_errors,
            )
            if explicit_termination_attempted:
                summary["terminated_explicit_pids"] = terminated_explicit_pids
                summary["surviving_explicit_pids"] = surviving_explicit_pids
            if error is not None:
                summary["error_type"] = type(error).__name__
                summary["error"] = str(error)
            _write_private_text(summary_path, _json_dumps(summary, indent=2, sort_keys=True) + "\n")
            print(_json_dumps(summary, indent=2, sort_keys=True))

    if error is not None:
        raise error
    if status == "safety_limit":
        return 125
    if status == "driver_error":
        return 126
    if status == "timeout":
        return 124
    if received_signal is not None:
        return 128 + received_signal
    if returncode is None:
        return 0
    return 128 - returncode if returncode < 0 else returncode


if __name__ == "__main__":
    sys.exit(main())
