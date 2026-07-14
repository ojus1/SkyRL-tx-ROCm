from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_LAUNCHER = _REPO / "rocm" / "start_qwen35.sh"
_READY_NAME = "engine-start.ready"
_RELEASE_NAME = "engine-start.release"
_TELEMETRY_NAME = "engine-start.watchdog.telemetry.jsonl"
_XLA_FLAGS = "--xla_gpu_enable_command_buffer="


def _gate_harness(tmp_path: Path) -> Path:
    source = _LAUNCHER.read_text(encoding="utf-8")
    start = source.index("validate_engine_start_gate_directory() {\n")
    end = source.index("\n# Default-off, compile-only static-bucket prewarm.", start)
    functions = source[start:end]
    profile_source = tmp_path / "profile_rocm.py"
    profile_source.write_text(
        """import os
import signal
import time


def replace_process(_signum, _frame):
    os.execv("/usr/bin/sleep", ["sleep", "30"])


signal.signal(signal.SIGUSR1, replace_process)
while True:
    time.sleep(0.1)
""",
        encoding="utf-8",
    )
    harness = tmp_path / "engine-start-gate-harness.sh"
    harness.write_text(
        """#!/bin/bash
set -euo pipefail
umask 077
engine_start_gate_dir="$1"
engine_start_gate_timeout_seconds="$2"
events="$3"
engine_start_profile_python="$4"
engine_start_profile_source="$5"
engine_start_profile_sha256="$6"
engine_start_telemetry="$engine_start_gate_dir/engine-start.watchdog.telemetry.jsonl"
engine_start_watchdog_pid=
engine_start_watchdog_start_ticks=
engine_start_watchdog_manifest_sha256=
engine_start_ready_payload=
engine_start_release_payload=
amd_card_names=(card1)
run_id=gate-test
bf16_rms_gate_up_lora_swiglu_contiguous=1
prewarm_audit_sha256=$(printf 'a%.0s' {1..64})
prewarm_handoff_sha256=$(printf 'b%.0s' {1..64})
export XLA_FLAGS=--xla_gpu_enable_command_buffer=
"""
        + functions
        + """
trap 'printf '"'"'signal-int\\n'"'"' >>"$events"; exit 130' INT
trap 'printf '"'"'signal-term\\n'"'"' >>"$events"; exit 143' TERM
printf 'before-wait\\n' >>"$events"
await_engine_start_release
printf 'released\\n' >>"$events"
/usr/bin/sleep "${FAKE_FINAL_REVALIDATE_DELAY:-0}"
if ! revalidate_engine_start_watchdog; then
  printf 'final-revalidation-failed\\n' >>"$events"
  exit 2
fi
printf 'after-release\\n' >>"$events"
""",
        encoding="utf-8",
    )
    harness.chmod(0o700)
    return harness


def _private_gate_dir(tmp_path: Path) -> Path:
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir(mode=0o700)
    return gate_dir


def _launcher_environment(tmp_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith(("PYTHON", "UV_")) or name in {
            "BASH_ENV",
            "ENV",
            "LD_AUDIT",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "__PYVENV_LAUNCHER__",
            "VIRTUAL_ENV",
            "VIRTUAL_ENV_PROMPT",
        }:
            environment.pop(name)
    environment.pop("JAX_ENABLE_PGLE", None)
    environment.pop("JAX_COMPILATION_CACHE_EXPECT_PGLE", None)
    environment["SKYRL_QWEN35_PREWARM_BUCKETS"] = "64"
    environment["SKYRL_QWEN35_RUN_ROOT"] = str(tmp_path / "runs")
    return environment


def _wait_for_path(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists() or path.is_symlink():
            return
        assert process.poll() is None, process.stderr.read() if process.stderr else ""
        time.sleep(0.01)
    process.kill()
    raise AssertionError(f"timed out waiting for {path}")


def _wait_for_event(events: Path, expected: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if events.exists() and expected in events.read_text(encoding="utf-8").splitlines():
            return
        assert process.poll() is None, process.stderr.read() if process.stderr else ""
        time.sleep(0.01)
    process.kill()
    raise AssertionError(f"timed out waiting for event {expected!r}")


def _wait_for_cmdline(pid: int, expected_first: str) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            first = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0", 1)[0].decode()
        except (FileNotFoundError, ProcessLookupError):
            first = ""
        if first == expected_first:
            return
        time.sleep(0.01)
    raise AssertionError(f"PID {pid} did not exec {expected_first!r}")


def _start_harness(
    harness: Path,
    gate_dir: Path,
    events: Path,
    timeout: int = 5,
    *,
    final_delay: float = 0,
) -> subprocess.Popen[str]:
    arguments = _harness_arguments(harness, gate_dir, events, timeout)
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = _XLA_FLAGS
    environment["FAKE_FINAL_REVALIDATE_DELAY"] = str(final_delay)
    return subprocess.Popen(
        arguments,
        cwd=_REPO,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _harness_arguments(harness: Path, gate_dir: Path, events: Path, timeout: int) -> list[str]:
    profile_source = harness.parent / "profile_rocm.py"
    return [
        str(harness),
        str(gate_dir),
        str(timeout),
        str(events),
        sys.executable,
        str(profile_source),
        hashlib.sha256(profile_source.read_bytes()).hexdigest(),
    ]


def _watchdog_start_ticks(pid: int) -> str:
    fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(") ", 1)[1]
    return fields.split()[19]


def _release_payload(
    ready_payload: str,
    *,
    watchdog_pid: int = os.getpid(),
    manifest_sha256: str = "c" * 64,
    watchdog_start_ticks: str | None = None,
) -> str:
    if watchdog_start_ticks is None:
        watchdog_start_ticks = _watchdog_start_ticks(watchdog_pid)
    ready_line = ready_payload.removesuffix("\n")
    release_line = ready_line.replace(
        "skyrl-qwen35-engine-start-ready-v1",
        "skyrl-qwen35-engine-start-release-v1",
        1,
    )
    return (
        f"{release_line} watchdog_pid={watchdog_pid} "
        f"watchdog_start_ticks={watchdog_start_ticks} "
        f"watchdog_manifest_sha256={manifest_sha256}\n"
    )


def _publish_release(
    gate_dir: Path,
    ready_payload: str,
    *,
    watchdog_pid: int = os.getpid(),
    manifest_sha256: str = "c" * 64,
    watchdog_start_ticks: str | None = None,
) -> None:
    staged = gate_dir / ".engine-start.release.test.tmp"
    staged.write_text(
        _release_payload(
            ready_payload,
            watchdog_pid=watchdog_pid,
            manifest_sha256=manifest_sha256,
            watchdog_start_ticks=watchdog_start_ticks,
        ),
        encoding="utf-8",
    )
    staged.chmod(0o600)
    os.replace(staged, gate_dir / _RELEASE_NAME)


def _profile_argv(gate_dir: Path, server_pid: int, profile_source: Path) -> list[str]:
    return [
        sys.executable,
        "-B",
        str(profile_source),
        "--output",
        str(gate_dir / _TELEMETRY_NAME),
        "--card",
        "card1",
        "--interval",
        "0.1",
        "--include-pid",
        f"server={server_pid}",
        "--terminate-included-on-safety",
        "--terminate-included-on-abort",
        "--sensor-grace-seconds",
        "60",
        "--max-junction-temp-c",
        "90",
        "--max-gpu-power-watts",
        "400",
        "--max-vram-gib",
        "24",
        "--min-host-available-gib",
        "0",
        "--max-swap-gib",
        "8",
        "--record-command",
    ]


def _ready_server_pid(ready_payload: str) -> int:
    field = next(part for part in ready_payload.split() if part.startswith("launcher_pid="))
    return int(field.split("=", 1)[1])


def _write_watchdog_telemetry(
    gate_dir: Path,
    ready_payload: str,
    watchdog_pid: int,
    profile_source: Path,
) -> str:
    server_pid = _ready_server_pid(ready_payload)
    manifest = {
        "record_type": "manifest",
        "interval_seconds": 0.1,
        "baseline_seconds": 0.0,
        "duration_seconds": None,
        "timeout_seconds": None,
        "safety_limits": {
            "max_junction_temp_c": 90.0,
            "max_gpu_power_watts": 400.0,
            "max_vram_bytes": 24 * 1024**3,
            "min_host_available_bytes": 0,
            "max_swap_bytes": 8 * 1024**3,
        },
        "sensor_grace_seconds": 60.0,
        "terminate_included_on_safety": True,
        "terminate_included_on_abort": True,
        "gpu": {"card": "card1", "device_id": "0x744c"},
        "runtime": {
            "script_sha256": hashlib.sha256(profile_source.read_bytes()).hexdigest(),
            "accelerator_environment": {"XLA_FLAGS": _XLA_FLAGS},
        },
        "explicit_processes": {
            "server": {
                "pid": server_pid,
                "executable": "/usr/bin/bash",
                "command": [str(_LAUNCHER), "gate-test"],
                "accelerator_environment": {"XLA_FLAGS": _XLA_FLAGS},
            }
        },
        "command": [],
        "command_recorded": True,
        "passed_file_descriptor_count": 0,
    }
    sample = {
        "record_type": "sample",
        "phase": "measured",
        "processes": {
            "server": {
                "root_pid": server_pid,
                "process_count": 1,
            }
        },
    }
    first_line = (json.dumps(manifest, separators=(",", ":")) + "\n").encode()
    telemetry = gate_dir / _TELEMETRY_NAME
    telemetry.write_bytes(first_line + (json.dumps(sample, separators=(",", ":")) + "\n").encode())
    telemetry.chmod(0o600)
    return hashlib.sha256(first_line).hexdigest()


def _tamper_watchdog_telemetry(gate_dir: Path, invalid_kind: str) -> str:
    telemetry = gate_dir / _TELEMETRY_NAME
    lines = telemetry.read_bytes().splitlines(keepends=True)
    manifest = json.loads(lines[0])
    if invalid_kind == "script-hash":
        manifest["runtime"]["script_sha256"] = "0" * 64
    elif invalid_kind == "limit":
        manifest["safety_limits"]["max_gpu_power_watts"] = 399.0
    elif invalid_kind == "server-pid":
        manifest["explicit_processes"]["server"]["pid"] += 1
    elif invalid_kind == "card":
        manifest["gpu"]["card"] = "card0"
    elif invalid_kind == "xla":
        manifest["runtime"]["accelerator_environment"]["XLA_FLAGS"] = "bad"
    elif invalid_kind == "measured-sample":
        sample = json.loads(lines[1])
        sample["phase"] = "preflight"
        lines[1] = (json.dumps(sample, separators=(",", ":")) + "\n").encode()
    else:
        raise AssertionError(f"unknown invalid telemetry kind: {invalid_kind}")
    first_line = (json.dumps(manifest, separators=(",", ":")) + "\n").encode()
    telemetry.write_bytes(first_line + b"".join(lines[1:]))
    telemetry.chmod(0o600)
    return hashlib.sha256(first_line).hexdigest()


def _start_authenticated_watchdog(
    harness: Path,
    gate_dir: Path,
    ready_payload: str,
) -> tuple[subprocess.Popen[str], str]:
    profile_source = harness.parent / "profile_rocm.py"
    environment = os.environ.copy()
    environment["XLA_FLAGS"] = _XLA_FLAGS
    watchdog = subprocess.Popen(
        _profile_argv(gate_dir, _ready_server_pid(ready_payload), profile_source),
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    manifest_sha256 = _write_watchdog_telemetry(
        gate_dir,
        ready_payload,
        watchdog.pid,
        profile_source,
    )
    return watchdog, manifest_sha256


def test_gate_is_after_postflight_cache_attestation_and_prewarm_only_exit() -> None:
    source = _LAUNCHER.read_text(encoding="utf-8")
    final_journal = source.rindex("if run_amdgpu_safety >/dev/null; then")
    final_status = source.index("if ((prewarm_status != 0))", final_journal)
    cache_hash = source.index(
        'prewarm_handoff_sha256="${prewarm_handoff_sha256_line%% *}"',
        final_status,
    )
    prewarm_only = source.index('if [[ "$prewarm_only" == "1" ]]', cache_hash)
    gate_call = source.index("\n  await_engine_start_release\n", prewarm_only)
    post_release_journal = source.index("if ! run_amdgpu_safety >/dev/null; then", gate_call)
    kfd_gate = source.index("if ! require_engine_start_kfd_unowned; then", gate_call)
    gate_environment_unset = source.index("unset SKYRL_QWEN35_ENGINE_START_GATE_DIR", kfd_gate)
    final_watchdog_gate = source.index("if ! revalidate_engine_start_watchdog; then", gate_environment_unset)
    clear_traps = source.index("\ntrap - EXIT INT TERM\n", final_watchdog_gate)
    stable_api = source.index("  exec /usr/bin/env -i \\\n", clear_traps)
    fallback_api = source.index(
        'exec "$uv_executable" run --active --no-sync -m skyrl.tinker.api',
        stable_api,
    )

    assert final_journal < final_status < cache_hash < prewarm_only < gate_call
    assert (
        gate_call
        < post_release_journal
        < kfd_gate
        < gate_environment_unset
        < final_watchdog_gate
        < clear_traps
        < stable_api
        < fallback_api
    )
    assert 'engine_start_gate_dir="${SKYRL_QWEN35_ENGINE_START_GATE_DIR-}"' in source
    assert 'if [[ -n "$engine_start_gate_dir" && "$prewarm_only" == "0" ]]' in source
    assert ("SKYRL_QWEN35_ENGINE_START_GATE_DIR requires " "SKYRL_QWEN35_ENGINE_T64_CACHE_ATTEST=1.") in source
    assert ('engine_start_telemetry="$engine_start_gate_dir/' 'engine-start.watchdog.telemetry.jsonl"') in source
    assert 'profile_source_blob_oid="$(source_blob_oid rocm/profile_rocm.py 100644)"' in source
    assert 'kfd_owners="$(/usr/bin/fuser /dev/kfd 2>&1)"' in source
    default_prewarm_dependencies = next(line for line in source.splitlines() if line.startswith("  for executable in "))
    gate_dependency_section = source[
        source.index("  for engine_start_gate_executable in ") : source.index("require_unset_or_exact()")
    ]
    for gate_only_executable in ("/usr/bin/mv", "/usr/bin/rm", "/usr/bin/sleep"):
        assert gate_only_executable not in default_prewarm_dependencies
        assert gate_only_executable in gate_dependency_section
    assert source.index("unset SKYRL_QWEN35_ENGINE_START_GATE_DIR", gate_call) < clear_traps
    subprocess.run(["bash", "-n", str(_LAUNCHER)], check=True)


@pytest.mark.parametrize(
    ("invalid_kind", "message"),
    [
        ("missing-cache-attestation", "requires SKYRL_QWEN35_ENGINE_T64_CACHE_ATTEST=1"),
        ("timeout", "must be an integer in [1, 3600]"),
        ("timeout-without-directory", "requires nonempty SKYRL_QWEN35_ENGINE_START_GATE_DIR"),
        ("prewarm-only", "requires SKYRL_QWEN35_PREWARM_ONLY=0"),
        ("directory-mode", "must be an absolute canonical, owned mode-0700 directory"),
        ("stale-ready", "refusing a stale engine-start gate marker"),
    ],
)
def test_launcher_rejects_invalid_gate_configuration_before_hardware_access(
    tmp_path: Path, invalid_kind: str, message: str
) -> None:
    gate_dir = _private_gate_dir(tmp_path)
    environment = _launcher_environment(tmp_path)
    if invalid_kind != "timeout-without-directory":
        environment["SKYRL_QWEN35_ENGINE_START_GATE_DIR"] = str(gate_dir)
    if invalid_kind != "missing-cache-attestation":
        environment["SKYRL_QWEN35_ENGINE_T64_CACHE_ATTEST"] = "1"
    if invalid_kind == "timeout":
        environment["SKYRL_QWEN35_ENGINE_START_GATE_TIMEOUT_SECONDS"] = "0"
    elif invalid_kind == "timeout-without-directory":
        environment["SKYRL_QWEN35_ENGINE_START_GATE_TIMEOUT_SECONDS"] = "5"
        environment.pop("SKYRL_QWEN35_ENGINE_T64_CACHE_ATTEST", None)
    elif invalid_kind == "prewarm-only":
        environment["SKYRL_QWEN35_PREWARM_ONLY"] = "1"
        environment.pop("SKYRL_QWEN35_ENGINE_T64_CACHE_ATTEST", None)
    elif invalid_kind == "directory-mode":
        gate_dir.chmod(0o750)
    elif invalid_kind == "stale-ready":
        (gate_dir / _READY_NAME).write_text("stale\n", encoding="utf-8")

    result = subprocess.run(
        [str(_LAUNCHER), "invalid-engine-gate"],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert "AMDGPU" not in result.stderr
    assert not (tmp_path / "runs").exists()


def test_gate_publishes_private_ready_and_accepts_exact_atomic_release(
    tmp_path: Path,
) -> None:
    harness = _gate_harness(tmp_path)
    gate_dir = _private_gate_dir(tmp_path)
    events = tmp_path / "events"
    process = _start_harness(harness, gate_dir, events)
    ready = gate_dir / _READY_NAME
    _wait_for_path(ready, process)

    payload = ready.read_text(encoding="utf-8")
    assert payload.startswith("skyrl-qwen35-engine-start-ready-v1 nonce=")
    for field in (
        " launcher_pid=",
        " boot_id=",
        " run_id=gate-test",
        " fused=1",
        f" prewarm_sha256={'a' * 64}",
        f" handoff_sha256={'b' * 64}",
    ):
        assert field in payload
    assert payload.endswith("\n")
    assert ready.stat().st_mode & 0o777 == 0o600
    assert ready.stat().st_nlink == 1
    assert events.read_text(encoding="utf-8") == "before-wait\n"

    watchdog, manifest_sha256 = _start_authenticated_watchdog(
        harness,
        gate_dir,
        payload,
    )
    try:
        _publish_release(
            gate_dir,
            payload,
            watchdog_pid=watchdog.pid,
            manifest_sha256=manifest_sha256,
        )
        stdout, stderr = process.communicate(timeout=5)
    finally:
        watchdog.terminate()
        watchdog.wait(timeout=5)

    assert process.returncode == 0, stderr
    assert stdout == ""
    assert stderr == ""
    assert events.read_text(encoding="utf-8") == ("before-wait\nreleased\nafter-release\n")
    assert (gate_dir / _RELEASE_NAME).stat().st_mode & 0o777 == 0o600
    assert sorted(path.name for path in gate_dir.iterdir()) == [
        _READY_NAME,
        _RELEASE_NAME,
        _TELEMETRY_NAME,
    ]


@pytest.mark.parametrize(
    "invalid_kind",
    ["token", "mode", "symlink", "hardlink", "watchdog_ticks", "manifest"],
)
def test_gate_fails_closed_on_invalid_release(tmp_path: Path, invalid_kind: str) -> None:
    harness = _gate_harness(tmp_path)
    gate_dir = _private_gate_dir(tmp_path)
    events = tmp_path / "events"
    process = _start_harness(harness, gate_dir, events)
    ready = gate_dir / _READY_NAME
    release = gate_dir / _RELEASE_NAME
    _wait_for_path(ready, process)
    payload = ready.read_text(encoding="utf-8")

    if invalid_kind == "token":
        release_payload = _release_payload(payload)
        nonce_start = release_payload.index(" nonce=") + len(" nonce=")
        replacement = "0" if release_payload[nonce_start] != "0" else "1"
        staged = gate_dir / ".bad-token.tmp"
        staged.write_text(
            release_payload[:nonce_start] + replacement + release_payload[nonce_start + 1 :],
            encoding="utf-8",
        )
        staged.chmod(0o600)
        os.replace(staged, release)
    elif invalid_kind == "mode":
        _publish_release(gate_dir, payload)
        release.chmod(0o640)
    elif invalid_kind == "symlink":
        release.symlink_to(ready)
    elif invalid_kind == "hardlink":
        os.link(ready, release)
    elif invalid_kind == "watchdog_ticks":
        bad_release = _release_payload(payload).replace(
            f"watchdog_start_ticks={_watchdog_start_ticks(os.getpid())}",
            "watchdog_start_ticks=1",
        )
        staged = gate_dir / ".bad-ticks.tmp"
        staged.write_text(bad_release, encoding="utf-8")
        staged.chmod(0o600)
        os.replace(staged, release)
    else:
        bad_release = _release_payload(payload).replace(
            f"watchdog_manifest_sha256={'c' * 64}",
            "watchdog_manifest_sha256=not-a-digest",
        )
        staged = gate_dir / ".bad-manifest.tmp"
        staged.write_text(bad_release, encoding="utf-8")
        staged.chmod(0o600)
        os.replace(staged, release)

    _stdout, stderr = process.communicate(timeout=5)

    assert process.returncode == 2
    assert "refusing an" in stderr
    assert "engine-start release marker" in stderr
    assert events.read_text(encoding="utf-8") == "before-wait\n"


def test_gate_rejects_arbitrary_same_uid_pid_and_manifest_hash(
    tmp_path: Path,
) -> None:
    harness = _gate_harness(tmp_path)
    gate_dir = _private_gate_dir(tmp_path)
    events = tmp_path / "events"
    process = _start_harness(harness, gate_dir, events)
    ready = gate_dir / _READY_NAME
    _wait_for_path(ready, process)
    payload = ready.read_text(encoding="utf-8")
    manifest_sha256 = _write_watchdog_telemetry(
        gate_dir,
        payload,
        os.getpid(),
        harness.parent / "profile_rocm.py",
    )

    _publish_release(
        gate_dir,
        payload,
        watchdog_pid=os.getpid(),
        manifest_sha256=manifest_sha256,
    )
    _stdout, stderr = process.communicate(timeout=5)

    assert process.returncode == 2
    assert "exact profile_rocm policy" in stderr
    assert events.read_text(encoding="utf-8") == "before-wait\n"


def test_gate_rejects_arbitrary_manifest_hash_for_exact_profiler(
    tmp_path: Path,
) -> None:
    harness = _gate_harness(tmp_path)
    gate_dir = _private_gate_dir(tmp_path)
    events = tmp_path / "events"
    process = _start_harness(harness, gate_dir, events)
    ready = gate_dir / _READY_NAME
    _wait_for_path(ready, process)
    payload = ready.read_text(encoding="utf-8")
    watchdog, _manifest_sha256 = _start_authenticated_watchdog(
        harness,
        gate_dir,
        payload,
    )
    try:
        _publish_release(
            gate_dir,
            payload,
            watchdog_pid=watchdog.pid,
            manifest_sha256="d" * 64,
        )
        _stdout, stderr = process.communicate(timeout=5)
    finally:
        watchdog.terminate()
        watchdog.wait(timeout=5)

    assert process.returncode == 2
    assert "watchdog manifest digest mismatch" in stderr
    assert events.read_text(encoding="utf-8") == "before-wait\n"


@pytest.mark.parametrize(
    "invalid_kind",
    ["script-hash", "limit", "server-pid", "card", "xla", "measured-sample"],
)
def test_gate_rejects_nonexact_profiler_telemetry_policy(tmp_path: Path, invalid_kind: str) -> None:
    harness = _gate_harness(tmp_path)
    gate_dir = _private_gate_dir(tmp_path)
    events = tmp_path / "events"
    process = _start_harness(harness, gate_dir, events)
    ready = gate_dir / _READY_NAME
    _wait_for_path(ready, process)
    payload = ready.read_text(encoding="utf-8")
    watchdog, _manifest_sha256 = _start_authenticated_watchdog(
        harness,
        gate_dir,
        payload,
    )
    manifest_sha256 = _tamper_watchdog_telemetry(gate_dir, invalid_kind)
    try:
        _publish_release(
            gate_dir,
            payload,
            watchdog_pid=watchdog.pid,
            manifest_sha256=manifest_sha256,
        )
        _stdout, stderr = process.communicate(timeout=5)
    finally:
        watchdog.terminate()
        watchdog.wait(timeout=5)

    assert process.returncode == 2, stderr
    assert events.read_text(encoding="utf-8") == "before-wait\n"


def test_gate_rejects_stopped_exact_profiler(tmp_path: Path) -> None:
    harness = _gate_harness(tmp_path)
    gate_dir = _private_gate_dir(tmp_path)
    events = tmp_path / "events"
    process = _start_harness(harness, gate_dir, events)
    ready = gate_dir / _READY_NAME
    _wait_for_path(ready, process)
    payload = ready.read_text(encoding="utf-8")
    watchdog, manifest_sha256 = _start_authenticated_watchdog(
        harness,
        gate_dir,
        payload,
    )
    start_ticks = _watchdog_start_ticks(watchdog.pid)
    os.kill(watchdog.pid, signal.SIGSTOP)
    try:
        _publish_release(
            gate_dir,
            payload,
            watchdog_pid=watchdog.pid,
            watchdog_start_ticks=start_ticks,
            manifest_sha256=manifest_sha256,
        )
        _stdout, stderr = process.communicate(timeout=5)
    finally:
        os.kill(watchdog.pid, signal.SIGCONT)
        watchdog.terminate()
        watchdog.wait(timeout=5)

    assert process.returncode == 2
    assert "without a live owned watchdog" in stderr


@pytest.mark.parametrize("identity_change", ["exited", "execed"])
def test_retained_watchdog_identity_is_revalidated_before_exec(tmp_path: Path, identity_change: str) -> None:
    harness = _gate_harness(tmp_path)
    gate_dir = _private_gate_dir(tmp_path)
    events = tmp_path / "events"
    process = _start_harness(harness, gate_dir, events, final_delay=1)
    ready = gate_dir / _READY_NAME
    _wait_for_path(ready, process)
    payload = ready.read_text(encoding="utf-8")
    watchdog, manifest_sha256 = _start_authenticated_watchdog(
        harness,
        gate_dir,
        payload,
    )
    _publish_release(
        gate_dir,
        payload,
        watchdog_pid=watchdog.pid,
        manifest_sha256=manifest_sha256,
    )
    _wait_for_event(events, "released", process)

    if identity_change == "exited":
        watchdog.terminate()
        watchdog.wait(timeout=5)
    else:
        watchdog.send_signal(signal.SIGUSR1)
        _wait_for_cmdline(watchdog.pid, "sleep")

    _stdout, stderr = process.communicate(timeout=5)
    if watchdog.poll() is None:
        watchdog.terminate()
        watchdog.wait(timeout=5)

    assert process.returncode == 2, stderr
    assert events.read_text(encoding="utf-8") == ("before-wait\nreleased\nfinal-revalidation-failed\n")


@pytest.mark.parametrize("invalid_kind", ["mode", "symlink"])
def test_gate_rejects_invalid_directory_before_ready_publication(tmp_path: Path, invalid_kind: str) -> None:
    harness = _gate_harness(tmp_path)
    real_dir = _private_gate_dir(tmp_path)
    if invalid_kind == "mode":
        real_dir.chmod(0o750)
        gate_dir = real_dir
    else:
        gate_dir = tmp_path / "gate-link"
        gate_dir.symlink_to(real_dir, target_is_directory=True)
    events = tmp_path / "events"

    result = subprocess.run(
        _harness_arguments(harness, gate_dir, events, 2),
        cwd=_REPO,
        env={**os.environ, "XLA_FLAGS": _XLA_FLAGS},
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 2
    assert "directory is no longer private and canonical" in result.stderr
    assert not (real_dir / _READY_NAME).exists()


def test_gate_timeout_is_bounded_and_leaves_ready_evidence(tmp_path: Path) -> None:
    harness = _gate_harness(tmp_path)
    gate_dir = _private_gate_dir(tmp_path)
    events = tmp_path / "events"
    started = time.monotonic()

    result = subprocess.run(
        _harness_arguments(harness, gate_dir, events, 1),
        cwd=_REPO,
        env={**os.environ, "XLA_FLAGS": _XLA_FLAGS},
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    elapsed = time.monotonic() - started
    assert result.returncode == 2
    assert "timed out waiting for the engine-start release marker" in result.stderr
    assert elapsed < 3
    assert (gate_dir / _READY_NAME).is_file()
    assert not (gate_dir / _RELEASE_NAME).exists()
    assert events.read_text(encoding="utf-8") == "before-wait\n"


def test_gate_wait_honors_existing_term_trap(tmp_path: Path) -> None:
    harness = _gate_harness(tmp_path)
    gate_dir = _private_gate_dir(tmp_path)
    events = tmp_path / "events"
    process = _start_harness(harness, gate_dir, events, timeout=30)
    _wait_for_path(gate_dir / _READY_NAME, process)

    process.send_signal(signal.SIGTERM)
    _stdout, stderr = process.communicate(timeout=5)

    assert process.returncode == 143, stderr
    assert events.read_text(encoding="utf-8") == "before-wait\nsignal-term\n"
    assert not (gate_dir / _RELEASE_NAME).exists()
