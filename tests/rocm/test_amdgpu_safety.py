from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).parents[2]
_SAFETY = _REPO / "rocm" / "amdgpu_safety.py"
_SPEC = importlib.util.spec_from_file_location("amdgpu_safety", _SAFETY)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
amdgpu_fatal_events_since_boot = _MODULE.amdgpu_fatal_events_since_boot
require_clean_amdgpu_boot = _MODULE.require_clean_amdgpu_boot


def _journal(returncode=0, stdout="", stderr=""):
    def run(arguments, **kwargs):
        assert arguments == [
            "journalctl",
            "-k",
            "-b",
            "--no-pager",
            "-o",
            "short-iso",
        ]
        assert kwargs == {
            "capture_output": True,
            "text": True,
            "timeout": 15,
            "check": False,
        }
        return SimpleNamespace(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    return run


def test_clean_boot_returns_manifest_fragment():
    result = require_clean_amdgpu_boot(
        run_fn=_journal(stdout="kernel: amdgpu: SMU is resumed successfully!\n")
    )

    assert result == {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}


@pytest.mark.parametrize(
    "line",
    [
        "kernel: [drm:gfx_v11_0_bad_op_irq [amdgpu]] *ERROR* Illegal opcode in command stream",
        "kernel: amdgpu: ring gfx_0.0.0 timeout, signaled seq=1",
        "kernel: amdgpu_job_timedout: ring gfx timeout",
        "kernel: amdgpu: GPU reset begin!",
        "kernel: amdgpu: VM_CONTEXT1_PROTECTION_FAULT_ADDR 0x0",
        "kernel: amdgpu: GPU page fault detected",
    ],
)
def test_fatal_driver_event_requires_reboot(line):
    with pytest.raises(RuntimeError, match="reboot before retrying"):
        require_clean_amdgpu_boot(run_fn=_journal(stdout=line + "\n"))


def test_unrelated_non_amdgpu_text_is_ignored():
    events = amdgpu_fatal_events_since_boot(
        run_fn=_journal(stdout="application: illegal opcode in command stream\n")
    )

    assert events == []


def test_journal_failure_is_fail_closed():
    with pytest.raises(RuntimeError, match="could not verify"):
        require_clean_amdgpu_boot(
            run_fn=_journal(returncode=1, stderr="permission denied")
        )


def test_journal_timeout_is_fail_closed():
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("journalctl", 15)

    with pytest.raises(RuntimeError, match="timed out"):
        require_clean_amdgpu_boot(run_fn=timeout)


def test_every_full_model_entrypoint_enforces_boot_quarantine():
    for relative_path in (
        "rocm/probe_bfc_preallocation.py",
        "rocm/probe_model_residency.py",
        "rocm/probe_sft_compile.py",
        "rocm/start_qwen35.sh",
    ):
        source = (_REPO / relative_path).read_text(encoding="utf-8")
        assert "amdgpu_safety" in source, relative_path
