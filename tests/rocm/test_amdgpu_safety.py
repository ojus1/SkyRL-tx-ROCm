from __future__ import annotations

import importlib.util
import os
import stat
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
acquire_qwen35_rocm_launch_lock = _MODULE.acquire_qwen35_rocm_launch_lock
guarded_qwen35_rocm_process = _MODULE.guarded_qwen35_rocm_process
require_clean_amdgpu_boot = _MODULE.require_clean_amdgpu_boot
require_headless_unowned_amdgpu = _MODULE.require_headless_unowned_amdgpu


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


def _headless_amd_drm(tmp_path: Path) -> Path:
    drm_root = tmp_path / "drm"
    (drm_root / "card1" / "device").mkdir(parents=True)
    (drm_root / "card1" / "device" / "vendor").write_text("0x1002\n")
    return drm_root


def _character_stat(_path):
    return SimpleNamespace(st_mode=stat.S_IFCHR)


def test_shared_hardware_preflight_accepts_only_headless_unowned_kfd(tmp_path):
    kfd = tmp_path / "dev" / "kfd"

    def unowned(arguments, **kwargs):
        assert arguments == ["/usr/bin/fuser", str(kfd)]
        assert kwargs == {
            "capture_output": True,
            "text": True,
            "timeout": 5,
            "check": False,
        }
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    result = require_headless_unowned_amdgpu(
        drm_root=_headless_amd_drm(tmp_path),
        kfd_path=kfd,
        stat_fn=_character_stat,
        access_fn=lambda *_args: True,
        which_fn=lambda _name: "/usr/bin/fuser",
        run_fn=unowned,
    )

    assert result == {
        "amd_cards": ["card1"],
        "connected_amd_connectors": [],
        "kfd_path": str(kfd),
        "kfd_accessible": True,
        "kfd_unowned": True,
    }


def test_shared_hardware_preflight_rejects_active_amd_display_before_kfd(tmp_path):
    drm_root = _headless_amd_drm(tmp_path)
    connector = drm_root / "card1-HDMI-A-1"
    connector.mkdir()
    (connector / "status").write_text("connected\n")

    with pytest.raises(RuntimeError, match="AMD display connector is active"):
        require_headless_unowned_amdgpu(drm_root=drm_root)


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "message"),
    [
        (0, "1234", "", "already owned: 1234"),
        (0, "", "/dev/kfd: kernel", "already owned"),
        (2, "", "permission denied", "could not verify exclusive"),
        (1, "", "unexpected output", "could not verify exclusive"),
    ],
)
def test_shared_hardware_preflight_fails_closed_on_fuser_outcomes(
    tmp_path, returncode, stdout, stderr, message
):
    with pytest.raises(RuntimeError, match=message):
        require_headless_unowned_amdgpu(
            drm_root=_headless_amd_drm(tmp_path),
            kfd_path=tmp_path / "dev" / "kfd",
            stat_fn=_character_stat,
            access_fn=lambda *_args: True,
            which_fn=lambda _name: "/usr/bin/fuser",
            run_fn=lambda *_args, **_kwargs: SimpleNamespace(
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            ),
        )


def test_shared_lock_serializes_with_existing_probe_locks(tmp_path, monkeypatch):
    from rocm.probe_bfc_preallocation import _acquire_global_lock

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    lock_fd = acquire_qwen35_rocm_launch_lock()
    try:
        with pytest.raises(RuntimeError, match="global launch lock"):
            _acquire_global_lock()
    finally:
        os.close(lock_fd)


def test_guard_releases_lock_when_hardware_preflight_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setattr(
        _MODULE,
        "require_clean_amdgpu_boot",
        lambda: {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []},
    )
    monkeypatch.setattr(
        _MODULE,
        "require_headless_unowned_amdgpu",
        lambda: (_ for _ in ()).throw(RuntimeError("KFD refusal")),
    )

    with pytest.raises(RuntimeError, match="KFD refusal"):
        with guarded_qwen35_rocm_process():
            pytest.fail("guard yielded after a failed preflight")

    released_fd = acquire_qwen35_rocm_launch_lock()
    os.close(released_fd)


def test_every_full_model_entrypoint_enforces_boot_quarantine():
    for relative_path in (
        "rocm/probe_bfc_preallocation.py",
        "rocm/probe_model_residency.py",
        "rocm/probe_sft_compile.py",
        "rocm/start_qwen35.sh",
    ):
        source = (_REPO / relative_path).read_text(encoding="utf-8")
        assert "amdgpu_safety" in source, relative_path
