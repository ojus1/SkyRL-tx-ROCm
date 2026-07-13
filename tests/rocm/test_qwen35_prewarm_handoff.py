from __future__ import annotations

import importlib.util
import json
import os
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

_REPO = Path(__file__).resolve().parents[2]
_HELPER_PATH = _REPO / "rocm" / "qwen35_prewarm_handoff.py"
_PROFILE_PATH = _REPO / "rocm" / "profile_rocm.py"
_LAUNCHER = _REPO / "rocm" / "start_qwen35.sh"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


handoff = _load("qwen35_prewarm_handoff_test", _HELPER_PATH)
profile = _load("qwen35_profile_pass_fd_test", _PROFILE_PATH)


@dataclass
class FakeHardware:
    drm_root: Path
    dev_root: Path
    pci_root: Path
    card_link: Path
    boot_id_path: Path
    vram: Path
    gtt: Path
    runtime_status: Path
    output: Path
    node_identities: dict[str, dict[str, str]]


@dataclass
class FakeClock:
    value: float = 0.0
    after_sleep: Callable[[float], None] | None = None

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds
        if self.after_sleep is not None:
            self.after_sleep(self.value)


def _fake_hardware(tmp_path: Path) -> FakeHardware:
    drm_root = tmp_path / "sys" / "class" / "drm"
    drm_root.mkdir(parents=True)
    pci_root = (
        tmp_path
        / "sys"
        / "devices"
        / "pci0000:00"
        / "0000:03:00.0"
    )
    card_sysfs = pci_root / "drm" / "card1"
    render_sysfs = pci_root / "drm" / "renderD128"
    card_sysfs.mkdir(parents=True)
    render_sysfs.mkdir()
    (card_sysfs / "device").symlink_to(pci_root, target_is_directory=True)
    card_link = drm_root / "card1"
    card_link.symlink_to(card_sysfs, target_is_directory=True)
    (pci_root / "power").mkdir()
    (pci_root / "vendor").write_text("0x1002\n", encoding="utf-8")
    (pci_root / "device").write_text("0x744c\n", encoding="utf-8")
    (card_sysfs / "dev").write_text("226:1\n", encoding="utf-8")
    (render_sysfs / "dev").write_text("226:128\n", encoding="utf-8")
    vram = pci_root / "mem_info_vram_used"
    gtt = pci_root / "mem_info_gtt_used"
    runtime_status = pci_root / "power" / "runtime_status"
    vram.write_text("100\n", encoding="utf-8")
    gtt.write_text("20\n", encoding="utf-8")
    runtime_status.write_text("suspended\n", encoding="utf-8")

    dev_root = tmp_path / "dev"
    (dev_root / "dri").mkdir(parents=True)
    (dev_root / "kfd").touch()
    (dev_root / "dri" / "card1").touch()
    (dev_root / "dri" / "renderD128").touch()

    node_identities = {
        str(dev_root / "kfd"): {
            "path": str(dev_root / "kfd"),
            "rdev": "236:0",
            "sysfs_dev": "236:0",
            "sysfs_target": str(tmp_path / "sys" / "devices" / "virtual" / "kfd" / "kfd"),
        },
        str(dev_root / "dri" / "card1"): {
            "path": str(dev_root / "dri" / "card1"),
            "rdev": "226:1",
            "sysfs_dev": "226:1",
            "sysfs_target": str(card_sysfs),
        },
        str(dev_root / "dri" / "renderD128"): {
            "path": str(dev_root / "dri" / "renderD128"),
            "rdev": "226:128",
            "sysfs_dev": "226:128",
            "sysfs_target": str(render_sysfs),
        },
    }

    boot_id_path = tmp_path / "proc" / "sys" / "kernel" / "random" / "boot_id"
    boot_id_path.parent.mkdir(parents=True)
    boot_id_path.write_text("54ccf56c-5f4f-4ef7-ac98-c13e0587b5b9\n", encoding="utf-8")

    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    run_dir.chmod(0o700)
    return FakeHardware(
        drm_root=drm_root,
        dev_root=dev_root,
        pci_root=pci_root,
        card_link=card_link,
        boot_id_path=boot_id_path,
        vram=vram,
        gtt=gtt,
        runtime_status=runtime_status,
        output=run_dir / "prewarm-handoff.jsonl",
        node_identities=node_identities,
    )


def _node_identity(fake: FakeHardware, path: Path) -> dict[str, str]:
    assert path.is_file()
    return dict(fake.node_identities[str(path)])


def _clean_boot() -> dict[str, object]:
    return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}


def _capture(fake: FakeHardware, owner_probe=lambda _path: ()) -> dict[str, object]:
    return handoff.capture_baseline(
        fake.output,
        drm_root=fake.drm_root,
        dev_root=fake.dev_root,
        boot_id_path=fake.boot_id_path,
        node_identity_probe=lambda path: _node_identity(fake, path),
        owner_probe=owner_probe,
        boot_validator=_clean_boot,
    )


def _settle(
    fake: FakeHardware,
    clock: FakeClock,
    owner_probe=lambda _path: (),
    *,
    timeout: float = 1.0,
) -> dict[str, object]:
    return handoff.settle_handoff(
        fake.output,
        timeout_seconds=timeout,
        poll_interval_seconds=1.0,
        drm_root=fake.drm_root,
        dev_root=fake.dev_root,
        boot_id_path=fake.boot_id_path,
        node_identity_probe=lambda path: _node_identity(fake, path),
        owner_probe=owner_probe,
        boot_validator=_clean_boot,
        monotonic_fn=clock.monotonic,
        sleep_fn=clock.sleep,
    )


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_character_node_identity_matches_rdev_to_resolved_sysfs_dev(
    tmp_path: Path,
) -> None:
    node = tmp_path / "dev" / "dri" / "renderD128"
    node.parent.mkdir(parents=True)
    node.touch()
    sysfs_target = tmp_path / "sys" / "devices" / "pci" / "drm" / "renderD128"
    sysfs_target.mkdir(parents=True)
    (sysfs_target / "dev").write_text("226:128\n", encoding="utf-8")
    sys_dev_char = tmp_path / "sys" / "dev" / "char"
    sys_dev_char.mkdir(parents=True)
    (sys_dev_char / "226:128").symlink_to(sysfs_target, target_is_directory=True)

    identity = handoff._character_node_identity(
        node,
        stat_fn=lambda _path: SimpleNamespace(
            st_mode=stat.S_IFCHR | 0o660,
            st_rdev=os.makedev(226, 128),
        ),
        access_fn=lambda _path, _mode: True,
        sys_dev_char_root=sys_dev_char,
    )

    assert identity == {
        "path": str(node),
        "rdev": "226:128",
        "sysfs_dev": "226:128",
        "sysfs_target": str(sysfs_target),
    }


@pytest.mark.parametrize(
    ("mode", "accessible", "sysfs_dev", "message"),
    (
        (stat.S_IFREG | 0o660, True, "226:128", "accessible character device"),
        (stat.S_IFCHR | 0o660, False, "226:128", "accessible character device"),
        (stat.S_IFCHR | 0o660, True, "226:129", "does not match sysfs"),
    ),
)
def test_character_node_identity_rejects_substitution_or_sysfs_mismatch(
    tmp_path: Path,
    mode: int,
    accessible: bool,
    sysfs_dev: str,
    message: str,
) -> None:
    node = tmp_path / "dev" / "kfd"
    node.parent.mkdir(parents=True)
    node.touch()
    sysfs_target = tmp_path / "sys" / "devices" / "virtual" / "kfd" / "kfd"
    sysfs_target.mkdir(parents=True)
    (sysfs_target / "dev").write_text(f"{sysfs_dev}\n", encoding="utf-8")
    sys_dev_char = tmp_path / "sys" / "dev" / "char"
    sys_dev_char.mkdir(parents=True)
    (sys_dev_char / "226:128").symlink_to(sysfs_target, target_is_directory=True)

    with pytest.raises(handoff.HandoffError, match=message):
        handoff._character_node_identity(
            node,
            stat_fn=lambda _path: SimpleNamespace(
                st_mode=mode,
                st_rdev=os.makedev(226, 128),
            ),
            access_fn=lambda _path, _mode: accessible,
            sys_dev_char_root=sys_dev_char,
        )


def test_delayed_release_requires_three_consecutive_exact_idle_samples(
    tmp_path: Path,
) -> None:
    fake = _fake_hardware(tmp_path)
    baseline = _capture(fake)
    fake.vram.write_text("700000000\n", encoding="utf-8")
    fake.gtt.write_text("21\n", encoding="utf-8")
    fake.runtime_status.write_text("active\n", encoding="utf-8")
    clock = FakeClock()

    def release_after_two_seconds(now: float) -> None:
        if now >= 2.0:
            fake.vram.write_text("100\n", encoding="utf-8")
            fake.gtt.write_text("20\n", encoding="utf-8")
            fake.runtime_status.write_text("suspended\n", encoding="utf-8")

    clock.after_sleep = release_after_two_seconds
    result = _settle(fake, clock, timeout=5.0)

    assert baseline["baseline"]["vram_used_bytes"] == 100
    assert result["status"] == "passed"
    assert result["final_ready_streak"] == 3
    assert result["accelerator_device_opened"] is False
    assert result["vram_tolerance_bytes"] == 0
    assert result["gtt_tolerance_bytes"] == 0
    records = _records(fake.output)
    assert records[0]["record_type"] == "prewarm_handoff_baseline"
    assert records[0]["release_contract"]["poll_interval_seconds"] == 1.0
    assert records[0]["accelerator_device_opened"] is False
    assert records[-1]["record_type"] == "prewarm_handoff_complete"
    ready_samples = [
        record
        for record in records
        if record["record_type"] == "prewarm_handoff_sample"
        and record["status"] == "ready_candidate"
    ]
    assert [sample["elapsed_seconds"] for sample in ready_samples[-3:]] == [
        2.0,
        3.0,
        4.0,
    ]
    assert stat.S_IMODE(fake.output.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("relative_node", "identity_field"),
    (
        ("kfd", "kfd_node"),
        ("dri/card1", "drm_node"),
        ("dri/renderD128", "render_node"),
    ),
)
def test_same_path_character_device_rebinding_fails_closed(
    tmp_path: Path, relative_node: str, identity_field: str
) -> None:
    fake = _fake_hardware(tmp_path)
    baseline = _capture(fake)
    node = fake.dev_root / relative_node
    identity = fake.node_identities[str(node)]
    identity["rdev"] = "240:7"
    identity["sysfs_dev"] = "240:7"
    identity["sysfs_target"] = str(tmp_path / "sys" / "devices" / "replacement")

    with pytest.raises(handoff.HandoffError, match="identity|rdev"):
        _settle(fake, FakeClock())

    assert baseline["device"][identity_field]
    terminal = _records(fake.output)[-1]
    assert terminal["record_type"] == "prewarm_handoff_error"
    assert terminal["accelerator_device_opened"] is False


def test_same_card_name_rebound_to_different_pci_bdf_fails_closed(
    tmp_path: Path,
) -> None:
    fake = _fake_hardware(tmp_path)
    baseline = _capture(fake)
    replacement_root = fake.pci_root.with_name("0000:04:00.0")
    fake.pci_root.rename(replacement_root)
    replacement_card = replacement_root / "drm" / "card1"
    replacement_render = replacement_root / "drm" / "renderD128"
    (replacement_card / "device").unlink()
    (replacement_card / "device").symlink_to(
        replacement_root, target_is_directory=True
    )
    fake.card_link.unlink()
    fake.card_link.symlink_to(replacement_card, target_is_directory=True)
    fake.node_identities[str(fake.dev_root / "dri" / "card1")][
        "sysfs_target"
    ] = str(replacement_card)
    fake.node_identities[str(fake.dev_root / "dri" / "renderD128")][
        "sysfs_target"
    ] = str(replacement_render)

    with pytest.raises(handoff.HandoffError, match="identity"):
        _settle(fake, FakeClock())

    assert baseline["device"]["pci_bdf"] == "0000:03:00.0"
    terminal = _records(fake.output)[-1]
    assert terminal["record_type"] == "prewarm_handoff_error"
    assert terminal["accelerator_device_opened"] is False


@pytest.mark.parametrize(
    ("field", "replacement"),
    (("vendor", "0x1234\n"), ("device", "0x9999\n")),
)
def test_pci_vendor_or_device_change_after_capture_fails_closed(
    tmp_path: Path, field: str, replacement: str
) -> None:
    fake = _fake_hardware(tmp_path)
    _capture(fake)
    (fake.pci_root / field).write_text(replacement, encoding="utf-8")

    with pytest.raises(handoff.HandoffError, match="AMD DRM GPU"):
        _settle(fake, FakeClock())

    terminal = _records(fake.output)[-1]
    assert terminal["record_type"] == "prewarm_handoff_error"
    assert terminal["accelerator_device_opened"] is False


def test_identity_change_during_bracketed_snapshot_fails_immediately(
    tmp_path: Path,
) -> None:
    fake = _fake_hardware(tmp_path)
    render = fake.dev_root / "dri" / "renderD128"

    def owner_probe(path: Path) -> tuple[int, ...]:
        if path.name == "renderD128":
            fake.node_identities[str(render)].update(
                {
                    "rdev": "240:7",
                    "sysfs_dev": "240:7",
                    "sysfs_target": str(
                        tmp_path / "sys" / "devices" / "replacement"
                    ),
                }
            )
        return ()

    with pytest.raises(handoff.HandoffError, match="identity|rdev"):
        _capture(fake, owner_probe=owner_probe)

    assert not fake.output.exists()


def test_keyboard_interrupt_during_settle_records_one_terminal_error(
    tmp_path: Path,
) -> None:
    fake = _fake_hardware(tmp_path)
    _capture(fake)
    clock = FakeClock()

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        handoff.settle_handoff(
            fake.output,
            timeout_seconds=5.0,
            poll_interval_seconds=1.0,
            drm_root=fake.drm_root,
            dev_root=fake.dev_root,
            boot_id_path=fake.boot_id_path,
            node_identity_probe=lambda path: _node_identity(fake, path),
            owner_probe=lambda _path: (),
            boot_validator=_clean_boot,
            monotonic_fn=clock.monotonic,
            sleep_fn=interrupt,
        )

    terminal = [
        record
        for record in _records(fake.output)
        if record["record_type"].endswith(("complete", "timeout", "error"))
    ]
    assert len(terminal) == 1
    assert terminal[0]["record_type"] == "prewarm_handoff_error"
    assert terminal[0]["error_type"] == "KeyboardInterrupt"


@pytest.mark.parametrize("owner_name", ("kfd", "renderD128"))
def test_persistent_accelerator_owner_times_out_and_never_passes(
    tmp_path: Path, owner_name: str
) -> None:
    fake = _fake_hardware(tmp_path)
    _capture(fake)
    clock = FakeClock()

    def owner_probe(path: Path) -> tuple[int, ...]:
        return (4321,) if path.name == owner_name else ()

    with pytest.raises(handoff.HandoffError, match="exact idle baseline"):
        _settle(fake, clock, owner_probe)

    records = _records(fake.output)
    assert records[-1]["record_type"] == "prewarm_handoff_timeout"
    assert records[-1]["accelerator_device_opened"] is False
    samples = [
        record
        for record in records
        if record["record_type"] == "prewarm_handoff_sample"
    ]
    assert samples
    owner_check = "kfd_unowned" if owner_name == "kfd" else "render_unowned"
    assert all(sample["checks"][owner_check] is False for sample in samples)
    assert not any(
        record["record_type"] == "prewarm_handoff_complete" for record in records
    )


def test_persistent_single_byte_vram_residue_times_out(tmp_path: Path) -> None:
    fake = _fake_hardware(tmp_path)
    _capture(fake)
    fake.vram.write_text("101\n", encoding="utf-8")

    with pytest.raises(handoff.HandoffError, match="exact idle baseline"):
        _settle(fake, FakeClock())

    records = _records(fake.output)
    assert records[-1]["record_type"] == "prewarm_handoff_timeout"
    assert records[-1]["checks"]["vram_no_higher_than_exact_baseline"] is False


@pytest.mark.parametrize("value", ["", "unreadable", "-1", "1.5", "nan"])
def test_unreadable_or_invalid_vram_fails_closed_immediately(
    tmp_path: Path, value: str
) -> None:
    fake = _fake_hardware(tmp_path)
    _capture(fake)
    fake.vram.write_text(value, encoding="utf-8")
    clock = FakeClock()

    with pytest.raises(handoff.HandoffError, match="VRAM used"):
        _settle(fake, clock)

    assert clock.value == 0
    records = _records(fake.output)
    assert records[-1]["record_type"] == "prewarm_handoff_error"
    assert records[-1]["status"] == "failed"


def test_missing_vram_sysfs_file_fails_closed_immediately(tmp_path: Path) -> None:
    fake = _fake_hardware(tmp_path)
    _capture(fake)
    fake.vram.unlink()
    clock = FakeClock()

    with pytest.raises(handoff.HandoffError, match="cannot read VRAM used"):
        _settle(fake, clock)

    assert clock.value == 0
    assert _records(fake.output)[-1]["record_type"] == "prewarm_handoff_error"


def test_runtime_active_until_deadline_records_timeout(tmp_path: Path) -> None:
    fake = _fake_hardware(tmp_path)
    _capture(fake)
    fake.runtime_status.write_text("active\n", encoding="utf-8")

    with pytest.raises(handoff.HandoffError, match="within 1 seconds"):
        _settle(fake, FakeClock())

    terminal = _records(fake.output)[-1]
    assert terminal["record_type"] == "prewarm_handoff_timeout"
    assert terminal["checks"]["runtime_suspended"] is False
    assert terminal["elapsed_seconds"] == 1.0


@pytest.mark.parametrize(
    ("timeout", "poll_interval", "message"),
    (
        (121.0, 1.0, "no greater than 120"),
        (120.0, 0.25, "exactly 1 second"),
    ),
)
def test_settle_rejects_weakened_release_window_before_hardware_discovery(
    tmp_path: Path, timeout: float, poll_interval: float, message: str
) -> None:
    with pytest.raises(handoff.HandoffError, match=message):
        handoff.settle_handoff(
            tmp_path / "not-created.jsonl",
            timeout_seconds=timeout,
            poll_interval_seconds=poll_interval,
        )


def test_fuser_pid_parser_does_not_treat_render_minor_as_owner_pid() -> None:
    result = subprocess.CompletedProcess(
        ["fuser", "/dev/dri/renderD128"],
        0,
        stdout=" 4567",
        stderr="/dev/dri/renderD128:",
    )

    owners = handoff._fuser_owner_pids(
        Path("/dev/dri/renderD128"),
        which_fn=lambda _name: "/usr/bin/fuser",
        run_fn=lambda *_args, **_kwargs: result,
    )

    assert owners == (4567,)


def test_fuser_no_owner_result_requires_completely_empty_output() -> None:
    result = subprocess.CompletedProcess(
        ["fuser", "/dev/kfd"],
        1,
        stdout="",
        stderr="",
    )

    assert (
        handoff._fuser_owner_pids(
            Path("/dev/kfd"),
            which_fn=lambda _name: "/usr/bin/fuser",
            run_fn=lambda *_args, **_kwargs: result,
        )
        == ()
    )


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    (
        ("pid=4567", "/dev/dri/renderD128:"),
        ("4567", "/dev/dri/renderD128:\nwarning: 999"),
        ("4567", ""),
    ),
)
def test_fuser_owner_result_rejects_extra_or_unparseable_output(
    stdout: str, stderr: str
) -> None:
    result = subprocess.CompletedProcess(
        ["fuser", "/dev/dri/renderD128"],
        0,
        stdout=stdout,
        stderr=stderr,
    )

    with pytest.raises(handoff.HandoffError, match="malformed owner output"):
        handoff._fuser_owner_pids(
            Path("/dev/dri/renderD128"),
            which_fn=lambda _name: "/usr/bin/fuser",
            run_fn=lambda *_args, **_kwargs: result,
        )


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity", "1e9999"))
def test_settle_rejects_nonfinite_json_numbers(tmp_path: Path, constant: str) -> None:
    fake = _fake_hardware(tmp_path)
    _capture(fake)
    payload = fake.output.read_text(encoding="utf-8")
    assert '"vram_used_bytes":100' in payload
    fake.output.write_text(
        payload.replace('"vram_used_bytes":100', f'"vram_used_bytes":{constant}', 1),
        encoding="utf-8",
    )

    with pytest.raises(handoff.HandoffError, match="artifact is malformed"):
        _settle(fake, FakeClock())


def test_settle_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    fake = _fake_hardware(tmp_path)
    _capture(fake)
    payload = fake.output.read_text(encoding="utf-8")
    assert '"schema_version":1' in payload
    fake.output.write_text(
        payload.replace(
            '"schema_version":1',
            '"schema_version":1,"schema_version":1',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(handoff.HandoffError, match="artifact is malformed"):
        _settle(fake, FakeClock())


def test_settle_rejects_unknown_baseline_field_and_records_no_open(
    tmp_path: Path,
) -> None:
    fake = _fake_hardware(tmp_path)
    _capture(fake)
    baseline = _records(fake.output)[0]
    baseline["unexpected"] = True
    fake.output.write_text(json.dumps(baseline) + "\n", encoding="utf-8")

    with pytest.raises(handoff.HandoffError, match="top-level schema"):
        _settle(fake, FakeClock())

    terminal = _records(fake.output)[-1]
    assert terminal["record_type"] == "prewarm_handoff_error"
    assert terminal["accelerator_device_opened"] is False


def test_settle_rejects_changed_release_contract(tmp_path: Path) -> None:
    fake = _fake_hardware(tmp_path)
    _capture(fake)
    baseline = _records(fake.output)[0]
    baseline["release_contract"]["poll_interval_seconds"] = 0.25
    fake.output.write_text(json.dumps(baseline) + "\n", encoding="utf-8")

    with pytest.raises(handoff.HandoffError, match="release contract"):
        _settle(fake, FakeClock())


def test_profile_pass_fd_validation_is_narrow_and_requires_open_descriptors() -> None:
    read_fd, write_fd = os.pipe()
    try:
        assert profile._validated_pass_fds([read_fd, write_fd]) == (
            read_fd,
            write_fd,
        )
        with pytest.raises(ValueError, match="duplicate"):
            profile._validated_pass_fds([read_fd, read_fd])
        with pytest.raises(ValueError, match="at least 3"):
            profile._validated_pass_fds([2])
    finally:
        os.close(read_fd)
        os.close(write_fd)
    with pytest.raises(ValueError, match="is not open"):
        profile._validated_pass_fds([read_fd])


def test_profile_forwards_only_requested_descriptor_to_wrapped_cpu_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "telemetry.jsonl"
    read_fd, write_fd = os.pipe()

    def sample(_device, _hwmon, _targets, _create_times, start, phase):
        return {
            "record_type": "sample",
            "phase": phase,
            "elapsed_seconds": profile.time.monotonic() - start,
            "gpu_power_watts": None,
            "gpu_junction_temp_c": None,
            "vram_used_bytes": 0,
            "gtt_used_bytes": 0,
            "host_memory_used_bytes": 0,
            "host_memory_available_bytes": 64 * 1024**3,
            "host_swap_used_bytes": 0,
            "processes": {},
        }

    monkeypatch.setattr(
        profile,
        "_find_gpu",
        lambda _card: (Path("/mock/device"), None, {"card": "mock"}),
    )
    monkeypatch.setattr(profile, "_sample", sample)
    monkeypatch.setattr(profile, "_kernel_driver_errors_since", lambda _start: [])
    monkeypatch.setattr(profile.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_PROFILE_PATH),
            "--output",
            str(output),
            "--interval",
            "0.01",
            "--pass-fd",
            str(read_fd),
            "--",
            sys.executable,
            "-c",
            (
                "import errno,os,sys\n"
                "os.fstat(int(sys.argv[1]))\n"
                "try:\n"
                "    os.fstat(int(sys.argv[2]))\n"
                "except OSError as error:\n"
                "    assert error.errno == errno.EBADF\n"
                "else:\n"
                "    raise RuntimeError('unrequested descriptor was inherited')\n"
            ),
            str(read_fd),
            str(write_fd),
        ],
    )
    try:
        assert profile.main() == 0
    finally:
        os.close(read_fd)
        os.close(write_fd)

    manifest = _records(output)[0]
    assert manifest["passed_file_descriptor_count"] == 1
    assert manifest["command_recorded"] is False
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert (
        stat.S_IMODE(output.with_suffix(output.suffix + ".summary.json").stat().st_mode)
        == 0o600
    )


@pytest.mark.parametrize("timeout", ("599", "14401", "nan", "3600.0", "999999"))
def test_launcher_rejects_invalid_long_timeout_before_hardware_access(
    tmp_path: Path, timeout: str
) -> None:
    run_root = tmp_path / "runs"
    environment = os.environ.copy()
    environment["SKYRL_QWEN35_PREWARM_TIMEOUT_SECONDS"] = timeout
    environment["SKYRL_QWEN35_RUN_ROOT"] = str(run_root)

    result = subprocess.run(
        ["bash", str(_LAUNCHER), "invalid-timeout"],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 2
    assert "must be an integer in [600, 14400]" in result.stderr
    assert not run_root.exists()


def test_launcher_profiles_then_always_settles_before_final_journal_and_api() -> None:
    source = _LAUNCHER.read_text(encoding="utf-8")
    capture = "python rocm/qwen35_prewarm_handoff.py capture"
    profiler = "python rocm/profile_rocm.py"
    child = "python rocm/prewarm_qwen35_buckets.py"
    settle = "python rocm/qwen35_prewarm_handoff.py settle"
    final_journal = "python3 -m rocm.amdgpu_safety >/dev/null"
    api = "exec uv run --active --no-sync -m skyrl.tinker.api"

    assert source.index(capture) < source.index(profiler) < source.index(child)
    cleanup_definition = source.index("finish_prewarm_once()")
    cleanup_call = source.index("  finish_prewarm_once\n", source.index(child))
    cleanup_journal = source.index(final_journal, cleanup_definition)
    assert cleanup_definition < source.index(settle) < cleanup_journal
    assert source.index(child) < cleanup_call
    handoff_gate = source.index("if ((prewarm_handoff_status != 0))", cleanup_call)
    termination_gate = source.index(
        "if ((prewarm_termination_status != 0))", handoff_gate
    )
    prewarm_gate = source.index("if ((prewarm_status != 0))", termination_gate)
    prewarm_only_gate = source.index('if [[ "$prewarm_only" == "1" ]]', prewarm_gate)
    assert (
        cleanup_call
        < handoff_gate
        < termination_gate
        < prewarm_gate
        < prewarm_only_gate
        < source.index(api)
    )
    assert 'prewarm_handoff_artifact="$run_dir/prewarm-handoff.jsonl"' in source
    assert (
        'prewarm_timeout_seconds="${SKYRL_QWEN35_PREWARM_TIMEOUT_SECONDS:-3600}"'
        in source
    )
    assert '--output "$run_dir/prewarm.telemetry.jsonl"' in source
    for exact_argument in (
        "--interval 0.1",
        "--baseline-seconds 2",
        '--timeout "$prewarm_timeout_seconds"',
        "--sensor-grace-seconds 60",
        "--max-junction-temp-c 90",
        "--max-gpu-power-watts 315",
        "--max-vram-gib 24",
        "--min-host-available-gib 0",
        "--max-swap-gib 8",
        '--pass-fd "$launch_lock_fd"',
        "--timeout-seconds 120",
        "--poll-interval-seconds 1",
    ):
        assert exact_argument in source
    assert source.count("export XLA_FLAGS=") == 1
    assert "export XLA_FLAGS=--xla_gpu_enable_command_buffer=" in source
    assert source.count("export JAX_ENABLE_PGLE=") == 1
    assert source.count("export JAX_COMPILATION_CACHE_EXPECT_PGLE=") == 1
    subprocess.run(["bash", "-n", str(_LAUNCHER)], check=True)


def _prewarm_trap_harness(tmp_path: Path) -> tuple[Path, Path]:
    source = _LAUNCHER.read_text(encoding="utf-8")
    start = source.index("prewarm_status=0\n")
    end = source.index("\nexec uv run --active", start)
    production_block = source[start:end]
    events = tmp_path / "events.txt"
    harness = tmp_path / "prewarm-trap-harness.sh"
    harness.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
events="$EVENTS"
run_dir="$(dirname "$events")"
prewarm_buckets=64
prewarm_optimizer=0
prewarm_only="${FAKE_PREWARM_ONLY:-0}"
prewarm_timeout_seconds=600
memory_mode=growth
SKYRL_ROCM_PALLAS_ATTENTION=0
model_path=/mock/qwen35
amd_card_names=(card1)
exec {launch_lock_fd}</dev/null

python() {
  local script="$1"
  if [[ "$script" == "rocm/qwen35_prewarm_handoff.py" ]]; then
    if [[ "$2" == "capture" ]]; then
      printf 'capture\n' >>"$events"
      return 0
    fi
    printf 'settle\n' >>"$events"
    sleep "${FAKE_SETTLE_DELAY:-0}"
    return "${FAKE_SETTLE_STATUS:-0}"
  fi
  if [[ "$script" == "rocm/profile_rocm.py" ]]; then
    printf 'profile_start\n' >>"$events"
    case "${FAKE_PROFILE_MODE:-exit0}" in
      exit0) printf 'profile_exit\n' >>"$events"; return 0 ;;
      exit7) printf 'profile_exit\n' >>"$events"; return 7 ;;
      timeout) printf 'profile_timeout\n' >>"$events"; return 124 ;;
      hang)
        trap 'printf "profile_term\\n" >>"$events"; exit 143' INT TERM
        while true; do sleep 0.05; done
        ;;
      *) return 99 ;;
    esac
  fi
  return 98
}

python3() {
  printf 'journal\n' >>"$events"
  sleep "${FAKE_JOURNAL_DELAY:-0}"
  return "${FAKE_JOURNAL_STATUS:-0}"
}
"""
        + production_block
        + "\nprintf 'api\\n' >>\"$events\"\n",
        encoding="utf-8",
    )
    harness.chmod(0o700)
    return harness, events


def _run_prewarm_trap_harness(
    tmp_path: Path, **environment_updates: str
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    harness, events = _prewarm_trap_harness(tmp_path)
    environment = os.environ.copy()
    environment["EVENTS"] = str(events)
    environment.update(environment_updates)
    result = subprocess.run(
        ["bash", str(harness)],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return result, events.read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize(
    ("environment_updates", "expected_status", "expected_events"),
    (
        (
            {"FAKE_PROFILE_MODE": "exit0"},
            0,
            ["capture", "profile_start", "profile_exit", "settle", "journal", "api"],
        ),
        (
            {"FAKE_PROFILE_MODE": "exit0", "FAKE_PREWARM_ONLY": "1"},
            0,
            ["capture", "profile_start", "profile_exit", "settle", "journal"],
        ),
        (
            {"FAKE_PROFILE_MODE": "exit7"},
            7,
            ["capture", "profile_start", "profile_exit", "settle", "journal"],
        ),
        (
            {"FAKE_PROFILE_MODE": "timeout"},
            124,
            ["capture", "profile_start", "profile_timeout", "settle", "journal"],
        ),
        (
            {"FAKE_PROFILE_MODE": "exit0", "FAKE_SETTLE_STATUS": "9"},
            2,
            ["capture", "profile_start", "profile_exit", "settle", "journal"],
        ),
        (
            {"FAKE_PROFILE_MODE": "exit0", "FAKE_JOURNAL_STATUS": "11"},
            2,
            ["capture", "profile_start", "profile_exit", "settle", "journal"],
        ),
    ),
)
def test_prewarm_cleanup_state_machine_status_and_order(
    tmp_path: Path,
    environment_updates: dict[str, str],
    expected_status: int,
    expected_events: list[str],
) -> None:
    result, events = _run_prewarm_trap_harness(tmp_path, **environment_updates)

    assert result.returncode == expected_status, result.stderr
    assert events == expected_events
    assert events.count("settle") == 1
    assert events.count("journal") == 1


@pytest.mark.parametrize(
    ("signal_number", "expected_status"),
    ((signal.SIGINT, 130), (signal.SIGTERM, 143)),
)
def test_launcher_signal_reaps_supervisor_then_settles_and_checks_journal_once(
    tmp_path: Path, signal_number: signal.Signals, expected_status: int
) -> None:
    harness, events_path = _prewarm_trap_harness(tmp_path)
    environment = os.environ.copy()
    environment.update(
        {
            "EVENTS": str(events_path),
            "FAKE_PROFILE_MODE": "hang",
            "FAKE_SETTLE_DELAY": "0.2",
        }
    )
    process = subprocess.Popen(
        ["bash", str(harness)],
        cwd=_REPO,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if events_path.exists() and "profile_start" in events_path.read_text(
            encoding="utf-8"
        ):
            break
        time.sleep(0.01)
    else:
        process.kill()
        process.wait(timeout=5)
        pytest.fail("profile harness never started")

    os.kill(process.pid, signal_number)
    time.sleep(0.05)
    os.kill(process.pid, signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=10)
    events = events_path.read_text(encoding="utf-8").splitlines()

    assert process.returncode == expected_status, (stdout, stderr)
    assert events == ["capture", "profile_start", "profile_term", "settle", "journal"]
    assert events.count("settle") == 1
    assert events.count("journal") == 1
    assert "api" not in events


@pytest.mark.parametrize(
    ("wait_for_event", "delay_environment"),
    (
        ("settle", {"FAKE_SETTLE_DELAY": "0.3"}),
        ("journal", {"FAKE_JOURNAL_DELAY": "0.3"}),
    ),
)
@pytest.mark.parametrize(
    ("signal_number", "expected_status"),
    ((signal.SIGINT, 130), (signal.SIGTERM, 143)),
)
def test_first_signal_during_cleanup_is_deferred_but_never_discarded(
    tmp_path: Path,
    wait_for_event: str,
    delay_environment: dict[str, str],
    signal_number: signal.Signals,
    expected_status: int,
) -> None:
    harness, events_path = _prewarm_trap_harness(tmp_path)
    environment = os.environ.copy()
    environment.update(
        {
            "EVENTS": str(events_path),
            "FAKE_PROFILE_MODE": "exit0",
            **delay_environment,
        }
    )
    process = subprocess.Popen(
        ["bash", str(harness)],
        cwd=_REPO,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if events_path.exists() and wait_for_event in events_path.read_text(
            encoding="utf-8"
        ).splitlines():
            break
        time.sleep(0.01)
    else:
        process.kill()
        process.wait(timeout=5)
        pytest.fail(f"cleanup harness never reached {wait_for_event}")

    os.kill(process.pid, signal_number)
    stdout, stderr = process.communicate(timeout=10)
    events = events_path.read_text(encoding="utf-8").splitlines()

    assert process.returncode == expected_status, (stdout, stderr)
    assert events == ["capture", "profile_start", "profile_exit", "settle", "journal"]
    assert events.count("settle") == 1
    assert events.count("journal") == 1
    assert "api" not in events


def test_helper_has_no_jax_rocm_graph_or_configurable_fake_root_path() -> None:
    source = _HELPER_PATH.read_text(encoding="utf-8")

    assert "import jax" not in source
    assert "hipGraph" not in source
    assert "cuda.graph" not in source
    assert "--drm-root" not in source
    assert "--dev-root" not in source
    assert "XLA_FLAGS" not in source
