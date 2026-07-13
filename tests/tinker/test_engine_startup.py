from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from skyrl.tinker.engine_startup import (
    ProcessIdentity,
    ProcessIdentityError,
    identities_from_launch_record,
    new_engine_launch_id,
    parse_linux_process_stat,
    process_identity_is_live,
    read_process_identity,
    validate_engine_claim_record,
    validate_engine_launch_id,
    validate_initializing_engine_launch,
    validate_ready_engine_launch,
    validate_runtime_attestation_handoff,
)

_BOOT_ID = "11111111-2222-3333-4444-555555555555"
_OTHER_BOOT_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
_LAUNCH_ID = "0123456789abcdef0123456789abcdef"


def _process_stat(
    pid: int,
    *,
    name: str = "skyrl engine",
    state: str = "S",
    start_ticks: int = 123_456,
) -> str:
    # The tokens after the command begin at Linux proc-stat field 3. There are
    # 18 fields between state (field 3) and starttime (field 22).
    fields = [state, *("0" for _ in range(18)), str(start_ticks)]
    return f"{pid} ({name}) {' '.join(fields)}\n"


def _write_proc_identity(
    proc_root: Path,
    *,
    pid: int,
    start_ticks: int,
    state: str = "S",
    boot_id: str = _BOOT_ID,
) -> None:
    boot_id_path = proc_root / "sys/kernel/random/boot_id"
    boot_id_path.parent.mkdir(parents=True, exist_ok=True)
    boot_id_path.write_text(f"{boot_id}\n", encoding="ascii")
    stat_path = proc_root / str(pid) / "stat"
    stat_path.parent.mkdir(parents=True, exist_ok=True)
    stat_path.write_text(
        _process_stat(
            pid,
            name="engine worker) pool) with spaces",
            state=state,
            start_ticks=start_ticks,
        ),
        encoding="utf-8",
    )


def _source_attestation(role: str = "api") -> dict[str, Any]:
    module_name = "api.py" if role == "api" else "engine.py"
    return {
        "status": "passed",
        "role": role,
        "git_head": "a" * 40,
        "git_tree": "b" * 40,
        "source_archive_path": "/private/source.tar",
        "source_archive_sha256": "c" * 64,
        "source_file_count": 1_090,
        "source_total_bytes": 35_333_482,
        "full_head_tree_validated": True,
        "source_root": "/private/source",
        "repo_root": "/repo",
        "working_directory": "/private/source",
        "module_origin": f"/private/source/skyrl/tinker/{module_name}",
        "package_origin": "/private/source/skyrl/__init__.py",
        "uv_executable": "/home/user/.local/bin/uv",
        "uv_sha256": "d" * 64,
        "launch_lock": _launch_lock_attestation(),
        "jax_compilation_cache": "/private/jax-cache",
        "memory_mode": "growth",
        "xla_flags": "--xla_gpu_enable_command_buffer=",
        "jax_enable_pgle": "false",
        "jax_compilation_cache_expect_pgle": "false",
        "pallas_attention": "1",
        "startup_cache_attestation": {"status": "not_required"},
        "dont_write_bytecode": True,
    }


def _launch_lock_attestation() -> dict[str, Any]:
    return {
        "status": "passed",
        "descriptor": 10,
        "path": "/run/user/1000/skyrl-qwen35-rocm-1000",
        "inheritable": True,
        "exclusive_lock_observed": True,
    }


def _required_cache_claim() -> dict[str, Any]:
    return {
        "status": "required-v1",
        "schema_name": "skyrl.qwen35.persistent-cache-attestation",
        "schema_version": 1,
        "seed": {"bucket": 64},
        "prewarm_audit": {"sha256": "e" * 64},
        "prewarm_handoff": {"sha256": "f" * 64},
    }


def _claim_record(**overrides: Any) -> SimpleNamespace:
    values = {
        "launch_id": _LAUNCH_ID,
        "backend": "jax",
        "status": "starting",
        "boot_id": _BOOT_ID,
        "api_pid": 101,
        "api_start_ticks": 1_001,
        "engine_launcher_pid": 202,
        "engine_launcher_start_ticks": 2_002,
        "engine_pid": None,
        "engine_start_ticks": None,
        "api_source_attestation": _source_attestation(),
        "api_launch_lock_attestation": _launch_lock_attestation(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _ready_record(**overrides: Any) -> SimpleNamespace:
    values = vars(_claim_record()).copy()
    engine_source = _source_attestation("engine")
    launch_lock = _launch_lock_attestation()
    values.update(
        {
            "status": "ready",
            "engine_pid": 303,
            "engine_start_ticks": 3_003,
            "engine_source_attestation": engine_source,
            "engine_launch_lock_attestation": launch_lock,
            "runtime_handoff_attestation": validate_runtime_attestation_handoff(
                values["api_source_attestation"],
                engine_source,
                values["api_launch_lock_attestation"],
                launch_lock,
            ),
            "cache_evidence_status": "not_required",
            "cache_evidence": {},
            "heartbeat_at": datetime.now(timezone.utc),
            "heartbeat_monotonic_ns": time.monotonic_ns(),
            "heartbeat_sequence": 1,
            "ready_at": datetime.now(timezone.utc),
        }
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _identity(
    pid: int,
    start_ticks: int,
    *,
    boot_id: str = _BOOT_ID,
    state: str = "S",
) -> ProcessIdentity:
    return ProcessIdentity(
        pid=pid,
        start_ticks=start_ticks,
        boot_id=boot_id,
        state=state,
    )


def test_parse_linux_process_stat_handles_spaces_and_right_parentheses() -> None:
    raw = _process_stat(
        4242,
        name="engine worker) pool) with spaces",
        state="R",
        start_ticks=987_654_321,
    )

    assert parse_linux_process_stat(raw, 4242) == ("R", 987_654_321)


@pytest.mark.parametrize(
    ("raw", "expected_pid", "message"),
    [
        ("12 malformed", 12, "malformed"),
        (_process_stat(12), 13, "PID mismatch"),
        ("abc (worker) S " + "0 " * 19, 12, "PID is malformed"),
        ("12 (worker) Q " + "0 " * 19, 12, "fields are truncated"),
        ("12 (worker) S " + "0 " * 19, 12, "positive"),
    ],
)
def test_parse_linux_process_stat_rejects_malformed_records(
    raw: str, expected_pid: int, message: str
) -> None:
    with pytest.raises(ProcessIdentityError, match=message):
        parse_linux_process_stat(raw, expected_pid)


def test_read_process_identity_uses_injected_proc_root(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    _write_proc_identity(proc_root, pid=77, start_ticks=8_888, state="I")

    assert read_process_identity(77, proc_root=proc_root) == ProcessIdentity(
        pid=77,
        start_ticks=8_888,
        boot_id=_BOOT_ID,
        state="I",
    )


def test_process_identity_liveness_rejects_dead_reused_and_other_boot(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path / "proc"
    expected = _identity(77, 8_888)

    _write_proc_identity(proc_root, pid=77, start_ticks=8_888)
    assert process_identity_is_live(expected, proc_root=proc_root)

    _write_proc_identity(proc_root, pid=77, start_ticks=8_888, state="Z")
    assert not process_identity_is_live(expected, proc_root=proc_root)

    _write_proc_identity(proc_root, pid=77, start_ticks=8_888, state="T")
    assert not process_identity_is_live(expected, proc_root=proc_root)

    _write_proc_identity(proc_root, pid=77, start_ticks=9_999)
    assert not process_identity_is_live(expected, proc_root=proc_root)

    _write_proc_identity(
        proc_root,
        pid=77,
        start_ticks=8_888,
        boot_id=_OTHER_BOOT_ID,
    )
    assert not process_identity_is_live(expected, proc_root=proc_root)

    (proc_root / "77/stat").unlink()
    assert not process_identity_is_live(expected, proc_root=proc_root)


@pytest.mark.parametrize(
    "launch_id",
    [
        "",
        "0" * 31,
        "0" * 33,
        "ABCDEF0123456789abcdef0123456789",
        "g" * 32,
        "01234567-89abcdef-0123456789abcdef",
    ],
)
def test_validate_engine_launch_id_rejects_noncanonical_values(
    launch_id: str,
) -> None:
    with pytest.raises(ValueError, match="32 lowercase hexadecimal digits"):
        validate_engine_launch_id(launch_id)


def test_new_engine_launch_id_is_canonical_and_unique() -> None:
    launch_ids = {new_engine_launch_id() for _ in range(32)}

    assert len(launch_ids) == 32
    assert all(
        validate_engine_launch_id(launch_id) == launch_id for launch_id in launch_ids
    )


def test_runtime_attestation_handoff_accepts_exact_shared_invariants() -> None:
    source = _source_attestation()
    launch_lock = _launch_lock_attestation()

    assert validate_runtime_attestation_handoff(
        source,
        _source_attestation("engine"),
        launch_lock,
        dict(launch_lock),
    ) == {
        "status": "passed",
        "source_status": "passed",
        "git_head": "a" * 40,
        "git_tree": "b" * 40,
        "source_root": "/private/source",
        "jax_compilation_cache": "/private/jax-cache",
        "startup_cache_attestation": {"status": "not_required"},
        "launch_lock": launch_lock,
    }


def test_runtime_attestation_handoff_rejects_source_or_lock_mismatch() -> None:
    source = _source_attestation()
    mismatched_source = _source_attestation("engine")
    mismatched_source["xla_flags"] = "--xla_gpu_enable_command_buffer=fusion"

    with pytest.raises(RuntimeError, match="runtime source attestation mismatch"):
        validate_runtime_attestation_handoff(
            source,
            mismatched_source,
            _launch_lock_attestation(),
            _launch_lock_attestation(),
        )

    mismatched_lock = _launch_lock_attestation()
    mismatched_lock["descriptor"] = 11
    with pytest.raises(RuntimeError, match="launch-lock attestations do not match"):
        validate_runtime_attestation_handoff(
            source,
            _source_attestation("engine"),
            _launch_lock_attestation(),
            mismatched_lock,
        )


def test_runtime_attestation_handoff_rejects_unknown_shared_field_drift() -> None:
    engine_source = _source_attestation("engine")
    engine_source["new_safety_policy"] = "relaxed"

    with pytest.raises(RuntimeError, match="runtime source attestation mismatch"):
        validate_runtime_attestation_handoff(
            _source_attestation(),
            engine_source,
            _launch_lock_attestation(),
            _launch_lock_attestation(),
        )


def test_runtime_attestation_handoff_binds_required_cache_claim() -> None:
    api_source = _source_attestation()
    engine_source = _source_attestation("engine")
    claim = _required_cache_claim()
    api_source["startup_cache_attestation"] = claim
    engine_source["startup_cache_attestation"] = dict(claim)

    handoff = validate_runtime_attestation_handoff(
        api_source,
        engine_source,
        _launch_lock_attestation(),
        _launch_lock_attestation(),
    )

    assert handoff["startup_cache_attestation"] == claim


def test_runtime_attestation_handoff_rejects_cache_claim_drift() -> None:
    api_source = _source_attestation()
    engine_source = _source_attestation("engine")
    api_source["startup_cache_attestation"] = _required_cache_claim()
    engine_source["startup_cache_attestation"] = _required_cache_claim()
    engine_source["startup_cache_attestation"]["prewarm_audit"] = {"sha256": "0" * 64}

    with pytest.raises(RuntimeError, match="runtime source attestation mismatch"):
        validate_runtime_attestation_handoff(
            api_source,
            engine_source,
            _launch_lock_attestation(),
            _launch_lock_attestation(),
        )


def test_runtime_attestation_handoff_distinguishes_generic_launches() -> None:
    assert (
        validate_runtime_attestation_handoff(
            {"status": "not_required", "role": "api"},
            {"status": "not_required", "role": "engine"},
            {"status": "not_required"},
            {"status": "not_required"},
        )["status"]
        == "matched_unattested"
    )

    assert (
        validate_runtime_attestation_handoff(
            {"status": "not_required", "role": "api"},
            {"status": "not_required", "role": "engine"},
            _launch_lock_attestation(),
            _launch_lock_attestation(),
        )["status"]
        == "matched_lock_only"
    )


@pytest.mark.parametrize("direct_exec", [False, True])
def test_validate_engine_claim_accepts_wrapper_child_and_direct_exec(
    direct_exec: bool,
) -> None:
    record = _claim_record()
    api_identity = _identity(101, 1_001)
    launcher_identity = _identity(202, 2_002)
    engine_identity = launcher_identity if direct_exec else _identity(303, 3_003)
    parent_identity = api_identity if direct_exec else launcher_identity
    live_identities: list[ProcessIdentity] = []

    def identity_is_live(identity: ProcessIdentity) -> bool:
        live_identities.append(identity)
        return True

    handoff = validate_engine_claim_record(
        record,
        launch_id=_LAUNCH_ID,
        backend="jax",
        engine_identity=engine_identity,
        parent_identity=parent_identity,
        engine_source_attestation=_source_attestation("engine"),
        engine_lock_attestation=_launch_lock_attestation(),
        identity_is_live=identity_is_live,
        process_group_reader=lambda _pid: 202,
    )

    assert handoff["status"] == "passed"
    assert [(identity.pid, identity.start_ticks) for identity in live_identities] == [
        (101, 1_001),
        (202, 2_002),
    ]


@pytest.mark.parametrize(
    ("record_overrides", "message"),
    [
        ({"status": "initializing"}, "not in STARTING state"),
        ({"backend": "fsdp"}, "backend does not match"),
        ({"boot_id": _OTHER_BOOT_ID}, "different boots"),
        ({"engine_launcher_pid": None}, "has not bound"),
        ({"engine_pid": 303}, "already contains an engine identity"),
        ({"api_source_attestation": {}}, "runtime attestations are absent"),
    ],
)
def test_validate_engine_claim_rejects_invalid_starting_rows(
    record_overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        validate_engine_claim_record(
            _claim_record(**record_overrides),
            launch_id=_LAUNCH_ID,
            backend="jax",
            engine_identity=_identity(303, 3_003),
            parent_identity=_identity(202, 2_002),
            engine_source_attestation=_source_attestation("engine"),
            engine_lock_attestation=_launch_lock_attestation(),
            identity_is_live=lambda _identity: True,
            process_group_reader=lambda _pid: 202,
        )


def test_validate_engine_claim_rejects_unrelated_or_stale_processes() -> None:
    with pytest.raises(RuntimeError, match="neither the bound launcher"):
        validate_engine_claim_record(
            _claim_record(),
            launch_id=_LAUNCH_ID,
            backend="jax",
            engine_identity=_identity(303, 3_003),
            parent_identity=_identity(404, 4_004),
            engine_source_attestation=_source_attestation("engine"),
            engine_lock_attestation=_launch_lock_attestation(),
            identity_is_live=lambda _identity: True,
            process_group_reader=lambda _pid: 202,
        )

    with pytest.raises(RuntimeError, match="parent is not the recorded API"):
        validate_engine_claim_record(
            _claim_record(),
            launch_id=_LAUNCH_ID,
            backend="jax",
            engine_identity=_identity(202, 2_002),
            parent_identity=_identity(404, 4_004),
            engine_source_attestation=_source_attestation("engine"),
            engine_lock_attestation=_launch_lock_attestation(),
            identity_is_live=lambda _identity: True,
            process_group_reader=lambda _pid: 202,
        )

    with pytest.raises(RuntimeError, match="no longer live"):
        validate_engine_claim_record(
            _claim_record(),
            launch_id=_LAUNCH_ID,
            backend="jax",
            engine_identity=_identity(303, 3_003),
            parent_identity=_identity(202, 2_002),
            engine_source_attestation=_source_attestation("engine"),
            engine_lock_attestation=_launch_lock_attestation(),
            identity_is_live=lambda identity: identity.pid != 101,
            process_group_reader=lambda _pid: 202,
        )


def test_validate_ready_launch_accepts_exact_live_launch() -> None:
    record = _ready_record()
    observed: list[ProcessIdentity] = []

    def identity_is_live(identity: ProcessIdentity) -> bool:
        observed.append(identity)
        return True

    identities = validate_ready_engine_launch(
        record,
        launch_id=_LAUNCH_ID,
        backend="jax",
        api_identity=_identity(101, 1_001),
        launcher_identity=_identity(202, 2_002),
        expected_api_source_attestation=_source_attestation(),
        expected_api_lock_attestation=_launch_lock_attestation(),
        identity_is_live=identity_is_live,
        process_group_reader=lambda _pid: 202,
    )

    assert identities == {
        "api": _identity(101, 1_001, state="?"),
        "engine_launcher": _identity(202, 2_002, state="?"),
        "engine": _identity(303, 3_003, state="?"),
    }
    assert observed == list(identities.values())


def test_validate_ready_launch_accepts_only_claim_bound_required_cache_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rocm import qwen35_cache_attestation

    api_source = _source_attestation()
    engine_source = _source_attestation("engine")
    claim = _required_cache_claim()
    api_source["startup_cache_attestation"] = claim
    engine_source["startup_cache_attestation"] = dict(claim)
    launch_lock = _launch_lock_attestation()
    handoff = validate_runtime_attestation_handoff(
        api_source, engine_source, launch_lock, launch_lock
    )
    evidence = {"proof": "exact"}
    observed: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def validate(claim_arg: Any, evidence_arg: Any) -> dict[str, Any]:
        observed.append((dict(claim_arg), dict(evidence_arg)))
        return dict(evidence_arg)

    monkeypatch.setattr(
        qwen35_cache_attestation, "validate_runtime_cache_evidence", validate
    )
    record = _ready_record(
        api_source_attestation=api_source,
        engine_source_attestation=engine_source,
        runtime_handoff_attestation=handoff,
        cache_evidence_status=qwen35_cache_attestation.RUNTIME_HIT_KIND,
        cache_evidence=evidence,
    )

    validate_ready_engine_launch(
        record,
        launch_id=_LAUNCH_ID,
        backend="jax",
        api_identity=_identity(101, 1_001),
        launcher_identity=_identity(202, 2_002),
        expected_api_source_attestation=api_source,
        expected_api_lock_attestation=launch_lock,
        identity_is_live=lambda _identity: True,
        process_group_reader=lambda _pid: 202,
    )

    assert observed == [(claim, evidence)]


def test_validate_ready_launch_rejects_missing_required_cache_evidence() -> None:
    api_source = _source_attestation()
    engine_source = _source_attestation("engine")
    claim = _required_cache_claim()
    api_source["startup_cache_attestation"] = claim
    engine_source["startup_cache_attestation"] = dict(claim)
    launch_lock = _launch_lock_attestation()
    handoff = validate_runtime_attestation_handoff(
        api_source, engine_source, launch_lock, launch_lock
    )

    with pytest.raises(RuntimeError, match="required cache evidence is absent"):
        validate_ready_engine_launch(
            _ready_record(
                api_source_attestation=api_source,
                engine_source_attestation=engine_source,
                runtime_handoff_attestation=handoff,
            ),
            launch_id=_LAUNCH_ID,
            backend="jax",
            api_identity=_identity(101, 1_001),
            launcher_identity=_identity(202, 2_002),
            expected_api_source_attestation=api_source,
            expected_api_lock_attestation=launch_lock,
            identity_is_live=lambda _identity: True,
            process_group_reader=lambda _pid: 202,
        )


def test_validate_initializing_launch_revalidates_claim_before_ready() -> None:
    record = _ready_record(status="initializing")

    handoff = validate_initializing_engine_launch(
        record,
        launch_id=_LAUNCH_ID,
        backend="jax",
        api_identity=_identity(101, 1_001, state="?"),
        launcher_identity=_identity(202, 2_002, state="?"),
        engine_identity=_identity(303, 3_003, state="?"),
        engine_source_attestation=_source_attestation("engine"),
        engine_lock_attestation=_launch_lock_attestation(),
        identity_is_live=lambda _identity: True,
        process_group_reader=lambda _pid: 202,
    )

    assert handoff == record.runtime_handoff_attestation


@pytest.mark.parametrize(
    ("record_overrides", "message"),
    [
        ({"status": "ready"}, "not INITIALIZING"),
        ({"backend": "fsdp"}, "backend changed"),
        ({"engine_start_ticks": 3_004}, "process identities changed"),
        ({"engine_source_attestation": {}}, "attestations changed"),
        (
            {"runtime_handoff_attestation": {"status": "passed"}},
            "runtime handoff changed",
        ),
    ],
)
def test_validate_initializing_launch_rejects_changed_claim(
    record_overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        validate_initializing_engine_launch(
            _ready_record(**{"status": "initializing", **record_overrides}),
            launch_id=_LAUNCH_ID,
            backend="jax",
            api_identity=_identity(101, 1_001, state="?"),
            launcher_identity=_identity(202, 2_002, state="?"),
            engine_identity=_identity(303, 3_003, state="?"),
            engine_source_attestation=_source_attestation("engine"),
            engine_lock_attestation=_launch_lock_attestation(),
            identity_is_live=lambda _identity: True,
            process_group_reader=lambda _pid: 202,
        )


def test_validate_ready_launch_rejects_unvalidated_strict_cache_hit_evidence() -> None:
    with pytest.raises(RuntimeError, match="cache evidence is incomplete"):
        validate_ready_engine_launch(
            _ready_record(cache_evidence_status="strict_public_monitoring_hit"),
            launch_id=_LAUNCH_ID,
            backend="jax",
            api_identity=_identity(101, 1_001),
            launcher_identity=_identity(202, 2_002),
            expected_api_source_attestation=_source_attestation(),
            expected_api_lock_attestation=_launch_lock_attestation(),
            identity_is_live=lambda _identity: True,
            process_group_reader=lambda _pid: 202,
        )


@pytest.mark.parametrize(
    ("record_overrides", "message"),
    [
        ({"status": "initializing"}, "not READY"),
        ({"backend": "fsdp"}, "backend does not match"),
        ({"boot_id": _OTHER_BOOT_ID}, "different boot"),
        ({"api_start_ticks": 1_002}, "different API process"),
        ({"engine_launcher_start_ticks": 2_003}, "different launcher process"),
        ({"engine_source_attestation": {}}, "attestations are incomplete"),
        (
            {"heartbeat_monotonic_ns": time.monotonic_ns() - 10_000_000_000},
            "heartbeat is stale",
        ),
        ({"heartbeat_sequence": 0}, "heartbeat is absent"),
        ({"ready_at": None}, "publication timestamp is absent"),
        (
            {"runtime_handoff_attestation": {"status": "failed"}},
            "handoff attestation is inconsistent",
        ),
        ({"cache_evidence_status": "miss"}, "cache evidence is incomplete"),
    ],
)
def test_validate_ready_launch_rejects_nonready_or_mismatched_rows(
    record_overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        validate_ready_engine_launch(
            _ready_record(**record_overrides),
            launch_id=_LAUNCH_ID,
            backend="jax",
            api_identity=_identity(101, 1_001),
            launcher_identity=_identity(202, 2_002),
            expected_api_source_attestation=_source_attestation(),
            expected_api_lock_attestation=_launch_lock_attestation(),
            identity_is_live=lambda _identity: True,
            process_group_reader=lambda _pid: 202,
        )


def test_validate_ready_launch_rejects_a_stale_engine_identity() -> None:
    with pytest.raises(RuntimeError, match="process identities are stale"):
        validate_ready_engine_launch(
            _ready_record(),
            launch_id=_LAUNCH_ID,
            backend="jax",
            api_identity=_identity(101, 1_001),
            launcher_identity=_identity(202, 2_002),
            expected_api_source_attestation=_source_attestation(),
            expected_api_lock_attestation=_launch_lock_attestation(),
            identity_is_live=lambda identity: identity.pid != 303,
            process_group_reader=lambda _pid: 202,
        )


def test_validate_ready_launch_rejects_escaped_process_group() -> None:
    with pytest.raises(RuntimeError, match="escaped its dedicated process group"):
        validate_ready_engine_launch(
            _ready_record(),
            launch_id=_LAUNCH_ID,
            backend="jax",
            api_identity=_identity(101, 1_001),
            launcher_identity=_identity(202, 2_002),
            expected_api_source_attestation=_source_attestation(),
            expected_api_lock_attestation=_launch_lock_attestation(),
            identity_is_live=lambda _identity: True,
            process_group_reader=lambda _pid: 999,
        )


def test_identities_from_launch_record_rejects_incomplete_identity() -> None:
    with pytest.raises(
        ProcessIdentityError, match="engine process identity is incomplete"
    ):
        identities_from_launch_record(_ready_record(engine_start_ticks=None))
