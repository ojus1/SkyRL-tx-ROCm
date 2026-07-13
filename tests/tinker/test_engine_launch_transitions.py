"""CPU-only tests for engine-side supervised launch transitions."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any

import pytest
from sqlmodel import Session, SQLModel, create_engine

import skyrl.tinker.engine as engine_module
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import (
    EngineLaunchDB,
    EngineLaunchStatus,
    EngineStateDB,
)
from skyrl.tinker.engine import TinkerEngine, _publish_engine_failure
from skyrl.tinker.engine_startup import (
    ProcessIdentity,
    validate_engine_claim_record,
    validate_initializing_engine_launch,
    validate_runtime_attestation_handoff,
)

_BOOT_ID = "11111111-2222-3333-4444-555555555555"
_LAUNCH_ID = "0123456789abcdef0123456789abcdef"
_OTHER_LAUNCH_ID = "fedcba9876543210fedcba9876543210"


def _identity(pid: int, start_ticks: int) -> ProcessIdentity:
    return ProcessIdentity(
        pid=pid,
        start_ticks=start_ticks,
        boot_id=_BOOT_ID,
        state="S",
    )


_API_IDENTITY = _identity(101, 1_001)
_LAUNCHER_IDENTITY = _identity(202, 2_002)
_ENGINE_IDENTITY = _identity(303, 3_003)
_API_SOURCE_ATTESTATION = {"status": "not_required", "role": "api"}
_ENGINE_SOURCE_ATTESTATION = {"status": "not_required", "role": "engine"}
_LOCK_ATTESTATION = {"status": "not_required"}
_HANDOFF_ATTESTATION = validate_runtime_attestation_handoff(
    _API_SOURCE_ATTESTATION,
    _ENGINE_SOURCE_ATTESTATION,
    _LOCK_ATTESTATION,
    _LOCK_ATTESTATION,
)


class _OneShotStop:
    def __init__(self) -> None:
        self.stopped = False

    def is_set(self) -> bool:
        return self.stopped

    def set(self) -> None:
        self.stopped = True


class _OneShotWake:
    def __init__(self, stop: _OneShotStop) -> None:
        self.stop = stop

    def wait(self, _timeout: float) -> bool:
        self.stop.set()
        return True

    def clear(self) -> None:
        return None


def _config(database_url: str, launch_id: str = _LAUNCH_ID) -> EngineConfig:
    return EngineConfig(
        base_model="test/model",
        backend="jax",
        database_url=database_url,
        startup_launch_id=launch_id,
    )


def _launch_row(
    *,
    launch_id: str = _LAUNCH_ID,
    status: EngineLaunchStatus = EngineLaunchStatus.INITIALIZING,
    engine_identity: ProcessIdentity | None = _ENGINE_IDENTITY,
) -> EngineLaunchDB:
    return EngineLaunchDB(
        launch_id=launch_id,
        backend="jax",
        status=status,
        boot_id=_BOOT_ID,
        api_pid=_API_IDENTITY.pid,
        api_start_ticks=_API_IDENTITY.start_ticks,
        engine_launcher_pid=_LAUNCHER_IDENTITY.pid,
        engine_launcher_start_ticks=_LAUNCHER_IDENTITY.start_ticks,
        engine_pid=None if engine_identity is None else engine_identity.pid,
        engine_start_ticks=(
            None if engine_identity is None else engine_identity.start_ticks
        ),
        api_source_attestation=_API_SOURCE_ATTESTATION,
        api_launch_lock_attestation=_LOCK_ATTESTATION,
        engine_source_attestation=_ENGINE_SOURCE_ATTESTATION,
        engine_launch_lock_attestation=_LOCK_ATTESTATION,
        runtime_handoff_attestation=_HANDOFF_ATTESTATION,
        heartbeat_at=(
            None
            if status == EngineLaunchStatus.STARTING
            else datetime.now(timezone.utc)
        ),
        heartbeat_monotonic_ns=(
            None if status == EngineLaunchStatus.STARTING else time.monotonic_ns()
        ),
        heartbeat_sequence=(0 if status == EngineLaunchStatus.STARTING else 1),
    )


@pytest.fixture()
def launch_database(tmp_path: Path):
    database_url = f"sqlite:///{tmp_path / 'engine-launches.db'}"
    db_engine = create_engine(database_url)
    SQLModel.metadata.create_all(db_engine)
    try:
        yield database_url, db_engine
    finally:
        db_engine.dispose()


@pytest.fixture()
def initializing_engine(launch_database, monkeypatch: pytest.MonkeyPatch):
    database_url, db_engine = launch_database
    with Session(db_engine) as session:
        session.add(_launch_row())
        session.commit()

    engine = object.__new__(TinkerEngine)
    engine.config = _config(database_url)
    engine.db_engine = db_engine
    engine.api_launch_identity = _API_IDENTITY
    engine.engine_launcher_identity = _LAUNCHER_IDENTITY
    engine.engine_launch_identity = _ENGINE_IDENTITY
    engine._engine_watchdog_db_engine = db_engine
    engine.runtime_source_attestation = _ENGINE_SOURCE_ATTESTATION
    engine.runtime_launch_lock_attestation = _LOCK_ATTESTATION
    engine.runtime_handoff_attestation = _HANDOFF_ATTESTATION
    engine.cache_evidence_status = "not_required"
    engine.cache_evidence = {}

    def validate_with_stubbed_liveness(*args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs["identity_is_live"] = lambda _identity: True
        kwargs["process_group_reader"] = lambda _pid: _LAUNCHER_IDENTITY.pid
        return validate_initializing_engine_launch(*args, **kwargs)

    monkeypatch.setattr(
        engine_module,
        "validate_initializing_engine_launch",
        validate_with_stubbed_liveness,
    )
    return engine


def test_claim_engine_launch_transitions_starting_and_clears_stale_proxy(
    launch_database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url, db_engine = launch_database
    with Session(db_engine) as session:
        session.add(
            _launch_row(
                status=EngineLaunchStatus.STARTING,
                engine_identity=None,
            )
        )
        session.add(
            EngineStateDB(
                singleton_id=1,
                inference_proxy_url="http://stale-engine.invalid",
            )
        )
        session.commit()

    engine = object.__new__(TinkerEngine)
    engine.config = _config(database_url)
    engine.db_engine = db_engine
    engine.runtime_source_attestation = _ENGINE_SOURCE_ATTESTATION
    engine.runtime_launch_lock_attestation = _LOCK_ATTESTATION

    monkeypatch.setattr(engine_module.os, "getppid", lambda: _LAUNCHER_IDENTITY.pid)
    monkeypatch.setattr(
        engine_module,
        "read_process_identity",
        lambda pid=None: _ENGINE_IDENTITY if pid is None else _LAUNCHER_IDENTITY,
    )

    def validate_claim_with_stubbed_os(*args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs["identity_is_live"] = lambda _identity: True
        kwargs["process_group_reader"] = lambda _pid: _LAUNCHER_IDENTITY.pid
        return validate_engine_claim_record(*args, **kwargs)

    monkeypatch.setattr(
        engine_module,
        "validate_engine_claim_record",
        validate_claim_with_stubbed_os,
    )

    engine._claim_engine_launch()

    with Session(db_engine) as session:
        record = session.get(EngineLaunchDB, _LAUNCH_ID)
        proxy_state = session.get(EngineStateDB, 1)
        assert record is not None
        assert record.status == EngineLaunchStatus.INITIALIZING
        assert (record.engine_pid, record.engine_start_ticks) == (
            _ENGINE_IDENTITY.pid,
            _ENGINE_IDENTITY.start_ticks,
        )
        assert record.engine_source_attestation == _ENGINE_SOURCE_ATTESTATION
        assert record.engine_launch_lock_attestation == _LOCK_ATTESTATION
        assert record.runtime_handoff_attestation == _HANDOFF_ATTESTATION
        assert proxy_state is not None
        assert proxy_state.inference_proxy_url is None

    assert engine.engine_launch_identity == _ENGINE_IDENTITY
    assert engine.engine_launcher_identity == ProcessIdentity(
        pid=_LAUNCHER_IDENTITY.pid,
        start_ticks=_LAUNCHER_IDENTITY.start_ticks,
        boot_id=_BOOT_ID,
        state="?",
    )
    assert engine.api_launch_identity == ProcessIdentity(
        pid=_API_IDENTITY.pid,
        start_ticks=_API_IDENTITY.start_ticks,
        boot_id=_BOOT_ID,
        state="?",
    )
    assert engine.runtime_handoff_attestation == _HANDOFF_ATTESTATION


def test_publish_ready_transitions_only_exact_initializing_launch(
    initializing_engine: TinkerEngine,
) -> None:
    initializing_engine._publish_engine_ready()

    with Session(initializing_engine.db_engine) as session:
        record = session.get(EngineLaunchDB, _LAUNCH_ID)
        assert record is not None
        assert record.status == EngineLaunchStatus.READY
        assert record.ready_at is not None
        assert record.cache_evidence_status == "not_required"
        assert record.cache_evidence == {}
        assert record.error_message is None


def test_publish_ready_rejects_cache_evidence_without_matching_requirement(
    initializing_engine: TinkerEngine,
) -> None:
    initializing_engine.cache_evidence_status = "strict_aot_t64_persistent_cache_hit_v1"
    initializing_engine.cache_evidence = {"unbound": True}

    with pytest.raises(RuntimeError, match="opt-out publication is inconsistent"):
        initializing_engine._publish_engine_ready()

    with Session(initializing_engine.db_engine) as session:
        record = session.get(EngineLaunchDB, _LAUNCH_ID)
        assert record is not None
        assert record.status == EngineLaunchStatus.INITIALIZING
        assert record.ready_at is None


@pytest.mark.parametrize(
    "status",
    [
        EngineLaunchStatus.STARTING,
        EngineLaunchStatus.READY,
        EngineLaunchStatus.FAILED,
        EngineLaunchStatus.STOPPED,
    ],
)
def test_publish_ready_rejects_every_noninitializing_state(
    initializing_engine: TinkerEngine,
    status: EngineLaunchStatus,
) -> None:
    with Session(initializing_engine.db_engine) as session:
        record = session.get(EngineLaunchDB, _LAUNCH_ID)
        assert record is not None
        record.status = status
        session.add(record)
        session.commit()

    with pytest.raises(RuntimeError, match="not INITIALIZING"):
        initializing_engine._publish_engine_ready()

    with Session(initializing_engine.db_engine) as session:
        record = session.get(EngineLaunchDB, _LAUNCH_ID)
        assert record is not None
        assert record.status == status
        assert record.ready_at is None


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("api_start_ticks", _API_IDENTITY.start_ticks + 1),
        ("engine_launcher_pid", _LAUNCHER_IDENTITY.pid + 1),
        ("engine_start_ticks", _ENGINE_IDENTITY.start_ticks + 1),
    ],
)
def test_publish_ready_rejects_changed_process_identity(
    initializing_engine: TinkerEngine,
    field: str,
    replacement: int,
) -> None:
    with Session(initializing_engine.db_engine) as session:
        record = session.get(EngineLaunchDB, _LAUNCH_ID)
        assert record is not None
        setattr(record, field, replacement)
        session.add(record)
        session.commit()

    with pytest.raises(RuntimeError, match="process identities changed"):
        initializing_engine._publish_engine_ready()

    with Session(initializing_engine.db_engine) as session:
        record = session.get(EngineLaunchDB, _LAUNCH_ID)
        assert record is not None
        assert record.status == EngineLaunchStatus.INITIALIZING
        assert record.ready_at is None


@pytest.mark.parametrize(
    "initial_status",
    [EngineLaunchStatus.INITIALIZING, EngineLaunchStatus.READY],
)
def test_publish_failure_marks_exact_launch_without_touching_unrelated_row(
    launch_database,
    monkeypatch: pytest.MonkeyPatch,
    initial_status: EngineLaunchStatus,
) -> None:
    database_url, db_engine = launch_database
    with Session(db_engine) as session:
        session.add(_launch_row(status=initial_status))
        session.add(
            _launch_row(
                launch_id=_OTHER_LAUNCH_ID,
                status=EngineLaunchStatus.READY,
            )
        )
        session.commit()

    monkeypatch.setattr(
        engine_module,
        "read_process_identity",
        lambda: _ENGINE_IDENTITY,
    )

    error = RuntimeError("backend initialization failed")
    _publish_engine_failure(_config(database_url), error)

    with Session(db_engine) as session:
        failed = session.get(EngineLaunchDB, _LAUNCH_ID)
        unrelated = session.get(EngineLaunchDB, _OTHER_LAUNCH_ID)
        assert failed is not None
        assert failed.status == EngineLaunchStatus.FAILED
        assert failed.error_message == "RuntimeError: backend initialization failed"
        assert unrelated is not None
        assert unrelated.status == EngineLaunchStatus.READY
        assert unrelated.error_message is None


def test_publish_failure_rejects_changed_engine_identity(
    launch_database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url, db_engine = launch_database
    changed_identity = _identity(
        _ENGINE_IDENTITY.pid,
        _ENGINE_IDENTITY.start_ticks + 1,
    )
    with Session(db_engine) as session:
        session.add(_launch_row(engine_identity=changed_identity))
        session.commit()

    monkeypatch.setattr(
        engine_module,
        "read_process_identity",
        lambda: _ENGINE_IDENTITY,
    )

    _publish_engine_failure(_config(database_url), RuntimeError("wrong process"))

    with Session(db_engine) as session:
        record = session.get(EngineLaunchDB, _LAUNCH_ID)
        assert record is not None
        assert record.status == EngineLaunchStatus.INITIALIZING
        assert record.error_message is None


def test_engine_launch_watchdog_heartbeats_identity_bound_row(
    launch_database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url, db_engine = launch_database
    with Session(db_engine) as session:
        session.add(_launch_row())
        session.commit()

    engine = object.__new__(TinkerEngine)
    engine.config = _config(database_url)
    engine.db_engine = db_engine
    engine.api_launch_identity = _API_IDENTITY
    engine.engine_launcher_identity = _LAUNCHER_IDENTITY
    engine.engine_launch_identity = _ENGINE_IDENTITY
    engine._engine_watchdog_db_engine = db_engine
    engine._engine_watchdog_stop = _OneShotStop()
    engine._engine_watchdog_wake = _OneShotWake(engine._engine_watchdog_stop)
    engine._engine_watchdog_probe_request = Event()
    engine._engine_watchdog_probe_ack = Event()
    engine._engine_watchdog_probe_request.set()
    engine_module._ENGINE_FAILURE_IN_PROGRESS.clear()
    monkeypatch.setattr(
        engine_module,
        "process_identity_is_live",
        lambda _identity: True,
    )
    monkeypatch.setattr(
        engine_module.os,
        "getpgid",
        lambda _pid: _LAUNCHER_IDENTITY.pid,
    )

    engine._engine_launch_watchdog()

    with Session(db_engine) as session:
        record = session.get(EngineLaunchDB, _LAUNCH_ID)
        assert record is not None
        assert record.heartbeat_sequence == 2
        assert record.heartbeat_at is not None
        assert record.heartbeat_monotonic_ns is not None
    assert engine._engine_watchdog_probe_ack.is_set()
    assert not engine._engine_watchdog_probe_request.is_set()


def test_engine_launch_watchdog_signals_on_lost_supervisor(
    launch_database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url, db_engine = launch_database
    with Session(db_engine) as session:
        session.add(_launch_row())
        session.commit()

    engine = object.__new__(TinkerEngine)
    engine.config = _config(database_url)
    engine.db_engine = db_engine
    engine.api_launch_identity = _API_IDENTITY
    engine.engine_launcher_identity = _LAUNCHER_IDENTITY
    engine.engine_launch_identity = _ENGINE_IDENTITY
    engine._engine_watchdog_stop = Event()
    engine._engine_watchdog_wake = Event()
    engine._engine_watchdog_probe_request = Event()
    engine._engine_watchdog_probe_ack = Event()
    observed_signals: list[tuple[int, int]] = []
    engine_module._ENGINE_FAILURE_IN_PROGRESS.clear()
    monkeypatch.setattr(
        engine_module,
        "process_identity_is_live",
        lambda identity: identity != _API_IDENTITY,
    )
    monkeypatch.setattr(
        engine_module.os,
        "killpg",
        lambda process_group_id, signal_number: observed_signals.append(
            (process_group_id, signal_number)
        ),
    )
    monkeypatch.setattr(engine_module.os, "getpgrp", lambda: _LAUNCHER_IDENTITY.pid)

    engine._engine_launch_watchdog()

    assert observed_signals == [(_LAUNCHER_IDENTITY.pid, engine_module.signal.SIGKILL)]


def test_engine_launch_watchdog_requires_fresh_probe_acknowledgements(
    launch_database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url, db_engine = launch_database
    with Session(db_engine) as session:
        session.add(_launch_row())
        session.commit()

    engine = object.__new__(TinkerEngine)
    engine.config = _config(database_url)
    engine.db_engine = db_engine
    engine.api_launch_identity = _API_IDENTITY
    engine.engine_launcher_identity = _LAUNCHER_IDENTITY
    engine.engine_launch_identity = _ENGINE_IDENTITY
    engine._engine_watchdog_stop = Event()
    engine._engine_watchdog_wake = Event()
    engine._engine_watchdog_probe_request = Event()
    engine._engine_watchdog_probe_ack = Event()
    engine._engine_watchdog_thread = None
    engine._engine_watchdog_db_engine = None
    engine_module._ENGINE_FAILURE_IN_PROGRESS.clear()
    monkeypatch.setattr(
        engine_module,
        "process_identity_is_live",
        lambda _identity: True,
    )
    monkeypatch.setattr(
        engine_module.os,
        "getpgid",
        lambda _pid: _LAUNCHER_IDENTITY.pid,
    )

    try:
        engine._start_engine_launch_watchdog()
        with Session(db_engine) as session:
            after_start = session.get(EngineLaunchDB, _LAUNCH_ID)
            assert after_start is not None
            startup_sequence = after_start.heartbeat_sequence
        engine._require_engine_watchdog_acknowledgement("post-backend-test")
        with Session(db_engine) as session:
            after_probe = session.get(EngineLaunchDB, _LAUNCH_ID)
            assert after_probe is not None
            assert after_probe.heartbeat_sequence > startup_sequence
    finally:
        engine._engine_watchdog_stop.set()
        engine._engine_watchdog_wake.set()
        assert engine._engine_watchdog_thread is not None
        engine._engine_watchdog_thread.join(timeout=2)
        assert not engine._engine_watchdog_thread.is_alive()
        assert engine._engine_watchdog_db_engine is not None
        engine._engine_watchdog_db_engine.dispose()


def test_engine_launch_watchdog_retries_one_transient_database_failure(
    launch_database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url, db_engine = launch_database
    with Session(db_engine) as session:
        session.add(_launch_row())
        session.commit()

    stop = _OneShotStop()

    class RetryWake:
        def wait(self, timeout: float) -> bool:
            if timeout >= engine_module._ENGINE_WATCHDOG_INTERVAL_SECONDS:
                stop.set()
            return True

        def clear(self) -> None:
            return None

    class FailingSession:
        def __enter__(self):
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def exec(self, _statement: Any) -> None:
            raise RuntimeError("database is locked")

    engine = object.__new__(TinkerEngine)
    engine.config = _config(database_url)
    engine.db_engine = db_engine
    engine.api_launch_identity = _API_IDENTITY
    engine.engine_launcher_identity = _LAUNCHER_IDENTITY
    engine.engine_launch_identity = _ENGINE_IDENTITY
    engine._engine_watchdog_db_engine = db_engine
    engine._engine_watchdog_stop = stop
    engine._engine_watchdog_wake = RetryWake()
    engine._engine_watchdog_probe_request = Event()
    engine._engine_watchdog_probe_ack = Event()
    engine._engine_watchdog_probe_request.set()
    engine_module._ENGINE_FAILURE_IN_PROGRESS.clear()
    real_session = engine_module.Session
    session_calls = 0

    def session_factory(bind: Any):
        nonlocal session_calls
        session_calls += 1
        if session_calls == 1:
            return FailingSession()
        return real_session(bind)

    monkeypatch.setattr(engine_module, "Session", session_factory)
    monkeypatch.setattr(
        engine_module,
        "process_identity_is_live",
        lambda _identity: True,
    )
    monkeypatch.setattr(
        engine_module.os,
        "getpgid",
        lambda _pid: _LAUNCHER_IDENTITY.pid,
    )
    monkeypatch.setattr(
        engine_module.os,
        "killpg",
        lambda *_args: pytest.fail("transient DB failure killed engine group"),
    )

    engine._engine_launch_watchdog()

    assert session_calls == 2
    assert engine._engine_watchdog_probe_ack.is_set()
    with Session(db_engine) as session:
        record = session.get(EngineLaunchDB, _LAUNCH_ID)
        assert record is not None
        assert record.heartbeat_sequence == 2
