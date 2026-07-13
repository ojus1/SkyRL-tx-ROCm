"""CPU-only tests for the API/engine readiness database handoff."""

import asyncio
import signal
import sqlite3
import sys
from contextlib import suppress
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import BigInteger
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import api as tinker_api
from skyrl.tinker.api import (
    _bind_engine_launcher,
    _insert_engine_launch,
    _mark_engine_launch,
    _wait_for_engine_ready,
)
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import (
    EngineLaunchDB,
    EngineLaunchStatus,
    enable_sqlite_wal,
)
from skyrl.tinker.engine_startup import ProcessIdentity

_BOOT_ID = "54ccf56c-5f4f-4ef7-ac98-c13e0587b5b9"
_API_SOURCE = {"status": "not_required", "role": "api"}
_API_LOCK = {"status": "not_required"}
_REPO = Path(__file__).resolve().parents[2]


def _asyncio_test(function: Any) -> Any:
    """Run one async test without requiring the optional pytest plugin."""

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(function(*args, **kwargs))

    return wrapper


def _identity(
    pid: int, start_ticks: int, *, boot_id: str = _BOOT_ID
) -> ProcessIdentity:
    return ProcessIdentity(
        pid=pid,
        start_ticks=start_ticks,
        boot_id=boot_id,
        state="S",
    )


async def _new_database(path: Path) -> AsyncEngine:
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    enable_sqlite_wal(engine.sync_engine)
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    return engine


def _config(path: Path) -> EngineConfig:
    return EngineConfig(
        base_model="test/model",
        backend="jax",
        database_url=f"sqlite:///{path}",
    )


def test_process_counters_use_cross_database_64_bit_columns() -> None:
    columns = EngineLaunchDB.__table__.c
    for name in (
        "api_start_ticks",
        "engine_launcher_start_ticks",
        "engine_start_ticks",
        "heartbeat_monotonic_ns",
        "heartbeat_sequence",
    ):
        assert isinstance(columns[name].type, BigInteger)


async def _get_launch(engine: AsyncEngine, launch_id: str) -> EngineLaunchDB:
    async with AsyncSession(engine) as session:
        record = await session.get(EngineLaunchDB, launch_id)
    assert record is not None
    return record


async def _insert_starting(
    engine: AsyncEngine,
    path: Path,
    *,
    launch_id: str = "a" * 32,
    api_identity: ProcessIdentity | None = None,
) -> ProcessIdentity:
    selected_identity = api_identity or _identity(101, 1_001)
    await _insert_engine_launch(
        engine,
        launch_id=launch_id,
        engine_config=_config(path),
        api_identity=selected_identity,
        source_attestation=_API_SOURCE,
        lock_attestation=_API_LOCK,
    )
    return selected_identity


@_asyncio_test
async def test_create_all_adds_launch_table_without_altering_legacy_engine_state(
    tmp_path: Path,
) -> None:
    """A live pre-readiness database gains a new table, not legacy columns."""
    path = tmp_path / "legacy.db"
    updated_at = datetime(2025, 1, 2, tzinfo=timezone.utc).isoformat()
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE engine_state (
                singleton_id INTEGER NOT NULL PRIMARY KEY,
                inference_proxy_url VARCHAR,
                updated_at DATETIME NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO engine_state VALUES (?, ?, ?)",
            (1, "http://127.0.0.1:8123", updated_at),
        )
        connection.commit()

    engine = await _new_database(path)
    await engine.dispose()

    with sqlite3.connect(path) as connection:
        legacy_columns = [
            row[1] for row in connection.execute("PRAGMA table_info(engine_state)")
        ]
        launch_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(engine_launches)")
        }
        legacy_row = connection.execute(
            "SELECT singleton_id, inference_proxy_url, updated_at FROM engine_state"
        ).fetchone()

    assert legacy_columns == [
        "singleton_id",
        "inference_proxy_url",
        "updated_at",
    ]
    assert {
        "launch_id",
        "status",
        "api_pid",
        "engine_launcher_pid",
        "engine_pid",
        "runtime_handoff_attestation",
        "cache_evidence_status",
    } <= launch_columns
    assert legacy_row == (1, "http://127.0.0.1:8123", updated_at)


@_asyncio_test
async def test_insert_engine_launch_persists_starting_row_and_rejects_collision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "insert.db"
    engine = await _new_database(path)
    launch_id = "b" * 32
    api_identity = _identity(102, 1_002)
    try:
        await _insert_starting(
            engine,
            path,
            launch_id=launch_id,
            api_identity=api_identity,
        )
        record = await _get_launch(engine, launch_id)

        assert record.status == EngineLaunchStatus.STARTING
        assert record.backend == "jax"
        assert (record.api_pid, record.api_start_ticks, record.boot_id) == (
            api_identity.pid,
            api_identity.start_ticks,
            api_identity.boot_id,
        )
        assert record.engine_launcher_pid is None
        assert record.engine_pid is None
        assert record.api_source_attestation == _API_SOURCE
        assert record.api_launch_lock_attestation == _API_LOCK

        with pytest.raises(RuntimeError, match="engine launch ID collision"):
            await _insert_starting(
                engine,
                path,
                launch_id=launch_id,
                api_identity=_identity(999, 9_999),
            )

        record_after_collision = await _get_launch(engine, launch_id)
        assert record_after_collision.api_pid == api_identity.pid
        assert record_after_collision.api_start_ticks == api_identity.start_ticks
    finally:
        await engine.dispose()


@_asyncio_test
async def test_bind_engine_launcher_is_a_single_conditional_transition(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bind.db"
    engine = await _new_database(path)
    launch_id = "c" * 32
    try:
        api_identity = await _insert_starting(
            engine,
            path,
            launch_id=launch_id,
        )
        launcher_identity = _identity(202, 2_002)
        await _bind_engine_launcher(
            engine,
            launch_id=launch_id,
            api_identity=api_identity,
            launcher_identity=launcher_identity,
        )

        record = await _get_launch(engine, launch_id)
        assert (record.engine_launcher_pid, record.engine_launcher_start_ticks) == (
            launcher_identity.pid,
            launcher_identity.start_ticks,
        )

        with pytest.raises(RuntimeError, match="conditional binding race"):
            await _bind_engine_launcher(
                engine,
                launch_id=launch_id,
                api_identity=api_identity,
                launcher_identity=_identity(203, 2_003),
            )

        with pytest.raises(RuntimeError, match="different boots"):
            await _bind_engine_launcher(
                engine,
                launch_id=launch_id,
                api_identity=api_identity,
                launcher_identity=_identity(
                    204,
                    2_004,
                    boot_id="9b77cf06-9d11-47d9-b65f-ae3a91c04e57",
                ),
            )

        unchanged = await _get_launch(engine, launch_id)
        assert unchanged.engine_launcher_pid == launcher_identity.pid
        assert unchanged.engine_launcher_start_ticks == launcher_identity.start_ticks
    finally:
        await engine.dispose()


@_asyncio_test
async def test_mark_engine_launch_is_identity_and_state_guarded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mark.db"
    engine = await _new_database(path)
    launch_id = "d" * 32
    try:
        api_identity = await _insert_starting(
            engine,
            path,
            launch_id=launch_id,
        )

        assert not await _mark_engine_launch(
            engine,
            launch_id=launch_id,
            api_identity=_identity(999, 9_999),
            status=EngineLaunchStatus.FAILED,
            expected_statuses=(EngineLaunchStatus.STARTING,),
            error_message="wrong owner",
        )
        assert await _mark_engine_launch(
            engine,
            launch_id=launch_id,
            api_identity=api_identity,
            status=EngineLaunchStatus.FAILED,
            expected_statuses=(EngineLaunchStatus.STARTING,),
            error_message="x" * 3_000,
        )
        assert not await _mark_engine_launch(
            engine,
            launch_id=launch_id,
            api_identity=api_identity,
            status=EngineLaunchStatus.STOPPED,
            expected_statuses=(EngineLaunchStatus.STARTING,),
        )

        record = await _get_launch(engine, launch_id)
        assert record.status == EngineLaunchStatus.FAILED
        assert record.error_message == "x" * 2_048
    finally:
        await engine.dispose()


@_asyncio_test
async def test_wait_for_engine_ready_returns_validated_exact_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "ready.db"
    engine = await _new_database(path)
    launch_id = "e" * 32
    api_identity = _identity(105, 1_005)
    launcher_identity = _identity(205, 2_005)
    engine_identity = _identity(305, 3_005)
    expected_identities = {
        "api": api_identity,
        "engine_launcher": launcher_identity,
        "engine": engine_identity,
    }
    validation: dict[str, Any] = {}

    def fake_validate(
        record: EngineLaunchDB, **kwargs: Any
    ) -> dict[str, ProcessIdentity]:
        validation["record"] = record
        validation["kwargs"] = kwargs
        return expected_identities

    monkeypatch.setattr(tinker_api, "validate_ready_engine_launch", fake_validate)
    exit_task = asyncio.create_task(asyncio.Event().wait())
    try:
        await _insert_starting(
            engine,
            path,
            launch_id=launch_id,
            api_identity=api_identity,
        )
        async with AsyncSession(engine) as session:
            record = await session.get(EngineLaunchDB, launch_id)
            assert record is not None
            record.status = EngineLaunchStatus.READY
            record.engine_launcher_pid = launcher_identity.pid
            record.engine_launcher_start_ticks = launcher_identity.start_ticks
            record.engine_pid = engine_identity.pid
            record.engine_start_ticks = engine_identity.start_ticks
            await session.commit()

        identities = await _wait_for_engine_ready(
            engine,
            launch_id=launch_id,
            backend="jax",
            api_identity=api_identity,
            launcher_identity=launcher_identity,
            expected_api_source_attestation=_API_SOURCE,
            expected_api_lock_attestation=_API_LOCK,
            background_engine=SimpleNamespace(returncode=None),
            engine_exit_task=exit_task,
            timeout_sec=1,
        )

        assert identities == expected_identities
        assert validation["record"].launch_id == launch_id
        assert validation["kwargs"] == {
            "launch_id": launch_id,
            "backend": "jax",
            "api_identity": api_identity,
            "launcher_identity": launcher_identity,
            "expected_api_source_attestation": _API_SOURCE,
            "expected_api_lock_attestation": _API_LOCK,
        }
    finally:
        exit_task.cancel()
        with suppress(asyncio.CancelledError):
            await exit_task
        await engine.dispose()


@_asyncio_test
async def test_wait_for_engine_ready_rejects_terminal_row_without_polling(
    tmp_path: Path,
) -> None:
    path = tmp_path / "failed.db"
    engine = await _new_database(path)
    launch_id = "f" * 32
    api_identity = _identity(106, 1_006)
    exit_task = asyncio.create_task(asyncio.Event().wait())
    try:
        await _insert_starting(
            engine,
            path,
            launch_id=launch_id,
            api_identity=api_identity,
        )
        assert await _mark_engine_launch(
            engine,
            launch_id=launch_id,
            api_identity=api_identity,
            status=EngineLaunchStatus.FAILED,
            expected_statuses=(EngineLaunchStatus.STARTING,),
            error_message="constructor failed",
        )

        with pytest.raises(
            RuntimeError,
            match="entered failed before readiness: constructor failed",
        ):
            await _wait_for_engine_ready(
                engine,
                launch_id=launch_id,
                backend="jax",
                api_identity=api_identity,
                launcher_identity=_identity(206, 2_006),
                expected_api_source_attestation=_API_SOURCE,
                expected_api_lock_attestation=_API_LOCK,
                background_engine=SimpleNamespace(returncode=None),
                engine_exit_task=exit_task,
                timeout_sec=1,
            )
    finally:
        exit_task.cancel()
        with suppress(asyncio.CancelledError):
            await exit_task
        await engine.dispose()


def test_server_waiters_poll_exact_engine_health_endpoint() -> None:
    shell_waiters = (
        "tests/train/gpu_e2e_test/gsm8k_tinker.sh",
        "tests/train/gpu_e2e_test/gsm8k_tinker_fully_async.sh",
        (
            "tests/train/gpu_e2e_test/official_tinker_vs_skyrl/"
            "gsm8k_official_tinker_qwen3-8b_baseline.sh"
        ),
        (
            "tests/train/gpu_e2e_test/official_tinker_vs_skyrl/"
            "gsm8k_tinker_colocated_llama-3.1-8b-instruct_baseline.sh"
        ),
        (
            "tests/train/gpu_e2e_test/official_tinker_vs_skyrl/"
            "gsm8k_tinker_fully_async_llama-3.1-8b-instruct_baseline.sh"
        ),
    )
    health_wait = (
        "until curl -sSf http://localhost:8000/api/v1/healthz >/dev/null 2>&1; do"
    )
    for relative_path in shell_waiters:
        source = (_REPO / relative_path).read_text(encoding="utf-8")
        assert health_wait in source
        assert "localhost:8000/docs" not in source

    modal_source = (_REPO / "examples/tinker/ppo/modal_run.py").read_text(
        encoding="utf-8"
    )
    assert 'f"{base_url}/api/v1/healthz"' in modal_source
    assert 'f"{base_url}/docs"' not in modal_source


@_asyncio_test
async def test_stop_background_engine_terminates_complete_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exit_future: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    group_alive = True
    observed_signals: list[tuple[int, int]] = []

    def signal_group(process_group_id: int, signal_number: int) -> bool:
        nonlocal group_alive
        observed_signals.append((process_group_id, signal_number))
        group_alive = False
        exit_future.set_result(-signal_number)
        return True

    monkeypatch.setattr(tinker_api, "_signal_engine_process_group", signal_group)
    monkeypatch.setattr(
        tinker_api,
        "_engine_process_group_exists",
        lambda _process_group_id: group_alive,
    )

    result = await tinker_api._stop_background_engine(
        SimpleNamespace(pid=202, returncode=None),
        exit_future,
        202,
        graceful_timeout_sec=0.01,
        forced_timeout_sec=0.01,
        poll_interval_sec=0,
    )

    assert result.exit_code == -signal.SIGTERM
    assert result.signals_sent == (signal.SIGTERM,)
    assert not result.exited_before_signal
    assert observed_signals == [(202, signal.SIGTERM)]
    assert not group_alive


@_asyncio_test
async def test_stop_background_engine_escalates_surviving_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exit_future: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    group_alive = True
    observed_signals: list[tuple[int, int]] = []

    def signal_group(process_group_id: int, signal_number: int) -> bool:
        nonlocal group_alive
        observed_signals.append((process_group_id, signal_number))
        if signal_number == signal.SIGKILL:
            group_alive = False
            exit_future.set_result(-signal_number)
        return True

    monkeypatch.setattr(tinker_api, "_signal_engine_process_group", signal_group)
    monkeypatch.setattr(
        tinker_api,
        "_engine_process_group_exists",
        lambda _process_group_id: group_alive,
    )

    result = await tinker_api._stop_background_engine(
        SimpleNamespace(pid=202, returncode=None),
        exit_future,
        202,
        graceful_timeout_sec=0,
        forced_timeout_sec=0.01,
        poll_interval_sec=0,
    )

    assert result.exit_code == -signal.SIGKILL
    assert result.signals_sent == (signal.SIGTERM, signal.SIGKILL)
    assert not result.exited_before_signal
    assert observed_signals == [
        (202, signal.SIGTERM),
        (202, signal.SIGKILL),
    ]
    assert not group_alive


@_asyncio_test
async def test_healthz_accepts_only_current_live_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "health.db"
    engine = await _new_database(path)
    launch_id = "1" * 32
    api_identity = _identity(111, 1_111)
    launcher_identity = _identity(211, 2_111)
    engine_identity = _identity(311, 3_111)
    accepted_identities = {
        "api": api_identity,
        "engine_launcher": launcher_identity,
        "engine": engine_identity,
    }
    exit_future: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    try:
        await _insert_starting(
            engine,
            path,
            launch_id=launch_id,
            api_identity=api_identity,
        )
        monkeypatch.setattr(
            tinker_api,
            "validate_ready_engine_launch",
            lambda _record, **_kwargs: accepted_identities,
        )
        state = SimpleNamespace(
            engine_launch_id=launch_id,
            engine_launch_identities=accepted_identities,
            background_engine=SimpleNamespace(returncode=None),
            engine_exit_task=exit_future,
            db_engine=engine,
            engine_config=_config(path),
            runtime_source_attestation=_API_SOURCE,
            runtime_launch_lock_attestation=_API_LOCK,
        )
        request = SimpleNamespace(app=SimpleNamespace(state=state))

        response = await tinker_api.healthz(request)
        assert response.status == "ok"

        exit_future.set_result(1)
        with pytest.raises(tinker_api.HTTPException) as error:
            await tinker_api.healthz(request)
        assert error.value.status_code == 503
        assert error.value.detail == "engine unavailable"
    finally:
        if not exit_future.done():
            exit_future.cancel()
        await engine.dispose()


@_asyncio_test
async def test_wait_for_engine_ready_makes_child_exit_win(
    tmp_path: Path,
) -> None:
    path = tmp_path / "child-exit.db"
    engine = await _new_database(path)
    exit_future: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    exit_future.set_result(17)
    try:
        with pytest.raises(RuntimeError, match="exited before readiness with code 17"):
            await _wait_for_engine_ready(
                engine,
                launch_id="2" * 32,
                backend="jax",
                api_identity=_identity(112, 1_112),
                launcher_identity=_identity(212, 2_112),
                expected_api_source_attestation=_API_SOURCE,
                expected_api_lock_attestation=_API_LOCK,
                background_engine=SimpleNamespace(returncode=17),
                engine_exit_task=exit_future,
                timeout_sec=1,
            )
    finally:
        await engine.dispose()


@_asyncio_test
async def test_wait_for_engine_ready_enforces_bounded_deadline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "timeout.db"
    engine = await _new_database(path)
    launch_id = "3" * 32
    exit_future: asyncio.Future[int] = asyncio.get_running_loop().create_future()
    api_identity = _identity(113, 1_113)
    try:
        await _insert_starting(
            engine,
            path,
            launch_id=launch_id,
            api_identity=api_identity,
        )
        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(TimeoutError, match="did not become ready"):
            await _wait_for_engine_ready(
                engine,
                launch_id=launch_id,
                backend="jax",
                api_identity=api_identity,
                launcher_identity=_identity(213, 2_113),
                expected_api_source_attestation=_API_SOURCE,
                expected_api_lock_attestation=_API_LOCK,
                background_engine=SimpleNamespace(returncode=None),
                engine_exit_task=exit_future,
                timeout_sec=0.02,
            )
        assert loop.time() - started < 0.5
    finally:
        exit_future.cancel()
        with suppress(asyncio.CancelledError):
            await exit_future
        await engine.dispose()


def test_process_group_ownership_rejects_pid_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _identity(214, 2_114)
    monkeypatch.setattr(
        tinker_api,
        "read_process_identity",
        lambda _pid: expected,
    )
    monkeypatch.setattr(tinker_api.os, "getpgid", lambda _pid: 214)
    assert tinker_api._identity_owns_engine_process_group(expected, 214)

    reused = _identity(expected.pid, expected.start_ticks + 1)
    monkeypatch.setattr(
        tinker_api,
        "read_process_identity",
        lambda _pid: reused,
    )
    assert not tinker_api._identity_owns_engine_process_group(expected, 214)


@pytest.mark.parametrize(
    "result",
    [
        tinker_api._EngineStopResult(0, (signal.SIGTERM,), False),
        tinker_api._EngineStopResult(-signal.SIGTERM, (signal.SIGTERM,), False),
        tinker_api._EngineStopResult(128 + signal.SIGTERM, (signal.SIGTERM,), False),
        tinker_api._EngineStopResult(
            -signal.SIGTERM,
            (signal.SIGTERM, signal.SIGKILL),
            False,
        ),
        tinker_api._EngineStopResult(-signal.SIGKILL, (signal.SIGTERM, signal.SIGKILL), False),
        tinker_api._EngineStopResult(
            128 + signal.SIGKILL,
            (signal.SIGTERM, signal.SIGKILL),
            False,
        ),
    ],
)
def test_expected_shutdown_results_require_causal_signal(
    result: tinker_api._EngineStopResult,
) -> None:
    assert tinker_api._expected_engine_shutdown_result(result)


@pytest.mark.parametrize(
    "result",
    [
        tinker_api._EngineStopResult(0, (), True),
        tinker_api._EngineStopResult(-signal.SIGKILL, (), True),
        tinker_api._EngineStopResult(-signal.SIGKILL, (signal.SIGTERM,), False),
        tinker_api._EngineStopResult(17, (signal.SIGTERM,), False),
    ],
)
def test_unexpected_shutdown_results_are_rejected(
    result: tinker_api._EngineStopResult,
) -> None:
    assert not tinker_api._expected_engine_shutdown_result(result)


@_asyncio_test
async def test_real_process_group_cleanup_kills_sigterm_ignoring_wrapper() -> None:
    script = """
import signal
import subprocess
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen(["/usr/bin/sleep", "60"])
print(child.pid, flush=True)
time.sleep(60)
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        start_new_session=True,
        stdout=asyncio.subprocess.PIPE,
    )
    assert process.stdout is not None
    exit_task = asyncio.create_task(process.wait())
    try:
        await asyncio.wait_for(process.stdout.readline(), timeout=2)
        result = await tinker_api._stop_background_engine(
            process,
            exit_task,
            process.pid,
            graceful_timeout_sec=0.05,
            forced_timeout_sec=2,
            poll_interval_sec=0.01,
        )
        assert result.exit_code == -signal.SIGKILL
        assert result.signals_sent == (signal.SIGTERM, signal.SIGKILL)
        assert not tinker_api._engine_process_group_exists(process.pid)
    finally:
        if tinker_api._engine_process_group_exists(process.pid):
            tinker_api._signal_engine_process_group(process.pid, signal.SIGKILL)
        if not exit_task.done():
            await asyncio.wait_for(exit_task, timeout=2)
