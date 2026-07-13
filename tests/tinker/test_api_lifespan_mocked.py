"""CPU-only contract tests for the complete API engine lifespan."""

import asyncio
import signal
from dataclasses import dataclass, field
from functools import wraps
from types import SimpleNamespace
from typing import Any, Literal

import pytest

from skyrl.tinker import api as tinker_api
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import EngineLaunchStatus
from skyrl.tinker.engine_startup import ProcessIdentity

_BOOT_ID = "54ccf56c-5f4f-4ef7-ac98-c13e0587b5b9"
_LAUNCH_ID = "7" * 32
_API_ATTESTATION = {"status": "not_required", "role": "api"}
_LOCK_ATTESTATION = {"status": "not_required"}


def _asyncio_test(function: Any) -> Any:
    """Run an async test without requiring the optional pytest plugin."""

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(function(*args, **kwargs))

    return wrapper


def _identity(pid: int, start_ticks: int) -> ProcessIdentity:
    return ProcessIdentity(
        pid=pid,
        start_ticks=start_ticks,
        boot_id=_BOOT_ID,
        state="S",
    )


class _FakeConnection:
    def __init__(self, observations: "_Observations") -> None:
        self._observations = observations

    async def run_sync(self, function: Any) -> None:
        self._observations.schema_functions.append(function)


class _FakeBegin:
    def __init__(self, observations: "_Observations") -> None:
        self._connection = _FakeConnection(observations)

    async def __aenter__(self) -> _FakeConnection:
        return self._connection

    async def __aexit__(self, *_args: Any) -> None:
        return None


class _FakeDatabaseEngine:
    def __init__(self, observations: "_Observations") -> None:
        self.sync_engine = object()
        self._observations = observations

    def begin(self) -> _FakeBegin:
        return _FakeBegin(self._observations)

    async def dispose(self) -> None:
        self._observations.cleanup_events.append(("dispose",))
        self._observations.dispose_calls += 1


class _FakeProcess:
    def __init__(
        self,
        observations: "_Observations",
        *,
        initial_exit_code: int | None,
    ) -> None:
        self.pid = 4_242
        self.returncode = initial_exit_code
        self.wait_calls = 0
        self.terminate_calls = 0
        self.kill_calls = 0
        self._observations = observations
        self._exit: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        if initial_exit_code is not None:
            self._exit.set_result(initial_exit_code)

    async def wait(self) -> int:
        self.wait_calls += 1
        task = asyncio.current_task()
        assert task is not None
        self._observations.process_wait_tasks.append(task)
        return await self._exit

    def finish(self, exit_code: int) -> None:
        self.returncode = exit_code
        if not self._exit.done():
            self._exit.set_result(exit_code)

    def terminate(self) -> None:
        self.terminate_calls += 1
        raise AssertionError("dedicated process group must be signalled")

    def kill(self) -> None:
        self.kill_calls += 1
        raise AssertionError("dedicated process group must be signalled")


@dataclass
class _Observations:
    scenario: Literal["ready", "child_exit", "timeout"]
    subprocess_calls: list[tuple[tuple[str, ...], dict[str, Any]]] = field(
        default_factory=list
    )
    schema_functions: list[Any] = field(default_factory=list)
    process_wait_tasks: list[asyncio.Task[int]] = field(default_factory=list)
    readiness_exit_tasks: list[asyncio.Task[int]] = field(default_factory=list)
    stop_exit_tasks: list[asyncio.Task[int]] = field(default_factory=list)
    cleanup_events: list[tuple[Any, ...]] = field(default_factory=list)
    terminal_writes: list[dict[str, Any]] = field(default_factory=list)
    child_configs: list[EngineConfig] = field(default_factory=list)
    dispose_calls: int = 0
    process: _FakeProcess | None = None


def _app() -> SimpleNamespace:
    config = EngineConfig(
        base_model="test/model",
        backend="jax",
        database_url="sqlite:///unused.db",
        engine_startup_timeout_sec=30,
    )
    return SimpleNamespace(state=SimpleNamespace(engine_config=config))


def _install_lifespan_mocks(
    monkeypatch: pytest.MonkeyPatch,
    scenario: Literal["ready", "child_exit", "timeout"],
) -> _Observations:
    observations = _Observations(scenario=scenario)
    database_engine = _FakeDatabaseEngine(observations)
    api_identity = _identity(1_001, 11_001)
    launcher_identity = _identity(4_242, 14_242)
    engine_identity = _identity(5_005, 15_005)
    accepted_identities = {
        "api": api_identity,
        "engine_launcher": launcher_identity,
        "engine": engine_identity,
    }

    monkeypatch.setattr(
        tinker_api,
        "revalidate_runtime_source",
        lambda **_kwargs: dict(_API_ATTESTATION),
    )
    monkeypatch.setattr(
        tinker_api,
        "revalidate_runtime_launch_lock",
        lambda _initial: dict(_LOCK_ATTESTATION),
    )
    monkeypatch.setattr(
        tinker_api,
        "create_async_engine",
        lambda *_args, **_kwargs: database_engine,
    )
    monkeypatch.setattr(tinker_api, "enable_sqlite_wal", lambda _engine: None)
    monkeypatch.setattr(tinker_api, "new_engine_launch_id", lambda: _LAUNCH_ID)
    monkeypatch.setattr(
        tinker_api,
        "read_process_identity",
        lambda pid=None: api_identity if pid is None else launcher_identity,
    )
    monkeypatch.setattr(
        tinker_api.psutil,
        "Process",
        lambda _pid: SimpleNamespace(cmdline=lambda: ["uv", "run", "api"]),
    )

    def build_command(
        _parent_cmd: list[str],
        child_config: EngineConfig,
        **_kwargs: Any,
    ) -> list[str]:
        observations.child_configs.append(child_config)
        return ["uv", "run", "-m", "skyrl.tinker.engine"]

    monkeypatch.setattr(tinker_api, "_build_uv_run_cmd_engine", build_command)

    async def create_subprocess(
        *command: str,
        **options: Any,
    ) -> _FakeProcess:
        observations.subprocess_calls.append((command, options))
        initial_exit_code = 17 if scenario == "child_exit" else None
        process = _FakeProcess(
            observations,
            initial_exit_code=initial_exit_code,
        )
        observations.process = process
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    monkeypatch.setattr(tinker_api.os, "getppid", lambda: 999)
    monkeypatch.setattr(tinker_api.os, "getpgid", lambda pid: pid)

    async def insert_launch(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def bind_launcher(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(tinker_api, "_insert_engine_launch", insert_launch)
    monkeypatch.setattr(tinker_api, "_bind_engine_launcher", bind_launcher)

    async def wait_for_ready(
        *_args: Any,
        engine_exit_task: asyncio.Task[int],
        **_kwargs: Any,
    ) -> dict[str, ProcessIdentity]:
        observations.readiness_exit_tasks.append(engine_exit_task)
        if scenario == "child_exit":
            exit_code = await asyncio.shield(engine_exit_task)
            raise RuntimeError(
                f"background engine exited before readiness with code {exit_code}"
            )
        if scenario == "timeout":
            raise TimeoutError("background engine did not become ready")
        return accepted_identities

    monkeypatch.setattr(tinker_api, "_wait_for_engine_ready", wait_for_ready)

    def signal_group(process_group_id: int, signal_number: int) -> bool:
        observations.cleanup_events.append(("signal", process_group_id, signal_number))
        return True

    monkeypatch.setattr(tinker_api, "_signal_engine_process_group", signal_group)

    async def stop_process(
        background_engine: _FakeProcess,
        engine_exit_task: asyncio.Task[int],
        process_group_id: int | None,
    ) -> tinker_api._EngineStopResult:
        observations.cleanup_events.append(
            ("signal", process_group_id, signal.SIGTERM)
        )
        observations.cleanup_events.append(("stop", process_group_id))
        observations.stop_exit_tasks.append(engine_exit_task)
        exited_before_signal = background_engine.returncode is not None
        if background_engine.returncode is None:
            background_engine.finish(-signal.SIGTERM)
        return tinker_api._EngineStopResult(
            exit_code=await asyncio.shield(engine_exit_task),
            signals_sent=(signal.SIGTERM,),
            exited_before_signal=exited_before_signal,
        )

    monkeypatch.setattr(tinker_api, "_stop_background_engine", stop_process)

    async def mark_launch(*_args: Any, **kwargs: Any) -> bool:
        observations.cleanup_events.append(("mark", kwargs["status"]))
        observations.terminal_writes.append(dict(kwargs))
        return True

    monkeypatch.setattr(tinker_api, "_mark_engine_launch", mark_launch)
    return observations


def _assert_common_cleanup(
    app: SimpleNamespace,
    observations: _Observations,
) -> _FakeProcess:
    process = observations.process
    assert process is not None
    assert observations.subprocess_calls == [
        (
            ("uv", "run", "-m", "skyrl.tinker.engine"),
            {"start_new_session": True},
        )
    ]
    assert observations.child_configs[0].startup_launch_id == _LAUNCH_ID
    assert process.wait_calls == 1
    assert len(observations.process_wait_tasks) == 1
    assert observations.readiness_exit_tasks == [app.state.engine_exit_task]
    assert observations.stop_exit_tasks == [app.state.engine_exit_task]
    assert observations.process_wait_tasks[0] is app.state.engine_exit_task
    assert observations.cleanup_events[0] == (
        "signal",
        process.pid,
        signal.SIGTERM,
    )
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert observations.dispose_calls == 1
    assert observations.cleanup_events[-1] == ("dispose",)
    return process


@_asyncio_test
async def test_lifespan_normal_ready_shutdown_reuses_wait_task_and_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _app()
    observations = _install_lifespan_mocks(monkeypatch, "ready")

    async with tinker_api.lifespan(app):
        process = observations.process
        assert process is not None
        assert app.state.engine_launch_id == _LAUNCH_ID
        assert app.state.engine_process_group_id == process.pid
        assert app.state.engine_launch_identities is not None
        assert not app.state.engine_exit_task.done()

    process = _assert_common_cleanup(app, observations)
    assert process.returncode == -signal.SIGTERM
    assert len(observations.terminal_writes) == 1
    terminal = observations.terminal_writes[0]
    assert terminal["status"] == EngineLaunchStatus.STOPPED
    assert terminal["expected_statuses"] == (EngineLaunchStatus.READY,)
    assert terminal["error_message"] is None


@_asyncio_test
async def test_lifespan_child_exit_before_ready_fails_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _app()
    observations = _install_lifespan_mocks(monkeypatch, "child_exit")

    with pytest.raises(
        RuntimeError,
        match="exited before readiness with code 17",
    ):
        async with tinker_api.lifespan(app):
            raise AssertionError("lifespan yielded after child exit")

    process = _assert_common_cleanup(app, observations)
    assert process.returncode == 17
    assert len(observations.terminal_writes) == 1
    terminal = observations.terminal_writes[0]
    assert terminal["status"] == EngineLaunchStatus.FAILED
    assert terminal["expected_statuses"] == (
        EngineLaunchStatus.STARTING,
        EngineLaunchStatus.INITIALIZING,
        EngineLaunchStatus.READY,
    )
    assert terminal["error_message"].startswith("RuntimeError:")


@_asyncio_test
async def test_lifespan_readiness_timeout_fails_stops_and_disposes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _app()
    observations = _install_lifespan_mocks(monkeypatch, "timeout")

    with pytest.raises(TimeoutError, match="did not become ready"):
        async with tinker_api.lifespan(app):
            raise AssertionError("lifespan yielded after readiness timeout")

    process = _assert_common_cleanup(app, observations)
    assert process.returncode == -signal.SIGTERM
    assert len(observations.terminal_writes) == 1
    terminal = observations.terminal_writes[0]
    assert terminal["status"] == EngineLaunchStatus.FAILED
    assert terminal["expected_statuses"] == (
        EngineLaunchStatus.STARTING,
        EngineLaunchStatus.INITIALIZING,
        EngineLaunchStatus.READY,
    )
    assert terminal["error_message"].startswith("TimeoutError:")
