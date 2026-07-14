# Runtime source attestation intentionally precedes every third-party import.
# ruff: noqa: E402

import asyncio
import hashlib
import json
import os
import random
import signal
import threading
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, AsyncGenerator, ClassVar, Literal
from uuid import uuid4

import skyrl as _skyrl_package
from skyrl.tinker.runtime_source import (
    revalidate_runtime_launch_lock,
    revalidate_runtime_source,
    validate_runtime_launch_lock,
    validate_runtime_source,
)

_RUNTIME_LAUNCH_LOCK_ATTESTATION = validate_runtime_launch_lock()
_RUNTIME_SOURCE_ATTESTATION = validate_runtime_source(
    role="api",
    module_file=Path(__file__),
    package_file=Path(_skyrl_package.__file__),
)

import fastapi
import psutil
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import (
    Base64Bytes,
    BaseModel,
    Discriminator,
    Field,
    Tag,
    model_validator,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.exc import TimeoutError as SATimeoutError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, func, select, update
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import types
from skyrl.tinker.config import EngineConfig, add_model, config_to_argv
from skyrl.tinker.db_models import (
    CheckpointDB,
    CheckpointStatus,
    EngineLaunchDB,
    EngineLaunchStatus,
    FutureDB,
    ModelDB,
    RequestDedupDB,
    RequestStatus,
    SamplingSessionDB,
    SessionDB,
    enable_sqlite_wal,
    get_async_database_url,
)
from skyrl.tinker.engine_startup import (
    ProcessIdentity,
    new_engine_launch_id,
    read_process_identity,
    validate_ready_engine_launch,
)
from skyrl.tinker.extra import (
    ExternalInferenceClient,
    SkyRLTrainInferenceForwardingClient,
)
from skyrl.utils.log import get_uvicorn_log_config, logger
from skyrl.utils.storage import download_file

# Validation patterns for train_run_ids, model_ids and checkpoint_ids
ID_PATTERN = r"^[a-zA-Z0-9_-]+$"
ID_MAX_LENGTH = 255

API_SERVER_STARTUP_ARGS = ["-m", "skyrl.tinker.api"]
_UV_RUN_OPTIONS_WITH_VALUES = {
    "--allow-insecure-host",
    "--cache-dir",
    "--color",
    "--config-file",
    "--config-setting",
    "--config-settings-package",
    "--default-index",
    "--directory",
    "--env-file",
    "--exclude-newer",
    "--exclude-newer-package",
    "--extra",
    "--extra-index-url",
    "--find-links",
    "--fork-strategy",
    "--group",
    "--index",
    "--index-strategy",
    "--index-url",
    "--keyring-provider",
    "--link-mode",
    "--no-binary-package",
    "--no-build-isolation-package",
    "--no-build-package",
    "--no-extra",
    "--no-group",
    "--no-sources-package",
    "--only-group",
    "--package",
    "--prerelease",
    "--project",
    "--python",
    "--python-platform",
    "--refresh-package",
    "--reinstall-package",
    "--resolution",
    "--upgrade-group",
    "--upgrade-package",
    "--with",
    "--with-editable",
    "--with-requirements",
    "-C",
    "-P",
    "-f",
    "-i",
    "-p",
    "-w",
}

# Timeout for graceful shutdown when engine crashes
SHUTDOWN_TIMEOUT_SECONDS = 20
STARTUP_DB_POLL_TIMEOUT_SECONDS = 2.0
TERMINAL_STATE_WRITE_TIMEOUT_SECONDS = 2.0
READINESS_DB_FAILURE_GRACE_SECONDS = 5.0


def _get_parent_uv_run_args(
    parent_cmd: list[str],
    runtime_source_attestation: dict[str, Any] | None = None,
) -> list[str]:
    """Extract parent `uv run <uv run args>` flags for the engine launch given the parent process's startup command

    `uv run` starts this Python API process as a child. To recover the original
    `uv run <uv run args> ...` flags, we inspect the parent process command line
    and extract all the uv run args before the script argument.
    """
    if (
        len(parent_cmd) < 4
        or Path(parent_cmd[0]).name not in {"uv", "uv.exe"}
        or parent_cmd[1] != "run"
    ):
        raise ValueError(
            f"Unable to parse tinker API server startup command: {parent_cmd}. "
            "Ensure that the tinker API server was started with `uv run <uv run args> -m skyrl.tinker.api`"
        )
    if runtime_source_attestation is not None and (
        runtime_source_attestation.get("status") == "passed"
    ):
        source_root = runtime_source_attestation["source_root"]
        expected_prefix = [
            runtime_source_attestation["uv_executable"],
            "run",
            "--active",
            "--no-sync",
            "--no-env-file",
            "--no-config",
            "--directory",
            source_root,
            "--project",
            source_root,
        ]
        if (
            parent_cmd[: len(expected_prefix)] != expected_prefix
            or parent_cmd[
                len(expected_prefix) : len(expected_prefix)
                + len(API_SERVER_STARTUP_ARGS)
            ]
            != API_SERVER_STARTUP_ARGS
        ):
            raise ValueError(
                "Hardened tinker API parent command does not match the exact "
                "snapshot uv policy"
            )
        return expected_prefix[2:]

    module_index = next(
        (
            index
            for index in range(2, len(parent_cmd) - 1)
            if parent_cmd[index : index + len(API_SERVER_STARTUP_ARGS)]
            == API_SERVER_STARTUP_ARGS
        ),
        None,
    )
    if module_index is None:
        raise ValueError(
            f"Unable to parse tinker API server startup command: {parent_cmd}. "
            "Ensure that the tinker API server was started with `uv run <uv run args> -m skyrl.tinker.api`"
        )
    stop_index = module_index
    if stop_index > 2 and parent_cmd[stop_index - 1] == "python":
        possible_option = parent_cmd[stop_index - 2]
        if possible_option in _UV_RUN_OPTIONS_WITH_VALUES:
            pass
        elif possible_option.startswith("-") and possible_option != "--":
            raise ValueError(
                "Unable to distinguish the Python command from an unknown uv "
                f"option value in startup command: {parent_cmd}"
            )
        else:
            stop_index -= 1
    if stop_index > 2 and parent_cmd[stop_index - 1] == "--":
        stop_index -= 1
    return parent_cmd[2:stop_index]


def _build_uv_run_cmd_engine(
    parent_cmd: list[str],
    engine_config: BaseModel,
    runtime_source_attestation: dict[str, Any] | None = None,
) -> list[str]:
    """Builds uv run command for the engine

    Args:
        parent_cmd: The command for the parent process starting the engine
        engine_config: Engine configuration
    Returns:
        cmd: The uv run command for the tinker engine
    """
    parent_flags = _get_parent_uv_run_args(
        parent_cmd, runtime_source_attestation=runtime_source_attestation
    )
    cmd = [parent_cmd[0], "run"]
    logger.debug(f"Detected API server uv run flags: {parent_flags}")
    cmd.extend(parent_flags)
    # NOTE: uv deduplicates extras so we can unconditionally add the tinker extra
    cmd.extend(["--extra", "tinker", "--extra", engine_config.backend])
    cmd.extend(["-m", "skyrl.tinker.engine"])
    cmd.extend(config_to_argv(engine_config))
    return cmd


async def _insert_engine_launch(
    db_engine: Any,
    *,
    launch_id: str,
    engine_config: EngineConfig,
    api_identity: ProcessIdentity,
    source_attestation: dict[str, Any],
    lock_attestation: dict[str, Any],
) -> None:
    """Insert the API-owned STARTING row without overwriting a collision."""
    record = EngineLaunchDB(
        launch_id=launch_id,
        backend=engine_config.backend,
        status=EngineLaunchStatus.STARTING,
        boot_id=api_identity.boot_id,
        api_pid=api_identity.pid,
        api_start_ticks=api_identity.start_ticks,
        api_source_attestation=dict(source_attestation),
        api_launch_lock_attestation=dict(lock_attestation),
    )
    async with AsyncSession(db_engine) as session:
        session.add(record)
        try:
            await session.commit()
        except IntegrityError as error:
            await session.rollback()
            raise RuntimeError("engine launch ID collision") from error


async def _bind_engine_launcher(
    db_engine: Any,
    *,
    launch_id: str,
    api_identity: ProcessIdentity,
    launcher_identity: ProcessIdentity,
) -> None:
    """Conditionally bind the uv wrapper identity to the STARTING launch."""
    if launcher_identity.boot_id != api_identity.boot_id:
        raise RuntimeError("API and engine launcher identities span different boots")
    async with AsyncSession(db_engine) as session:
        statement = (
            update(EngineLaunchDB)
            .where(
                EngineLaunchDB.launch_id == launch_id,
                EngineLaunchDB.status == EngineLaunchStatus.STARTING,
                EngineLaunchDB.api_pid == api_identity.pid,
                EngineLaunchDB.api_start_ticks == api_identity.start_ticks,
                EngineLaunchDB.engine_launcher_pid.is_(None),
                EngineLaunchDB.engine_launcher_start_ticks.is_(None),
            )
            .values(
                engine_launcher_pid=launcher_identity.pid,
                engine_launcher_start_ticks=launcher_identity.start_ticks,
                updated_at=datetime.now(timezone.utc),
            )
        )
        result = await session.exec(statement)
        if result.rowcount != 1:
            raise RuntimeError(
                "engine launcher identity lost its conditional binding race"
            )
        await session.commit()


async def _mark_engine_launch(
    db_engine: Any,
    *,
    launch_id: str,
    api_identity: ProcessIdentity,
    status: EngineLaunchStatus,
    expected_statuses: tuple[EngineLaunchStatus, ...],
    error_message: str | None = None,
) -> bool:
    """Apply one terminal launch transition without reviving stale rows."""
    now = datetime.now(timezone.utc)
    values: dict[str, Any] = {
        "status": status,
        "updated_at": now,
        "error_message": error_message[:2048] if error_message else None,
    }
    async with AsyncSession(db_engine) as session:
        statement = (
            update(EngineLaunchDB)
            .where(
                EngineLaunchDB.launch_id == launch_id,
                EngineLaunchDB.boot_id == api_identity.boot_id,
                EngineLaunchDB.api_pid == api_identity.pid,
                EngineLaunchDB.api_start_ticks == api_identity.start_ticks,
                EngineLaunchDB.status.in_(expected_statuses),
            )
            .values(**values)
        )
        result = await session.exec(statement)
        await session.commit()
        return result.rowcount == 1


async def _wait_for_engine_ready(
    db_engine: Any,
    *,
    launch_id: str,
    backend: str,
    api_identity: ProcessIdentity,
    launcher_identity: ProcessIdentity,
    expected_api_source_attestation: dict[str, Any],
    expected_api_lock_attestation: dict[str, Any],
    background_engine: asyncio.subprocess.Process,
    engine_exit_task: asyncio.Task[int],
    timeout_sec: float,
) -> dict[str, ProcessIdentity]:
    """Wait for this exact launch's READY row while racing child exit."""
    deadline = time.monotonic() + timeout_sec
    while True:
        if engine_exit_task.done() or background_engine.returncode is not None:
            exit_code = await asyncio.shield(engine_exit_task)
            raise RuntimeError(
                f"background engine exited before readiness with code {exit_code}"
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"background engine did not become ready within {timeout_sec:g} seconds"
            )

        async def read_launch_record() -> EngineLaunchDB | None:
            async with AsyncSession(db_engine) as session:
                return await session.get(EngineLaunchDB, launch_id)

        try:
            record = await asyncio.wait_for(
                read_launch_record(),
                timeout=min(STARTUP_DB_POLL_TIMEOUT_SECONDS, remaining),
            )
        except asyncio.TimeoutError:
            if engine_exit_task.done() or background_engine.returncode is not None:
                exit_code = await asyncio.shield(engine_exit_task)
                raise RuntimeError(
                    "background engine exited while its readiness row was blocked "
                    f"with code {exit_code}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "background engine readiness deadline expired during DB read"
                ) from None
            continue

        if record is None:
            raise RuntimeError("current engine launch row disappeared")
        else:
            status = str(getattr(record.status, "value", record.status))
            if status == EngineLaunchStatus.READY.value:
                identities = validate_ready_engine_launch(
                    record,
                    launch_id=launch_id,
                    backend=backend,
                    api_identity=api_identity,
                    launcher_identity=launcher_identity,
                    expected_api_source_attestation=(
                        expected_api_source_attestation
                    ),
                    expected_api_lock_attestation=expected_api_lock_attestation,
                )
                if (
                    engine_exit_task.done()
                    or background_engine.returncode is not None
                ):
                    exit_code = await asyncio.shield(engine_exit_task)
                    raise RuntimeError(
                        "background engine exited during readiness validation "
                        f"with code {exit_code}"
                    )
                return identities
            if status in {
                EngineLaunchStatus.FAILED.value,
                EngineLaunchStatus.STOPPED.value,
            }:
                detail = record.error_message or "no failure detail was published"
                raise RuntimeError(
                    f"background engine entered {status} before readiness: {detail}"
                )
            if status not in {
                EngineLaunchStatus.STARTING.value,
                EngineLaunchStatus.INITIALIZING.value,
            }:
                raise RuntimeError(f"background engine published invalid state {status!r}")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"background engine did not become ready within {timeout_sec:g} seconds"
            )
        poll_task = asyncio.create_task(asyncio.sleep(min(0.05, remaining)))
        done, _ = await asyncio.wait(
            {engine_exit_task, poll_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if engine_exit_task in done:
            if not poll_task.done():
                poll_task.cancel()
                with suppress(asyncio.CancelledError):
                    await poll_task
            exit_code = await asyncio.shield(engine_exit_task)
            raise RuntimeError(
                f"background engine exited before readiness with code {exit_code}"
            )
        await poll_task


def _engine_process_group_exists(process_group_id: int) -> bool:
    """Probe the dedicated child group without treating EPERM as absence."""
    if type(process_group_id) is not int or process_group_id <= 0:
        raise RuntimeError("engine process-group ID is invalid")
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError as error:
        raise RuntimeError("cannot inspect the engine process group") from error
    return True


def _signal_engine_process_group(process_group_id: int, signal_number: int) -> bool:
    """Signal the isolated engine tree; return false only if it is already gone."""
    try:
        os.killpg(process_group_id, signal_number)
    except ProcessLookupError:
        return False
    except PermissionError as error:
        raise RuntimeError("cannot signal the engine process group") from error
    return True


def _identity_owns_engine_process_group(
    identity: ProcessIdentity | None,
    process_group_id: int,
) -> bool:
    """Bind group signalling to a non-reused recorded process identity."""
    if identity is None:
        return False
    try:
        current = read_process_identity(identity.pid)
        current_process_group = os.getpgid(identity.pid)
    except (OSError, RuntimeError):
        return False
    return (
        current.start_ticks == identity.start_ticks
        and current.boot_id == identity.boot_id
        and current.state not in {"X", "Z", "x"}
        and current_process_group == process_group_id
    )


@dataclass(frozen=True)
class _EngineStopResult:
    exit_code: int
    signals_sent: tuple[int, ...]
    exited_before_signal: bool


def _expected_engine_shutdown_result(result: _EngineStopResult) -> bool:
    """Require causal evidence that API-delivered shutdown ended the wrapper."""
    if result.exited_before_signal or not result.signals_sent:
        return False
    if signal.SIGTERM in result.signals_sent and result.exit_code in {
        0,
        -signal.SIGTERM,
        128 + signal.SIGTERM,
    }:
        return True
    if signal.SIGKILL in result.signals_sent:
        return result.exit_code in {-signal.SIGKILL, 128 + signal.SIGKILL}
    return False


async def _stop_background_engine(
    background_engine: asyncio.subprocess.Process,
    engine_exit_task: asyncio.Task[int],
    process_group_id: int | None,
    *,
    graceful_timeout_sec: float = 5.0,
    forced_timeout_sec: float = 5.0,
    poll_interval_sec: float = 0.05,
) -> _EngineStopResult:
    """Terminate the isolated engine tree, reap its wrapper, and prove exit."""
    exited_before_signal = (
        engine_exit_task.done() or background_engine.returncode is not None
    )
    signals_sent: list[int] = []
    if process_group_id is None:
        try:
            background_engine.terminate()
            signals_sent.append(signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            exit_code = await asyncio.wait_for(
                asyncio.shield(engine_exit_task), timeout=graceful_timeout_sec
            )
            return _EngineStopResult(
                exit_code=exit_code,
                signals_sent=tuple(signals_sent),
                exited_before_signal=exited_before_signal,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Background engine (PID %d) did not terminate gracefully, killing",
                background_engine.pid,
            )
            try:
                background_engine.kill()
                signals_sent.append(signal.SIGKILL)
            except ProcessLookupError:
                pass
        exit_code = await asyncio.wait_for(
            asyncio.shield(engine_exit_task), timeout=forced_timeout_sec
        )
        return _EngineStopResult(
            exit_code=exit_code,
            signals_sent=tuple(signals_sent),
            exited_before_signal=exited_before_signal,
        )

    if _signal_engine_process_group(process_group_id, signal.SIGTERM):
        signals_sent.append(signal.SIGTERM)
    graceful_deadline = time.monotonic() + graceful_timeout_sec
    while time.monotonic() < graceful_deadline:
        if engine_exit_task.done() and not _engine_process_group_exists(
            process_group_id
        ):
            return _EngineStopResult(
                exit_code=await asyncio.shield(engine_exit_task),
                signals_sent=tuple(signals_sent),
                exited_before_signal=exited_before_signal,
            )
        await asyncio.sleep(poll_interval_sec)

    if _engine_process_group_exists(process_group_id):
        logger.warning(
            "Engine process group %d did not terminate gracefully, killing",
            process_group_id,
        )
        if _signal_engine_process_group(process_group_id, signal.SIGKILL):
            signals_sent.append(signal.SIGKILL)
    try:
        exit_code = await asyncio.wait_for(
            asyncio.shield(engine_exit_task), timeout=forced_timeout_sec
        )
    except asyncio.TimeoutError as error:
        raise RuntimeError("engine launcher could not be reaped after SIGKILL") from error
    forced_deadline = time.monotonic() + forced_timeout_sec
    while _engine_process_group_exists(process_group_id):
        if time.monotonic() >= forced_deadline:
            raise RuntimeError("engine process group survived SIGKILL")
        await asyncio.sleep(poll_interval_sec)
    return _EngineStopResult(
        exit_code=exit_code,
        signals_sent=tuple(signals_sent),
        exited_before_signal=exited_before_signal,
    )


def _validate_required_startup_cache_engine_config(
    source_attestation: dict[str, object], engine_config: EngineConfig
) -> None:
    """Reject a required T64 claim before constructing a mismatched backend."""
    claim = source_attestation.get("startup_cache_attestation")
    if claim is None:
        if source_attestation.get("status") == "not_required":
            return
        raise RuntimeError("hardened runtime startup cache claim is absent")
    if not isinstance(claim, dict):
        raise RuntimeError("runtime startup cache claim has an invalid type")
    requirement = claim.get("status")
    if requirement == "not_required":
        if claim != {"status": "not_required"}:
            raise RuntimeError("runtime cache opt-out claim has unexpected fields")
        return
    if requirement != "required-v1":
        raise RuntimeError("runtime startup cache requirement is invalid")
    seed = claim.get("seed")
    if not isinstance(seed, dict):
        raise RuntimeError("required runtime startup cache seed is absent")
    memory_mode = source_attestation.get("memory_mode")
    if memory_mode not in {"growth", "preallocate85"}:
        raise RuntimeError("required runtime memory mode is invalid")
    expected_backend_config = {
        "max_lora_adapters": 2,
        "max_lora_rank": 8,
        "train_micro_batch_size": 1,
        "sample_max_num_sequences": 1,
        "gradient_checkpointing": True,
        "loss_chunk_size": 64,
        "qwen35_bf16_down_lora_residual": True,
        "abstract_model_load": memory_mode == "preallocate85",
    }
    if (
        engine_config.backend != "jax"
        or engine_config.backend_config != expected_backend_config
        or str(engine_config.base_model) != seed.get("model_path")
        or engine_config.external_inference_url is not None
    ):
        raise RuntimeError(
            "required T64 cache attestation does not match the exact JAX engine config"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    runtime_source_attestation = revalidate_runtime_source(
        initial=_RUNTIME_SOURCE_ATTESTATION,
        role="api",
        module_file=Path(__file__),
        package_file=Path(_skyrl_package.__file__),
    )
    runtime_launch_lock_attestation = revalidate_runtime_launch_lock(
        _RUNTIME_LAUNCH_LOCK_ATTESTATION
    )
    app.state.runtime_source_attestation = runtime_source_attestation
    app.state.runtime_launch_lock_attestation = runtime_launch_lock_attestation
    logger.info("API runtime source attestation: %s", runtime_source_attestation)
    logger.info(
        "API runtime launch-lock attestation: %s",
        runtime_launch_lock_attestation,
    )
    engine_config: EngineConfig = app.state.engine_config
    if engine_config.startup_launch_id is not None:
        raise RuntimeError(
            "startup_launch_id is internal; API configuration must leave it unset"
        )
    _validate_required_startup_cache_engine_config(
        runtime_source_attestation, engine_config
    )

    db_url = get_async_database_url(engine_config.database_url)
    db_engine = create_async_engine(db_url, echo=False)
    enable_sqlite_wal(db_engine.sync_engine)
    app.state.db_engine = db_engine
    app.state.external_inference_client = None
    app.state.background_engine = None
    app.state.engine_exit_task = None
    app.state.engine_launch_id = None
    app.state.engine_launch_identities = None
    app.state.engine_process_group_id = None

    api_identity: ProcessIdentity | None = None
    launcher_identity: ProcessIdentity | None = None
    launch_id: str | None = None
    launch_created = False
    startup_ready = False
    shutting_down = False
    unexpected_engine_exit = False
    lifecycle_failed = False
    background_engine: asyncio.subprocess.Process | None = None
    engine_exit_task: asyncio.Task[int] | None = None
    engine_process_group_id: int | None = None
    monitor_task: asyncio.Task[None] | None = None
    force_exit_timer: threading.Timer | None = None
    startup_error = "API startup did not complete"

    try:
        async with db_engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        # Setup external inference client if configured. The colocated path
        # stays on the engine because only its synchronous sample path knows
        # how to wake an inference worker that sleeps during training.
        backend_name = engine_config.backend
        backend_cfg = engine_config.backend_config or {}
        is_colocated = bool(
            backend_cfg.get("trainer.placement.colocate_all", True)
        )
        if engine_config.external_inference_url:
            app.state.external_inference_client = ExternalInferenceClient(
                engine_config, db_engine
            )
            logger.info(
                "External engine configured: %s",
                engine_config.external_inference_url,
            )
        elif backend_name in ("megatron", "fsdp") and not is_colocated:
            app.state.external_inference_client = (
                SkyRLTrainInferenceForwardingClient(engine_config, db_engine)
            )
            logger.info(
                "SkyRL-Train inference forwarding client enabled for "
                "non-colocated backend=%s",
                backend_name,
            )
        else:
            logger.info("Using internal engine for inference")

        api_identity = read_process_identity()
        launch_id = new_engine_launch_id()
        await _insert_engine_launch(
            db_engine,
            launch_id=launch_id,
            engine_config=engine_config,
            api_identity=api_identity,
            source_attestation=runtime_source_attestation,
            lock_attestation=runtime_launch_lock_attestation,
        )
        launch_created = True
        child_config = engine_config.model_copy(
            update={"startup_launch_id": launch_id}
        )
        engine_startup_deadline = time.monotonic() + float(
            engine_config.engine_startup_timeout_sec
        )

        parent_cmd = psutil.Process(os.getppid()).cmdline()
        cmd = _build_uv_run_cmd_engine(
            parent_cmd,
            child_config,
            runtime_source_attestation=runtime_source_attestation,
        )
        subprocess_options: dict[str, Any] = {}
        subprocess_options["start_new_session"] = True
        if runtime_launch_lock_attestation["status"] == "passed":
            subprocess_options["pass_fds"] = (
                runtime_launch_lock_attestation["descriptor"],
            )
        background_engine = await asyncio.create_subprocess_exec(
            *cmd, **subprocess_options
        )
        app.state.background_engine = background_engine
        engine_exit_task = asyncio.create_task(background_engine.wait())
        app.state.engine_exit_task = engine_exit_task
        engine_process_group_id = background_engine.pid
        app.state.engine_process_group_id = engine_process_group_id
        if os.getpgid(background_engine.pid) != background_engine.pid:
            raise RuntimeError("engine launcher did not enter a dedicated process group")
        logger.info(
            "Started background engine launcher with PID %d for launch %s "
            "(backend=%s)",
            background_engine.pid,
            launch_id,
            backend_name,
        )

        launcher_identity = read_process_identity(background_engine.pid)
        bind_timeout = engine_startup_deadline - time.monotonic()
        if bind_timeout <= 0:
            raise TimeoutError("engine startup deadline expired before launcher bind")
        await asyncio.wait_for(
            _bind_engine_launcher(
                db_engine,
                launch_id=launch_id,
                api_identity=api_identity,
                launcher_identity=launcher_identity,
            ),
            timeout=bind_timeout,
        )
        readiness_timeout = engine_startup_deadline - time.monotonic()
        if readiness_timeout <= 0:
            raise TimeoutError("engine startup deadline expired before readiness wait")
        identities = await _wait_for_engine_ready(
            db_engine,
            launch_id=launch_id,
            backend=backend_name,
            api_identity=api_identity,
            launcher_identity=launcher_identity,
            expected_api_source_attestation=runtime_source_attestation,
            expected_api_lock_attestation=runtime_launch_lock_attestation,
            background_engine=background_engine,
            engine_exit_task=engine_exit_task,
            timeout_sec=readiness_timeout,
        )
        app.state.engine_launch_id = launch_id
        app.state.engine_launch_identities = identities

        async def monitor_engine() -> None:
            """Fail this launch on child exit or stale operational readiness."""
            nonlocal force_exit_timer, startup_error, unexpected_engine_exit
            failure_reason: str | None = None
            database_failure_started: float | None = None
            while not shutting_down:
                poll_task = asyncio.create_task(asyncio.sleep(1.0))
                try:
                    done, _ = await asyncio.wait(
                        {engine_exit_task, poll_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                except BaseException:
                    poll_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await poll_task
                    raise
                if engine_exit_task in done:
                    if not poll_task.done():
                        poll_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await poll_task
                    exit_code = await asyncio.shield(engine_exit_task)
                    failure_reason = (
                        f"background engine exited with code {exit_code}"
                    )
                    break
                await poll_task
                async def read_monitor_record() -> EngineLaunchDB | None:
                    async with AsyncSession(db_engine) as session:
                        return await session.get(EngineLaunchDB, launch_id)

                try:
                    record = await asyncio.wait_for(
                        read_monitor_record(),
                        timeout=STARTUP_DB_POLL_TIMEOUT_SECONDS,
                    )
                except Exception as error:
                    now = time.monotonic()
                    if database_failure_started is None:
                        database_failure_started = now
                    if (
                        now - database_failure_started
                        <= READINESS_DB_FAILURE_GRACE_SECONDS
                    ):
                        logger.warning(
                            "Engine readiness DB monitor retry after transient "
                            "failure: %s",
                            error,
                        )
                        continue
                    failure_reason = (
                        "engine readiness DB monitor exceeded its retry window: "
                        f"{error}"
                    )
                    break
                database_failure_started = None
                try:
                    current_identities = validate_ready_engine_launch(
                        record,
                        launch_id=launch_id,
                        backend=backend_name,
                        api_identity=api_identity,
                        launcher_identity=launcher_identity,
                        expected_api_source_attestation=(
                            runtime_source_attestation
                        ),
                        expected_api_lock_attestation=(
                            runtime_launch_lock_attestation
                        ),
                    )
                    if current_identities != identities:
                        raise RuntimeError(
                            "engine launch identities changed after readiness"
                        )
                except Exception as error:
                    failure_reason = f"engine readiness monitor failed: {error}"
                    break
            if shutting_down or failure_reason is None:
                return
            unexpected_engine_exit = True
            startup_error = failure_reason
            logger.error("%s; exiting API server", failure_reason)
            if engine_process_group_id is not None and (
                _identity_owns_engine_process_group(
                    launcher_identity, engine_process_group_id
                )
                or _identity_owns_engine_process_group(
                    identities["engine"], engine_process_group_id
                )
            ):
                try:
                    _signal_engine_process_group(
                        engine_process_group_id, signal.SIGTERM
                    )
                except Exception:
                    logger.exception(
                        "Failed to signal unhealthy engine process group"
                    )
            try:
                await asyncio.wait_for(
                    _mark_engine_launch(
                        db_engine,
                        launch_id=launch_id,
                        api_identity=api_identity,
                        status=EngineLaunchStatus.FAILED,
                        expected_statuses=(EngineLaunchStatus.READY,),
                        error_message=failure_reason,
                    ),
                    timeout=TERMINAL_STATE_WRITE_TIMEOUT_SECONDS,
                )
            except Exception:
                logger.exception("Failed to persist unexpected engine exit")

            # Uvicorn can wait for active tasks after SIGTERM. A daemon timer
            # retains the existing hard upper bound for that graceful path.
            def force_exit() -> None:
                logger.warning("Graceful shutdown timed out, forcing exit")
                os._exit(1)

            timer = threading.Timer(SHUTDOWN_TIMEOUT_SECONDS, force_exit)
            timer.daemon = True
            timer.start()
            force_exit_timer = timer
            os.kill(os.getpid(), signal.SIGTERM)

        monitor_task = asyncio.create_task(monitor_engine())
        startup_ready = True
        logger.info(
            "Accepted READY engine launch %s (launcher PID %d, engine PID %d)",
            launch_id,
            launcher_identity.pid,
            identities["engine"].pid,
        )
        yield
    except BaseException as error:
        lifecycle_failed = True
        startup_error = f"{type(error).__name__}: {error}"
        raise
    finally:
        if (
            startup_ready
            and engine_exit_task is not None
            and background_engine is not None
            and (
                engine_exit_task.done()
                or background_engine.returncode is not None
            )
        ):
            unexpected_engine_exit = True
            exit_code = (
                engine_exit_task.result()
                if engine_exit_task.done()
                else background_engine.returncode
            )
            startup_error = f"background engine exited with code {exit_code}"
        shutting_down = True
        if monitor_task is not None:
            monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await monitor_task

        accepted_identities = app.state.engine_launch_identities
        accepted_engine_identity = (
            None
            if accepted_identities is None
            else accepted_identities.get("engine")
        )
        app.state.engine_launch_identities = None
        engine_stop_task: asyncio.Task[_EngineStopResult] | None = None
        group_ownership_error = False
        if background_engine is not None and engine_exit_task is not None:
            owned_process_group_id = None
            if engine_process_group_id is not None and (
                _identity_owns_engine_process_group(
                    launcher_identity, engine_process_group_id
                )
                or _identity_owns_engine_process_group(
                    accepted_engine_identity, engine_process_group_id
                )
            ):
                owned_process_group_id = engine_process_group_id
            elif engine_process_group_id is not None:
                try:
                    group_ownership_error = _engine_process_group_exists(
                        engine_process_group_id
                    )
                except Exception:
                    group_ownership_error = True
                    logger.exception("Could not validate engine process-group ownership")
            engine_stop_task = asyncio.create_task(
                _stop_background_engine(
                    background_engine,
                    engine_exit_task,
                    owned_process_group_id,
                )
            )
            # Let the stop task deliver TERM before any potentially blocked DB
            # or client cleanup operation begins.
            await asyncio.sleep(0)

        normal_shutdown = (
            startup_ready
            and not unexpected_engine_exit
            and not lifecycle_failed
        )
        if (
            not normal_shutdown
            and launch_created
            and launch_id is not None
            and api_identity is not None
        ):
            try:
                marked = await asyncio.wait_for(
                    _mark_engine_launch(
                        db_engine,
                        launch_id=launch_id,
                        api_identity=api_identity,
                        status=EngineLaunchStatus.FAILED,
                        expected_statuses=(
                            EngineLaunchStatus.STARTING,
                            EngineLaunchStatus.INITIALIZING,
                            EngineLaunchStatus.READY,
                        ),
                        error_message=startup_error,
                    ),
                    timeout=TERMINAL_STATE_WRITE_TIMEOUT_SECONDS,
                )
                if not marked:
                    logger.warning(
                        "Engine launch %s was already terminal during cleanup",
                        launch_id,
                    )
            except Exception:
                logger.exception("Failed to persist terminal engine launch state")

        inference_client = getattr(
            app.state, "external_inference_client", None
        )
        aclose = getattr(inference_client, "aclose", None)
        if aclose is not None:
            with suppress(Exception):
                await asyncio.wait_for(
                    aclose(), timeout=TERMINAL_STATE_WRITE_TIMEOUT_SECONDS
                )

        stop_error: str | None = (
            "engine process-group ownership could not be verified"
            if group_ownership_error
            else None
        )
        shutdown_result: _EngineStopResult | None = None
        if engine_stop_task is not None and background_engine is not None:
            logger.info(
                "Stopping background engine launcher (PID %d)",
                background_engine.pid,
            )
            try:
                shutdown_result = await asyncio.shield(engine_stop_task)
                logger.info(
                    "Background engine launcher stopped with code %d",
                    shutdown_result.exit_code,
                )
            except Exception:
                stop_error = "background engine could not be stopped and reaped"
                logger.exception("Failed to stop or reap background engine launcher")
        if (
            normal_shutdown
            and stop_error is None
            and shutdown_result is not None
            and not _expected_engine_shutdown_result(shutdown_result)
        ):
            stop_error = (
                "background engine exited unexpectedly during shutdown with code "
                f"{shutdown_result.exit_code}"
            )
        if (
            normal_shutdown
            and launch_created
            and launch_id is not None
            and api_identity is not None
        ):
            try:
                marked = await asyncio.wait_for(
                    _mark_engine_launch(
                        db_engine,
                        launch_id=launch_id,
                        api_identity=api_identity,
                        status=(
                            EngineLaunchStatus.STOPPED
                            if stop_error is None
                            else EngineLaunchStatus.FAILED
                        ),
                        expected_statuses=(EngineLaunchStatus.READY,),
                        error_message=stop_error,
                    ),
                    timeout=TERMINAL_STATE_WRITE_TIMEOUT_SECONDS,
                )
                if not marked:
                    logger.warning(
                        "Engine launch %s was already terminal after shutdown",
                        launch_id,
                    )
            except Exception:
                logger.exception("Failed to persist final engine launch state")
        try:
            await asyncio.wait_for(
                db_engine.dispose(), timeout=TERMINAL_STATE_WRITE_TIMEOUT_SECONDS
            )
        except Exception:
            logger.exception("Failed to dispose API database engine cleanly")
        if force_exit_timer is not None:
            force_exit_timer.cancel()


app = FastAPI(title="Tinker API Mock", version="0.0.1", lifespan=lifespan)


async def get_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get a database session."""
    async with AsyncSession(request.app.state.db_engine) as session:
        yield session


async def get_model(session: AsyncSession, model_id: str) -> ModelDB:
    """Fetch a model by ID, raising 404 if not found."""
    statement = select(ModelDB).where(ModelDB.model_id == model_id)
    result = await session.exec(statement)
    model = result.first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


async def create_future(
    session: AsyncSession,
    request_type: types.RequestType,
    model_id: str | None,
    request_data: BaseModel,
) -> int:
    """Create a FutureDB entry and return its auto-generated request_id."""
    future_db = FutureDB(
        request_type=request_type,
        model_id=model_id,
        request_data=request_data.model_dump(mode="json"),
        status=RequestStatus.PENDING,
    )
    session.add(future_db)
    await session.flush()  # Flush to generate auto-increment request_id
    assert future_db.request_id
    return future_db.request_id


def _sequence_request_identity(scope: str, sequence_id: int | None) -> str | None:
    if sequence_id is None:
        return None
    encoded = json.dumps(
        {
            "scope": scope,
            "sequence_id": sequence_id,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _http_request_identity(scope: str, request: Request) -> str | None:
    values = request.headers.getlist("x-idempotency-key")
    if not values:
        return None
    if len(values) != 1 or not values[0] or len(values[0].encode("utf-8")) > 512:
        raise HTTPException(
            status_code=400,
            detail="X-Idempotency-Key must be one nonempty bounded value",
        )
    encoded = json.dumps(
        {"scope": scope, "idempotency_key": values[0]},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _request_payload_sha256(
    request: BaseModel, *, ignored_fields: frozenset[str] = frozenset()
) -> str:
    payload = request.model_dump(mode="json")
    for field in ignored_fields:
        payload.pop(field, None)
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=422,
            detail="request payload is not canonically serializable",
        ) from error
    return hashlib.sha256(encoded).hexdigest()


async def _load_deduplicated_response(
    session: AsyncSession,
    *,
    request_key: str | None,
    request_type: str,
    payload_sha256: str,
) -> dict[str, object] | None:
    if request_key is None:
        return None
    record = await session.get(RequestDedupDB, request_key)
    if record is None:
        return None
    if (
        record.request_type != request_type
        or record.payload_sha256 != payload_sha256
    ):
        # 409 is retryable for five minutes in the supported Tinker SDK.
        raise HTTPException(
            status_code=400,
            detail="request sequence identity conflicts with a prior payload",
        )
    if record.response_data is None:
        raise HTTPException(
            status_code=503,
            detail="request sequence identity has no committed response",
        )
    return dict(record.response_data)


async def _reserve_deduplicated_request(
    session: AsyncSession,
    *,
    request_key: str | None,
    request_type: str,
    payload_sha256: str,
) -> dict[str, object] | None:
    if request_key is None:
        return None
    # Insert first: on SQLite/WAL a SELECT-then-INSERT transaction can fail its
    # read-to-write upgrade with SQLITE_BUSY_SNAPSHOT under concurrent retries.
    session.add(
        RequestDedupDB(
            request_key=request_key,
            request_type=request_type,
            payload_sha256=payload_sha256,
            response_data=None,
        )
    )
    try:
        await session.flush()
    except IntegrityError as error:
        await session.rollback()
        replay = await _load_deduplicated_response(
            session,
            request_key=request_key,
            request_type=request_type,
            payload_sha256=payload_sha256,
        )
        if replay is None:
            raise HTTPException(
                status_code=503,
                detail="request sequence identity could not be reserved",
            ) from error
        return replay
    return None


async def _commit_deduplicated_response(
    session: AsyncSession,
    *,
    request_key: str | None,
    request_type: str,
    payload_sha256: str,
    response_data: dict[str, object],
) -> tuple[dict[str, object], bool]:
    if request_key is None:
        await session.commit()
        return response_data, False
    record = await session.get(RequestDedupDB, request_key)
    if record is None or (
        record.request_type != request_type
        or record.payload_sha256 != payload_sha256
        or record.response_data is not None
    ):
        await session.rollback()
        raise HTTPException(
            status_code=503,
            detail="request sequence reservation changed before commit",
        )
    record.response_data = response_data
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise
    return response_data, False


async def create_checkpoint(
    session: AsyncSession,
    model_id: str,
    checkpoint_id: str,
    checkpoint_type: types.CheckpointType,
):
    """Create a pending CheckpointDB entry, relying on database constraints for validation."""
    checkpoint_db = CheckpointDB(
        model_id=model_id,
        checkpoint_id=checkpoint_id,
        checkpoint_type=checkpoint_type,
        status=CheckpointStatus.PENDING,
    )
    session.add(checkpoint_db)

    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        # Determine which constraint failed by checking if the model exists
        statement = select(ModelDB).where(ModelDB.model_id == model_id)
        result = await session.exec(statement)

        if not result.first():
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Checkpoint '{checkpoint_id}' already exists for model '{model_id}'",
            )


class LoRAConfig(BaseModel):
    rank: int
    seed: int | None = Field(
        default=None, description="Seed for LoRA weight initialization. If None, a random seed is used."
    )
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True


class CreateModelRequest(BaseModel):
    session_id: str
    model_seq_id: int = Field(ge=0)
    base_model: str
    lora_config: LoRAConfig
    user_metadata: dict[str, Any] | None = None
    model_role: str = "policy"
    type: Literal["create_model"] = "create_model"


class CreateModelResponse(BaseModel):
    model_id: str
    base_model: str
    lora_config: LoRAConfig | None = None
    status: str = "created"
    request_id: str


class UnloadModelRequest(BaseModel):
    model_id: str
    type: Literal["unload_model"] = "unload_model"


class UnloadModelResponse(BaseModel):
    request_id: str
    model_id: str


class ModelData(BaseModel):
    base_model: str
    lora_config: LoRAConfig | None = None
    model_name: str | None = None


class ModelInfoResponse(BaseModel):
    model_id: str
    status: str
    model_data: ModelData


class Checkpoint(BaseModel):
    checkpoint_id: str
    checkpoint_type: Literal["training", "sampler"]
    time: datetime
    tinker_path: str


class TrainingRun(BaseModel):
    training_run_id: str
    base_model: str
    model_owner: str = "default"
    is_lora: bool = True
    corrupted: bool = False
    lora_rank: int | None = None
    last_request_time: datetime
    last_checkpoint: Checkpoint | None = None
    last_sampler_checkpoint: Checkpoint | None = None
    user_metadata: dict[str, str] | None = None


class EncodedTextChunk(BaseModel):
    type: Literal["encoded_text"] = "encoded_text"
    tokens: list[int]

    def to_types(self) -> types.EncodedTextChunk:
        return types.EncodedTextChunk(tokens=self.tokens)


class ImageChunk(BaseModel):
    type: Literal["image"] = "image"
    data: Base64Bytes
    format: Literal["png", "jpeg"]
    expected_tokens: int | None = None

    def to_types(self) -> types.ImageChunk:
        return types.ImageChunk.model_construct(
            data=self.data,
            format=self.format,
            expected_tokens=self.expected_tokens,
        )


class ImageAssetPointerChunk(BaseModel):
    type: Literal["image_asset_pointer"] = "image_asset_pointer"
    format: Literal["png", "jpeg"]
    location: str = Field(min_length=1)
    expected_tokens: int | None = None

    def to_types(self) -> types.ImageAssetPointerChunk:
        return types.ImageAssetPointerChunk(
            format=self.format,
            location=self.location,
            expected_tokens=self.expected_tokens,
        )


def _get_model_chunk_type(v: Any) -> str:
    if isinstance(v, dict):
        if "type" in v:
            return v["type"]
        is_encoded_text = "tokens" in v
        is_image_asset_pointer = "location" in v
        is_image = "data" in v

        if sum([is_encoded_text, is_image_asset_pointer, is_image]) > 1:
            raise ValueError(
                "Ambiguous model chunk type: must be exactly one of 'encoded_text', 'image_asset_pointer', or 'image'"
            )
        if is_encoded_text:
            return "encoded_text"
        if is_image_asset_pointer:
            return "image_asset_pointer"
        if is_image:
            return "image"
    return getattr(v, "type", "encoded_text")


ModelInputChunk = Annotated[
    Annotated[EncodedTextChunk, Tag("encoded_text")]
    | Annotated[ImageAssetPointerChunk, Tag("image_asset_pointer")]
    | Annotated[ImageChunk, Tag("image")],
    Discriminator(_get_model_chunk_type),
]


class ModelInput(BaseModel):
    chunks: list[ModelInputChunk]

    def to_types(self) -> types.ModelInput:
        return types.ModelInput(chunks=[chunk.to_types() for chunk in self.chunks])


class TensorData(BaseModel):
    data: list[int] | list[float]

    def to_types(self) -> types.TensorData:
        return types.TensorData(data=self.data)


class Datum(BaseModel):
    loss_fn_inputs: dict[str, TensorData]
    model_input: ModelInput

    def to_types(self) -> types.Datum:
        inp = self.loss_fn_inputs

        if "weights" not in inp:
            weights = types.TensorData(data=[1.0] * len(inp["target_tokens"].data))
        else:
            weights = inp["weights"].to_types()

        return types.Datum(
            loss_fn_inputs=types.LossFnInputs(
                target_tokens=inp["target_tokens"].to_types(),
                weights=weights,
                advantages=inp["advantages"].to_types() if "advantages" in inp else types.TensorData(data=[]),
                logprobs=inp["logprobs"].to_types() if "logprobs" in inp else types.TensorData(data=[]),
                values=inp["values"].to_types() if "values" in inp else types.TensorData(data=[]),
                returns=inp["returns"].to_types() if "returns" in inp else types.TensorData(data=[]),
            ),
            model_input=self.model_input.to_types(),
        )


class ForwardBackwardInput(BaseModel):
    _ALLOWED_KEYS_BY_LOSS_FN: ClassVar[dict[str, set[str]]] = {
        "cross_entropy": set(),
        "importance_sampling": set(),
        "ppo": {"clip_low_threshold", "clip_high_threshold", "value_clip"},
        "cispo": {"clip_low_threshold", "clip_high_threshold"},
        "ppo_critic": {"value_clip"},
        "dppo": {"delta_low", "delta_high"},
    }

    data: list[Datum]
    loss_fn: Literal["cross_entropy", "importance_sampling", "ppo", "cispo", "ppo_critic", "dppo"]
    loss_fn_config: dict[str, float] | None = None

    @model_validator(mode="after")
    def validate_loss_fn_config_keys(self):
        """Validate loss_fn_config keys based on the selected loss function."""
        if self.loss_fn_config is None:
            return self

        allowed_keys = self._ALLOWED_KEYS_BY_LOSS_FN[self.loss_fn]
        invalid_keys = sorted(set(self.loss_fn_config.keys()) - allowed_keys)
        if invalid_keys:
            if allowed_keys:
                raise ValueError(
                    f"Invalid loss_fn_config keys for loss_fn='{self.loss_fn}': {invalid_keys}. "
                    f"Allowed keys: {sorted(allowed_keys)}."
                )
            raise ValueError(
                f"loss_fn='{self.loss_fn}' does not accept loss_fn_config keys. " f"Received: {invalid_keys}."
            )
        return self

    def to_types(self) -> types.ForwardBackwardInput:
        return types.ForwardBackwardInput(
            data=[datum.to_types() for datum in self.data],
            loss_fn=self.loss_fn,
            loss_fn_config=self.loss_fn_config,
        )


class ForwardBackwardRequest(BaseModel):
    model_id: str
    forward_backward_input: ForwardBackwardInput
    seq_id: int = Field(ge=0)


class ForwardRequest(BaseModel):
    model_id: str
    forward_input: ForwardBackwardInput
    seq_id: int = Field(ge=0)


class AdamParams(BaseModel):
    learning_rate: float = Field(default=1e-4, ge=0.0)
    beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    beta2: float = Field(default=0.95, ge=0.0, lt=1.0)
    eps: float = Field(default=1e-12, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    grad_clip_norm: Literal[0.0] = 0.0

    def to_types(self) -> types.AdamParams:
        return types.AdamParams(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )


class OptimStepRequest(BaseModel):
    model_id: str
    adam_params: AdamParams
    seq_id: int = Field(ge=0)


class SaveWeightsForSamplerRequest(BaseModel):
    model_id: str
    path: str | None = Field(default=None, pattern=ID_PATTERN, max_length=ID_MAX_LENGTH)
    sampling_session_seq_id: int | None = Field(default=None, ge=0)
    seq_id: int = Field(ge=0)
    type: Literal["save_weights_for_sampler"] = "save_weights_for_sampler"

    @model_validator(mode="after")
    def check_path_or_ids(self):
        if not self.path and (self.sampling_session_seq_id is None or self.seq_id is None):
            raise ValueError("Either 'path' or both 'sampling_session_seq_id' and 'seq_id' must be provided")
        return self


class SamplingParams(BaseModel):
    max_tokens: int | None = None
    seed: int | None = None
    stop: list[int] | list[str] | None = None
    temperature: float = 1
    top_k: int = -1
    top_p: float = 1

    def to_types(self) -> types.SamplingParams:
        if self.max_tokens is None:
            raise HTTPException(status_code=400, detail="max_tokens is currently required")
        if self.max_tokens <= 0:
            raise HTTPException(status_code=400, detail="max_tokens must be a positive number")

        # Generate a random seed if not provided
        seed = self.seed if self.seed is not None else random.randint(0, 2**31 - 1)

        # Determine if stop values are token IDs (int) or strings
        stop_tokens = None
        stop_strings = None
        if self.stop:
            if all(isinstance(s, int) for s in self.stop):
                stop_tokens = list(self.stop)
            elif all(isinstance(s, str) for s in self.stop):
                stop_strings = list(self.stop)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="stop must be either all integers (token IDs) or all strings, not mixed",
                )

        return types.SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=seed,
            stop_tokens=stop_tokens,
            stop_strings=stop_strings,
            top_k=self.top_k,
            top_p=self.top_p,
        )


class SampleRequest(BaseModel):
    num_samples: int = 1
    prompt: ModelInput
    sampling_params: SamplingParams
    base_model: str | None = None
    model_path: str | None = None
    sampling_session_id: str | None = None
    seq_id: int | None = Field(default=None, ge=0)
    prompt_logprobs: bool | None = None
    topk_prompt_logprobs: int = 0
    type: Literal["sample"] = "sample"

    @model_validator(mode="after")
    def validate_model_source(self):
        """Valid if:
        - sampling_session_id is provided AND seq_id is provided
        - OR exactly one of base_model or model_path is provided
        """
        if self.sampling_session_id is not None:
            if self.seq_id is None:
                raise ValueError("'seq_id' must be provided when 'sampling_session_id' is used")
            return self
        if (self.base_model is None) == (self.model_path is None):
            raise ValueError(
                "When 'sampling_session_id' is not provided, exactly one of 'base_model' or 'model_path' must be provided"
            )
        return self


class SaveWeightsRequest(BaseModel):
    model_id: str
    path: str = Field(..., pattern=ID_PATTERN, max_length=ID_MAX_LENGTH)
    seq_id: int = Field(ge=0)
    type: Literal["save_weights"] = "save_weights"


class LoadWeightsRequest(BaseModel):
    model_id: str
    path: str
    optimizer: bool | None = None
    seq_id: int = Field(ge=0)
    type: Literal["load_weights"] = "load_weights"


class FutureResponse(BaseModel):
    future_id: str
    status: str = "pending"
    request_id: str


class TelemetryEvent(BaseModel):
    event: str
    event_id: str
    event_session_index: int
    severity: str
    timestamp: str
    properties: dict[str, Any] | None = None


class TelemetryRequest(BaseModel):
    events: list[TelemetryEvent]
    platform: str
    sdk_version: str
    session_id: str


class TelemetryResponse(BaseModel):
    status: Literal["accepted"] = "accepted"


class HealthResponse(BaseModel):
    status: Literal["ok"]


class CreateSessionRequest(BaseModel):
    tags: list[str]
    user_metadata: dict[str, Any] | None = None
    sdk_version: str
    type: Literal["create_session"] = "create_session"


class CreateSessionResponse(BaseModel):
    type: Literal["create_session"] = "create_session"
    info_message: str | None = None
    warning_message: str | None = None
    error_message: str | None = None
    session_id: str


class SessionHeartbeatRequest(BaseModel):
    session_id: str
    type: Literal["session_heartbeat"] = "session_heartbeat"


class SessionHeartbeatResponse(BaseModel):
    type: Literal["session_heartbeat"] = "session_heartbeat"


class CreateSamplingSessionRequest(BaseModel):
    session_id: str
    sampling_session_seq_id: int
    base_model: str | None = None
    model_path: str | None = None
    type: Literal["create_sampling_session"] = "create_sampling_session"


class CreateSamplingSessionResponse(BaseModel):
    type: Literal["create_sampling_session"] = "create_sampling_session"
    sampling_session_id: str


class GetSamplerResponse(BaseModel):
    sampler_id: str
    base_model: str
    model_path: str | None = None


class SupportedModel(BaseModel):
    model_name: str


class GetServerCapabilitiesResponse(BaseModel):
    supported_models: list[SupportedModel]


class ListCheckpointsResponse(BaseModel):
    checkpoints: list[Checkpoint]


class Cursor(BaseModel):
    offset: int
    limit: int
    total_count: int


class TrainingRunsResponse(BaseModel):
    training_runs: list[TrainingRun]
    cursor: Cursor


class WeightsInfoRequest(BaseModel):
    tinker_path: str


class WeightsInfoResponse(BaseModel):
    """Minimal information for loading public checkpoints."""

    # from: https://github.com/thinking-machines-lab/tinker/blob/main/src/tinker/types/weights_info_response.py
    base_model: str
    is_lora: bool
    lora_rank: int | None = None


class ClientConfigResponse(BaseModel):
    pjwt_auth_enabled: bool = False
    # Holder-level submit retries retain the same sample seq_id and are
    # deduplicated server-side.  Disabling the outer whole-operation retry
    # prevents the SDK from allocating a new seq_id after an ambiguous result.
    sample_no_retries: Literal[True] = True


@app.post("/api/v1/client/config", response_model=ClientConfigResponse)
async def client_config():
    """Stub for tinker SDK client_config handshake."""
    return ClientConfigResponse()


@app.get("/api/v1/healthz", response_model=HealthResponse)
async def healthz(request: Request):
    """Return healthy only for the exact accepted and still-live launch."""
    try:
        launch_id = request.app.state.engine_launch_id
        accepted_identities = request.app.state.engine_launch_identities
        background_engine = request.app.state.background_engine
        engine_exit_task = request.app.state.engine_exit_task
        if (
            launch_id is None
            or accepted_identities is None
            or background_engine is None
            or engine_exit_task is None
            or engine_exit_task.done()
            or background_engine.returncode is not None
        ):
            raise RuntimeError("no live engine launch has been accepted")
        async def read_health_record() -> EngineLaunchDB | None:
            async with AsyncSession(request.app.state.db_engine) as session:
                return await session.get(EngineLaunchDB, launch_id)

        record = await asyncio.wait_for(
            read_health_record(), timeout=STARTUP_DB_POLL_TIMEOUT_SECONDS
        )
        current_identities = validate_ready_engine_launch(
            record,
            launch_id=launch_id,
            backend=request.app.state.engine_config.backend,
            api_identity=accepted_identities["api"],
            launcher_identity=accepted_identities["engine_launcher"],
            expected_api_source_attestation=(
                request.app.state.runtime_source_attestation
            ),
            expected_api_lock_attestation=(
                request.app.state.runtime_launch_lock_attestation
            ),
        )
        if current_identities != accepted_identities:
            raise RuntimeError("ready engine identities changed after startup")
        if engine_exit_task.done() or background_engine.returncode is not None:
            raise RuntimeError("engine exited during health validation")
    except Exception as error:
        logger.warning("Engine health validation failed: %s", error)
        raise HTTPException(
            status_code=503,
            detail="engine unavailable",
        ) from error
    return HealthResponse(status="ok")


@app.post("/api/v1/create_session", response_model=CreateSessionResponse)
async def create_session(
    request: CreateSessionRequest,
    req: Request,
    session: AsyncSession = Depends(get_session),
):
    """Create a new session + persist in DB"""
    request_type = "create_session"
    request_key = _http_request_identity("create-session", req)
    payload_sha256 = _request_payload_sha256(request)
    replay = await _reserve_deduplicated_request(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
    )
    if replay is not None:
        return CreateSessionResponse.model_validate(replay)
    session_id = f"session_{uuid4().hex[:8]}"
    session_db = SessionDB(
        session_id=session_id,
        tags=request.tags,
        user_metadata=request.user_metadata or {},
        sdk_version=request.sdk_version,
        status="active",
    )
    session.add(session_db)
    response = CreateSessionResponse(session_id=session_id)
    committed, _ = await _commit_deduplicated_response(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
        response_data=response.model_dump(mode="json"),
    )
    return CreateSessionResponse.model_validate(committed)


@app.post("/api/v1/session_heartbeat", response_model=SessionHeartbeatResponse)
async def session_heartbeat(request: SessionHeartbeatRequest, session: AsyncSession = Depends(get_session)):
    """Heartbeat for an active session to keep it alive."""
    session_db = await session.get(SessionDB, request.session_id)
    if session_db is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session_db.last_heartbeat_at = datetime.now(timezone.utc)
    session_db.heartbeat_count += 1
    await session.commit()
    return SessionHeartbeatResponse()


@app.post("/api/v1/create_sampling_session", response_model=CreateSamplingSessionResponse)
async def create_sampling_session(request: CreateSamplingSessionRequest, session: AsyncSession = Depends(get_session)):
    """Create a new sampling session within an existing session."""
    request_type = "create_sampling_session"
    request_key = _sequence_request_identity(
        f"sampling-client:{request.session_id}",
        request.sampling_session_seq_id,
    )
    payload_sha256 = _request_payload_sha256(request)
    replay = await _reserve_deduplicated_request(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
    )
    if replay is not None:
        return CreateSamplingSessionResponse.model_validate(replay)
    session_db = await session.get(SessionDB, request.session_id)
    if session_db is None:
        raise HTTPException(status_code=404, detail="Session not found")
    # Exactly one of base_model or model_path must be provided
    if (request.base_model is None) == (request.model_path is None):
        raise HTTPException(status_code=400, detail="Exactly one of base_model or model_path must be provided")
    sampling_session_id = f"sampling_{uuid4().hex[:8]}"
    sampling_db = SamplingSessionDB(
        sampling_session_id=sampling_session_id,
        session_id=request.session_id,
        sampling_session_seq_id=request.sampling_session_seq_id,
        base_model=request.base_model,
        model_path=request.model_path,
    )
    session.add(sampling_db)
    response = CreateSamplingSessionResponse(
        sampling_session_id=sampling_session_id
    )
    committed, _ = await _commit_deduplicated_response(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
        response_data=response.model_dump(mode="json"),
    )
    return CreateSamplingSessionResponse.model_validate(committed)


@app.get("/api/v1/samplers/{sampler_id}", response_model=GetSamplerResponse)
async def get_sampler(sampler_id: str, session: AsyncSession = Depends(get_session)):
    """Get sampler (sampling session) information."""
    sampling_db = await session.get(SamplingSessionDB, sampler_id)
    if sampling_db is None:
        raise HTTPException(status_code=404, detail="Sampler not found")
    if sampling_db.base_model is not None:
        base_model = sampling_db.base_model
    else:
        # Sampling session was created from a model_path — resolve the
        # underlying base model from the source training run so the SDK can
        # load the matching tokenizer.
        path = types.TinkerPath.parse(sampling_db.model_path)
        model = await get_model(session, path.primary_id)
        base_model = model.base_model
    return GetSamplerResponse(
        sampler_id=sampling_db.sampling_session_id,
        base_model=base_model,
        model_path=sampling_db.model_path,
    )


@app.post("/api/v1/create_model", response_model=CreateModelResponse)
async def create_model(request: CreateModelRequest, session: AsyncSession = Depends(get_session)):
    """Create a new model, optionally with a LoRA adapter."""
    request_type = types.RequestType.CREATE_MODEL.value
    request_key = _sequence_request_identity(
        f"training-client:{request.session_id}", request.model_seq_id
    )
    payload_sha256 = _request_payload_sha256(request)
    replay = await _reserve_deduplicated_request(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
    )
    if replay is not None:
        return CreateModelResponse.model_validate(replay)

    # Validate session exists
    session_db = await session.get(SessionDB, request.session_id)
    if session_db is None:
        raise HTTPException(status_code=404, detail="Session not found")

    model_id = f"model_{uuid4().hex[:8]}"

    # alpha = 32 seems to be the tinker default (see https://thinkingmachines.ai/blog/lora/)
    # Generate a random seed if not provided
    seed = request.lora_config.seed if request.lora_config.seed is not None else random.randint(0, 2**31 - 1)
    lora_config = types.LoraConfig(
        rank=request.lora_config.rank,
        alpha=32.0,
        seed=seed,
        train_mlp=request.lora_config.train_mlp,
        train_attn=request.lora_config.train_attn,
        train_unembed=request.lora_config.train_unembed,
    )
    request_id = await create_future(
        session=session,
        request_type=types.RequestType.CREATE_MODEL,
        model_id=model_id,
        request_data=types.CreateModelInput(lora_config=lora_config, model_role=request.model_role),
    )

    model_db = ModelDB(
        model_id=model_id,
        base_model=request.base_model,
        lora_config=lora_config.model_dump(),
        status="created",
        request_id=request_id,
        session_id=request.session_id,
    )
    session.add(model_db)

    response = CreateModelResponse(
        model_id=model_id,
        base_model=request.base_model,
        lora_config=request.lora_config,
        status="created",
        request_id=str(request_id),
    )
    committed, _ = await _commit_deduplicated_response(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
        response_data=response.model_dump(mode="json"),
    )
    return CreateModelResponse.model_validate(committed)


@app.post("/api/v1/unload_model", response_model=UnloadModelResponse)
async def unload_model(request: UnloadModelRequest, session: AsyncSession = Depends(get_session)):
    """Unload a model and free all associated resources."""
    request_type = types.RequestType.UNLOAD_MODEL.value
    request_key = _sequence_request_identity(
        f"model-lifecycle:{request.model_id}", 0
    )
    payload_sha256 = _request_payload_sha256(request)
    replay = await _reserve_deduplicated_request(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
    )
    if replay is not None:
        return UnloadModelResponse.model_validate(replay)
    # Validate model exists
    model_db = await session.get(ModelDB, request.model_id)
    if model_db is None:
        raise HTTPException(status_code=404, detail="Model not found")

    # Update model status
    model_db.status = "unloading"

    # Create future request
    request_id = await create_future(
        session=session,
        request_type=types.RequestType.UNLOAD_MODEL,
        model_id=request.model_id,
        request_data=types.UnloadModelInput(),
    )

    response = UnloadModelResponse(
        request_id=str(request_id), model_id=request.model_id
    )
    committed, _ = await _commit_deduplicated_response(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
        response_data=response.model_dump(mode="json"),
    )
    return UnloadModelResponse.model_validate(committed)


class GetInfoRequest(BaseModel):
    model_id: str
    type: str | None = None


@app.post("/api/v1/get_info", response_model=ModelInfoResponse)
async def get_model_info(request: GetInfoRequest, session: AsyncSession = Depends(get_session)):
    """Retrieve information about the current model."""
    model = await get_model(session, request.model_id)

    lora_config = types.LoraConfig.model_validate(model.lora_config)
    model_data = ModelData(
        base_model=model.base_model, lora_config=LoRAConfig(rank=lora_config.rank), model_name=model.base_model
    )

    return ModelInfoResponse(model_id=model.model_id, status=model.status, model_data=model_data)


@app.get("/api/v1/training_runs/{model_id}", response_model=TrainingRun)
async def get_training_run(model_id: str, session: AsyncSession = Depends(get_session)):
    """Get training run for session resumption."""
    model = await get_model(session, model_id)

    lora_config = types.LoraConfig.model_validate(model.lora_config)

    return TrainingRun(
        training_run_id=model.model_id,
        base_model=model.base_model,
        model_owner="default",
        is_lora=True,
        corrupted=False,
        lora_rank=lora_config.rank,
        # TODO: Once we track modified_at timestamps, update this
        last_request_time=model.created_at,
        last_checkpoint=None,
        last_sampler_checkpoint=None,
        user_metadata=None,
    )


@app.post("/api/v1/forward_backward", response_model=FutureResponse)
async def forward_backward(request: ForwardBackwardRequest, session: AsyncSession = Depends(get_session)):
    """Compute and accumulate gradients."""
    request_type = types.RequestType.FORWARD_BACKWARD.value
    request_key = _sequence_request_identity(
        f"training:{request.model_id}", request.seq_id
    )
    payload_sha256 = _request_payload_sha256(request)
    replay = await _reserve_deduplicated_request(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
    )
    if replay is not None:
        return FutureResponse.model_validate(replay)
    await get_model(session, request.model_id)

    request_id = await create_future(
        session=session,
        request_type=types.RequestType.FORWARD_BACKWARD,
        model_id=request.model_id,
        request_data=request.forward_backward_input.to_types(),
    )

    response = FutureResponse(
        future_id=str(request_id), status="pending", request_id=str(request_id)
    )
    committed, _ = await _commit_deduplicated_response(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
        response_data=response.model_dump(mode="json"),
    )
    return FutureResponse.model_validate(committed)


@app.post("/api/v1/forward", response_model=FutureResponse)
async def forward(request: ForwardRequest, session: AsyncSession = Depends(get_session)):
    """Forward pass to obtain logprobs without accumulating gradients"""
    request_type = types.RequestType.FORWARD.value
    request_key = _sequence_request_identity(
        f"training:{request.model_id}", request.seq_id
    )
    payload_sha256 = _request_payload_sha256(request)
    replay = await _reserve_deduplicated_request(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
    )
    if replay is not None:
        return FutureResponse.model_validate(replay)
    await get_model(session, request.model_id)

    request_id = await create_future(
        session=session,
        request_type=types.RequestType.FORWARD,
        model_id=request.model_id,
        request_data=request.forward_input.to_types(),
    )

    response = FutureResponse(
        future_id=str(request_id), status="pending", request_id=str(request_id)
    )
    committed, _ = await _commit_deduplicated_response(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
        response_data=response.model_dump(mode="json"),
    )
    return FutureResponse.model_validate(committed)


@app.post("/api/v1/optim_step", response_model=FutureResponse)
async def optim_step(request: OptimStepRequest, session: AsyncSession = Depends(get_session)):
    """Update model using accumulated gradients."""
    request_type = types.RequestType.OPTIM_STEP.value
    request_key = _sequence_request_identity(
        f"training:{request.model_id}", request.seq_id
    )
    payload_sha256 = _request_payload_sha256(request)
    replay = await _reserve_deduplicated_request(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
    )
    if replay is not None:
        return FutureResponse.model_validate(replay)
    await get_model(session, request.model_id)

    request_id = await create_future(
        session=session,
        request_type=types.RequestType.OPTIM_STEP,
        model_id=request.model_id,
        request_data=types.OptimStepInput(adam_params=request.adam_params.to_types()),
    )

    response = FutureResponse(
        future_id=str(request_id), status="pending", request_id=str(request_id)
    )
    committed, _ = await _commit_deduplicated_response(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
        response_data=response.model_dump(mode="json"),
    )
    return FutureResponse.model_validate(committed)


@app.post("/api/v1/load_weights", response_model=FutureResponse)
async def load_weights(request: LoadWeightsRequest, req: Request, session: AsyncSession = Depends(get_session)):
    """Loads weights and training state."""
    request_type = types.RequestType.LOAD_WEIGHTS.value
    request_key = _sequence_request_identity(
        f"training:{request.model_id}", request.seq_id
    )
    payload_sha256 = _request_payload_sha256(request)
    replay = await _reserve_deduplicated_request(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
    )
    if replay is not None:
        return FutureResponse.model_validate(replay)
    await get_model(session, request.model_id)

    path = types.TinkerPath.parse(request.path)
    if (
        not path
        or path.kind != "weights"
        or not (source_model_id := path.primary_id)
        or not (checkpoint_id := path.secondary_id)
    ):
        raise HTTPException(
            status_code=400, detail="request.path must be in format tinker://source_model_id/weights/checkpoint_id"
        )

    await validate_checkpoint(req, source_model_id, checkpoint_id, types.CheckpointType.TRAINING, session)

    request_id = await create_future(
        session=session,
        request_type=types.RequestType.LOAD_WEIGHTS,
        model_id=request.model_id,
        request_data=types.LoadWeightsInput(source_model_id=source_model_id, checkpoint_id=checkpoint_id),
    )

    response = FutureResponse(
        future_id=str(request_id), status="pending", request_id=str(request_id)
    )
    committed, _ = await _commit_deduplicated_response(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
        response_data=response.model_dump(mode="json"),
    )
    return FutureResponse.model_validate(committed)


@app.post("/api/v1/save_weights", response_model=FutureResponse)
async def save_weights(request: SaveWeightsRequest, session: AsyncSession = Depends(get_session)):
    """Saves weights and training state."""
    request_type = types.RequestType.SAVE_WEIGHTS.value
    request_key = _sequence_request_identity(
        f"training:{request.model_id}", request.seq_id
    )
    payload_sha256 = _request_payload_sha256(request)
    replay = await _reserve_deduplicated_request(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
    )
    if replay is not None:
        return FutureResponse.model_validate(replay)
    # Create pending checkpoint entry (validates model exists)
    await create_checkpoint(
        session=session,
        model_id=request.model_id,
        checkpoint_id=request.path,
        checkpoint_type=types.CheckpointType.TRAINING,
    )

    request_id = await create_future(
        session=session,
        request_type=types.RequestType.SAVE_WEIGHTS,
        model_id=request.model_id,
        request_data=types.SaveWeightsInput(path=request.path),
    )

    response = FutureResponse(
        future_id=str(request_id), status="pending", request_id=str(request_id)
    )
    committed, _ = await _commit_deduplicated_response(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
        response_data=response.model_dump(mode="json"),
    )
    return FutureResponse.model_validate(committed)


@app.post("/api/v1/save_weights_for_sampler", response_model=FutureResponse)
async def save_weights_for_sampler(request: SaveWeightsForSamplerRequest, session: AsyncSession = Depends(get_session)):
    """Saves weights in a format compatible with sampling/inference servers."""
    request_type = types.RequestType.SAVE_WEIGHTS_FOR_SAMPLER.value
    request_key = _sequence_request_identity(
        f"training:{request.model_id}", request.seq_id
    )
    # tinker 0.22.4 allocates this session counter inside its outer retry
    # closure.  The stable training seq_id defines the logical request; replay
    # must return the first committed session even when this allocator value
    # advances after a lost response.
    ignored_fields = (
        frozenset({"sampling_session_seq_id"})
        if request.path is None
        else frozenset()
    )
    payload_sha256 = _request_payload_sha256(
        request, ignored_fields=ignored_fields
    )
    replay = await _reserve_deduplicated_request(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
    )
    if replay is not None:
        return FutureResponse.model_validate(replay)

    # Get the model (validates it exists and gives us the session_id)
    model = await get_model(session, request.model_id)

    checkpoint_id = request.path or f"ss{request.sampling_session_seq_id}_seq{request.seq_id}"
    sampling_session_id = None
    if request.sampling_session_seq_id is not None and request.seq_id is not None:
        # Create the sampling session using the model's session
        sampling_session_id = f"sampling_{uuid4().hex[:8]}"
        sampling_db = SamplingSessionDB(
            sampling_session_id=sampling_session_id,
            session_id=model.session_id,
            sampling_session_seq_id=request.sampling_session_seq_id,
            base_model=None,
            model_path=f"tinker://{request.model_id}/sampler_weights/{checkpoint_id}",
        )
        session.add(sampling_db)

    # Create pending checkpoint entry
    await create_checkpoint(
        session=session,
        model_id=request.model_id,
        checkpoint_id=checkpoint_id,
        checkpoint_type=types.CheckpointType.SAMPLER,
    )

    request_id = await create_future(
        session=session,
        request_type=types.RequestType.SAVE_WEIGHTS_FOR_SAMPLER,
        model_id=request.model_id,
        request_data=types.SaveWeightsForSamplerInput(
            path=checkpoint_id,
            sampling_session_seq_id=request.sampling_session_seq_id,
            seq_id=request.seq_id,
            sampling_session_id=sampling_session_id,
        ),
    )

    response = FutureResponse(
        future_id=str(request_id), status="pending", request_id=str(request_id)
    )
    committed, _ = await _commit_deduplicated_response(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
        response_data=response.model_dump(mode="json"),
    )
    return FutureResponse.model_validate(committed)


async def get_sampling_model(request: SampleRequest, session: AsyncSession) -> (str | None, str | None):
    """Return (base_model, model_path) for a sampling request."""
    # Resolve model/base from sampling_session_id if provided
    if request.sampling_session_id is not None:
        sampling_session = await session.get(SamplingSessionDB, request.sampling_session_id)
        if sampling_session is None:
            raise HTTPException(status_code=404, detail="Sampling session not found")
        return (sampling_session.base_model, sampling_session.model_path)
    return (request.base_model, request.model_path)


@app.post("/api/v1/asample", response_model=FutureResponse)
async def asample(request: SampleRequest, req: Request, session: AsyncSession = Depends(get_session)):
    """Generates samples from the model (async version)."""
    if request.sampling_session_id is not None and ":" in request.sampling_session_id:
        raise HTTPException(
            status_code=400,
            detail="sampling_session_id must not contain ':' (the routing-key delimiter)",
        )

    request_type = (
        types.RequestType.EXTERNAL.value
        if req.app.state.external_inference_client
        else types.RequestType.SAMPLE.value
    )
    request_key = (
        _sequence_request_identity(
            f"sampling:{request.sampling_session_id}", request.seq_id
        )
        if request.sampling_session_id is not None
        else _http_request_identity("sample-direct", req)
    )
    payload_sha256 = _request_payload_sha256(request)
    replay = await _reserve_deduplicated_request(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
    )
    if replay is not None:
        return FutureResponse.model_validate(replay)

    base_model, model_path = await get_sampling_model(request, session)

    if base_model:
        model_id = checkpoint_id = ""
    else:
        assert model_path is not None
        path = types.TinkerPath.parse(model_path)
        if (
            not path
            # Accept either tinker://model_id/checkpoint_id or tinker://model_id/sampler_weights/checkpoint_id
            or path.kind not in ("", "sampler_weights")
            or not (model_id := path.primary_id)
            or not (checkpoint_id := path.secondary_id)
        ):
            raise HTTPException(
                status_code=400,
                detail="model_path must be tinker://model_id/checkpoint_id or tinker://model_id/sampler_weights/checkpoint_id",
            )
        await get_model(session, model_id)
        # Validate that the checkpoint exists and is ready
        await validate_checkpoint(req, model_id, checkpoint_id, types.CheckpointType.SAMPLER, session)

    request_id = await create_future(
        session=session,
        request_type=types.RequestType(request_type),
        model_id=model_id,
        request_data=types.SampleInput(
            base_model=base_model,
            prompt=request.prompt.to_types(),
            sampling_params=request.sampling_params.to_types(),
            num_samples=request.num_samples,
            checkpoint_id=checkpoint_id,
            prompt_logprobs=request.prompt_logprobs if request.prompt_logprobs is not None else False,
            seq_id=request.seq_id,
            sampling_session_id=request.sampling_session_id,
        ),
    )

    response = FutureResponse(
        future_id=str(request_id), status="pending", request_id=str(request_id)
    )
    committed, _ = await _commit_deduplicated_response(
        session,
        request_key=request_key,
        request_type=request_type,
        payload_sha256=payload_sha256,
        response_data=response.model_dump(mode="json"),
    )

    if req.app.state.external_inference_client:
        asyncio.create_task(
            req.app.state.external_inference_client.call_and_store_result(
                request_id, request, model_id, checkpoint_id, base_model=base_model
            )
        )

    return FutureResponse.model_validate(committed)


@app.get("/api/v1/get_server_capabilities", response_model=GetServerCapabilitiesResponse)
async def get_server_capabilities(request: Request):
    """Retrieve information about supported models and server capabilities."""
    supported_models = [
        SupportedModel(model_name=request.app.state.engine_config.base_model),
    ]
    return GetServerCapabilitiesResponse(supported_models=supported_models)


class RetrieveFutureRequest(BaseModel):
    request_id: str


@app.post("/api/v1/retrieve_future")
async def retrieve_future(request: RetrieveFutureRequest, req: Request):
    """Retrieve the result of an async operation, waiting until it's available."""
    timeout = 300  # 5 minutes
    deadline = time.perf_counter() + timeout

    # Start with 100ms, grow to 1s
    poll = 0.1
    max_poll = 1.0

    while time.perf_counter() < deadline:
        try:
            async with AsyncSession(req.app.state.db_engine) as session:
                # First, only query the status to avoid deserializing JSON data
                statement = select(FutureDB.status).where(FutureDB.request_id == int(request.request_id))
                result = await session.exec(statement)
                status = result.first()

                if not status:
                    raise HTTPException(status_code=404, detail="Future not found")

                # Only fetch full record if status is terminal (completed or failed)
                if status in (RequestStatus.COMPLETED, RequestStatus.FAILED):
                    statement = select(FutureDB).where(FutureDB.request_id == int(request.request_id))
                    result = await session.exec(statement)
                    future = result.first()

                    if future.status == RequestStatus.COMPLETED:
                        return future.result_data

                    if future.status == RequestStatus.FAILED:
                        # Return 400 for handled errors (validation, etc.), 500 for unexpected failures
                        if future.result_data and "error" in future.result_data:
                            raise HTTPException(status_code=400, detail=future.result_data["error"])
                        else:
                            raise HTTPException(status_code=500, detail="Unknown error")
        except SATimeoutError:
            pass

        # Exponential backoff
        await asyncio.sleep(poll)
        poll = min(poll * 1.5, max_poll)

    raise HTTPException(status_code=408, detail="Timeout waiting for result")


@app.post("/api/v1/telemetry", response_model=TelemetryResponse)
async def send_telemetry(request: TelemetryRequest):
    """Accept batches of SDK telemetry events for analytics and diagnostics."""
    # Just acknowledge receipt without doing anything
    return TelemetryResponse(status="accepted")


async def validate_checkpoint(
    request: Request, unique_id: str, checkpoint_id: str, checkpoint_type: types.CheckpointType, session: AsyncSession
):
    """Validate that a model and checkpoint exist in the database, returning the checkpoint path."""
    checkpoint_db = await session.get(CheckpointDB, (unique_id, checkpoint_id, checkpoint_type))

    if not checkpoint_db:
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {unique_id}/{checkpoint_id}")

    if checkpoint_db.status == CheckpointStatus.PENDING:
        raise HTTPException(status_code=425, detail="Checkpoint is still being created")

    if checkpoint_db.status == CheckpointStatus.FAILED:
        raise HTTPException(status_code=500, detail=f"Checkpoint creation failed: {checkpoint_db.error_message}")

    subdir = "sampler_weights" if checkpoint_type == types.CheckpointType.SAMPLER else ""
    return request.app.state.engine_config.checkpoints_base / unique_id / subdir / f"{checkpoint_id}.tar.gz"


@app.get("/api/v1/training_runs")
async def list_training_runs(
    limit: int = 20, offset: int = 0, session: AsyncSession = Depends(get_session)
) -> TrainingRunsResponse:
    """List all training runs"""

    # Use window function to get total count alongside paginated results in a single query
    statement = select(ModelDB, func.count().over().label("total_count")).offset(offset).limit(limit)
    result = await session.exec(statement)
    rows = result.all()

    total_count = rows[0].total_count if rows else 0

    training_runs = []
    for row in rows:
        model = row.ModelDB
        lora_config = types.LoraConfig.model_validate(model.lora_config)

        training_runs.append(
            TrainingRun(
                training_run_id=model.model_id,
                base_model=model.base_model,
                model_owner="default",
                is_lora=True,
                corrupted=False,
                lora_rank=lora_config.rank,
                last_request_time=model.created_at,  # TODO: Once we track modified_at timestamps, update this
                last_checkpoint=None,
                last_sampler_checkpoint=None,
                user_metadata=None,
            )
        )

    return TrainingRunsResponse(
        training_runs=training_runs, cursor=Cursor(offset=offset, limit=limit, total_count=total_count)
    )


@app.get("/api/v1/training_runs/{unique_id}/checkpoints/{checkpoint_id}/archive")
async def get_checkpoint_archive_url(
    request: Request,
    unique_id: str = fastapi.Path(..., pattern=ID_PATTERN, max_length=ID_MAX_LENGTH),
    checkpoint_id: str = fastapi.Path(..., pattern=ID_PATTERN, max_length=ID_MAX_LENGTH),
    session: AsyncSession = Depends(get_session),
):
    """Return a 302 redirect to the download URL (SDK expects this pattern)"""
    await validate_checkpoint(request, unique_id, checkpoint_id, types.CheckpointType.SAMPLER, session)

    # Generate URL to the download endpoint and return 302 redirect
    download_url = str(request.url_for("download_checkpoint_archive", unique_id=unique_id, checkpoint_id=checkpoint_id))
    expires = datetime.utcnow() + timedelta(minutes=120)

    response = RedirectResponse(url=download_url, status_code=302)
    response.headers["Expires"] = expires.strftime("%a, %d %b %Y %H:%M:%S GMT")
    return response


@app.get("/api/v1/training_runs/{unique_id}/checkpoints/{checkpoint_id}/download")
async def download_checkpoint_archive(
    request: Request,
    unique_id: str = fastapi.Path(..., pattern=ID_PATTERN, max_length=ID_MAX_LENGTH),
    checkpoint_id: str = fastapi.Path(..., pattern=ID_PATTERN, max_length=ID_MAX_LENGTH),
    session: AsyncSession = Depends(get_session),
):
    """Actually download the checkpoint archive bytes"""
    checkpoint_path = await validate_checkpoint(
        request, unique_id, checkpoint_id, types.CheckpointType.SAMPLER, session
    )

    file_buffer = await asyncio.to_thread(download_file, checkpoint_path)

    filename = f"{unique_id}_{checkpoint_id}.tar.gz"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Length": str(file_buffer.getbuffer().nbytes),
    }

    return StreamingResponse(file_buffer, media_type="application/octet-stream", headers=headers)


@app.get("/api/v1/training_runs/{unique_id}/checkpoints")
async def list_checkpoints(
    unique_id: str = fastapi.Path(..., pattern=ID_PATTERN, max_length=ID_MAX_LENGTH),
    session: AsyncSession = Depends(get_session),
):
    """List checkpoints for a model."""
    statement = (
        select(CheckpointDB)
        .where(CheckpointDB.model_id == unique_id)
        .where(CheckpointDB.status == CheckpointStatus.COMPLETED)
    )
    result = await session.exec(statement)

    checkpoints = []
    for checkpoint in result.all():
        # Construct tinker_path based on checkpoint type
        path_kind = "weights" if checkpoint.checkpoint_type == types.CheckpointType.TRAINING else "sampler_weights"
        tinker_path = f"tinker://{unique_id}/{path_kind}/{checkpoint.checkpoint_id}"

        checkpoints.append(
            Checkpoint(
                checkpoint_id=checkpoint.checkpoint_id,
                checkpoint_type=checkpoint.checkpoint_type.value,
                time=checkpoint.completed_at,
                tinker_path=tinker_path,
            )
        )

    return ListCheckpointsResponse(checkpoints=checkpoints)


@app.get("/api/v1/models/{unique_id}/checkpoints")
async def list_checkpoints_models(
    unique_id: str = fastapi.Path(..., pattern=ID_PATTERN, max_length=ID_MAX_LENGTH),
    session: AsyncSession = Depends(get_session),
):
    """Just to be compatible with tinker SDK"""
    return await list_checkpoints(unique_id=unique_id, session=session)


@app.post("/api/v1/weights_info", response_model=WeightsInfoResponse)
async def get_weights_info(request: WeightsInfoRequest, req: Request, session: AsyncSession = Depends(get_session)):
    """Get information about weights/checkpoint from a tinker path."""
    path = types.TinkerPath.parse(request.tinker_path)

    if not path or path.kind != "weights":
        raise HTTPException(
            status_code=400, detail="Invalid tinker path format. Expected: tinker://model_id/weights/checkpoint_id"
        )

    model_id = path.primary_id
    checkpoint_id = path.secondary_id

    # Get model info (this will raise 404 if model doesn't exist)
    model = await get_model(session, model_id)

    # Validate checkpoint exists and is completed
    await validate_checkpoint(req, model_id, checkpoint_id, types.CheckpointType.TRAINING, session)

    lora_config = types.LoraConfig.model_validate(model.lora_config)
    is_lora = lora_config.rank > 0

    return WeightsInfoResponse(
        base_model=model.base_model,
        is_lora=is_lora,
        lora_rank=lora_config.rank,
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Tinker API Mock",
        "version": "0.0.1",
        "endpoints": {
            "models": ["/api/v1/create_model", "/api/v1/get_info", "/api/v1/training_runs/{model_id}"],
            "training": ["/api/v1/forward_backward", "/api/v1/optim_step"],
            "futures": ["/api/v1/retrieve_future"],
            "service": ["/api/v1/get_server_capabilities"],
            "telemetry": ["/api/v1/telemetry"],
            "checkpoints": ["/api/v1/training_runs/{unique_id}/checkpoints"],
            "download": [
                "/api/v1/training_runs/{unique_id}/checkpoints/{checkpoint_id}/archive",
                "/api/v1/training_runs/{unique_id}/checkpoints/{checkpoint_id}/download",
            ],
        },
    }


if __name__ == "__main__":
    import argparse

    import uvicorn

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="SkyRL tinker API server")
    add_model(parser, EngineConfig)
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    # Create EngineConfig from parsed arguments (only EngineConfig fields)
    engine_config = EngineConfig.model_validate({k: v for k, v in vars(args).items() if k in EngineConfig.model_fields})
    if engine_config.startup_launch_id is not None:
        parser.error("--startup-launch-id is internal and cannot be supplied to the API")

    # Store config in app.state so lifespan can access it
    app.state.engine_config = engine_config

    uvicorn.run(app, host=args.host, port=args.port, log_config=get_uvicorn_log_config())
