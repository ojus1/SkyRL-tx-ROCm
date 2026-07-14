from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parents[2] / "rocm" / "local_server_attestation.py"
_SPEC = importlib.util.spec_from_file_location("local_server_attestation", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_ATTEST = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _ATTEST
_SPEC.loader.exec_module(_ATTEST)

_BOOT_ID = "12345678-1234-5678-9abc-123456789abc"
_LAUNCH_ID = "a" * 32
_PORT = 8001
_LISTENER_INODE = 424242
_EXPECTED_GIT_HEAD = "b" * 40
_EXPECTED_GIT_TREE = "c" * 40
_PRODUCTION_MODEL_BLOBS = dict(_ATTEST._EXPECTED_MODEL_BLOBS)


def _fake_model_payload(name: str) -> bytes:
    return f"fixture model blob for {name}\n".encode()


def _git_blob_oid(payload: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


@pytest.fixture(autouse=True)
def _use_content_addressed_fixture_model(monkeypatch):
    fixture_ids = {
        name: (
            hashlib.sha256(_fake_model_payload(name)).hexdigest()
            if len(production_id) == 64
            else _git_blob_oid(_fake_model_payload(name))
        )
        for name, production_id in _PRODUCTION_MODEL_BLOBS.items()
    }
    monkeypatch.setattr(_ATTEST, "_EXPECTED_MODEL_BLOBS", fixture_ids)


class _HealthProbe:
    def __init__(self, result=None, error: BaseException | None = None):
        self.result = {"status": "ok"} if result is None else result
        self.error = error
        self.calls: list[tuple[str, int]] = []

    def __call__(self, host: str, port: int):
        self.calls.append((host, port))
        if self.error is not None:
            raise self.error
        return self.result


def _stat_record(
    pid: int,
    *,
    ppid: int,
    process_group: int,
    session: int,
    start_ticks: int,
    state: str = "S",
) -> str:
    fields = [state, str(ppid), str(process_group), str(session), *(["0"] * 15)]
    fields.append(str(start_ticks))
    assert len(fields) == 20
    return f"{pid} (fake process {pid}) {' '.join(fields)}\n"


def _nul(values: list[str]) -> bytes:
    return b"\0".join(value.encode("utf-8") for value in values) + b"\0"


@dataclass
class _ServerFixture:
    root: Path
    proc_root: Path
    database_path: Path
    model_path: Path
    source_root: Path
    source_archive: Path
    account_home: Path
    repo_root: Path
    cache_directory: Path
    runtime_root: Path
    lock_directory: Path
    held_lock_fd: int
    uv_path: Path
    python_path: Path
    network_namespace_path: Path
    uid: int
    outer_pid: int
    api_pid: int
    engine_launcher_pid: int
    engine_pid: int
    direct_engine_exec: bool

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{_PORT}"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.database_path}"

    @property
    def backend_config(self) -> dict[str, object]:
        return {
            "max_lora_adapters": 2,
            "max_lora_rank": 8,
            "train_micro_batch_size": 1,
            "sample_max_num_sequences": 1,
            "gradient_checkpointing": True,
            "loss_chunk_size": 64,
            "abstract_model_load": False,
        }

    def api_options(self, backend_config_raw: str) -> list[str]:
        return [
            "--base-model",
            str(self.model_path),
            "--backend",
            "jax",
            "--backend-config",
            backend_config_raw,
            "--host",
            "127.0.0.1",
            "--port",
            str(_PORT),
            "--checkpoints-base",
            str(self.database_path.parent / "checkpoints"),
            "--engine-startup-timeout-sec",
            "3600",
            "--database-url",
            self.database_url,
        ]

    def engine_options(self, backend_config_raw: str) -> list[str]:
        return [
            "--base-model",
            str(self.model_path),
            "--backend",
            "jax",
            "--backend-config",
            backend_config_raw,
            "--checkpoints-base",
            str(self.database_path.parent / "checkpoints"),
            "--database-url",
            self.database_url,
            "--external-inference-api-key",
            "EMPTY",
            "--external-inference-lora-base",
            "/tmp/lora_models",
            "--session-cleanup-interval-sec",
            "60",
            "--session-timeout-sec",
            "300",
            "--startup-launch-id",
            _LAUNCH_ID,
            "--engine-startup-timeout-sec",
            "3600",
        ]

    def process_root(self, pid: int) -> Path:
        return self.proc_root / str(pid)

    def write_cmdline(self, pid: int, argv: list[str]) -> None:
        (self.process_root(pid) / "cmdline").write_bytes(_nul(argv))

    def read_cmdline(self, pid: int) -> list[str]:
        raw = (self.process_root(pid) / "cmdline").read_bytes()
        return list(_ATTEST._parse_nul_argv(raw, f"test PID {pid}"))

    def replace_option(self, pid: int, name: str, value: str) -> None:
        argv = self.read_cmdline(pid)
        index = argv.index(name)
        argv[index + 1] = value
        self.write_cmdline(pid, argv)

    def install_commands(
        self,
        backend_config_raw: str | None = None,
        *,
        engine_backend_config_raw: str | None = None,
    ) -> None:
        api_raw = backend_config_raw or json.dumps(
            self.backend_config, separators=(",", ":")
        )
        engine_raw = engine_backend_config_raw or json.dumps(json.loads(api_raw))
        api_options = self.api_options(api_raw)
        engine_options = self.engine_options(engine_raw)
        self.write_cmdline(
            self.outer_pid,
            [
                str(self.uv_path),
                "run",
                "--active",
                "--no-sync",
                "--no-env-file",
                "--no-config",
                "--directory",
                str(self.source_root),
                "--project",
                str(self.source_root),
                "-m",
                _ATTEST.API_MODULE,
                *api_options,
            ],
        )
        self.write_cmdline(
            self.api_pid,
            [str(self.python_path), "-m", _ATTEST.API_MODULE, *api_options],
        )
        if self.direct_engine_exec:
            launcher_argv = [
                str(self.python_path),
                "-m",
                _ATTEST.ENGINE_MODULE,
                *engine_options,
            ]
        else:
            launcher_argv = [
                str(self.uv_path),
                "run",
                "--active",
                "--no-sync",
                "--no-env-file",
                "--no-config",
                "--directory",
                str(self.source_root),
                "--project",
                str(self.source_root),
                "--extra",
                "tinker",
                "--extra",
                "jax",
                "-m",
                _ATTEST.ENGINE_MODULE,
                *engine_options,
            ]
        self.write_cmdline(self.engine_launcher_pid, launcher_argv)
        if not self.direct_engine_exec:
            self.write_cmdline(
                self.engine_pid,
                [
                    str(self.python_path),
                    "-m",
                    _ATTEST.ENGINE_MODULE,
                    *engine_options,
                ],
            )
        if self.database_path.exists():
            parsed_config = json.loads(api_raw)
            self.update_source_claim(
                memory_mode=(
                    "preallocate85"
                    if parsed_config["abstract_model_load"]
                    else "growth"
                )
            )

    def write_environment(
        self,
        pid: int,
        *,
        xla_flags: str | None = _ATTEST.EXPECTED_XLA_FLAGS,
        jax_platforms: str | None = "rocm",
        rocr_visible_devices: str | None = "0",
        preallocate: str | None = "false",
        memory_environment: dict[str, str] | None = None,
        pallas_attention: str | None = "0",
        extra_environment: dict[str, str] | None = None,
        remove_environment: tuple[str, ...] = (),
        duplicate: bool = False,
    ) -> None:
        if pid == self.outer_pid:
            role = "outer"
        elif pid == self.engine_pid:
            role = "engine"
        else:
            role = "api"
        source_memory_mode = "growth"
        startup_cache = {"status": "not_required"}
        if self.database_path.exists():
            with closing(sqlite3.connect(self.database_path)) as connection:
                row = connection.execute(
                    "SELECT api_source_attestation FROM engine_launches"
                ).fetchone()
            if row is not None:
                source_record = json.loads(row[0])
                source_memory_mode = source_record["memory_mode"]
                startup_cache = source_record["startup_cache_attestation"]
        venv_bin = self.repo_root / ".venv/bin"
        environment = {
            "HOME": str(self.account_home),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": (
                "/opt/rocm/bin:/usr/bin:/bin"
                if role == "outer"
                else ":".join(
                    [str(venv_bin)] * (2 if role == "engine" else 1)
                    + ["/opt/rocm/bin", "/usr/bin", "/bin"]
                )
            ),
            "VIRTUAL_ENV": str(self.repo_root / ".venv"),
            "XDG_RUNTIME_DIR": str(self.runtime_root),
            "SKYRL_QWEN35_RUNTIME_GIT_HEAD": _EXPECTED_GIT_HEAD,
            "SKYRL_QWEN35_RUNTIME_MEMORY_MODE": source_memory_mode,
            "SKYRL_QWEN35_RUNTIME_REPO_ROOT": str(self.repo_root),
            "SKYRL_QWEN35_RUNTIME_SOURCE_ROOT": str(self.source_root),
            "SKYRL_QWEN35_RUNTIME_UV_EXECUTABLE": str(self.uv_path),
            **_ATTEST._REQUIRED_RUNTIME_ENVIRONMENT,
            "JAX_COMPILATION_CACHE_DIR": str(self.cache_directory),
            "SKYRL_ROCM_PALLAS_ATTENTION": "0",
            "SKYRL_QWEN35_LAUNCH_LOCK_FD": "10",
        }
        if role != "outer":
            environment["UV"] = str(self.uv_path)
            environment["UV_RUN_RECURSION_DEPTH"] = "2" if role == "engine" else "1"
        if startup_cache["status"] == "required-v1":
            environment.update(
                {
                    "SKYRL_QWEN35_RUNTIME_T64_CACHE_ATTEST": "required-v1",
                    "SKYRL_QWEN35_RUNTIME_PREWARM_AUDIT_PATH": startup_cache[
                        "prewarm_audit"
                    ]["path"],
                    "SKYRL_QWEN35_RUNTIME_PREWARM_AUDIT_SHA256": startup_cache[
                        "prewarm_audit"
                    ]["sha256"],
                    "SKYRL_QWEN35_RUNTIME_PREWARM_HANDOFF_PATH": startup_cache[
                        "prewarm_handoff"
                    ]["path"],
                    "SKYRL_QWEN35_RUNTIME_PREWARM_HANDOFF_SHA256": startup_cache[
                        "prewarm_handoff"
                    ]["sha256"],
                }
            )
        if xla_flags is not None:
            environment["XLA_FLAGS"] = xla_flags
        else:
            environment.pop("XLA_FLAGS", None)
        if jax_platforms is not None:
            environment["JAX_PLATFORMS"] = jax_platforms
        else:
            environment.pop("JAX_PLATFORMS", None)
        if rocr_visible_devices is not None:
            environment["ROCR_VISIBLE_DEVICES"] = rocr_visible_devices
        else:
            environment.pop("ROCR_VISIBLE_DEVICES", None)
        if preallocate is not None:
            environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = preallocate
        else:
            environment.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
        for name, value in (memory_environment or {}).items():
            environment[name] = value
        if pallas_attention is None:
            environment.pop("SKYRL_ROCM_PALLAS_ATTENTION", None)
        else:
            environment["SKYRL_ROCM_PALLAS_ATTENTION"] = pallas_attention
        environment.update(extra_environment or {})
        for name in remove_environment:
            environment.pop(name, None)
        values = [f"{name}={value}" for name, value in environment.items()]
        if duplicate:
            values.append(f"XLA_FLAGS={environment['XLA_FLAGS']}")
        (self.process_root(pid) / "environ").write_bytes(_nul(values))

    def rewrite_stat(
        self,
        pid: int,
        *,
        ppid: int,
        process_group: int,
        session: int,
        start_ticks: int,
        state: str = "S",
    ) -> None:
        (self.process_root(pid) / "stat").write_text(
            _stat_record(
                pid,
                ppid=ppid,
                process_group=process_group,
                session=session,
                start_ticks=start_ticks,
                state=state,
            ),
            encoding="utf-8",
        )

    def update_launch(self, **values) -> None:
        assignments = ", ".join(f"{name} = ?" for name in values)
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                f"UPDATE engine_launches SET {assignments}",  # noqa: S608 - test-controlled names
                tuple(values.values()),
            )
            connection.commit()

    def update_source_claim(
        self,
        *,
        memory_mode: str | None = None,
        pallas_attention: str | None = None,
    ) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                "SELECT api_source_attestation, engine_source_attestation "
                "FROM engine_launches"
            ).fetchone()
            assert row is not None
            api_source, engine_source = (json.loads(value) for value in row)
            for source in (api_source, engine_source):
                if memory_mode is not None:
                    source["memory_mode"] = memory_mode
                if pallas_attention is not None:
                    source["pallas_attention"] = pallas_attention
            connection.execute(
                "UPDATE engine_launches SET api_source_attestation = ?, "
                "engine_source_attestation = ?",
                (json.dumps(api_source), json.dumps(engine_source)),
            )
            connection.commit()


def _write_process(
    fixture: _ServerFixture,
    pid: int,
    *,
    ppid: int,
    process_group: int,
    session: int,
    start_ticks: int,
    executable: Path,
) -> None:
    process_root = fixture.process_root(pid)
    (process_root / "ns").mkdir(parents=True)
    (process_root / "fd").mkdir()
    (process_root / "net").mkdir()
    fixture.rewrite_stat(
        pid,
        ppid=ppid,
        process_group=process_group,
        session=session,
        start_ticks=start_ticks,
    )
    (process_root / "status").write_text(
        f"Name:\tfake\nUid:\t{fixture.uid}\t{fixture.uid}\t{fixture.uid}\t{fixture.uid}\n",
        encoding="ascii",
    )
    (process_root / "ns/net").symlink_to(fixture.network_namespace_path)
    (process_root / "exe").symlink_to(executable)
    (process_root / "cwd").symlink_to(fixture.source_root)
    (process_root / "fd/10").symlink_to(fixture.lock_directory)
    fixture.write_environment(pid)


def _create_database(fixture: _ServerFixture) -> None:
    lock_record = {
        "status": "passed",
        "descriptor": 10,
        "path": str(fixture.lock_directory),
        "inheritable": True,
        "exclusive_lock_observed": True,
    }
    source_shared = {
        "status": "passed",
        "git_head": _EXPECTED_GIT_HEAD,
        "git_tree": _EXPECTED_GIT_TREE,
        "source_archive_path": str(fixture.source_archive),
        "source_archive_sha256": hashlib.sha256(
            fixture.source_archive.read_bytes()
        ).hexdigest(),
        "source_file_count": 3,
        "source_total_bytes": sum(
            (fixture.source_root / path).stat().st_size
            for path in (
                "skyrl/__init__.py",
                "skyrl/tinker/api.py",
                "skyrl/tinker/engine.py",
            )
        ),
        "full_head_tree_validated": True,
        "source_root": str(fixture.source_root),
        "repo_root": str(fixture.repo_root),
        "working_directory": str(fixture.source_root),
        "package_origin": str(fixture.source_root / "skyrl/__init__.py"),
        "uv_executable": str(fixture.uv_path),
        "uv_sha256": hashlib.sha256(fixture.uv_path.read_bytes()).hexdigest(),
        "launch_lock": lock_record,
        "jax_compilation_cache": str(fixture.cache_directory),
        "memory_mode": "growth",
        "xla_flags": _ATTEST.EXPECTED_XLA_FLAGS,
        "jax_enable_pgle": "false",
        "jax_compilation_cache_expect_pgle": "false",
        "pallas_attention": "0",
        "startup_cache_attestation": {"status": "not_required"},
        "dont_write_bytecode": True,
    }
    api_source_record = {
        **source_shared,
        "role": "api",
        "module_origin": str(fixture.source_root / "skyrl/tinker/api.py"),
    }
    engine_source_record = {
        **source_shared,
        "role": "engine",
        "module_origin": str(fixture.source_root / "skyrl/tinker/engine.py"),
    }
    handoff_record = {
        "status": "passed",
        "source_status": "passed",
        "git_head": source_shared["git_head"],
        "git_tree": source_shared["git_tree"],
        "source_root": source_shared["source_root"],
        "jax_compilation_cache": source_shared["jax_compilation_cache"],
        "startup_cache_attestation": source_shared["startup_cache_attestation"],
        "launch_lock": lock_record,
    }
    timestamp = "2026-07-14T08:00:00+00:00"
    with closing(sqlite3.connect(fixture.database_path)) as connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        connection.execute(
            """
            CREATE TABLE engine_launches (
                launch_id VARCHAR NOT NULL PRIMARY KEY,
                backend VARCHAR NOT NULL,
                status VARCHAR(12) NOT NULL,
                boot_id VARCHAR NOT NULL,
                api_pid INTEGER NOT NULL,
                api_start_ticks BIGINT NOT NULL,
                engine_launcher_pid INTEGER,
                engine_launcher_start_ticks BIGINT,
                engine_pid INTEGER,
                engine_start_ticks BIGINT,
                api_source_attestation JSON NOT NULL,
                api_launch_lock_attestation JSON NOT NULL,
                engine_source_attestation JSON NOT NULL,
                engine_launch_lock_attestation JSON NOT NULL,
                runtime_handoff_attestation JSON NOT NULL,
                cache_evidence_status VARCHAR NOT NULL,
                cache_evidence JSON NOT NULL,
                error_message VARCHAR,
                heartbeat_at DATETIME,
                heartbeat_monotonic_ns BIGINT,
                heartbeat_sequence BIGINT NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                ready_at DATETIME
            )
            """
        )
        connection.execute(
            "CREATE INDEX ix_engine_launches_status ON engine_launches (status)"
        )
        connection.execute(
            "CREATE INDEX ix_engine_launches_cache_evidence_status "
            "ON engine_launches (cache_evidence_status)"
        )
        connection.execute(
            """
            CREATE TABLE request_deduplication (
                request_key VARCHAR NOT NULL PRIMARY KEY,
                request_type VARCHAR NOT NULL,
                payload_sha256 VARCHAR NOT NULL,
                response_data JSON,
                created_at DATETIME NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO engine_launches VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                _LAUNCH_ID,
                "jax",
                "READY",
                _BOOT_ID,
                fixture.api_pid,
                200,
                fixture.engine_launcher_pid,
                300,
                fixture.engine_pid,
                300 if fixture.direct_engine_exec else 400,
                json.dumps(api_source_record),
                json.dumps(lock_record),
                json.dumps(engine_source_record),
                json.dumps(lock_record),
                json.dumps(handoff_record),
                "not_required",
                json.dumps({}),
                None,
                timestamp,
                time.monotonic_ns(),
                7,
                timestamp,
                timestamp,
                timestamp,
            ),
        )
        connection.commit()
    fixture.database_path.chmod(0o600)


def _make_fixture(
    tmp_path: Path, *, direct_engine_exec: bool = False
) -> _ServerFixture:
    uid = os.getuid()
    tmp_path.chmod(0o700)
    proc_root = tmp_path / "proc"
    (proc_root / "sys/kernel/random").mkdir(parents=True)
    (proc_root / "sys/kernel/random/boot_id").write_text(
        _BOOT_ID + "\n", encoding="ascii"
    )
    network_namespace = tmp_path / "network-namespace"
    network_namespace.write_bytes(b"namespace")
    (proc_root / "self/ns").mkdir(parents=True)
    (proc_root / "self/ns/net").symlink_to(network_namespace)

    account_home = tmp_path / "home"
    account_home.mkdir(mode=0o700)
    uv_path = account_home / ".local/bin/uv"
    uv_path.parent.mkdir(parents=True)
    repo_root = tmp_path / "repo"
    python_path = repo_root / ".venv/bin/python3.12"
    python_path.parent.mkdir(parents=True)
    uv_path.write_bytes(b"fake exact uv payload\n")
    python_path.write_bytes(b"fake exact Python payload\n")
    _ATTEST.EXPECTED_UV_SHA256 = hashlib.sha256(uv_path.read_bytes()).hexdigest()

    source_commit_root = (
        account_home / ".cache/skyrl-source-snapshots-private-v1" / _EXPECTED_GIT_HEAD
    )
    source_root = source_commit_root / "source-head"
    source_root.mkdir(parents=True, mode=0o700)
    for directory in (
        account_home / ".cache",
        account_home / ".cache/skyrl-source-snapshots-private-v1",
        source_commit_root,
        source_root,
    ):
        directory.chmod(0o700)
    source_payloads = {
        "skyrl/__init__.py": b"# fixture package\n",
        "skyrl/tinker/api.py": b"# fixture API\n",
        "skyrl/tinker/engine.py": b"# fixture engine\n",
    }
    for relative_path, payload in source_payloads.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(0o600)
    source_archive = source_commit_root / "source-head.tar"
    source_archive.write_bytes(
        json.dumps(
            {name: payload.hex() for name, payload in sorted(source_payloads.items())},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    source_archive.chmod(0o600)
    cache_directory = tmp_path / "jax-cache"
    cache_directory.mkdir(mode=0o700)
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(mode=0o700)
    lock_directory = runtime_root / f"skyrl-qwen35-rocm-{uid}"
    lock_directory.mkdir(mode=0o700)
    held_lock_fd = os.open(lock_directory, os.O_RDONLY | os.O_DIRECTORY)
    fcntl.flock(held_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    model_cache = tmp_path / _ATTEST.EXPECTED_MODEL_CACHE_DIRECTORY
    blobs = model_cache / "blobs"
    model_path = model_cache / "snapshots" / _ATTEST.EXPECTED_MODEL_REVISION
    blobs.mkdir(parents=True)
    model_path.mkdir(parents=True)
    for name, blob_id in _ATTEST._EXPECTED_MODEL_BLOBS.items():
        (blobs / blob_id).write_bytes(_fake_model_payload(name))
        (model_path / name).symlink_to(f"../../blobs/{blob_id}")

    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    (run_dir / "checkpoints").mkdir(mode=0o700)
    database_path = run_dir / "tinker.db"
    outer_pid, api_pid, engine_launcher_pid = 100, 101, 102
    engine_pid = engine_launcher_pid if direct_engine_exec else 103
    fixture = _ServerFixture(
        root=tmp_path,
        proc_root=proc_root,
        database_path=database_path,
        model_path=model_path,
        source_root=source_root,
        source_archive=source_archive,
        account_home=account_home,
        repo_root=repo_root,
        cache_directory=cache_directory,
        runtime_root=runtime_root,
        lock_directory=lock_directory,
        held_lock_fd=held_lock_fd,
        uv_path=uv_path,
        python_path=python_path,
        network_namespace_path=network_namespace,
        uid=uid,
        outer_pid=outer_pid,
        api_pid=api_pid,
        engine_launcher_pid=engine_launcher_pid,
        engine_pid=engine_pid,
        direct_engine_exec=direct_engine_exec,
    )
    _write_process(
        fixture,
        outer_pid,
        ppid=50,
        process_group=50,
        session=50,
        start_ticks=100,
        executable=uv_path,
    )
    _write_process(
        fixture,
        api_pid,
        ppid=outer_pid,
        process_group=50,
        session=50,
        start_ticks=200,
        executable=python_path,
    )
    _write_process(
        fixture,
        engine_launcher_pid,
        ppid=api_pid,
        process_group=engine_launcher_pid,
        session=engine_launcher_pid,
        start_ticks=300,
        executable=python_path if direct_engine_exec else uv_path,
    )
    if not direct_engine_exec:
        _write_process(
            fixture,
            engine_pid,
            ppid=engine_launcher_pid,
            process_group=engine_launcher_pid,
            session=engine_launcher_pid,
            start_ticks=400,
            executable=python_path,
        )
    fixture.install_commands()
    tcp_header = (
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when "
        "retrnsmt   uid  timeout inode\n"
    )
    tcp_row = (
        f"   0: 0100007F:{_PORT:04X} 00000000:0000 0A "
        f"00000000:00000000 00:00000000 00000000 {uid} 0 "
        f"{_LISTENER_INODE} 1\n"
    )
    (fixture.process_root(api_pid) / "net/tcp").write_text(
        tcp_header + tcp_row, encoding="ascii"
    )
    (fixture.process_root(api_pid) / "fd/7").symlink_to(f"socket:[{_LISTENER_INODE}]")
    _create_database(fixture)
    return fixture


def _fixture_source_validator(
    fixture: _ServerFixture,
    repo_root: Path,
    git_head: str,
    account_home: Path,
):
    assert repo_root == fixture.repo_root
    assert git_head == _EXPECTED_GIT_HEAD
    assert account_home == fixture.account_home
    relative_paths = (
        "skyrl/__init__.py",
        "skyrl/tinker/api.py",
        "skyrl/tinker/engine.py",
    )
    payloads = {
        name: (fixture.source_root / name).read_bytes().hex() for name in relative_paths
    }
    expected_archive = json.dumps(
        payloads, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    if fixture.source_archive.read_bytes() != expected_archive:
        raise RuntimeError("fixture source/archive mismatch")
    return {
        "cache_status": "validated",
        "format": "skyrl-private-source-cache-v1",
        "git_head": _EXPECTED_GIT_HEAD,
        "git_tree": _EXPECTED_GIT_TREE,
        "source_archive_path": str(fixture.source_archive),
        "source_archive_sha256": hashlib.sha256(expected_archive).hexdigest(),
        "source_file_count": len(relative_paths),
        "source_snapshot_root": str(fixture.source_root),
        "source_total_bytes": sum(
            (fixture.source_root / name).stat().st_size for name in relative_paths
        ),
        "full_head_tree_validated": True,
    }


def _attestation_kwargs(
    fixture: _ServerFixture, health: _HealthProbe | None = None
) -> dict[str, object]:
    return {
        "expected_git_head": _EXPECTED_GIT_HEAD,
        "expected_git_tree": _EXPECTED_GIT_TREE,
        "expected_repo_root": fixture.repo_root,
        "expected_python_sha256": hashlib.sha256(
            fixture.python_path.read_bytes()
        ).hexdigest(),
        "proc_root": fixture.proc_root,
        "health_probe": health or _HealthProbe(),
        "expected_uid": fixture.uid,
        "runtime_root": fixture.runtime_root,
        "source_cache_validator": lambda repo, head, home: _fixture_source_validator(
            fixture, repo, head, home
        ),
        "cache_evidence_validator": (
            lambda _repo, _source, _claim, _evidence, _boot: None
        ),
    }


def _attest_fixture(fixture: _ServerFixture, health: _HealthProbe | None = None):
    kwargs = _attestation_kwargs(fixture, health)
    selected_health = kwargs["health_probe"]
    seal = _ATTEST.attest_local_server(
        server_pid=fixture.outer_pid,
        base_url=fixture.base_url,
        **kwargs,
    )
    return seal, selected_health


def test_attestation_and_revalidation_bind_exact_wrapper_process_tree(tmp_path):
    fixture = _make_fixture(tmp_path)
    seal, health = _attest_fixture(fixture)

    record = seal.as_record()
    assert record["status"] == "passed"
    assert record["endpoint"] == {
        "host": "127.0.0.1",
        "port": 8001,
        "listener_inode_sha256": record["endpoint"]["listener_inode_sha256"],
    }
    assert record["backend"]["name"] == "jax"
    assert record["backend"]["sample_max_num_sequences"] == 1
    assert record["backend"]["abstract_model_load"] is False
    assert record["environment"]["XLA_FLAGS"] == ("--xla_gpu_enable_command_buffer=")
    assert record["environment"]["JAX_PLATFORMS"] == "rocm"
    assert record["environment"]["ROCR_VISIBLE_DEVICES"] == "0"
    assert record["environment"]["SKYRL_ROCM_PALLAS_ATTENTION"] == "0"
    assert record["environment"]["memory_mode"] == "growth"
    assert record["environment"]["memory_environment"] == {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
    }
    assert record["processes"]["engine_launcher"]["pid"] == 102
    assert record["processes"]["engine"]["pid"] == 103
    assert record["source_attestation_status"] == "passed"
    assert record["model"]["revision"] == _ATTEST.EXPECTED_MODEL_REVISION
    assert seal.verified_model_snapshot() == fixture.model_path
    assert record["storage"]["sqlite_journal_mode"] == "wal"
    assert record["storage"]["ingress_sequence_deduplication"] == {
        "transactional_future_enqueue": True,
        "backend_effect_exactly_once_across_engine_crash": False,
        "external_sampling_dispatch_covered": False,
    }
    assert len(seal.contract_sha256) == 64
    assert health.calls == [("127.0.0.1", 8001)] * 2

    revalidation = _ATTEST.revalidate_local_server(
        seal,
        proc_root=fixture.proc_root,
        health_probe=health,
        expected_uid=fixture.uid,
    )

    assert revalidation == {
        "record_type": "server_revalidation",
        "schema_version": 3,
        "status": "passed",
        "scope": "hardened_local_procfs_point_in_time_v3",
        "revalidated_at_unix_ns": revalidation["revalidated_at_unix_ns"],
        "attestation_sha256": seal.contract_sha256,
    }
    assert health.calls == [("127.0.0.1", 8001)] * 4


def test_attestation_accepts_engine_launcher_direct_exec(tmp_path):
    fixture = _make_fixture(tmp_path, direct_engine_exec=True)

    seal, _ = _attest_fixture(fixture)

    processes = seal.as_record()["processes"]
    assert processes["engine_launcher"] == processes["engine"]
    assert processes["engine"]["pid"] == fixture.engine_launcher_pid


def test_source_cache_is_fully_revalidated_at_both_ends_of_each_seal(tmp_path):
    fixture = _make_fixture(tmp_path)
    health = _HealthProbe()
    kwargs = _attestation_kwargs(fixture, health)
    original = kwargs["source_cache_validator"]
    calls = []

    def counting_validator(repo, head, home):
        calls.append((repo, head, home))
        return original(repo, head, home)

    kwargs["source_cache_validator"] = counting_validator
    seal = _ATTEST.attest_local_server(
        server_pid=fixture.outer_pid,
        base_url=fixture.base_url,
        **kwargs,
    )
    assert len(calls) == 2

    _ATTEST.revalidate_local_server(
        seal,
        proc_root=fixture.proc_root,
        health_probe=health,
        expected_uid=fixture.uid,
    )
    assert len(calls) == 4


def test_public_record_contains_only_allowlisted_hashes_and_safe_values(tmp_path):
    fixture = _make_fixture(tmp_path)
    seal, _ = _attest_fixture(fixture)

    serialized = json.dumps(seal.as_record(), sort_keys=True)

    for private_path in (
        fixture.database_path,
        fixture.database_path.parent / "checkpoints",
        fixture.model_path,
        fixture.source_root,
        fixture.source_archive,
        fixture.cache_directory,
        fixture.lock_directory,
        fixture.uv_path,
        fixture.python_path,
        fixture.repo_root,
    ):
        assert str(private_path) not in serialized
    assert _LAUNCH_ID not in serialized
    assert "PATH=/usr/bin:/bin" not in serialized
    assert "--backend-config" not in serialized
    assert "cmdline_sha256" in serialized
    assert "executable_sha256" in serialized
    assert "runtime_handoff_sha256" in serialized


@pytest.mark.parametrize("role", ["api", "engine"])
def test_attestation_rejects_python_tokens_before_module(tmp_path, role):
    fixture = _make_fixture(tmp_path)
    pid = fixture.api_pid if role == "api" else fixture.engine_pid
    argv = fixture.read_cmdline(pid)
    fixture.write_cmdline(pid, [argv[0], "-c", "arbitrary_code()", *argv[1:]])

    with pytest.raises(
        _ATTEST.LocalServerAttestationError, match="exactly Python -m|API-module child"
    ):
        _attest_fixture(fixture)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda argv: [argv[0], "not-run", *argv[2:]],
        lambda argv: [*argv[:2], "--env-file", "/tmp/evil", *argv[2:]],
        lambda argv: [value for value in argv if value not in {"tinker"}],
    ],
)
def test_attestation_rejects_noncanonical_engine_uv_wrapper(tmp_path, mutation):
    fixture = _make_fixture(tmp_path)
    fixture.write_cmdline(
        fixture.engine_launcher_pid,
        mutation(fixture.read_cmdline(fixture.engine_launcher_pid)),
    )

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="wrapper|module"):
        _attest_fixture(fixture)


def test_attestation_rejects_nonhardened_outer_uv_form(tmp_path):
    fixture = _make_fixture(tmp_path)
    argv = fixture.read_cmdline(fixture.outer_pid)
    module_index = argv.index("-m")
    fixture.write_cmdline(
        fixture.outer_pid,
        [argv[0], "run", "--active", "--no-sync", *argv[module_index:]],
    )

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="hardened"):
        _attest_fixture(fixture)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--external-inference-api-key", "secret"),
        ("--external-inference-lora-base", "/tmp/other-loras"),
        ("--session-cleanup-interval-sec", "61"),
        ("--session-timeout-sec", "301"),
    ],
)
def test_attestation_rejects_changed_engine_defaults(tmp_path, option, value):
    fixture = _make_fixture(tmp_path)
    for pid in {fixture.engine_launcher_pid, fixture.engine_pid}:
        fixture.replace_option(pid, option, value)

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="configuration"):
        _attest_fixture(fixture)


def test_attestation_rejects_arbitrary_model_snapshot(tmp_path):
    fixture = _make_fixture(tmp_path)
    arbitrary = tmp_path / "not-qwen35"
    arbitrary.mkdir()
    for pid in {
        fixture.outer_pid,
        fixture.api_pid,
        fixture.engine_launcher_pid,
        fixture.engine_pid,
    }:
        fixture.replace_option(pid, "--base-model", str(arbitrary))

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="pinned Qwen3.5"):
        _attest_fixture(fixture)


def test_attestation_rejects_changed_required_model_blob(tmp_path):
    fixture = _make_fixture(tmp_path)
    entry = fixture.model_path / "config.json"
    entry.unlink()
    entry.symlink_to("../../blobs/not-the-pinned-object")

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="model entry"):
        _attest_fixture(fixture)


def test_revalidation_rejects_same_size_model_blob_mutation(tmp_path):
    fixture = _make_fixture(tmp_path)
    seal, health = _attest_fixture(fixture)
    blob_id = _ATTEST._EXPECTED_MODEL_BLOBS["config.json"]
    blob_path = fixture.model_path.parents[1] / "blobs" / blob_id
    payload = bytearray(blob_path.read_bytes())
    payload[0] ^= 1
    blob_path.write_bytes(payload)

    with pytest.raises(
        _ATTEST.LocalServerAttestationError, match="content verification"
    ):
        _ATTEST.revalidate_local_server(
            seal,
            proc_root=fixture.proc_root,
            health_probe=health,
            expected_uid=fixture.uid,
        )


def test_attestation_rejects_nested_model_blob_symlink(tmp_path):
    fixture = _make_fixture(tmp_path)
    blob_id = _ATTEST._EXPECTED_MODEL_BLOBS["config.json"]
    blob_path = fixture.model_path.parents[1] / "blobs" / blob_id
    outside = tmp_path / "outside-model-blob"
    blob_path.rename(outside)
    blob_path.symlink_to(outside)

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="model blob"):
        _attest_fixture(fixture)


def test_attestation_rejects_unexpected_recognized_model_sidecar(tmp_path):
    fixture = _make_fixture(tmp_path)
    (fixture.model_path / "special_tokens_map.json").write_text(
        '{"additional_special_tokens":["<changed-workload>"]}',
        encoding="utf-8",
    )

    with pytest.raises(
        _ATTEST.LocalServerAttestationError, match="unexpected top-level nodes"
    ):
        _attest_fixture(fixture)


def test_attestation_rereads_processes_after_final_model_stream(
    tmp_path, monkeypatch
):
    fixture = _make_fixture(tmp_path)
    original_validate = _ATTEST._validate_model_snapshot
    calls = 0

    def mutate_after_second_model_validation(raw, expected_uid):
        nonlocal calls
        result = original_validate(raw, expected_uid)
        calls += 1
        if calls == 2:
            fixture.rewrite_stat(
                fixture.outer_pid,
                ppid=50,
                process_group=50,
                session=50,
                start_ticks=101,
            )
        return result

    monkeypatch.setattr(
        _ATTEST,
        "_validate_model_snapshot",
        mutate_after_second_model_validation,
    )

    with pytest.raises(
        _ATTEST.LocalServerAttestationError, match="changed during attestation"
    ):
        _attest_fixture(fixture)


def test_attestation_rejects_checkpoint_directory_not_sibling_of_database(tmp_path):
    fixture = _make_fixture(tmp_path)
    other = tmp_path / "other-checkpoints"
    other.mkdir(mode=0o700)
    for pid in {
        fixture.outer_pid,
        fixture.api_pid,
        fixture.engine_launcher_pid,
        fixture.engine_pid,
    }:
        fixture.replace_option(pid, "--checkpoints-base", str(other))

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="database sibling"):
        _attest_fixture(fixture)


def test_attestation_rejects_reordered_launcher_options(tmp_path):
    fixture = _make_fixture(tmp_path)
    for pid in (fixture.outer_pid, fixture.api_pid):
        argv = fixture.read_cmdline(pid)
        backend_index = argv.index("--backend")
        backend_pair = argv[backend_index : backend_index + 2]
        del argv[backend_index : backend_index + 2]
        base_index = argv.index("--base-model")
        argv[base_index:base_index] = backend_pair
        fixture.write_cmdline(pid, argv)

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="option order"):
        _attest_fixture(fixture)


@pytest.mark.parametrize("pallas", [None, "2"])
def test_attestation_rejects_missing_or_invalid_pallas_mode(tmp_path, pallas):
    fixture = _make_fixture(tmp_path)
    fixture.write_environment(fixture.engine_pid, pallas_attention=pallas)

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="PALLAS|environment"):
        _attest_fixture(fixture)


def test_attestation_rejects_mixed_role_pallas_mode(tmp_path):
    fixture = _make_fixture(tmp_path)
    fixture.write_environment(fixture.engine_pid, pallas_attention="1")

    with pytest.raises(
        _ATTEST.LocalServerAttestationError, match="source|Pallas|roles"
    ):
        _attest_fixture(fixture)


def test_pallas_mode_is_public_and_changes_same_server_contract(tmp_path):
    fixture = _make_fixture(tmp_path)
    seal_zero, _ = _attest_fixture(fixture)
    for pid in {
        fixture.outer_pid,
        fixture.api_pid,
        fixture.engine_launcher_pid,
        fixture.engine_pid,
    }:
        fixture.write_environment(pid, pallas_attention="1")
    fixture.update_source_claim(pallas_attention="1")

    seal_one, _ = _attest_fixture(fixture)

    assert seal_zero.contract_sha256 != seal_one.contract_sha256
    assert seal_one.as_record()["environment"]["SKYRL_ROCM_PALLAS_ATTENTION"] == "1"
    with pytest.raises(_ATTEST.LocalServerAttestationError, match="contract changed"):
        _ATTEST.revalidate_local_server(
            seal_zero,
            proc_root=fixture.proc_root,
            health_probe=_HealthProbe(),
            expected_uid=fixture.uid,
        )


@pytest.mark.parametrize(
    "extra_environment",
    [
        {"JAX_DISABLE_JIT": "1"},
        {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.25"},
        {"PYTHONPATH": "/tmp/injected"},
        {"LD_PRELOAD": "/tmp/injected.so"},
    ],
)
def test_attestation_rejects_unexpected_accelerator_or_injection_environment(
    tmp_path, extra_environment
):
    fixture = _make_fixture(tmp_path)
    fixture.write_environment(fixture.engine_pid, extra_environment=extra_environment)

    with pytest.raises(
        _ATTEST.LocalServerAttestationError, match="environment|injection"
    ):
        _attest_fixture(fixture)


def test_public_record_recursively_redacts_unrelated_environment(tmp_path):
    fixture = _make_fixture(tmp_path)
    secret = "do-not-publish-this-sentinel"
    for pid in {
        fixture.outer_pid,
        fixture.api_pid,
        fixture.engine_launcher_pid,
        fixture.engine_pid,
    }:
        fixture.write_environment(pid, extra_environment={"PRIVATE_SENTINEL": secret})

    seal, _ = _attest_fixture(fixture)

    assert secret not in json.dumps(seal.as_record(), sort_keys=True)


def test_attestation_rejects_delete_journal_database(tmp_path):
    fixture = _make_fixture(tmp_path)
    with sqlite3.connect(fixture.database_path) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
        assert connection.execute("PRAGMA journal_mode=DELETE").fetchone() == (
            "delete",
        )

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="WAL"):
        _attest_fixture(fixture)


def test_attestation_rejects_nonexact_engine_launch_schema(tmp_path):
    fixture = _make_fixture(tmp_path)
    with sqlite3.connect(fixture.database_path) as connection:
        connection.execute("ALTER TABLE engine_launches ADD COLUMN injected TEXT")

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="schema"):
        _attest_fixture(fixture)


def test_attestation_rejects_engine_launch_table_trigger(tmp_path):
    fixture = _make_fixture(tmp_path)
    with closing(sqlite3.connect(fixture.database_path)) as connection:
        connection.execute(
            "CREATE TRIGGER mutate_ready AFTER UPDATE ON engine_launches BEGIN "
            "SELECT 1; END"
        )
        connection.commit()

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="trigger"):
        _attest_fixture(fixture)


def test_attestation_requires_transactional_request_deduplication_table(tmp_path):
    fixture = _make_fixture(tmp_path)
    with sqlite3.connect(fixture.database_path) as connection:
        connection.execute("DROP TABLE request_deduplication")

    with pytest.raises(
        _ATTEST.LocalServerAttestationError,
        match="request-deduplication schema",
    ):
        _attest_fixture(fixture)


def test_fixture_uses_exact_production_engine_launch_schema(tmp_path):
    fixture = _make_fixture(tmp_path)
    with closing(sqlite3.connect(fixture.database_path)) as connection:
        observed = connection.execute("PRAGMA table_info(engine_launches)").fetchall()

    assert tuple(observed) == _ATTEST._ENGINE_LAUNCH_SCHEMA


def test_attestation_rejects_extra_launch_row_in_fresh_database(tmp_path):
    fixture = _make_fixture(tmp_path)
    remaining_columns = ", ".join(_ATTEST._ENGINE_LAUNCH_SCHEMA_COLUMNS[1:])
    with closing(sqlite3.connect(fixture.database_path)) as connection:
        connection.execute(
            f"INSERT INTO engine_launches SELECT ?, {remaining_columns} "  # noqa: S608 - fixed test schema
            "FROM engine_launches",
            ("e" * 32,),
        )
        connection.commit()

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="one launch row"):
        _attest_fixture(fixture)


def test_attestation_rejects_suffix_ready_status(tmp_path):
    fixture = _make_fixture(tmp_path)
    fixture.update_launch(status="evil.READY")

    with pytest.raises(
        _ATTEST.LocalServerAttestationError, match="exact matching READY"
    ):
        _attest_fixture(fixture)


def test_same_seal_revalidation_rejects_database_inode_replacement(tmp_path):
    fixture = _make_fixture(tmp_path)
    seal, _ = _attest_fixture(fixture)
    replacement = fixture.database_path.parent / "replacement.db"
    with (
        sqlite3.connect(fixture.database_path) as source,
        sqlite3.connect(replacement) as destination,
    ):
        source.backup(destination)
        assert destination.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
    replacement.chmod(0o600)
    os.replace(replacement, fixture.database_path)

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="contract changed"):
        _ATTEST.revalidate_local_server(
            seal,
            proc_root=fixture.proc_root,
            health_probe=_HealthProbe(),
            expected_uid=fixture.uid,
        )


def test_same_seal_allows_only_dynamic_heartbeat_updates(tmp_path):
    fixture = _make_fixture(tmp_path)
    seal, _ = _attest_fixture(fixture)
    timestamp = "2026-07-14T08:00:01+00:00"
    fixture.update_launch(
        heartbeat_at=timestamp,
        heartbeat_monotonic_ns=time.monotonic_ns(),
        heartbeat_sequence=8,
        updated_at=timestamp,
    )

    result = _ATTEST.revalidate_local_server(
        seal,
        proc_root=fixture.proc_root,
        health_probe=_HealthProbe(),
        expected_uid=fixture.uid,
    )

    assert result["status"] == "passed"


def test_attestation_rejects_forged_runtime_handoff(tmp_path):
    fixture = _make_fixture(tmp_path)
    fixture.update_launch(runtime_handoff_attestation=json.dumps({"status": "passed"}))

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="handoff"):
        _attest_fixture(fixture)


def test_attestation_rejects_process_only_unhardened_source_claim(tmp_path):
    fixture = _make_fixture(tmp_path)
    fixture.update_launch(
        api_source_attestation=json.dumps({"status": "not_required", "role": "api"}),
        engine_source_attestation=json.dumps(
            {"status": "not_required", "role": "engine"}
        ),
    )

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="source"):
        _attest_fixture(fixture)


def test_revalidation_rejects_source_snapshot_mutation(tmp_path):
    fixture = _make_fixture(tmp_path)
    seal, health = _attest_fixture(fixture)
    source = fixture.source_root / "skyrl/tinker/api.py"
    payload = bytearray(source.read_bytes())
    payload[-2] ^= 1
    source.write_bytes(payload)
    source.chmod(0o600)

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="source-cache"):
        _ATTEST.revalidate_local_server(
            seal,
            proc_root=fixture.proc_root,
            health_probe=health,
            expected_uid=fixture.uid,
        )


def test_attestation_rejects_server_from_stale_expected_head(tmp_path):
    fixture = _make_fixture(tmp_path)
    kwargs = _attestation_kwargs(fixture)
    kwargs["expected_git_head"] = "e" * 40

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="source"):
        _ATTEST.attest_local_server(
            server_pid=fixture.outer_pid,
            base_url=fixture.base_url,
            **kwargs,
        )


def test_attestation_rejects_python_payload_changed_after_client_pin(tmp_path):
    fixture = _make_fixture(tmp_path)
    kwargs = _attestation_kwargs(fixture)
    payload = bytearray(fixture.python_path.read_bytes())
    payload[0] ^= 1
    fixture.python_path.write_bytes(payload)

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="Python payload"):
        _ATTEST.attest_local_server(
            server_pid=fixture.outer_pid,
            base_url=fixture.base_url,
            **kwargs,
        )


def test_attestation_rejects_noncanonical_runtime_lock_root(tmp_path):
    fixture = _make_fixture(tmp_path)
    kwargs = _attestation_kwargs(fixture)
    kwargs["runtime_root"] = tmp_path

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="launch-lock"):
        _ATTEST.attest_local_server(
            server_pid=fixture.outer_pid,
            base_url=fixture.base_url,
            **kwargs,
        )


@pytest.mark.parametrize(
    "environment_change",
    [
        {"SKYRL_QWEN35_RUNTIME_GIT_HEAD": "e" * 40},
        {"SKYRL_QWEN35_RUNTIME_REPO_ROOT": "/tmp/other-repo"},
        {"SKYRL_QWEN35_RUNTIME_UNEXPECTED": "1"},
    ],
)
def test_attestation_rejects_runtime_source_environment_drift(
    tmp_path, environment_change
):
    fixture = _make_fixture(tmp_path)
    fixture.write_environment(
        fixture.engine_pid,
        extra_environment=environment_change,
    )

    with pytest.raises(
        _ATTEST.LocalServerAttestationError, match="environment|runtime-source"
    ):
        _attest_fixture(fixture)


def test_attestation_rejects_unexpected_outer_child(tmp_path):
    fixture = _make_fixture(tmp_path)
    extra_pid = 104
    _write_process(
        fixture,
        extra_pid,
        ppid=fixture.outer_pid,
        process_group=50,
        session=50,
        start_ticks=500,
        executable=fixture.python_path,
    )

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="child set"):
        _attest_fixture(fixture)


def test_attestation_rejects_unexpected_engine_child(tmp_path):
    fixture = _make_fixture(tmp_path)
    extra_pid = 104
    _write_process(
        fixture,
        extra_pid,
        ppid=fixture.engine_pid,
        process_group=fixture.engine_launcher_pid,
        session=fixture.engine_launcher_pid,
        start_ticks=500,
        executable=fixture.python_path,
    )

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="leaf process"):
        _attest_fixture(fixture)


def test_attestation_requires_full_startup_cache_when_requested(tmp_path):
    fixture = _make_fixture(tmp_path)
    kwargs = _attestation_kwargs(fixture)
    kwargs["require_startup_cache"] = True

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="startup-cache"):
        _ATTEST.attest_local_server(
            server_pid=fixture.outer_pid,
            base_url=fixture.base_url,
            **kwargs,
        )


def _required_startup_cache(fixture: _ServerFixture) -> dict[str, object]:
    return {
        "status": "required-v1",
        "schema_name": "skyrl.qwen35.persistent-cache-attestation",
        "schema_version": 1,
        "seed": {
            "source_git_head": _EXPECTED_GIT_HEAD,
            "source_git_tree": _EXPECTED_GIT_TREE,
            "cache_path": str(fixture.cache_directory),
            "attention_backend": "xla",
            "construction": "eager",
            "model": "Qwen/Qwen3.5-4B",
            "model_revision": _ATTEST.EXPECTED_MODEL_REVISION,
            "model_path": str(fixture.model_path),
            "compile_target": "train_bucket_forward_backward_accumulate",
            "bucket": 64,
            "batch_size": 1,
            "xla_flags": _ATTEST.EXPECTED_XLA_FLAGS,
            "graph_api_used": False,
            "command_buffer_used": False,
            "backend_config": fixture.backend_config,
            "backend_config_sha256": "3" * 64,
        },
        "prewarm_audit": {
            "path": str(fixture.database_path.parent / "prewarm.jsonl"),
            "sha256": "1" * 64,
        },
        "prewarm_handoff": {
            "path": str(fixture.database_path.parent / "prewarm-handoff.jsonl"),
            "sha256": "2" * 64,
        },
    }


def _install_required_startup_cache(
    fixture: _ServerFixture, startup_cache: dict[str, object]
) -> None:
    with closing(sqlite3.connect(fixture.database_path)) as connection:
        row = connection.execute(
            "SELECT api_source_attestation, engine_source_attestation, "
            "runtime_handoff_attestation FROM engine_launches"
        ).fetchone()
        assert row is not None
        api_source, engine_source, handoff = (json.loads(value) for value in row)
        api_source["startup_cache_attestation"] = startup_cache
        engine_source["startup_cache_attestation"] = startup_cache
        handoff["startup_cache_attestation"] = startup_cache
        connection.execute(
            "UPDATE engine_launches SET api_source_attestation = ?, "
            "engine_source_attestation = ?, runtime_handoff_attestation = ?, "
            "cache_evidence_status = ?, cache_evidence = ?",
            (
                json.dumps(api_source),
                json.dumps(engine_source),
                json.dumps(handoff),
                _ATTEST.RUNTIME_CACHE_HIT_KIND,
                json.dumps({"kind": _ATTEST.RUNTIME_CACHE_HIT_KIND}),
            ),
        )
        connection.commit()
    for pid in {
        fixture.outer_pid,
        fixture.api_pid,
        fixture.engine_launcher_pid,
        fixture.engine_pid,
    }:
        fixture.write_environment(pid)


def test_required_startup_cache_invokes_full_validator_initially_and_finally(
    tmp_path,
):
    fixture = _make_fixture(tmp_path)
    startup_cache = _required_startup_cache(fixture)
    _install_required_startup_cache(fixture, startup_cache)

    events = []

    def validate_cache(repo, source_root, claim, evidence, boot_id_path):
        events.append("cache")
        assert repo == fixture.repo_root
        assert source_root == fixture.source_root
        assert claim == startup_cache
        assert evidence == {"kind": _ATTEST.RUNTIME_CACHE_HIT_KIND}
        assert boot_id_path == fixture.proc_root / "sys/kernel/random/boot_id"

    kwargs = _attestation_kwargs(fixture)
    original_source_validator = kwargs["source_cache_validator"]

    def validate_source(repo, head, home):
        events.append("source")
        return original_source_validator(repo, head, home)

    kwargs["require_startup_cache"] = True
    kwargs["source_cache_validator"] = validate_source
    kwargs["cache_evidence_validator"] = validate_cache

    seal = _ATTEST.attest_local_server(
        server_pid=fixture.outer_pid,
        base_url=fixture.base_url,
        **kwargs,
    )

    assert seal.as_record()["cache"]["required_for_gate"] is True
    assert events == ["source", "cache", "source", "cache"]


@pytest.mark.parametrize(
    ("field", "mutated"),
    [
        ("source_git_head", "d" * 40),
        ("source_git_tree", "e" * 40),
        ("cache_path", "/tmp/wrong-cache"),
        ("attention_backend", "pallas"),
        ("construction", "abstract-load"),
        ("model_path", "/tmp/wrong-model"),
        ("backend_config.max_lora_rank", 16),
    ],
)
def test_required_cache_seed_must_match_observed_server_contract(
    tmp_path, field, mutated
):
    fixture = _make_fixture(tmp_path)
    startup_cache = _required_startup_cache(fixture)
    seed = startup_cache["seed"]
    assert isinstance(seed, dict)
    if field.startswith("backend_config."):
        backend = seed["backend_config"]
        assert isinstance(backend, dict)
        backend[field.split(".", 1)[1]] = mutated
    else:
        seed[field] = mutated
    _install_required_startup_cache(fixture, startup_cache)
    validator_called = False

    def must_not_validate(*args):
        nonlocal validator_called
        validator_called = True

    kwargs = _attestation_kwargs(fixture)
    kwargs["require_startup_cache"] = True
    kwargs["cache_evidence_validator"] = must_not_validate

    with pytest.raises(
        _ATTEST.LocalServerAttestationError,
        match="cache seed is not cross-bound",
    ):
        _ATTEST.attest_local_server(
            server_pid=fixture.outer_pid,
            base_url=fixture.base_url,
            **kwargs,
        )

    assert validator_called is False


def test_default_cache_validator_loads_attested_private_snapshot_helper(tmp_path):
    fixture = _make_fixture(tmp_path)
    helper = fixture.source_root / "rocm/qwen35_cache_attestation.py"
    helper.parent.mkdir(mode=0o700)
    helper.write_text(
        "def revalidate_startup_cache_claim(claim, *, boot_id_path):\n"
        "    assert claim == {'claim': 'exact'}\n"
        "    assert boot_id_path.name == 'boot_id'\n"
        "    return {'rebuilt': True}\n"
        "def validate_runtime_cache_evidence(claim, evidence):\n"
        "    assert claim == {'rebuilt': True}\n"
        "    assert evidence == {'evidence': 'exact'}\n",
        encoding="ascii",
    )
    helper.chmod(0o600)
    nodes_before = sorted(
        path.relative_to(fixture.source_root)
        for path in fixture.source_root.rglob("*")
    )

    _ATTEST._default_cache_evidence_validator(
        fixture.repo_root,
        fixture.source_root,
        {"claim": "exact"},
        {"evidence": "exact"},
        fixture.proc_root / "sys/kernel/random/boot_id",
    )
    nodes_after = sorted(
        path.relative_to(fixture.source_root)
        for path in fixture.source_root.rglob("*")
    )

    assert nodes_after == nodes_before
    assert not any("__pycache__" in path.parts for path in nodes_after)


def test_attestation_rejects_api_process_group_or_session_drift(tmp_path):
    fixture = _make_fixture(tmp_path)
    fixture.rewrite_stat(
        fixture.api_pid,
        ppid=fixture.outer_pid,
        process_group=999,
        session=999,
        start_ticks=200,
    )

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="ancestry"):
        _attest_fixture(fixture)


def test_base_url_rejects_port_zero(tmp_path):
    fixture = _make_fixture(tmp_path)

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="base URL"):
        _ATTEST.attest_local_server(
            server_pid=fixture.outer_pid,
            base_url="http://127.0.0.1:0",
            **_attestation_kwargs(fixture),
        )


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8001",
        "https://127.0.0.1:8001",
        "http://127.0.0.1:08001",
        "http://user@127.0.0.1:8001",
        "http://127.0.0.1:8001/",
        "http://127.0.0.1:8001?x=1",
    ],
)
def test_attestation_rejects_noncanonical_loopback_urls(tmp_path, base_url):
    fixture = _make_fixture(tmp_path)

    with pytest.raises(
        _ATTEST.LocalServerAttestationError, match="base URL must be exactly"
    ):
        _ATTEST.attest_local_server(
            server_pid=fixture.outer_pid,
            base_url=base_url,
            **_attestation_kwargs(fixture),
        )


@pytest.mark.parametrize(
    "raw_config",
    [
        (
            '{"max_lora_adapters":2,"max_lora_rank":8,'
            '"train_micro_batch_size":1,"sample_max_num_sequences":true,'
            '"gradient_checkpointing":true,"loss_chunk_size":64,'
            '"abstract_model_load":false}'
        ),
        (
            '{"max_lora_adapters":2,"max_lora_rank":8,'
            '"train_micro_batch_size":1,"sample_max_num_sequences":2,'
            '"gradient_checkpointing":true,"loss_chunk_size":64,'
            '"abstract_model_load":false}'
        ),
        (
            '{"max_lora_adapters":2,"max_lora_rank":8,'
            '"train_micro_batch_size":1,"sample_max_num_sequences":1,'
            '"gradient_checkpointing":true,"loss_chunk_size":64,'
            '"abstract_model_load":false,"unexpected":0}'
        ),
        (
            '{"max_lora_adapters":2,"max_lora_rank":8,'
            '"train_micro_batch_size":1,"sample_max_num_sequences":1,'
            '"gradient_checkpointing":true,"loss_chunk_size":64,'
            '"abstract_model_load":false,"sample_max_num_sequences":1}'
        ),
    ],
)
def test_attestation_rejects_nonexact_backend_config(tmp_path, raw_config):
    fixture = _make_fixture(tmp_path)
    fixture.install_commands(raw_config)

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="backend config"):
        _attest_fixture(fixture)


@pytest.mark.parametrize("role_pid", [100, 101, 102, 103])
def test_attestation_requires_exact_xla_flags_in_every_process(tmp_path, role_pid):
    fixture = _make_fixture(tmp_path)
    fixture.write_environment(
        role_pid, xla_flags="--xla_gpu_enable_command_buffer= --xla_dump_to=/tmp"
    )

    with pytest.raises(
        _ATTEST.LocalServerAttestationError, match="XLA_FLAGS|API child"
    ):
        _attest_fixture(fixture)


def test_attestation_rejects_duplicate_xla_environment_entries(tmp_path):
    fixture = _make_fixture(tmp_path)
    fixture.write_environment(fixture.engine_pid, duplicate=True)

    with pytest.raises(
        _ATTEST.LocalServerAttestationError, match="duplicate XLA_FLAGS"
    ):
        _attest_fixture(fixture)


@pytest.mark.parametrize(
    ("environment_changes", "message"),
    [
        ({"jax_platforms": "cpu"}, "JAX_PLATFORMS"),
        ({"jax_platforms": None}, "JAX_PLATFORMS"),
        ({"rocr_visible_devices": "1"}, "ROCR_VISIBLE_DEVICES"),
        ({"rocr_visible_devices": None}, "ROCR_VISIBLE_DEVICES"),
    ],
)
def test_attestation_requires_rocm_device_zero_in_every_process(
    tmp_path, environment_changes, message
):
    fixture = _make_fixture(tmp_path)
    fixture.write_environment(fixture.engine_pid, **environment_changes)

    with pytest.raises(_ATTEST.LocalServerAttestationError, match=message):
        _attest_fixture(fixture)


def test_growth_mode_requires_preallocation_disabled(tmp_path):
    fixture = _make_fixture(tmp_path)
    fixture.write_environment(fixture.engine_pid, preallocate="true")

    with pytest.raises(
        _ATTEST.LocalServerAttestationError,
        match="XLA_PYTHON_CLIENT_PREALLOCATE.*growth",
    ):
        _attest_fixture(fixture)


def test_preallocate85_mode_requires_and_records_exact_allocator_contract(tmp_path):
    fixture = _make_fixture(tmp_path)
    config = fixture.backend_config
    config["abstract_model_load"] = True
    fixture.install_commands(json.dumps(config, separators=(",", ":")))
    memory_environment = {
        "XLA_PYTHON_CLIENT_ALLOCATOR": "bfc",
        "XLA_CLIENT_MEM_FRACTION": "0.85",
        "HIP_VISIBLE_DEVICES": "0",
        "GPU_DEVICE_ORDINAL": "0",
    }
    for pid in {
        fixture.outer_pid,
        fixture.api_pid,
        fixture.engine_launcher_pid,
        fixture.engine_pid,
    }:
        fixture.write_environment(
            pid,
            preallocate="true",
            memory_environment=memory_environment,
        )

    seal, _ = _attest_fixture(fixture)

    record = seal.as_record()
    assert record["backend"]["abstract_model_load"] is True
    assert record["environment"]["memory_mode"] == "preallocate85"
    assert record["environment"]["memory_environment"] == {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        **memory_environment,
    }


def test_preallocate85_mode_rejects_missing_allocator_contract(tmp_path):
    fixture = _make_fixture(tmp_path)
    config = fixture.backend_config
    config["abstract_model_load"] = True
    fixture.install_commands(json.dumps(config, separators=(",", ":")))
    for pid in {
        fixture.outer_pid,
        fixture.api_pid,
        fixture.engine_launcher_pid,
        fixture.engine_pid,
    }:
        fixture.write_environment(pid, preallocate="true")

    with pytest.raises(
        _ATTEST.LocalServerAttestationError,
        match="preallocate85.*XLA_PYTHON_CLIENT_ALLOCATOR",
    ):
        _attest_fixture(fixture)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("backend", "fsdp", "backend or boot"),
        ("status", "FAILED", "one exact matching READY"),
        ("engine_start_ticks", 401, "start identity"),
        ("engine_pid", 999, "cannot inspect engine process"),
    ],
)
def test_attestation_rejects_readiness_identity_mismatches(
    tmp_path, column, value, message
):
    fixture = _make_fixture(tmp_path)
    fixture.update_launch(**{column: value})

    with pytest.raises(_ATTEST.LocalServerAttestationError, match=message):
        _attest_fixture(fixture)


def test_attestation_rejects_listener_not_owned_by_api(tmp_path):
    fixture = _make_fixture(tmp_path)
    (fixture.process_root(fixture.api_pid) / "fd/7").unlink()

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="not held"):
        _attest_fixture(fixture)


def test_attestation_ignores_only_vanished_unrelated_api_fd(tmp_path, monkeypatch):
    fixture = _make_fixture(tmp_path)
    transient = fixture.process_root(fixture.api_pid) / "fd/8"
    transient.symlink_to("socket:[999999]")
    original_readlink = os.readlink
    vanished = False

    def race_readlink(path):
        nonlocal vanished
        if Path(path) == transient and not vanished:
            vanished = True
            raise FileNotFoundError(path)
        return original_readlink(path)

    monkeypatch.setattr(_ATTEST.os, "readlink", race_readlink)

    seal, _ = _attest_fixture(fixture)

    assert vanished is True
    assert seal.as_record()["status"] == "passed"


@pytest.mark.parametrize(
    "health", [{"status": "starting"}, {"status": "ok", "extra": 1}]
)
def test_attestation_rejects_nonexact_health_response(tmp_path, health):
    fixture = _make_fixture(tmp_path)

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="exactly"):
        _attest_fixture(fixture, _HealthProbe(result=health))


def test_revalidation_rejects_process_identity_change(tmp_path):
    fixture = _make_fixture(tmp_path)
    seal, _ = _attest_fixture(fixture)
    fixture.rewrite_stat(
        fixture.engine_pid,
        ppid=fixture.engine_launcher_pid,
        process_group=fixture.engine_launcher_pid,
        session=fixture.engine_launcher_pid,
        start_ticks=401,
    )
    fixture.update_launch(engine_start_ticks=401)

    with pytest.raises(_ATTEST.LocalServerAttestationError, match="contract changed"):
        _ATTEST.revalidate_local_server(
            seal,
            proc_root=fixture.proc_root,
            health_probe=_HealthProbe(),
            expected_uid=fixture.uid,
        )
