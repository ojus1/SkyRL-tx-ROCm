from __future__ import annotations

import os
from pathlib import Path

import pytest

from skyrl.tinker import runtime_source

_TEST_JAX_CACHE_RELATIVE = Path(
    ".cache/skyrl-jax-rocm-private-v1/test-stack-namespace"
)


def _write(path: Path, payload: bytes = b"# source\n", mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)
    return path


@pytest.fixture(autouse=True)
def _stub_full_source_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    def validate_full_source_cache(
        *,
        repo_root: Path,
        head: str,
        account_home: Path,
        source_root: Path,
        jax_cache: Path,
    ) -> dict[str, object]:
        assert repo_root.name == "repo"
        assert source_root.name == "source-head"
        if jax_cache != account_home / _TEST_JAX_CACHE_RELATIVE:
            raise runtime_source.RuntimeSourceError(
                "runtime JAX compilation cache is not the exact stack namespace"
            )
        return {
            "cache_status": "validated",
            "format": "skyrl-private-source-cache-v1",
            "git_head": head,
            "git_tree": "b" * len(head),
            "source_archive_path": str(source_root.parent / "source-head.tar"),
            "source_archive_sha256": "c" * 64,
            "source_file_count": 1090,
            "source_snapshot_root": str(source_root),
            "source_total_bytes": 35_333_482,
            "full_head_tree_validated": True,
            "expected_jax_cache": str(account_home / _TEST_JAX_CACHE_RELATIVE),
        }

    monkeypatch.setattr(
        runtime_source,
        "_validate_full_source_cache",
        validate_full_source_cache,
    )
    monkeypatch.setattr(
        runtime_source,
        "validate_runtime_launch_lock",
        lambda _environment=None: {
            "status": "passed",
            "descriptor": 10,
            "path": "/run/user/1000/skyrl-qwen35-rocm-1000",
            "inheritable": True,
            "exclusive_lock_observed": True,
        },
    )
    monkeypatch.setattr(
        runtime_source,
        "_validate_uv_payload",
        lambda _path: runtime_source._UV_SHA256,
    )


def _fixture(tmp_path: Path, role: str = "api") -> dict[str, object]:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    account_cache = home / ".cache"
    account_cache.mkdir(mode=0o700)
    head = "a" * 40
    cache_root = account_cache / "skyrl-source-snapshots-private-v1"
    cache_root.mkdir(mode=0o700)
    commit_root = cache_root / head
    commit_root.mkdir(mode=0o700)
    source_root = commit_root / "source-head"
    source_root.mkdir(mode=0o700)
    package = _write(source_root / "skyrl/__init__.py")
    module = _write(source_root / runtime_source._ROLE_PATHS[role])
    for directory in (source_root / "skyrl", source_root / "skyrl/tinker"):
        directory.chmod(0o700)

    repo = tmp_path / "repo"
    repo.mkdir(mode=0o700)
    venv = repo / ".venv"
    (venv / "bin").mkdir(parents=True, mode=0o700)
    uv_executable = _write(home / ".local/bin/uv", b"uv\n", mode=0o755)
    (home / ".local").chmod(0o700)
    (home / ".local/bin").chmod(0o700)
    jax_cache = home / _TEST_JAX_CACHE_RELATIVE
    jax_cache.mkdir(parents=True, mode=0o700)
    (jax_cache.parent).chmod(0o700)
    uv_depth = "1" if role == "api" else "2"
    environment = {
        **runtime_source._REQUIRED_ENVIRONMENT,
        **runtime_source._MEMORY_MODE_ENVIRONMENT["growth"],
        "HOME": str(home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": ":".join(
            [f"{venv}/bin"] * int(uv_depth)
            + ["/opt/rocm/bin", "/usr/bin", "/bin"]
        ),
        "UV": str(uv_executable),
        "UV_RUN_RECURSION_DEPTH": uv_depth,
        "VIRTUAL_ENV": str(venv),
        "JAX_COMPILATION_CACHE_DIR": str(jax_cache),
        "SKYRL_QWEN35_RUNTIME_GIT_HEAD": head,
        "SKYRL_QWEN35_RUNTIME_MEMORY_MODE": "growth",
        "SKYRL_QWEN35_RUNTIME_REPO_ROOT": str(repo),
        "SKYRL_QWEN35_RUNTIME_SOURCE_ROOT": str(source_root),
        "SKYRL_QWEN35_RUNTIME_UV_EXECUTABLE": str(uv_executable),
        "SKYRL_ROCM_PALLAS_ATTENTION": "1",
    }
    return {
        "environment": environment,
        "head": head,
        "jax_cache": jax_cache,
        "module": module,
        "package": package,
        "repo": repo,
        "source_root": source_root,
        "uv_executable": uv_executable,
    }


@pytest.mark.parametrize("role", ["api", "engine"])
def test_runtime_source_claim_binds_full_tree_cwd_and_graph_free_cache_policy(
    tmp_path: Path, role: str
) -> None:
    fixture = _fixture(tmp_path, role)

    result = runtime_source.validate_runtime_source(
        role=role,
        module_file=fixture["module"],
        package_file=fixture["package"],
        environment=fixture["environment"],
        cwd=fixture["source_root"],
        dont_write_bytecode=True,
    )

    assert result == {
        "status": "passed",
        "role": role,
        "git_head": fixture["head"],
        "git_tree": "b" * 40,
        "source_archive_path": str(
            Path(fixture["source_root"]).parent / "source-head.tar"
        ),
        "source_archive_sha256": "c" * 64,
        "source_file_count": 1090,
        "source_total_bytes": 35_333_482,
        "full_head_tree_validated": True,
        "source_root": str(fixture["source_root"]),
        "repo_root": str(fixture["repo"]),
        "working_directory": str(fixture["source_root"]),
        "module_origin": str(fixture["module"]),
        "package_origin": str(fixture["package"]),
        "uv_executable": str(fixture["uv_executable"]),
        "uv_sha256": runtime_source._UV_SHA256,
        "launch_lock": {
            "status": "passed",
            "descriptor": 10,
            "path": "/run/user/1000/skyrl-qwen35-rocm-1000",
            "inheritable": True,
            "exclusive_lock_observed": True,
        },
        "jax_compilation_cache": str(fixture["jax_cache"]),
        "memory_mode": "growth",
        "xla_flags": "--xla_gpu_enable_command_buffer=",
        "jax_enable_pgle": "false",
        "jax_compilation_cache_expect_pgle": "false",
        "pallas_attention": "1",
        "rocprof_attach_enabled": False,
        "startup_cache_attestation": {"status": "not_required"},
        "dont_write_bytecode": True,
    }


def test_generic_runtime_without_claims_remains_unchanged() -> None:
    assert runtime_source.validate_runtime_source(
        role="api",
        module_file=Path("not-used"),
        package_file=Path("not-used"),
        environment={},
    ) == {"status": "not_required", "role": "api"}


def test_runtime_source_accepts_exact_rocprof_attach_mode(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["environment"][runtime_source._ROCPROF_ATTACH_ENV] = "1"

    result = runtime_source.validate_runtime_source(
        role="api",
        module_file=fixture["module"],
        package_file=fixture["package"],
        environment=fixture["environment"],
        cwd=fixture["source_root"],
        dont_write_bytecode=True,
    )

    assert result["rocprof_attach_enabled"] is True


@pytest.mark.parametrize("value", ["", "0", "true", "2"])
def test_runtime_source_rejects_nonexact_rocprof_attach_mode(
    tmp_path: Path, value: str
) -> None:
    fixture = _fixture(tmp_path)
    fixture["environment"][runtime_source._ROCPROF_ATTACH_ENV] = value

    with pytest.raises(runtime_source.RuntimeSourceError, match="ROCP_TOOL_ATTACH"):
        runtime_source.validate_runtime_source(
            role="api",
            module_file=fixture["module"],
            package_file=fixture["package"],
            environment=fixture["environment"],
            cwd=fixture["source_root"],
            dont_write_bytecode=True,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ROCPROF_OUTPUT_PATH", "/tmp/unattested"),
        ("ROCPROFILER_REGISTER_ENABLED", "0"),
    ],
)
def test_runtime_source_rejects_rocprofiler_implementation_environment(
    tmp_path: Path, name: str, value: str
) -> None:
    fixture = _fixture(tmp_path)
    fixture["environment"][name] = value

    with pytest.raises(
        runtime_source.RuntimeSourceError,
        match="unexpected_accelerator_environment",
    ):
        runtime_source.validate_runtime_source(
            role="api",
            module_file=fixture["module"],
            package_file=fixture["package"],
            environment=fixture["environment"],
            cwd=fixture["source_root"],
            dont_write_bytecode=True,
        )


def _cache_claim_environment() -> dict[str, str]:
    return {
        runtime_source._T64_CACHE_ATTEST_ENV: "required-v1",
        runtime_source._PREWARM_AUDIT_PATH_ENV: "/private/run/prewarm.jsonl",
        runtime_source._PREWARM_AUDIT_SHA256_ENV: "d" * 64,
        runtime_source._PREWARM_HANDOFF_PATH_ENV: (
            "/private/run/prewarm-handoff.jsonl"
        ),
        runtime_source._PREWARM_HANDOFF_SHA256_ENV: "e" * 64,
    }


@pytest.mark.parametrize("missing", runtime_source._CACHE_ATTESTATION_ENVIRONMENT)
def test_runtime_cache_claim_is_all_or_none(missing: str) -> None:
    environment = _cache_claim_environment()
    del environment[missing]

    with pytest.raises(runtime_source.RuntimeSourceError, match="incomplete"):
        runtime_source.validate_runtime_source(
            role="api",
            module_file=Path("unused"),
            package_file=Path("unused"),
            environment=environment,
        )


def test_runtime_cache_claim_cannot_exist_without_hardened_source() -> None:
    with pytest.raises(runtime_source.RuntimeSourceError, match="hardened source"):
        runtime_source.validate_runtime_source(
            role="api",
            module_file=Path("unused"),
            package_file=Path("unused"),
            environment=_cache_claim_environment(),
        )


def test_runtime_cache_claim_mode_is_exact() -> None:
    environment = _cache_claim_environment()
    environment[runtime_source._T64_CACHE_ATTEST_ENV] = "required-v2"

    with pytest.raises(runtime_source.RuntimeSourceError, match="required-v1"):
        runtime_source.validate_runtime_source(
            role="api",
            module_file=Path("unused"),
            package_file=Path("unused"),
            environment=environment,
        )


def test_required_runtime_cache_claim_is_shared_source_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = _fixture(tmp_path, "api")
    fixture["environment"].update(_cache_claim_environment())
    observed: dict[str, object] = {}
    expected = {
        "status": "required-v1",
        "schema_version": 1,
        "seed": {"bucket": 64},
    }

    def validate(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(runtime_source, "_validate_startup_cache_claim", validate)

    result = runtime_source.validate_runtime_source(
        role="api",
        module_file=fixture["module"],
        package_file=fixture["package"],
        environment=fixture["environment"],
        cwd=fixture["source_root"],
        dont_write_bytecode=True,
    )

    assert result["startup_cache_attestation"] == expected
    assert observed["source_root"] == fixture["source_root"]
    assert observed["git_head"] == fixture["head"]
    assert observed["git_tree"] == "b" * 40
    assert observed["jax_cache"] == fixture["jax_cache"]
    assert observed["attention_backend"] == "1"


def test_runtime_source_revalidation_rejects_attestation_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_source,
        "validate_runtime_source",
        lambda **_kwargs: {"status": "passed", "role": "api"},
    )

    with pytest.raises(runtime_source.RuntimeSourceError, match="changed after"):
        runtime_source.revalidate_runtime_source(
            initial={"status": "not_required", "role": "api"},
            role="api",
            module_file="unused",
            package_file="unused",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("head", "lowercase object ID"),
        ("cwd", "working directory"),
        ("xla", "policy mismatch"),
        ("bytecode", "policy mismatch"),
        ("module", "did not resolve"),
        ("jax_cache_mode", "exact mode 0700"),
        ("jax_cache_namespace", "exact stack namespace"),
        ("snapshot_venv", "must not contain"),
        ("unexpected_jax", "unexpected_accelerator_environment"),
        ("unexpected_uv", "unexpected names"),
        ("uv", "fixed account path"),
    ],
)
def test_runtime_source_claim_fails_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    fixture = _fixture(tmp_path)
    environment = dict(fixture["environment"])
    cwd = fixture["source_root"]
    module = fixture["module"]
    dont_write_bytecode = True
    if mutation == "head":
        environment["SKYRL_QWEN35_RUNTIME_GIT_HEAD"] = "A" * 40
    elif mutation == "cwd":
        cwd = tmp_path
    elif mutation == "xla":
        environment["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer=fusion"
    elif mutation == "bytecode":
        dont_write_bytecode = False
    elif mutation == "module":
        module = _write(Path(fixture["source_root"]) / "skyrl/tinker/wrong.py")
    elif mutation == "jax_cache_mode":
        Path(fixture["jax_cache"]).chmod(0o755)
    elif mutation == "jax_cache_namespace":
        wrong_cache = Path(fixture["jax_cache"]).parent / "wrong"
        wrong_cache.mkdir(mode=0o700)
        environment["JAX_COMPILATION_CACHE_DIR"] = str(wrong_cache)
    elif mutation == "snapshot_venv":
        (Path(fixture["source_root"]) / ".venv").mkdir(mode=0o700)
    elif mutation == "unexpected_jax":
        environment["JAX_DISABLE_JIT"] = "1"
    elif mutation == "unexpected_uv":
        environment["UV_PROJECT"] = "/tmp/redirect"
    elif mutation == "uv":
        wrong_uv = _write(tmp_path / "uv", b"uv\n", mode=0o755)
        environment["SKYRL_QWEN35_RUNTIME_UV_EXECUTABLE"] = str(wrong_uv)
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(runtime_source.RuntimeSourceError, match=message):
        runtime_source.validate_runtime_source(
            role="api",
            module_file=module,
            package_file=fixture["package"],
            environment=environment,
            cwd=cwd,
            dont_write_bytecode=dont_write_bytecode,
        )


def test_runtime_source_rejects_incomplete_claims(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    environment = dict(fixture["environment"])
    environment.pop("SKYRL_QWEN35_RUNTIME_GIT_HEAD")

    with pytest.raises(runtime_source.RuntimeSourceError, match="incomplete"):
        runtime_source.validate_runtime_source(
            role="api",
            module_file=fixture["module"],
            package_file=fixture["package"],
            environment=environment,
            cwd=fixture["source_root"],
            dont_write_bytecode=True,
        )


def test_runtime_source_files_must_be_private_and_singly_linked(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    module = Path(fixture["module"])
    os.link(module, tmp_path / "second-module-link")

    with pytest.raises(runtime_source.RuntimeSourceError, match="singly linked"):
        runtime_source.validate_runtime_source(
            role="api",
            module_file=module,
            package_file=fixture["package"],
            environment=fixture["environment"],
            cwd=fixture["source_root"],
            dont_write_bytecode=True,
        )


def test_preallocate_runtime_requires_every_allocator_and_visibility_setting(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    environment = dict(fixture["environment"])
    environment["SKYRL_QWEN35_RUNTIME_MEMORY_MODE"] = "preallocate85"
    environment.pop("XLA_PYTHON_CLIENT_PREALLOCATE")
    environment.update(runtime_source._MEMORY_MODE_ENVIRONMENT["preallocate85"])

    runtime_source.validate_runtime_source(
        role="api",
        module_file=fixture["module"],
        package_file=fixture["package"],
        environment=environment,
        cwd=fixture["source_root"],
        dont_write_bytecode=True,
    )

    environment.pop("XLA_PYTHON_CLIENT_ALLOCATOR")
    with pytest.raises(runtime_source.RuntimeSourceError, match="policy mismatch"):
        runtime_source.validate_runtime_source(
            role="api",
            module_file=fixture["module"],
            package_file=fixture["package"],
            environment=environment,
            cwd=fixture["source_root"],
            dont_write_bytecode=True,
        )
