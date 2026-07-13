from __future__ import annotations

import ast
import importlib.util
import json
import os
import stat
import subprocess
import sys
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOOL = _REPO / "rocm" / "prewarm_qwen35_buckets.py"
_LAUNCHER = _REPO / "rocm" / "start_qwen35.sh"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_TOOL_SPEC = importlib.util.spec_from_file_location("qwen35_bucket_prewarm", _TOOL)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
prewarm = importlib.util.module_from_spec(_TOOL_SPEC)
_TOOL_SPEC.loader.exec_module(prewarm)


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in (
        "JAX_PLATFORMS",
        "JAX_COMPILATION_CACHE_DIR",
        "SKYRL_ROCM_PALLAS_ATTENTION",
        "XLA_FLAGS",
    ):
        environment.pop(name, None)
    return subprocess.run(
        [sys.executable, str(_TOOL), *arguments],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def test_default_is_cpu_only_plan_without_jax_or_gpu() -> None:
    result = _run()

    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert [record["record_type"] for record in records] == ["manifest", "plan"]
    manifest, plan = records
    assert manifest["mode"] == "cpu_plan_only"
    assert manifest["buckets"] == [64, 256]
    assert manifest["command_buffers_required_disabled"] == (
        "--xla_gpu_enable_command_buffer="
    )
    assert manifest["compiled_callable_invocations"] == 0
    assert manifest["optimizer_step_invocations"] == 0
    assert manifest["optimizer_compile_requested"] is False
    assert manifest["optimizer_compile_calls_planned"] == 0
    assert manifest["executable_export_used"] is False
    assert manifest["graph_api_used"] is False
    assert manifest["source_attestation"] == {
        "status": "cpu_plan_only_self_hash",
        "prewarm_source_sha256": prewarm._file_sha256(_TOOL),
        "launcher_lock_fd_claim_present": False,
    }
    assert plan["jax_imported"] is False
    assert plan["gpu_accessed"] is False
    assert "cannot populate the ROCm executable cache" in plan["note"]


def _fake_verified_source_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, dict[str, str], dict[str, object]]:
    original_repo = tmp_path / "repo"
    original_repo.mkdir(mode=0o700)
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    snapshot = run_dir / "source-head"
    snapshot.mkdir(mode=0o700)
    pycache = run_dir / "python-cache-empty"
    pycache.mkdir(mode=0o700)
    archive = run_dir / "source-head.tar"
    archive.write_bytes(b"exact private HEAD archive")
    archive.chmod(0o600)
    site_packages = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)

    records = {
        "rocm/start_qwen35.sh": {
            "path": "rocm/start_qwen35.sh",
            "git_oid": "1" * 40,
            "sha256": "a" * 64,
        },
        "rocm/prewarm_qwen35_buckets.py": {
            "path": "rocm/prewarm_qwen35_buckets.py",
            "git_oid": "2" * 40,
            "sha256": "b" * 64,
        },
        "rocm/verified_source_bootstrap.py": {
            "path": "rocm/verified_source_bootstrap.py",
            "git_oid": "3" * 40,
            "sha256": "c" * 64,
        },
    }
    manifest: dict[str, object] = {
        "status": "passed",
        "files": list(records.values()),
        "git_head": "d" * 40,
        "git_tree": "e" * 40,
        "git_object_format": "sha1",
        "original_repo_root": str(original_repo),
        "snapshot_root": str(snapshot),
        "source_manifest_sha256": "f" * 64,
        "file_count": 1_087,
        "total_source_bytes": 36_000_000,
        "venv_site_packages": str(site_packages),
        "threat_model_excludes": ["malicious process running as the same UID"],
    }
    monkeypatch.setattr(
        prewarm,
        "_runtime_isolation_checks",
        lambda: {
            "isolated": True,
            "no_site": True,
            "dont_write_bytecode": True,
            "ignore_environment": True,
            "safe_path": True,
        },
    )
    monkeypatch.setattr(sys, "pycache_prefix", str(pycache))
    environment = {
        "SKYRL_QWEN35_GIT_HEAD": manifest["git_head"],
        "SKYRL_QWEN35_GIT_TREE": manifest["git_tree"],
        "SKYRL_QWEN35_GIT_WORKTREE_CLEAN": "true",
        "SKYRL_QWEN35_LAUNCHER_BLOB_OID": records["rocm/start_qwen35.sh"]["git_oid"],
        "SKYRL_QWEN35_LAUNCHER_SHA256": records["rocm/start_qwen35.sh"]["sha256"],
        "SKYRL_QWEN35_PREWARM_BLOB_OID": records["rocm/prewarm_qwen35_buckets.py"]["git_oid"],
        "SKYRL_QWEN35_PREWARM_SHA256": records["rocm/prewarm_qwen35_buckets.py"]["sha256"],
        "SKYRL_QWEN35_BOOTSTRAP_BLOB_OID": records["rocm/verified_source_bootstrap.py"]["git_oid"],
        "SKYRL_QWEN35_BOOTSTRAP_SHA256": records["rocm/verified_source_bootstrap.py"]["sha256"],
        "SKYRL_QWEN35_SOURCE_ARCHIVE_PATH": str(archive),
        "SKYRL_QWEN35_SOURCE_ARCHIVE_SHA256": prewarm._file_sha256(archive),
        "SKYRL_QWEN35_SOURCE_INTERPRETER": sys.executable,
        "SKYRL_QWEN35_SOURCE_INTERPRETER_FLAGS": "-I,-S,-B,-P",
        "SKYRL_QWEN35_SOURCE_PYCACHE_PREFIX": str(pycache),
        "SKYRL_QWEN35_SOURCE_REPO_ROOT": str(original_repo),
        "SKYRL_QWEN35_SOURCE_SNAPSHOT_ROOT": str(snapshot),
        "SKYRL_QWEN35_SOURCE_VENV_SITE_PACKAGES": str(site_packages),
        "SKYRL_VERIFIED_SOURCE_GIT_HEAD": manifest["git_head"],
        "SKYRL_VERIFIED_SOURCE_GIT_TREE": manifest["git_tree"],
        "SKYRL_VERIFIED_SOURCE_MANIFEST_SHA256": manifest[
            "source_manifest_sha256"
        ],
        "SKYRL_VERIFIED_SOURCE_RUNTIME_POLICY": "true",
        "SKYRL_VERIFIED_SOURCE_SNAPSHOT_ROOT": str(snapshot),
    }
    return snapshot, environment, manifest


def test_source_attestation_binds_full_head_snapshot_and_runtime_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot, environment, manifest = _fake_verified_source_contract(
        monkeypatch, tmp_path
    )
    calls: list[dict[str, object]] = []

    def validator(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return manifest

    result = prewarm._validated_source_attestation(
        launcher_required=True,
        repo_root=snapshot,
        environment=environment,
        snapshot_validator=validator,
    )

    assert len(calls) == 1
    assert calls[0]["require_runtime_policy"] is False
    assert calls[0]["target_module"] == "rocm.prewarm_qwen35_buckets"
    assert result["status"] == "passed"
    assert result["git_head"] == "d" * 40
    assert result["git_tree"] == "e" * 40
    assert result["source_manifest_sha256"] == "f" * 64
    assert result["source_file_count"] == 1_087
    assert result["full_head_tree_validated"] is True
    assert result["site_initialization_blocked_by_preimport_bootstrap"] is True
    assert result["launcher_source_role"] == "claimed_launcher_reference"
    assert result["launcher_environment_names"] == list(
        prewarm._SOURCE_ENVIRONMENT_NAMES
    )
    assert set(result["runtime_sources"]) == {"launcher", "prewarm", "bootstrap"}


def test_direct_operational_source_attestation_is_disabled() -> None:
    with pytest.raises(RuntimeError, match="direct ROCm prewarm is disabled"):
        prewarm._validated_source_attestation(
            launcher_required=False,
            environment={},
        )


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("SKYRL_QWEN35_GIT_HEAD", "0" * 40),
        ("SKYRL_QWEN35_GIT_WORKTREE_CLEAN", "false"),
        ("SKYRL_QWEN35_LAUNCHER_SHA256", "0" * 64),
        ("SKYRL_QWEN35_PREWARM_BLOB_OID", "0" * 40),
        ("SKYRL_QWEN35_BOOTSTRAP_SHA256", "0" * 64),
        ("SKYRL_VERIFIED_SOURCE_RUNTIME_POLICY", "false"),
    ),
)
def test_source_attestation_rejects_environment_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
    value: str,
) -> None:
    snapshot, environment, manifest = _fake_verified_source_contract(
        monkeypatch, tmp_path
    )
    environment[name] = value

    with pytest.raises(RuntimeError, match="source attestation mismatch"):
        prewarm._validated_source_attestation(
            launcher_required=True,
            repo_root=snapshot,
            environment=environment,
            snapshot_validator=lambda **_kwargs: manifest,
        )


def test_source_attestation_rejects_missing_claim_before_snapshot_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot, environment, _manifest = _fake_verified_source_contract(
        monkeypatch, tmp_path
    )
    environment.pop("SKYRL_QWEN35_GIT_TREE")

    with pytest.raises(RuntimeError, match="environment is incomplete"):
        prewarm._validated_source_attestation(
            launcher_required=True,
            repo_root=snapshot,
            environment=environment,
            snapshot_validator=lambda **_kwargs: pytest.fail(
                "validator must not run with incomplete claims"
            ),
        )


def test_source_attestation_rejects_hardlinked_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot, environment, manifest = _fake_verified_source_contract(
        monkeypatch, tmp_path
    )
    os.link(environment["SKYRL_QWEN35_SOURCE_ARCHIVE_PATH"], tmp_path / "archive-link")

    with pytest.raises(RuntimeError, match="singly linked"):
        prewarm._validated_source_attestation(
            launcher_required=True,
            repo_root=snapshot,
            environment=environment,
            snapshot_validator=lambda **_kwargs: manifest,
        )


def test_source_attestation_rejects_runtime_policy_regression(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot, environment, manifest = _fake_verified_source_contract(
        monkeypatch, tmp_path
    )
    monkeypatch.setattr(
        prewarm,
        "_runtime_isolation_checks",
        lambda: {
            "isolated": True,
            "no_site": False,
            "dont_write_bytecode": True,
            "ignore_environment": True,
            "safe_path": True,
        },
    )

    with pytest.raises(RuntimeError, match="source attestation mismatch"):
        prewarm._validated_source_attestation(
            launcher_required=True,
            repo_root=snapshot,
            environment=environment,
            snapshot_validator=lambda **_kwargs: manifest,
        )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "comma-separated"),
        ("64,", "comma-separated"),
        ("64,64", "unique and strictly increasing"),
        ("256,64", "unique and strictly increasing"),
        ("63", "not a canonical"),
        ("16", "must be in"),
        ("3072", "must be in"),
    ],
)
def test_bucket_list_fails_closed(value: str, message: str) -> None:
    result = _run("--buckets", value)

    assert result.returncode == 2
    assert result.stdout == ""
    assert message in result.stderr


def test_canonical_representative_buckets_are_preserved() -> None:
    args = prewarm._parse_args(
        [
            "--buckets",
            "32,48,96,384,768,1536,2048",
            "--attention-backend",
            "pallas",
        ]
    )

    assert args.buckets == (32, 48, 96, 384, 768, 1536, 2048)


def test_optimizer_compile_requires_literal_cli_flag_and_stays_plan_only() -> None:
    result = _run("--compile-optimizer")

    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert records[0]["optimizer_compile_requested"] is True
    assert records[0]["optimizer_compile_calls_planned"] == 1
    assert records[1]["optimizer_compile_planned"] is True
    assert records[1]["jax_imported"] is False

    malformed = _run("--compile-optimizer=1")
    assert malformed.returncode == 2
    assert malformed.stdout == ""
    assert "ignored explicit argument" in malformed.stderr


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--execute-rocm",), "requires both"),
        (("--allow-gpu",), "requires both"),
        (
            ("--execute-rocm", "--allow-gpu"),
            "requires --model-path",
        ),
        (
            (
                "--execute-rocm",
                "--allow-gpu",
                "--model-path",
                "/tmp/model",
            ),
            "requires --output",
        ),
        (
            ("--buckets", "64,512"),
            "buckets >=512 require --attention-backend pallas",
        ),
        (
            (
                "--execute-rocm",
                "--allow-gpu",
                "--model-path",
                "/tmp/model",
                "--output",
                "relative.jsonl",
            ),
            "--output must be absolute",
        ),
    ],
)
def test_rocm_execution_requires_explicit_safe_contract(
    arguments: tuple[str, ...], message: str
) -> None:
    result = _run(*arguments)

    assert result.returncode == 2
    assert result.stdout == ""
    assert message in result.stderr


def test_operational_rocm_cli_requires_verified_launcher_lock(tmp_path: Path) -> None:
    result = _run(
        "--execute-rocm",
        "--allow-gpu",
        "--model-path",
        str(tmp_path / "model"),
        "--output",
        str(tmp_path / "audit.jsonl"),
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "requires the verified launcher lock descriptor" in result.stderr


def _exact_environment(monkeypatch: pytest.MonkeyPatch, cache: Path) -> None:
    values = {
        "JAX_PLATFORMS": "rocm",
        "ROCR_VISIBLE_DEVICES": "0",
        "SKYRL_ROCM_PALLAS_ATTENTION": "0",
        "JAX_COMPILATION_CACHE_DIR": str(cache),
        "JAX_ENABLE_COMPILATION_CACHE": "true",
        "JAX_ENABLE_PGLE": "false",
        "JAX_COMPILATION_CACHE_EXPECT_PGLE": "false",
        "JAX_RAISE_PERSISTENT_CACHE_ERRORS": "true",
        "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": (
            "xla_gpu_per_fusion_autotune_cache_dir"
        ),
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
        "JAX_COMPILATION_CACHE_MAX_SIZE": "17179869184",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_FLAGS": "--xla_gpu_enable_command_buffer=",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_cache_and_graph_environment_must_match_launcher_exactly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "private-cache"
    cache.mkdir()
    _exact_environment(monkeypatch, cache)
    monkeypatch.setattr(prewarm, "prepare_cache", lambda maximum: cache)

    assert prewarm._validate_environment("xla", "eager") == cache

    monkeypatch.setenv("JAX_RAISE_PERSISTENT_CACHE_ERRORS", "false")
    with pytest.raises(RuntimeError, match="JAX_RAISE_PERSISTENT_CACHE_ERRORS"):
        prewarm._validate_environment("xla", "eager")
    monkeypatch.setenv("JAX_RAISE_PERSISTENT_CACHE_ERRORS", "true")

    for name in ("JAX_ENABLE_PGLE", "JAX_COMPILATION_CACHE_EXPECT_PGLE"):
        monkeypatch.setenv(name, "true")
        with pytest.raises(RuntimeError, match=name):
            prewarm._validate_environment("xla", "eager")
        monkeypatch.delenv(name)
        with pytest.raises(RuntimeError, match=name):
            prewarm._validate_environment("xla", "eager")
        monkeypatch.setenv(name, "false")

    monkeypatch.setenv(
        "XLA_FLAGS",
        "--xla_gpu_enable_command_buffer= --xla_gpu_kernel_cache_file=/tmp/unsafe",
    )
    with pytest.raises(RuntimeError, match="only the exact empty command-buffer"):
        prewarm._validate_environment("xla", "eager")


def test_model_path_must_equal_exact_local_huggingface_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshots" / prewarm._MODEL_REVISION
    snapshot.mkdir(parents=True)
    calls: list[tuple[str, dict[str, object]]] = []

    def snapshot_download(repo: str, **kwargs: object) -> str:
        calls.append((repo, kwargs))
        return str(snapshot)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)

    assert prewarm._validate_stack_and_model(snapshot) == snapshot.resolve()
    assert calls == [
        (
            "Qwen/Qwen3.5-4B",
            {
                "revision": prewarm._MODEL_REVISION,
                "allow_patterns": (
                    "*.safetensors",
                    "*.json",
                    "*.txt",
                    "*.jinja",
                ),
                "local_files_only": True,
            },
        )
    ]

    impostor = tmp_path / "impostor" / prewarm._MODEL_REVISION
    impostor.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="does not equal huggingface_hub"):
        prewarm._validate_stack_and_model(impostor)


def _drm_card(root: Path, name: str, vendor: str, device: str) -> None:
    device_dir = root / name / "device"
    device_dir.mkdir(parents=True)
    (device_dir / "vendor").write_text(vendor, encoding="utf-8")
    (device_dir / "device").write_text(device, encoding="utf-8")


def test_sysfs_hardware_binding_requires_sole_amd_744c(tmp_path: Path) -> None:
    drm = tmp_path / "drm"
    _drm_card(drm, "card1", "0x1002\n", "0x744c\n")
    _drm_card(drm, "card2", "0x8086\n", "0x9bc5\n")

    result = prewarm._validate_sysfs_amd_gpu(drm)

    assert result["sysfs_amd_gpu_count"] == 1
    assert result["sysfs_amd_gpu"]["vendor_id"] == "0x1002"
    assert result["sysfs_amd_gpu"]["device_id"] == "0x744c"

    (drm / "card1" / "device" / "device").write_text("0x73bf\n")
    with pytest.raises(RuntimeError, match="exactly one AMD DRM GPU"):
        prewarm._validate_sysfs_amd_gpu(drm)

    (drm / "card1" / "device" / "device").write_text("0x744c\n")
    _drm_card(drm, "card3", "0x1002\n", "0x744c\n")
    with pytest.raises(RuntimeError, match="exactly one AMD DRM GPU"):
        prewarm._validate_sysfs_amd_gpu(drm)


@pytest.mark.parametrize("kind", ["gfx1100", "AMD Radeon RX 7900 XTX"])
def test_visible_jax_device_accepts_exact_target_identifiers(kind: str) -> None:
    device = SimpleNamespace(platform="gpu", device_kind=kind)
    jax = SimpleNamespace(devices=lambda: [device])

    result = prewarm._validate_visible_jax_device(jax)

    assert result["jax_visible_device_count"] == 1
    assert result["jax_visible_device_kind"] == kind
    assert result["jax_visible_device_kind_validation"] == (
        "matched_gfx1100_or_rx7900xtx"
    )


def test_visible_jax_device_fails_closed_on_count_platform_or_kind() -> None:
    with pytest.raises(RuntimeError, match="exactly one visible"):
        prewarm._validate_visible_jax_device(SimpleNamespace(devices=lambda: []))
    with pytest.raises(RuntimeError, match="expected JAX GPU"):
        prewarm._validate_visible_jax_device(
            SimpleNamespace(
                devices=lambda: [SimpleNamespace(platform="cpu", device_kind="cpu")]
            )
        )
    with pytest.raises(RuntimeError, match="does not identify"):
        prewarm._validate_visible_jax_device(
            SimpleNamespace(
                devices=lambda: [
                    SimpleNamespace(platform="gpu", device_kind="Radeon VII")
                ]
            )
        )


def test_cache_is_revalidated_after_compilation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "private-cache"
    cache.mkdir()
    observed_maxima: list[int] = []

    def validate(maximum: int) -> Path:
        observed_maxima.append(maximum)
        return cache

    monkeypatch.setattr(prewarm, "prepare_cache", validate)
    prewarm._revalidate_cache_after_compile(cache)

    assert observed_maxima == [4 * 1024**3]

    monkeypatch.setattr(prewarm, "prepare_cache", lambda _maximum: tmp_path / "other")
    with pytest.raises(RuntimeError, match="namespace changed"):
        prewarm._revalidate_cache_after_compile(cache)


class _FakeMonitoring:
    def __init__(self) -> None:
        self.listeners: list[object] = []
        self.duration_listeners: list[object] = []

    def register_event_listener(self, listener: object) -> None:
        self.listeners.append(listener)

    def unregister_event_listener(self, listener: object) -> None:
        self.listeners.remove(listener)

    def register_event_duration_secs_listener(self, listener: object) -> None:
        self.duration_listeners.append(listener)

    def unregister_event_duration_listener(self, listener: object) -> None:
        self.duration_listeners.remove(listener)

    def emit(self, event: str) -> None:
        for listener in tuple(self.listeners):
            listener(event)

    def emit_duration(
        self, event: str, duration_secs: object, **metadata: str | int
    ) -> None:
        for listener in tuple(self.duration_listeners):
            listener(event, duration_secs, **metadata)


def test_per_bucket_cache_evidence_uses_public_events_and_directory_delta(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    monitoring = _FakeMonitoring()
    jax = SimpleNamespace(monitoring=monitoring)
    monkeypatch.setattr(prewarm, "prepare_cache", lambda _maximum: cache)
    postflights: list[str] = []

    def compile_bucket() -> tuple[object, float, float]:
        monitoring.emit("/jax/compilation_cache/compile_requests_use_cache")
        monitoring.emit("/jax/compilation_cache/cache_misses")
        (cache / "jit_model-deadbeef-cache").write_bytes(b"compiled")
        return object(), 0.1, 0.2

    def postflight() -> dict[str, object]:
        postflights.append("clean")
        return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}

    output = StringIO()
    _compiled, _lower, _compile, evidence = prewarm._run_bucket_compile_attempt(
        bucket=64,
        cache_path=cache,
        jax=jax,
        compile_fn=compile_bucket,
        boot_validator=postflight,
        output=output,
    )

    assert postflights == ["clean"]
    assert evidence["classification"] == (
        "strict_public_monitoring_miss_with_single_cache_add"
    )
    assert evidence["public_monitoring_events"] == {
        "compile_requests_use_cache": 1,
        "cache_hits": 0,
        "cache_misses": 1,
    }
    assert evidence["top_level_executable_cache"]["added_entries"] == [
        "jit_model-deadbeef-cache"
    ]
    assert "do not prove which key" in evidence["evidence_limit"]
    assert [
        json.loads(line)["record_type"] for line in output.getvalue().splitlines()
    ] == ["bucket_postflight", "bucket_cache_evidence"]


@pytest.mark.parametrize(
    ("mutation", "classification"),
    [
        ("two_added", "ambiguous_miss_with_multiple_added_cache_entries"),
        ("changed_only", "ambiguous_miss_with_changed_cache_entry"),
    ],
)
def test_strict_miss_rejects_multiple_adds_or_changed_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
    classification: str,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    existing = cache / "jit_existing-cache"
    if mutation == "changed_only":
        existing.write_bytes(b"before")
    monitoring = _FakeMonitoring()
    monkeypatch.setattr(prewarm, "prepare_cache", lambda _maximum: cache)

    def compile_bucket() -> tuple[object, float, float]:
        monitoring.emit("/jax/compilation_cache/compile_requests_use_cache")
        monitoring.emit("/jax/compilation_cache/cache_misses")
        if mutation == "two_added":
            (cache / "jit_first-cache").write_bytes(b"first")
            (cache / "jit_second-cache").write_bytes(b"second")
        else:
            existing.write_bytes(b"changed-and-longer")
        return object(), 0.1, 0.2

    output = StringIO()
    with pytest.raises(RuntimeError, match=classification):
        prewarm._run_bucket_compile_attempt(
            bucket=64,
            cache_path=cache,
            jax=SimpleNamespace(monitoring=monitoring),
            compile_fn=compile_bucket,
            boot_validator=lambda: {
                "amdgpu_boot_clean": True,
                "fatal_amdgpu_events": [],
            },
            output=output,
        )

    evidence = json.loads(output.getvalue().splitlines()[-1])["evidence"]
    assert evidence["classification"] == classification


def test_strict_public_cache_hit_is_promotable_without_directory_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "jit_model-hit-cache").write_bytes(b"cached")
    monitoring = _FakeMonitoring()
    jax = SimpleNamespace(monitoring=monitoring)
    monkeypatch.setattr(prewarm, "prepare_cache", lambda _maximum: cache)

    def compile_bucket() -> tuple[object, float, float]:
        monitoring.emit("/jax/compilation_cache/compile_requests_use_cache")
        monitoring.emit("/jax/compilation_cache/cache_hits")
        monitoring.emit_duration("/jax/compilation_cache/compile_time_saved_sec", 2.5)
        monitoring.emit_duration(
            "/jax/compilation_cache/cache_retrieval_time_sec", 0.01
        )
        return object(), 0.1, 0.01

    result = prewarm._run_bucket_compile_attempt(
        bucket=64,
        cache_path=cache,
        jax=jax,
        compile_fn=compile_bucket,
        boot_validator=lambda: {
            "amdgpu_boot_clean": True,
            "fatal_amdgpu_events": [],
        },
        output=StringIO(),
    )

    evidence = result[-1]
    assert evidence["classification"] == "strict_public_monitoring_hit"
    assert evidence["public_monitoring_duration_events"] == {
        "compile_time_saved_sec": [2.5],
        "cache_retrieval_time_sec": [0.01],
    }


@pytest.mark.parametrize("mutation", ["added", "changed"])
def test_public_hit_with_top_level_cache_mutation_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    existing = cache / "jit_model-hit-cache"
    existing.write_bytes(b"cached")
    monitoring = _FakeMonitoring()
    monkeypatch.setattr(prewarm, "prepare_cache", lambda _maximum: cache)

    def compile_bucket() -> tuple[object, float, float]:
        monitoring.emit("/jax/compilation_cache/compile_requests_use_cache")
        monitoring.emit("/jax/compilation_cache/cache_hits")
        monitoring.emit_duration("/jax/compilation_cache/compile_time_saved_sec", 2.5)
        monitoring.emit_duration(
            "/jax/compilation_cache/cache_retrieval_time_sec", 0.01
        )
        if mutation == "added":
            (cache / "jit_unexpected-added-cache").write_bytes(b"unexpected")
        else:
            existing.write_bytes(b"unexpected-change")
        return object(), 0.1, 0.01

    output = StringIO()
    with pytest.raises(RuntimeError, match="ambiguous_hit_with_top_level"):
        prewarm._run_bucket_compile_attempt(
            bucket=64,
            cache_path=cache,
            jax=SimpleNamespace(monitoring=monitoring),
            compile_fn=compile_bucket,
            boot_validator=lambda: {
                "amdgpu_boot_clean": True,
                "fatal_amdgpu_events": [],
            },
            output=output,
        )

    evidence = json.loads(output.getvalue().splitlines()[-1])["evidence"]
    assert evidence["classification"] == ("ambiguous_hit_with_top_level_cache_mutation")


@pytest.mark.parametrize(
    ("duration", "metadata", "classification"),
    [
        (float("nan"), {}, "malformed_public_monitoring_evidence"),
        (-0.1, {}, "malformed_public_monitoring_evidence"),
        (
            1.0,
            {"module": "untrusted-attribution"},
            "malformed_public_monitoring_evidence",
        ),
    ],
)
def test_malformed_public_duration_evidence_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    duration: float,
    metadata: dict[str, str],
    classification: str,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    monitoring = _FakeMonitoring()
    monkeypatch.setattr(prewarm, "prepare_cache", lambda _maximum: cache)

    def compile_bucket() -> tuple[object, float, float]:
        monitoring.emit("/jax/compilation_cache/compile_requests_use_cache")
        monitoring.emit("/jax/compilation_cache/cache_hits")
        monitoring.emit_duration(
            "/jax/compilation_cache/compile_time_saved_sec",
            duration,
            **metadata,
        )
        monitoring.emit_duration(
            "/jax/compilation_cache/cache_retrieval_time_sec", 0.01
        )
        return object(), 0.1, 0.01

    output = StringIO()
    with pytest.raises(RuntimeError, match="evidence is not promotable"):
        prewarm._run_bucket_compile_attempt(
            bucket=64,
            cache_path=cache,
            jax=SimpleNamespace(monitoring=monitoring),
            compile_fn=compile_bucket,
            boot_validator=lambda: {
                "amdgpu_boot_clean": True,
                "fatal_amdgpu_events": [],
            },
            output=output,
        )

    evidence = json.loads(output.getvalue().splitlines()[-1])["evidence"]
    assert evidence["classification"] == classification
    assert evidence["public_monitoring_schema_issues"]
    assert monitoring.listeners == []
    assert monitoring.duration_listeners == []


def test_duplicate_hit_duration_is_ambiguous_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    monitoring = _FakeMonitoring()
    monkeypatch.setattr(prewarm, "prepare_cache", lambda _maximum: cache)

    def compile_bucket() -> tuple[object, float, float]:
        monitoring.emit("/jax/compilation_cache/compile_requests_use_cache")
        monitoring.emit("/jax/compilation_cache/cache_hits")
        monitoring.emit_duration("/jax/compilation_cache/compile_time_saved_sec", 2.0)
        monitoring.emit_duration("/jax/compilation_cache/compile_time_saved_sec", 2.0)
        monitoring.emit_duration(
            "/jax/compilation_cache/cache_retrieval_time_sec", 0.01
        )
        return object(), 0.1, 0.01

    output = StringIO()
    with pytest.raises(RuntimeError, match="hit_without_exact"):
        prewarm._run_bucket_compile_attempt(
            bucket=64,
            cache_path=cache,
            jax=SimpleNamespace(monitoring=monitoring),
            compile_fn=compile_bucket,
            boot_validator=lambda: {
                "amdgpu_boot_clean": True,
                "fatal_amdgpu_events": [],
            },
            output=output,
        )


@pytest.mark.parametrize(
    ("events", "add_entry", "classification"),
    [
        ((), False, "no_public_hit_or_miss_event_observed"),
        (
            (
                "/jax/compilation_cache/compile_requests_use_cache",
                "/jax/compilation_cache/cache_misses",
            ),
            False,
            "miss_event_without_top_level_cache_change",
        ),
        (
            (
                "/jax/compilation_cache/compile_requests_use_cache",
                "/jax/compilation_cache/compile_requests_use_cache",
                "/jax/compilation_cache/cache_misses",
            ),
            True,
            "non_strict_miss_events_with_cache_change",
        ),
        (
            ("/jax/compilation_cache/compile_requests_use_cache",),
            False,
            "mixed_or_incomplete_public_monitoring_events",
        ),
    ],
)
def test_non_strict_cache_evidence_hard_fails_promotion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    events: tuple[str, ...],
    add_entry: bool,
    classification: str,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    monitoring = _FakeMonitoring()
    jax = SimpleNamespace(monitoring=monitoring)
    monkeypatch.setattr(prewarm, "prepare_cache", lambda _maximum: cache)

    def compile_bucket() -> tuple[object, float, float]:
        for event in events:
            monitoring.emit(event)
        if add_entry:
            (cache / "jit_model-unproven-cache").write_bytes(b"candidate")
        return object(), 0.1, 0.2

    output = StringIO()
    with pytest.raises(RuntimeError, match="evidence is not promotable"):
        prewarm._run_bucket_compile_attempt(
            bucket=256,
            cache_path=cache,
            jax=jax,
            compile_fn=compile_bucket,
            boot_validator=lambda: {
                "amdgpu_boot_clean": True,
                "fatal_amdgpu_events": [],
            },
            output=output,
        )

    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [record["record_type"] for record in records] == [
        "bucket_postflight",
        "bucket_cache_evidence",
    ]
    assert records[-1]["status"] == "rejected"
    assert records[-1]["evidence"]["classification"] == classification
    assert not any(record.get("record_type") == "bucket_compiled" for record in records)


def test_compile_exception_still_runs_cache_and_journal_postflight_without_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    monitoring = _FakeMonitoring()
    jax = SimpleNamespace(monitoring=monitoring)
    order: list[str] = []

    def validate(_maximum: int) -> Path:
        order.append("cache_revalidated")
        return cache

    monkeypatch.setattr(prewarm, "prepare_cache", validate)

    def compile_bucket() -> tuple[object, float, float]:
        order.append("compile_attempt")
        raise RuntimeError("compiler exploded")

    def postflight() -> dict[str, object]:
        order.append("journal_postflight")
        return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}

    output = StringIO()
    with pytest.raises(RuntimeError, match="failed after clean postflight"):
        prewarm._run_bucket_compile_attempt(
            bucket=256,
            cache_path=cache,
            jax=jax,
            compile_fn=compile_bucket,
            boot_validator=postflight,
            output=output,
        )

    assert order == ["compile_attempt", "cache_revalidated", "journal_postflight"]
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [record["record_type"] for record in records] == ["bucket_postflight"]
    assert records[0]["compile_succeeded"] is False
    assert records[0]["cache_revalidated"] is True
    assert not any(
        record.get("record_type") == "bucket_compiled"
        and record.get("status") == "passed"
        for record in records
    )


def test_backend_ready_artifact_includes_exact_hardware_preflight() -> None:
    output = StringIO()
    hardware = {
        "amdgpu_boot_clean": True,
        "fatal_amdgpu_events": [],
        "amd_card_count": 1,
        "connected_amd_connectors": [],
        "kfd_accessible": True,
        "kfd_unowned": True,
    }

    prewarm._emit_backend_ready({"timestamp": "now"}, hardware, output)

    record = json.loads(output.getvalue())
    assert record["record_type"] == "backend_ready"
    assert record["hardware_preflight"] == hardware


def _patch_rocm_execution_for_postflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    postflight_error: Exception | None,
) -> tuple[SimpleNamespace, list[str]]:
    order: list[str] = []
    cache = tmp_path / "cache"
    cache.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    lock_dir = tmp_path / "lock"
    lock_dir.mkdir()
    lock_fd = os.open(lock_dir, os.O_RDONLY | os.O_DIRECTORY)
    args = SimpleNamespace(
        execute_rocm=True,
        compile_optimizer=False,
        buckets=(64,),
        attention_backend="xla",
        model_path=model,
        construction="eager",
        launcher_lock_fd=None,
    )

    def source_attestation(**_kwargs: object) -> dict[str, object]:
        order.append("source_attestation")
        return {"git_head": "a" * 40, "git_worktree_clean": True}

    monkeypatch.setattr(
        prewarm, "_validated_source_attestation", source_attestation
    )

    monkeypatch.setattr(prewarm, "_validate_stack_and_model", lambda _path: model)
    monkeypatch.setattr(
        prewarm,
        "_validate_environment",
        lambda _attention, _construction: cache,
    )
    monkeypatch.setattr(prewarm.compile_probe, "_acquire_global_lock", lambda: lock_fd)
    sysfs_hardware = {
        "sysfs_amd_gpu_count": 1,
        "sysfs_amd_gpu": {"vendor_id": "0x1002", "device_id": "0x744c"},
    }
    monkeypatch.setattr(prewarm, "_validate_sysfs_amd_gpu", lambda: sysfs_hardware)

    boot_calls = 0

    def boot_check() -> dict[str, object]:
        nonlocal boot_calls
        boot_calls += 1
        order.append("boot_preflight" if boot_calls == 1 else "boot_postflight")
        if boot_calls == 2 and postflight_error is not None:
            raise postflight_error
        return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}

    from rocm import amdgpu_safety

    monkeypatch.setattr(amdgpu_safety, "require_clean_amdgpu_boot", boot_check)

    def hardware_preflight() -> dict[str, object]:
        order.append("hardware_preflight")
        return {"kfd_unowned": True}

    monkeypatch.setattr(
        prewarm.compile_probe, "_hardware_preflight", hardware_preflight
    )

    def validate_cache(maximum: int) -> Path:
        assert maximum == 4 * 1024**3
        order.append("cache_revalidated")
        return cache

    monkeypatch.setattr(prewarm, "prepare_cache", validate_cache)

    def compile_only(
        _args: object,
        _model: Path,
        observed_cache: Path,
        hardware: dict[str, object],
        output: object,
        _boot_validator: object,
    ) -> int:
        assert observed_cache == cache
        assert hardware == {
            "amdgpu_boot_clean": True,
            "fatal_amdgpu_events": [],
            **sysfs_hardware,
            "kfd_unowned": True,
        }
        order.append("bucket_compiled")
        output.write('{"record_type":"bucket_compiled"}\n')
        prewarm._revalidate_cache_after_compile(observed_cache)
        return 0

    monkeypatch.setattr(prewarm, "_run_rocm", compile_only)
    return args, order


def test_clean_postflight_follows_compile_and_cache_before_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, order = _patch_rocm_execution_for_postflight(
        monkeypatch, tmp_path, postflight_error=None
    )
    output = StringIO()

    assert prewarm._execute(args, output) == 0

    assert order == [
        "source_attestation",
        "boot_preflight",
        "hardware_preflight",
        "bucket_compiled",
        "cache_revalidated",
        "source_attestation",
        "boot_postflight",
    ]
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [record["record_type"] for record in records] == [
        "manifest",
        "bucket_compiled",
        "hardware_postflight",
        "complete",
    ]
    assert records[0]["source_attestation"] == {
        "git_head": "a" * 40,
        "git_worktree_clean": True,
    }
    assert records[-2]["status"] == "clean"
    assert records[-2]["amdgpu_boot_clean"] is True
    assert records[-2]["source_attestation_revalidated"] is True
    assert records[-2]["inherited_launcher_lock_validated"] is False
    assert records[-1]["amdgpu_postflight_clean"] is True
    assert records[-1]["source_attestation_revalidated"] is True
    assert records[-1]["inherited_launcher_lock_validated"] is False


def test_changed_source_attestation_after_compile_fails_before_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, _order = _patch_rocm_execution_for_postflight(
        monkeypatch, tmp_path, postflight_error=None
    )
    source_calls = 0

    def changing_source(**_kwargs: object) -> dict[str, object]:
        nonlocal source_calls
        source_calls += 1
        return {
            "git_head": "a" * 40,
            "git_worktree_clean": True,
            "prewarm_source_sha256": str(source_calls),
        }

    monkeypatch.setattr(
        prewarm, "_validated_source_attestation", changing_source
    )
    output = StringIO()

    assert prewarm._execute(args, output) == 1

    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [record["record_type"] for record in records] == [
        "manifest",
        "bucket_compiled",
        "hardware_postflight",
        "error",
    ]
    assert records[-2]["source_attestation_revalidated"] is False
    assert records[-1]["stage"] == "source_postflight"
    assert "changed during operational prewarm" in records[-1]["message"]
    assert not any(record["record_type"] == "complete" for record in records)


def test_later_failure_preserves_successful_inherited_lock_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, _order = _patch_rocm_execution_for_postflight(
        monkeypatch, tmp_path, postflight_error=None
    )
    inherited_lock = tmp_path / "inherited-lock"
    inherited_lock.mkdir()
    inherited_fd = os.open(inherited_lock, os.O_RDONLY | os.O_DIRECTORY)
    args.launcher_lock_fd = inherited_fd
    monkeypatch.setattr(
        prewarm,
        "_validate_inherited_lock",
        lambda descriptor: descriptor,
    )
    monkeypatch.setattr(
        prewarm,
        "_run_rocm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("late compile failure")
        ),
    )
    output = StringIO()

    assert prewarm._execute(args, output) == 1

    records = [json.loads(line) for line in output.getvalue().splitlines()]
    postflight = next(
        record for record in records if record["record_type"] == "hardware_postflight"
    )
    assert postflight["inherited_launcher_lock_validated"] is True
    assert records[-1]["stage"] == "rocm_compile_only"


def test_fatal_postflight_fails_without_complete_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, order = _patch_rocm_execution_for_postflight(
        monkeypatch,
        tmp_path,
        postflight_error=RuntimeError("fatal AMDGPU event after compile"),
    )
    output = StringIO()

    assert prewarm._execute(args, output) == 1

    assert order[-4:] == [
        "bucket_compiled",
        "cache_revalidated",
        "source_attestation",
        "boot_postflight",
    ]
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert "complete" not in {record["record_type"] for record in records}
    assert "hardware_postflight" not in {record["record_type"] for record in records}
    error = records[-1]
    assert error["record_type"] == "error"
    assert error["stage"] == "hardware_postflight"
    assert error["status"] == "failed"
    assert "fatal AMDGPU event after compile" in error["message"]


@pytest.mark.parametrize(
    "failure_point",
    ["source_attestation", "snapshot", "jax_import", "backend_setup", "signature"],
)
def test_tool_wide_postflight_covers_all_gpu_capable_failure_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_point: str,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    args = SimpleNamespace(
        execute_rocm=True,
        compile_optimizer=False,
        buckets=(64,),
        attention_backend="xla",
        model_path=model,
        construction="eager",
        launcher_lock_fd=None,
    )
    boot_calls: list[str] = []

    def boot_check() -> dict[str, object]:
        boot_calls.append("journal_check")
        return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}

    monkeypatch.setattr(prewarm, "_require_clean_amdgpu_boot", boot_check)
    monkeypatch.setattr(
        prewarm,
        "_validated_source_attestation",
        lambda **_kwargs: {"git_head": "a" * 40, "git_worktree_clean": True},
    )
    if failure_point == "source_attestation":
        monkeypatch.setattr(
            prewarm,
            "_validated_source_attestation",
            lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("source_attestation fault")
            ),
        )
        monkeypatch.setattr(
            prewarm,
            "_file_sha256",
            lambda _path: pytest.fail("source failure fallback must not re-read source"),
        )
    elif failure_point == "snapshot":
        monkeypatch.setattr(
            prewarm,
            "_validate_stack_and_model",
            lambda _path: (_ for _ in ()).throw(RuntimeError("snapshot fault")),
        )
    else:
        monkeypatch.setattr(prewarm, "_validate_stack_and_model", lambda _path: model)
        lock_dir = tmp_path / "lock"
        lock_dir.mkdir()
        lock_fd = os.open(lock_dir, os.O_RDONLY | os.O_DIRECTORY)
        monkeypatch.setattr(
            prewarm.compile_probe, "_acquire_global_lock", lambda: lock_fd
        )
        monkeypatch.setattr(
            prewarm,
            "_validate_sysfs_amd_gpu",
            lambda: {"sysfs_amd_gpu_count": 1},
        )
        monkeypatch.setattr(
            prewarm.compile_probe,
            "_hardware_preflight",
            lambda: {"kfd_unowned": True},
        )
        monkeypatch.setattr(
            prewarm,
            "_validate_environment",
            lambda _attention, _construction: cache,
        )

        def fail_run(*_args: object, **_kwargs: object) -> int:
            raise RuntimeError(f"{failure_point} fault")

        monkeypatch.setattr(prewarm, "_run_rocm", fail_run)

    output = StringIO()
    assert prewarm._execute(args, output) == 1

    assert len(boot_calls) == (
        0
        if failure_point == "source_attestation"
        else 1
        if failure_point == "snapshot"
        else 2
    )
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [record["record_type"] for record in records] == (
        ["manifest", "error"]
        if failure_point == "source_attestation"
        else ["manifest", "hardware_postflight", "error"]
    )
    if failure_point != "source_attestation":
        assert records[-2]["operation_succeeded"] is False
        assert records[-2]["amdgpu_boot_clean"] is True
    else:
        assert records[0]["source_attestation"]["prewarm_source_sha256"] is None
        assert (
            records[0]["source_attestation"]["source_hash_retried_after_failure"]
            is False
        )
    assert records[-1]["stage"] == (
        "source_attestation"
        if failure_point == "source_attestation"
        else "static_validation"
        if failure_point == "snapshot"
        else "rocm_compile_only"
    )
    assert f"{failure_point} fault" in records[-1]["message"]
    assert not any(record["record_type"] == "complete" for record in records)


def test_inherited_lock_descriptor_must_name_private_global_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    lock_dir = tmp_path / f"skyrl-qwen35-rocm-{os.getuid()}"
    lock_dir.mkdir(mode=0o700)
    lock_fd = os.open(lock_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert prewarm._validate_inherited_lock(lock_fd) == lock_fd
    finally:
        os.close(lock_fd)

    wrong_dir = tmp_path / "wrong"
    wrong_dir.mkdir(mode=0o700)
    wrong_fd = os.open(wrong_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(RuntimeError, match="not the private global launch lock"):
            prewarm._validate_inherited_lock(wrong_fd)
    finally:
        os.close(wrong_fd)


def test_execution_audit_output_is_private_and_exclusive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "nested" / "prewarm.jsonl"
    output.parent.mkdir(mode=0o700)
    fake_args = SimpleNamespace(output=output)

    def execute(_args: object, stream: object) -> int:
        stream.write('{"record_type":"test"}\n')
        return 0

    monkeypatch.setattr(prewarm, "_parse_args", lambda _argv: fake_args)
    monkeypatch.setattr(prewarm, "_execute", execute)

    assert prewarm.main([]) == 0
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert json.loads(output.read_text(encoding="utf-8")) == {"record_type": "test"}

    assert prewarm.main([]) == 2


def test_rocm_output_requires_absolute_preexisting_private_real_parent(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="absolute"):
        prewarm._open_private_output(Path("relative/prewarm.jsonl"))
    with pytest.raises(RuntimeError, match="already exist"):
        prewarm._open_private_output(tmp_path / "missing" / "prewarm.jsonl")

    public_parent = tmp_path / "public"
    public_parent.mkdir(mode=0o755)
    public_parent.chmod(0o755)
    with pytest.raises(RuntimeError, match="mode-0700"):
        prewarm._open_private_output(public_parent / "prewarm.jsonl")

    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    symlink_parent = tmp_path / "linked-private"
    symlink_parent.symlink_to(private_parent, target_is_directory=True)
    with pytest.raises(RuntimeError, match="must not contain a symlink"):
        prewarm._open_private_output(symlink_parent / "prewarm.jsonl")

    descriptor = prewarm._open_private_output(private_parent / "prewarm.jsonl")
    os.close(descriptor)
    artifact = private_parent / "prewarm.jsonl"
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        prewarm._open_private_output(artifact)


class _FakeCompiled:
    def __call__(self, *_args: object) -> None:
        raise AssertionError("compiled executable was invoked")


class _FakeLowered:
    def __init__(self) -> None:
        self.compile_calls = 0

    def compile(self) -> _FakeCompiled:
        self.compile_calls += 1
        return _FakeCompiled()


class _FakeModelPass:
    def __init__(self, lowered: _FakeLowered) -> None:
        self.lowered = lowered
        self.lower_calls: list[tuple[object, ...]] = []

    def lower(self, *signature: object) -> _FakeLowered:
        self.lower_calls.append(signature)
        return self.lowered

    def __call__(self, *_args: object) -> None:
        raise AssertionError("model pass was invoked")


class _FakeJax:
    @staticmethod
    @contextmanager
    def set_mesh(mesh: object):
        yield mesh


def test_compile_step_only_lowers_and_compiles_without_replay() -> None:
    lowered = _FakeLowered()
    model_pass = _FakeModelPass(lowered)
    backend = SimpleNamespace(
        mesh="mesh",
        accumulated_grads="grads",
        lora_params="lora",
        non_lora_params="base",
    )

    compiled, lower_seconds, compile_seconds = prewarm._lower_and_compile_bucket(
        _FakeJax(), model_pass, backend, ("inputs", "loss-config")
    )

    assert isinstance(compiled, _FakeCompiled)
    assert model_pass.lower_calls == [
        ("grads", "lora", "base", "inputs", "loss-config")
    ]
    assert lowered.compile_calls == 1
    assert lower_seconds >= 0
    assert compile_seconds >= 0


def test_optimizer_compile_only_uses_exact_backend_signature_without_update() -> None:
    lowered = _FakeLowered()
    optimizer_pass = _FakeModelPass(lowered)
    optimizer = object()
    backend = SimpleNamespace(
        mesh="mesh",
        accumulated_grads="grads",
        lora_params="lora",
    )

    compiled, lower_seconds, compile_seconds = prewarm._lower_and_compile_optimizer(
        _FakeJax(), optimizer_pass, backend, optimizer, "adapter-index"
    )

    assert isinstance(compiled, _FakeCompiled)
    assert optimizer_pass.lower_calls == [("grads", "lora", optimizer, "adapter-index")]
    assert lowered.compile_calls == 1
    assert lower_seconds >= 0
    assert compile_seconds >= 0


def test_optimizer_compile_attempt_has_separate_cache_and_postflight_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    monitoring = _FakeMonitoring()
    monkeypatch.setattr(prewarm, "prepare_cache", lambda _maximum: cache)

    def compile_optimizer() -> tuple[object, float, float]:
        monitoring.emit("/jax/compilation_cache/compile_requests_use_cache")
        monitoring.emit("/jax/compilation_cache/cache_misses")
        (cache / "jit_optimizer-deadbeef-cache").write_bytes(b"compiled")
        return _FakeCompiled(), 0.2, 0.4

    output = StringIO()
    result = prewarm._run_optimizer_compile_attempt(
        cache_path=cache,
        jax=SimpleNamespace(monitoring=monitoring),
        compile_fn=compile_optimizer,
        boot_validator=lambda: {
            "amdgpu_boot_clean": True,
            "fatal_amdgpu_events": [],
        },
        output=output,
    )

    assert isinstance(result[0], _FakeCompiled)
    records = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [record["record_type"] for record in records] == [
        "optimizer_postflight",
        "optimizer_cache_evidence",
    ]
    assert {record["compile_target"] for record in records} == {
        "sequence_independent_compute_grads_and_update"
    }
    assert all(record["optimizer_step_invocations"] == 0 for record in records)


def test_optimizer_compile_failure_cleans_monitoring_and_runs_postflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    monitoring = _FakeMonitoring()
    monkeypatch.setattr(prewarm, "prepare_cache", lambda _maximum: cache)
    order: list[str] = []

    def fail_compile() -> tuple[object, float, float]:
        order.append("compile")
        raise RuntimeError("optimizer compiler fault")

    def postflight() -> dict[str, object]:
        order.append("postflight")
        return {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}

    output = StringIO()
    with pytest.raises(RuntimeError, match="optimizer compiler fault"):
        prewarm._run_optimizer_compile_attempt(
            cache_path=cache,
            jax=SimpleNamespace(monitoring=monitoring),
            compile_fn=fail_compile,
            boot_validator=postflight,
            output=output,
        )

    assert order == ["compile", "postflight"]
    assert monitoring.listeners == []
    assert monitoring.duration_listeners == []
    record = json.loads(output.getvalue())
    assert record["record_type"] == "optimizer_postflight"
    assert record["compile_succeeded"] is False
    assert record["optimizer_step_invocations"] == 0


def test_installed_jax_accepts_exact_bucket_signature_on_cpu() -> None:
    environment = {**os.environ, "JAX_PLATFORMS": "cpu"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from types import SimpleNamespace
import jax
import jax.numpy as jnp
from rocm.prewarm_qwen35_buckets import _shape_signature

mesh = jax.make_mesh((1, 1, 1), ("fsdp", "ep", "tp"))
signature = _shape_signature(jax, jnp, SimpleNamespace(mesh=mesh), 256)
assert len(signature) == 9
assert [tuple(value.shape) for value in signature[:8]] == [
    (1, 256), (1, 256), (1,), (1, 256),
    (1, 256), (1,), (1, 256), (1, 256),
]
assert str(signature[0].dtype) == "int32"
assert str(signature[4].dtype) == "float32"
assert tuple(signature[8].clip_low_threshold.shape) == (1,)
assert tuple(signature[8].clip_high_threshold.shape) == (1,)
""",
        ],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_installed_jax_public_monitoring_listener_api_on_cpu() -> None:
    environment = {**os.environ, "JAX_PLATFORMS": "cpu"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from jax import monitoring

seen = []
durations = []
def listener(event, **metadata):
    seen.append((event, metadata))
def duration_listener(event, duration_secs, **metadata):
    durations.append((event, duration_secs, metadata))

monitoring.register_event_listener(listener)
monitoring.register_event_duration_secs_listener(duration_listener)
try:
    monitoring.record_event("/skyrl/prewarm/test", bucket=64)
    monitoring.record_event_duration_secs("/skyrl/prewarm/duration", 1.25)
finally:
    monitoring.unregister_event_duration_listener(duration_listener)
    monitoring.unregister_event_listener(listener)
assert seen == [("/skyrl/prewarm/test", {"bucket": 64})]
assert durations == [("/skyrl/prewarm/duration", 1.25, {})]
""",
        ],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_tool_has_no_top_level_jax_import_export_or_executable_call() -> None:
    source = _TOOL.read_text(encoding="utf-8")
    module = ast.parse(source)
    top_level_imports = [
        node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported_roots = {
        alias.name.partition(".")[0]
        for node in top_level_imports
        for alias in node.names
    }

    assert "jax" not in imported_roots
    assert "flax" not in imported_roots
    assert "serialize_executable" not in source
    assert "jax.export" not in source
    assert source.count("compiled = lowered.compile()") == 1
    assert "compiled(" not in source
    assert "model_pass(" not in source
    assert "optimizer_pass(" not in source
    assert ".optim_step(" not in source
    assert "optimizer.update(" not in source
    assert "hipGraph" not in source
    assert "cuda.graph" not in source
    forbidden_direct_calls = {
        "compiled",
        "model_pass",
        "optimizer_pass",
        "_compute_grads_and_update",
    }
    assert not any(
        isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id in forbidden_direct_calls
            or isinstance(node.func, ast.Attribute)
            and node.func.attr in forbidden_direct_calls
        )
        for node in ast.walk(module)
    )
    run_section = source[source.index("def _run_rocm(") : source.index("def _execute(")]
    execute_section = source[source.index("def _execute(") : source.index("def main(")]
    attempt_section = source[
        source.index("def _run_compile_attempt(") : source.index(
            "def _validate_inherited_lock("
        )
    ]
    assert "_run_bucket_compile_attempt(" in run_section
    assert "_run_optimizer_compile_attempt(" in run_section
    assert run_section.index("for bucket in args.buckets:") < run_section.index(
        "if args.compile_optimizer:"
    )
    assert "_revalidate_cache_after_compile(cache_path)" in attempt_section
    assert execute_section.index("_run_rocm(") < execute_section.index(
        "postflight = _require_clean_amdgpu_boot()"
    )
    assert execute_section.index("postflight = _require_clean_amdgpu_boot()") < (
        execute_section.index('"record_type": "complete"')
    )
    assert '"hardware_preflight": hardware' in source
    assert execute_section.index("_validate_sysfs_amd_gpu()") < execute_section.index(
        "_validate_environment("
    )
    assert run_section.index("_validate_visible_jax_device(jax)") < run_section.index(
        "JaxBackendImpl("
    )


def test_launcher_integration_is_default_off_and_after_cache_graph_policy() -> None:
    source = _LAUNCHER.read_text(encoding="utf-8")

    opt_in = 'prewarm_buckets="${SKYRL_QWEN35_PREWARM_BUCKETS:-}"'
    optimizer_opt_in = 'prewarm_optimizer="${SKYRL_QWEN35_PREWARM_OPTIMIZER-0}"'
    prewarm_only_opt_in = 'prewarm_only="${SKYRL_QWEN35_PREWARM_ONLY-0}"'
    invocation = "--module rocm.prewarm_qwen35_buckets"
    final_journal_gate = "if run_amdgpu_safety >/dev/null; then"
    api_exec = 'exec "$uv_executable" run --active --no-sync -m skyrl.tinker.api'
    assert opt_in in source
    assert optimizer_opt_in in source
    assert prewarm_only_opt_in in source
    assert "0|1) ;;" in source
    assert "prewarm_optimizer_args=(--compile-optimizer)" in source
    assert '"${prewarm_optimizer_args[@]}"' in source
    assert "export JAX_ENABLE_PGLE=false" in source
    assert "export JAX_COMPILATION_CACHE_EXPECT_PGLE=false" in source
    assert source.count("export JAX_ENABLE_PGLE=") == 1
    assert source.count("export JAX_COMPILATION_CACHE_EXPECT_PGLE=") == 1
    source_gate = source.index('if [[ -n "$prewarm_buckets" ]]')
    assert source.index("export JAX_ENABLE_PGLE=false") < source_gate
    initial_journal_gate = source.index("if ! run_amdgpu_safety >/dev/null; then")
    assert source_gate < initial_journal_gate
    assert "/usr/bin/git" in source[source_gate : source.index("run_root=")]
    assert "/usr/bin/sha256sum" in source[source_gate : source.index("run_root=")]
    assert "GIT_CONFIG_NOSYSTEM=1" in source
    assert "GIT_CONFIG_GLOBAL=/dev/null" in source
    assert "GIT_NO_REPLACE_OBJECTS=1" in source
    assert "GIT_OPTIONAL_LOCKS=0" in source
    assert "HOME=/nonexistent" in source
    assert "XDG_CONFIG_HOME=/nonexistent" in source
    assert "core.fsmonitor=false" in source
    assert "core.untrackedCache=false" in source
    assert "--ignore-submodules=none" in source
    assert 'export SKYRL_QWEN35_GIT_HEAD="$source_git_head"' in source
    assert 'export SKYRL_QWEN35_GIT_TREE="$source_git_tree"' in source
    assert "export SKYRL_QWEN35_GIT_WORKTREE_CLEAN=true" in source
    assert 'export SKYRL_QWEN35_LAUNCHER_SHA256="$launcher_source_sha256"' in source
    assert 'export SKYRL_QWEN35_PREWARM_SHA256="$prewarm_source_sha256"' in source
    assert 'export SKYRL_QWEN35_BOOTSTRAP_SHA256="$bootstrap_source_sha256"' in source
    assert 'source_blob_oid rocm/start_qwen35.sh 100755' in source
    assert 'source_blob_oid rocm/prewarm_qwen35_buckets.py 100644' in source
    assert 'source_blob_oid rocm/verified_source_bootstrap.py 100644' in source
    assert '"${source_git[@]}" archive' in source
    assert '--no-same-permissions' in source
    assert 'source_snapshot="$run_dir/source-head"' in source
    assert '/usr/bin/python3.12' in source
    for flag in ("-I", "-S", "-B", "-P"):
        assert f"    {flag}\n" in source
    assert '"$source_snapshot/rocm/verified_source_bootstrap.py"' in source
    assert 'run_verified_source_module rocm.profile_rocm' in source
    assert 'run_verified_source_module rocm.qwen35_prewarm_handoff' in source
    assert "source .venv/bin/activate" not in source
    assert 'runpy.run_path("/dev/stdin"' not in source
    assert "sys.path.insert(0, sys.argv.pop(1))" in source
    source_claim_unset = source.index("unset SKYRL_QWEN35_GIT_HEAD")
    for claim_name in prewarm._SOURCE_ENVIRONMENT_NAMES:
        assert f"unset {claim_name}" in source
    final_prewarmer_gate = source.index(
        "if ((prewarm_status != 0))",
        source.rindex(final_journal_gate),
    )
    assert final_prewarmer_gate < source_claim_unset
    assert source_claim_unset < source.index(api_exec)
    assert 'if [[ -n "$prewarm_buckets" ]]' in source
    assert "--execute-rocm" in source
    assert "--allow-gpu" in source
    assert '--launcher-lock-fd "$launch_lock_fd"' in source
    assert '--output "$run_dir/prewarm.jsonl"' in source
    assert source.index(
        'export JAX_COMPILATION_CACHE_DIR="$jax_cache_dir"'
    ) < source.index("export XLA_FLAGS=--xla_gpu_enable_command_buffer=")
    assert source.index(
        "export XLA_FLAGS=--xla_gpu_enable_command_buffer="
    ) < source.index(invocation)
    assert source.index("export JAX_ENABLE_PGLE=false") < source.index(invocation)
    assert source.index("export JAX_COMPILATION_CACHE_EXPECT_PGLE=false") < (
        source.index(invocation)
    )
    assert source.index(invocation) < source.index(api_exec)
    assert source.index(invocation) < source.rindex(final_journal_gate)
    assert source.rindex(final_journal_gate) < source.index(api_exec)
    assert "prewarm_status=$?" in source
    assert "final_journal_status=$?" in source
    assert source.index("prewarm_status=$?") < source.rindex(final_journal_gate)
    prewarm_status_gate = source.index(
        "if ((prewarm_status != 0))", source.rindex(final_journal_gate)
    )
    prewarm_only_gate = source.index(
        'if [[ "$prewarm_only" == "1" ]]', prewarm_status_gate
    )
    assert source.rindex(final_journal_gate) < prewarm_status_gate
    assert prewarm_status_gate < prewarm_only_gate < source.index(api_exec)
    between_gate_and_api = source[
        source.rindex(final_journal_gate) + len(final_journal_gate) : source.index(
            api_exec
        )
    ]
    assert "prewarm_qwen35_buckets.py" not in between_gate_and_api
    assert source.count("export XLA_FLAGS=") == 1
    assert 'amd_device_ids+=("$(<"$card_path/device/device")")' in source
    assert '[[ "${amd_device_ids[0]:-}" != "0x744c" ]]' in source
    assert source.index('[[ "${amd_device_ids[0]:-}" != "0x744c" ]]') < source.index(
        "run_verified_source_module rocm.prepare_jax_cache_dir"
    )
    subprocess.run(["bash", "-n", str(_LAUNCHER)], check=True)


def test_launcher_prewarm_rejects_dirty_source_before_run_directory_or_hardware(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    rocm = repo / "rocm"
    rocm.mkdir(parents=True)
    launcher = rocm / "start_qwen35.sh"
    launcher.write_bytes(_LAUNCHER.read_bytes())
    launcher.chmod(0o700)
    subprocess.run(["/usr/bin/git", "init", "-q", str(repo)], check=True)
    subprocess.run(["/usr/bin/git", "-C", str(repo), "add", "--", str(launcher)], check=True)
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(repo),
            "-c",
            "user.name=Source Test",
            "-c",
            "user.email=source@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        check=True,
    )
    launcher.write_bytes(launcher.read_bytes() + b"\n# dirty after commit\n")
    run_root = tmp_path / "runs"
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("PYTHON") or name in {
            "__PYVENV_LAUNCHER__",
            "VIRTUAL_ENV",
            "VIRTUAL_ENV_PROMPT",
        }:
            environment.pop(name)
    environment.pop("JAX_ENABLE_PGLE", None)
    environment.pop("JAX_COMPILATION_CACHE_EXPECT_PGLE", None)
    environment["SKYRL_QWEN35_PREWARM_BUCKETS"] = "64"
    environment["SKYRL_QWEN35_RUN_ROOT"] = str(run_root)

    result = subprocess.run(
        [str(launcher), "dirty-source"],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 2
    assert "dirty worktree" in result.stderr
    assert "AMDGPU" not in result.stderr
    assert not run_root.exists()


@pytest.mark.parametrize(
    "name",
    ["PYTHONPATH", "PYTHONWARNINGS", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__"],
)
def test_launcher_rejects_python_startup_injection_before_run_directory(
    tmp_path: Path, name: str
) -> None:
    environment = os.environ.copy()
    for inherited in tuple(environment):
        if inherited.startswith("PYTHON") or inherited in {
            "__PYVENV_LAUNCHER__",
            "VIRTUAL_ENV",
            "VIRTUAL_ENV_PROMPT",
        }:
            environment.pop(inherited)
    environment.pop("JAX_ENABLE_PGLE", None)
    environment.pop("JAX_COMPILATION_CACHE_EXPECT_PGLE", None)
    environment[name] = "/tmp/hostile-python-startup"
    environment["SKYRL_QWEN35_PREWARM_BUCKETS"] = "64"
    environment["SKYRL_QWEN35_RUN_ROOT"] = str(tmp_path / "runs")

    result = subprocess.run(
        [str(_LAUNCHER), "injected-python-startup"],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 2
    assert f"operational prewarm: {name}" in result.stderr
    assert not (tmp_path / "runs").exists()


def test_privileged_launcher_shebang_does_not_execute_bash_env(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "bash-env-ran"
    bash_env = tmp_path / "bash-env.sh"
    bash_env.write_text(f"touch {marker}\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["BASH_ENV"] = str(bash_env)

    result = subprocess.run(
        [str(_LAUNCHER), "bash-env-injection"],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 2
    assert "injection variable: BASH_ENV" in result.stderr
    assert not marker.exists()


def test_source_archive_extraction_strips_inherited_tar_options(
    tmp_path: Path,
) -> None:
    payload_root = tmp_path / "payload"
    payload_root.mkdir()
    (payload_root / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    archive = tmp_path / "source.tar"
    subprocess.run(
        [
            "/usr/bin/env",
            "-i",
            "LC_ALL=C",
            "PATH=/usr/bin:/bin",
            "/usr/bin/tar",
            "--create",
            f"--file={archive}",
            f"--directory={payload_root}",
            "tracked.txt",
        ],
        check=True,
    )
    hostile_environment = os.environ.copy()
    hostile_environment["TAR_OPTIONS"] = "--definitely-not-a-valid-tar-option"
    control = tmp_path / "control"
    control.mkdir()
    control_result = subprocess.run(
        [
            "/usr/bin/tar",
            "--extract",
            f"--file={archive}",
            f"--directory={control}",
        ],
        env=hostile_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert control_result.returncode != 0

    destination = tmp_path / "sanitized"
    destination.mkdir()
    result = subprocess.run(
        [
            "/usr/bin/env",
            "-i",
            "LC_ALL=C",
            "PATH=/usr/bin:/bin",
            "/usr/bin/tar",
            "--extract",
            "--no-same-owner",
            "--no-same-permissions",
            f"--file={archive}",
            f"--directory={destination}",
        ],
        env=hostile_environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (destination / "tracked.txt").read_text(encoding="utf-8") == "tracked\n"
    launcher = _LAUNCHER.read_text(encoding="utf-8")
    assert "LC_ALL=C PATH=/usr/bin:/bin /usr/bin/tar" in launcher


@pytest.mark.parametrize(
    ("optimizer_value", "buckets", "message"),
    [
        ("", "64", "must be exactly 0 or 1"),
        ("true", "64", "must be exactly 0 or 1"),
        ("2", "64", "must be exactly 0 or 1"),
        ("1", "", "requires nonempty SKYRL_QWEN35_PREWARM_BUCKETS"),
    ],
)
def test_launcher_optimizer_opt_in_fails_closed_before_hardware_access(
    optimizer_value: str, buckets: str, message: str
) -> None:
    environment = os.environ.copy()
    environment.pop("JAX_ENABLE_PGLE", None)
    environment.pop("JAX_COMPILATION_CACHE_EXPECT_PGLE", None)
    environment["SKYRL_QWEN35_PREWARM_OPTIMIZER"] = optimizer_value
    environment["SKYRL_QWEN35_PREWARM_BUCKETS"] = buckets

    result = subprocess.run(
        [str(_LAUNCHER), "invalid-opt-in-test"],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert "AMDGPU" not in result.stderr


@pytest.mark.parametrize(
    ("prewarm_only", "buckets", "message"),
    (
        ("", "64", "must be exactly 0 or 1"),
        ("true", "64", "must be exactly 0 or 1"),
        ("2", "64", "must be exactly 0 or 1"),
        ("1", "", "requires nonempty SKYRL_QWEN35_PREWARM_BUCKETS"),
    ),
)
def test_launcher_prewarm_only_fails_closed_before_hardware_access(
    prewarm_only: str, buckets: str, message: str
) -> None:
    environment = os.environ.copy()
    environment.pop("JAX_ENABLE_PGLE", None)
    environment.pop("JAX_COMPILATION_CACHE_EXPECT_PGLE", None)
    environment["SKYRL_QWEN35_PREWARM_ONLY"] = prewarm_only
    environment["SKYRL_QWEN35_PREWARM_BUCKETS"] = buckets

    result = subprocess.run(
        [str(_LAUNCHER), "invalid-prewarm-only-test"],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 2
    assert message in result.stderr
    assert "AMDGPU" not in result.stderr


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("JAX_ENABLE_PGLE", "true"),
        ("JAX_ENABLE_PGLE", "1"),
        ("JAX_COMPILATION_CACHE_EXPECT_PGLE", "true"),
        ("JAX_COMPILATION_CACHE_EXPECT_PGLE", "0"),
    ],
)
def test_launcher_rejects_conflicting_pgle_environment_before_hardware_access(
    name: str, value: str
) -> None:
    environment = os.environ.copy()
    environment.pop("JAX_ENABLE_PGLE", None)
    environment.pop("JAX_COMPILATION_CACHE_EXPECT_PGLE", None)
    environment[name] = value
    environment["SKYRL_QWEN35_PREWARM_OPTIMIZER"] = "0"
    environment["SKYRL_QWEN35_PREWARM_BUCKETS"] = ""

    result = subprocess.run(
        [str(_LAUNCHER), "invalid-pgle-test"],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 2
    assert f"{name}={value} conflicts with graph-free startup" in result.stderr
    assert "AMDGPU" not in result.stderr


def test_launcher_status_pattern_postflights_failed_child_and_never_executes_api() -> (
    None
):
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
set -euo pipefail
prewarm_status=0
if bash -c 'exit 7'; then
  prewarm_status=0
else
  prewarm_status=$?
fi
printf 'final-journal\n'
final_journal_status=0
if ((final_journal_status != 0)); then
  exit 2
fi
if ((prewarm_status != 0)); then
  exit "$prewarm_status"
fi
printf 'api-exec\n'
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 7
    assert result.stdout.splitlines() == ["final-journal"]


def test_lock_directory_test_assumption_is_private(tmp_path: Path) -> None:
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
