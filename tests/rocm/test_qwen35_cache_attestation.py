from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_HELPER = _REPO / "rocm" / "qwen35_cache_attestation.py"
_SPEC = importlib.util.spec_from_file_location("qwen35_cache_attestation_test", _HELPER)
assert _SPEC is not None and _SPEC.loader is not None
attestation = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = attestation
_SPEC.loader.exec_module(attestation)


_KEY = "jit_forward_backward_and_accumulate-" + "a" * 64
_BOOT_ID = "54ccf56c-5f4f-4ef7-ac98-c13e0587b5b9"


@pytest.fixture(autouse=True)
def _trusted_handoff_source(monkeypatch: pytest.MonkeyPatch) -> None:
    digest = hashlib.sha256(
        (_REPO / "rocm/qwen35_prewarm_handoff.py").read_bytes()
    ).hexdigest()
    monkeypatch.setattr(
        attestation, "_sibling_source_sha256", lambda _name: digest
    )


def _pair(cache: Path, key: str, payload: bytes, logical_atime_ns: int) -> None:
    (cache / f"{key}-cache").write_bytes(payload)
    (cache / f"{key}-atime").write_bytes(logical_atime_ns.to_bytes(8, "little"))


def _trace(*ordered: str) -> attestation.MonitoringTrace:
    events: dict[str, int] = {}
    durations: dict[str, list[float]] = {}
    for event in ordered:
        if event.endswith("_sec"):
            durations.setdefault(event, []).append(2.0 if "saved" in event else 0.1)
        else:
            events[event] = events.get(event, 0) + 1
    return attestation.MonitoringTrace(
        ordered_events=ordered,
        events=events,
        durations={name: tuple(values) for name, values in durations.items()},
        schema_issues=(),
    )


_HIT_TRACE = (
    "/jax/compilation_cache/compile_requests_use_cache",
    "/jax/compilation_cache/cache_hits",
    "/jax/compilation_cache/compile_time_saved_sec",
    "/jax/compilation_cache/cache_retrieval_time_sec",
)
_MISS_TRACE = (
    "/jax/compilation_cache/compile_requests_use_cache",
    "/jax/compilation_cache/cache_misses",
)


def test_exact_hit_attributes_one_cache_key(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    cache.chmod(0o700)
    _pair(cache, _KEY, b"serialized executable", 100)
    before = attestation.snapshot_cache(cache)

    (cache / f"{_KEY}-atime").write_bytes((150).to_bytes(8, "little"))
    after = attestation.snapshot_cache(cache)
    evidence = attestation.compare_cache_transition(
        before,
        after,
        _trace(*_HIT_TRACE),
        operation_wall_start_ns=125,
        operation_wall_end_ns=175,
        expected_key=_KEY,
        require_hit=True,
    )

    assert evidence["classification"] == attestation.PREWARM_SEED_HIT
    assert evidence["target_cache_entry"]["key"] == _KEY
    assert evidence["target_atime_transition"] == {
        "name": f"{_KEY}-atime",
        "before_logical_atime_ns": 100,
        "after_logical_atime_ns": 150,
        "before_sha256": hashlib.sha256((100).to_bytes(8, "little")).hexdigest(),
        "after_sha256": hashlib.sha256((150).to_bytes(8, "little")).hexdigest(),
        "transition": "rewritten",
    }
    assert evidence["snapshots"]["executable_changed"] == []
    assert evidence["snapshots"]["logical_atime_changed"] == [_KEY]


def test_exact_miss_requires_one_new_paired_entry(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    cache.chmod(0o700)
    before = attestation.snapshot_cache(cache)
    _pair(cache, _KEY, b"new executable", 150)
    autotune = cache / "xla_gpu_per_fusion_autotune_cache_dir"
    autotune.mkdir()
    autotune.chmod(0o700)
    (autotune / "fusion-cache").write_bytes(b"compile-side autotune data")
    after = attestation.snapshot_cache(cache)

    evidence = attestation.compare_cache_transition(
        before,
        after,
        _trace(*_MISS_TRACE),
        operation_wall_start_ns=125,
        operation_wall_end_ns=175,
    )

    assert evidence["classification"] == attestation.PREWARM_SEED_MISS
    assert evidence["evidence_limit"] == attestation.MISS_EVIDENCE_LIMIT
    assert evidence["target_atime_transition"]["transition"] == "added"
    assert evidence["snapshots"]["executable_added"] == [_KEY]
    assert (
        evidence["snapshots"]["auxiliary_manifest_before_sha256"]
        != evidence["snapshots"]["auxiliary_manifest_after_sha256"]
    )


@pytest.mark.parametrize("mutation", ["executable", "extra_atime", "missing_atime"])
def test_hit_rejects_every_non_target_mutation(tmp_path: Path, mutation: str) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    cache.chmod(0o700)
    other = "jit_other-" + "b" * 64
    _pair(cache, _KEY, b"target", 100)
    _pair(cache, other, b"other", 90)
    before = attestation.snapshot_cache(cache)
    (cache / f"{_KEY}-atime").write_bytes((150).to_bytes(8, "little"))
    if mutation == "executable":
        (cache / f"{other}-cache").write_bytes(b"changed")
    elif mutation == "extra_atime":
        (cache / f"{other}-atime").write_bytes((151).to_bytes(8, "little"))
    else:
        (cache / f"{other}-atime").unlink()
    if mutation == "missing_atime":
        with pytest.raises(attestation.CacheAttestationError, match="orphaned"):
            attestation.snapshot_cache(cache)
        return
    after = attestation.snapshot_cache(cache)
    with pytest.raises(attestation.CacheAttestationError, match="not an exact hit"):
        attestation.compare_cache_transition(
            before,
            after,
            _trace(*_HIT_TRACE),
            operation_wall_start_ns=125,
            operation_wall_end_ns=175,
            expected_key=_KEY,
            require_hit=True,
        )


def test_hit_requires_strictly_increasing_atime_inside_wall_bracket(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    cache.chmod(0o700)
    _pair(cache, _KEY, b"target", 100)
    before = attestation.snapshot_cache(cache)
    (cache / f"{_KEY}-atime").write_bytes((250).to_bytes(8, "little"))
    after = attestation.snapshot_cache(cache)

    with pytest.raises(attestation.CacheAttestationError, match="not an exact hit"):
        attestation.compare_cache_transition(
            before,
            after,
            _trace(*_HIT_TRACE),
            operation_wall_start_ns=125,
            operation_wall_end_ns=175,
            expected_key=_KEY,
            require_hit=True,
        )


def test_hit_detects_new_empty_auxiliary_directory(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    cache.chmod(0o700)
    _pair(cache, _KEY, b"target", 100)
    before = attestation.snapshot_cache(cache)
    (cache / f"{_KEY}-atime").write_bytes((150).to_bytes(8, "little"))
    auxiliary = cache / "xla_gpu_per_fusion_autotune_cache_dir"
    auxiliary.mkdir()
    auxiliary.chmod(0o700)
    after = attestation.snapshot_cache(cache)

    assert before.auxiliary_manifest_sha256 != after.auxiliary_manifest_sha256
    with pytest.raises(attestation.CacheAttestationError, match="not an exact hit"):
        attestation.compare_cache_transition(
            before,
            after,
            _trace(*_HIT_TRACE),
            operation_wall_start_ns=125,
            operation_wall_end_ns=175,
            expected_key=_KEY,
            require_hit=True,
        )


def test_hit_detects_new_empty_nested_auxiliary_directory(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    cache.chmod(0o700)
    auxiliary = cache / "xla_gpu_per_fusion_autotune_cache_dir"
    auxiliary.mkdir()
    auxiliary.chmod(0o700)
    _pair(cache, _KEY, b"target", 100)
    before = attestation.snapshot_cache(cache)
    (cache / f"{_KEY}-atime").write_bytes((150).to_bytes(8, "little"))
    (auxiliary / "tmp").mkdir()
    (auxiliary / "tmp").chmod(0o700)
    after = attestation.snapshot_cache(cache)

    assert before.auxiliary_manifest_sha256 != after.auxiliary_manifest_sha256
    with pytest.raises(attestation.CacheAttestationError, match="not an exact hit"):
        attestation.compare_cache_transition(
            before,
            after,
            _trace(*_HIT_TRACE),
            operation_wall_start_ns=125,
            operation_wall_end_ns=175,
            expected_key=_KEY,
            require_hit=True,
        )


def test_snapshot_rejects_unreadable_auxiliary_directory(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    cache.chmod(0o700)
    auxiliary = cache / "xla_gpu_per_fusion_autotune_cache_dir"
    auxiliary.mkdir()
    auxiliary.chmod(0o700)
    unreadable = auxiliary / "unreadable"
    unreadable.mkdir()
    unreadable.chmod(0o000)
    try:
        with pytest.raises(attestation.CacheAttestationError, match="untrusted"):
            attestation.snapshot_cache(cache)
    finally:
        unreadable.chmod(0o700)


@pytest.mark.parametrize("size", [0, 7, 9])
def test_snapshot_rejects_malformed_logical_atime(tmp_path: Path, size: int) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    cache.chmod(0o700)
    (cache / f"{_KEY}-cache").write_bytes(b"target")
    (cache / f"{_KEY}-atime").write_bytes(b"x" * size)
    with pytest.raises(attestation.CacheAttestationError, match="logical-atime"):
        attestation.snapshot_cache(cache)


def test_snapshot_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o700)
    cache.chmod(0o700)
    outside = tmp_path / "outside"
    outside.write_bytes(b"target")
    (cache / f"{_KEY}-cache").symlink_to(outside)
    (cache / f"{_KEY}-atime").write_bytes((100).to_bytes(8, "little"))
    with pytest.raises(attestation.CacheAttestationError, match="unexpected"):
        attestation.snapshot_cache(cache)

    (cache / f"{_KEY}-cache").unlink()
    os.link(outside, cache / f"{_KEY}-cache")
    with pytest.raises(attestation.CacheAttestationError, match="singly linked"):
        attestation.snapshot_cache(cache)


class _Monitoring:
    def __init__(self, *, fail_duration_registration: bool = False) -> None:
        self.event_listeners: list[object] = []
        self.duration_listeners: list[object] = []
        self.fail_duration_registration = fail_duration_registration

    def register_event_listener(self, listener: object) -> None:
        self.event_listeners.append(listener)

    def unregister_event_listener(self, listener: object) -> None:
        self.event_listeners.remove(listener)

    def register_event_duration_secs_listener(self, listener: object) -> None:
        if self.fail_duration_registration:
            raise RuntimeError("registration failed")
        self.duration_listeners.append(listener)

    def unregister_event_duration_listener(self, listener: object) -> None:
        self.duration_listeners.remove(listener)

    def event(self, name: str, **metadata: object) -> None:
        for listener in tuple(self.event_listeners):
            listener(name, **metadata)

    def duration(self, name: str, value: object, **metadata: object) -> None:
        for listener in tuple(self.duration_listeners):
            listener(name, value, **metadata)


def test_public_monitoring_capture_preserves_order_and_cleans_up() -> None:
    monitoring = _Monitoring()
    capture = attestation.PublicCacheMonitoringCapture(monitoring)
    with capture:
        monitoring.event(_HIT_TRACE[0])
        monitoring.event(_HIT_TRACE[1])
        monitoring.duration(_HIT_TRACE[2], 2.0)
        monitoring.duration(_HIT_TRACE[3], 0.1)

    trace = capture.trace()
    assert trace.ordered_events == _HIT_TRACE
    assert monitoring.event_listeners == []
    assert monitoring.duration_listeners == []


def test_partial_listener_registration_is_cleaned_up() -> None:
    monitoring = _Monitoring(fail_duration_registration=True)
    capture = attestation.PublicCacheMonitoringCapture(monitoring)
    with pytest.raises(RuntimeError, match="registration failed"):
        capture.__enter__()
    assert monitoring.event_listeners == []
    assert monitoring.duration_listeners == []


@pytest.mark.parametrize(
    "payload",
    [
        b'{"a":1,"a":2}\n',
        b'{"value":NaN}\n',
        b'{"value":Infinity}\n',
        b'{"value":1e9999}\n',
        b'{"value":1}',
        b"[]\n",
        b"\xff\n",
    ],
)
def test_strict_jsonl_rejects_ambiguous_or_malformed_input(payload: bytes) -> None:
    with pytest.raises(attestation.CacheAttestationError):
        attestation.strict_jsonl(payload)


def _jsonl(records: list[dict[str, object]]) -> bytes:
    return b"".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for record in records
    )


def _fused_mlp_route(
    *, enabled: bool, bucket: int = attestation.BUCKET
) -> dict[str, object]:
    if not enabled:
        status = attestation.FUSED_MLP_ROUTE_STATUS_DISABLED_GLOBAL
        rationale = attestation.FUSED_MLP_ROUTE_RATIONALE_DISABLED_GLOBAL
        indices: list[int] = []
    elif bucket != attestation.BUCKET:
        status = attestation.FUSED_MLP_ROUTE_STATUS_DISABLED_BUCKET
        rationale = attestation.FUSED_MLP_ROUTE_RATIONALE_DISABLED_BUCKET
        indices = []
    else:
        status = attestation.FUSED_MLP_ROUTE_STATUS_QUALIFIED
        rationale = attestation.FUSED_MLP_ROUTE_RATIONALE_QUALIFIED
        indices = list(attestation.FUSED_MLP_ROUTE_LAYER_INDICES)
    return {
        "kind": attestation.FUSED_MLP_ROUTE_KIND,
        "enabled": enabled,
        "bucket": bucket,
        "status": status,
        "expected_layer_indices": indices,
        "expected_layer_count": len(indices),
        "qualified_layer_indices": list(indices),
        "qualified_layer_count": len(indices),
        "static_python_routing_rationale": rationale,
    }


def _prewarm_records(cache_path: Path) -> list[dict[str, object]]:
    backend_config = {
        "abstract_model_load": False,
        "enforce_eager": False,
        "qwen35_bf16_down_lora_residual": False,
        "qwen35_bf16_rms_gate_up_lora_swiglu_contiguous": False,
    }
    backend_hash = attestation.canonical_json_sha256(
        backend_config, domain="skyrl-qwen35-resolved-jax-backend-config-v1"
    )
    target_entry = {
        "key": _KEY,
        "name": f"{_KEY}-cache",
        "device": 1,
        "inode": 2,
        "mode": 0o600,
        "uid": os.getuid(),
        "link_count": 1,
        "size_bytes": 12,
        "mtime_ns": 10,
        "ctime_ns": 11,
        "sha256": "d" * 64,
    }
    target_atime = {
        "name": f"{_KEY}-atime",
        "before_logical_atime_ns": 100,
        "after_logical_atime_ns": 150,
        "before_sha256": hashlib.sha256((100).to_bytes(8, "little")).hexdigest(),
        "after_sha256": hashlib.sha256((150).to_bytes(8, "little")).hexdigest(),
        "transition": "rewritten",
    }
    monitoring = {
        "ordered_events": list(_HIT_TRACE),
        "compile_requests_use_cache": 1,
        "cache_hits": 1,
        "cache_misses": 0,
        "compile_time_saved_sec": [2.0],
        "cache_retrieval_time_sec": [0.1],
        "schema_issues": [],
    }
    snapshots = {
        "executable_manifest_before_sha256": "1" * 64,
        "executable_manifest_after_sha256": "1" * 64,
        "logical_atime_manifest_before_sha256": "2" * 64,
        "logical_atime_manifest_after_sha256": "3" * 64,
        "auxiliary_manifest_before_sha256": "4" * 64,
        "auxiliary_manifest_after_sha256": "4" * 64,
        "executable_added": [],
        "executable_removed": [],
        "executable_changed": [],
        "logical_atime_added": [],
        "logical_atime_removed": [],
        "logical_atime_changed": [_KEY],
    }
    evidence = {
        "schema_name": attestation.SCHEMA_NAME,
        "schema_version": attestation.SCHEMA_VERSION,
        "classification": attestation.PREWARM_SEED_HIT,
        "target_cache_entry": target_entry,
        "target_atime_transition": target_atime,
        "monitoring": monitoring,
        "snapshots": snapshots,
        "operation_wall_start_ns": 125,
        "operation_wall_end_ns": 175,
        "evidence_limit": attestation.HIT_EVIDENCE_LIMIT,
        "public_monitoring_events": {
            "compile_requests_use_cache": 1,
            "cache_hits": 1,
            "cache_misses": 0,
        },
        "public_monitoring_duration_events": {
            "compile_time_saved_sec": [2.0],
            "cache_retrieval_time_sec": [0.1],
        },
        "public_monitoring_schema_issues": [],
        "top_level_executable_cache": {
            "entries_before": 1,
            "entries_after": 1,
            "bytes_before": 12,
            "bytes_after": 12,
            "added_entries": [],
            "changed_entries": [],
            "removed_entries": [],
            "post_manifest_sha256": snapshots[
                "executable_manifest_after_sha256"
            ],
        },
    }
    manifest = {
        "record_type": "manifest",
        "artifact_schema_name": attestation.SCHEMA_NAME,
        "artifact_schema_version": attestation.SCHEMA_VERSION,
        "timestamp": "2026-07-13T00:00:00+00:00",
        "mode": "rocm_compile_only",
        "model": attestation.MODEL,
        "model_revision": attestation.MODEL_REVISION,
        "construction": "eager",
        "buckets": [64],
        "batch_size": 1,
        "attention_backend": "pallas",
        "optimizer_compile_requested": False,
        "train_bucket_lower_calls_planned": 1,
        "train_bucket_compile_calls_planned": 1,
        "optimizer_lower_calls_planned": 0,
        "optimizer_compile_calls_planned": 0,
        "command_buffers_required_disabled": attestation.DISABLE_COMMAND_BUFFERS,
        "graph_api_used": False,
        "executable_export_used": False,
        "compiled_callable_invocations": 0,
        "optimizer_step_invocations": 0,
        "source_attestation": {
            "status": "passed",
            "git_head": "a" * 40,
            "git_tree": "b" * 40,
            "full_head_tree_validated": True,
        },
    }
    backend = {
        "record_type": "backend_ready",
        "timestamp": "2026-07-13T00:00:01+00:00",
        "model": attestation.MODEL,
        "model_revision": attestation.MODEL_REVISION,
        "model_path": "/models/pinned",
        "construction": "eager",
        "cache_path": str(cache_path),
        "platform_resolved": "gpu",
        "platform_version": "ROCm 7.2.4",
        "jax_version": "0.10.2",
        "jaxlib_version": "0.10.2",
        "adapter_index": 1,
        "setup_seconds": 1.0,
        "setup_dispatch_caveat": "setup only; no training executable ran",
        "optimizer_compile_requested": False,
        "train_bucket_lower_calls": 0,
        "train_bucket_compile_calls": 0,
        "optimizer_lower_calls": 0,
        "optimizer_compile_calls": 0,
        "hardware_preflight": {},
        "backend_config": backend_config,
        "backend_config_sha256": backend_hash,
        "model_pass_executable_invocations": 0,
        "optimizer_step_invocations": 0,
    }
    postflight = {
        "record_type": "bucket_postflight",
        "timestamp": "2026-07-13T00:00:02+00:00",
        "bucket": 64,
        "compile_target": attestation.COMPILE_TARGET,
        "status": "clean",
        "compile_succeeded": True,
        "cache_revalidated": True,
        "amdgpu_boot_clean": True,
        "fatal_amdgpu_events": [],
        "model_pass_executable_invocations": 0,
        "optimizer_step_invocations": 0,
    }
    cache_record = {
        "record_type": "bucket_cache_evidence",
        "timestamp": "2026-07-13T00:00:03+00:00",
        "bucket": 64,
        "compile_target": attestation.COMPILE_TARGET,
        "status": "accepted",
        "evidence": evidence,
        "model_pass_executable_invocations": 0,
        "optimizer_step_invocations": 0,
    }
    compiled = {
        "record_type": "bucket_compiled",
        "timestamp": "2026-07-13T00:00:04+00:00",
        "bucket": 64,
        "batch_size": 1,
        "attention_backend": "pallas",
        "compile_target": attestation.COMPILE_TARGET,
        "status": "passed",
        "persistent_cache_evidence": evidence,
        "lower_seconds": 1.0,
        "compile_seconds": 2.0,
        "compiled_memory": {},
        "fused_mlp_route_attestation": _fused_mlp_route(enabled=False),
        "train_bucket_lower_calls": 1,
        "train_bucket_compile_calls": 1,
        "optimizer_lower_calls": 0,
        "optimizer_compile_calls": 0,
        "model_pass_executable_invocations": 0,
        "optimizer_step_invocations": 0,
    }
    hardware = {
        "record_type": "hardware_postflight",
        "timestamp": "2026-07-13T00:00:05+00:00",
        "status": "clean",
        "operation_succeeded": True,
        "source_attestation_revalidated": True,
        "inherited_launcher_lock_validated": True,
        "amdgpu_boot_clean": True,
        "fatal_amdgpu_events": [],
        "model_pass_executable_invocations": 0,
        "optimizer_step_invocations": 0,
    }
    complete = {
        "record_type": "complete",
        "artifact_schema_name": attestation.SCHEMA_NAME,
        "artifact_schema_version": attestation.SCHEMA_VERSION,
        "timestamp": "2026-07-13T00:00:06+00:00",
        "status": "passed",
        "buckets": [64],
        "optimizer_compiled": False,
        "train_bucket_lower_calls": 1,
        "train_bucket_compile_calls": 1,
        "optimizer_lower_calls": 0,
        "optimizer_compile_calls": 0,
        "amdgpu_postflight_clean": True,
        "source_attestation_revalidated": True,
        "inherited_launcher_lock_validated": True,
        "cache_revalidated_after_each_compile": True,
        "model_pass_executable_invocations": 0,
        "optimizer_step_invocations": 0,
    }
    return [manifest, backend, postflight, cache_record, compiled, hardware, complete]


def _prewarm_miss_records(cache_path: Path) -> list[dict[str, object]]:
    records = json.loads(json.dumps(_prewarm_records(cache_path)))
    evidence = records[3]["evidence"]
    evidence["classification"] = attestation.PREWARM_SEED_MISS
    evidence["target_atime_transition"].update(
        {
            "before_logical_atime_ns": None,
            "before_sha256": None,
            "transition": "added",
        }
    )
    evidence["monitoring"].update(
        {
            "ordered_events": list(_MISS_TRACE),
            "cache_hits": 0,
            "cache_misses": 1,
            "compile_time_saved_sec": [],
            "cache_retrieval_time_sec": [],
        }
    )
    evidence["snapshots"].update(
        {
            "executable_manifest_before_sha256": "0" * 64,
            "auxiliary_manifest_after_sha256": "5" * 64,
            "executable_added": [_KEY],
            "logical_atime_added": [_KEY],
            "logical_atime_changed": [],
        }
    )
    evidence["evidence_limit"] = attestation.MISS_EVIDENCE_LIMIT
    evidence["public_monitoring_events"].update(
        {"cache_hits": 0, "cache_misses": 1}
    )
    evidence["public_monitoring_duration_events"].update(
        {"compile_time_saved_sec": [], "cache_retrieval_time_sec": []}
    )
    evidence["top_level_executable_cache"].update(
        {
            "entries_before": 0,
            "entries_after": 1,
            "bytes_before": 0,
            "bytes_after": 12,
            "added_entries": [f"{_KEY}-cache"],
        }
    )
    records[4]["persistent_cache_evidence"] = json.loads(json.dumps(evidence))
    return records


def _handoff_records() -> list[dict[str, object]]:
    device = {
        "drm_card": "card1",
        "vendor_id": "0x1002",
        "device_id": "0x744c",
        "pci_bdf": "0000:03:00.0",
        "pci_sysfs_path": "/sys/devices/pci0000:00/0000:03:00.0",
        "drm_sysfs_path": "/sys/devices/pci0000:00/0000:03:00.0/drm/card1",
        "drm_sysfs_dev": "226:1",
        "render_sysfs_path": (
            "/sys/devices/pci0000:00/0000:03:00.0/drm/renderD128"
        ),
        "render_sysfs_dev": "226:128",
        "drm_node": {
            "path": "/dev/dri/card1",
            "rdev": "226:1",
            "sysfs_dev": "226:1",
            "sysfs_target": (
                "/sys/devices/pci0000:00/0000:03:00.0/drm/card1"
            ),
        },
        "kfd_node": {
            "path": "/dev/kfd",
            "rdev": "236:0",
            "sysfs_dev": "236:0",
            "sysfs_target": "/sys/devices/virtual/kfd/kfd",
        },
        "render_node": {
            "path": "/dev/dri/renderD128",
            "rdev": "226:128",
            "sysfs_dev": "226:128",
            "sysfs_target": (
                "/sys/devices/pci0000:00/0000:03:00.0/drm/renderD128"
            ),
        },
    }
    baseline = {
        "boot_id": _BOOT_ID,
        "device_identity": device,
        "vram_used_bytes": 0,
        "gtt_used_bytes": 0,
        "runtime_status": "suspended",
        "kfd_owner_pids": [],
        "render_owner_pids": [],
    }
    baseline_record = {
        "record_type": "prewarm_handoff_baseline",
        "schema_version": 1,
        "timestamp": "2026-07-13T00:00:00+00:00",
        "status": "passed",
        "device": device,
        "baseline": baseline,
        "release_contract": {
            "timeout_seconds": 120.0,
            "poll_interval_seconds": 1.0,
            "required_consecutive_ready_samples": 3,
            "vram_tolerance_bytes": 0,
            "gtt_tolerance_bytes": 0,
            "runtime_status_required": "suspended",
        },
        "script_sha256": hashlib.sha256(
            (_REPO / "rocm/qwen35_prewarm_handoff.py").read_bytes()
        ).hexdigest(),
        "amdgpu_boot_clean": True,
        "fatal_amdgpu_events": [],
        "graph_api_used": False,
        "command_buffer_used": False,
        "accelerator_device_opened": False,
    }
    samples = [
        {
            "record_type": "prewarm_handoff_sample",
            "schema_version": 1,
            "timestamp": f"2026-07-13T00:00:0{index + 1}+00:00",
            "sample_index": index,
            "elapsed_seconds": float(index),
            "snapshot": baseline,
            "checks": {
                "same_boot": True,
                "same_device_identity": True,
                "kfd_unowned": True,
                "render_unowned": True,
                "vram_no_higher_than_exact_baseline": True,
                "gtt_no_higher_than_exact_baseline": True,
                "runtime_suspended": True,
            },
            "ready_streak": index + 1,
            "required_ready_streak": 3,
            "status": "ready_candidate",
            "accelerator_device_opened": False,
        }
        for index in range(3)
    ]
    complete = {
        "record_type": "prewarm_handoff_complete",
        "schema_version": 1,
        "timestamp": "2026-07-13T00:00:04+00:00",
        "status": "passed",
        "elapsed_seconds": 3.0,
        "baseline": baseline,
        "final_snapshot": baseline,
        "checks": {
            "same_boot": True,
            "same_device_identity": True,
            "kfd_unowned": True,
            "render_unowned": True,
            "vram_no_higher_than_exact_baseline": True,
            "gtt_no_higher_than_exact_baseline": True,
            "runtime_suspended": True,
        },
        "sample_count": 3,
        "final_ready_streak": 3,
        "vram_tolerance_bytes": 0,
        "gtt_tolerance_bytes": 0,
        "amdgpu_boot_clean": True,
        "fatal_amdgpu_events": [],
        "graph_api_used": False,
        "command_buffer_used": False,
        "accelerator_device_opened": False,
    }
    return [baseline_record, *samples, complete]


def _write_private(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    path.chmod(0o600)
    return hashlib.sha256(payload).hexdigest()


def test_build_and_revalidate_startup_claim_without_jax(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    cache = tmp_path / "cache"
    cache.mkdir()
    prewarm = run_dir / "prewarm.jsonl"
    handoff = run_dir / "prewarm-handoff.jsonl"
    prewarm_sha = _write_private(prewarm, _jsonl(_prewarm_records(cache)))
    handoff_sha = _write_private(handoff, _jsonl(_handoff_records()))
    boot_id = tmp_path / "boot_id"
    boot_id.write_text(_BOOT_ID + "\n", encoding="ascii")

    claim = attestation.build_startup_cache_claim(
        prewarm_path=prewarm,
        prewarm_sha256=prewarm_sha,
        handoff_path=handoff,
        handoff_sha256=handoff_sha,
        expected_git_head="a" * 40,
        expected_git_tree="b" * 40,
        expected_cache_path=str(cache),
        expected_attention_backend="pallas",
        boot_id_path=boot_id,
    )

    assert claim["status"] == attestation.REQUIREMENT
    assert claim["seed"]["target_cache_entry"]["key"] == _KEY
    assert claim["seed"]["fused_mlp_route_attestation"] == _fused_mlp_route(
        enabled=False
    )
    assert (
        attestation.revalidate_startup_cache_claim(claim, boot_id_path=boot_id) == claim
    )


def test_startup_claim_accepts_seed_miss_auxiliary_mutation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    cache = tmp_path / "cache"
    cache.mkdir()
    prewarm = run_dir / "prewarm.jsonl"
    handoff = run_dir / "prewarm-handoff.jsonl"
    prewarm_sha = _write_private(prewarm, _jsonl(_prewarm_miss_records(cache)))
    handoff_sha = _write_private(handoff, _jsonl(_handoff_records()))
    boot_id = tmp_path / "boot_id"
    boot_id.write_text(_BOOT_ID + "\n", encoding="ascii")

    claim = attestation.build_startup_cache_claim(
        prewarm_path=prewarm,
        prewarm_sha256=prewarm_sha,
        handoff_path=handoff,
        handoff_sha256=handoff_sha,
        expected_git_head="a" * 40,
        expected_git_tree="b" * 40,
        expected_cache_path=str(cache),
        expected_attention_backend="pallas",
        boot_id_path=boot_id,
    )

    assert claim["seed"]["prewarm_seed_kind"] == attestation.PREWARM_SEED_MISS


@pytest.mark.parametrize(
    ("record_index", "field", "value"),
    [
        (0, "artifact_schema_version", 1.0),
        (0, "batch_size", True),
        (0, "untrusted", "extra"),
        (1, "adapter_index", True),
        (2, "bucket", 64.0),
        (3, "bucket", 64.0),
        (4, "bucket", 64.0),
        (5, "untrusted", "extra"),
        (-1, "artifact_schema_name", "wrong"),
        (-1, "artifact_schema_version", 999),
    ],
)
def test_prewarm_rejects_nonexact_outer_records(
    tmp_path: Path, record_index: int, field: str, value: object
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    records = _prewarm_records(cache)
    records[record_index][field] = value

    with pytest.raises(attestation.CacheAttestationError):
        attestation.validate_prewarm_t64_artifact(
            _jsonl(records),
            expected_git_head="a" * 40,
            expected_git_tree="b" * 40,
            expected_cache_path=str(cache),
            expected_attention_backend="pallas",
        )


def test_prewarm_rejects_authoritative_record_reordering(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    records = _prewarm_records(cache)
    records[3], records[4] = records[4], records[3]

    with pytest.raises(
        attestation.CacheAttestationError, match="record structure|out of order"
    ):
        attestation.validate_prewarm_t64_artifact(
            _jsonl(records),
            expected_git_head="a" * 40,
            expected_git_tree="b" * 40,
            expected_cache_path=str(cache),
            expected_attention_backend="pallas",
        )


def test_prewarm_rejects_rehashed_unpromoted_down_fusion(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    records = _prewarm_records(cache)
    backend_config = records[1]["backend_config"]
    assert isinstance(backend_config, dict)
    backend_config["qwen35_bf16_down_lora_residual"] = True
    records[1]["backend_config_sha256"] = attestation.canonical_json_sha256(
        backend_config,
        domain="skyrl-qwen35-resolved-jax-backend-config-v1",
    )

    with pytest.raises(
        attestation.CacheAttestationError, match="unpromoted BF16 down fusion"
    ):
        attestation.validate_prewarm_t64_artifact(
            _jsonl(records),
            expected_git_head="a" * 40,
            expected_git_tree="b" * 40,
            expected_cache_path=str(cache),
            expected_attention_backend="pallas",
        )


@pytest.mark.parametrize("value", [None, 0, "false"])
def test_prewarm_requires_exact_contiguous_fused_mlp_bool(
    tmp_path: Path, value: object
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    records = _prewarm_records(cache)
    backend_config = records[1]["backend_config"]
    assert isinstance(backend_config, dict)
    key = "qwen35_bf16_rms_gate_up_lora_swiglu_contiguous"
    if value is None:
        backend_config.pop(key)
    else:
        backend_config[key] = value
    records[1]["backend_config_sha256"] = attestation.canonical_json_sha256(
        backend_config,
        domain="skyrl-qwen35-resolved-jax-backend-config-v1",
    )

    with pytest.raises(attestation.CacheAttestationError, match="not an exact bool"):
        attestation.validate_prewarm_t64_artifact(
            _jsonl(records),
            expected_git_head="a" * 40,
            expected_git_tree="b" * 40,
            expected_cache_path=str(cache),
            expected_attention_backend="pallas",
        )


def test_prewarm_seed_preserves_enabled_contiguous_fused_mlp_policy(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    records = _prewarm_records(cache)
    backend_config = records[1]["backend_config"]
    assert isinstance(backend_config, dict)
    key = "qwen35_bf16_rms_gate_up_lora_swiglu_contiguous"
    backend_config[key] = True
    records[1]["backend_config_sha256"] = attestation.canonical_json_sha256(
        backend_config,
        domain="skyrl-qwen35-resolved-jax-backend-config-v1",
    )
    records[4]["fused_mlp_route_attestation"] = _fused_mlp_route(enabled=True)

    seed = attestation.validate_prewarm_t64_artifact(
        _jsonl(records),
        expected_git_head="a" * 40,
        expected_git_tree="b" * 40,
        expected_cache_path=str(cache),
        expected_attention_backend="pallas",
    )

    assert seed["backend_config"][key] is True
    assert seed["fused_mlp_route_attestation"] == _fused_mlp_route(enabled=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("qualified_layer_count", 31, "does not match its exact bucket policy"),
        ("status", "claimed", "does not match its exact bucket policy"),
        ("unexpected", False, "schema is invalid"),
    ],
)
def test_prewarm_rejects_nonexact_fused_mlp_route_attestation(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    records = _prewarm_records(cache)
    route = records[4]["fused_mlp_route_attestation"]
    assert isinstance(route, dict)
    route[field] = value

    with pytest.raises(attestation.CacheAttestationError, match=message):
        attestation.validate_prewarm_t64_artifact(
            _jsonl(records),
            expected_git_head="a" * 40,
            expected_git_tree="b" * 40,
            expected_cache_path=str(cache),
            expected_attention_backend="pallas",
        )


def test_fused_mlp_route_validator_requires_non_t64_disabled_state() -> None:
    route = _fused_mlp_route(enabled=True, bucket=256)

    assert attestation._validate_fused_mlp_route_attestation(
        route,
        enabled=True,
        bucket=256,
    ) == route

    route["qualified_layer_indices"] = [0]
    route["qualified_layer_count"] = 1
    with pytest.raises(
        attestation.CacheAttestationError,
        match="does not match its exact bucket policy",
    ):
        attestation._validate_fused_mlp_route_attestation(
            route,
            enabled=True,
            bucket=256,
        )


def test_prewarm_rejects_impossible_declared_bucket_structure(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    records = _prewarm_records(cache)
    records[0]["buckets"] = [64, 128]
    records[0]["train_bucket_lower_calls_planned"] = 2
    records[0]["train_bucket_compile_calls_planned"] = 2
    records[-1]["buckets"] = [64, 128]
    records[-1]["train_bucket_lower_calls"] = 2
    records[-1]["train_bucket_compile_calls"] = 2

    with pytest.raises(attestation.CacheAttestationError, match="record structure"):
        attestation.validate_prewarm_t64_artifact(
            _jsonl(records),
            expected_git_head="a" * 40,
            expected_git_tree="b" * 40,
            expected_cache_path=str(cache),
            expected_attention_backend="pallas",
        )


def test_startup_claim_rejects_incomplete_handoff(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    cache = tmp_path / "cache"
    cache.mkdir()
    prewarm = run_dir / "prewarm.jsonl"
    handoff = run_dir / "prewarm-handoff.jsonl"
    prewarm_sha = _write_private(prewarm, _jsonl(_prewarm_records(cache)))
    handoff_sha = _write_private(handoff, _jsonl(_handoff_records()[:-1]))
    boot_id = tmp_path / "boot_id"
    boot_id.write_text(_BOOT_ID + "\n", encoding="ascii")

    with pytest.raises(attestation.CacheAttestationError, match="samples|terminal"):
        attestation.build_startup_cache_claim(
            prewarm_path=prewarm,
            prewarm_sha256=prewarm_sha,
            handoff_path=handoff,
            handoff_sha256=handoff_sha,
            expected_git_head="a" * 40,
            expected_git_tree="b" * 40,
            expected_cache_path=str(cache),
            expected_attention_backend="pallas",
            boot_id_path=boot_id,
        )


@pytest.mark.parametrize(
    ("record_index", "field", "value"),
    [
        (0, "schema_version", 1.0),
        (1, "schema_version", 1.0),
        (1, "required_ready_streak", 3.0),
        (-1, "schema_version", 1.0),
        (-1, "sample_count", 3.0),
        (-1, "vram_tolerance_bytes", False),
    ],
)
def test_handoff_rejects_noninteger_integer_fields(
    record_index: int, field: str, value: object
) -> None:
    records = _handoff_records()
    records[record_index][field] = value

    with pytest.raises(attestation.CacheAttestationError):
        attestation.validate_completed_handoff_artifact(
            _jsonl(records), expected_boot_id=_BOOT_ID
        )


def test_private_artifact_rejects_mode_hardlink_and_digest_drift(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    artifact = run_dir / "prewarm.jsonl"
    payload = b'{"record_type":"manifest"}\n'
    digest = _write_private(artifact, payload)
    artifact.chmod(0o644)
    with pytest.raises(attestation.CacheAttestationError, match="0600"):
        attestation.stable_private_artifact(artifact, digest, maximum_bytes=1024)
    artifact.chmod(0o600)
    os.link(artifact, run_dir / "alias")
    with pytest.raises(attestation.CacheAttestationError, match="singly linked"):
        attestation.stable_private_artifact(artifact, digest, maximum_bytes=1024)
    (run_dir / "alias").unlink()
    with pytest.raises(attestation.CacheAttestationError, match="SHA-256"):
        attestation.stable_private_artifact(artifact, "0" * 64, maximum_bytes=1024)


def test_attestation_helper_has_no_jax_import_at_module_scope() -> None:
    source = Path(attestation.__file__).read_text(encoding="utf-8")
    prefix = source.split("def _shape_signature", maxsplit=1)[0]
    assert "import jax" not in prefix
    assert "import flax" not in prefix
