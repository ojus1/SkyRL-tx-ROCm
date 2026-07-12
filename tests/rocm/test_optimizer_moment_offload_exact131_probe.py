from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

_REPO = Path(__file__).parents[2]
_PROBE_PATH = _REPO / "rocm" / "probe_optimizer_moment_offload_exact131.py"
_SPEC = importlib.util.spec_from_file_location(
    "probe_optimizer_moment_offload_exact131_test", _PROBE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)
_SOURCE = _PROBE_PATH.read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE)
_CLEAN = {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}


def _records(output: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in output.getvalue().splitlines()]


def _accelerator_modules() -> set[str]:
    return {
        name
        for name in sys.modules
        if name in {"jax", "jaxlib", "flax", "skyrl.tx.utils.offload"}
        or name.startswith(("jax.", "jaxlib.", "flax."))
    }


def test_default_refuses_without_new_jax_flax_numpy_or_offload_import():
    before = _accelerator_modules()
    output = io.StringIO()

    result = _PROBE._execute(
        SimpleNamespace(platform="abstract", allow_gpu=False, case=None), output
    )

    assert result == 0
    assert _accelerator_modules() == before
    manifest, refused = _records(output)
    assert manifest["scope"] == "abstract_refusal"
    assert manifest["case"] is None
    assert manifest["inventory_source_proof"]["passed"] is True
    assert manifest["inventory_source_sha256"] == _PROBE._INVENTORY_SOURCE_SHA256
    assert manifest["contract"]["leaves"]["selected_bytes_total"] == 138_051_584
    assert (
        manifest["probe_source_sha256"]
        == hashlib.sha256(_PROBE_PATH.read_bytes()).hexdigest()
    )
    assert refused["record_type"] == "refused"
    assert refused["jax_imported"] is False
    assert refused["flax_imported"] is False
    assert refused["offload_module_imported"] is False
    assert refused["counters"] == _PROBE._zero_counters()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--platform", "rocm"), "requires the explicit --allow-gpu"),
        (
            ("--platform", "rocm", "--allow-gpu"),
            "requires the explicit --case exact131",
        ),
        (
            ("--platform", "rocm", "--allow-gpu", "--case", "exact131"),
            "requires --output for a private JSONL artifact",
        ),
        (("--allow-gpu",), "only valid with --platform rocm"),
        (("--case", "exact131"), "only valid with --platform rocm"),
        (("--platform", "rocm", "--allow-gpu", "--case", "mid64"), "invalid choice"),
        (("--cycles", "4"), "unrecognized arguments"),
        (("--leaf-count", "64"), "unrecognized arguments"),
        (("--optimizer-update",), "unrecognized arguments"),
    ],
)
def test_parser_rejects_implicit_gpu_and_scope_changes(arguments, message, capsys):
    with pytest.raises(SystemExit) as raised:
        _PROBE._parse_args(list(arguments))

    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_private_output_is_mode_0600_and_exclusive(tmp_path):
    output = tmp_path / "exact131.jsonl"

    assert _PROBE.main(["--output", str(output)]) == 0
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert [
        record["record_type"]
        for record in map(json.loads, output.read_text().splitlines())
    ] == ["manifest", "refused"]
    with pytest.raises(SystemExit):
        _PROBE._parse_args(["--output", str(output)])


def test_literal_shape_inventory_exact_totals_revision_and_source_binding():
    proof = _PROBE._verify_inventory_source()
    contract = _PROBE._exact_contract()

    assert proof == {
        "passed": True,
        "source_path_repo_relative": "rocm/probe_jax_optimizer.py",
        "source_sha256": _PROBE._INVENTORY_SOURCE_SHA256,
        "source_commit": "816183190987a66916ff6155a4254eea852e17aa",
        "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "shape_groups": [
            {"name": name, "shape": list(shape), "count": count}
            for name, shape, count in _PROBE._SHAPE_GROUPS
        ],
        "parameter_leaf_count": 402,
        "parameter_elements": 34_512_896,
        "two_moment_leaf_count": 804,
        "two_bfloat16_moment_bytes": 138_051_584,
    }
    assert len(_PROBE._PARAMETER_INVENTORY) == 402
    assert len(_PROBE._SELECTED_PATHS) == len(_PROBE._SELECTED_SHAPES) == 804
    assert sum(np.prod(shape) for shape in _PROBE._SELECTED_SHAPES) == 69_025_792
    assert _PROBE._SELECTED_BYTES == 138_051_584
    assert _PROBE._SELECTED_BYTES / 1024**2 == 131.65625
    assert contract["provenance"]["inventory_source_sha256"] == proof["source_sha256"]
    assert contract["path_contract"]["mu_leaf_count"] == 402
    assert contract["path_contract"]["nu_leaf_count"] == 402
    assert contract["transfer_plan"]["construction_tuple_leaves"] == 805
    assert contract["transfer_plan"]["timed_cycles"] == 3
    assert contract["transfer_plan"]["transactional_manager_batches_total"] == 11
    assert contract["transfer_plan"]["selected_leaf_directional_copies_total"] == 8844
    assert contract["outer_profiler_required"] == {
        "max_vram_gib": 2,
        "max_junction_temp_c": 70,
        "max_power_w": 200,
        "min_host_available_gib": 8,
        "timeout_seconds": 120,
        "sensor_grace_seconds": 15,
        "swap_growth_permitted": False,
    }


def test_inventory_source_hash_or_literal_drift_fails_closed(monkeypatch, tmp_path):
    original = _PROBE._INVENTORY_SOURCE_PATH.read_text(encoding="utf-8")
    changed = tmp_path / "probe_jax_optimizer.py"
    changed.write_text(
        original.replace("_EXPECTED_LEAVES = 402", "_EXPECTED_LEAVES = 401")
    )
    monkeypatch.setattr(_PROBE, "_INVENTORY_SOURCE_PATH", changed)

    with pytest.raises(RuntimeError, match="SHA-256 differs"):
        _PROBE._verify_inventory_source()

    changed_sha = hashlib.sha256(changed.read_bytes()).hexdigest()
    monkeypatch.setattr(_PROBE, "_INVENTORY_SOURCE_SHA256", changed_sha)
    with pytest.raises(RuntimeError, match="leaf count differs"):
        _PROBE._verify_inventory_source()


_ENV_NAMES = (
    "JAX_PLATFORMS",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "JAX_ROCM_VISIBLE_DEVICES",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_CLIENT_MEM_FRACTION",
    "HSA_OVERRIDE_GFX_VERSION",
    "XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB",
    "JAX_PJRT_CLIENT_CREATE_OPTIONS",
    "JAX_MOCK_GPU_TOPOLOGY",
    "MOCK_NUM_GPU_PROCESSES",
    "TF_FORCE_UNIFIED_MEMORY",
    "XLA_FLAGS",
)


def _clear_environment(monkeypatch):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def _valid_environment(monkeypatch):
    _clear_environment(monkeypatch)
    return _PROBE._configure_rocm_environment()


def test_delegated_environment_proves_bfc_no_unified_and_empty_command_buffers(
    monkeypatch,
):
    environment = _valid_environment(monkeypatch)
    manifest = _PROBE._environment_manifest(environment)
    proof = _PROBE._prove_command_buffers_disabled(environment)

    assert manifest["bfc_growth_allocator"] is True
    assert manifest["unified_memory_disabled"] is True
    assert manifest["raw_xla_flags_emitted"] is False
    assert proof["command_buffer_assignment_count"] == 1
    assert proof["sole_assignment_is_exact_empty"] is True
    assert proof["raw_xla_flags_emitted"] is False


def test_error_record_redacts_message(monkeypatch):
    secret = "password=/home/private/exact131-secret"
    monkeypatch.setattr(
        _PROBE,
        "_assert_fresh_accelerator_process",
        lambda: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    output = io.StringIO()

    result = _PROBE._execute(
        SimpleNamespace(platform="rocm", allow_gpu=True, case="exact131"), output
    )

    assert result == 1
    artifact = output.getvalue()
    error = _records(output)[-1]
    assert secret not in artifact
    assert "/home/private" not in artifact
    assert error["record_type"] == "error"
    assert error["message_redacted"] is True
    assert error["message_sha256"] == hashlib.sha256(secret.encode()).hexdigest()


class _FakeSharding:
    def __init__(self, device, memory_kind="device"):
        self.device = device
        self.memory_kind = memory_kind

    def with_memory_kind(self, memory_kind):
        return _FakeSharding(self.device, memory_kind)

    def __eq__(self, other):
        return isinstance(other, _FakeSharding) and (self.device, self.memory_kind) == (
            other.device,
            other.memory_kind,
        )


class _FakeArray:
    def __init__(self, value, sharding):
        self.value = np.asarray(value)
        self.shape = self.value.shape
        self.dtype = self.value.dtype
        self.nbytes = self.value.nbytes
        self.sharding = sharding
        self.committed = True
        self.is_fully_addressable = True


class _FakeVariable:
    def __init__(self, value):
        self.value = value

    def get_raw_value(self):
        return self.value


class _FakeNnx:
    OptState = _FakeVariable


class _ConstructionJax:
    def __init__(self):
        self.device = SimpleNamespace(platform="gpu")
        self.sharding = SimpleNamespace(
            SingleDeviceSharding=lambda device: _FakeSharding(device)
        )
        self.calls = []

    def devices(self):
        return [self.device]

    def device_put(self, values, targets, *, donate, may_alias):
        self.calls.append((values, targets, donate, may_alias))
        return tuple(
            _FakeArray(value, target)
            for value, target in zip(values, targets, strict=True)
        )

    @staticmethod
    def block_until_ready(values):
        return values


def _install_tiny_inventory(monkeypatch):
    inventory = (
        ("leaf_000", "lora_A", (2, 8)),
        ("leaf_001", "lora_B", (4, 8)),
    )
    paths = tuple(
        ("opt_state", 0, slot, leaf_name)
        for slot in ("mu", "nu")
        for leaf_name, _group, _shape in inventory
    )
    shapes = tuple(shape for _name, _group, shape in inventory) * 2
    monkeypatch.setattr(_PROBE, "_PARAMETER_INVENTORY", inventory)
    monkeypatch.setattr(_PROBE, "_PARAMETER_LEAF_COUNT", 2)
    monkeypatch.setattr(_PROBE, "_PARAMETER_ELEMENTS", 48)
    monkeypatch.setattr(_PROBE, "_LEAF_COUNT", 4)
    monkeypatch.setattr(_PROBE, "_MU_PATHS", paths[:2])
    monkeypatch.setattr(_PROBE, "_NU_PATHS", paths[2:])
    monkeypatch.setattr(_PROBE, "_SELECTED_PATHS", paths)
    monkeypatch.setattr(_PROBE, "_SELECTED_SHAPES", shapes)
    monkeypatch.setattr(_PROBE, "_SELECTED_BYTES", 192)


def test_construction_is_one_805_semantics_batch_with_nonzero_distinct_hashes(
    monkeypatch,
):
    _install_tiny_inventory(monkeypatch)
    jax = _ConstructionJax()
    counters = _PROBE._zero_counters()

    tree, expected, identity = _PROBE._construct_tree(
        jax, np, ml_dtypes, _FakeNnx, counters
    )

    assert len(jax.calls) == 1
    values, targets, donate, may_alias = jax.calls[0]
    assert len(values) == len(targets) == 5
    assert donate is False
    assert may_alias is False
    assert len(expected) == 4
    assert len({entry["sha256"] for entry in expected}) == 4
    assert all(np.count_nonzero(value) == value.size for value in values[:-1])
    assert tuple(entry["path"] for entry in expected) == _PROBE._SELECTED_PATHS
    assert tuple(entry["shape"] for entry in expected) == _PROBE._SELECTED_SHAPES
    assert len(identity["variables"]) == 4
    assert _PROBE._variable_at(tree, _PROBE._COUNT_PATH) is identity["count_variable"]
    assert identity["count_variable"].get_raw_value() is identity["count_raw"]
    assert counters["construction_device_put_attempts"] == 1
    assert counters["construction_device_put_completions"] == 1


def test_all_804_identity_codes_are_nonzero_and_distinct_without_large_shapes():
    digests = []
    for leaf_index in range(804):
        value = _PROBE._host_leaf(np, ml_dtypes, (16,), leaf_index)
        assert np.count_nonzero(value) == value.size
        digests.append(_PROBE._host_sha256(np, value))
    assert len(set(digests)) == 804


def test_allocator_requires_pool_stats_and_95_percent_every_direction():
    with pytest.raises(RuntimeError, match="pool_bytes"):
        _PROBE._allocator_snapshot(
            SimpleNamespace(memory_stats=lambda: {"bytes_in_use": 1})
        )

    selected = _PROBE._SELECTED_BYTES
    enough = int(np.ceil(0.95 * selected))
    device = {"bytes_in_use": 300 * 1024**2, "pool_bytes": 400 * 1024**2}
    host = {
        "bytes_in_use": device["bytes_in_use"] - enough,
        "pool_bytes": device["pool_bytes"],
    }
    assert (
        _PROBE._allocator_transition("release", device, host, "release")["passed"]
        is True
    )
    assert (
        _PROBE._allocator_transition("allocate", host, device, "allocate")["passed"]
        is True
    )
    with pytest.raises(RuntimeError, match="95% gate"):
        _PROBE._allocator_transition(
            "short",
            device,
            {
                "bytes_in_use": device["bytes_in_use"] - enough + 1,
                "pool_bytes": device["pool_bytes"],
            },
            "release",
        )


def _plateau(value=100):
    return {
        "vram": {
            "observed": value is not None,
            "minimum_bytes": value,
            "maximum_bytes": value,
        },
        "gtt": {
            "observed": value is not None,
            "minimum_bytes": value,
            "maximum_bytes": value,
        },
    }


def test_physical_release_remains_separate_and_informational():
    transition = _PROBE._physical_transition(
        "release", _plateau(100), _plateau(100), "release"
    )
    assert transition["physical_release_target_met"] is False
    assert transition["gate_effect"] == "informational_only"
    assert transition["failure_effect"] == "none_bfc_may_retain_physical_pages"


class _CycleHandle:
    def __init__(self, calls):
        self.calls = calls
        self.phase = "offloaded"

    def stage_back(self):
        assert self.phase == "offloaded"
        self.calls.append("stage_back")
        self.phase = "staged_back"
        return self

    def reoffload(self):
        assert self.phase == "staged_back"
        self.calls.append("reoffload")
        self.phase = "complete"
        return _CycleHandle(self.calls)


class _RuntimeJax:
    __version__ = "fake-jax"

    def __init__(self):
        self.device = SimpleNamespace(platform="gpu")
        self.block_calls = 0

    @staticmethod
    def default_backend():
        return "gpu"

    def devices(self):
        return [self.device]

    def block_until_ready(self, values):
        self.block_calls += 1
        return values

    @staticmethod
    def device_get(values):
        return values


def _runtime_tree():
    leaves = [
        _FakeVariable(np.asarray([index + 1], dtype=ml_dtypes.bfloat16))
        for index in range(804)
    ]
    return {
        "opt_state": (
            {
                "count": _FakeVariable(
                    np.asarray(_PROBE._COUNT_SENTINEL, dtype=np.int32)
                ),
                "mu": {f"leaf_{index:03d}": leaves[index] for index in range(402)},
                "nu": {
                    f"leaf_{index:03d}": leaves[402 + index] for index in range(402)
                },
            },
        )
    }


def _install_runtime_mocks(monkeypatch):
    tree = _runtime_tree()
    calls = []
    high = 400 * 1024**2
    low = high - _PROBE._SELECTED_BYTES
    state_values = [high if index % 2 == 0 else low for index in range(12)]
    snapshots = iter(
        [{"bytes_in_use": 1, "pool_bytes": 512 * 1024**2}]
        + [
            {"bytes_in_use": value, "pool_bytes": 512 * 1024**2}
            for value in state_values
        ]
    )

    def construct(_jax, _np, _ml, _nnx, counters):
        counters["construction_device_put_attempts"] += 1
        counters["construction_device_put_completions"] += 1
        expected = tuple(
            {"path": path, "shape": shape, "sha256": f"{index:064x}"}
            for index, (path, shape) in enumerate(
                zip(_PROBE._SELECTED_PATHS, _PROBE._SELECTED_SHAPES, strict=True)
            )
        )
        return tree, expected, {"mock": True}

    def offload(received_tree, *, paths, memory_kind):
        assert received_tree is tree
        assert paths == _PROBE._SELECTED_PATHS
        assert memory_kind == "pinned_host"
        calls.append("initial_offload")
        return _CycleHandle(calls)

    def oracle(_jax, _np, received_tree, expected, counters):
        assert received_tree is tree
        assert len(expected) == 804
        counters["device_get_attempts"] += 1
        counters["device_get_completions"] += 1
        calls.append("device_get")
        return {"passed": True, "selected_leaf_count": 804}

    monkeypatch.setattr(_PROBE, "_construct_tree", construct)
    monkeypatch.setattr(
        _PROBE, "_validate_tree_phase", lambda *_args, **_kwargs: {"validated": True}
    )
    monkeypatch.setattr(_PROBE, "_allocator_snapshot", lambda _device: next(snapshots))
    monkeypatch.setattr(_PROBE, "_sample_physical_plateau", lambda _card: _plateau(100))
    monkeypatch.setattr(_PROBE, "_device_get_oracle", oracle)
    ticks = iter(index * 0.001 for index in range(200))
    monkeypatch.setattr(_PROBE.time, "perf_counter", lambda: next(ticks))
    return calls, offload


def test_mocked_runtime_executes_warmup_three_cycles_full_oracle_and_final_reoffload(
    monkeypatch,
):
    environment = _valid_environment(monkeypatch)
    calls, offload = _install_runtime_mocks(monkeypatch)
    jax = _RuntimeJax()
    dependencies = (
        jax,
        SimpleNamespace(__version__="fake-jaxlib"),
        SimpleNamespace(
            get_backend=lambda: SimpleNamespace(platform_version="ROCm 7.2")
        ),
        np,
        ml_dtypes,
        _FakeNnx,
        offload,
    )
    output = io.StringIO()
    counters = _PROBE._zero_counters()

    result = _PROBE._run_rocm(
        output,
        lambda: dict(_CLEAN),
        counters,
        environment=environment,
        amd_card="card1",
        _dependencies=dependencies,
    )

    assert result == 0
    assert calls == [
        "initial_offload",
        "stage_back",
        "reoffload",
        "stage_back",
        "reoffload",
        "stage_back",
        "reoffload",
        "stage_back",
        "reoffload",
        "stage_back",
        "device_get",
        "reoffload",
    ]
    assert counters == _PROBE._completed_counters()
    assert jax.block_calls == 10
    records = _records(output)
    methods = [
        record for record in records if record["record_type"] == "manager_method"
    ]
    assert len(methods) == 10
    assert (
        len([record for record in methods if record["timing_class"] == "measured"]) == 6
    )
    transitions = [
        record for record in records if record["record_type"] == "allocator_transition"
    ]
    physical = [
        record for record in records if record["record_type"] == "physical_transition"
    ]
    assert len(transitions) == len(physical) == 11
    assert all(record["transition"]["passed"] for record in transitions)
    summary = next(
        record for record in records if record["record_type"] == "performance_summary"
    )
    assert len(summary["directions"]["stage_back"]["each"]) == 3
    assert len(summary["directions"]["reoffload"]["each"]) == 3
    assert summary["directions"]["stage_back"]["median_binary_gib_per_second"] > 0
    assert summary["directions"]["stage_back"]["minimum_binary_gib_per_second"] > 0
    assert records[-1]["record_type"] == "exact131_passed"


def test_method_and_post_barrier_limits_fail_closed(monkeypatch):
    tree = _runtime_tree()
    handle = _CycleHandle([])
    jax = _RuntimeJax()
    counters = _PROBE._zero_counters()
    ticks = iter((0.0, 0.101))
    monkeypatch.setattr(_PROBE.time, "perf_counter", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="100 ms"):
        _PROBE._checked_stage_back(jax, tree, handle, counters, "warmup_stage_back")

    handle = _CycleHandle([])
    counters = _PROBE._zero_counters()
    ticks = iter((0.0, 0.001, 0.002, 0.02))
    monkeypatch.setattr(_PROBE.time, "perf_counter", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="10 ms"):
        _PROBE._checked_stage_back(jax, tree, handle, counters, "warmup_stage_back")


def test_complete_oracle_rejects_one_of_804_leaf_hashes(monkeypatch):
    monkeypatch.setattr(_PROBE, "_SELECTED_SHAPES", ((1,),) * 804)
    tree = _runtime_tree()
    expected = tuple(
        {
            "path": path,
            "shape": (1,),
            "sha256": _PROBE._host_sha256(np, _PROBE._raw_at(tree, path)),
        }
        for path in _PROBE._SELECTED_PATHS
    )
    corrupted = list(expected)
    corrupted[403] = {**corrupted[403], "sha256": "0" * 64}

    with pytest.raises(RuntimeError, match="complete host bitwise SHA"):
        _PROBE._device_get_oracle(
            _RuntimeJax(), np, tree, tuple(corrupted), _PROBE._zero_counters()
        )


def test_ast_proves_no_eager_accelerator_import_or_hidden_work():
    forbidden = {"jax", "jaxlib", "flax", "numpy", "ml_dtypes", "skyrl"}
    module_imports = []
    for node in _TREE.body:
        if isinstance(node, ast.Import):
            module_imports.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_imports.append(node.module.split(".", 1)[0])
    assert forbidden.isdisjoint(module_imports)

    calls = [node for node in ast.walk(_TREE) if isinstance(node, ast.Call)]
    attributes = [
        node.func.attr for node in calls if isinstance(node.func, ast.Attribute)
    ]
    names = [node.func.id for node in calls if isinstance(node.func, ast.Name)]
    assert attributes.count("stage_back") == 1
    assert attributes.count("reoffload") == 1
    assert attributes.count("device_get") == 1
    assert attributes.count("jit") == 0
    assert attributes.count("compile") == 0
    assert "optimizer.update" not in _SOURCE
    assert names.count("offload_optimizer_moments") == 1
    assert names.count("_checked_stage_back") == 3
    assert names.count("_checked_reoffload") == 3
    assert names.count("_device_get_oracle") == 1

    run_start = _SOURCE.index("def _run_rocm")
    run_source = _SOURCE[run_start : _SOURCE.index("def _execute", run_start)]
    assert run_source.index("_verify_inventory_source") < run_source.index("import jax")
    assert run_source.index("_prove_command_buffers_disabled") < run_source.index(
        "import jax"
    )
    assert run_source.index("offload_optimizer_moments(") < run_source.index(
        "warmup_stage_back"
    )
    assert run_source.index("warmup_stage_back") < run_source.index(
        "for cycle in range"
    )
    assert run_source.index("for cycle in range") < run_source.index("final_stage_back")
    assert run_source.index("final_stage_back") < run_source.index(
        "_device_get_oracle("
    )
    assert run_source.index("_device_get_oracle(") < run_source.index("final_reoffload")


def test_default_subprocess_refuses_without_accelerator_imports():
    program = f"""
import contextlib, importlib.util, io, json, sys
path = {str(_PROBE_PATH)!r}
before = set(sys.modules)
spec = importlib.util.spec_from_file_location('isolated_exact131_probe', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    result = module.main([])
new = set(sys.modules) - before
forbidden = sorted(name for name in new if name in {{'jax','jaxlib','flax','numpy','ml_dtypes','skyrl.tx.utils.offload'}} or name.startswith(('jax.','jaxlib.','flax.','numpy.')))
print(json.dumps({{'result': result, 'forbidden': forbidden, 'records': [json.loads(line) for line in captured.getvalue().splitlines()]}}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(_REPO)

    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=_REPO,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["result"] == 0
    assert report["forbidden"] == []
    assert [record["record_type"] for record in report["records"]] == [
        "manifest",
        "refused",
    ]
