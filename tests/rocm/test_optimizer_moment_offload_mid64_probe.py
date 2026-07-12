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
_PROBE_PATH = _REPO / "rocm" / "probe_optimizer_moment_offload_mid64.py"
_SPEC = importlib.util.spec_from_file_location(
    "probe_optimizer_moment_offload_mid64_test", _PROBE_PATH
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


def test_default_refuses_without_new_jax_flax_or_offload_import():
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
    assert (
        manifest["probe_source_sha256"]
        == hashlib.sha256(_PROBE_PATH.read_bytes()).hexdigest()
    )
    assert (
        manifest["delegated_smoke_probe_source_sha256"]
        == hashlib.sha256(Path(_PROBE._smoke_probe().__file__).read_bytes()).hexdigest()
    )
    assert (
        manifest["offload_source_sha256"]
        == hashlib.sha256(
            (_REPO / "skyrl" / "tx" / "utils" / "offload.py").read_bytes()
        ).hexdigest()
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
        (("--platform", "rocm", "--allow-gpu"), "requires the explicit --case mid64"),
        (
            ("--platform", "rocm", "--allow-gpu", "--case", "mid64"),
            "requires --output for a private JSONL artifact",
        ),
        (("--allow-gpu",), "only valid with --platform rocm"),
        (("--case", "mid64"), "only valid with --platform rocm"),
        (("--platform", "rocm", "--allow-gpu", "--case", "smoke8"), "invalid choice"),
        (("--cycles", "4"), "unrecognized arguments"),
        (("--leaf-count", "128"), "unrecognized arguments"),
        (("--optimizer-update",), "unrecognized arguments"),
    ],
)
def test_parser_rejects_implicit_gpu_and_scope_changes(arguments, message, capsys):
    with pytest.raises(SystemExit) as raised:
        _PROBE._parse_args(list(arguments))

    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_private_output_is_mode_0600_and_exclusive(tmp_path):
    output = tmp_path / "mid64.jsonl"

    assert _PROBE.main(["--output", str(output)]) == 0
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert [
        record["record_type"]
        for record in map(json.loads, output.read_text().splitlines())
    ] == [
        "manifest",
        "refused",
    ]
    with pytest.raises(SystemExit):
        _PROBE._parse_args(["--output", str(output)])


def test_contract_is_exact_mid64_fixed_three_cycle_rung():
    contract = _PROBE._exact_contract()

    assert contract["case"] == "mid64"
    assert len(contract["selected_paths"]) == 64
    assert contract["selected_paths"][:2] == [
        ["opt_state", 0, "mu", "leaf_00"],
        ["opt_state", 0, "mu", "leaf_01"],
    ]
    assert contract["selected_paths"][31:34] == [
        ["opt_state", 0, "mu", "leaf_31"],
        ["opt_state", 0, "nu", "leaf_00"],
        ["opt_state", 0, "nu", "leaf_01"],
    ]
    assert contract["path_contract"]["mu_leaf_count"] == 32
    assert contract["path_contract"]["nu_leaf_count"] == 32
    assert contract["leaves"]["bytes_each"] == 1024**2
    assert contract["leaves"]["selected_bytes_total"] == 64 * 1024**2
    assert contract["transfer_plan"]["construction_tuple_leaves"] == 65
    assert contract["transfer_plan"]["timed_cycles"] == 3
    assert contract["transfer_plan"]["transactional_manager_batches_total"] == 11
    assert contract["transfer_plan"]["selected_leaf_directional_copies_total"] == 704
    assert contract["transfer_plan"]["device_get_calls"] == 1
    assert contract["per_handle_method_gate"] == {
        "method_seconds_strictly_below": 0.1,
        "post_method_tuple_barrier_seconds_strictly_below": 0.01,
        "warmup_and_final_methods_gated_but_excluded_from_performance_summary": True,
    }
    assert contract["physical_accounting"]["release_is_informational"] is True
    assert contract["outer_profiler_required"] == {
        "max_vram_gib": 2,
        "max_junction_temp_c": 70,
        "max_power_w": 200,
        "min_host_available_gib": 8,
        "timeout_seconds": 90,
        "sensor_grace_seconds": 15,
        "swap_growth_permitted": False,
    }


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
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
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


def test_command_buffer_process_mismatch_fails_before_runtime(monkeypatch):
    environment = _valid_environment(monkeypatch)
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_command_buffer=true")

    with pytest.raises(RuntimeError, match="exactly match"):
        _PROBE._prove_command_buffers_disabled(environment)


def test_error_record_redacts_message(monkeypatch):
    secret = "password=/home/private/mid64-secret"
    monkeypatch.setattr(
        _PROBE,
        "_assert_fresh_accelerator_process",
        lambda: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    output = io.StringIO()

    result = _PROBE._execute(
        SimpleNamespace(platform="rocm", allow_gpu=True, case="mid64"), output
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


def test_construction_is_one_65_leaf_nonaliasing_batch_with_64_distinct_hashes(
    monkeypatch,
):
    monkeypatch.setattr(_PROBE, "_LEAF_SHAPE", (8, 8))
    monkeypatch.setattr(_PROBE, "_LEAF_BYTES", 8 * 8 * 2)
    monkeypatch.setattr(_PROBE, "_SELECTED_BYTES", 64 * 8 * 8 * 2)
    jax = _ConstructionJax()
    counters = _PROBE._zero_counters()

    tree, expected, identity = _PROBE._construct_tree(
        jax, np, ml_dtypes, _FakeNnx, counters
    )

    assert len(jax.calls) == 1
    values, targets, donate, may_alias = jax.calls[0]
    assert len(values) == len(targets) == 65
    assert donate is False
    assert may_alias is False
    assert len(expected) == 64
    assert len({entry["sha256"] for entry in expected}) == 64
    assert all(np.count_nonzero(value) == value.size for value in values[:-1])
    assert tuple(entry["path"] for entry in expected) == _PROBE._SELECTED_PATHS
    assert len(identity["variables"]) == 64
    assert _PROBE._variable_at(tree, _PROBE._COUNT_PATH) is identity["count_variable"]
    assert identity["count_variable"].get_raw_value() is identity["count_raw"]
    assert counters["construction_device_put_attempts"] == 1
    assert counters["construction_device_put_completions"] == 1


def test_allocator_requires_pool_stats_and_95_percent_every_direction():
    with pytest.raises(RuntimeError, match="pool_bytes"):
        _PROBE._allocator_snapshot(
            SimpleNamespace(memory_stats=lambda: {"bytes_in_use": 1})
        )

    selected = _PROBE._SELECTED_BYTES
    enough = int(np.ceil(0.95 * selected))
    device = {"bytes_in_use": 200 * 1024**2, "pool_bytes": 300 * 1024**2}
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
    assert (
        enough
        == _PROBE._exact_contract()["allocator_gate"]["minimum_directional_delta_bytes"]
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
        for index in range(64)
    ]
    return {
        "opt_state": (
            {
                "count": _FakeVariable(
                    np.asarray(_PROBE._COUNT_SENTINEL, dtype=np.int32)
                ),
                "mu": {f"leaf_{index:02d}": leaves[index] for index in range(32)},
                "nu": {f"leaf_{index:02d}": leaves[32 + index] for index in range(32)},
            },
        )
    }


def _install_runtime_mocks(monkeypatch):
    tree = _runtime_tree()
    calls = []
    state_values = [
        200 * 1024**2 if index % 2 == 0 else 130 * 1024**2 for index in range(12)
    ]
    snapshots = iter(
        [{"bytes_in_use": 1, "pool_bytes": 300 * 1024**2}]
        + [
            {"bytes_in_use": value, "pool_bytes": 300 * 1024**2}
            for value in state_values
        ]
    )

    def construct(_jax, _np, _ml, _nnx, counters):
        counters["construction_device_put_attempts"] += 1
        counters["construction_device_put_completions"] += 1
        expected = tuple(
            {"path": path, "sha256": f"{index:064x}"}
            for index, path in enumerate(_PROBE._SELECTED_PATHS)
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
        assert len(expected) == 64
        counters["device_get_attempts"] += 1
        counters["device_get_completions"] += 1
        calls.append("device_get")
        return {"passed": True, "selected_leaf_count": 64}

    monkeypatch.setattr(_PROBE, "_construct_tree", construct)
    monkeypatch.setattr(
        _PROBE, "_validate_tree_phase", lambda *_args, **_kwargs: {"validated": True}
    )
    monkeypatch.setattr(_PROBE, "_allocator_snapshot", lambda _device: next(snapshots))
    monkeypatch.setattr(_PROBE, "_sample_physical_plateau", lambda _card: _plateau(100))
    monkeypatch.setattr(_PROBE, "_device_get_oracle", oracle)
    ticks = iter(index * 0.001 for index in range(200))
    monkeypatch.setattr(_PROBE.time, "perf_counter", lambda: next(ticks))
    return tree, calls, offload


def test_mocked_runtime_executes_exact_warmup_three_timed_cycles_oracle_and_final_cycle(
    monkeypatch,
):
    environment = _valid_environment(monkeypatch)
    _tree, calls, offload = _install_runtime_mocks(monkeypatch)
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
    assert (
        len(
            [
                record
                for record in methods
                if "warmup_excluded" in record["timing_class"]
            ]
        )
        == 2
    )
    assert (
        len(
            [
                record
                for record in methods
                if "oracle_cycle_excluded" in record["timing_class"]
            ]
        )
        == 2
    )
    assert all(record["timing"]["seconds"] < 0.1 for record in methods)
    assert all(record["timing"]["post_barrier_seconds"] < 0.01 for record in methods)
    transitions = [
        record for record in records if record["record_type"] == "allocator_transition"
    ]
    physical = [
        record for record in records if record["record_type"] == "physical_transition"
    ]
    assert len(transitions) == len(physical) == 11
    assert all(record["transition"]["passed"] for record in transitions)
    assert all(
        record["transition"]["gate_effect"] == "informational_only"
        for record in physical
    )
    summary = next(
        record for record in records if record["record_type"] == "performance_summary"
    )
    assert len(summary["directions"]["stage_back"]["each"]) == 3
    assert len(summary["directions"]["reoffload"]["each"]) == 3
    assert summary["throughput_unit"] == "binary_GiB_per_second"
    assert records[-1]["record_type"] == "mid64_passed"


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


def test_timing_summary_reports_each_median_max_and_binary_throughput():
    records = []
    for cycle, seconds in enumerate((0.010, 0.020, 0.030), start=1):
        records.append(
            {
                "cycle": cycle,
                "method": "stage_back",
                "seconds": seconds,
                "post_barrier_seconds": 0.001,
                "binary_gib_per_second": 0.0625 / seconds,
            }
        )
        records.append(
            {
                "cycle": cycle,
                "method": "reoffload",
                "seconds": seconds / 2,
                "post_barrier_seconds": 0.001,
                "binary_gib_per_second": 0.0625 / (seconds / 2),
            }
        )

    summary = _PROBE._timing_summary(records)

    assert summary["stage_back"]["median_seconds"] == pytest.approx(0.020)
    assert summary["stage_back"]["maximum_seconds"] == pytest.approx(0.030)
    assert summary["reoffload"]["median_seconds"] == pytest.approx(0.010)
    assert len(summary["stage_back"]["each"]) == len(summary["reoffload"]["each"]) == 3
    assert summary["stage_back"]["minimum_binary_gib_per_second"] > 0
    assert summary["reoffload"]["maximum_binary_gib_per_second"] > 0


def test_complete_oracle_rejects_one_leaf_hash_mismatch(monkeypatch):
    monkeypatch.setattr(_PROBE, "_LEAF_SHAPE", (1,))
    tree = _runtime_tree()
    expected = tuple(
        {
            "path": path,
            "sha256": _PROBE._host_sha256(np, _PROBE._raw_at(tree, path)),
        }
        for path in _PROBE._SELECTED_PATHS
    )
    corrupted = list(expected)
    corrupted[17] = {**corrupted[17], "sha256": "0" * 64}

    with pytest.raises(RuntimeError, match="complete host bitwise SHA"):
        _PROBE._device_get_oracle(
            _RuntimeJax(),
            np,
            tree,
            tuple(corrupted),
            _PROBE._zero_counters(),
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
spec = importlib.util.spec_from_file_location('isolated_mid64_probe', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    result = module.main([])
new = set(sys.modules) - before
forbidden = sorted(name for name in new if name in {{'jax','jaxlib','flax','skyrl.tx.utils.offload'}} or name.startswith(('jax.','jaxlib.','flax.')))
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
