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
_PROBE_PATH = _REPO / "rocm" / "probe_optimizer_moment_offload.py"
_SPEC = importlib.util.spec_from_file_location("probe_optimizer_moment_offload_test", _PROBE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)
_SOURCE = _PROBE_PATH.read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE)
_CLEAN = {"amdgpu_boot_clean": True, "fatal_amdgpu_events": []}
_HEADLESS = {
    **_CLEAN,
    "amd_cards": ["card1"],
    "connected_amd_connectors": [],
    "kfd_path": "/dev/kfd",
    "kfd_accessible": True,
    "kfd_unowned": True,
}


def _records(output: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in output.getvalue().splitlines()]


def _accelerator_modules() -> set[str]:
    return {
        name
        for name in sys.modules
        if name in {"jax", "jaxlib", "flax", "skyrl.tx.utils.offload"} or name.startswith(("jax.", "jaxlib.", "flax."))
    }


def test_default_is_refusal_without_new_jax_flax_or_offload_import():
    before = _accelerator_modules()
    output = io.StringIO()

    result = _PROBE._execute(SimpleNamespace(platform="abstract", allow_gpu=False, case=None), output)

    assert result == 0
    assert _accelerator_modules() == before
    manifest, refused = _records(output)
    assert manifest["record_type"] == "manifest"
    assert manifest["scope"] == "abstract_refusal"
    assert manifest["fresh_process_required"] is True
    assert manifest["probe_source_sha256"] == hashlib.sha256(_PROBE_PATH.read_bytes()).hexdigest()
    assert (
        manifest["offload_source_sha256"]
        == hashlib.sha256((_REPO / "skyrl" / "tx" / "utils" / "offload.py").read_bytes()).hexdigest()
    )
    assert refused == {
        "record_type": "refused",
        "timestamp": refused["timestamp"],
        "status": "no_gpu_abstract_manifest_only",
        "reason": (
            "pass --platform rocm --allow-gpu --case smoke8 --output explicitly under "
            "profile_rocm.py in a fresh process"
        ),
        "jax_imported": False,
        "flax_imported": False,
        "offload_module_imported": False,
        "counters": _PROBE._zero_counters(),
    }


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--platform", "rocm"), "requires the explicit --allow-gpu"),
        (("--platform", "rocm", "--allow-gpu"), "requires the explicit --case smoke8"),
        (
            ("--platform", "rocm", "--allow-gpu", "--case", "smoke8"),
            "requires --output for a private JSONL artifact",
        ),
        (("--allow-gpu",), "only valid with --platform rocm"),
        (("--case", "smoke8"), "only valid with --platform rocm"),
        (("--platform", "rocm", "--allow-gpu", "--case", "mid64"), "invalid choice"),
        (("--platform", "rocm", "--allow-gpu", "--case", "exact"), "invalid choice"),
        (("--optimizer-update",), "unrecognized arguments"),
        (("--repeats", "2"), "unrecognized arguments"),
    ],
)
def test_parser_rejects_implicit_gpu_or_scope_broadening(arguments, message, capsys):
    with pytest.raises(SystemExit) as raised:
        _PROBE._parse_args(list(arguments))

    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_private_output_is_exclusive_and_exact_mode(tmp_path):
    output = tmp_path / "smoke.jsonl"

    assert _PROBE.main(["--output", str(output)]) == 0
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert [record["record_type"] for record in map(json.loads, output.read_text().splitlines())] == [
        "manifest",
        "refused",
    ]
    with pytest.raises(SystemExit):
        _PROBE._parse_args(["--output", str(output)])


def test_contract_is_exact_smoke8_and_keeps_accounting_distinct():
    contract = _PROBE._exact_contract()

    assert contract["case"] == "smoke8"
    assert contract["selected_paths"] == [
        ["opt_state", 0, "mu", "weight"],
        ["opt_state", 0, "nu", "weight"],
    ]
    assert contract["unselected_sentinel_path"] == ["opt_state", 0, "count"]
    assert contract["leaves"]["bytes_each"] == 4 * 1024**2
    assert contract["leaves"]["selected_bytes_total"] == 8 * 1024**2
    assert contract["memory_kinds"] == {"source": "device", "offload": "pinned_host"}
    assert contract["transfer_plan"]["transactional_manager_batches_total"] == 5
    assert contract["transfer_plan"]["selected_leaf_directional_copies_total"] == 10
    assert contract["transfer_plan"]["optimizer_updates"] == 0
    assert contract["transfer_plan"]["command_buffer_invocations"] == 0
    assert contract["physical_accounting"]["release_is_informational"] is True
    assert contract["allocator_gate"]["initial_offload_release_required"] is True
    assert contract["outer_profiler_required"] == {
        "max_vram_gib": 2,
        "max_junction_temp_c": 70,
        "max_power_w": 200,
        "min_host_available_gib": 8,
        "swap_growth_permitted": False,
    }
    assert contract["not_implemented"] == [
        "mid64",
        "exact_model_state",
        "optimizer_update",
        "overlap",
        "throughput_sweep",
    ]


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


def _clear_probe_environment(monkeypatch):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_environment_is_bfc_growth_single_device_no_unified_and_redacts_flags(monkeypatch):
    _clear_probe_environment(monkeypatch)
    secret = "/private/token-path"
    monkeypatch.setenv("XLA_FLAGS", f"--xla_dump_to={secret}")

    environment = _PROBE._configure_rocm_environment()
    manifest = _PROBE._environment_manifest(environment)
    proof = _PROBE._prove_command_buffers_disabled(environment)

    assert os.environ["JAX_PLATFORMS"] == "rocm"
    assert os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "bfc"
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert "TF_FORCE_UNIFIED_MEMORY" not in os.environ
    assert manifest["bfc_growth_allocator"] is True
    assert manifest["unified_memory_disabled"] is True
    assert manifest["raw_xla_flags_emitted"] is False
    assert proof["command_buffer_assignment_count"] == 1
    assert proof["sole_assignment_is_exact_empty"] is True
    artifact = json.dumps({"manifest": manifest, "proof": proof})
    assert secret not in artifact
    assert environment["XLA_FLAGS_original"] not in artifact
    assert environment["XLA_FLAGS_effective"] not in artifact


@pytest.mark.parametrize(
    "flags",
    [
        "--xla_gpu_enable_command_buffer=true",
        "--xla_gpu_enable_command_buffer=false",
        "--xla_gpu_enable_command_buffer",
        "--noxla_gpu_enable_command_buffer",
        "--xla_gpu_enable_command_buffer= --xla_gpu_enable_command_buffer=",
        "--xla_gpu_enable_command_buffer= --xla_gpu_enable_command_buffer=true",
    ],
)
def test_conflicting_or_duplicate_command_buffer_flags_fail_closed(monkeypatch, flags):
    _clear_probe_environment(monkeypatch)
    monkeypatch.setenv("XLA_FLAGS", flags)

    with pytest.raises(RuntimeError, match="command-buffer flags conflict"):
        _PROBE._configure_rocm_environment()


def test_command_buffer_proof_checks_process_environment_not_return_only(monkeypatch):
    _clear_probe_environment(monkeypatch)
    environment = _PROBE._configure_rocm_environment()
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_command_buffer=true")

    with pytest.raises(RuntimeError, match="exactly match"):
        _PROBE._prove_command_buffers_disabled(environment)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("XLA_PYTHON_CLIENT_ALLOCATOR", "platform"),
        ("XLA_PYTHON_CLIENT_PREALLOCATE", "true"),
        ("TF_FORCE_UNIFIED_MEMORY", "1"),
        ("JAX_MOCK_GPU_TOPOLOGY", "2x2x1"),
        ("MOCK_NUM_GPU_PROCESSES", "2"),
    ],
)
def test_allocator_topology_or_unified_memory_conflicts_are_rejected(monkeypatch, name, value):
    _clear_probe_environment(monkeypatch)
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError):
        _PROBE._configure_rocm_environment()


def test_public_preflight_requires_one_headless_card_and_unowned_kfd():
    assert _PROBE._public_safety_preflight(_HEADLESS) == _HEADLESS
    for mutation in (
        {"amd_cards": []},
        {"amd_cards": ["card1", "card2"]},
        {"amd_cards": ["../card1"]},
        {"connected_amd_connectors": ["card1-DP-1"]},
        {"kfd_path": "/private/kfd"},
        {"kfd_accessible": False},
        {"kfd_unowned": False},
    ):
        with pytest.raises(RuntimeError):
            _PROBE._public_safety_preflight({**_HEADLESS, **mutation})


def test_fresh_process_guard_rejects_prior_accelerator_import_without_leaking_name(monkeypatch):
    monkeypatch.setitem(sys.modules, "jax.synthetic_private_module", object())

    with pytest.raises(RuntimeError) as raised:
        _PROBE._assert_fresh_accelerator_process()

    assert "already imported module count" in str(raised.value)
    assert "synthetic_private_module" not in str(raised.value)


def test_error_artifact_redacts_exception_text(monkeypatch):
    secret = "credential=/private/very-secret-token"
    monkeypatch.setattr(
        _PROBE, "_assert_fresh_accelerator_process", lambda: (_ for _ in ()).throw(RuntimeError(secret))
    )
    output = io.StringIO()

    result = _PROBE._execute(SimpleNamespace(platform="rocm", allow_gpu=True, case="smoke8"), output)

    assert result == 1
    artifact = output.getvalue()
    error = _records(output)[-1]
    assert secret not in artifact
    assert "/private/very-secret-token" not in artifact
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
        self._value = value

    def get_raw_value(self):
        return self._value


class _FakeNnx:
    OptState = _FakeVariable


class _ConstructionJax:
    def __init__(self):
        self.device = SimpleNamespace(platform="gpu")
        self.sharding = SimpleNamespace(SingleDeviceSharding=lambda device: _FakeSharding(device))
        self.calls = []

    def devices(self):
        return [self.device]

    def device_put(self, values, shardings, *, donate, may_alias):
        self.calls.append((values, shardings, donate, may_alias))
        return tuple(_FakeArray(value, sharding) for value, sharding in zip(values, shardings, strict=True))

    @staticmethod
    def block_until_ready(values):
        return values


def test_tree_construction_is_one_nonaliasing_batch_with_distinct_nonzero_data(monkeypatch):
    monkeypatch.setattr(_PROBE, "_LEAF_SHAPE", (8, 16))
    monkeypatch.setattr(_PROBE, "_LEAF_BYTES", 8 * 16 * 2)
    jax = _ConstructionJax()
    counters = _PROBE._zero_counters()

    tree, hashes, identity = _PROBE._construct_tree(jax, np, ml_dtypes, _FakeNnx, counters)

    assert len(jax.calls) == 1
    values, shardings, donate, may_alias = jax.calls[0]
    assert len(values) == len(shardings) == 3
    assert donate is False
    assert may_alias is False
    assert hashes["mu"] != hashes["nu"]
    assert np.count_nonzero(values[0]) == values[0].size
    assert np.count_nonzero(values[1]) == values[1].size
    assert not np.array_equal(values[0], values[1])
    assert _PROBE._variable_at(tree, _PROBE._MU_PATH) is identity["variables"]["mu"]
    assert _PROBE._variable_at(tree, _PROBE._NU_PATH) is identity["variables"]["nu"]
    assert _PROBE._variable_at(tree, _PROBE._COUNT_PATH) is identity["variables"]["count"]
    assert counters["construction_device_put_attempts"] == 1
    assert counters["construction_device_put_completions"] == 1


def _placement_fixture():
    device = _FakeSharding("gpu0", "device")
    pinned = device.with_memory_kind("pinned_host")

    def array(shape, dtype, nbytes, sharding):
        return SimpleNamespace(
            shape=shape,
            dtype=dtype,
            nbytes=nbytes,
            sharding=sharding,
            committed=True,
            is_fully_addressable=True,
        )

    mu_value = array(_PROBE._LEAF_SHAPE, "bfloat16", _PROBE._LEAF_BYTES, pinned)
    nu_value = array(_PROBE._LEAF_SHAPE, "bfloat16", _PROBE._LEAF_BYTES, pinned)
    count_value = array((), "int32", 4, device)
    mu_variable = _FakeVariable(mu_value)
    nu_variable = _FakeVariable(nu_value)
    count_variable = _FakeVariable(count_value)
    tree = {
        "opt_state": (
            {
                "count": count_variable,
                "mu": {"weight": mu_variable},
                "nu": {"weight": nu_variable},
            },
        )
    }
    leaves = tuple(
        SimpleNamespace(
            path=path,
            moment_slot=slot,
            shape=_PROBE._LEAF_SHAPE,
            dtype="bfloat16",
            nbytes=_PROBE._LEAF_BYTES,
            device_sharding=device,
            offload_sharding=pinned,
            device_memory_kind="device",
            offload_memory_kind="pinned_host",
        )
        for path, slot in zip(_PROBE._SELECTED_PATHS, ("mu", "nu"), strict=True)
    )
    handle = SimpleNamespace(
        phase="offloaded",
        manifest=leaves,
        leaf_count=2,
        total_bytes=_PROBE._SELECTED_BYTES,
    )
    identity = {
        "variables": {"mu": mu_variable, "nu": nu_variable, "count": count_variable},
        "count_raw_id": id(count_value),
        "device_sharding": device,
    }
    return tree, handle, identity


def test_placement_validator_rejects_variable_identity_and_sharding_changes():
    tree, handle, identity = _placement_fixture()
    assert _PROBE._validate_tree_phase(tree, handle, identity, "offloaded")["exact_selected_shardings"] is True

    original_mu = tree["opt_state"][0]["mu"]["weight"]
    tree["opt_state"][0]["mu"]["weight"] = _FakeVariable(original_mu.get_raw_value())
    with pytest.raises(RuntimeError, match="Variable identity changed"):
        _PROBE._validate_tree_phase(tree, handle, identity, "offloaded")

    tree["opt_state"][0]["mu"]["weight"] = original_mu
    identity["variables"]["nu"].get_raw_value().sharding = identity["device_sharding"]
    with pytest.raises(RuntimeError, match="placement validation"):
        _PROBE._validate_tree_phase(tree, handle, identity, "offloaded")


def test_allocator_95_percent_gate_checks_each_direction():
    selected = _PROBE._SELECTED_BYTES
    enough = int(np.ceil(0.95 * selected))
    base = {"bytes_in_use": 100 * 1024**2}
    released = {"bytes_in_use": base["bytes_in_use"] - enough}

    assert _PROBE._allocator_transition("offload", base, released, "release")["passed"] is True
    assert _PROBE._allocator_transition("stage", released, base, "allocate")["passed"] is True
    with pytest.raises(RuntimeError, match="95% gate"):
        _PROBE._allocator_transition(
            "short_release",
            base,
            {"bytes_in_use": base["bytes_in_use"] - enough + 1},
            "release",
        )


def test_post_method_barrier_must_prove_synchronous_completion(monkeypatch):
    ticks = iter((0.0, _PROBE._MAX_POST_METHOD_BARRIER_SECONDS))
    monkeypatch.setattr(_PROBE.time, "perf_counter", lambda: next(ticks))

    with pytest.raises(RuntimeError, match="synchronous completion"):
        _PROBE._timed_tuple_barrier(
            _RuntimeJax(),
            _runtime_tree(),
            _PROBE._zero_counters(),
        )


def test_effective_bandwidth_uses_binary_gib_and_rejects_invalid_inputs():
    assert _PROBE._effective_gib_per_second(1024**3, 0.5) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="byte count"):
        _PROBE._effective_gib_per_second(0, 1.0)
    with pytest.raises(ValueError, match="duration"):
        _PROBE._effective_gib_per_second(1, 0.0)


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_physical_plateau_is_exact_50ms_over_500ms(tmp_path):
    device = tmp_path / "card1" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n")
    (device / "mem_info_vram_used").write_text("123456\n")
    (device / "mem_info_gtt_used").write_text("654321\n")
    clock = _FakeClock()

    plateau = _PROBE._sample_physical_plateau(
        "card1",
        drm_root=tmp_path,
        clock=clock,
        sleep=clock.sleep,
    )

    assert plateau["sample_count"] == 11
    assert plateau["observed_window_seconds"] == pytest.approx(0.5)
    assert [sample["relative_seconds"] for sample in plateau["samples"]] == pytest.approx(
        [index * 0.05 for index in range(11)]
    )
    assert plateau["vram"]["readable_samples"] == 11
    assert plateau["vram"]["last_bytes"] == 123456
    assert plateau["gtt"]["last_bytes"] == 654321


def _plateau(vram, gtt):
    return {
        "vram": {
            "observed": vram is not None,
            "minimum_bytes": vram,
            "maximum_bytes": vram,
        },
        "gtt": {
            "observed": gtt is not None,
            "minimum_bytes": gtt,
            "maximum_bytes": gtt,
        },
    }


def test_physical_release_target_is_informational_not_allocator_gate():
    unchanged = _PROBE._physical_transition("offload", _plateau(100, 200), _plateau(100, 200), "release")
    missing = _PROBE._physical_transition("offload", _plateau(None, None), _plateau(None, None), "release")

    assert unchanged["physical_release_target_met"] is False
    assert unchanged["gate_effect"] == "informational_only"
    assert unchanged["failure_effect"] == "none_bfc_may_retain_physical_pages"
    assert missing["physical_release_target_met"] is None
    assert missing["vram_observed"] is False


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
    __version__ = "PRIVATE_JAX_VERSION"

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
    device = SimpleNamespace(memory_kind="device")
    return {
        "opt_state": (
            {
                "count": _FakeVariable(np.asarray(173, dtype=np.int32)),
                "mu": {"weight": _FakeVariable(_FakeArray(np.ones((2,), dtype=np.float32), device))},
                "nu": {"weight": _FakeVariable(_FakeArray(np.ones((2,), dtype=np.float32), device))},
            },
        )
    }


def _valid_environment(monkeypatch):
    _clear_probe_environment(monkeypatch)
    return _PROBE._configure_rocm_environment()


def _install_run_mocks(monkeypatch):
    tree = _runtime_tree()
    calls = []
    snapshots = iter(
        {"bytes_in_use": value}
        for value in (
            1,
            100 * 1024**2,
            91 * 1024**2,
            100 * 1024**2,
            91 * 1024**2,
            100 * 1024**2,
            91 * 1024**2,
        )
    )

    def construct(_jax, _np, _ml, _nnx, counters):
        counters["construction_device_put_attempts"] += 1
        counters["construction_device_put_completions"] += 1
        return tree, {"mu": "a" * 64, "nu": "b" * 64}, {"mock": True}

    def offload(received_tree, *, paths, memory_kind):
        assert received_tree is tree
        assert paths == _PROBE._SELECTED_PATHS
        assert memory_kind == "pinned_host"
        calls.append("initial_offload")
        return _CycleHandle(calls)

    def oracle(_jax, _np, received_tree, expected, counters):
        assert received_tree is tree
        assert expected == {"mu": "a" * 64, "nu": "b" * 64}
        counters["device_get_attempts"] += 1
        counters["device_get_completions"] += 1
        calls.append("device_get")
        return {"passed": True}

    monkeypatch.setattr(_PROBE, "_construct_tree", construct)
    monkeypatch.setattr(_PROBE, "_validate_tree_phase", lambda *_args: {"validated": True})
    monkeypatch.setattr(_PROBE, "_allocator_snapshot", lambda _device: next(snapshots))
    monkeypatch.setattr(_PROBE, "_sample_physical_plateau", lambda _card: _plateau(100, 200))
    monkeypatch.setattr(_PROBE, "_device_get_oracle", oracle)
    tick = iter(index * 0.001 for index in range(100))
    monkeypatch.setattr(_PROBE.time, "perf_counter", lambda: next(tick))
    return tree, calls, offload


def test_mocked_runtime_has_exact_transfer_counts_order_journals_and_no_hidden_invocation(monkeypatch):
    environment = _valid_environment(monkeypatch)
    _tree, calls, offload = _install_run_mocks(monkeypatch)
    jax = _RuntimeJax()
    dependencies = (
        jax,
        SimpleNamespace(__version__="PRIVATE_JAXLIB_VERSION"),
        SimpleNamespace(get_backend=lambda: SimpleNamespace(platform_version="ROCm PRIVATE PLATFORM PATH")),
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
    assert calls == ["initial_offload", "stage_back", "reoffload", "stage_back", "device_get", "reoffload"]
    assert counters == _PROBE._completed_counters()
    records = _records(output)
    artifact = output.getvalue()
    assert "PRIVATE_JAX_VERSION" not in artifact
    assert "PRIVATE_JAXLIB_VERSION" not in artifact
    assert "ROCm PRIVATE PLATFORM PATH" not in artifact
    assert [record["method"] for record in records if record["record_type"] == "timed_transfer"] == [
        "stage_back",
        "reoffload",
    ]
    timed = [record for record in records if record["record_type"] == "timed_transfer"]
    assert all(record["effective_gib_per_second"] > 0 for record in timed)
    assert all(
        record["post_method_tuple_barrier_seconds"] < _PROBE._MAX_POST_METHOD_BARRIER_SECONDS for record in timed
    )
    assert jax.block_calls == 2
    assert [record["stage"] for record in records if record["record_type"] == "journal_checkpoint"] == [
        "after_backend_initialization_attempt",
        "after_tree_construction_attempt",
        "after_initial_offload_attempt",
        "after_timed_stage_back_attempt",
        "after_timed_reoffload_attempt",
        "after_final_stage_back_attempt",
        "after_device_get_attempt",
        "after_final_reoffload_attempt",
    ]
    transitions = [record for record in records if record["record_type"] == "memory_transition"]
    assert len(transitions) == 5
    assert all(record["allocator"]["passed"] for record in transitions)
    assert all(record["physical"]["gate_effect"] == "informational_only" for record in transitions)
    assert records[-1]["record_type"] == "smoke8_passed"
    assert records[-1]["round_trip_effective_gib_per_second"] > 0
    assert records[-1]["counters"] == _PROBE._completed_counters()


def test_command_buffer_failure_precedes_backend_and_all_transfers(monkeypatch):
    environment = _valid_environment(monkeypatch)
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_command_buffer=true")
    jax = _RuntimeJax()
    backend_calls = []
    jax.default_backend = lambda: backend_calls.append(True) or "gpu"
    output = io.StringIO()

    with pytest.raises(RuntimeError, match="exactly match"):
        _PROBE._run_rocm(
            output,
            lambda: dict(_CLEAN),
            _PROBE._zero_counters(),
            environment=environment,
            amd_card="card1",
            _dependencies=(
                jax,
                SimpleNamespace(__version__="fake"),
                SimpleNamespace(get_backend=lambda: SimpleNamespace(platform_version="ROCm")),
                np,
                ml_dtypes,
                _FakeNnx,
                lambda *_args, **_kwargs: None,
            ),
        )

    assert backend_calls == []
    assert output.getvalue() == ""


def test_device_get_oracle_fails_on_bitwise_hash_mismatch(monkeypatch):
    monkeypatch.setattr(_PROBE, "_LEAF_SHAPE", (2,))
    tree = {
        "opt_state": (
            {
                "count": _FakeVariable(np.asarray(173, dtype=np.int32)),
                "mu": {"weight": _FakeVariable(np.asarray([1, 2], dtype=ml_dtypes.bfloat16))},
                "nu": {"weight": _FakeVariable(np.asarray([3, 4], dtype=ml_dtypes.bfloat16))},
            },
        )
    }
    expected = {
        "mu": _PROBE._hash_host_array(np.asarray([1, 2], dtype=ml_dtypes.bfloat16)),
        "nu": "0" * 64,
    }

    with pytest.raises(RuntimeError, match="bitwise SHA"):
        _PROBE._device_get_oracle(_RuntimeJax(), np, tree, expected, _PROBE._zero_counters())


def test_module_scope_and_ast_prove_no_eager_accelerator_import_or_extra_calls():
    forbidden_roots = {"jax", "jaxlib", "flax", "numpy", "ml_dtypes", "skyrl"}
    module_imports = []
    for node in _TREE.body:
        if isinstance(node, ast.Import):
            module_imports.extend(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_imports.append(node.module.split(".", 1)[0])
    assert forbidden_roots.isdisjoint(module_imports)

    calls = [node for node in ast.walk(_TREE) if isinstance(node, ast.Call)]
    attributes = [node.func.attr for node in calls if isinstance(node.func, ast.Attribute)]
    names = [node.func.id for node in calls if isinstance(node.func, ast.Name)]
    assert attributes.count("stage_back") == 2
    assert attributes.count("reoffload") == 2
    assert attributes.count("device_get") == 1
    assert attributes.count("jit") == 0
    assert attributes.count("compile") == 0
    assert names.count("offload_optimizer_moments") == 1

    run_start = _SOURCE.index("def _run_rocm")
    run_source = _SOURCE[run_start : _SOURCE.index("def _execute", run_start)]
    assert run_source.index("_prove_command_buffers_disabled") < run_source.index("import jax")
    assert run_source.index("offload_optimizer_moments(") < run_source.index("_timed_stage_back(")
    assert run_source.index("_timed_stage_back(") < run_source.index("_timed_reoffload(")
    assert run_source.index("successor.stage_back()") < run_source.index("_device_get_oracle(")
    assert run_source.index("_device_get_oracle(") < run_source.index("successor.reoffload()")


def test_default_subprocess_refuses_without_importing_accelerator_stack():
    program = f"""
import contextlib, importlib.util, io, json, sys
path = {str(_PROBE_PATH)!r}
before = set(sys.modules)
spec = importlib.util.spec_from_file_location('isolated_offload_probe', path)
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
    assert [record["record_type"] for record in report["records"]] == ["manifest", "refused"]
