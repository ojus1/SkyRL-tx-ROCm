import os
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import safetensors.numpy
import torch
from cloudpathlib import CloudPath, implementation_registry
from cloudpathlib.local import local_s3_implementation
from flax import nnx
from jax.tree_util import DictKey
from peft import PeftModel
from safetensors import safe_open as real_safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from skyrl.tinker.types import LoraConfig
from skyrl.tx.layers.lora import FusedLoRALinear, init_lora_adapter
from skyrl.tx.models.configs import Qwen3Config
from skyrl.tx.models.qwen3 import Qwen3ForCausalLM
from skyrl.tx.utils import models
from skyrl.tx.utils.models import (
    _fuse_checkpoint_arrays_numpy,
    extract_adapter_state,
    insert_adapter_state,
    is_stacked_path,
)
from skyrl.utils.storage import download_and_unpack


def create_test_model(base_model_name: str, rank: int, alpha: int, adapter_index: int):
    """Create a small Qwen3 model for testing with LoRA enabled."""
    base_config = AutoConfig.from_pretrained(base_model_name)
    # Make it smaller for testing
    base_config.num_hidden_layers = 1
    base_config.hidden_size = 64
    base_config.intermediate_size = 128
    base_config.num_attention_heads = 2
    base_config.num_key_value_heads = 2
    # transformers >=5.4 validates len(layer_types) == num_hidden_layers.
    layer_types = getattr(base_config, "layer_types", None)
    if layer_types is not None:
        base_config.layer_types = list(layer_types[: base_config.num_hidden_layers])

    config = Qwen3Config(base_config, max_lora_adapters=5, max_lora_rank=32, shard_attention_heads=True)

    mesh = jax.make_mesh((1, 1), ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2)
    with jax.set_mesh(mesh):
        model = Qwen3ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        init_lora_adapter(model, adapter_index=adapter_index, lora_config=LoraConfig(rank=rank, alpha=alpha, seed=0))

    return config, base_config, model


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_fuse_checkpoint_arrays_numpy_matches_runtime_layout(batch_shape):
    group_sizes = (4, 2, 2)
    num_groups = 3
    arrays = [
        np.arange(np.prod((*batch_shape, num_groups * size)), dtype=np.float32).reshape(*batch_shape, num_groups * size)
        + component * 1_000
        for component, size in enumerate(group_sizes)
    ]

    expected = np.asarray(FusedLoRALinear.fuse(*arrays, group_sizes=group_sizes))
    actual = _fuse_checkpoint_arrays_numpy(arrays, group_sizes)

    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, expected)


def test_fuse_checkpoint_arrays_numpy_does_not_dispatch_to_jax(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("checkpoint layout transform dispatched to JAX")

    monkeypatch.setattr(jax, "device_put", fail)
    monkeypatch.setattr(jnp, "concatenate", fail)

    actual = _fuse_checkpoint_arrays_numpy(
        [np.arange(12).reshape(2, 6), np.arange(4).reshape(2, 2)],
        (3, 1),
    )

    assert isinstance(actual, np.ndarray)
    assert actual.shape == (2, 8)


@pytest.mark.parametrize(
    "arrays,group_sizes,error",
    [
        ([np.ones((2, 4))], (2, 2), "one nonempty"),
        ([np.ones((2, 4))], (0,), "positive"),
        ([np.ones(())], (1,), "at least one dimension"),
        ([np.ones((2, 5))], (2,), "not divisible"),
        ([np.ones((2, 4)), np.ones((3, 2))], (2, 1), "batch shape"),
        ([np.ones((2, 4)), np.ones((2, 3))], (2, 1), "last dim"),
    ],
)
def test_fuse_checkpoint_arrays_numpy_rejects_invalid_layout(arrays, group_sizes, error):
    with pytest.raises(ValueError, match=error):
        _fuse_checkpoint_arrays_numpy(arrays, group_sizes)


class _TinyCheckpointLeaf(nnx.Module):
    def __init__(self, shape: tuple[int, ...]):
        self.kernel = nnx.Param(jnp.zeros(shape, dtype=jnp.float32))


class _TinyCheckpointModel(nnx.Module):
    def __init__(self, *, name: str = "proj", shape: tuple[int, ...] = (2, 3)):
        setattr(self, name, _TinyCheckpointLeaf(shape))


class _TinyTwoCheckpointModel(nnx.Module):
    def __init__(self):
        self.first = _TinyCheckpointLeaf((2, 3))
        self.second = _TinyCheckpointLeaf((2, 3))


class _TinyExpertModel(nnx.Module):
    def __init__(self):
        self.experts = _TinyCheckpointLeaf((2, 2, 3))


class _TinyExpertConfig:
    def get_num_experts(self):
        return 2


class _TinyFusedLoraLeaf(nnx.Module):
    def __init__(self):
        self.lora_A = nnx.Param(jnp.zeros((2, 3), dtype=jnp.float32))
        self.lora_B = nnx.Param(jnp.zeros((2, 6), dtype=jnp.float32))


class _TinyFusedLoraModel(nnx.Module):
    def __init__(self):
        self.fused = _TinyFusedLoraLeaf()


class _TinyAdapterLeaf(nnx.Module):
    def __init__(self):
        self.lora_A = nnx.Param(jnp.zeros((3, 2, 4), dtype=jnp.float32))
        self.lora_B = nnx.Param(jnp.zeros((3, 4, 3), dtype=jnp.float32))


class _TinyAdapterModel(nnx.Module):
    def __init__(self):
        self.proj = _TinyAdapterLeaf()


class _TrackedSafetensorReader:
    def __init__(self, reader, shard: str, events: list[tuple]):
        self._reader = reader
        self._shard = shard
        self._events = events

    def keys(self):
        return self._reader.keys()

    def get_tensor(self, key: str):
        array = self._reader.get_tensor(key)
        self._events.append(("load", self._shard, key))
        payload = array
        while isinstance(payload.base, np.ndarray):
            payload = payload.base
        weakref.finalize(
            payload,
            self._events.append,
            ("release", self._shard, key),
        )
        return array


def _instrument_safe_open(monkeypatch, events: list[tuple]) -> None:
    @contextmanager
    def tracked_safe_open(filename, framework, **kwargs):
        shard = Path(filename).name
        events.append(("open", shard, framework, kwargs.copy()))
        with real_safe_open(filename, framework=framework, **kwargs) as reader:
            try:
                yield _TrackedSafetensorReader(reader, shard, events)
            finally:
                events.append(("close", shard))

    monkeypatch.setattr(models, "safe_open", tracked_safe_open)


def _assert_pread_readers_closed(events: list[tuple]) -> None:
    opens = [event for event in events if event[0] == "open"]
    closes = [event for event in events if event[0] == "close"]
    assert opens
    assert all(framework == "numpy" and kwargs == {"backend": "pread"} for _, _, framework, kwargs in opens)
    assert sorted(shard for _, shard, _, _ in opens) == sorted(shard for _, shard in closes)


def _write_tiny_safetensor(path: Path, tensors: dict[str, np.ndarray]) -> None:
    safetensors.numpy.save_file(
        {key: np.ascontiguousarray(value) for key, value in tensors.items()},
        path,
    )


def _load_tiny_checkpoint(
    checkpoint_dir: Path,
    model: nnx.Module,
    *,
    prefix: str = "",
    config=None,
    **kwargs,
) -> None:
    models.load_safetensors(
        checkpoint_dir,
        config or SimpleNamespace(),
        model,
        prefix=prefix,
        **kwargs,
    )


def test_load_safetensors_streams_a_real_tiny_checkpoint(tmp_path, monkeypatch):
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    _write_tiny_safetensor(
        tmp_path / "model-00001.safetensors",
        {"proj.weight": expected.T},
    )
    model = _TinyCheckpointModel()
    events: list[tuple] = []
    _instrument_safe_open(monkeypatch, events)

    _load_tiny_checkpoint(tmp_path, model)

    np.testing.assert_array_equal(np.asarray(model.proj.kernel[...]), expected)
    _assert_pread_readers_closed(events)


def test_load_safetensors_blocks_and_releases_before_loading_next_payload(tmp_path, monkeypatch):
    first = np.arange(6, dtype=np.float32).reshape(2, 3)
    second = first + 100
    _write_tiny_safetensor(
        tmp_path / "model.safetensors",
        {"first.weight": first.T, "second.weight": second.T},
    )
    model = _TinyTwoCheckpointModel()
    events: list[tuple] = []
    _instrument_safe_open(monkeypatch, events)
    original_device_put = jax.device_put
    original_block_until_ready = jax.block_until_ready

    def tracked_device_put(value, *args, **kwargs):
        events.append(("device_put",))
        # Do not let the CPU backend retain the reader-owned input via zero-copy.
        return original_device_put(np.array(value, copy=True), *args, **kwargs)

    def tracked_block_until_ready(value):
        result = original_block_until_ready(value)
        events.append(("block_done",))
        return result

    monkeypatch.setattr(jax, "device_put", tracked_device_put)
    monkeypatch.setattr(jax, "block_until_ready", tracked_block_until_ready)

    _load_tiny_checkpoint(tmp_path, model)

    first_load = events.index(("load", "model.safetensors", "first.weight"))
    first_device_put = events.index(("device_put",), first_load)
    first_block_done = events.index(("block_done",), first_device_put)
    first_release = events.index(("release", "model.safetensors", "first.weight"))
    second_load = events.index(("load", "model.safetensors", "second.weight"))
    assert first_load < first_device_put < first_block_done < first_release < second_load, events
    np.testing.assert_array_equal(np.asarray(model.first.kernel[...]), first)
    np.testing.assert_array_equal(np.asarray(model.second.kernel[...]), second)
    _assert_pread_readers_closed(events)


def test_load_safetensors_normalizes_prefix_lazily(tmp_path):
    expected = np.arange(6, dtype=np.float32).reshape(2, 3) + 10
    _write_tiny_safetensor(
        tmp_path / "adapter.safetensors",
        {"base_model.model.proj.weight": expected.T},
    )
    model = _TinyCheckpointModel()

    _load_tiny_checkpoint(tmp_path, model, prefix="base_model.model.")

    np.testing.assert_array_equal(np.asarray(model.proj.kernel[...]), expected)


def test_load_safetensors_materializes_only_current_fused_components(tmp_path, monkeypatch):
    first = np.arange(8, dtype=np.float32).reshape(2, 4)
    second = np.arange(4, dtype=np.float32).reshape(2, 2) + 100
    _write_tiny_safetensor(
        tmp_path / "fused.safetensors",
        {"first.weight": first.T, "second.weight": second.T},
    )
    model = _TinyCheckpointModel(name="fused", shape=(2, 6))
    expected = _fuse_checkpoint_arrays_numpy([first, second], (2, 1))
    original_fuse = models._fuse_checkpoint_arrays_numpy
    events: list[str] = []

    def checked_numpy_fuse(arrays, group_sizes):
        assert all(isinstance(array, np.ndarray) for array in arrays)
        events.append("numpy_fuse")
        return original_fuse(arrays, group_sizes)

    original_device_put = jax.device_put

    def tracked_device_put(*args, **kwargs):
        events.append("device_put")
        return original_device_put(*args, **kwargs)

    monkeypatch.setattr(
        models,
        "get_fused_info",
        lambda _model: {"fused": (("first", "second"), (2, 1))},
    )
    monkeypatch.setattr(models, "_fuse_checkpoint_arrays_numpy", checked_numpy_fuse)
    monkeypatch.setattr(jax, "device_put", tracked_device_put)

    _load_tiny_checkpoint(tmp_path, model)

    assert events == ["numpy_fuse", "device_put"]
    np.testing.assert_array_equal(np.asarray(model.fused.kernel[...]), expected)


def test_load_safetensors_preserves_expert_stack_values(tmp_path):
    expected = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    _write_tiny_safetensor(
        tmp_path / "experts.safetensors",
        {
            "experts.0.kernel": expected[0].T,
            "experts.1.kernel": expected[1].T,
        },
    )
    model = _TinyExpertModel()

    _load_tiny_checkpoint(tmp_path, model, config=_TinyExpertConfig())

    np.testing.assert_array_equal(np.asarray(model.experts.kernel[...]), expected)


def test_load_safetensors_preserves_fused_lora_a_and_b_values(tmp_path, monkeypatch):
    shared_a = np.arange(6, dtype=np.float32).reshape(2, 3)
    first_b = np.arange(8, dtype=np.float32).reshape(2, 4) + 100
    second_b = np.arange(4, dtype=np.float32).reshape(2, 2) + 200
    _write_tiny_safetensor(
        tmp_path / "adapter.safetensors",
        {
            "first.lora_A.weight": shared_a.T,
            "second.lora_A.weight": shared_a.T,
            "first.lora_B.weight": first_b.T,
            "second.lora_B.weight": second_b.T,
        },
    )
    model = _TinyFusedLoraModel()
    monkeypatch.setattr(
        models,
        "get_fused_info",
        lambda _model: {"fused": (("first", "second"), (2, 1))},
    )

    _load_tiny_checkpoint(tmp_path, model, skip_lora=False)

    np.testing.assert_array_equal(np.asarray(model.fused.lora_A[...]), shared_a)
    np.testing.assert_array_equal(
        np.asarray(model.fused.lora_B[...]),
        _fuse_checkpoint_arrays_numpy([first_b, second_b], (2, 1)),
    )


def test_load_safetensors_preserves_adapter_slot_values(tmp_path):
    expected_a = np.arange(4, dtype=np.float32).reshape(2, 2) + 300
    expected_b = np.arange(6, dtype=np.float32).reshape(2, 3) + 400
    _write_tiny_safetensor(
        tmp_path / "adapter.safetensors",
        {
            "proj.lora_A.weight": expected_a.T,
            "proj.lora_B.weight": expected_b.T,
        },
    )
    model = _TinyAdapterModel()

    _load_tiny_checkpoint(
        tmp_path,
        model,
        skip_lora=False,
        adapter_index=1,
        rank=2,
    )

    np.testing.assert_array_equal(np.asarray(model.proj.lora_A[...][1, :, :2]), expected_a)
    np.testing.assert_array_equal(np.asarray(model.proj.lora_B[...][1, :2, :]), expected_b)
    assert np.count_nonzero(np.asarray(model.proj.lora_A[...][0])) == 0
    assert np.count_nonzero(np.asarray(model.proj.lora_A[...][2])) == 0
    assert np.count_nonzero(np.asarray(model.proj.lora_B[...][0])) == 0
    assert np.count_nonzero(np.asarray(model.proj.lora_B[...][2])) == 0


def test_adapter_arrayref_write_blocks_host_conversion_and_parent_update(monkeypatch):
    from skyrl.tx.layers.stacked import ArrayRef

    parent = nnx.Param(jnp.zeros((2, 3, 2, 4), dtype=jnp.float32))
    param = ArrayRef(parent, 1)
    tensor = np.arange(4, dtype=np.float32).reshape(2, 2)
    blocked_shapes = []
    original_block_until_ready = jax.block_until_ready

    def tracked_block_until_ready(value):
        result = original_block_until_ready(value)
        blocked_shapes.append(value.shape)
        return result

    monkeypatch.setattr(jax, "block_until_ready", tracked_block_until_ready)

    models._write_checkpoint_adapter_slice(
        param,
        tensor,
        (1, slice(None), slice(None, 2)),
    )

    assert blocked_shapes == [(2, 2), (3, 2, 4), (3, 2, 4)]
    np.testing.assert_array_equal(
        np.asarray(parent[...][1, 1, :, :2]),
        tensor,
    )
    assert np.count_nonzero(np.asarray(parent[...][0])) == 0


def test_load_safetensors_rejects_duplicate_normalized_keys_deterministically(tmp_path, monkeypatch):
    value = np.arange(6, dtype=np.float32).reshape(3, 2)
    _write_tiny_safetensor(tmp_path / "z-shard.safetensors", {"proj.weight": value})
    _write_tiny_safetensor(
        tmp_path / "a-shard.safetensors",
        {"base_model.model.proj.weight": value + 1},
    )
    model = _TinyCheckpointModel()
    events: list[tuple] = []
    _instrument_safe_open(monkeypatch, events)

    with pytest.raises(
        RuntimeError,
        match=(
            "Duplicate safetensors key after prefix normalization 'proj.weight': "
            "a-shard.safetensors:'base_model.model.proj.weight' and "
            "z-shard.safetensors:'proj.weight'"
        ),
    ):
        _load_tiny_checkpoint(tmp_path, model, prefix="base_model.model.")

    _assert_pread_readers_closed(events)


def test_load_safetensors_preserves_missing_key_failure(tmp_path):
    _write_tiny_safetensor(
        tmp_path / "model.safetensors",
        {"different.weight": np.ones((3, 2), dtype=np.float32)},
    )

    with pytest.raises(RuntimeError, match="Missing key while loading .*: proj.weight"):
        _load_tiny_checkpoint(tmp_path, _TinyCheckpointModel())


@pytest.mark.parametrize(
    ("failure", "error_type", "error"),
    [
        ("missing", RuntimeError, "Missing key while loading .*: second.weight"),
        ("shape", AssertionError, "shape mismatch for second.weight"),
        ("device_put", RuntimeError, "injected second device_put failure"),
    ],
)
def test_load_safetensors_closes_pread_readers_after_partial_load_failure(
    tmp_path, monkeypatch, failure, error_type, error
):
    first = np.arange(6, dtype=np.float32).reshape(2, 3)
    second = first + 100
    tensors = {"first.weight": first.T}
    if failure == "shape":
        tensors["second.weight"] = np.ones((4, 2), dtype=np.float32)
    elif failure == "device_put":
        tensors["second.weight"] = second.T
    _write_tiny_safetensor(tmp_path / "model.safetensors", tensors)
    model = _TinyTwoCheckpointModel()
    events: list[tuple] = []
    _instrument_safe_open(monkeypatch, events)

    if failure == "device_put":
        original_device_put = jax.device_put
        call_count = 0

        def fail_second_device_put(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("injected second device_put failure")
            return original_device_put(*args, **kwargs)

        monkeypatch.setattr(jax, "device_put", fail_second_device_put)

    with pytest.raises(error_type, match=error):
        _load_tiny_checkpoint(tmp_path, model)

    assert ("load", "model.safetensors", "first.weight") in events
    if failure != "missing":
        assert ("load", "model.safetensors", "second.weight") in events
    _assert_pread_readers_closed(events)


def test_load_safetensors_never_uses_eager_load_file(tmp_path, monkeypatch):
    expected = np.arange(6, dtype=np.float32).reshape(2, 3) + 20
    _write_tiny_safetensor(tmp_path / "model.safetensors", {"proj.weight": expected.T})

    def fail_eager_load(*_args, **_kwargs):
        raise AssertionError("eager safetensors load_file path was used")

    monkeypatch.setattr(safetensors.numpy, "load_file", fail_eager_load)
    model = _TinyCheckpointModel()

    _load_tiny_checkpoint(tmp_path, model)

    np.testing.assert_array_equal(np.asarray(model.proj.kernel[...]), expected)


def test_residency_probe_forces_command_buffer_disable_last():
    from rocm.probe_model_residency import (
        _DISABLE_COMMAND_BUFFERS,
        _force_xla_flag,
    )

    effective = _force_xla_flag(
        "--unrelated=1 --xla_gpu_enable_command_buffer=CUBLAS " "--xla_gpu_enable_command_buffer=",
        _DISABLE_COMMAND_BUFFERS,
    )

    assert effective.split() == ["--unrelated=1", _DISABLE_COMMAND_BUFFERS]


@pytest.mark.parametrize("name", ["ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"])
def test_residency_probe_rejects_a_different_visible_gpu(monkeypatch, name):
    import argparse

    from rocm.probe_model_residency import _configure_environment

    for variable in (
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv(name, "1")
    with pytest.raises(RuntimeError, match=rf"{name}=0"):
        _configure_environment(argparse.Namespace(platform="rocm"))


def test_residency_probe_gpu_preflight_requires_headless_unowned_kfd(tmp_path):
    import stat

    from rocm.probe_model_residency import _gpu_preflight

    drm_root = tmp_path / "drm"
    (drm_root / "card1" / "device").mkdir(parents=True)
    (drm_root / "card1" / "device" / "vendor").write_text("0x1002\n")
    (drm_root / "card1-Writeback-1").mkdir()
    (drm_root / "card1-Writeback-1" / "status").write_text("connected\n")
    kfd_path = tmp_path / "dev" / "kfd"

    preflight = _gpu_preflight(
        drm_root=drm_root,
        kfd_path=kfd_path,
        stat_fn=lambda _path: SimpleNamespace(st_mode=stat.S_IFCHR),
        access_fn=lambda *_args: True,
        which_fn=lambda _name: "/usr/bin/fuser",
        run_fn=lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr=""),
    )

    assert preflight == {
        "amd_cards": ["card1"],
        "connected_amd_connectors": [],
        "kfd_path": str(kfd_path),
        "kfd_accessible": True,
        "kfd_unowned": True,
    }

    (drm_root / "card1-HDMI-A-1").mkdir()
    (drm_root / "card1-HDMI-A-1" / "status").write_text("connected\n")
    with pytest.raises(RuntimeError, match="AMD display connector is active"):
        _gpu_preflight(drm_root=drm_root)


def test_residency_probe_gpu_preflight_rejects_existing_kfd_owner(tmp_path):
    import stat

    from rocm.probe_model_residency import _gpu_preflight

    drm_root = tmp_path / "drm"
    (drm_root / "card1" / "device").mkdir(parents=True)
    (drm_root / "card1" / "device" / "vendor").write_text("0x1002\n")

    with pytest.raises(RuntimeError, match="already owned: 1234"):
        _gpu_preflight(
            drm_root=drm_root,
            kfd_path=tmp_path / "dev" / "kfd",
            stat_fn=lambda _path: SimpleNamespace(st_mode=stat.S_IFCHR),
            access_fn=lambda *_args: True,
            which_fn=lambda _name: "/usr/bin/fuser",
            run_fn=lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="1234", stderr=""),
        )


def test_residency_probe_accepts_exact_preallocate85_environment(monkeypatch):
    import argparse

    from rocm.probe_model_residency import _configure_environment

    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "bfc")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    monkeypatch.setenv("XLA_CLIENT_MEM_FRACTION", "0.85")
    monkeypatch.setenv("SKYRL_QWEN35_MEMORY_MODE", "preallocate85")

    _configure_environment(argparse.Namespace(platform="rocm", allocator_mode="preallocate85"))

    assert os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "bfc"
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert os.environ["XLA_CLIENT_MEM_FRACTION"] == "0.85"


@pytest.mark.parametrize(
    "allocator_mode,environment,error",
    [
        (
            "growth",
            {"SKYRL_QWEN35_MEMORY_MODE": "preallocate85"},
            "conflicts with --allocator-mode growth",
        ),
        (
            "preallocate85",
            {"XLA_PYTHON_CLIENT_ALLOCATOR": "platform"},
            "requires the BFC allocator",
        ),
        (
            "preallocate85",
            {"XLA_CLIENT_MEM_FRACTION": "0.9"},
            "must be exactly 0.85",
        ),
        (
            "preallocate85",
            {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85"},
            "is deprecated",
        ),
    ],
)
def test_residency_probe_rejects_conflicting_allocator_environment(monkeypatch, allocator_mode, environment, error):
    import argparse

    from rocm.probe_model_residency import _configure_environment

    for name in (
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "SKYRL_QWEN35_MEMORY_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=error):
        _configure_environment(argparse.Namespace(platform="rocm", allocator_mode=allocator_mode))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_abstract_checkpoint_load_matches_eager_model(tmp_path, dtype):
    from transformers import (
        Qwen3_5Config as HFQwen3_5Config,
    )
    from transformers import (
        Qwen3_5TextConfig,
    )

    from skyrl.tx.models.configs import Qwen3_5Config
    from skyrl.tx.models.qwen3_5 import Qwen3_5ForCausalLM

    text_config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        layer_types=["linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        tie_word_embeddings=True,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
            "mrope_section": [3, 3, 2],
        },
    )
    base_config = HFQwen3_5Config(
        text_config=text_config,
        tie_word_embeddings=True,
    )
    config = Qwen3_5Config(
        base_config,
        max_lora_adapters=2,
        max_lora_rank=4,
        shard_attention_heads=True,
    )
    mesh = jax.make_mesh(
        (1, 1, 1),
        ("fsdp", "ep", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * 3,
    )

    def model_factory():
        return Qwen3_5ForCausalLM(config, dtype=dtype, rngs=nnx.Rngs(0))

    def is_base_parameter(path):
        return not any(name in path for name in ("lora_A", "lora_B", "lora_scaling", "lora_ranks"))

    with jax.set_mesh(mesh), nnx.use_eager_sharding(True):
        eager_model = model_factory()
        models.save_safetensors(
            config.get_text_config(),
            eager_model,
            tmp_path / "model.safetensors",
            filter_fn=is_base_parameter,
        )

        loaded_model = models.load_qwen3_5_safetensors_abstract(
            tmp_path,
            config,
            Qwen3_5ForCausalLM,
            dtype=dtype,
            mesh=mesh,
        )

    eager_state = nnx.to_flat_state(nnx.state(eager_model))
    loaded_state = nnx.to_flat_state(nnx.state(loaded_model))
    assert [path for path, _ in eager_state] == [path for path, _ in loaded_state]
    for (path, expected), (_, actual) in zip(eager_state, loaded_state):
        expected_value = expected.get_raw_value()
        actual_value = actual.get_raw_value()
        assert actual_value.dtype == expected_value.dtype
        assert actual_value.sharding == expected_value.sharding
        np.testing.assert_array_equal(
            np.asarray(actual_value),
            np.asarray(expected_value),
            err_msg=str(path),
        )

    input_ids = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32)
    attention_mask = jnp.ones_like(input_ids)
    with jax.set_mesh(mesh):
        eager_output = eager_model(
            input_ids,
            attention_mask=attention_mask,
            is_training=True,
        ).last_hidden_state
        loaded_output = loaded_model(
            input_ids,
            attention_mask=attention_mask,
            is_training=True,
        ).last_hidden_state
    np.testing.assert_array_equal(
        np.asarray(loaded_output),
        np.asarray(eager_output),
    )


@pytest.mark.parametrize(
    "model_class,model_type,layer_types,error",
    [
        (object, "qwen3_5_text", ["full_attention"], "Qwen3.5 causal-LM model class"),
        (None, "qwen3", ["full_attention"], "requires a Qwen3.5 text config"),
        (None, "qwen3_5_text", ["future_attention"], "does not support Qwen3.5 layer types"),
    ],
)
def test_abstract_checkpoint_load_rejects_unsupported_family_or_config(
    tmp_path, model_class, model_type, layer_types, error
):
    from skyrl.tx.models.qwen3_5 import Qwen3_5ForCausalLM

    class UnsupportedConfig:
        max_lora_rank = 4

        def __init__(self):
            self.text_config = SimpleNamespace(model_type=model_type)

        def get_config(self):
            return SimpleNamespace(layer_types=layer_types)

    if model_class is None:
        model_class = Qwen3_5ForCausalLM
    mesh = jax.make_mesh((1,), ("tp",), axis_types=(jax.sharding.AxisType.Auto,))

    with pytest.raises(ValueError, match=error):
        models.load_qwen3_5_safetensors_abstract(
            tmp_path,
            UnsupportedConfig(),
            model_class,
            dtype=jnp.float32,
            mesh=mesh,
        )


def test_abstract_lora_materialization_leaves_unknown_state_abstract():
    class TinyState(nnx.Module):
        def __init__(self):
            self.base = nnx.Param(jax.ShapeDtypeStruct((2,), jnp.float32))
            self.lora_A = nnx.Param(jax.ShapeDtypeStruct((1, 2, 1), jnp.float32))

    model = TinyState()
    models.materialize_abstract_lora_initializers(model, max_lora_rank=1)

    assert isinstance(model.base.get_raw_value(), jax.ShapeDtypeStruct)
    assert not isinstance(model.lora_A.get_raw_value(), jax.ShapeDtypeStruct)
    with pytest.raises(RuntimeError, match=r"1 abstract state leaves: base"):
        models.assert_model_fully_materialized(model)


@pytest.mark.parametrize("storage_type", ["local", "cloud"])
def test_save_load_lora_checkpoint(storage_type: str, monkeypatch, tmp_path: Path):
    base_model_name = "Qwen/Qwen3-0.6B"
    # Setup output path for tar.gz file based on storage type
    if storage_type == "cloud":
        monkeypatch.setitem(implementation_registry, "s3", local_s3_implementation)
        client = local_s3_implementation.client_class(local_storage_dir=tmp_path)
        output_path = CloudPath("s3://bucket/checkpoint.tar.gz", client=client)
    else:
        output_path = tmp_path / "checkpoint.tar.gz"

    rank, alpha, adapter_index = 8, 16, 2
    config, base_config, model = create_test_model(base_model_name, rank, alpha, adapter_index)
    adapter_config = LoraConfig(rank=rank, alpha=alpha, seed=0)

    # Set LoRA weights to random values for testing (to catch transpose bugs)
    qkv_proj = model.model.layers[0].self_attn.qkv_proj
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(42))
    qkv_proj.lora_A[...] = jax.random.normal(rng1, qkv_proj.lora_A[...].shape)
    qkv_proj.lora_B[...] = jax.random.normal(rng2, qkv_proj.lora_B[...].shape)

    # Store expected values (trimmed to rank and transposed)
    # The fused qkv_proj lora_A is shared, so q_proj gets the same lora_A
    expected_lora_A = np.array(qkv_proj.lora_A[...][adapter_index, :, :rank].T)
    # For lora_B, we need to unpack the fused output and get just the q portion
    fused_lora_B = np.array(qkv_proj.lora_B[...][adapter_index, :rank, :])
    q_lora_B, _, _ = FusedLoRALinear.split(fused_lora_B, qkv_proj.group_sizes)
    expected_lora_B = q_lora_B.T

    # Save and verify checkpoint exists
    models.save_lora_checkpoint(model, base_model_name, adapter_config, adapter_index, output_path, rank=0)
    assert output_path.exists()

    # Load with peft and verify
    with download_and_unpack(output_path) as extracted_dir:
        base_model = AutoModelForCausalLM.from_config(base_config)
        peft_model = PeftModel.from_pretrained(base_model, extracted_dir)

        assert peft_model.peft_config["default"].r == rank
        assert peft_model.peft_config["default"].lora_alpha == alpha

        q_proj_adapter = peft_model.base_model.model.model.layers[0].self_attn.q_proj
        lora_A = q_proj_adapter.lora_A["default"].weight
        lora_B = q_proj_adapter.lora_B["default"].weight

        assert torch.allclose(lora_A, torch.from_numpy(expected_lora_A), atol=1e-6)
        assert torch.allclose(lora_B, torch.from_numpy(expected_lora_B), atol=1e-6)


@pytest.mark.parametrize(
    "path,expected",
    [
        # Stacked paths (DictKey) — real NNX paths include _stacked
        (
            (
                DictKey(key="model"),
                DictKey(key="layers"),
                DictKey(key="_stacked"),
                DictKey(key="self_attn"),
                DictKey(key="lora_A"),
            ),
            True,
        ),
        (
            (
                DictKey(key="model"),
                DictKey(key="layers"),
                DictKey(key="layer_groups"),
                DictKey(key="_stacked"),
                DictKey(key="self_attn"),
                DictKey(key="lora_A"),
            ),
            True,
        ),
        # Non-stacked paths (DictKey)
        ((DictKey(key="model"), DictKey(key="embed_tokens"), DictKey(key="lora_A")), False),
        ((DictKey(key="lm_head"), DictKey(key="lora_A")), False),
        # String paths
        (("model", "layers", "_stacked", "self_attn", "lora_A"), True),
        (("model", "embed_tokens", "lora_A"), False),
    ],
    ids=["stacked_layers", "multi_stacked_layers", "embed_tokens", "lm_head", "str_stacked", "str_embed"],
)
def test_is_stacked_path(path, expected):
    """Test is_stacked_path correctly identifies stacked vs non-stacked paths."""
    assert is_stacked_path(path) is expected


def test_extract_insert_adapter_state_roundtrip():
    """Test that extract_adapter_state and insert_adapter_state are inverses."""
    base_model_name = "Qwen/Qwen3-0.6B"
    rank, alpha, adapter_index = 8, 16, 2
    _, _, model = create_test_model(base_model_name, rank, alpha, adapter_index)

    # Set LoRA weights to random values
    qkv_proj = model.model.layers[0].self_attn.qkv_proj
    rng1, rng2 = jax.random.split(jax.random.PRNGKey(123))
    qkv_proj.lora_A[...] = jax.random.normal(rng1, qkv_proj.lora_A[...].shape)
    qkv_proj.lora_B[...] = jax.random.normal(rng2, qkv_proj.lora_B[...].shape)

    # Split model to get lora_params
    _, lora_params, _ = nnx.split(model, model.is_lora_param, ...)

    # Store original values for comparison
    original_lora_A = np.array(qkv_proj.lora_A[...][adapter_index, :, :rank])
    original_lora_B = np.array(qkv_proj.lora_B[...][adapter_index, :rank, :])

    # Extract adapter state
    extracted = extract_adapter_state(adapter_index, lora_params, rank)

    # Verify extracted shape is correct (no adapter dimension)
    for path, leaf in jax.tree.leaves_with_path(extracted):
        key = path[-2].key if hasattr(path[-2], "key") else str(path[-2])
        if key in {"lora_A", "lora_B"}:
            # Stacked: should have (num_layers, ...) not (num_layers, num_adapters, ...)
            if is_stacked_path(path):
                assert leaf.shape[0] == 1  # num_layers
                assert leaf.ndim == 3  # (layers, in_dim, rank) or (layers, rank, out_dim)

    # Zero out the adapter's weights
    qkv_proj.lora_A[...] = qkv_proj.lora_A[...].at[adapter_index].set(0)
    qkv_proj.lora_B[...] = qkv_proj.lora_B[...].at[adapter_index].set(0)

    # Verify weights are zeroed
    assert np.allclose(qkv_proj.lora_A[...][adapter_index], 0)
    assert np.allclose(qkv_proj.lora_B[...][adapter_index], 0)

    # Re-split to get updated lora_params
    _, lora_params, _ = nnx.split(model, model.is_lora_param, ...)

    # Insert extracted state back (modifies lora_params in-place via nnx.update)
    insert_adapter_state(adapter_index, lora_params, extracted, rank)

    # Verify weights are restored by checking lora_params directly
    for path, leaf in jax.tree.leaves_with_path(lora_params):
        key = path[-2].key if hasattr(path[-2], "key") else str(path[-2])
        # leaf is a state wrapper with .value, or can be an array directly
        arr = leaf.value if hasattr(leaf, "value") else leaf
        if "qkv_proj" in str(path) and key == "lora_A":
            restored_lora_A = np.array(arr[0, adapter_index, :, :rank])
        elif "qkv_proj" in str(path) and key == "lora_B":
            restored_lora_B = np.array(arr[0, adapter_index, :rank, :])

    assert np.allclose(original_lora_A, restored_lora_A), "lora_A not restored correctly"
    assert np.allclose(original_lora_B, restored_lora_B), "lora_B not restored correctly"
