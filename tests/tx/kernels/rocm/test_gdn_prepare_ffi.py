from __future__ import annotations

import ast
import hashlib
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from skyrl.tx.kernels.rocm import gdn_prepare_ffi as prepare
from skyrl.tx.kernels.rocm import gdn_prepare_oracle as oracle

_REPO = Path(__file__).parents[4]
_PYTHON_SOURCE = _REPO / "skyrl" / "tx" / "kernels" / "rocm" / "gdn_prepare_ffi.py"
_ORACLE_SOURCE = _REPO / "skyrl" / "tx" / "kernels" / "rocm" / "gdn_prepare_oracle.py"
_HIP_SOURCE = (
    _REPO / "skyrl" / "tx" / "kernels" / "rocm" / "ffi" / "gdn_prepare_s512.hip"
)
_BUILD_SCRIPT = (
    _REPO
    / "skyrl"
    / "tx"
    / "kernels"
    / "rocm"
    / "ffi"
    / "build_gdn_prepare_s512_gfx1100.sh"
)


@pytest.fixture(autouse=True)
def _reset_registration(monkeypatch):
    monkeypatch.setattr(prepare, "_registration_state", None)
    monkeypatch.setattr(prepare._sealed_loader, "_library_lifetime_handles", [])
    retained_fds: list[int] = []
    monkeypatch.setattr(
        prepare._sealed_loader,
        "_library_lifetime_fds",
        retained_fds,
    )
    yield
    for descriptor in retained_fds:
        try:
            os.close(descriptor)
        except OSError:
            pass


class _FakeSymbol:
    restype: Any = None
    argtypes: Any = None


class _FakeLibrary:
    def __init__(self, *, with_symbol: bool = True) -> None:
        self.symbol = _FakeSymbol()
        if with_symbol:
            setattr(self, prepare.GDN_PREPARE_S512_TARGET, self.symbol)


class _FakeFfi:
    def __init__(self) -> None:
        self.capsule_calls: list[Any] = []
        self.registration_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.ffi_call_builds: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.invocations: list[tuple[Any, ...]] = []

    def pycapsule(self, symbol):
        self.capsule_calls.append(symbol)
        return ("capsule", symbol)

    def register_ffi_target(self, *args, **kwargs):
        self.registration_calls.append((args, kwargs))

    def ffi_call(self, *args, **kwargs):
        self.ffi_call_builds.append((args, kwargs))

        def invoke(*values):
            self.invocations.append(values)
            return ("prepared-u", "prepared-w", "gamma")

        return invoke


class _FakeJax:
    def __init__(self) -> None:
        self.ffi = _FakeFfi()
        self.shape_specs: list[tuple[tuple[int, ...], Any]] = []

    def ShapeDtypeStruct(self, shape, dtype):
        spec = SimpleNamespace(shape=tuple(shape), dtype=dtype)
        self.shape_specs.append((tuple(shape), dtype))
        return spec


def _library_file(tmp_path: Path) -> Path:
    path = tmp_path / "libskyrl_gdn_prepare_s512_gfx1100.so"
    path.write_bytes(b"mock exact prepare shared object")
    path.chmod(0o600)
    return path.resolve()


def _library_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_dependencies(monkeypatch):
    jax = _FakeJax()
    jnp = SimpleNamespace(float32="f32")
    library = _FakeLibrary()
    snapshots: list[tuple[Path, str]] = []
    loads: list[Any] = []
    snapshot = SimpleNamespace(
        sha256=None,
        size_bytes=31,
        mode=0o600,
        seals=15,
        fd=-1,
    )

    monkeypatch.setattr(prepare, "_import_jax", lambda: (jax, jnp))

    def seal(path, digest):
        snapshots.append((path, digest))
        snapshot.sha256 = digest
        prepare._sealed_loader._library_lifetime_fds.append(snapshot.fd)
        return snapshot

    def load(item):
        loads.append(item)
        return library

    monkeypatch.setattr(prepare._sealed_loader, "_snapshot_library", seal)
    monkeypatch.setattr(prepare._sealed_loader, "_load_cdll", load)
    return jax, jnp, library, snapshot, snapshots, loads


def _abstract_inputs(dtype: Any = "f32") -> tuple[SimpleNamespace, ...]:
    return (
        SimpleNamespace(shape=prepare.GDN_PREPARE_S512_KEY_SHAPE, dtype=dtype),
        SimpleNamespace(shape=prepare.GDN_PREPARE_S512_VALUE_SHAPE, dtype=dtype),
        SimpleNamespace(shape=prepare.GDN_PREPARE_S512_GATE_SHAPE, dtype=dtype),
        SimpleNamespace(shape=prepare.GDN_PREPARE_S512_GATE_SHAPE, dtype=dtype),
    )


def test_registration_and_operation_are_default_off_without_jax_or_loading(
    monkeypatch,
):
    monkeypatch.setattr(
        prepare,
        "_import_jax",
        lambda: (_ for _ in ()).throw(AssertionError("JAX must remain lazy")),
    )
    monkeypatch.setattr(
        prepare._sealed_loader,
        "_snapshot_library",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not snapshot")),
    )

    with pytest.raises(RuntimeError, match="registration is disabled by default"):
        prepare.register_gdn_prepare_s512()
    with pytest.raises(RuntimeError, match="disabled by default"):
        prepare.gdn_prepare_s512(*_abstract_inputs())
    with pytest.raises(ValueError, match="invalid while.*disabled"):
        prepare.gdn_prepare_s512(
            *_abstract_inputs(),
            library_path="/private/not-loaded.so",
        )


@pytest.mark.parametrize("enabled", [1, 0, None, "true"])
def test_opt_in_requires_an_exact_bool(enabled):
    with pytest.raises(TypeError, match="exact bool"):
        prepare.register_gdn_prepare_s512(enabled=enabled)
    with pytest.raises(TypeError, match="exact bool"):
        prepare.gdn_prepare_s512(*_abstract_inputs(), enabled=enabled)


@pytest.mark.parametrize(
    "digest",
    [None, "", "0" * 63, "0" * 65, "G" * 64, b"0" * 64],
)
def test_enabled_paths_require_exact_lowercase_sha_before_jax(
    monkeypatch,
    tmp_path,
    digest,
):
    path = _library_file(tmp_path)
    monkeypatch.setattr(
        prepare,
        "_import_jax",
        lambda: (_ for _ in ()).throw(AssertionError("JAX must remain lazy")),
    )
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        prepare.register_gdn_prepare_s512(
            path,
            library_sha256=digest,
            enabled=True,
        )


def test_library_path_must_be_exact_absolute_canonical_private_file(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        prepare.register_gdn_prepare_s512(
            "libskyrl_gdn_prepare_s512_gfx1100.so",
            library_sha256="0" * 64,
            enabled=True,
        )

    wrong = tmp_path / "wrong.so"
    wrong.write_bytes(b"wrong")
    wrong.chmod(0o600)
    with pytest.raises(ValueError, match="exact name"):
        prepare.register_gdn_prepare_s512(
            wrong.resolve(),
            library_sha256=_library_sha256(wrong),
            enabled=True,
        )

    library = _library_file(tmp_path)
    library.chmod(0o620)
    with pytest.raises(ValueError, match="group- or world-writable"):
        prepare.register_gdn_prepare_s512(
            library,
            library_sha256=_library_sha256(library),
            enabled=True,
        )


def test_symlink_library_is_rejected_before_snapshot(tmp_path):
    target = tmp_path / "private-target.so"
    target.write_bytes(b"target")
    target.chmod(0o600)
    link = tmp_path / "libskyrl_gdn_prepare_s512_gfx1100.so"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        prepare.register_gdn_prepare_s512(
            link.absolute(),
            library_sha256=hashlib.sha256(b"target").hexdigest(),
            enabled=True,
        )


def test_registration_reuses_sealed_loader_and_retains_exact_identity(
    monkeypatch,
    tmp_path,
):
    path = _library_file(tmp_path)
    digest = _library_sha256(path)
    jax, _jnp, library, snapshot, snapshots, loads = _fake_dependencies(monkeypatch)

    registration = prepare.register_gdn_prepare_s512(
        path,
        library_sha256=digest,
        enabled=True,
    )

    assert snapshots == [(path, digest)]
    assert loads == [snapshot]
    assert prepare._sealed_loader._library_lifetime_handles == [library]
    assert registration.library_path == path
    assert registration.library_sha256 == digest
    assert registration.snapshot_sha256 == digest
    assert registration.snapshot_mode == 0o600
    assert registration.snapshot_seals == 15
    assert registration.sealed_snapshot is True
    assert registration.snapshot_fd_retained is True
    assert jax.ffi.capsule_calls == [library.symbol]
    assert jax.ffi.registration_calls == [
        (
            (
                prepare.GDN_PREPARE_S512_TARGET,
                ("capsule", library.symbol),
            ),
            {"platform": "ROCM", "api_version": 1},
        )
    ]
    assert library.symbol.restype is not None
    assert library.symbol.argtypes is not None


def test_registration_is_idempotent_only_for_exact_path_and_digest(
    monkeypatch,
    tmp_path,
):
    path = _library_file(tmp_path)
    digest = _library_sha256(path)
    jax, _jnp, _library, _snapshot, snapshots, _loads = _fake_dependencies(monkeypatch)

    first = prepare.register_gdn_prepare_s512(
        path,
        library_sha256=digest,
        enabled=True,
    )
    second = prepare.register_gdn_prepare_s512(
        path,
        library_sha256=digest,
        enabled=True,
    )
    assert first is second
    assert len(snapshots) == 1
    assert len(jax.ffi.registration_calls) == 1

    with pytest.raises(RuntimeError, match="different library identity"):
        prepare.register_gdn_prepare_s512(
            path,
            library_sha256="0" * 64,
            enabled=True,
        )


def test_missing_exact_symbol_retains_loaded_library_without_registration(
    monkeypatch,
    tmp_path,
):
    path = _library_file(tmp_path)
    jax, _jnp, _library, _snapshot, _snapshots, _loads = _fake_dependencies(monkeypatch)
    missing = _FakeLibrary(with_symbol=False)
    monkeypatch.setattr(prepare._sealed_loader, "_load_cdll", lambda _item: missing)

    with pytest.raises(RuntimeError, match="missing its exact handler symbol"):
        prepare.register_gdn_prepare_s512(
            path,
            library_sha256=_library_sha256(path),
            enabled=True,
        )
    assert prepare._sealed_loader._library_lifetime_handles == [missing]
    assert jax.ffi.registration_calls == []
    assert prepare._registration_state is None


@pytest.mark.parametrize(
    ("index", "shape"),
    [
        (0, (1, 511, 16, 128)),
        (0, (1, 512, 32, 128)),
        (1, (1, 512, 16, 128)),
        (1, (1, 512, 32, 64)),
        (2, (1, 512, 16)),
        (3, (512, 32)),
    ],
)
def test_every_input_shape_is_exactly_gated_before_snapshot(
    monkeypatch,
    tmp_path,
    index,
    shape,
):
    path = _library_file(tmp_path)
    _jax, _jnp, _library, _snapshot, snapshots, _loads = _fake_dependencies(monkeypatch)
    inputs = list(_abstract_inputs())
    inputs[index] = SimpleNamespace(shape=shape, dtype="f32")

    with pytest.raises(ValueError, match="shape must be exactly"):
        prepare.gdn_prepare_s512(
            *inputs,
            enabled=True,
            library_path=path,
            library_sha256=_library_sha256(path),
        )
    assert snapshots == []


@pytest.mark.parametrize("index", range(4))
def test_every_input_requires_exact_fp32_before_snapshot(
    monkeypatch,
    tmp_path,
    index,
):
    path = _library_file(tmp_path)
    _jax, _jnp, _library, _snapshot, snapshots, _loads = _fake_dependencies(monkeypatch)
    inputs = list(_abstract_inputs())
    inputs[index] = SimpleNamespace(shape=inputs[index].shape, dtype="bf16")

    with pytest.raises(TypeError, match="dtype must be exactly float32"):
        prepare.gdn_prepare_s512(
            *inputs,
            enabled=True,
            library_path=path,
            library_sha256=_library_sha256(path),
        )
    assert snapshots == []


def test_enabled_call_builds_exact_three_result_typed_ffi_contract(
    monkeypatch,
    tmp_path,
):
    path = _library_file(tmp_path)
    digest = _library_sha256(path)
    jax, _jnp, _library, _snapshot, snapshots, _loads = _fake_dependencies(monkeypatch)
    inputs = _abstract_inputs()

    result = prepare.gdn_prepare_s512(
        *inputs,
        enabled=True,
        library_path=path,
        library_sha256=digest,
    )

    assert result == ("prepared-u", "prepared-w", "gamma")
    assert snapshots == [(path, digest)]
    assert jax.shape_specs == [
        (prepare.GDN_PREPARE_S512_VALUE_SHAPE, "f32"),
        (prepare.GDN_PREPARE_S512_VALUE_SHAPE, "f32"),
        (prepare.GDN_PREPARE_S512_GATE_SHAPE, "f32"),
    ]
    args, kwargs = jax.ffi.ffi_call_builds[0]
    assert args[0] == prepare.GDN_PREPARE_S512_TARGET
    assert tuple(spec.shape for spec in args[1]) == (
        prepare.GDN_PREPARE_S512_VALUE_SHAPE,
        prepare.GDN_PREPARE_S512_VALUE_SHAPE,
        prepare.GDN_PREPARE_S512_GATE_SHAPE,
    )
    assert kwargs == {
        "has_side_effect": False,
        "vmap_method": "sequential",
        "input_layouts": (
            (0, 1, 2, 3),
            (0, 1, 2, 3),
            (0, 1, 2),
            (0, 1, 2),
        ),
        "output_layouts": ((0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2)),
        "input_output_aliases": None,
        "custom_call_api_version": 4,
    }
    assert jax.ffi.invocations == [inputs]


def test_wrapper_has_no_eager_jax_or_insecure_pathname_loader():
    source = _PYTHON_SOURCE.read_text(encoding="utf-8")
    module = ast.parse(source)
    imports = [
        node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported_roots = {
        alias.name.partition(".")[0] for node in imports for alias in node.names
    }

    assert "jax" not in imported_roots
    assert "jaxlib" not in imported_roots
    assert "_sealed_loader._snapshot_library" in source
    assert "_sealed_loader._load_cdll(snapshot)" in source
    assert "_sealed_loader._library_lifetime_handles.append(library)" in source
    assert "memfd_create" not in source
    assert "ctypes.CDLL" not in source
    assert "JAX_PLATFORMS" not in source
    assert "HIP_VISIBLE_DEVICES" not in source


def test_hip_source_encodes_exact_geometry_pairing_and_wy_equations():
    source = _HIP_SOURCE.read_text(encoding="utf-8")

    for marker in (
        "kTokens = 512",
        "kChunks = 8",
        "kChunk = 64",
        "kKeyHeads = 16",
        "kValueHeads = 32",
        "kHeadsPerKey = 2",
        "kHeadDimension = 128",
        "kThreads = 128",
        "kWorkgroups = kChunks * kKeyHeads",
        "kLdsBytes == 49'664",
        "value_head = kHeadsPerKey * key_head + head_in_pair",
        "beta_chunk[row] * key_gram[row][column]",
        "expf(prefix[row] - prefix[column])",
        "beta_chunk[row] * expf(prefix[row])",
        "solution[row][local_thread] = rhs - correction",
    ):
        assert marker in source
    assert "column < row" in source
    assert "previous < row" in source
    assert "chunk_index = blockIdx.x / kKeyHeads" in source
    assert "key_head = blockIdx.x % kKeyHeads" in source
    assert source.count("hipLaunchKernelGGL") == 1


def test_hip_handler_gates_all_shapes_bytes_and_distinct_buffers():
    source = _HIP_SOURCE.read_text(encoding="utf-8")

    assert "BufferR4<xla::ffi::F32> key" in source
    assert "BufferR4<xla::ffi::F32> value" in source
    assert "BufferR3<xla::ffi::F32> g" in source
    assert "BufferR3<xla::ffi::F32> beta" in source
    assert source.count(".Arg<xla::ffi::") == 4
    assert source.count(".Ret<xla::ffi::") == 3
    assert "key.size_bytes() != kKeyBytes" in source
    assert "value.size_bytes() != kValueBytes" in source
    assert "g.size_bytes() != kGateBytes" in source
    assert "beta.size_bytes() != kGateBytes" in source
    assert "prepared_u->size_bytes() != kValueBytes" in source
    assert "prepared_w->size_bytes() != kValueBytes" in source
    assert "gamma->size_bytes() != kGateBytes" in source
    assert "std::array<const void*, 7>" in source
    assert "AreDistinctNonNull(buffers)" in source
    assert prepare.GDN_PREPARE_S512_TARGET in source
    assert "PlatformStream<hipStream_t>" in source


def test_hip_source_has_one_async_launch_no_sync_allocator_or_fast_math():
    source = _HIP_SOURCE.read_text(encoding="utf-8")

    assert source.count("hipGetLastError") == 2
    assert "hipDeviceSynchronize" not in source
    assert "hipStreamSynchronize" not in source
    assert "hipMalloc" not in source
    assert "hipFree" not in source
    assert "malloc(" not in source
    assert "operator new" not in source
    assert "new float" not in source
    assert "__expf" not in source
    assert "use_fast_math" not in source


def test_build_rule_is_exact_gfx1100_external_artifact_without_fast_math():
    source = _BUILD_SCRIPT.read_text(encoding="utf-8")

    assert 'EXPECTED_ARCH="gfx1100"' in source
    assert 'EXPECTED_BASENAME="libskyrl_gdn_prepare_s512_gfx1100.so"' in source
    assert 'EXPECTED_SYMBOL="skyrl_gdn_prepare_s512_f32_v1"' in source
    assert '"--offload-arch=$EXPECTED_ARCH"' in source
    assert "-fPIC -shared" in source
    assert "-fno-fast-math" in source
    assert "-ffast-math" not in source
    assert "output must be outside the source repository" in source
    assert "/dev/kfd" not in source
    assert "rocminfo" not in source
    assert "hipDeviceSynchronize" not in source
    assert stat.S_IMODE(os.stat(_BUILD_SCRIPT).st_mode) & stat.S_IXUSR


def test_oracle_dense_solve_matches_direct_per_pair_equations():
    key = np.asarray(
        [[[[0.4, -0.2], [0.1, 0.3], [-0.5, 0.25]]]],
        dtype=np.float32,
    )
    value = np.asarray(
        [
            [
                [
                    [[0.2, 0.1], [-0.3, 0.4], [0.6, -0.2]],
                    [[-0.1, 0.5], [0.7, -0.4], [0.2, 0.3]],
                ]
            ]
        ],
        dtype=np.float32,
    )
    g = np.asarray([[[[-0.02, -0.03, -0.01], [-0.04, 0.0, -0.02]]]], dtype=np.float32)
    beta = np.asarray(
        [[[[0.2, 0.3, 0.4], [0.5, 0.1, 0.25]]]],
        dtype=np.float32,
    )

    actual_u, actual_w, actual_gamma = oracle._dense_prepare_chunks_numpy(
        key,
        value,
        g,
        beta,
    )
    gram = key[0, 0] @ key[0, 0].T
    for pair in range(2):
        prefix = np.cumsum(g[0, 0, pair], dtype=np.float32)
        gamma = np.exp(prefix).astype(np.float32)
        decay = np.tril(np.exp(prefix[:, None] - prefix[None, :]))
        lower = np.tril(beta[0, 0, pair, :, None] * gram * decay, k=-1)
        rhs = np.concatenate(
            (
                beta[0, 0, pair, :, None] * value[0, 0, pair],
                beta[0, 0, pair, :, None] * gamma[:, None] * key[0, 0],
            ),
            axis=-1,
        )
        expected = np.linalg.solve(
            np.eye(3, dtype=np.float32) + lower,
            rhs,
        )
        np.testing.assert_allclose(actual_u[0, 0, pair], expected[:, :2])
        np.testing.assert_allclose(actual_w[0, 0, pair], expected[:, 2:])
        np.testing.assert_allclose(actual_gamma[0, 0, pair], gamma)


def test_exact_oracle_indexing_and_head_pairs_with_identity_solve(monkeypatch):
    key = np.zeros(prepare.GDN_PREPARE_S512_KEY_SHAPE, dtype=np.float32)
    value = np.arange(
        np.prod(prepare.GDN_PREPARE_S512_VALUE_SHAPE),
        dtype=np.float32,
    ).reshape(prepare.GDN_PREPARE_S512_VALUE_SHAPE)
    value %= np.float32(257.0)
    g = np.zeros(prepare.GDN_PREPARE_S512_GATE_SHAPE, dtype=np.float32)
    beta = np.ones(prepare.GDN_PREPARE_S512_GATE_SHAPE, dtype=np.float32)
    solve_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def identity_solve(matrix, rhs):
        solve_shapes.append((matrix.shape, rhs.shape))
        np.testing.assert_array_equal(
            matrix,
            np.broadcast_to(np.eye(64, dtype=np.float32), matrix.shape),
        )
        return rhs

    monkeypatch.setattr(oracle.np.linalg, "solve", identity_solve)
    prepared_u, prepared_w, gamma = oracle.gdn_prepare_s512_numpy(
        key,
        value,
        g,
        beta,
    )

    assert solve_shapes == [((8, 16, 2, 64, 64), (8, 16, 2, 64, 256))]
    np.testing.assert_array_equal(prepared_u, value)
    np.testing.assert_array_equal(prepared_w, np.zeros_like(prepared_w))
    np.testing.assert_array_equal(gamma, np.ones_like(gamma))


def test_dense_oracle_transformed_interior_mask_has_zero_uw_and_decay_plateau():
    key = np.asarray(
        [[[[0.2, -0.1], [0.3, 0.4], [0.0, 0.0], [-0.2, 0.5]]]],
        dtype=np.float32,
    )
    value = np.asarray(
        [
            [
                [
                    [[0.1, 0.2], [0.3, -0.1], [0.0, 0.0], [0.2, 0.4]],
                    [[-0.2, 0.1], [0.5, 0.2], [0.0, 0.0], [-0.3, 0.6]],
                ]
            ]
        ],
        dtype=np.float32,
    )
    g = np.asarray(
        [[[[-0.02, -0.03, 0.0, -0.01], [-0.01, -0.04, 0.0, -0.02]]]],
        dtype=np.float32,
    )
    beta = np.asarray(
        [[[[0.2, 0.3, 0.0, 0.4], [0.1, 0.25, 0.0, 0.35]]]],
        dtype=np.float32,
    )

    prepared_u, prepared_w, gamma = oracle._dense_prepare_chunks_numpy(
        key,
        value,
        g,
        beta,
    )

    np.testing.assert_array_equal(
        prepared_u[..., 2, :],
        np.zeros_like(prepared_u[..., 2, :]),
    )
    np.testing.assert_array_equal(
        prepared_w[..., 2, :],
        np.zeros_like(prepared_w[..., 2, :]),
    )
    np.testing.assert_array_equal(gamma[..., 2], gamma[..., 1])


def test_transformed_mask_contract_zeros_only_declared_ffi_inputs():
    key = np.ones(prepare.GDN_PREPARE_S512_KEY_SHAPE, dtype=np.float32)
    value = np.ones(prepare.GDN_PREPARE_S512_VALUE_SHAPE, dtype=np.float32)
    g = -np.ones(prepare.GDN_PREPARE_S512_GATE_SHAPE, dtype=np.float32)
    beta = np.ones(prepare.GDN_PREPARE_S512_GATE_SHAPE, dtype=np.float32)
    mask = np.ones((1, 512), dtype=np.bool_)
    mask[:, (0, 63, 64, 255, 511)] = False

    transformed = oracle.transform_gdn_prepare_s512_inputs_numpy(
        key,
        value,
        g,
        beta,
        mask,
    )
    for item, expected in zip(
        transformed,
        (1.0, 1.0, -1.0, 1.0),
        strict=True,
    ):
        assert item.dtype == np.float32
        np.testing.assert_array_equal(
            item[:, ~mask[0]],
            np.zeros_like(item[:, ~mask[0]]),
        )
        np.testing.assert_array_equal(
            item[:, mask[0]],
            np.full_like(item[:, mask[0]], expected),
        )
    assert all(
        item is not original
        for item, original in zip(
            transformed,
            (key, value, g, beta),
            strict=True,
        )
    )


@pytest.mark.parametrize(
    ("name", "array", "match"),
    [
        ("key", np.zeros((1, 511, 16, 128), dtype=np.float32), "shape"),
        ("key", np.zeros((1, 512, 16, 128), dtype=np.float16), "dtype"),
    ],
)
def test_oracle_rejects_non_exact_public_inputs(name, array, match):
    inputs = {
        "key": np.zeros(prepare.GDN_PREPARE_S512_KEY_SHAPE, dtype=np.float32),
        "value": np.zeros(prepare.GDN_PREPARE_S512_VALUE_SHAPE, dtype=np.float32),
        "g": np.zeros(prepare.GDN_PREPARE_S512_GATE_SHAPE, dtype=np.float32),
        "beta": np.zeros(prepare.GDN_PREPARE_S512_GATE_SHAPE, dtype=np.float32),
    }
    inputs[name] = array
    with pytest.raises((TypeError, ValueError), match=match):
        oracle.gdn_prepare_s512_numpy(**inputs)


def test_oracle_is_numpy_only_dense_solve_not_runtime_fallback():
    source = _ORACLE_SOURCE.read_text(encoding="utf-8")
    module = ast.parse(source)
    imports = [
        node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported_roots = {
        alias.name.partition(".")[0] for node in imports for alias in node.names
    }

    assert "jax" not in imported_roots
    assert "jaxlib" not in imported_roots
    assert "np.linalg.solve" in source
    assert "np.tril" in source
    assert "np.cumsum" in source
    assert "gdn_prepare_ffi" not in source


def test_no_prepare_binary_is_committed_next_to_sources():
    unexpected = [
        path.name
        for path in _HIP_SOURCE.parent.iterdir()
        if path.is_file() and path.suffix not in {".hip", ".sh", ".md"}
    ]
    assert unexpected == []
