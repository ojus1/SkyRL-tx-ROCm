from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from skyrl.tx.kernels.rocm import gdn_reverse_ffi as reverse_ffi
from skyrl.tx.kernels.rocm import gdn_superblock_reverse_oracle as oracle

_REPO = Path(__file__).parents[2]
_PYTHON_SOURCE = _REPO / "skyrl" / "tx" / "kernels" / "rocm" / "gdn_reverse_ffi.py"


@pytest.fixture(autouse=True)
def _reset_registration(monkeypatch):
    monkeypatch.setattr(reverse_ffi, "_registration_state", None)
    monkeypatch.setattr(reverse_ffi._sealed_loader, "_library_lifetime_handles", [])
    retained_fds: list[int] = []
    monkeypatch.setattr(
        reverse_ffi._sealed_loader,
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
            setattr(self, reverse_ffi.GDN_REVERSE_S512_TARGET, self.symbol)


class _FakeFfi:
    def __init__(self) -> None:
        self.capsule_calls: list[Any] = []
        self.registration_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.registration_states_during_calls: list[Any] = []
        self.ffi_call_builds: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.invocations: list[tuple[Any, ...]] = []

    def pycapsule(self, symbol):
        self.capsule_calls.append(symbol)
        return ("capsule", symbol)

    def register_ffi_target(self, *args, **kwargs):
        self.registration_calls.append((args, kwargs))
        self.registration_states_during_calls.append(reverse_ffi._registration_state)

    def ffi_call(self, *args, **kwargs):
        self.ffi_call_builds.append((args, kwargs))

        def invoke(*values):
            self.invocations.append(values)
            return (
                "query-gradient",
                "key-gradient",
                "value-gradient",
                "g-gradient",
                "beta-gradient",
                "initial-state-gradient",
                "hidden-u8-scratch",
            )

        return invoke


class _FakeJax:
    def __init__(self) -> None:
        self.ffi = _FakeFfi()
        self.shape_specs: list[tuple[tuple[int, ...], Any]] = []

    def ShapeDtypeStruct(self, shape, dtype):
        spec = SimpleNamespace(shape=tuple(shape), dtype=dtype)
        self.shape_specs.append((tuple(shape), dtype))
        return spec


class _ExposedHostValue:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: Any,
        *,
        itemsize: int,
        address: int,
        nbytes: int | None = None,
        strides: tuple[int, ...] | None = None,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.nbytes = nbytes if nbytes is not None else _shape_bytes(shape, itemsize)
        self.strides = (
            reverse_ffi._expected_c_strides(shape, itemsize)
            if strides is None
            else strides
        )
        self.__array_interface__ = {
            "shape": shape,
            "typestr": "|u1",
            "data": (address, False),
            "version": 3,
        }


def _shape_bytes(shape: tuple[int, ...], itemsize: int) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * itemsize


def _library_file(
    tmp_path: Path, contents: bytes = b"mock reverse shared object"
) -> Path:
    path = tmp_path / "libskyrl_gdn_reverse_s512_gfx1100.so"
    path.write_bytes(contents)
    path.chmod(0o600)
    return path.resolve()


def _library_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _abstract_inputs() -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(shape=shape, dtype=dtype)
        for shape, dtype in (
            (reverse_ffi.GDN_REVERSE_S512_QUERY_SHAPE, "f32"),
            (reverse_ffi.GDN_REVERSE_S512_QUERY_SHAPE, "f32"),
            (reverse_ffi.GDN_REVERSE_S512_VALUE_SHAPE, "f32"),
            (reverse_ffi.GDN_REVERSE_S512_GATE_SHAPE, "f32"),
            (reverse_ffi.GDN_REVERSE_S512_GATE_SHAPE, "f32"),
            (reverse_ffi.GDN_REVERSE_S512_STATE_SHAPE, "f32"),
            (reverse_ffi.GDN_REVERSE_S512_OUTPUT_COTANGENT_SHAPE, "bf16"),
            (reverse_ffi.GDN_REVERSE_S512_STATE_SHAPE, "f32"),
        )
    )


def _fake_dependencies(monkeypatch):
    jax = _FakeJax()
    jnp = SimpleNamespace(float32="f32", bfloat16="bf16", uint8="u8")
    library = _FakeLibrary()
    snapshots: list[tuple[Path, str]] = []
    loads: list[Any] = []
    snapshot = SimpleNamespace(
        sha256=None,
        size_bytes=64,
        mode=0o600,
        seals=15,
        fd=-1,
    )
    monkeypatch.setattr(reverse_ffi, "_import_jax", lambda: (jax, jnp))

    def seal(path, digest):
        snapshots.append((path, digest))
        snapshot.sha256 = digest
        reverse_ffi._sealed_loader._library_lifetime_fds.append(snapshot.fd)
        return snapshot

    def load(item):
        loads.append(item)
        return library

    monkeypatch.setattr(reverse_ffi._sealed_loader, "_snapshot_library", seal)
    monkeypatch.setattr(reverse_ffi._sealed_loader, "_load_cdll", load)
    return jax, jnp, library, snapshot, snapshots, loads


def test_declared_boundary_shapes_and_payload_arithmetic_match_composed_oracle():
    assert reverse_ffi.GDN_REVERSE_S512_QUERY_SHAPE == (
        oracle.GDN_SUPERBLOCK_S512_QUERY_SHAPE
    )
    assert reverse_ffi.GDN_REVERSE_S512_VALUE_SHAPE == (
        oracle.GDN_SUPERBLOCK_S512_VALUE_SHAPE
    )
    assert reverse_ffi.GDN_REVERSE_S512_GATE_SHAPE == (
        oracle.GDN_SUPERBLOCK_S512_GATE_SHAPE
    )
    assert reverse_ffi.GDN_REVERSE_S512_STATE_SHAPE == (
        oracle.GDN_SUPERBLOCK_S512_STATE_SHAPE
    )
    assert reverse_ffi.GDN_REVERSE_S512_OUTPUT_COTANGENT_SHAPE == (
        oracle.GDN_SUPERBLOCK_S512_OUTPUT_SHAPE
    )
    assert reverse_ffi.GDN_REVERSE_S512_INPUT_BYTES == (
        oracle.GDN_SUPERBLOCK_S512_REVERSE_INPUT_BYTES
    )
    assert reverse_ffi.GDN_REVERSE_S512_GRADIENT_BYTES == (
        oracle.GDN_SUPERBLOCK_S512_GRADIENT_BYTES
    )
    assert reverse_ffi.GDN_REVERSE_S512_SCRATCH_SHAPE == (34_144_256,)
    assert reverse_ffi.GDN_REVERSE_S512_LOW_LEVEL_OUTPUT_BYTES == (
        reverse_ffi.GDN_REVERSE_S512_GRADIENT_BYTES
        + reverse_ffi.GDN_REVERSE_S512_SCRATCH_BYTES
    )
    assert reverse_ffi.GDN_REVERSE_S512_TOTAL_BUFFER_BYTES == (
        reverse_ffi.GDN_REVERSE_S512_INPUT_BYTES
        + reverse_ffi.GDN_REVERSE_S512_LOW_LEVEL_OUTPUT_BYTES
    )
    assert reverse_ffi.GDN_REVERSE_S512_TOTAL_BUFFER_BYTES == 78_446_592
    assert reverse_ffi.GDN_REVERSE_S512_INPUT_COUNT == 8
    assert reverse_ffi.GDN_REVERSE_S512_PUBLIC_GRADIENT_COUNT == 6
    assert reverse_ffi.GDN_REVERSE_S512_LOW_LEVEL_RESULT_COUNT == 7
    assert reverse_ffi.GDN_REVERSE_S512_TUPLE_POINTER_TABLE_BYTES == 7 * 8
    assert reverse_ffi.GDN_REVERSE_S512_REQUIRED_SCRATCH_BASE_ALIGNMENT_BYTES == 16


def test_hidden_scratch_region_specs_are_typed_contiguous_and_unambiguous():
    expected = (
        (
            "prepared_u",
            0,
            8_388_608,
            "float32",
            (1, 512, 32, 128),
            (0, 1, 2, 3),
        ),
        (
            "prepared_w",
            8_388_608,
            8_388_608,
            "float32",
            (1, 512, 32, 128),
            (0, 1, 2, 3),
        ),
        (
            "gamma",
            16_777_216,
            65_536,
            "float32",
            (1, 512, 32),
            (0, 1, 2),
        ),
        (
            "replay_states_s1_s7",
            16_842_752,
            14_680_064,
            "float32",
            (7, 32, 128, 128),
            (0, 1, 2, 3),
        ),
        (
            "odd_value_head_query_gradient_chunk",
            31_522_816,
            524_288,
            "float32",
            (16, 64, 128),
            (0, 1, 2),
        ),
        (
            "odd_value_head_key_gradient_chunk",
            32_047_104,
            524_288,
            "float32",
            (16, 64, 128),
            (0, 1, 2),
        ),
        (
            "prepared_w_gradient_chunk",
            32_571_392,
            1_048_576,
            "float32",
            (32, 64, 128),
            (0, 1, 2),
        ),
        (
            "attention_gradient_chunk",
            33_619_968,
            524_288,
            "float32",
            (32, 64, 64),
            (0, 1, 2),
        ),
    )
    observed = tuple(
        (
            region.name,
            region.offset_bytes,
            region.nbytes,
            region.dtype,
            region.shape,
            region.row_major_physical_order,
        )
        for region in reverse_ffi.GDN_REVERSE_S512_SCRATCH_LAYOUT
    )
    assert observed == expected
    cursor = 0
    for region in reverse_ffi.GDN_REVERSE_S512_SCRATCH_LAYOUT:
        assert isinstance(region, reverse_ffi.GdnReverseS512ScratchRegion)
        assert region.offset_bytes == cursor
        assert region.offset_bytes % 16 == 0
        assert region.nbytes > 0 and region.nbytes % 16 == 0
        assert region.dtype == "float32"
        assert region.row_major_physical_order == tuple(range(len(region.shape)))
        assert _shape_bytes(region.shape, 4) == region.nbytes
        assert region.lifetime
        assert region.use_and_reuse
        cursor = region.offset_bytes + region.nbytes
    assert cursor == reverse_ffi.GDN_REVERSE_S512_SCRATCH_BYTES

    regions = {
        region.name: region for region in reverse_ffi.GDN_REVERSE_S512_SCRATCH_LAYOUT
    }
    for name in (
        "odd_value_head_query_gradient_chunk",
        "odd_value_head_key_gradient_chunk",
        "prepared_w_gradient_chunk",
        "attention_gradient_chunk",
    ):
        assert "One reverse chunk c" in regions[name].lifetime
        assert "Reuse for c-1 requires same-stream ordering" in (
            regions[name].use_and_reuse
        )
    assert "overwrites it in place" in (
        regions["prepared_w_gradient_chunk"].use_and_reuse
    )
    assert "writes dA" in regions["attention_gradient_chunk"].use_and_reuse


def test_source_marks_memory_alignment_and_numerics_as_unverified_requirements():
    source = _PYTHON_SOURCE.read_text(encoding="utf-8")
    normalized_source = " ".join(source.split())

    for marker in (
        "does not prove compiler allocation",
        "actual peak-memory accounting",
        "address stability",
        "base alignment",
        "require compiled-HLO",
        "must validate that the U8 base is aligned to at least 16 bytes",
        "ABI requirements, not measurements",
        "assumes seven 64-bit result",
        "does not implement or prove gradient arithmetic",
        "required to widen dO before reverse arithmetic",
    ):
        assert marker in normalized_source


def test_registration_and_operation_are_default_off_without_jax_or_loading(
    monkeypatch,
):
    monkeypatch.setattr(
        reverse_ffi,
        "_import_jax",
        lambda: (_ for _ in ()).throw(AssertionError("JAX must remain lazy")),
    )
    monkeypatch.setattr(
        reverse_ffi._sealed_loader,
        "_snapshot_library",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not snapshot")),
    )

    with pytest.raises(RuntimeError, match="registration is disabled by default"):
        reverse_ffi.register_gdn_reverse_s512()
    with pytest.raises(RuntimeError, match="disabled by default"):
        reverse_ffi.gdn_reverse_s512(*_abstract_inputs())
    with pytest.raises(ValueError, match="invalid while.*disabled"):
        reverse_ffi.gdn_reverse_s512(
            *_abstract_inputs(),
            library_path="/private/not-loaded.so",
        )


@pytest.mark.parametrize("enabled", [1, 0, None, "true"])
def test_opt_in_requires_an_exact_bool(enabled):
    with pytest.raises(TypeError, match="exact bool"):
        reverse_ffi.register_gdn_reverse_s512(enabled=enabled)
    with pytest.raises(TypeError, match="exact bool"):
        reverse_ffi.gdn_reverse_s512(*_abstract_inputs(), enabled=enabled)


@pytest.mark.parametrize(
    "digest",
    [None, "", "0" * 63, "0" * 65, "G" * 64, b"0" * 64],
)
def test_digest_is_exact_and_checked_before_lazy_jax(monkeypatch, tmp_path, digest):
    path = _library_file(tmp_path)
    monkeypatch.setattr(
        reverse_ffi,
        "_import_jax",
        lambda: (_ for _ in ()).throw(AssertionError("JAX must remain lazy")),
    )
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        reverse_ffi.register_gdn_reverse_s512(
            path,
            library_sha256=digest,
            enabled=True,
        )


def test_library_path_is_canonical_private_exact_basename_and_not_a_symlink(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        reverse_ffi.register_gdn_reverse_s512(
            "libskyrl_gdn_reverse_s512_gfx1100.so",
            library_sha256="0" * 64,
            enabled=True,
        )

    wrong = tmp_path / "wrong.so"
    wrong.write_bytes(b"wrong")
    wrong.chmod(0o600)
    with pytest.raises(ValueError, match="exact name"):
        reverse_ffi.register_gdn_reverse_s512(
            wrong.resolve(),
            library_sha256=_library_sha256(wrong),
            enabled=True,
        )

    library = _library_file(tmp_path)
    library.chmod(0o620)
    with pytest.raises(ValueError, match="group- or world-writable"):
        reverse_ffi.register_gdn_reverse_s512(
            library,
            library_sha256=_library_sha256(library),
            enabled=True,
        )

    target = tmp_path / "private-target.so"
    target.write_bytes(b"target")
    target.chmod(0o600)
    library.unlink()
    library.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        reverse_ffi.register_gdn_reverse_s512(
            library.absolute(),
            library_sha256=hashlib.sha256(b"target").hexdigest(),
            enabled=True,
        )


def test_registration_uses_one_sealed_snapshot_and_rejects_conflicting_identity(
    monkeypatch,
    tmp_path,
):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = _library_file(first_dir, b"first reverse library")
    second = _library_file(second_dir, b"second reverse library")
    jax, _jnp, library, snapshot, snapshots, loads = _fake_dependencies(monkeypatch)

    registration = reverse_ffi.register_gdn_reverse_s512(
        first,
        library_sha256=_library_sha256(first),
        enabled=True,
    )
    repeated = reverse_ffi.register_gdn_reverse_s512(
        first,
        library_sha256=_library_sha256(first),
        enabled=True,
    )

    assert repeated is registration
    assert snapshots == [(first, _library_sha256(first))]
    assert loads == [snapshot]
    assert reverse_ffi._sealed_loader._library_lifetime_handles == [library]
    assert registration.snapshot_sha256 == _library_sha256(first)
    assert registration.sealed_snapshot is True
    assert registration.snapshot_fd_retained is True
    assert jax.ffi.registration_calls == [
        (
            (
                reverse_ffi.GDN_REVERSE_S512_TARGET,
                ("capsule", library.symbol),
            ),
            {"platform": "ROCM", "api_version": 1},
        )
    ]
    assert len(jax.ffi.registration_states_during_calls) == 1
    pending = jax.ffi.registration_states_during_calls[0]
    assert isinstance(pending, reverse_ffi._PoisonedRegistration)
    assert pending.library_path == first
    assert pending.library_sha256 == _library_sha256(first)
    assert isinstance(reverse_ffi._registration_state, reverse_ffi._RegistrationState)

    with pytest.raises(RuntimeError, match="different library identity"):
        reverse_ffi.register_gdn_reverse_s512(
            second,
            library_sha256=_library_sha256(second),
            enabled=True,
        )
    assert snapshots == [(first, _library_sha256(first))]


def test_loader_failure_and_missing_exact_symbol_fail_closed(monkeypatch, tmp_path):
    path = _library_file(tmp_path)
    digest = _library_sha256(path)
    _jax, _jnp, _library, _snapshot, snapshots, _loads = _fake_dependencies(monkeypatch)
    monkeypatch.setattr(
        reverse_ffi._sealed_loader,
        "_load_cdll",
        lambda _snapshot: (_ for _ in ()).throw(OSError("deliberate failure")),
    )

    with pytest.raises(RuntimeError, match="could not load the sealed"):
        reverse_ffi.register_gdn_reverse_s512(
            path,
            library_sha256=digest,
            enabled=True,
        )
    assert reverse_ffi._registration_state is None
    assert reverse_ffi._sealed_loader._library_lifetime_handles == []
    assert snapshots == [(path, digest)]

    missing = _FakeLibrary(with_symbol=False)
    monkeypatch.setattr(reverse_ffi._sealed_loader, "_load_cdll", lambda _item: missing)
    with pytest.raises(RuntimeError, match="missing its exact handler symbol"):
        reverse_ffi.register_gdn_reverse_s512(
            path,
            library_sha256=digest,
            enabled=True,
        )
    assert reverse_ffi._registration_state is None
    assert reverse_ffi._sealed_loader._library_lifetime_handles == [missing]


def test_pycapsule_failure_retains_loaded_lifetime_without_publishing_state(
    monkeypatch,
    tmp_path,
):
    path = _library_file(tmp_path)
    digest = _library_sha256(path)
    jax, _jnp, library, snapshot, snapshots, loads = _fake_dependencies(monkeypatch)
    working_pycapsule = jax.ffi.pycapsule

    def fail_capsule(symbol):
        jax.ffi.capsule_calls.append(symbol)
        raise RuntimeError("deliberate pycapsule failure")

    monkeypatch.setattr(jax.ffi, "pycapsule", fail_capsule)
    with pytest.raises(RuntimeError, match="deliberate pycapsule failure"):
        reverse_ffi.register_gdn_reverse_s512(
            path,
            library_sha256=digest,
            enabled=True,
        )

    assert snapshots == [(path, digest)]
    assert loads == [snapshot]
    assert jax.ffi.capsule_calls == [library.symbol]
    assert jax.ffi.registration_calls == []
    assert reverse_ffi._sealed_loader._library_lifetime_handles == [library]
    assert snapshot.fd in reverse_ffi._sealed_loader._library_lifetime_fds
    assert reverse_ffi._registration_state is None

    monkeypatch.setattr(jax.ffi, "pycapsule", working_pycapsule)
    registration = reverse_ffi.register_gdn_reverse_s512(
        path,
        library_sha256=digest,
        enabled=True,
    )
    assert registration.library_path == path
    assert snapshots == [(path, digest), (path, digest)]
    assert loads == [snapshot, snapshot]
    assert jax.ffi.capsule_calls == [library.symbol, library.symbol]
    assert len(jax.ffi.registration_calls) == 1
    assert reverse_ffi._sealed_loader._library_lifetime_handles == [library, library]
    assert isinstance(reverse_ffi._registration_state, reverse_ffi._RegistrationState)


def test_target_registration_failure_poison_is_permanent_for_all_identities(
    monkeypatch,
    tmp_path,
):
    path = _library_file(tmp_path)
    digest = _library_sha256(path)
    jax, _jnp, library, snapshot, snapshots, loads = _fake_dependencies(monkeypatch)

    def fail_registration(*args, **kwargs):
        jax.ffi.registration_calls.append((args, kwargs))
        jax.ffi.registration_states_during_calls.append(reverse_ffi._registration_state)
        raise RuntimeError("deliberate target registration failure")

    monkeypatch.setattr(jax.ffi, "register_ffi_target", fail_registration)
    with pytest.raises(RuntimeError, match="deliberate target registration failure"):
        reverse_ffi.register_gdn_reverse_s512(
            path,
            library_sha256=digest,
            enabled=True,
        )

    assert snapshots == [(path, digest)]
    assert loads == [snapshot]
    assert jax.ffi.capsule_calls == [library.symbol]
    assert jax.ffi.registration_calls == [
        (
            (
                reverse_ffi.GDN_REVERSE_S512_TARGET,
                ("capsule", library.symbol),
            ),
            {"platform": "ROCM", "api_version": 1},
        )
    ]
    assert reverse_ffi._sealed_loader._library_lifetime_handles == [library]
    assert snapshot.fd in reverse_ffi._sealed_loader._library_lifetime_fds
    poison = reverse_ffi._registration_state
    assert isinstance(poison, reverse_ffi._PoisonedRegistration)
    assert poison.library_path == path
    assert poison.library_sha256 == digest
    assert poison.target_name == reverse_ffi.GDN_REVERSE_S512_TARGET
    assert poison.failed_stage == "register_ffi_target"
    assert jax.ffi.registration_states_during_calls == [poison]

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other = _library_file(other_dir, b"other reverse library")
    monkeypatch.setattr(
        reverse_ffi,
        "_import_jax",
        lambda: (_ for _ in ()).throw(
            AssertionError("poison rejection must precede another JAX import")
        ),
    )
    attempts = (
        (path, digest),
        (other, _library_sha256(other)),
    )
    for candidate, candidate_digest in attempts:
        with pytest.raises(RuntimeError, match="registration is poisoned"):
            reverse_ffi.register_gdn_reverse_s512(
                candidate,
                library_sha256=candidate_digest,
                enabled=True,
            )
        with pytest.raises(RuntimeError, match="registration is poisoned"):
            reverse_ffi.gdn_reverse_s512(
                *_abstract_inputs(),
                enabled=True,
                library_path=candidate,
                library_sha256=candidate_digest,
            )

    with pytest.raises(RuntimeError, match="registration is poisoned"):
        reverse_ffi.register_gdn_reverse_s512()
    with pytest.raises(RuntimeError, match="registration is poisoned"):
        reverse_ffi.gdn_reverse_s512(*_abstract_inputs())
    with pytest.raises(RuntimeError, match="registration is poisoned"):
        reverse_ffi.register_gdn_reverse_s512(enabled=1)
    with pytest.raises(RuntimeError, match="registration is poisoned"):
        reverse_ffi.gdn_reverse_s512(*_abstract_inputs(), enabled=None)

    assert snapshots == [(path, digest)]
    assert loads == [snapshot]
    assert jax.ffi.capsule_calls == [library.symbol]
    assert len(jax.ffi.registration_calls) == 1
    assert reverse_ffi._registration_state is poison


@pytest.mark.parametrize(
    ("index", "shape"),
    [
        (0, (1, 511, 16, 128)),
        (1, (1, 512, 32, 128)),
        (2, (1, 512, 16, 128)),
        (3, (1, 512, 16)),
        (4, (1, 512, 32, 1)),
        (5, (1, 16, 128, 128)),
        (6, (1, 511, 32, 128)),
        (7, (1, 32, 128, 64)),
    ],
)
def test_every_input_shape_is_gated_before_snapshot(
    monkeypatch,
    tmp_path,
    index,
    shape,
):
    path = _library_file(tmp_path)
    _jax, _jnp, _library, _snapshot, snapshots, _loads = _fake_dependencies(monkeypatch)
    inputs = list(_abstract_inputs())
    inputs[index] = SimpleNamespace(shape=shape, dtype=inputs[index].dtype)

    with pytest.raises(ValueError, match="shape must be exactly"):
        reverse_ffi.gdn_reverse_s512(
            *inputs,
            enabled=True,
            library_path=path,
            library_sha256=_library_sha256(path),
        )
    assert snapshots == []


@pytest.mark.parametrize("index", range(8))
def test_every_input_dtype_is_gated_before_snapshot(monkeypatch, tmp_path, index):
    path = _library_file(tmp_path)
    _jax, _jnp, _library, _snapshot, snapshots, _loads = _fake_dependencies(monkeypatch)
    inputs = list(_abstract_inputs())
    wrong_dtype = "f32" if index == 6 else "bf16"
    inputs[index] = SimpleNamespace(shape=inputs[index].shape, dtype=wrong_dtype)

    with pytest.raises(TypeError, match="dtype must be exactly"):
        reverse_ffi.gdn_reverse_s512(
            *inputs,
            enabled=True,
            library_path=path,
            library_sha256=_library_sha256(path),
        )
    assert snapshots == []


@pytest.mark.parametrize("index", range(8))
def test_every_exposed_stride_and_size_is_gated_before_snapshot(
    monkeypatch,
    tmp_path,
    index,
):
    path = _library_file(tmp_path)
    _jax, _jnp, _library, _snapshot, snapshots, _loads = _fake_dependencies(monkeypatch)
    inputs = list(_abstract_inputs())
    shape = tuple(inputs[index].shape)
    itemsize = 2 if index == 6 else 4
    strides = list(reverse_ffi._expected_c_strides(shape, itemsize))
    strides[-1] = itemsize * 2
    inputs[index] = _ExposedHostValue(
        shape,
        inputs[index].dtype,
        itemsize=itemsize,
        address=0x1000_0000 + index * 0x1000_0000,
        strides=tuple(strides),
    )

    with pytest.raises(ValueError, match="strides must be exactly"):
        reverse_ffi.gdn_reverse_s512(
            *inputs,
            enabled=True,
            library_path=path,
            library_sha256=_library_sha256(path),
        )
    assert snapshots == []

    inputs = list(_abstract_inputs())
    inputs[index] = _ExposedHostValue(
        shape,
        inputs[index].dtype,
        itemsize=itemsize,
        address=0x1000_0000 + index * 0x1000_0000,
        nbytes=_shape_bytes(shape, itemsize) - 1,
    )
    with pytest.raises(ValueError, match="nbytes must be exactly"):
        reverse_ffi.gdn_reverse_s512(
            *inputs,
            enabled=True,
            library_path=path,
            library_sha256=_library_sha256(path),
        )
    assert snapshots == []


def test_input_identity_and_exposed_host_overlap_fail_before_snapshot(
    monkeypatch,
    tmp_path,
):
    path = _library_file(tmp_path)
    _jax, _jnp, _library, _snapshot, snapshots, _loads = _fake_dependencies(monkeypatch)
    inputs = list(_abstract_inputs())
    inputs[1] = inputs[0]
    with pytest.raises(ValueError, match="query overlaps key"):
        reverse_ffi.gdn_reverse_s512(
            *inputs,
            enabled=True,
            library_path=path,
            library_sha256=_library_sha256(path),
        )
    assert snapshots == []

    inputs = list(_abstract_inputs())
    query_bytes = _shape_bytes(reverse_ffi.GDN_REVERSE_S512_QUERY_SHAPE, 4)
    inputs[0] = _ExposedHostValue(
        reverse_ffi.GDN_REVERSE_S512_QUERY_SHAPE,
        "f32",
        itemsize=4,
        address=0x4000_0000,
    )
    inputs[1] = _ExposedHostValue(
        reverse_ffi.GDN_REVERSE_S512_QUERY_SHAPE,
        "f32",
        itemsize=4,
        address=0x4000_0000 + query_bytes - 16,
    )
    with pytest.raises(ValueError, match="query overlaps key"):
        reverse_ffi.gdn_reverse_s512(
            *inputs,
            enabled=True,
            library_path=path,
            library_sha256=_library_sha256(path),
        )
    assert snapshots == []


def test_enabled_call_declares_seven_results_but_returns_only_six_buffers(
    monkeypatch,
    tmp_path,
):
    path = _library_file(tmp_path)
    digest = _library_sha256(path)
    jax, _jnp, _library, _snapshot, snapshots, _loads = _fake_dependencies(monkeypatch)
    inputs = _abstract_inputs()

    result = reverse_ffi.gdn_reverse_s512(
        *inputs,
        enabled=True,
        library_path=path,
        library_sha256=digest,
    )

    assert result == (
        "query-gradient",
        "key-gradient",
        "value-gradient",
        "g-gradient",
        "beta-gradient",
        "initial-state-gradient",
    )
    assert "hidden-u8-scratch" not in result
    assert snapshots == [(path, digest)]
    assert jax.shape_specs == [
        (reverse_ffi.GDN_REVERSE_S512_QUERY_SHAPE, "f32"),
        (reverse_ffi.GDN_REVERSE_S512_QUERY_SHAPE, "f32"),
        (reverse_ffi.GDN_REVERSE_S512_VALUE_SHAPE, "f32"),
        (reverse_ffi.GDN_REVERSE_S512_GATE_SHAPE, "f32"),
        (reverse_ffi.GDN_REVERSE_S512_GATE_SHAPE, "f32"),
        (reverse_ffi.GDN_REVERSE_S512_STATE_SHAPE, "f32"),
        (reverse_ffi.GDN_REVERSE_S512_SCRATCH_SHAPE, "u8"),
    ]
    args, kwargs = jax.ffi.ffi_call_builds[0]
    assert args[0] == reverse_ffi.GDN_REVERSE_S512_TARGET
    assert len(args[1]) == 7
    assert kwargs == {
        "has_side_effect": False,
        "vmap_method": "sequential",
        "input_layouts": (
            (0, 1, 2, 3),
            (0, 1, 2, 3),
            (0, 1, 2, 3),
            (0, 1, 2),
            (0, 1, 2),
            (0, 1, 2, 3),
            (0, 1, 2, 3),
            (0, 1, 2, 3),
        ),
        "output_layouts": (
            (0, 1, 2, 3),
            (0, 1, 2, 3),
            (0, 1, 2, 3),
            (0, 1, 2),
            (0, 1, 2),
            (0, 1, 2, 3),
            (0,),
        ),
        "input_output_aliases": None,
        "custom_call_api_version": 4,
    }
    assert jax.ffi.invocations == [inputs]


def test_wrapper_is_import_light_sealed_and_contains_no_runtime_fallback():
    source = _PYTHON_SOURCE.read_text(encoding="utf-8")
    module = ast.parse(source)
    imports = [
        node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported_roots = {
        alias.name.partition(".")[0] for node in imports for alias in node.names
    }

    assert imported_roots.isdisjoint({"jax", "jaxlib", "numpy", "torch", "hip"})
    assert "_sealed_loader._snapshot_library" in source
    assert "_sealed_loader._load_cdll(snapshot)" in source
    assert "_sealed_loader._library_lifetime_handles.append(library)" in source
    assert "ctypes.CDLL" not in source
    assert "input_output_aliases=None" in source
    assert "_hidden_scratch" in source
    assert "fallback" in source.lower()
    for forbidden in (
        "hipgraph",
        "cudagraph",
        "command_buffer",
        "streambegincapture",
        "streamendcapture",
        "graphlaunch",
    ):
        assert forbidden not in source.lower()
