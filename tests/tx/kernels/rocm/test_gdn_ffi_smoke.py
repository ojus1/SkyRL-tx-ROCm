from __future__ import annotations

import ast
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from skyrl.tx.kernels.rocm import gdn_ffi_smoke as smoke

_REPO = Path(__file__).parents[4]
_PYTHON_SOURCE = _REPO / "skyrl" / "tx" / "kernels" / "rocm" / "gdn_ffi_smoke.py"
_HIP_SOURCE = _REPO / "skyrl" / "tx" / "kernels" / "rocm" / "ffi" / "gdn_ffi_smoke.hip"
_BUILD_SCRIPT = (
    _REPO
    / "skyrl"
    / "tx"
    / "kernels"
    / "rocm"
    / "ffi"
    / "build_gdn_ffi_smoke_gfx1100.sh"
)


@pytest.fixture(autouse=True)
def _reset_registration(monkeypatch):
    monkeypatch.setattr(smoke, "_registration_state", None)
    monkeypatch.setattr(smoke, "_library_lifetime_handles", [])


class _FakeSymbol:
    restype: Any = None
    argtypes: Any = None


class _FakeLibrary:
    def __init__(self, *, with_symbol: bool = True) -> None:
        self.symbol = _FakeSymbol()
        if with_symbol:
            setattr(self, smoke.GDN_FFI_SMOKE_TARGET, self.symbol)


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
            return ("ffi-result", values[0])

        return invoke


class _FakeJax:
    def __init__(self) -> None:
        self.ffi = _FakeFfi()
        self.shape_specs: list[tuple[tuple[int, ...], Any]] = []

    def ShapeDtypeStruct(self, shape, dtype):
        spec = SimpleNamespace(shape=tuple(shape), dtype=dtype)
        self.shape_specs.append((tuple(shape), dtype))
        return spec


def _fake_dependencies(monkeypatch):
    jax = _FakeJax()
    jnp = SimpleNamespace(bfloat16="bf16")
    library = _FakeLibrary()
    loads: list[Path] = []

    monkeypatch.setattr(smoke, "_import_jax", lambda: (jax, jnp))

    def load(path):
        loads.append(path)
        return library

    monkeypatch.setattr(smoke, "_load_cdll", load)
    return jax, jnp, library, loads


def _library_file(
    tmp_path: Path, name: str = "libskyrl_gdn_ffi_smoke_gfx1100.so"
) -> Path:
    path = tmp_path / name
    path.write_bytes(b"mock shared object")
    path.chmod(0o600)
    return path.resolve()


def test_import_and_default_fallback_do_not_import_jax_or_load_a_library(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        smoke,
        "_import_jax",
        lambda: (_ for _ in ()).throw(AssertionError("JAX must not be imported")),
    )
    monkeypatch.setattr(
        smoke,
        "_load_cdll",
        lambda _path: (_ for _ in ()).throw(AssertionError("library must not load")),
    )

    assert smoke.gdn_ffi_smoke_copy(sentinel) is sentinel
    with pytest.raises(ValueError, match="invalid while.*disabled"):
        smoke.gdn_ffi_smoke_copy(sentinel, library_path="/private/not-loaded.so")


def test_registration_is_default_off_before_path_jax_or_loader(monkeypatch):
    monkeypatch.setattr(
        smoke,
        "_import_jax",
        lambda: (_ for _ in ()).throw(AssertionError("JAX must not be imported")),
    )
    monkeypatch.setattr(
        smoke,
        "_load_cdll",
        lambda _path: (_ for _ in ()).throw(AssertionError("library must not load")),
    )

    with pytest.raises(RuntimeError, match="disabled by default"):
        smoke.register_gdn_ffi_smoke()


@pytest.mark.parametrize("value", [1, 0, None, "true"])
def test_opt_in_must_be_an_exact_bool(value):
    with pytest.raises(TypeError, match="exact bool"):
        smoke.register_gdn_ffi_smoke(enabled=value)
    with pytest.raises(TypeError, match="exact bool"):
        smoke.gdn_ffi_smoke_copy(object(), enabled=value)


def test_enabled_path_requires_exact_absolute_owned_private_regular_library(tmp_path):
    with pytest.raises(ValueError, match="explicit"):
        smoke.register_gdn_ffi_smoke(enabled=True)
    with pytest.raises(ValueError, match="absolute"):
        smoke.register_gdn_ffi_smoke("libskyrl_gdn_ffi_smoke_gfx1100.so", enabled=True)

    wrong_name = _library_file(tmp_path, "wrong.so")
    with pytest.raises(ValueError, match="exact name"):
        smoke.register_gdn_ffi_smoke(wrong_name, enabled=True)

    directory = tmp_path / "libskyrl_gdn_ffi_smoke_gfx1100.so"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        smoke.register_gdn_ffi_smoke(directory.resolve(), enabled=True)


def test_group_or_world_writable_and_symlink_libraries_are_rejected(tmp_path):
    writable = _library_file(tmp_path)
    writable.chmod(0o620)
    with pytest.raises(ValueError, match="group- or world-writable"):
        smoke.register_gdn_ffi_smoke(writable, enabled=True)

    writable.unlink()
    target = tmp_path / "private-target.so"
    target.write_bytes(b"mock")
    target.chmod(0o600)
    link = tmp_path / "libskyrl_gdn_ffi_smoke_gfx1100.so"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        smoke.register_gdn_ffi_smoke(link.absolute(), enabled=True)


def test_registration_is_rocm_api_v1_and_retains_cdll(monkeypatch, tmp_path):
    path = _library_file(tmp_path)
    jax, _jnp, library, loads = _fake_dependencies(monkeypatch)

    registration = smoke.register_gdn_ffi_smoke(path, enabled=True)

    assert registration == smoke.GdnFfiSmokeRegistration(library_path=path)
    assert loads == [path]
    assert jax.ffi.capsule_calls == [library.symbol]
    assert len(jax.ffi.registration_calls) == 1
    args, kwargs = jax.ffi.registration_calls[0]
    assert args == (smoke.GDN_FFI_SMOKE_TARGET, ("capsule", library.symbol))
    assert kwargs == {"platform": "ROCM", "api_version": 1}
    assert library.symbol.restype is not None
    assert library.symbol.argtypes is not None
    assert smoke._registration_state is not None
    assert smoke._registration_state.library is library
    assert smoke._library_lifetime_handles == [library]


def test_registration_is_idempotent_only_for_same_canonical_path(monkeypatch, tmp_path):
    first = _library_file(tmp_path)
    jax, _jnp, _library, loads = _fake_dependencies(monkeypatch)

    one = smoke.register_gdn_ffi_smoke(first, enabled=True)
    two = smoke.register_gdn_ffi_smoke(first, enabled=True)
    assert one is two
    assert loads == [first]
    assert len(jax.ffi.registration_calls) == 1

    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second = _library_file(second_dir)
    with pytest.raises(RuntimeError, match="different library"):
        smoke.register_gdn_ffi_smoke(second, enabled=True)


def test_missing_exact_handler_symbol_fails_without_registration(monkeypatch, tmp_path):
    path = _library_file(tmp_path)
    jax = _FakeJax()
    monkeypatch.setattr(
        smoke, "_import_jax", lambda: (jax, SimpleNamespace(bfloat16="bf16"))
    )
    monkeypatch.setattr(
        smoke, "_load_cdll", lambda _path: _FakeLibrary(with_symbol=False)
    )

    with pytest.raises(RuntimeError, match="missing its exact handler symbol"):
        smoke.register_gdn_ffi_smoke(path, enabled=True)
    assert jax.ffi.registration_calls == []
    assert smoke._registration_state is None


@pytest.mark.parametrize(
    "value",
    [
        SimpleNamespace(shape=(1, 512, 32, 128), dtype="bf16"),
        SimpleNamespace(shape=(1, 1024, 16, 128), dtype="bf16"),
        SimpleNamespace(shape=(1, 1024, 32, 64), dtype="bf16"),
        SimpleNamespace(shape=(1, 1024, 32, 128, 1), dtype="bf16"),
    ],
)
def test_enabled_copy_rejects_every_shape_drift_before_library_load(
    monkeypatch, tmp_path, value
):
    path = _library_file(tmp_path)
    _jax, _jnp, _library, loads = _fake_dependencies(monkeypatch)

    with pytest.raises(ValueError, match="shape must be exactly"):
        smoke.gdn_ffi_smoke_copy(value, enabled=True, library_path=path)
    assert loads == []


def test_enabled_copy_rejects_non_bf16_before_library_load(monkeypatch, tmp_path):
    path = _library_file(tmp_path)
    _jax, _jnp, _library, loads = _fake_dependencies(monkeypatch)
    value = SimpleNamespace(shape=smoke.GDN_FFI_SMOKE_SHAPE, dtype="float16")

    with pytest.raises(TypeError, match="dtype must be exactly bfloat16"):
        smoke.gdn_ffi_smoke_copy(value, enabled=True, library_path=path)
    assert loads == []


def test_enabled_copy_builds_exact_abstract_typed_ffi_call(monkeypatch, tmp_path):
    path = _library_file(tmp_path)
    jax, _jnp, _library, loads = _fake_dependencies(monkeypatch)
    value = SimpleNamespace(shape=smoke.GDN_FFI_SMOKE_SHAPE, dtype="bf16")

    result = smoke.gdn_ffi_smoke_copy(value, enabled=True, library_path=path)

    assert result == ("ffi-result", value)
    assert loads == [path]
    assert jax.shape_specs == [(smoke.GDN_FFI_SMOKE_SHAPE, "bf16")]
    assert len(jax.ffi.ffi_call_builds) == 1
    args, kwargs = jax.ffi.ffi_call_builds[0]
    assert args[0] == smoke.GDN_FFI_SMOKE_TARGET
    assert args[1].shape == smoke.GDN_FFI_SMOKE_SHAPE
    assert args[1].dtype == "bf16"
    assert kwargs == {
        "has_side_effect": False,
        "vmap_method": "sequential",
        "input_layouts": ((0, 1, 2, 3),),
        "output_layouts": (0, 1, 2, 3),
        "input_output_aliases": None,
        "custom_call_api_version": 4,
    }
    assert jax.ffi.invocations == [(value,)]


def test_python_module_has_no_eager_jax_import_or_library_load():
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
    top_level_calls = [
        node
        for node in module.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert top_level_calls == []
    assert "JAX_PLATFORMS" not in source
    assert "HIP_VISIBLE_DEVICES" not in source


def test_hip_handler_is_exact_shape_asynchronous_stream_smoke_only():
    source = _HIP_SOURCE.read_text(encoding="utf-8")

    assert "Typed-FFI stream/lifetime smoke" in source
    assert "kTokens = 1024" in source
    assert "kValueHeads = 32" in source
    assert "kHeadDimension = 128" in source
    assert "kElements == 4'194'304" in source
    assert source.count("hipLaunchKernelGGL") == 1
    assert "PlatformStream<hipStream_t>" in source
    assert "XLA_FFI_DEFINE_HANDLER_SYMBOL" in source
    assert smoke.GDN_FFI_SMOKE_TARGET in source
    assert source.count("hipGetLastError") == 2
    assert "hipPeekAtLastError" not in source
    assert "hipDeviceSynchronize" not in source
    assert "hipStreamSynchronize" not in source
    assert "triangular_solve" not in source
    assert "recurrence" not in source.lower()


def test_build_rule_is_deterministic_gfx1100_shared_library_only():
    source = _BUILD_SCRIPT.read_text(encoding="utf-8")

    assert 'EXPECTED_ARCH="gfx1100"' in source
    assert 'EXPECTED_BASENAME="libskyrl_gdn_ffi_smoke_gfx1100.so"' in source
    assert '"--offload-arch=$EXPECTED_ARCH"' in source
    assert "-fPIC -shared" in source
    assert "JAXLIB_INCLUDE_DIR" in source
    assert "output must be outside the source repository" in source
    assert "--dynamic --defined-only" in source
    assert "/dev/kfd" not in source
    assert "rocminfo" not in source
    assert "hipDeviceSynchronize" not in source
    assert "--genco" not in source


def test_no_compiled_binary_is_committed_next_to_sources():
    ffi_dir = _HIP_SOURCE.parent
    allowed_suffixes = {".hip", ".sh", ".md"}
    unexpected = [
        path.name
        for path in ffi_dir.iterdir()
        if path.is_file() and path.suffix not in allowed_suffixes
    ]
    assert unexpected == []
    assert stat.S_IMODE(os.stat(_BUILD_SCRIPT).st_mode) & stat.S_IXUSR
