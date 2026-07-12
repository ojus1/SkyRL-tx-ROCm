from __future__ import annotations

import ast
import fcntl
import hashlib
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
    retained_fds: list[int] = []
    monkeypatch.setattr(smoke, "_library_lifetime_fds", retained_fds)
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
    loads: list[smoke._SealedLibrarySnapshot] = []

    monkeypatch.setattr(smoke, "_import_jax", lambda: (jax, jnp))

    def load(snapshot):
        loads.append(snapshot)
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


def _library_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    with pytest.raises(ValueError, match="invalid while.*disabled"):
        smoke.gdn_ffi_smoke_copy(sentinel, library_sha256="0" * 64)


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


@pytest.mark.parametrize(
    "library_sha256",
    [None, "", "0" * 63, "0" * 65, "G" * 64, b"0" * 64],
)
def test_enabled_public_paths_require_exact_lowercase_sha_before_jax_or_snapshot(
    monkeypatch, tmp_path, library_sha256
):
    path = _library_file(tmp_path)
    monkeypatch.setattr(
        smoke,
        "_import_jax",
        lambda: (_ for _ in ()).throw(AssertionError("JAX must not be imported")),
    )
    monkeypatch.setattr(
        smoke,
        "_snapshot_library",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not snapshot")),
    )
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        smoke.register_gdn_ffi_smoke(path, library_sha256=library_sha256, enabled=True)
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        smoke.gdn_ffi_smoke_copy(
            object(),
            library_path=path,
            library_sha256=library_sha256,
            enabled=True,
        )


def test_unapproved_hash_never_reaches_cdll_and_rejected_snapshot_is_sealed_retained(
    monkeypatch, tmp_path
):
    path = _library_file(tmp_path)
    jax = _FakeJax()
    monkeypatch.setattr(
        smoke, "_import_jax", lambda: (jax, SimpleNamespace(bfloat16="bf16"))
    )
    monkeypatch.setattr(
        smoke,
        "_load_cdll",
        lambda _snapshot: (_ for _ in ()).throw(
            AssertionError("unapproved bytes must never reach CDLL")
        ),
    )

    with pytest.raises(RuntimeError, match="does not match library_sha256"):
        smoke.register_gdn_ffi_smoke(path, library_sha256="0" * 64, enabled=True)

    assert len(smoke._library_lifetime_fds) == 1
    descriptor = smoke._library_lifetime_fds[0]
    observed_seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
    assert observed_seals & smoke._REQUIRED_SEALS == smoke._REQUIRED_SEALS
    assert stat.S_IMODE(os.fstat(descriptor).st_mode) == 0o600
    assert smoke._library_lifetime_handles == []


def test_dlopen_reads_approved_sealed_snapshot_despite_path_replacement(
    monkeypatch, tmp_path
):
    approved = b"approved exact shared object bytes"
    unapproved = b"replacement bytes must not load"
    path = _library_file(tmp_path)
    path.write_bytes(approved)
    path.chmod(0o600)
    digest = hashlib.sha256(approved).hexdigest()
    jax = _FakeJax()
    library = _FakeLibrary()
    loaded: list[bytes] = []
    monkeypatch.setattr(
        smoke, "_import_jax", lambda: (jax, SimpleNamespace(bfloat16="bf16"))
    )

    def replace_then_load(snapshot):
        replacement = path.with_name("replacement.so")
        replacement.write_bytes(unapproved)
        replacement.chmod(0o600)
        os.replace(replacement, path)
        loaded.append(Path(snapshot.proc_path).read_bytes())
        assert (
            fcntl.fcntl(snapshot.fd, fcntl.F_GET_SEALS) & smoke._REQUIRED_SEALS
            == smoke._REQUIRED_SEALS
        )
        return library

    monkeypatch.setattr(smoke, "_load_cdll", replace_then_load)
    registration = smoke.register_gdn_ffi_smoke(
        path, library_sha256=digest, enabled=True
    )

    assert path.read_bytes() == unapproved
    assert loaded == [approved]
    assert registration.snapshot_sha256 == digest
    assert smoke._registration_state is not None
    assert smoke._registration_state.snapshot.fd in smoke._library_lifetime_fds


def test_missing_required_seal_fails_before_cdll_and_retains_fd(monkeypatch, tmp_path):
    path = _library_file(tmp_path)
    digest = _library_sha256(path)
    real_fcntl = fcntl.fcntl
    monkeypatch.setattr(
        smoke,
        "_load_cdll",
        lambda _snapshot: (_ for _ in ()).throw(
            AssertionError("unsealed snapshot must never reach CDLL")
        ),
    )

    def omit_write_seal(descriptor, command, argument=0):
        if command == fcntl.F_GET_SEALS:
            return smoke._REQUIRED_SEALS & ~fcntl.F_SEAL_WRITE
        return real_fcntl(descriptor, command, argument)

    monkeypatch.setattr(smoke.fcntl, "fcntl", omit_write_seal)
    with pytest.raises(RuntimeError, match="every required seal"):
        smoke.register_gdn_ffi_smoke(path, library_sha256=digest, enabled=True)
    assert len(smoke._library_lifetime_fds) == 1


def test_source_fd_identity_mutation_is_rejected_before_cdll(monkeypatch, tmp_path):
    path = _library_file(tmp_path)
    digest = _library_sha256(path)
    real_read = os.read
    mutated = False
    monkeypatch.setattr(
        smoke,
        "_load_cdll",
        lambda _snapshot: (_ for _ in ()).throw(
            AssertionError("unstable source must never reach CDLL")
        ),
    )

    def mutate_after_first_read(descriptor, count):
        nonlocal mutated
        chunk = real_read(descriptor, count)
        if chunk and not mutated:
            mutated = True
            path.write_bytes(b"changed source identity")
            path.chmod(0o600)
        return chunk

    monkeypatch.setattr(smoke.os, "read", mutate_after_first_read)
    with pytest.raises(RuntimeError, match="changed while being snapshotted"):
        smoke.register_gdn_ffi_smoke(path, library_sha256=digest, enabled=True)
    assert mutated is True
    assert len(smoke._library_lifetime_fds) == 1


def test_snapshot_fd_survives_cdll_failure(monkeypatch, tmp_path):
    path = _library_file(tmp_path)
    digest = _library_sha256(path)
    monkeypatch.setattr(
        smoke,
        "_load_cdll",
        lambda _snapshot: (_ for _ in ()).throw(OSError("private loader failure")),
    )
    with pytest.raises(RuntimeError, match="sealed.*snapshot"):
        smoke.register_gdn_ffi_smoke(path, library_sha256=digest, enabled=True)
    assert len(smoke._library_lifetime_fds) == 1
    assert os.fstat(smoke._library_lifetime_fds[0]).st_size == path.stat().st_size


def test_enabled_path_requires_exact_absolute_owned_private_regular_library(tmp_path):
    with pytest.raises(ValueError, match="library_sha256"):
        smoke.register_gdn_ffi_smoke(enabled=True)
    with pytest.raises(ValueError, match="absolute"):
        smoke.register_gdn_ffi_smoke(
            "libskyrl_gdn_ffi_smoke_gfx1100.so",
            library_sha256="0" * 64,
            enabled=True,
        )

    wrong_name = _library_file(tmp_path, "wrong.so")
    with pytest.raises(ValueError, match="exact name"):
        smoke.register_gdn_ffi_smoke(
            wrong_name, library_sha256=_library_sha256(wrong_name), enabled=True
        )

    directory = tmp_path / "libskyrl_gdn_ffi_smoke_gfx1100.so"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        smoke.register_gdn_ffi_smoke(
            directory.resolve(), library_sha256="0" * 64, enabled=True
        )


def test_group_or_world_writable_and_symlink_libraries_are_rejected(tmp_path):
    writable = _library_file(tmp_path)
    writable.chmod(0o620)
    with pytest.raises(ValueError, match="group- or world-writable"):
        smoke.register_gdn_ffi_smoke(
            writable, library_sha256=_library_sha256(writable), enabled=True
        )

    writable.unlink()
    target = tmp_path / "private-target.so"
    target.write_bytes(b"mock")
    target.chmod(0o600)
    link = tmp_path / "libskyrl_gdn_ffi_smoke_gfx1100.so"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        smoke.register_gdn_ffi_smoke(
            link.absolute(),
            library_sha256=hashlib.sha256(b"mock").hexdigest(),
            enabled=True,
        )


def test_registration_is_rocm_api_v1_and_retains_cdll(monkeypatch, tmp_path):
    path = _library_file(tmp_path)
    digest = _library_sha256(path)
    jax, _jnp, library, loads = _fake_dependencies(monkeypatch)

    registration = smoke.register_gdn_ffi_smoke(
        path, library_sha256=digest, enabled=True
    )

    assert registration.library_path == path
    assert registration.library_sha256 == digest
    assert registration.snapshot_sha256 == digest
    assert registration.snapshot_size_bytes == path.stat().st_size
    assert registration.snapshot_mode == 0o600
    assert registration.snapshot_seals & smoke._REQUIRED_SEALS == smoke._REQUIRED_SEALS
    assert registration.sealed_snapshot is True
    assert registration.snapshot_fd_retained is True
    assert len(loads) == 1
    assert loads[0].original_path == path
    assert loads[0].sha256 == digest
    assert loads[0].fd in smoke._library_lifetime_fds
    assert jax.ffi.capsule_calls == [library.symbol]
    assert len(jax.ffi.registration_calls) == 1
    args, kwargs = jax.ffi.registration_calls[0]
    assert args == (smoke.GDN_FFI_SMOKE_TARGET, ("capsule", library.symbol))
    assert kwargs == {"platform": "ROCM", "api_version": 1}
    assert library.symbol.restype is not None
    assert library.symbol.argtypes is not None
    assert smoke._registration_state is not None
    assert smoke._registration_state.library is library
    assert smoke._registration_state.snapshot is loads[0]
    assert smoke._library_lifetime_handles == [library]


def test_registration_is_idempotent_only_for_same_canonical_path(monkeypatch, tmp_path):
    first = _library_file(tmp_path)
    first_digest = _library_sha256(first)
    jax, _jnp, _library, loads = _fake_dependencies(monkeypatch)

    one = smoke.register_gdn_ffi_smoke(first, library_sha256=first_digest, enabled=True)
    two = smoke.register_gdn_ffi_smoke(first, library_sha256=first_digest, enabled=True)
    assert one is two
    assert len(loads) == 1
    assert len(jax.ffi.registration_calls) == 1

    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second = _library_file(second_dir)
    with pytest.raises(RuntimeError, match="different library"):
        smoke.register_gdn_ffi_smoke(
            second, library_sha256=_library_sha256(second), enabled=True
        )

    with pytest.raises(RuntimeError, match="different library"):
        smoke.register_gdn_ffi_smoke(first, library_sha256="0" * 64, enabled=True)


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
        smoke.register_gdn_ffi_smoke(
            path, library_sha256=_library_sha256(path), enabled=True
        )
    assert jax.ffi.registration_calls == []
    assert smoke._registration_state is None
    assert len(smoke._library_lifetime_fds) == 1
    assert len(smoke._library_lifetime_handles) == 1


def test_fd_and_cdll_survive_partial_target_registration_failure(monkeypatch, tmp_path):
    path = _library_file(tmp_path)
    jax, _jnp, library, loads = _fake_dependencies(monkeypatch)
    monkeypatch.setattr(
        jax.ffi,
        "register_ffi_target",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("private registry failure")
        ),
    )
    with pytest.raises(RuntimeError, match="private registry failure"):
        smoke.register_gdn_ffi_smoke(
            path, library_sha256=_library_sha256(path), enabled=True
        )
    assert smoke._registration_state is None
    assert len(loads) == 1
    assert loads[0].fd in smoke._library_lifetime_fds
    assert smoke._library_lifetime_handles == [library]


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
        smoke.gdn_ffi_smoke_copy(
            value,
            enabled=True,
            library_path=path,
            library_sha256=_library_sha256(path),
        )
    assert loads == []


def test_enabled_copy_rejects_non_bf16_before_library_load(monkeypatch, tmp_path):
    path = _library_file(tmp_path)
    _jax, _jnp, _library, loads = _fake_dependencies(monkeypatch)
    value = SimpleNamespace(shape=smoke.GDN_FFI_SMOKE_SHAPE, dtype="float16")

    with pytest.raises(TypeError, match="dtype must be exactly bfloat16"):
        smoke.gdn_ffi_smoke_copy(
            value,
            enabled=True,
            library_path=path,
            library_sha256=_library_sha256(path),
        )
    assert loads == []


def test_enabled_copy_builds_exact_abstract_typed_ffi_call(monkeypatch, tmp_path):
    path = _library_file(tmp_path)
    jax, _jnp, _library, loads = _fake_dependencies(monkeypatch)
    value = SimpleNamespace(shape=smoke.GDN_FFI_SMOKE_SHAPE, dtype="bf16")

    digest = _library_sha256(path)
    result = smoke.gdn_ffi_smoke_copy(
        value,
        enabled=True,
        library_path=path,
        library_sha256=digest,
    )

    assert result == ("ffi-result", value)
    assert len(loads) == 1
    assert loads[0].original_path == path
    assert loads[0].sha256 == digest
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
    assert "os.memfd_create" in source
    assert "F_ADD_SEALS" in source
    assert "F_GET_SEALS" in source
    assert "F_SEAL_WRITE" in source
    assert "F_SEAL_GROW" in source
    assert "F_SEAL_SHRINK" in source
    assert "F_SEAL_SEAL" in source
    assert "ctypes.CDLL(snapshot.proc_path" in source
    assert "ctypes.CDLL(str(library_path)" not in source


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
