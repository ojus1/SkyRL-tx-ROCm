from __future__ import annotations

import ast
import hashlib
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from skyrl.tx.kernels.rocm import gdn_execute_ffi as execute
from skyrl.tx.kernels.rocm import gdn_execute_oracle as oracle

_REPO = Path(__file__).parents[2]
_PYTHON_SOURCE = _REPO / "skyrl" / "tx" / "kernels" / "rocm" / "gdn_execute_ffi.py"
_HIP_SOURCE = (
    _REPO / "skyrl" / "tx" / "kernels" / "rocm" / "ffi" / "gdn_execute_s512.hip"
)
_BUILD_SCRIPT = (
    _REPO
    / "skyrl"
    / "tx"
    / "kernels"
    / "rocm"
    / "ffi"
    / "build_gdn_execute_s512_gfx1100.sh"
)


@pytest.fixture(autouse=True)
def _reset_registration(monkeypatch):
    monkeypatch.setattr(execute, "_registration_state", None)
    monkeypatch.setattr(execute._sealed_loader, "_library_lifetime_handles", [])
    retained_fds: list[int] = []
    monkeypatch.setattr(
        execute._sealed_loader,
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
            setattr(self, execute.GDN_EXECUTE_S512_TARGET, self.symbol)


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
            return ("bf16-output", "fp32-final-state")

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
    path = tmp_path / "libskyrl_gdn_execute_s512_gfx1100.so"
    path.write_bytes(b"mock exact execute shared object")
    path.chmod(0o600)
    return path.resolve()


def _library_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_dependencies(monkeypatch):
    jax = _FakeJax()
    jnp = SimpleNamespace(float32="f32", bfloat16="bf16")
    library = _FakeLibrary()
    snapshots: list[tuple[Path, str]] = []
    loads: list[Any] = []
    snapshot = SimpleNamespace(
        sha256=None,
        size_bytes=32,
        mode=0o600,
        seals=15,
        fd=-1,
    )

    monkeypatch.setattr(execute, "_import_jax", lambda: (jax, jnp))

    def seal(path, digest):
        snapshots.append((path, digest))
        snapshot.sha256 = digest
        execute._sealed_loader._library_lifetime_fds.append(snapshot.fd)
        return snapshot

    def load(item):
        loads.append(item)
        return library

    monkeypatch.setattr(execute._sealed_loader, "_snapshot_library", seal)
    monkeypatch.setattr(execute._sealed_loader, "_load_cdll", load)
    return jax, jnp, library, snapshot, snapshots, loads


def _abstract_inputs(dtype: Any = "f32") -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(shape=shape, dtype=dtype)
        for shape in (
            execute.GDN_EXECUTE_S512_QUERY_SHAPE,
            execute.GDN_EXECUTE_S512_QUERY_SHAPE,
            execute.GDN_EXECUTE_S512_PREPARED_SHAPE,
            execute.GDN_EXECUTE_S512_PREPARED_SHAPE,
            execute.GDN_EXECUTE_S512_GAMMA_SHAPE,
            execute.GDN_EXECUTE_S512_STATE_SHAPE,
        )
    )


def test_wrapper_constants_are_identical_to_committed_oracle_boundary():
    assert execute.GDN_EXECUTE_S512_QUERY_SHAPE == oracle.GDN_EXECUTE_S512_QUERY_SHAPE
    assert (
        execute.GDN_EXECUTE_S512_PREPARED_SHAPE
        == oracle.GDN_EXECUTE_S512_PREPARED_SHAPE
    )
    assert execute.GDN_EXECUTE_S512_GAMMA_SHAPE == oracle.GDN_EXECUTE_S512_GAMMA_SHAPE
    assert execute.GDN_EXECUTE_S512_STATE_SHAPE == oracle.GDN_EXECUTE_S512_STATE_SHAPE
    assert execute.GDN_EXECUTE_S512_OUTPUT_SHAPE == oracle.GDN_EXECUTE_S512_OUTPUT_SHAPE
    assert execute.GDN_EXECUTE_S512_INPUT_BYTES == oracle.GDN_EXECUTE_S512_INPUT_BYTES
    assert (
        execute.GDN_EXECUTE_S512_OUTPUT_BYTES
        == oracle.GDN_EXECUTE_S512_OUTPUT_TENSOR_BYTES
    )


def test_registration_and_operation_are_default_off_without_jax_or_loading(monkeypatch):
    monkeypatch.setattr(
        execute,
        "_import_jax",
        lambda: (_ for _ in ()).throw(AssertionError("JAX must remain lazy")),
    )
    monkeypatch.setattr(
        execute._sealed_loader,
        "_snapshot_library",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not snapshot")),
    )

    with pytest.raises(RuntimeError, match="registration is disabled by default"):
        execute.register_gdn_execute_s512()
    with pytest.raises(RuntimeError, match="disabled by default"):
        execute.gdn_execute_s512(*_abstract_inputs())
    with pytest.raises(ValueError, match="invalid while.*disabled"):
        execute.gdn_execute_s512(
            *_abstract_inputs(),
            library_path="/private/not-loaded.so",
        )


@pytest.mark.parametrize("enabled", [1, 0, None, "true"])
def test_opt_in_requires_an_exact_bool(enabled):
    with pytest.raises(TypeError, match="exact bool"):
        execute.register_gdn_execute_s512(enabled=enabled)
    with pytest.raises(TypeError, match="exact bool"):
        execute.gdn_execute_s512(*_abstract_inputs(), enabled=enabled)


@pytest.mark.parametrize(
    "digest",
    [None, "", "0" * 63, "0" * 65, "G" * 64, b"0" * 64],
)
def test_enabled_paths_require_exact_lowercase_sha_before_jax(
    monkeypatch, tmp_path, digest
):
    path = _library_file(tmp_path)
    monkeypatch.setattr(
        execute,
        "_import_jax",
        lambda: (_ for _ in ()).throw(AssertionError("JAX must remain lazy")),
    )
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        execute.register_gdn_execute_s512(
            path,
            library_sha256=digest,
            enabled=True,
        )


def test_library_path_is_exact_absolute_canonical_private_file(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        execute.register_gdn_execute_s512(
            "libskyrl_gdn_execute_s512_gfx1100.so",
            library_sha256="0" * 64,
            enabled=True,
        )

    wrong = tmp_path / "wrong.so"
    wrong.write_bytes(b"wrong")
    wrong.chmod(0o600)
    with pytest.raises(ValueError, match="exact name"):
        execute.register_gdn_execute_s512(
            wrong.resolve(),
            library_sha256=_library_sha256(wrong),
            enabled=True,
        )

    library = _library_file(tmp_path)
    library.chmod(0o620)
    with pytest.raises(ValueError, match="group- or world-writable"):
        execute.register_gdn_execute_s512(
            library,
            library_sha256=_library_sha256(library),
            enabled=True,
        )


def test_symlink_library_is_rejected_before_snapshot(tmp_path):
    target = tmp_path / "private-target.so"
    target.write_bytes(b"target")
    target.chmod(0o600)
    link = tmp_path / "libskyrl_gdn_execute_s512_gfx1100.so"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        execute.register_gdn_execute_s512(
            link.absolute(),
            library_sha256=hashlib.sha256(b"target").hexdigest(),
            enabled=True,
        )


def test_registration_is_isolated_sealed_and_exact(monkeypatch, tmp_path):
    path = _library_file(tmp_path)
    digest = _library_sha256(path)
    jax, _jnp, library, snapshot, snapshots, loads = _fake_dependencies(monkeypatch)

    registration = execute.register_gdn_execute_s512(
        path,
        library_sha256=digest,
        enabled=True,
    )

    assert snapshots == [(path, digest)]
    assert loads == [snapshot]
    assert execute._sealed_loader._library_lifetime_handles == [library]
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
                execute.GDN_EXECUTE_S512_TARGET,
                ("capsule", library.symbol),
            ),
            {"platform": "ROCM", "api_version": 1},
        )
    ]
    assert library.symbol.restype is not None
    assert library.symbol.argtypes is not None


def test_repeated_identical_registration_reuses_one_sealed_identity(
    monkeypatch, tmp_path
):
    path = _library_file(tmp_path)
    digest = _library_sha256(path)
    jax, _jnp, _library, snapshot, snapshots, loads = _fake_dependencies(monkeypatch)

    first = execute.register_gdn_execute_s512(path, library_sha256=digest, enabled=True)
    second = execute.register_gdn_execute_s512(
        path, library_sha256=digest, enabled=True
    )

    assert first is second
    assert snapshots == [(path, digest)]
    assert loads == [snapshot]
    assert len(jax.ffi.registration_calls) == 1


def test_conflicting_registration_identity_fails_before_second_snapshot(
    monkeypatch, tmp_path
):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = _library_file(first_dir)
    second = _library_file(second_dir)
    second.write_bytes(b"different exact execute shared object")
    second.chmod(0o600)
    _jax, _jnp, _library, _snapshot, snapshots, _loads = _fake_dependencies(monkeypatch)

    execute.register_gdn_execute_s512(
        first, library_sha256=_library_sha256(first), enabled=True
    )
    with pytest.raises(RuntimeError, match="different library identity"):
        execute.register_gdn_execute_s512(
            second, library_sha256=_library_sha256(second), enabled=True
        )
    assert snapshots == [(first, _library_sha256(first))]


def test_loader_failure_and_missing_handler_symbol_fail_closed(monkeypatch, tmp_path):
    path = _library_file(tmp_path)
    digest = _library_sha256(path)
    _jax, _jnp, _library, _snapshot, snapshots, _loads = _fake_dependencies(monkeypatch)
    monkeypatch.setattr(
        execute._sealed_loader,
        "_load_cdll",
        lambda _snapshot: (_ for _ in ()).throw(OSError("deliberate load failure")),
    )

    with pytest.raises(RuntimeError, match="could not load the sealed"):
        execute.register_gdn_execute_s512(path, library_sha256=digest, enabled=True)
    assert execute._registration_state is None
    assert execute._sealed_loader._library_lifetime_handles == []
    assert snapshots == [(path, digest)]

    missing = _FakeLibrary(with_symbol=False)
    monkeypatch.setattr(execute._sealed_loader, "_load_cdll", lambda _snapshot: missing)
    with pytest.raises(RuntimeError, match="missing its exact handler symbol"):
        execute.register_gdn_execute_s512(path, library_sha256=digest, enabled=True)
    assert execute._registration_state is None
    assert execute._sealed_loader._library_lifetime_handles == [missing]


@pytest.mark.parametrize(
    ("index", "shape"),
    [
        (0, (1, 511, 16, 128)),
        (1, (1, 512, 32, 128)),
        (2, (1, 512, 16, 128)),
        (3, (1, 512, 32, 64)),
        (4, (1, 512, 16)),
        (5, (1, 16, 128, 128)),
    ],
)
def test_every_input_shape_is_gated_before_snapshot(
    monkeypatch, tmp_path, index, shape
):
    path = _library_file(tmp_path)
    _jax, _jnp, _library, _snapshot, snapshots, _loads = _fake_dependencies(monkeypatch)
    inputs = list(_abstract_inputs())
    inputs[index] = SimpleNamespace(shape=shape, dtype="f32")

    with pytest.raises(ValueError, match="shape must be exactly"):
        execute.gdn_execute_s512(
            *inputs,
            enabled=True,
            library_path=path,
            library_sha256=_library_sha256(path),
        )
    assert snapshots == []


@pytest.mark.parametrize("index", range(6))
def test_every_input_requires_exact_fp32_before_snapshot(monkeypatch, tmp_path, index):
    path = _library_file(tmp_path)
    _jax, _jnp, _library, _snapshot, snapshots, _loads = _fake_dependencies(monkeypatch)
    inputs = list(_abstract_inputs())
    inputs[index] = SimpleNamespace(shape=inputs[index].shape, dtype="bf16")

    with pytest.raises(TypeError, match="dtype must be exactly float32"):
        execute.gdn_execute_s512(
            *inputs,
            enabled=True,
            library_path=path,
            library_sha256=_library_sha256(path),
        )
    assert snapshots == []


@pytest.mark.parametrize("index", range(6))
def test_every_exposed_host_stride_must_be_exact_row_major(
    monkeypatch, tmp_path, index
):
    path = _library_file(tmp_path)
    _jax, _jnp, _library, _snapshot, snapshots, _loads = _fake_dependencies(monkeypatch)
    inputs = list(_abstract_inputs())
    shape = tuple(inputs[index].shape)
    wrong = list(execute._expected_c_strides(shape))
    wrong[-1] = 8
    inputs[index] = SimpleNamespace(shape=shape, dtype="f32", strides=tuple(wrong))

    with pytest.raises(ValueError, match="strides must be exactly"):
        execute.gdn_execute_s512(
            *inputs,
            enabled=True,
            library_path=path,
            library_sha256=_library_sha256(path),
        )
    assert snapshots == []


def test_enabled_call_builds_exact_two_result_typed_ffi_contract(monkeypatch, tmp_path):
    path = _library_file(tmp_path)
    digest = _library_sha256(path)
    jax, _jnp, _library, _snapshot, snapshots, _loads = _fake_dependencies(monkeypatch)
    inputs = _abstract_inputs()

    result = execute.gdn_execute_s512(
        *inputs,
        enabled=True,
        library_path=path,
        library_sha256=digest,
    )

    assert result == ("bf16-output", "fp32-final-state")
    assert snapshots == [(path, digest)]
    assert jax.shape_specs == [
        (execute.GDN_EXECUTE_S512_OUTPUT_SHAPE, "bf16"),
        (execute.GDN_EXECUTE_S512_STATE_SHAPE, "f32"),
    ]
    args, kwargs = jax.ffi.ffi_call_builds[0]
    assert args[0] == execute.GDN_EXECUTE_S512_TARGET
    assert tuple(spec.shape for spec in args[1]) == (
        execute.GDN_EXECUTE_S512_OUTPUT_SHAPE,
        execute.GDN_EXECUTE_S512_STATE_SHAPE,
    )
    assert kwargs == {
        "has_side_effect": False,
        "vmap_method": "sequential",
        "input_layouts": (
            (0, 1, 2, 3),
            (0, 1, 2, 3),
            (0, 1, 2, 3),
            (0, 1, 2, 3),
            (0, 1, 2),
            (0, 1, 2, 3),
        ),
        "output_layouts": ((0, 1, 2, 3), (0, 1, 2, 3)),
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
    assert "gdn_prepare_ffi" not in source
    assert "_sealed_loader._snapshot_library" in source
    assert "_sealed_loader._load_cdll(snapshot)" in source
    assert "_sealed_loader._library_lifetime_handles.append(library)" in source
    assert "ctypes.CDLL" not in source
    assert "JAX_PLATFORMS" not in source
    assert "HIP_VISIBLE_DEVICES" not in source


def test_hip_source_encodes_exact_geometry_head_pairing_and_recurrence():
    source = _HIP_SOURCE.read_text(encoding="utf-8")

    for marker in (
        "kTokens = 512",
        "kChunks = 8",
        "kChunk = 64",
        "kKeyHeads = 16",
        "kValueHeads = 32",
        "kHeadsPerKey = 2",
        "kHeadDimension = 128",
        "kWorkgroups = kValueHeads",
        "kLdsBytes == 49'408",
        "key_head = value_head / kHeadsPerKey",
        "prepared_u[ValueOffset(token, value_head, output_column)] - correction",
        "attention[row][previous] *",
        "reverse_decay[output_column] =",
        "final_gamma /",
        "reverse_decay[row] * corrected[row][output_column]",
        "final_gamma * final_state[state_index] + update",
        "const hip_bfloat16 converted(inter_chunk + intra_chunk)",
    ):
        assert marker in source
    assert "column <= row" in source
    assert source.count("final_gamma /") == 1
    assert source.count("hipLaunchKernelGGL") == 1


def test_hip_handler_gates_typed_shapes_bytes_layout_contract_and_disjoint_ranges():
    source = _HIP_SOURCE.read_text(encoding="utf-8")

    assert source.count("BufferR4<xla::ffi::F32>") >= 7
    assert "BufferR3<xla::ffi::F32> gamma" in source
    assert "ResultBufferR4<xla::ffi::BF16> output" in source
    assert source.count(".Arg<xla::ffi::") == 6
    assert source.count(".Ret<xla::ffi::") == 2
    for marker in (
        "query.size_bytes() != kQueryBytes",
        "key.size_bytes() != kQueryBytes",
        "prepared_u.size_bytes() != kPreparedBytes",
        "prepared_w.size_bytes() != kPreparedBytes",
        "gamma.size_bytes() != kGammaBytes",
        "initial_state.size_bytes() != kStateBytes",
        "output->size_bytes() != kOutputBytes",
        "final_state->size_bytes() != kStateBytes",
        "std::array<BufferRange, 8>",
        "AreDisjointNonNull(buffers)",
        "physical row-major strides are fixed",
        "PlatformStream<hipStream_t>",
    ):
        assert marker in source
    assert execute.GDN_EXECUTE_S512_TARGET in source


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


def test_execute_slice_contains_no_graph_or_command_buffer_path():
    combined = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in (_PYTHON_SOURCE, _HIP_SOURCE, _BUILD_SCRIPT)
    )
    for forbidden in (
        "hipgraph",
        "cudagraph",
        "command_buffer",
        "streambegincapture",
        "streamendcapture",
        "graphlaunch",
    ):
        assert forbidden not in combined


def test_build_rule_is_deterministic_fixed_gfx1100_and_external_only():
    source = _BUILD_SCRIPT.read_text(encoding="utf-8")

    assert 'EXPECTED_ARCH="gfx1100"' in source
    assert 'EXPECTED_BASENAME="libskyrl_gdn_execute_s512_gfx1100.so"' in source
    assert 'EXPECTED_SYMBOL="skyrl_gdn_execute_s512_f32_bf16_v1"' in source
    assert 'COMPILATION_UNIT_ID="736b79726c5f6764"' in source
    assert "SOURCE_DATE_EPOCH=0" in source
    assert '"--offload-arch=$EXPECTED_ARCH"' in source
    assert '"-cuid=$COMPILATION_UNIT_ID"' in source
    assert '"-ffile-prefix-map=$REPO_ROOT=/skyrl"' in source
    assert "-Wl,--build-id=sha1" in source
    assert "-fPIC -shared" in source
    assert "-fno-fast-math" in source
    assert "-ffast-math" not in source
    assert "output must be outside the source repository" in source
    assert "/dev/kfd" not in source
    assert "rocminfo" not in source
    assert "hipDeviceSynchronize" not in source
    assert stat.S_IMODE(os.stat(_BUILD_SCRIPT).st_mode) & stat.S_IXUSR
    syntax = subprocess.run(
        ["bash", "-n", str(_BUILD_SCRIPT)],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr


def test_build_rule_refuses_wrong_arch_missing_argument_and_unsafe_outputs(tmp_path):
    wrong_arch = subprocess.run(
        [str(_BUILD_SCRIPT)],
        cwd=_REPO,
        env={**os.environ, "SKYRL_ROCM_ARCH": "gfx1101"},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert wrong_arch.returncode == 2
    assert "refusing architecture" in wrong_arch.stderr

    missing_argument = subprocess.run(
        [str(_BUILD_SCRIPT)],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert missing_argument.returncode == 2
    assert "usage:" in missing_argument.stderr

    inside_repo = _REPO / "libskyrl_gdn_execute_s512_gfx1100.so"
    assert not inside_repo.exists()
    unsafe_parent = subprocess.run(
        [str(_BUILD_SCRIPT), str(inside_repo)],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert unsafe_parent.returncode == 2
    assert "outside the source repository" in unsafe_parent.stderr

    existing = _library_file(tmp_path)
    overwrite = subprocess.run(
        [str(_BUILD_SCRIPT), str(existing)],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert overwrite.returncode == 2
    assert "refusing to overwrite output" in overwrite.stderr
