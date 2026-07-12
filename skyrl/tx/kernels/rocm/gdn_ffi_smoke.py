"""Explicitly opt-in ROCm typed-FFI ABI smoke for a future GDN kernel.

This module is not a Gated DeltaNet implementation and is not a performance
path.  Its only accelerated operation copies one exact, contiguous BF16 GDN
superblock-shaped array to a distinct output buffer.  The smoke establishes
shared-library lifetime, typed-FFI registration, row-major layout, and use of
JAX's supplied ROCm stream before any recurrence math is added.

The default :func:`gdn_ffi_smoke_copy` path is an identity fallback.  Importing
this module and using that fallback do not import JAX, load a shared library,
register a target, or inspect an accelerator.  FFI use requires both an exact
``enabled=True`` opt-in and an explicit absolute path to a locally built
library plus its exact lowercase SHA-256.  The approved bytes are loaded only
from a private sealed in-memory snapshot, never from the validated pathname.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import importlib
import operator
import os
import re
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GDN_FFI_SMOKE_TARGET = "skyrl_gdn_ffi_smoke_bf16_copy_v1"
GDN_FFI_SMOKE_SHAPE = (1, 1024, 32, 128)
GDN_FFI_SMOKE_BYTES = 8 * 1024**2

_REGISTRATION_API_VERSION = 1
_CUSTOM_CALL_API_VERSION = 4
_ROCM_PLATFORM = "ROCM"
_ROW_MAJOR_LAYOUT = (0, 1, 2, 3)
_LIBRARY_BASENAME = "libskyrl_gdn_ffi_smoke_gfx1100.so"
_SNAPSHOT_MODE = 0o600
_COPY_CHUNK_BYTES = 1024 * 1024
_MFD_FLAGS = getattr(os, "MFD_CLOEXEC", 0x0001) | getattr(
    os, "MFD_ALLOW_SEALING", 0x0002
)
_REQUIRED_SEALS = (
    getattr(fcntl, "F_SEAL_WRITE", 0x0008)
    | getattr(fcntl, "F_SEAL_GROW", 0x0004)
    | getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
    | getattr(fcntl, "F_SEAL_SEAL", 0x0001)
)


@dataclass(frozen=True, slots=True)
class GdnFfiSmokeRegistration:
    """Public, library-handle-free description of one process registration."""

    library_path: Path
    library_sha256: str
    snapshot_sha256: str
    snapshot_size_bytes: int
    snapshot_mode: int
    snapshot_seals: int
    sealed_snapshot: bool
    snapshot_fd_retained: bool
    target_name: str = GDN_FFI_SMOKE_TARGET
    platform: str = _ROCM_PLATFORM
    registration_api_version: int = _REGISTRATION_API_VERSION
    custom_call_api_version: int = _CUSTOM_CALL_API_VERSION


@dataclass(slots=True)
class _RegistrationState:
    public: GdnFfiSmokeRegistration
    library: Any
    snapshot: _SealedLibrarySnapshot


@dataclass(frozen=True, slots=True)
class _SealedLibrarySnapshot:
    original_path: Path
    source_identity: tuple[int, int, int, int, int, int, int]
    sha256: str
    size_bytes: int
    mode: int
    seals: int
    fd: int

    @property
    def proc_path(self) -> str:
        return f"/proc/self/fd/{self.fd}"


_registration_lock = threading.Lock()
_registration_state: _RegistrationState | None = None
_library_lifetime_handles: list[Any] = []
_library_lifetime_fds: list[int] = []


def _require_exact_bool(value: bool, name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact bool")


def _validate_library_sha256(library_sha256: str | None) -> str:
    if (
        not isinstance(library_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", library_sha256) is None
    ):
        raise ValueError(
            "gdn_ffi_smoke library_sha256 must be exactly 64 lowercase hexadecimal digits"
        )
    return library_sha256


def _validate_library_path(library_path: str | os.PathLike[str] | None) -> Path:
    if library_path is None:
        raise ValueError("gdn_ffi_smoke requires an explicit shared-library path")
    if isinstance(library_path, bytes) or not isinstance(
        library_path, (str, os.PathLike)
    ):
        raise TypeError("library_path must be a string or path-like object")

    try:
        candidate = Path(library_path)
    except TypeError as error:
        raise TypeError(
            "library_path must resolve to a text filesystem path"
        ) from error
    if not candidate.is_absolute():
        raise ValueError("gdn_ffi_smoke library_path must be absolute")
    if candidate.name != _LIBRARY_BASENAME:
        raise ValueError(
            "gdn_ffi_smoke library must use the exact name "
            "libskyrl_gdn_ffi_smoke_gfx1100.so"
        )
    try:
        info = candidate.lstat()
    except FileNotFoundError as error:
        raise ValueError("gdn_ffi_smoke shared library does not exist") from error
    except OSError as error:
        raise ValueError("gdn_ffi_smoke shared library cannot be inspected") from error
    if stat.S_ISLNK(info.st_mode):
        raise ValueError("gdn_ffi_smoke shared library must not be a symbolic link")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("gdn_ffi_smoke shared library must be a regular file")
    if info.st_uid != os.getuid():
        raise ValueError(
            "gdn_ffi_smoke shared library must be owned by the current user"
        )
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError(
            "gdn_ffi_smoke shared library must not be group- or world-writable"
        )

    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError("gdn_ffi_smoke shared library cannot be resolved") from error
    if resolved != candidate:
        raise ValueError(
            "gdn_ffi_smoke library_path must not traverse symbolic links or '..'"
        )
    return resolved


def _import_jax() -> tuple[Any, Any]:
    """Import JAX lazily, after explicit opt-in and path validation."""
    return importlib.import_module("jax"), importlib.import_module("jax.numpy")


def _source_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
        int(info.st_uid),
        int(info.st_mode),
    )


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise RuntimeError("could not copy the approved FFI library snapshot")
        view = view[written:]


def _snapshot_library(
    library_path: Path, expected_sha256: str
) -> _SealedLibrarySnapshot:
    """Copy, hash, seal, and retain one immutable dlopen input."""
    if not callable(getattr(os, "memfd_create", None)):
        raise RuntimeError("gdn_ffi_smoke requires Linux memfd_create")
    for name in (
        "F_ADD_SEALS",
        "F_GET_SEALS",
        "F_SEAL_WRITE",
        "F_SEAL_GROW",
        "F_SEAL_SHRINK",
        "F_SEAL_SEAL",
    ):
        if not isinstance(getattr(fcntl, name, None), int):
            raise RuntimeError("gdn_ffi_smoke requires Linux memfd sealing support")

    path_info = library_path.lstat()
    source_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    source_fd = os.open(library_path, source_flags)
    try:
        source_before = os.fstat(source_fd)
        if _source_identity(source_before) != _source_identity(path_info):
            raise RuntimeError("gdn_ffi_smoke library changed before snapshot open")
        if not stat.S_ISREG(source_before.st_mode):
            raise ValueError("gdn_ffi_smoke snapshot source must be a regular file")
        if source_before.st_uid != os.getuid():
            raise ValueError("gdn_ffi_smoke snapshot source must be user-owned")
        if source_before.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError("gdn_ffi_smoke snapshot source must remain private")
        if source_before.st_size <= 0:
            raise ValueError("gdn_ffi_smoke shared library must be non-empty")

        snapshot_fd = os.memfd_create("skyrl-gdn-ffi-smoke", _MFD_FLAGS)
        # Never let a snapshot descriptor disappear while process-global FFI
        # state may have observed it, including all later failure paths.
        _library_lifetime_fds.append(snapshot_fd)
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_fd, _COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            _write_all(snapshot_fd, chunk)
            copied += len(chunk)

        source_after = os.fstat(source_fd)
        source_identity = _source_identity(source_before)
        if (
            _source_identity(source_after) != source_identity
            or copied != source_before.st_size
        ):
            raise RuntimeError("gdn_ffi_smoke library changed while being snapshotted")

        os.fchmod(snapshot_fd, _SNAPSHOT_MODE)
        fcntl.fcntl(snapshot_fd, fcntl.F_ADD_SEALS, _REQUIRED_SEALS)
        observed_seals = int(fcntl.fcntl(snapshot_fd, fcntl.F_GET_SEALS))
        if observed_seals & _REQUIRED_SEALS != _REQUIRED_SEALS:
            raise RuntimeError("gdn_ffi_smoke memfd does not have every required seal")
        snapshot_info = os.fstat(snapshot_fd)
        if (
            not stat.S_ISREG(snapshot_info.st_mode)
            or stat.S_IMODE(snapshot_info.st_mode) != _SNAPSHOT_MODE
            or snapshot_info.st_size != copied
        ):
            raise RuntimeError("gdn_ffi_smoke sealed snapshot metadata is invalid")
        observed_sha256 = digest.hexdigest()
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                "gdn_ffi_smoke sealed snapshot SHA-256 does not match library_sha256"
            )
        os.lseek(snapshot_fd, 0, os.SEEK_SET)
        return _SealedLibrarySnapshot(
            original_path=library_path,
            source_identity=source_identity,
            sha256=observed_sha256,
            size_bytes=copied,
            mode=stat.S_IMODE(snapshot_info.st_mode),
            seals=observed_seals,
            fd=snapshot_fd,
        )
    finally:
        os.close(source_fd)


def _load_cdll(snapshot: _SealedLibrarySnapshot) -> Any:
    mode = getattr(os, "RTLD_LOCAL", 0) | getattr(os, "RTLD_NOW", 0)
    return ctypes.CDLL(snapshot.proc_path, mode=mode)


def _ffi_namespace(jax_module: Any) -> Any:
    ffi = getattr(jax_module, "ffi", None)
    required = ("ffi_call", "pycapsule", "register_ffi_target")
    if ffi is None or any(not callable(getattr(ffi, name, None)) for name in required):
        raise RuntimeError("installed JAX does not expose the required typed-FFI API")
    if not callable(getattr(jax_module, "ShapeDtypeStruct", None)):
        raise RuntimeError("installed JAX does not expose ShapeDtypeStruct")
    return ffi


def _register_enabled(
    library_path: Path, library_sha256: str, jax_module: Any
) -> GdnFfiSmokeRegistration:
    global _registration_state

    ffi = _ffi_namespace(jax_module)
    with _registration_lock:
        if _registration_state is not None:
            if (
                _registration_state.public.library_path != library_path
                or _registration_state.public.library_sha256 != library_sha256
            ):
                raise RuntimeError(
                    "gdn_ffi_smoke target is already registered from a different library identity"
                )
            return _registration_state.public

        snapshot = _snapshot_library(library_path, library_sha256)
        try:
            library = _load_cdll(snapshot)
        except OSError as error:
            raise RuntimeError(
                "could not load the sealed gdn_ffi_smoke library snapshot"
            ) from error
        # Retain even across a later symbol/capsule/registration failure.  A
        # third-party registry must never be left with a pointer into an
        # unloaded object if its error path partially accepted the capsule.
        _library_lifetime_handles.append(library)
        try:
            handler = getattr(library, GDN_FFI_SMOKE_TARGET)
        except AttributeError as error:
            raise RuntimeError(
                "gdn_ffi_smoke library is missing its exact handler symbol"
            ) from error
        handler.restype = ctypes.c_void_p
        handler.argtypes = (ctypes.c_void_p,)
        capsule = ffi.pycapsule(handler)
        ffi.register_ffi_target(
            GDN_FFI_SMOKE_TARGET,
            capsule,
            platform=_ROCM_PLATFORM,
            api_version=_REGISTRATION_API_VERSION,
        )
        public = GdnFfiSmokeRegistration(
            library_path=library_path,
            library_sha256=library_sha256,
            snapshot_sha256=snapshot.sha256,
            snapshot_size_bytes=snapshot.size_bytes,
            snapshot_mode=snapshot.mode,
            snapshot_seals=snapshot.seals,
            sealed_snapshot=True,
            snapshot_fd_retained=snapshot.fd in _library_lifetime_fds,
        )
        # The PyCapsule contains only a function pointer.  Retain CDLL for the
        # complete process lifetime so that pointer can never outlive its code.
        _registration_state = _RegistrationState(
            public=public, library=library, snapshot=snapshot
        )
        return public


def register_gdn_ffi_smoke(
    library_path: str | os.PathLike[str] | None = None,
    *,
    library_sha256: str | None = None,
    enabled: bool = False,
) -> GdnFfiSmokeRegistration:
    """Load and register the smoke handler only after exact explicit opt-in.

    Registration is process-global and idempotent only for the same canonical
    path and exact digest.  It is intentionally restricted to platform
    ``ROCM`` and typed-FFI registration API version 1.
    """
    _require_exact_bool(enabled, "enabled")
    if not enabled:
        raise RuntimeError("gdn_ffi_smoke registration is disabled by default")
    validated_sha256 = _validate_library_sha256(library_sha256)
    validated_path = _validate_library_path(library_path)
    jax_module, _ = _import_jax()
    return _register_enabled(validated_path, validated_sha256, jax_module)


def _validate_smoke_value(value: Any, jnp_module: Any) -> None:
    raw_shape = getattr(value, "shape", None)
    if raw_shape is None:
        raise TypeError("gdn_ffi_smoke input must expose shape and dtype")
    try:
        dimensions = []
        for dimension in raw_shape:
            if type(dimension) is bool:
                raise TypeError("boolean dimension")
            dimensions.append(operator.index(dimension))
        shape = tuple(dimensions)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "gdn_ffi_smoke input shape must contain concrete integers"
        ) from error
    if shape != GDN_FFI_SMOKE_SHAPE:
        raise ValueError(
            f"gdn_ffi_smoke input shape must be exactly {GDN_FFI_SMOKE_SHAPE}, got {shape}"
        )
    if getattr(value, "dtype", None) != jnp_module.bfloat16:
        raise TypeError("gdn_ffi_smoke input dtype must be exactly bfloat16")


def _call_registered(value: Any, jax_module: Any, jnp_module: Any) -> Any:
    ffi = _ffi_namespace(jax_module)
    result = jax_module.ShapeDtypeStruct(GDN_FFI_SMOKE_SHAPE, jnp_module.bfloat16)
    call = ffi.ffi_call(
        GDN_FFI_SMOKE_TARGET,
        result,
        has_side_effect=False,
        vmap_method="sequential",
        input_layouts=(_ROW_MAJOR_LAYOUT,),
        output_layouts=_ROW_MAJOR_LAYOUT,
        input_output_aliases=None,
        custom_call_api_version=_CUSTOM_CALL_API_VERSION,
    )
    return call(value)


def gdn_ffi_smoke_copy(
    value: Any,
    *,
    enabled: bool = False,
    library_path: str | os.PathLike[str] | None = None,
    library_sha256: str | None = None,
) -> Any:
    """Return ``value`` by default or call the opt-in ROCm copy smoke.

    The disabled identity fallback deliberately performs no validation because
    it must remain a zero-dependency, zero-registration path.  The enabled path
    accepts only exact BF16 ``[1, 1024, 32, 128]`` values, an explicit local
    library path, and its exact lowercase digest.  It has no custom
    differentiation rule and must not be wired into model execution.
    """
    _require_exact_bool(enabled, "enabled")
    if not enabled:
        if library_path is not None or library_sha256 is not None:
            raise ValueError(
                "library_path/library_sha256 are invalid while gdn_ffi_smoke is disabled"
            )
        return value

    validated_sha256 = _validate_library_sha256(library_sha256)
    validated_path = _validate_library_path(library_path)
    jax_module, jnp_module = _import_jax()
    _validate_smoke_value(value, jnp_module)
    _register_enabled(validated_path, validated_sha256, jax_module)
    return _call_registered(value, jax_module, jnp_module)
