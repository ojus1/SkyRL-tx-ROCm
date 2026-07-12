"""Default-off typed FFI wrapper for exact S512 GDN WY preparation.

This experimental operation accepts already transformed FP32 inputs: K is
masked and L2-normalized, while V/g/beta are masked and promoted before this
boundary.  It prepares ``U``, ``W``, and within-chunk cumulative decay ``G``
for eight 64-token chunks.  It does not execute the recurrence, define a custom
VJP, or participate in model dispatch.

The enabled path requires an explicit canonical shared-library path and exact
lowercase SHA-256.  It reuses the previously gated GDN smoke module's sealed
``memfd`` snapshot loader and process-lifetime retention.  The approved object
is therefore loaded from ``/proc/self/fd/<retained-fd>``, never from its
validated pathname.  JAX remains a lazy dependency after exact opt-in plus
path and digest validation; the library is snapshotted only after exact input
validation.
"""

from __future__ import annotations

import ctypes
import importlib
import operator
import os
import re
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skyrl.tx.kernels.rocm import gdn_ffi_smoke as _sealed_loader

GDN_PREPARE_S512_TARGET = "skyrl_gdn_prepare_s512_f32_v1"
GDN_PREPARE_S512_KEY_SHAPE = (1, 512, 16, 128)
GDN_PREPARE_S512_VALUE_SHAPE = (1, 512, 32, 128)
GDN_PREPARE_S512_GATE_SHAPE = (1, 512, 32)
GDN_PREPARE_S512_KEY_BYTES = 4 * 1024**2
GDN_PREPARE_S512_VALUE_BYTES = 8 * 1024**2
GDN_PREPARE_S512_GATE_BYTES = 64 * 1024

_REGISTRATION_API_VERSION = 1
_CUSTOM_CALL_API_VERSION = 4
_ROCM_PLATFORM = "ROCM"
_LIBRARY_BASENAME = "libskyrl_gdn_prepare_s512_gfx1100.so"
_ROW_MAJOR_R4 = (0, 1, 2, 3)
_ROW_MAJOR_R3 = (0, 1, 2)


@dataclass(frozen=True, slots=True)
class GdnPrepareS512Registration:
    """Library-handle-free identity for one process-global registration."""

    library_path: Path
    library_sha256: str
    snapshot_sha256: str
    snapshot_size_bytes: int
    snapshot_mode: int
    snapshot_seals: int
    sealed_snapshot: bool
    snapshot_fd_retained: bool
    target_name: str = GDN_PREPARE_S512_TARGET
    platform: str = _ROCM_PLATFORM
    registration_api_version: int = _REGISTRATION_API_VERSION
    custom_call_api_version: int = _CUSTOM_CALL_API_VERSION


@dataclass(slots=True)
class _RegistrationState:
    public: GdnPrepareS512Registration
    library: Any
    snapshot: Any


_registration_lock = threading.Lock()
_registration_state: _RegistrationState | None = None


def _require_exact_bool(value: bool, name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact bool")


def _validate_library_sha256(library_sha256: str | None) -> str:
    if (
        not isinstance(library_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", library_sha256) is None
    ):
        raise ValueError(
            "gdn_prepare_s512 library_sha256 must be exactly 64 lowercase "
            "hexadecimal digits"
        )
    return library_sha256


def _validate_library_path(library_path: str | os.PathLike[str] | None) -> Path:
    if library_path is None:
        raise ValueError("gdn_prepare_s512 requires an explicit shared-library path")
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
        raise ValueError("gdn_prepare_s512 library_path must be absolute")
    if candidate.name != _LIBRARY_BASENAME:
        raise ValueError(
            "gdn_prepare_s512 library must use the exact name "
            "libskyrl_gdn_prepare_s512_gfx1100.so"
        )
    try:
        info = candidate.lstat()
    except FileNotFoundError as error:
        raise ValueError("gdn_prepare_s512 shared library does not exist") from error
    except OSError as error:
        raise ValueError(
            "gdn_prepare_s512 shared library cannot be inspected"
        ) from error
    if stat.S_ISLNK(info.st_mode):
        raise ValueError("gdn_prepare_s512 shared library must not be a symbolic link")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("gdn_prepare_s512 shared library must be a regular file")
    if info.st_uid != os.getuid():
        raise ValueError("gdn_prepare_s512 shared library must be user-owned")
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError(
            "gdn_prepare_s512 shared library must not be group- or world-writable"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(
            "gdn_prepare_s512 shared library cannot be resolved"
        ) from error
    if resolved != candidate:
        raise ValueError(
            "gdn_prepare_s512 library_path must not traverse symbolic links or '..'"
        )
    return resolved


def _import_jax() -> tuple[Any, Any]:
    """Import JAX only after explicit opt-in, digest, and path checks."""
    return importlib.import_module("jax"), importlib.import_module("jax.numpy")


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
) -> GdnPrepareS512Registration:
    global _registration_state

    ffi = _ffi_namespace(jax_module)
    with _registration_lock:
        if _registration_state is not None:
            if (
                _registration_state.public.library_path != library_path
                or _registration_state.public.library_sha256 != library_sha256
            ):
                raise RuntimeError(
                    "gdn_prepare_s512 target is already registered from a different "
                    "library identity"
                )
            return _registration_state.public

        # Reuse the independently gated one-pass hash, sealed-memfd snapshot,
        # and retained descriptor implementation. Do not pathname-dlopen here.
        snapshot = _sealed_loader._snapshot_library(library_path, library_sha256)
        try:
            library = _sealed_loader._load_cdll(snapshot)
        except OSError as error:
            raise RuntimeError(
                "could not load the sealed gdn_prepare_s512 library snapshot"
            ) from error
        # Retain even if symbol lookup or third-party registration fails.
        _sealed_loader._library_lifetime_handles.append(library)
        try:
            handler = getattr(library, GDN_PREPARE_S512_TARGET)
        except AttributeError as error:
            raise RuntimeError(
                "gdn_prepare_s512 library is missing its exact handler symbol"
            ) from error
        handler.restype = ctypes.c_void_p
        handler.argtypes = (ctypes.c_void_p,)
        capsule = ffi.pycapsule(handler)
        ffi.register_ffi_target(
            GDN_PREPARE_S512_TARGET,
            capsule,
            platform=_ROCM_PLATFORM,
            api_version=_REGISTRATION_API_VERSION,
        )
        public = GdnPrepareS512Registration(
            library_path=library_path,
            library_sha256=library_sha256,
            snapshot_sha256=snapshot.sha256,
            snapshot_size_bytes=snapshot.size_bytes,
            snapshot_mode=snapshot.mode,
            snapshot_seals=snapshot.seals,
            sealed_snapshot=True,
            snapshot_fd_retained=(snapshot.fd in _sealed_loader._library_lifetime_fds),
        )
        _registration_state = _RegistrationState(
            public=public,
            library=library,
            snapshot=snapshot,
        )
        return public


def register_gdn_prepare_s512(
    library_path: str | os.PathLike[str] | None = None,
    *,
    library_sha256: str | None = None,
    enabled: bool = False,
) -> GdnPrepareS512Registration:
    """Register the exact prepare handler after explicit opt-in."""
    _require_exact_bool(enabled, "enabled")
    if not enabled:
        raise RuntimeError("gdn_prepare_s512 registration is disabled by default")
    validated_sha256 = _validate_library_sha256(library_sha256)
    validated_path = _validate_library_path(library_path)
    jax_module, _ = _import_jax()
    return _register_enabled(validated_path, validated_sha256, jax_module)


def _concrete_shape(value: Any, name: str) -> tuple[int, ...]:
    raw_shape = getattr(value, "shape", None)
    if raw_shape is None:
        raise TypeError(f"gdn_prepare_s512 {name} must expose shape and dtype")
    try:
        dimensions = []
        for dimension in raw_shape:
            if type(dimension) is bool:
                raise TypeError("boolean dimension")
            dimensions.append(operator.index(dimension))
        return tuple(dimensions)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"gdn_prepare_s512 {name} shape must contain concrete integers"
        ) from error


def _validate_value(
    value: Any,
    name: str,
    expected_shape: tuple[int, ...],
    float32_dtype: Any,
) -> None:
    shape = _concrete_shape(value, name)
    if shape != expected_shape:
        raise ValueError(
            f"gdn_prepare_s512 {name} shape must be exactly {expected_shape}, "
            f"got {shape}"
        )
    if getattr(value, "dtype", None) != float32_dtype:
        raise TypeError(f"gdn_prepare_s512 {name} dtype must be exactly float32")


def _call_registered(
    key: Any,
    value: Any,
    g: Any,
    beta: Any,
    jax_module: Any,
    jnp_module: Any,
) -> tuple[Any, Any, Any]:
    ffi = _ffi_namespace(jax_module)
    results = (
        jax_module.ShapeDtypeStruct(GDN_PREPARE_S512_VALUE_SHAPE, jnp_module.float32),
        jax_module.ShapeDtypeStruct(GDN_PREPARE_S512_VALUE_SHAPE, jnp_module.float32),
        jax_module.ShapeDtypeStruct(GDN_PREPARE_S512_GATE_SHAPE, jnp_module.float32),
    )
    call = ffi.ffi_call(
        GDN_PREPARE_S512_TARGET,
        results,
        has_side_effect=False,
        vmap_method="sequential",
        input_layouts=(
            _ROW_MAJOR_R4,
            _ROW_MAJOR_R4,
            _ROW_MAJOR_R3,
            _ROW_MAJOR_R3,
        ),
        output_layouts=(_ROW_MAJOR_R4, _ROW_MAJOR_R4, _ROW_MAJOR_R3),
        input_output_aliases=None,
        custom_call_api_version=_CUSTOM_CALL_API_VERSION,
    )
    prepared_u, prepared_w, gamma = call(key, value, g, beta)
    return prepared_u, prepared_w, gamma


def gdn_prepare_s512(
    key: Any,
    value: Any,
    g: Any,
    beta: Any,
    *,
    enabled: bool = False,
    library_path: str | os.PathLike[str] | None = None,
    library_sha256: str | None = None,
) -> tuple[Any, Any, Any]:
    """Prepare exact FP32 S512 ``(U, W, G)`` through opt-in ROCm FFI.

    Inputs must already have masked positions zeroed in K, V, g, and beta, and
    unmasked K rows must already have the model's L2 normalization. K is kept
    at 16 key heads; value/g/beta retain 32 value heads, paired as
    ``value_head // 2``. The operation has no fallback because manufacturing
    prepared values on the disabled path could silently select different math.
    """
    _require_exact_bool(enabled, "enabled")
    if not enabled:
        if library_path is not None or library_sha256 is not None:
            raise ValueError(
                "library_path/library_sha256 are invalid while "
                "gdn_prepare_s512 is disabled"
            )
        raise RuntimeError("gdn_prepare_s512 is disabled by default")

    validated_sha256 = _validate_library_sha256(library_sha256)
    validated_path = _validate_library_path(library_path)
    jax_module, jnp_module = _import_jax()
    for item, name, shape in (
        (key, "key", GDN_PREPARE_S512_KEY_SHAPE),
        (value, "value", GDN_PREPARE_S512_VALUE_SHAPE),
        (g, "g", GDN_PREPARE_S512_GATE_SHAPE),
        (beta, "beta", GDN_PREPARE_S512_GATE_SHAPE),
    ):
        _validate_value(item, name, shape, jnp_module.float32)

    _register_enabled(validated_path, validated_sha256, jax_module)
    return _call_registered(key, value, g, beta, jax_module, jnp_module)
