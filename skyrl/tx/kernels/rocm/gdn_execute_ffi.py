"""Default-off typed FFI wrapper for exact S512 GDN execution.

This experimental operation consumes the exact FP32 boundary established by
``gdn_execute_oracle``: normalized/scaled query and key, prepared U/W, the
within-chunk cumulative decay gamma, and an incoming recurrent state.  It
returns a BF16 token-major output and an FP32 final state.  It does not prepare
U/W/gamma, define a transpose, or participate in model dispatch.

The enabled path requires an explicit canonical shared-library path and exact
lowercase SHA-256.  It reuses the previously gated GDN smoke module's sealed
``memfd`` snapshot loader and process-lifetime retention.  JAX remains a lazy
dependency until exact opt-in, library validation, and input ABI validation.
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

GDN_EXECUTE_S512_TARGET = "skyrl_gdn_execute_s512_f32_bf16_v1"
GDN_EXECUTE_S512_QUERY_SHAPE = (1, 512, 16, 128)
GDN_EXECUTE_S512_PREPARED_SHAPE = (1, 512, 32, 128)
GDN_EXECUTE_S512_GAMMA_SHAPE = (1, 512, 32)
GDN_EXECUTE_S512_STATE_SHAPE = (1, 32, 128, 128)
GDN_EXECUTE_S512_OUTPUT_SHAPE = GDN_EXECUTE_S512_PREPARED_SHAPE
GDN_EXECUTE_S512_INPUT_BYTES = 27_328_512
GDN_EXECUTE_S512_OUTPUT_BYTES = 6_291_456

_REGISTRATION_API_VERSION = 1
_CUSTOM_CALL_API_VERSION = 4
_ROCM_PLATFORM = "ROCM"
_LIBRARY_BASENAME = "libskyrl_gdn_execute_s512_gfx1100.so"
_ROW_MAJOR_R4 = (0, 1, 2, 3)
_ROW_MAJOR_R3 = (0, 1, 2)
_F32_ITEMSIZE = 4


@dataclass(frozen=True, slots=True)
class GdnExecuteS512Registration:
    """Library-handle-free identity for one process-global registration."""

    library_path: Path
    library_sha256: str
    snapshot_sha256: str
    snapshot_size_bytes: int
    snapshot_mode: int
    snapshot_seals: int
    sealed_snapshot: bool
    snapshot_fd_retained: bool
    target_name: str = GDN_EXECUTE_S512_TARGET
    platform: str = _ROCM_PLATFORM
    registration_api_version: int = _REGISTRATION_API_VERSION
    custom_call_api_version: int = _CUSTOM_CALL_API_VERSION


@dataclass(slots=True)
class _RegistrationState:
    public: GdnExecuteS512Registration
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
            "gdn_execute_s512 library_sha256 must be exactly 64 lowercase "
            "hexadecimal digits"
        )
    return library_sha256


def _validate_library_path(library_path: str | os.PathLike[str] | None) -> Path:
    if library_path is None:
        raise ValueError("gdn_execute_s512 requires an explicit shared-library path")
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
        raise ValueError("gdn_execute_s512 library_path must be absolute")
    if candidate.name != _LIBRARY_BASENAME:
        raise ValueError(
            "gdn_execute_s512 library must use the exact name "
            "libskyrl_gdn_execute_s512_gfx1100.so"
        )
    try:
        info = candidate.lstat()
    except FileNotFoundError as error:
        raise ValueError("gdn_execute_s512 shared library does not exist") from error
    except OSError as error:
        raise ValueError(
            "gdn_execute_s512 shared library cannot be inspected"
        ) from error
    if stat.S_ISLNK(info.st_mode):
        raise ValueError("gdn_execute_s512 shared library must not be a symbolic link")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("gdn_execute_s512 shared library must be a regular file")
    if info.st_uid != os.getuid():
        raise ValueError("gdn_execute_s512 shared library must be user-owned")
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError(
            "gdn_execute_s512 shared library must not be group- or world-writable"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(
            "gdn_execute_s512 shared library cannot be resolved"
        ) from error
    if resolved != candidate:
        raise ValueError(
            "gdn_execute_s512 library_path must not traverse symbolic links or '..'"
        )
    return resolved


def _import_jax() -> tuple[Any, Any]:
    """Import JAX only after explicit opt-in and library checks."""
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
) -> GdnExecuteS512Registration:
    global _registration_state

    ffi = _ffi_namespace(jax_module)
    with _registration_lock:
        if _registration_state is not None:
            if (
                _registration_state.public.library_path != library_path
                or _registration_state.public.library_sha256 != library_sha256
            ):
                raise RuntimeError(
                    "gdn_execute_s512 target is already registered from a "
                    "different library identity"
                )
            return _registration_state.public

        snapshot = _sealed_loader._snapshot_library(library_path, library_sha256)
        try:
            library = _sealed_loader._load_cdll(snapshot)
        except OSError as error:
            raise RuntimeError(
                "could not load the sealed gdn_execute_s512 library snapshot"
            ) from error
        # Retain even if a later lookup or third-party registration step fails.
        _sealed_loader._library_lifetime_handles.append(library)
        try:
            handler = getattr(library, GDN_EXECUTE_S512_TARGET)
        except AttributeError as error:
            raise RuntimeError(
                "gdn_execute_s512 library is missing its exact handler symbol"
            ) from error
        handler.restype = ctypes.c_void_p
        handler.argtypes = (ctypes.c_void_p,)
        capsule = ffi.pycapsule(handler)
        ffi.register_ffi_target(
            GDN_EXECUTE_S512_TARGET,
            capsule,
            platform=_ROCM_PLATFORM,
            api_version=_REGISTRATION_API_VERSION,
        )
        public = GdnExecuteS512Registration(
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


def register_gdn_execute_s512(
    library_path: str | os.PathLike[str] | None = None,
    *,
    library_sha256: str | None = None,
    enabled: bool = False,
) -> GdnExecuteS512Registration:
    """Register the exact execute handler after explicit opt-in."""
    _require_exact_bool(enabled, "enabled")
    if not enabled:
        raise RuntimeError("gdn_execute_s512 registration is disabled by default")
    validated_sha256 = _validate_library_sha256(library_sha256)
    validated_path = _validate_library_path(library_path)
    jax_module, _ = _import_jax()
    return _register_enabled(validated_path, validated_sha256, jax_module)


def _concrete_shape(value: Any, name: str) -> tuple[int, ...]:
    raw_shape = getattr(value, "shape", None)
    if raw_shape is None:
        raise TypeError(f"gdn_execute_s512 {name} must expose shape and dtype")
    try:
        dimensions = []
        for dimension in raw_shape:
            if type(dimension) is bool:
                raise TypeError("boolean dimension")
            dimensions.append(operator.index(dimension))
        return tuple(dimensions)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"gdn_execute_s512 {name} shape must contain concrete integers"
        ) from error


def _expected_c_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = _F32_ITEMSIZE
    reversed_strides = []
    for dimension in reversed(shape):
        reversed_strides.append(stride)
        stride *= dimension
    return tuple(reversed(reversed_strides))


def _validate_exposed_strides(
    value: Any, name: str, expected_shape: tuple[int, ...]
) -> None:
    """Reject non-row-major host views; XLA fixes absent device strides.

    JAX arrays and tracers generally expose no byte-stride attribute.  Their
    native ABI is nevertheless fixed by ``ffi_call``'s explicit physical
    layouts.  NumPy and other host views that do expose strides must already be
    exact C-order buffers and cannot rely on an implicit materializing copy.
    """
    raw_strides = getattr(value, "strides", None)
    if raw_strides is None:
        return
    try:
        strides = tuple(operator.index(stride) for stride in raw_strides)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"gdn_execute_s512 {name} strides must contain concrete integers"
        ) from error
    expected = _expected_c_strides(expected_shape)
    if strides != expected:
        raise ValueError(
            f"gdn_execute_s512 {name} strides must be exactly {expected}, got {strides}"
        )


def _validate_value(
    value: Any,
    name: str,
    expected_shape: tuple[int, ...],
    float32_dtype: Any,
) -> None:
    shape = _concrete_shape(value, name)
    if shape != expected_shape:
        raise ValueError(
            f"gdn_execute_s512 {name} shape must be exactly {expected_shape}, "
            f"got {shape}"
        )
    if getattr(value, "dtype", None) != float32_dtype:
        raise TypeError(f"gdn_execute_s512 {name} dtype must be exactly float32")
    _validate_exposed_strides(value, name, expected_shape)


def _call_registered(
    query: Any,
    key: Any,
    prepared_u: Any,
    prepared_w: Any,
    gamma: Any,
    initial_state: Any,
    jax_module: Any,
    jnp_module: Any,
) -> tuple[Any, Any]:
    ffi = _ffi_namespace(jax_module)
    results = (
        jax_module.ShapeDtypeStruct(GDN_EXECUTE_S512_OUTPUT_SHAPE, jnp_module.bfloat16),
        jax_module.ShapeDtypeStruct(GDN_EXECUTE_S512_STATE_SHAPE, jnp_module.float32),
    )
    call = ffi.ffi_call(
        GDN_EXECUTE_S512_TARGET,
        results,
        has_side_effect=False,
        vmap_method="sequential",
        input_layouts=(
            _ROW_MAJOR_R4,
            _ROW_MAJOR_R4,
            _ROW_MAJOR_R4,
            _ROW_MAJOR_R4,
            _ROW_MAJOR_R3,
            _ROW_MAJOR_R4,
        ),
        output_layouts=(_ROW_MAJOR_R4, _ROW_MAJOR_R4),
        input_output_aliases=None,
        custom_call_api_version=_CUSTOM_CALL_API_VERSION,
    )
    output, final_state = call(query, key, prepared_u, prepared_w, gamma, initial_state)
    return output, final_state


def gdn_execute_s512(
    query: Any,
    key: Any,
    prepared_u: Any,
    prepared_w: Any,
    gamma: Any,
    initial_state: Any,
    *,
    enabled: bool = False,
    library_path: str | os.PathLike[str] | None = None,
    library_sha256: str | None = None,
) -> tuple[Any, Any]:
    """Execute exact S512 GDN recurrence through opt-in ROCm typed FFI.

    Every input is FP32 and must use the oracle's token-major shape.  Query and
    key must already be masked, normalized, and query-scaled; U/W/gamma must be
    outputs of the qualified prepare boundary.  There is deliberately no
    fallback because silently selecting different recurrence math is unsafe.
    """
    _require_exact_bool(enabled, "enabled")
    if not enabled:
        if library_path is not None or library_sha256 is not None:
            raise ValueError(
                "library_path/library_sha256 are invalid while "
                "gdn_execute_s512 is disabled"
            )
        raise RuntimeError("gdn_execute_s512 is disabled by default")

    validated_sha256 = _validate_library_sha256(library_sha256)
    validated_path = _validate_library_path(library_path)
    jax_module, jnp_module = _import_jax()
    for item, name, shape in (
        (query, "query", GDN_EXECUTE_S512_QUERY_SHAPE),
        (key, "key", GDN_EXECUTE_S512_QUERY_SHAPE),
        (prepared_u, "prepared_u", GDN_EXECUTE_S512_PREPARED_SHAPE),
        (prepared_w, "prepared_w", GDN_EXECUTE_S512_PREPARED_SHAPE),
        (gamma, "gamma", GDN_EXECUTE_S512_GAMMA_SHAPE),
        (initial_state, "initial_state", GDN_EXECUTE_S512_STATE_SHAPE),
    ):
        _validate_value(item, name, shape, jnp_module.float32)

    _register_enabled(validated_path, validated_sha256, jax_module)
    return _call_registered(
        query,
        key,
        prepared_u,
        prepared_w,
        gamma,
        initial_state,
        jax_module,
        jnp_module,
    )
