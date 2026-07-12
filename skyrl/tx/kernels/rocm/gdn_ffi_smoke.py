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
library.
"""

from __future__ import annotations

import ctypes
import importlib
import operator
import os
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


@dataclass(frozen=True, slots=True)
class GdnFfiSmokeRegistration:
    """Public, library-handle-free description of one process registration."""

    library_path: Path
    target_name: str = GDN_FFI_SMOKE_TARGET
    platform: str = _ROCM_PLATFORM
    registration_api_version: int = _REGISTRATION_API_VERSION
    custom_call_api_version: int = _CUSTOM_CALL_API_VERSION


@dataclass(slots=True)
class _RegistrationState:
    public: GdnFfiSmokeRegistration
    library: Any


_registration_lock = threading.Lock()
_registration_state: _RegistrationState | None = None
_library_lifetime_handles: list[Any] = []


def _require_exact_bool(value: bool, name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact bool")


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
    if candidate.name != "libskyrl_gdn_ffi_smoke_gfx1100.so":
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


def _load_cdll(library_path: Path) -> Any:
    mode = getattr(os, "RTLD_LOCAL", 0) | getattr(os, "RTLD_NOW", 0)
    return ctypes.CDLL(str(library_path), mode=mode)


def _ffi_namespace(jax_module: Any) -> Any:
    ffi = getattr(jax_module, "ffi", None)
    required = ("ffi_call", "pycapsule", "register_ffi_target")
    if ffi is None or any(not callable(getattr(ffi, name, None)) for name in required):
        raise RuntimeError("installed JAX does not expose the required typed-FFI API")
    if not callable(getattr(jax_module, "ShapeDtypeStruct", None)):
        raise RuntimeError("installed JAX does not expose ShapeDtypeStruct")
    return ffi


def _register_enabled(library_path: Path, jax_module: Any) -> GdnFfiSmokeRegistration:
    global _registration_state

    ffi = _ffi_namespace(jax_module)
    with _registration_lock:
        if _registration_state is not None:
            if _registration_state.public.library_path != library_path:
                raise RuntimeError(
                    "gdn_ffi_smoke target is already registered from a different library"
                )
            return _registration_state.public

        try:
            library = _load_cdll(library_path)
        except OSError as error:
            raise RuntimeError(
                "could not load the explicit gdn_ffi_smoke library"
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
        public = GdnFfiSmokeRegistration(library_path=library_path)
        # The PyCapsule contains only a function pointer.  Retain CDLL for the
        # complete process lifetime so that pointer can never outlive its code.
        _registration_state = _RegistrationState(public=public, library=library)
        return public


def register_gdn_ffi_smoke(
    library_path: str | os.PathLike[str] | None = None,
    *,
    enabled: bool = False,
) -> GdnFfiSmokeRegistration:
    """Load and register the smoke handler only after exact explicit opt-in.

    Registration is process-global and idempotent only for the same canonical
    path.  It is intentionally restricted to platform ``ROCM`` and typed-FFI
    registration API version 1.
    """
    _require_exact_bool(enabled, "enabled")
    if not enabled:
        raise RuntimeError("gdn_ffi_smoke registration is disabled by default")
    validated_path = _validate_library_path(library_path)
    jax_module, _ = _import_jax()
    return _register_enabled(validated_path, jax_module)


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
) -> Any:
    """Return ``value`` by default or call the opt-in ROCm copy smoke.

    The disabled identity fallback deliberately performs no validation because
    it must remain a zero-dependency, zero-registration path.  The enabled path
    accepts only exact BF16 ``[1, 1024, 32, 128]`` values and an explicit local
    library path.  It has no custom differentiation rule and must not be wired
    into model execution.
    """
    _require_exact_bool(enabled, "enabled")
    if not enabled:
        if library_path is not None:
            raise ValueError("library_path is invalid while gdn_ffi_smoke is disabled")
        return value

    validated_path = _validate_library_path(library_path)
    jax_module, jnp_module = _import_jax()
    _validate_smoke_value(value, jnp_module)
    _register_enabled(validated_path, jax_module)
    return _call_registered(value, jax_module, jnp_module)
