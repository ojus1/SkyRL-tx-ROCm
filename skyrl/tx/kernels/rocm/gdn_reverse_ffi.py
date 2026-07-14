"""Default-off typed-FFI declaration for the S512 GDN transpose boundary.

This stage-zero module fixes a proposed native ABI before a HIP implementation
is admitted.  The declared boundary consumes the six transformed FP32 forward
primals plus the BF16 output and FP32 final-state cotangents used by
:mod:`gdn_superblock_reverse_oracle`.  Its low-level signature declares seven
results: six public FP32 gradient buffers and one hidden U8 scratch slab.  The
wrapper deliberately omits that seventh value from its public return.

The hidden 34,144,256-byte result is intended to make workspace visible to XLA
and to provide one base address to a future native handler.  This Python-only
stage does not prove compiler allocation, actual peak-memory accounting,
address stability, or base alignment.  Those properties require compiled-HLO
inspection plus isolated native qualification.  An admitted handler must
validate that the U8 base is aligned to at least 16 bytes before applying the
declared region offsets.

There is no CPU fallback and no partial native implementation in this module.
Registration and execution are default-off, require an explicitly identified
sealed shared object, and import JAX only after opt-in and identity checks.
Inputs are validated before any snapshot or load.  The shared object is loaded
from the existing sealed memfd snapshot loader rather than from its validated
pathname.  Because JAX exposes no target-registration rollback, an exception
from ``register_ffi_target`` permanently poisons this process-local operation;
only a process restart can make another registration attempt safe.
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

GDN_REVERSE_S512_TARGET = "skyrl_gdn_reverse_s512_f32_bf16_v1"
GDN_REVERSE_S512_QUERY_SHAPE = (1, 512, 16, 128)
GDN_REVERSE_S512_VALUE_SHAPE = (1, 512, 32, 128)
GDN_REVERSE_S512_GATE_SHAPE = (1, 512, 32)
GDN_REVERSE_S512_STATE_SHAPE = (1, 32, 128, 128)
GDN_REVERSE_S512_OUTPUT_COTANGENT_SHAPE = GDN_REVERSE_S512_VALUE_SHAPE
GDN_REVERSE_S512_SCRATCH_SHAPE = (34_144_256,)

# Declarative payload arithmetic derived from the shapes and dtypes below.
# These constants are ABI requirements, not measurements of compiler or device
# allocation.  The tuple-table value additionally assumes seven 64-bit result
# pointers; it is metadata, not tensor payload, and requires native-build
# verification.
GDN_REVERSE_S512_INPUT_BYTES = 25_296_896
GDN_REVERSE_S512_GRADIENT_BYTES = 19_005_440
GDN_REVERSE_S512_SCRATCH_BYTES = 34_144_256
GDN_REVERSE_S512_LOW_LEVEL_OUTPUT_BYTES = 53_149_696
GDN_REVERSE_S512_TOTAL_BUFFER_BYTES = 78_446_592
GDN_REVERSE_S512_INPUT_COUNT = 8
GDN_REVERSE_S512_PUBLIC_GRADIENT_COUNT = 6
GDN_REVERSE_S512_LOW_LEVEL_RESULT_COUNT = 7
GDN_REVERSE_S512_TUPLE_POINTER_TABLE_BYTES = 56
GDN_REVERSE_S512_REQUIRED_SCRATCH_BASE_ALIGNMENT_BYTES = 16


@dataclass(frozen=True, slots=True)
class GdnReverseS512ScratchRegion:
    """One typed region required within the proposed hidden U8 result.

    ``row_major_physical_order`` is major-to-minor order, matching the Python
    FFI layout convention.  Offset alignment only implies address alignment if
    the unverified U8 result base satisfies the separately declared 16-byte
    requirement.
    """

    name: str
    offset_bytes: int
    nbytes: int
    dtype: str
    shape: tuple[int, ...]
    row_major_physical_order: tuple[int, ...]
    lifetime: str
    use_and_reuse: str


# Fixed proposed regions within the hidden U8 result.  These declarations do
# not establish that a compiler allocated the slab or honored base alignment.
GDN_REVERSE_S512_SCRATCH_LAYOUT = (
    GdnReverseS512ScratchRegion(
        name="prepared_u",
        offset_bytes=0,
        nbytes=8_388_608,
        dtype="float32",
        shape=(1, 512, 32, 128),
        row_major_physical_order=(0, 1, 2, 3),
        lifetime="After PrepareAll through completion of reverse chunk zero.",
        use_and_reuse=(
            "PrepareAll writes all chunks once; replay and execute-reverse "
            "launches read them. The region is never repurposed."
        ),
    ),
    GdnReverseS512ScratchRegion(
        name="prepared_w",
        offset_bytes=8_388_608,
        nbytes=8_388_608,
        dtype="float32",
        shape=(1, 512, 32, 128),
        row_major_physical_order=(0, 1, 2, 3),
        lifetime="After PrepareAll through completion of reverse chunk zero.",
        use_and_reuse=(
            "PrepareAll writes all chunks once; replay and execute-reverse "
            "launches read them. The region is never repurposed."
        ),
    ),
    GdnReverseS512ScratchRegion(
        name="gamma",
        offset_bytes=16_777_216,
        nbytes=65_536,
        dtype="float32",
        shape=(1, 512, 32),
        row_major_physical_order=(0, 1, 2),
        lifetime="After PrepareAll through completion of reverse chunk zero.",
        use_and_reuse=(
            "PrepareAll writes all chunks once; replay and execute-reverse "
            "launches read them. The region is never repurposed."
        ),
    ),
    GdnReverseS512ScratchRegion(
        name="replay_states_s1_s7",
        offset_bytes=16_842_752,
        nbytes=14_680_064,
        dtype="float32",
        shape=(7, 32, 128, 128),
        row_major_physical_order=(0, 1, 2, 3),
        lifetime="From seven replay writes through their reverse-chunk reads.",
        use_and_reuse=(
            "ReplayStateChunk(c), c=0..6, writes slot c as S(c+1). Reverse "
            "chunk c reads input S0 when c=0 and slot c-1 otherwise. Slots "
            "are not repurposed during the handler."
        ),
    ),
    GdnReverseS512ScratchRegion(
        name="odd_value_head_query_gradient_chunk",
        offset_bytes=31_522_816,
        nbytes=524_288,
        dtype="float32",
        shape=(16, 64, 128),
        row_major_physical_order=(0, 1, 2),
        lifetime="One reverse chunk c at a time, in descending order 7..0.",
        use_and_reuse=(
            "ExecuteReverseB(c) writes the odd-value-head dQ partials; "
            "PairReduce(c) reads them after even partials were written to the "
            "public dQ chunk. Reuse for c-1 requires same-stream ordering "
            "after PairReduce(c)."
        ),
    ),
    GdnReverseS512ScratchRegion(
        name="odd_value_head_key_gradient_chunk",
        offset_bytes=32_047_104,
        nbytes=524_288,
        dtype="float32",
        shape=(16, 64, 128),
        row_major_physical_order=(0, 1, 2),
        lifetime="One reverse chunk c at a time, in descending order 7..0.",
        use_and_reuse=(
            "ExecuteReverseB(c) writes the odd-value-head dK partials; "
            "PairReduce(c) reads them after even partials were written to the "
            "public dK chunk. Reuse for c-1 requires same-stream ordering "
            "after PairReduce(c)."
        ),
    ),
    GdnReverseS512ScratchRegion(
        name="prepared_w_gradient_chunk",
        offset_bytes=32_571_392,
        nbytes=1_048_576,
        dtype="float32",
        shape=(32, 64, 128),
        row_major_physical_order=(0, 1, 2),
        lifetime="One reverse chunk c at a time, in descending order 7..0.",
        use_and_reuse=(
            "ExecuteReverseB(c) writes dW; PrepareTransposeSolve(c) overwrites "
            "it in place with RHS W-bar; PreparePropagate(c) reads it. Reuse "
            "for c-1 requires same-stream ordering after PreparePropagate(c)."
        ),
    ),
    GdnReverseS512ScratchRegion(
        name="attention_gradient_chunk",
        offset_bytes=33_619_968,
        nbytes=524_288,
        dtype="float32",
        shape=(32, 64, 64),
        row_major_physical_order=(0, 1, 2),
        lifetime="One reverse chunk c at a time, in descending order 7..0.",
        use_and_reuse=(
            "ExecuteReverseA(c) writes dA and ExecuteReverseB(c) reads it. "
            "Reuse for c-1 requires same-stream ordering after "
            "ExecuteReverseB(c)."
        ),
    ),
)

_REGISTRATION_API_VERSION = 1
_CUSTOM_CALL_API_VERSION = 4
_ROCM_PLATFORM = "ROCM"
_LIBRARY_BASENAME = "libskyrl_gdn_reverse_s512_gfx1100.so"
_ROW_MAJOR_R4 = (0, 1, 2, 3)
_ROW_MAJOR_R3 = (0, 1, 2)
_ROW_MAJOR_R1 = (0,)
_F32_ITEMSIZE = 4
_BF16_ITEMSIZE = 2


@dataclass(frozen=True, slots=True)
class GdnReverseS512Registration:
    """Library-handle-free identity for one process-global registration."""

    library_path: Path
    library_sha256: str
    snapshot_sha256: str
    snapshot_size_bytes: int
    snapshot_mode: int
    snapshot_seals: int
    sealed_snapshot: bool
    snapshot_fd_retained: bool
    target_name: str = GDN_REVERSE_S512_TARGET
    platform: str = _ROCM_PLATFORM
    registration_api_version: int = _REGISTRATION_API_VERSION
    custom_call_api_version: int = _CUSTOM_CALL_API_VERSION


@dataclass(slots=True)
class _RegistrationState:
    public: GdnReverseS512Registration
    library: Any
    snapshot: Any


@dataclass(frozen=True, slots=True)
class _PoisonedRegistration:
    """Identity whose target-registry outcome is no longer knowable."""

    library_path: Path
    library_sha256: str
    target_name: str = GDN_REVERSE_S512_TARGET
    failed_stage: str = "register_ffi_target"


_registration_lock = threading.Lock()
_registration_state: _RegistrationState | _PoisonedRegistration | None = None


def _raise_if_registration_poisoned() -> None:
    """Require process restart after an indeterminate registry mutation."""
    with _registration_lock:
        poisoned = isinstance(_registration_state, _PoisonedRegistration)
    if poisoned:
        raise RuntimeError(
            "gdn_reverse_s512 process registration is poisoned after an "
            "indeterminate register_ffi_target failure; restart the process"
        )


def _require_exact_bool(value: bool, name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact bool")


def _validate_library_sha256(library_sha256: str | None) -> str:
    if (
        not isinstance(library_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", library_sha256) is None
    ):
        raise ValueError(
            "gdn_reverse_s512 library_sha256 must be exactly 64 lowercase "
            "hexadecimal digits"
        )
    return library_sha256


def _validate_library_path(library_path: str | os.PathLike[str] | None) -> Path:
    if library_path is None:
        raise ValueError("gdn_reverse_s512 requires an explicit shared-library path")
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
        raise ValueError("gdn_reverse_s512 library_path must be absolute")
    if candidate.name != _LIBRARY_BASENAME:
        raise ValueError(
            "gdn_reverse_s512 library must use the exact name "
            "libskyrl_gdn_reverse_s512_gfx1100.so"
        )
    try:
        info = candidate.lstat()
    except FileNotFoundError as error:
        raise ValueError("gdn_reverse_s512 shared library does not exist") from error
    except OSError as error:
        raise ValueError(
            "gdn_reverse_s512 shared library cannot be inspected"
        ) from error
    if stat.S_ISLNK(info.st_mode):
        raise ValueError("gdn_reverse_s512 shared library must not be a symbolic link")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("gdn_reverse_s512 shared library must be a regular file")
    if info.st_uid != os.getuid():
        raise ValueError("gdn_reverse_s512 shared library must be user-owned")
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError(
            "gdn_reverse_s512 shared library must not be group- or world-writable"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(
            "gdn_reverse_s512 shared library cannot be resolved"
        ) from error
    if resolved != candidate:
        raise ValueError(
            "gdn_reverse_s512 library_path must not traverse symbolic links or '..'"
        )
    return resolved


def _import_jax() -> tuple[Any, Any]:
    """Import JAX only after explicit opt-in and library identity checks."""
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
    library_path: Path,
    library_sha256: str,
    jax_module: Any,
) -> GdnReverseS512Registration:
    global _registration_state

    ffi = _ffi_namespace(jax_module)
    with _registration_lock:
        if isinstance(_registration_state, _PoisonedRegistration):
            raise RuntimeError(
                "gdn_reverse_s512 process registration is poisoned after an "
                "indeterminate register_ffi_target failure; restart the process"
            )
        if _registration_state is not None:
            if (
                _registration_state.public.library_path != library_path
                or _registration_state.public.library_sha256 != library_sha256
            ):
                raise RuntimeError(
                    "gdn_reverse_s512 target is already registered from a "
                    "different library identity"
                )
            return _registration_state.public

        snapshot = _sealed_loader._snapshot_library(library_path, library_sha256)
        try:
            library = _sealed_loader._load_cdll(snapshot)
        except OSError as error:
            raise RuntimeError(
                "could not load the sealed gdn_reverse_s512 library snapshot"
            ) from error
        # Retain even if symbol lookup or third-party registration fails.
        _sealed_loader._library_lifetime_handles.append(library)
        try:
            handler = getattr(library, GDN_REVERSE_S512_TARGET)
        except AttributeError as error:
            raise RuntimeError(
                "gdn_reverse_s512 library is missing its exact handler symbol"
            ) from error
        handler.restype = ctypes.c_void_p
        handler.argtypes = (ctypes.c_void_p,)
        capsule = ffi.pycapsule(handler)

        # JAX exposes no rollback API for target registration. Publish a poison
        # identity before crossing that boundary: an exception can follow a
        # partial registry mutation. Only an unequivocal successful return and
        # completed public-state construction below may replace this sentinel.
        _registration_state = _PoisonedRegistration(
            library_path=library_path,
            library_sha256=library_sha256,
        )
        ffi.register_ffi_target(
            GDN_REVERSE_S512_TARGET,
            capsule,
            platform=_ROCM_PLATFORM,
            api_version=_REGISTRATION_API_VERSION,
        )
        public = GdnReverseS512Registration(
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


def register_gdn_reverse_s512(
    library_path: str | os.PathLike[str] | None = None,
    *,
    library_sha256: str | None = None,
    enabled: bool = False,
) -> GdnReverseS512Registration:
    """Register the declared reverse-handler identity after explicit opt-in."""
    _raise_if_registration_poisoned()
    _require_exact_bool(enabled, "enabled")
    if not enabled:
        raise RuntimeError("gdn_reverse_s512 registration is disabled by default")
    validated_sha256 = _validate_library_sha256(library_sha256)
    validated_path = _validate_library_path(library_path)
    jax_module, _ = _import_jax()
    return _register_enabled(validated_path, validated_sha256, jax_module)


def _concrete_shape(value: Any, name: str) -> tuple[int, ...]:
    raw_shape = getattr(value, "shape", None)
    if raw_shape is None:
        raise TypeError(f"gdn_reverse_s512 {name} must expose shape and dtype")
    try:
        dimensions = []
        for dimension in raw_shape:
            if type(dimension) is bool:
                raise TypeError("boolean dimension")
            dimensions.append(operator.index(dimension))
        return tuple(dimensions)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"gdn_reverse_s512 {name} shape must contain concrete integers"
        ) from error


def _expected_c_strides(
    shape: tuple[int, ...],
    itemsize: int,
) -> tuple[int, ...]:
    stride = itemsize
    reversed_strides = []
    for dimension in reversed(shape):
        reversed_strides.append(stride)
        stride *= dimension
    return tuple(reversed(reversed_strides))


def _validate_exposed_strides(
    value: Any,
    name: str,
    expected_shape: tuple[int, ...],
    itemsize: int,
) -> None:
    """Reject non-row-major host views; XLA fixes absent device strides."""
    raw_strides = getattr(value, "strides", None)
    if raw_strides is None:
        return
    try:
        strides = tuple(operator.index(stride) for stride in raw_strides)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"gdn_reverse_s512 {name} strides must contain concrete integers"
        ) from error
    expected = _expected_c_strides(expected_shape, itemsize)
    if strides != expected:
        raise ValueError(
            f"gdn_reverse_s512 {name} strides must be exactly {expected}, got {strides}"
        )


def _validate_value(
    value: Any,
    name: str,
    expected_shape: tuple[int, ...],
    expected_dtype: Any,
    itemsize: int,
) -> None:
    shape = _concrete_shape(value, name)
    if shape != expected_shape:
        raise ValueError(
            f"gdn_reverse_s512 {name} shape must be exactly {expected_shape}, "
            f"got {shape}"
        )
    if getattr(value, "dtype", None) != expected_dtype:
        raise TypeError(
            f"gdn_reverse_s512 {name} dtype must be exactly "
            f"{'bfloat16' if itemsize == _BF16_ITEMSIZE else 'float32'}"
        )
    _validate_exposed_strides(value, name, expected_shape, itemsize)
    raw_nbytes = getattr(value, "nbytes", None)
    if raw_nbytes is not None:
        try:
            nbytes = operator.index(raw_nbytes)
        except TypeError as error:
            raise TypeError(
                f"gdn_reverse_s512 {name} nbytes must be a concrete integer"
            ) from error
        expected_nbytes = itemsize
        for dimension in expected_shape:
            expected_nbytes *= dimension
        if nbytes != expected_nbytes:
            raise ValueError(
                f"gdn_reverse_s512 {name} nbytes must be exactly "
                f"{expected_nbytes}, got {nbytes}"
            )


def _exposed_host_range(value: Any, name: str) -> tuple[int, int] | None:
    """Return a host buffer range without importing NumPy, when exposed."""
    interface = getattr(value, "__array_interface__", None)
    if interface is None:
        return None
    if not isinstance(interface, dict):
        raise TypeError(f"gdn_reverse_s512 {name} has an invalid array interface")
    data = interface.get("data")
    if not isinstance(data, tuple) or not data:
        raise TypeError(f"gdn_reverse_s512 {name} has an invalid array data pointer")
    try:
        address = operator.index(data[0])
        nbytes = operator.index(getattr(value, "nbytes"))
    except (AttributeError, TypeError) as error:
        raise TypeError(
            f"gdn_reverse_s512 {name} must expose a concrete host range"
        ) from error
    if address <= 0 or nbytes <= 0:
        raise ValueError(f"gdn_reverse_s512 {name} host range must be non-null")
    end = address + nbytes
    if end <= address:
        raise ValueError(f"gdn_reverse_s512 {name} host range overflows")
    return address, end


def _validate_distinct_inputs(values: tuple[Any, ...], names: tuple[str, ...]) -> None:
    """Reject known aliases; an admitted native handler must gate all 15 ranges."""
    ranges = tuple(
        _exposed_host_range(value, name)
        for value, name in zip(values, names, strict=True)
    )
    for left_index, (left_name, left, left_range) in enumerate(
        zip(names, values, ranges, strict=True)
    ):
        for right_name, right, right_range in zip(
            names[left_index + 1 :],
            values[left_index + 1 :],
            ranges[left_index + 1 :],
            strict=True,
        ):
            aliased = left is right
            if left_range is not None and right_range is not None:
                aliased = aliased or (
                    left_range[0] < right_range[1] and right_range[0] < left_range[1]
                )
            if aliased:
                raise ValueError(
                    "gdn_reverse_s512 inputs must use distinct, non-overlapping "
                    f"buffers; {left_name} overlaps {right_name}"
                )


def _call_registered(
    query: Any,
    key: Any,
    value: Any,
    g: Any,
    beta: Any,
    initial_state: Any,
    output_cotangent: Any,
    final_state_cotangent: Any,
    jax_module: Any,
    jnp_module: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    ffi = _ffi_namespace(jax_module)
    results = (
        jax_module.ShapeDtypeStruct(GDN_REVERSE_S512_QUERY_SHAPE, jnp_module.float32),
        jax_module.ShapeDtypeStruct(GDN_REVERSE_S512_QUERY_SHAPE, jnp_module.float32),
        jax_module.ShapeDtypeStruct(GDN_REVERSE_S512_VALUE_SHAPE, jnp_module.float32),
        jax_module.ShapeDtypeStruct(GDN_REVERSE_S512_GATE_SHAPE, jnp_module.float32),
        jax_module.ShapeDtypeStruct(GDN_REVERSE_S512_GATE_SHAPE, jnp_module.float32),
        jax_module.ShapeDtypeStruct(GDN_REVERSE_S512_STATE_SHAPE, jnp_module.float32),
        jax_module.ShapeDtypeStruct(GDN_REVERSE_S512_SCRATCH_SHAPE, jnp_module.uint8),
    )
    call = ffi.ffi_call(
        GDN_REVERSE_S512_TARGET,
        results,
        has_side_effect=False,
        vmap_method="sequential",
        input_layouts=(
            _ROW_MAJOR_R4,
            _ROW_MAJOR_R4,
            _ROW_MAJOR_R4,
            _ROW_MAJOR_R3,
            _ROW_MAJOR_R3,
            _ROW_MAJOR_R4,
            _ROW_MAJOR_R4,
            _ROW_MAJOR_R4,
        ),
        output_layouts=(
            _ROW_MAJOR_R4,
            _ROW_MAJOR_R4,
            _ROW_MAJOR_R4,
            _ROW_MAJOR_R3,
            _ROW_MAJOR_R3,
            _ROW_MAJOR_R4,
            _ROW_MAJOR_R1,
        ),
        input_output_aliases=None,
        custom_call_api_version=_CUSTOM_CALL_API_VERSION,
    )
    low_level_results = call(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        output_cotangent,
        final_state_cotangent,
    )
    (
        query_gradient,
        key_gradient,
        value_gradient,
        g_gradient,
        beta_gradient,
        initial_state_gradient,
        _hidden_scratch,
    ) = low_level_results
    return (
        query_gradient,
        key_gradient,
        value_gradient,
        g_gradient,
        beta_gradient,
        initial_state_gradient,
    )


def gdn_reverse_s512(
    query: Any,
    key: Any,
    value: Any,
    g: Any,
    beta: Any,
    initial_state: Any,
    output_cotangent: Any,
    final_state_cotangent: Any,
    *,
    enabled: bool = False,
    library_path: str | os.PathLike[str] | None = None,
    library_sha256: str | None = None,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Dispatch six declared transformed-boundary results through opt-in FFI.

    Q/K/V/g/beta/S0 and dS8 use FP32 result/input declarations; dO uses BF16.
    This stage does not implement or prove gradient arithmetic.  An admitted
    native handler is required to widen dO before reverse arithmetic and match
    the separately qualified oracle. Inputs must be disjoint. Masking and
    normalization transposes remain outside this proposed native boundary.
    There is deliberately no fallback.
    """
    _raise_if_registration_poisoned()
    _require_exact_bool(enabled, "enabled")
    if not enabled:
        if library_path is not None or library_sha256 is not None:
            raise ValueError(
                "library_path/library_sha256 are invalid while "
                "gdn_reverse_s512 is disabled"
            )
        raise RuntimeError("gdn_reverse_s512 is disabled by default")

    validated_sha256 = _validate_library_sha256(library_sha256)
    validated_path = _validate_library_path(library_path)
    jax_module, jnp_module = _import_jax()
    values = (
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        output_cotangent,
        final_state_cotangent,
    )
    names = (
        "query",
        "key",
        "value",
        "g",
        "beta",
        "initial_state",
        "output_cotangent",
        "final_state_cotangent",
    )
    specs = (
        (GDN_REVERSE_S512_QUERY_SHAPE, jnp_module.float32, _F32_ITEMSIZE),
        (GDN_REVERSE_S512_QUERY_SHAPE, jnp_module.float32, _F32_ITEMSIZE),
        (GDN_REVERSE_S512_VALUE_SHAPE, jnp_module.float32, _F32_ITEMSIZE),
        (GDN_REVERSE_S512_GATE_SHAPE, jnp_module.float32, _F32_ITEMSIZE),
        (GDN_REVERSE_S512_GATE_SHAPE, jnp_module.float32, _F32_ITEMSIZE),
        (GDN_REVERSE_S512_STATE_SHAPE, jnp_module.float32, _F32_ITEMSIZE),
        (
            GDN_REVERSE_S512_OUTPUT_COTANGENT_SHAPE,
            jnp_module.bfloat16,
            _BF16_ITEMSIZE,
        ),
        (GDN_REVERSE_S512_STATE_SHAPE, jnp_module.float32, _F32_ITEMSIZE),
    )
    for item, name, (shape, dtype, itemsize) in zip(
        values,
        names,
        specs,
        strict=True,
    ):
        _validate_value(item, name, shape, dtype, itemsize)
    _validate_distinct_inputs(values, names)

    _register_enabled(validated_path, validated_sha256, jax_module)
    return _call_registered(*values, jax_module, jnp_module)
