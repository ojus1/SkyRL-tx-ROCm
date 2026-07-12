"""Opt-in ROCm kernel experiments.

Importing this package does not load a shared library, register an FFI target,
initialize JAX, or select an accelerator backend.
"""

from skyrl.tx.kernels.rocm.gdn_ffi_smoke import (
    GDN_FFI_SMOKE_BYTES,
    GDN_FFI_SMOKE_SHAPE,
    GDN_FFI_SMOKE_TARGET,
    GdnFfiSmokeRegistration,
    gdn_ffi_smoke_copy,
    register_gdn_ffi_smoke,
)

__all__ = (
    "GDN_FFI_SMOKE_BYTES",
    "GDN_FFI_SMOKE_SHAPE",
    "GDN_FFI_SMOKE_TARGET",
    "GdnFfiSmokeRegistration",
    "gdn_ffi_smoke_copy",
    "register_gdn_ffi_smoke",
)
