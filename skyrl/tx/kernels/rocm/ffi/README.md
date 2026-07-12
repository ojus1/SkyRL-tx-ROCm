# `gdn_ffi_smoke`

This directory contains a minimal typed-FFI prerequisite for a future ROCm
Qwen3.5 GDN superblock. It is not GDN math, not a model path, and not a speed
kernel. The only device operation copies one exact contiguous BF16
`[1, 1024, 32, 128]` array (8 MiB) to a distinct output.

The handler validates dtype, rank, every dimension, and byte count. It launches
one bounded grid-stride HIP kernel on JAX's supplied `hipStream_t`, checks launch
submission, and returns without `hipDeviceSynchronize` or
`hipStreamSynchronize`. XLA owns input/output lifetime and stream ordering.

Build to an explicit path outside the source tree:

```bash
JAXLIB_INCLUDE_DIR=/absolute/path/to/jaxlib/include \
  skyrl/tx/kernels/rocm/ffi/build_gdn_ffi_smoke_gfx1100.sh \
  /absolute/private/build/libskyrl_gdn_ffi_smoke_gfx1100.so
```

The script is fixed to `gfx1100`, creates no default build artifact, refuses to
overwrite, and only compiles/inspects files. It does not enumerate a device or
open `/dev/kfd`. Record the complete lowercase digest of that exact output:

```bash
sha256sum /absolute/private/build/libskyrl_gdn_ffi_smoke_gfx1100.so
```

Python use is explicitly opt-in:

```python
from skyrl.tx.kernels.rocm.gdn_ffi_smoke import gdn_ffi_smoke_copy

# Default, dependency-free identity fallback:
y = gdn_ffi_smoke_copy(x)

# Experimental typed FFI; compilation/execution is ROCm GPU work:
y = gdn_ffi_smoke_copy(
    x,
    enabled=True,
    library_path="/absolute/private/build/libskyrl_gdn_ffi_smoke_gfx1100.so",
    library_sha256="<exact 64-character lowercase sha256>",
)
```

The enabled path validates the canonical user-owned source path, opens it with
`O_NOFOLLOW`, and copies it once into a private `memfd` while computing the
required digest. It verifies the source file identity stayed unchanged, checks
the digest before loading, and applies and verifies `F_SEAL_WRITE`,
`F_SEAL_GROW`, `F_SEAL_SHRINK`, and `F_SEAL_SEAL`. `ctypes` loads only the
immutable `/proc/self/fd/<retained-fd>` snapshot, never the validated pathname.
The snapshot descriptor and any `ctypes.CDLL` are retained for process lifetime,
including later registration failures.

The exported handler is wrapped with `jax.ffi.pycapsule`, and the target is
registered only for platform `ROCM` with typed-FFI registration API version 1.
The `ffi_call` uses custom-call API version 4 and explicit row-major layouts.

There is no custom VJP, recurrence, masking, state carry, scratch allocation,
numerical promotion, or performance claim. Do not wire this smoke into model
execution. A guarded hardware probe and independent stream/lifetime review are
required before it can serve as plumbing for real GDN kernels.
