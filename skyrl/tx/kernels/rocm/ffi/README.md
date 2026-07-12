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

## Guarded gfx1100 result

The independently audited `copy8` gate passed from commit `19eba202` on ROCm
7.2.4/JAX 0.10.2. The 63,912-byte shared object was built for `gfx1100`,
exported the exact handler once, and had SHA-256
`341f3a7b0c8b0ba2a3e92f8e6ae7a5c07a253a1aee806d306f2ec884ff4dced6`.
The wrapper copied and hashed those bytes into a retained mode-`0600` memfd,
verified write/grow/shrink/seal protections (mask 15), and passed only
`/proc/self/fd/<fd>` to `dlopen`.

StableHLO and optimized HLO each contained exactly one call to
`skyrl_gdn_ffi_smoke_bf16_copy_v1`, with no other custom call, outer loop, or
alias metadata. Compiler accounting was exactly 8,388,608 B of arguments,
8,388,608 B of distinct output, and zero temporary/alias bytes. One checked,
synchronized dispatch took 5.090134 ms and copied all 8,388,608 bytes
bit-for-bit; input and output SHA-256 were both
`c27a2d6d0a518172311646aae22fa49fa11f575a1a6b4baf87f2866b0d002c3f`.
This one cold measurement is not a bandwidth or performance claim.

The outer profiler completed in 39.303168 s with return code zero. Peak
physical VRAM, junction temperature, and power were 728,805,376 B, 49 C, and
128 W; minimum host-available memory was 61,946,482,688 B and swap stayed zero.
All ten child journal checkpoints and both safety flights were clean, and the
card returned to runtime suspend.

All artifacts are mode `0600`:

- `/tmp/gdn-ffi-copy8-boot54ccf56c-run1.jsonl`
- `/tmp/gdn-ffi-copy8-boot54ccf56c-run1.telemetry.jsonl`
- `/tmp/gdn-ffi-copy8-boot54ccf56c-run1.telemetry.jsonl.summary.json`

Their SHA-256 values are respectively
`f10d695caed1030233193abe85f05890f272bb705d93b6fb57c52e46ec42bd29`,
`fe7058908250fa20368a0527a98a6927f16b95a9987a377780501b2bff930dca`,
and `c2b8b61c40a516a335b3d2862973c93c5e3ab1edac5f51e6aa3414cb705af0cd`.

This clears only typed-FFI ABI registration, sealed code-object lifetime,
JAX-supplied stream launch ordering, distinct output allocation, and exact
BF16 copy. Real GDN prepare/execute/reverse kernels still require independent
math, scratch, numerical, backward, and model-integration gates.
