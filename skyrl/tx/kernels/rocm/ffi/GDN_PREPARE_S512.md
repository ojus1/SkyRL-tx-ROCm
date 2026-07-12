# Experimental exact S512 GDN prepare FFI

This is the first GDN math rung after the typed-FFI stream/lifetime smoke. It
prepares WY tensors for exactly one Qwen3.5-4B 512-token superblock; it does not
execute the recurrent state transition, implement reverse mode, or select a
model path.

The fixed FP32 ABI is:

| Role | Shape | Bytes |
|---|---:|---:|
| transformed K | `[1,512,16,128]` | 4 MiB |
| transformed V | `[1,512,32,128]` | 8 MiB |
| transformed g | `[1,512,32]` | 64 KiB |
| transformed beta | `[1,512,32]` | 64 KiB |
| prepared U | `[1,512,32,128]` | 8 MiB |
| prepared W | `[1,512,32,128]` | 8 MiB |
| cumulative G | `[1,512,32]` | 64 KiB |

Masking is deliberately not a fifth ABI argument. Before this boundary, K has
the model's L2 normalization and FP32 promotion, and masked K, V, g, and beta
entries are zero. `gdn_prepare_oracle.py` exposes the masking transformation
for semantic tests; it intentionally treats K as already normalized. A masked
row therefore has zero U/W, while G remains the cumulative exponential of
transformed g and need not be zero.

For each of eight 64-token chunks and 16 key heads, one 128-thread workgroup
computes a key Gram matrix once and reuses it for value heads `2*hk` and
`2*hk+1`. For each value head it forms the strict-lower correction

`L[i,j] = beta[i] * dot(K[i], K[j]) * exp(prefix[i] - prefix[j])`, for `j < i`,

then solves

`(I + L) [U | W] = [beta * V | beta * exp(prefix) * K]`.

The kernel uses 49,664 bytes of statically bounded LDS: shared key Gram and
strict-lower matrices, one 64-column forward-solve tile, and prefix/beta
vectors. This is below the 64 KiB workgroup limit targeted by the design. The
native-build result below records compile and code-object resource evidence;
runtime occupancy and latency, numerical error, and watchdog safety remain
unmeasured.

There is one HIP launch on JAX's supplied stream, no device or stream
synchronization, no device-side scratch allocation, and no fast-math build
flag. The handler checks exact ranks, dimensions, byte counts, and seven
pairwise-distinct non-null buffers before submitting the launch.

Build to a new canonical path outside the repository:

```bash
JAXLIB_INCLUDE_DIR=/absolute/path/to/jaxlib/include \
  skyrl/tx/kernels/rocm/ffi/build_gdn_prepare_s512_gfx1100.sh \
  /absolute/private/build/libskyrl_gdn_prepare_s512_gfx1100.so
sha256sum /absolute/private/build/libskyrl_gdn_prepare_s512_gfx1100.so
```

The script is fixed to `gfx1100`, refuses overwrite, emits no default artifact,
and never opens `/dev/kfd`. Python registration is default-off and requires the
explicit canonical path plus exact lowercase SHA-256:

```python
from skyrl.tx.kernels.rocm.gdn_prepare_ffi import gdn_prepare_s512

u, w, cumulative_g = gdn_prepare_s512(
    transformed_key,
    transformed_value,
    transformed_g,
    transformed_beta,
    enabled=True,
    library_path=(
        "/absolute/private/build/libskyrl_gdn_prepare_s512_gfx1100.so"
    ),
    library_sha256="<exact 64-character lowercase sha256>",
)
```

The wrapper reuses the gated smoke module's one-pass hashing copy into a
mode-`0600`, fully sealed memfd. Only the retained `/proc/self/fd/<fd>` snapshot
is passed to `dlopen`; the source path is never loaded. The descriptor and CDLL
are retained for process lifetime even across later registration failures.

JAX compile and runtime gates are pending. Do not build or invoke this
operation outside the guarded probe protocol, and do not wire it into the
model or a custom VJP. The next rungs require an independently reviewed
compile-only gate, an analytic plus dense-oracle runtime gate at S512, then
separate execute and explicit reverse-superblock kernels.

## ROCm 7.2.4 external build result

The independently reviewed sources built successfully with the fixed gfx1100
script into a new mode-`0600` artifact outside the repository:

`/tmp/skyrl-gdn-prepare-s512-build-boot54ccf56c.Stpvf1/libskyrl_gdn_prepare_s512_gfx1100.so`

The shared object is 80,232 bytes with SHA-256
`56f667eee1eddac6b881feba18fc9a3315bc2b22e8fdfe08effa36e32e6315ef`.
It contains one HIP offload bundle for exact target
`amdgcn-amd-amdhsa--gfx1100`; the extracted 10,064-byte code object has
SHA-256
`cba64f5c7e645f7ea3d83f8f95a12440fbeb526036891ee159ac179824296aac`.
The dynamic symbol table contains the exact typed-FFI handler once.

AMDGPU code-object metadata reports one kernel with a 128-thread maximum
workgroup, wavefront size 32, 49,664 bytes of fixed group/LDS storage, zero
bytes of fixed private/scratch storage, 48 SGPRs, 45 VGPRs, zero SGPR spills,
zero VGPR spills, and no dynamic stack. This confirms the source-level LDS
calculation and clears the first compile/resource gate; the large LDS footprint
still limits each compute unit to one such workgroup and must be evaluated in
runtime occupancy and latency measurements.

The build took 3.57 seconds wall time, used at most 181,552 KiB host RSS, and
reported zero swaps. `/dev/kfd` and the AMD render node were unowned before and
after the build, so no GPU program was submitted. This build result does not
validate JAX registration, physical HLO layouts, one-call lowering, numerical
correctness, launch safety, or performance. Those remain separate guarded
compile and runtime probes; the operation is still default-off and unwired.
