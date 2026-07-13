# Experimental exact S512 GDN execute FFI

This is a default-off, exact-shape forward kernel for the recurrent execute
stage of one Qwen3.5-4B GDN 512-token superblock. It consumes already prepared
FP32 tensors, carries the FP32 state through eight 64-token chunks, and returns
a BF16 output plus the FP32 final state. It does not prepare `U`, `W`, or
`gamma`, implement reverse mode, select a model path, or authorize replay.

The fixed ABI is:

| Role | Shape | Dtype | Bytes |
|---|---:|---:|---:|
| query | `[1,512,16,128]` | FP32 | 4 MiB |
| key | `[1,512,16,128]` | FP32 | 4 MiB |
| prepared U | `[1,512,32,128]` | FP32 | 8 MiB |
| prepared W | `[1,512,32,128]` | FP32 | 8 MiB |
| gamma | `[1,512,32]` | FP32 | 64 KiB |
| initial state | `[1,32,128,128]` | FP32 | 2 MiB |
| output | `[1,512,32,128]` | BF16 | 4 MiB |
| final state | `[1,32,128,128]` | FP32 | 2 MiB |

One 128-thread workgroup owns one value head and all eight recurrent chunks.
Value head `hv` reads key head `hv // 2`; thread `c` owns output/state column
`c`. Each chunk forms the causal gamma-scaled query/key product, computes
`corrected = U - W @ state`, emits the inter- plus intra-chunk output, and then
updates the recurrent state. Attention, corrected values, and the 64-element
reverse-decay vector share 49,408 bytes of LDS. The launch has 32 workgroups,
uses JAX's supplied stream, allocates no dynamic device scratch, and performs
no device- or stream-wide synchronization.

The typed handler rejects any rank, dimension, byte-count, or dtype mismatch
and requires all six inputs and two outputs to be pairwise disjoint and
non-null. The Python wrapper is opt-in and loads only a hashed, sealed private
snapshot of the supplied native library.

Build to a fresh canonical path outside the repository:

```bash
JAXLIB_INCLUDE_DIR=/absolute/path/to/jaxlib/include \
  skyrl/tx/kernels/rocm/ffi/build_gdn_execute_s512_gfx1100.sh \
  /absolute/private/build/libskyrl_gdn_execute_s512_gfx1100.so
sha256sum /absolute/private/build/libskyrl_gdn_execute_s512_gfx1100.so
```

The build script is fixed to `gfx1100`, refuses overwrite, and does not access
the GPU. Enabled use requires an exact external path and digest:

```python
from skyrl.tx.kernels.rocm.gdn_execute_ffi import gdn_execute_s512

output, final_state = gdn_execute_s512(
    query,
    key,
    prepared_u,
    prepared_w,
    gamma,
    initial_state,
    enabled=True,
    library_path="/absolute/private/build/libskyrl_gdn_execute_s512_gfx1100.so",
    library_sha256="<exact 64-character lowercase sha256>",
)
```

`rocm/probe_gdn_execute_s512_ffi.py` is the guarded qualification entrypoint.
Its default path only emits an abstract refusal. The ROCm path requires the
exact one-shot case, private output, pinned artifact, direct
`profile_rocm.py` parent, headless and exclusive GPU, clean current-boot
journal, and solely:

```bash
XLA_FLAGS=--xla_gpu_enable_command_buffer=
```

The mandatory `--compile-diagnostic` rung constructs only abstract shape/dtype
descriptors, lowers, compiles, inspects both IR dialects and compiler memory,
and destroys the unreleased handle. The numerical rung independently repeats
that gate, constructs and hashes its complete CPU oracle before device
placement, releases a one-use capability, performs exactly one tuple device
put, one checked invocation, one output readiness barrier, and one tuple device
get. It has no warmup, replay, graph, device reference, reduction, backward,
VJP, or model call.

The headless card remains runtime-suspended while the CPU oracle is built, so
the profiler permits up to 60 seconds for thermal sensors to appear. Numeric
temperature and power readings are still enforced during that interval;
VRAM, host-memory, swap, timeout, journal, and process limits are immediate.
Completion without ever observing required sensors fails closed.

## ROCm 7.2.4 build and compile result

The exact sources built reproducibly into an 80,128-byte mode-`0600` shared
object with SHA-256
`435487abce7299bf9a835840f195cb95f0a644804fc2c41843ba9f5621ebd53b`.
The code object contains only `gfx1100`. Metadata reports 49,408 bytes of LDS,
30 SGPRs, 66 VGPRs, no SGPR/VGPR spill, no private segment, no dynamic stack,
wave size 32, and a 128-thread maximum workgroup. Device disassembly contains
the expected two static IEEE-division sites: attention decay and one shared
reverse-decay precomputation per row.

From commit `5308ca72`, the zero-invocation compile diagnostic completed under
the 90 C, 315 W, 24 GiB, and 300-second supervisor. Lowering took
0.016764 seconds and compilation 0.015176 seconds. StableHLO and optimized HLO
each contained exactly one direct typed call with the six ordered inputs and
two ordered outputs, exact row-major layouts, no other custom call, no loop,
and no alias. Compiler memory was exactly 27,328,512 argument bytes,
6,291,472 tuple-root-inclusive output bytes, zero temporary bytes, zero alias
bytes, and 1,920 generated-code bytes.

All runtime, oracle, transfer, capability-release, invocation, warmup, replay,
graph, backward, and model counters were zero. The supervised process completed
in 42.0545 seconds; peak junction temperature, power, and physical VRAM were
47 C, 128 W, and 711,987,200 bytes, with zero swap and a clean postflight.
The private artifacts and SHA-256 values are:

- `/tmp/gdn-execute-s512-compile-diagnostic-boot54ccf56c-run1.jsonl`:
  `d56765d8b57edbc02757fc22053d991972f26e10eea13d24ae2d719b86783c1e`;
- telemetry:
  `f342f8e4e5fba230ab480bbbfc03b9e4739d9f03d75d3e4e6d774bd0fbe0952d`;
- profiler summary:
  `4a307fb32b190cf3b65bd46bd182a15c4706fabeea6ea9671b7ce84df11fb51a`.

## ROCm 7.2.4 exact one-shot forward result

The first numerical attempt deliberately constructed the CPU oracle before
ROCm initialization. The headless card's thermal sensor remained unavailable
past the then-15-second grace, so the supervisor terminated it at 17.049
seconds. VRAM stayed exactly at the 27,947,008-byte idle baseline, GTT was
unchanged, and every backend, compile, capability, transfer, and dispatch
counter remained zero. This was a fail-closed guard precursor, not a kernel
failure or invocation. Its child, telemetry, and summary SHA-256 values are
`60902ced02fd811b1429a2429a90ead65e9e6d1bcdc6d2861c546b13c9d44fca`,
`aa7c7c4a13658a809f5f9f200f3550ad80c496bae81e8d5bfc99ab135d25df12`,
and `676b251507b1c09cb1ab09645356c70331ceaf6d0797fd4202ea6e1f62577e13`.

After independent review of the sensor-grace-only correction, commit
`4c9e7877` completed exactly one fresh-process numerical invocation. The
candidate took 0.008414445 seconds including its output readiness barrier.
Against the committed BF16/FP32 CPU oracle:

| Result | Relative L2 | Cosine | Maximum absolute error |
|---|---:|---:|---:|
| BF16 output | `2.33682e-5` | `0.999999999727` | `9.53674e-7` |
| FP32 final state | `1.23831e-7` | `0.999999999999972` | `1.16415e-9` |

All values were finite, shapes/dtypes/byte counts were exact, outputs were
distinct and contiguous, and the six host inputs retained their committed
hashes. Counters prove one capability release, one six-leaf tuple device put,
one input barrier, one checked executable invocation, one output barrier, one
two-leaf tuple device get, and zero warmup, replay, graph, GPU reference,
reduction, backward, or model calls. VJP was neither exercised nor validated.
The repeated compiler gate again reported exact argument/output memory with
zero temporary and alias bytes.

The supervisor returned zero after 72.2149 seconds. Peak observed junction
temperature, power, and physical VRAM were 50 C, 129 W, and 779,153,408 bytes;
minimum host-available memory was 58,909,396,992 bytes and swap stayed zero.
The first 215 measured samples had unavailable GPU sensors; 485 consecutive
complete samples followed and covered dispatch through process exit. The
postflight journal was clean, the process released KFD, and the card later
returned to runtime suspend and the exact 27,947,008-byte idle VRAM baseline.
The private artifacts and SHA-256 values are:

- `/tmp/gdn-execute-s512-runtime-boot54ccf56c-run2.jsonl`:
  `c9bd178848a85071ac0734b08f88ada3677aed58e1b63f2679a8110f93393cb4`;
- telemetry:
  `ff5817eafa9f36181c88caeaa7d7368738869cab87559754ae1f2493c3d785ab`;
- profiler summary:
  `92051af62e43ea0d669b1a63c24d20c976c6550c1fbd284fd7038ddbd16066ff`.

This qualifies one exact standalone forward invocation. Its 8.414 ms is the
first candidate invocation after backend compilation and input placement, not
end-to-end cold startup, repeated throughput, or model speedup. It does not
qualify preparation plus execute composition, reverse mode, VJP, tails, other
sequence lengths, model integration, warmup, replay, or production use. The
next kernel gate is an explicit reverse superblock, followed by composed
prepare/execute/reverse numerical tests and one-layer fixed-data integration.
