# gfx1100 quantized LoRA FFI route

This note turns the grouped W8/W4 semantics in
`skyrl/tx/kernels/quantized_lora.py` into a concrete ROCm compilation plan. It
is scoped to Qwen3.5-4B on the RX 7900 XTX (`gfx1100`). Nothing described here
is enabled in model code yet.

## What is proven locally

The installed stack is ROCm 7.2.4, Clang 22, Composable Kernel 1.2.0,
rocWMMA 2.2.0, hipBLASLt 1.2.2, and JAX/JAXlib 0.10.2. Inspection and
compile-only checks establish the following:

- Clang accepts gfx1100's signed IU8 and IU4 WMMA builtins. Both accumulate
  into eight lane-local INT32 values for a 16x16x16 wave32 operation. IU8 takes
  two `int32x4` fragments; IU4 takes two `int32x2` packed-nibble fragments.
- `rocwmma/internal/wmma_impl.hpp` supplies the gfx11 IU8 wrapper, including
  `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`. The installed public rocWMMA
  implementation has no corresponding IU4 wrapper.
- CK's `ck/tensor_operation/gpu/warp/wmma_gemm.hpp` names IU4 behind
  `CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4`, but the installed file has no IU4
  `wmma_type` implementation. It is not a ready W4A4 GEMM path in this package.
- hipBLASLt ships gfx1100 INT8 Tensile code objects, including I8-input/I32
  output variants. Its scale modes do not express the oracle's scale for every
  `(row, K-group)` and `(K-group, output)` pair. Its epilogues also do not
  implement an arbitrary rank-8 LoRA branch.
- JAXlib supplies typed-FFI headers at
  `jaxlib/include/xla/ffi/api/{api.h,ffi.h}`. They expose typed buffers,
  `PlatformStream<hipStream_t>`, and `ScratchAllocator`; no pybind11 dependency
  is required to register an exported handler through `ctypes` and
  `jax.ffi.pycapsule`.

The compile proof is:

```bash
rocm/compile_probes/compile_quant_wmma_gfx1100.sh
```

It runs `hipcc -O3 --genco --no-gpu-bundle-output
--offload-arch=gfx1100`, writes only to a private directory under `${TMPDIR:-/tmp}`,
and disassembles the code object. A post-upgrade rerun on ROCm 7.2.4 on
2026-07-13 reproduced a 6,304-byte HSACO containing both:

```text
v_wmma_i32_16x16x16_iu8 v[0:7], v[8:11], v[12:15], v[0:7]
v_wmma_i32_16x16x16_iu4 v[0:7], v[8:9],  v[10:11], v[0:7]
```

The script does not enumerate a device, open `/dev/kfd`, or launch a kernel. It
refuses an architecture other than gfx1100 and fails if hipcc, LLVM inspection,
either kernel symbol, or either instruction is unavailable. Set
`SKYRL_KEEP_COMPILE_PROBE=1` only when the temporary HSACO/disassembly must be
retained for inspection.

This proves compiler and ISA availability only. It says nothing about correct
fragment layout, numerical quality, occupancy, latency, or training speed.

## Route decision

| Route | Concrete implementation | Decision |
|---|---|---|
| W8A8 group-64 | Custom HIP wave32 GEMM using the proven IU8 builtin | First native-integer speed prototype |
| W4A4 group-64 | Custom HIP wave32 GEMM using the proven IU4 builtin | Second prototype, after W8A8 is stable |
| Whole-K hipBLASLt INT8 | Vendor GEMM with coarser scalar/outer-vector scales | Useful comparison, but a different quantization format |
| Group-64 hipBLASLt | One GEMM per group plus a separate reduction | Reject: too many calls or an `M x groups x N` partial buffer |
| Installed CK IU4 | Experimental selector without a complete installed implementation | Reject for this environment |
| Single whole-layer kernel | Persistent kernel with cross-stage global synchronization | Reject; use bounded subdispatches |

Direct builtins are the shortest path for both formats. rocWMMA can be used to
prototype IU8 tile movement, but using the same explicit fragment loader for
IU8 and IU4 avoids making W4 depend on an incomplete wrapper. The packed W4
checkpoint layout is not automatically a WMMA fragment layout: an offline or
load-time permutation must create the canonical device weight. Do not perform
that permutation on every forward.

## Default-off Pallas W8A8 feasibility rung

`skyrl/tx/kernels/rocm/w8a8_lora.py` now provides a smaller experiment before
the production HIP work. JAX 0.10.2's local Pallas primitive assigns signed
integer dots an INT32 result, and its Triton lowering accepts signed INT8
inputs and emits a Triton `dot`. That source-level fact does **not** prove the
gfx1100 backend selects an IU8 matrix instruction; only retained code-object
disassembly or equivalent ISA evidence can establish that.

The prototype is explicit opt-in and is not imported or selected by model
code. It keeps W8 group-64 codes/scales compact, quantizes A8 once per bounded
row superblock, computes one INT32 dot per K group, applies the row and weight
scales in FP32, and adds rank-8 LoRA in the same epilogue before one BF16 cast.
It also exposes a base-only custom-VJP route so a future compact layer can use
the existing `LoRAMixin.apply_lora` implementation for no-adapter and genuine
multi-adapter batches; the fused epilogue is only the already-selected
single-adapter fast path.
Its relaxed input pullback dequantizes one W8 tile to BF16 and uses a BF16 dot
with FP32 accumulation. The source maps all forward and backward row work in
at most 512 rows by default, with an enforced 2,048-row maximum; one small
Pallas program is not incorrectly treated as proof of a bounded whole-grid
launch. Each forward program still scans the capped K domain, up to 144
group-64 dots, and each input-pullback workgroup scans the capped N domain, up
to 18,432 columns. The physical duration of both scans remains unproven.
The experiment also caps K at 9,216 and N at 18,432, the largest non-vocabulary
Qwen3.5-4B projection dimensions. The tied 248,320-column head is deliberately
outside this path.

CPU Pallas-interpreter tests currently prove only semantics. The fixed
`M=3,K=64,N=17` tail case is bitwise equal to the portable W8A8 forward. Against
the stronger portable FP32-dequant backward, relative L2 errors were 0.3341%
for `dX`, 0.2435% for `dA`, 0.3220% for `dB`, and 0.0000703% for `dscale`, all
inside the retained 1% CPU regression gate. A `M=1025` JAXPR test proves a
three-iteration row scan whose sole forward Pallas grid covers exactly 512
rows per iteration. A separate `M=33,K=128,N=19` interpreter test executes
three 16-row superblocks and checks bitwise forward equality plus all four
pullbacks against the same 1% gate. These are CPU structural/numerical results,
not GPU lowering, runtime, memory, or speed results.

The first useful model family remains all 32 MLP gate/up matrices, each exactly
`K=2560,N=18432`. Compact W8 codes plus BF16 group-64 scales save exactly
1,462,763,520 bytes (1.362304688 GiB) versus their BF16 kernels. This one shared
callsite is the smallest projection family with more than 1 GiB capacity
upside. It must retain no BF16 shadow and remains blocked on guarded gfx1100
compile, ISA, correctness, memory, and throughput gates before NNX/checkpoint
integration.

### Guarded gfx1100 qualification entrypoint

`rocm/run_w8a8_lora_forward_gate.py` and
`rocm/probe_w8a8_lora_forward.py` implement only the first hardware rung. Their
default invocations emit an abstract refusal without importing JAX. The ROCm
controller holds the global launch lock across an exact suspended/unowned
RX 7900 XTX baseline, hash-bound `profile_rocm.py` supervision, the child, three
consecutive one-second idle-handoff samples, and a final whole-boot journal
check. The profiler is itself loaded from hash-verified source under
`python -I -S -B`, with its Linux `psutil` runtime hash-bound before imports.
The child separately validates the inherited lock, exact PCI/render
identity, disconnected AMD display, unowned KFD/render nodes, fixed stack and
selected native-library/source hashes, and the exact sole environment value:

```text
XLA_FLAGS=--xla_gpu_enable_command_buffer=
```

The fixed source requests the base-W8A8 plus LoRA-B/scaling Pallas epilogue at
`M=3,K=64,N=17`, group 64, `block_m=block_n=16`, and physical grid `1x2`.
Activation quantization and the LoRA-A contraction remain ordinary JAX work
outside that Pallas call, so this is not an entire-forward megakernel. The only
enabled hardware phase is an unpromoted compile diagnostic; it invokes no
returned executable. `--phase execute` is rejected until retained gfx1100 ISA
is qualified in a later source revision. The diagnostic contains no backward,
warmup, repetition, replay, graph, model, or benchmark work.

The controller fixes sampled-sysfs thresholds of 2 GiB VRAM, 90 C junction,
and 315 W `power1_average`, plus an 85 C child launch gate, 300-second child
timeout, 330-second independent supervisor watchdog, and relaxed 0 GiB
host-available/8 GiB swap limits. It records and bounds the largest observed
sample gap, but this remains reactive evidence rather than a continuous
measurement. The child also refuses a reported hardware `power1_cap` above
315 W; that configuration still does not prove that every instantaneous
electrical transient stayed below the cap. The process uses a fresh private
7.5%-of-VRAM BFC allocator limit and fresh JAX,
Triton, compiler-dump, and `TF_XLA_HSACO_CACHE_DIR` directories. Longer compile
startup is intentional and cannot be replaced by command-buffer capture. The
BFC setting and XLA buffer-plan estimate are allocator/compiler constraints,
not independent proof of peak physical VRAM.

The safe defaults are:

```bash
.venv/bin/python rocm/run_w8a8_lora_forward_gate.py
.venv/bin/python rocm/probe_w8a8_lora_forward.py
```

After committing a clean source tree, a fresh compile-only artifact can be
requested with:

```bash
umask 077
run_dir="/tmp/skyrl-w8a8-compile-$(date +%s)"
XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  .venv/bin/python -I -S -B \
    -X "pycache_prefix=$run_dir/python-cache" \
    rocm/run_w8a8_lora_forward_gate.py \
    --platform rocm --phase compile --allow-gpu --run-dir "$run_dir"
```

Controller completion by itself does not establish runtime correctness or
native INT8 ISA. It establishes only an unpromoted compile diagnostic whose
private probe, telemetry, compiler, handoff, and final-journal artifacts must
be reviewed together. Optimized HLO proving one Triton custom call is not an
INT8 ISA proof, and surrounding optimized-HLO fusions may still launch
separate kernels. The probe therefore persists raw StableHLO and optimized HLO
and marks the result `passed_compile_diagnostic_unpromoted`. A separate offline
audit must correlate an actual retained code object with the forward symbol,
gfx1100 disassembly, and resource use. Runtime promotion additionally requires
a guarded dispatch trace and host numerical comparison. If that chain is not
available, the result remains only a Pallas Triton custom-call compile proof.
Version/origin checks and selected binary hashes do not constitute a complete
hash closure over every JAX, NumPy, ML-dtypes, ROCm, and system runtime file;
the untouched installed stack remains a recorded trust assumption.

The structural gate treats raw forward-name occurrences as diagnostics only:
XLA may repeat that name in op metadata and the embedded Triton configuration
for one custom call. Both the probe and controller instead mask quoted payloads
and comments, require the exact scoped `mhlo.backend_config`/`backend_config`
`name`, preserve duplicate target/name attributes as failures, bind the sole
call to the unique public entry, and require its SSA result to reach the entry
return/ROOT through data operands. Metadata, nested-map, comment, payload,
dead-helper, dead-result, and control-predecessor decoys all fail closed.

### Qualified tiny compile and native ISA result

Exact revision `da9bf1ee6195921fd7c8cf9055a3dc8d4a1ed704` completed the
compile-only rung in `/tmp/skyrl-w8a8-compile-1783972228`. The controller
returned `passed_compile_diagnostic_unpromoted`, the worktree remained clean,
and the returned executable was invoked zero times. Lowering and compilation
took 1.134153 and 0.719444 seconds. The compiler-reported argument, output, and
temporary sizes were 2,806, 102, and 3,600 bytes. StableHLO and optimized HLO
had SHA-256 values
`0e57123dd6c1d4355b7ccbbf0f7908db686ceaafc6d26afbb237811f8861ecdf`
and
`7adf78dc72eebed7a4eaa0c1e88f7ba43ecea0199d54817fcaeb2fcc3b5aa0dc`.
Both independently parse as one entry-owned Triton custom call whose result
feeds the public output, with the exact forward name and no backward, outer
loop, graph, capture, command-buffer, or replay marker.

`rocm/inspect_w8a8_lora_isa.py` reproduces the retained-code audit without
importing JAX or opening a GPU device. It binds the private cache entry by
SHA-256, bounds zstd and Snappy output before accepting it, decodes the exact
IFRT/Riegeli split record structure, inventories every embedded ELF, and uses
hash-pinned ROCm LLVM tools. The decoded record sizes are
`[56, 0, 1920, 0, 52425]`. The 1,920-byte wrapper record contains an empty
1,904-byte ELF plus a 16-byte trailer; its `.text` is empty and it has no
kernel, so it is explicitly rejected as ISA evidence. The final record
contains five ELFs. Selection is by the unique exact function symbol rather
than record position.

The retained result can be rechecked offline with:

```bash
cache=/tmp/skyrl-w8a8-compile-1783972228/compiler-artifacts/jax-cache/\
jit_candidate-1d44db98de8b8ab596105a0991d6a240efebc1d3d29ab28606b6365a828cb645-cache
/usr/bin/python3 -I -S -B rocm/inspect_w8a8_lora_isa.py "$cache" \
  --expected-cache-sha256 \
  d00d54b9e684852a1d933eafb332e058c5b232d79f687dc936951887e3f646a2
```

The selected code object is exactly 8,440 bytes with SHA-256
`606a80a508317af303966e5c2ca357d138d08828949c0dbfdcd73ccde1726389`.
It binds `skyrl_qwen35_w8a8_lora_forward` to
`amdgcn--amdhsa-amdgiz-gfx1100`, reports 34 SGPRs, 62 VGPRs, zero SGPR/VGPR
spills, zero private segment, wave32, maximum workgroup size 128, and maximum
grid `1x2x1`. Its disassembly contains exactly four
`v_wmma_i32_16x16x16_iu8` instructions, all with signed-operand metadata
`neg_lo:[1,1,0]`. Paired with the exact HLO/custom-call evidence, this proves
native RDNA3 signed-INT8 WMMA code generation for the fixed tiny Pallas
forward. It does **not** prove that the code was dispatched, that its result is
correct, or that it is faster than BF16. The Riegeli internal checksum
algorithm is not reimplemented; the retained artifact is instead anchored by
the whole-cache SHA-256 plus exact boundaries and decompression return codes.

The guarded compile consumed at most 867,360,768 bytes of physical VRAM,
reached 51 C junction temperature and 113 W sampled average power, and used no
additional swap. All 1,090 measured telemetry samples were within the fixed
limits, with a maximum sample gap of 0.0899 seconds. The process returned to
the exact 27,947,008-byte VRAM and 15,966,208-byte GTT baseline, remained
headless and unowned, reached runtime suspend for three consecutive handoff
samples, and left the whole-boot AMDGPU journal clean. These are compile and
native-code-generation results, not performance measurements.

The next separate rung is exactly one host-checked invocation of the same tiny
forward. Only after that source and its fresh nested-ELF gate are independently
reviewed will the ladder change one risk dimension at a time: signed base-only
semantics;
`K=128`; three row superblocks; small base/fused VJPs; K-scan lengths 8/40/144;
N-scan lengths 8/32/128/288; then the first real `K=2560,N=18432` gate/up
rectangle. The artificial `K=9216,N=18432` rectangle is not a model shape and
must not be run. Every rung retains a strict sub-100-ms candidate-dispatch gate
and a fresh-process exact-idle handoff. The user's 3% allowance applies to
Pallas attention gradients, not this quantized projection; the W8 CPU and
forward gates deliberately remain at 1%.

## Forward operation

For flattened input `X[M,K]`, rank `R=8`, and group size `G=64`, the JAX-visible
operation should enqueue several short kernels on JAX's existing HIP stream:

1. Quantize each BF16 input row/group once, producing packed `Xq` and FP32
   `sx[M,K/G]`. This avoids rescanning the same input for every output tile.
2. Compute `Z = X A` from the original input and BF16 LoRA A with FP32
   accumulation. Keep `Z[M,8]` in FP32 to match the semantic oracle.
3. Run tiled integer base GEMM. For each output tile and each K-group, issue
   four K=16 WMMA operations into INT32, convert that group accumulator to
   FP32, multiply by `sx[m,g] * sw[g,n]`, and add it to the FP32 output
   accumulator.
4. In the output epilogue, add `scaling * sum_r Z[m,r] * B[r,n]` in FP32 and
   cast once to the requested BF16 output. Optional bias/residual/gate fusion
   must be a separate, explicitly tested ABI variant.

At 32K tokens and K=2,560, materialized A8 uses 80 MiB plus 5 MiB of FP32
row/group scales; packed A4 uses 40 MiB plus the same scales. FP32 `Z` is only
1 MiB. These are per-live-projection workspaces, not tensors to retain across
all 32 layers. With rematerialization, backward should recompute them.

Group-local INT32 accumulation is comfortably safe. Symmetric W8 codes in the
oracle are in `[-127,127]`, so the worst absolute G=64 accumulator is
`64*127*127 = 1,032,256`. W4 is `64*7*7 = 3,136`. Reset the INT32 accumulator
at every group because the distinct scales require an FP32 rescale before
summing groups; a whole-K integer accumulator is not equivalent.

Quantization must reproduce the oracle's FP32 absolute-max reduction,
zero-group scale of one, round-to-nearest-even, symmetric clipping, and signed
two's-complement nibbles. Weight scales remain compact BF16 but are converted
to FP32 before multiplication. RMSNorm, attention/GDN state, and loss
reductions remain FP32/BF16 as specified in `MEGAKERNELS.md`; A4/A8 applies
only at the base-linear boundary.

## Backward contract

The first production backward should preserve the existing straight-through
contract, not invent a fast but different quantized gradient:

```text
dX_base = dY @ dequant(W).T
R       = scaling * dY @ B.T
dX      = dX_base + R @ A.T
dA      = X.T @ R
dB      = scaling * (X @ A).T @ dY
dscale  = sum(dY * ((X @ A) @ B))
```

- Implement the initial `dX_base` exactly: load a compact weight tile, convert
  its codes and BF16 scales to FP32, and contract it with FP32 `dY` using FP32
  products and accumulation. Never materialize a complete BF16/FP32 base
  weight or save it in the VJP residual. A dequantize-to-BF16 WMMA path changes
  the documented gradient because `code * scale` is not generally exactly
  representable in BF16; treat that as a separately gated relaxed-backward
  experiment rather than the oracle-preserving implementation.
- Keep `X`, compact codes/scales, A, B, and scaling as the custom-VJP residual.
  Frozen codes/scales have no cotangent.
- Compute `R`, `dA`, `dB`, and `dscale` from the original BF16 values with FP32
  products/reductions. Token superblocks should write unique partial `dA`/`dB`
  buffers followed by a fixed-order reduction; do not make nondeterministic
  atomics the default.
- Cast final `dX`, `dA`, and `dB` once to their declared dtypes. Quantizing dY
  for a native integer transpose GEMM is a later accuracy experiment, not the
  W8A8/W4A4 forward implementation.

This means initial training speedups will be smaller than forward-only GEMM
speedups. Backward still benefits from compact frozen-weight residency, but
the exact tiled FP32 dequantize/dot path has no IU4/IU8 matrix-rate advantage.

## Typed-FFI boundary

Use separate targets for W8 and W4 rather than a hot-path bit-width branch.
The forward ABI should take row-major BF16 X, signed W8 or packed U8 W4 codes,
BF16 weight scales, BF16 A/B, and FP32 scaling, and return BF16 Y. If internal
subdispatches need `Xq`, `sx`, or `Z`, make them explicit auxiliary FFI outputs
in the first implementation so XLA owns their stream-safe lifetime. Only move
them to `ScratchAllocator` after its asynchronous lifetime is verified on this
JAX/ROCm plugin; never synchronize the device merely to free scratch.

The C++ handler shape is:

```cpp
xla::ffi::Error Forward(
    xla::ffi::BufferR2<xla::ffi::BF16> x,
    xla::ffi::BufferR2<xla::ffi::S8> w8,  // U8 for packed W4 target
    xla::ffi::BufferR2<xla::ffi::BF16> sw,
    xla::ffi::BufferR2<xla::ffi::BF16> a,
    xla::ffi::BufferR2<xla::ffi::BF16> b,
    xla::ffi::BufferR0<xla::ffi::F32> scaling,
    xla::ffi::ResultBufferR2<xla::ffi::BF16> y,
    hipStream_t stream);
```

Bind the stream with `Ctx<PlatformStream<hipStream_t>>()`, export with
`XLA_FFI_DEFINE_HANDLER_SYMBOL`, load by `ctypes`, wrap with
`jax.ffi.pycapsule`, and register for platform `ROCM` with typed-FFI API version
1. The Python call uses `jax.ffi.ffi_call` (custom-call API version 4) and
explicit row-major layouts. Wrap forward/backward calls in `jax.custom_vjp`;
raw FFI has no autodiff rule. Retain the `ctypes.CDLL` object for the lifetime
of the process; the capsule contains only a function pointer and must not
outlive the loaded shared library.

Handlers enqueue work on the supplied stream and return without
`hipDeviceSynchronize`. Check launch submission with `hipPeekAtLastError` and
convert failure to `xla::ffi::Error::Internal`. Validate every dimension,
group-size/rank invariant, buffer byte count, and alignment in both Python and
the handler.

## Bounded dispatch and acceptance gates

One handler may enqueue multiple dependency-ordered kernels, but no individual
kernel should cover an unbounded full projection. Start with conservative M/N
superblocks (for example M<=2,048 and N<=1,024), profile their worst Qwen shape,
and only enlarge them while the guarded profiler keeps every dispatch below the
chosen watchdog budget. Tail kernels handle nonmultiples; K is padded to 64.
There is no inter-block synchronization inside a launch.

The compile proof is not a reason to enable either route. Required gates are:

1. Exact pack/unpack, quantization, forward, and VJP comparisons to
   `quantized_lora.py`, including zeros, tails, and repeated determinism.
2. Fresh-process isolated GPU checks at 64-512 tokens, then serialized 1K-16K
   shapes under `profile_rocm.py`; 24K/32K comes only after clean smaller runs.
3. No BF16 shadow weight and no full dequantized weight in XLA buffer
   assignment. Peak VRAM must realize the documented 3.2-3.8 GiB W8 or
   4.9-5.8 GiB W4 base-model saving, depending on whether the tied matrix is
   kept BF16.
4. Treat at least 1.2x isolated projection throughput and a measurable
   end-to-end step gain as the W8 continuation gate. W4 should materially beat
   W8, not merely save memory, before accepting its higher numerical risk.
5. SFT loss/gradient and GRPO logprob/KL/reward checks must pass independently
   for each projection family before model-wide activation.

The only current performance conclusion is feasibility: W8A8 and W4A4 can be
compiled natively on this gfx1100 toolchain, the exact tiny Pallas W8A8 forward
emits signed-IU8 WMMA, and the compact formats offer large, already-calculated
residency savings. No Pallas W8 executable has run. Its speed advantage remains
an experiment because runtime correctness, input quantization, group
rescaling, LoRA work, backward, and bounded-launch overhead have not been
benchmarked against the complete BF16 projection.
