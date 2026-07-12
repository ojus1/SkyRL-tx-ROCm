# gfx1100 quantized LoRA FFI route

This note turns the grouped W8/W4 semantics in
`skyrl/tx/kernels/quantized_lora.py` into a concrete ROCm compilation plan. It
is scoped to Qwen3.5-4B on the RX 7900 XTX (`gfx1100`). Nothing described here
is enabled in model code yet.

## What is proven locally

The installed stack is HIP/ROCm 7.2.0, Clang 22, Composable Kernel 1.2.0,
rocWMMA 2.2.0, hipBLASLt 1.2.1, and JAX/JAXlib 0.10.2. Inspection and
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
and disassembles the code object. On the installed stack it produced a
6,304-byte HSACO containing both:

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
compiled natively on this gfx1100 toolchain, and the compact formats offer
large, already-calculated residency savings. Their speed advantage remains an
experiment because input quantization, group rescaling, LoRA work, backward,
and bounded-launch overhead are not represented by the fragment compile probe.
