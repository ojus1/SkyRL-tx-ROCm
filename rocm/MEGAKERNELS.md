# Qwen3.5-4B ROCm fused-kernel design

This document is an implementation plan for LoRA SFT/GRPO on one Radeon RX
7900 XTX (`gfx1100`, 24 GiB) with JAX 0.10.2 and ROCm 7.2. It covers the text
model only and the exact Qwen3.5-4B geometry:

- hidden size 2,560, intermediate size 9,216, rank-8 LoRA;
- 32 layers: 24 Gated DeltaNet (GDN) and 8 full-attention layers;
- full attention: 16 query heads, 4 KV heads, head dimension 256;
- GDN: 16 key heads, 32 value heads, key/value dimension 128, chunk size 64;
- tied 248,320-token embedding/LM head;
- sequence buckets through 32,768 tokens.

Geometry and checkpoint byte totals refer to `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

Measured facts are identified as such. FLOP counts, time projections, and
memory savings are estimates until verified with profiler traces and XLA buffer
assignments.

## Rejected design: one whole-layer or whole-model dispatch

A literal whole-layer or whole-model GPU dispatch is rejected, including a HIP
implementation. Transformer stages need grid-wide synchronization between
projections, attention or recurrence, and subsequent projections. Pallas/Triton
does not provide a safe cooperative grid barrier, while a persistent HIP kernel
would have poor occupancy and run long enough to recreate the GPU-watchdog risk.
One exact BF16 full-attention layer streams about 205 MiB (0.200 GiB) of frozen
weights and one GDN layer about 215 MiB (0.210 GiB); the complete text model is
7.834 GiB. The problem is repeated, globally synchronized layer work, not
"several GiB" in one layer.

Here, a "megakernel" means a JAX operation or FFI handler composed of several
short, dependency-ordered GPU dispatches. Every underlying dispatch must have a
bounded work estimate and an independently measured duration.

## Current source hooks

- LoRA routing and low-rank products: `skyrl/tx/layers/lora.py`,
  `LoRAMixin.apply_lora` (currently around line 111).
- Generic multi-adapter sorting: `skyrl/tx/layers/util.py:66`,
  `prepare_routing`.
- Full QKV, gate, attention, and O projection:
  `skyrl/tx/models/qwen3_5.py:296-377`, `Qwen3_5Attention`.
- Chunked FP32 GDN: `skyrl/tx/models/qwen3_5.py:101-255`,
  `chunk_gated_delta_rule`.
- GDN projections and output: `skyrl/tx/models/qwen3_5.py:380-595`,
  `Qwen3_5GatedDeltaNet`.
- Fused gate/up and down projections: `skyrl/tx/models/qwen3_5.py:598-639`,
  `Qwen3_5MLP`.
- Norm and residual boundaries: `skyrl/tx/models/qwen3_5.py:642-699`,
  `Qwen3_5DecoderLayer`.
- Per-layer training rematerialization: `skyrl/tx/models/qwen3_5.py:702-721`
  and its training call around lines 790-802.
- Final RMSNorm: `skyrl/tx/models/qwen3_5.py:846`.
- Tied LM head: `skyrl/tx/models/qwen3_5.py:884-902`.
- Chunked logprob computation: `skyrl/tx/utils/logits_processor.py:82-130`.
- Training loss integration: `skyrl/backends/jax.py:334-411`.
- Portable grouped-quantization semantics:
  `skyrl/tx/kernels/quantized_lora.py` (reference only, not a GPU speed path).

Line numbers describe the worktree when this document was written; symbols are
the stable integration points.

## Exact equations used by custom VJPs

### Frozen-base LoRA linear

For scale `s`, frozen base weight `W`, and trainable `A`, `B`:

```text
U  = X A
Y  = X W + s U B

R  = s dY B^T
dX = dY W^T + R A^T
dA = X^T R
dB = s U^T dY
```

No `dW` is required. Accumulate `dA` and `dB` in FP32. Token superblocks may
write unique partial gradients followed by a fixed-order reduction; atomics are
not required.

For input embedding token IDs `t_i`, the LoRA form and gradients are:

```text
Y_i       = E[t_i] + s A[t_i] B
dB        = s sum_i A[t_i]^T dY_i
dA[t_i]  += s dY_i B^T
```

`E` is frozen. Repeated token IDs require a deterministic segmented reduction
for `dA`; a plain parallel scatter is not deterministic.

### RMSNorm

For width `H`, `r = rsqrt(mean(x^2) + eps)`, and `y = w * x * r`:

```text
g  = dy * w
dx = r * (g - x * r^2 * mean(g * x))
```

Here `w` means the effective scale. For `Qwen3_5RMSNorm`, whose checkpoint
parameter is a delta, the effective scale is `1 + weight`; the gated GDN norm
stores its scale directly.

### SwiGLU and attention output gate

```text
p  = silu(g) * u
s  = sigmoid(g)
du = dp * silu(g)
dg = dp * u * (s + g * s * (1-s))

z     = O * sigmoid(gate)
dO    = dz * sigmoid(gate)
dgate = dz * O * sigmoid(gate) * (1-sigmoid(gate))
```

### GDN recurrence

After Q/K normalization, with `q` also scaled by `1/sqrt(Dk)` and state shaped
`[Dk,Dv]`:

```text
alpha = exp(g)
D     = alpha * S_prev
m     = k^T D
delta = beta * (v-m)
S     = D + k delta^T
o     = q^T S
```

Reverse one token, after adding the future-state cotangent to `barS`:

```text
barS += q baro^T
barq += S baro

barD     += barS
bark     += barS delta
bardelta += barS^T k

barbeta += dot(bardelta, v-m)
barv    += beta * bardelta
barm    -= beta * bardelta

bark += D barm
barD += k barm^T

baralpha  = sum(barD * S_prev)
barS_prev = alpha * barD
barg      = alpha * baralpha
```

The optimized chunked implementation may use the existing WY equations, but
its custom VJP must remain equivalent to this recurrence and retain FP32 state.

### Tied linear target logprob

For tied embedding row `E_j`, target `y`, and upstream cotangent `c`:

```text
z_j    = h E_j
logp_y = z_y - logsumexp_j(z_j)
dz_j   = c * (1[j=y] - softmax(z)_j)
dh     = sum_j dz_j E_j
```

The tied embedding is frozen, so the training kernel returns `dh` but not
`dE`. The outer CE, importance-sampling, PPO, or CISPO expression supplies `c`;
the fused linear-logprob operation need not know the loss type.

## Bounded-dispatch policy

The clean isolated 16K BF16 Pallas attention measurement was:

- forward median: 184.4 ms;
- forward plus backward median: 895.0 ms;
- all finite, no driver reset or log error.

Those results imply about 11.9 TFLOP/s for its forward and 6.2 TFLOP/s for its
backward. The following estimates use 6 TFLOP/s as a deliberately conservative
planning rate. They are not runtime guarantees.

Start dense projection work at 512 tokens per dispatch. Autotune upward only
after validation, with 2,048 tokens as the initial hard maximum:

| Underlying dispatch | Work at proposed maximum | Estimate at 6 TFLOP/s |
|---|---:|---:|
| QKV projection, `M=2048` | 107.4 GFLOP | 17.9 ms |
| O projection, `M=2048` | 42.9 GFLOP | 7.2 ms |
| Gate/up projection, `M=2048` | 193.3 GFLOP | 32.2 ms |
| Down projection, `M=2048` | 96.6 GFLOP | 16.1 ms |
| Concatenated GDN input projection, `M=2048` | 129.5 GFLOP | 21.6 ms |
| GDN output projection, `M=2048` | 42.9 GFLOP | 7.2 ms |
| Native-GQA query chunk, `Cq=128`, `T=32K` | 68.7 GFLOP | 11.5 ms |
| Tied-head token chunk, `M=128` | 162.7 GFLOP | 27.1 ms |
| GDN 1,024-token superblock | about 5-6 GFLOP | below 1 ms compute |

Base transpose GEMMs in backward have the same bound. Gate/up recomputation
and its transpose GEMM must remain separate underlying dispatches. Require an
observed duration below 100 ms for every new custom dispatch; otherwise shrink
its token/query/vocabulary superblock.

## Operation and custom-VJP boundaries

Build a sequence of bounded fused operations, not one model-wide custom call.
Each row below is one JAX-visible operation with an explicit `custom_vjp`; its
HIP implementation may enqueue a vendor GEMM and one or more short kernels on
the supplied stream. A raw FFI call is never an autodiff boundary by itself.

| JAX-visible operation | Values intentionally materialized for its consumer |
|---|---|
| Input embedding | first hidden state after frozen lookup plus embedding LoRA |
| Full-attention input | normalized/rotated Q, K, V, and gate |
| Native GQA | attention output (online-softmax LSE only if its VJP needs it) |
| Gated O projection | first residual hidden state |
| RMS + gate/up + SwiGLU | 9,216-wide product |
| Down projection | second residual/layer output |
| GDN input | canonical Q/K/V, z, a, and b; no repeated heads |
| GDN preparation/core | U/W plus state checkpoints at bounded superblocks |
| GDN gated output | first residual hidden state |
| Final tied head | target logprobs only |

The original layer input remains available for residual addition. Frozen base
weights and frozen quantization scales have no cotangent. Each projection VJP
returns `dX`, `dA`, and `dB`; reductions across token superblocks happen in a
separate deterministic dispatch. Norm, RoPE, gate, convolution, recurrence, and
quantize/dequantize derivatives belong to the operation that owns their forward
calculation. This partition keeps every recomputation and failure attributable
to one bounded stage and permits a stage-by-stage rollback to the JAX reference.

## Stage fusion map

### 1. LoRA routing

Use direct A/B selection and ordinary GEMMs for the common batch-size-one path.
Keep the sorted ragged-dot path for true mixed-adapter batches. If larger mixed
batches become important, prepare their token permutation and group sizes once
per model pass rather than once per LoRA layer.

For backward, each token superblock writes unique FP32 partial `dA` and `dB`.
A deterministic second-stage tree reduction avoids contended atomics.

Treat the input embedding as its own fused operation: load the frozen embedding
row (or dequantize its W8/W4 row), load `A[token_id]`, apply the rank-8 B
product and scale, and write one hidden vector. Its VJP uses a stable token-ID
sort/segment reduction for repeated-row `dA` plus a bounded FP32 reduction for
`dB`. The tied output-head operation remains frozen and does not apply the
embedding LoRA, matching the current model.

### 2. Full-attention input projection

Target interface:

```text
(Q, K, V, gate) = qkv_lora_rope(
    hidden,
    input_rms_weight,
    q_norm_weight,
    k_norm_weight,
    Wqkv, A, B,
    positions, adapter_index)
```

The fused operation first applies the decoder's 2,560-wide input RMSNorm while
leaving the unnormalized `hidden` available for the later residual. The
projection epilogue should add LoRA, split the existing interleaved QKV layout,
apply the distinct per-head `q_norm` and `k_norm` delta weights, apply
64-dimensional partial RoPE, and write canonical Q, K, V, and gate directly.
It must not write a raw 10,240-wide QKV tensor and transform it in a later
kernel. All three ordinary RMSNorm scales in this operation use the checkpoint
contract `1 + weight`; `q_norm_weight` and `k_norm_weight` are not interchangeable.

Backward applies inverse RoPE, the distinct per-head Q/K RMS VJPs, reassembles
raw dQKV, then uses the frozen-base/LoRA equations and finally the input RMS
VJP. A standard library GEMM followed by a Pallas epilogue cannot eliminate the
raw projection buffer; production fusion requires the custom kernel to own the
GEMM.

### 3. Watchdog-bounded native GQA

Use a custom VJP with absolute causal positions and no K/V head repeat:

```text
for q_start in range(0, T, Cq):
    O_chunk, LSE_chunk = rectangular_forward(
        Q[q_start:q_start+Cq], K, V, key_mask, q_start)
```

Start with `Cq=128`. Forward, dQ, and dKV are separate bounded dispatches.
Keep online-softmax statistics and dK/dV accumulation in FP32. Map query head
`hq` to KV head `hq // 4`. Accumulate dK/dV through aliased FP32 buffers in a
fixed query-chunk order; do not use atomics.

### 4. Attention gate, O projection, and first residual

Target interface:

```text
hidden1 = gated_o_lora_residual(O, gate, residual, Wo, Ao, Bo)
```

Load `O` and gate, apply sigmoid gating while loading projection tiles, add the
LoRA contribution, and add the residual in the output epilogue. This removes
the full gated-attention tensor and the pre-residual projection result.

### 5. Post-attention RMS, gate/up projection, and SwiGLU

The existing `(1,1)` interleaved gate/up layout is suitable for a paired
epilogue:

```text
product = rms_gateup_swiglu_lora(
    hidden1, rms_weight, Wgateup, Agateup, Bgateup)
```

Write only the 9,216-wide SwiGLU product. For memory-efficient backward, do not
save the 18,432-wide gate/up tensor. Recompute gate/up one 512-2,048-token block
at a time, immediately form `dg`/`du`, and run the frozen-base and LoRA
backward equations.

Do not fuse the down projection into this stage. It reduces over all 9,216
product features; recomputing gate/up for each down-output tile would multiply
the expensive projection work. The SwiGLU product is a required global-memory
boundary.

### 6. Down projection and second residual

```text
hidden2 = down_lora_residual(product, hidden1, Wdown, Adown, Bdown)
```

The residual is a projection-output epilogue. The product remains materialized.

### 7. GDN projection, convolution, and head mapping

Target interface:

```text
(Q, K, V, z, a, b) = gdn_input(
    hidden, input_rms_weight,
    Wqkv, Wz, Wa, Wb,
    Aqkv, Bqkv, Az, Bz, Aa, Ba, Ab, Bb,
    adapter_index, attention_mask)
```

The decoder input RMSNorm is part of this operation, while the original
unnormalized `hidden` remains the residual boundary. Prepack the frozen QKV
(8,192), z (4,096), a (32), and b (32) weights as one 12,352-wide logical
projection while retaining four independent LoRA pairs. The stronger kernel
computes the QKV projection for a token tile plus its three-token halo, applies
depthwise causal convolution and SiLU per channel, and writes canonical Q/K/V.
Halo recomputation at tile boundaries is negligible. Backward must include the
input RMS VJP after summing cotangents from all four projection branches.

Do not repeat 16 Q/K heads to 32. Value head `hv` reads key head `hv // 2`
inside the GDN kernel.

### 8. GDN core

This operation owns the pre-recurrence transforms: L2-normalize Q/K and compute
`beta=sigmoid(b)` in the input dtype; cast Q/K/V/beta to FP32; scale Q by
`1/sqrt(128)`; and compute `g=-exp(A_log)*softplus(a+dt_bias)` in FP32. A masked
token uses `beta=0,g=0`, making its state transition an identity. `A_log` and
`dt_bias` are frozen, but the VJP must propagate through sigmoid/softplus and
Q/K normalization to the LoRA-bearing qkv/a/b projections. Head `hv` selects
Q/K head `hv//2` without a materialized repeat.

Use three bounded stages:

1. Parallel per-64-token WY preparation and 64x64 correction solve, producing
   only U, W, and minimal gamma data.
2. State/output execution for 8-16 chunks per dispatch, carrying the 2 MiB FP32
   state through HBM at each 512-1,024-token boundary.
3. Reverse-superblock custom VJP using the exact recurrence above.

Saving FP32 state every 1,024 tokens costs about 34 MiB at 16K and 66 MiB at
32K, instead of saving one state for every 64-token chunk.

### 9. GDN gated norm, output projection, and residual

The GDN output projection should load core output and z, calculate per-head FP32
RMS statistics plus SiLU gating, and perform the 4,096-to-2,560 base+LoRA
projection with residual addition. Do not write a separate gated-normalized
4,096-wide tensor.

### 10. Final RMS and tied linear logprob

The launcher currently sets `loss_chunk_size=64`, which already bounds live
logits and does not materialize full `[T,V]` logits. The current logical
per-chunk logits are 30.3125 MiB in BF16 or 60.625 MiB in FP32. A fused
target-logprob custom VJP eliminates this per-chunk tensor, not a full-sequence
tensor. `M=128` is the first proposed fused-head bucket, not the current setting;
its BF16/FP32 logical logits would be 60.625/121.25 MiB.

The tied embedding is approximately 1.184 GiB, larger than cache. With token
chunk 64 it is nominally streamed at least 256 times at 16K (about 303.1 GiB)
and 512 times at 32K (about 606.3 GiB) per forward, before backward or remat.
Fusion does not itself remove those repeated vocabulary reads; it permits a
larger token chunk without a larger logits buffer. `M=128` halves those counts
to 128/256 (about 151.6/303.1 GiB). After that path is validated, test 256
tokens (325.5 GFLOP, estimated 54 ms at 6 TFLOP/s).

Forward should use split-vocabulary online max/sum and emit only target
logprobs. A watchdog-safe, deterministic backward can make each vocabulary
superblock write a unique partial dHidden and reduce the partials in a second
kernel. With 4,096 vocabulary items per superblock and 128 tokens, that FP32
workspace is 76.25 MiB. Alternatively, scan 61 vocabulary superblocks through
an aliased 1.25 MiB dHidden accumulator. Neither design needs atomics.

Keep the elementwise CE/PPO/CISPO/importance-sampling loss outside this custom
VJP. The `[B,T]` logprobs are small, and the separation keeps one kernel valid
for every loss type.

## Logical temporary-memory opportunities

These are theoretical logical tensor sizes. They are not additive peak-memory
claims; XLA may alias or reuse some storage. Confirm realized savings from XLA
buffer assignments and fresh-process allocator telemetry.

| Candidate eliminated or avoided | 16K | 32K |
|---|---:|---:|
| Each materialized hidden/RMS output | 80 MiB | 160 MiB |
| Native GQA removes repeated K+V | 192 MiB | 384 MiB |
| Attention gated temporary | 128 MiB | 256 MiB |
| Raw gate+up tensor | 576 MiB | 1,152 MiB |
| Each pre-residual hidden result | 80 MiB | 160 MiB |
| Native grouped GDN removes repeated FP32 Q+K | 256 MiB | 512 MiB |
| GDN k_beta + v_beta | 512 MiB | 1,024 MiB |
| GDN decay mask + correction L | 256 MiB | 512 MiB |
| GDN correction RHS | 512 MiB | 1,024 MiB |
| GDN intra-attention + U-minus-state | 384 MiB | 768 MiB |
| Current per-loss-chunk BF16 logits (`M=64`) | 30.3125 MiB | 30.3125 MiB |
| Proposed fused-head BF16 logits (`M=128`) | 60.625 MiB | 60.625 MiB |

GDN still needs U/W, totaling about 512 MiB at 16K and 1 GiB at 32K, unless a
later on-the-fly design proves faster. Fusing preparation and state stages can
avoid roughly 1.6/3.3 GiB of other logical FP32 intermediates at 16K/32K, but
the actual peak delta will be smaller if XLA already reuses buffers.

## Quantization routes for `gfx1100`

### Hardware constraints

The official ROCm precision table lists INT8, FP16, and BF16 matrix-core support
for RDNA3, including the RX 7900 XTX class, but no native FP8 or floating-point
FP4 matrix-core support. Integer four-bit is a separate case: the official
RDNA3 ISA lists `V_WMMA_I32_16X16X16_IU4`, and AMD's RX 7900 XTX WMMA guide
documents both signed/unsigned four-bit inputs with INT32 accumulation. The
installed Composable Kernel headers also contain gfx11 IU4 WMMA paths. rocWMMA
supports `gfx1100` as a wave32 target and exposes INT8 fragments; hipBLASLt
advertises INT8 types at the library API. Every exact
transpose/layout/scale/epilogue combination still needs an algorithm-support
query or a custom HIP kernel and benchmark on this installed stack.

Consequently, "8-bit" means signed INT8 here, not FP8. W8A8 can use native
INT8-to-INT32 matrix instructions. W8A16 must dequantize weights into a
BF16/FP16 compute tile. W4A4 can use native integer-four-bit WMMA; W4A8 must
unpack weights to INT8, and W4A16 must dequantize to BF16. AMD's published
instruction-rate table gives the RX 7900 XTX the same theoretical
512 operations/clock/CU for BF16 and IU8, versus 1,024 for IU4. Therefore W8's
speed opportunity is reduced memory/register traffic rather than a higher math
ceiling; W4A4 has a theoretical 2x matrix-instruction ceiling, before group
rescaling, LoRA, reductions, and quality constraints.

Primary hardware references:

- [ROCm data types and precision support](https://rocm.docs.amd.com/en/latest/reference/precision-support.html)
- [rocWMMA API and supported architectures](https://rocm.docs.amd.com/projects/rocWMMA/en/develop/api-reference/api-reference-guide.html)
- [hipBLASLt precision support](https://rocm.docs.amd.com/projects/hipBLASLt/en/develop/reference/data-type-support.html)
- [AMD GPUOpen: RDNA3 WMMA on RX 7900 XTX](https://gpuopen.com/learn/wmma_on_rdna3/)
- [AMD RDNA3 shader ISA](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf)

### Frozen-model memory estimates

The local safetensor headers contain 8,411,510,272 bytes (7.8338 GiB) of
Qwen3.5-4B language-model parameters, excluding vision/MTP tensors: 249
two-dimensional frozen matrices with
4,204,789,760 elements plus 1,930,752 bytes of other tensors. The estimates
below keep non-matrix tensors in their checkpoint types. Both W8 and W4 use one
BF16 scale per 64 input values per output; W4 stores packed signed nibbles. They
exclude LoRA/optimizer/activation storage and are not allocator-peak
measurements.

| Frozen-weight route | Estimated model storage | Saving from BF16 |
|---|---:|---:|
| BF16 checkpoint baseline | 8,411,510,272 B (7.8338 GiB) | - |
| W8 group-64, including tied embedding/head | 4,338,120,192 B (4.0402 GiB) | 3.7936 GiB |
| W8 group-64, tied embedding/head kept BF16 | 4,953,953,792 B (4.6137 GiB) | 3.2201 GiB |
| W4 group-64, including tied embedding/head | 2,235,725,312 B (2.0822 GiB) | 5.7516 GiB |
| W4 group-64, tied embedding/head kept BF16 | 3,169,408,512 B (2.9517 GiB) | 4.8821 GiB |

These savings exist only if the packed representation is the canonical device
parameter and kernels dequantize one tile at a time. Keeping a BF16 shadow copy
on device, or allowing XLA to constant-fold a full dequantized matrix, forfeits
the gain. Represent W4 as explicitly packed `uint8` nibbles plus scale buffers;
do not assume JAX's logical four-bit dtype occupies half a byte in device
buffers. Loader and FFI layouts are therefore part of the quantized-kernel work,
not a later optimization.

Quantize ordinary frozen projections first while keeping the tied
embedding/head BF16. That isolates quality changes and still saves 3.2-4.9 GiB.
The tied matrix is then the highest-leverage bandwidth experiment: at the
current 64-token loss bucket its 1.184 GiB BF16 representation is nominally
read about 303/606 GiB at 16K/32K per forward. W8 approximately halves that
weight traffic to about 156/313 GiB; group-64 W4 reduces it to about 81/161 GiB
after scales. Those are logical traffic bounds, not end-to-end speedup
predictions. Because the matrix is tied, quantizing it changes both token lookup
and the vocabulary head; duplicating a BF16 embedding just for lookup would
reduce the memory saving and must be reported explicitly.

### Ranked implementation routes

1. **W8A16O16, frozen base only.** Keep BF16 activations, residuals, LoRA A/B,
   and outputs; dequantize each W8 tile into the BF16 GEMM path. This is the
   safest first production route and captures roughly half-weight residency.
   Because `gfx1100` has no mixed INT8-by-BF16 matrix instruction, speed can be
   neutral or negative if dequantization is not hidden behind weight loads.
2. **W8A8O16 projection inputs.** Quantize each 64-value input group with a
   dynamic row/group scale, use matching output/group W8 scales, accumulate
   INT32 inside each group, and sum rescaled groups in FP32 before the BF16
   output/LoRA/residual epilogue. This is the lower-risk native-integer speed
   route, although its published matrix-instruction ceiling equals BF16; gains
   must come from reduced traffic and fragment/register footprint. Keep the
   LoRA branch on BF16 input and accumulate `dA`/`dB` in FP32. Start backward
   `dX_base` with W8A16; quantizing `dY` to obtain an INT8 transpose GEMM is a
   separate accuracy gate. Group-64 scales force an FP32 rescale/sum after each
   K group, so one stock whole-K INT8 GEMM does not implement this contract. A
   coarser per-row/per-output scale scheme that permits one hipBLASLt GEMM is a
   separate format and must pass the same quality gates.
3. **W4A16/W4A8 group-64 base weights.** Implement W4A16 by unpacking and
   dequantizing per K group, and W4A8 by expanding nibbles to INT8 tiles. Group
   scales require separately rescaled partial accumulations, so these paths
   trade more instructions for the largest residency and tied-head bandwidth
   saving. Treat them as memory routes until profiling proves a speed gain.
4. **W4A4O16 native IU4.** Dynamically quantize each activation K group to
   signed four-bit, feed packed W4/A4 fragments to the native IU4-to-INT32 WMMA,
   rescale each group in FP32, and emit BF16. This is the only ranked route with
   a higher published matrix-instruction ceiling than BF16 on RX 7900 XTX, but
   it is also the highest-risk projection path numerically. Validate tied head,
   MLP, attention projections, and backward straight-through behavior
   independently; do not turn on model-wide A4 in its first experiment. Start
   base `dX` from the dequantized W4 straight-through reference; a native IU4
   transpose path also quantizes `dY` to A4 and is a later, separate gate.
5. **O8 layer-boundary checkpoints.** Quantize only after the residual addition
   and store a per-token or small-block scale, then dequantize before the next
   layer. At 32K, one `[1,T,2560]` boundary falls from 160 MiB BF16 to about
   80 MiB plus scales. Do not store attention softmax data, Q/K normalization
   statistics, GDN U/W, or recurrent state in O8. The benefit must be measured
   with rematerialization enabled because XLA may not retain every boundary.
6. **Blockwise INT8 Adam moments.** The exact two-adapter, rank-8 shape has
   34,512,896 trainable LoRA elements. Its two BF16 moment arrays occupy about
   131.7 MiB; two INT8 moment arrays would use about 65.8 MiB plus scales, saving
   only about 65.8 MiB. Perform the update and bias correction in FP32. This is
   lower priority than compacting inactive adapter slots or quantizing the base.
7. **O4 boundaries and four-bit optimizer state.** Defer. Integer-four-bit
   matrix support does not make four-bit residual checkpoints or Adam moments
   numerically safe, and those encodings risk corrupting residual, norm,
   softmax, recurrence, and optimizer dynamics for modest additional memory
   beyond the ranked routes. User acceptance makes them valid experiments, not
   valid defaults without the same model-level gates.

If projection input codes are materialized for reuse, a 32K-by-2,560 tensor is
80 MiB in A8 plus 2.5 MiB of group-64 BF16 scales, or 40 MiB in packed A4 plus
the same 2.5 MiB of scales, versus 160 MiB in BF16. An on-tile design can avoid
that buffer, but must not rescan and requantize the same K values for every N
tile; compare both choices in the buffer assignment and trace.

Regardless of route, keep RMS/softmax reductions, online-softmax LSE, GDN
decays and state, deterministic gradient reductions, and optimizer update math
in FP32. Keep canonical attention Q/K/V and GDN Q/K/V in BF16 initially. A8/A4
are input encodings at base-GEMM boundaries, not permission to make the GDN
state or attention reduction four/eight-bit.

### Quantized projection VJP contract

For K-group `g`, W8A8 uses dynamic row/group activation scale `sx[m,g]` and
frozen output/group weight scale `sw[g,n]`:

```text
Xq[m,g,:] = quantize_int8(X[m,g,:] / sx[m,g])
Wq[g,:,n] = frozen quantized W[g,:,n]
P[m,g,n]  = int32_dot(Xq[m,g,:], Wq[g,:,n])
Y_base    = sum_g float(P[m,g,n]) * sx[m,g] * sw[g,n]
Y         = Y_base + s * (X A) B
```

W4A4 uses the same grouped equation with signed four-bit codes and native IU4
dot products; W4A8 expands the weight code to INT8 before the dot product.

The fused operation defines the activation quantizer's backward convention
explicitly. The semantic baseline uses a straight-through base `dX` against the
dequantized frozen weight; a clipped or separately quantized transpose route is
a new accuracy experiment, not an interchangeable implementation detail.
Frozen `Wq` and scales return no cotangent. The VJP returns LoRA `dX`, `dA`, and
`dB` from the BF16 inputs with FP32 reductions. Do not silently use the
quantized `Xq` to form LoRA gradients. W4 group scales require one rescaled
accumulator per K group before their FP32 sum.

[`skyrl/tx/kernels/quantized_lora.py`](../skyrl/tx/kernels/quantized_lora.py)
is the portable JAX semantic oracle for this contract. It implements group-64
W8/W4 storage, signed nibble packing, dynamic A8/A4 quantization,
INT32-within-group/FP32-across-group forward arithmetic, and a custom VJP with
dequantized-base straight-through `dX` plus FP32-computed LoRA gradients. Its focused
tests are in
[`tests/tx/kernels/test_quantized_lora.py`](../tests/tx/kernels/test_quantized_lora.py).
The oracle uses portable JAX integer operations and streams compact groups in
its reference VJP; it is not a GPU speed path and must never be selected as the
production model implementation.

O8 is a separate layer-boundary quantize/dequantize operation with its own
custom VJP and scale metadata. Mixing projection A8 and boundary O8 in the same
first experiment would make quality regressions impossible to attribute.

## Pallas versus HIP/FFI

| Stage | Pallas/JAX status | HIP/FFI decision |
|---|---|---|
| RMS, RoPE, masks, sigmoid, residual, depthwise conv | Suitable now | Usually unnecessary |
| Rank-8 LoRA products and deterministic reductions | Suitable now | Optional |
| Bounded native GQA | Preferred first production route after validation | Fallback only if Pallas remains unstable |
| Base QKV/O GEMM with nonstandard epilogue | Correctness prototype possible | Recommended for production speed and true buffer elimination |
| Gate/up GEMM with paired SwiGLU epilogue | Prototype possible | Recommended |
| GDN 64x64 solve and state superblocks | CPU/Pallas correctness prototype | Recommended for LDS/register control |
| Split-vocabulary tied linear-logprob VJP | Multi-dispatch prototype possible | Recommended for production WMMA/reduction quality |
| W8A16 dequantized base GEMM | Weak Pallas prototype | HIP required for a production fused load/dequant path |
| W8A8 base GEMM | No dependable Pallas INT8-matrix path | HIP/rocWMMA or a proven hipBLASLt algorithm |
| W4A16/W4A8 base GEMM | Weak unpack/dequant prototype | HIP; dequantize to BF16 or unpack to INT8 |
| W4A4 base GEMM | Portable oracle only | HIP native IU4 WMMA or a proven installed CK path |
| O8 checkpoint boundary | Suitable correctness prototype | HIP optional after measured benefit |
| Whole layer/model single dispatch | Rejected | Rejected |

No mathematical stage requires FFI if multiple Pallas calls are allowed. FFI is
the production choice where owning the base GEMM, explicit LDS scheduling, or a
deterministic global reduction is necessary to beat a vendor GEMM plus
materialized intermediates.

The local toolchain already supplies `hipcc`, `libamdhip64`, `librocblas`,
`libhipblaslt`, JAX typed-FFI headers, and `jax.ffi.ffi_call` aliases/layouts.
An FFI handler should obtain JAX's stream with
`xla::ffi::PlatformStream<hipStream_t>`, enqueue bounded kernels, and use
`xla::ffi::ScratchAllocator` for temporary storage. Since `pybind11` is not
installed, exported handler symbols can be loaded with `ctypes`, wrapped by
`jax.ffi.pycapsule`, and registered for platform `ROCM`. Commit sources and
build scripts, not compiled binaries.

Suggested future source tree:

```text
skyrl/tx/kernels/rocm/
  pallas_attention.py
  pallas_lora.py
  pallas_logprob.py
  ffi/
    bindings.cc
    dense_epilogues.hip
    gdn.hip
    linear_logprob.hip
    CMakeLists.txt
    __init__.py
```

Every forward/backward FFI pair must be wrapped in an explicit JAX
`custom_vjp`; raw `ffi_call` does not provide autodiff.

## Staged validation gates

Every candidate must have an explicit config switch and retain a callable BF16
reference at the same operation boundary. Use fixed inputs and paired runs; do
not compare unrelated training trajectories.

1. **CPU semantics:** run Pallas in `interpret=True`; compare every forward and
   VJP against the current JAX implementation in FP16/BF16, including padding,
   chunk boundaries, rank-8 A/B gradients, and repeated-run determinism.
2. **Small isolated GPU:** one subprocess per shape at 64-512 tokens; validate
   outputs and q/k/v or LoRA gradients against the reference.
3. **Progressive synthetic GPU:** serialize 1K, 2K, 4K, 8K, and 16K runs. Fail
   on nonfinite values, page faults, ring timeouts, resets, or unexpected timing.
4. **32K safety:** run 24K and 32K forward once, inspect system logs, then run
   backward once. Add warmups/repeats only after a clean single run.
5. **Layer integration:** validate one full-attention layer, one GDN layer, and
   one MLP with exact loss and every active LoRA gradient.
6. **Model integration:** use fresh server processes for increasing-context SFT,
   then GRPO. Compare loss, logprobs, gradient norm, peak allocator bytes, system
   VRAM, wall time, and tokens/s.
7. **Autotuning:** increase one block dimension at a time while keeping every
   underlying custom dispatch below 100 ms. Never tune by returning to a
   monolithic 32K attention or whole-layer dispatch.
8. **Publication gate:** require independent code review, clean driver logs,
   deterministic correctness evidence, and a material measured gain before
   enabling any experimental path by default.

Initial numeric promotion thresholds are:

| Candidate | Required paired result versus current BF16 JAX |
|---|---|
| BF16 fused stage | forward relative L2 at most `5e-3` and cosine at least `0.9999`; every LoRA/input-gradient relative L2 at most `1e-2` and cosine at least `0.999` |
| FP32 GDN state/reduction | relative L2 at most `1e-4` on small recurrence references and at most `1e-3` through the longest tested sequence; no state-dtype demotion |
| W8/A8/O8 stage | output cosine at least `0.995`, relative L2 at most `3e-2`, and no clipping beyond the scheme's recorded dynamic/calibration rule |
| W4/A4 isolated oracle | output cosine at least `0.98` and relative L2 at most `0.20`; this looser unit gate does not relax the full-model gate below |
| Quantized full model | target-logprob MAE at most `0.02` nat, p99 absolute error at most `0.1` nat, held-out NLL degradation at most 1%, LoRA-gradient cosine at least `0.99`, and gradient-norm ratio in `[0.95,1.05]` on the same batches |
| INT8 optimizer moments | first-update direction cosine at least `0.999`, update-norm error at most 1%, all finite through a 100-step CPU replay, then the same short SFT loss gate as BF16 Adam |

Treat these as minimum engineering gates, not evidence that quantization has no
task-quality effect. A short fixed-data SFT replay must preserve loss ordering
and convergence. A fixed-rollout GRPO replay must preserve finite importance
ratios, keep the clipped-token fraction within one percentage point, and keep
KL/reward within measured BF16 seed-to-seed variation before any quantized
route is used for a real run.

For performance promotion, require every underlying dispatch below 100 ms and
clean kernel/driver logs. Enable a path by default only if it improves
end-to-end tokens/s by at least 15% at a target bucket, or reduces measured peak
VRAM by at least 1 GiB without a material throughput regression. A slower path
is memory-first only if it enables a previously impossible bucket. It must not
regress the 64-512-token control buckets by more than 5% unless explicitly
selected in that memory-first mode.

Rollback is per operation: disable the stage's config switch and restart from a
fresh process using the BF16 implementation. Never catch a GPU fault and
continue in the same process. A quantized checkpoint must record scheme,
group/axis, scale dtype, source revision, and a tensor checksum; incompatible or
missing metadata fails closed. Any nonfinite result, threshold miss,
nondeterminism, driver error, dispatch above the watchdog bound, or failure to
meet the material-gain gate keeps the route experimental and off by default.

Moving the monitor to the iGPU limits desktop disruption, but it does not make
an unsafe compute dispatch acceptable. GPU experiments must remain serialized.

## Recommended implementation order

1. Single-adapter LoRA fast path and reusable routing.
2. Watchdog-bounded native-GQA Pallas custom VJP.
3. Split-vocabulary tied linear-logprob Pallas prototype.
4. Gate/up+SwiGLU Pallas prototype and exact custom VJP.
5. Minimal typed HIP/FFI reference handler and CPU fallback.
6. Production BF16 dense epilogues, including input RMS and exact LoRA VJPs.
7. Production GDN preparation/state kernels.
8. Larger loss chunks and final-RMS fusion.
9. W8A16 frozen projections with the tied matrix initially BF16; measure the
   3.2 GiB residency opportunity before changing more numerics.
10. W8 tied embedding/head, then W8A8 projection inputs for native INT8 speed.
11. O8 residual-boundary checkpoints only if measured remat storage remains
    material at the target bucket.
12. W4 group-64 storage/unpack after W8 establishes loader, metadata, and
    quality baselines.
13. Native W4A4 IU4 projection kernels one stage at a time, starting with the
    tied head; retain BF16 outputs and FP32 reductions.
14. INT8 optimizer moments only if the roughly 66 MiB saving still matters.
