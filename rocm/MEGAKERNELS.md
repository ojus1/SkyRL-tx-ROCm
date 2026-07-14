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
assignments. Every throughput percentage or speedup range in this document is
an engineering estimate, not a measured result, unless it is explicitly
labelled as measured.

## Rejected design: one whole-layer or whole-model dispatch

A literal whole-layer or whole-model GPU dispatch is rejected, including a HIP
implementation. Transformer stages need grid-wide synchronization between
projections, attention or recurrence, and subsequent projections. Pallas/Triton
does not provide a safe cooperative grid barrier, while a persistent HIP kernel
would have poor occupancy and run long enough to recreate the GPU-watchdog risk.
One exact BF16 full-attention layer streams about 205 MiB (0.200 GiB) of frozen
weights and one GDN layer about 215 MiB (0.210 GiB); the complete text model is
7.834 GiB. The problem is repeated, globally synchronized layer work, not
"several GiB" in one layer. At 32K, causal attention forward for one of the
eight full-attention layers is about 8.80 TFLOP, or roughly 1.47 s at the
conservative 6 TFLOP/s planning rate, before projections or backward. Moving
the display to the iGPU removes desktop-reset coupling; it does not make a
watchdog-scale single dispatch safe.

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
- Portable full-attention QKV-stage equations and explicit precision-policy
  VJP: `skyrl/tx/kernels/qwen3_5_qkv_lora.py` (unwired experiment only; its
  portable backward increases remat work and is not a memory/speed path).

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
| Native-GQA query chunk, `Cq=512`, `T=32K` | 274.9 GFLOP | 45.8 ms |
| Tied-head token chunk, `M=128` | 162.7 GFLOP | 27.1 ms |
| GDN 1,024-token superblock | about 5-6 GFLOP | below 1 ms compute |

Base transpose GEMMs in backward have the same bound. Gate/up recomputation
and its transpose GEMM must remain separate underlying dispatches. Require an
observed duration below 100 ms for every new custom dispatch; otherwise shrink
its token/query/vocabulary superblock.

## Whole-model projection launch accounting

The source topology has exactly 200 LoRA-bearing dense projections:

- 8 full-attention layers times QKV, O, gate/up, and down gives 32;
- 24 GDN layers times QKV, z, a, b, output, gate/up, and down gives 168.

The current batch-size-one path nominally issues a frozen-base GEMM plus the
two rank-8 LoRA GEMMs for each projection, or 600 dense GEMMs per forward. A
training execution with rematerialized projection work and its VJPs is about
2,200 dense GEMMs. These are operation-level launch counts, not profiler
measurements: XLA may fuse small operations, and a library call may enqueue
more than one device kernel.

The proposed stage map combines the four GDN input branches into one logical
dense group. It therefore has 128 fused groups: 32 in the full-attention layers
and 96 in the GDN layers. For launch budgeting with `M<=2048`, the assumed
target schedule models forward as
`128 * (1 + ceil(T/2048))` underlying launches and a rematerialized training
execution as `128 * (5 + 3*ceil(T/2048))`. The fixed term covers the fused
LoRA/epilogue and VJP work; the sequence-dependent term is the bounded base
projection work. These formulas assume the implementation succeeds in
collapsing that fixed work to the stated launch count; they are a design target,
not an audited enqueue trace.

| Tokens | Current forward | Bounded fused forward | Current training | Bounded fused training |
|---:|---:|---:|---:|---:|
| 64 | 600 | 256 | about 2,200 | 1,024 |
| 512 | 600 | 256 | about 2,200 | 1,024 |
| 8,192 | 600 | 640 | about 2,200 | 2,176 |
| 32,768 | 600 | 2,176 | about 2,200 | 6,784 |

This table counts dense projection work only. It excludes the GDN core, full
attention, tied head, norms, convolutions, and reductions. At long context the
bounded design deliberately trades more launches for watchdog safety; its gain
must come from eliminating intermediates, reducing weight/activation traffic,
and using better epilogues rather than from a lower launch count.

The following end-to-end BF16 throughput uplifts are estimates to test, not
measurements or commitments:

| Bucket | Estimated BF16 fusion uplift |
|---:|---:|
| 64 | 5-15% |
| 512 | 3-10% |
| 8,192 | 5-20% |
| 32,768 | Enablement-first; no defensible speed range before the GDN and tied-head paths are measured |

A fully optimized native W4 long-context path has an estimated realistic
throughput range of 1.2-1.5x versus BF16. That estimate assumes compact weights
remain canonical, IU4 work is useful rather than conversion-bound, and the
bounded stage overhead is controlled. The exact training backward specified
below has no IU4 matrix-rate advantage, so training may fall below that range.

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

Direct A/B selection and ordinary GEMMs for the common batch-size-one path are
already implemented by `LoRAMixin._apply_single_adapter_lora`; this is the
baseline, not remaining megakernel work. Keep the sorted ragged-dot path for
true mixed-adapter batches. If larger mixed batches become important, prepare
their token permutation and group sizes once per model pass rather than once
per LoRA layer.

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

The reviewed prototype starts with `Cq=512`, giving 64 query ranges at 32K,
but this is only a compile/resource hypothesis. Forward, dQ, and dKV are
separate bounded dispatches. Reduce `Cq` from 512 to 256 and then 128 if any
dispatch approaches 100 ms, spills, or exceeds gfx1100 resources; abandon this
Pallas route if 128 still fails. Keep online-softmax statistics and dK/dV
accumulation in FP32. Map query head `hq` to KV head `hq // 4`. Accumulate
dK/dV through aliased FP32 buffers in a fixed query-chunk order; do not use
atomics.

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

The first exact-shape BF16 prototype is now a measured no-go. Its two Pallas
launches use `BM16/BN32/BK64`; the forward hides the normalized activation and
raw 18,432-wide projection, but its correctness-first custom VJP recomputes the
complete dense JAX forward before applying library-dot pullbacks. A guarded
four-sample smoke measurement at `B1/T64` found only `0.695x` forward,
`0.597x` forward-plus-VJP, and `0.633x` rematerialized-stage speed relative to
the BF16 JAX boundary. The smoke rung is not a qualifying benchmark, but the
miss is large enough that the unchanged 160-dispatch confirmation run is not
justified. This exact implementation remains default-off and must not be wired
into the decoder.

The next candidate must remove the dense-forward recomputation from the custom
VJP and own the projection/backward boundary. Larger Pallas N tiles are useful
only as a bounded diagnostic after that structural fix; the production route
remains a HIP/typed-FFI dense epilogue with an exact custom VJP. The existing
smoke telemetry is dominated by JAX BFC preallocation and therefore establishes
no allocator or model-capacity saving.

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
   only U, W, and minimal gamma data for one superblock.
2. State/output execution for 8-16 chunks per dispatch, consuming that
   superblock's U/W immediately and carrying the 2 MiB FP32 state through HBM
   at each 512-1,024-token boundary.
3. Reverse-superblock custom VJP using the exact recurrence above.

Saving FP32 state every 1,024 tokens costs about 34 MiB at 16K and 66 MiB at
32K, instead of saving one state for every 64-token chunk.

The launch-count reason to replace the current scan is larger than the dense
projection count. The existing chunk equations contain about five
matrix-producing operations per 64-token chunk per GDN layer. Across 24 GDN
layers their nominal forward count is therefore
`5 * ceil(T/64) * 24`. A 1,024-token schedule with one preparation dispatch and
one state/output dispatch per superblock instead uses
`2 * ceil(T/1024) * 24` forward dispatches:

| Tokens | Current GDN scan matrix launches | Two-dispatch superblocks |
|---:|---:|---:|
| 64 | about 120 | 48 |
| 512 | about 960 | 48 |
| 8,192 | about 15,360 | 384 |
| 32,768 | about 61,440 | 1,536 |

These are launch-accounting estimates, not traced device-kernel counts, and
they exclude projection, normalization, convolution, and backward work. The
backward should recompute preparation for one superblock and then run one
reverse dispatch; retaining full-sequence U/W merely to reduce recomputation is
not the intended design.

U and W are both FP32 `[B,T,32,128]` tensors. Materializing them for a whole
active layer versus streaming at most one 1,024-token superblock gives:

| Tokens | Full-sequence U+W | Streamed U+W scratch |
|---:|---:|---:|
| 64 | 2 MiB | 2 MiB |
| 512 | 16 MiB | 16 MiB |
| 8,192 | 256 MiB | at most 32 MiB |
| 32,768 | 1,024 MiB | at most 32 MiB |

The scratch is per active layer and is reused across superblocks. The 992 MiB
32K logical difference must not be multiplied by 24 or added blindly to every
other opportunity: per-layer rematerialization, aliasing, and allocator reuse
determine the realized peak.

### 9. GDN gated norm, output projection, and residual

The GDN output projection should load core output and z, calculate per-head FP32
RMS statistics plus SiLU gating, and perform the 4,096-to-2,560 base+LoRA
projection with residual addition. Do not write a separate gated-normalized
4,096-wide tensor.

### 10. Final RMS and tied linear logprob

The launcher currently sets `loss_chunk_size=64`, which already bounds live
logits and does not materialize full `[T,V]` logits. The current logical
per-chunk logits are 30.3125 MiB in BF16 or 60.625 MiB in FP32. The default-off
split target-logprob custom VJP is now wired for explicit `M=64/128/256` plus a
vocabulary superblock and eliminates this full-vocabulary per-chunk tensor, not
a full-sequence tensor. Its online forward forms each vocabulary tile once.
`M=128` remains the first GPU qualification bucket; it is not the launcher
default. Dense BF16/FP32 logits at that size would be 60.625/121.25 MiB.

The tied embedding is approximately 1.184 GiB, larger than cache. With token
chunk 64 it is reread once per token chunk in forward. Training nominally uses
three embedding dot operands per chunk. In the online custom path these are one
forward logits dot plus the backward logits-reconstruction and expected-
embedding dots; the former two-pass prototype had four. The table below is
estimated logical traffic, not measured HBM traffic; it applies `ceil(T/M)`
and includes group-64 scale bytes for W8/W4:

| Tied-head route | 64 tokens | 512 tokens | 8,192 tokens | 32,768 tokens |
|---|---:|---:|---:|---:|
| BF16, `M=64` | 3.55 GiB | 28.42 GiB | 454.69 GiB | 1,818.75 GiB |
| BF16, `M=256` | 3.55 GiB | 7.10 GiB | 113.67 GiB | 454.69 GiB |
| W8 group-64, `M=256` | 1.83 GiB | 3.66 GiB | 58.61 GiB | 234.45 GiB |
| W4 group-64, `M=256` | 0.94 GiB | 1.89 GiB | 30.19 GiB | 120.78 GiB |

Fusion does not itself remove vocabulary reads; it permits a larger token
chunk without a larger logits buffer. `M=128` remains the first fused-head
bucket. After it is validated, test `M=256`: its work is 325.5 GFLOP and its
duration is estimated at 54 ms at the conservative 6 TFLOP/s planning rate.
The portable online value-plus-VJP lowering has three dot bodies; a production
fused backward that reuses each live embedding tile for both backward products
would reduce the BF16 `M=256`, 32K target from 454.69 to 303.125 GiB. The
W8/W4 rows are bandwidth opportunities, not end-to-end speedup claims.

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

Full-sequence GDN preparation creates 512 MiB/1 GiB of U/W at 16K/32K. The
streamed superblock design above caps its logical U/W scratch at 32 MiB by
recomputing in backward. Fusing preparation and state stages can also avoid
roughly 1.6/3.3 GiB of other logical FP32 intermediates at 16K/32K. None of
these logical figures are additive peak claims: the actual delta can be much
smaller when XLA aliases or reuses the same storage.

### O8 layer-boundary accounting

O8 means that each completed decoder-layer residual is quantized once, stored
as signed INT8 codes plus a per-token or small-block scale, and dequantized to
the model dtype before the next layer consumes it. The custom VJP treats the
round/clip operation with the declared straight-through rule and returns the
incoming cotangent to the pre-quantized boundary; codes and stopped-gradient
scales are not trainable. It is not permission to quantize an in-progress
residual sum, attention/GDN state, or the GDN U/W workspace.

To realize any checkpoint saving while preserving the STE gradient, wrap the
whole next-layer/recompute boundary in one `custom_vjp` that takes the BF16
layer input. Its forward quantizes and dequantizes that input, evaluates the
next layer from the dequantized value, and stores only `(codes, scales)` in
place of the BF16 input among its activation residuals. Ordinary layer
operands and metadata remain available to compute their required cotangents.
Its backward dequantizes the compact residual, recomputes the layer, applies
the layer VJP, and returns the resulting cotangent to the original BF16 input
under the declared STE rule.

Do not expose integer codes as the output of one independently differentiated
quantizer and the input to a separate checkpointed dequantizer/layer. JAX
integer arrays carry `float0` cotangents, so that arrangement cannot bridge
autodiff back to the BF16 input. A quantize/dequantize identity outside the
custom-VJP recompute boundary followed by rematerializing a layer whose
argument is BF16 also makes XLA retain the BF16 layer input and realizes no O8
boundary saving.

Ignoring the small scale metadata, replacing retained BF16 values across all
32 `[1,T,2560]` decoder boundaries has this logical upper bound:

| Tokens | BF16-to-O8 saving across 32 boundaries |
|---:|---:|
| 64 | about 5 MiB |
| 512 | about 40 MiB |
| 8,192 | about 640 MiB |
| 32,768 | about 2,560 MiB |

These figures assume all 32 boundaries would otherwise contribute to the live
training set. Rematerialization may retain fewer, scales reduce the saving
slightly, and allocator reuse may make the measured peak delta much smaller.
The 2.5 GiB 32K opportunity must not be added to every hidden-state row in the
table above.

## Host-offload floor

Host offload is an enablement fallback, not a speed optimization. At 32K,
moving a 5 GiB retained set to host and later bringing it back transfers 5 GiB
in each direction. Even an optimistic effective 20-25 GiB/s link implies an
estimated 0.4-0.5 seconds of aggregate transfer time per offload/reload event,
before synchronization, allocation, or staging overhead. Pinned memory and
overlap can hide some wall time but cannot remove this byte floor. Offloading
many layer boundaries independently is therefore not viable; first reduce or
stream the GDN/tied-head working sets and quantify the remaining retained set.

## Quantization routes for `gfx1100`

### Hardware constraints

The official ROCm precision table lists INT8, FP16, and BF16 matrix-core support
for RDNA3, including the RX 7900 XTX class, but no native FP8 or floating-point
FP4 matrix-core support. Integer four-bit is a separate case: the official
RDNA3 ISA lists `V_WMMA_I32_16X16X16_IU4`, and AMD's RX 7900 XTX WMMA guide
documents both signed/unsigned four-bit inputs with INT32 accumulation. A local
compile-only audit proved that Clang emits the gfx1100 IU8 and IU4 instructions
when their builtins are called directly. It did not enumerate the GPU or launch
a kernel. The installed rocWMMA wraps IU8 but has no matching public IU4
wrapper. The installed Composable Kernel source names an experimental IU4
selector but lacks the corresponding `wmma_type`/intrinsic implementation, so
it is not a ready IU4 path. hipBLASLt advertises INT8 types at the library API,
but every exact transpose/layout/scale/epilogue combination still needs an
algorithm-support query or a custom HIP kernel and benchmark.

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

The installed-stack evidence and exact fragment ABI are recorded in
[`QUANTIZED_FFI.md`](QUANTIZED_FFI.md). Its compile-only proof is
[`compile_quant_wmma_gfx1100.sh`](compile_probes/compile_quant_wmma_gfx1100.sh).
Compiler/ISA feasibility is proven; layout correctness, quality, occupancy,
and speed are not.

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
The tied matrix is then the highest-leverage bandwidth experiment; the
three-pass training traffic estimates for BF16/W8/W4 and `M=64/256` are in the
final-head section above. Those are logical traffic bounds, not end-to-end
speedup predictions. Because the matrix is tied, quantizing it changes both
token lookup and the vocabulary head; duplicating a BF16 embedding just for
lookup would reduce the memory saving and must be reported explicitly.

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
   rescale each group in FP32, and emit BF16. The installed implementation route
   is a custom HIP kernel using the compile-proven Clang builtin; neither the
   installed rocWMMA nor CK supplies a complete IU4 wrapper. This is the only
   ranked route with a higher published matrix-instruction ceiling than BF16 on
   RX 7900 XTX, but it is also the highest-risk projection path numerically.
   Validate tied head, MLP, attention projections, and backward
   straight-through behavior independently; do not turn on model-wide A4 in
   its first experiment. Start base `dX` from the dequantized W4
   straight-through reference; a native IU4 transpose path also quantizes `dY`
   to A4 and is a later, separate gate.
5. **O8 layer-boundary checkpoints.** Quantize only after the residual addition
   and store signed codes plus a per-token or small-block scale, then dequantize
   before the next layer. Its explicit straight-through VJP and the upper-bound
   5/40/640/2,560 MiB savings across all 32 boundaries are specified above. Do
   not store attention softmax data, Q/K normalization statistics, GDN U/W, or
   recurrent state in O8. The benefit must be measured with rematerialization
   enabled because XLA may not retain every boundary.
6. **Blockwise INT8 Adam moments.** The implemented semantic prototype is
   negative evidence for the current BF16 training path. For the exact
   two-adapter, rank-8 inventory, its byte payloads plus block-16 FP32
   scales/offsets use 90.51 MiB and save only 41.14 MiB versus BF16 moments.
   Its fixed 100-step BF16 comparison reached 5.239612% worst relative
   update-norm error and failed the 1% gate. Keep it unwired; a materially
   different encoding would be a new experiment. This remains lower priority
   than compacting inactive adapter slots or quantizing the frozen base.
7. **O4 boundaries and four-bit optimizer state.** Defer. Integer-four-bit
   matrix support does not make four-bit residual checkpoints or Adam moments
   numerically safe, and those encodings risk corrupting residual, norm,
   softmax, recurrence, and optimizer dynamics for modest additional memory
   beyond the ranked routes. User acceptance makes them valid experiments, not
   valid defaults without the same model-level gates.

### Adapter-slot and optimizer-state arithmetic

The 34,512,896-element figure includes two equal adapter slots. If only one is
active, removing the inactive half saves about 131.7 MiB across its BF16
parameters, BF16 gradients, and two BF16 Adam moments. This is a pure capacity
win when checkpoint/serving semantics do not require the spare slot, and is
larger than quantizing all moments from BF16 to INT8.

| State change | Logical saving before scale metadata |
|---|---:|
| Remove one of two inactive adapter slots: params + grads + two moments | 131.7 MiB |
| Quantize both adapters' two BF16 Adam moments to INT8 | 65.8 MiB |
| Quantize both adapters' two BF16 Adam moments to packed W4 | 98.7 MiB |

The W4 moment number is capacity arithmetic, not a recommendation or speed
estimate. Both quantized optimizer routes still require FP32 dequantization,
update, bias correction, and model-level convergence gates; scales make their
real savings slightly smaller.

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
| W4A4 base GEMM | Portable oracle only | Custom HIP using the compile-proven native IU4 builtin; installed rocWMMA/CK are incomplete for IU4 |
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

| Candidate | Required paired result and reference boundary |
|---|---|
| Other BF16 fused stage | forward relative L2 at most `5e-3` and cosine at least `0.9999`; every LoRA/input-gradient relative L2 at most `1e-2` and cosine at least `0.999` versus current BF16 JAX |
| Pallas attention versus FP32 oracle | forward-output relative L2 strictly below 1% (`<1%`, `1e-2`); each dQ/dK/dV relative L2 strictly below 3% (`<3%`, `3e-2`); established FP32-delta implementation retains its gradient regression gate of strictly below 1% (`<1%`, `1e-2`) |
| FP32 GDN state/reduction | relative L2 at most `1e-4` on small recurrence references and at most `1e-3` through the longest tested sequence; no state-dtype demotion |
| W8 implementation fidelity | custom kernel versus portable grouped-W8 oracle: output and every defined VJP relative L2 strictly below 1% (`<1%`, `1e-2`); forward cosine at least `0.9999` and forward maximum absolute error at most `0.25`; this excludes inherent quantization error |
| W8/A8 quantization quality | portable quantized oracle versus BF16: output cosine at least `0.995`, relative L2 strictly below 3% (`<3%`, `3e-2`), and no clipping beyond the scheme's recorded dynamic/calibration rule; full-model gates still apply; O8 requires a separate oracle and qualification gate |
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

1. Preserve the completed single-adapter LoRA fast path; add reusable routing
   only if true mixed-adapter batches become important.
2. Complete and qualify the watchdog-bounded native-GQA Pallas custom VJP.
3. Split-vocabulary tied linear-logprob Pallas prototype.
4. Replace the measured-no-go gate/up+SwiGLU Pallas VJP with an exact backward
   that does not recompute the dense forward; use larger Pallas N tiles only as
   bounded diagnostics.
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
