# Split-vocabulary tied target-logprob prototype

Status: equation and lowering experiment only. Nothing in this document or in
`skyrl/tx/kernels/tied_logprob.py` is wired into a model, selected on GPU, or a
measured speed kernel.

## Boundary and semantics

Qwen3.5-4B ties its output head to the frozen input embedding. SkyRL currently
computes target logprobs as:

```text
logits[m, v] = hidden[m, :] dot embedding[v, :]
logprob[m] = logits[m, target[m]] - logsumexp_v(logits[m, v])
```

The prototype preserves those equations while splitting work along both axes:

- tokens use a static `M` of 64, 128, or 256;
- vocabulary rows use a configurable superblock `VB`;
- forward retains only one `M x VB` logits tile and `O(M)` normalizer state;
- a clamped final slice overlaps and masks earlier rows when `V % VB != 0`,
  avoiding a second, padded embedding-shaped logical buffer;
- backward recomputes one tile at a time and returns only `dHidden`;
- the tied embedding is explicitly frozen by the custom VJP.

The JAX oracle uses two vocabulary passes in forward: one for the global maximum
and target gather, and one for the exponential denominator. This mirrors the
current dense `jax.nn.logsumexp` precision policy closely, but it rereads the
embedding and is not a performance implementation. A future GPU kernel should
use a numerically stable online `(max, sumexp)` update in one forward pass.
Reduction reassociation means the split forward is numerically close rather
than bit-identical. Its low-precision VJP is deliberately strengthened: it
forms probabilities and accumulates the expected embedding in FP32, then casts
`dHidden` to the model dtype. BF16/FP16 agreement therefore uses explicit
tolerances against autodiff through the current dense equation.

The main API is unmasked and therefore preserves the observable
`compute_logprobs` contract. An optional boolean mask can predicate rows for
experiments. A separate fixed-capacity compaction helper gathers only nonzero
loss-mask rows and scatters zeros back. That is loss/VJP-equivalent after
SkyRL's `safe_loss_mask`, but it changes auxiliary logprobs at inactive tokens,
so it must remain study-only until consumers accept that contract.

The stage does not perform a causal shift. SkyRL's backend already supplies one
target ID for each aligned hidden-state row (for SFT, datum construction shifts
the original token sequence before it reaches `compute_logprobs`). Padding
targets are currently zero-filled and their losses are suppressed later by
`safe_loss_mask`; the unmasked API deliberately computes those auxiliary rows.
The target gather also retains JAX's `take_along_axis` behavior: IDs in
`[-V, V)` are accepted, while farther out-of-range active IDs produce NaN.

## Logical memory and traffic at Qwen3.5-4B geometry

For `V=248,320`, `H=2,560`, BF16, and 32,768 target tokens:

| Item | Logical size/traffic |
|---|---:|
| Frozen tied embedding | 1.1841 GiB |
| Current `M=64` full logits chunk | 30.3125 MiB |
| Full logits at `M=256` | 121.25 MiB |
| Split `M=256, VB=4,096` logits tile | 2.0 MiB |
| One-pass forward reads, `M=64` (future kernel) | 606.25 GiB |
| One-pass forward reads, `M=128` (future kernel) | 303.125 GiB |
| One-pass forward reads, `M=256` (future kernel) | 151.5625 GiB |

The read figures are logical GEMM operand traffic, not measured VRAM traffic.
They explain the main opportunity: a small vocabulary tile permits `M=256`
without a 121 MiB logits allocation, cutting repeated embedding reads by 4x
relative to the current `M=64`. Actual cache behavior, matrix occupancy, and
the fixed arithmetic count limit realized speedup, so this needs isolated GPU
measurement before integration.

The future online forward plus recomputing backward would read the embedding
twice per token chunk. The present JAX oracle reads it three times (two forward,
one backward) and must not be benchmarked as the proposed implementation. To
realize two reads after integration, the fused head must also bypass SkyRL's
existing per-chunk `jax.checkpoint`: its compact normalizer residual replaces
the reason for rematerializing the dense logits path. Leaving that checkpoint
around the custom stage can rematerialize forward and restore a third read.

Quantized frozen embeddings can stack with this boundary:

- W8 storage is about 0.5920 GiB before scale overhead;
- W4 storage is about 0.2960 GiB before scale overhead;
- W8/W4 reduces frozen-weight traffic only if the ROCm kernel consumes packed
  weights directly; dequantizing the whole embedding defeats the purpose;
- quantized target gather and `dHidden` need their own accuracy oracle before
  sharing production code with the BF16 semantic path.

The frozen VJP itself does not eliminate a baseline `dEmbedding` allocation:
SkyRL already differentiates only LoRA state. Its benefit is making that
contract explicit while controlling logits/probability intermediates.

## Future bounded ROCm stage

A production implementation should be a JAX-visible stage backed by bounded
subdispatches, not one whole-sequence megakernel:

1. For each `M` token chunk, stream vocabulary superblocks through an online
   max/sumexp reduction and target-logit selection.
2. Emit one target logprob and the compact normalizer state per token.
3. In the custom VJP, stream superblocks again, reconstruct softmax tiles, and
   accumulate `dHidden` in FP32 before the model-dtype cast.
4. Predicate inactive compact-buffer slots. Only compact real loss-mask tokens
   if the caller explicitly accepts zero auxiliary outputs elsewhere.
5. Put a wall-time bound on each launch and split vocabulary work further if a
   dispatch approaches the gfx1100 watchdog budget.

HIP/CK is the safer first implementation route on this machine. Pallas can be
used in interpret mode to validate tile indexing, but its ROCm execution path
must remain opt-in and independently guarded. The portable JAX loops in the
prototype are the semantic oracle for both routes, not a fallback kernel.

## Verification

CPU-only tests cover all three token chunk sizes, non-divisible token/vocabulary
padding, exact tail-row ownership, negative and out-of-range target gathers,
BF16/FP16 agreement, arbitrary-cotangent `dHidden` (including the normalizer
cotangent for an out-of-range gather), frozen embedding, masking, fixed-capacity
active compaction, and validation failures:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q \
  tests/tx/kernels/test_tied_logprob.py
```

The StableHLO audit lowers an independent dense reference first and confirms it
contains `65 x 97` logits. It then lowers split forward and value-plus-VJP and
confirms both contain `64 x 32` tiles while containing neither `65 x 97` nor
`64 x 97` floating buffers. This is a logical-buffer result only; allocator
peak memory must be measured on the eventual GPU implementation.

A second compile-only audit uses the exact Qwen3.5-4B head geometry. For both
forward and value-plus-`dHidden`, the lowered IR contains a `256 x 4,096` BF16
logits tile and no `256 x 248,320` BF16 or FP32 logits buffer. The backward IR
contains the expected `256 x 2,560` FP32 `dHidden` accumulator. Lowering uses
shape descriptors on the CPU; it neither allocates these model tensors nor
initializes ROCm. It also rejects the `249,856 x 2,560` shape that would reveal
a full embedding padded to 61 vocabulary superblocks.
