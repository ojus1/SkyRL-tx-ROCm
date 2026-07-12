# EGGROLL optimizer and FlashQLA feasibility on RX 7900 XTX

## Decision

Do **not** replace SkyRL-tx's gradient/Adam path exclusively with EGGROLL for
Qwen3.5-4B on this machine. EGGROLL is a promising separate training
algorithm, but it is not an `optim_step` substitute: it replaces
forward/backward and gradient accumulation with a population of perturbed
fitness evaluations. The released implementation is an early JAX/RWKV research
library, not an arbitrary-model wrapper. Its demonstrated near-inference
throughput uses hundreds to 1,024 simultaneous population members; that regime
is not available beside long-context Qwen on this 24 GiB GPU.

Do use FlashQLA as an algorithm and operation-boundary reference for the Gated
DeltaNet port. Do **not** try to install or call its kernels on ROCm. The
released package dispatches CUDA/TileLang implementations for NVIDIA SM90,
SM100, and a partial SM120 path. The corresponding AMD work is the bounded GDN
prepare/execute/reverse design in [`GDN_SUPERBLOCK.md`](GDN_SUPERBLOCK.md), not
a source-level backend switch.

Likewise, do not build one literal whole-model or whole-layer forward/backward
kernel. FlashQLA itself deliberately uses several fused kernels rather than one
kernel for the complete flow. On gfx1100, bounded stages with explicit saved
state and custom VJPs are the feasible definition of a "megakernel."

## Audited revisions and scope

- Qwen model: `Qwen/Qwen3.5-4B` revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- EGGROLL code: `ESHyperscale/HyperscaleES` revision
  `b77f7d6f91238fd575313e946b9cad21e0a74b32`.
- FlashQLA code: `QwenLM/FlashQLA` revision
  `40b7527f6c6e2ed8ed65350103e3ca64174f53f3` (`v0.1.2-2-g40b7527`; the package
  and README identify the July 2026 v0.1.2 release line).
- Local hardware/software: one RX 7900 XTX (`gfx1100`, 24 GiB), ROCm 7.2,
  JAX/JAXlib 0.10.2, command buffers disabled.

No EGGROLL or FlashQLA GPU program was executed. FlashQLA was inspected as
source only. The EGGROLL assessment is interface and arithmetic analysis, not
a Qwen quality benchmark.

Primary upstream references are the
[EGGROLL project](https://eshyperscale.github.io/),
[EGGROLL paper](https://arxiv.org/abs/2511.16652),
[HyperscaleES research code](https://github.com/ESHyperscale/HyperscaleES), and
[FlashQLA repository](https://github.com/QwenLM/FlashQLA).

## Why EGGROLL is not a drop-in optimizer

SkyRL's current learner contract is:

```text
forward -> token logprobs/loss -> VJP of LoRA state -> accumulated gradients
        -> Adam update of one adapter
```

EGGROLL's contract is instead:

```text
generate rank-r perturbation for every evolved matrix and population member
        -> run perturbed forward/rollout -> reduce scalar fitnesses
        -> reconstruct a dense matrix-shaped update estimate -> apply an Optax solver
```

Consequences:

1. It cannot consume SkyRL's already accumulated gradients. Selecting it in
   `optim_step` would leave the expensive backward pass in place and would not
   implement EGGROLL.
2. SFT must become population fitness optimization of NLL. GRPO must become a
   direct reward/fitness algorithm rather than the current clipped
   importance-ratio learner. This changes the algorithm and quality metrics,
   not just the optimizer.
3. HyperscaleES models explicitly route every evolved matrix multiplication
   through `Noiser.do_mm`/`do_Tmm` and carry a per-parameter PRNG tree and
   evolution classification. SkyRL's NNX Qwen implementation exposes none of
   that interface.
4. The released LLM model code is RWKV6/RWKV7-specific; the active pretrained
   registry entries are RWKV7, while the quantized-RWKV6 entries are commented
   out. The EGGROLL noiser's embedding perturbation is `NotImplemented`, and
   there is no Qwen/Transformer model or SkyRL backend adapter.
5. The code is labelled a research preview and is GPL-3.0-only, while SkyRL is
   Apache-2.0. Apache-2.0 code can be combined into a GPLv3 work, but copying
   HyperscaleES code into SkyRL while continuing to distribute the combined
   work as Apache-2.0-only is not an available route. Retaining SkyRL's current
   license calls for an independently written implementation of the published
   algorithm, subject to normal legal review.

The project page also explains why its reasoning experiments use RWKV:
constant recurrent state permits large parallel populations, whereas a
Transformer's growing KV cache bottlenecks the number of simultaneous
generations. That limitation is directly relevant to Qwen3.5.

## EGGROLL memory opportunity and cost

The exact two-slot rank-8 SkyRL LoRA tree contains 34,512,896 BF16 elements.
Its explicit train state is small relative to the frozen model:

| State removed by a stateless ES learner | Logical saving |
|---|---:|
| Two BF16 Adam moments | 131.7 MiB |
| Accumulated BF16 LoRA gradients | 65.8 MiB |
| Both together | 197.5 MiB |

Those are upper bounds before any EGGROLL state. HyperscaleES defaults to
Optax SGD, whose state is negligible, but it can also accept a stateful Optax
solver. Replacing Adam therefore saves only about 132 MiB; this is nowhere near
the multi-GiB capacity problem at context 2,048.

Backprop-free training could save substantially more activation and reverse
workspace memory. That is EGGROLL's real capacity opportunity. It is offset by
three costs on this machine:

- population activations, recurrent/KV state, and outputs scale with the
  number of simultaneous perturbations;
- sequentially microbatching a population preserves memory but multiplies
  forward/rollout time; and
- the released update constructs a full matrix-shaped update for every evolved
  matrix. Evolving all Qwen base matrices can therefore introduce a
  model-sized update tree unless donation and buffer assignment prove an
  in-place schedule.

Applying ES only to the existing LoRA A/B arrays avoids a full base-weight
update, but it also gives up EGGROLL's main full-parameter claim. The upstream
project distinguishes the dense, potentially full-rank update formed by
summing low-rank perturbations from directly optimizing a fixed LoRA adapter,
whose resulting base-model change remains rank-limited. For an `N`-member,
rank-`r` population, the EGGROLL matrix update has rank at most
`min(N*r, m, n)`; full rank therefore requires `N*r >= min(m, n)`.

For this exact backend, one stored rank-8 adapter slot is about 32.9 MiB. A
naive adapter-per-population implementation would consume roughly 33 GiB for a
population of 1,024 before activations. EGGROLL avoids those stored copies by
regenerating rank-r factors. SkyRL would need a new population-aware model path
to realize that property, and likely fused linear kernels to make that path
competitive; a custom kernel is not a prerequisite merely to regenerate noise
from seeds.

## Quantization does not reverse the optimizer decision

User acceptance of 4/8-bit model, activation, and optimizer experiments creates
real capacity options, but they are not EGGROLL-specific. From the exact local
safetensor inventory, canonical group-64 frozen-weight storage is estimated as:

| Frozen base route | Storage | Saving from BF16 |
|---|---:|---:|
| BF16 | 7.8338 GiB | - |
| W8, tied embedding/head kept BF16 | 4.6137 GiB | 3.2201 GiB |
| W8, including tied embedding/head | 4.0402 GiB | 3.7936 GiB |
| W4, tied embedding/head kept BF16 | 2.9517 GiB | 4.8821 GiB |
| W4, including tied embedding/head | 2.0822 GiB | 5.7516 GiB |

Those are logical packed-storage estimates, not measured allocator peaks. They
require packed weights to be the canonical device representation and a kernel
that dequantizes tiles rather than retaining a BF16 shadow. The same W8/W4
forward kernels benefit the existing Adam learner, so these savings do not
argue for changing the learning algorithm. They could make a small ES
population easier to fit, but not the hundreds-member, long-context regime.

Likewise, quantizing both BF16 Adam moments to INT8 or packed four-bit saves
only about 65.8 MiB or 98.7 MiB before scale metadata; stateless SGD removes
the roughly 131.7 MiB moments entirely. A8/A4 population activations could
lower forward residency, but there is no quantized Qwen EGGROLL path or
quality/speed evidence for it. The upstream pure-int8 result uses a bespoke
recurrent EGG model and cannot be treated as evidence for quantized
Qwen3.5. The implementation and numerical gates for the shared quantized
projection path are in [`MEGAKERNELS.md`](MEGAKERNELS.md).

## EGGROLL speed evidence and expected result here

The upstream headline is a gain over **naive ES**, not a measured gain over
SkyRL's Qwen LoRA learner. The paper's controlled throughput figure uses one
GH200 (described as equivalent to one H100), a BF16 linear model of dimension
8,192, and a maximum batch/population of 1,024. Normalized to pure inference at
100, it reports EGGROLL at 91 with pre-generated noise and 69 while regenerating
noise, versus a PPO reference at 34. That is about 2.0x over that PPO reference
with online noise, under its high-population NVIDIA test conditions. The
project also reports roughly 100x over unstructured naive ES. The paper's
roofline analysis estimates that this 8,192-wide, rank-1 case needs a batch of
352 to reach 300 operations per byte, so the headline result is explicitly a
large-population result rather than a small-population forecast.

None of those ratios transfers to Qwen3.5 on gfx1100:

- the RX 7900 XTX has no room for a 1,024-member long-context Transformer
  population;
- Qwen's full attention and GDN state do much more work than the benchmarked
  isolated linear layer;
- ROCm/gfx1100 kernel efficiency for the additional batched low-rank branches
  is unmeasured; and
- SFT convergence per fitness evaluation and GRPO reward sample efficiency are
  different from backpropagation and must be included in time-to-quality.

The defensible expected outcome is therefore:

| Route | Memory expectation | Speed expectation on this machine |
|---|---|---|
| Use stateless SGD instead of Adam (not EGGROLL) | Save about 132 MiB | The isolated Adam replay is about 70 ms; any saving is bounded and changes optimizer behavior |
| LoRA-only EGGROLL, small population | Lower reverse memory | Likely slower per useful update due repeated forwards |
| Full-parameter EGGROLL | Potentially inference-like activation memory | No Qwen integration or fit evidence; model-sized update and population risks |
| Upstream large-population regime | Strong published NVIDIA result | Not reachable with 24 GiB and Qwen KV/state geometry |

There is no evidence of a big enough end-to-end gain to justify replacing the
working optimizer. Keep EGGROLL as an optional research algorithm, not the
default or exclusive learner.

## If an EGGROLL experiment is still desired

The smallest honest gate is a clean-room, LoRA-only experiment after the
ordinary learner is stable at longer contexts:

1. Use fixed context 64 and fixed SFT data; do not mix the experiment with
   sampling or 32K kernel work.
2. Use antithetic populations 2, 4, then 8, regenerated from seeds. Never store
   one adapter per population member.
3. Perturb only the active LoRA A/B state and use scalar NLL as fitness. Keep
   the frozen base identical and command buffers disabled.
4. Compare wall time, peak VRAM, NLL reduction per token, and update variance
   against the existing Adam control over enough updates to measure learning,
   not one execution.
5. Stop if it fails finiteness/reproducibility, exceeds the Adam peak, or does
   not reach equivalent loss improvement within 1.2x of Adam wall time. Only
   then consider a direct-reward experiment.

That experiment answers whether small-population ES has local value. It does
not validate the upstream hyperscale performance claim.

## FlashQLA portability result

FlashQLA's README summarizes 2-3x forward and about 2x backward speed over its
FLA Triton comparison across multiple NVIDIA Hopper/Blackwell scenarios. The
checked-in CUDA-graph tables are more shape-dependent. For the exact
`35B/9B/4B TP1`, `h_qk=16`, `h_v=32`, one-sequence geometry, the 32K forward
speedup over FLA is 2.34x on H200, 2.39x on GB200, and 1.66x on RTX 5090. At 2K
the same rows report 1.30x, 1.59x, and 0.81x respectively. The backward tables
do not include this exact `16/32` head geometry. These are GDN operation
timings with 10 warmups and 100 CUDA-graph replays, not end-to-end Qwen
training ratios and not an AMD forecast.

The source cannot execute on this AMD stack:

- the documented runtime requires CUDA 12.8+, PyTorch 2.8+, and NVIDIA
  SM90/SM100, with v0.1.2 adding a forward-only SM120 path;
- dispatch calls TileLang's `nvcc` compute-version query and selects only
  `9.0`, `10.0`, or a partial `12.0` implementation;
- runtime utilities use `torch.cuda`, CUDA devices/events, NVIDIA
  multiprocessor properties, warpgroup specialization, and NVIDIA-specific
  layouts; and
- there is no HIP/gfx1100 implementation or JAX FFI boundary.

Attempting to install it would add a CUDA/PyTorch kernel stack without creating
an AMD path. Source inspection is sufficient to reject direct use. FlashQLA
itself is MIT-licensed, so a port can reuse source subject to its notice; the
blocker is the NVIDIA-specific implementation, not a copyleft conflict.

Its architecture is nevertheless the right reference. FlashQLA explicitly
rejects both a decomposition into many independent elementary operations and a
single kernel for the entire computation. It fuses several high-value GDN
stages, carries bounded state, and supplies explicit forward/backward kernels
on SM90/SM100 (SM120 is forward-only). That maps to the local three-stage plan:

1. bounded WY/KKT preparation;
2. bounded state/output execution with native Q-head to V-head grouping; and
3. explicit reverse-superblock recomputation/custom VJP.

The CPU semantic and compiler evidence for that AMD-specific boundary is in
[`GDN_SUPERBLOCK.md`](GDN_SUPERBLOCK.md). At 32K it removes a full-sequence
temporary-growth term in the CPU compiler estimate; at 512/1,024 its portable
automatic VJP is worse, so the oracle must not be wired as a production
fallback.

## Feasible megakernel boundaries

The production unit should be a JAX-visible stage that may enqueue several
bounded HIP/Pallas subdispatches. The ranked boundaries are:

1. split-vocabulary tied target-logprob forward/custom VJP for immediate head
   bandwidth and logits memory;
2. query-bounded native-GQA forward/backward, required before any new 32K
   attention execution;
3. GDN prepare/execute/reverse superblocks, the largest long-context temporary
   reduction and the FlashQLA adaptation target;
4. W8/W4 frozen projection plus BF16 LoRA/residual epilogues; and
5. optional fused QKV/RMSNorm/LoRA stages after their custom VJP avoids
   full-size FP32 residuals.

A literal whole layer/model kernel is rejected because it would keep too many
weights, activations, recurrent states, and cotangents live for gfx1100
register/LDS limits; it would also create watchdog-scale dispatches, prevent
bounded checkpoint/recompute scheduling, and make XLA buffer donation opaque.
The operation-specific prototypes are documented in
[`TIED_LOGPROB_PROTOTYPE.md`](TIED_LOGPROB_PROTOTYPE.md),
[`QUERY_BOUNDED_GQA.md`](QUERY_BOUNDED_GQA.md), and
[`GDN_SUPERBLOCK.md`](GDN_SUPERBLOCK.md).
