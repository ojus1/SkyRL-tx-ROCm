# EGGROLL optimizer and FlashQLA feasibility on RX 7900 XTX

## Decision

Do **not** replace SkyRL-tx's gradient/Adam path exclusively with EGGROLL for
Qwen3.5-4B on this machine. EGGROLL is a promising separate training
algorithm, but it is not an `optim_step` substitute: it replaces
forward/backward and gradient accumulation with a population of perturbed
fitness evaluations. Upstream now has both the original JAX/RWKV research
library and an official PyTorch/vLLM Qwen runtime. Neither is an arbitrary-model
wrapper or a JAX/SkyRL optimizer.

The current 24 June 2026 OpenReview revision materially strengthens the Qwen
case without reversing that decision. On one GH200 for nine hours, the authors
match hardware and effective wall clock against GRPO for Qwen3-4B, rank-1 LoRA
perturbations, population 256, and maximum response length 8,192. EGGROLL
reports a five-benchmark average of 49.8 versus 47.5 for GRPO on DeepScaleR and
48.4 versus 44.2 on ORZ. This is credible same-budget quality evidence.
Population 4,096 and 16,384 results retain 256 members per GPU and scale the
GPU count, so they are larger-compute cluster results. None demonstrates
Qwen3.5, ROCm, 32K responses, or a large population on one 24 GiB card. Keep
Adam as the production SFT/GRPO learner and prototype EGGROLL only as an
optional, separately gated ES-RLVR backend.

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
  [`b77f7d6f91238fd575313e946b9cad21e0a74b32`](https://github.com/ESHyperscale/HyperscaleES/tree/b77f7d6f91238fd575313e946b9cad21e0a74b32).
- EGGROLL Transformer runtime: `ESHyperscale/eggroll-vllm` revision
  [`bcc215e8784f5f44d24985145c0a71e74283cf1f`](https://github.com/ESHyperscale/eggroll-vllm/tree/bcc215e8784f5f44d24985145c0a71e74283cf1f).
- EGGROLL paper: the authoritative [OpenReview revision](https://openreview.net/forum?id=bfVJ4GsHrO),
  modified 2026-06-24, and its [current PDF](https://openreview.net/pdf?id=bfVJ4GsHrO).
  [arXiv v2](https://arxiv.org/abs/2511.16652), revised 2026-02-16, is retained
  only as a historical revision.
- FlashQLA code: `QwenLM/FlashQLA` revision
  [`e9c5836c4c1f3fa92532e1e568eacacabf45542d`](https://github.com/QwenLM/FlashQLA/tree/e9c5836c4c1f3fa92532e1e568eacacabf45542d)
  (`v0.1.2-4-ge9c5836`; the package and README identify the July 2026 v0.1.2
  release line).
- Local hardware/software: one RX 7900 XTX (`gfx1100`, 24 GiB), ROCm 7.2.4,
  AMDGPU 6.16.13, JAX/JAXlib 0.10.2, command buffers disabled.

No EGGROLL or FlashQLA GPU program was executed. FlashQLA was inspected as
source only. No local Qwen EGGROLL quality benchmark was run; cited Qwen
results are upstream.

Primary upstream references are the
[EGGROLL project](https://eshyperscale.github.io/), the pinned paper and both
codebases above, and the
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
4. The JAX library's released LLM code remains RWKV6/RWKV7-specific. Its noiser
   has no embedding implementation and the supplied model must route evolved
   multiplications through `Noiser`. The separate
   [eggroll-vLLM runtime](https://github.com/ESHyperscale/eggroll-vllm/blob/bcc215e8784f5f44d24985145c0a71e74283cf1f/README.md)
   supports Qwen-style `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
   `up_proj`, and `down_proj`. It has no Qwen3.5 GDN mapping for `in_proj_qkv`,
   `in_proj_z`, `in_proj_a`, `in_proj_b`, or `out_proj`, and it rejects unknown
   mappings.
5. HyperscaleES is labelled a research preview, and both implementations are
   GPL-3.0-only with no tagged release, while SkyRL is Apache-2.0. Apache-2.0
   code can be combined into a GPLv3 work, but copying either implementation
   into SkyRL while continuing to distribute the combined work as
   Apache-2.0-only is not an available route. Retaining SkyRL's current license
   calls for an independently written implementation of the published
   algorithm, subject to normal legal review.

The JAX experiments use RWKV because constant recurrent state permits large
parallel populations, whereas a Transformer's growing KV cache limits
simultaneous generations. The official Transformer runtime instead uses
PyTorch, vLLM 0.17, Ray, PEFT, CUDA, NCCL/PyNccl, multi-LoRA serving, tensor
parallelism within a node, and population parallelism across nodes. It streams
one dense FP32 layer update, constructed from rank-r population perturbations,
at a time, casts it to model dtype, applies it to mutable base-model weights,
and saves a full-model checkpoint; population LoRAs are transient. That is
useful systems evidence, but it is a separate CUDA trainer rather than a local
cache solution or a JAX/SkyRL backend.

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
solver. Against the accepted alternatives of BF16 gradients plus INT8 or
packed four-bit moments, the corresponding logical upper bounds are about
131.7 MiB and 98.7 MiB before scale metadata. Thus the guaranteed local
train-state opportunity is only 98.7-197.5 MiB. Switching Adam to stateless
SGD without changing the learner removes the 131.7 MiB BF16 moments by itself.
None of these figures resolves the multi-GiB model and activation capacity
problem at long context.

Backprop-free training could save substantially more activation and reverse
workspace memory. That is EGGROLL's real capacity opportunity. It is offset by
three costs on this machine:

- population activations, recurrent/KV state, and outputs scale with the
  number of simultaneous perturbations;
- sequentially microbatching a population preserves memory but multiplies
  forward/rollout time; and
- HyperscaleES constructs matrix-shaped updates, while eggroll-vLLM avoids a
  simultaneous model-sized update tree by applying one FP32 layer update at a
  time. That runtime still requires mutable base weights and a full-model
  checkpoint and has no verified ROCm memory schedule.

Long-context rollout state is a second, independent limit. At the exact
Qwen3.5-4B geometry, one simultaneously active 32K divergent trajectory has a
projected raw cache of about 1.05 GiB: 1.00 GiB of BF16 K/V for eight full
attention layers plus 48 MiB of FP32 matrix state for the 24 GDN layers. This
is arithmetic, not an allocator measurement, and excludes convolution state,
cache metadata, fragmentation, weights, and workspaces. Prefix caching may
reuse pages only when adapter identity and prompt state are compatible; it
cannot share divergent responses across perturbations. The current Qwen
same-budget experiment stops at an 8,192-token maximum response and population
256 on one GH200. Population 4,096/16,384 keeps 256 members per GPU and scales
GPU count; it does not demonstrate a large local 32K population.

Applying ES only to the existing LoRA A/B arrays avoids mutable full base
weights, but it also gives up EGGROLL's main full-parameter claim. The official
Transformer runtime instead uses transient low-rank population perturbations
to form and apply a potentially higher-rank base-matrix update. For an
`N`-member, rank-`r` population, that update has rank at most
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
lower forward residency, but eggroll-vLLM uses `dtype=auto` and has no
implemented quantization configuration. The paper's INT8 result uses
per-output-channel symmetric matmul weights and integer updates for an RWKV
distillation recipe, with BF16 scales/moments. It does not establish W8/W4 Qwen
weights, quantized activations/optimizer state, or Qwen3.5 quality. The
implementation and numerical gates for the shared quantized projection path
are in [`MEGAKERNELS.md`](MEGAKERNELS.md).

There is now a more actionable exact-hardware inference path. vLLM merged a
[native HIP W4A16 kernel for gfx1100](https://github.com/vllm-project/vllm/pull/41394)
on 2026-05-29. The upstream test used an RX 7900 XTX (`gfx1100`) with a
Qwen3.6-27B GPTQ group-32 checkpoint and reports 205-345 token/s for BF16
activations, 2.5-4.2x its Triton BF16 baseline. Those are upstream inference
measurements on another model and shapes, not local Qwen3.5-4B or training
results. Nevertheless, this merged gfx1100 implementation is a stronger
starting point for quantized GRPO rollout generation and kernel-design review
than either porting FlashQLA wholesale or changing the learner to EGGROLL. Its
PyTorch/vLLM HIP operator is not directly callable from the JAX learner.

## EGGROLL speed evidence and expected result here

The current paper supplies a paired Qwen quality/budget result. Its Qwen3-4B,
rank-1, maximum-response-8,192 run uses population 256, one GH200, and nine
hours. The authors match hardware and effective wall clock against GRPO. The
reported five-benchmark averages are EGGROLL 49.8 versus GRPO 47.5 on
DeepScaleR and 48.4 versus 44.2 on ORZ. EGGROLL performs three optimization
steps versus GRPO's 200, but this is not a 67x speedup: each ES step evaluates
hundreds of population members. The paper does not give a numeric Qwen peak
VRAM comparison. Larger population 4,096/16,384 results scale GPU count while
retaining 256 members per GPU and therefore spend more aggregate compute.

The upstream headline is a gain over **naive ES**, not a measured gain over
SkyRL's Qwen LoRA learner. The paper's controlled throughput figure uses one
GH200 (described as equivalent to one H100), a BF16 linear model of dimension
8,192, and a maximum batch/population of 1,024. Normalized to pure inference at
100, it reports EGGROLL at 91 with pre-generated noise and 69 while regenerating
noise, versus a PPO reference at 34. Those are respectively 2.68x and 2.03x
over that PPO kernel path under high-population NVIDIA test conditions. They
are synthetic single-BF16-linear-layer throughputs, not Qwen wall-clock
speedups. The project also reports roughly 100x over unstructured naive ES,
not over GRPO. The paper's
roofline analysis estimates that this 8,192-wide, rank-1 case needs a batch of
352 to reach 300 operations per byte, so the headline result is explicitly a
large-population result rather than a small-population forecast.

None of those ratios transfers to Qwen3.5 on gfx1100:

- the RX 7900 XTX has no room for a large simultaneous long-context Transformer
  population; the official runtime distributes and schedules its population
  across vLLM workers rather than demonstrating 24 GiB one-card residency;
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
| Full-parameter EGGROLL | Potentially inference-like activation memory | Official Qwen3 runtime streams layer updates, but lacks Qwen3.5/ROCm/32K support and local fit evidence |
| Current single-GH200 Qwen ES-RLVR | Inference-like activation claim, no numeric Qwen peak | Matched-budget quality beats the paper's GRPO controls; no isolated end-to-end speed ratio |
| Upstream large-population regime | Strong published NVIDIA result | Not reachable with 24 GiB and Qwen KV/state geometry |

There is no evidence of a big enough end-to-end gain to justify replacing the
working optimizer. Keep EGGROLL as an optional ES-RLVR research algorithm, not
the SFT path or the default/exclusive GRPO learner.

## If an EGGROLL experiment is still desired

The Qwen evidence now justifies a separate ES-RLVR prototype after the ordinary
learner and rollout engine are stable. Use the official GPL runtime as a
behavioral reference or separate work, and use a clean-room implementation if
it must remain inside Apache-licensed SkyRL. Do not replace SFT with ES. The
first local gate should:

1. add complete Qwen3.5 mappings for both full-attention projections and GDN
   `in_proj_*`/`out_proj` matrices;
2. establish ROCm/RCCL/vLLM correctness on gfx1100 before measuring quality;
3. use antithetic populations 2, 4, then 8, regenerated from seeds and executed
   in bounded serialized waves, never as one stored adapter per member;
4. compare the same short, verifiable-reward prompt set against a GRPO control
   on identical BF16 or quantized frozen weights, recording quality, generated
   tokens/s, wall time, peak VRAM/RAM, disk traffic, and energy; and
5. stop on failed finiteness/reproducibility, higher peak memory, or failure to
   reach equivalent reward improvement within 1.2x of GRPO wall time, then
   repeat at longer contexts up to 32K only after the short gate passes.

This tests whether a small, serialized population has local value. It neither
reproduces the paper's 256-members-per-GH200 regime nor validates its synthetic
linear-layer throughput claim.

## FlashQLA portability result

FlashQLA's README summarizes 2-3x forward and about 2x backward speed over its
FLA Triton comparison across multiple NVIDIA Hopper/Blackwell scenarios. The
[checked-in CUDA-graph tables](https://github.com/QwenLM/FlashQLA/tree/e9c5836c4c1f3fa92532e1e568eacacabf45542d/benchmark)
are more shape-dependent. For the exact
`35B/9B/4B TP1`, `h_qk=16`, `h_v=32`, one-sequence geometry, the current
forward speedup over FLA is:

| NVIDIA device | 2K | 4K | 8K | 16K | 32K |
|---|---:|---:|---:|---:|---:|
| H200 | 1.30x | 1.54x | 1.82x | 2.13x | 2.34x |
| GB200 | 1.59x | 1.65x | 1.67x | 2.02x | 2.39x |
| RTX 5090 | 0.92x | 1.29x | 1.61x | 1.83x | 1.98x |
| RTX PRO 6000 Blackwell Server Edition | 0.99x | 1.62x | 1.92x | 2.38x | 2.52x |

The new PRO 6000 table strengthens the long-context NVIDIA evidence without
making short-context gains universal. The backward tables do not include this
exact `16/32` head geometry. These are GDN-operation timings with 10 warmups
and 100 CUDA-graph replays, not end-to-end Qwen training ratios and not an AMD
forecast. Command-buffer/graph capture is prohibited in this local stack, so
the benchmark method itself is also not transferable.

The two commits after the prior audit only change the SM120 forward path and
benchmarks: they fix no-initial-state handling, revise the SM120 KKT schedule,
and add the RTX PRO 6000 results. They add no HIP/gfx1100 dispatch, JAX FFI,
or SM120 backward. The portability decision is unchanged. The local NumPy
algebra oracle intentionally remains pinned to the earlier unchanged
`40b7527f` source equations rather than silently changing its reference.

At the audited revision, FlashQLA provides chunked-prefill forward/backward,
not a merged recurrent decode path. The only current decode/verify work is an
[open pull request](https://github.com/QwenLM/FlashQLA/pull/20) for SM90
Hopper, so it adds neither released decode support nor AMD evidence.

The source cannot execute on this AMD stack:

- the [documented runtime](https://github.com/QwenLM/FlashQLA/blob/e9c5836c4c1f3fa92532e1e568eacacabf45542d/README.md)
  requires CUDA 12.8+, PyTorch 2.8+, and NVIDIA SM90/SM100, with v0.1.2 adding
  a forward-only SM120 path;
- [dispatch](https://github.com/QwenLM/FlashQLA/blob/e9c5836c4c1f3fa92532e1e568eacacabf45542d/flash_qla/ops/gated_delta_rule/chunk/__init__.py)
  calls TileLang's `nvcc` compute-version query and selects only `9.0`, `10.0`,
  or a partial `12.0` implementation;
- runtime utilities use `torch.cuda`, CUDA devices/events, NVIDIA
  multiprocessor properties, warpgroup specialization, and NVIDIA-specific
  layouts; and
- there is no HIP/gfx1100 implementation or JAX FFI boundary.

Attempting to install it would add a CUDA/PyTorch kernel stack without creating
an AMD path. Source inspection is sufficient to reject direct use. FlashQLA
itself is MIT-licensed, so a port can reuse source subject to its notice; the
blocker is the NVIDIA-specific implementation, not a copyleft conflict.

TileLang itself is now at
[v0.1.12](https://github.com/tile-ai/tilelang/releases/tag/v0.1.12);
[v0.1.10](https://github.com/tile-ai/tilelang/releases/tag/v0.1.10) added
RDNA3/RDNA3.5 WMMA and ROCm 7.2 support. FlashQLA still
[pins TileLang 0.1.9](https://github.com/QwenLM/FlashQLA/blob/e9c5836c4c1f3fa92532e1e568eacacabf45542d/setup.py)
and manually selects NVIDIA kernels and layouts. Upgrading that dependency
would therefore not port FlashQLA, though the new gfx11 backend makes TileLang
one candidate for independently implementing the bounded AMD stages. A JAX
production path would additionally need a typed FFI/custom-call boundary and
an explicit custom VJP; FlashQLA's PyTorch API is not such a boundary.

Its architecture is nevertheless the right reference. FlashQLA explicitly
rejects both a decomposition into many independent elementary operations and a
single kernel for the entire computation. It fuses several high-value GDN
stages, carries bounded state, and supplies explicit forward/backward kernels
on SM90/SM100 (SM120 is forward-only). That maps to the local three-stage plan:

1. bounded WY/KKT preparation;
2. bounded state/output execution with native Q-head to V-head grouping; and
3. explicit reverse-superblock recomputation/custom VJP.

Commit `f3f711d4` adds an import-light CPU gate for the previously unresolved
representation change. `gdn_forward_oracle.py` independently evaluates the
token recurrence, FlashQLA's no-decay KKT inverse followed by its decayed
`Ag`, and SkyRL's decay-folded U/W triangular solve. It proves the two chunk
representations are related by diagonal decay conjugation and produce the same
output and final state.

The exact Qwen `Hk=16`, `Hv=32`, `Dk=Dv=128` cases passed at S=64 and S=128
with nonzero initial state, and right-padding boundaries 63/64/65 passed with
native `hv -> hv // 2` grouping. Across the exact-geometry tests, worst
absolute and relative drift from the direct recurrence were respectively
`4.47e-8` and `5.2914e-7`; an additional 72-case CPU stress matrix had worst
absolute drift `4.10e-8`. Ten committed tests passed, the default import loaded
no JAX, JAXlib, PyTorch, or TileLang, and no accelerator was used. The oracle
and test SHA-256 values are
`1a6394e24ec2dc2cc5481569e92b3eb5f75c2f001c19cea68c8de531edffe87f`
and `5f2a8d994c9d291dbf62cde6da66787a89f4cb7e6192597857facf19cc400567`.

This clears FP32 forward algebra for a source-level adaptation; it does not
validate BF16 storage or scheduling, HIP numerics, FlashQLA's analytic
backward, or an integrated custom VJP. The wider CPU semantic and compiler
evidence for the AMD-specific boundary remains in
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
