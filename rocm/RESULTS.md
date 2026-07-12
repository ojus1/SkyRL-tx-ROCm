# Qwen3.5 ROCm validation record

This file records only experiments that completed on the local Radeon RX 7900
XTX (`gfx1100`) with the monitor attached to the Intel iGPU. It is not a claim
that larger context buckets are safe or that synthetic SFT loss is a quality
measurement. Every GPU run used a fresh process, private telemetry files, and
fatal AMDGPU journal monitoring.

## Current safety configuration

- ROCm 7.2 and JAX/JAXlib/ROCm plugin/PJRT 0.10.2.
- `XLA_FLAGS=--xla_gpu_enable_command_buffer=`. Command-buffer replay is
  disabled because the second full-model Adam replay previously produced an
  HSA invalid-packet/illegal-opcode failure; repeated updates pass with it off.
- JAX preallocation is disabled and the Pallas attention path is opt-in.
- Junction temperature is capped at 90 C, VRAM at 23 GiB, minimum available
  host memory at 4 GiB, and swap use at zero.
- The AMD display connectors must be disconnected and `/dev/kfd` unowned
  before launch. A 60-second cold-start sensor grace accommodates a
  runtime-suspended headless GPU.

## Full Qwen3.5-4B SFT control

On 2026-07-12, revision `431ee0b3fb9ae04b4c56d41fee95d593b7476cbc`
completed one cold and five measured forward/backward/Adam steps for
`Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The configuration was batch 1,
context 64, rank-8 LoRA, two adapter slots, per-layer rematerialization, and a
64-token loss chunk.

| Result | Measured value |
|---|---:|
| Cold forward/backward plus first Adam | 87.230 s |
| Measured median step | 0.5931 s |
| Measured p95 step | 1.0707 s |
| Median useful throughput | 107.90 tokens/s |
| Maximum system VRAM used | 18,067,054,592 B (16.83 GiB) |
| Maximum process RSS / PSS / USS | 16.57 / 16.30 / 16.04 GiB |
| Maximum / p95 junction temperature | 73 / 61 C |
| Maximum / p95 GPU power | 271 / 137 W |
| Maximum host memory used | 16.79 GiB |
| Maximum swap used | 0 B |

All six losses and gradient norms were finite. The model unload endpoint
succeeded, no fatal AMDGPU journal event was observed, VRAM returned to the
27,947,008-byte idle baseline, and no process retained `/dev/kfd`. The local
private artifacts are:

- `/tmp/postfix2-t64-1783854419.sft.jsonl`
- `/tmp/postfix2-t64-1783854419.telemetry.jsonl`
- `/tmp/postfix2-t64-1783854419.telemetry.jsonl.summary.json`

The client-observed step time includes the local HTTP/database/future path. It
must not be compared directly with isolated device-event kernel timing.

## Full Qwen3.5-4B SFT at context 512

On 2026-07-12, revision `82e038f61565e820d8dc86b5fd285eb77a07a301`
completed one cold and five measured forward/backward/Adam steps at batch 1,
context 512, using the opt-in Pallas attention path. LoRA, rematerialization,
loss chunking, model revision, and command-buffer settings matched the
64-token control.

| Result | Measured value |
|---|---:|
| Cold forward/backward plus first Adam | 170.225 s |
| Server-reported train JIT | 157.79 s |
| Measured median / p95 step | 1.4281 / 1.4299 s |
| Median useful throughput | 358.52 tokens/s |
| Maximum system VRAM used | 18,530,045,952 B (17.26 GiB) |
| Maximum process RSS / PSS / USS | 16.13 / 15.77 / 15.59 GiB |
| Maximum / p95 junction temperature | 80 / 66 C |
| Maximum / p95 GPU power | 358 / 143 W |
| Maximum host memory used | 17.35 GiB |
| Maximum swap used | 0 B |

All six losses and gradient norms were finite, the adapter unload succeeded,
the fatal-journal query was empty, VRAM returned to the 27,947,008-byte idle
baseline, and `/dev/kfd` was unowned after exit. Relative to the T64 control,
the end-to-end client median used 2.41x the time for 8x the useful tokens, or
3.32x the throughput. This comparison includes polling/API overhead and also
changes attention implementation at the 512-token safety boundary; it is not
an isolated Pallas speedup.

The manifest had no tracked source diff and identified the exact commit above.
It did list three untracked, unwired GRPO-benchmark files being reviewed in
parallel; none is imported by the server or SFT client. Local private artifacts
are:

- `/tmp/t512-sft-1783856707.sft.jsonl`
- `/tmp/t512-sft-1783856707.telemetry.jsonl`
- `/tmp/t512-sft-1783856707.telemetry.jsonl.summary.json`

### Context-1024 progression

The same guarded configuration then completed one cold and five measured
steps at context 1,024. The server-reported JIT was 160.24 s; the client cold
step including the first Adam was 172.242 s. Median/p95 step time was
2.1917/2.2023 s, or 467.21 useful tokens/s. Peak system VRAM was
18,823,901,184 B (17.53 GiB), maximum process RSS/PSS/USS was
16.93/16.72/16.47 GiB, and maximum host use was 16.90 GiB. Maximum/p95
junction temperature was 85/72 C, maximum/p95 GPU power was 358/160 W, and
swap remained unused.

All losses and gradient norms were finite, adapter unload succeeded, the fatal
journal query was empty, and the device returned to its idle VRAM/KFD state.
The local artifacts are:

- `/tmp/t1024-sft-1783857150.sft.jsonl`
- `/tmp/t1024-sft-1783857150.telemetry.jsonl`
- `/tmp/t1024-sft-1783857150.telemetry.jsonl.summary.json`

### Context-2048 default-allocator capacity boundary

The first context-2,048 full-model attempt used the default JAX device-memory
fraction, a stricter 85 C junction cap, and five-second untimed cooling delays.
Attention had already passed its isolated forward/backward gate, but the full
training executable did not fit. XLA reported that rematerialization reduced
its estimate from 10.60 GiB to 10.48 GiB but could not reach the 9.59 GiB
target; the allocator then rejected a 10.56 GiB request. No training step ran.

The failure was a handled `RESOURCE_EXHAUSTED` result: adapter cleanup
succeeded, peak observed VRAM was 18,294,276,096 B, maximum junction
temperature was 84 C, swap remained zero, the fatal-driver journal query was
empty, and KFD/VRAM returned to idle. This does not move the full-model
validation frontier beyond 1,024. It establishes memory residency/buffer
assignment, rather than attention correctness, as the next blocker. Artifacts
are:

- `/tmp/t2048-sft-1783858067.sft.jsonl`
- `/tmp/t2048-sft-1783858067.telemetry.jsonl`
- `/tmp/t2048-sft-1783858067.telemetry.jsonl.summary.json`

A fresh-process retry explicitly raised
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` while lowering the profiler's system-VRAM
cap to 22 GiB. It failed on the same 10.56 GiB allocation, unloaded cleanly,
and produced no driver event. Increasing the allocator fraction therefore does
not solve this boundary; compact model residency and/or a smaller executable
working set is required. Retry artifacts are
`/tmp/t2048-fraction90-sft-1783858376.{sft.jsonl,telemetry.jsonl}` (with the
usual `.summary.json` beside telemetry).

An initialization-only server run then set
`XLA_PYTHON_CLIENT_ALLOCATOR=platform` and received no client request. It still
settled at 17,899,212,800 B of physical VRAM, indistinguishable from the
default route. The process was stopped normally after observation, reached
68 C, used no swap, and produced no fatal driver event. This installed ROCm
JAX stack therefore does not provide a useful deallocating platform-allocator
escape hatch. Telemetry is in
`/tmp/t2048-platform-sft-1783858638.telemetry.jsonl`.

### Model-residency accounting

`rocm/probe_model_residency.py` separates abstract state size, eager model
initialization, checkpoint loading, NNX splitting, and settling without
compiling or executing a model pass. The exact abstract state is
8,480,538,476 B (7.8981 GiB): 8,411,510,272 B of non-LoRA state and
69,028,204 B of LoRA metadata/parameters. After normal initialization and
loading, `jax.live_arrays()` reported 16,892,048,748 B because the module and
split states both reference arrays, but pointer deduplication found exactly
8,480,538,476 B across 1,182 unique buffers. Thus `nnx.split` aliases the base
buffers; it does not duplicate the model.

The allocator view explains the much larger physical number. With the
requested `platform` allocator, backend-ready `bytes_in_use` was
9,127,142,144 B while `pool_bytes` was 17,181,966,336 B and observed physical
VRAM was 17,899,282,432 B. An experimental abstract-model/direct-load route,
with fused checkpoint interleaving performed in NumPy rather than accidentally
through JAX, reported settled `bytes_in_use` of 8,604,524,288 B and peak
`bytes_in_use` of 8,674,484,224 B. That run used a different allocator setting,
so the counter difference is confounded and is not a proven loader saving.
The decisive counters are unchanged: the route had the same 8,480,538,476 B
unique live state, BFC still grew its pool to 17,179,869,184 B, and physical
VRAM still reached 17,895,243,776 B. Direct loading alone therefore does
**not** reduce settled model state or reclaim the roughly 8 GiB pool growth.

These are initialization-only measurements, not a validated backend change.
The direct-load route is now wired behind the default-off
`abstract_model_load` flag, but it has not passed a ROCm model forward or
training step. The evidence rules out duplicate NNX state as the primary cause
and makes allocator-pool geometry plus the 10.56 GiB executable arena the
capacity problem to solve. Artifacts are:

- `/tmp/residency-bfc-1783859376.{jsonl,telemetry.jsonl}`
- `/tmp/residency-platform-1783859495.{jsonl,telemetry.jsonl}`
- `/tmp/residency-abstract-load4-1783860134.{jsonl,telemetry.jsonl}`

For the default 75% BFC limit, allocator-live state plus the requested arena is
already impossible even with perfect placement: approximately
`8.500 + 10.560 > 17.988 GiB`. The 90% growth-mode retry has aggregate room,
but neither its cached free region nor its remaining extension budget is large
enough for one 10.56 GiB allocation. The next allocator experiment is a fixed
85% preallocated BFC arena combined with direct loading, which should create
one 20.387 GiB region and avoid replacement holes. That is quantitative
feasibility only; allocation-only, load-only, compile-only, and one-step gates
must pass in separate fresh processes before another multi-step run.

### Fixed-BFC allocation and direct-load gates

The reviewed fixed-85% BFC allocation-only probe passed in a fresh guarded
process. XLA reported `bytes_limit == pool_bytes == peak_pool_bytes ==
21,892,169,728 B` (20.3887 GiB), with only the 256-byte probe array live. The
value is exactly one 2 MiB granule above the raw sysfs-times-0.85 estimate;
OpenXLA explicitly rounds the BFC GPU limit upward to 2 MiB. Physical VRAM
peaked at 22,605,082,624 B (21.0526 GiB), junction temperature at 49 C, and
swap stayed at its pre-existing 622,592-byte baseline. The process returned to
27,947,008 B idle VRAM with unowned KFD and no fatal driver event. Artifacts:

- `/tmp/bfc85-allocation-1783863595.jsonl`
- `/tmp/bfc85-allocation-1783863595.telemetry.jsonl`
- `/tmp/bfc85-allocation-1783863595.telemetry.jsonl.summary.json`

The next fresh process loaded the pinned checkpoint through the abstract
constructor without a model pass. `loaded` contained exactly 1,182 unique
buffers and 8,480,538,476 live bytes; `nnx.split` left both unique-buffer deltas
at zero. Settled BFC usage was 8,484,341,504 B with a peak of 8,553,896,960 B,
leaving 13,407,828,224 B (12.4870 GiB) inside the fixed pool—1.9270 GiB beyond
the nominal 10.56 GiB failed arena request. Physical VRAM peaked at
22,610,644,992 B, junction temperature at 54 C, swap did not grow, artifacts
were mode 0600, and exit again restored idle KFD/VRAM without a fatal driver
event. Artifacts:

- `/tmp/bfc85-residency-1783864015.jsonl`
- `/tmp/bfc85-residency-1783864015.telemetry.jsonl`
- `/tmp/bfc85-residency-1783864015.telemetry.jsonl.summary.json`

These two gates establish allocation and residency feasibility only. They do
not execute forward/backward, prove compiled temporary placement, or move the
validated full-model frontier beyond context 1,024.

The first context-2,048 compile-control invocation then completed exact backend
setup but deliberately stopped during lowering: the attention selector refused
the quadratic XLA fallback because the probe had not explicitly selected the
Pallas route. No model-pass callable or optimizer step ran. Peak physical VRAM
was 22,612,500,480 B, junction temperature was 59 C, swap did not grow, and the
process returned cleanly to idle without a driver event. The probe now exposes
an exact attention selector: effective contexts below 512 use `xla`, while
512--16,384 require explicit `pallas`. This handled control failure is not a
compile pass. Artifacts are
`/tmp/bfc85-compile-t2048-1783864343.{jsonl,telemetry.jsonl}` with the telemetry
summary alongside them.

## Fixed-rollout GRPO learner control

Revision `31800cf001c0c982e56231f386182f0cb02c163c` completed one cold
and five measured `importance_sampling` forward/backward/Adam steps for one
deterministic two-rollout group at context 64. Each rollout had 16 action
tokens, rewards `[0,1]`, advantages `[-0.5,+0.5]`, rank-8 LoRA, and a scalar
synthetic old log-probability of -5. No sampling, grading, checkpoint export,
or sampler synchronization was performed.

| Result | Measured value |
|---|---:|
| Cold learner step | 44.267 s |
| Server-reported train JIT | 28.36 s |
| Measured median / p95 step | 0.9258 / 0.9436 s |
| Median learner throughput | 138.26 tokens/s |
| Median action-token throughput | 34.57 tokens/s |
| Maximum system VRAM used | 18,067,034,112 B (16.83 GiB) |
| Maximum process RSS / PSS / USS | 17.10 / 16.82 / 16.57 GiB |
| Maximum / p95 junction temperature | 77 / 68 C |
| Maximum / p95 GPU power | 318 / 145 W |
| Maximum host memory used | 16.95 GiB |
| Maximum swap used | 0 B |

All policy metrics and optimizer gradient norms remained finite, the adapter
unloaded, the fatal-journal query was empty, and KFD/VRAM returned to idle. The
scalar synthetic old log-probability is deliberately not sampled from the
model: action importance ratios spanned roughly `9e-5` to `148`, and gradient
norms ranged from 15.3 to 35.5. This is learner-path and numerical-finiteness
evidence, not a realistic ratio/KL distribution or policy-quality result.

Local private artifacts are:

- `/tmp/grpo-t64-g2-1783857586.grpo.jsonl`
- `/tmp/grpo-t64-g2-1783857586.telemetry.jsonl`
- `/tmp/grpo-t64-g2-1783857586.telemetry.jsonl.summary.json`

## Isolated evidence and known failures

- The bounded correctness probe completed 512-token BF16 Pallas attention for
  both an all-valid sequence and a 385-valid/127-padded sequence. For the
  all-valid case, median forward/backward latency was 0.850/1.824 ms after one
  compile run, JAX peak allocation was 171,963,648 B, maximum junction
  temperature was 60 C, and no fatal driver event occurred. Against the
  16-query-token-block FP32 reference, output relative L2/cosine was
  0.00202/0.999998. The `dq` and `dk` relative-L2 errors were 0.0113 and
  0.0108, slightly outside the initial 0.01 promotion gate, so this remains
  experimental despite its otherwise clean result. The padded result was
  materially the same. Telemetry is in
  `/tmp/pallas512-reference-1783855014.telemetry.jsonl` and
  `/tmp/pallas512-pad385-1783855051.telemetry.jsonl`.
- The same bounded reference at 1,024 tokens completed with 1.194/4.645 ms
  median forward/backward, 339,747,840 B JAX peak allocation, and no driver
  event. Output relative L2 was 0.00206; `dq`/`dk` remained 0.0113/0.0107, so
  the numerical promotion decision did not change. Telemetry is in
  `/tmp/pallas1024-reference-1783857107.telemetry.jsonl`.
- At 2,048 tokens the isolated path also passed: median forward/backward was
  2.942/15.347 ms, JAX peak allocation was 948,325,888 B, maximum junction was
  67 C, and there was no driver event. Output/`dq`/`dk` relative L2 remained
  0.00210/0.0113/0.0107. Telemetry is in
  `/tmp/pallas2048-reference-1783857879.telemetry.jsonl`.
- The exact synthetic Qwen3.5 LoRA optimizer tree contains 402 leaves and
  34,512,896 BF16 parameter elements. Three repeated GPU updates passed with
  command buffers disabled, with about 70 ms steady optimizer time and
  1.806 GiB peak VRAM.
- The opt-in monolithic Pallas attention path previously completed BF16
  batch-1, 16,384-token, 16-query-head/4-KV-head, dimension-256 forward and
  backward in about 184 ms and 895 ms respectively, with 1.544 GiB JAX peak
  allocation. This is isolated evidence, not permission to exceed the hard
  16K cap.
- A 32,768-token monolithic attention backward caused an AMDGPU ring timeout
  and reset. The implementation now refuses that shape. A query-bounded
  forward/backward kernel is required before any new 24K/32K attention test.
- A former full-model run completed its first update and the second forward,
  then failed during the second Adam command-buffer replay with invalid packet
  format and illegal opcode. Disabling all XLA GPU command buffers is therefore
  mandatory on this machine.

## Validation frontier

The post-fix full-model SFT validation frontier is currently context 1,024. A
fixed-rollout GRPO learner control is verified at context 64; real sampling,
fixed-real-rollout replay, KL/reward comparison, and end-to-end GRPO remain
unverified. Contexts above 1,024, quantized model execution, and fused-kernel
execution also remain unverified. The Pallas gradient result remains just
outside its initial promotion gate, so it is still opt-in despite the clean
integrated runs. The staged gates in [`MEGAKERNELS.md`](MEGAKERNELS.md) must be
followed; no quantized or custom-kernel path is enabled by default.
