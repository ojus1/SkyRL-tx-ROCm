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
