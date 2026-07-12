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

The post-fix full-model validation frontier is currently context 64. Context
512 and larger, GRPO, quantized model execution, and fused-kernel execution
remain unverified. The staged gates in [`MEGAKERNELS.md`](MEGAKERNELS.md) must
be followed; no quantized or custom-kernel path is enabled by default.
