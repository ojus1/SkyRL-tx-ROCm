# Exact Qwen3.5 SFT compile-only probe

`probe_sft_compile.py` isolates the next context-2,048 capacity gate without
sending a Tinker request or executing the lowered model pass. Its default path
forces `JAX_PLATFORMS=cpu`, emits a refusal record, and exits before importing
JAX. The initial implementation was CPU-reviewed before the staged ROCm gates;
subsequent hardware results are recorded in `RESULTS.md`.

The explicit ROCm path reproduces the normal batch-one SFT lifecycle up to the
first model pass:

1. Select the pinned `Qwen/Qwen3.5-4B` revision
   `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
2. Construct `JaxBackendImpl` with two adapter slots, maximum rank 8,
   microbatch 1, per-layer checkpointing, and loss chunk size 64. The default
   construction is the pending `abstract_model_load=True` direct-load route;
   `--construction eager` is available as a control.
3. Call the normal `create_model` method for adapter index 1 with rank 8,
   alpha 32, and seed 0. This intentionally makes the Adam state and global
   accumulated-gradient tree resident, matching the API lifecycle that
   preceded the original failure.
4. Build the exact sharded abstract input signature: batch-one integer token,
   mask, adapter, and target arrays; FP32 loss-mask, sampling-logprob, advantage,
   and loss-config arrays; and the real backend parameter/state trees.
5. Call `_forward_backward_and_accumulate.lower(...)`, summarize StableHLO,
   call `lowered.compile()`, and report `CompiledMemoryStats`, cost analysis,
   optimized-HLO metadata, allocator counters, and timings.

The compiled callable is never invoked, and `optim_step` is never called.
Every record carries `model_pass_executable_invocations: 0`.

The probe flushes a minimal `setup_stage` record before every potentially
blocking action. It separately synchronizes constructor state, newly created
optimizer/adapter state, and the combined final state before marking each phase
complete. After the July 12 illegal-opcode event, every guarded Qwen3.5
full-model entrypoint also refuses a current boot whose kernel journal contains
a fatal AMDGPU event; rebooting is the only way to clear that quarantine.
`--stop-after-backend-ready` provides a setup-only diagnostic and exits before
lowering. That diagnostic subsequently passed after the ROCm 7.2.4 upgrade and
reboot; it remains useful for isolating setup from compilation.

ROCm effective (bucketed) context 512 and above must explicitly select
`--attention-backend pallas`; the code refuses the quadratic XLA fallback.
The isolated 2,048-token Pallas path passed the hardware safety gate, but its
`dq`/`dk` relative L2 remained about 1.1%, above the 1% numerical promotion
threshold. A successful compile therefore proves capacity only, not numerical
promotion for training.

The attention integration now installs a narrow, fail-closed JAX 0.10.2 patch
that casts only the Pallas backward delta preprocess's loaded `O` and `dO`
tiles to FP32 before their product and reduction. In CPU interpret mode at the
exact Qwen3.5 geometry `T=64, Hq=16, Hkv=4, D=256`, seed 0 `dq`/`dk`
relative L2 improved from `0.00763/0.00782` to `0.00317/0.00420`; forward,
`dV`, and the bounded FP32 reference were unchanged. Both patched gradients
pass the existing 1% gate without relaxing it. This is a deterministic CPU
regression result, not hardware promotion: the isolated 512/1,024/2,048-token
ROCm accuracy ladder must be repeated with the patched preprocess before the
long-context training block is removed.

## ROCm 7.2.4 hardware results

The post-upgrade qualification advanced one fresh guarded process at a time.
All three wrappers returned status 0, the current-boot fatal-driver scan stayed
empty, swap did not grow, and `/dev/kfd` plus physical VRAM returned to idle
after exit.

| Gate | Setup / lower / compile | Compiled temporary | Allocator peak-live | Physical VRAM peak | Junction / power max |
|---|---:|---:|---:|---:|---:|
| Context-64 setup only | 96.173134 / n/a / n/a s | n/a | 8,829,511,424 B | 22,608,961,536 B | 61 C / 137 W |
| Context-64 XLA compile only | 39.642890 / 6.228467 / 52.449389 s | 196,247,088 B | 10,076,861,696 B | 22,655,623,168 B | 73 C / 206 W |
| Context-2,048 Pallas compile only | 39.177387 / 6.350830 / 94.074129 s | 11,390,298,880 B | 12,522,268,672 B | 23,003,750,400 B | 79 C / 311 W |

The setup-only JSONL contains `backend_ready` and `stopped` but no `lowered` or
`compiled` record. Both compile-only JSONLs contain a final `compiled` record
with `status: passed`. Most importantly, the context-2,048 record reports
`model_pass_executable_invocations: 0` and
`optimizer_step_invocations: 0`: neither the returned full-model callable nor
the optimizer ran. Compilation may have executed only the bounded
autotuning/profiling work described above. Artifacts:

- `/tmp/postrocm-backend-setup-1783872188.jsonl`
- `/tmp/postrocm-backend-setup-1783872188.telemetry.jsonl`
- `/tmp/postrocm-backend-setup-1783872188.telemetry.jsonl.summary.json`
- `/tmp/postrocm-compile-t64-1783872326.jsonl`
- `/tmp/postrocm-compile-t64-1783872326.telemetry.jsonl`
- `/tmp/postrocm-compile-t64-1783872326.telemetry.jsonl.summary.json`
- `/tmp/postrocm-compile-t2048-pallas-1783872544.jsonl`
- `/tmp/postrocm-compile-t2048-pallas-1783872544.telemetry.jsonl`
- `/tmp/postrocm-compile-t2048-pallas-1783872544.telemetry.jsonl.summary.json`

This is compile-only capacity evidence. `bench_sft.py --one-update-gate` has
passed independent source review, CPU/mocked protocol tests, and its first
context-64 hardware validation on ROCm 7.2.4. That result is recorded in
`RESULTS.md`; it does not authorize invoking the 2,048-token executable.
Separate Pallas numerical qualification also remains required: the
approximately 1.1% isolated `dq`/`dk` relative-L2 result still exceeds the 1%
promotion threshold.

## Safety contract

ROCm requires both `--platform rocm` and `--allow-gpu`. The probe also:

- fixes BFC preallocation at 85% using `XLA_CLIENT_MEM_FRACTION=0.85`;
- disables command buffers before importing JAX;
- pins all accelerator visibility variables to device 0;
- refuses conflicting inherited allocator variables;
- shares the global Qwen3.5 launch lock with the server launcher;
- refuses an occupied `/dev/kfd`; and
- refuses an active physical AMD display connector.

Neither hardware guard has an override in this probe.

Contexts above 2,048 additionally require `--allow-large-context` and are not
the first gate. Compilation can still allocate most of VRAM, consume substantial
host RAM, or fail with `RESOURCE_EXHAUSTED`; compile-only does not mean
resource-free.

The backend constructor and `create_model` necessarily perform checkpoint
device transfers, BFC preallocation, LoRA initialization, and optimizer/gradient
setup. Those may dispatch small setup operations. Here, “compile-only” means
that no SFT forward/backward executable or optimizer executable is run. This
caveat is recorded in the `backend_ready` JSON record rather than hidden.
`lowered.compile()` may also execute bounded GPU autotuning/profiling kernels
and allocate representative buffers. It therefore still needs the telemetry
guard even though the returned full model-pass callable is never invoked; this
second caveat is recorded in the manifest.

## Usage

The safe default is:

```bash
.venv/bin/python rocm/probe_sft_compile.py
```

It emits `manifest` and `refused` records only. The setup-only reproducer must
run under telemetry:

```bash
.venv/bin/python rocm/profile_rocm.py \
  --output /tmp/qwen35-backend-setup.telemetry.jsonl \
  --interval 0.25 --timeout 600 --sensor-grace-seconds 60 \
  --max-junction-temp-c 80 --max-vram-gib 22 \
  --min-host-available-gib 8 --max-swap-gib 0.001 -- \
  .venv/bin/python rocm/probe_sft_compile.py \
    --platform rocm --allow-gpu --context 64 \
    --attention-backend xla --stop-after-backend-ready \
    --output /tmp/qwen35-backend-setup.jsonl
```

The setup-only sequence and subsequent context-2,048 Pallas compile-only gate
have now passed cleanly after reboot. This does not authorize invoking the
compiled callable. The reviewed exact-one-update client passed its context-64
hardware gate, but effective context 2,048 still carries the Pallas numerical
qualification above.

Setup-only success produces `manifest`, the flushed `setup_stage` sequence,
`backend_ready`, and `stopped`; it produces no `lowered` or `compiled` record.
Compile-only success additionally produces `lowered` and `compiled`.
A guarded Python failure produces an `error` record and a nonzero exit. A fatal
driver event can terminate the process before it can emit that final error
record. Compiler memory reports are accounting, not physical peak VRAM;
interpret them alongside telemetry.

## CPU verification

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q \
  tests/rocm/test_sft_compile_probe.py
```

The tests cover the CPU-only default, explicit GPU acknowledgement, context
limits and bucketing, fixed allocator conflicts, command-buffer overriding,
and a source-level guard that the compiled callable is never invoked.
