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

ROCm effective (bucketed) context 512 and above must explicitly select
`--attention-backend pallas`; the code refuses the quadratic XLA fallback.
The isolated 2,048-token Pallas path passed the hardware safety gate, but its
`dq`/`dk` relative L2 remained about 1.1%, above the 1% numerical promotion
threshold. A successful compile therefore proves capacity only, not numerical
promotion for training.

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

It emits `manifest` and `refused` records only. The first real compile should
run in a fresh process under telemetry:

```bash
.venv/bin/python rocm/profile_rocm.py \
  --output /tmp/qwen35-sft-compile-t2048.telemetry.jsonl \
  --interval 0.25 --timeout 1800 --sensor-grace-seconds 60 \
  --max-junction-temp-c 80 --max-vram-gib 22 \
  --min-host-available-gib 8 --max-swap-gib 0.001 -- \
  .venv/bin/python rocm/probe_sft_compile.py \
    --platform rocm --allow-gpu --context 2048 \
    --attention-backend pallas \
    --output /tmp/qwen35-sft-compile-t2048.jsonl
```

Success produces `manifest`, `backend_ready`, `lowered`, and `compiled`
records. A guarded failure produces an `error` record and a nonzero exit. The
memory report is compiler accounting, not physical peak VRAM; interpret it
alongside the telemetry file.

## CPU verification

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q \
  tests/rocm/test_sft_compile_probe.py
```

The tests cover the CPU-only default, explicit GPU acknowledgement, context
limits and bucketing, fixed allocator conflicts, command-buffer overriding,
and a source-level guard that the compiled callable is never invoked.
