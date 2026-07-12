# SkyRL Tinker on ROCm

This directory contains the reproducible local setup and smoke benchmark used
for an AMD Radeon RX 7900 XTX (`gfx1100`) with ROCm 7.2.4 and Python 3.12.

## Environments

Use separate virtual environments for the backend and cookbook client. Current
SkyRL requires Transformers 5.6.1 or newer while the current Tinker Cookbook
caps Transformers at 5.5.4.

```bash
# From this repository.
uv venv --python 3.12 .venv
UV_TORCH_BACKEND=cpu uv pip install --python .venv/bin/python -e '.[jax,tinker]'
uv pip install --python .venv/bin/python 'jax-rocm7-plugin==0.10.2'

# Clone the cookbook as a sibling of this repository.
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git ../tinker-cookbook
uv venv --python 3.12 ../tinker-cookbook/.venv
UV_TORCH_BACKEND=cpu uv pip install \
  --python ../tinker-cookbook/.venv/bin/python -e ../tinker-cookbook
```

Do not install SkyRL's `gpu` extra on AMD; it selects CUDA JAX. The ROCm plugin
provides both `jax-rocm7-plugin` and `jax-rocm7-pjrt`.

## Run the verified smoke benchmark

Start the server and wait for `Starting background engine...`:

```bash
./rocm/start_skyrl.sh
```

In another terminal:

```bash
TINKER_API_KEY=tml-dummy ../tinker-cookbook/.venv/bin/python \
  rocm/run_cookbook_smoke.py
```

The one-step rank-8 LoRA SFT run on `Qwen/Qwen3-0.6B` completed on the RX 7900
XTX with 30 tokens, `train_mean_nll=2.650795`, and gradient norm `12.1875`.

## ROCm compatibility fix

JAX reports both CUDA and ROCm devices through the generic platform name `gpu`.
SkyRL previously selected its explicitly cuDNN-backed fused attention for every
GPU, causing `cuDNN is not detected` on ROCm. The backend selector now checks
JAX's platform version and uses portable XLA attention for ROCm.

## Known limitation

Forward/backward, the optimizer step, and checkpoint saves are verified. A
post-checkpoint generation probe on JAX 0.10.2 hit
`HSA_STATUS_ERROR_INVALID_PACKET_FORMAT` while loading a LoRA sampler on
`gfx1100`. Local sampling on this exact combination remains unverified.

## Qwen3.5-4B training experiments

The Qwen3.5 path is deliberately stricter than the original smoke test. Keep
the display attached to the iGPU and ensure that no process owns `/dev/kfd`.
The launcher refuses an active AMD display connector, a second ROCm server,
an occupied port, an unsafe/reused run directory, or an existing KFD owner:

```bash
./rocm/start_qwen35.sh t64-control
```

It resolves the tested Qwen3.5-4B revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` from the local cache and pins
`JAX_PLATFORMS=rocm`; its default
`SKYRL_QWEN35_MEMORY_MODE=growth` disables JAX preallocation, preserving the
validated baseline. It also enables per-layer rematerialization, uses
rank-8/two-slot LoRA and 64-token loss chunks, and disables all XLA GPU command
buffers. The latter is required on this machine:
an exact 402-leaf Qwen3.5 LoRA Adam probe passed repeated GPU updates with
command buffers disabled, while the earlier full-model failure occurred on
the second replay of the same optimizer executable.

The ROCm 7.2.4 post-reboot floor was re-established before any full-model
probe: a no-preallocation JIT `float32[1]` add returned exact `[3.25]` for
`[1.25] + 2`, and a separate BF16 `16x16` all-ones matrix multiply plus
`value_and_grad(sum)` returned exact FP32 loss `4096.0` and gradient extrema
`16.0/16.0`. These shell checks have no saved artifacts and their process wall
times include startup, so they are correctness/safety evidence rather than
kernel benchmarks.

Run the synthetic end-to-end SFT control from the separate Cookbook
environment:

```bash
TINKER_API_KEY=tml-dummy ../tinker-cookbook/.venv/bin/python \
  rocm/bench_sft.py \
  --base-url http://127.0.0.1:8001 \
  --context 64 \
  --warmup-steps 1 \
  --measured-steps 5 \
  --run-id t64-control \
  --output /tmp/qwen35-t64-control.sft.jsonl
```

This benchmark uses deterministic synthetic tokens. It measures the local
HTTP/database/future path plus forward, backward, and Adam; it is throughput
and stability evidence, not SFT quality evidence.

For staged long-context capacity checks, `--inter-step-delay-seconds 5` inserts
an untimed cooling interval between updates. Runs using it remain valid for
per-step latency and peak-memory checks, but not for continuous-duty throughput
or thermal comparisons.

### Fixed-BFC allocator experiment

The alternative `SKYRL_QWEN35_MEMORY_MODE=preallocate85` is default-off. It
forces the BFC allocator, fixed preallocation, `XLA_CLIENT_MEM_FRACTION=0.85`,
and abstract checkpoint construction. The launcher rejects inherited allocator,
fraction, preallocation, or device-selection settings that conflict with that
contract. Allocation and one direct load passed, but a later repeated backend
setup on the prior driver/boot hit an AMDGPU illegal opcode. After the ROCm
7.2.4 upgrade and reboot, bounded setup-only and compile-only gates passed, as
recorded below. This remains an experimental capacity path, not a
production-ready training configuration. Every guarded Qwen3.5 full-model
entrypoint still refuses any boot whose journal contains a fatal AMDGPU event.

The allocation probe defaults to CPU and is safe to run without an accelerator
acknowledgement:

```bash
.venv/bin/python rocm/probe_bfc_preallocation.py --settle-seconds 0
```

The ROCm form makes one 256-byte device transfer. Its dominant effect is the
intentional fixed 85% BFC allocation, so run it alone in a fresh process under
the telemetry guard. Before importing JAX, the probe independently requires an
accessible and unowned `/dev/kfd`, an AMD DRM card, and no connected
non-Writeback AMD display; this probe has no safety override:

```bash
.venv/bin/python rocm/profile_rocm.py \
  --output /tmp/qwen35-bfc85-allocation.telemetry.jsonl \
  --baseline-seconds 2 \
  --timeout 120 \
  --sensor-grace-seconds 60 \
  --max-junction-temp-c 80 \
  --max-vram-gib 22 \
  --min-host-available-gib 8 \
  --max-swap-gib 0.001 \
  -- .venv/bin/python rocm/probe_bfc_preallocation.py \
       --platform rocm --allow-gpu \
       --fraction 0.85 --settle-seconds 5 \
       --output /tmp/qwen35-bfc85-allocation.jsonl
```

On this 25,753,026,560-byte device, the observed allocation gate is
`pool_bytes == bytes_limit == 21,892,169,728` B (20.3887 GiB). This is one
2 MiB granule above the raw 85% sysfs estimate: pinned OpenXLA deliberately
[rounds the GPU BFC limit upward to 2 MiB](https://raw.githubusercontent.com/openxla/xla/5a9e73cbd92530cac2ac36f4736a774b2412afe2/xla/pjrt/gpu/gpu_helpers.cc).
Also require physical VRAM below 21.2 GiB, no swap growth from the measured
baseline, no fatal AMDGPU journal event, and return to idle KFD/VRAM after the
probe exits. Any other mismatch stops the experiment; increasing the fraction
is not a substitute for understanding it.

After that passes, advance one fresh process at a time:

1. **Load only:** run `probe_model_residency.py` with
   `--platform rocm --allow-gpu --construction abstract-load
   --allocator-mode preallocate85 --output <private.jsonl>` under the same
   80 C/22 GiB telemetry limits. Require the pinned revision, exact
   8,480,538,476-byte unique state, no duplicate base buffers, at least 0.5 GiB
   beyond the measured 10.56 GiB arena request, and a clean shutdown.
2. **Compile only:** run the reviewed `probe_sft_compile.py` control described
   in [`SFT_COMPILE_PROBE.md`](SFT_COMPILE_PROBE.md). It lowers and compiles the
   exact context-2,048 training executable but never invokes the returned full
   model-pass callable or optimizer step. XLA may run bounded autotuning kernels,
   so the same guard remains mandatory. Re-read allocator state and require
   `pool_bytes - bytes_in_use >= compiled_temp_bytes + 0.5 GiB`.
3. **One update:** issue exactly one context-2,048 forward/backward/Adam request,
   then unload immediately and verify telemetry, the driver journal, KFD, and
   idle VRAM. The steady-state `bench_sft.py` harness intentionally requires six
   updates and must not be used as a substitute for this gate.
4. **Cooled validation:** only after the one-update gate passes, run the normal
   one-warmup/five-measured protocol with
   `--inter-step-delay-seconds 5`.

On ROCm 7.2.4, the staged full-model setup and compilation gates passed in
fresh guarded processes. Context-64 setup-only took 96.173134 s and stopped
before lowering. The context-64 XLA compile took 39.642890/6.228467/52.449389 s
for setup/lowering/compilation with 196,247,088 B compiled temporary memory.
The context-2,048 Pallas compile took 39.177387/6.350830/94.074129 s with
11,390,298,880 B compiled temporary memory. Its physical VRAM peak was
23,003,750,400 B, maximum junction temperature was 79 C, and swap remained
zero. The returned callable and optimizer were not invoked:
`model_pass_executable_invocations == optimizer_step_invocations == 0`.
Exact artifacts are listed in [`RESULTS.md`](RESULTS.md), including
`/tmp/postrocm-backend-setup-1783872188.jsonl`,
`/tmp/postrocm-compile-t64-1783872326.jsonl`, and
`/tmp/postrocm-compile-t2048-pallas-1783872544.jsonl` with their corresponding
`.telemetry.jsonl` and `.telemetry.jsonl.summary.json` files.

The independently reviewed exact-one-update protocol is available through the
same client. It rejects explicit warmup/measured counts, performs exactly one
cold forward/backward/Adam update, records cleanup success or failure, unloads,
and emits no steady-state throughput claim:

```bash
TINKER_API_KEY=tml-dummy ../tinker-cookbook/.venv/bin/python \
  rocm/bench_sft.py \
  --base-url http://127.0.0.1:8001 \
  --context 64 \
  --one-update-gate \
  --run-id one-update-t64 \
  --output /tmp/one-update-t64.sft.jsonl
```

Its fail-closed protocol, cleanup paths, direct-script invocation, and normal
1+5 compatibility passed CPU/mocked review. The first hardware gate then passed
at context 64 on ROCm 7.2.4: exactly one update completed, cleanup unloaded the
adapter, and the device returned to idle without a fatal driver event. This
validates the protocol and the short-context post-upgrade model path, not
steady-state throughput. Context-2,048 execution remains blocked until Pallas
is numerically qualified: isolated `dq`/`dk` relative-L2 error is about 1.1%,
above the 1% promotion threshold. Exact results are in [`RESULTS.md`](RESULTS.md).

The fixed-rollout GRPO learner harness follows the Cookbook's causal shift,
group-mean advantage, mask-removal, `importance_sampling`, and Adam call order
without performing sampling or grading. Its smallest nondegenerate control is:

```bash
TINKER_API_KEY=tml-dummy ../tinker-cookbook/.venv/bin/python \
  rocm/bench_grpo.py \
  --base-url http://127.0.0.1:8001 \
  --context 64 \
  --completion-tokens 16 \
  --group-size 2 \
  --warmup-steps 1 \
  --measured-steps 5 \
  --run-id grpo-t64-g2 \
  --output /tmp/grpo-t64-g2.jsonl
```

The rollouts, old log-probabilities, rewards, and advantages are deterministic
synthetic inputs. This isolates learner execution; it is not rollout-quality,
reward, KL, or policy-improvement evidence.

Use the telemetry wrapper for every GPU experiment. It writes private JSONL
and summary files, watches fatal AMDGPU journal events once per second, and
terminates wrapped/deliberately included process trees on configured limits:

```bash
server_pid="$(fuser 8001/tcp 2>/dev/null | xargs)"
TINKER_API_KEY=tml-dummy .venv/bin/python rocm/profile_rocm.py \
  --output /tmp/qwen35-profile.jsonl \
  --include-pid "server=$server_pid" \
  --terminate-included-on-safety \
  --sensor-grace-seconds 60 \
  --max-junction-temp-c 90 \
  --max-gpu-power-watts 315 \
  --max-vram-gib 23 \
  --min-host-available-gib 4 \
  --timeout 900 \
  -- ../tinker-cookbook/.venv/bin/python rocm/bench_sft.py \
       --context 64 --warmup-steps 1 --measured-steps 5 \
       --run-id profiled-t64 --output /tmp/profiled-t64.sft.jsonl
```

The explicit 60-second sensor grace is needed after the headless AMD GPU has
runtime-suspended: its hwmon files can remain temporarily unreadable while the
first ROCm context starts. Safety limits still apply as soon as the sensors
become readable. A successful wrapped command or attach-only measurement is
also rejected if any configured junction-temperature or power sensor produced
no measured value at all, so a short run cannot silently finish entirely inside
the grace window. See [`RESULTS.md`](RESULTS.md) for the exact validated runs
and the context buckets that remain untested.

Large-context runs may use the full practical VRAM budget and explicit
CPU/RAM/disk offload. Keep offload buffers and transfer timing observable;
uncontrolled swap is not a substitute for a deliberate offload path. Smaller
isolated probes retain tighter workload-specific limits so an unexpected
allocation or power excursion fails early.

The exact optimizer replay probe defaults to CPU. GPU use requires two
explicit acknowledgements and should only be run in a fresh process with the
profiler:

```bash
.venv/bin/python rocm/probe_jax_optimizer.py \
  --platform gpu --allow-gpu --command-buffer-mode disable --steps 3
```

The exact post-upgrade replay passed all three updates for 402 LoRA leaves and
34,512,896 BF16 parameter elements. Lowering took 3.266979 s, the cold
compile/update 16.688593 s, and the two ordinary replays 0.077036/0.105031 s;
all state and sentinel checks were finite and every checksum changed. These are
isolated optimizer timings, not full-model SFT throughput. Artifacts are
`/tmp/postrocm-opt-1783872102.probe.jsonl`,
`/tmp/postrocm-opt-1783872102.telemetry.jsonl`, and
`/tmp/postrocm-opt-1783872102.telemetry.jsonl.summary.json`.

ROCm causal self-attention at 512 tokens or longer cannot silently use the
quadratic XLA fallback. The currently validated Pallas geometry is opt-in with
`SKYRL_ROCM_PALLAS_ATTENTION=1` and hard-capped at 16,384 tokens. Inputs above
16K remain refused until query-bounded forward and backward kernels replace the
monolithic launch. Do not bypass this cap to attempt 32K.

The fused-stage, GDN/FlashQLA adaptation, native-GQA, tied-head, W8/W4/A8/A4,
activation-checkpoint, and custom-VJP design is in
[`MEGAKERNELS.md`](MEGAKERNELS.md). The decision not to replace the learner
with EGGROLL, the exact state/speed opportunity, and the source-level FlashQLA
portability audit are in
[`ES_FLASHQLA_FEASIBILITY.md`](ES_FLASHQLA_FEASIBILITY.md). The three bounded
stage prototypes have separate promotion records:
[`GDN_SUPERBLOCK.md`](GDN_SUPERBLOCK.md),
[`QUERY_BOUNDED_GQA.md`](QUERY_BOUNDED_GQA.md), and
[`TIED_LOGPROB_PROTOTYPE.md`](TIED_LOGPROB_PROTOTYPE.md).

All of those implementations remain unwired. The quantized LoRA implementation
in `skyrl/tx/kernels/quantized_lora.py` is a CPU semantic oracle only; it is not
selected by model code and is not a GPU performance path. The native gfx1100
IU8/IU4 compile proof and production FFI requirements are in
[`QUANTIZED_FFI.md`](QUANTIZED_FFI.md). Likewise,
`skyrl/tx/kernels/qwen3_5_qkv_lora.py` is an unwired equation/precision
experiment; its portable custom VJP is explicitly not a memory or speed path.
