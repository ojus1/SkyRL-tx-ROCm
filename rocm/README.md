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

The launcher is intended to trade startup and disk for potential repeat-startup
savings. It pins a private cache namespace to JAX/JAXlib/ROCm plugin/PJRT
0.10.2, ROCm 7.2.4,
AMDGPU 6.16.13, and gfx1100; rejects an inherited cache path; and stores every
eligible executable plus per-fusion autotuning data. Serialized executable
entries use a 16 GiB JAX LRU. The separately unbounded autotune subtree is
scanned for unsafe objects and must remain at or below 4 GiB at each startup.
Persistent-cache read/write errors are fatal rather than downgraded to warnings.
The cache path and all ancestors are owner/mode/symlink validated because JAX
treats cache entries as trusted executable content. This mechanism does not
enable XLA command buffers or HIP Graphs, does not reduce steady-state VRAM by
itself, and does not replace shape-specific warmup. It is designed to let later
static-bucket precompile and ordinary host-driven warmups amortize compilation;
each additional hit, bucket, and warmup path requires separate qualification.
The active trusted namespace
`~/.cache/skyrl-jax-rocm-private-v1/jax0.10.2-jaxlib0.10.2-rocm-plugin0.10.2-pjrt0.10.2-rocm7.2.4-amdgpu6.16.13-gfx1100-v1`
was initially empty. The legacy 42 MiB `~/.cache/skyrl-jax` tree is outside this
namespace and is neither read nor credited as startup evidence.

Static-bucket prewarm is available but default-off. A nonempty canonical bucket
list runs one compile-only process before the API starts and records a private
`prewarm.jsonl` in the new run directory:

```bash
# Default-off, launcher-supervised compile-only prewarm.
SKYRL_QWEN35_PREWARM_BUCKETS=64,256 ./rocm/start_qwen35.sh prewarm-t64-t256

# Also compile the one sequence-independent Adam update after both train buckets.
SKYRL_QWEN35_PREWARM_BUCKETS=64,256 \
SKYRL_QWEN35_PREWARM_OPTIMIZER=1 \
  ./rocm/start_qwen35.sh prewarm-t64-t256-adam
```

Buckets at 512 or above additionally require the explicitly enabled Pallas
attention path. The tool constructs the exact backend and rank-8 LoRA/Adam
state, then lowers and compiles forward/backward/accumulation for batch one. It
does not invoke those executables, precompile the sampler, export an executable,
or replace data-dependent warmup. The optimizer extension is separately
default-off: the launcher accepts only literal `0` or `1`, and `1` requires a
nonempty bucket list. After all train buckets pass, it calls `lower().compile()`
once on the backend's exact `_compute_grads_and_update` JIT using accumulated
gradients, LoRA state, the live NNX Adam object, and the scalar adapter index.
This operation is independent of sequence length. The compiled handle is only
inspected for memory metadata and then discarded; it is never called, so no
Adam update, gradient reset, or model-state mutation through an optimizer step
occurs.

Backend/model construction, pinned-weight loading, LoRA parameter and Adam-state
initialization, array placement, and the explicit `block_until_ready` may still
perform ordinary setup array work. Compilation can initialize ROCm, allocate
representative buffers, and run XLA autotuning kernels, so allow a much longer
startup. The launcher now wraps this child in `profile_rocm.py`, with a default
3600-second timeout configurable only from 600 through 14400 seconds, a 90 C
junction limit, 315 W power limit, 24 GiB VRAM limit, no positive host-RAM
floor, and an 8 GiB swap limit. It preserves only the already-held global lock
descriptor, records `prewarm.telemetry.jsonl` plus its private summary, and
captures an exact idle handoff baseline before the child. On success, failure,
timeout, `INT`, or `TERM`, the launcher reaps the profiler, requires three
consecutive one-second samples with the exact PCI/DRM/KFD identity unowned and
VRAM/GTT no higher than baseline, and then performs the final boot-journal gate
before it can start the API. Neither path enables command buffers, HIP Graphs,
PGLE, executable export, an actual warmup/replay, or an in-engine prewarm. The
launcher exports both `JAX_ENABLE_PGLE=false` and
`JAX_COMPILATION_CACHE_EXPECT_PGLE=false`, and the direct prewarm tool requires
those exact values. Running
`python rocm/prewarm_qwen35_buckets.py` without ROCm acknowledgements is a
CPU-only plan and never imports JAX.

The exact cold T64-plus-Adam direct-tool population and one matched compile-only
cache-hit process described below are now hardware-qualified. The multi-bucket
path, launcher-to-API transition, additional/repeated hits, and actual warmup
remain unqualified. The launcher examples above provide the required telemetry,
headless-display, exclusive-KFD, fatal-journal, and cleanup gates; any direct
tool invocation still requires equivalent external supervision. The API's
current health endpoint does not prove that the nested engine has finished
loading, inherited the intended cache policy, or consumed a particular cached
executable. Therefore prewarm remains default-off until an engine readiness and
startup-attestation contract is added and separately qualified. Each
train-bucket and optimizer compile has separate timing, compiled-memory,
postflight, counter, and cache-evidence records. They report JAX 0.10.2's public
process-level persistent-cache hit/miss events, numeric `compile_time_saved_sec`
and `cache_retrieval_time_sec` events on hits, and top-level cache-directory
deltas. Those public callbacks contain no module or cache key, so this is useful
process-level evidence but not per-key proof of a cache write or hit. Startup
accepts only one monitored hit with exactly one finite nonnegative value for
each duration and no top-level executable-cache mutation, or one monitored miss
paired with exactly one newly added top-level executable-cache entry, no changed
or removed entry, and no duration. Unexpected metadata, malformed/duplicate
durations, cache additions or changes during a hit, changed-only or multiple-add
miss deltas, cache removals, and otherwise ambiguous, missing, or mixed evidence
fail closed.

### ROCm 7.2.4 cold T64 plus Adam cache-population result

Using the audited relevant sources from commit
`a414cec194ad7d724d15fbef4828527fc7f4d4c0` on boot
`54ccf56c-5f4f-4ef7-ac98-c13e0587b5b9`, a separate pre-run filesystem check
found the versioned trusted namespace empty before process start. The first
output path used `/tmp` directly and was refused before `_execute`, JAX import,
or GPU access because the child artifact requires a private mode-`0700` parent.
That validation precursor returned 2 in
2.009 s, created no child artifact, and had zero measured VRAM/GTT or write-byte
delta. Its profiler telemetry and summary SHA-256 values are
`82215b202fc8b87d2df0f57004bc199c80b49f4ba038e14234efa077f39ef169` and
`5c0eb79185ccc0314908d5c86ce525c681ed2991c6446465a9197edc8ea4f206`.
The refusal cause is reconstructed from the pinned source and attempted path;
the redacted telemetry does not embed stderr.

The corrected fresh process used a private directory and compiled one T64 XLA
forward/backward/accumulation executable followed by the sequence-independent
Adam executable. Backend/model setup took `39.588969 s`. It legitimately
populated 185 small top-level setup entries totaling 879,185 bytes before the
two monitored targets. These entries came from backend/model construction,
pinned-weight loading, LoRA and Adam initialization, array placement, and
synchronization; they are not hidden train or optimizer-step invocations.

The T64 target lowered in `6.398382 s` and compiled in `48.291319 s`.
`CompiledMemoryStats` reported 8,549,564,220 argument bytes, 69,029,424 output
bytes, 196,230,704 temporary bytes, and 69,025,800 alias bytes. Its one public
cache request was one strict miss with no hit or duration event, adding exactly
one 3,834,842-byte executable entry and changing or removing none. That entry's
SHA-256 is
`4bc6a2472759483ea593737d5ee5f04bc982d3179c25dccd0e30ee4697bf1877`.

Adam then lowered in `3.028201 s` and compiled in `17.290585 s`. Compiler memory
was 276,103,204 argument bytes, 345,145,162 output bytes, 120,360 temporary
bytes, and zero aliases. Its separately monitored request was also exactly one
strict miss with no hit or duration event, adding one 880,688-byte entry and
changing or removing none. Its SHA-256 is
`4880755376c24b921b4db1f34ef1ce5ec589ad88e990db898adb388b13d2199a`.
Both targets recorded exactly one lower and compile and zero model-pass or
optimizer-step executable invocations. Per-target and final AMDGPU postflights
were clean.

The final namespace contained 187 executable entries totaling 5,594,715 bytes,
187 atime files totaling 1,496 bytes, and 365 per-fusion autotune textprotos
totaling 47,775 bytes. The complete 739-file payload was 5,643,986 bytes with no
symlink or special object; autotune content remained far below its 4 GiB
ceiling. The artifact's final executable-cache manifest SHA-256 was
`35a04f7f55949e110f5ba93fd276464576bc9fa464836c40b415ff891b68cafa`.
JAX's callbacks remain process-level evidence and do not cryptographically bind
the observed event to a filename.

The profiler returned zero with no signal in `143.439143 s`. Across 1,389
measured samples, peak physical VRAM was 17,929,101,312 bytes (69.62% of the
card), junction temperature 72 C, board power 235 W, minimum host-available
memory 55,799,439,360 bytes, and swap zero. Sensors were unavailable for the
first 101 measured samples but recovered at 12.225 s within the 60 s grace.
The process exited and the final postflight was clean. Final sampled VRAM was
519,909,376 bytes versus a 27,947,008-byte baseline, so full return to baseline
was not established before the profiler stopped.

The independently audited mode-`0600` evidence under the private mode-`0700`
directory is:

- `/tmp/skyrl-qwen35-prewarm-a414cec1/qwen35-prewarm-t64-adam-cold-boot54ccf56c-run1.jsonl`:
  `11882a98a89475c65ef3c97f56acefa5c8607a9d31e05669c3145d528739fe66`;
- its telemetry:
  `4708bc44470f55cc4a847830d8751caa2641f3c45914379555d57a069d727004`;
- its summary:
  `45144fde088d53b4f3a16f635979f313536bf1fd63427f84f60c617c19ee59cd`.

Command-buffer disabling is explicit in the child. PGLE disabling is indirect
artifact-and-pinned-source evidence: reaching `backend_ready` required the
pinned validator to accept both PGLE variables as `false`. The profiler records
its own pinned script hash, but the child schema does not yet embed the prewarm
script hash or Git HEAD, so source identity was reconstructed externally from
the audited relevant files matching that commit and the run-directory name.
This run proves cold compile-only population and its resource cost. It does not
measure a cache hit or startup time saved, invoke a training pass or Adam
update, validate optimizer-step correctness, authorize a warmup or replay, or
establish a steady-state speed or memory gain.

### ROCm 7.2.4 matched T64 plus Adam cache-hit result

One separately authorized fresh process repeated the exact T64-plus-Adam
compile-only path against the unchanged populated cache. HEAD `292387b1`
changed only documentation relative to the cold run; prewarm, launcher,
backend, cache helper, profiler, and test bytes were unchanged. Before the
process, the executable manifest still contained 187 entries totaling
5,594,715 bytes with SHA-256
`35a04f7f55949e110f5ba93fd276464576bc9fa464836c40b415ff891b68cafa`.
The T64 and Adam executable hashes remained `4bc6a2472759483ea593737d5ee5f04bc982d3179c25dccd0e30ee4697bf1877`
and `4880755376c24b921b4db1f34ef1ce5ec589ad88e990db898adb388b13d2199a`.

Both monitored targets passed the strict hit contract independently: one cache
request, one hit, zero misses, exactly one finite nonnegative saved-time and
retrieval-time event, no schema issue, and no executable-cache addition,
change, or removal. Entry count, byte count, and manifest stayed exact before
and after each target. All 365 autotune files totaling 47,775 bytes retained
their pre-hit mtimes. Exactly 185 setup atime companions were refreshed before
`backend_ready`, followed by the T64 and Adam atimes in their target windows;
these are expected cache-access bookkeeping writes, not executable or autotune
content mutations and not target timing evidence.

The paired cold-to-hit compile results were:

| Target | Cold compile | Cache-hit compile | Reduction | Speedup |
|---|---:|---:|---:|---:|
| T64 forward/backward/accumulate | `48.291319 s` | `4.720658 s` | `90.22%` | `10.23x` |
| Adam update | `17.290585 s` | `1.628718 s` | `90.58%` | `10.62x` |
| Combined target compilation | `65.581903 s` | `6.349376 s` | `90.32%` | `10.33x` |

Including lowering, the two targets fell from `75.008486 s` to `15.466490 s`,
a 79.38% reduction or 4.85x speedup. Matched profiler elapsed time fell from
`143.439143 s` to `75.865997 s`, saving `67.573147 s`: a 47.11% reduction or
1.89x paired profiled direct-tool speedup. Backend/model setup itself changed from
`39.588969 s` to `32.429409 s`; this 18.08% observed reduction was outside the
target listeners, so the setup atime reads support cache-access attribution but
not a public per-target duration claim.

JAX's callbacks separately reported `41.386720 s` saved and `4.613280 s`
retrieval for T64, plus `14.470795 s` saved and `1.529205 s` retrieval for Adam.
The combined callback values are `55.857515 s` saved and `6.142485 s`
retrieval. Those internal estimates use different accounting and must not be
equated with the paired wall-time difference.

Compiled argument, output, temporary, and alias memory fields matched the cold
run. Deserialized handles reported generated-code sizes 16 bytes larger for
each target, so the complete memory-analysis records were not byte-identical.
Peak physical VRAM was 17,900,191,744 bytes versus 17,929,101,312 bytes cold,
only 0.16% lower and not a demonstrated memory saving. Peak process RSS fell
from 5,759,397,888 to 4,211,974,144 bytes (26.87%), and command-tree writes fell
from 54,505,472 to 1,134,592 bytes (97.92%). These are startup host-resource
observations, not steady-state training measurements.

The profiler completed with return zero and no signal. Across 708 measured
samples, peak junction temperature was 55 C, peak board power 168 W, minimum
host-available memory 56,863,875,072 bytes, and swap zero. Per-target and final
AMDGPU postflights were clean. Telemetry stopped shortly after process exit at
520,364,032 bytes VRAM, but a bounded independent settle check later found the
process gone, KFD unowned, the card suspended, no fatal journal event, and VRAM
exactly back at its 27,947,008-byte baseline.

The independently audited mode-`0600` artifacts under a fresh private
mode-`0700` directory are:

- `/tmp/skyrl-qwen35-prewarm-hit-292387b1/qwen35-prewarm-t64-adam-hit-boot54ccf56c-run1.jsonl`:
  `ae600260a0d949e5988f0ae2b913ec026ff3e6f76d71b461465aa168e08acdab`;
- its telemetry:
  `1674af56660ccc6fcecb44a21e2623d7d079076630e194b2fc2533c9cc9415e1`;
- its summary:
  `cc900f388ecd50ce81e8673eea0f6af46a35cd743dae0bafcf455ca580df8092`.

This is one paired compile-only result on the exact pinned stack, source, and
cache. Public callbacks remain process-level rather than per-key proof. It does
not establish repeatability, steady-state training throughput, optimizer-update
speed, sampler performance, another bucket, actual warmup safety, or a runtime
VRAM gain. No additional hit, warmup, or compiled-call invocation is authorized
by this result.

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
