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

# Run the same gates, then exit before the hardware-unqualified API transition.
SKYRL_QWEN35_PREWARM_BUCKETS=512 \
SKYRL_QWEN35_PREWARM_OPTIMIZER=1 \
SKYRL_QWEN35_PREWARM_ONLY=1 \
SKYRL_ROCM_PALLAS_ATTENTION=1 \
  ./rocm/start_qwen35.sh qualify-t512-adam
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

`SKYRL_QWEN35_PREWARM_ONLY=1` is also default-off, accepts only literal `0` or
`1`, and requires a nonempty bucket list. It does not weaken or skip any
prewarm, handoff, or final-journal gate; after all of them pass, it disarms the
launcher traps and exits zero before the API command. This is the qualification
route while source-aligned engine startup and cache consumption remain
hardware-unqualified.

Every operational launcher prewarm also requires a clean Git worktree before
the first hardware or JAX access. Fixed, sanitized Git commands bind the exact
commit and raw launcher/bootstrap/prewarm blobs to both their Git object IDs and
SHA-256 digests. Source-cache preparation feeds the exact content-addressed
bootstrap HEAD blob from sanitized Git directly to isolated Python over standard
input; it does not reopen mutable working-tree Python by pathname. That
stdlib-only bootstrap creates or revalidates the entire tracked HEAD tree at the
fixed private path
`$ACCOUNT_HOME/.cache/skyrl-source-snapshots-private-v1/$GIT_HEAD/source-head`.
The absolute path depends on the commit, never the run ID; this preserves Python
`__file__`/code filenames and therefore JAX executable-cache keys across fresh
processes. Every ROCm helper then starts from that normalized snapshot under
Python `-I -S -B -P` with an empty per-run private bytecode-cache prefix and an
allowlisted environment. For those Python helpers and module imports, this
prevents executable `.pth`, `sitecustomize`, user-site, stale bytecode, ignored
import shadows, and mutable working-tree modules from running before the full
snapshot is checked. The launcher itself necessarily
begins as working-tree shell code; it uses `/bin/bash -p`, no pre-gate external
path lookup, and an exact HEAD-blob check rather than claiming snapshot
execution. A tracked regular `rocm` package plus explicit package-search and
target-origin checks prevent mutable site-packages from replacing an allowlisted
module. The child records the snapshot-manifest hash in its first manifest,
revalidates the same tree after compilation, and refuses `complete` if source or
Git state changed.
Operational direct execution without the verified launcher lock is rejected;
the unprivileged CPU planning mode remains directly usable. The launcher removes
all helper-only claims before API exec. It then supplies a separate minimal set
of runtime claims through a sanitized `env -i`; API and engine accept those
claims only after independently revalidating the Git/archive/full-tree source,
exact accelerator environment, memory mode, cache namespace, cwd, and origins.
The transition also pins `uv 0.11.8` to SHA-256
`646adf5cf12ba17d1a41fa77c8dd6496f73651dcfeeed6b5f4ec019b36bc7153`.
All guarded entrypoints use the single fixed
`/run/user/$UID/skyrl-qwen35-rocm-$UID` lock namespace rather than inherited
`XDG_RUNTIME_DIR`; the API validates and explicitly passes that already-locked
directory descriptor through nested `uv` to the engine, so an orphan engine
continues to exclude a second guarded GPU process.

This is an operational integrity boundary, not protection against a malicious
same-UID process, root/kernel compromise, or compromised Git/Python binaries.
The parent process and dynamic-loader environment are also outside the boundary:
`LD_PRELOAD`/`LD_AUDIT` can act while `/bin/bash` is loading, before the launcher
can reject those inherited variables.
The explicitly added virtualenv `site-packages` directory supplies an
exact-version-checked JAX stack, but its dependency file bytes are not yet
cryptographically attested. Verified Python helper startup/execution in the
prewarm-only path uses isolated Python. The later API and nested engine use the
normal virtualenv interpreter under fixed `uv --active --no-sync`, not the same
isolated bootstrap; their source tree and runtime policy are revalidated, while
third-party dependency bytes remain outside the attestation boundary.

Initial archive creation/extraction and repeated full-tree Git-blob,
deterministic-archive, and SHA-256 validation deliberately spend extra startup
CPU, wall time, RAM, and disk. Commit `5fb2b220` measured initial preparation at
1.59 seconds and reuse validation at 1.18 seconds; the additional API/engine
revalidation overhead in this new transition has not yet been hardware-run or
timed end to end.
The tar archive and extracted snapshot consume roughly twice the tracked-tree
bytes once per retained commit, rather than once per run. Existing complete
entries are validation-only and are never repaired or rewritten; partial,
altered, symlinked, hardlinked, foreign-owned, or incorrectly permissioned
entries fail closed before helper or hardware access.

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
`JAX_COMPILATION_CACHE_EXPECT_PGLE=false`, and the operational child requires
those exact values. Running
`python rocm/prewarm_qwen35_buckets.py` without ROCm acknowledgements is a
CPU-only plan and never imports JAX.

The exact cold T64-plus-Adam direct-tool pair, supervised T512 Pallas pair, and
commit-keyed T1024 Pallas pair described below are now hardware-qualified. The
launcher-to-API/nested-engine source-path alignment is implemented and
CPU-qualified, but engine cache consumption, the multi-bucket path, additional
repeated hits, and actual warmup remain hardware-unqualified. The launcher
examples above provide the
required telemetry, headless-display, exclusive-KFD, fatal-journal, and cleanup
gates. The historical T64 direct-tool artifacts used equivalent external
supervision; current operational ROCm prewarm is launcher-only. The API now
waits for an exact launch-ID/PID/start-tick/boot/source/lock-bound engine row,
published only after backend construction, before it serves requests. Its
health endpoint revalidates that row, a one-second engine watchdog heartbeat,
the isolated API-spawned process group, and all three live process identities.
Shutdown signals the complete group immediately and does not report STOPPED
until the wrapper is reaped and the group is absent. This readiness protocol is
bounded to 3600 seconds in both Qwen launcher branches. It is CPU-qualified but
has not yet been exercised with the real ROCm backend, and it
still does not prove consumption of a particular cached executable. Therefore
prewarm remains default-off until the in-engine cache-evidence contract is
implemented and separately qualified. The containment result is specifically
for the pinned non-Ray, single-process JAX path: a POSIX process group is not a
cgroup, so Ray/FSDP/Megatron workers, daemonized descendants, or any child that
calls `setsid()` require separate containment and qualification. Each
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
its own pinned script hash, but that historical child schema did not embed the
prewarm script hash or Git HEAD, so source identity was reconstructed externally
from the audited relevant files matching that commit and the run-directory
name.
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

### ROCm 7.2.4 T512 Pallas population and matched cache-hit result

Two distinct supervised prewarm runs, correlated to commit
`ab962c799caf09bd85b7454cca46413d6deb88c3` but not directly Git-attested by
their artifacts, qualified the next training bucket on boot
`54ccf56c-5f4f-4ef7-ac98-c13e0587b5b9`. Both used growth/eager model
construction, batch one, rank-8 LoRA, 64-token loss chunks, Pallas attention,
the exact sole `XLA_FLAGS=--xla_gpu_enable_command_buffer=`, and the optional
Adam compile under a 7,200-second watchdog. Operationally,
`SKYRL_QWEN35_PREWARM_ONLY=1` ran completed telemetry supervision plus the
reap/settle, identity, and final-journal gates and exited before the API. The
launcher mode and command are not embedded; this pre-API exit is reconstructed
from the artifact set and byte-matching launcher. No compiled model pass or
optimizer step was invoked.

This was not a fully cold T512-plus-Adam pair. The fixed trusted namespace
already contained 187 entries totaling 5,594,715 bytes from the qualified
T64-plus-Adam work, including the sequence-independent Adam executable. The
first process therefore truthfully produced one strict T512 miss and one strict
Adam hit. The T512 request added exactly one 5,969,258-byte executable and no
change or removal. Its filename is
`jit_forward_backward_and_accumulate-1caaece3bcfd971ef407426aa8e52a1d135776a45776ad3bef8c1f2f852fb5b6-cache`
and the still-present file currently has SHA-256
`ac7a95686357008047971717f68cd5dcb95b10bb08348b0a4ff18867559524c8`.
The resulting 188-entry namespace totaled 11,563,973 bytes with manifest
SHA-256 `a6847015bd5aee2a23d5f4da26cd511c773dffc9a950b4949188350aa2a65904`.
The content hash was measured after the run; the embedded manifest binds sorted
filename, size, and mtime tuples rather than executable content bytes.

T512 lowered in `6.939800 s` and compiled in `93.328001 s`. Compiler memory
analysis reported 8,549,574,972 argument bytes, 69,032,112 output bytes,
2,940,601,600 temporary bytes, and 69,025,800 alias bytes. The existing Adam
hit lowered in `2.808095 s` and compiled in `1.738305 s`; its public callback
reported `14.359669 s` saved and `1.640331 s` retrieval. Backend/model setup took
`32.287817 s`.

The matched second process accepted one strict hit for each target: exactly one
cache request and hit, zero misses, one finite nonnegative saved-time and
retrieval-time value, no schema issue, and no executable-cache addition,
change, or removal. Entry count, byte count, and manifest remained exact. The
paired target results were:

| Target | First process | Matched hit | Reduction | Speedup |
|---|---:|---:|---:|---:|
| T512 forward/backward/accumulate compile | `93.328001 s` | `5.617719 s` | `93.98%` | `16.61x` |
| Existing Adam-hit compile | `1.738305 s` | `1.609458 s` | `7.41%` | `1.08x` |
| Combined target compile | `95.066305 s` | `7.227177 s` | `92.40%` | `13.15x` |
| Both targets including lowering | `104.814200 s` | `16.931410 s` | `83.85%` | `6.19x` |
| Entire profiled child | `169.407536 s` | `80.572651 s` | `52.44%` | `2.10x` |

Only T512 is a miss-to-hit comparison. Adam is hit-to-hit variation;
consequently, the combined rows blend that variation with the T512 cache effect
and are descriptive paired-process reductions, not independent Adam cache-gain
measurements.

JAX separately reported `84.490213 s` saved with `5.509787 s` retrieval for
T512, and `14.484431 s` saved with `1.515569 s` retrieval for Adam. These
process-level callbacks do not name a module or key and use different internal
accounting from paired wall time. T512 argument, output, temporary, and alias
memory fields matched across processes; the deserialized handle reported a
16-byte larger generated-code field. Backend setup was also effectively flat
at `32.486707 s` in the hit process.

The first/hit profiler maxima were respectively 18,151,886,848 and
17,901,137,920 bytes VRAM, 6,114,643,968 and 4,372,545,536 bytes command-tree RSS,
69/64 C junction temperature, 199/192 W board power, and zero swap. Thus the hit
observed 0.234 GiB lower peak VRAM, 1.622 GiB lower peak RSS, and 98.40% fewer
command-tree write bytes. These are compile-startup observations, not
steady-state training-memory gains. Each post-child handoff independently
returned in about 8.8 seconds to the exact 27,947,008-byte VRAM and
15,966,208-byte GTT baselines with the same PCI/DRM/KFD identity, KFD and render
unowned, runtime suspended, and a clean journal.

The private mode-`0600` artifacts under mode-`0700` run directories are:

- population child, handoff, telemetry, and summary:
  `425624d66934a3389b925066ad790b06066c929f86222778253fd10bec111ddb`,
  `2ddbed0165ac44668653d9db92b19444787ff5a266146f34dcbcf74c1c794bf8`,
  `fbec617df8cbc2129dde84ee8efabb5a51d6f770c07177bece1242a97316e468`,
  and `768eb6d9b7285338eb41250cea1fa99743853ca3e026576b7709a7574cb0735e`
  under `/tmp/skyrl-qwen35-runs/t512-adam-populate-ab962c79-r1`;
- matched-hit child, handoff, telemetry, and summary:
  `7c216e1956b2b58c45c9e5077d45b0b0dea4ff55bbdfd2a7ed1a10259f0d327e`,
  `7937510fe199b3df039f4cb46d12044c8bad7112909046e1d9968d4a20712d42`,
  `a40902933b6a7d50e6fe436978cbc054b36a0d854c6c47758c30ed4d6da45d64`,
  and `d017fcf0525361a9a29097ca3ecc8e7432718e334048f443fb4ed6cf61f8715a`
  under `/tmp/skyrl-qwen35-runs/t512-adam-hit-ab962c79-r1`.

Both telemetry manifests embed and exactly match profiler SHA-256
`ed230758101a2a540b3a09e7f84ac92256d2bb41c70dbc399b9466fe0b979684`;
both handoff baselines embed and exactly match helper SHA-256
`4d6c7e665219ce125d840e68b0e2cb7e8b1b5f98552ff65a2d07a153b3cd1392`.
Those historical child artifacts did not embed the prewarm-tool, launcher, or
Git hashes. Their correlation to commit `ab962c79` was reconstructed at audit
time from runtime source bytes matching HEAD, observed pre/post-run Git checks,
commit/run timestamps, and run-directory names rather than direct runtime
attestation.
At that audit, the prewarm tool and launcher matched SHA-256
`0ce082bbed81ee9013cae036c0619c3e2de5d15b0ed1efce5fcc7958fb303d7c`
and `118d1ed67a1dd3662d2b43aad86e28ef0bba73d141e07e51ef3a1b0540fcb8f0`,
and the README was the sole tracked modification. Those external checks are not
embedded proof that the worktree stayed clean during execution.

This qualifies one cold T512 cache population and one matched compile-only hit;
it is not a repeated-sample timing distribution. It does not prove per-key
callback attribution, execute T512 training, establish steady-state throughput
or memory, validate an engine transition, or authorize graph capture/replay.

### ROCm 7.2.4 commit-keyed T1024 population and matched hit

Commit `5fb2b220bfbf202ac2b9295efb9e9a072cc00135`, tree
`4c560736eefd6395baedff692828db63ba86ce4d`, qualified the first exact
same-source-path T1024 pair. Both launcher-supervised fresh processes used the
same private snapshot at
`$ACCOUNT_HOME/.cache/skyrl-source-snapshots-private-v1/$GIT_HEAD/source-head`.
Independent reconstruction matched all 1,090 Git blobs, 239 directories, and
35,333,482 source bytes; the source-manifest SHA-256 was
`5179a0e25214a50ee8cb74b7793323fd0189759c8de9b63d23ec1e727caf23f0`.
The retained deterministic archive was 36,290,560 bytes with SHA-256
`3222c78008e0e6bcf23ca034a96f4e7c075567e2a53381639598c48594f6547b`.

Separate CPU-only preparation created that archive/snapshot in `1.59 s` with
141,588 KiB peak RSS, then revalidated it in `1.18 s` with 141,028 KiB peak RSS.
The pair consumes 68.306 MiB per retained commit. Reuse changed none of the
1,332 file/directory inode, mode, link, size, mtime, ctime, or content-hash
records; the per-run empty bytecode-cache directory remained outside the stable
snapshot.

The first GPU process began with 190 top-level executable entries totaling
23,553,082 bytes. T1024 reported one strict public-monitoring miss and added
exactly one 6,002,623-byte entry, changing or removing none:
`jit_forward_backward_and_accumulate-f2c446a981cb5237c15fe0550221ed6b046c5dff8829de56d9a5cb1045c21fa6-cache`.
Its SHA-256 is
`a5576049fc7109e508c17c43e2f2911ff1150266548774a7b99409c667a51fd4`.
The matched fresh process then reported exactly one hit, zero misses,
`37.202892 s` saved, `6.797108 s` retrieval, and no addition, change, or removal.
The namespace stayed at 191 entries / 29,555,705 bytes with manifest SHA-256
`f8987b8f629512eb89bad4c8a79aff7f86c4b605ed579a41e345e05171311ff4`.
The public callbacks remain process-level and do not expose a request-to-key
mapping; the single-entry delta and unchanged hit manifest are the available
key-level filesystem evidence.

| Matched metric | Population miss | Stable-path hit | Observed change |
|---|---:|---:|---:|
| T1024 compile | `47.361373 s` | `6.897813 s` | `85.44%` less; `6.87x` |
| T1024 lowering | `7.515965 s` | `7.514467 s` | effectively unchanged |
| T1024 lower + compile | `54.877338 s` | `14.412279 s` | `73.74%` less; `3.81x` |
| Backend/model setup | `32.224181 s` | `32.637875 s` | effectively unchanged |
| Entire profiled child | `135.804416 s` | `95.294999 s` | `29.83%` less; `1.43x` |
| Peak command-tree RSS | `5,777,256,448 B` | `4,404,682,752 B` | `1.278 GiB` / `23.76%` less |
| Peak physical VRAM | `17,901,096,960 B` | `17,901,109,248 B` | `+12 KiB`; no capacity gain |

Argument, output, temporary, and alias memory fields matched exactly. As in the
independent T64/T512/Adam cache loads, the deserialized handle's generated-code
field was 16 bytes larger (`2,196` to `2,212` bytes); this is treated as fixed
deserialization-accounting metadata, not byte-identical memory analysis. The
population/hit maxima were 61/63 C junction, 255/192 W sampled board power, and
zero swap. The hit started warmer and had higher mean/p95 power despite its lower
maximum, so this pair does not establish a steady-power reduction.

Both children used the exact sole
`XLA_FLAGS=--xla_gpu_enable_command_buffer=`, recorded graph and command-buffer
use as false, and invoked neither a compiled model pass nor an optimizer step.
Both exact handoffs returned in about 8.3 seconds to suspended runtime,
27,947,008-byte VRAM, 15,966,208-byte GTT, empty KFD/render owner lists, and a
clean whole-boot journal. Private artifact SHA-256 values are:

- population child / handoff / telemetry / summary under
  `/tmp/skyrl-qwen35-runs/t1024-stable-populate-adamhit-5fb2b220-r1`:
  `b684bb92e2d314c40eb226db7d64fb706bfe2b0fce41d4677a3cec55b390cc09`,
  `3d85e55fe02a5f8bd754f9d0473209cfd09f911c52a59a0b6708530bddba52f4`,
  `613147cc6d59def57f7485b231a9a07d929b3b4f8aaa20009a458f494ee9cf76`,
  and `a317fb874cb2f23e29f873d5f90a2bcf9e98ce247f35d05b8b3e8a3fdec5dd05`;
- hit child / handoff / telemetry / summary under
  `/tmp/skyrl-qwen35-runs/t1024-stable-hit-adamhit-5fb2b220-r1`:
  `d46564fb6c68383c30b971f6f0a13f45accc92b6438f5ee6bfab06486f31d0dc`,
  `f81458764be19b927be9a45254de4b42eeadc374510b94a5b5966c7dc0bf21af`,
  `9a0cacd074f3d0530580c846137a0af81a0720bccfb45dc568ba841fe7d946b1`,
  and `8f4b2cc2e0dac0aa4886bee941b51047ef5d6731d10a00bdf823ee97709997a7`.

This is a compile-only prewarm-process result, not a server-ready or training
throughput result. At evidence commit `5fb2b220`, the API and nested engine still
launched through `uv` from the mutable checkout while prewarm imported from the
commit-keyed snapshot. The current launcher closes that source-path gap in code:
its final API uses a sanitized `env -i`, fixed `uv`, explicit snapshot
directory/project, disabled env/config discovery and bytecode, and the nested
engine must reproduce the same exact prefix. Both processes revalidate the Git
HEAD, deterministic archive, full snapshot layout/content, accelerator policy,
memory mode, stack-versioned JAX cache, working directory, and module origins
before backend construction. Disposable CPU tests prove API/engine origin
selection, archive/full-tree binding, environment behavior, no snapshot
mutation, rejection of shadow modules, exact engine readiness transitions,
stale/PID-reused/stopped process rejection, heartbeat freshness, child-exit and
timeout behavior, and terminate-to-kill whole-group cleanup. This new
transition has not yet run on the GPU, so real-backend readiness and engine
cache consumption remain unproven.
Ordinary warmup, graph capture/replay, and any compiled-call invocation remain
unauthorized. The older
T1024 `93.144 s` then `47.655 s` runs were two misses with different source
paths; their roughly 2x change is consistent with per-fusion autotune reuse, not
a top-level hit. Consequently, the descriptive `93.144` to `6.898 s` gap must
not be presented as a controlled cache-only `13.50x` result.

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
steady-state throughput. The isolated Pallas numerical blocker is now cleared
through context 2,048. The user-authorized gradient relative-L2 outer ceiling
is 3%, while the guarded FP32-delta path retains and passes the stricter 1%
regression gate: its worst observed result was dK at `0.003628` (about 0.363%).
Context-2,048 execution still requires the separate full-model and safety gates.
Exact results are in [`RESULTS.md`](RESULTS.md).

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
