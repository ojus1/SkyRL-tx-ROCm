# Query-bounded native GQA prototype

Status: CPU/Pallas-interpret correctness plus exact T=512 guarded ROCm compile,
analytic single-forward, and factorized nonzero candidate/replay promotion. The
prototype is not connected to the model dispatcher.

The prototype in `skyrl/tx/kernels/query_bounded_gqa.py` is the next safe
attention architecture for Qwen3.5-4B's `B=1, Hq=16, Hkv=4, D=256` training
shape. It replaces one monolithic attention call with a static sequence of
bounded Pallas calls and maps each group of four query heads directly to one KV
head. It does not use `jnp.repeat` on K or V.

## Exact semantics

For query position `i`, query head `h`, and KV head `g = h // 4`, the forward
operation is

```text
score(i, j, h) = dot(Q[i, h], K[j, g]) / sqrt(D)
valid(i, j) = (j <= i) and key_mask[j]
O[i, h] = softmax_j(where(valid, score, -inf)) @ V[:, g]
```

`query_start` is a static attribute of every chunk call. A query at local
position `r` in a later chunk uses global position `query_start + r` for its
causal comparison. Treating each chunk as a fresh sequence would be incorrect
because it would hide all earlier keys; a hand-constructed prefix-average test
specifically covers this failure mode.

The mask is applied to keys only, matching SkyRL's portable attention call.
The public prototype requires a nonempty right-padded mask (ones followed by
zeros). That structural property is a caller precondition, not a runtime array
value check: JAX must be able to trace the function without copying a mask back
to the host. Padded query rows still attend valid earlier keys, exactly as the
portable path does; the SFT/GRPO loss mask remains responsible for excluding
those rows. Unlike the current segment-ID adaptation, padding does not create a
second attention segment.

All sequence and tile dimensions currently require exact divisibility. There
is no partial final query chunk or key tile: unsupported tails fail validation
instead of being silently dropped. This is sufficient for the 512-token bucket
ladder through 32K, but a future general dispatcher must either pad to a tested
bucket or add explicit tail masks.

The forward saves base-2 log-sum-exp values. The custom VJP implements the
standard FlashAttention equations with FP32 softmax state and accumulators. The
CPU suite compares arbitrary output cotangents for `dQ`, `dK`, and `dV`, not
just a scalar sum loss.

## Bounded dispatches

The conservative 32K starting configuration is:

| Item | Value |
| --- | ---: |
| Query range per call | 512 tokens |
| Forward tiles | Q=64, K=64 |
| Backward tiles | Q=32, K=32 |
| Query ranges | 64 |
| Forward calls per layer | 64 |
| dQ calls per layer | 64 |
| dK/dV calls per layer | 64 |
| Forward programs per call | 128 |
| dQ programs per call | 256 |
| dK/dV programs per call | 4096 |
| Maximum forward key tiles per program | 512 |
| Maximum dQ key tiles per program | 1024 |
| Query-block/head iterations per dK/dV program | 16 x 4 |

Forward and dQ calls own only one 512-token query range. A dK/dV call owns one
query range but grids across the full KV sequence; each program is the sole
writer for one 32-token KV tile. This bounds the work in every dispatch while
retaining full causal attention.

The 32-token backward tiles follow the already-tested JAX Pallas backward tile
size and keep the two logical FP32 `dK`/`dV` tile accumulators to 64 KiB total
before other live values. This is register-value arithmetic, not a claim that
the compiler places a 64 KiB object in LDS or avoids spills. A 64-token KV tile
would double those logical accumulators to 128 KiB and is not the safe initial
gfx1100 choice.

At 32K, the existing MHA adaptation expands both K and V from 4 to 16 heads.
Avoiding that expansion removes 384 MiB of BF16 materialization:

```text
2 tensors * 32768 tokens * (16 - 4) heads * 256 values * 2 bytes = 384 MiB
```

The prototype adds a 2 MiB FP32 LSE residual and uses persistent FP32 `dK` and
`dV` accumulators totaling 256 MiB. Those accumulators are threaded through the
64 query ranges with input/output aliases and cast to the original dtype only
after the final range.

Those byte counts are logical tensor arithmetic, not a measured peak-memory
reduction. The forward output/LSE and backward dQ chunks are concatenated after
the bounded calls; optimized GPU buffer assignment must still prove that it
does not retain all chunk outputs and then allocate a second full-size result.
Likewise, the alias declarations must survive lowering before the 256 MiB
accumulators can be assumed to occupy one buffer pair rather than successive
copies.

## Deterministic dK/dV reduction

There are no atomics. For every KV tile, one program visits query blocks in
ascending order and the four associated query heads in lexical order. Across
query ranges, the previous FP32 `dK`/`dV` arrays are explicit inputs aliased to
the next outputs. That data dependency fixes the chunk order and prevents two
calls from racing on an accumulator. CPU interpret tests verify bitwise-identical
results across repeated VJPs.

## Preventing re-fusion into a monolithic launch

The query-range loop is statically unrolled in Python. Every range is a
separate `pallas_call` with a distinct static `query_start`, operation name, and
metadata. A non-interpret Pallas call lowers as an opaque GPU custom-call
boundary, which ordinary XLA fusion cannot absorb into a producer or combine
with the adjacent Pallas call. The dK/dV alias chain adds a data dependency
between its calls as well.

The CPU Jaxpr gates prove two forward `pallas_call` primitives and six
forward-plus-VJP primitives for a two-chunk numerical test. A separate 32K
abstract trace proves 64 forward boundaries and 192 boundaries in the pullback
trace (64 saved-forward, 64 dQ, and 64 dK/dV calls). This is necessary but not
sufficient evidence for ROCm. Before invoking any returned attention
executable, a guarded compile-only ROCm gate must inspect optimized HLO/LLVM
and assert all of the following. The gate itself is GPU work because
`compile()` may dispatch bounded autotuning/profiling kernels; only the
returned attention executable remains uninvoked:

1. Exactly 64 forward, 64 dQ, and 64 dK/dV custom calls exist at 32K.
2. No GPU `while` or single wrapper custom call encloses all query ranges.
3. K and V retain four heads; no `[1, 32768, 16, 256]` K/V temporary exists.
4. The dK/dV input/output aliases survive buffer assignment.
5. Chunk concatenations do not introduce a second full-size output or dQ
   buffer at peak.
6. Resource metadata shows no spills or tile allocation beyond the gfx1100
   limits.

Command buffers must remain disabled. Even separate custom calls could be
replayed unsafely if a command buffer captured the complete step, which is a
separate failure mode already observed in the optimizer path.

## Validation performed without the GPU

`tests/tx/kernels/test_query_bounded_gqa.py` currently covers:

- exact FP32 forward and arbitrary-cotangent VJP parity against
  `jax.nn.dot_product_attention` at valid lengths 1, 7, 13, and 16;
- BF16 and FP16 forward/VJP parity at dtype-appropriate tolerances;
- the exact Qwen `Hq=16, Hkv=4, D=256, BF16` geometry in interpret mode;
- global causal offsets across query ranges;
- direct Q-head to KV-head mapping with no K/V repeat;
- padded-query behavior;
- explicit rejection of partial tail chunks;
- bitwise deterministic repeated VJPs;
- separate Pallas primitive counts; and
- the exact 32K dispatch plan, abstract forward/VJP output shapes, and 64/192
  primitive counts.

All tests use `JAX_PLATFORMS=cpu` and `interpret=True`.

## Fail-closed 512-token compile-only gate

`rocm/probe_query_bounded_gqa_compile.py` is the first GPU promotion gate. It
has one immutable signature: Qwen3.5-4B BF16 `B=1, T=512, Hq=16, Hkv=4,
D=256`, including the forward output and an arbitrary-cotangent VJP returning
`dQ`, `dK`, and `dV`. The probe passes only `ShapeDtypeStruct` inputs to
`lower()` and `compile()`. It never constructs user Q/K/V/mask/cotangent
arrays, calls the lowered callable, or calls the compiled executable.
Nevertheless, `compile()` may dispatch bounded GPU autotuning/profiling kernels
and allocate compiler-managed buffers. This is GPU work, not a CPU-safe source
inspection.

The default path imports no JAX and emits an abstract refusal manifest:

```bash
.venv/bin/python rocm/probe_query_bounded_gqa_compile.py \
  --output /tmp/query-bounded-gqa-abstract.jsonl
```

ROCm compilation requires both acknowledgements, a new artifact path, and the
telemetry wrapper. Use fresh paths for both artifacts:

```bash
.venv/bin/python rocm/profile_rocm.py \
  --output /tmp/query-bounded-gqa-t512-compile.telemetry.jsonl \
  --baseline-seconds 2 \
  --timeout 300 \
  --sensor-grace-seconds 60 \
  --max-junction-temp-c 80 \
  --max-vram-gib 4 \
  --min-host-available-gib 8 \
  --max-swap-gib 0.001 \
  -- .venv/bin/python rocm/probe_query_bounded_gqa_compile.py \
       --platform rocm --allow-gpu \
       --output /tmp/query-bounded-gqa-t512-compile.jsonl
```

The file is created exclusively with mode `0600`; an existing path is never
overwritten. Each stage is flushed before the next operation. The ROCm path
reuses the isolated Pallas probe's bounded environment: device zero only, BFC
growth allocation capped at a 0.75 fraction, unified-memory and allocator
bypasses rejected, and command buffers forced off. It also holds
`guarded_qwen35_rocm_process` across backend initialization and compilation and
rechecks the current boot journal before releasing the guard.

The final artifact contains StableHLO and optimized-HLO hashes, custom-call
counts, and exact named Pallas attribution for the one forward, one dQ, and one
dK/dV boundary, plus compiler memory and cost analysis where available. Each
dialect must expose exactly three Pallas/Triton calls, each expected name must
belong to exactly one of those calls, every call must have exactly one expected
name, and no outer HLO `while` may remain. A mismatch fails closed after
preserving the metadata. "Compile-only" means that the returned model callable
is not invoked; it does not mean no GPU kernels run during compilation, nor is
it evidence of numerical correctness or runtime watchdog safety.

### ROCm 7.2.4 compile-only result

The first guarded hardware gate passed on commit `a3a40d41` and clean boot
`54ccf56c-5f4f-4ef7-ac98-c13e0587b5b9`. The exact BF16
`B=1, T=512, Hq=16, Hkv=4, D=256` forward-plus-arbitrary-VJP signature lowered
in 0.349206 s and compiled in 3.037847 s. Both StableHLO and optimized HLO
contained exactly three `__gpu$xla.gpu.triton` custom calls: one exactly named
forward boundary, one dQ boundary, and one dK/dV boundary. Both dialects had
zero outer `while` operations, and every exact-name/bijection check passed.
Their SHA-256 digests were respectively
`f62ffb9ae654d13152031dece070e05cf1e44e4621b45cfd614c0092583e6ab5` and
`6b18b3834fac2a85926f1009516d75b28cbb2f029ca7e26bcf023e5c545b6c88`.

`CompiledMemoryStats` reported 10,487,808 B of arguments, 10,485,792 B of
outputs, 2,130,688 B of temporary storage, and no aliases. The telemetry
wrapper completed in 10.010465 s with return code zero: physical VRAM peaked
at 711,987,200 B, junction temperature at 53 C, power at 132 W, and swap at
zero. The kernel log stayed available and fault-free. The private probe,
telemetry, and summary artifacts are:

- `/tmp/query-bounded-gqa-t512-compile-boot54ccf56c-run1.jsonl`
- `/tmp/query-bounded-gqa-t512-compile-boot54ccf56c-run1.telemetry.jsonl`
- `/tmp/query-bounded-gqa-t512-compile-boot54ccf56c-run1.telemetry.jsonl.summary.json`

All three files are mode `0600`. The manifest, compiled record, and completion
record report zero lowered-callable and compiled-executable invocations; there
is no error record. The guarded journal postflight passed. A separate
point-in-time postcheck found `/dev/kfd` and the AMD render node unowned and
the card back in runtime suspend.

That result promoted only the exact T=512 structure through compile. It did
not by itself promote execution, numerical correctness, padding, repeated
launches, or any larger bucket; the separate first runtime gate follows below.

## Exact T=512 single-forward runtime gate

`rocm/probe_query_bounded_gqa_runtime.py` implements the first execution gate.
Its default path imports no JAX and emits only an abstract refusal:

```bash
.venv/bin/python rocm/probe_query_bounded_gqa_runtime.py \
  --output /tmp/query-bounded-gqa-t512-runtime-abstract.jsonl
```

The first hardware run must use fresh artifact paths and the tighter
expected-footprint anomaly limits below. Separately reviewed larger experiments
may use ceilings up to 90 C, 315 W, and the full practical VRAM budget; this
one-call attention probe is expected to remain far below them:

```bash
.venv/bin/python rocm/profile_rocm.py \
  --output /tmp/query-bounded-gqa-t512-runtime.telemetry.jsonl \
  --baseline-seconds 2 \
  --timeout 120 \
  --sensor-grace-seconds 60 \
  --max-junction-temp-c 70 \
  --max-vram-gib 2 \
  --max-gpu-power-watts 315 \
  --min-host-available-gib 8 \
  --max-swap-gib 0.001 \
  -- .venv/bin/python rocm/probe_query_bounded_gqa_runtime.py \
       --platform rocm --allow-gpu \
       --output /tmp/query-bounded-gqa-t512-runtime.jsonl
```

The probe does not consume the prior compile artifact. In its guarded process
it freshly lowers and compiles the immutable BF16
`B=1, T=512, Hq=16, Hkv=4, D=256` forward. Before an executable is exposed,
both StableHLO and optimized HLO must contain exactly one custom call total.
Its target set must be exactly `{__gpu$xla.gpu.triton}`, its sole marker must
be the exact forward name, and there may be no dQ, dK/dV, or outer `while`.
Compiler memory analysis is mandatory: temporary memory must be at most 64 MiB
and arguments plus output plus temporary memory at most 128 MiB.

Only after those gates pass does it build hashed host BF16 inputs: zero Q/K,
deterministic position-plus-KV-head V values in approximately `[-1, 1]`, and
an all-ones mask. The candidate executable is invoked exactly once and must
complete in less than 100 ms, with a clean journal check immediately after the
dispatch. A flushed `dispatch_started` record must first show one attempt and
zero completions. The result is copied to the host and compared entirely in
NumPy to the analytic cumulative-mean causal GQA result mapped by
`query_head // 4`.
The full output must be finite with relative L2 below 1%, cosine at least
0.9999, and maximum absolute error at most 0.02.

There is no GPU reference, random input, replay, backward, padding case, or
model-dispatcher connection in this first runtime gate. Any compile, memory,
journal, duration, transfer, or numerical failure is fatal. Replay and the
broader input matrix remain separate later probes.

### ROCm 7.2.4 runtime result

The gate passed on clean commit `f421039b` and boot
`54ccf56c-5f4f-4ef7-ac98-c13e0587b5b9`. Fresh lowering took 0.264439 s and
compilation took 2.035000 s. StableHLO and optimized HLO each contained one
custom call total, exactly targeted `__gpu$xla.gpu.triton`, with the exact
forward marker and no dQ, dK/dV, or outer `while`. Their SHA-256 digests were
`cee355417f5e3220893f3f4ddd85299cfdb055e46e9fe3d6b4e8de4ce9fb3b62` and
`6ee442dffa82364b6f2feb453eb6030877012e328d3718df19b9eeef12d5c0d1`.

Compiler memory analysis reported 6,293,504 B of arguments, 4,194,304 B of
output, and 33,024 B of temporary storage (10,520,832 B combined), passing the
mandatory release gate. The one checked invocation completed in 0.006422197 s.
Its output was finite; against the host analytic result, relative L2 was
0.001571651, cosine was 0.999958456, mean absolute error was 0.000613824, and
maximum absolute error was 0.001953125. The flushed invocation record showed
one attempt and zero completions before dispatch; the final counters were
exactly one attempt and one completion, with zero replay, backward, GPU
reference, device error-reduction, or lowered-callable invocations.

The telemetry wrapper completed in 19.649497 s with return code zero. Physical
VRAM peaked at 765,751,296 B and swap remained at zero. Among the 141 readable
sensor samples, observed junction temperature and power reached 50 C and
130 W. Temperature and power were unavailable for the first 34 of 175 measured
samples (3.42 s of the measured window), so those values are observed maxima,
not guaranteed full-run peaks. The backend-initialization, compile,
input-transfer, candidate-dispatch, device-get, and final guarded journal
checks were all clean. A separate, unarchived point-in-time operator postcheck
found `/dev/kfd` and the AMD render node unowned and the card back in runtime
suspend; the telemetry artifacts themselves prove process exit, not GPU
cooldown. All artifacts are mode `0600`:

- `/tmp/query-bounded-gqa-t512-runtime-boot54ccf56c-run1.jsonl`
- `/tmp/query-bounded-gqa-t512-runtime-boot54ccf56c-run1.telemetry.jsonl`
- `/tmp/query-bounded-gqa-t512-runtime-boot54ccf56c-run1.telemetry.jsonl.summary.json`

This promotes only one all-valid analytic forward at T=512. It does not
promote replay, nonzero Q/K, random inputs, padding, a GPU reference, backward,
larger buckets, or model integration.

## Exact T=512 nonzero replay gate

`rocm/probe_query_bounded_gqa_replay.py` is a separate next-rung probe. Its
default path imports no JAX and emits only an abstract refusal. The guarded
ROCm path must run in a fresh process under the profiler:

```bash
.venv/bin/python rocm/profile_rocm.py \
  --output /tmp/query-bounded-gqa-t512-replay.telemetry.jsonl \
  --card card1 \
  --interval 0.1 \
  --baseline-seconds 2 \
  --timeout 120 \
  --sensor-grace-seconds 5 \
  --max-junction-temp-c 70 \
  --max-gpu-power-watts 315 \
  --max-vram-gib 2 \
  --min-host-available-gib 8 \
  --max-swap-gib 0.001 \
  -- .venv/bin/python rocm/probe_query_bounded_gqa_replay.py \
       --platform rocm --allow-gpu \
       --output /tmp/query-bounded-gqa-t512-replay.jsonl
```

The immutable input is dense, nonzero BF16 Q/K/V built on the host from a
fixed seed. Token/head scalars and seeded sign directions make the input
random but deliberately factorized, allowing an independent FP32 host causal
GQA oracle without a GPU reference or a billion-operation CPU matmul. The
probe compiles once and inherits the exact one-forward-call IR gate and the
64 MiB temporary/128 MiB total compiled-memory limits from the promoted
single-forward gate. Command buffers are proven disabled before backend use.

The executable first runs as a candidate. Its synchronized duration must be
below 100 ms; output must be finite with relative L2 below 1%, cosine at least
0.9999, and maximum absolute error at most 0.02. Exact invocation counters and
a clean current-boot journal checkpoint must also pass. Only then does the
probe issue an opaque one-shot replay authorization, bound by object identity
to that exact executable and to clean candidate dispatch/device-get proofs.
The first replay attempt consumes it, including an invalid attempt. The one
ordinary host-driven replay must pass the same duration/numerical gates and
match the candidate output byte-for-byte. The private JSONL records a flushed
pre-dispatch attempt for each launch and journal checkpoints after backend
initialization, compile, input transfer, each dispatch, each device transfer,
and final postflight.

This probe has no GPU reference, device-side error reduction, command-buffer
replay, backward work, padding case, recompilation, or model integration. It
promotes nothing until its artifacts and telemetry receive an independent
audit.

### ROCm 7.2.4 replay result

The gate passed from clean commit `3361d15d` on boot
`54ccf56c-5f4f-4ef7-ac98-c13e0587b5b9`. The artifact's probe and delegated
runtime-probe SHA-256 values match the committed source. Lowering took
0.274011 s and compilation took 2.054230 s. StableHLO and optimized HLO each
contained exactly one custom call, targeted only
`__gpu$xla.gpu.triton`, with one forward marker, no dQ/dK-dV marker, and no
outer `while`. Their SHA-256 values were
`cee355417f5e3220893f3f4ddd85299cfdb055e46e9fe3d6b4e8de4ce9fb3b62`
and `2a56b206f4c424e18ba9c415ca172782e2622682bca5b830094d0e5671489643`.

Compiler analysis reported 6,293,504 B of arguments, 4,194,304 B of output,
and 33,024 B of temporary storage (10,520,832 B combined). The candidate took
6.275332 ms and the ordinary replay took 0.993285 ms. Both produced the same
BF16 bytes, with output SHA-256
`764e3416be5ae78a889dff83748da180e0377b4081ce8ac9a26a5ccffaf4ab2f`.
Against the independent FP32 host oracle, both had relative L2 0.002079095,
cosine 0.999966979, mean absolute error 0.000062428, and maximum absolute error
0.001205862. Final counters were exactly two forward attempts/completions, one
candidate attempt/completion, one replay attempt/completion, and zero lowered
callable invocations.

The profiler completed in 28.703052 s with return code zero. Observed physical
VRAM peaked at 765,743,104 B and swap remained zero. Temperature and power
became readable 3.920438 s after measured sampling began, before backend-ready,
and all subsequent samples remained readable through compile and both
launches; their observed maxima were 49 C and 130 W. Candidate and replay began
9.825239 s and 14.974710 s after the first readable sensor sample. These are
sampled maxima, not continuous guarantees: the nominal interval was 100 ms,
the longest gap was 151.834 ms, and neither sub-7-ms dispatch was guaranteed to
contain a sample.

All seven child journal checkpoints, its preflight/postflight, and the
profiler's periodic/final driver checks were clean. A separate, unarchived
point-in-time operator postcheck found `/dev/kfd` and the AMD render node
unowned and the card in runtime suspend. All artifacts are mode `0600`:

- `/tmp/query-bounded-gqa-t512-replay-boot54ccf56c-run1.jsonl`
- `/tmp/query-bounded-gqa-t512-replay-boot54ccf56c-run1.telemetry.jsonl`
- `/tmp/query-bounded-gqa-t512-replay-boot54ccf56c-run1.telemetry.jsonl.summary.json`

Independent audits approved the 29-record child artifact and 286-sample
telemetry. This promotes only all-valid, forward-only T=512 with dense nonzero
but deliberately factorized seeded inputs and one ordinary replay. It does not
promote fully IID per-feature inputs, padding, backward, a GPU reference,
larger buckets, latency distributions, or model integration.

## Exact T=512 fully-IID forward gate

`rocm/probe_query_bounded_gqa_iid.py` is the next separate forward-only rung.
Its Q, K, and V scalars are independently drawn by host PCG64 from the same
nonzero BF16 grid, uniformly over signed magnitudes 1 through 48 divided by
128. It builds a complete independent FP32 dense causal-GQA oracle on the host,
then permits exactly one checked candidate invocation. There is no replay,
padding case, backward work, GPU reference, device-side error reduction, or
model integration. The NumPy emulation of the kernel's BF16 probability path
is informational and cannot authorize execution or success.

The default mode imports no JAX and emits an abstract refusal. The guarded
ROCm path must run in a fresh profiler-controlled process:

```bash
.venv/bin/python rocm/profile_rocm.py \
  --output /tmp/query-bounded-gqa-t512-iid.telemetry.jsonl \
  --card card1 \
  --interval 0.1 \
  --baseline-seconds 2 \
  --timeout 120 \
  --sensor-grace-seconds 5 \
  --max-junction-temp-c 70 \
  --max-gpu-power-watts 315 \
  --max-vram-gib 2 \
  --min-host-available-gib 8 \
  --max-swap-gib 0.001 \
  -- .venv/bin/python rocm/probe_query_bounded_gqa_iid.py \
       --platform rocm --allow-gpu \
       --output /tmp/query-bounded-gqa-t512-iid.jsonl
```

The probe delegates the already-audited exact one-forward-call StableHLO,
optimized-HLO, and compiled-memory release gates. Its synchronized candidate
duration must be below 100 ms; output must be finite with relative L2 below
1%, cosine at least 0.9999, and maximum absolute error at most 0.02. It remains
unpromoted until the private child artifact and profiler telemetry receive an
independent audit.

### ROCm 7.2.4 fully-IID result

The gate passed from clean commit `fef0a442` on boot
`54ccf56c-5f4f-4ef7-ac98-c13e0587b5b9`. The manifest's probe, delegated
replay/runtime probe, and profiler SHA-256 values all match the committed
sources. Lowering took 0.282275 s and compilation took 1.992297 s. StableHLO
and optimized HLO each contained exactly one custom call targeted only at
`__gpu$xla.gpu.triton`, one forward marker, no dQ/dK-dV marker, and no outer
`while`. Their SHA-256 values were
`cee355417f5e3220893f3f4ddd85299cfdb055e46e9fe3d6b4e8de4ce9fb3b62`
and `1e22fcc17ce391dd39a6c8016c4852d49c9da783c13ab25605f770f5f536e112`.

Compiler analysis reported 6,293,504 B of arguments, 4,194,304 B of output,
and 33,024 B of temporary storage (10,520,832 B combined). The only candidate
invocation took 14.230550 ms. Its BF16 output SHA-256 was
`e3454e82a4c4d3b3775b9045fdb3535c81a234a47fd8958847fc3f67a85ef048`.
Against the complete independent FP32 host oracle, relative L2 was
0.001852000, cosine was 0.999966025, mean absolute error was 0.000029012, and
maximum absolute error was 0.001300991. Final counters were exactly one
forward attempt/completion and zero lowered-callable invocations. The input
construction used the same signed-magnitude distribution independently for
every Q, K, and V feature; no accelerator RNG or reference was used.

The profiler completed in 30.002400 s with return code zero. It recorded 20
baseline, one preflight, and 279 measured samples. Observed physical VRAM
peaked at 765,743,104 B, junction temperature at 48 C, and power at 129 W;
swap remained zero and host-available memory stayed at or above
62,372,548,608 B. Temperature and power first became jointly readable 4.018 s
after measured sampling began, within the 5 s grace period and well before the
candidate; neither sensor was missing afterwards. The longest measured sample
gap was 141.925 ms, so these are observed run maxima and the 14 ms dispatch was
not continuously sampled.

All seven child journal checkpoints, child preflight/postflight, profiler
driver checks, and a separate operator postcheck were clean; `/dev/kfd` and
the AMD render node were unowned and the card returned to runtime suspend.
All artifacts are mode `0600`:

- `/tmp/query-bounded-gqa-t512-iid-boot54ccf56c-run1.jsonl`
- `/tmp/query-bounded-gqa-t512-iid-boot54ccf56c-run1.telemetry.jsonl`
- `/tmp/query-bounded-gqa-t512-iid-boot54ccf56c-run1.telemetry.jsonl.summary.json`

Their SHA-256 values are respectively
`682ad68ff4542dc44827621c6667039253b2330ac56ac58b4b48a67c4ea52ce0`,
`768ff3035a0b1ff0dbcdb6540b72d8af024e7c1c068915cc807dd9cb9f384b66`,
and `73e0e0941b8c037ba57166437d077a75122d53eb21f858ed12347b36a352205a`.
This promotes only one all-valid, forward-only T=512 candidate with fully IID
per-feature bounded-grid inputs. It does not promote replay, padding,
backward, other lengths, latency distributions, or model integration.

## C=256/T=512 last-chunk compile-only gate

Longer sequences require timing one Pallas call at a time: the full public
forward contains `T/C` static calls, so an aggregate synchronized time cannot
identify or bound an individual launch once it exceeds the watchdog target.
`query_bounded_gqa_forward_chunk` is therefore a forward-only experimental
entry point for one query range against longer K/V. It has no concatenation or
custom VJP and remains disconnected from model dispatch.

`rocm/probe_query_bounded_gqa_chunk_compile.py` is the first C=256 gate. Its
default mode imports no JAX and emits only an abstract refusal. The guarded
path lowers exactly `[1,256,16,256]` BF16 queries at global `query_start=256`
against `[1,512,4,256]` BF16 K/V, then compiles once and discards the
executable without returning or invoking it:

```bash
.venv/bin/python rocm/profile_rocm.py \
  --output /tmp/query-bounded-gqa-c256-t512-compile.telemetry.jsonl \
  --card card1 \
  --interval 0.1 \
  --baseline-seconds 2 \
  --timeout 120 \
  --sensor-grace-seconds 15 \
  --max-junction-temp-c 70 \
  --max-gpu-power-watts 315 \
  --max-vram-gib 2 \
  --min-host-available-gib 8 \
  --max-swap-gib 0.001 \
  -- .venv/bin/python rocm/probe_query_bounded_gqa_chunk_compile.py \
       --platform rocm --allow-gpu \
       --output /tmp/query-bounded-gqa-c256-t512-compile.jsonl
```

StableHLO and optimized HLO must independently contain exactly one custom
call, with the sole target `__gpu$xla.gpu.triton`, the exact full marker
`query_bounded_gqa_forward_q256`, no other forward/backward/lookalike marker,
no outer `while`, and exact query metadata wherever preserved. Compiler memory
analysis must report exactly 4,196,352 B of arguments and 2,097,152 B of
output, with at most 64 MiB of temporary storage. Compilation may issue
bounded GPU profiling work, but every executable/lowered-call counter remains
zero. This gate promotes nothing until its source, artifact, and telemetry are
independently audited.

### ROCm 7.2.4 C=256 compile-only result

The gate passed from clean commit `dd5e45e4` on boot
`54ccf56c-5f4f-4ef7-ac98-c13e0587b5b9`. Every source hash in the manifest
matches that checkout. Lowering took 0.250435 s and compilation took 1.988343
s. StableHLO and optimized HLO each contained exactly one total custom/Pallas
call with the sole target `__gpu$xla.gpu.triton`, the exact q256 marker, exact
decoded `query_start=256` and `query_size=256`, no other forward/backward
kernel marker, and no outer `while`. Their SHA-256 values were
`f604694d4de7d8e2c24b1e6dd94adf1cd566736bf99d78d464e91d2d821b8e57`
and `6df7e0ccfad755c6ab88ce3eb9e6ea3c217742bc917fb3e8f2fe39b4437c23dc`.

Compiler analysis reported the exact 4,196,352 B of arguments and 2,097,152 B
of output, plus 16,640 B of temporary storage (6,310,144 B combined). All
forward-attempt/completion, lowered-callable, and compiled-executable counters
remained zero. The compiled executable was deleted without being returned or
invoked.

The profiler completed in 22.577327 s with return code zero. Its 20 baseline,
one preflight, and 206 measured samples observed at most 711,974,912 B physical
VRAM, 49 C junction temperature, and 129 W power. Host-available memory stayed
at or above 62,351,974,400 B and swap remained zero. Temperature and power
first became jointly readable 6.925 s after measured sampling began, within
the deliberately extended 15 s grace needed for the child preflight and
pre-backend journal check; neither was missing afterwards. The longest
measured sample gap was 105.407 ms. Values are sampled run maxima, not
continuous bounds on compiler profiling kernels.

All four child journal checkpoints, preflight/postflight, profiler driver
checks, and a separate operator postcheck were clean. `/dev/kfd` and the AMD
render node were unowned and the card returned to runtime suspend. All
artifacts are mode `0600`:

- `/tmp/query-bounded-gqa-c256-t512-compile-boot54ccf56c-run3.jsonl`
- `/tmp/query-bounded-gqa-c256-t512-compile-boot54ccf56c-run3.telemetry.jsonl`
- `/tmp/query-bounded-gqa-c256-t512-compile-boot54ccf56c-run3.telemetry.jsonl.summary.json`

Their SHA-256 values are respectively
`c1d272756fc05c0cec34424687da8764bd9c40db30bbdcf4b7aec6e419b00875`,
`556396c1e53be2470878c8c1e693ebfb7d0267ca2728a18d362e7f247ca7fd24`,
and `0f3e461b5846ebd175513edcd5fbccdc74e1d1ef7c401ee867470c6d7ea42b2e`.

Two precursor attempts are deliberately not promotion evidence. Run 1 was
terminated before backend initialization when the old 5 s sensor grace
expired during pre-backend safety work; VRAM stayed at idle and the boot
remained clean. Run 2 compiled successfully with zero invocations but was
rejected by an over-strict pre-fix parser that did not decode canonical MLIR
metadata. This result promotes only compilation of the exact C=256/T=512 last
chunk. It does not authorize executing it or qualify any runtime, other
query-start, longer length, padding, backward path, or model integration.

### ROCm 7.2.4 C=256 analytic runtime result

The independently audited one-shot runtime gate passed from committed source
`25c91623` on the same clean boot. It compiled the exact last chunk at
`query_start=256`, then invoked one checked executable exactly once. Q and K
were zero, the mask was all-valid, and V used deterministic high-contrast
nonlinear BF16 values spanning -10.375 through 10.375 and varying across token,
KV head, and feature. The independent host oracle was the inclusive global
FP32 prefix mean for positions 256 through 511 with `query_head // 4` GQA
mapping.

The one synchronized candidate took 6.337533 ms, below both the 75 ms
promotion ceiling and 100 ms hard limit. Its output relative L2 was
0.001540165, cosine 0.999984086, maximum absolute error 0.001953125, and mean
absolute error 0.000335445; every value was finite and shape/dtype/byte counts
were exact. As a sensitivity control, using the exclusive prefix instead
would produce relative L2 0.07911505, cosine 0.99686277, and maximum error
0.04070252, failing the gate decisively. Feature and KV-head permutations also
fail the CPU gate.

StableHLO and optimized HLO again contained exactly one total/Pallas custom
call with the exact q256 marker, ROCm Triton target, and query metadata, with
no other query-bounded marker or outer `while`. Their SHA-256 values were
`f604694d4de7d8e2c24b1e6dd94adf1cd566736bf99d78d464e91d2d821b8e57`
and `bce513f87faffbbf5a60c5664956cd80a5c90951f42e5c3b975addae9f63c45a`.
Compiler memory was the exact 4,196,352 B of arguments, 2,097,152 B of output,
zero alias bytes, and 16,640 B of temporary storage.

The profiler completed in 34.915053 s with return code zero. Across 40
baseline, one preflight, and 644 measured samples, maximum physical VRAM was
745,201,664 B, maximum junction temperature 50 C, maximum power 130 W, minimum
host-available memory 62,074,712,064 B, and swap stayed zero. Sensors appeared
within 4.408 s and none was later missing. All eight journal checkpoints and
both safety flights were clean; `/dev/kfd` was unowned after exit and the card
returned to runtime suspend.

All artifacts are mode `0600`:

- `/tmp/query-bounded-gqa-c256-t512-runtime-boot54ccf56c-run1.jsonl`
- `/tmp/query-bounded-gqa-c256-t512-runtime-boot54ccf56c-run1.telemetry.jsonl`
- `/tmp/query-bounded-gqa-c256-t512-runtime-boot54ccf56c-run1.telemetry.jsonl.summary.json`

Their SHA-256 values are respectively
`9bcd4acc151aec402696ba8bbc1176b2efba1c38d3c23cc6920a9b712c99c0c6`,
`0bda6e8c69f54f9f97787f16c2a8da40458d6ee2576fbfd4178e5bb0b56f992b`,
and `027b212ac371ef128e7ba14b58415f8e2235ceecce3f9870a65b80e4c8e6c87a`.

This promotes only one BF16 batch-1, all-valid, zero-Q/K C=256 forward chunk
at T=512. It validates global causal offset, GQA value-head mapping, and
nonconstant value accumulation. It does not validate nonzero QK logits or
scale, padding, another chunk or length, replay, backward, model integration,
or sustained throughput. Those remain separate fresh-process gates.

## C=256 analytic final-chunk length ladder through 32K

`rocm/probe_query_bounded_gqa_chunk_length.py` extends the audited analytic
case through exactly one fresh process at each of
`T={1024,2048,4096,8192,16384,24576,32768}`. Every rung uses the final
256-query range, zero Q/K, an all-valid mask, high-contrast nonlinear V, and
an independent host FP32 global-prefix oracle. Sequential promotion is
mandatory; a passing rung authorizes only the next one.

Every independently audited rung lowered and compiled exactly one
`__gpu$xla.gpu.triton` call with marker `q{T-256}`, exact preserved
`query_start=T-256` and `query_size=256` metadata, no other bounded-attention
marker, no outer `while`, and zero alias bytes. Compiler memory remained the
exact `2,097,152 + 4100*T` argument bytes, 2,097,152 output bytes, and 16,640
temporary bytes. Each executable was invoked once.

| T | Query marker | Arguments MiB | Candidate ms | Relative L2 | Cosine | Peak VRAM MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | q768 | 6.004 | 6.556177 | 0.001534396 | 0.999998450 | 710.699 |
| 2,048 | q1792 | 10.008 | 7.292170 | 0.001534178 | see note | 726.691 |
| 4,096 | q3840 | 18.016 | 8.361281 | 0.001532413 | 0.999998826 | 730.707 |
| 8,192 | q7936 | 34.031 | 9.700540 | 0.001534719 | 0.999998822 | 762.699 |
| 16,384 | q16128 | 66.062 | 13.386882 | 0.001453649 | 0.999998944 | 826.695 |
| 24,576 | q24320 | 98.094 | 17.007953 | 0.001318760 | 0.999999130 | 954.699 |
| 32,768 | q32512 | 130.125 | 19.581098 | 0.001221016 | 0.999999255 | 954.688 |

The T=2048 artifact's million-element FP32 diagnostic reduction reported
cosine 1.000010133. Independent audit showed this was a 10.13-ppm reduction
artifact: its relative L2 implies true cosine at least 0.999998823. Before
T=4096, commit `26b84570` changed dot products and norms to FP64, retained raw
cosine, and clipped the gated value to [-1,1]. The reporting-only revision was
reviewed without rerunning T=2048; all later values in the table are the exact
FP64 reports. Maximum absolute error was 0.0009765625 through 16K and
0.0009765923 at 24K/32K, far below the 0.02 gate.

Across the complete ladder, the largest sampled physical VRAM was
1,001,074,688 B, largest junction temperature 59 C, largest power sample
217 W, minimum host-available memory 60,949,962,752 B, and swap remained
zero. Every profiler returned zero, every required child journal checkpoint
and safety flight was clean, every process exited, `/dev/kfd` and the render
node became unowned, and the card returned to runtime suspend. Candidate
latency remained below the 75 ms promotion ceiling at every length.

All artifacts are private mode `0600`. The child/telemetry/summary SHA-256
triples are:

- T=1024:
  `d6361de3ea1934a0cf0d53814452c5e22b235fbe0a69267c212990cf884a3b38`,
  `8451d5b1e7877efd29155d63cb2baa100b8d95fe983efda17a2779e092a1b1c6`,
  `be5e310e62dfeff65076f397e2995d45fd835b8e8b167771f509eb412d35e654`.
- T=2048:
  `022b141a72bfae723673905138e978d6bb7c6f8ae66778d924333e281da9a4c1`,
  `771b9a970a1de2dea78e21dcfb4ac1c951ca6e6499772d89514ceeab7d3c28e4`,
  `6cdde6ffe2fdfa28259ea9c444b9bf0722d34cf38f4b8ab681ce3456993e6284`.
- T=4096:
  `4646210a2c898dcfdc505dbefedb077326fc252918daa8d738fa2d287b4ef75c`,
  `344c9a9223b66afe3047bb21f6e39f67dfe642ecdab0cb0690e0370b8e28698a`,
  `0e174c41b5ba32fc9b45ee8a3ca11f37afd6dfac87d2191f470b90a563e444d7`.
- T=8192:
  `dae93da4bd6897f11a2b347b8b779db44d52422fd04d9947c9031612f7e7a441`,
  `8803d6f7fe43831a266ee0f43b4c71cad5ff804de392923945ce076550e629f5`,
  `40ee288ed1f52ba987b744cb4f1fd098747ed9a7260c28aad7ccaf4788c8709e`.
- T=16384:
  `3ffbc1eb24c94896182a9298f4dec024233fe036f87307e348568179c3eef6c7`,
  `c20049df35c1cf0fe8fd8d68c67f4099a342ee0534df321bed66e0d641a23ac9`,
  `75e94adbd2857240acc6692977fbd8182e164cab6b331f4f6434125b1adc9e40`.
- T=24576:
  `cf5aee3f05069cf3cec22b3524deeab5561f74f88f383b590f47d9c07599e72c`,
  `73ad9ac2d5ee6898c992eb75fb85f9c947a113634e2c58ef37c8449b4c2b424a`,
  `f04eef9c5bc9588ed6f93c8b0d3b702d592de29a34a19fbf3a771f3398720b26`.
- T=32768:
  `3fc9bbede1f9d88c940228092cde51da7a74d77755e5604c2d05403c6cf7b5a4`,
  `d295f781211f453242e31338caedce08eca69ba95a945b0001d8c21c02261a77`,
  `2594d6fb3b91ebfd033d8a8234740a3edff0ba8a43f48757ad26269058755bc7`.

This promotes the analytic final-C256 chunk only through 32K. It proves global
causal offsets, grouped-head V mapping, bounded one-dispatch latency, and exact
memory scaling for that input family. It does **not** promote nonzero Q/K
logits or scale behavior, padding, earlier query chunks, the full static call
schedule, replay, backward, model integration, SFT, or GRPO. Those require
separate fresh-process gates; this ladder authorizes no additional GPU work.

## Exact T=1024/C=256 nonzero-scale sentinel gate

`rocm/probe_query_bounded_gqa_nonzero_scale.py` is a separate, default-refusing
forward gate for the first limitation left by the zero-Q/K length ladder. It
uses the final 256 queries at T=1024, an all-valid mask, and independent
per-feature host PCG64 BF16 Q/K/V values from nonzero signed magnitudes 1
through 96 divided by 128. The candidate scale is passed explicitly as the
exact binary fraction 3/32. A required host-only wrong-scale control at 1/16
has relative L2 about 0.098449 against the candidate-scale oracle, so the input
detects a missing or incorrect scale decisively.

The independent FP32 stable-softmax oracle streams one query head over
32-query and 64-key tiles; it does not allocate a full 256-by-1024 logits or
probability matrix. CPU calibration observes a maximum absolute valid logit of
about 1.431713. Its conservative explicit NumPy-array scratch accounting is
332,288 bytes, including simultaneously live validity, gathered-logit,
logit/probability, tile, accumulator-update, row-state, and position arrays.
That is accounting rather than a measured allocator peak and excludes
Python/NumPy object overhead and internal BLAS workspace. The exact
production-size oracle is also checked in unit tests against a separate dense
implementation restricted to small shapes.

Before environment setup or JAX import, the guarded path pins the delegated
length-probe, compile-helper, safety-helper, and query-kernel sources and binds,
validates, and retains the exact safety callables. It then
allows one lower, one compile, one input-tuple device placement, one checked
executable invocation, and one device retrieval. StableHLO and optimized HLO
must each prove exactly one q768 ROCm Triton custom call, no other bounded
marker, no outer `while`, explicitly present exact `query_start=768` and
`query_size=256` metadata without lookalikes, duplicates, prefixes, or value
suffixes. An independent raw-IR check requires canonical integer text, so forms
such as `768x`, `768garbage`, `768e2`, and `768.0` fail closed even if a
delegated numeric-prefix parser recognizes `768`. Compiler memory must prove
6,295,552 argument bytes,
2,097,152 output bytes, zero alias bytes, and at most 64 MiB temporary memory.
There is no warmup, replay, backward path, accelerator reference or reduction,
or model invocation.

The default mode imports no JAX and only emits a refusal manifest. After
independent source review, the following command was executed exactly once in
a fresh supervised process. It must not be replayed as part of reproducing
this documentation change:

```bash
.venv/bin/python rocm/profile_rocm.py \
  --output /tmp/query-bounded-gqa-c256-t1024-nonzero-scale.telemetry.jsonl \
  --card card1 \
  --interval 0.1 \
  --baseline-seconds 2 \
  --timeout 120 \
  --sensor-grace-seconds 15 \
  --max-junction-temp-c 70 \
  --max-gpu-power-watts 315 \
  --max-vram-gib 2 \
  --min-host-available-gib 8 \
  --max-swap-gib 0.001 \
  -- .venv/bin/python rocm/probe_query_bounded_gqa_nonzero_scale.py \
       --platform rocm --allow-gpu \
       --output /tmp/query-bounded-gqa-c256-t1024-nonzero-scale.jsonl
```

Promotion required finite output, relative L2 below 0.01, cosine at least
0.9999, maximum absolute error at most 0.02, candidate duration below the
100 ms hard limit and 75 ms promotion ceiling, and a clean outer profiler.
The guarded run passed all gates. Its sole checked candidate dispatch took
`6.87041300261626 ms`. The BF16 output SHA-256 was
`590e462e2c0fc03e56ce72ee1aabd33ebbae09f8f1d309803ea9e486100f7f91`;
against the independent FP32 oracle, relative L2 was
`0.002357022718211298`, cosine was `0.9999972222493951`, and maximum
absolute error was `0.00022630393505096436`. The required wrong-scale control
retained relative L2 `0.09844902832132886` before device placement.

Both IR dialects contained one sole q768 ROCm Triton call, no outer `while`,
and exact parsed and raw `query_start=768`/`query_size=256` metadata. Compiler
memory was 6,295,552 argument bytes, 2,097,152 output bytes, 16,640 temporary
bytes, and zero alias bytes. Final accounting recorded exactly one lower,
compile, input-tuple placement, checked invocation, and output retrieval. It
recorded zero warmups, replays, lowered-callable invocations, backward calls,
accelerator references or reductions, and model calls.

The profiler completed with return code zero and no signal. Across 392
measured samples, peak physical VRAM was `745213952` bytes, peak junction
temperature was `48 C`, peak board power was `128 W`, minimum host-available
memory was `60000137216` bytes, and swap use remained zero. All eight child
journal checkpoints and the child preflight/postflight were clean.

The mode-`0600` evidence artifacts and SHA-256 values are:

- `/tmp/query-bounded-gqa-c256-t1024-nonzero-scale.jsonl`:
  `504ed6136215b9871be7077174ab4c40aaa4eae57a79630753516535e6c89e5d`;
- `/tmp/query-bounded-gqa-c256-t1024-nonzero-scale.telemetry.jsonl`:
  `438aedd4622b28bbb3f4fc59f2429fd37c78d35679043eeaaa78632ae672684d`;
- `/tmp/query-bounded-gqa-c256-t1024-nonzero-scale.telemetry.jsonl.summary.json`:
  `1545f35f7c77abf0fd94037755dcac9240ae1478e9dd3c89d04bed1daf2409a3`.

This promotes only the exact all-valid final-C256 T=1024 nonzero-scale forward
sentinel. It does not authorize padding, another query chunk or length, replay,
backward, model integration, SFT, GRPO, or a latency-distribution claim.

## Pending T=1024/C=256 right-padding gates

`rocm/probe_query_bounded_gqa_padding.py` defines the next isolated forward
gate but has **not** been run on the GPU. Its exact case enum is
`valid768`, `valid769`, `valid831`, `valid832`, `valid833`, and `valid1023`.
One explicit case is admitted per fresh process; the probe has no multi-case
GPU path. These cases place the first masked key immediately before the final
query chunk, immediately after its first query, on either side of the next
64-key tile boundary, and at the last sequence position.

Q/K/V and scale are unchanged from the promoted all-valid sentinel. Their
SHA-256 values remain respectively
`16aa12a02e88387223f000513febba987c23490016e5a1c9fe32019a862afc5d`,
`85ba4ec243a74b9a2019d30c94c4a0edf62e2204a6d3148fe551113605134841`,
and `60132fd8c7733d2f02381f90d59e5a3dc7d740c09f25e1833540d7c956771b8a`;
the explicit scale is still 3/32. Only the int32 key mask changes to a
nonempty prefix of ones. The mask applies to keys only: query rows at and
after the padding transition remain defined and must still be removed by the
training loss mask when appropriate.

The independent FP32 oracle streams over 32-query and 64-key tiles without a
full 256-by-1024 matrix. Conservative explicit NumPy-array scratch accounting
is 334,400 bytes, including separate causal, key, and combined validity masks;
it remains accounting rather than a measured allocator peak and excludes
Python/NumPy object overhead and internal BLAS workspace. All six exact
production-shape results match a separate dense CPU oracle within the fixed
test tolerance.

Each case pins its mask and reference hashes and requires an all-valid-mask
control to fail on both all affected rows and the first affected row with
relative L2 above 0.02. This row-local gate prevents the single affected row
in `valid1023` from being hidden by 255 unaffected rows: its informational
whole-output relative L2 is only about 0.001954, while its one affected row is
0.0333121893.

| Case | Mask SHA-256 | Reference SHA-256 | Affected-row wrong-mask rel L2 | First-affected-row rel L2 |
|---|---|---|---:|---:|
| `valid768` | `41b4c32a488d09f1b6487b1d89e4b78d93e02dea44b0cf6dbf65fb1dd4286c53` | `017970569e9232a1d289ef1bc084c187a3e266a46250741962f3e2008d963ec1` | 0.3656457575 | 0.0372590756 |
| `valid769` | `cbc965937f4fc2c6b9a151a0024f4283e18ae1e3781198800d58f23e31b059f3` | `92c9ff029e1ded9b7e08f56d8e3b04e9ded198f6289221de47385fff667c5845` | 0.3649149871 | 0.0374931669 |
| `valid831` | `6c0ad7401a09409a707d995e0cc35ac1a7db579c8738262c847f9d6b8cd5b4a2` | `8ae299a67efdb7e8abb3f4d52842e248341d9d8076de2a8e9033da8357b538f5` | 0.3199291223 | 0.0323933496 |
| `valid832` | `53c0cfc02674331795d5243e14a0c67386fc9b2d901e4d9076938aac2c3fd5d9` | `1c01b2f7ed46b6b0a948b6873755989a572520bdb68a5de3ad31612db078c4b9` | 0.3187053494 | 0.0348590429 |
| `valid833` | `ce0382fba2e5be6f36fee68b6be584555b804b7168fbbfb311653e1aecad8a80` | `f56f62eb56fb8e5f0b7989f04cbd548fac24142e282eda9d5334e32374962617` | 0.3173608216 | 0.0326644662 |
| `valid1023` | `b70afc4fbdb3d8248e081b57a3b0ba543c0be859ea9c82e46d969fe669abcfc9` | `c5199af1c94d6afd600b1b6147699acf10854f43bbb42dc891170874d5d5af79` | 0.0333121893 | 0.0333121893 |

Runtime validation records aggregate metrics, the worst relative L2, cosine,
maximum error, and mean error over individual query rows, plus exact metrics
for the last unaffected, first affected, and second affected rows when those
rows exist in the final chunk. Both aggregate and every-row gates require
relative L2 below 0.01, cosine at least 0.9999, and maximum absolute error at
most 0.02. Candidate duration must be below the 100 ms hard limit and 75 ms
promotion ceiling.

The compile path pins and delegates the promoted nonzero-scale probe. It still
requires one q768 ROCm Triton call in each IR dialect, exact parsed and
independently checked raw `query_start=768`/`query_size=256` metadata, no outer
`while`, 6,295,552 argument bytes, 2,097,152 output bytes, zero alias bytes,
and no more than 64 MiB temporary memory. Counters permit exactly one lower,
compile, tuple placement, checked invocation, and retrieval, with no warmup,
replay, backward, accelerator reference/reduction, or model call.

After independent source review, each case would require a separate command
of this form. This is a pending command template, not evidence that it ran:

```bash
.venv/bin/python rocm/profile_rocm.py \
  --output /tmp/query-bounded-gqa-c256-t1024-valid768.telemetry.jsonl \
  --card card1 \
  --interval 0.1 \
  --baseline-seconds 2 \
  --timeout 120 \
  --sensor-grace-seconds 15 \
  --max-junction-temp-c 70 \
  --max-gpu-power-watts 315 \
  --max-vram-gib 2 \
  --min-host-available-gib 8 \
  --max-swap-gib 0.001 \
  -- .venv/bin/python rocm/probe_query_bounded_gqa_padding.py \
       --platform rocm --allow-gpu --case valid768 \
       --output /tmp/query-bounded-gqa-c256-t1024-valid768.jsonl
```

No case is promoted by this documentation or its CPU tests. A passing private
child artifact and profiler telemetry must be independently audited before
advancing to another case or to backward/model integration.

## GPU promotion gates

This prototype should remain disconnected from `dot_product_attention` until
all gates pass in fresh, profiler-controlled processes:

1. Compile only the Qwen shape at T=512 and inspect the generated artifacts as
   described above. The returned attention executable must not be invoked.
2. Run the exact single-forward T=512 analytic gate above. Do not add a replay,
   random input, GPU reference, padding case, or backward work to that process.
3. In fresh later gates, qualify the remaining T=512 forward-only input matrix
   before advancing through 1K, 2K, 4K, 8K, 16K, 24K, and 32K.
4. Run an arbitrary-cotangent VJP at the same buckets and compare sampled/full
   gradients where the reference fits.
5. Repeat padding boundaries including valid lengths 1, `T-1`, and `T`.
6. Repeat every promoted bucket enough times to expose replay and cleanup
   faults, with command buffers disabled and kernel/journal monitoring active.
7. Record maximum individual dispatch duration. The initial safety target is
   below 100 ms; reduce the query range to 256 if any call approaches it.
8. Only after 32K isolated attention passes, add a separate environment opt-in
   (for example `SKYRL_ROCM_QUERY_BOUNDED_GQA=1`) while retaining the current
   16K monolithic limit and default-off behavior for the old kernel.

The existing 16K monolithic measurement suggests 512-query ranges are a
plausible sub-100-ms starting point, but that is an inference, not a benchmark
of this implementation. Pallas/Triton code generation, register use, aliasing,
compile time from 192 calls per layer, and end-to-end launch overhead all remain
unmeasured. If the compile-only or 512-token GPU gate fails, the prototype must
not be enabled merely because CPU interpret correctness passed.
