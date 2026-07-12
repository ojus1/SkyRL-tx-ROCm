# Query-bounded native GQA prototype

Status: CPU/Pallas-interpret correctness plus exact T=512 guarded ROCm
compile-only structural prototype. Compilation has run, but the returned
attention executable has never been invoked and the prototype is not connected
to the model dispatcher.

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

This promotes only the exact T=512 structure through compile. It does not
promote execution, numerical correctness, padding, repeated launches, or any
larger bucket. The prototype remains disconnected from model attention.

## GPU promotion gates

This prototype should remain disconnected from `dot_product_attention` until
all gates pass in fresh, profiler-controlled processes:

1. Compile only the Qwen shape at T=512 and inspect the generated artifacts as
   described above. The returned attention executable must not be invoked.
2. Run forward-only at 512, 1K, 2K, 4K, 8K, 16K, 24K, and 32K, comparing finite
   outputs and sampled rows to a bounded reference.
3. Run an arbitrary-cotangent VJP at the same buckets and compare sampled/full
   gradients where the reference fits.
4. Repeat padding boundaries including valid lengths 1, `T-1`, and `T`.
5. Repeat every bucket enough times to expose replay and cleanup faults, with
   command buffers disabled and kernel/journal monitoring active.
6. Record maximum individual dispatch duration. The initial safety target is
   below 100 ms; reduce the query range to 256 if any call approaches it.
7. Only after 32K isolated attention passes, add a separate environment opt-in
   (for example `SKYRL_ROCM_QUERY_BOUNDED_GQA=1`) while retaining the current
   16K monolithic limit and default-off behavior for the old kernel.

The existing 16K monolithic measurement suggests 512-query ranges are a
plausible sub-100-ms starting point, but that is an inference, not a benchmark
of this implementation. Pallas/Triton code generation, register use, aliasing,
compile time from 192 calls per layer, and end-to-end launch overhead all remain
unmeasured. If the compile-only or 512-token GPU gate fails, the prototype must
not be enabled merely because CPU interpret correctness passed.
