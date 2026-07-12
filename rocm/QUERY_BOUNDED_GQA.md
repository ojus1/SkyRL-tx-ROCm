# Query-bounded native GQA prototype

Status: CPU/Pallas-interpret correctness prototype only. It has never been
compiled or executed on the GPU and is not connected to the model dispatcher.

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
sufficient evidence for ROCm. Before any GPU execution, a compile-only ROCm
gate must inspect optimized HLO/LLVM and assert all of the following:

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

## GPU promotion gates

This prototype should remain disconnected from `dot_product_attention` until
all gates pass in fresh, profiler-controlled processes:

1. Compile only the Qwen shape at T=512 and inspect the generated artifacts as
   described above. Compilation must not initialize a running benchmark.
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
