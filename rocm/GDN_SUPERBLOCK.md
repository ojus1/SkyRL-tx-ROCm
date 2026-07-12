# Qwen3.5 GDN superblock CPU prototype

`skyrl/tx/kernels/qwen3_5_gdn_superblock.py` is an unwired semantic and
scheduling prototype for the highest-value bounded Gated DeltaNet stage. It is
not a GPU kernel and is not called by the model. The prototype keeps the exact
Qwen3.5-4B recurrence contract while exposing the intended 512/1,024-token
operation boundary:

- Q/K remain `[B,T,16,128]`. Value heads are represented as
  `[16 key heads, 2 value heads per key head]`; the prototype maps the size-two
  axis instead of forming `[B,T,32,128]` repeated Q/K tensors.
- an outer `lax.scan` carries the `[B,32,128,128]` FP32 state at superblock
  boundaries;
- each outer body prepares and immediately consumes the 8 or 16 chunks' WY
  `U`/`W` values; and
- `jax.checkpoint(..., policy=nothing_saveable)` covers normalization, WY
  preparation, and state/output execution together. Automatic reverse mode
  therefore recomputes `U`/`W` for one superblock instead of retaining them for
  the complete sequence.

Portable `lax.map` and `lax.scan` describe association and scheduling only. A
production HIP/FFI implementation should execute the two value-head groups in
one bounded launch and needs an explicit custom VJP; this prototype makes no
GPU launch-count or speed claim.

## Semantic evidence

The focused CPU suite exercises both 8- and 16-chunk schedules with real
64-token chunks. It compares output, final FP32 state, and nonuniform
scalar-loss VJPs of Q, K, V, g, beta, and initial state against both
`chunk_gated_delta_rule` and `recurrent_gated_delta_rule`. Covered lengths
include 511, 512, 513, and 1,025, with right-padding ending inside a chunk and
on either side of the 512/1,024 superblock boundary. Interior holes, left
padding, and an all-masked sequence are checked as identity transitions, and
masked-token input gradients must be exactly zero. The suite also checks the
BF16-output/FP32-state contract and a BF16 scalar-loss VJP against the current
chunk rule.

```bash
JAX_PLATFORMS=cpu .venv/bin/pytest -q \
  tests/tx/kernels/test_qwen3_5_gdn_superblock.py
```

Result on JAX/JAXlib 0.10.2: `11 passed`. The largest FP32 VJP differences from
the chunk reference were below `5e-7` in the tested 1,025-token case; the
largest differences from the recurrent reference were below `1.4e-6`.

The no-repeat grouping changes the reduction order for the two value-head
contributions to each Q/K cotangent. In the BF16 VJP test, Q and K gradients
differed from the repeated-head chunk rule by 0.656% and 0.658% relative L2;
the test gates each gradient at 2%. V, g, beta, and initial-state gradients were
identical in that case. This is a numerical gate, not evidence that the
portable automatic transpose is production-ready. The explicit reverse kernel
should accumulate the two Q/K contributions in FP32 before casting at its
interface.

## Logical per-superblock buffers

These are individual FP32 tensor sizes at exact batch-one Qwen3.5-4B geometry,
not an additive peak. `rhs` and `solution` each contain both U and W widths;
XLA may reuse them, while a tiled kernel need not materialize every listed
matrix.

| Logical tensor | 512 tokens / 8 chunks | 1,024 tokens / 16 chunks |
|---|---:|---:|
| FP32 boundary state | 2 MiB | 2 MiB |
| U + W | 16 MiB | 32 MiB |
| gamma | 0.0625 MiB | 0.125 MiB |
| decay mask | 4 MiB | 8 MiB |
| strict-lower correction | 4 MiB | 8 MiB |
| RHS | 16 MiB | 32 MiB |
| triangular-solve result | 16 MiB | 32 MiB |
| intra-chunk attention | 4 MiB | 8 MiB |
| corrected values | 8 MiB | 16 MiB |
| FP32 block output before BF16 boundary cast | 8 MiB | 16 MiB |

Avoiding the explicit BF16 Q/K head repeat removes 4 MiB of arguments at 512,
8 MiB at 1,024, and 256 MiB at 32K. It does not remove per-value-head W or
correction matrices: beta and g differ for each value head, so those tensors
are mathematically head-specific.

## Saved reverse residuals

JAX 0.10.2's private saved-residual inspection reports blocked primal inputs,
the FP32 scan-carry checkpoints, and a few scalar pad constants. It reports no
U, W, gamma, 64x64 correction, or FP32 full-sequence output residual. The exact
residual totals are:

| Sequence | Superblocks | Boundary-state residual | All reported residuals |
|---:|---:|---:|---:|
| 512 | 1 x 512 | 2 MiB | 10.094 MiB |
| 1,024 | 1 x 1,024 | 2 MiB | 18.188 MiB |
| 2,048 | 2 x 1,024 | 4 MiB | 36.377 MiB |
| 32,768 | 32 x 1,024 | 64 MiB | 582.031 MiB |

At 32K, full-sequence U+W would be 1,024 MiB. The 64 MiB state residual is the
intended reverse schedule: one 2 MiB checkpoint per 1,024-token superblock.
The other 518.031 MiB is blocked Q/K/V/g/beta/mask input data. Enclosing
whole-layer rematerialization determines whether those projection outputs are
stored or recomputed in an integrated model; this isolated result must not be
treated as model peak memory.

## CPU StableHLO and compiler memory evidence

`rocm/probe_gdn_superblock.py` forces the CPU backend before importing JAX. It
compiles but does not execute exact-shape programs and emits JSON containing
JAX/JAXlib versions, StableHLO counts, compiler memory estimates, logical
buffers, and saved residuals:

```bash
.venv/bin/python rocm/probe_gdn_superblock.py \
  --sequence-length 1024 --chunks-per-superblock 16
```

Lengths above 4,096 require `--allow-large-compile`. That flag does not enable
GPU execution; it only acknowledges that compiling the exact current-rule VJP
at long context can consume substantial host RAM. Reproducing the 32K row
therefore requires the flag and should be done only on a host with enough free
memory.

The following are JAX/JAXlib 0.10.2 CPU `CompiledMemoryStats` in MiB, not ROCm
allocator or speed measurements. "Current" receives already repeated 32-head
Q/K, matching the present model boundary. "Superblock" receives 16-head Q/K.

| T | Program | Arguments | Outputs | Forward temp | VJP temp |
|---:|---|---:|---:|---:|---:|
| 512 | Superblock | 10.094 | 6.000 | 53.168 | 299.851 |
| 512 | Current chunk rule | 14.094 | 6.000 | 56.125 | 266.408 |
| 1,024 | Superblock | 18.188 | 10.000 | 104.329 | 594.200 |
| 1,024 | Current chunk rule | 26.188 | 10.000 | 112.250 | 530.815 |
| 2,048 | Superblock | 34.375 | 18.000 | 175.831 | 721.148 |
| 2,048 | Current chunk rule | 50.375 | 18.000 | 224.500 | 1,059.629 |
| 32,768 | Superblock | 520.000 | 258.000 | 1,544.039 | 2,444.427 |
| 32,768 | Current chunk rule | 776.000 | 258.000 | 3,592.000 | 16,924.063 |

The long-context scaling is the useful result: the current chunk function
prepares full-sequence WY tensors before its chunk scan, while the outer
superblock scan bounds preparation. At 32K, the CPU estimate is 2,047.961 MiB
lower for forward temporary storage and 14,479.636 MiB lower for VJP temporary
storage. These figures cannot be transferred directly to GPU peak VRAM, but
they establish that the operation boundary removes the full-sequence storage
term.

The short-context result is equally important: automatic VJP temporary storage
is 33.443 MiB larger at 512 and 63.385 MiB larger at 1,024. Rematerialization
and the portable size-two `lax.map` introduce reverse machinery that only pays
off after multiple superblocks. StableHLO makes that cost visible:

| Exact 1,024-token lowering | Lines | `dot_general` | `while` |
|---|---:|---:|---:|
| Superblock forward | 466 | 6 | 3 |
| Current forward | 224 | 6 | 1 |
| Superblock VJP | 1,702 | 20 | 9 |
| Current VJP | 713 | 17 | 2 |

The superblock VJP contains one optimization barrier from the checkpoint. Its
inner tensor types use `[B,16,...]` Q/K and a mapped size-two value group; there
is no repeated Q/K input. The final `[B,T,32,128]` type is the required output,
not a Q/K expansion.

## Production blockers and decision

This is a viable, high-value long-context operation boundary, but the portable
automatic VJP is not the production implementation. A ROCm path still needs:

1. A bounded preparation kernel which tiles the 64x64 correction and
   triangular solve without simultaneously materializing RHS, solution, L,
   decay mask, and attention matrices.
2. A bounded state/output kernel which maps `hv -> hv // 2` directly and emits
   BF16 at the superblock boundary.
3. An explicit reverse-superblock custom VJP which reloads one 2 MiB boundary
   state, recomputes that block's U/W, reconstructs or locally checkpoints its
   eight/sixteen chunk states, accumulates the two value-group dQ/dK
   contributions in FP32, and propagates dV/dg/dbeta in a fixed order.
   Delegating this transpose to JAX leaves a 594.2 MiB CPU temporary at 1,024
   tokens and adds nine while loops.
4. Isolated GPU correctness and watchdog tests at 512 then 1,024 before any
   32K integration. This CPU work provides no evidence about HIP duration,
   occupancy, numerical drift from a tiled solve, or driver safety.
5. Tail scheduling. The portable oracle pads a physical tail to the configured
   512/1,024-token superblock, which is intentional for static semantic tests
   but wasteful for lengths just beyond a boundary. Production should choose
   an 8-chunk tail bucket when possible and use the existing 64-token path for
   a smaller remainder.

The recommended next implementation is therefore a three-stage FFI operation
(prepare, execute, reverse) with one-superblock scratch reuse, not wiring this
portable oracle into the model and not a whole-layer persistent kernel.
