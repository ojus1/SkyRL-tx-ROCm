# gfx1100 quantized LoRA FFI route

This note turns the grouped W8/W4 semantics in
`skyrl/tx/kernels/quantized_lora.py` into a concrete ROCm compilation plan. It
is scoped to Qwen3.5-4B on the RX 7900 XTX (`gfx1100`). Nothing described here
is enabled in model code yet.

## What is proven locally

The installed stack is ROCm 7.2.4, Clang 22, Composable Kernel 1.2.0,
rocWMMA 2.2.0, hipBLASLt 1.2.2, and JAX/JAXlib 0.10.2. Inspection and
compile-only checks establish the following:

- Clang accepts gfx1100's signed IU8 and IU4 WMMA builtins. Both accumulate
  into eight lane-local INT32 values for a 16x16x16 wave32 operation. IU8 takes
  two `int32x4` fragments; IU4 takes two `int32x2` packed-nibble fragments.
- `rocwmma/internal/wmma_impl.hpp` supplies the gfx11 IU8 wrapper, including
  `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`. The installed public rocWMMA
  implementation has no corresponding IU4 wrapper.
- CK's `ck/tensor_operation/gpu/warp/wmma_gemm.hpp` names IU4 behind
  `CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4`, but the installed file has no IU4
  `wmma_type` implementation. It is not a ready W4A4 GEMM path in this package.
- hipBLASLt ships gfx1100 INT8 Tensile code objects, including I8-input/I32
  output variants. Its scale modes do not express the oracle's scale for every
  `(row, K-group)` and `(K-group, output)` pair. Its epilogues also do not
  implement an arbitrary rank-8 LoRA branch.
- JAXlib supplies typed-FFI headers at
  `jaxlib/include/xla/ffi/api/{api.h,ffi.h}`. They expose typed buffers,
  `PlatformStream<hipStream_t>`, and `ScratchAllocator`; no pybind11 dependency
  is required to register an exported handler through `ctypes` and
  `jax.ffi.pycapsule`.

The compile proof is:

```bash
rocm/compile_probes/compile_quant_wmma_gfx1100.sh
```

It runs `hipcc -O3 --genco --no-gpu-bundle-output
--offload-arch=gfx1100`, writes only to a private directory under `${TMPDIR:-/tmp}`,
and disassembles the code object. A post-upgrade rerun on ROCm 7.2.4 on
2026-07-13 reproduced a 6,304-byte HSACO containing both:

```text
v_wmma_i32_16x16x16_iu8 v[0:7], v[8:11], v[12:15], v[0:7]
v_wmma_i32_16x16x16_iu4 v[0:7], v[8:9],  v[10:11], v[0:7]
```

The script does not enumerate a device, open `/dev/kfd`, or launch a kernel. It
refuses an architecture other than gfx1100 and fails if hipcc, LLVM inspection,
either kernel symbol, or either instruction is unavailable. Set
`SKYRL_KEEP_COMPILE_PROBE=1` only when the temporary HSACO/disassembly must be
retained for inspection.

This proves compiler and ISA availability only. It says nothing about correct
fragment layout, numerical quality, occupancy, latency, or training speed.

## Route decision

| Route | Concrete implementation | Decision |
|---|---|---|
| W8A8 group-64 | Custom HIP wave32 GEMM using the proven IU8 builtin | First native-integer speed prototype |
| W4A4 group-64 | Custom HIP wave32 GEMM using the proven IU4 builtin | Second prototype, after W8A8 is stable |
| Whole-K hipBLASLt INT8 | Vendor GEMM with coarser scalar/outer-vector scales | Useful comparison, but a different quantization format |
| Group-64 hipBLASLt | One GEMM per group plus a separate reduction | Reject: too many calls or an `M x groups x N` partial buffer |
| Installed CK IU4 | Experimental selector without a complete installed implementation | Reject for this environment |
| Single whole-layer kernel | Persistent kernel with cross-stage global synchronization | Reject; use bounded subdispatches |

Direct builtins are the shortest path for both formats. rocWMMA can be used to
prototype IU8 tile movement, but using the same explicit fragment loader for
IU8 and IU4 avoids making W4 depend on an incomplete wrapper. The packed W4
checkpoint layout is not automatically a WMMA fragment layout: an offline or
load-time permutation must create the canonical device weight. Do not perform
that permutation on every forward.

## Default-off Pallas W8A8 feasibility rung

`skyrl/tx/kernels/rocm/w8a8_lora.py` now provides a smaller experiment before
the production HIP work. JAX 0.10.2's local Pallas primitive assigns signed
integer dots an INT32 result, and its Triton lowering accepts signed INT8
inputs and emits a Triton `dot`. That source-level fact does **not** prove the
gfx1100 backend selects an IU8 matrix instruction; only retained code-object
disassembly or equivalent ISA evidence can establish that.

The prototype is explicit opt-in and is not imported or selected by model
code. It keeps W8 group-64 codes/scales compact, quantizes A8 once per bounded
row superblock, computes one INT32 dot per K group, applies the row and weight
scales in FP32, and adds rank-8 LoRA in the same epilogue before one BF16 cast.
It also exposes a base-only custom-VJP route so a future compact layer can use
the existing `LoRAMixin.apply_lora` implementation for no-adapter and genuine
multi-adapter batches; the fused epilogue is only the already-selected
single-adapter fast path.
Its relaxed input pullback dequantizes one W8 tile to BF16 and uses a BF16 dot
with FP32 accumulation. The source maps all forward and backward row work in
at most 512 rows by default, with an enforced 2,048-row maximum; one small
Pallas program is not incorrectly treated as proof of a bounded whole-grid
launch. Each forward program still scans the capped K domain, up to 144
group-64 dots, and each input-pullback workgroup scans the capped N domain, up
to 18,432 columns. The physical duration of both scans remains unproven.
The experiment also caps K at 9,216 and N at 18,432, the largest non-vocabulary
Qwen3.5-4B projection dimensions. The tied 248,320-column head is deliberately
outside this path.

CPU Pallas-interpreter tests currently prove only semantics. Logical output
tails are now zero-padded outside Pallas to complete physical `block_n` tiles;
the kernels themselves contain no row/column tail predicates. The fixed
`M=3,K=64,N=17` case therefore calls Pallas with physical `N=32`, slices back to
17 columns, and remains bitwise equal to the portable W8A8 forward. Against the
stronger portable FP32-dequant backward, relative L2 errors were 0.3341%
for `dX`, 0.2435% for `dA`, 0.3220% for `dB`, and 0.0000703% for `dscale`, all
inside the retained 1% CPU regression gate. A `M=1025` JAXPR test proves a
three-iteration row scan whose sole forward Pallas grid covers exactly 512
rows per iteration. A separate `M=33,K=128,N=19` interpreter test executes
three 16-row superblocks and checks bitwise forward equality plus all four
pullbacks against the same 1% gate. These are CPU structural/numerical results,
not GPU lowering, runtime, memory, or speed results.

The first useful model family remains all 32 MLP gate/up matrices, each exactly
`K=2560,N=18432`. Compact W8 codes plus BF16 group-64 scales save exactly
1,462,763,520 bytes (1.362304688 GiB) versus their BF16 kernels. This one shared
callsite is the smallest projection family with more than 1 GiB capacity
upside. It must retain no BF16 shadow. The tiny full-tile case has now passed
guarded gfx1100 compilation and an offline ISA/thunk audit, but runtime
correctness, realized memory, throughput, backward, and model-shape gates still
block NNX/checkpoint integration.

### Guarded gfx1100 qualification entrypoint

`rocm/run_w8a8_lora_forward_gate.py` and
`rocm/probe_w8a8_lora_forward.py` implement the compile-only rung and a
separately guarded one-shot execute rung. Their default invocations emit an
abstract refusal without importing JAX. The ROCm
controller holds the global launch lock across an exact suspended/unowned
RX 7900 XTX baseline, hash-bound `profile_rocm.py` supervision, the child, three
consecutive one-second idle-handoff samples, and a final whole-boot journal
check. The profiler is itself loaded from hash-verified source under
`python -I -S -B`, with its Linux `psutil` runtime hash-bound before imports.
The child separately validates the inherited lock, exact PCI/render
identity, disconnected AMD display, unowned KFD/render nodes, fixed stack and
selected native-library/source hashes, and the exact sole environment value:

```text
XLA_FLAGS=--xla_gpu_enable_command_buffer=
```

The fixed source requests the base-W8A8 plus LoRA-B/scaling Pallas epilogue at
logical `M=3,K=64,N=17`, physical `M=16,N=32`, group 64,
`block_m=block_n=16`, and physical grid `1x2`.
Activation quantization and the LoRA-A contraction remain ordinary JAX work
outside that Pallas call, so this is not an entire-forward megakernel. The
compile phase invokes no returned executable. A separately guarded
`--phase execute` branch is implemented. Its first released invocation of the
former masked-`N=17` artifact hit the exact five-second dispatch watchdog and
was killed/reaped without an AMDGPU fault or reset; the bracket contained five
GPU kernels plus output readiness, so it does not prove that Pallas itself
stalled. The corrected full-tile source has since been freshly compiled twice
and audited offline, but it has not run. The exact nested gfx1100 object and
six-thunk inventory were committed at `e4152`; the latest compile then exposed
one caller-derived run path inside the otherwise pinned executable record. The
equal-length normalized whole-record pins, CPU regressions, and independent
source review are now complete. This still does not release an executable. A
separately authorized rung may permit one six-leaf input transfer, one compiled
forward invocation with readiness, one output transfer, and a host-only
comparison against the immutable oracle. The gate exposes no backward,
warmup, repetition, replay, GPU reference, device-side error reduction, graph,
model, or benchmark path, and it cannot promote the runtime or model path.

The controller fixes sampled-sysfs thresholds of 2 GiB VRAM, 90 C junction,
and 400 W `power1_average`, plus an 85 C child launch gate, 300-second child
timeout, 330-second independent supervisor watchdog, and relaxed 0 GiB
host-available/8 GiB swap limits. It records and bounds the largest observed
sample gap, but this remains reactive evidence rather than a continuous
measurement. The child also refuses a reported hardware `power1_cap` above
400 W; that configuration still does not prove that every instantaneous
electrical transient stayed below the cap. The process uses a fresh private
7.5%-of-VRAM BFC allocator limit and fresh JAX,
Triton, compiler-dump, and `TF_XLA_HSACO_CACHE_DIR` directories. Longer compile
startup is intentional and cannot be replaced by command-buffer capture. The
BFC setting and XLA buffer-plan estimate are allocator/compiler constraints,
not independent proof of peak physical VRAM.

The safe defaults are:

```bash
.venv/bin/python rocm/run_w8a8_lora_forward_gate.py
.venv/bin/python rocm/probe_w8a8_lora_forward.py
```

After committing a clean source tree, a fresh compile-only artifact can be
requested with:

```bash
umask 077
run_dir="/tmp/skyrl-w8a8-compile-$(date +%s)"
XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  .venv/bin/python -I -S -B \
    -X "pycache_prefix=$run_dir/python-cache" \
    rocm/run_w8a8_lora_forward_gate.py \
    --platform rocm --phase compile --allow-gpu --run-dir "$run_dir"
```

The exact former masked execute capability was consumed and must not be
retried. The corrected source is committed and freshly compiled. Its exact
object/thunk and caller-path-normalized record pins are updated, CPU-tested,
and independently reviewed. That offline GO is not execution authorization;
only a separate one-shot request may use this command form:

```bash
umask 077
run_dir="/tmp/skyrl-w8a8-runtime-$(date +%s)"
XLA_FLAGS=--xla_gpu_enable_command_buffer= \
  .venv/bin/python -I -S -B \
    -X "pycache_prefix=$run_dir/python-cache" \
    rocm/run_w8a8_lora_forward_gate.py \
    --platform rocm --phase execute --allow-gpu --run-dir "$run_dir"
```

Even a passing invocation would establish only fixed logical
`M=3,K=64,N=17`, physical `M=16,N=32`, group-64, rank-8 forward correctness
under the retained W8 gate. It would not
establish promotion, throughput, memory savings, backward correctness,
warmup/replay safety, or model integration.

Compile-phase controller completion by itself does not establish runtime
correctness or native INT8 ISA. It establishes only an unpromoted compile
diagnostic whose private probe, telemetry, compiler, handoff, and final-journal
artifacts must be reviewed together. Optimized HLO proving one Triton custom
call is not an INT8 ISA proof, and surrounding optimized-HLO fusions may still
launch separate kernels. The probe therefore persists raw StableHLO and
optimized HLO and marks the result `passed_compile_diagnostic_unpromoted`. A
separate offline audit must correlate an actual retained code object with the
forward symbol, gfx1100 disassembly, and resource use. Runtime promotion
additionally requires a guarded dispatch trace and host numerical comparison.
If that chain is not available, the result remains only a Pallas Triton
custom-call compile proof.
Version/origin checks and selected binary hashes do not constitute a complete
hash closure over every JAX, NumPy, ML-dtypes, ROCm, and system runtime file;
the untouched installed stack remains a recorded trust assumption.

The structural gate treats raw forward-name occurrences as diagnostics only:
XLA may repeat that name in op metadata and the embedded Triton configuration
for one custom call. Both the probe and controller instead mask quoted payloads
and comments, require the exact scoped `mhlo.backend_config`/`backend_config`
`name`, preserve duplicate target/name attributes as failures, bind the sole
call to the unique public entry, and require its SSA result to reach the entry
return/ROOT through data operands. Metadata, nested-map, comment, payload,
dead-helper, dead-result, and control-predecessor decoys all fail closed.

### Qualified tiny compile and native ISA result

Exact revision `da9bf1ee6195921fd7c8cf9055a3dc8d4a1ed704` completed the
compile-only rung in `/tmp/skyrl-w8a8-compile-1783972228`. The controller
returned `passed_compile_diagnostic_unpromoted`, the worktree remained clean,
and the returned executable was invoked zero times. Lowering and compilation
took 1.134153 and 0.719444 seconds. The compiler-reported argument, output, and
temporary sizes were 2,806, 102, and 3,600 bytes. StableHLO and optimized HLO
had SHA-256 values
`0e57123dd6c1d4355b7ccbbf0f7908db686ceaafc6d26afbb237811f8861ecdf`
and
`7adf78dc72eebed7a4eaa0c1e88f7ba43ecea0199d54817fcaeb2fcc3b5aa0dc`.
Both independently parse as one entry-owned Triton custom call whose result
feeds the public output, with the exact forward name and no backward, outer
loop, graph, capture, command-buffer, or replay marker.

`rocm/inspect_w8a8_lora_isa.py` reproduces the retained-code audit without
importing JAX or opening a GPU device. It binds the private cache entry by
SHA-256, bounds zstd and Snappy output before accepting it, decodes the exact
IFRT/Riegeli split record structure, inventories every embedded ELF, and uses
hash-pinned ROCm LLVM tools. The decoded record sizes are
`[56, 0, 1920, 0, 52425]`. The 1,920-byte wrapper record contains an empty
1,904-byte ELF plus a 16-byte ROCm module identifier; its `.text` is empty and
it has no kernel, so it is explicitly rejected as ISA evidence. OpenXLA
deliberately appends a nanosecond timestamp and random 64-bit value to ROCm
binaries so separately loaded identical HSACOs receive different module hashes
([upstream implementation](https://github.com/openxla/xla/blob/5befee6e1873bddf3280d01e2a7c0c78e46e12be/xla/service/gpu/gpu_executable.cc#L498-L504)).
The verifier therefore pins the deterministic 1,904-byte empty ELF, requires
the exact 16-byte identifier shape, reports both values and its digest, and
continues to integrity-bind every byte through the caller-supplied whole-cache
SHA-256. The final record contains five ELFs. Selection is by the unique exact
function/descriptor pair rather than record position; an ELF that defines the
expected symbol alongside any decoy now fails closed.

The retained result can be rechecked offline with:

```bash
cache=/tmp/skyrl-w8a8-compile-1783972228/compiler-artifacts/jax-cache/\
jit_candidate-1d44db98de8b8ab596105a0991d6a240efebc1d3d29ab28606b6365a828cb645-cache
/usr/bin/python3 -I -S -B rocm/inspect_w8a8_lora_isa.py "$cache" \
  --expected-cache-sha256 \
  d00d54b9e684852a1d933eafb332e058c5b232d79f687dc936951887e3f646a2
```

The selected code object is exactly 8,440 bytes with SHA-256
`606a80a508317af303966e5c2ca357d138d08828949c0dbfdcd73ccde1726389`.
It binds `skyrl_qwen35_w8a8_lora_forward` to
`amdgcn--amdhsa-amdgiz-gfx1100`, reports 34 SGPRs, 62 VGPRs, zero SGPR/VGPR
spills, zero private segment, wave32, maximum workgroup size 128, and maximum
grid `1x2x1`. Its disassembly contains exactly four
`v_wmma_i32_16x16x16_iu8` instructions, all with signed-operand metadata
`neg_lo:[1,1,0]`. Paired with the exact HLO/custom-call evidence, this proves
native RDNA3 signed-INT8 WMMA code generation for the fixed tiny Pallas
forward. It does **not** prove that the code was dispatched, that its result is
correct, or that it is faster than BF16. The Riegeli internal checksum
algorithm is not reimplemented; the retained artifact is instead anchored by
the whole-cache SHA-256 plus exact boundaries and decompression return codes.

The guarded compile consumed at most 867,360,768 bytes of physical VRAM,
reached 51 C junction temperature and 113 W sampled average power, and used no
additional swap. All 1,090 measured telemetry samples were within the fixed
limits, with a maximum sample gap of 0.0899 seconds. The process returned to
the exact 27,947,008-byte VRAM and 15,966,208-byte GTT baseline, remained
headless and unowned, reached runtime suspend for three consecutive handoff
samples, and left the whole-boot AMDGPU journal clean. These are compile and
native-code-generation results, not performance measurements.

### First guarded forward attempt: failed closed before release

Exact revision `2b38bfb380728e7a1920d599e27b5c2ab268a0be` made the first
guarded one-shot attempt in `/tmp/skyrl-w8a8-runtime-1783980961`. Lowering and
compilation completed in 1.119482 and 0.757109 seconds, but the then-current
offline inspector incorrectly pinned the complete empty-wrapper record. The
underlying 1,904-byte ELF was byte-identical to the qualified object with
SHA-256
`bf465081edca1fa73a8d1d73e9cc0a354d22038c47f21bc9f4e418388c8fd563`;
only OpenXLA's deliberately variable 16-byte module identifier differed. The
fresh nested object still contained the unique exact 8,440-byte W8 HSACO with
SHA-256
`606a80a508317af303966e5c2ca357d138d08828949c0dbfdcd73ccde1726389`.

The probe emitted `compiled_unreleased` and then `error`. It emitted no
`fresh_isa_qualification`, `executable_released`, `input_device_put`,
`dispatch_started`, `dispatch`, `device_get`, numerical-validation, or
completion record, and the compiled executable invocation count remained
zero. This attempt is therefore neither runtime-correctness nor promotion
evidence. The controller killed and reaped the complete scope in 67.1 ms,
restored the exact VRAM/GTT idle baseline with the runtime suspended and device
nodes unowned, and observed a clean whole-boot AMDGPU journal. Independently
recomputed compile peaks were 867,332,096 bytes physical VRAM, 51 C junction,
and 101 W sampled average power.

After correcting only the nondeterministic-wrapper invariant, the CPU-only
inspector passes both the prior cache
`d00d54b9e684852a1d933eafb332e058c5b232d79f687dc936951887e3f646a2`
and the failed-attempt cache
`b1009c207c0c78bfe7a54c4fae568133034f2e6d0381367365cba10531bdec19`.
No GPU access or executable dispatch is involved in that offline regression.

### Masked timeout and corrected full-tile compile

The one guarded retry of the old masked object was consumed at revision
`8eb3c7d7` in `/tmp/skyrl-w8a8-runtime-1783982504`. Its five kernels and final
D2D slice did not complete inside the five-second bracket around the compiled
call and output readiness. The controller killed and reaped the complete
scope, restored the exact suspended/unowned baseline, and found no AMDGPU
fault or reset. This cannot identify the stalled stage and supplies no
correctness or performance result. The exact masked 8,440-byte object and this
timed-out executable must never be retried.

Exact revision `f6b26775456e5c2aa92dd621b26f3752f111ad00` first completed the
corrected full-tile compile. After committing the exact six-thunk and native
object pins at `e4152e47900086bfe2f58acf94d9f9a765b99aa5`, a fresh
compile-only qualification completed in
`/tmp/skyrl-w8a8-compile-1783987495`. The controller returned
`passed_compile_diagnostic_unpromoted`, with zero ISA qualifications,
returned-executable invocations, device transfers, dispatches, or releases.
Lowering took 1.112806747 seconds and compilation took 0.720117240 seconds.
The 15,351-byte StableHLO has SHA-256
`3eac9283c56d35e9df7f3fb42d7ab62fba851b74ca7f5ea09ee746f1587f1772`;
the 22,853-byte optimized HLO has SHA-256
`f05cc7cefb15832b90a24088fad868175034874534b8193e82430d4840899406`.
The optimized-HLO difference from the first corrected compile is source-line
metadata. Both prove one physical `[16,32]` Pallas result followed by the final
`wrapped_slice` to logical `[3,17]`. Compiler accounting remains 4,036 bytes
of arguments, 102 bytes of output, 3,600 bytes of temporaries, 7,738 bytes
total, and 1,920 bytes of generated code.

The fresh retained cache is 19,332 bytes with SHA-256
`bf40785e7dd4d0a07336ba814e8d7fde59c2d5525e50f8317134999119802604`.
Its decoded record sizes are `[56, 0, 1920, 0, 52909]`. The raw executable
record is 52,909 bytes with path-dependent SHA-256
`696fa3761f183e49139ca957c1d8bd337699451560f8f233df5e0b4cbdea556b`.
The CPU-only inspector requires exactly one caller-derived canonical 101-byte
autotune-cache path at field-one/record offsets 19,901/19,905 and replaces only
that byte range with an equal-length neutral token. The normalized record is
still a valid 52,909-byte protobuf and has SHA-256
`8060df67a90b7e0827672aa4c349d66f51a50b13120345e698ea95454c6acc08`;
its normalized 20,600-byte field one has SHA-256
`1cac7332465fe69bd9d4ae2a53dbd0454a5f3ca4fd28bbdbd400a66a30dde1cd`.
The whole normalized-record hash pins every non-path byte, framing, thunk, and
embedded ELF. The prior corrected cache fails this fresh normalized hash due
to its different probe source-line metadata. Because the two builds used
different source, they do not empirically prove same-source path-only
reproducibility. A bounded wire/ELF audit found exactly six ordered
custom-kernel thunks:

| Order | Kernel | Grid | Threads | LDS (bytes) | ELF bytes / SHA-256 |
|---:|---|---|---:|---:|---|
| 0 | `input_pad_reduce_fusion` | `2x1x1` | `256x1x1` | 0 | 4,080 / `06a6035fabadbc8de4d7d201fa51ad2b9383a37faa84e4a0b51d9587fa3d8c7f` |
| 1 | `loop_convert_fusion` | `8x1x1` | `128x1x1` | 0 | 3,944 / `8db071b2d0e93475f713c566d984a940155ed293e63adf53b62a53288fada685` |
| 2 | `gemm_fusion_dot_general_1` | `1x1x1` | `128x1x1` | 8,192 | 7,408 / `c45a0fb7f236f7b16dbdfedb905dd116a02006c16c43d6dd687c30ccedf2eaf1` |
| 3 | `loop_select_fusion` | `1x1x1` | `16x1x1` | 0 | 3,424 / `8e7f454a584324b303ab299d22e4a3d4ee956f29bc36c64c030256fd24068a71` |
| 4 | `skyrl_qwen35_w8a8_lora_forward` | `1x2x1` | `128x1x1` | 1,024 | 7,160 / `87a2ae903547258a4b107fad17797147c417d8ca35cc600bc35d77e46323368f` |
| 5 | `wrapped_slice` | `1x1x1` | `51x1x1` | 0 | 3,416 / `476174a6aa35385fa65e84356f63b196540840c4ac782985b6ecf744b30c4799` |

The later fresh compile at commit
`5981d493d0edefd9b01c92ef8cfff6c6c2be4cee` in
`/tmp/skyrl-w8a8-compile-1784038127` retained the identical 15,351-byte
StableHLO and exact W8 Pallas object, but the per-fusion autotuner selected
`block_m=block_n=32` rather than 16 for the separate LoRA-A BF16 GEMM. Its
cache is 18,839 bytes with SHA-256
`13d349e2bbbde57f1b84e2116ca120cab217b89ef49db13964660c02abe657dc`;
the normalized executable is 53,166 bytes with SHA-256
`4a7fc5e78b508cca93db2abfe209100a56153a123372bb25aa964c0cbb124985`,
and its normalized 20,600-byte HLO field has SHA-256
`9978dc0830323f4331dcb7c537fcbc56d8263be070d6f545b43191a8e651085b`.
Only thunk 2 changes: its serialization is 7,922 bytes with SHA-256
`0a6c802363f4c8dc30ddfebb195cccc66d4dadc4138db5860736c879de132a2d`,
its dynamic LDS is 16,384 bytes, and its 7,664-byte ELF has SHA-256
`9ab0e3abac1983fcb44279f9fe1b40da01186a6d674e271b6c103cbb69c40b2a`.
The auxiliary ELF uses 14 SGPRs, 159 VGPRs, and no spills; the BM16 object
uses 14 SGPRs, 164 VGPRs, and no spills.

The first allowlist revision recognized exactly two atomic historical-artifact
variants: `lora_gemm_bm16_bn16` with complete-contract SHA-256
`b7d543d6bf2aff9913221b1a438851fc7eec825d98cc9427b7178804d143db57`,
and `lora_gemm_bm32_bn32` with complete-contract SHA-256
`75ce7e3c82b4219a17391f3f3019c3fbef84dfdf6c924cb3051d4b7d884ae0c7`.
Those hashes remain historical evidence rather than current accepted inputs.

The allowlist source layout was then frozen at commit `0fec1b67`, with the
candidate definition fixed at line 2,174. Three clean compile-only seeds under
that identical layout completed in
`/tmp/skyrl-w8a8-compile-1784041485`,
`/tmp/skyrl-w8a8-compile-1784041690`, and
`/tmp/skyrl-w8a8-compile-1784041853`. The first two independently selected
BM32 and produced the same normalized 53,166-byte record SHA-256
`989798f1183a243fe074491578827e4b04bf2d0eb25ca127f0a3b06f93050b94`
and normalized 20,600-byte HLO SHA-256
`577bdf1c685ce7553f3af8ff8ab6b2125247f2cc7a15ee47f4c0e7916277b03f`.
The third selected BM16, with normalized 52,909-byte record SHA-256
`94e1a986416c6b1b0b3d249b5ff41c2fc11dec215612a66c21d28a15968d49bc`
and normalized HLO SHA-256
`aca6770fd14a7d002ad465bfe8ac09c22f77b33b5d38ca0b2bd95c13734349d5`.
All thunk serializations, launches, nested ELFs, and the exact W8 object match
their historical BM16/BM32 counterparts.

The verifier now accepts only these two current-layout atomic contracts:
BM16 contract SHA-256
`d2859c3fd661ca42f7ce2231d3b090af59e53210fad860519df7695a7856a947`
and BM32 contract SHA-256
`749fff3d982c91f738c7b7c5c44d4d7120c9d24f439d10df90f20ae7a5890766`.
Unknown configurations, historical top hashes, and mixed BM16/BM32 tuples
fail closed. Pin substitutions are equal-length and preserve every probe
source line. Runtime remains blocked until a clean post-pin compile passes the
child pre-dispatch inspector; the controller repeats that inspection
independently as postflight attestation after the supervised child exits.

Across the three seeds, candidate executable invocations remained zero. Peak
sampled power was 118 W, peak junction temperature 61 C, peak VRAM 867,364,864
bytes, and swap remained 24,576 bytes. Every run ended with a clean AMDGPU
journal, no `/dev/kfd` owner, and the required idle handoff.

Across the two earlier BM16 corrected builds, all six thunk serializations and
all seven embedded ELFs are byte-identical. The first four objects also remain
byte-identical to the masked build. The corrected Pallas object targets
gfx1100, uses 34 SGPRs and 105 VGPRs, reports no spills, and contains exactly
four signed IU8 WMMAs and nine barriers. Its sole forward branch is after all
barriers. The former final D2D copy is the actual `wrapped_slice` kernel. This
complete offline evidence gives a GO for the normalized exact
inspector/probe/controller pins, not a GO to dispatch. It proves no runtime
correctness, throughput, realized memory saving, backward, or promotion.

The fresh compile peaked at 867,352,576 bytes physical VRAM, 22,405,120 bytes
GTT, 6,423,089,152 bytes host RAM used, 51 C junction temperature, and 113 W
sampled average power. Swap remained exactly 20,480 bytes, the maximum sample
gap was 0.082857711 seconds, the whole-boot AMDGPU journal remained clean, and
three handoff samples restored the exact suspended/unowned 27,947,008-byte
VRAM and 15,966,208-byte GTT baseline.

After the normalized pins, tests, and independent review, any separately
authorized rung remains exactly one host-checked invocation of the corrected
tiny forward. Only after that result will the ladder change one risk dimension
at a time: signed base-only semantics;
`K=128`; three row superblocks; small base/fused VJPs; K-scan lengths 8/40/144;
N-scan lengths 8/32/128/288; then the first real `K=2560,N=18432` gate/up
rectangle. The artificial `K=9216,N=18432` rectangle is not a model shape and
must not be run. Every rung retains a strict sub-100-ms candidate-dispatch gate
and a fresh-process exact-idle handoff. The user's 3% allowance applies to
Pallas attention gradients, not this quantized projection; the W8 CPU and
forward gates deliberately remain at 1%.

## Forward operation

For flattened input `X[M,K]`, rank `R=8`, and group size `G=64`, the JAX-visible
operation should enqueue several short kernels on JAX's existing HIP stream:

1. Quantize each BF16 input row/group once, producing packed `Xq` and FP32
   `sx[M,K/G]`. This avoids rescanning the same input for every output tile.
2. Compute `Z = X A` from the original input and BF16 LoRA A with FP32
   accumulation. Keep `Z[M,8]` in FP32 to match the semantic oracle.
3. Run tiled integer base GEMM. For each output tile and each K-group, issue
   four K=16 WMMA operations into INT32, convert that group accumulator to
   FP32, multiply by `sx[m,g] * sw[g,n]`, and add it to the FP32 output
   accumulator.
4. In the output epilogue, add `scaling * sum_r Z[m,r] * B[r,n]` in FP32 and
   cast once to the requested BF16 output. Optional bias/residual/gate fusion
   must be a separate, explicitly tested ABI variant.

At 32K tokens and K=2,560, materialized A8 uses 80 MiB plus 5 MiB of FP32
row/group scales; packed A4 uses 40 MiB plus the same scales. FP32 `Z` is only
1 MiB. These are per-live-projection workspaces, not tensors to retain across
all 32 layers. With rematerialization, backward should recompute them.

Group-local INT32 accumulation is comfortably safe. Symmetric W8 codes in the
oracle are in `[-127,127]`, so the worst absolute G=64 accumulator is
`64*127*127 = 1,032,256`. W4 is `64*7*7 = 3,136`. Reset the INT32 accumulator
at every group because the distinct scales require an FP32 rescale before
summing groups; a whole-K integer accumulator is not equivalent.

Quantization must reproduce the oracle's FP32 absolute-max reduction,
zero-group scale of one, round-to-nearest-even, symmetric clipping, and signed
two's-complement nibbles. Weight scales remain compact BF16 but are converted
to FP32 before multiplication. RMSNorm, attention/GDN state, and loss
reductions remain FP32/BF16 as specified in `MEGAKERNELS.md`; A4/A8 applies
only at the base-linear boundary.

## Backward contract

The first production backward should preserve the existing straight-through
contract, not invent a fast but different quantized gradient:

```text
dX_base = dY @ dequant(W).T
R       = scaling * dY @ B.T
dX      = dX_base + R @ A.T
dA      = X.T @ R
dB      = scaling * (X @ A).T @ dY
dscale  = sum(dY * ((X @ A) @ B))
```

- Implement the initial `dX_base` exactly: load a compact weight tile, convert
  its codes and BF16 scales to FP32, and contract it with FP32 `dY` using FP32
  products and accumulation. Never materialize a complete BF16/FP32 base
  weight or save it in the VJP residual. A dequantize-to-BF16 WMMA path changes
  the documented gradient because `code * scale` is not generally exactly
  representable in BF16; treat that as a separately gated relaxed-backward
  experiment rather than the oracle-preserving implementation.
- Keep `X`, compact codes/scales, A, B, and scaling as the custom-VJP residual.
  Frozen codes/scales have no cotangent.
- Compute `R`, `dA`, `dB`, and `dscale` from the original BF16 values with FP32
  products/reductions. Token superblocks should write unique partial `dA`/`dB`
  buffers followed by a fixed-order reduction; do not make nondeterministic
  atomics the default.
- Cast final `dX`, `dA`, and `dB` once to their declared dtypes. Quantizing dY
  for a native integer transpose GEMM is a later accuracy experiment, not the
  W8A8/W4A4 forward implementation.

This means initial training speedups will be smaller than forward-only GEMM
speedups. Backward still benefits from compact frozen-weight residency, but
the exact tiled FP32 dequantize/dot path has no IU4/IU8 matrix-rate advantage.

## Typed-FFI boundary

Use separate targets for W8 and W4 rather than a hot-path bit-width branch.
The forward ABI should take row-major BF16 X, signed W8 or packed U8 W4 codes,
BF16 weight scales, BF16 A/B, and FP32 scaling, and return BF16 Y. If internal
subdispatches need `Xq`, `sx`, or `Z`, make them explicit auxiliary FFI outputs
in the first implementation so XLA owns their stream-safe lifetime. Only move
them to `ScratchAllocator` after its asynchronous lifetime is verified on this
JAX/ROCm plugin; never synchronize the device merely to free scratch.

The C++ handler shape is:

```cpp
xla::ffi::Error Forward(
    xla::ffi::BufferR2<xla::ffi::BF16> x,
    xla::ffi::BufferR2<xla::ffi::S8> w8,  // U8 for packed W4 target
    xla::ffi::BufferR2<xla::ffi::BF16> sw,
    xla::ffi::BufferR2<xla::ffi::BF16> a,
    xla::ffi::BufferR2<xla::ffi::BF16> b,
    xla::ffi::BufferR0<xla::ffi::F32> scaling,
    xla::ffi::ResultBufferR2<xla::ffi::BF16> y,
    hipStream_t stream);
```

Bind the stream with `Ctx<PlatformStream<hipStream_t>>()`, export with
`XLA_FFI_DEFINE_HANDLER_SYMBOL`, load by `ctypes`, wrap with
`jax.ffi.pycapsule`, and register for platform `ROCM` with typed-FFI API version
1. The Python call uses `jax.ffi.ffi_call` (custom-call API version 4) and
explicit row-major layouts. Wrap forward/backward calls in `jax.custom_vjp`;
raw FFI has no autodiff rule. Retain the `ctypes.CDLL` object for the lifetime
of the process; the capsule contains only a function pointer and must not
outlive the loaded shared library.

Handlers enqueue work on the supplied stream and return without
`hipDeviceSynchronize`. Check launch submission with `hipPeekAtLastError` and
convert failure to `xla::ffi::Error::Internal`. Validate every dimension,
group-size/rank invariant, buffer byte count, and alignment in both Python and
the handler.

## Bounded dispatch and acceptance gates

One handler may enqueue multiple dependency-ordered kernels, but no individual
kernel should cover an unbounded full projection. Start with conservative M/N
superblocks (for example M<=2,048 and N<=1,024), profile their worst Qwen shape,
and only enlarge them while the guarded profiler keeps every dispatch below the
chosen watchdog budget. Tail kernels handle nonmultiples; K is padded to 64.
There is no inter-block synchronization inside a launch.

The compile proof is not a reason to enable either route. Required gates are:

1. Exact pack/unpack, quantization, forward, and VJP comparisons to
   `quantized_lora.py`, including zeros, tails, and repeated determinism.
2. Fresh-process isolated GPU checks at 64-512 tokens, then serialized 1K-16K
   shapes under `profile_rocm.py`; 24K/32K comes only after clean smaller runs.
3. No BF16 shadow weight and no full dequantized weight in XLA buffer
   assignment. Peak VRAM must realize the documented 3.2-3.8 GiB W8 or
   4.9-5.8 GiB W4 base-model saving, depending on whether the tied matrix is
   kept BF16.
4. Treat at least 1.2x isolated projection throughput and a measurable
   end-to-end step gain as the W8 continuation gate. W4 should materially beat
   W8, not merely save memory, before accepting its higher numerical risk.
5. SFT loss/gradient and GRPO logprob/KL/reward checks must pass independently
   for each projection family before model-wide activation.

The only current performance conclusion is feasibility: W8A8 and W4A4 can be
compiled natively on this gfx1100 toolchain, the exact tiny Pallas W8A8 forward
emits signed-IU8 WMMA, and the compact formats offer large, already-calculated
residency savings. The guarded one-shot execute branch exists, but no Pallas W8
result has completed or been validated: the old masked top-level invocation
timed out, and the corrected object has not been executed. Its speed advantage
remains an experiment because runtime correctness, input quantization, group
rescaling, LoRA work, backward, and bounded-launch overhead have not been
benchmarked against the complete BF16 projection.
