# Qwen3.5 ROCm validation record

This file records only experiments that completed on the local Radeon RX 7900
XTX (`gfx1100`) with the monitor attached to the Intel iGPU. It is not a claim
that larger context buckets are safe or that synthetic SFT loss is a quality
measurement. Every GPU run used a fresh process, private telemetry files, and
fatal AMDGPU journal monitoring.

## Current safety configuration

- ROCm 7.2.4 and JAX/JAXlib/ROCm plugin/PJRT 0.10.2.
- `XLA_FLAGS=--xla_gpu_enable_command_buffer=`. Command-buffer replay is
  disabled because the second full-model Adam replay previously produced an
  HSA invalid-packet/illegal-opcode failure; repeated updates pass with it off.
- The validated growth-mode training baseline disables JAX preallocation and
  keeps Pallas attention opt-in. Fixed-85% BFC preallocation is used only by
  the explicitly guarded capacity probes described below.
- Junction temperature is capped at 90 C and configured GPU power at 400 W.
  Full VRAM plus CPU/RAM/disk offload are permitted; individual qualification
  rungs may retain tighter workload-specific caps when their expected footprint
  is much smaller. There is no longer a global zero-swap or 4 GiB
  host-available promotion rule.
- The AMD display connectors must be disconnected and `/dev/kfd` unowned
  before launch. A 60-second cold-start sensor grace accommodates a
  runtime-suspended headless GPU.

## Post-upgrade ROCm 7.2.4 qualification

After installing ROCm 7.2.4 and AMDGPU 30.30.4, rebooting into the newly built
6.16.13 DKMS module, and moving the display to the Intel iGPU, qualification
advanced in bounded fresh processes with command buffers disabled. The first
two probes also disabled JAX preallocation: a one-element JIT `float32` add
mapped `[1.25] + 2` to exactly `[3.25]` on `rocm:0`, and a separate BF16
`16x16` all-ones matrix multiply plus `value_and_grad(sum)` returned exactly
FP32 loss `4096.0` and gradient minimum/maximum `16.0/16.0`. These two shell
probes did not write artifacts; their process wall times included startup and
are not kernel-performance measurements. After each, the entire-boot guard and
fatal-journal query were clean, `/dev/kfd` was unowned, and the device returned
to runtime suspend.

The exact 402-leaf Qwen3.5 rank-8/two-slot LoRA AdamW replay then passed three
ordinary GPU dispatches with command buffers disabled. It covered 34,512,896
BF16 parameter elements (69,025,792 B). Lowering took 3.266979 s; the cold
compile/update took 16.688593 s and replays took 0.077036 and 0.105031 s. Every
step changed its sentinel checksum, all sentinels and final parameter/optimizer
states were finite, and each gradient norm was `4.056726455688477`. Peak
physical VRAM was 1,804,165,120 B, peak process RSS/PSS/USS was
3,554,209,792/3,170,508,800/2,797,559,808 B, maximum junction temperature was
56 C, maximum power was 137 W, and swap remained zero. The guarded process
exited with status 0, no fatal driver event, and restored idle KFD/runtime-PM
state. Artifacts:

- `/tmp/postrocm-opt-1783872102.probe.jsonl`
- `/tmp/postrocm-opt-1783872102.telemetry.jsonl`
- `/tmp/postrocm-opt-1783872102.telemetry.jsonl.summary.json`

The fixed-85% BFC full-model gates then progressed without executing an SFT
update:

| Gate | Exact result | Physical peak / safety |
|---|---|---|
| Context-64 setup only | Setup 96.173134 s; allocator live/peak/pool 8,691,296,256 / 8,829,511,424 / 21,892,169,728 B | VRAM 22,608,961,536 B; junction 61 C; power 137 W; swap 0 B |
| Context-64 XLA compile only | Setup/lower/compile 39.642890 / 6.228467 / 52.449389 s; compiled temporary 196,247,088 B; allocator peak-live 10,076,861,696 B | VRAM 22,655,623,168 B; junction 73 C; power 206 W; swap 0 B |
| Context-2,048 Pallas compile only | Setup/lower/compile 39.177387 / 6.350830 / 94.074129 s; compiled temporary 11,390,298,880 B; allocator peak-live 12,522,268,672 B | VRAM 23,003,750,400 B; junction 79 C; power 311 W; swap 0 B |

The setup-only gate stopped before lowering. Both compile gates returned
`status: passed`; compilation may itself dispatch bounded autotuning kernels,
but the returned model-pass callable was never invoked. In particular, the
context-2,048 artifact records
`model_pass_executable_invocations: 0` and
`optimizer_step_invocations: 0`. Every wrapper returned status 0, swap did not
grow, the full current-boot fatal-journal scan stayed empty, and KFD/VRAM
returned to idle after each process. Artifacts:

- `/tmp/postrocm-backend-setup-1783872188.jsonl`
- `/tmp/postrocm-backend-setup-1783872188.telemetry.jsonl`
- `/tmp/postrocm-backend-setup-1783872188.telemetry.jsonl.summary.json`
- `/tmp/postrocm-compile-t64-1783872326.jsonl`
- `/tmp/postrocm-compile-t64-1783872326.telemetry.jsonl`
- `/tmp/postrocm-compile-t64-1783872326.telemetry.jsonl.summary.json`
- `/tmp/postrocm-compile-t2048-pallas-1783872544.jsonl`
- `/tmp/postrocm-compile-t2048-pallas-1783872544.telemetry.jsonl`
- `/tmp/postrocm-compile-t2048-pallas-1783872544.telemetry.jsonl.summary.json`

This advances context 2,048 to **compile-only capacity**, not validated
training. The exact-one-update client protocol has now passed independent
source review plus CPU/mocked tests and the context-64 hardware gate below. It
is not permission to invoke the context-2,048 executable. The isolated Pallas
numerical blocker has since been cleared through 2,048 tokens. The
user-authorized outer gradient gate requires relative L2 strictly below 3%
(`<3%`), while the guarded forward output remains strictly below 1% (`<1%`)
and the FP32-delta path retains and passes its gradient regression gate of
strictly below 1% (`<1%`): its worst observed result was dK at `0.003628`
(about 0.363%). The separate full-model and safety gates remain; neither
compile success nor bounded autotuning qualifies the complete training path.

### Exact W8A8 Pallas compile and ISA qualification

Revision `da9bf1ee6195921fd7c8cf9055a3dc8d4a1ed704` completed the fixed
`M=3,K=64,N=17`, group-64, rank-8 LoRA compile diagnostic with zero returned
executable invocations. Lowering and compilation took 1.134153 and 0.719444
seconds. StableHLO and optimized HLO each contain exactly one entry-owned
Triton custom call with the exact forward name and live result. Their SHA-256
values are
`0e57123dd6c1d4355b7ccbbf0f7908db686ceaafc6d26afbb237811f8861ecdf`
and
`7adf78dc72eebed7a4eaa0c1e88f7ba43ecea0199d54817fcaeb2fcc3b5aa0dc`.
The structured compiler memory report totals 6,508 bytes excluding generated
code.

An independent, CPU-only parse of the retained JAX cache
`d00d54b9e684852a1d933eafb332e058c5b232d79f687dc936951887e3f646a2`
distinguishes the empty top-level wrapper from the real nested Pallas object.
The unique symbol-matched object is 8,440 bytes with SHA-256
`606a80a508317af303966e5c2ca357d138d08828949c0dbfdcd73ccde1726389`.
ROCm LLVM reports target `amdgcn--amdhsa-amdgiz-gfx1100`, 34 SGPRs, 62
VGPRs, zero compiler-reported spills, and zero private segment. Disassembly
contains exactly four signed `v_wmma_i32_16x16x16_iu8` instructions. Paired
with the HLO call identity, this qualifies native gfx1100 signed-INT8 WMMA
code generation for this exact Pallas microcase. It does not qualify runtime
correctness or speed.

The first guarded one-shot forward attempt at revision `2b38bfb3` failed
closed before executable release in
`/tmp/skyrl-w8a8-runtime-1783980961`. Lowering and compilation succeeded, but
the offline inspector over-bound the complete top-level wrapper record even
though its deterministic 1,904-byte empty ELF was unchanged; only OpenXLA's
deliberately variable 16-byte ROCm module identifier differed. The probe
emitted no executable-release, device-put, dispatch-started, dispatch,
device-get, or runtime `numerical_validation` record, and its
compiled-executable invocation count remained zero. The controller killed and
reaped the scope in 67.1 ms and
restored exact suspended/unowned idle state with a clean AMDGPU journal.
Compilation peaked at 867,332,096 bytes physical VRAM, 51 C junction, and 101
W sampled average power. This is failed-closed safety evidence, not runtime
correctness or promotion evidence.

The verifier now pins the deterministic empty ELF and exact 16-byte identifier
shape while retaining caller-bound whole-cache SHA-256 integrity. CPU-only
regression inspection passes both the previously qualified cache and the
failed-attempt cache, which contain the same exact 8,440-byte W8 HSACO. A GPU
retry remained a separate guarded rung.

That one guarded retry was consumed at revision `8eb3c7d7` in
`/tmp/skyrl-w8a8-runtime-1783982504`. The exact five-kernel executable plus its
final device-to-device slice did not complete inside the controller's five
second dispatch watchdog. The probe emitted `dispatch_started`, but no
post-attempt checkpoint, dispatch completion, device-get, numerical-validation,
or completion record. The controller killed the complete cgroup, reaped it in
0.071485 seconds, and restored the exact suspended/unowned VRAM/GTT baseline
after three consecutive idle samples. The whole-boot AMDGPU journal remained
clean: this was a fail-closed executable timeout, not evidence of a driver reset
or another machine crash.

The retained executable contains, in order, input pad/reduce, A8 conversion,
BF16 LoRA-A GEMM, scale select, the Pallas W8 call, and a final slice. Because
the watchdog bracket covered `compiled(...)` and output readiness together, it
does not identify which stage stalled. GPU-busy telemetry is also not
attributable: it reached 100% about 56 seconds before dispatch and showed the
same pre-dispatch behavior in the earlier run that never released an
executable. The exact Pallas thunk used grid `1x2x1`, 128 threads, and 1,024
bytes of dynamic LDS. Its 8,440-byte HSACO has four signed-IU8 WMMAs, nine
barriers, no spills/private stack, and extensive EXEC-mask control for the
single-column `N=17` tail. These facts make the masked multi-wave lowering a
plausible hypothesis, not a proven cause.

The replacement source therefore zero-pads logical output features to complete
physical `block_n` tiles outside Pallas, omits row/column tail predicates from
the forward and base-input-VJP kernels, and slices only after the custom call.
For the qualification case this changes logical `N=17` to physical `N=32`;
aligned Qwen projection sizes require no padding. CPU/interpreter and mock
coverage now proves bitwise tail-forward equivalence, the retained W8
implementation-fidelity VJP gates requiring relative L2 strictly below 1%
(`<1%`) against the portable grouped-W8 oracle, physical call shapes, and
absence of `lt`/`and` tail predicates in both kernel JAXPRs. A proposed W8
quantization-versus-BF16 quality gate of strictly below 3% (`<3%`) is distinct
and not yet qualified.

Revision `f6b26775456e5c2aa92dd621b26f3752f111ad00` first completed the
corrected full-tile compile. After the six-thunk, embedded-object, resource,
and control-flow pins were committed at
`e4152e47900086bfe2f58acf94d9f9a765b99aa5`, a second fresh compile-only
qualification completed in `/tmp/skyrl-w8a8-compile-1783987495`. The
controller returned `passed_compile_diagnostic_unpromoted`; there were zero
ISA qualifications, input/output transfers, returned-executable invocations,
or releases. Lowering and compilation took 1.112806747 and 0.720117240
seconds. The 15,351-byte StableHLO has SHA-256
`3eac9283c56d35e9df7f3fb42d7ab62fba851b74ca7f5ea09ee746f1587f1772`;
the 22,853-byte optimized HLO has SHA-256
`f05cc7cefb15832b90a24088fad868175034874534b8193e82430d4840899406`.
The optimized-HLO change from the first corrected compile is confined to
source-location line metadata. Both artifacts bind the sole Pallas call to
physical BF16 `[16,32]` and the public result to logical BF16 `[3,17]` through
the final `wrapped_slice`. Compiler accounting remains 4,036 bytes of
arguments, 102 bytes of output, 3,600 bytes of temporaries, 7,738 bytes total,
and 1,920 bytes of generated code.

The fresh retained cache is 19,332 bytes with SHA-256
`bf40785e7dd4d0a07336ba814e8d7fde59c2d5525e50f8317134999119802604`
and decoded record sizes `[56, 0, 1920, 0, 52909]`. Its raw 52,909-byte
executable record has the run-path-dependent SHA-256
`696fa3761f183e49139ca957c1d8bd337699451560f8f233df5e0b4cbdea556b`.
A bounded CPU-only schema audit requires the sole embedded autotune-cache path
to equal the caller-derived canonical 101-byte path at field-one/record
offsets 19,901/19,905, then replaces exactly those 101 bytes with an
equal-length neutral token. The still-parseable normalized record is 52,909
bytes with SHA-256
`8060df67a90b7e0827672aa4c349d66f51a50b13120345e698ea95454c6acc08`;
its normalized 20,600-byte field one has SHA-256
`1cac7332465fe69bd9d4ae2a53dbd0454a5f3ca4fd28bbdbd400a66a30dde1cd`.
The whole normalized-record hash remains authoritative and therefore pins all
non-path bytes, protobuf framing, thunks, and embedded ELFs. The older cache
fails this fresh normalized hash because its probe source-line metadata is
different. These two builds are not a same-source reproducibility pair, so
path-only variability under identical source remains inferred rather than
empirically repeated. The fresh audit decoded exactly these six ordered
custom-kernel thunks:

| Order | Kernel | Grid | Threads | LDS (bytes) | ELF bytes / SHA-256 |
|---:|---|---|---:|---:|---|
| 0 | `input_pad_reduce_fusion` | `2x1x1` | `256x1x1` | 0 | 4,080 / `06a6035fabadbc8de4d7d201fa51ad2b9383a37faa84e4a0b51d9587fa3d8c7f` |
| 1 | `loop_convert_fusion` | `8x1x1` | `128x1x1` | 0 | 3,944 / `8db071b2d0e93475f713c566d984a940155ed293e63adf53b62a53288fada685` |
| 2 | `gemm_fusion_dot_general_1` | `1x1x1` | `128x1x1` | 8,192 | 7,408 / `c45a0fb7f236f7b16dbdfedb905dd116a02006c16c43d6dd687c30ccedf2eaf1` |
| 3 | `loop_select_fusion` | `1x1x1` | `16x1x1` | 0 | 3,424 / `8e7f454a584324b303ab299d22e4a3d4ee956f29bc36c64c030256fd24068a71` |
| 4 | `skyrl_qwen35_w8a8_lora_forward` | `1x2x1` | `128x1x1` | 1,024 | 7,160 / `87a2ae903547258a4b107fad17797147c417d8ca35cc600bc35d77e46323368f` |
| 5 | `wrapped_slice` | `1x1x1` | `51x1x1` | 0 | 3,416 / `476174a6aa35385fa65e84356f63b196540840c4ac782985b6ecf744b30c4799` |

Fresh compile `/tmp/skyrl-w8a8-compile-1784038127` exposed one bounded
autotune variant. StableHLO and the W8 Pallas thunk/ELF remain byte-identical,
while the separate LoRA-A BF16 GEMM selected `block_m=block_n=32`. The
normalized executable is 53,166 bytes with SHA-256
`4a7fc5e78b508cca93db2abfe209100a56153a123372bb25aa964c0cbb124985`;
its normalized HLO hash is
`9978dc0830323f4331dcb7c537fcbc56d8263be070d6f545b43191a8e651085b`.
Only thunk 2 differs: 7,922 bytes / SHA-256
`0a6c802363f4c8dc30ddfebb195cccc66d4dadc4138db5860736c879de132a2d`,
16,384 bytes LDS, and a 7,664-byte ELF / SHA-256
`9ab0e3abac1983fcb44279f9fe1b40da01186a6d674e271b6c103cbb69c40b2a`.

The first allowlist revision pinned two complete historical-artifact contracts:
`lora_gemm_bm16_bn16` (`b7d543d6bf2aff9913221b1a438851fc7eec825d98cc9427b7178804d143db57`)
and `lora_gemm_bm32_bn32`
(`75ce7e3c82b4219a17391f3f3019c3fbef84dfdf6c924cb3051d4b7d884ae0c7`).
Those two hashes are retained as historical evidence, not accepted by the
current verifier.

Three clean compile-only seeds from the frozen `0fec1b67` source layout then
captured both same-layout outcomes. Runs `1784041485` and `1784041690` selected
BM32 and reproduced normalized record/HLO SHA-256 values
`989798f1183a243fe074491578827e4b04bf2d0eb25ca127f0a3b06f93050b94` /
`577bdf1c685ce7553f3af8ff8ab6b2125247f2cc7a15ee47f4c0e7916277b03f`.
Run `1784041853` selected BM16 with normalized record/HLO SHA-256 values
`94e1a986416c6b1b0b3d249b5ff41c2fc11dec215612a66c21d28a15968d49bc` /
`aca6770fd14a7d002ad465bfe8ac09c22f77b33b5d38ca0b2bd95c13734349d5`.
The current accepted complete-contract hashes are
`d2859c3fd661ca42f7ce2231d3b090af59e53210fad860519df7695a7856a947`
(BM16) and
`749fff3d982c91f738c7b7c5c44d4d7120c9d24f439d10df90f20ae7a5890766`
(BM32). Equal-length pin changes preserved the candidate at source line 2,174.
All three seeds compiled but invoked the candidate zero times; they peaked at
118 W, 61 C, and 867,364,864 bytes VRAM with unchanged 24,576-byte swap and
clean postflight/handoff.

The selected name must match its complete normalized record/HLO, ordered
thunks, launches, and nested ELFs in the inspector, child, and independent
controller. Unknown, historical, or cross-paired variants are rejected.
Clean post-pin compile-only run `1784042397` selected BM16 and passed the
public offline verifier with zero invocations. It peaked at 118 W, 59 C, and
867,364,864 bytes VRAM, retained 24,576-byte swap, and ended with clean
postflight/handoff.

Exactly one guarded runtime was then attempted in run `1784042574`. The child
passed every pre-dispatch BM16 ISA check, released the executable, transferred
all six inputs, and emitted `dispatch_started`, but the dispatch never emitted
a post-attempt record. The five-second watchdog failed, killed the complete
cgroup in 0.071221 seconds, and produced no device-get or numerical output.
Peak telemetry was 105 W, 61 C, and 867,450,880 bytes VRAM; swap remained
24,576 bytes. The journal stayed clean, three handoff samples restored the
exact idle baseline, and `/dev/kfd` was unowned. W8 runtime is therefore
NO-GO, must not be retried unchanged, and is not a SkyRL optimizer/model
integration candidate. The child check is pre-dispatch; the controller's
independent repeat would be postflight after a successful child exit. No W8
ISA check was relaxed.

Across the two earlier BM16 corrected builds, all six thunk serializations and
all seven ELFs are byte-identical. The first four ELFs are also byte-identical to the
masked executable. The corrected Pallas object is 7,160 bytes, targets
gfx1100, uses 34 SGPRs and 105 VGPRs, and reports no spills. Its disassembly
has exactly four signed IU8-WMMA instructions and nine barriers; its sole
forward branch occurs after every barrier. The former final D2D copy is the
real `wrapped_slice` kernel shown above. This is an offline GO for the exact
normalized inspector and release-gate pins. It is not permission to execute
the object and proves no runtime correctness, speed, memory saving, backward,
or promotion.

The fresh compile peaked at 867,352,576 bytes physical VRAM, 22,405,120 bytes
GTT, 6,423,089,152 bytes host RAM used, 51 C junction temperature, and 113 W
sampled average power. Swap stayed exactly 20,480 bytes, the maximum measured
sample gap was 0.082857711 seconds, the AMDGPU journal remained clean, and
handoff restored the exact suspended/unowned baseline of 27,947,008 bytes VRAM
and 15,966,208 bytes GTT after three samples.

For the original compile-only artifacts under
`/tmp/skyrl-w8a8-compile-1783972228`, peak observed physical VRAM, junction
temperature, and sampled average power were 867,360,768 bytes, 51 C, and 113 W.
Swap did not grow, all monitored limits passed, no driver fault appeared, and
the device returned to its exact suspended/unowned idle baseline. That
historical compile evidence justified the now-consumed one-shot rung; it does
not authorize another invocation of the same artifact. The old masked
8,440-byte object and its timed-out executable must never be retried. The
corrected object has not been executed. The normalized pins and their CPU-only
regressions have received independent review in this source revision. That
offline GO is not execution authorization; any guarded request remains a
separate one-shot decision.

### Stable-source T1024 executable-cache qualification

Exact commit `5fb2b220bfbf202ac2b9295efb9e9a072cc00135` completed one
launcher-supervised T1024 top-level-cache population and one matched fresh-process
hit from the identical private commit-keyed snapshot. The population added only
`jit_forward_backward_and_accumulate-f2c446a981cb5237c15fe0550221ed6b046c5dff8829de56d9a5cb1045c21fa6-cache`
(6,002,623 bytes); the hit reported one hit, zero misses, `37.202892 s` saved,
`6.797108 s` retrieval, and no cache addition, change, or removal.

Matched T1024 compile time fell from `47.361373 s` to `6.897813 s` (85.44%,
6.87x), while the complete profiled child fell from `135.804416 s` to
`95.294999 s` (29.83%, 1.43x). Peak command-tree RSS fell by 1.278 GiB / 23.76%.
Peak VRAM remained effectively identical at 17,901,096,960 versus
17,901,109,248 bytes, so this is not a capacity gain. Both runs used zero swap,
stayed below 63 C and 255 W, invoked no compiled model pass or optimizer step,
and returned exactly to suspended, unowned baseline state with a clean journal.

The stable snapshot contained 1,090 files / 35,333,482 bytes with source
manifest `5179a0e25214a50ee8cb74b7793323fd0189759c8de9b63d23ec1e727caf23f0`.
Its deterministic archive hash was
`3222c78008e0e6bcf23ca034a96f4e7c075567e2a53381639598c48594f6547b`.
Full timings, cache hashes, artifact hashes, and public-monitoring limitations
are recorded in [`README.md`](README.md). The result proves repeated prewarm
process reuse only. API/nested-engine source-path alignment is now implemented
and CPU-qualified with sanitized environment, exact `uv` arguments, and full
Git/archive/snapshot revalidation. The exact uv payload and one fixed per-UID
lock namespace are pinned, and CPU probes confirm the locked descriptor reaches
the nested engine. This transition has not yet consumed the cache in a hardware
run. An exact per-launch readiness table, dedicated child process group,
engine-owned watchdog heartbeat, bounded startup wait, and fail-closed health
endpoint are now implemented and CPU-qualified for the pinned non-Ray JAX
path. Real ROCm backend readiness,
actual cache consumption, ordinary warmup, and steady-state throughput remain
unverified.

### Exact S512 GDN execute forward gate

Revision `4c9e7877` qualified exactly one standalone typed-FFI forward
invocation for the S512 GDN recurrent execute stage. After backend compilation
and input placement, the one checked candidate took 8.414 ms including its
output readiness barrier. BF16 output relative-L2/cosine/max-absolute error was
`2.33682e-5` / `0.999999999727` / `9.53674e-7`; the FP32 final-state result was
`1.23831e-7` / `0.999999999999972` / `1.16415e-9`.

The guarded process performed exactly one capability release, tuple device
put, executable invocation, output barrier, and tuple device get. It performed
no warmup, replay, graph, backward, model, device-reference, or reduction call;
VJP was not exercised. Peak observed junction temperature, power, and physical
VRAM were 50 C, 129 W, and 779,153,408 bytes, swap stayed zero, and KFD/VRAM
returned to the exact idle state with a clean current-boot journal.

This is forward correctness and first-candidate latency evidence, not repeated
throughput or an end-to-end model speedup. Reverse mode, composed
prepare/execute, model integration, and other shapes remain pending. Exact
kernel design, build metadata, compiler evidence, artifact hashes, and scope
are recorded in
[GDN_EXECUTE_S512.md](../skyrl/tx/kernels/rocm/ffi/GDN_EXECUTE_S512.md).

### Post-upgrade exact-one-update hardware gate

Pushed revision `8437dd7739f16ca0c42832cbbca9858f3ced7875` completed the
reviewed `--one-update-gate` at context 64 in the growth allocator mode. The
artifact contains exactly `manifest`, `step`, `cleanup`, and `summary` records:
one requested update, no warmup or measured throughput samples, one cold
forward/backward/Adam step, `adapter_unloaded: true`, and
`updates_completed: 1`. The repository manifest was clean at that exact
revision.

| Result | Measured value |
|---|---:|
| Client-observed cold update | 103.215228 s |
| Server training JIT | 69.06 s |
| Server optimizer request | 33.316 s |
| Synthetic mean NLL | 0.4008960724 |
| Optimizer gradient norm | 4.90625 |
| Maximum system VRAM used | 18,066,522,112 B (16.83 GiB) |
| Maximum process RSS / PSS / USS | 17,610,248,192 / 17,316,330,496 / 17,040,822,272 B |
| Maximum junction temperature / power | 67 C / 261 W |
| Maximum host memory used / swap | 16,740,802,560 B / 0 B |

After the successful client summary, the long-running server wrapper was
stopped intentionally with SIGINT; its telemetry summary therefore records
`status: signal`, `received_signal: 2`, and wrapper return code 143 rather than
a natural server exit. Uvicorn completed application shutdown and stopped the
background engine. The client artifact proves adapter unload before that
signal. The subsequent entire-boot guard was clean, port 8001 was free,
`/dev/kfd` was unowned, and the GPU returned to runtime suspend. Artifacts:

- `/tmp/postrocm-one-t64-1783873721.sft.jsonl`
- `/tmp/postrocm-one-t64-1783873721.telemetry.jsonl`
- `/tmp/postrocm-one-t64-1783873721.telemetry.jsonl.summary.json`

This re-establishes one full Qwen3.5-4B forward/backward/Adam execution at
context 64 on ROCm 7.2.4. It is a safety/correctness gate, not a throughput
baseline and not evidence for contexts above the previously validated 1,024.

## Full Qwen3.5-4B SFT control

On 2026-07-12, revision `431ee0b3fb9ae04b4c56d41fee95d593b7476cbc`
completed one cold and five measured forward/backward/Adam steps for
`Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The configuration was batch 1,
context 64, rank-8 LoRA, two adapter slots, per-layer rematerialization, and a
64-token loss chunk.

| Result | Measured value |
|---|---:|
| Cold forward/backward plus first Adam | 87.230 s |
| Measured median step | 0.5931 s |
| Measured p95 step | 1.0707 s |
| Median useful throughput | 107.90 tokens/s |
| Maximum system VRAM used | 18,067,054,592 B (16.83 GiB) |
| Maximum process RSS / PSS / USS | 16.57 / 16.30 / 16.04 GiB |
| Maximum / p95 junction temperature | 73 / 61 C |
| Maximum / p95 GPU power | 271 / 137 W |
| Maximum host memory used | 16.79 GiB |
| Maximum swap used | 0 B |

All six losses and gradient norms were finite. The model unload endpoint
succeeded, no fatal AMDGPU journal event was observed, VRAM returned to the
27,947,008-byte idle baseline, and no process retained `/dev/kfd`. The local
private artifacts are:

- `/tmp/postfix2-t64-1783854419.sft.jsonl`
- `/tmp/postfix2-t64-1783854419.telemetry.jsonl`
- `/tmp/postfix2-t64-1783854419.telemetry.jsonl.summary.json`

The client-observed step time includes the local HTTP/database/future path. It
must not be compared directly with isolated device-event kernel timing.

## Full Qwen3.5-4B SFT at context 512

On 2026-07-12, revision `82e038f61565e820d8dc86b5fd285eb77a07a301`
completed one cold and five measured forward/backward/Adam steps at batch 1,
context 512, using the opt-in Pallas attention path. LoRA, rematerialization,
loss chunking, model revision, and command-buffer settings matched the
64-token control.

| Result | Measured value |
|---|---:|
| Cold forward/backward plus first Adam | 170.225 s |
| Server-reported train JIT | 157.79 s |
| Measured median / p95 step | 1.4281 / 1.4299 s |
| Median useful throughput | 358.52 tokens/s |
| Maximum system VRAM used | 18,530,045,952 B (17.26 GiB) |
| Maximum process RSS / PSS / USS | 16.13 / 15.77 / 15.59 GiB |
| Maximum / p95 junction temperature | 80 / 66 C |
| Maximum / p95 GPU power | 358 / 143 W |
| Maximum host memory used | 17.35 GiB |
| Maximum swap used | 0 B |

All six losses and gradient norms were finite, the adapter unload succeeded,
the fatal-journal query was empty, VRAM returned to the 27,947,008-byte idle
baseline, and `/dev/kfd` was unowned after exit. Relative to the T64 control,
the end-to-end client median used 2.41x the time for 8x the useful tokens, or
3.32x the throughput. This comparison includes polling/API overhead and also
changes attention implementation at the 512-token safety boundary; it is not
an isolated Pallas speedup.

The manifest had no tracked source diff and identified the exact commit above.
It did list three untracked, unwired GRPO-benchmark files being reviewed in
parallel; none is imported by the server or SFT client. Local private artifacts
are:

- `/tmp/t512-sft-1783856707.sft.jsonl`
- `/tmp/t512-sft-1783856707.telemetry.jsonl`
- `/tmp/t512-sft-1783856707.telemetry.jsonl.summary.json`

### Context-1024 progression

The same guarded configuration then completed one cold and five measured
steps at context 1,024. The server-reported JIT was 160.24 s; the client cold
step including the first Adam was 172.242 s. Median/p95 step time was
2.1917/2.2023 s, or 467.21 useful tokens/s. Peak system VRAM was
18,823,901,184 B (17.53 GiB), maximum process RSS/PSS/USS was
16.93/16.72/16.47 GiB, and maximum host use was 16.90 GiB. Maximum/p95
junction temperature was 85/72 C, maximum/p95 GPU power was 358/160 W, and
swap remained unused.

All losses and gradient norms were finite, adapter unload succeeded, the fatal
journal query was empty, and the device returned to its idle VRAM/KFD state.
The local artifacts are:

- `/tmp/t1024-sft-1783857150.sft.jsonl`
- `/tmp/t1024-sft-1783857150.telemetry.jsonl`
- `/tmp/t1024-sft-1783857150.telemetry.jsonl.summary.json`

### Context-2048 default-allocator capacity boundary

The first context-2,048 full-model attempt used the default JAX device-memory
fraction, a stricter 85 C junction cap, and five-second untimed cooling delays.
Attention had already passed its isolated forward/backward gate, but the full
training executable did not fit. XLA reported that rematerialization reduced
its estimate from 10.60 GiB to 10.48 GiB but could not reach the 9.59 GiB
target; the allocator then rejected a 10.56 GiB request. No training step ran.

The failure was a handled `RESOURCE_EXHAUSTED` result: adapter cleanup
succeeded, peak observed VRAM was 18,294,276,096 B, maximum junction
temperature was 84 C, swap remained zero, the fatal-driver journal query was
empty, and KFD/VRAM returned to idle. This does not move the full-model
validation frontier beyond 1,024. It establishes memory residency/buffer
assignment, rather than attention correctness, as the next blocker. Artifacts
are:

- `/tmp/t2048-sft-1783858067.sft.jsonl`
- `/tmp/t2048-sft-1783858067.telemetry.jsonl`
- `/tmp/t2048-sft-1783858067.telemetry.jsonl.summary.json`

A fresh-process retry explicitly raised
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` while lowering the profiler's system-VRAM
cap to 22 GiB. It failed on the same 10.56 GiB allocation, unloaded cleanly,
and produced no driver event. Increasing the allocator fraction therefore does
not solve this boundary; compact model residency and/or a smaller executable
working set is required. Retry artifacts are
`/tmp/t2048-fraction90-sft-1783858376.{sft.jsonl,telemetry.jsonl}` (with the
usual `.summary.json` beside telemetry).

An initialization-only server run then set
`XLA_PYTHON_CLIENT_ALLOCATOR=platform` and received no client request. It still
settled at 17,899,212,800 B of physical VRAM, indistinguishable from the
default route. The process was stopped normally after observation, reached
68 C, used no swap, and produced no fatal driver event. This installed ROCm
JAX stack therefore does not provide a useful deallocating platform-allocator
escape hatch. Telemetry is in
`/tmp/t2048-platform-sft-1783858638.telemetry.jsonl`.

### Model-residency accounting

`rocm/probe_model_residency.py` separates abstract state size, eager model
initialization, checkpoint loading, NNX splitting, and settling without
compiling or executing a model pass. The exact abstract state is
8,480,538,476 B (7.8981 GiB): 8,411,510,272 B of non-LoRA state and
69,028,204 B of LoRA metadata/parameters. After normal initialization and
loading, `jax.live_arrays()` reported 16,892,048,748 B because the module and
split states both reference arrays, but pointer deduplication found exactly
8,480,538,476 B across 1,182 unique buffers. Thus `nnx.split` aliases the base
buffers; it does not duplicate the model.

The allocator view explains the much larger physical number. With the
requested `platform` allocator, backend-ready `bytes_in_use` was
9,127,142,144 B while `pool_bytes` was 17,181,966,336 B and observed physical
VRAM was 17,899,282,432 B. An experimental abstract-model/direct-load route,
with fused checkpoint interleaving performed in NumPy rather than accidentally
through JAX, reported settled `bytes_in_use` of 8,604,524,288 B and peak
`bytes_in_use` of 8,674,484,224 B. That run used a different allocator setting,
so the counter difference is confounded and is not a proven loader saving.
The decisive counters are unchanged: the route had the same 8,480,538,476 B
unique live state, BFC still grew its pool to 17,179,869,184 B, and physical
VRAM still reached 17,895,243,776 B. Direct loading alone therefore does
**not** reduce settled model state or reclaim the roughly 8 GiB pool growth.

These are initialization-only measurements, not a validated backend change.
The direct-load route is now wired behind the default-off
`abstract_model_load` flag, but it has not passed a ROCm model forward or
training step. The evidence rules out duplicate NNX state as the primary cause
and makes allocator-pool geometry plus the 10.56 GiB executable arena the
capacity problem to solve. Artifacts are:

- `/tmp/residency-bfc-1783859376.{jsonl,telemetry.jsonl}`
- `/tmp/residency-platform-1783859495.{jsonl,telemetry.jsonl}`
- `/tmp/residency-abstract-load4-1783860134.{jsonl,telemetry.jsonl}`

For the default 75% BFC limit, allocator-live state plus the requested arena is
already impossible even with perfect placement: approximately
`8.500 + 10.560 > 17.988 GiB`. The 90% growth-mode retry has aggregate room,
but neither its cached free region nor its remaining extension budget is large
enough for one 10.56 GiB allocation. The next allocator experiment is a fixed
85% preallocated BFC arena combined with direct loading, which should create
one 20.387 GiB region and avoid replacement holes. That is quantitative
feasibility only; allocation-only, load-only, compile-only, and one-step gates
must pass in separate fresh processes before another multi-step run.

### Fixed-BFC allocation and direct-load gates

The reviewed fixed-85% BFC allocation-only probe passed in a fresh guarded
process. XLA reported `bytes_limit == pool_bytes == peak_pool_bytes ==
21,892,169,728 B` (20.3887 GiB), with only the 256-byte probe array live. The
value is exactly one 2 MiB granule above the raw sysfs-times-0.85 estimate;
OpenXLA explicitly rounds the BFC GPU limit upward to 2 MiB. Physical VRAM
peaked at 22,605,082,624 B (21.0526 GiB), junction temperature at 49 C, and
swap stayed at its pre-existing 622,592-byte baseline. The process returned to
27,947,008 B idle VRAM with unowned KFD and no fatal driver event. Artifacts:

- `/tmp/bfc85-allocation-1783863595.jsonl`
- `/tmp/bfc85-allocation-1783863595.telemetry.jsonl`
- `/tmp/bfc85-allocation-1783863595.telemetry.jsonl.summary.json`

The next fresh process loaded the pinned checkpoint through the abstract
constructor without a model pass. `loaded` contained exactly 1,182 unique
buffers and 8,480,538,476 live bytes; `nnx.split` left both unique-buffer deltas
at zero. Settled BFC usage was 8,484,341,504 B with a peak of 8,553,896,960 B,
leaving 13,407,828,224 B (12.4870 GiB) inside the fixed pool—1.9270 GiB beyond
the nominal 10.56 GiB failed arena request. Physical VRAM peaked at
22,610,644,992 B, junction temperature at 54 C, swap did not grow, artifacts
were mode 0600, and exit again restored idle KFD/VRAM without a fatal driver
event. Artifacts:

- `/tmp/bfc85-residency-1783864015.jsonl`
- `/tmp/bfc85-residency-1783864015.telemetry.jsonl`
- `/tmp/bfc85-residency-1783864015.telemetry.jsonl.summary.json`

These two gates establish allocation and residency feasibility only. They do
not execute forward/backward, prove compiled temporary placement, or move the
validated full-model frontier beyond context 1,024.

The first context-2,048 compile-control invocation then completed exact backend
setup but deliberately stopped during lowering: the attention selector refused
the quadratic XLA fallback because the probe had not explicitly selected the
Pallas route. No model-pass callable or optimizer step ran. Peak physical VRAM
was 22,612,500,480 B, junction temperature was 59 C, swap did not grow, and the
process returned cleanly to idle without a driver event. The probe now exposes
an exact attention selector: effective contexts below 512 use `xla`, while
512--16,384 require explicit `pallas`. This handled control failure is not a
compile pass. Artifacts are
`/tmp/bfc85-compile-t2048-1783864343.{jsonl,telemetry.jsonl}` with the telemetry
summary alongside them.

A separate context-64 XLA compile-only control subsequently passed. Backend
setup used 8,691,296,256 B, lowering took 5.945 s, and compilation took
41.811 s. Compiler accounting reported 196,234,800 B (187.14 MiB) of temporary
memory; no returned callable or optimizer step was invoked. Physical VRAM
peaked at 22,642,417,664 B, junction temperature at 71 C, swap did not grow,
and the process returned cleanly to idle. Artifacts:

- `/tmp/bfc85-compile-t64-1783864767.jsonl`
- `/tmp/bfc85-compile-t64-1783864767.telemetry.jsonl`
- `/tmp/bfc85-compile-t64-1783864767.telemetry.jsonl.summary.json`

The following fresh process was configured for context 2,048 with Pallas, but
it did **not** reach Pallas lowering or compilation. Its private JSONL contains
only the flushed manifest—no `backend_ready`, `lowered`, or `compiled` record.
At 19:32:28 local time, during common backend/model/optimizer setup, the kernel
reported `Illegal opcode in command stream`; ROCm also surfaced
an invalid-packet diagnostic in the live terminal. That terminal line was not
persisted in the local artifacts, so the kernel journal event is the durable
evidence. The profiler immediately terminated the child. Peak physical VRAM
was 22,610,432,000 B, maximum junction was 60 C,
and swap did not grow, so this was not a resource-limit trip. The desktop stayed
up on the iGPU, KFD and VRAM returned to idle, and no ring timeout, VM fault,
reset, or reboot followed. Artifacts:

- `/tmp/bfc85-compile-t2048-pallas-1783864906.jsonl`
- `/tmp/bfc85-compile-t2048-pallas-1783864906.telemetry.jsonl`
- `/tmp/bfc85-compile-t2048-pallas-1783864906.telemetry.jsonl.summary.json`

The exact offending asynchronous setup operation is not instrumented in that
artifact, and attributing the event to Pallas or context length would be wrong.
Postmortem inspection also found an earlier illegal-opcode event at 15:11:54 in
the same boot, predating these fixed-BFC gates; moving the display to the iGPU
did not clear potentially suspect driver state. This is an important confounder
and is why the new guard scans the entire current-boot journal.
The observed failure occurred during repeated fresh-process fixed-arena setup;
the earlier same-boot event means that path is not established as the original
cause. Command buffers were disabled but were not sufficient protection. No
further GPU work was permitted in that boot. The ROCm 7.2.4 post-reboot
setup/compile qualification is recorded above; it does not retroactively prove
the cause of this event or authorize a 2K model step or integrated megakernel
run.

## Fixed-rollout GRPO learner control

Revision `31800cf001c0c982e56231f386182f0cb02c163c` completed one cold
and five measured `importance_sampling` forward/backward/Adam steps for one
deterministic two-rollout group at context 64. Each rollout had 16 action
tokens, rewards `[0,1]`, advantages `[-0.5,+0.5]`, rank-8 LoRA, and a scalar
synthetic old log-probability of -5. No sampling, grading, checkpoint export,
or sampler synchronization was performed.

| Result | Measured value |
|---|---:|
| Cold learner step | 44.267 s |
| Server-reported train JIT | 28.36 s |
| Measured median / p95 step | 0.9258 / 0.9436 s |
| Median learner throughput | 138.26 tokens/s |
| Median action-token throughput | 34.57 tokens/s |
| Maximum system VRAM used | 18,067,034,112 B (16.83 GiB) |
| Maximum process RSS / PSS / USS | 17.10 / 16.82 / 16.57 GiB |
| Maximum / p95 junction temperature | 77 / 68 C |
| Maximum / p95 GPU power | 318 / 145 W |
| Maximum host memory used | 16.95 GiB |
| Maximum swap used | 0 B |

All policy metrics and optimizer gradient norms remained finite, the adapter
unloaded, the fatal-journal query was empty, and KFD/VRAM returned to idle. The
scalar synthetic old log-probability is deliberately not sampled from the
model: action importance ratios spanned roughly `9e-5` to `148`, and gradient
norms ranged from 15.3 to 35.5. This is learner-path and numerical-finiteness
evidence, not a realistic ratio/KL distribution or policy-quality result.

Local private artifacts are:

- `/tmp/grpo-t64-g2-1783857586.grpo.jsonl`
- `/tmp/grpo-t64-g2-1783857586.telemetry.jsonl`
- `/tmp/grpo-t64-g2-1783857586.telemetry.jsonl.summary.json`

## Isolated evidence and known failures

The next three bullets preserve the pre-FP32-delta measurements, whose dQ/dK
relative-L2 errors were about 1.1%. They describe the former unpatched
implementation under the initial gradient promotion criterion of strictly
below 1% (`<1%`), not the current guarded path. Forward output remains strictly
below 1% (`<1%`); the user-authorized outer gradient gate also requires
strictly below 3% (`<3%`). The current FP32-delta path retains its gradient
regression gate of strictly below 1% (`<1%`) and passed it through 2,048 tokens,
with worst observed dK relative L2 `0.003628` (about 0.363%).

- Before the FP32-delta patch, the bounded correctness probe completed
  512-token BF16 Pallas attention for
  both an all-valid sequence and a 385-valid/127-padded sequence. For the
  all-valid case, median forward/backward latency was 0.850/1.824 ms after one
  compile run, JAX peak allocation was 171,963,648 B, maximum junction
  temperature was 60 C, and no fatal driver event occurred. Against the
  16-query-token-block FP32 reference, output relative L2/cosine was
  0.00202/0.999998. The `dq` and `dk` relative-L2 errors were 0.0113 and
  0.0108, slightly outside the initial gradient promotion gate of strictly
  below 1% (`<1%`), so the then-unpatched path remained experimental despite
  its otherwise clean result. The padded result was materially the same.
  Telemetry is in
  `/tmp/pallas512-reference-1783855014.telemetry.jsonl` and
  `/tmp/pallas512-pad385-1783855051.telemetry.jsonl`.
- The same bounded reference at 1,024 tokens completed with 1.194/4.645 ms
  median forward/backward, 339,747,840 B JAX peak allocation, and no driver
  event. Output relative L2 was 0.00206; `dq`/`dk` remained 0.0113/0.0107, so
  the pre-patch numerical promotion decision did not change. Telemetry is in
  `/tmp/pallas1024-reference-1783857107.telemetry.jsonl`.
- At 2,048 tokens the isolated path also passed: median forward/backward was
  2.942/15.347 ms, JAX peak allocation was 948,325,888 B, maximum junction was
  67 C, and there was no driver event. Output/`dq`/`dk` relative L2 remained
  0.00210/0.0113/0.0107. Telemetry is in
  `/tmp/pallas2048-reference-1783857879.telemetry.jsonl`.
- The exact synthetic Qwen3.5 LoRA optimizer tree contains 402 leaves and
  34,512,896 BF16 parameter elements. Three repeated GPU updates passed with
  command buffers disabled, with about 70 ms steady optimizer time and
  1.806 GiB peak VRAM.
- The opt-in monolithic Pallas attention path previously completed BF16
  batch-1, 16,384-token, 16-query-head/4-KV-head, dimension-256 forward and
  backward in about 184 ms and 895 ms respectively, with 1.544 GiB JAX peak
  allocation. This is isolated evidence, not permission to exceed the hard
  16K cap.
- A 32,768-token monolithic attention backward caused an AMDGPU ring timeout
  and reset. The implementation now refuses that shape. A query-bounded
  forward/backward kernel is required before any new 24K/32K attention test.
- A former full-model run completed its first update and the second forward,
  then failed during the second Adam command-buffer replay with invalid packet
  format and illegal opcode. Disabling all XLA GPU command buffers is therefore
  mandatory on this machine.

## BF16 RMS/gate-up/LoRA/SwiGLU smoke no-go

Revision `60f43fa1ec9340bcd7924d006cbdbb40d59f78c1` completed the fully
guarded smoke rung for the exact `B1/T64/M64/K2560/N9216/rank8` two-launch
Pallas candidate using `BM16/BN32/BK64`. Compile-only and one-shot numerics
evidence were bound into the run before any timed dispatch. The numerical
comparison passed the user-authorized 3% outer gradient gate: forward relative
L2/cosine were `0.00021385/0.99999998`, VJP output relative L2/cosine were
`0.00015202/0.99999999`, and the `x`, LoRA-A, and LoRA-B gradients were
bit-identical to the BF16 reference in this case.

The smoke schedule completed 2 warmup and 4 measured Latin-rotation
supercycles. All 24 program dispatches plus guarded setup and teardown had
fresh external five-second cgroup-wide watchdogs; all 26 operations completed.
The raw 16 timing samples produced these attested medians:

| Boundary | BF16 JAX reference | Pallas candidate | Reference/candidate |
|---|---:|---:|---:|
| Forward | 0.59384 ms | 0.85451 ms | 0.695x |
| Forward + VJP | 0.89514 ms | 1.49858 ms | 0.597x |
| Rematerialized stage | 1.48897 ms | 2.35308 ms | 0.633x |

This four-sample smoke rung is deliberately non-qualifying, but every measured
boundary misses the promotion threshold by a wide margin. The unchanged
160-dispatch benchmark was therefore not run: more repetitions cannot
reasonably turn these approximately 30--40% throughput deficits into the required
`1.10x` forward-plus-VJP and `1.15x` rematerialized-stage gains. The current
custom VJP also recomputes the complete dense reference forward, so decoder
integration and deterministic end-to-end testing would measure a structurally
noncompetitive path. The exact candidate remains default-off and is a no-go
for model integration.

The profile completed in 40.411 s with return code zero, no signal, no safety
violation, no kernel/driver error, and a clean current-boot postflight. Peaks
were 62 C junction, 158 W, 20,053,643,264 B physical VRAM, and 24,576 B swap;
the scope was collected successfully and `/dev/kfd` returned idle. Because
fixed JAX BFC preallocation dominates the roughly 18.68 GiB VRAM reading, this
run makes no memory-saving claim. A separate fresh-process growth-allocator
A/B experiment is required for such a claim.

Private mode-0600 evidence is in
`/tmp/skyrl-bf16-benchmark-smoke.lMmiKTzu`. SHA-256 bindings are raw result
`847ca8f17c98e3daae0f143ac60579c1e0b568647c739b37336e74ca62740f59`,
progress `b1db049d37449d828d84a986729d72adeeb0890bdffc888d221ad33592745139`,
telemetry `3eb4f1779a19db36b1adbd36c83b5fd7092924ffd2f6a4e889413992e71d2d2a`,
profile summary `1e0fbca1b14c3594afb573c4d5a3e7bc23095b57dec619453042cecfaa542989`,
and CPU attestation
`3cd6e6391dcb2632e3dbcba146f6cb03cce0db0f2e24919b1d3e2c2864e56eff`.

## Contiguous BF16 RMS/gate-up/LoRA/SwiGLU successor

Revision `a135780c89a6ec701fc996e830a21890c97d4fd4` replaced the strided
no-go above with an exact, still default-off contiguous-layout candidate for
`B1/T64/M64/K2560/physical-N18432/product-N9216/rank8`. It preserves the
checkpoint's adjacent `(gate0, up0, gate1, up1, ...)` columns. One bounded
Pallas call materializes normalized BF16 X, rank-8 LoRA-A output, and the FP32
RMS vectors; a second consumes contiguous physical-N tiles and writes the BF16
SwiGLU product. The custom VJP saves the fused BF16 projection and uses five
library GEMMs without replaying the dense forward. This is a two-dispatch
operation boundary, not a literal single GPU megakernel.

The guarded compile-only rung lowered and compiled the reference and candidate
forward and forward-plus-VJP programs exactly once, with zero returned
executable invocations. The candidate forward contained the two expected
Pallas calls and one library dot; its forward-plus-VJP contained those two
Pallas calls and six total dots. The reference contained no Pallas call and
three/eight dots respectively.

The next fresh process invoked each of those four programs exactly once under
an independent five-second cgroup watchdog. Against the BF16 JAX boundary, all
candidate results passed the user-authorized 3% gradient gate and the output
cosine gate:

| Result | Relative L2 | Cosine where gated |
|---|---:|---:|
| Forward output | 0.0002374 | 0.999999972 |
| VJP output | 0.0002811 | 0.999999960 |
| dX | 0.0005684 | report-only |
| dLoRA-A | 0.0012808 | report-only |
| dLoRA-B | 0.0007217 | report-only |

The full guarded benchmark then completed 8 warmup and 32 measured
supercycles. All 160 program dispatches, setup, and teardown completed their
independent watchdogs, and every one of the 32 paired cycles favored the
candidate:

| Boundary | BF16 JAX reference | Contiguous candidate | Speedup / latency reduction |
|---|---:|---:|---:|
| Forward | 0.584814 ms | 0.450090 ms | 1.2993x / 23.04% |
| Forward + VJP | 0.902324 ms | 0.652571 ms | 1.3827x / 27.68% |
| Rematerialized stage | 1.487138 ms | 1.102660 ms | 1.3487x / 25.85% |

The profile peaked at 64 C, 161 W, and 20,053,565,440 B physical VRAM, with
24,576 B swap and no driver error. The shared fixed-BFC process plateau cannot
separate candidate from reference residency, so this run makes no comparative
memory-saving claim. Explicit saved backward state is 2,360,832 B, which is a
logical state inventory rather than a measured allocation delta.

This clears isolated speed and numerical promotion for opt-in model
integration. It does **not** authorize default enablement: a second independent
numerics process, integrated route proof, and paired end-to-end T64 learner run
are still required. The older strided candidate remains a historical no-go;
its result must not be used to reject this different contiguous schedule.

Private evidence directories are
`/tmp/skyrl-bf16-contiguous-compile.rJ1qDQ7A`,
`/tmp/skyrl-bf16-contiguous-numerics.MnNEQzTf`,
`/tmp/skyrl-bf16-contiguous-benchmark-smoke.5UHStzBC`, and
`/tmp/skyrl-bf16-contiguous-benchmark.BITmwSRv`. The full benchmark SHA-256
bindings are raw result
`0c3e19901160d4f25242794364c4dd112446244281dfd2e98244101050566684`,
progress
`3be8cd180c94a7503e88ef90f1e93943a1031a4713e3f06fea548e443d67a077`,
telemetry
`0bf1c7780f8faf57488c820c1675bc9542a3c8e9137e9444fd56f7e97d7f6eaa`,
profile summary
`9927f1b58663f671b9e6da420baffeb57fb6d1a8c469e7ffae46ef7c7da0968d`,
and CPU attestation
`784225620db7dce86a733aa88886154629b94544783cf24cf390a31c799285c3`.

## Validation frontier

The post-fix full-model SFT validation frontier is currently context 1,024. A
fixed-rollout GRPO learner control is verified at context 64; real sampling,
fixed-real-rollout replay, KL/reward comparison, and end-to-end GRPO remain
unverified. The contiguous T64 MLP-up operation is isolated-speed/numerics
qualified but not yet end-to-end qualified. Contexts above 1,024, quantized
model execution, and all other fused-kernel model execution remain unverified.
The isolated Pallas numerical blocker is
cleared through 2,048 tokens: the user-authorized outer gradient gate requires
strictly below 3% (`<3%`), and the forward-output gate remains strictly below
1% (`<1%`). The current guarded FP32-delta path's worst observed dK result,
`0.003628` (about 0.363%), passes its retained gradient regression gate of
strictly below 1% (`<1%`). Pallas remains opt-in while the separate full-model,
long-context, and safety matrix is incomplete.
The staged gates in [`MEGAKERNELS.md`](MEGAKERNELS.md) must be followed; no
quantized or custom-kernel path is enabled by default.
