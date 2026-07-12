# Bounded BF16 down-LoRA AOT proof

This directory contains an offline HIP AOT compilation proof for one bounded
Qwen3.5 down-projection stage:

```text
U = X A
Y = residual + X W + scale * U B
```

`W` is frozen. `A` and `B` are the trainable LoRA matrices. The proof targets
the Qwen3.5-4B maximum down-projection dimensions (`K=9216`, `N=2560`) and an
active LoRA rank no larger than eight. It deliberately limits one invocation to
at most 256 token rows; a 32K sequence must be split into bounded row tiles.

## What it proves

Run:

```bash
rocm/compile_probes/compile_down_lora_residual_gfx1100.sh
```

The script calls `hipcc --genco --no-gpu-bundle-output
--offload-arch=gfx1100`, then uses ROCm's LLVM inspection tools to verify:

- the output is an ELF64 AMDGPU `gfx1100` code object;
- all four versioned C kernel symbols are exported;
- their caller-owned argument prefixes match the machine-readable ABI header,
  and their full kernarg sizes include the expected compiler-generated hidden
  launch-field area;
- every kernel is limited to 64 threads, wave32, has no dynamic stack, and has
  no reported SGPR or VGPR spills; and
- every exported symbol has a disassembled body and no unresolved function.

Compilation and inspection do not enumerate a GPU, open `/dev/kfd`, register an
FFI target, load the code object, or submit work. The temporary directory is
mode-private and removed by default. Set `SKYRL_KEEP_COMPILE_PROBE=1` only when
the HSACO, metadata, and disassembly are needed for manual inspection.

This does **not** prove numerical correctness on a GPU, speed, occupancy,
watchdog safety, JAX FFI integration, or runtime stability. The loops are
scalar and reference-oriented; a production kernel should replace the dense
dot-product bodies with measured gfx1100 tiling while preserving this ABI's
bounded dispatch and tail rules.

## Storage and equations

All dense tensors are row-major. BF16 inputs and outputs use native
`hip_bfloat16`; every dot product accumulates in FP32 and rounds once on a BF16
store. The two compact workspaces are FP32 and padded to rank capacity eight:

| Value | Shape | Storage |
|---|---:|---:|
| `X` | `[M,K]` | BF16 |
| `residual`, `Y`, `dY`, `dResidual` | `[M,N]` | BF16 |
| frozen `W` | `[K,N]` | BF16 |
| `A`, `dA` | `[K,8]` | BF16; columns `rank:8` are padding |
| `B`, `dB` | `[8,N]` | BF16; rows `rank:8` are padding |
| saved `U = X A` | `[M,8]` | FP32; inactive ranks are zero |
| prepared `V = scale * dY B^T` | `[M,8]` | FP32; inactive ranks are zero |
| `dX` | `[M,K]` | BF16 |

The backward sequence implements the frozen-base VJP:

```text
dResidual = dY
V         = scale * dY B^T
dX        = dY W^T + V A^T
dA        = X^T V
dB        = scale * U^T dY
```

There is intentionally no `dW` and no gradient for the constant LoRA scale.
The active adapter slice must be selected before this boundary; this ABI does
not perform per-row adapter routing.

## Bounds, launches, and tails

Every launch uses `block=(64,1,1)`. The kernels validate their complete grid and
shape before writing; a mismatched launch returns without a status value, so a
future wrapper must validate the same contract on the host.

| Exported symbol | Grid | Outputs |
|---|---|---|
| `skyrl_down_lora_residual_fwd_bf16_gfx1100_v1` | `(ceil(N/64), M, 1)` | `Y`, saved `U` |
| `skyrl_down_lora_residual_bwd_prepare_bf16_gfx1100_v1` | `(M, 1, 1)` | `dResidual`, prepared `V` |
| `skyrl_down_lora_residual_bwd_dx_da_bf16_gfx1100_v1` | `(ceil(K/64), max(M,8), 1)` | `dX`, `dA` |
| `skyrl_down_lora_residual_bwd_db_bf16_gfx1100_v1` | `(ceil(N/64), 8, 1)` | `dB` |

The accepted runtime bounds are `1 <= M <= 256`, `1 <= K <= 9216`,
`1 <= N <= 2560`, and `1 <= rank <= 8`. Feature tiles predicate their final
lane, and inactive padded LoRA gradients are explicitly zeroed. Inputs and
outputs must be naturally aligned, allocated for the full documented shape,
and non-overlapping because the kernel pointers use the restrict contract.

The explicit argument-prefix structures and fixed constants live in
`down_lora_residual_gfx1100_abi.h`. The `_v1` suffix is part of the ABI: any
argument reorder, dtype change, layout change, or bound change requires a new
version rather than silently reusing these symbols. A future launcher must use
a supported HIP API to populate the compiler-generated hidden portion of the
AMDGPU kernarg; it must not treat the header structs as the entire raw buffer.
