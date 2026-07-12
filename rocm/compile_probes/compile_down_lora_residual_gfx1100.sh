#!/usr/bin/env bash
# Compile and statically inspect the bounded BF16 down-LoRA ABI. Never loads a GPU.

set -euo pipefail

umask 077

readonly EXPECTED_ARCH="gfx1100"
readonly REQUESTED_ARCH="${SKYRL_ROCM_ARCH:-$EXPECTED_ARCH}"
readonly KEEP_OUTPUT="${SKYRL_KEEP_COMPILE_PROBE:-0}"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SOURCE="$SCRIPT_DIR/down_lora_residual_gfx1100.hip"
readonly ABI_HEADER="$SCRIPT_DIR/down_lora_residual_gfx1100_abi.h"

readonly -a KERNELS=(
  skyrl_down_lora_residual_fwd_bf16_gfx1100_v1
  skyrl_down_lora_residual_bwd_prepare_bf16_gfx1100_v1
  skyrl_down_lora_residual_bwd_dx_da_bf16_gfx1100_v1
  skyrl_down_lora_residual_bwd_db_bf16_gfx1100_v1
)
readonly -a EXPLICIT_ARG_SIZES=(80 48 72 40)
# Accessing gridDim/blockDim makes Clang reserve its 256-byte AMDGPU hidden
# launch-field area after the caller-owned argument prefix.
readonly -a FULL_KERNARG_SIZES=(336 304 328 296)

if [[ "$REQUESTED_ARCH" != "$EXPECTED_ARCH" ]]; then
  printf 'refusing architecture %q: this proof is valid only for gfx1100\n' \
    "$REQUESTED_ARCH" >&2
  exit 2
fi

if [[ "$KEEP_OUTPUT" != "0" && "$KEEP_OUTPUT" != "1" ]]; then
  printf 'SKYRL_KEEP_COMPILE_PROBE must be 0 or 1, got %q\n' "$KEEP_OUTPUT" >&2
  exit 2
fi

for input in "$SOURCE" "$ABI_HEADER"; do
  if [[ ! -f "$input" || -L "$input" ]]; then
    printf 'compile-probe input is missing or is a symbolic link: %s\n' "$input" >&2
    exit 2
  fi
done

HIPCC="${HIPCC:-$(command -v hipcc || true)}"
if [[ -z "$HIPCC" || ! -x "$HIPCC" ]]; then
  printf 'hipcc is required\n' >&2
  exit 2
fi

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
OBJDUMP="${LLVM_OBJDUMP:-$ROCM_PATH/lib/llvm/bin/llvm-objdump}"
READOBJ="${LLVM_READOBJ:-$ROCM_PATH/lib/llvm/bin/llvm-readobj}"
for tool in "$OBJDUMP" "$READOBJ"; do
  if [[ ! -x "$tool" ]]; then
    printf 'required ROCm LLVM tool is unavailable: %s\n' "$tool" >&2
    exit 2
  fi
done

TMP_PARENT="${TMPDIR:-/tmp}"
if [[ ! -d "$TMP_PARENT" || -L "$TMP_PARENT" || ! -w "$TMP_PARENT" ]]; then
  printf 'TMPDIR must be a writable, real directory: %s\n' "$TMP_PARENT" >&2
  exit 2
fi

BUILD_DIR="$(mktemp -d -- "$TMP_PARENT/skyrl-down-lora-gfx1100.XXXXXX")"
readonly BUILD_DIR
readonly CODE_OBJECT="$BUILD_DIR/down_lora_residual_gfx1100.hsaco"
readonly METADATA="$BUILD_DIR/down_lora_residual_gfx1100.metadata"
readonly SYMBOLS="$BUILD_DIR/down_lora_residual_gfx1100.symbols"
readonly DISASSEMBLY="$BUILD_DIR/down_lora_residual_gfx1100.disasm"

cleanup() {
  if [[ "$KEEP_OUTPUT" == "0" ]]; then
    rm -rf -- "$BUILD_DIR"
  fi
}
trap cleanup EXIT HUP INT TERM

if ! HIP_VERSION="$("$HIPCC" --version 2>&1)"; then
  printf 'the selected hipcc cannot report its version: %s\n' "$HIPCC" >&2
  exit 2
fi
if ! grep -Fq 'HIP version:' <<<"$HIP_VERSION"; then
  printf 'the selected compiler does not identify itself as hipcc:\n%s\n' \
    "$HIP_VERSION" >&2
  exit 2
fi

printf 'compiler: %s\n' "$HIPCC"
printf 'target: %s\n' "$EXPECTED_ARCH"
printf 'build directory: %s\n' "$BUILD_DIR"
printf 'compile command:'
printf ' %q' "$HIPCC" -std=c++17 -O3 -Wall -Wextra -Werror --genco \
  --no-gpu-bundle-output "--offload-arch=$EXPECTED_ARCH" "$SOURCE" -o "$CODE_OBJECT"
printf '\n'

# --genco emits an AOT code object. No command in this script opens /dev/kfd,
# enumerates devices, links a host executable, loads the HSACO, or launches it.
"$HIPCC" -std=c++17 -O3 -Wall -Wextra -Werror --genco \
  --no-gpu-bundle-output "--offload-arch=$EXPECTED_ARCH" \
  "$SOURCE" -o "$CODE_OBJECT"

"$READOBJ" --file-headers --notes "$CODE_OBJECT" >"$METADATA"
"$READOBJ" --symbols --elf-output-style=GNU "$CODE_OBJECT" >"$SYMBOLS"
"$OBJDUMP" --disassemble --mcpu="$EXPECTED_ARCH" "$CODE_OBJECT" >"$DISASSEMBLY"

grep -Fq 'Format: elf64-amdgpu' "$METADATA" || {
  printf 'output is not an ELF64 AMDGPU code object\n' >&2
  exit 1
}
grep -Fq 'EF_AMDGPU_MACH_AMDGCN_GFX1100' "$METADATA" || {
  printf 'code-object header does not target gfx1100\n' >&2
  exit 1
}

for index in "${!KERNELS[@]}"; do
  kernel="${KERNELS[$index]}"
  expected_explicit_size="${EXPLICIT_ARG_SIZES[$index]}"
  expected_full_size="${FULL_KERNARG_SIZES[$index]}"
  grep -Eq "FUNC +GLOBAL +PROTECTED +[0-9]+ ${kernel}$" "$SYMBOLS" || {
    printf 'missing exported kernel function: %s\n' "$kernel" >&2
    exit 1
  }
  grep -Fq ".name:           $kernel" "$METADATA" || {
    printf 'AMDGPU metadata is missing kernel: %s\n' "$kernel" >&2
    exit 1
  }
  argument_sizes="$({
    awk -v target="$kernel" '
      /^  - \.args:/ { explicit = ""; kernarg = ""; last_offset = "" }
      /^[[:space:]]+- \.offset:/ { last_offset = $3 }
      /\.value_kind:[[:space:]]+hidden_block_count_x/ { explicit = last_offset }
      /    \.kernarg_segment_size:/ { kernarg = $2 }
      /    \.name:/ && $2 == target { print explicit, kernarg }
    ' "$METADATA"
  } || true)"
  read -r actual_explicit_size actual_full_size <<<"$argument_sizes"
  if [[ "$actual_explicit_size" != "$expected_explicit_size" ||
        "$actual_full_size" != "$expected_full_size" ]]; then
    printf 'kernel %s has explicit/full kernarg sizes %q/%q, expected %s/%s\n' \
      "$kernel" "$actual_explicit_size" "$actual_full_size" \
      "$expected_explicit_size" "$expected_full_size" >&2
    exit 1
  fi
  grep -Fq "<$kernel>:" "$DISASSEMBLY" || {
    printf 'disassembly is missing kernel body: %s\n' "$kernel" >&2
    exit 1
  }
done

require_metadata_count() {
  local expected="$1"
  local text="$2"
  local actual
  actual="$(grep -Fc "$text" "$METADATA" || true)"
  if [[ "$actual" != "$expected" ]]; then
    printf 'metadata field %q occurred %s times, expected %s\n' \
      "$text" "$actual" "$expected" >&2
    exit 1
  fi
}

require_metadata_count 4 '.max_flat_workgroup_size: 64'
require_metadata_count 4 '.wavefront_size: 32'
require_metadata_count 4 '.uses_dynamic_stack: false'
require_metadata_count 4 '.sgpr_spill_count: 0'
require_metadata_count 4 '.vgpr_spill_count: 0'
require_metadata_count 4 '.private_segment_fixed_size: 0'
require_metadata_count 4 '.kernarg_segment_align: 8'
require_metadata_count 4 '.uniform_work_group_size: 1'
require_metadata_count 1 '.group_segment_fixed_size: 32'
require_metadata_count 3 '.group_segment_fixed_size: 0'

if grep -Eq 'FUNC +GLOBAL +[^ ]+ +UND ' "$SYMBOLS"; then
  printf 'code object contains an unresolved global function\n' >&2
  exit 1
fi

printf 'artifact bytes: %s\n' "$(stat -c '%s' -- "$CODE_OBJECT")"
printf 'verified exported kernels:\n'
for kernel in "${KERNELS[@]}"; do
  printf '  %s\n' "$kernel"
done
printf 'verified ABI: BF16 storage, FP32 accumulation/workspaces, rank<=8, '
printf 'rows<=256, K<=9216, N<=2560, 64-thread tail-safe tiles\n'

if [[ "$KEEP_OUTPUT" == "1" ]]; then
  printf 'artifacts retained under: %s\n' "$BUILD_DIR"
else
  printf 'offline compile proof passed; temporary artifacts will be removed\n'
fi
