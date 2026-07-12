#!/usr/bin/env bash
# Compile and inspect the gfx1100 IU8/IU4 WMMA proof without opening a GPU.

set -euo pipefail

umask 077

readonly EXPECTED_ARCH="gfx1100"
readonly REQUESTED_ARCH="${SKYRL_ROCM_ARCH:-$EXPECTED_ARCH}"
readonly KEEP_OUTPUT="${SKYRL_KEEP_COMPILE_PROBE:-0}"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SOURCE="$SCRIPT_DIR/quant_wmma_gfx1100.hip"

if [[ "$REQUESTED_ARCH" != "$EXPECTED_ARCH" ]]; then
  printf 'refusing architecture %q: this probe is valid only for gfx1100\n' "$REQUESTED_ARCH" >&2
  exit 2
fi

if [[ "$KEEP_OUTPUT" != "0" && "$KEEP_OUTPUT" != "1" ]]; then
  printf 'SKYRL_KEEP_COMPILE_PROBE must be 0 or 1, got %q\n' "$KEEP_OUTPUT" >&2
  exit 2
fi

if [[ ! -f "$SOURCE" || -L "$SOURCE" ]]; then
  printf 'probe source is missing or is a symbolic link: %s\n' "$SOURCE" >&2
  exit 2
fi

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

BUILD_DIR="$(mktemp -d -- "$TMP_PARENT/skyrl-quant-wmma-gfx1100.XXXXXX")"
readonly BUILD_DIR
readonly CODE_OBJECT="$BUILD_DIR/quant_wmma_gfx1100.hsaco"
readonly DISASSEMBLY="$BUILD_DIR/quant_wmma_gfx1100.disasm"
readonly SYMBOLS="$BUILD_DIR/quant_wmma_gfx1100.symbols"

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
  printf 'the selected compiler does not identify itself as hipcc:\n%s\n' "$HIP_VERSION" >&2
  exit 2
fi

printf 'compiler: %s\n' "$HIPCC"
printf 'target: %s\n' "$EXPECTED_ARCH"
printf 'build directory: %s\n' "$BUILD_DIR"
printf 'compile command:'
printf ' %q' "$HIPCC" -O3 --genco --no-gpu-bundle-output \
  "--offload-arch=$EXPECTED_ARCH" "$SOURCE" -o "$CODE_OBJECT"
printf '\n'

# --genco performs compilation/code-object generation only. It neither probes
# the installed devices nor submits work to /dev/kfd.
"$HIPCC" -O3 --genco --no-gpu-bundle-output \
  "--offload-arch=$EXPECTED_ARCH" "$SOURCE" -o "$CODE_OBJECT"

"$OBJDUMP" --disassemble --mcpu="$EXPECTED_ARCH" "$CODE_OBJECT" >"$DISASSEMBLY"
"$READOBJ" --symbols "$CODE_OBJECT" >"$SYMBOLS"

grep -Fq 'skyrl_wmma_iu8_fragment_compile_probe' "$SYMBOLS" || {
  printf 'compiled code object is missing the IU8 probe kernel\n' >&2
  exit 1
}
grep -Fq 'skyrl_wmma_iu4_fragment_compile_probe' "$SYMBOLS" || {
  printf 'compiled code object is missing the IU4 probe kernel\n' >&2
  exit 1
}
grep -Eq 'v_wmma_i32_16x16x16_iu8' "$DISASSEMBLY" || {
  printf 'gfx1100 disassembly contains no IU8 WMMA instruction\n' >&2
  exit 1
}
grep -Eq 'v_wmma_i32_16x16x16_iu4' "$DISASSEMBLY" || {
  printf 'gfx1100 disassembly contains no IU4 WMMA instruction\n' >&2
  exit 1
}

printf 'artifact bytes: %s\n' "$(stat -c '%s' -- "$CODE_OBJECT")"
printf 'defined kernels:\n'
grep -E 'skyrl_wmma_iu(8|4)_fragment_compile_probe' "$SYMBOLS"
printf 'verified instructions:\n'
grep -E -m 2 'v_wmma_i32_16x16x16_iu(8|4)' "$DISASSEMBLY"

if [[ "$KEEP_OUTPUT" == "1" ]]; then
  printf 'artifacts retained under: %s\n' "$BUILD_DIR"
else
  printf 'compile-only proof passed; temporary artifacts will be removed\n'
fi
