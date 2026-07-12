#!/usr/bin/env bash
# Build the typed-FFI smoke as a host shared library plus gfx1100 device code.
# This script compiles and inspects files only; it never enumerates or opens a GPU.

set -euo pipefail
umask 077

readonly EXPECTED_ARCH="gfx1100"
readonly REQUESTED_ARCH="${SKYRL_ROCM_ARCH:-$EXPECTED_ARCH}"
readonly EXPECTED_BASENAME="libskyrl_gdn_ffi_smoke_gfx1100.so"
readonly EXPECTED_SYMBOL="skyrl_gdn_ffi_smoke_bf16_copy_v1"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../../.." && pwd -P)"
readonly SOURCE="$SCRIPT_DIR/gdn_ffi_smoke.hip"

if [[ "$REQUESTED_ARCH" != "$EXPECTED_ARCH" ]]; then
  printf 'refusing architecture %q: gdn_ffi_smoke is fixed to %s\n' \
    "$REQUESTED_ARCH" "$EXPECTED_ARCH" >&2
  exit 2
fi
if [[ "$#" != 1 ]]; then
  printf 'usage: JAXLIB_INCLUDE_DIR=/absolute/jaxlib/include %s /absolute/%s\n' \
    "$0" "$EXPECTED_BASENAME" >&2
  exit 2
fi

readonly OUTPUT="$1"
if [[ "$OUTPUT" != /* || "$(basename -- "$OUTPUT")" != "$EXPECTED_BASENAME" ]]; then
  printf 'output must be an absolute path ending in %s\n' "$EXPECTED_BASENAME" >&2
  exit 2
fi
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  printf 'refusing to overwrite output\n' >&2
  exit 2
fi

readonly OUTPUT_PARENT="$(dirname -- "$OUTPUT")"
if [[ ! -d "$OUTPUT_PARENT" || -L "$OUTPUT_PARENT" || ! -w "$OUTPUT_PARENT" ]]; then
  printf 'output parent must be a writable real directory\n' >&2
  exit 2
fi
readonly CANONICAL_PARENT="$(cd -- "$OUTPUT_PARENT" && pwd -P)"
if [[ "$OUTPUT" != "$CANONICAL_PARENT/$EXPECTED_BASENAME" ]]; then
  printf 'output path must be canonical and must not traverse symlinks or ..\n' >&2
  exit 2
fi
if [[ "$CANONICAL_PARENT" == "$REPO_ROOT" || "$CANONICAL_PARENT" == "$REPO_ROOT/"* ]]; then
  printf 'output must be outside the source repository\n' >&2
  exit 2
fi

readonly JAX_INCLUDE_REQUESTED="${JAXLIB_INCLUDE_DIR:-}"
if [[ -z "$JAX_INCLUDE_REQUESTED" || "$JAX_INCLUDE_REQUESTED" != /* ||
      ! -f "$JAX_INCLUDE_REQUESTED/xla/ffi/api/ffi.h" ]]; then
  printf 'JAXLIB_INCLUDE_DIR must be an absolute jaxlib include directory\n' >&2
  exit 2
fi
readonly JAX_INCLUDE="$(cd -- "$JAX_INCLUDE_REQUESTED" && pwd -P)"
if [[ "$JAX_INCLUDE" != "$JAX_INCLUDE_REQUESTED" ]]; then
  printf 'JAXLIB_INCLUDE_DIR must be canonical and must not traverse symlinks or ..\n' >&2
  exit 2
fi

if [[ ! -f "$SOURCE" || -L "$SOURCE" ]]; then
  printf 'gdn_ffi_smoke source is missing or is a symbolic link\n' >&2
  exit 2
fi

readonly HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
readonly LLVM_NM="${LLVM_NM:-/usr/bin/nm}"
if [[ ! -x "$HIPCC" || ! -x "$LLVM_NM" ]]; then
  printf 'executable hipcc and llvm-nm are required\n' >&2
  exit 2
fi

BUILD_OUTPUT="$(mktemp --suffix=.so -- "$CANONICAL_PARENT/.gdn-ffi-smoke.XXXXXX")"
readonly BUILD_OUTPUT
cleanup() {
  rm -f -- "$BUILD_OUTPUT"
}
trap cleanup EXIT HUP INT TERM

"$HIPCC" -std=c++17 -O3 -fPIC -shared -Wall -Wextra -Werror \
  "--offload-arch=$EXPECTED_ARCH" -isystem "$JAX_INCLUDE" \
  -Wl,-z,defs -Wl,-z,relro -Wl,-z,now \
  "$SOURCE" -o "$BUILD_OUTPUT"

SYMBOL_COUNT="$({
  "$LLVM_NM" --dynamic --defined-only "$BUILD_OUTPUT" |
    grep -Ec " [TW] ${EXPECTED_SYMBOL}$" || true
})"
readonly SYMBOL_COUNT
if [[ "$SYMBOL_COUNT" != 1 ]]; then
  printf 'built library does not export exactly one %s symbol\n' "$EXPECTED_SYMBOL" >&2
  exit 1
fi

chmod 0600 -- "$BUILD_OUTPUT"
if ! ln -- "$BUILD_OUTPUT" "$OUTPUT"; then
  printf 'could not publish output without overwriting an existing file\n' >&2
  exit 1
fi
rm -f -- "$BUILD_OUTPUT"
printf 'built %s for %s\n' "$OUTPUT" "$EXPECTED_ARCH"
