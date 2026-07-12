#!/usr/bin/env bash
set -euo pipefail
umask 077

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

if [[ $# -ne 1 || ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "usage: $0 RUN_ID" >&2
  echo "RUN_ID must start with a letter or digit and then use only letters, digits, dot, underscore, or dash." >&2
  exit 2
fi

run_id="$1"
run_root="${SKYRL_QWEN35_RUN_ROOT:-/tmp/skyrl-qwen35-runs}"
run_dir="$run_root/$run_id"
port="${SKYRL_QWEN35_PORT:-8001}"
if [[ ! "$port" =~ ^[1-9][0-9]{0,4}$ ]] || ((10#$port > 65535)); then
  echo "SKYRL_QWEN35_PORT must be an integer in [1, 65535]." >&2
  exit 2
fi

if [[ -z "$run_root" || "$run_root" != /* ]]; then
  echo "SKYRL_QWEN35_RUN_ROOT must be an absolute path." >&2
  exit 2
fi
if [[ -L "$run_root" || (-e "$run_root" && (! -d "$run_root" || ! -O "$run_root")) ]]; then
  echo "refusing unsafe run root (must be a real directory owned by this user): $run_root" >&2
  exit 2
fi
mkdir -p "$run_root"
if [[ -L "$run_root" || ! -d "$run_root" || ! -O "$run_root" ]]; then
  echo "refusing unsafe run root after creation: $run_root" >&2
  exit 2
fi
chmod 700 "$run_root"

if ! command -v flock >/dev/null 2>&1; then
  echo "refusing to launch because flock is unavailable" >&2
  exit 2
fi
lock_parent="${XDG_RUNTIME_DIR:-/tmp}"
lock_dir="$lock_parent/skyrl-qwen35-rocm-$UID"
if [[ -L "$lock_dir" || (-e "$lock_dir" && (! -d "$lock_dir" || ! -O "$lock_dir")) ]]; then
  echo "refusing unsafe global launch-lock directory: $lock_dir" >&2
  exit 2
fi
if ! mkdir -m 700 "$lock_dir" 2>/dev/null && [[ ! -d "$lock_dir" ]]; then
  echo "could not create global launch-lock directory: $lock_dir" >&2
  exit 2
fi
if [[ -L "$lock_dir" || ! -d "$lock_dir" || ! -O "$lock_dir" ]]; then
  echo "refusing unsafe global launch-lock directory after creation: $lock_dir" >&2
  exit 2
fi
chmod 700 "$lock_dir"
exec {launch_lock_fd}<"$lock_dir"
if ! flock -n "$launch_lock_fd"; then
  echo "refusing to launch while another Qwen3.5 ROCm server holds the global launch lock" >&2
  exit 2
fi

if [[ ! -c /dev/kfd || ! -r /dev/kfd || ! -w /dev/kfd ]]; then
  echo "refusing to launch because /dev/kfd is missing or inaccessible" >&2
  exit 2
fi

# Keep the desktop on the iGPU: a compute reset must not also take down the
# active display. Ignore virtual writeback connectors, whose state is unknown.
amd_cards=0
connected_amd_connectors=()
for card_path in /sys/class/drm/card[0-9]*; do
  [[ "${card_path##*/}" =~ ^card[0-9]+$ ]] || continue
  [[ -r "$card_path/device/vendor" ]] || continue
  [[ "$(<"$card_path/device/vendor")" == "0x1002" ]] || continue
  ((amd_cards += 1))
  for status_path in "$card_path"-*/status; do
    [[ -r "$status_path" ]] || continue
    [[ "${status_path%/status}" == *Writeback* ]] && continue
    if [[ "$(<"$status_path")" == "connected" ]]; then
      connected_amd_connectors+=("${status_path%/status}")
    fi
  done
done
if ((amd_cards == 0)); then
  echo "refusing to launch because no AMD DRM card was found" >&2
  exit 2
fi
if ((${#connected_amd_connectors[@]} > 0)) && [[ "${SKYRL_ALLOW_AMD_DISPLAY:-0}" != "1" ]]; then
  echo "refusing to launch while an AMD display connector is active:" >&2
  printf '  %s\n' "${connected_amd_connectors[@]}" >&2
  echo "Move the display to the iGPU, or set SKYRL_ALLOW_AMD_DISPLAY=1 after accepting reset risk." >&2
  exit 2
fi

if [[ "${SKYRL_ALLOW_EXISTING_KFD:-0}" != "1" ]]; then
  if ! command -v fuser >/dev/null 2>&1; then
    echo "refusing to launch because fuser is unavailable for the /dev/kfd ownership check" >&2
    exit 2
  fi
  kfd_owners="$(fuser /dev/kfd 2>/dev/null || true)"
  if [[ -n "${kfd_owners//[[:space:]]/}" ]]; then
    echo "refusing to launch while /dev/kfd is already owned by PID(s): $kfd_owners" >&2
    echo "Set SKYRL_ALLOW_EXISTING_KFD=1 only after verifying that sharing is intentional." >&2
    exit 2
  fi
fi

if ! python3 - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
PY
then
  echo "refusing to launch because 127.0.0.1:$port is unavailable" >&2
  exit 2
fi

if [[ ! -r .venv/bin/activate ]] || ! command -v uv >/dev/null 2>&1; then
  echo "refusing to launch because the project virtualenv or uv is unavailable" >&2
  exit 2
fi

if ! mkdir "$run_dir"; then
  echo "refusing to reuse existing run directory: $run_dir" >&2
  exit 2
fi
mkdir "$run_dir/checkpoints"

source .venv/bin/activate
export LLVM_PATH=/opt/rocm/llvm
export JAX_PLATFORMS=rocm
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/skyrl-jax}"
export HF_XET_HIGH_PERFORMANCE=1
export SKYRL_ROCM_PALLAS_ATTENTION="${SKYRL_ROCM_PALLAS_ATTENTION:-0}"

# The first full-size Adam execution completed, but replay of the same XLA
# executable produced HSA_STATUS_ERROR_INVALID_PACKET_FORMAT and an illegal
# gfx1100 command-stream opcode. Disable XLA HIP-graph command-buffer capture
# until the isolated replay probe establishes a narrower safe configuration.
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_command_buffer="

exec uv run --active --no-sync -m skyrl.tinker.api \
  --base-model Qwen/Qwen3.5-4B \
  --backend jax \
  --backend-config '{"max_lora_adapters":2,"max_lora_rank":8,"train_micro_batch_size":1,"sample_max_num_sequences":1,"gradient_checkpointing":true,"loss_chunk_size":64}' \
  --host 127.0.0.1 \
  --port "$port" \
  --checkpoints-base "$run_dir/checkpoints" \
  --database-url "sqlite:///$run_dir/tinker.db"
