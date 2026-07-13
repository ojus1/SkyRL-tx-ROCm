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
model_repo="Qwen/Qwen3.5-4B"
model_revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
memory_mode="${SKYRL_QWEN35_MEMORY_MODE:-growth}"
prewarm_buckets="${SKYRL_QWEN35_PREWARM_BUCKETS:-}"
prewarm_optimizer="${SKYRL_QWEN35_PREWARM_OPTIMIZER:-0}"
backend_config='{"max_lora_adapters":2,"max_lora_rank":8,"train_micro_batch_size":1,"sample_max_num_sequences":1,"gradient_checkpointing":true,"loss_chunk_size":64}'

case "$prewarm_optimizer" in
  0|1) ;;
  *)
    echo "SKYRL_QWEN35_PREWARM_OPTIMIZER must be exactly 0 or 1." >&2
    exit 2
    ;;
esac
if [[ "$prewarm_optimizer" == "1" && -z "$prewarm_buckets" ]]; then
  echo "SKYRL_QWEN35_PREWARM_OPTIMIZER=1 requires nonempty SKYRL_QWEN35_PREWARM_BUCKETS." >&2
  exit 2
fi

require_unset_or_exact() {
  local name="$1"
  local expected="$2"
  local value
  if [[ -v "$name" ]]; then
    value="${!name}"
    if [[ "$value" != "$expected" ]]; then
      echo "$name=$value conflicts with SKYRL_QWEN35_MEMORY_MODE=preallocate85 (required: $expected)." >&2
      exit 2
    fi
  fi
}

require_unset_or_false() {
  local name="$1"
  local value
  if [[ -v "$name" ]]; then
    value="${!name}"
    if [[ "$value" != "false" ]]; then
      echo "$name=$value conflicts with graph-free startup (required: false)." >&2
      exit 2
    fi
  fi
}

require_unset_or_false JAX_ENABLE_PGLE
require_unset_or_false JAX_COMPILATION_CACHE_EXPECT_PGLE
export JAX_ENABLE_PGLE=false
export JAX_COMPILATION_CACHE_EXPECT_PGLE=false

case "$memory_mode" in
  growth)
    ;;
  preallocate85)
    require_unset_or_exact JAX_PLATFORMS rocm
    require_unset_or_exact XLA_PYTHON_CLIENT_ALLOCATOR bfc
    if [[ -v XLA_PYTHON_CLIENT_PREALLOCATE ]] \
      && [[ "${XLA_PYTHON_CLIENT_PREALLOCATE,,}" != "true" ]] \
      && [[ "$XLA_PYTHON_CLIENT_PREALLOCATE" != "1" ]]; then
      echo "XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE conflicts with SKYRL_QWEN35_MEMORY_MODE=preallocate85." >&2
      exit 2
    fi
    require_unset_or_exact XLA_CLIENT_MEM_FRACTION 0.85
    if [[ -n "${XLA_PYTHON_CLIENT_MEM_FRACTION:-}" ]]; then
      echo "XLA_PYTHON_CLIENT_MEM_FRACTION is deprecated and conflicts with SKYRL_QWEN35_MEMORY_MODE=preallocate85." >&2
      exit 2
    fi
    require_unset_or_exact ROCR_VISIBLE_DEVICES 0
    require_unset_or_exact HIP_VISIBLE_DEVICES 0
    require_unset_or_exact GPU_DEVICE_ORDINAL 0
    backend_config='{"max_lora_adapters":2,"max_lora_rank":8,"train_micro_batch_size":1,"sample_max_num_sequences":1,"gradient_checkpointing":true,"loss_chunk_size":64,"abstract_model_load":true}'
    ;;
  *)
    echo "SKYRL_QWEN35_MEMORY_MODE must be growth or preallocate85." >&2
    exit 2
    ;;
esac

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

if ! python3 -m rocm.amdgpu_safety >/dev/null; then
  echo "refusing to launch until the fatal AMDGPU boot quarantine is cleared by a reboot" >&2
  exit 2
fi

if [[ ! -c /dev/kfd || ! -r /dev/kfd || ! -w /dev/kfd ]]; then
  echo "refusing to launch because /dev/kfd is missing or inaccessible" >&2
  exit 2
fi

# Keep the desktop on the iGPU: a compute reset must not also take down the
# active display. Ignore virtual writeback connectors, whose state is unknown.
amd_cards=0
amd_device_ids=()
connected_amd_connectors=()
for card_path in /sys/class/drm/card[0-9]*; do
  [[ "${card_path##*/}" =~ ^card[0-9]+$ ]] || continue
  [[ -r "$card_path/device/vendor" ]] || continue
  [[ "$(<"$card_path/device/vendor")" == "0x1002" ]] || continue
  ((amd_cards += 1))
  if [[ ! -r "$card_path/device/device" ]]; then
    echo "refusing to launch because the AMD PCI device ID is unreadable at $card_path" >&2
    exit 2
  fi
  amd_device_ids+=("$(<"$card_path/device/device")")
  for status_path in "$card_path"-*/status; do
    [[ -r "$status_path" ]] || continue
    [[ "${status_path%/status}" == *Writeback* ]] && continue
    if [[ "$(<"$status_path")" == "connected" ]]; then
      connected_amd_connectors+=("${status_path%/status}")
    fi
  done
done
if ((amd_cards != 1)) || [[ "${amd_device_ids[0]:-}" != "0x744c" ]]; then
  echo "refusing to launch without exactly one AMD DRM GPU at PCI device 0x744c (observed: ${amd_device_ids[*]:-none})" >&2
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
  if kfd_owners="$(fuser /dev/kfd 2>&1)"; then
    echo "refusing to launch while /dev/kfd is already owned by PID(s): $kfd_owners" >&2
    echo "Set SKYRL_ALLOW_EXISTING_KFD=1 only after verifying that sharing is intentional." >&2
    exit 2
  else
    fuser_status=$?
    if ((fuser_status != 1)) || [[ -n "${kfd_owners//[[:space:]]/}" ]]; then
      echo "could not verify exclusive /dev/kfd ownership with fuser: ${kfd_owners:-return code $fuser_status}" >&2
      exit 2
    fi
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
if ! model_path="$({
  python - "$model_repo" "$model_revision" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

model_repo, revision = sys.argv[1:]
path = Path(
    snapshot_download(
        model_repo,
        revision=revision,
        allow_patterns=("*.safetensors", "*.json", "*.txt", "*.jinja"),
        local_files_only=True,
    )
).resolve()
if path.name != revision:
    raise RuntimeError(f"resolved revision {path.name!r}, expected {revision!r}")
print(path)
PY
} 2>/dev/null)"; then
  echo "refusing to launch because pinned $model_repo revision $model_revision is not fully cached" >&2
  exit 2
fi
export LLVM_PATH=/opt/rocm/llvm
export JAX_PLATFORMS=rocm
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
if [[ "$memory_mode" == "growth" ]]; then
  # Preserve the validated baseline behavior and inherited allocator/fraction.
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
else
  export ROCR_VISIBLE_DEVICES=0
  export HIP_VISIBLE_DEVICES=0
  export GPU_DEVICE_ORDINAL=0
  export XLA_PYTHON_CLIENT_ALLOCATOR=bfc
  export XLA_PYTHON_CLIENT_PREALLOCATE=true
  export XLA_CLIENT_MEM_FRACTION=0.85
  unset XLA_PYTHON_CLIENT_MEM_FRACTION
fi

# Spend startup/disk to avoid repeating backend compilation and per-fusion
# autotuning after a restart.  The cache contains trusted executable content,
# so keep it private and namespace it by every installed GPU-stack component.
# Source/HLO changes still participate in JAX's cache key.  Future external FFI
# libraries must additionally version this namespace by their exact hash.
expected_jax_stack="0.10.2,0.10.2,0.10.2,0.10.2"
if ! installed_jax_stack="$(python - <<'PY'
from importlib.metadata import version

print(
    ",".join(
        version(package)
        for package in ("jax", "jaxlib", "jax-rocm7-plugin", "jax-rocm7-pjrt")
    )
)
PY
)"; then
  echo "refusing to configure the trusted JAX cache because stack-version discovery failed" >&2
  exit 2
fi
if [[ "$installed_jax_stack" != "$expected_jax_stack" ]]; then
  echo "refusing stale JAX cache namespace for stack $installed_jax_stack (expected $expected_jax_stack)" >&2
  exit 2
fi
if [[ ! -r /opt/rocm/.info/version || "$(</opt/rocm/.info/version)" != "7.2.4" ]]; then
  echo "refusing stale JAX cache namespace because ROCm is not exactly 7.2.4" >&2
  exit 2
fi
if [[ ! -r /sys/module/amdgpu/version || "$(</sys/module/amdgpu/version)" != "6.16.13" ]]; then
  echo "refusing stale JAX cache namespace because AMDGPU is not exactly 6.16.13" >&2
  exit 2
fi
if [[ -v JAX_COMPILATION_CACHE_DIR ]]; then
  echo "refusing inherited JAX_COMPILATION_CACHE_DIR; the trusted cache path is stack-versioned by the launcher" >&2
  exit 2
fi
if ! jax_cache_dir="$(
  python rocm/prepare_jax_cache_dir.py \
    --max-autotune-bytes 4294967296
)"; then
  echo "refusing to start without a private, validated JAX compilation cache" >&2
  exit 2
fi
export JAX_COMPILATION_CACHE_DIR="$jax_cache_dir"
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_RAISE_PERSISTENT_CACHE_ERRORS=true
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=xla_gpu_per_fusion_autotune_cache_dir
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
# This 16 GiB JAX LRU covers top-level serialized executable entries only.
# prepare_jax_cache_dir.py separately refuses an autotune subtree above 4 GiB
# at startup; XLA's per-fusion textproto files are not part of this LRU.
export JAX_COMPILATION_CACHE_MAX_SIZE=17179869184
export HF_XET_HIGH_PERFORMANCE=1
export SKYRL_ROCM_PALLAS_ATTENTION="${SKYRL_ROCM_PALLAS_ATTENTION:-0}"

# The first full-size Adam execution completed, but replay of the same XLA
# executable produced HSA_STATUS_ERROR_INVALID_PACKET_FORMAT and an illegal
# gfx1100 command-stream opcode. Disable XLA HIP-graph command-buffer capture
# until the isolated replay probe establishes a narrower safe configuration.
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

# Default-off, compile-only static-bucket prewarm. This populates the trusted
# persistent cache before the API starts, but never invokes a compiled model
# pass or optimizer step. ROCm compilation may still run bounded autotuning.
prewarm_status=0
if [[ -n "$prewarm_buckets" ]]; then
  prewarm_optimizer_args=()
  if [[ "$prewarm_optimizer" == "1" ]]; then
    prewarm_optimizer_args=(--compile-optimizer)
  fi
  prewarm_construction=eager
  if [[ "$memory_mode" == "preallocate85" ]]; then
    prewarm_construction=abstract-load
  fi
  case "$SKYRL_ROCM_PALLAS_ATTENTION" in
    0) prewarm_attention=xla ;;
    1) prewarm_attention=pallas ;;
    *)
      echo "SKYRL_ROCM_PALLAS_ATTENTION must be 0 or 1 for startup prewarm." >&2
      exit 2
      ;;
  esac
  if python rocm/prewarm_qwen35_buckets.py \
      --execute-rocm \
      --allow-gpu \
      "${prewarm_optimizer_args[@]}" \
      --buckets "$prewarm_buckets" \
      --model-path "$model_path" \
      --construction "$prewarm_construction" \
      --attention-backend "$prewarm_attention" \
      --launcher-lock-fd "$launch_lock_fd" \
      --output "$run_dir/prewarm.jsonl"; then
    prewarm_status=0
  else
    prewarm_status=$?
  fi
fi

final_journal_status=0
if python3 -m rocm.amdgpu_safety >/dev/null; then
  final_journal_status=0
else
  final_journal_status=$?
fi
if ((final_journal_status != 0)); then
  echo "refusing API start because the final AMDGPU boot-journal postflight failed" >&2
  exit 2
fi
if ((prewarm_status != 0)); then
  echo "refusing API start because startup prewarm failed with status $prewarm_status" >&2
  exit "$prewarm_status"
fi

exec uv run --active --no-sync -m skyrl.tinker.api \
  --base-model "$model_path" \
  --backend jax \
  --backend-config "$backend_config" \
  --host 127.0.0.1 \
  --port "$port" \
  --checkpoints-base "$run_dir/checkpoints" \
  --database-url "sqlite:///$run_dir/tinker.db"
