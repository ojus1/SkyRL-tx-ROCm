#!/bin/bash -p
set -euo pipefail
umask 077

if [[ "$-" != *p* ]]; then
  echo "refusing non-privileged Bash startup; execute this launcher directly so its /bin/bash -p shebang is honored" >&2
  exit 2
fi
for injection_name in BASH_ENV ENV LD_AUDIT LD_LIBRARY_PATH LD_PRELOAD; do
  if [[ -v "$injection_name" && -n "${!injection_name}" ]]; then
    echo "refusing inherited interpreter or dynamic-loader injection variable: $injection_name" >&2
    exit 2
  fi
done
for internal_claim_name in \
  SKYRL_QWEN35_RUNTIME_T64_CACHE_ATTEST \
  SKYRL_QWEN35_RUNTIME_PREWARM_AUDIT_PATH \
  SKYRL_QWEN35_RUNTIME_PREWARM_AUDIT_SHA256 \
  SKYRL_QWEN35_RUNTIME_PREWARM_HANDOFF_PATH \
  SKYRL_QWEN35_RUNTIME_PREWARM_HANDOFF_SHA256; do
  if [[ -v "$internal_claim_name" ]]; then
    echo "refusing inherited launcher-owned cache-attestation claim: $internal_claim_name" >&2
    exit 2
  fi
done
while IFS= read -r exported_name; do
  if [[ "$exported_name" == BASH_FUNC_*%% ]]; then
    echo "refusing inherited exported Bash function: $exported_name" >&2
    exit 2
  fi
done < <(compgen -e)

launcher_invocation="${BASH_SOURCE[0]}"
if [[ "$launcher_invocation" != /* ]]; then
  launcher_invocation="$(pwd -P)/$launcher_invocation"
fi
if [[ -L "$launcher_invocation" || ! -f "$launcher_invocation" || ! -O "$launcher_invocation" ]]; then
  echo "refusing a symlinked, non-regular, or foreign-owned launcher" >&2
  exit 2
fi
launcher_directory="$(cd -P -- "${launcher_invocation%/*}" && pwd -P)"
launcher_path="$launcher_directory/${launcher_invocation##*/}"
repo="$(cd -P -- "$launcher_directory/.." && pwd -P)"
if [[ "$launcher_path" != "$repo/rocm/start_qwen35.sh" ]]; then
  echo "refusing launcher path outside the canonical repository location" >&2
  exit 2
fi
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
prewarm_optimizer="${SKYRL_QWEN35_PREWARM_OPTIMIZER-0}"
prewarm_only="${SKYRL_QWEN35_PREWARM_ONLY-0}"
prewarm_timeout_seconds="${SKYRL_QWEN35_PREWARM_TIMEOUT_SECONDS:-3600}"
engine_t64_cache_attest="${SKYRL_QWEN35_ENGINE_T64_CACHE_ATTEST-0}"
engine_start_gate_dir="${SKYRL_QWEN35_ENGINE_START_GATE_DIR-}"
engine_start_gate_timeout_seconds="${SKYRL_QWEN35_ENGINE_START_GATE_TIMEOUT_SECONDS-300}"
engine_start_gate_enabled=0
bf16_rms_gate_up_lora_swiglu_contiguous="${SKYRL_QWEN35_BF16_RMS_GATE_UP_LORA_SWIGLU_CONTIGUOUS-0}"
runtime_cache_attestation_environment=()

case "$prewarm_optimizer" in
  0|1) ;;
  *)
    echo "SKYRL_QWEN35_PREWARM_OPTIMIZER must be exactly 0 or 1." >&2
    exit 2
    ;;
esac
case "$prewarm_only" in
  0|1) ;;
  *)
    echo "SKYRL_QWEN35_PREWARM_ONLY must be exactly 0 or 1." >&2
    exit 2
    ;;
esac
case "$engine_t64_cache_attest" in
  0|1) ;;
  *)
    echo "SKYRL_QWEN35_ENGINE_T64_CACHE_ATTEST must be exactly 0 or 1." >&2
    exit 2
    ;;
esac
case "$bf16_rms_gate_up_lora_swiglu_contiguous" in
  0) bf16_rms_gate_up_lora_swiglu_contiguous_json=false ;;
  1) bf16_rms_gate_up_lora_swiglu_contiguous_json=true ;;
  *)
    echo "SKYRL_QWEN35_BF16_RMS_GATE_UP_LORA_SWIGLU_CONTIGUOUS must be exactly 0 or 1." >&2
    exit 2
    ;;
esac
backend_config='{"max_lora_adapters":2,"max_lora_rank":8,"train_micro_batch_size":1,"sample_max_num_sequences":1,"gradient_checkpointing":true,"loss_chunk_size":64,"qwen35_bf16_down_lora_residual":false,"qwen35_bf16_rms_gate_up_lora_swiglu_contiguous":'"$bf16_rms_gate_up_lora_swiglu_contiguous_json"',"abstract_model_load":false}'
if [[ "$prewarm_optimizer" == "1" && -z "$prewarm_buckets" ]]; then
  echo "SKYRL_QWEN35_PREWARM_OPTIMIZER=1 requires nonempty SKYRL_QWEN35_PREWARM_BUCKETS." >&2
  exit 2
fi
if [[ "$prewarm_only" == "1" && -z "$prewarm_buckets" ]]; then
  echo "SKYRL_QWEN35_PREWARM_ONLY=1 requires nonempty SKYRL_QWEN35_PREWARM_BUCKETS." >&2
  exit 2
fi
if [[ "${engine_t64_cache_attest:-0}" == "1" ]]; then
  if [[ "$prewarm_only" != "0" ]]; then
    echo "SKYRL_QWEN35_ENGINE_T64_CACHE_ATTEST=1 requires SKYRL_QWEN35_PREWARM_ONLY=0." >&2
    exit 2
  fi
  compact_prewarm_buckets="${prewarm_buckets//[[:space:]]/}"
  case ",$compact_prewarm_buckets," in
    *,64,*) ;;
    *)
      echo "SKYRL_QWEN35_ENGINE_T64_CACHE_ATTEST=1 requires exact bucket 64 in SKYRL_QWEN35_PREWARM_BUCKETS." >&2
      exit 2
      ;;
  esac
fi
if [[ ! "$prewarm_timeout_seconds" =~ ^[0-9]{3,5}$ ]] \
  || ((10#$prewarm_timeout_seconds < 600 || 10#$prewarm_timeout_seconds > 14400)); then
  echo "SKYRL_QWEN35_PREWARM_TIMEOUT_SECONDS must be an integer in [600, 14400]." >&2
  exit 2
fi
if [[ -z "$engine_start_gate_dir" \
  && -v SKYRL_QWEN35_ENGINE_START_GATE_TIMEOUT_SECONDS ]]; then
  echo "SKYRL_QWEN35_ENGINE_START_GATE_TIMEOUT_SECONDS requires nonempty SKYRL_QWEN35_ENGINE_START_GATE_DIR." >&2
  exit 2
fi
if [[ -n "$engine_start_gate_dir" && "$prewarm_only" != "0" ]]; then
  echo "SKYRL_QWEN35_ENGINE_START_GATE_DIR requires SKYRL_QWEN35_PREWARM_ONLY=0." >&2
  exit 2
fi
if [[ -n "$engine_start_gate_dir" && "$prewarm_only" == "0" ]]; then
  engine_start_gate_enabled=1
  if [[ "$engine_t64_cache_attest" != "1" ]]; then
    echo "SKYRL_QWEN35_ENGINE_START_GATE_DIR requires SKYRL_QWEN35_ENGINE_T64_CACHE_ATTEST=1." >&2
    exit 2
  fi
  if [[ -z "$prewarm_buckets" ]]; then
    echo "SKYRL_QWEN35_ENGINE_START_GATE_DIR requires nonempty SKYRL_QWEN35_PREWARM_BUCKETS." >&2
    exit 2
  fi
  if [[ ! "$engine_start_gate_timeout_seconds" =~ ^[0-9]{1,4}$ ]] \
    || ((10#$engine_start_gate_timeout_seconds < 1 \
      || 10#$engine_start_gate_timeout_seconds > 3600)); then
    echo "SKYRL_QWEN35_ENGINE_START_GATE_TIMEOUT_SECONDS must be an integer in [1, 3600]." >&2
    exit 2
  fi
  if [[ "$engine_start_gate_dir" != /* || "$engine_start_gate_dir" == */ ]] \
    || [[ -L "$engine_start_gate_dir" || ! -d "$engine_start_gate_dir" ]] \
    || [[ "$(cd -P -- "$engine_start_gate_dir" && pwd -P)" \
      != "$engine_start_gate_dir" ]] \
    || [[ "$(/usr/bin/stat -c '%u:%a:%F' -- "$engine_start_gate_dir")" \
      != "$EUID:700:directory" ]]; then
    echo "SKYRL_QWEN35_ENGINE_START_GATE_DIR must be an absolute canonical, owned mode-0700 directory without symlinks." >&2
    exit 2
  fi
  for engine_start_gate_marker in \
    "$engine_start_gate_dir/engine-start.ready" \
    "$engine_start_gate_dir/engine-start.release" \
    "$engine_start_gate_dir/engine-start.watchdog.telemetry.jsonl" \
    "$engine_start_gate_dir/engine-start.watchdog.telemetry.jsonl.summary.json"; do
    if [[ -e "$engine_start_gate_marker" || -L "$engine_start_gate_marker" ]]; then
      echo "refusing a stale engine-start gate marker: $engine_start_gate_marker" >&2
      exit 2
    fi
  done
  for engine_start_gate_executable in \
    /usr/bin/fuser /usr/bin/mv /usr/bin/rm /usr/bin/sleep; do
    if [[ ! -x "$engine_start_gate_executable" ]]; then
      echo "refusing engine-start gate because a required executable is unavailable: $engine_start_gate_executable" >&2
      exit 2
    fi
  done
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
    backend_config='{"max_lora_adapters":2,"max_lora_rank":8,"train_micro_batch_size":1,"sample_max_num_sequences":1,"gradient_checkpointing":true,"loss_chunk_size":64,"qwen35_bf16_down_lora_residual":false,"qwen35_bf16_rms_gate_up_lora_swiglu_contiguous":'"$bf16_rms_gate_up_lora_swiglu_contiguous_json"',"abstract_model_load":true}'
    ;;
  *)
    echo "SKYRL_QWEN35_MEMORY_MODE must be growth or preallocate85." >&2
    exit 2
    ;;
esac

if [[ -n "${ROCR_VISIBLE_DEVICES:-}" && "$ROCR_VISIBLE_DEVICES" != "0" ]]; then
  echo "ROCR_VISIBLE_DEVICES must be unset or exactly 0 for the qualified Qwen3.5 server." >&2
  exit 2
fi
case "${SKYRL_ROCM_PALLAS_ATTENTION:-0}" in
  0|1) ;;
  *)
    echo "SKYRL_ROCM_PALLAS_ATTENTION must be exactly 0 or 1." >&2
    exit 2
    ;;
esac

if [[ -n "$prewarm_buckets" ]]; then
  while IFS= read -r exported_name; do
    case "$exported_name" in
      PYTHON*|UV_*|__PYVENV_LAUNCHER__|VIRTUAL_ENV|VIRTUAL_ENV_PROMPT)
        echo "refusing inherited Python, uv, or virtualenv startup variable during operational prewarm: $exported_name" >&2
        exit 2
        ;;
    esac
  done < <(compgen -e)
  for executable in /usr/bin/chmod /usr/bin/env /usr/bin/getent /usr/bin/git /usr/bin/mkdir /usr/bin/python3.12 /usr/bin/sha256sum /usr/bin/stat /usr/bin/tar; do
    if [[ ! -x "$executable" ]]; then
      echo "refusing prewarm because required source-attestation executable is unavailable: $executable" >&2
      exit 2
    fi
  done
  source_git=(
    /usr/bin/env -i
    GIT_CONFIG_GLOBAL=/dev/null
    GIT_CONFIG_NOSYSTEM=1
    GIT_NO_REPLACE_OBJECTS=1
    GIT_OPTIONAL_LOCKS=0
    HOME=/nonexistent
    LC_ALL=C
    PATH=/usr/bin:/bin
    XDG_CONFIG_HOME=/nonexistent
    /usr/bin/git
    -c core.fsmonitor=false
    -c core.untrackedCache=false
    -C "$repo"
  )
  if ! source_git_top_level="$("${source_git[@]}" rev-parse --show-toplevel 2>&1)" \
    || [[ "$source_git_top_level" != "$repo" ]]; then
    echo "refusing prewarm because Git top-level does not match the repository" >&2
    exit 2
  fi
  if ! source_git_head="$("${source_git[@]}" rev-parse --verify 'HEAD^{commit}' 2>&1)" \
    || [[ ! "$source_git_head" =~ ^[0-9a-f]{40}$|^[0-9a-f]{64}$ ]]; then
    echo "refusing prewarm because Git HEAD commit is unavailable or malformed" >&2
    exit 2
  fi
  if ! source_git_status="$("${source_git[@]}" status --porcelain=v1 --untracked-files=all --ignore-submodules=none 2>&1)"; then
    echo "refusing prewarm because Git worktree status is unavailable" >&2
    exit 2
  fi
  if [[ -n "$source_git_status" ]]; then
    echo "refusing operational ROCm prewarm from a dirty worktree" >&2
    exit 2
  fi

  source_index_flags="$("${source_git[@]}" ls-files -v -- rocm/prewarm_qwen35_buckets.py rocm/profile_rocm.py rocm/start_qwen35.sh rocm/verified_source_bootstrap.py 2>&1)" || {
    echo "refusing prewarm because runtime-source index flags are unavailable" >&2
    exit 2
  }
  expected_source_index_flags=$'H rocm/prewarm_qwen35_buckets.py\nH rocm/profile_rocm.py\nH rocm/start_qwen35.sh\nH rocm/verified_source_bootstrap.py'
  if [[ "$source_index_flags" != "$expected_source_index_flags" ]]; then
    echo "refusing prewarm because runtime source has assume-unchanged, skip-worktree, or unexpected index state" >&2
    exit 2
  fi

  source_git_tree="$("${source_git[@]}" rev-parse --verify "$source_git_head^{tree}" 2>&1)" || {
    echo "refusing prewarm because the exact Git tree is unavailable" >&2
    exit 2
  }
  if [[ ! "$source_git_tree" =~ ^[0-9a-f]{40}$|^[0-9a-f]{64}$ ]]; then
    echo "refusing prewarm because the exact Git tree object ID is malformed" >&2
    exit 2
  fi

  source_blob_oid() {
    local relative_path="$1"
    local expected_mode="$2"
    local absolute_path="$repo/$relative_path"
    local tree_record
    local head_oid
    local working_oid
    if ! tree_record="$("${source_git[@]}" ls-tree "$source_git_head" -- "$relative_path" 2>&1)"; then
      return 1
    fi
    head_oid="${tree_record#"$expected_mode blob "}"
    head_oid="${head_oid%%$'\t'*}"
    if [[ ! "$head_oid" =~ ^[0-9a-f]{40}$|^[0-9a-f]{64}$ ]] \
      || [[ "$tree_record" != "$expected_mode blob $head_oid"$'\t'"$relative_path" ]]; then
      return 1
    fi
    if ! working_oid="$("${source_git[@]}" hash-object --no-filters -- "$absolute_path" 2>&1)" \
      || [[ "$working_oid" != "$head_oid" ]]; then
      return 1
    fi
    printf '%s\n' "$head_oid"
  }

  source_sha256() {
    local source_path="$1"
    local hash_output
    local digest
    if ! hash_output="$(/usr/bin/env -i LC_ALL=C PATH=/usr/bin:/bin /usr/bin/sha256sum -- "$source_path" 2>&1)"; then
      return 1
    fi
    digest="${hash_output%% *}"
    if [[ ! "$digest" =~ ^[0-9a-f]{64}$ ]] \
      || [[ "$hash_output" != "$digest  $source_path" ]]; then
      return 1
    fi
    printf '%s\n' "$digest"
  }
  source_blob_sha256() {
    local blob_oid="$1"
    local hash_output
    local digest
    if ! hash_output="$("${source_git[@]}" cat-file blob "$blob_oid" | /usr/bin/env -i LC_ALL=C PATH=/usr/bin:/bin /usr/bin/sha256sum 2>&1)"; then
      return 1
    fi
    digest="${hash_output%% *}"
    if [[ ! "$digest" =~ ^[0-9a-f]{64}$ ]] || [[ "$hash_output" != "$digest  -" ]]; then
      return 1
    fi
    printf '%s\n' "$digest"
  }
  launcher_source_path="$launcher_path"
  prewarm_source_path="$repo/rocm/prewarm_qwen35_buckets.py"
  profile_source_path="$repo/rocm/profile_rocm.py"
  bootstrap_source_path="$repo/rocm/verified_source_bootstrap.py"
  if ! launcher_source_blob_oid="$(source_blob_oid rocm/start_qwen35.sh 100755)" \
    || ! prewarm_source_blob_oid="$(source_blob_oid rocm/prewarm_qwen35_buckets.py 100644)" \
    || ! profile_source_blob_oid="$(source_blob_oid rocm/profile_rocm.py 100644)" \
    || ! bootstrap_source_blob_oid="$(source_blob_oid rocm/verified_source_bootstrap.py 100644)" \
    || ! launcher_source_sha256="$(source_sha256 "$launcher_source_path")" \
    || ! prewarm_source_sha256="$(source_sha256 "$prewarm_source_path")" \
    || ! profile_source_sha256="$(source_sha256 "$profile_source_path")" \
    || ! bootstrap_source_sha256="$(source_sha256 "$bootstrap_source_path")" \
    || ! launcher_blob_sha256="$(source_blob_sha256 "$launcher_source_blob_oid")" \
    || ! prewarm_blob_sha256="$(source_blob_sha256 "$prewarm_source_blob_oid")" \
    || ! profile_blob_sha256="$(source_blob_sha256 "$profile_source_blob_oid")" \
    || ! bootstrap_blob_sha256="$(source_blob_sha256 "$bootstrap_source_blob_oid")" \
    || [[ "$launcher_source_sha256" != "$launcher_blob_sha256" ]] \
    || [[ "$prewarm_source_sha256" != "$prewarm_blob_sha256" ]] \
    || [[ "$profile_source_sha256" != "$profile_blob_sha256" ]] \
    || [[ "$bootstrap_source_sha256" != "$bootstrap_blob_sha256" ]]; then
    echo "refusing prewarm because exact runtime source does not match its HEAD blob" >&2
    exit 2
  fi
  export SKYRL_QWEN35_GIT_HEAD="$source_git_head"
  export SKYRL_QWEN35_GIT_TREE="$source_git_tree"
  export SKYRL_QWEN35_GIT_WORKTREE_CLEAN=true
  export SKYRL_QWEN35_LAUNCHER_BLOB_OID="$launcher_source_blob_oid"
  export SKYRL_QWEN35_LAUNCHER_SHA256="$launcher_source_sha256"
  export SKYRL_QWEN35_PREWARM_BLOB_OID="$prewarm_source_blob_oid"
  export SKYRL_QWEN35_PREWARM_SHA256="$prewarm_source_sha256"
  export SKYRL_QWEN35_BOOTSTRAP_BLOB_OID="$bootstrap_source_blob_oid"
  export SKYRL_QWEN35_BOOTSTRAP_SHA256="$bootstrap_source_sha256"
fi
if [[ -n "$prewarm_buckets" ]]; then
  export PATH=/opt/rocm/bin:/usr/bin:/bin
fi

run_root="${SKYRL_QWEN35_RUN_ROOT:-/tmp/skyrl-qwen35-runs}"
run_dir="$run_root/$run_id"
verified_python=()
verified_basic_environment=()
verified_runtime_environment=()
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

if [[ ! -x /usr/bin/flock ]]; then
  echo "refusing to launch because flock is unavailable" >&2
  exit 2
fi
lock_parent="/run/user/$UID"
if [[ -L "$lock_parent" || ! -d "$lock_parent" || ! -O "$lock_parent" \
  || "$(/usr/bin/stat -c '%a' -- "$lock_parent")" != "700" ]]; then
  echo "refusing unsafe fixed global launch-lock parent: $lock_parent" >&2
  exit 2
fi
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
if ! /usr/bin/flock -n "$launch_lock_fd"; then
  echo "refusing to launch while another Qwen3.5 ROCm server holds the global launch lock" >&2
  exit 2
fi
export SKYRL_QWEN35_LAUNCH_LOCK_FD="$launch_lock_fd"

if ! /usr/bin/mkdir "$run_dir"; then
  echo "refusing to reuse existing run directory: $run_dir" >&2
  exit 2
fi
/usr/bin/mkdir "$run_dir/checkpoints"

if [[ -n "$prewarm_buckets" ]]; then
  if ! passwd_record="$(/usr/bin/env -i LC_ALL=C PATH=/usr/bin:/bin /usr/bin/getent passwd "$UID")"; then
    echo "refusing prewarm because the account home cannot be resolved" >&2
    exit 2
  fi
  IFS=: read -r passwd_name _passwd_token passwd_uid passwd_gid _passwd_gecos source_account_home passwd_shell <<<"$passwd_record"
  if [[ -z "$passwd_name" || "$passwd_uid" != "$UID" || -z "$passwd_gid" \
    || "$source_account_home" != /* || -z "$passwd_shell" \
    || -L "$source_account_home" || ! -d "$source_account_home" \
    || ! -O "$source_account_home" ]]; then
    echo "refusing prewarm because the account database returned an unsafe home" >&2
    exit 2
  fi
  source_account_home="$(cd -P -- "$source_account_home" && pwd -P)"
  venv_site_packages="$repo/.venv/lib/python3.12/site-packages"
  if [[ -L "$venv_site_packages" || ! -d "$venv_site_packages" \
    || ! -O "$venv_site_packages" \
    || "$(cd -P -- "$venv_site_packages" && pwd -P)" != "$venv_site_packages" ]]; then
    echo "refusing prewarm because the exact Python 3.12 virtualenv site-packages directory is unavailable" >&2
    exit 2
  fi

  source_pycache_prefix="$run_dir/python-cache-empty"
  /usr/bin/mkdir -m 700 "$source_pycache_prefix"
  if ! source_cache_record="$(
    "${source_git[@]}" cat-file blob "$bootstrap_source_blob_oid" |
      /usr/bin/env -i \
        HOME="$source_account_home" \
        LC_ALL=C \
        PATH=/usr/bin:/bin \
        /usr/bin/python3.12 -I -S -B -P \
          -X "pycache_prefix=$source_pycache_prefix" \
          - \
          --prepare-source-cache \
          --repo-root "$repo" \
          --git-head "$source_git_head" \
          --account-home "$source_account_home"
  )"; then
    echo "refusing prewarm because the private commit-keyed source cache could not be prepared" >&2
    exit 2
  fi
  mapfile -t source_cache_fields <<<"$source_cache_record"
  if ((${#source_cache_fields[@]} != 4)); then
    echo "refusing prewarm because source-cache preparation returned malformed output" >&2
    exit 2
  fi
  source_archive="${source_cache_fields[0]}"
  source_archive_sha256="${source_cache_fields[1]}"
  source_snapshot="${source_cache_fields[2]}"
  source_cache_status="${source_cache_fields[3]}"
  source_cache_root="$source_account_home/.cache/skyrl-source-snapshots-private-v1"
  source_commit_root="$source_cache_root/$source_git_head"
  case "$source_cache_status" in
    created|reused) ;;
    *)
      echo "refusing prewarm because source-cache preparation returned an invalid status" >&2
      exit 2
      ;;
  esac
  if [[ "$source_archive" != "$source_commit_root/source-head.tar" \
    || "$source_snapshot" != "$source_commit_root/source-head" \
    || ! "$source_archive_sha256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "refusing prewarm because source-cache preparation escaped the exact commit key" >&2
    exit 2
  fi
  if [[ "$(/usr/bin/stat -c '%u:%a:%F' -- "$source_cache_root")" \
      != "$UID:700:directory" \
    || "$(/usr/bin/stat -c '%u:%a:%F' -- "$source_commit_root")" \
      != "$UID:700:directory" \
    || "$(/usr/bin/stat -c '%u:%a:%F' -- "$source_snapshot")" \
      != "$UID:700:directory" ]]; then
    echo "refusing prewarm because the commit-keyed source-cache directories are not private" >&2
    exit 2
  fi
  if [[ "$(/usr/bin/stat -c '%u:%h:%a:%F' -- "$source_archive")" \
      != "$UID:1:600:regular file" ]]; then
    echo "refusing prewarm because the source archive is not a private regular file" >&2
    exit 2
  fi
  if ! observed_source_archive_sha256="$(source_sha256 "$source_archive")" \
    || [[ "$observed_source_archive_sha256" != "$source_archive_sha256" ]]; then
    echo "refusing prewarm because the source archive could not be hashed" >&2
    exit 2
  fi
  if ! snapshot_prewarm_sha256="$(source_sha256 "$source_snapshot/rocm/prewarm_qwen35_buckets.py")" \
    || ! snapshot_bootstrap_sha256="$(source_sha256 "$source_snapshot/rocm/verified_source_bootstrap.py")" \
    || [[ "$snapshot_prewarm_sha256" != "$prewarm_source_sha256" ]] \
    || [[ "$snapshot_bootstrap_sha256" != "$bootstrap_source_sha256" ]]; then
    echo "refusing prewarm because extracted runtime source does not match HEAD" >&2
    exit 2
  fi
  export SKYRL_QWEN35_SOURCE_ARCHIVE_SHA256="$source_archive_sha256"
  export SKYRL_QWEN35_SOURCE_ARCHIVE_PATH="$source_archive"
  export SKYRL_QWEN35_SOURCE_REPO_ROOT="$repo"
  export SKYRL_QWEN35_SOURCE_SNAPSHOT_ROOT="$source_snapshot"
  export SKYRL_QWEN35_SOURCE_PYCACHE_PREFIX="$source_pycache_prefix"
  export SKYRL_QWEN35_SOURCE_VENV_SITE_PACKAGES="$venv_site_packages"
  export SKYRL_QWEN35_SOURCE_INTERPRETER=/usr/bin/python3.12
  export SKYRL_QWEN35_SOURCE_INTERPRETER_FLAGS=-I,-S,-B,-P

  verified_python=(
    /usr/bin/python3.12
    -I
    -S
    -B
    -P
    -X "pycache_prefix=$source_pycache_prefix"
    "$source_snapshot/rocm/verified_source_bootstrap.py"
    --repo-root "$repo"
    --git-head "$source_git_head"
    --snapshot-root "$source_snapshot"
    --venv-site-packages "$venv_site_packages"
  )
  verified_basic_environment=(
    HOME="$source_account_home"
    LANG=C.UTF-8
    LC_ALL=C.UTF-8
    PATH=/opt/rocm/bin:/usr/bin:/bin
    XDG_RUNTIME_DIR="$lock_parent"
    SKYRL_QWEN35_GIT_HEAD="$SKYRL_QWEN35_GIT_HEAD"
    SKYRL_QWEN35_GIT_TREE="$SKYRL_QWEN35_GIT_TREE"
    SKYRL_QWEN35_GIT_WORKTREE_CLEAN="$SKYRL_QWEN35_GIT_WORKTREE_CLEAN"
    SKYRL_QWEN35_LAUNCHER_BLOB_OID="$SKYRL_QWEN35_LAUNCHER_BLOB_OID"
    SKYRL_QWEN35_LAUNCHER_SHA256="$SKYRL_QWEN35_LAUNCHER_SHA256"
    SKYRL_QWEN35_PREWARM_BLOB_OID="$SKYRL_QWEN35_PREWARM_BLOB_OID"
    SKYRL_QWEN35_PREWARM_SHA256="$SKYRL_QWEN35_PREWARM_SHA256"
    SKYRL_QWEN35_BOOTSTRAP_BLOB_OID="$SKYRL_QWEN35_BOOTSTRAP_BLOB_OID"
    SKYRL_QWEN35_BOOTSTRAP_SHA256="$SKYRL_QWEN35_BOOTSTRAP_SHA256"
    SKYRL_QWEN35_SOURCE_ARCHIVE_SHA256="$SKYRL_QWEN35_SOURCE_ARCHIVE_SHA256"
    SKYRL_QWEN35_SOURCE_ARCHIVE_PATH="$SKYRL_QWEN35_SOURCE_ARCHIVE_PATH"
    SKYRL_QWEN35_SOURCE_REPO_ROOT="$SKYRL_QWEN35_SOURCE_REPO_ROOT"
    SKYRL_QWEN35_SOURCE_SNAPSHOT_ROOT="$SKYRL_QWEN35_SOURCE_SNAPSHOT_ROOT"
    SKYRL_QWEN35_SOURCE_PYCACHE_PREFIX="$SKYRL_QWEN35_SOURCE_PYCACHE_PREFIX"
    SKYRL_QWEN35_SOURCE_VENV_SITE_PACKAGES="$SKYRL_QWEN35_SOURCE_VENV_SITE_PACKAGES"
    SKYRL_QWEN35_SOURCE_INTERPRETER="$SKYRL_QWEN35_SOURCE_INTERPRETER"
    SKYRL_QWEN35_SOURCE_INTERPRETER_FLAGS="$SKYRL_QWEN35_SOURCE_INTERPRETER_FLAGS"
  )
fi

run_verified_source_module() {
  local target_module="$1"
  shift
  /usr/bin/env -i \
    "${verified_basic_environment[@]}" \
    "${verified_runtime_environment[@]}" \
    "${verified_python[@]}" \
    --module "$target_module" \
    -- \
    "$@"
}

run_amdgpu_safety() {
  if [[ -n "$prewarm_buckets" ]]; then
    run_verified_source_module rocm.amdgpu_safety
  else
    python3 -m rocm.amdgpu_safety
  fi
}

if ! run_amdgpu_safety >/dev/null; then
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
amd_card_names=()
amd_device_ids=()
connected_amd_connectors=()
for card_path in /sys/class/drm/card[0-9]*; do
  [[ "${card_path##*/}" =~ ^card[0-9]+$ ]] || continue
  [[ -r "$card_path/device/vendor" ]] || continue
  [[ "$(<"$card_path/device/vendor")" == "0x1002" ]] || continue
  ((amd_cards += 1))
  amd_card_names+=("${card_path##*/}")
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
  if [[ ! -x /usr/bin/fuser ]]; then
    echo "refusing to launch because fuser is unavailable for the /dev/kfd ownership check" >&2
    exit 2
  fi
  if kfd_owners="$(/usr/bin/fuser /dev/kfd 2>&1)"; then
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

if ! /usr/bin/env -i LC_ALL=C.UTF-8 PATH=/usr/bin:/bin \
    /usr/bin/python3.12 -I -S -B -P -X pycache_prefix=/dev/null - "$port" <<'PY'
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

if [[ -n "$prewarm_buckets" ]]; then
  uv_executable="$source_account_home/.local/bin/uv"
  if [[ -L "$uv_executable" || ! -f "$uv_executable" \
    || ! -x "$uv_executable" || ! -O "$uv_executable" ]]; then
    echo "refusing to launch because the fixed user uv executable is unavailable" >&2
    exit 2
  fi
  expected_uv_version="uv 0.11.8 (x86_64-unknown-linux-gnu)"
  expected_uv_sha256="646adf5cf12ba17d1a41fa77c8dd6496f73651dcfeeed6b5f4ec019b36bc7153"
  if ! uv_sha256="$(/usr/bin/sha256sum -- "$uv_executable")" \
    || [[ "$uv_sha256" != "$expected_uv_sha256  $uv_executable" ]]; then
    echo "refusing unqualified uv executable; expected exact uv 0.11.8 payload" >&2
    exit 2
  fi
  if ! uv_version="$(/usr/bin/env -i LC_ALL=C PATH=/usr/bin:/bin "$uv_executable" --version)" \
    || [[ "$uv_version" != "$expected_uv_version" ]]; then
    echo "refusing unqualified uv version output after payload verification" >&2
    exit 2
  fi
else
  uv_executable="$(command -v uv || true)"
fi
if [[ ! -x .venv/bin/python || ! -d .venv/lib/python3.12/site-packages \
  || -z "$uv_executable" ]]; then
  echo "refusing to launch because the project virtualenv or uv is unavailable" >&2
  exit 2
fi

venv_site_packages="$repo/.venv/lib/python3.12/site-packages"
export VIRTUAL_ENV="$repo/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
dependency_pycache_prefix="${source_pycache_prefix:-/dev/null}"
dependency_home="${source_account_home:-${HOME:-/nonexistent}}"
if ! model_path="$({
  /usr/bin/env -i \
    HOME="$dependency_home" \
    LC_ALL=C.UTF-8 \
    PATH=/opt/rocm/bin:/usr/bin:/bin \
    /usr/bin/python3.12 -I -S -B -P \
      -X "pycache_prefix=$dependency_pycache_prefix" \
      - \
      "$venv_site_packages" \
      "$model_repo" \
      "$model_revision" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv.pop(1))
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
export ROCR_VISIBLE_DEVICES=0
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
if ! installed_jax_stack="$(
  /usr/bin/env -i \
    HOME="$dependency_home" \
    LC_ALL=C.UTF-8 \
    PATH=/opt/rocm/bin:/usr/bin:/bin \
    /usr/bin/python3.12 -I -S -B -P \
      -X "pycache_prefix=$dependency_pycache_prefix" \
      -c 'import sys; sys.path.insert(0, sys.argv[1]); from importlib.metadata import version; print(",".join(version(package) for package in ("jax", "jaxlib", "jax-rocm7-plugin", "jax-rocm7-pjrt")))' \
      "$venv_site_packages"
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
if [[ -n "$prewarm_buckets" ]]; then
  if ! jax_cache_dir="$(
    run_verified_source_module rocm.prepare_jax_cache_dir \
      --max-autotune-bytes 4294967296
  )"; then
    echo "refusing to start without a private, validated JAX compilation cache" >&2
    exit 2
  fi
elif ! jax_cache_dir="$(
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

if [[ -n "$prewarm_buckets" ]]; then
  verified_runtime_environment=(
    HF_XET_HIGH_PERFORMANCE="$HF_XET_HIGH_PERFORMANCE"
    JAX_COMPILATION_CACHE_DIR="$JAX_COMPILATION_CACHE_DIR"
    JAX_COMPILATION_CACHE_EXPECT_PGLE="$JAX_COMPILATION_CACHE_EXPECT_PGLE"
    JAX_COMPILATION_CACHE_MAX_SIZE="$JAX_COMPILATION_CACHE_MAX_SIZE"
    JAX_ENABLE_COMPILATION_CACHE="$JAX_ENABLE_COMPILATION_CACHE"
    JAX_ENABLE_PGLE="$JAX_ENABLE_PGLE"
    JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES="$JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"
    JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="$JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"
    JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="$JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"
    JAX_PLATFORMS="$JAX_PLATFORMS"
    JAX_RAISE_PERSISTENT_CACHE_ERRORS="$JAX_RAISE_PERSISTENT_CACHE_ERRORS"
    LLVM_PATH="$LLVM_PATH"
    ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
    SKYRL_ROCM_PALLAS_ATTENTION="$SKYRL_ROCM_PALLAS_ATTENTION"
    XLA_FLAGS="$XLA_FLAGS"
    XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PYTHON_CLIENT_PREALLOCATE"
  )
  if [[ "$memory_mode" == "preallocate85" ]]; then
    verified_runtime_environment+=(
      GPU_DEVICE_ORDINAL="$GPU_DEVICE_ORDINAL"
      HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES"
      XLA_CLIENT_MEM_FRACTION="$XLA_CLIENT_MEM_FRACTION"
      XLA_PYTHON_CLIENT_ALLOCATOR="$XLA_PYTHON_CLIENT_ALLOCATOR"
    )
  fi
fi

engine_start_watchdog_pid=
engine_start_watchdog_start_ticks=
engine_start_watchdog_manifest_sha256=
engine_start_ready_payload=
engine_start_release_payload=
if ((engine_start_gate_enabled != 0)); then
  engine_start_profile_python="$repo/.venv/bin/python"
  engine_start_profile_source="$repo/rocm/profile_rocm.py"
  engine_start_profile_sha256="$profile_source_sha256"
  engine_start_telemetry="$engine_start_gate_dir/engine-start.watchdog.telemetry.jsonl"
fi

validate_engine_start_gate_directory() {
  local physical_path
  local metadata
  if [[ "$engine_start_gate_dir" != /* || "$engine_start_gate_dir" == */ ]] \
    || [[ -L "$engine_start_gate_dir" || ! -d "$engine_start_gate_dir" ]]; then
    return 1
  fi
  physical_path="$(cd -P -- "$engine_start_gate_dir" && pwd -P)" || return 1
  [[ "$physical_path" == "$engine_start_gate_dir" ]] || return 1
  metadata="$(/usr/bin/stat -c '%u:%a:%F' -- "$engine_start_gate_dir")" \
    || return 1
  [[ "$metadata" == "$EUID:700:directory" ]]
}

read_engine_start_gate_marker() {
  local marker_path="$1"
  local expected_size
  local marker_size
  local marker_payload
  local metadata
  if [[ -L "$marker_path" || ! -f "$marker_path" ]]; then
    return 1
  fi
  metadata="$(/usr/bin/stat -c '%u:%a:%F:%h:%s' -- "$marker_path")" || return 1
  [[ "$metadata" =~ ^$EUID:600:regular\ file:1:([1-9][0-9]{0,3})$ ]] \
    || return 1
  marker_size="${BASH_REMATCH[1]}"
  ((10#$marker_size <= 2048)) || return 1
  IFS= read -r marker_payload <"$marker_path" || return 1
  expected_size=$((${#marker_payload} + 1))
  [[ "$metadata" == "$EUID:600:regular file:1:$expected_size" ]] || return 1
  printf '%s' "$marker_payload"
}

validate_engine_start_gate_marker() {
  local marker_path="$1"
  local expected_payload="$2"
  local marker_payload
  marker_payload="$(read_engine_start_gate_marker "$marker_path")" || return 1
  [[ "$marker_payload" == "$expected_payload" ]]
}

read_watchdog_start_ticks() {
  local watchdog_pid="$1"
  local stat_line
  local stat_rest
  local -a stat_fields
  [[ -r "/proc/$watchdog_pid/stat" ]] || return 1
  stat_line="$(<"/proc/$watchdog_pid/stat")" || return 1
  [[ "${stat_line%% *}" == "$watchdog_pid" && "$stat_line" == *") "* ]] \
    || return 1
  stat_rest="${stat_line##*) }"
  read -r -a stat_fields <<<"$stat_rest" || return 1
  ((${#stat_fields[@]} >= 20)) || return 1
  [[ "${stat_fields[0]}" == R || "${stat_fields[0]}" == S \
    || "${stat_fields[0]}" == D ]] || return 1
  [[ "${stat_fields[19]}" =~ ^[1-9][0-9]*$ ]] || return 1
  printf '%s' "${stat_fields[19]}"
}

validate_engine_start_watchdog() {
  local watchdog_pid="$1"
  local expected_start_ticks="$2"
  local first_start_ticks
  local second_start_ticks
  local status_key
  local status_value
  local real_uid=
  local effective_uid=
  local saved_uid=
  local filesystem_uid=
  local launcher_cgroup
  local watchdog_cgroup

  [[ "$watchdog_pid" != "$$" && "$watchdog_pid" =~ ^[1-9][0-9]{0,9}$ ]] \
    || return 1
  first_start_ticks="$(read_watchdog_start_ticks "$watchdog_pid")" || return 1
  [[ "$first_start_ticks" == "$expected_start_ticks" ]] || return 1
  [[ -r "/proc/$watchdog_pid/status" ]] || return 1
  while IFS=: read -r status_key status_value; do
    if [[ "$status_key" == Uid ]]; then
      read -r real_uid effective_uid saved_uid filesystem_uid <<<"$status_value" \
        || return 1
      break
    fi
  done <"/proc/$watchdog_pid/status"
  [[ "$real_uid" == "$EUID" && "$effective_uid" == "$EUID" \
    && "$saved_uid" == "$EUID" && "$filesystem_uid" == "$EUID" ]] \
    || return 1
  [[ -r "/proc/$$/cgroup" && -r "/proc/$watchdog_pid/cgroup" ]] || return 1
  launcher_cgroup="$(<"/proc/$$/cgroup")" || return 1
  watchdog_cgroup="$(<"/proc/$watchdog_pid/cgroup")" || return 1
  [[ -n "$launcher_cgroup" && "$watchdog_cgroup" == "$launcher_cgroup" ]] \
    || return 1
  second_start_ticks="$(read_watchdog_start_ticks "$watchdog_pid")" || return 1
  [[ "$second_start_ticks" == "$expected_start_ticks" ]]
}

validate_engine_start_profiler_evidence() {
  local watchdog_pid="$1"
  local watchdog_start_ticks="$2"
  local watchdog_manifest_sha256="$3"
  /usr/bin/python3.12 -I -S -B -P - \
    "$engine_start_telemetry" \
    "$EUID" \
    "$watchdog_pid" \
    "$watchdog_start_ticks" \
    "$engine_start_profile_source" \
    "$engine_start_profile_sha256" \
    "$engine_start_profile_python" \
    "${amd_card_names[0]}" \
    "$$" \
    "$XLA_FLAGS" \
    "$watchdog_manifest_sha256" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path


def refuse(message):
    raise SystemExit(message)


def strict_json(payload):
    def reject_constant(value):
        refuse(f"non-finite JSON constant: {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                refuse(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            payload,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        refuse(f"invalid telemetry JSON: {error}")


(
    telemetry_arg,
    expected_uid_arg,
    watchdog_pid_arg,
    watchdog_start_ticks,
    profile_source_arg,
    profile_sha256,
    profile_python_arg,
    card,
    server_pid_arg,
    xla_flags,
    manifest_sha256,
) = sys.argv[1:]
telemetry = Path(telemetry_arg)
expected_uid = int(expected_uid_arg)
watchdog_pid = int(watchdog_pid_arg)
server_pid = int(server_pid_arg)
profile_source = Path(profile_source_arg)
profile_python = Path(profile_python_arg)

if telemetry != telemetry.parent / "engine-start.watchdog.telemetry.jsonl":
    refuse("unexpected watchdog telemetry path")
try:
    telemetry_stat = telemetry.lstat()
except OSError as error:
    refuse(f"watchdog telemetry is unavailable: {error}")
if (
    not stat.S_ISREG(telemetry_stat.st_mode)
    or stat.S_IMODE(telemetry_stat.st_mode) != 0o600
    or telemetry_stat.st_uid != expected_uid
    or telemetry_stat.st_nlink != 1
):
    refuse("watchdog telemetry is not a private singly-linked regular file")

flags = os.O_RDONLY | os.O_CLOEXEC
if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
try:
    descriptor = os.open(telemetry, flags)
except OSError as error:
    refuse(f"watchdog telemetry could not be opened safely: {error}")
with os.fdopen(descriptor, "rb") as source:
    opened_stat = os.fstat(source.fileno())
    if (opened_stat.st_dev, opened_stat.st_ino) != (
        telemetry_stat.st_dev,
        telemetry_stat.st_ino,
    ):
        refuse("watchdog telemetry identity changed while opening")
    first_line = source.readline(65537)
    if not first_line.endswith(b"\n") or len(first_line) > 65536:
        refuse("watchdog manifest line is missing or oversized")
    if hashlib.sha256(first_line).hexdigest() != manifest_sha256:
        refuse("watchdog manifest digest mismatch")
    manifest = strict_json(first_line)
    measured_sample = None
    for _index in range(256):
        line = source.readline(262145)
        if not line:
            break
        if not line.endswith(b"\n") or len(line) > 262144:
            refuse("watchdog sample line is incomplete or oversized")
        record = strict_json(line)
        if record.get("record_type") == "sample" and record.get("phase") == "measured":
            measured_sample = record
            break
if measured_sample is None:
    refuse("watchdog has not durably recorded a measured sample")

try:
    current_profile_sha256 = hashlib.sha256(profile_source.read_bytes()).hexdigest()
except OSError as error:
    refuse(f"profile source is unavailable: {error}")
if current_profile_sha256 != profile_sha256:
    refuse("profile source does not match its attested Git blob")

expected_argv = [
    str(profile_python),
    "-B",
    str(profile_source),
    "--output",
    str(telemetry),
    "--card",
    card,
    "--interval",
    "0.1",
    "--include-pid",
    f"server={server_pid}",
    "--terminate-included-on-safety",
    "--terminate-included-on-abort",
    "--sensor-grace-seconds",
    "60",
    "--max-junction-temp-c",
    "90",
    "--max-gpu-power-watts",
    "400",
    "--max-vram-gib",
    "24",
    "--min-host-available-gib",
    "0",
    "--max-swap-gib",
    "8",
    "--record-command",
]
try:
    raw_cmdline = Path(f"/proc/{watchdog_pid}/cmdline").read_bytes()
    observed_argv = [
        value.decode("utf-8", "strict")
        for value in raw_cmdline.removesuffix(b"\0").split(b"\0")
    ]
    observed_executable = Path(f"/proc/{watchdog_pid}/exe").resolve(strict=True)
    expected_executable = profile_python.resolve(strict=True)
    watchdog_cgroup = Path(f"/proc/{watchdog_pid}/cgroup").read_bytes()
    server_cgroup = Path(f"/proc/{server_pid}/cgroup").read_bytes()
    raw_environment = Path(f"/proc/{watchdog_pid}/environ").read_bytes()
    raw_server_environment = Path(f"/proc/{server_pid}/environ").read_bytes()
except (OSError, UnicodeError) as error:
    refuse(f"watchdog process evidence is unavailable: {error}")
if observed_argv != expected_argv or observed_executable != expected_executable:
    refuse("watchdog command identity is not the exact profile_rocm policy")
if not watchdog_cgroup or watchdog_cgroup != server_cgroup:
    refuse("watchdog is outside the server service cgroup")
watchdog_xla = [
    item.split(b"=", 1)[1].decode("utf-8", "strict")
    for item in raw_environment.split(b"\0")
    if item.startswith(b"XLA_FLAGS=")
]
if watchdog_xla != [xla_flags]:
    refuse("watchdog process does not have the exact XLA policy")
server_xla = [
    item.split(b"=", 1)[1].decode("utf-8", "strict")
    for item in raw_server_environment.split(b"\0")
    if item.startswith(b"XLA_FLAGS=")
]
if server_xla != [xla_flags]:
    refuse("server process does not have the exact XLA policy")

expected_limits = {
    "max_junction_temp_c": 90.0,
    "max_gpu_power_watts": 400.0,
    "max_vram_bytes": 24 * 1024**3,
    "min_host_available_bytes": 0,
    "max_swap_bytes": 8 * 1024**3,
}
if manifest.get("record_type") != "manifest":
    refuse("watchdog first line is not a manifest")
if (
    manifest.get("interval_seconds") != 0.1
    or manifest.get("baseline_seconds") != 0.0
    or manifest.get("duration_seconds") is not None
    or manifest.get("timeout_seconds") is not None
    or manifest.get("sensor_grace_seconds") != 60.0
    or manifest.get("safety_limits") != expected_limits
    or manifest.get("terminate_included_on_safety") is not True
    or manifest.get("terminate_included_on_abort") is not True
    or manifest.get("command") != []
    or manifest.get("command_recorded") is not True
    or manifest.get("passed_file_descriptor_count") != 0
):
    refuse("watchdog manifest does not match the exact attach-only safety policy")
gpu = manifest.get("gpu")
if not isinstance(gpu, dict) or gpu.get("card") != card or gpu.get("device_id") != "0x744c":
    refuse("watchdog manifest is bound to a different GPU")
runtime = manifest.get("runtime")
if (
    not isinstance(runtime, dict)
    or runtime.get("script_sha256") != profile_sha256
    or not isinstance(runtime.get("accelerator_environment"), dict)
    or runtime["accelerator_environment"].get("XLA_FLAGS") != xla_flags
):
    refuse("watchdog runtime manifest is not exact")
explicit_processes = manifest.get("explicit_processes")
if not isinstance(explicit_processes, dict) or set(explicit_processes) != {"server"}:
    refuse("watchdog manifest does not have exactly one server target")
server = explicit_processes["server"]
if (
    not isinstance(server, dict)
    or server.get("pid") != server_pid
    or server.get("unavailable") is not None
    or not isinstance(server.get("command"), list)
    or server.get("command") == ["<arguments omitted>"]
    or not isinstance(server.get("accelerator_environment"), dict)
    or server["accelerator_environment"].get("XLA_FLAGS") != xla_flags
):
    refuse("watchdog manifest is not attached to the exact server process")
sample_processes = measured_sample.get("processes")
sample_server = sample_processes.get("server") if isinstance(sample_processes, dict) else None
if (
    not isinstance(sample_server, dict)
    or sample_server.get("root_pid") != server_pid
    or not isinstance(sample_server.get("process_count"), int)
    or sample_server["process_count"] < 1
):
    refuse("watchdog measured sample does not cover the server process")

# The shell independently validates the live start ticks before and after this
# check. Keep the release binding in this validator too.
stat_fields = Path(f"/proc/{watchdog_pid}/stat").read_text().rsplit(") ", 1)[1].split()
if len(stat_fields) < 20 or stat_fields[19] != watchdog_start_ticks:
    refuse("watchdog PID start ticks changed during evidence validation")
PY
}

revalidate_engine_start_watchdog() {
  validate_engine_start_gate_directory \
    && validate_engine_start_gate_marker \
      "$engine_start_gate_dir/engine-start.ready" "$engine_start_ready_payload" \
    && validate_engine_start_gate_marker \
      "$engine_start_gate_dir/engine-start.release" "$engine_start_release_payload" \
    && validate_engine_start_watchdog \
      "$engine_start_watchdog_pid" "$engine_start_watchdog_start_ticks" \
    && validate_engine_start_profiler_evidence \
      "$engine_start_watchdog_pid" \
      "$engine_start_watchdog_start_ticks" \
      "$engine_start_watchdog_manifest_sha256"
}

require_engine_start_kfd_unowned() {
  local kfd_owners
  local fuser_status
  if kfd_owners="$(/usr/bin/fuser /dev/kfd 2>&1)"; then
    echo "refusing API start because /dev/kfd is owned after prewarm handoff: $kfd_owners" >&2
    return 1
  fi
  fuser_status=$?
  if ((fuser_status != 1)) || [[ -n "${kfd_owners//[[:space:]]/}" ]]; then
    echo "refusing API start because exact /dev/kfd ownership could not be verified" >&2
    return 1
  fi
}

await_engine_start_release() {
  local boot_id
  local gate_token
  local payload
  local ready_marker="$engine_start_gate_dir/engine-start.ready"
  local release_marker="$engine_start_gate_dir/engine-start.release"
  local telemetry_summary="$engine_start_telemetry.summary.json"
  local marker_path
  local ready_tmp
  local release_payload
  local watchdog_manifest_sha256
  local watchdog_pid
  local watchdog_start_ticks
  local wait_started

  if ! validate_engine_start_gate_directory; then
    echo "refusing engine-start gate because its directory is no longer private and canonical" >&2
    return 2
  fi
  for marker_path in \
    "$ready_marker" "$release_marker" "$engine_start_telemetry" "$telemetry_summary"; do
    if [[ -e "$marker_path" || -L "$marker_path" ]]; then
      echo "refusing a stale engine-start gate marker: $marker_path" >&2
      return 2
    fi
  done
  IFS= read -r gate_token </proc/sys/kernel/random/uuid || {
    echo "refusing engine-start gate because a kernel token is unavailable" >&2
    return 2
  }
  if [[ ! "$gate_token" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$ ]]; then
    echo "refusing engine-start gate because its kernel token is malformed" >&2
    return 2
  fi
  IFS= read -r boot_id </proc/sys/kernel/random/boot_id || {
    echo "refusing engine-start gate because the current boot ID is unavailable" >&2
    return 2
  }
  if [[ ! "$boot_id" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] \
    || [[ ! "$prewarm_audit_sha256" =~ ^[0-9a-f]{64}$ ]] \
    || [[ ! "$prewarm_handoff_sha256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "refusing engine-start gate because its runtime identity is malformed" >&2
    return 2
  fi
  payload="skyrl-qwen35-engine-start-ready-v1 nonce=$gate_token launcher_pid=$$ boot_id=$boot_id run_id=$run_id fused=$bf16_rms_gate_up_lora_swiglu_contiguous prewarm_sha256=$prewarm_audit_sha256 handoff_sha256=$prewarm_handoff_sha256"
  ready_tmp="$engine_start_gate_dir/.engine-start.ready.$gate_token.tmp"
  if [[ -e "$ready_tmp" || -L "$ready_tmp" ]] \
    || ! (set -o noclobber; printf '%s\n' "$payload" >"$ready_tmp") \
    || ! /usr/bin/chmod 600 -- "$ready_tmp" \
    || ! validate_engine_start_gate_marker "$ready_tmp" "$payload"; then
    /usr/bin/rm -f -- "$ready_tmp"
    echo "refusing engine-start gate because its ready marker could not be staged" >&2
    return 2
  fi
  if ! /usr/bin/mv --no-clobber --no-target-directory \
      "$ready_tmp" "$ready_marker" \
    || [[ -e "$ready_tmp" || -L "$ready_tmp" ]] \
    || ! validate_engine_start_gate_marker "$ready_marker" "$payload"; then
    /usr/bin/rm -f -- "$ready_tmp"
    echo "refusing engine-start gate because its ready marker could not be published atomically" >&2
    return 2
  fi

  wait_started=$SECONDS
  while true; do
    if [[ -e "$release_marker" || -L "$release_marker" ]]; then
      if ! validate_engine_start_gate_directory \
        || ! validate_engine_start_gate_marker "$ready_marker" "$payload" \
        || ! release_payload="$(read_engine_start_gate_marker "$release_marker")" \
        || [[ ! "$release_payload" =~ ^skyrl-qwen35-engine-start-release-v1\ nonce=([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})\ launcher_pid=([1-9][0-9]{0,9})\ boot_id=([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\ run_id=([A-Za-z0-9][A-Za-z0-9._-]*)\ fused=([01])\ prewarm_sha256=([0-9a-f]{64})\ handoff_sha256=([0-9a-f]{64})\ watchdog_pid=([1-9][0-9]{0,9})\ watchdog_start_ticks=([1-9][0-9]*)\ watchdog_manifest_sha256=([0-9a-f]{64})$ ]]; then
        echo "refusing an invalid engine-start release marker" >&2
        return 2
      fi
      if [[ "${BASH_REMATCH[1]}" != "$gate_token" \
        || "${BASH_REMATCH[2]}" != "$$" \
        || "${BASH_REMATCH[3]}" != "$boot_id" \
        || "${BASH_REMATCH[4]}" != "$run_id" \
        || "${BASH_REMATCH[5]}" != "$bf16_rms_gate_up_lora_swiglu_contiguous" \
        || "${BASH_REMATCH[6]}" != "$prewarm_audit_sha256" \
        || "${BASH_REMATCH[7]}" != "$prewarm_handoff_sha256" ]]; then
        echo "refusing an engine-start release marker bound to different runtime evidence" >&2
        return 2
      fi
      watchdog_pid="${BASH_REMATCH[8]}"
      watchdog_start_ticks="${BASH_REMATCH[9]}"
      watchdog_manifest_sha256="${BASH_REMATCH[10]}"
      if [[ ! "$watchdog_manifest_sha256" =~ ^[0-9a-f]{64}$ ]] \
        || ! validate_engine_start_watchdog \
          "$watchdog_pid" "$watchdog_start_ticks" \
        || ! validate_engine_start_profiler_evidence \
          "$watchdog_pid" \
          "$watchdog_start_ticks" \
          "$watchdog_manifest_sha256"; then
        echo "refusing an engine-start release marker without a live owned watchdog" >&2
        return 2
      fi
      engine_start_watchdog_pid="$watchdog_pid"
      engine_start_watchdog_start_ticks="$watchdog_start_ticks"
      engine_start_watchdog_manifest_sha256="$watchdog_manifest_sha256"
      engine_start_ready_payload="$payload"
      engine_start_release_payload="$release_payload"
      return 0
    fi
    if ((SECONDS - wait_started >= 10#$engine_start_gate_timeout_seconds)); then
      echo "timed out waiting for the engine-start release marker" >&2
      return 2
    fi
    /usr/bin/sleep 0.1
  done
}

# Default-off, compile-only static-bucket prewarm. This populates the trusted
# persistent cache before the API starts, but never invokes a compiled model
# pass or optimizer step. ROCm compilation may still run bounded autotuning.
prewarm_status=0
prewarm_handoff_status=0
final_journal_status=0
prewarm_termination_status=0
prewarm_started=0
prewarm_supervisor_pid=
prewarm_supervisor_waited=1
prewarm_cleanup_state=0

record_prewarm_cleanup_signal() {
  if ((prewarm_termination_status == 0)); then
    prewarm_termination_status="$1"
  fi
}

finish_prewarm_once() {
  if ((prewarm_cleanup_state != 0)); then
    return 0
  fi
  prewarm_cleanup_state=1

  # A signal during cleanup must be remembered without recursing or skipping
  # the bounded handoff and final-journal gates. The caller exits afterward.
  trap 'record_prewarm_cleanup_signal 130' INT
  trap 'record_prewarm_cleanup_signal 143' TERM

  if ((prewarm_started != 0 && prewarm_supervisor_waited == 0)); then
    if [[ -z "$prewarm_supervisor_pid" && -n "${!:-}" ]]; then
      prewarm_supervisor_pid="$!"
    fi
    if [[ -n "$prewarm_supervisor_pid" ]]; then
      kill -TERM "$prewarm_supervisor_pid" 2>/dev/null || true
      if wait "$prewarm_supervisor_pid"; then
        prewarm_status=0
      else
        prewarm_status=$?
      fi
    else
      prewarm_status=143
    fi
    prewarm_supervisor_waited=1
  fi

  if ((prewarm_started != 0)); then
    if run_verified_source_module rocm.qwen35_prewarm_handoff settle \
        --output "$prewarm_handoff_artifact" \
        --timeout-seconds 120 \
        --poll-interval-seconds 1; then
      prewarm_handoff_status=0
    else
      prewarm_handoff_status=$?
    fi
  fi

  if run_amdgpu_safety >/dev/null; then
    final_journal_status=0
  else
    final_journal_status=$?
  fi

  prewarm_cleanup_state=2
  trap 'on_prewarm_signal 130' INT
  trap 'on_prewarm_signal 143' TERM
}

select_prewarm_exit_status() {
  local incoming_status="$1"
  if ((final_journal_status != 0 || prewarm_handoff_status != 0)); then
    selected_prewarm_exit_status=2
  elif ((prewarm_termination_status != 0)); then
    selected_prewarm_exit_status="$prewarm_termination_status"
  elif ((prewarm_status != 0)); then
    selected_prewarm_exit_status="$prewarm_status"
  else
    selected_prewarm_exit_status="$incoming_status"
  fi
}

on_prewarm_signal() {
  if ((prewarm_termination_status == 0)); then
    prewarm_termination_status="$1"
  fi
  finish_prewarm_once
  select_prewarm_exit_status "$prewarm_termination_status"
  trap - EXIT INT TERM
  exit "$selected_prewarm_exit_status"
}

on_prewarm_exit() {
  local incoming_status="$1"
  finish_prewarm_once
  select_prewarm_exit_status "$incoming_status"
  trap - EXIT INT TERM
  exit "$selected_prewarm_exit_status"
}

if [[ -n "$prewarm_buckets" ]]; then
  prewarm_optimizer_args=()
  if [[ "$prewarm_optimizer" == "1" ]]; then
    prewarm_optimizer_args=(--compile-optimizer)
  fi
  prewarm_bf16_rms_gate_up_args=()
  if [[ "$bf16_rms_gate_up_lora_swiglu_contiguous" == "1" ]]; then
    prewarm_bf16_rms_gate_up_args=(
      --qwen35-bf16-rms-gate-up-lora-swiglu-contiguous
    )
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
  prewarm_handoff_artifact="$run_dir/prewarm-handoff.jsonl"
  if ! run_verified_source_module rocm.qwen35_prewarm_handoff capture \
      --output "$prewarm_handoff_artifact"; then
    echo "refusing startup prewarm because the idle AMDGPU baseline could not be captured" >&2
    exit 2
  fi
  trap 'on_prewarm_signal 130' INT
  trap 'on_prewarm_signal 143' TERM
  trap 'on_prewarm_exit "$?"' EXIT
  prewarm_started=1
  prewarm_supervisor_waited=0
  run_verified_source_module rocm.profile_rocm \
      --output "$run_dir/prewarm.telemetry.jsonl" \
      --card "${amd_card_names[0]}" \
      --interval 0.1 \
      --baseline-seconds 2 \
      --timeout "$prewarm_timeout_seconds" \
      --sensor-grace-seconds 60 \
      --max-junction-temp-c 90 \
      --max-gpu-power-watts 400 \
      --max-vram-gib 24 \
      --min-host-available-gib 0 \
      --max-swap-gib 8 \
      --pass-fd "$launch_lock_fd" \
      -- \
      "${verified_python[@]}" \
        --module rocm.prewarm_qwen35_buckets \
        -- \
        --execute-rocm \
        --allow-gpu \
        "${prewarm_optimizer_args[@]}" \
        "${prewarm_bf16_rms_gate_up_args[@]}" \
        --buckets "$prewarm_buckets" \
        --model-path "$model_path" \
        --construction "$prewarm_construction" \
        --attention-backend "$prewarm_attention" \
        --launcher-lock-fd "$launch_lock_fd" \
        --output "$run_dir/prewarm.jsonl" &
  prewarm_supervisor_pid=$!
  if wait "$prewarm_supervisor_pid"; then
    prewarm_status=0
  else
    prewarm_status=$?
  fi
  prewarm_supervisor_waited=1
  finish_prewarm_once
fi

if [[ -z "$prewarm_buckets" ]]; then
  if run_amdgpu_safety >/dev/null; then
    final_journal_status=0
  else
    final_journal_status=$?
  fi
fi
if ((final_journal_status != 0)); then
  echo "refusing API start because the final AMDGPU boot-journal postflight failed" >&2
  exit 2
fi
if ((prewarm_handoff_status != 0)); then
  echo "refusing API start because the prewarm AMDGPU handoff failed with status $prewarm_handoff_status" >&2
  exit 2
fi
if ((prewarm_termination_status != 0)); then
  echo "refusing API start because startup prewarm was interrupted" >&2
  exit "$prewarm_termination_status"
fi
if ((prewarm_status != 0)); then
  echo "refusing API start because startup prewarm failed with status $prewarm_status" >&2
  exit "$prewarm_status"
fi

if [[ "${engine_t64_cache_attest:-0}" == "1" ]]; then
  prewarm_audit_artifact="$run_dir/prewarm.jsonl"
  if [[ "$prewarm_handoff_artifact" != "$run_dir/prewarm-handoff.jsonl" ]] \
    || [[ "$(/usr/bin/stat -c '%u:%a:%F' -- "$run_dir")" \
      != "$EUID:700:directory" ]]; then
    echo "refusing runtime cache attestation because its artifact parent is not the exact private run directory" >&2
    exit 2
  fi
  for artifact in "$prewarm_audit_artifact" "$prewarm_handoff_artifact"; do
    if [[ -L "$artifact" ]] \
      || [[ "$(/usr/bin/stat -c '%u:%a:%F:%h' -- "$artifact")" \
        != "$EUID:600:regular file:1" ]]; then
      echo "refusing runtime cache attestation because an artifact is not an owned, singly linked mode-0600 regular file" >&2
      exit 2
    fi
  done
  if ! prewarm_audit_sha256_line="$(/usr/bin/sha256sum -- "$prewarm_audit_artifact")" \
    || ! prewarm_handoff_sha256_line="$(/usr/bin/sha256sum -- "$prewarm_handoff_artifact")"; then
    echo "refusing runtime cache attestation because its artifacts could not be hashed" >&2
    exit 2
  fi
  prewarm_audit_sha256="${prewarm_audit_sha256_line%% *}"
  prewarm_handoff_sha256="${prewarm_handoff_sha256_line%% *}"
  if [[ ! "$prewarm_audit_sha256" =~ ^[0-9a-f]{64}$ ]] \
    || [[ ! "$prewarm_handoff_sha256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "refusing runtime cache attestation because an artifact digest is malformed" >&2
    exit 2
  fi
  runtime_cache_attestation_environment=(
    SKYRL_QWEN35_RUNTIME_T64_CACHE_ATTEST=required-v1
    SKYRL_QWEN35_RUNTIME_PREWARM_AUDIT_PATH="$prewarm_audit_artifact"
    SKYRL_QWEN35_RUNTIME_PREWARM_AUDIT_SHA256="$prewarm_audit_sha256"
    SKYRL_QWEN35_RUNTIME_PREWARM_HANDOFF_PATH="$prewarm_handoff_artifact"
    SKYRL_QWEN35_RUNTIME_PREWARM_HANDOFF_SHA256="$prewarm_handoff_sha256"
  )
fi
unset SKYRL_QWEN35_ENGINE_T64_CACHE_ATTEST

unset SKYRL_QWEN35_GIT_HEAD
unset SKYRL_QWEN35_GIT_TREE
unset SKYRL_QWEN35_GIT_WORKTREE_CLEAN
unset SKYRL_QWEN35_LAUNCHER_BLOB_OID
unset SKYRL_QWEN35_LAUNCHER_SHA256
unset SKYRL_QWEN35_PREWARM_BLOB_OID
unset SKYRL_QWEN35_PREWARM_SHA256
unset SKYRL_QWEN35_BOOTSTRAP_BLOB_OID
unset SKYRL_QWEN35_BOOTSTRAP_SHA256
unset SKYRL_QWEN35_SOURCE_ARCHIVE_PATH
unset SKYRL_QWEN35_SOURCE_ARCHIVE_SHA256
unset SKYRL_QWEN35_SOURCE_INTERPRETER
unset SKYRL_QWEN35_SOURCE_INTERPRETER_FLAGS
unset SKYRL_QWEN35_SOURCE_PYCACHE_PREFIX
unset SKYRL_QWEN35_SOURCE_REPO_ROOT
unset SKYRL_QWEN35_SOURCE_SNAPSHOT_ROOT
unset SKYRL_QWEN35_SOURCE_VENV_SITE_PACKAGES
unset SKYRL_VERIFIED_SOURCE_GIT_HEAD
unset SKYRL_VERIFIED_SOURCE_GIT_TREE
unset SKYRL_VERIFIED_SOURCE_MANIFEST_SHA256
unset SKYRL_VERIFIED_SOURCE_RUNTIME_POLICY
unset SKYRL_VERIFIED_SOURCE_SNAPSHOT_ROOT

if [[ "$prewarm_only" == "1" ]]; then
  trap - EXIT INT TERM
  exit 0
fi

if ((engine_start_gate_enabled != 0)); then
  await_engine_start_release
  if ! run_amdgpu_safety >/dev/null; then
    echo "refusing API start because the post-release AMDGPU boot-journal gate failed" >&2
    exit 2
  fi
  if ! require_engine_start_kfd_unowned; then
    exit 2
  fi
fi
unset SKYRL_QWEN35_ENGINE_START_GATE_DIR
unset SKYRL_QWEN35_ENGINE_START_GATE_TIMEOUT_SECONDS

if ((engine_start_gate_enabled != 0)); then
  if ! revalidate_engine_start_watchdog; then
    echo "refusing API start because the engine-start watchdog did not survive final safety checks" >&2
    exit 2
  fi
fi

trap - EXIT INT TERM
if [[ -n "$prewarm_buckets" ]]; then
  cd -- "$source_snapshot"
  exec /usr/bin/env -i \
    HOME="$source_account_home" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/rocm/bin:/usr/bin:/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV="$repo/.venv" \
    XDG_RUNTIME_DIR="$lock_parent" \
    SKYRL_QWEN35_RUNTIME_GIT_HEAD="$source_git_head" \
    SKYRL_QWEN35_RUNTIME_MEMORY_MODE="$memory_mode" \
    SKYRL_QWEN35_RUNTIME_REPO_ROOT="$repo" \
    SKYRL_QWEN35_RUNTIME_SOURCE_ROOT="$source_snapshot" \
    SKYRL_QWEN35_RUNTIME_UV_EXECUTABLE="$uv_executable" \
    SKYRL_QWEN35_LAUNCH_LOCK_FD="$SKYRL_QWEN35_LAUNCH_LOCK_FD" \
    "${runtime_cache_attestation_environment[@]}" \
    "${verified_runtime_environment[@]}" \
    "$uv_executable" run \
    --active \
    --no-sync \
    --no-env-file \
    --no-config \
    --directory "$source_snapshot" \
    --project "$source_snapshot" \
    -m skyrl.tinker.api \
    --base-model "$model_path" \
    --backend jax \
    --backend-config "$backend_config" \
    --host 127.0.0.1 \
    --port "$port" \
    --checkpoints-base "$run_dir/checkpoints" \
    --engine-startup-timeout-sec 3600 \
    --database-url "sqlite:///$run_dir/tinker.db"
fi
exec "$uv_executable" run --active --no-sync -m skyrl.tinker.api \
  --base-model "$model_path" \
  --backend jax \
  --backend-config "$backend_config" \
  --host 127.0.0.1 \
  --port "$port" \
  --checkpoints-base "$run_dir/checkpoints" \
  --engine-startup-timeout-sec 3600 \
  --database-url "sqlite:///$run_dir/tinker.db"
