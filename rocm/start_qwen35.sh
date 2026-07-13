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
backend_config='{"max_lora_adapters":2,"max_lora_rank":8,"train_micro_batch_size":1,"sample_max_num_sequences":1,"gradient_checkpointing":true,"loss_chunk_size":64}'

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
if [[ "$prewarm_optimizer" == "1" && -z "$prewarm_buckets" ]]; then
  echo "SKYRL_QWEN35_PREWARM_OPTIMIZER=1 requires nonempty SKYRL_QWEN35_PREWARM_BUCKETS." >&2
  exit 2
fi
if [[ "$prewarm_only" == "1" && -z "$prewarm_buckets" ]]; then
  echo "SKYRL_QWEN35_PREWARM_ONLY=1 requires nonempty SKYRL_QWEN35_PREWARM_BUCKETS." >&2
  exit 2
fi
if [[ ! "$prewarm_timeout_seconds" =~ ^[0-9]{3,5}$ ]] \
  || ((10#$prewarm_timeout_seconds < 600 || 10#$prewarm_timeout_seconds > 14400)); then
  echo "SKYRL_QWEN35_PREWARM_TIMEOUT_SECONDS must be an integer in [600, 14400]." >&2
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

if [[ -n "$prewarm_buckets" ]]; then
  while IFS= read -r exported_name; do
    case "$exported_name" in
      PYTHON*|__PYVENV_LAUNCHER__|VIRTUAL_ENV|VIRTUAL_ENV_PROMPT)
        echo "refusing inherited Python or virtualenv startup variable during operational prewarm: $exported_name" >&2
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

  source_index_flags="$("${source_git[@]}" ls-files -v -- rocm/prewarm_qwen35_buckets.py rocm/start_qwen35.sh rocm/verified_source_bootstrap.py 2>&1)" || {
    echo "refusing prewarm because runtime-source index flags are unavailable" >&2
    exit 2
  }
  expected_source_index_flags=$'H rocm/prewarm_qwen35_buckets.py\nH rocm/start_qwen35.sh\nH rocm/verified_source_bootstrap.py'
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
  bootstrap_source_path="$repo/rocm/verified_source_bootstrap.py"
  if ! launcher_source_blob_oid="$(source_blob_oid rocm/start_qwen35.sh 100755)" \
    || ! prewarm_source_blob_oid="$(source_blob_oid rocm/prewarm_qwen35_buckets.py 100644)" \
    || ! bootstrap_source_blob_oid="$(source_blob_oid rocm/verified_source_bootstrap.py 100644)" \
    || ! launcher_source_sha256="$(source_sha256 "$launcher_source_path")" \
    || ! prewarm_source_sha256="$(source_sha256 "$prewarm_source_path")" \
    || ! bootstrap_source_sha256="$(source_sha256 "$bootstrap_source_path")" \
    || ! launcher_blob_sha256="$(source_blob_sha256 "$launcher_source_blob_oid")" \
    || ! prewarm_blob_sha256="$(source_blob_sha256 "$prewarm_source_blob_oid")" \
    || ! bootstrap_blob_sha256="$(source_blob_sha256 "$bootstrap_source_blob_oid")" \
    || [[ "$launcher_source_sha256" != "$launcher_blob_sha256" ]] \
    || [[ "$prewarm_source_sha256" != "$prewarm_blob_sha256" ]] \
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
if ! /usr/bin/flock -n "$launch_lock_fd"; then
  echo "refusing to launch while another Qwen3.5 ROCm server holds the global launch lock" >&2
  exit 2
fi

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

  source_archive="$run_dir/source-head.tar"
  source_snapshot="$run_dir/source-head"
  source_pycache_prefix="$run_dir/python-cache-empty"
  /usr/bin/mkdir -m 700 "$source_snapshot" "$source_pycache_prefix"
  if ! "${source_git[@]}" archive \
      --format=tar \
      --output="$source_archive" \
      "$source_git_head"; then
    echo "refusing prewarm because the exact HEAD source archive could not be created" >&2
    exit 2
  fi
  if [[ "$(/usr/bin/stat -c '%u:%h:%a:%F' -- "$source_archive")" \
      != "$UID:1:600:regular file" ]]; then
    echo "refusing prewarm because the source archive is not a private regular file" >&2
    exit 2
  fi
  if ! source_archive_sha256="$(source_sha256 "$source_archive")"; then
    echo "refusing prewarm because the source archive could not be hashed" >&2
    exit 2
  fi
  if ! /usr/bin/env -i LC_ALL=C PATH=/usr/bin:/bin /usr/bin/tar \
      --extract \
      --no-same-owner \
      --no-same-permissions \
      --file="$source_archive" \
      --directory="$source_snapshot"; then
    echo "refusing prewarm because the exact HEAD source archive could not be extracted" >&2
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
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}"
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
      --max-gpu-power-watts 315 \
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

trap - EXIT INT TERM
exec "$uv_executable" run --active --no-sync -m skyrl.tinker.api \
  --base-model "$model_path" \
  --backend jax \
  --backend-config "$backend_config" \
  --host 127.0.0.1 \
  --port "$port" \
  --checkpoints-base "$run_dir/checkpoints" \
  --database-url "sqlite:///$run_dir/tinker.db"
