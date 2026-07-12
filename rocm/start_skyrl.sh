#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="$(cd "$repo/.." && pwd)"
cd "$repo"

source .venv/bin/activate
export LLVM_PATH=/opt/rocm/llvm
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_XET_HIGH_PERFORMANCE=1

exec uv run --active --no-sync -m skyrl.tinker.api \
  --base-model Qwen/Qwen3-0.6B \
  --backend jax \
  --backend-config '{"max_lora_adapters":4,"max_lora_rank":8,"train_micro_batch_size":1,"sample_max_num_sequences":4,"gradient_checkpointing":true}' \
  --host 127.0.0.1 \
  --port 8000 \
  --checkpoints-base "$workspace/checkpoints" \
  --database-url "sqlite:///$workspace/skyrl-tinker.db"
