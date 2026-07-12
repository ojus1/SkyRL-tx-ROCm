# SkyRL Tinker on ROCm

This directory contains the reproducible local setup and smoke benchmark used
for an AMD Radeon RX 7900 XTX (`gfx1100`) with ROCm 7.2 and Python 3.12.

## Environments

Use separate virtual environments for the backend and cookbook client. Current
SkyRL requires Transformers 5.6.1 or newer while the current Tinker Cookbook
caps Transformers at 5.5.4.

```bash
# From this repository.
uv venv --python 3.12 .venv
UV_TORCH_BACKEND=cpu uv pip install --python .venv/bin/python -e '.[jax,tinker]'
uv pip install --python .venv/bin/python 'jax-rocm7-plugin==0.10.2'

# Clone the cookbook as a sibling of this repository.
git clone https://github.com/thinking-machines-lab/tinker-cookbook.git ../tinker-cookbook
uv venv --python 3.12 ../tinker-cookbook/.venv
UV_TORCH_BACKEND=cpu uv pip install \
  --python ../tinker-cookbook/.venv/bin/python -e ../tinker-cookbook
```

Do not install SkyRL's `gpu` extra on AMD; it selects CUDA JAX. The ROCm plugin
provides both `jax-rocm7-plugin` and `jax-rocm7-pjrt`.

## Run the verified smoke benchmark

Start the server and wait for `Starting background engine...`:

```bash
./rocm/start_skyrl.sh
```

In another terminal:

```bash
TINKER_API_KEY=tml-dummy ../tinker-cookbook/.venv/bin/python \
  rocm/run_cookbook_smoke.py
```

The one-step rank-8 LoRA SFT run on `Qwen/Qwen3-0.6B` completed on the RX 7900
XTX with 30 tokens, `train_mean_nll=2.650795`, and gradient norm `12.1875`.

## ROCm compatibility fix

JAX reports both CUDA and ROCm devices through the generic platform name `gpu`.
SkyRL previously selected its explicitly cuDNN-backed fused attention for every
GPU, causing `cuDNN is not detected` on ROCm. The backend selector now checks
JAX's platform version and uses portable XLA attention for ROCm.

## Known limitation

Forward/backward, the optimizer step, and checkpoint saves are verified. A
post-checkpoint generation probe on JAX 0.10.2 hit
`HSA_STATUS_ERROR_INVALID_PACKET_FORMAT` while loading a LoRA sampler on
`gfx1100`. Local sampling on this exact combination remains unverified.

## Qwen3.5-4B training experiments

The Qwen3.5 path is deliberately stricter than the original smoke test. Keep
the display attached to the iGPU and ensure that no process owns `/dev/kfd`.
The launcher refuses an active AMD display connector, a second ROCm server,
an occupied port, an unsafe/reused run directory, or an existing KFD owner:

```bash
./rocm/start_qwen35.sh t64-control
```

It pins `JAX_PLATFORMS=rocm`, disables JAX preallocation, enables per-layer
rematerialization, uses rank-8/two-slot LoRA and 64-token loss chunks, and
disables all XLA GPU command buffers. The latter is required on this machine:
an exact 402-leaf Qwen3.5 LoRA Adam probe passed repeated GPU updates with
command buffers disabled, while the earlier full-model failure occurred on
the second replay of the same optimizer executable.

Run the synthetic end-to-end SFT control from the separate Cookbook
environment:

```bash
TINKER_API_KEY=tml-dummy ../tinker-cookbook/.venv/bin/python \
  rocm/bench_sft.py \
  --base-url http://127.0.0.1:8001 \
  --context 64 \
  --warmup-steps 1 \
  --measured-steps 5 \
  --run-id t64-control \
  --output /tmp/qwen35-t64-control.sft.jsonl
```

This benchmark uses deterministic synthetic tokens. It measures the local
HTTP/database/future path plus forward, backward, and Adam; it is throughput
and stability evidence, not SFT quality evidence.

Use the telemetry wrapper for every GPU experiment. It writes private JSONL
and summary files, watches fatal AMDGPU journal events once per second, and
terminates wrapped/deliberately included process trees on configured limits:

```bash
server_pid="$(fuser 8001/tcp 2>/dev/null | xargs)"
TINKER_API_KEY=tml-dummy .venv/bin/python rocm/profile_rocm.py \
  --output /tmp/qwen35-profile.jsonl \
  --include-pid "server=$server_pid" \
  --terminate-included-on-safety \
  --sensor-grace-seconds 60 \
  --max-junction-temp-c 90 \
  --max-vram-gib 23 \
  --min-host-available-gib 4 \
  --timeout 900 \
  -- ../tinker-cookbook/.venv/bin/python rocm/bench_sft.py \
       --context 64 --warmup-steps 1 --measured-steps 5 \
       --run-id profiled-t64 --output /tmp/profiled-t64.sft.jsonl
```

The explicit 60-second sensor grace is needed after the headless AMD GPU has
runtime-suspended: its hwmon files can remain temporarily unreadable while the
first ROCm context starts. Safety limits still apply as soon as the sensors
become readable. See [`RESULTS.md`](RESULTS.md) for the exact validated runs
and the context buckets that remain untested.

The exact optimizer replay probe defaults to CPU. GPU use requires two
explicit acknowledgements and should only be run in a fresh process with the
profiler:

```bash
.venv/bin/python rocm/probe_jax_optimizer.py \
  --platform gpu --allow-gpu --command-buffer-mode disable --steps 3
```

ROCm causal self-attention at 512 tokens or longer cannot silently use the
quadratic XLA fallback. The currently validated Pallas geometry is opt-in with
`SKYRL_ROCM_PALLAS_ATTENTION=1` and hard-capped at 16,384 tokens. Inputs above
16K remain refused until query-bounded forward and backward kernels replace the
monolithic launch. Do not bypass this cap to attempt 32K.

The fused-stage, GDN/FlashQLA adaptation, native-GQA, tied-head, W8/W4/A8/A4,
activation-checkpoint, and custom-VJP design is in
[`MEGAKERNELS.md`](MEGAKERNELS.md). The quantized LoRA implementation currently
in `skyrl/tx/kernels/quantized_lora.py` is a CPU semantic oracle only; it is not
selected by model code and is not a GPU performance path. The native gfx1100
IU8/IU4 compile proof and production FFI requirements are in
[`QUANTIZED_FFI.md`](QUANTIZED_FFI.md). Likewise,
`skyrl/tx/kernels/qwen3_5_qkv_lora.py` is an unwired equation/precision
experiment; its portable custom VJP is explicitly not a memory or speed path.
