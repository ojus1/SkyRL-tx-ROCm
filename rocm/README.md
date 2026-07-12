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
