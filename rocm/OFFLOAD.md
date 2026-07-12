# Explicit offload and checkpoint staging

## Phase 1: streamed safetensor loading

The Qwen3.5 loader now indexes checkpoint metadata with deterministically
sorted `safe_open(..., framework="numpy", backend="pread")` readers and
materializes only the tensor or fused projection group needed by the current
model parameter. The `pread` backend avoids memory-mapping the full checkpoint
files. Fused checkpoint projections remain a NumPy-only layout transform; the
final `device_put` and parameter sharding are unchanged. Each placement is
synchronously blocked before its host payload is released and the next model
parameter is loaded; adapter conversion, slice update, and ArrayRef
write-through are blocked as well. Prefix normalization is compatible with
existing adapter checkpoints, but any normalized key collision across shards
fails before a device transfer. Readers are scoped to the load and closed on
success or failure.

The complete local `Qwen/Qwen3.5-4B` safetensor inventory is 9,319,737,856
bytes (8.6797 GiB); the language-model subset ultimately loaded into this
backend is 8,411,510,272 bytes (7.8338 GiB). The former loader materialized and
retained the complete 8.6797 GiB inventory in one Python dictionary while
walking the language model, including checkpoint tensors the backend never
consumed, plus any temporary fused arrays. The streamed path removes that
full-inventory retained dictionary. Its actual peak-RSS saving is not yet
measured and will be smaller than 8.6797 GiB because the current tensor or
fusion group, synchronous transfer staging, the model process, and filesystem
page cache still consume memory. No RSS reduction beyond removal of that
retained dictionary is claimed without measurement.

This phase is checkpoint staging only. It does **not** change the final model
values, dtypes, shardings, steady device residency, executable working set, or
training speed. It is not active weight, optimizer, or activation offload.
Linux swap and memory-mapped page eviction are likewise not explicit offload
and must not be counted as GPU-memory savings.

## Phase 2 primitive: explicit pinned-host placement

`skyrl/tx/utils/offload.py` provides synchronous, policy-free primitives for
copying a committed, fully addressable JAX array to an explicitly requested
memory kind and for replacing a basic NNX Variable only after that copy is
ready. The target is derived from the source sharding with
`with_memory_kind`; `device_put` is passed the exact source sharding with
donation and aliasing disabled. Shape, dtype, memory kind, device assignment,
partition spec, and exact target sharding are checked after synchronization.
The source remains valid. Unsupported/global/uncommitted/deleted arrays and
NNX Variable subclasses with custom raw-value setters fail closed.

The module also provides an unwired transactional moment-tree manager. The
caller must explicitly select exact `nnx.OptState` leaves below literal `mu`
or `nu` path components and must hold exclusive ownership of the NNX tree for
each operation. A complete selected tree is copied with one tuple
`device_put`, synchronized, and fully validated before the first Variable is
replaced. The consumable handle stages every selected leaf back to its exact
recorded device sharding for one optimizer update, then re-offloads the
possibly updated values and returns a successor handle. Synchronous failures
attempt to restore every replaced raw value; incomplete rollback fails
explicitly. This is not an atomicity guarantee for asynchronous process
termination or external concurrent mutation.

CPU-only tests validate exact device-to-`pinned_host`-to-device round trips,
NamedSharding preservation, transactional failure handling, and a three-update
AdamW replay with selected moment state staged before each update and
re-offloaded afterwards. These tests establish API and state semantics only.
They do not measure ROCm DMA bandwidth, VRAM release, overlap, optimizer
latency, or end-to-end SFT/GRPO performance, and the manager is not connected
to the backend.

For the current two-slot rank-8 Qwen3.5 LoRA inventory, the two BF16 Adam
moment trees contain exactly 138,051,584 bytes (131.65625 MiB). Offloading both
would save that persistent VRAM between updates but require at least
276,103,168 bytes (263.3125 MiB) of H2D+D2H traffic per step. At an assumed
20--25 GiB/s effective PCIe rate, transfer alone is approximately
10.29--12.86 ms, before dispatch and per-leaf overhead. This is therefore an
OOM-boundary option, not an expected speed optimization; W8/W4 frozen-weight
residency and activation work have much larger capacity upside.

Production wiring remains gated on trainer integration for update,
save/load/delete, and failure paths, plus measured gfx1100 transfer behavior.
The unwired manager performs batched tree transfers, but actual pinned-host
support, physical VRAM release, PCIe throughput, transfer/compute overlap, and
step-time impact have not yet been measured on this ROCm system. No production
offload or performance benefit is claimed until those gates pass.

## Guarded gfx1100 smoke gate

`rocm/probe_optimizer_moment_offload.py` starts with one deliberately small
case: two distinct 4 MiB BF16 `nnx.OptState` leaves under exact `mu` and `nu`
paths. Its default mode imports no JAX or Flax and emits only an abstract
refusal. The guarded case performs an initial offload, one timed stage-back and
re-offload, then an untimed stage-back for a complete host bitwise oracle and a
final re-offload:

```bash
.venv/bin/python rocm/profile_rocm.py \
  --output /tmp/optimizer-moment-offload-smoke8.telemetry.jsonl \
  --card card1 \
  --interval 0.05 \
  --baseline-seconds 2 \
  --timeout 60 \
  --sensor-grace-seconds 15 \
  --max-junction-temp-c 70 \
  --max-gpu-power-watts 200 \
  --max-vram-gib 2 \
  --min-host-available-gib 8 \
  --max-swap-gib 0.001 \
  -- .venv/bin/python rocm/probe_optimizer_moment_offload.py \
       --platform rocm --allow-gpu --case smoke8 \
       --output /tmp/optimizer-moment-offload-smoke8.jsonl
```

Each timed manager method and its following synchronization barrier must finish
below 100 ms and 10 ms respectively. The artifact reports effective binary
GiB/s, exact placement/identity/sharding/phase proofs, and the final selected
leaf SHA-256 values. It keeps three accounting claims separate:

- pinned-host placement and a bitwise round trip;
- reusable JAX allocator capacity, requiring at least 95% of the selected
  8 MiB to move out of and back into `bytes_in_use` on every transition;
- physical sysfs VRAM/GTT plateaus, sampled every 50 ms for 500 ms and reported
  as informational because BFC may retain pages after buffers become reusable.

This case contains no optimizer update, model, overlap, or production-sized
state. It promotes nothing until its committed source, private child artifact,
telemetry, and post-exit cleanup receive independent audits. A passing result
only authorizes larger 64 MiB and exact 131.65625 MiB transfer gates.
