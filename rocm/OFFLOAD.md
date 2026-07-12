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
