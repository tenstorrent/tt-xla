# KV Cache Size Decode Performance Regression — Investigation

**Date:** 2026-04-10
**Status:** Root cause identified, fix paths outlined

## Reproduction

Confirmed with OPT-125M on single P150 chip, varying only `gpu_memory_utilization`:

| gpu_memory_utilization | KV cache tokens | Decode tok/s | Relative |
|---|---|---|---|
| 0.005 | 1,728 | 68.9 | 1.00x |
| 0.01 | 3,488 | 60.0 | 0.87x |
| 0.05 | 17,472 | 21.3 | 0.31x |

Same prompt, same `max_tokens=64`, `temperature=0`. Decode performance is inversely proportional to total cache pool size.

## Root Cause

The regression is caused by **per-step full-cache copies in the compiled decode graph**, not by the paged attention kernel itself.

### Isolated kernel shows no regression

The `paged_scaled_dot_product_attention_decode` TTNN kernel was benchmarked in isolation with 64 vs 2048 blocks (32x difference). Ratio: **1.0x** — the kernel only accesses active blocks via the page table. The kernel is not the problem.

### Full model graph copies entire cache 3x per layer per step

The combined K+V cache layout `(2, num_blocks, num_kv_heads, block_size, head_dim)` forces a slice-update-concat pattern in every decode step. From the TTNN IR:

```
# Per layer, per decode step:
%3  = slice_static(%2, [0:1,...])         # Extract K from combined cache  → FULL COPY
%24 = slice_static(%2, [1:2,...])         # Extract V from combined cache  → FULL COPY
paged_update_cache(%4, new_kv, ...)       # In-place update (void, no copy)
paged_attention_decode(q, %4, %25, ...)   # Attention (efficient, paged)
%29 = concat(%4, %25)                    # Re-combine K+V               → FULL COPY
%30 = reshape(%29)                       # Return combined cache
```

That's **3 full-cache copies per layer per decode step**. For OPT-125M (12 layers):
- 36 full-cache copies per decode step
- Small (54 blocks): 36 × 5.3 MB = **191 MB** of data movement
- Large (546 blocks): 36 × 53.7 MB = **1,933 MB** of data movement

### No input/output aliasing

The XLA graph has `mhlo.input_output_alias = []`. The runtime cannot alias input and output cache buffers — it allocates fresh output buffers and copies results, adding further overhead proportional to cache size.

### Source code trace

The slice/concat comes from `attention.py:451-502`:

```python
def _handle_paged_attention(self, inputs, kv_cache, attn_metadata):
    k_cache = kv_cache[0]      # Line 455: slice → full copy in XLA
    v_cache = kv_cache[1]      # Line 456: slice → full copy in XLA

    k_cache = torch.ops.tt.paged_update_cache(k_cache, ...)  # In-place at TTNN level
    v_cache = torch.ops.tt.paged_update_cache(v_cache, ...)

    new_kv_cache = torch.stack([k_cache, v_cache], dim=0)  # Line 501: concat → full copy
    kv_cache.copy_(new_kv_cache)                            # Line 502: copy back
```

The combined `(2, N, ...)` cache shape is defined at `attention.py:104`:
```python
def get_kv_cache_shape(...):
    return (2, num_blocks, num_kv_heads, block_size, head_size)
```

## Data

### Decode graph structure (OPT-125M)

Graph `SyncTensorsGraph.3665`:
- **164 input args**: 12 combined KV caches (`2×N×12×32×64`), ~80 weight tensors, dynamic inputs
- **13 return values**: 12 updated KV caches (`2×N×12×32×64`), 1 model output
- **24 `paged_update_cache` ops** (12 layers × K + V)
- **12 `paged_attention_decode` ops** (12 layers)

### Cache tensor sizes

| Config | num_blocks | Per-cache (bf16) | All 12 layers |
|---|---|---|---|
| 0.005 | 54 | 5.3 MB | 63.7 MB |
| 0.05 | 546 | 53.7 MB | 644.1 MB |

## Fix Paths (ordered by expected impact)

### 1. Separate K and V cache tensors (eliminate slice/concat)

Store K and V as separate tensors instead of combined `(2, N, ...)`. This eliminates the 2 slice + 1 concat copies per layer. Requires changing:
- `get_kv_cache_shape()` to return separate K and V shapes
- `_handle_paged_attention()` to accept separate tensors
- `_compute_decode_attention()` to accept separate tensors
- `initialize_kv_cache()` in model_runner to allocate separate tensors

**Impact:** Eliminates 36 full-cache copies per step (the dominant cost).

### 2. Enable input/output aliasing for cache tensors

Set `mhlo.input_output_alias` to alias cache inputs to outputs. Since `paged_update_cache` is in-place at the TTNN level, the runtime could reuse the same buffer. This would eliminate the output allocation + copy at the graph execution boundary.

**Impact:** Eliminates 12 additional cache-sized allocations per step. May require changes in the PJRT plugin's graph execution path.

### 3. Avoid returning caches from the graph

If the caches are already on-device and updated in-place by `ttnn.paged_update_cache`, they don't need to be graph outputs. The graph could return only the model logits, and the caches would persist on device between steps.

**Impact:** Would eliminate all cache data from the return path. Requires careful handling of XLA's functional semantics — the runtime needs to know the buffers were modified.

### 4. `torch.stack` → in-place write-back

If the combined layout must stay, replace `torch.stack` + `copy_` with in-place slice writes:
```python
kv_cache[0].copy_(k_cache)
kv_cache[1].copy_(v_cache)
```
This might compile to in-place writes instead of full concat + copy.

**Impact:** Would eliminate the concat copy (1 of 3 per layer). The slices might still copy.

## Next Steps

1. **Quantify the fix**: Prototype fix path #1 (separate K/V tensors) on OPT-125M and measure the improvement
2. **Check if TPU backend has the same issue**: vLLM's TPU backend (`vllm/attention/backends/pallas.py`) may use the same combined layout — compare their approach
3. **Profile with `TTMLIR_ENABLE_PERF_TRACE`**: Confirm that slice/concat ops dominate decode time in the trace
4. **Verify `input_output_alias` feasibility**: Check if the TT PJRT plugin supports buffer aliasing and what's needed to enable it

## Artifacts

- `bench_kv_cache_size.py` — Benchmark script (in repo root)
- `bench_kv_cache_size.log` — Benchmark output
- `/tmp/ir_small.log` — Full DEBUG IR dump at gpu_memory_utilization=0.005
- `/tmp/ir_large.log` — Full DEBUG IR dump at gpu_memory_utilization=0.05
- `tests/torch/ops/test_kv_cache_size_perf.py` — Unit tests (kernel-level + correctness)
