# KV Cache Bleed — Root Cause Confirmed (#3899)

## Root Cause

**The TTNN `scaled_dot_product_attention_decode` kernel (both paged and non-paged) does not properly mask positions beyond `cur_pos` in cache blocks.** Non-zero data at masked positions leaks into the attention output.

This is a **core tt-metal kernel bug**, not specific to paged attention or tt-xla.

## Standalone TTNN Proof

`test_ttnn_sdpa_decode_adjacency.py` — pure TTNN test, no XLA/vLLM:

```
Padding leak test (dirty vs clean padding):
  User 0: max_diff=4.704102 LEAK
  User 1: max_diff=5.220703 LEAK
  ...
  Overall max: 5.220703
  Padding masked: FAIL — PADDING LEAKS!
```

The leak occurs:
- For ALL `cur_pos` values (4, 8, 14, 15, 16, 24, 30)
- For ALL non-zero padding patterns (uniform, random, constant)
- In BOTH paged and non-paged SDPA decode kernels
- On both Wormhole and Blackhole architectures
- In the latest tt-metal main branch (same code)

## Impact

Any use of `scaled_dot_product_attention_decode` or `paged_scaled_dot_product_attention_decode`
where cache blocks contain non-zero data beyond `cur_pos` will produce incorrect attention output.

In vLLM, this manifests as cross-user response contamination ("bleed") because `min_context_len`
padding fills cache blocks with non-zero KV data from padding tokens. The bug affects all batch
sizes but is more visible at larger batches.

## Workaround Status

| Workaround | Effectiveness |
|-----------|---------------|
| min_context_len=64 + batch=8 | 0/20 failures |
| min_context_len=64 + batch=16 | 10/10 failures |
| min_context_len=128 + batch=16 | 1/10 failures |
| Default config + batch=16 | 4/10 failures |

**No `min_context_len` value fully eliminates the bug at high batch sizes.** The proper fix
must be in the tt-metal SDPA decode kernel.

## Kernel Analysis

The mask generation code (`generate_mask`, `fill_tile_partial` in `dataflow_common.hpp`)
**looks correct on manual review** — it generates a proper causal mask with -inf at positions
beyond `cur_pos`. The mask is applied via `add_block_inplace` or matmul fusion in the compute
kernel. CB synchronization between writer (mask generation) and compute (mask application) is
present.

Despite the code looking correct, the empirical test proves the mask doesn't work. Possible
explanations:
1. The `DYNAMIC_CHUNK_SIZE` matmul fusion path applies the mask incorrectly
2. A tile format or face layout issue causes the mask to not align with the data
3. The `NEG_INF` constant (0xFF80FF80) may not be properly handled as -inf in bf16

## Files

### TTNN kernel (tt-metal)
- `ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp` — mask generation
- `ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp` — mask application
- `ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp` — core/batch mapping

### Standalone test
- `kv_cache_bleed_isolation/test_ttnn_sdpa_decode_adjacency.py` — reproduces with pure TTNN API
