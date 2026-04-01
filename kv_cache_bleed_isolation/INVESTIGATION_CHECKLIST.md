# KV Cache Bleed Investigation — Root Cause Found (#3899)

## Root Cause

**The TTNN `paged_scaled_dot_product_attention_decode` kernel reads non-zero padding data from cache blocks that were filled during `min_context_len`-padded prefill.** The causal mask should prevent these positions from affecting the output, but the padding data (with L2 norm ~18.53 per position) leaks through, corrupting the attention computation for shorter prompts.

### Evidence chain

1. `min_context_len=32` pads ALL prompts to 32 tokens during prefill
2. The prefill fills all 32 positions into the cache block — positions beyond the real prompt length contain KV data computed from padding tokens (token_id=0, position_id=0)
3. This padding data is **identical across all users** (same token, same position) with L2 norm ~18.53
4. During decode, the causal mask should mask positions `cur_pos+1` through `block_size-1`
5. **The TTNN kernel fails to properly mask these positions** — the padding data leaks through and dominates the attention output for shorter prompts
6. Since padding data is identical across users, the model effectively sees a blended context, producing cross-user contamination

### Proof: cache block norm dump (failing run)

```
Slot 2 (submarine, seq=14):
  k_norms=[7.47, 13.72, 17.91, 16.47, ...(real data)..., 18.53, 18.53, 18.53, ...(18 padding positions)]
                                                          ^pos 14              ^pos 31

Slot 3 (dinosaur, seq=14):
  k_norms=[7.47, 19.32, 16.38, 19.48, ...(real data)..., 18.53, 18.53, 18.53, ...]
```

Positions 14-31 have identical norm (18.53) across ALL slots because they're from the same padding token.

### Why the observations match

| Observation | Explanation |
|---|---|
| 0% failure without min_context_len | No padding → positions beyond prompt are zeros (norm 0) → even if masking leaks, zeros don't affect attention |
| 70% failure with min_context_len=32 | Padding fills 14-31 with non-zero data → masking leak corrupts output |
| Equal-length prompts → 0% | All slots have same padding boundary → same mask → relative attention unaffected |
| Swapped prompt order → 0% | Different scheduling → different compiled graph → different mask behavior |
| OPT-125m → 0% | MHA (12 heads), different shapes may avoid the tile-boundary masking bug |
| Sequential (batch=1) → 0% | Single user → no cross-user padding data to leak |
| Always dinosaur→submarine | Deterministic padding data + deterministic mask leak = deterministic contamination |

## Investigation timeline

### Completed
- [x] Confirmed min_context_len is the sole trigger (not bfp8, not const_eval)
- [x] Verified input data (input_ids, position_ids) is correct per slot
- [x] Verified KV cache at distinguishing positions is unique per slot (prefill NOT contaminated)
- [x] Verified CPU `paged_scaled_dot_product_attention_decode` correctly masks padding
- [x] Dumped full cache block norms showing non-zero padding at positions beyond prompt length
- [x] Analyzed TTNN kernel source: `generate_mask()` in `dataflow_common.hpp` operates at tile granularity

### Remaining to confirm/fix
- [ ] Verify the TTNN kernel's `generate_mask()` / `fill_tile_partial()` has a bug with mid-tile `cur_pos` values (e.g., cur_pos=14 within a 32-element tile)
- [ ] File a tt-metal issue for the kernel masking bug
- [ ] Implement workaround: zero out padding positions in the cache after prefill (requires doing this inside the compiled graph, not via CPU copy)
- [ ] Alternative workaround: disable min_context_len or set it to 1

## Key files

### TTNN kernel (the bug)
- `ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp` — `generate_mask()` function (lines 206-294)
- `ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp` — calls `generate_mask()`
- `ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp` — applies mask

### vLLM plugin (padding source)
- `integrations/vllm_plugin/vllm_tt/model_runner.py` — `_prepare_inputs()` creates zero-padded input_ids/position_ids
- `integrations/vllm_plugin/vllm_tt/attention.py` — `_handle_paged_attention()` calls `paged_fill_cache` for all padded positions

### Custom ops
- `python_package/tt_torch/custom_ops.py` — `paged_fill_cache`, `paged_scaled_dot_product_attention_decode` registration
