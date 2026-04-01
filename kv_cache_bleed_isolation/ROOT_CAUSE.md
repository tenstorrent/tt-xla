# KV Cache Bleed — Status and Findings (#3899)

**Status: Root cause narrowing — adjacent-slot decode interaction with padding**

## Latest finding (critical)

The cache data is IDENTICAL between passing and failing runs. The only difference is **which slot each prompt is assigned to**:

- **PASS**: submarine and dinosaur (both seq_len=14) at slots 0 and 7 (non-adjacent)
- **FAIL**: submarine and dinosaur at slots 2 and 3 (adjacent)

This means the bug is NOT in the cache write (prefill) — it's in the decode attention kernel's handling of **adjacent batch slots** when both have the same `cur_pos` and their cache blocks contain non-zero padding.

## Trigger conditions (all required)

1. `min_context_len == block_size == 32` (padding fills entire cache block)
2. Two prompts with the same length at adjacent batch slots
3. Cache blocks contain non-zero padding data (norm ~18.53 per position)

## Confirmed NOT the cause

- The KV cache data after prefill is correct and identical between pass/fail
- The scheduling pattern (7+1 prefill) is identical between pass/fail
- bfp8, const_eval settings are irrelevant

## Hypothesis

The TTNN `paged_scaled_dot_product_attention_decode` kernel shares some intermediate state (mask, partial results) between adjacent batch items. When two adjacent users have the same `cur_pos` value AND non-zero padding data in their cache blocks, this shared state causes cross-contamination.

With `min_context_len=64` (2 blocks per user), the padding is in block 1 which is fully masked, so the kernel never processes the contaminated data. With `min_context_len=32` (1 block), the kernel processes the partial block with padding, and the adjacent-batch interaction causes the leak.

## Immediate workaround

```python
additional_config={'min_context_len': 64}  # or 128, or omit for default
```

## Kernel analysis (so far)

- `paged_scaled_dot_product_attention_decode` uses 1 core per KV head per batch item (for our config)
- K multicast is NOT active (requires `q_heads_parallel_factor > 1`)
- Causal mask generation (`fill_tile_partial`) looks correct for mid-tile `cur_pos=14`
- `k_chunk_size=0` (dynamic) → resolves to `Sk_chunk_t=1` at runtime for `cur_pos=14`
- Each core reads its own `cur_pos_tensor[cur_batch]` independently
- No obvious shared state between adjacent batch items in the kernel code

## Remaining unknowns

- The `paged_update_cache` kernel (tt-metal experimental) might have an adjacent-batch bug
- The `paged_fill_cache` kernel might write to wrong positions for adjacent batch items
- There might be a race condition in the NOC (network-on-chip) reads/writes between cores
  processing adjacent batches on the same physical cores (if the core grid wraps around)
- The compiled graph might have a batch-dimension-dependent bug in how it prepares inputs
  for the paged ops

## Next steps

1. Build a targeted test with ONLY 2 prompts at controlled slot positions (adjacent vs non-adjacent)
   to isolate the adjacency condition with minimal compilation overhead
2. Check if `paged_update_cache` has the same adjacency sensitivity by:
   - Comparing cache BEFORE and AFTER the first decode step
   - Checking if position 14 contains the right per-user data
3. Inspect the TTNN experimental `paged_update_cache` kernel source for batch-adjacency bugs
4. File a tt-metal issue with the reproduction and adjacency finding
