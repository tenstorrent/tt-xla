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

## Leading hypothesis: NOC write conflict in paged_update_cache

The `paged_update_cache` kernel uses **height-sharded** input tensors. Each user's
update data is on a separate core. Each core:
1. Reads its user's new KV data from L1 (local shard)
2. Reads the page_table entry to find the physical block in DRAM
3. Writes the KV data to the physical block in DRAM via NOC

Adjacent batch items (users) are on adjacent cores. Adjacent cores share NOC
paths. If two adjacent cores write to nearby DRAM addresses (adjacent physical
blocks) simultaneously, a NOC write conflict could corrupt one of the writes.

This would explain:
- Why adjacent slots cause the bug (adjacent cores, shared NOC)
- Why non-adjacent slots don't (different NOC paths)
- Why the cache data appears correct at the time of reading (the corruption
  happens during the write, not the read)
- Why `min_context_len=32` triggers it (single-block updates concentrate
  writes into a small DRAM region)

## Remaining unknowns

- Whether the NOC actually has this write arbitration issue
- Whether `paged_fill_cache` (prefill) has the same adjacency sensitivity
- Whether the corruption is in the update (decode) or the fill (prefill)

## Next steps

1. Build a targeted test with ONLY 2 prompts at controlled slot positions (adjacent vs non-adjacent)
   to isolate the adjacency condition with minimal compilation overhead
2. Check if `paged_update_cache` has the same adjacency sensitivity by:
   - Comparing cache BEFORE and AFTER the first decode step
   - Checking if position 14 contains the right per-user data
3. Inspect the TTNN experimental `paged_update_cache` kernel source for batch-adjacency bugs
4. File a tt-metal issue with the reproduction and adjacency finding
