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

## Next steps

1. Check if the TTNN kernel uses multi-core processing where adjacent batches share cores
2. Inspect the `num_cores_per_head` and batch-to-core mapping in the SDPA decode program factory
3. Build a targeted test: 2 users with same seq_len at adjacent vs non-adjacent slots
