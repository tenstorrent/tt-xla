# KV Cache Bleed — Status and Findings (#3899)

**Status: Root cause NOT yet confirmed. Narrowed to specific trigger condition.**

## Trigger condition (confirmed)

The bleed occurs ONLY when `min_context_len == block_size == 32`. Specifically:

| min_context_len | blocks/user | failures |
|----------------|-------------|----------|
| 32 | 1 | 7/10 |
| 64 | 2 | 0/10 |
| 128 (default) | 4 | 0/10 |
| None (no padding) | varies | 0/10 |

Other settings (bfp8, const_eval) are NOT required to trigger the bug.

## Immediate workaround

Use any `min_context_len` value other than 32 (or the default 128):
```python
additional_config={'min_context_len': 64}  # or 128, or omit entirely
```

## What we know

1. The bleed is deterministic: always dinosaur→submarine, identical text every run
2. Input data (input_ids, position_ids) is verified correct for all slots
3. KV cache at real token positions (0 to prompt_len-1) appears correct per slot
4. Cache padding positions (prompt_len to block_size-1) have non-zero data from prefill
5. The CPU `paged_scaled_dot_product_attention_decode` correctly masks padding positions
6. The TTNN mask generation code (`fill_tile_partial`) appears correct on manual review
7. Positions 15-16 in submarine's cache block show anomalous norms (differ from the uniform padding at 17-31) — unexplained

## What we DON'T know

- Whether the TTNN attention kernel actually applies the mask correctly at runtime
- Why positions 15-16 have different data from 17-31 in layer 0 cache
- Why the bug is specific to 1-block-per-user (min_context_len=32) vs multi-block
- The exact mechanism of cross-user contamination

## Key concern: anomalous positions 15-16

For submarine (14 real tokens), layer 0 cache block norms:
```
pos 0-13: varying (real data)
pos 14:   15.66 (decode token, overwritten by paged_update_cache)
pos 15:   20.00 ← differs from uniform padding
pos 16:   13.59 ← differs from uniform padding
pos 17-31: 18.53 (uniform — expected for padding with same token/position)
```

For layer 0, ALL padding positions should have identical KV values (same token_id=0, position_id=0, no attention dependency). Positions 15-16 having different norms suggests either:
- paged_fill_cache wrote wrong data to these positions
- Something else modified them after prefill
- Cross-user data leaked into these specific positions

## Next steps

1. **Inspect flatbuffer with ttrt**: Check compile-time params for the SDPA decode kernel
2. **Dump cache BEFORE first decode step**: Catch the cache right after prefill, before paged_update_cache runs, to see if positions 14-16 already have anomalous data
3. **Compare positions 15-16 across users**: Check if submarine's positions 15-16 match another user's real data at those positions (would confirm cross-user leak)
