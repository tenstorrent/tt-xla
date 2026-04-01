# KV Cache Bleed Root Cause (#3899)

## TL;DR

The TTNN `paged_scaled_dot_product_attention_decode` kernel leaks non-zero padding data from cache blocks filled during `min_context_len`-padded prefill. The causal mask fails to fully exclude positions beyond `cur_pos` within a partially-filled tile, causing cross-user contamination.

## The mechanism

1. `min_context_len=32` pads all prompts to 32 tokens during prefill
2. `paged_fill_cache` writes KV data for all 32 positions (14 real + 18 padding for a 14-token prompt)
3. Padding positions get KV from token_id=0/position_id=0 — **identical across all users**, L2 norm ~18.53
4. During decode, `paged_scaled_dot_product_attention_decode` should mask positions 15-31 for a 14-token prompt (cur_pos=14 after first decode token)
5. The TTNN kernel's `generate_mask()` fails to properly mask these mid-tile positions
6. Unmasked padding data (identical across users) dominates the attention, blending all users' contexts

## Immediate workaround

Set `min_context_len` to `None` or `1` in the vLLM config:
```python
additional_config={'min_context_len': 1}  # or omit entirely
```

## Proper fix needed

The `generate_mask()` function in `dataflow_common.hpp` (tt-metal TTNN kernel) needs to correctly handle `cur_pos` values that fall mid-tile (not on tile boundaries). The `fill_tile_partial()` call should zero/mask positions `cur_pos+1` through the end of the tile.

## Verification

```
# Reproduces (70% failure):
min_context_len=32, Llama-3.2-1B, batch=8

# Fixed (0% failure):
min_context_len=None, same model and batch
```
