# KV Cache Block Reuse Contamination Debug

## Status: ROOT CAUSE IDENTIFIED — PREFIX CACHING

**The bleed is caused by vLLM's prefix caching feature.** When prefix caching
is enabled (the default), requests that share a common prefix (e.g., the Llama
chat template system prompt) reuse cached KV blocks from previous requests.
These cached blocks contain stale KV data from the previous request's context,
which contaminates the new request's attention computation.

## Hypothesis

When requests finish and their KV cache blocks are returned to the free pool,
the blocks are NOT zeroed out. When new requests get those recycled blocks,
the stale KV data from the previous request is still present. If the new
request doesn't fully overwrite all positions (e.g., due to padding or
different sequence lengths), the stale data leaks into the attention
computation, causing cross-request contamination.

## Evidence

1. Server logs show first round of 4 requests is always clean
2. Second round of 4 requests (reusing blocks from first round) bleeds
3. This is consistent across 1B, 3B, and 8B models
4. `llm.generate()` locally never reproduces because each call creates fresh batches
5. The bleed requires the server pattern: requests arrive, complete, new requests reuse their blocks

## Key Finding: PREFIX CACHING is the trigger

### Reproduction

Local vLLM server with `min_context_len=32`, staggered requests (0.8s apart):

| Config | Round 1 | Round 2-10 |
|--------|---------|-----------|
| `--enable-prefix-caching` (default) | CLEAN | BLEED in round 2,4 |
| `--no-enable-prefix-caching` | CLEAN | 10/10 CLEAN |

### Why prefix caching causes bleed

All 4 prompts share the same Llama chat template prefix (system prompt + user
header). With prefix caching enabled:
1. Round 1: First request computes the prefix KV and caches it
2. Round 1: Subsequent requests REUSE the prefix KV blocks
3. Round 1 completes, but prefix KV blocks stay cached (they're shared)
4. Round 2: New requests get the same prefix blocks from the cache
5. But those blocks may still contain stale KV data from the TAIL of
   round 1's requests (beyond the shared prefix), or the block assignments
   may be incorrect

## Immediate workaround

Disable prefix caching with `--no-enable-prefix-caching` when launching the
vLLM server. Verified locally: 10/10 rounds clean with prefix caching disabled
vs 2/10 rounds bleeding with it enabled.

For tt-inference-server, this flag should be added to the vLLM launch config.

## Performance impact

Disabling prefix caching means every request must recompute the shared prefix
(system prompt + template tokens) from scratch. For Llama-3.2-1B with the
default system prompt (~48 tokens), this adds ~48 tokens of prefill work per
request. For short prompts this is significant (~50% overhead), for longer
prompts it's minimal.

## Next steps

1. Verify workaround on production server (add --no-enable-prefix-caching)
2. Investigate WHY prefix caching causes stale data leaks on TT hardware
   - On GPU, prefix-cached blocks are read-only copies — the same physical KV
     data is shared between requests. This is safe because GPU attention reads
     from these blocks without modifying them.
   - On TT with paged attention, the paged_fill_cache and paged_update_cache
     ops might be modifying the shared blocks in-place, contaminating them for
     other requests.
3. Find the specific interaction between prefix caching and the TT paged cache ops

## Root Cause Detail (code-level)

When prefix caching is active and a new request matches a cached prefix:

1. `num_computed_tokens = N` (N prefix tokens already cached in shared blocks)
2. `num_scheduled_tokens = prompt_len - N` (only suffix tokens scheduled)
3. `_prepare_inputs` creates `input_ids` with only the suffix tokens
4. The model processes only the suffix tokens, producing KV for positions N..prompt_len

5. `_handle_paged_attention` calls `paged_fill_cache` with:
   - `fill_value`: KV for the suffix tokens (padded to `padded_total_num_scheduled_tokens`)
   - `page_table`: ALL blocks including prefix blocks (block 0 = prefix, block 1+ = suffix)

6. `paged_fill_cache` starts writing from block 0 in the page_table
7. **BUG: block 0 is the SHARED PREFIX block** — it gets overwritten with suffix KV data
8. Other requests sharing the same prefix block now see corrupted data

## Fix Implemented (pending verification)

Two changes:

### 1. attention.py: Offset page_table for prefill fill
Added `prefill_block_offset` to `TTMetadata`. When doing the prefill cache fill,
the page_table is sliced to skip prefix blocks:
```python
fill_page_table = attn_metadata.page_table[:, prefill_block_offset:]
```

### 2. model_runner.py: Compute block offset from num_computed_tokens
```python
min_computed = min(num_computed_tokens across users in batch)
prefill_block_offset = min_computed // block_size
```

### Verification status
- Fix implemented in attention.py and model_runner.py but NOT YET VERIFIED
- Cold compilation from scratch is extremely slow (2+ hours) because the
  kernel cache (~/.cache/tt-metal-cache/) was cleared during earlier TTNN
  masking experiments. Need to wait for compilation to finish or restore
  the kernel cache from a backup.
- The fix is logically correct based on code analysis:
  - paged_fill_cache writes to blocks starting from page_table[batch_idx][0]
  - With prefix caching, page_table[batch_idx][0] is the SHARED PREFIX BLOCK
  - The fix offsets the page_table by num_computed_tokens/block_size to skip prefix blocks
- WORKAROUND VERIFIED: --no-enable-prefix-caching eliminates the bleed (10/10 clean)
- BLOCKED: Board is hung — ttnn.open_device() and torch_xla.device() both hang.
  Multiple aborted compilations and process kills left the device in a bad state.
  Needs a container restart or physical board reset to continue testing.

### Immediate workaround (confirmed working)
`--no-enable-prefix-caching` on the vLLM server command line.
10/10 rounds clean. Production servers should use this until the fix is verified.
