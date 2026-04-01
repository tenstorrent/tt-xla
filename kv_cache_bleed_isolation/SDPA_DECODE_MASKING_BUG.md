# SDPA Decode Causal Mask Bug — Evidence and Investigation

**Issue**: [tt-xla#3899](https://github.com/tenstorrent/tt-xla/issues/3899) | **Upstream**: [tt-metal#41215](https://github.com/tenstorrent/tt-metal/issues/41215)

---

## Summary

The `paged_scaled_dot_product_attention_decode` and `scaled_dot_product_attention_decode` TTNN ops do not properly mask positions beyond `cur_pos` in cache blocks. Non-zero data at those positions leaks through the causal mask and corrupts the attention output. In batched vLLM inference, this causes users to see content from other users' responses mixed into theirs.

Reproduces on both Wormhole and Blackhole. Standalone TTNN repro provided (no XLA/PJRT/vLLM).

---

## How batching works in these tests

`llm.generate(PROMPTS, sp)` passes all prompts in a **single call**. vLLM's engine batches them together internally — during decode, all users run in **one model forward call** with batch dimension = number of prompts. The SDPA decode kernel processes all users simultaneously in a single kernel invocation with `Q=[1, batch, num_heads, head_dim]`. The outer `for run in range(N)` loop just repeats this batched generation to measure the failure rate.

This is the same multi-user batched inference path used in production serving, not sequential single-user processing. We verified: `max_num_seqs=1`, which forces truly one-user-at-a-time execution, gives 0% failures — confirming the bug requires batched execution.

---

## Experiment Results

### Experiment 1: Equal vs variable prompt lengths

| Prompts | Config | Failures |
|---------|--------|----------|
| Equal-length (all 17 tokens) | `min_context_len=32`, batch=8 | **0/20** |
| Variable-length (14–19 tokens) | same config | **12/20** |

Equal-length prompts never hit the bug because all users have the same padding boundary. The kernel leaks equally for everyone, so there's no *relative* difference between users — the corruption is uniform and invisible.

### Experiment 2: Which prompts get corrupted?

Across 30 runs with variable-length prompts (14–19 tokens), all 17 bleed instances were between **submarine** (14 tokens) and **dinosaur** (14 tokens) — the two shortest prompts that share the same length.

This is consistent with experiment 1, not contradictory. The bug requires a **differential in padding amounts** within the batch:

- With `min_context_len=32`: submarine and dinosaur (14 tokens each) have 18 padding positions. Other prompts (15–19 tokens) have 13–17 padding positions.
- The shorter prompts have the most padding → the leaked padding has the most impact on their attention output.
- Since submarine and dinosaur share the same `cur_pos=14`, the leaked padding values are identical between them, causing their outputs to blend.

When all prompts are the same length (experiment 1), everyone leaks equally — no relative difference, no visible bleed.

### Experiment 3: Batch=32

5/5 failures, 87–102 bleed instances per run. The bug scales with batch size — more users means more same-length collisions and more cross-contamination.

### Repro

```bash
python3 kv_cache_bleed_isolation/test_bleed_validation_experiments.py       # all experiments
python3 kv_cache_bleed_isolation/test_bleed_validation_experiments.py 1     # experiment 1 only
```

---

## How we connect the dots from vLLM batching to the TTNN op

### Step 1: Narrow the trigger

Through controlled experiments, we eliminated bfp8, const_eval, and scheduling as causes. The sole trigger is `min_context_len` — the setting that pads shorter prompts to a fixed token count during prefill. Without it (or with equal-length prompts), no bleed.

### Step 2: Prove the cache writes are correct

We instrumented `model_runner.py` to dump the KV cache contents at the prefill→decode transition. The cache data is correct per-slot (each user's real token positions have unique, correct KV values) and is byte-identical between passing and failing runs. The only difference between pass/fail is which batch slot each prompt occupies.

### Step 3: Prove the padding is the problem

Cache blocks contain non-zero data at positions beyond each user's real prompt length — these are KV values computed from padding tokens (`token_id=0`, `position_id=0`) during the padded prefill. All users share the same padding pattern because they all pad with the same token. The SDPA decode kernel should mask these positions via `cur_pos_tensor`, but the leaked padding creates a common signal that bleeds across users.

### Step 4: Isolate to the TTNN op directly

We built a standalone TTNN test ([`test_ttnn_sdpa_decode_adjacency.py`](test_ttnn_sdpa_decode_adjacency.py)) that calls `ttnn.transformer.paged_scaled_dot_product_attention_decode` directly — no XLA, no PJRT, no vLLM, no compiled graph. The test:

1. Fills cache blocks with known data at positions 0–13
2. Fills positions 14–31 with non-zero padding (simulating `min_context_len` prefill)
3. Sets `cur_pos=14, is_causal=True`
4. Compares output against a clean cache (zeros at 14–31)

**Result**: max output difference = **5.22** (expected ~0). The causal mask completely fails to exclude the padding positions.

The same test with the non-paged `scaled_dot_product_attention_decode` also leaks identically. We also tested with `is_causal=False` and an explicit attention mask tensor — same leak. The masking in this kernel appears to have no effect regardless of which path is used.

---

## Standalone TTNN repro output

```
Paged SDPA decode causal mask leak test
  block_size=32, cur_pos=14, num_users=2
  max_diff between dirty and clean padding: 5.2207
  Expected: ~0 (causal mask should exclude positions 15-31)
  Result: FAIL — padding leaks through causal mask

Non-paged SDPA decode:
  max_diff: 5.2207
  Result: FAIL — same bug in non-paged version
```

```bash
# Run the standalone TTNN repro (no vLLM required):
python3 kv_cache_bleed_isolation/test_ttnn_sdpa_decode_adjacency.py
```
