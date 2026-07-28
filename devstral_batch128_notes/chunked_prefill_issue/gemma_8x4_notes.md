# Gemma-4-31B on 8×4 BH galaxy — `flat_model_io`, and why it hits the tilize error but NOT our `128≤120` assert

_Context: colleague (ddilbaz) runs Gemma-4-31B DP+TP on an 8×4 Blackhole galaxy and hits the rank-1
`mesh_partition` tilize `TT_FATAL`, but never the `128≤120` worker-grid assert we hit on Devstral.
This note explains the relationship. Some of it is verified; the "why no assert" part has an honest
open question flagged at the end._

## Her config (relevant bits)
- `google/gemma-4-31B-it`, `mesh_shape=[8,4]` → **DP=8, TP=4**
- `enable_data_parallel=True`, `enable_tensor_parallel=True`
- `shard_weights_on_batch_axis=True`  ← (ours is `False`)
- `flat_model_io=True`                ← (ours is `False`)
- `max_num_seqs=8`, `max_model_len=128` (reportedly ran batch up to 128)
- **Hits:** the rank-1 `mesh_partition` tilize FATAL (tt-metal issue #48303).
- **Does NOT hit:** our `128≤120` assert.

## The two failures are INDEPENDENT conditions
1. **`128≤120` assert** (`deriveCanonicalL1CoreRangeSet`, HeightSharded case): fires when a *decode* op
   height-shards **>120 LOCAL users** onto the BH worker grid (120 cores). It's a pure local-user-count
   check — independent of mesh shape.
2. **Rank-1 tilize FATAL (#48303)**: `mesh_partition` (the DP split) of a **rank-1 TILE** tensor lowers to
   a `slice` that validates the *padded height* dim (`shape[-2]` = 1 for a 1-D tensor) → `1 % 32 != 0` →
   FATAL — **even when the split dimension divides cleanly**. Triggered by any rank-1 tiled tensor that is
   DP-partitioned *in-graph*.

Orthogonal: a small/sharded batch avoids #1; a rank-1 tiled in-graph DP-partition triggers #2.

## What `flat_model_io` actually does (it CAUSES her tilize, doesn't avoid the assert)
- With `flat_model_io=True`, `_prepare_model_call_tensors` (`model_runner.py:~1593`) does `reshape(-1)` on
  `input_ids`/`positions`, collapsing `[max_num_reqs, tokens]` → **rank-1**.
- A flattened rank-1 input, DP-partitioned in-graph while TILE-laid → exactly the #48303 FATAL.
- Her failing tensor is ~**len 5376** (≈ `max_num_reqs × padded_tokens`, e.g. 8×672 or 32×168) — **same bug
  class** as our Devstral `paged_fill_cache` `batch_idx`, just a **different tensor** (a flattened input vs.
  an `arange`).
- So `flat_model_io` is the **cause** of her tilize error, not the reason she avoids the assert.

## Why she avoids the `128≤120` assert — evidence + OPEN question
**Evidence she has real DP at the decode op:** the precompile compiles the decode-shaped graph
(`num_tokens=1`) *before* the prefill graphs, yet she only dies later, in prefill, on the tilize. So her
decode op did **not** trip `128≤120` → it sees **≤120 users** → her batch sharding **reaches the decode op**
(true DP, ~128/8 = **16 users/device**). That's the same end-state our Devstral `position_ids` fix produces,
reached by a different route.

**Open question — WHY her DP lands without our `position_ids` fix (NOT fully verified):**
- `shard_weights_on_batch_axis=True` marks weights `("model", "batch")` (`vllm_distributed_utils.py:108-111`)
  → the weight's **input/contraction dim is sharded on the DP axis** (FSDP-style). This distributes
  activations across DP and *may* stop Shardy from collapsing the batch back to replicated — a different
  reconciliation than our weights-TP-only (`shard_weights_on_batch_axis=False`) config, where the unpinned
  `position_ids` forced an all-gather of the hidden states back to the full batch (→ our assert).
- Her `input_ids` batch mark may also propagate to decode without that gather, possibly aided by the above
  and/or `flat_model_io`.

**Correction to an earlier (wrong) claim:** `shard_weights_on_batch_axis` does **not** "keep `position_ids`
sharded." It shards **weights** (input dim, on DP). `position_ids` sharding is still governed by the input
marks — i.e. the same `_pin_input_shardings` survival gap we fixed on Devstral. Her `position_ids` may well
be replicated like ours was; it just may not matter, given the weight-side batch distribution.

**To settle it, check her post-Shardy IR:** does the decode **query** / `position_ids` carry `_axis_0`?
What is the decode op's **users count** (16 vs 128)? That distinguishes "weights anchor the batch" from
"input_ids sharding propagates."

## Relationship to our Devstral work
- Our `128≤120` fix = mark `position_ids` via `_pin_input_shardings` (keeps the backbone batch-sharded; no
  post-embedding all-gather). Verified sufficient on its own (ablation).
- Our `batch_idx` tilize = the **same #48303 class** as her flattened-input tilize.
- **Durable cross-config fix** (covers both her and us, and `flat_model_io` variants): tt-metal **#48303**
  (make `mesh_partition`/slice of a rank-1 TILE tensor legal), or a blanket policy of keeping
  DP-partitioned rank-1 index/flattened tensors **ROW_MAJOR**. Per-tensor fixes (our `batch_idx`
  `sharding_constraint`, or hoisting) unblock a specific config but are whack-a-mole across configs.
