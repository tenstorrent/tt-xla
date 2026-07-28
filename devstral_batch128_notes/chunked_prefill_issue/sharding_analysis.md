# Sharding analysis — Devstral-2-123B on BH galaxy DP+TP mesh [4, 8]

> Living document. Answers ONE central question: can the TP-axis RowParallel
> all-reduce (`o_proj`, `down_proj`; `cluster_axis=1`) — the CCL that as a fused
> `ttnn.all_reduce` hangs `end_trace_capture` on galaxy — be **avoided or
> minimized** by a better sharding strategy, rather than merely decomposed to
> `reduce_scatter + all_gather`?
>
> **Answer up front:** No — the cross-TP *reduction* is fundamental to dense
> Megatron TP and cannot be sharded away. What *can* be avoided is the fused
> `ttnn.all_reduce` **op**, and that is already handled by the tt-mlir
> decomposition workaround (measured: `all_reduce=0, reduce_scatter=8,
> all_gather=12`, `end_trace_capture` SUCCEEDS — `devstral_batch128_notes/report.md`,
> blocker #3/D47–D52). Sequence parallelism would emit the *same two collective
> types* and is therefore no better for the trace hang, while carrying a known
> lowering risk in this path. The genuine sharding wins are two redundant-CCL
> cleanups (embedding DP round-trip; KV-cache replication), analyzed in §3.

---

## Evidence status legend

- **[MEASURED]** — observed on hardware / in exported counts this session
  (`report.md`).
- **[CODE]** — read directly from source (file:line cited).
- **[PROPOSED]** — a recommendation pending IR/hardware verification; CCL deltas
  are *expected/logical*, not measured. I ran no hardware and read no exported IR
  for this document.

---

## 1. Model & target

- **Model / variant:** `mistralai/Devstral-2-123B-Instruct-2512` (Mistral-Large
  lineage, dense decoder-only). `tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py:125,326`.
- **Bring-up config under debug:** `num_hidden_layers=2`, `prefill_chunk_size=128`,
  batch 128, bfp8 weights+KV, trace on, chunked prefill, const-eval
  (`test_...py:314,368,372`). The 2-layer truncation is a bring-up device; the
  production model is ~88 layers.
- **Architecture (full model, typical Mistral-Large — verify against the HF
  config before quoting production numbers):** hidden ≈ 12288, ≈96 Q heads,
  ≈8 KV heads (GQA), FFN inner ≈28672, vocab ≈131072. **[flag: dims not verified
  from HF config in this session; layer *structure* is the standard dense
  Llama/Mistral block, which is all the sharding argument needs.]**
- **Target mesh:** `[4, 8]` on BH galaxy (32 chips). Axis 0 = `"batch"` = **DP=4**
  (`cluster_axis=0`); axis 1 = `"model"` = **TP=8** (`cluster_axis=1`). Matches
  `references/mesh_shapes.md` (galaxy is 32 chips) reshaped to a 2D DP+TP mesh.

Per-transformer-layer Megatron column→row pairs (the CCL-relevant structure):

| Pair | Column-parallel (no CCL) | Row-parallel (1 reduction) | Weight spec (code) |
|---|---|---|---|
| Attention | `q/k/v_proj` `("model", batch)` | `o_proj` `(batch, "model")` | `XlaQKVParallelLinear._shard_weight` 202-204 / `partition_row_parallel_linear` 307 |
| MLP | `gate/up_proj` `("model", batch)` | `down_proj` `(batch, "model")` | `XlaMergedColumnParallelLinear._shard_weight` 111 / 307 |

All line numbers refer to `integrations/vllm_plugin/vllm_tt/vllm_distributed_utils.py`
unless noted. This is textbook Megatron-LM §3 (arXiv:1909.08053) and matches the
repo's own reference implementations (`tests/jax/multi_chip/bounties/qwen2_5_7b/README.md`
lines 20-33: "2 all_reduce ops per decoder layer — one after attention, one after
MLP").

---

## 2. Recommended strategy (summary)

Keep the current **DP=4 × Megatron-TP=8** scheme. Do **not** attempt to eliminate
the `o_proj`/`down_proj` reduction — it is fundamental (§3.1). Keep the tt-mlir
`TTNNAllReduceWorkarounds` decomposition that turns the fused `ttnn.all_reduce`
into `reduce_scatter + all_gather` **[MEASURED fix, report.md D50/D52]** as the
pragmatic answer to the trace hang. Reject sequence parallelism as the fix (§3.2):
it emits the same two collective types, adds a known lowering risk in this path,
and its only benefit (activation memory) is marginal in decode.

The real, low-risk wins are two **redundant-CCL cleanups** independent of the
all-reduce (§3.3):
- **(a) [PROPOSED]** change the embedding forward-hook constraint from
  `(None, None, None)` (fully replicated) to `("batch", None, None)` — removes a
  per-forward DP-axis all_gather + re-partition round-trip.
- **(b) [PROPOSED, partly blocked]** DP-batch shard the KV cache. Blocked today by
  `ttir.paged_update_cache` (the documented follow-up); current full replication
  is the correct-but-memory-heavy fallback.

---

## 3. Analysis

### 3.1 Is the `o_proj`/`down_proj` TP all-reduce fundamental? — **Yes.**

**Distinguish two things the question conflates:**

1. **The cross-TP reduction (the math): CANNOT be avoided in dense Megatron TP.**
   A column→row matmul pair shards the contraction dim of the second matmul across
   the TP axis, so each TP device computes a **partial sum** over its slice of the
   contracted dimension. Producing the correct full result *requires* summing those
   partials across the TP axis — **one reduction per column→row pair is the
   theoretical minimum** (`references/general_sharding.md` lines 16-18: "row-parallel …
   produces a partial sum → **one all-reduce** to combine"; `references/ccl_cheatsheet.md`
   line 16; Megatron-LM §3, arXiv:1909.08053). There is no dense-TP sharding that
   makes `o_proj`/`down_proj` produce a complete result without a cross-TP exchange.
   The only ways to have *no* cross-TP reduction are to (i) not shard the layer
   across TP at all (replicate — a 123B model will not fit / defeats TP), or
   (ii) switch paradigms (expert/pipeline parallelism — not applicable to these
   dense projections). Neither is a "better sharding of the same layer."

2. **The fused `ttnn.all_reduce` *op*: CAN be avoided/reformed.** The reduction
   must happen, but it need not be emitted as a single fused `ttnn.all_reduce`.
   The standard identity `all_reduce ≡ reduce_scatter + all_gather`
   (`references/ccl_cheatsheet.md` lines 22-27) lets the compiler realize the same
   reduction as two collectives. This is exactly what the re-added tt-mlir
   `TTNNAllReduceWorkarounds` pass does, and it is what fixed the trace hang:
   **[MEASURED]** `all_reduce=0, reduce_scatter=8, all_gather=12`, and
   `end_trace_capture` SUCCEEDS (`report.md` blocker #3 / D47, D49, D50, D52; root
   cause pinned to tt-mlir commit `1d91fcf556`/#8961 which had *dropped* that
   decomposition). With 2 layers and 2 pairs/layer that is 4 reductions → 8
   reduce_scatter + (8 + embedding/lm-head) all_gather, consistent with the counts.

**Bottom line for §1:** the reduction is fundamental; the fused op is not. The
already-landed decomposition removes the fused op. A "better sharding strategy"
cannot do better than *reform* this reduction — it cannot delete it.

**Rejected reform — column-parallel second matmul + `all_gather`.** One could keep
the activation TP-sharded and make `o_proj`/`down_proj` *column-parallel*
(replicated weight), then `all_gather` the input once — 1 all_gather instead of
the reduction (`references/video_vae.md` lines 145-153 documents this exact
trade: col-parallel result needs 1 all_gather vs a row-parallel all_reduce's 2
ccls). **Rejected:** it forces the *second* matmul's weight to be **replicated**
across TP. For a 123B model the `down_proj` (FFN inner ≈28672 → hidden) and
`o_proj` weights are among the largest; replicating them across TP=8 defeats the
entire memory rationale for TP. Trading ~half a collective's bandwidth for an 8×
weight-memory blow-up is a net loss here.

### 3.2 Sequence parallelism (SP) — does it emit **no** fused all_reduce? Yes, but it is **not the right tool for this problem.**

**What SP does (Megatron-LM SP, Korthikanti et al. arXiv:2205.05198; and the
repo's own `references/video_dit.md`):** SP shards the *sequence* dim of
activations in the norm/residual regions that sit *between* TP matmul pairs. The
`g`/`g-bar` conjugate pair replaces each TP-region all_reduce with an
`all_gather` at region entry and a `reduce_scatter` at region exit — activations
stay sequence-sharded across the norms. Total communication **volume is
identical** to all_reduce (that is the whole point of the identity in
`ccl_cheatsheet.md` line 22).

**Does SP emit a fused `ttnn.all_reduce`? No** — it emits `reduce_scatter` +
`all_gather`, so no fused all_reduce op, so no trace hang from *that* op.

**But that is exactly what the shipped decomposition already emits.** SP and the
`TTNNAllReduceWorkarounds` decomposition produce the **same two collective
types**. The **[MEASURED]** result (`all_reduce=0, reduce_scatter=8,
all_gather=12`) is already the SP-equivalent collective profile. So **SP cannot
help the trace hang beyond what is already shipped** — it is strictly no better
for the central problem.

**Is SP / reduce-scatter-kept-sharded supported by tt-mlir today?** — **Not
categorically unsupported, but fragile in *this* path.** Being precise, because
the evidence cuts both ways:
- **Pro (it lowers somewhere):** `references/video_dit.md` is a *production* SP
  example in the torch path — sequence-axis `all_gather` of K/V plus the
  double-`sharding_constraint` reshape workaround (lines 152-207). SP is **not**
  listed as unsupported in `references/compiler_support.md`.
- **Con (it is risky here):** in *this* vLLM decode path, a reduce_scatter feeding
  a norm is a documented hazard. `model_runner.py:1775-1777` **[CODE]**: pinning
  `position_ids` under pure-TP "perturbs GSPMD into a batch-axis
  **reduce_scatter → ttnn.rms_norm(k_norm)** graph that hits a tt-mlir
  redundant-to_layout bug (`TTNNDecomposeLayouts`)." SP's defining pattern is
  precisely reduce_scatter output flowing into a norm region — so it would invite
  the same class of bug in this model.

**Activation-memory benefit is marginal here.** SP's real payoff is memory: it
keeps the norm-region activations sequence-sharded. In **decode** the sequence
dim is ≈1, so there is essentially nothing to shard. In **prefill** with
`prefill_chunk_size=128` **[CODE test:368]**, a 128-token chunk sharded over TP=8
saves a modest amount, and the memory footprint is dominated by the KV cache
(§3.3b) and weights, not the transient chunk activations.

**Conclusion:** SP is rejected as the fix — not because it "can't lower," but
because it is **strictly no better than the already-landed decomposition** for
the trace hang (same collective types), **adds a known lowering risk** in this
path (reduce_scatter→rms_norm, `model_runner.py:1776`), and its **only real
benefit is negligible in decode**. Do not design around it.

#### CCL accounting — SP vs current, per transformer layer

Per layer there are 2 column→row pairs (attention, MLP), i.e. 2 fundamental
reductions.

| Scheme | Fused all_reduce | reduce_scatter | all_gather | Total collectives/layer | Emits fused AR op? |
|---|---|---|---|---|---|
| Current, **pre**-decomposition | 2 | 0 | 0 | 2 | **Yes** → hangs `end_trace_capture` [MEASURED] |
| Current, **with** decomposition (shipped) | 0 | 2 | 2 | 4 | No [MEASURED: 8 RS / ~12 AG at 2 layers] |
| **Sequence parallelism** | 0 | 2 | 2 | 4 | No — *same two types as decomposition* |

SP is column-identical to the shipped decomposition in every column that matters
for the hang. Its only differentiator (activation memory) does not show up in this
table and is marginal in decode.

### 3.3 Redundant / unnecessary CCLs in the CURRENT sharding (the real cleanup win)

These are independent of the fundamental reduction and are where a better
sharding actually removes collectives.

#### (a) Embedding output forced fully replicated → redundant DP-axis round-trip — **[PROPOSED fix]**

**Current [CODE]:** `partition_vocab_parallel_embedding`
(`vllm_distributed_utils.py:339-350`) shards the embedding weight `(None, "model")`
(vocab replicated, hidden on TP — vocab is deliberately *not* sharded to avoid a
`CollectivePermute` tt-mlir can't lower, tt-mlir #3370, lines 343-346), then
registers a forward hook constraining the **output** to `(None, None, None)` —
fully replicated on **both** axes (line 347).

Meanwhile the model inputs are pinned batch-sharded on the DP axis:
`_pin_input_shardings` marks `input_ids` `(batch, None)` and `inputs_embeds`
`(batch, None, None)` (`model_runner.py:1770-1774` **[CODE]**).

**The waste:** the embedding output naturally lands as
`(batch, None, "model")` — batch sharded on DP, hidden sharded on TP. Forcing
`(None, None, None)` does **two** gathers:
1. **all_gather on the TP/`model` axis** to materialize the full hidden dim
   (`cluster_axis=1`) — this one is **fundamental**: the first `q/k/v_proj` is
   column-parallel and needs the full hidden vector replicated across TP to compute
   its head slice (`references/general_sharding.md` line 15; the STATE-A
   "TP-replicated D" of `references/video_dit.md` lines 64-66). Keep it.
2. **all_gather on the DP/`batch` axis** (`cluster_axis=0`) — gathers the batch
   from 32/device to the full 128, *then* the very next batch-sharded op forces a
   `mesh_partition` re-shard back to 32/device (128→32). This DP round-trip is
   **pure waste**: nothing between the embedding and the first layer needs a
   replicated batch, and the whole stack runs batch-sharded on DP.

**Proposed fix:** change the hook target to `("batch", None, None)`. This keeps
the necessary TP all_gather (hidden still gathered for column-parallel QKV) and
**deletes the DP all_gather(32→128) + mesh_partition(128→32)** round-trip. It is
consistent with the already-pinned `(batch, None)` inputs.

**Frequency:** this happens **once per forward pass** (the model-entry embedding),
**not per layer** — so the budget line is per-forward, not ×layers. Still, the
gathered tensor is `[128, seq, hidden]` at full width, so the eliminated DP
all_gather is large.

**Hedge [PROPOSED]:** `(None, None, None)` may have been a *defensive replicated
anchor* (same function already dodges the vocab-shard CollectivePermute issue
#3370). Before landing, **verify in exported IR** that `("batch", None, None)`
does not let Shardy back-propagate a batch shard *into* the embedding gather
(analogous to the reshape back-prop hazard in `references/compiler_support.md`
"Shardy back-propagating a shard through a reshape" and the double-constraint
workaround). If it does, apply the back-to-back replicated-anchor-then-shard
pattern (`references/shardy_sharding.md` lines 96-115).

**CCL delta (expected, per forward):** −1 DP all_gather (large) − 1 DP
mesh_partition/reshard; TP all_gather unchanged.

#### (b) KV cache replicated in DP+TP — **[PROPOSED, blocked by paged_update_cache]**

**Current [CODE]:** in `DATA_TENSOR_PARALLEL` the KV cache is left un-annotated,
i.e. **fully replicated** across all 32 chips (`model_runner.py:3444-3449`):

> "DP+TP: leave the KV cache un-annotated (replicated under SPMD); each device
> writes its own K/V slice via paged_update_cache. The TP-only spec puts
> block_size on the DP axis and fails `ttir.paged_update_cache`. Tracked as a
> follow-up."

Contrast the pure-TP branch (`model_runner.py:3450-3463` **[CODE]**), which *does*
shard: `(None, "model", None, None)` — the KV-head dim on the TP axis (guarded by
`num_kv_heads % tp_size == 0` at 3407-3415). Cache layout is
`[num_blocks, block_size, num_kv_heads, head_dim]`
(`TTAttentionBackend.get_kv_cache_shape`, 3416-3421 **[CODE]**).

**Is a proper DP-batch shard feasible / what blocks it?** The paged KV layout has
**no batch/sequence axis** — it is a global block pool indexed by a page table.
The natural DP-shard would therefore have to fall on `num_blocks` or `block_size`,
and that is exactly what breaks `ttir.paged_update_cache`: the write op indexes
blocks via the page table, and sharding the block/`block_size` dim across the DP
axis makes those indices device-local in a way the op does not support (the
documented "puts block_size on the DP axis and fails ttir.paged_update_cache"
follow-up). So a clean DP-batch shard of the paged cache is **not achievable
today** without a `paged_update_cache` change in tt-metal/tt-mlir. This is a
**compiler/runtime blocker, not a Shardy-annotation choice.**

**Would fixing it remove the chunked-path `page_table` `cluster_axis=0`
mesh_partition?** The page tables are already batch-sharded on DP in all three
prepare paths: `safe_mark_sharding(page_table, self.mesh, ("batch", None))` at
`model_runner.py:1453`, `2300`, `2703` **[CODE]** (with the comment
"page_table / cache_position / batch_idx must share the K/V input's" sharding,
1449-1453). A DP-sharded cache whose blocks were partitioned to match the
batch-sharded page tables *could* in principle let the page-table indices stay
device-local and remove a resharding of the page table. **However, whether that
eliminates a specific `cluster_axis=0` mesh_partition is UNVERIFIABLE from source
alone — it needs the exported IR for the chunked path.** I flag this as an
expected-but-unconfirmed benefit, not a claim.

**Net:** the current replication is the correct fallback; it costs memory (the
block pool is not TP-head-sharded here, unlike the pure-TP branch) but adds no
CCL. A DP-batch shard is desirable (memory) and *might* trim a page-table reshard,
but is **blocked** on `ttir.paged_update_cache`. Recommend tracking the tt-mlir/
tt-metal follow-up; do not attempt it via Shardy annotation alone.

**Out-of-scope flag (not chased):** the logits replication constraints
`sharding_constraint_tensor(..., (None, None))` at `model_runner.py:3497` and
`3514` **[CODE]** have the same DP-round-trip smell as (a) — a full replicate on a
batch-sharded tensor. Outside the task's enumerated (a)/(b) scope; flagged for a
future look, not analyzed here.

---

## 4. Recommendation, rejected alternatives, and total CCL accounting

### Recommendation (in priority order)

1. **Keep the tt-mlir `TTNNAllReduceWorkarounds` decomposition** (reduce_scatter +
   all_gather) as the fix for the fused-all_reduce trace hang. **[MEASURED to work;
   compiler-supported today — it is the shipped fix.]** The reduction is
   fundamental (§3.1); this is the pragmatic and correct resolution.
2. **Land the embedding hook change `(None,None,None) → ("batch",None,None)`**
   (§3.3a) after verifying in exported IR that no batch shard back-propagates into
   the embedding gather. **[PROPOSED; low risk; removes a per-forward DP
   round-trip.]**
3. **Track the KV-cache DP-shard as a tt-mlir/tt-metal follow-up** on
   `ttir.paged_update_cache` (§3.3b). **[PROPOSED; BLOCKED today.]** Do not attempt
   via annotation alone.
4. **Do not adopt sequence parallelism** as the all-reduce fix (§3.2). **[Rejected:
   no better than the decomposition for the hang; known lowering risk here;
   marginal decode benefit.]**

### Rejected alternatives (summary)

| Alternative | Why rejected | Source |
|---|---|---|
| Eliminate the `o_proj`/`down_proj` reduction by re-sharding | Impossible in dense Megatron TP — the reduction is the math | general_sharding.md 16-18; Megatron §3 |
| Column-parallel 2nd matmul + `all_gather` (1 CCL vs the reduction) | Forces replicating the largest weights across TP=8 → memory blow-up defeats TP | video_vae.md 145-153 |
| Sequence parallelism to replace the all_reduce | Same 2 collective types as the shipped decomposition; reduce_scatter→rms_norm bug risk here; marginal decode memory benefit | §3.2; model_runner.py:1776; ccl_cheatsheet.md 22 |
| Replicate the whole layer (no TP) | 123B will not fit; defeats the purpose | general_sharding.md 9-13 |

### Total CCL accounting — current (shipped) vs recommended

Per **transformer layer** (2 pairs), the fundamental collective count is unchanged
by any recommendation — the reduction is fundamental:

| Component | Collective | Count/layer | Notes |
|---|---|:---:|---|
| Attention `o_proj` reduction | reduce_scatter + all_gather | 1 + 1 | fundamental; fused-AR-free [MEASURED] |
| MLP `down_proj` reduction | reduce_scatter + all_gather | 1 + 1 | fundamental; fused-AR-free [MEASURED] |
| **Per-layer total** | | **2 RS + 2 AG** | identical for shipped-decomposition and SP |

Per **forward pass** (model-level, once), the recommendation changes the entry/exit:

| Component | Current (shipped) | Recommended | Delta |
|---|---|---|---|
| Embedding output (§3.3a) | TP all_gather + **DP all_gather + DP reshard** | TP all_gather only | **−1 DP all_gather − 1 DP reshard** [PROPOSED] |
| KV cache (§3.3b) | replicated (0 CCL, high memory) | DP-shard (memory win; page-table reshard maybe removed) | **BLOCKED**; CCL delta unverifiable |
| `o_proj`/`down_proj` reduction | 2 RS + 2 AG per layer | unchanged | 0 (fundamental) |

**Model total (recommended vs current):** the only *deletable* collectives are the
per-forward embedding DP round-trip (recommendation 2). The per-layer reductions —
the dominant CCL cost, scaling with ~88 layers — are fundamental and stay as the
decomposed `reduce_scatter + all_gather`. **Honest headline: the all_reduce cannot
be eliminated, only reformed/decomposed; the decomposition already in place is the
pragmatic fix; the genuine sharding win is the redundant-CCL cleanup (embedding DP
round-trip), with the KV-cache DP-shard a memory win blocked on
`ttir.paged_update_cache`.**

### Compiler-support status of each recommendation

| Recommendation | Supported today? |
|---|---|
| Keep decomposition (rec 1) | **Yes — shipped & MEASURED** (report.md D50/D52) |
| Embedding `("batch",None,None)` (rec 2) | **Expected yes** — plain activation batch-shard already used at inputs (1770-1774); verify no back-prop into embedding gather [PROPOSED] |
| KV-cache DP-shard (rec 3) | **No — blocked** by `ttir.paged_update_cache` (model_runner.py:3446-3448) |
| Sequence parallelism (rec 4, rejected) | Lowers in torch path (video_dit.md) but **fragile here** (reduce_scatter→rms_norm, model_runner.py:1776); rejected regardless |

---

## 5. Bottlenecks & compiler limitations

- **Fused `ttnn.all_reduce` hangs `end_trace_capture` on galaxy** — root cause
  tt-mlir `1d91fcf556`/#8961 dropped `TTNNAllReduceWorkarounds`; re-adding it
  (decompose to reduce_scatter + all_gather) fixes it. **[MEASURED, report.md #3.]**
- **reduce_scatter → `ttnn.rms_norm` hits `TTNNDecomposeLayouts` bug** in this
  path (`model_runner.py:1776`) — the reason SP is risky here and `position_ids`
  is not pinned under pure-TP.
- **Vocab-sharded embedding needs a `CollectivePermute` tt-mlir can't lower**
  (tt-mlir #3370) — why the embedding weight keeps vocab replicated
  (`vllm_distributed_utils.py:343-346`).
- **`ttir.paged_update_cache` blocks a DP-batch KV-cache shard**
  (`model_runner.py:3446-3448`) — the §3.3b follow-up.
- **Shardy back-prop through reshape** (`references/compiler_support.md`) — the
  hazard to check when landing the §3.3a embedding change; mitigated by the
  back-to-back replicated-anchor→shard pattern (`references/shardy_sharding.md`).

## 6. Sources

- Megatron-LM tensor parallelism — arXiv:1909.08053 (§3: column→row pair = 1
  all_reduce).
- Megatron-LM sequence parallelism — Korthikanti et al., arXiv:2205.05198.
- `references/general_sharding.md` (column/row parallel, 1 all-reduce/pair).
- `references/ccl_cheatsheet.md` (pattern→collective; `all_reduce ≡
  reduce_scatter + all_gather`).
- `references/compiler_support.md` (reshape back-prop; conv halo — n/a here).
- `references/mesh_shapes.md` (galaxy = 32 chips).
- `references/video_dit.md` (production SP in torch path; STATE-A/B; double
  constraint).
- `references/video_vae.md` (col-parallel + all_gather vs row-parallel all_reduce
  trade).
- `references/shardy_sharding.md` (mark_sharding / back-to-back constraint pattern).
- `integrations/vllm_plugin/vllm_tt/vllm_distributed_utils.py` (partition fns:
  202-204, 292-309, 318, 339-350).
- `integrations/vllm_plugin/vllm_tt/model_runner.py` (input pinning 1770-1782;
  reduce_scatter→rms_norm note 1775-1777; page_table batch-shard 1453/2300/2703;
  KV cache 3406-3463; logits 3497/3514).
- `devstral_batch128_notes/report.md` (**MEASURED** decomposition fix: all_reduce=0,
  reduce_scatter=8, all_gather=12, end_trace_capture SUCCEEDS; blockers D45-D52).
- `tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py`
  (config: 125, 314, 326, 368-372).
- `tests/jax/multi_chip/bounties/qwen2_5_7b/README.md` (reference Megatron TP: 2
  all_reduce/layer).
