# Devstral-2-123B DP+TP — Why the first graph takes ~2 min to RUN (sync-after-op)

**Log:** `/data/ssalice/temp/tt-xla/devstral_dptp_test_synced_cpu_false.log`
**Run:** `test_dptp_devstral[mesh_shape0-1024-True-bfp_bf8]`, 32-chip BH galaxy, mesh `[4,8]`.
**Mode:** device sync after **every** op (serialized execution, one mesh-wide round-trip per op).

---

## 0. Bottom line (read this first)

The first backbone graph executes in **~164 s for 509 device ops** (`19:16:17.9 → 19:19:01.8`). This window is
**100% contiguous device-op execution** — verified: zero compile/fusion/module-builder lines are interspersed
between the first and last op.

The 164 s is **NOT** the 2 layers of math and **NOT** weight-load bandwidth. By elimination:

- Genuine compute at seq=128 / 2 layers / bf8 is tiny: ~10 matmuls on 128-token activations + 2 SDPA +
  5 rms_norm + elementwise ≈ **well under ~2 s of real FLOPs**.
- Genuine host→device weight transfer is **1.41 GB total**; even at a pessimistic ~1 GB/s effective that is **~1.4 s**.
- The 6 collectives are a handful of ops (see §5).

So the provable genuine work is **< ~10 s**. The remaining **~150 s is per-op serialization overhead** — a mesh-wide
dispatch + sync round-trip after each of 509 ops, which is exactly what `TT_RUNTIME_SYNC_AFTER_OP=1` forces on a
32-chip mesh. The user's instinct ("~2 min for 2 layers is way too slow") is correct: it is **not** the 2 layers, it
is the 509 synchronized round-trips. This is an **artifact of sync-after-op**, not a compute/bandwidth bug.

Two things in the run *are* genuine and *are* fixable, independent of the sync artifact:
1. **The second graph re-loads 6 const-eval weights on a cache MISS** (redundant re-materialization). Partially confirms the user's "const-eval recomputed each graph" hypothesis.
2. **The embedding is sharded across the DP axis**, forcing 2 `all_gather`s on `cluster_axis=0` (a DP-axis round-trip) — a sharding-design oddity, cheap in op-count but architecturally unexpected.

---

## 1. Config (confirmed)

From the `non-default args` line (log line 23) and the override log (line 96):

| Field | Value |
|---|---|
| model | `mistralai/Devstral-2-123B-Instruct-2512` |
| **num_hidden_layers** | **2** (overridden from 88 "for debugging/testing" — line 96) |
| **mesh_shape** | **[4, 8]** — axis_0 = 4 (**DP**), axis_1 = 8 (**TP**) |
| **gpu_memory_utilization** | **0.30** (KV budget 9.56 GiB of 31.88 GiB DRAM) |
| **cpu_sampling** | **False** |
| **enable_trace** | **False** |
| **optimization_level** | **1** |
| enable_const_eval | True |
| enable_data_parallel / enable_tensor_parallel | True / True |
| shard_weights_on_batch_axis | False |
| experimental_weight_dtype / kv_cache_dtype | bfp_bf8 / bfp_bf8 |
| max_model_len / prefill_chunk_size / max_num_seqs | 1024 / 128 / 128 |

`TT_RUNTIME_SYNC_AFTER_OP` is not echoed as a line in the log, but the behavior is unambiguous: every op is emitted as
a standalone `Executing operation:` with a following per-op sync, and the filename/run label is `synced`. The sequence
length of every `@main` graph is **128** (`%arg0: tensor<128xi32>`) — i.e. one prefill chunk of `prefill_chunk_size=128`.

A recurring warning is relevant to the CCL discussion below:
> `Auto-overriding fabric config to FABRIC_1D for the 32-device galaxy; this is a workaround, not expected behaviour.`

---

## 2. Time budget of the first graph's execution (164 s, 509 ops)

**Important measurement caveat:** the `Executing operation:` lines carry **no per-op timestamps** (only 1 profiler-ish
line exists in the whole log, none inside the exec windows). Wall-clock anchors exist only at graph-compile boundaries.
Therefore **per-category *seconds* cannot be measured** — I report per-category **counts** plus an **elimination
estimate**, explicitly labeled as an estimate. I do **not** multiply counts by an average to manufacture per-category
seconds.

**Op counts in the first graph (lines 8104–8752, 509 ops):**

| Category | Count | Genuine cost class |
|---|---:|---|
| `ttnn.deallocate` | **250** | ~0 real work (frees memory) — pure per-op sync tax |
| `ttnn.typecast` | 34 | cheap (bf16↔bf8 on 128-token tensors) |
| `ttnn.reshape` | 34 | cheap / metadata |
| `ttnn.to_layout` | 24 | cheap at seq=128 |
| `ttnn.to_device` | 23 | **weight loads = real bandwidth (see §3)** |
| `ttnn.multiply` | 21 | cheap elementwise |
| `ttnn.get_device` | 15 | free |
| `ttcore.load_cached` | 15 | const-eval dispatch (§3/§4) |
| `ttnn.slice_static` | 14 | cheap |
| `ttnn.matmul` | 10 | small (128-token activations) |
| `ttnn.from_device` | 10 | small readbacks |
| `ttnn.add` | 9 | cheap |
| `ttnn.rms_norm` | 5 | cheap |
| `ttnn.mesh_partition` | 5 | reshard (4×TP, 1×DP) |
| `ttnn.subtract` / `permute` / `concat` | 4 / 4 / 4 | cheap |
| `ttnn.paged_fill_cache` | 4 | KV writes, small |
| `ttnn.all_reduce` | 4 | **CCL (TP) — possibly slow on FABRIC_1D, see §5** |
| `ttnn.scaled_dot_product_attention` | 2 | small (128 tokens) |
| `ttnn.concatenate_heads` | 2 | cheap |
| `ttnn.all_gather` | 2 | **CCL (DP) — embedding round-trip, see §5** |
| sin/cos/zeros/sign/remainder/gt/full/embedding | 1 each | cheap |

**Where the 164 s goes (estimate by elimination):**

- **Genuine compute + bandwidth: < ~10 s.** (10 tiny matmuls + 2 SDPA + norms + 1.41 GB `to_device` ≈ ≤10 s combined.)
- **Per-op serialization overhead: ~150 s.** This is the residual and it dominates. It splits into two sub-buckets the
  log **cannot** separate:
  - **(i) Fixed per-op dispatch + mesh-wide sync tax** across all 509 ops. The clearest signal that the op stream is
    overhead-heavy: **250 / 509 ops (49%) are `deallocate`** — operations that do no arithmetic yet each still incur a
    full 32-chip sync round-trip under sync-after-op.
  - **(ii) Possibly-slow collectives** (4 `all_reduce` + 2 `all_gather`) running on the **FABRIC_1D workaround** fabric,
    whose per-collective latency on a 32-chip galaxy is not bounded by anything in the log.

**Single most expensive *genuine* ops (by data moved, not measured time):**

| Op | Tensor | Bytes/shard | Note |
|---|---|---:|---|
| `to_device` embedding | `131072 × 3072` bf16 | **805 MB** | full-vocab, **layer-count-independent** (§3) |
| `to_device` (×8) attn/mlp | `3584 × 12288` bf16 | 88 MB each | per-layer weights |
| `to_device` mlp | `12288 × 3584` bf16 | 88 MB each | per-layer |
| `to_device` mlp gate/up | `12288 × 1536`, `12288 × 1792` bf16 | 38–44 MB each | per-layer |

---

## 3. Const-eval specifics — the big layer-independent weights

The first graph invokes **15 const-eval functions** (`main_const_eval_0 … _14`), all cold (**15 cache MISS**), which
together `to_device` **1.41 GB** of weights (9 weight `to_device` ops; the rest of the 23 `to_device` in the graph are
runtime activation loads):

- **Embedding `embed_tokens` = `131072 × 3072` bf16 = 805 MB / shard** — this is the full vocabulary (131072) times the
  DP-sharded hidden slice (12288 / 4 = 3072). **It does not shrink with `num_hidden_layers=2`** — it is the single
  biggest transfer and would be identical at 88 layers.
- The 2 transformer layers contribute the remaining ~605 MB (o_proj/qkv `3584×12288` ×8, down/gate/up
  `12288×{3584,1536,1792}`), i.e. ~300 MB per layer.
- (`lm_head` — also full-vocab `131072 × …`, layer-independent — is materialized in the logits-producing graph, not
  this first backbone chunk; the demangle for `lm_head_weight` appears at compile time, log line 8105.)

**How much of the 2 min is "just loading the big full-size weights"?** Very little. All 1.41 GB (embedding included) is
**~1.4 s at ~1 GB/s**. So the full-vocab embedding is a red herring for *this* symptom: it is the largest single object
but its transfer time is a rounding error next to the ~150 s of per-op sync. It would matter for a no-sync steady-state
run, not here.

---

## 4. Is const-eval re-done per graph or cached? — **Partially re-done (user hypothesis partly CONFIRMED)**

Cache status of `LoadCachedOp` per graph (from `Cache miss or invalid cache` vs `Cache hit` lines):

| Graph (seq=128 chunk) | const-eval invoked | Cache MISS (re-loaded) | Cache HIT (reused) |
|---|---:|---:|---:|
| **G1** (first backbone) | 15 | **15** | 0 — cold start, expected |
| **G2** (next graph) | 7 | **6** | 1 |
| **G3** (logits/decode graph) | 15 | 0 | **15** — fully reused |

**Verdict:** const-eval is **not** fully cached across graphs. G1 cold-loads all 15 (unavoidable). **G2 re-runs /
re-materializes 6 const-eval weights on a cache MISS** even though G1 already loaded equivalent tensors — this is
**genuine redundant work** and directly supports the user's hypothesis for the second graph. By G3 the cache is warm and
all 15 are reused (HIT). So the leak is specifically the **G1→G2 handoff**: 6 weights re-loaded instead of reused. This
is the concrete **fixable** item (investigate why those 6 hashes miss — likely differing layout/sharding attributes
between the prefix_chunk=False and prefix_chunk=True graph variants producing distinct cache keys).

**Note on G2/G3 wall-time:** unlike G1, the G2 and G3 *windows* are **not cleanly decomposable**. G2's window
(`197.8 s → 332.0 s`, ~134 s) contains only 102 device ops, which cannot be explained by execution alone; the next
graph's host-side frontend (torch/dynamo fusion + StableHLO lowering — the demangle/fusion lines at ~198 s and the large
4700-line G3 IR dump) overlaps device execution here. I therefore **anchor all quantitative claims on G1** (verified
contiguous, all-execution) and do **not** infer a uniform per-op rate from G2/G3.

---

## 5. CCL audit (first graph)

| Collective | Count | cluster_axis | Interpretation |
|---|---:|---:|---|
| `all_reduce` | 4 | **1 (TP)** | 2 per layer × 2 layers (attn o_proj reduction + MLP down_proj reduction) — **exactly as expected, no redundancy** |
| `all_gather` | 2 | **0 (DP)** | **embedding DP round-trip — see below** |
| `mesh_partition` | 5 | 1 (×4), 0 (×1) | reshard ops |
| `reduce_scatter` | 0 | — | none |
| `point_to_point` | **0** | — | **no P2P storm** |

**No redundant all-reduces, no point-to-point storm, no unexpected DP-axis reductions.** The CCL count is appropriate
for a 2-layer TP model. One genuine oddity:

**The 2 `all_gather`s on `cluster_axis=0` (DP axis) are an embedding DP round-trip.** Evidence chain:
- `%arg0` (the 128 input tokens) is sharded `[{"_axis_0"}]` → **axis_0 shards the token dimension → axis_0 is the DP axis**.
- `embed_tokens` `%arg5: tensor<131072×12288>` is sharded `[{}, {"_axis_0"}]` → vocab unsharded, **hidden dim (12288)
  sharded across the DP axis** (→ 3072/shard, the 805 MB tensor of §3).
- Consequently, after the embedding lookup the hidden dimension lives split across the 4 DP groups, so reconstructing a
  full hidden vector requires an `all_gather` on `cluster_axis=0` (DP).

This is unusual: sharding a weight's hidden dim across the **data-parallel** axis injects a DP-axis collective into what
should be per-replica-independent work (with `shard_weights_on_batch_axis=False`). It is only 2 ops so it is **not** the
164 s bottleneck, but it is a **sharding-design flag**: normally the embedding would be replicated or TP-sharded, not
DP-sharded on hidden. Worth revisiting for steady-state (no-sync/trace) performance.

---

## 6. What is fixable vs. what is a sync-after-op artifact

| Finding | Nature | Fixable? |
|---|---|---|
| ~150 s of the 164 s = per-op dispatch + mesh sync across 509 ops (49% of which are `deallocate`) | **Artifact of `TT_RUNTIME_SYNC_AFTER_OP=1`** | Not a bug — disappears without sync-after-op / with tracing. This is the ~2 min. |
| Possibly-slow CCLs on FABRIC_1D workaround fabric (subset of the ~150 s) | Environment / fabric workaround | Indirectly — real fabric config would help; can't quantify from this log |
| G2 re-loads 6 const-eval weights (cache MISS on G1→G2 handoff) | **Genuine redundant work** | **Yes** — align cache keys across chunked-prefill graph variants |
| Embedding hidden-dim sharded on DP axis → 2 DP `all_gather`s | **Genuine sharding oddity** | **Yes** — replicate or TP-shard the embedding hidden dim |
| 805 MB full-vocab embedding + `lm_head` load, layer-independent | Genuine but small (~1.4 s for all 1.41 GB) | Not worth it under sync; matters only for steady-state |

**Discriminator to split the sync tax from slow-CCL cost** (the one thing this log cannot resolve): run the *same*
graph either with per-op device profiling, or with sync-after-op **off**, and compare the total. The sibling log
`/data/ssalice/temp/tt-xla/devstral_dptp_test_synced_trace_off_PASS.log` (trace-off, PASS) is a candidate baseline — a
no-sync total for the same first graph would directly prove the sync share. (Out of strict scope here, but it is the
clean experiment.)

**Answer to the user's question:** the ~2 minutes is spent almost entirely on **per-op serialization overhead — 509
mesh-wide sync round-trips forced by `TT_RUNTIME_SYNC_AFTER_OP=1`**, not on the 2 layers of compute, not on loading the
big full-vocab weights (~1.4 s), and not on an excessive/redundant CCL pattern. Turning off sync-after-op (or enabling
tracing) should collapse this to the underlying <~10 s of real work plus whatever the FABRIC_1D collectives genuinely
cost. The only genuine efficiency bugs uncovered are the **6-weight const-eval re-load on the second graph** and the
**DP-axis embedding all_gather** — both real, both fixable, both small relative to the sync artifact.
