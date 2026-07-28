# Layer-by-layer sharding map — Devstral-2-123B, `test_dptp_devstral[mesh_shape0-True-bfp_bf8]`

> Theoretical, sequential (matmul → CCL) view of ONE forward pass under the exact
> test config. Read-only analysis — no code changed, nothing run on hardware.
> Builds on `../chunked_prefill_issue/sharding_analysis.md` (the "is the TP
> all-reduce fundamental" question) by giving the per-op execution-order view the
> user asked for. Evidence tags: **[CODE]** read from source (file cited);
> **[IR]** anchored to the ground-truth TTNN shapes supplied in the task;
> **[CFG]** from the HF `config.json`; **[FLAG]** divergence / unverifiable.

---

## 0. Config, dims, mesh (the load-bearing facts)

**Model dims — [CFG]** (`~/.cache/huggingface/.../Devstral-2-123B-Instruct-2512/config.json`,
`architectures=["Ministral3ForCausalLM"]`, dense decoder, GQA, YaRN RoPE, RMSNorm):

| dim | value | notes |
|---|---|---|
| `hidden_size` | **12288** | matches ground-truth IR hidden [IR] |
| `num_attention_heads` | **96** | q_proj out = 96×128 = 12288 (= hidden, coincidence) |
| `num_key_value_heads` | **8** | GQA; `num_queries_per_kv = 96/8 = 12` |
| `head_dim` | **128** | TT requires head_size % 32 == 0 → ok |
| `intermediate_size` | **28672** | MLP inner |
| `vocab_size` | **131072** | |
| `num_hidden_layers` | 88 (prod) / **2** (test, `num_hidden_layers:2`) | structure repeats per layer |
| `rms_norm_eps` | 1e-5 | | 
| `sliding_window` | null | full attention every layer |

**Mesh — [CODE]** `model_runner.py:314` `xs.Mesh(ids, mesh_shape, ("batch","model"))`,
`mesh_shape=[4,8]`:
- **axis 0 = `"batch"` = DP = 4** → `cluster_axis=0`. Batch 128 ⇒ **32 seqs / DP replica** (`dp_size=mesh_shape[0]=4`, `model_runner.py:322`).
- **axis 1 = `"model"` = TP = 8** → `cluster_axis=1`. Shards `hidden 12288 → 1536`/device [IR].

**THE load-bearing config fact — `shard_weights_on_batch_axis=False`** (test line 359,
threaded into every partition fn as `batch_axis=None`). Weights are **TP-sharded and
DP-replicated** — *not* FSDP-sharded on the batch axis. Consequences, used throughout:
- Every per-layer weight spec that would carry a `batch` axis instead carries `None`
  there: QKV/merged-column → `("model", None)`; row-parallel → `(None, "model")`.
- Each DP replica holds a full copy of the (TP-sharded) weights and runs its own 32
  seqs end to end. **⇒ there is ZERO `cluster_axis=0` (DP) collective anywhere inside
  the decoder backbone.** DP replicas are embarrassingly parallel across the whole
  stack. The only DP-axis CCLs live at the model *entry* (embedding, now removed) and
  *exit* (sampling), analyzed in §1 and §6.

**Prefill token geometry (chunked, this config).** `prefill_chunk_size=128`,
`min_context_len=32`, `max_model_len=1024` (`%256==0` ⇒ chunked-SDPA path armed,
`_chunked_sdpa_active`, `model_runner.py:432`). Per DP replica a prefill step carries
**32 users × 128 chunk tokens = 4096 tokens** — exactly the ground-truth o_proj
`[4096 × 1536]` per-device matmul [IR]. Below I give per-device activation shapes as
`[users=32, tok, feat]`; the matmul sees them flattened to `[32·tok, feat]`
(`[4096, feat]` at a full 128-token chunk).

**Local vs global matmul (definitions used below):**
- **LOCAL** = each device computes its full output shard with no cross-device
  contraction. Pattern: *replicated-along-contraction input × output-sharded weight*
  (Megatron column-parallel). No CCL needed before it, none after it to correct the
  math.
- **GLOBAL / partial** = the contraction dim is TP-sharded, so each device yields a
  *partial sum*. Pattern: *contraction-sharded input × input-sharded weight* (Megatron
  row-parallel). Requires a **reduction across the TP axis** afterward.

---

## 1. Embedding (model entry) — gather op, not a matmul

**[CODE]** `partition_vocab_parallel_embedding` (`vllm_distributed_utils.py:339-357`).

- **Weight** `VocabParallelEmbedding.weight` `[vocab, hidden] = [131072, 12288]`, spec
  **`(None, "model")`** → per device `[131072, 1536]`. **Vocab replicated; hidden
  TP-sharded.**
- **Input** `input_ids` `[32, tok]` (DP-batch-sharded, `("batch", None)`,
  `_pin_input_shardings` `model_runner.py:1774`).
- **Op**: row-gather (embedding lookup), not a matmul. Output lands hidden-sharded:
  `[32, tok, 1536]`/device.
- **Output hook** `("batch", None, None)` (line 354) forces hidden → replicated:
  ⇒ **all_gather, cluster_axis=1 (TP), hidden dim, 1536 → 12288.** Result per device
  `[32, tok, 12288]`, DP-batch-sharded, hidden replicated. **[IR-consistent]**

**[FLAG — textbook divergence #1].** Textbook `VocabParallelEmbedding` shards the
*vocab* dim, does a masked lookup, and finishes with an **all_reduce**. Here vocab is
deliberately *replicated* and *hidden* is sharded instead, so the collective is an
**all_gather on a different axis/dim, not a reduce** — chosen to dodge the vocab-shard
`CollectivePermute` that tt-mlir can't lower (#3370, comment lines 343-346).

**[CODE — DP round-trip removed]** The hook was `(None, None, None)` (full replicate),
which added a spurious **DP all_gather (cluster_axis=0) 32→128 + mesh_partition
128→32** every forward. Changing it to `("batch", None, None)` keeps the necessary TP
hidden gather and deletes the DP round-trip (lines 347-353). ⇒ **no DP CCL at entry.**

---

## 2. Decoder layer — sequential per-op map (repeats ×`num_hidden_layers`)

Residual stream entering the layer: **`[32, tok, 12288]`/device — DP-batch-sharded
(axis 0), hidden replicated across TP.** Call this the "**full-hidden replicated**"
state; every column-parallel matmul consumes it and every reduction restores it.

### 2a. Input RMSNorm
- Weight `[12288]` replicated. Normalizes over hidden, which is fully present on each
  device. **Local, no CCL.** Output = full-hidden replicated `[32, tok, 12288]`.

### 2b. Attention — q_proj / k_proj / v_proj (column-parallel) — **LOCAL**
**[CODE]** `XlaQKVParallelLinear._shard_weight` 197-204 (split into 3 separate
`F.linear`s; `assert tp_size==1` — TP is done by SPMD, not vLLM's own TP).

| proj | weight `[out, in]` | spec | per-device weight | per-device output |
|---|---|---|---|---|
| q_proj | `[12288, 12288]` | `("model", None)` | `[1536, 12288]` = 12 heads×128 | `[32, tok, 1536]` (12 q heads) |
| k_proj | `[1024, 12288]` | `("model", None)` | `[128, 12288]` = 1 kv head×128 | `[32, tok, 128]` (1 kv head) |
| v_proj | `[1024, 12288]` | `("model", None)` | `[128, 12288]` = 1 kv head×128 | `[32, tok, 128]` (1 kv head) |

- **Local** (replicated full-hidden input × output/head-sharded weight). Contiguous
  dim-0 slicing puts **q heads `[12d, 12d+12)` and kv head `d` on device `d`** — the
  GQA group `h → h//12` is fully device-local.
- **CCL after: NONE.** Head-sharded q/k/v is exactly what SDPA wants. This is the
  column half of the Megatron pair — the pairing needs a CCL only after the *row*
  op (o_proj, §2f).

### 2c. RoPE (YaRN)
- Per-head elementwise rotation on q and (its 1) k head. **Local, no CCL.**

### 2d. KV-cache write (paged_fill_cache / paged_update_cache)
**[CODE]** `attention.py:_handle_paged_attention` 461-515.
- Prefill: `paged_fill_cache(k_cache, key, fill_page_table, batch_idx)`. `batch_idx`
  built on CPU, rebased `% local_batch` when `dp_size>1` (line 497-499) so global batch
  ids map to per-replica local slots.
- Cache layout `[num_blocks, num_kv_heads=8, block_size=32, head_dim=128]`
  (`get_kv_cache_shape` 104-112).
- **page_table / cache_position / batch_idx**: `safe_mark_sharding(page_table, mesh,
  ("batch", None))` — **DP-batch-sharded (cluster_axis=0)** (`model_runner.py:1455,
  2302, 2705`). Each replica indexes only its own 32 seqs' blocks.
- **KV cache tensor: un-annotated ⇒ replicated across all 32 chips** in DP+TP
  (`model_runner.py:3446-3451`), *unlike* the pure-TP branch which head-shards
  `(None, "model", None, None)` (line 3461).

**[FLAG — divergence #2, unverifiable from source].** The written K/V are TP-head-
sharded (1 kv head/device from k/v_proj), yet the cache is nominally *replicated*.
Whether GSPMD (i) emits an implicit **TP all_gather** to fill a truly-replicated cache,
or (ii) lets each device write its own head-slice into its own copy (no CCL, replicas
diverge by design), **cannot be determined from source** — the code comment only says
"each device writes its own K/V slice … the TP-only spec puts block_size on the DP axis
and fails `ttir.paged_update_cache`. Tracked as a follow-up." **Verify in exported IR
before asserting a CCL here.** No explicit CCL is marked; the current replication is
the correct-but-memory-heavy fallback (sharding_analysis §3.3b).

### 2e. SDPA — chunked or full (paged) — no matmul weight, no CCL
**[CODE]** `_compute_full_attention` 517-598.
- Inputs per device: q `[32, 12 heads, tok, 128]`, and its 1 kv head. GQA
  `num_queries_per_kv=12` — **the whole attention group is local to the device**, so
  **no KV replication or gather across TP is required** (this is the clean consequence
  of `num_kv_heads(8) == TP(8)`).
- **Standard prefill / decode**: `scaled_dot_product_attention` / `paged_..._decode`.
- **Chunked-prefix path (the extra graph)**: when `chunk_start_idx is not None and
  has_paged_cache and not shared_kv_mode` (line 545), calls
  `chunked_scaled_dot_product_attention(q, k_cache, v_cache, page_table,
  chunk_start_idx)` — the current chunk attends over the already-cached prefix in the
  paged buffer, mask+offset internal (no host mask, no dense gather). Because the
  trigger is Python-level, it **traces as its own distinct graph** (`prefix_chunk=True`,
  precompiled alongside `prefix_chunk=False`, `model_runner.py:2563-2592`). Sharding of
  q/k/v/page_table is identical to standard prefill; **no new CCL** — the second graph
  just changes the SDPA op + reads the prefix from the (DP-sharded page_table into the
  replicated) cache.
- Output per device `[32, tok, 1536]` (12 heads). **No CCL.**

### 2f. o_proj (row-parallel) — **GLOBAL / partial** → reduction #1
**[CODE]** `partition_row_parallel_linear` 302-309, spec `(batch_axis, "model")` =
**`(None, "model")`**.
- **Weight** `[hidden, n_heads·head_dim] = [12288, 12288]` → per device `[12288, 1536]`
  (out replicated, **in-dim TP-sharded**).
- **Input** = SDPA output `[32, tok, 1536]`/device — **contraction dim (1536)
  TP-sharded.**
- **Global**: each device computes a **partial** `[32, tok, 12288]` over its 1536 slice
  (IR: `matmul([4096×1536], [12288×1536]^T) → [4096×12288]`).
- **CCL after: all_reduce over TP (cluster_axis=1), hidden dim** — the fundamental
  Megatron reduction. **Decomposed [IR] to `reduce_scatter(cluster_axis=1,
  scatter_dim=hidden) → [.,.,1536]` + paired `all_gather(cluster_axis=1) → [.,.,12288]`**
  (the fused `ttnn.all_reduce` is avoided; it hangs `end_trace_capture` on galaxy —
  sharding_analysis §3.1). **Why**: the next ops (residual add, post-attn RMSNorm) need
  the complete hidden vector.

### 2g. Residual add + post-attention RMSNorm
- Add (both operands full-hidden replicated) + RMSNorm over hidden. **Local, no CCL.**
  Output = full-hidden replicated `[32, tok, 12288]`.

### 2h. MLP gate_proj / up_proj (column-parallel, merged) — **LOCAL**
**[CODE]** `XlaMergedColumnParallelLinear._shard_weight` 107-111, spec
`("model", batch_axis)` = **`("model", None)`** (two separate `F.linear`s, 148-161).

| proj | weight `[out, in]` | per-device weight | per-device output |
|---|---|---|---|
| gate_proj | `[28672, 12288]` | `[3584, 12288]` | `[32, tok, 3584]` |
| up_proj | `[28672, 12288]` | `[3584, 12288]` | `[32, tok, 3584]` |

- **Local** (replicated full-hidden input × output-sharded weight). **CCL after: NONE**
  (column half of the MLP pair).

### 2i. Activation (SiLU) + gate·up
- Elementwise on the TP-sharded intermediate. **Local, no CCL.** Output `[32, tok, 3584]`.

### 2j. down_proj (row-parallel) — **GLOBAL / partial** → reduction #2
**[CODE]** same `partition_row_parallel_linear`, spec **`(None, "model")`**.
- **Weight** `[hidden, intermediate] = [12288, 28672]` → per device `[12288, 3584]`
  (out replicated, **in-dim 28672 TP-sharded**).
- **Input** `[32, tok, 3584]`/device — contraction dim TP-sharded.
- **Global**: partial `[32, tok, 12288]`.
- **CCL after: all_reduce over TP (cluster_axis=1), hidden dim** — decomposed to
  `reduce_scatter + all_gather` [IR]. **Why**: the residual add + next layer's input
  RMSNorm need full hidden.

### 2k. Residual add → next layer
- Local. Output = full-hidden replicated `[32, tok, 12288]` → feeds §2a of the next
  layer.

---

## 3. Final RMSNorm
- Over full hidden (replicated). **Local, no CCL.** Output `[32, tok, 12288]`/device.

## 4. lm_head (ParallelLMHead, column-parallel) — **LOCAL matmul, then gather**
**[CODE]** `partition_parallel_lm_head` 325-331, spec **`("model", None)`**;
`compute_logits` 3509-3517.
- Precede: `select_hidden_states` 3495-3500 gathers the last token/seq →
  `[num_reqs, hidden]`, with `sharding_constraint_tensor(result, mesh, (None, None))`
  under 2D mesh (line 3499) — see §6 for the DP CCL this triggers.
- **Weight** `[vocab, hidden] = [131072, 12288]` → per device `[16384, 12288]`
  (**vocab TP-sharded**, hidden replicated).
- **Input** `[num_reqs, 12288]` full-hidden replicated. **Local matmul** (column-
  parallel): per device `[num_reqs, 16384]` (vocab-sharded).
- **CCL after**: `is_sharded_compute_logits=True` (`model_runner.py:2217`, TP on +
  ParallelLMHead present) ⇒ `sharding_constraint_tensor(logits, mesh, (None, None))`
  (line 3516) forces logits fully replicated ⇒ **all_gather over TP (cluster_axis=1),
  vocab dim, 16384 → 131072.**

## 5. Sampling
- `cpu_sampling=True` (test line 373; required, #4387/#4440). Logits `.cpu()`
  (`model_runner.py:1999`) then sampled on host. **No device CCL** in the sampler
  itself; the postprocess runs as its own compiled graph.

---

## 6. DP-axis (cluster_axis=0) behavior — summary

- **Inside the backbone: no DP CCL at all.** `shard_weights_on_batch_axis=False`
  ⇒ weights DP-replicated, activations DP-batch-sharded; each replica runs its 32
  seqs independently (§0).
- **Entry (embedding): DP CCL removed.** The `(None,None,None)→("batch",None,None)`
  hook change deleted the per-forward DP all_gather(32→128)+mesh_partition(128→32)
  round-trip (§1).
- **Exit (sampling): one surviving DP CCL.** `select_hidden_states` →
  `sharding_constraint_tensor(result, mesh, (None, None))` (line 3499): the selected
  hidden `[32, hidden]` is DP-batch-sharded per replica; `(None, None)` forces it
  replicated ⇒ **DP all_gather (cluster_axis=0), 32 → 128.** Once per forward, on the
  post-backbone sampling graph — **not per layer.** (The prior doc flagged this
  `(None,None)` constraint's "DP round-trip smell" as out of scope; here it is named as
  the one backbone-external DP collective that remains.)

---

## 7. Summary table — one row per matmul, execution order

Per-device shapes; batch=32/replica, `tok`=chunk tokens (128 at full chunk ⇒ 4096
flattened). "CCL after" = the collective needed to make the input correct for the NEXT
op. Reductions shown as their decomposed `reduce_scatter+all_gather` (fused all_reduce
avoided).

| # | op | local/global | input shard (per-dev) | weight shard (per-dev) | output shard (per-dev) | CCL after (type / axis / dim) |
|---|---|---|---|---|---|---|
| — | embedding (lookup) | — (gather) | `input_ids [32,tok]` DP-batch | `[131072,1536]` `(None,"model")` | `[32,tok,1536]` hidden-shard | **all_gather / TP(1) / hidden 1536→12288** |
| 1 | q_proj | **local** (col) | `[32,tok,12288]` hid-replicated | `[1536,12288]` `("model",None)` | `[32,tok,1536]` 12 q-heads | none |
| 2 | k_proj | **local** (col) | `[32,tok,12288]` hid-replicated | `[128,12288]` `("model",None)` | `[32,tok,128]` 1 kv-head | none |
| 3 | v_proj | **local** (col) | `[32,tok,12288]` hid-replicated | `[128,12288]` `("model",None)` | `[32,tok,128]` 1 kv-head | none (SDPA is device-local; KV write cache **[FLAG §2d]**) |
| 4 | o_proj | **global** (row) | `[32,tok,1536]` head/TP-shard | `[12288,1536]` `(None,"model")` | `[32,tok,12288]` **partial** | **all_reduce / TP(1) / hidden** = reduce_scatter(→1536)+all_gather(→12288) |
| 5 | gate_proj | **local** (col) | `[32,tok,12288]` hid-replicated | `[3584,12288]` `("model",None)` | `[32,tok,3584]` inter/TP-shard | none |
| 6 | up_proj | **local** (col) | `[32,tok,12288]` hid-replicated | `[3584,12288]` `("model",None)` | `[32,tok,3584]` inter/TP-shard | none |
| 7 | down_proj | **global** (row) | `[32,tok,3584]` inter/TP-shard | `[12288,3584]` `(None,"model")` | `[32,tok,12288]` **partial** | **all_reduce / TP(1) / hidden** = reduce_scatter(→1536)+all_gather(→12288) |
| 8 | lm_head | **local** (col) | `[num_reqs,12288]` hid-replicated | `[16384,12288]` `("model",None)` | `[num_reqs,16384]` vocab/TP-shard | **all_gather / TP(1) / vocab 16384→131072** (after a DP all_gather 32→128 in `select_hidden_states`) |

Rows 1-7 repeat per decoder layer; embedding/lm_head are once per forward.

---

## 8. Theoretical minimum CCL count

**Per decoder layer: exactly 2 reductions — the 2 Megatron row-parallel reductions.**
- 1 after **o_proj** (cluster_axis=1, hidden) — attention column→row pair.
- 1 after **down_proj** (cluster_axis=1, hidden) — MLP column→row pair.
- Decomposed, that is **2 reduce_scatter + 2 all_gather per layer** (fused all_reduce
  avoided for the trace hang; identity `all_reduce ≡ reduce_scatter + all_gather`).
- The 4 column-parallel matmuls (q/k/v/gate/up) and both RMSNorms/RoPE/SDPA need **no**
  CCL. This is the theoretical minimum for dense Megatron TP — the reduction is the math
  of a TP-sharded contraction and cannot be sharded away (sharding_analysis §3.1).

**Per forward, model-level (once):**
- +1 TP all_gather at embedding (hidden 1536→12288).
- +1 DP all_gather (32→128) + 1 TP all_gather (vocab 16384→131072) at the sampling head.
- 0 DP collectives inside the backbone.

**Clean arithmetic (this 2-layer test):** backbone = 2 layers × 2 reductions ×
(1 RS + 1 AG) = **4 RS + 4 AG**; plus embedding AG + lm_head AG = **4 RS + 6 AG per
compiled graph variant**.

**[FLAG — reconciliation with the MEASURED 8 RS / 12 AG]** The prior report measured
`reduce_scatter=8, all_gather=12` (fused `all_reduce=0`). That is **exactly 2×** the
clean per-graph count above. The likely explanation is that the export aggregates
collectives across the **two backbone graph variants this config compiles** — standard
prefill (`prefix_chunk=False`) and the chunked-prefix graph (`prefix_chunk=True`), the
extra graph the chunked-SDPA path adds (§2e). *This 2× is a plausible reconciliation,
not a certainty* — confirm which graphs the IR export actually covered before treating
it as fact. (Do **not** adopt the prior doc's "4 reductions → 8 RS" phrasing: it
implies 2 RS per reduction, which contradicts the 1 RS + 1 AG identity.)

---

## 9. Divergences from textbook Megatron (collected)

1. **Embedding** (§1): vocab *replicated*, hidden *sharded* → finishes with an
   **all_gather (TP, hidden)**, not the textbook vocab-sharded masked-lookup
   **all_reduce**. Motivated by dodging the vocab-shard CollectivePermute (#3370).
2. **KV cache in DP+TP** (§2d): **un-annotated/replicated** while the pure-TP branch
   head-shards `(None,"model",None,None)`. K/V writes are head-sharded but the cache is
   nominally replicated — the implied fill collective (if any) is **unverifiable from
   source**; blocked on `ttir.paged_update_cache`. Verify in IR.
3. **Weights DP-replicated, not FSDP** (§0): `shard_weights_on_batch_axis=False` — a
   deliberate choice that makes the backbone DP-collective-free at the cost of
   replicating the TP-sharded weights across the 4 DP replicas.
4. **Chunked prefill** (§2e): adds a **second traced backbone graph**
   (`prefix_chunk=True`) with identical sharding but the `chunked_scaled_dot_product_
   attention` op — the most likely source of the 2× measured-collective count (§8).
