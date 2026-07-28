# Paged-attention sharding map — Devstral-2-123B, `test_dptp_devstral[mesh_shape0-True-bfp_bf8]`

> **Read-only theoretical map.** No code changed, nothing run on hardware. All
> shapes are the *logical/global* shapes that appear in the traced XLA graph;
> per-device shapes are the global shape divided by the sharded mesh-axis size.
> Line numbers are from the current working tree (incl. uncommitted edits).

## 0. Config under analysis

Parametrize id decode (`test_data_tensor_parallel_generation.py:296-312`):

| id token | binds to | value |
|---|---|---|
| `mesh_shape0` | `mesh_shape` | `[4, 8]` |
| `True` | `enable_const_eval` | `True` |
| `bfp_bf8` | `experimental_weight_dtype` | `bfp_bf8` (**weight** dtype) |

The **KV-cache** dtype is set separately and unconditionally in the same test:
`experimental_kv_cache_dtype="bfp_bf8"` (line 361). `prefill_chunk_size=128`
(line 365) is the single switch that turns chunked prefill on; `enable_trace=True`
(364); `num_hidden_layers=2` bring-up truncation.

**Mesh / parallelism** (`model_runner.py:310-324`):
`xs.Mesh(range(32), [4, 8], ("batch", "model"))`.

| Mesh axis | index | name | role | `cluster_axis` |
|---|---|---|---|---|
| 0 | `[4,…]` | `"batch"` | **DP = 4** replicas | 0 |
| 1 | `[…,8]` | `"model"` | **TP = 8** (Megatron) | 1 |

`parallel_mode = DATA_TENSOR_PARALLEL`; `dp_size = mesh_shape[0] = 4`
(3 22); `use_2d_mesh = (1 not in mesh_shape) = True` (323).

**Derived scalars for this config**

| Quantity | Value | Source |
|---|---|---|
| Global batch (`max_num_reqs`) | **128** (already `%4==0`) | test knobs; DP round-up 327-336 |
| `dp_size` | 4 | 322 |
| `local_batch` = batch / dp_size | **32** seqs/replica | derived; used in-graph at `attention.py:498` |
| `block_size` | **32** (forced) | `platform.py` `block_size=32`; `attention.py:152` `get_page_size` returns 32 |
| `max_model_len` | **1024** (`TT_DEVSTRAL_MAX_MODEL_LEN` default) | test env |
| `max_num_blocks_per_req` = cdiv(1024,32) | **32** | `model_runner.py:389` |
| `max_num_blocks_per_req % 8` | `32 % 8 == 0` ✔ | gate 433 |
| `prefill_chunk_budget` = min(128, 1024) | **128** | 421-428 |
| `_chunked_sdpa_active` = (128 < 1024) ∧ (32%8==0) | **True** | 432-434 |
| `num_kv_heads` (GQA) | **8**† | `attn_module.num_kv_heads` (symbolic) |
| `head_size` | **128**† | `attn_module.head_size` (symbolic) |
| `num_query_heads` | ~96† | symbolic |

† **Not verifiable from the runner** — it reads `attn_module.num_kv_heads` /
`head_size` symbolically from the HF config. The values above are the
Mistral-Large-lineage expectation; **confirm against the model's `config.json`**.
The only hard constraint the code enforces is `num_kv_heads % tp_size == 0`
(`model_runner.py:3414`), i.e. `8 % 8 == 0` ⇒ **1 KV head per TP device**.

---

## 1. Input tensors

### 1.1 The pinning function — `_pin_input_shardings` (`model_runner.py:1757-1784`)

In `DATA_TENSOR_PARALLEL` with `use_2d_mesh=True`, `batch_axis = "batch"`
(the DP axis). It pins:

| Tensor | Global shape | Shard spec | Per-device shape | Notes |
|---|---|---|---|---|
| `input_ids` | `[128, T]` | `("batch", None)` | `[32, T]` | 1774; `T` = padded tokens (chunk-bucketed) |
| `inputs_embeds` | `[128, T, H]` | `("batch", None, None)` | `[32, T, H]` | 1776, **only if not None** — see §1.3 |
| `position_ids` | `[128, T]` | `("batch", None)` | `[32, T]` | 1784, DP-modes only |

`T` (`padded_total_num_scheduled_tokens`) is bucketed from the token-padding
ladder (`num_tokens_paddings`, 452-465); for a prefill chunk it caps at
`prefill_chunk_budget = 128`. `H` is the full hidden dim (replicated, not TP-split
at the input boundary).

**Why batch (DP) sharded:** each of the 4 DP replicas owns a disjoint block of
32 sequences of the 128-wide batch. Pinning `("batch", None[, None])` makes
GSPMD keep each replica's own 32 rows local and prevents recompilation drift
between warm-up and inference (the docstring's stated purpose: "both paths
produce identical graphs"). No mesh-axis is placed on the token/hidden dims, so
**no CCL touches the inputs** at the pin — it is a pure placement annotation.

### 1.2 `position_ids` pinned only in DP modes — the pure-TP hazard it dodges

`position_ids` is deliberately *not* pinned under pure-TP (guard 1780-1783):
under pure-TP, pinning it "perturbs GSPMD into a batch-axis
`reduce_scatter → ttnn.rms_norm(k_norm)` graph that hits a tt-mlir
redundant-`to_layout` bug (`TTNNDecomposeLayouts`)" (comment 1777-1779). In our
`DATA_TENSOR_PARALLEL` config the pin **is** applied (`("batch", None)`), because
the batch axis is a genuine DP axis here, not a TP axis.

### 1.3 `inputs_embeds` vs the embedding *output* activation

For text-only Devstral, the runner passes `input_ids` and computes the embedding
on-device; `inputs_embeds`-the-input is typically **None**, so the guard at 1775
skips it. The tensor that actually carries a batch shard through the embedding is
the **embedding output activation**, pinned by a forward hook — see §1.5. Do not
conflate the two.

### 1.4 `cache_position`, `batch_idx`, `num_scheduled_tokens`

Built in `_prepare_inputs` and marked in the DP branch (`model_runner.py:1438-1459`):

| Tensor | Global shape | dtype | Shard spec | Per-device | Role |
|---|---|---|---|---|---|
| `cache_position` | `[num_reqs]` (≤128) | int32 | `("batch",)` | `[32]` | decode write index = `seq_len − 1` per user (`1369`); feeds `paged_update_cache` |
| `batch_idx` | `[num_reqs]` = `arange(128)` | int32 | `("batch",)` | `[32]` = replica-r slice `[32r … 32r+31]` | user→slot map for `paged_fill_cache` (prefill) |
| `chunk_start_idx` | `[1]` | int32 | *unmarked* (scalar, replicated) | `[1]` | cached-prefix offset; see §2.3 |

`batch_idx` is a **global** `arange(max_num_reqs)` allocated once
(`batch_idx_max_reqs = torch.arange(128)`, 745-747; prefill variant 751-757),
then annotated `("batch",)`. So on DP replica *r* it physically holds
`[32r … 32r+31]`. The in-graph `% local_batch` rebase (§4.2) turns these back
into local `0…31`.

**`num_scheduled_tokens`** is **not** a sharded device tensor. It is host-side
scheduler metadata (`scheduler_output.num_scheduled_tokens`, a dict keyed by
req-id) that drives token counting, padding-bucket selection, the b1-prefill cap,
and the decode-vs-prefill branch (`1177-1238`). It never gets a `mark_sharding`
call; it only *shapes* the device tensors above. No shard spec applies.

The marking is guarded by a DP check and carries the comment (1449-1455):
*"page_table / cache_position / batch_idx must share the K/V input's [batch]
sharding … batch_idx feeds paged_fill_cache, whose verifier"* requires the
per-device dim-0 to match the (batch-sharded) key tensor. **CCL:** none — pure
placement.

### 1.5 The embedding-output DP round-trip fix — `("batch",None,None)` **[LANDED]**

`partition_vocab_parallel_embedding` (`vllm_distributed_utils.py:339-357`):

- **Weight** `[vocab, hidden]` marked `(None, "model")` — hidden on TP, **vocab
  un-sharded** (sharding vocab needs a `CollectivePermute` tt-mlir #3370 can't
  lower yet).
- **Output** activation `[128, T, H]`: a forward hook now constrains it to
  **`("batch", None, None)`** (line 354).

Prior analysis (`sharding_analysis.md` §3.3a) recorded this as `[PROPOSED]`
changing `(None,None,None)`→`("batch",None,None)`. **Divergence to flag: it is
now LANDED in the working tree** (354, with the round-trip rationale inlined).

Why: the old `(None,None,None)` forced full replication, so GSPMD did **two**
gathers — the legit TP-axis hidden all_gather (`cluster_axis=1`, 1536→12288, kept
because the first `q/k/v_proj` is column-parallel and needs full hidden per TP
device) **plus** a redundant DP-axis all_gather (32→128) immediately followed by
a `mesh_partition` back to 32 for the next batch-sharded op. The new
`("batch",None,None)` keeps each replica's 32 users local and **deletes the
DP-axis all_gather + re-partition** while preserving the TP hidden gather. This
is the one place a `cluster_axis=0` (DP) collective was *removed*.

---

## 2. `page_table` (read) and `fill_page_table` (write/suffix)

### 2.1 Shapes and dtype

Both are `(num_reqs, max_num_blocks_per_req)` int32 (dtype = `block_table_cpu.dtype`,
allocated at `model_runner.py:614-635`). For this config, `max_num_blocks_per_req = 32`.

| Tensor | Global shape | dtype | Shard spec | Per-device shape |
|---|---|---|---|---|
| `page_table` (read) | `[128, 32]` | int32 | `("batch", None)` | `[32, 32]` |
| `fill_page_table` (write) | `[128, 32]` | int32 | `("batch", None)` | `[32, 32]` |

Marked in all three prepare paths: `_prepare_inputs` (`1455`, `1459`),
`_dummy_run` (`2302`), and the chunked precompile path (`2705`). Each entry
`page_table[i]` lists the physical block ids that hold user *i*'s KV.

### 2.2 The `%8` alignment constraint on `max_num_blocks_per_req`

`_chunked_sdpa_active` requires `max_num_blocks_per_req % 8 == 0`
(`model_runner.py:432-434`) because the ttnn page-table "stick" must be
32-B-aligned. With `block_size=32` this is equivalent to
**`max_model_len % 256 == 0`** (256 = 8 × 32). If chunked prefill is opted-in but
this fails, the runner raises `NotImplementedError` (439-449) rather than
silently degrading. For `max_model_len=1024`: `1024/32 = 32`, `32 % 8 == 0` ✔,
so the chunked SDPA op is enabled.

### 2.3 Standard prefill vs cached-prefix chunked path

- **`fill_page_table` roll (suffix write).** With prefix caching, some of a
  user's leading blocks are already filled. `_prepare_inputs` clones the
  page_table and `torch.roll`s each row left by `num_computed_tokens // block_size`
  so `paged_fill_cache` writes the **suffix** blocks instead of clobbering shared
  prefix blocks (`1376-1395`). Zero-scheduled (already-prefilled, re-batched) rows
  are redirected to the null block 0 so they don't clobber earlier KV
  (`1397-1408`). When no roll is needed, `fill_page_table is page_table` (same
  tensor, single mark).
- **`chunk_start_idx`** (`1461-1475`): set to `[num_computed_tokens[0]]` **only**
  when `prefix_chunk_step` (padded tokens > 1 **and** any `num_computed > 0`)
  **and** `_chunked_sdpa_active`. Decode (L==1) and the first prefill chunk keep
  it **None**. Because the trigger is a Python-level flag, the chunked-prefix
  branch traces as a **distinct graph** (no data-dependent control flow) — the
  `prefix_chunk ∈ {False, True}` precompile buckets at `2306`/`2716`.
- **How the chunked path changes the read.** In standard first-chunk prefill,
  attention is dense SDPA over the in-flight chunk. In the cached-prefix chunk,
  `_compute_full_attention` (`attention.py:517-560`) routes to
  `chunked_scaled_dot_product_attention(q, k_cache, v_cache, page_table,
  chunk_start_idx, scale)` — it reads the prior KV straight out of the paged
  cache via `page_table`, applying the causal mask and the `chunk_start_idx`
  offset **internally** (no host attn-mask, no gather). So `page_table` is reused
  as the *read* index and `chunk_start_idx` supplies "where in the sequence this
  chunk starts." `fill_page_table` remains the *write* index for the new chunk's
  KV.

**CCL touching page_table / fill_page_table:** none directly — they are
batch-sharded placement annotations. (Prior analysis §3.3b speculated a
DP-sharded cache *might* remove a page-table reshard; that remains
**unverifiable from source** — see §4.4.)

---

## 3. Blocks / block table

### 3.1 `MultiGroupBlockTable` (host-side logical→physical map)

`InputBatch` holds a `MultiGroupBlockTable` (`input_batch.py:18, 83-90`),
one `BlockTable` per KV-cache group. It is **host/CPU state**, not a device
tensor, and carries no shard annotation. `_prepare_inputs` reads
`self.input_batch.block_table[0].get_cpu_tensor()` (`1352`, `1363`) to fill the
per-step `page_table`. `num_computed_tokens_cpu` (`input_batch.py:74-80`) tracks
how much of each request is already cached and drives the `fill_page_table` roll
(§2.3) and `chunk_start_idx`.

### 3.2 Logical→physical block mapping and `num_blocks` sizing

`_get_slot_mapping_metadata` (`model_runner.py:1101-1167`) computes, per request,
`global_block_start_idx = req_idx * max_num_blocks_per_req + local_block_start`,
gathers `block_numbers` from the flattened block table, and produces
`(kv_cache_start_index, new_kv_start_index, slice_len)` triples — i.e. logical
token ranges map to `block_number * block_size + offset` inside the flat paged
pool. `block_size = 32`.

`num_gpu_blocks` (the pool size) is derived at KV-cache init:
`num_blocks = tensor_size // kv_cache_spec.page_size_bytes` (`3394`), where
`tensor_size` comes from `TTWorker.determine_available_memory` (PJRT
`dram_size_bytes × gpu_memory_utilization`, `gpu_util=0.3` for the galaxy entry).
`page_size_bytes` is computed from the **uint8 accounting spec** (§4.2), so vLLM
budgets blocks against the BFP8 on-device footprint.

### 3.3 Is the block pool replicated ×DP? — **Yes (open item)**

The block **pool** (the KV-cache tensors, §4) is left un-annotated under
`DATA_TENSOR_PARALLEL`, i.e. **replicated on all 32 chips**. Its dim-0 is
`num_blocks` (a global block index), **not** a batch dim — so there is no natural
axis on which to carry a DP shard of the pool. This is the "physical DRAM
de-replication" open item: every device allocates a full-size pool even though
each DP replica only ever writes/reads its own 32 users' blocks (§4.3-4.4).

---

## 4. KV cache — the crux

### 4.1 Spec and physical layout

`get_kv_cache_spec` (`model_runner.py:1021-1099`) builds a **`FullAttentionSpec`**
per attention layer (dense GQA, `attn_type == DECODER`, no sliding window):
`FullAttentionSpec(block_size=32, num_kv_heads=8†, head_size=128†,
dtype=self.kv_cache_spec_dtype)`.

Physical per-layer tensors are allocated as **two separate K and V tensors**
(`model_runner.py:3428-3433`) with shape from
`TTAttentionBackend.get_kv_cache_shape` (`attention.py:104-112`):

```
(num_blocks, num_kv_heads, block_size, head_size)
          = (num_blocks, 8†, 32, 128†)
```

> **Divergence to flag — prior analysis had the middle dims swapped.**
> `sharding_analysis.md` §3.3b wrote the layout as
> `[num_blocks, block_size, num_kv_heads, head_dim]`. The **code** returns
> `(num_blocks, num_kv_heads, block_size, head_size)` — `num_kv_heads` is
> **dim 1**, `block_size` is **dim 2**. The prior analysis' *intent* ("KV-head
> dim on the TP axis") is correct; its written tuple is wrong.

### 4.2 dtype handling for `bfp_bf8` — uint8 accounting vs bf16 staged buffer

Two distinct dtypes (`model_runner.py:357-367`, `3424-3431`):

| Purpose | dtype | Where |
|---|---|---|
| **Accounting spec** (`kv_cache_spec_dtype`) | `torch.uint8` (1-byte stand-in) | 360 — makes vLLM budget blocks for the BFP8 on-device footprint |
| **Staged buffer** (`kv_cache_dtype`) | **bf16** | 3426 (`dtype = self.kv_cache_dtype`), `k_cache/v_cache = torch.zeros(shape, bf16)` |

So the block-count math uses uint8 `page_size_bytes`, but the actual staged K/V
tensors on device are bf16 (converted to BFP8 on device). `bfp_bf4` is explicitly
`NotImplementedError` (361-364, tt-xla #5011).

### 4.3 Sharding — replicated on DP, correctness via device-local page_table

`model_runner.py:3446-3465`:

| Mode | KV cache mark | Effect |
|---|---|---|
| `DATA_TENSOR_PARALLEL` (**our config**) | *none* (`pass`, 3446-3451) | **replicated on all 32 chips** |
| pure-TP (`enable_tensor_parallel`, pair) | `(None, "model", None, None)` (3460-3462) | shards **dim 1 = num_kv_heads** on TP (8→1/device) |
| pure-TP (MLA latent, single) | `(None, None, None, None)` | replicated |

Note that under the **actual** layout, the pure-TP spec `(None,"model",None,None)`
shards `num_kv_heads` — which is exactly what you want for TP (guarded by
`num_kv_heads % tp_size == 0`, 3414). Under our DP+TP config that branch is
**not** taken; the cache is left fully replicated.

**The central question — is the cache physically replicated on DP but
correctness-DP-sharded via a global `batch_idx` + in-graph `% local_batch`?
Yes. Confirmed from code, with one refinement.**

**(a) The primary correctness mechanism is the batch-sharded `page_table`, not
`batch_idx`.** Every paged op indexes the cache through the device-local
page_table:
- prefill write → `paged_fill_cache(…, fill_page_table, batch_idx=…)` (`attention.py:500-511`),
- decode write → `paged_update_cache(…, cache_position, page_table)` (`attention.py:473-484`) — **no `batch_idx` at all**,
- decode read → `paged_scaled_dot_product_attention_decode(…, page_table)` (`attention.py:658-662`),
- chunked-prefix read → `chunked_scaled_dot_product_attention(…, page_table, chunk_start_idx)` (`attention.py:551-558`).

Because `page_table`/`fill_page_table` are sharded `("batch", None)`, on DP
replica *r* their 32 rows are *only* replica *r*'s sequences, pointing at blocks
that only replica *r* writes. So each replica touches only its own users' blocks
inside its replicated copy of the pool — for **all** paged ops. The pool is
"replicated" as an *allocation*, but each physical copy holds *different*
(replica-local) data. That is what makes replication correct without a CCL.

**(b) `batch_idx % local_batch` is the prefill-write-specific adapter**
(`attention.py:490-511`):

```python
batch_idxs = attn_metadata.batch_idx          # global arange(128), sharded ("batch",)
if attn_metadata.dp_size > 1:
    local_batch = key_for_update.shape[0] // attn_metadata.dp_size   # 128 // 4 = 32
    batch_idxs = batch_idxs % local_batch       # global [32r..32r+31] -> local [0..31]
```

**Load-bearing SPMD assumption (state it explicitly):** under torch_xla GSPMD you
write **global logical** shapes and sharding is an annotation. So
`key_for_update.shape[0]` is the **global** batch **128**, not the per-device 32,
giving `local_batch = 128 // 4 = 32`. `batch_idx` is the global `arange(128)`
annotated `("batch",)`, so replica *r* physically holds `[32r … 32r+31]`; `% 32`
rebases those to local `[0 … 31]`, which lines up with the locally-shaped 32-row
`key_for_update`/`fill_page_table`. If one wrongly assumed *local* shapes, the
math looks broken (`32 // 4 = 8`) — it is correct **only** under global-shape
tracing. This is a "reconcile a global-annotated index tensor with
locally-sharded operands" pattern; the rebase is a no-op when `dp_size == 1`.

So: **prefill** uses `batch_idx % local_batch` + `fill_page_table`; **decode**
uses `cache_position` + `page_table` with no batch_idx. Both are correct because
the underlying page_table rows are device-local. Each DP replica writes and reads
only its own 32 users' KV.

**CCL touching the KV cache:** **none.** Full replication + device-local
page_table indexing means no cross-device exchange for reads or writes. The cost
is memory (full pool ×32), not communication.

### 4.4 The `ttir.paged_update_cache` limitation — what it blocks

Verbatim comment (`model_runner.py:3446-3450`):

> "DP+TP: leave the KV cache un-annotated (replicated under SPMD); each device
> writes its own K/V slice via paged_update_cache. The TP-only spec puts
> block_size on the DP axis and fails ttir.paged_update_cache. Tracked as a
> follow-up."

**Root invariant (verifiable):** the paged layout
`(num_blocks, num_kv_heads, block_size, head_size)` has **no batch/sequence
dimension** — user identity lives only in the page_table, not in the cache
tensor's axes. A DP/batch shard therefore has no natural tensor dim to land on;
it would have to fall on `num_blocks` or `block_size`, and `ttir.paged_update_cache`
(the write op, which indexes blocks via the page_table) does not support having
that indexed dim sharded across a mesh axis. Hence the pool stays **replicated**
— a compiler/runtime blocker, **not** a Shardy-annotation choice.

**Flag — do not over-read the comment.** The specific phrase "puts block_size on
the DP axis" does **not** cleanly match the pure-TP spec `(None,"model",None,None)`
under the real layout (that spec shards `num_kv_heads`, dim 1). It reads as
loosely-worded or as describing a hypothetical DP-shard attempt. The exact
"block_size on DP" mechanism is **unverifiable from source** and should not be
reverse-engineered; the reliable statement is the no-batch-axis invariant above.

**What it blocks:** physical DRAM de-replication of the block pool (each device
would hold only its `1/DP` share of blocks). Blocked today; would need a
`paged_update_cache` change in tt-metal/tt-mlir. Whether a DP-sharded pool would
*also* eliminate a `cluster_axis=0` page-table reshard is **unverifiable from
source alone** (needs exported IR for the chunked path) — flagged as
expected-but-unconfirmed, matching prior analysis.

---

## 5. Where DP-axis (`cluster_axis=0`) collectives appear or were removed

| Site | DP-axis CCL | Status |
|---|---|---|
| Embedding output activation (§1.5) | DP all_gather(32→128) + `mesh_partition`(128→32) | **removed** — hook changed `(None,None,None)`→`("batch",None,None)` (`vllm_distributed_utils.py:354`) **[LANDED]** |
| Inputs `input_ids`/`positions`/embeds (§1.1) | none | pure placement |
| `page_table`/`cache_position`/`batch_idx` (§1.4, §2) | none | pure placement |
| KV cache read/write (§4.3) | none | replicated + device-local page_table |
| Per-layer `o_proj`/`down_proj` (Megatron row-parallel) | **TP-axis** (`cluster_axis=1`) reduce_scatter+all_gather | fundamental TP reduction; not DP — out of this doc's scope (see `sharding_analysis.md` §3.1) |

The only DP-axis collective in the paged-attention data path was the embedding
round-trip, and it is gone. Everything else on the DP axis is placement-only.

---

## 6. Divergences from the prior analysis (`sharding_analysis.md`)

1. **KV-cache layout tuple was wrong.** Prior:
   `[num_blocks, block_size, num_kv_heads, head_dim]`. Code:
   `(num_blocks, num_kv_heads, block_size, head_size)` (`attention.py:112`).
   Consequence: the pure-TP spec `(None,"model",None,None)` shards `num_kv_heads`
   (correct intent), but the dim it lands on is dim 1, not dim 2 as the prior
   tuple implied.
2. **Embedding fix is LANDED, not PROPOSED.** Prior §3.3a marked
   `("batch",None,None)` as `[PROPOSED]`; the working tree has it applied
   (`vllm_distributed_utils.py:354`).
3. **Correctness-DP-sharding confirmed and refined.** Prior concluded it is
   "already implemented, only physical de-replication open" — confirmed. Refinement:
   the *primary* mechanism is the **batch-sharded page_table** used by every paged
   op (fill/update/decode-read/chunked-read); `batch_idx % local_batch` is the
   **prefill-write-only** adapter, and it is correct only under GSPMD global-shape
   tracing (`local_batch = 128//4 = 32`).
4. **"block_size on DP axis" flagged unverifiable.** The prior analysis repeated
   the code comment's phrasing; this doc flags that the mechanism doesn't match
   the real layout and grounds the blocker on the no-batch-axis invariant instead.

---

## 7. Sources

- `integrations/vllm_plugin/vllm_tt/model_runner.py`: mesh/dp_size 310-324; KV
  dtype 357-367; `max_num_blocks_per_req` 389; chunked gate 432-449; buffers
  614-646; `batch_idx` arange 742-757; `get_kv_cache_spec` 1021-1099;
  `_get_slot_mapping_metadata` 1101-1167; `_prepare_inputs` page_table/fill/roll
  1346-1459; `chunk_start_idx` 1461-1475; `_pin_input_shardings` 1757-1784;
  `_dummy_run` marks 2278-2328; chunked precompile 2684-2729; KV alloc+sharding
  3380-3465; logits replicate 3496-3517.
- `integrations/vllm_plugin/vllm_tt/attention_impls/attention.py`:
  `get_kv_cache_shape` 104-112; `TTMetadata` 178-218; `_handle_paged_attention`
  461-515 (prefill `%local_batch` 490-511; decode 468-484);
  `_compute_full_attention` / chunked read 517-560; decode read 658-662.
- `integrations/vllm_plugin/vllm_tt/input_batch.py`: `MultiGroupBlockTable`
  18, 83-90; `num_computed_tokens` 74-80, 254.
- `integrations/vllm_plugin/vllm_tt/vllm_distributed_utils.py`:
  `partition_vocab_parallel_embedding` 339-357 (embedding fix 346, 354);
  `partition_parallel_lm_head` 325-336; `MODULE_TYPE_TO_WRAPPING_FUNC` 390-404.
- `tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py`:
  parametrize 296-312; knobs 357-373 (KV dtype 361, prefill_chunk_size 365).
- Prior: `devstral_batch128_notes/chunked_prefill_issue/sharding_analysis.md`;
  `devstral_batch128_notes/high_seq_length_support/vllm_tt_reference.md`.
