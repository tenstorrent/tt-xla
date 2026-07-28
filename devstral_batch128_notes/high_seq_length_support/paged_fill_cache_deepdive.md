# `paged_fill_cache` DP+TP hang — operand/KV/sharding deep-dive

> **Read-only code investigation.** Nothing changed, nothing run on hardware. Line numbers are the
> current working tree. Shapes marked *global* are the logical shapes in the pre-partition StableHLO;
> *per-device* = global ÷ sharded-axis size. The IR quotes are from the real run
> `devstral_dptp_test_synced_trace_off_PASS.log` (the `num_tokens=128, num_reqs=128` chunked bucket,
> which is exactly the bucket that hung in `devstral_dptp_test_synced.log`, H8).

## Context recap (established, not re-litigated)
- **Root cause (H9/H17, `pagetable_workaround_and_mesh_analysis.md`):** at `optimization_level>=1` the four
  paged/SDPA ops are **absent** from `enabledOpsForWorkaroundWithOptimizer`
  (tt-mlir `TTNNWorkaroundsPatterns.cpp`), so the `page_table` operand of `ttnn.paged_fill_cache` is
  **never forced RowMajor** and reaches the kernel in **TILE** layout → misread block indices.
- **The bug is mesh-INDEPENDENT** (the enable gate keys on op name, not mesh). Fix validated (H11).
- This document answers the 5 configuration-dependence questions **given** that bug is active.

---

## Q1 — Full operand shard specs into `paged_fill_cache(cache, key, page_table, batch_idx)` for `[4,8]`

The prefill write is issued at `attention.py:500-511`:
```python
k_cache = torch.ops.tt.paged_fill_cache(k_cache, key_for_update,
                                         attn_metadata.fill_page_table, batch_idx=batch_idxs)
```
where `key_for_update = inputs.key.transpose(1,2)` (`attention.py:487`).

Mesh (`model_runner.py:314`): `xs.Mesh(range(32), [4,8], ("batch","model"))` →
`_axis_0="batch"`=DP=4, `_axis_1="model"`=TP=8; `dp_size=4` (`model_runner.py:322`).

**Verbatim from the run IR** (`...synced_trace_off_PASS.log`), inner `sdy.manual_computation`
(per-device), e.g. line 4327:
```
%48 = stablehlo.custom_call @tt.paged_fill_cache(%47, %35, %37, %46)
   : (tensor<78336x1x32x128xbf16>,  // cache      %arg8  sliced 8->1 on TP
      tensor<32x1x128x128xbf16>,    // key        %39    (32 users, 1 kv-head, 128 tok, 128 head)
      tensor<32x32xi32>,            // page_table %38
      tensor<32xi32>)               // batch_idx  %40
   -> tensor<78336x1x32x128xbf16>
```

| Operand | Global shape | dtype | Layout expected by kernel | Sharding (`sdy`) | Per-device shape | Marked where |
|---|---|---|---|---|---|---|
| **cache** `%arg8` | `[78336, 8, 32, 128]` | bf16 (staged; on-device BFP8) | tiled | replicated on DP (`_axis_0`), **all-slice 8→1 on TP** (`_axis_1`) | `[78336, 1, 32, 128]` | **not marked in tt-xla** (`model_runner.py:3446-3451`, DP+TP `pass`); the SPMD partitioner inserts `sdy.all_slice`/`all_to_all` on the TP axis (IR lines 4326/4597-4602) |
| **key** (`%39`, fill_value) | `[128, 8, 128, 128]` | bf16 | tiled | **batch** on DP + **model** on kv-head (TP) | `[32, 1, 128, 128]` | inherited from the q/k/v-proj TP column-parallel + DP-pinned input activation flow (`vllm_distributed_utils.py:197-214`, `_pin_input_shardings`); no direct mark on the K tensor |
| **page_table / fill_page_table** (`%38`) | `[128, 32]` | int32 | **RowMajor** (bug: gets TILE) | `("batch", None)` | **`[32, 32]`** | `model_runner.py:1455` (read) / `1459` (fill), `_dummy_run` 2302, chunked precompile 2705 |
| **batch_idx** (`%40`) | `[128]` = `arange(128)` | int32 | RowMajor | `("batch",)` | `[32]` holding `[32r..32r+31]` | `model_runner.py:1457`, 2304, 2707; then `% local_batch` → `[0..31]` at `attention.py:497-499` |

**Batch-sharded (`"batch"`/`_axis_0`, per-device dim0 = 32):** `key`, `page_table`, `fill_page_table`,
`cache_position`, `batch_idx`, `input_ids`, `position_ids`.
**TP-sharded (`"model"`/`_axis_1`):** `key`/`cache` num_kv_heads (8→1); the KV **cache** is sliced 8→1
on TP by the partitioner even though tt-xla leaves it un-annotated.
**Replicated:** KV cache on the DP axis (full `78336`-block pool on every one of the 32 chips —
`kv_pagetable_shard_map.md` §4.3); `chunk_start_idx` `[1]`.

The comment at `model_runner.py:1451-1454` states the intent: `page_table`/`cache_position`/`batch_idx`
must share the K/V input's per-device leading dim (32), because `paged_fill_cache`'s verifier requires
`batch_idx.dim0 == per-device batch`.

**Decisive fact for Q5:** the hanging bucket's per-device `page_table` is the **full `[32,32]`** — 32
real users × `max_num_blocks_per_req=32` (`arg1 ... local_shape = tensor<32x32xi32>`, IR line 4272).
It is **not** a tile-padded `[1,32]`; every one of the 1024 tile slots is a written value (a valid
block id or the zero/null block). This confirms the task's "`[32,32]` for both" premise for the
Devstral run and is central to the Q5 conclusion.

---

## Q2 — KV cache lifecycle across the two precompiled graphs (user hypothesis: **REFUTED**)

**One persistent buffer per layer, allocated once, shared by both graphs and all steps.**

- **Single allocation.** `initialize_kv_cache` (`model_runner.py:3331`) allocates, per attention layer,
  exactly one `[k_cache, v_cache]` pair via `torch.zeros(kv_cache_shape, dtype).to(device)`
  (`model_runner.py:3430-3433`), then `bind_kv_cache(kv_caches, static_forward_context, self.kv_caches)`
  (`3440-3444`). There is no per-graph or per-run reallocation anywhere; `_dummy_run` /
  `_get_dummy_inputs` build only `page_table`/`batch_idx`/etc. — never a KV tensor.
- **Bound before warmup.** vLLM V1 order (worker): `determine_available_memory` (`worker.py:237`, runs
  with a `torch.tensor([0])` placeholder, no real KV) → `initialize_from_config` →
  `initialize_kv_cache` (`worker.py:409`) → `compile_or_warm_up_model` → `capture_model`
  (`worker.py:381`). So the real persistent buffers are bound into `static_forward_context` **before**
  both graphs are traced. The `kv_cache[0].numel() > 0` guard in `TTAttentionBackendImpl.forward`
  (`attention.py:309-313`) is the switch between the profiling placeholder and the bound cache.
- **Both graphs read the *same* tensor.** The `prefix_chunk=False` and `prefix_chunk=True` graphs are two
  traces over the **same** forward context; the attention layer pulls `kv_cache` from the bound layer
  attribute, **not** from `attn_metadata` (`attn_metadata` only carries `page_table`/`chunk_start_idx`).
  The IR confirms it: in **both** graph dumps the cache enters as the same bound func arg
  (`%arg8`/`%arg11` = `...model_layers_0_self_attn_attn___kv_cache_{0,1}`, IR line 4272), not a
  freshly-materialized constant.
- **Identity preserved across steps.** After each write the impl does
  `kv_cache[0].copy_(k_cache); kv_cache[1].copy_(v_cache)` (`attention.py:513-515`) *specifically* to
  keep the tensor identity so XLA reuses the traced graph and the buffer persists across `generate()`
  calls. `torch.ops.tt.paged_fill_cache` is registered `mutates_args=[]` and returns a new value
  aliased back by this `copy_` (`custom_ops.py:1017`), and the func returns the cache
  (`output_operand_aliases` in the vhlo, IR line 2324) — an in-place update, not a new allocation.

**Conclusion:** no aliasing bug, no double-allocation, no divergent per-graph cache, no per-step
reallocation. The two chunked/non-chunked graphs demonstrably operate on the **same** persistent KV
buffer. The hang is not a cache-binding/lifecycle problem. *(The DP-axis replication of that buffer is
intended, `model_runner.py:3446-3451`; each of the 4 replicas holds a full 78336-block pool and writes
only its own 32 users' blocks via the batch-sharded page_table — `kv_pagetable_shard_map.md` §4.3.)*

---

## Q3 — Input sharding for everything `paged_fill_cache` depends on; `[4,8]`/`[2,4]` vs `[1,4]`

`_pin_input_shardings` (`model_runner.py:1757-1784`): `batch_axis = "batch" if use_2d_mesh else "model"`
(`1772`). Marks `input_ids` `(batch_axis, None)` (1774), `inputs_embeds` `(batch_axis, None, None)` (1776,
usually None for text-only Devstral — the embedding is computed on-device), and `position_ids`
`(batch_axis, None)` **only in DP modes** (1780-1784).

`page_table` / `fill_page_table` / `cache_position` / `batch_idx` are marked `("batch", …)` **only** in
`DATA_PARALLEL_ONLY` / `DATA_TENSOR_PARALLEL` (guard `model_runner.py:1447-1450`, marks 1455-1459; mirrored
in `_dummy_run` 2295-2304 and `_get_dummy_inputs` 2698-2707).

| Tensor | `[4,8]` DP+TP (`use_2d_mesh=True`) | `[2,4]` DP+TP (`use_2d_mesh=True`) | `[1,4]` pure-TP-1D (Gemma) |
|---|---|---|---|
| parallel_mode | `DATA_TENSOR_PARALLEL` | `DATA_TENSOR_PARALLEL` | `TENSOR_PARALLEL_ONLY_1D` (`1 in mesh_shape` → 296-299) |
| `dp_size` | 4 | 2 | **1** (322 only bumps in DP modes) |
| `batch_axis` | `"batch"` | `"batch"` | `"model"` |
| `input_ids`/`position_ids` | `("batch",None)` | `("batch",None)` | `input_ids ("model",None)`; **`position_ids` NOT pinned** (guard 1780) |
| `page_table`/`fill`/`cache_pos`/`batch_idx` | `("batch",…)` per-dev dim0=32 | `("batch",…)` per-dev dim0=32 | **NOT marked** → replicated across the 4 TP chips, full `max_num_seqs` leading dim |
| `batch_idx % local_batch` | applied (`dp_size=4`) | applied (`dp_size=2`) | **skipped** (`dp_size=1`, `attention.py:497`) |

**`[4,8]` and `[2,4]` take the identical code path** (both `DATA_TENSOR_PARALLEL`, both `use_2d_mesh=True`,
both mark `("batch",…)`); they differ only in `dp_size` (4 vs 2) and hence per-device batch (128/4=32 vs
64/2=32 — same 32 under the task's premise). The **`[1,4]` path is materially different**: page_table &
friends are replicated (not batch-sharded) and `position_ids` is deliberately unpinned (pure-TP hazard,
comment 1777-1779). So the `[1,4]`-vs-2D difference is real but, per H17, is **not** what protects a
non-hanging config — tilization of the page_table is unconditional either way.

---

## Q4 — `shard_weights_on_batch_axis` True vs False — does it touch the fill path? **No.**

Threaded only into the **weight** partition fns (`vllm_distributed_utils.py`): `batch_axis = "batch" if
shard_weights_on_batch_axis else None`, used in `XlaMergedColumnParallelLinear._shard_weight` (108-111),
`XlaQKVParallelLinear._shard_weight` (198-214), and `partition_row_parallel_linear` (306-307). It sets the
**second** axis of a weight's spec (`("model", batch_axis)` for column/QKV; `(batch_axis, "model")` for row).
`ColumnParallelLinear`/`nn.Linear`/`ParallelLMHead`/`VocabParallelEmbedding`/`FusedMoE` ignore it
(`274-357`). In pure-TP it is forced `True` (`model_runner.py:2181-2189`).

- **`False` (all three configs):** weights carry no DP-axis shard → **replicated across DP replicas**
  (FSDP off). `True` → weights additionally sharded on the "batch"/DP axis (memory win, adds a DP-axis
  weight reshard).
- It does **not** appear in `_prepare_inputs`, `_pin_input_shardings`, `_dummy_run`,
  `initialize_kv_cache`, or any `page_table`/`batch_idx`/`cache_position`/KV-cache mark. Those specs are
  set independently (Q1/Q3). The KV cache branch (`3446-3465`) never reads it.
- Activations: the K/V that feeds `paged_fill_cache` are batch-sharded via the DP-pinned inputs and the
  TP column-parallel q/k/v-proj, regardless of this flag. Changing it alters where *weights* live and one
  DP-axis weight collective — **not** the page_table, batch_idx, or KV-cache operand layouts.

**Conclusion:** orthogonal to the `paged_fill_cache` / page_table / KV path. Flipping it would not change
the tilized-page_table bug or its operands. (It *would* change weight memory and DP-axis weight comms —
relevant to OOM/perf, not to this hang.)

---

## Q5 — The discriminator: why `[4,8]` hangs but `[2,4]` doesn't (identical `[32,32]` page_table, same opt1/no-workaround)

### Premise check (surface, don't inherit silently)
- The task's "**same `[32,32]` for both**" is **confirmed for Devstral `[4,8]`** by the run IR
  (`arg1 local_shape = tensor<32x32xi32>`, 32 real users × 32 blocks/req — a *fully populated* tile).
- It is **assumed** for the Qwen3-8B `[2,4]` variant. That variant (`gmu=0.01`, "32 users/device") matches
  **no committed test** — the committed `[2,4]` smallmesh test (`test_prefill.py:236`) uses **Qwen3-0.6B,
  `gmu=0.3`, `max_num_seqs=8`** (→ 4 users/device, page_table `[4,32]` *tile-padded* to `[32,32]` with
  garbage padding rows). So it is the **user's own variant**, and its real `max_num_seqs` decides whether
  the tile is full `[32,32]` or padded. **This single unknown swings the answer** and should be nailed down
  (dump `page_table` local_shape from that run's IR).
- The two notes disagree: `kv_pagetable_shard_map.md`/this IR say per-device `[32,32]`; H10 recorded
  differing shapes (`[32,32]`/`[64,32]`/`[128,32]`). They cannot both be load-bearing — see below.

### The mechanism problem (the pivot the ranking must respect)
A **fully-populated `[32,32]` int32 tile** = exactly one TT tile (1024 elems, no physical padding). Read
row-major-when-stored-tiled, the tile-face swizzle **permutes** the 1024 entries. But every entry is
either a valid assigned block id (`0 ≤ id < num_blocks`) or the zero/null block — so a permuted entry is
**still a valid-but-wrong index → in-range → SILENT KV CORRUPTION, not a hang.** This predicts the *same*
outcome (silent corruption) for **both** `[4,8]` and `[2,4]` under an identical full `[32,32]` tile. **The
plain tile-swizzle therefore does not, by itself, explain why one hangs and the other doesn't.** An
out-of-range NoC address (→ the observed `15-3/15-2` device timeout) requires the misread to yield a value
that was *never* a valid index (uninitialized tile padding, or a byte-misaligned de-tilization producing
arbitrary 32-bit garbage). **This escalation is a tt-metal-kernel / tile-internal detail that is NOT
provable from this repo** (flagged already in `pagetable_workaround_and_mesh_analysis.md` §2 and H10).

Consequently I do **not** rank one clean primary. Two live discriminators remain, both repo-unprovable:

### Ranked candidates
**Co-leading (a) block-pool size / block-id magnitude / fill-iteration count.** *Code-provable part:*
`num_blocks = available_memory // page_size_bytes`, `available_memory ≈ device_DRAM × gpu_memory_utilization`
(`worker.py:298-330`, `model_runner.py:3394`). The two configs differ by **~2 orders of magnitude**:
  - Devstral `[4,8]`, BH galaxy: DRAM ≈ 31.88 GiB, `gmu=0.3` → KV budget ≈ 9.56 GiB (H16), and the IR shows
    **`num_blocks = 78336`** — block ids span a 17-bit range.
  - Qwen `[2,4]`, n300 Wormhole: DRAM ≈ 12–12.85 GiB, `gmu=0.01` → KV budget ≈ ~0.12 GiB → `num_blocks` on
    the order of **tens–low hundreds** — block ids are tiny; short `max_model_len` → most tile columns are
    the zero/null block.
  *Why it plausibly discriminates:* with a huge pool + longer/more-block sequences, any byte-misaligned
  de-tilization or padding read is far likelier to produce a value/address outside the allocated KV region
  (→ hang), and there are more fill iterations to hit one; with a tiny pool of small ids (mostly zeros), a
  misread almost always lands on a valid small block or the null block (→ silent corruption). *Not provable
  from repo* that the escalation actually fires — it is the most likely of the unprovable stories.

**Co-leading (b) arch NoC out-of-range behavior (WH n300 vs BH-galaxy Blackhole).** The `paged_fill_cache`
kernel and its NoC address handling are tt-metal/arch code, not in this repo. If WH silently wraps/faults
where BH times out (the observed symptom is exactly a BH `TT_METAL_OPERATION_TIMEOUT` at `15-3/15-2`), then
even an identical misread hangs on BH and doesn't on WH. This is a necessary condition for the *symptom* and
cannot be separated from (a) without a controlled experiment.

**Secondary (d) model dims — an amplifier, not a cause.** Devstral: 8 kv-heads global, TP8 → **1 kv-head/dev**;
Qwen3-8B: TP4 → 2 kv-heads/dev. The per-block stride `page_size_bytes = block_size × num_combined_kv_heads ×
padded_head_size × dtype_bits/8` (`attention.py:757-771`) scales the address `= base + block_id × stride`, so
a misread id lands *further* out for a larger stride — pushing in the same direction as (a), but far too
weak to be the discriminator alone.

**Ruled out (c) mesh/fabric size (8-chip vs 32-chip FABRIC_1D).** H8's `TT_RUNTIME_SYNC_AFTER_OP=1` run
localized the hang to `paged_fill_cache` — a **device-local write** into the DP-replicated pool, indexed by
the device-local page_table (`attention.py:500-511`; cache replicated on DP, `model_runner.py:3446-3451`).
It is not a collective; fabric/CCL was the async red herring (H5→H8). Chip count does not change this local
write. **Lowest.**

### The single experiment that separates the top two
Run the **Qwen `[2,4]` config on the same n300, only bumping `gpu_memory_utilization` 0.01 → 0.3** (and, if
its `max_num_seqs` is not already 32/dev, set it so the page_table is the full `[32,32]`, matching Devstral):
- **Now hangs** → pool-size / block-id-magnitude **(a)** is the discriminator.
- **Still no hang** → arch **(b)** (WH-vs-BH NoC out-of-range handling) is the discriminator, since pool size
  and page_table shape are then matched to the hanging Devstral case.

Cheap corroborations: (i) grep the `KV cache sizing:` log line (`worker.py:323-329`) for `num_blocks` on both
runs to confirm the ~100× gap; (ii) **re-enable `assert_output_coherent`/check PCC on the Qwen `[2,4]` run** —
the mechanism predicts it is **silently corrupt**, so "no hang" ≠ "correct" (its `assert_output_coherent` is
commented out — task, and the Devstral test likewise comments it at `test_...:407`). (iii) confirm the Qwen
variant's real `max_num_seqs`/page_table local_shape from its IR — a *padded* `[N,32]→[32,32]` tile has
genuine uninitialized padding and would make the out-of-range story provable for it too.

### Bottom line
The tilized-page_table bug is identical and mesh-independent for both configs and reliably produces *wrong*
indices for both. Under a genuinely identical full `[32,32]` page_table, the swizzle alone predicts **silent
corruption for both**, so hang-vs-no-hang is decided by whether a misread escalates to an **out-of-range NoC
address**, governed by **(a) pool size / block-id magnitude** and **(b) WH-vs-BH NoC handling** — co-leading,
neither provable from this repo. **(d)** amplifies **(a)**; **(c)** is ruled out (local op, not a CCL). The
`gmu 0.01→0.3` swap on the same n300 is the one experiment that breaks the (a)-vs-(b) tie. Regardless of which
wins, the validated fix (H11: add the four ops to `enabledOpsForWorkaroundWithOptimizer` → RowMajor page_table)
makes **both** correct, and Qwen `[2,4]`'s current "pass" is most likely unverified silent corruption.

---

## File:line index
- `attention.py`: `paged_fill_cache` call + `% local_batch` 490-511; `get_kv_cache_shape` 104-112; forward KV
  guard 309-313; identity-preserving `copy_` 513-515; `get_page_size_bytes` 757-771.
- `model_runner.py`: mesh/dp_size 310-324; parallel-mode 284-301; KV dtype 357-367; `max_num_blocks_per_req`
  389; chunked gate 432-449; `batch_idx` arange 742-757; `get_kv_cache_spec` 1021-1099; page_table/fill roll
  1376-1436; DP marks 1447-1459; `chunk_start_idx` 1461-1475; `_pin_input_shardings` 1757-1784; `_dummy_run`
  marks 2273-2304; `_get_dummy_inputs` marks 2698-2707; `shard_on_batch_axis` 2181-2191; `initialize_kv_cache`
  + bind 3331-3444; KV sharding branch 3446-3465; `capture_model` 3189-3221.
- `worker.py`: `determine_available_memory` / num_blocks + sizing log 237-330; `initialize_kv_cache` 409;
  `capture_model` (warmup, after KV bind) 381.
- `vllm_distributed_utils.py`: `safe_mark_sharding` 32-72; QKV/MergedCol/Row weight `batch_axis` 108-111,
  198-214, 306-307; embedding 339-357; `shard_model` 413-455.
- `custom_ops.py`: `paged_fill_cache` op (XLA custom_call, `mutates_args=[]`) 1017-1104.
- `tests/.../test_data_tensor_parallel_generation.py`: Devstral `test_dptp_devstral` gmu=0.3/max_num_seqs=128/
  chunk=128 378-396, `assert_output_coherent` commented 407.
- `tests/.../test_prefill.py`: committed `[2,4]` smallmesh (Qwen3-0.6B, gmu=0.3, max_num_seqs=8) 236-316.
- IR: `devstral_dptp_test_synced_trace_off_PASS.log` — hanging bucket num_reqs 2213; per-device
  `paged_fill_cache` 4327/4355; arg shapes/local_shapes 4272; TP all-slice of cache 4326/4597-4602.
- tt-mlir root cause: `pagetable_workaround_and_mesh_analysis.md`; sharding map: `kv_pagetable_shard_map.md`;
  timeline: `decisions.md` H8/H9/H10/H11/H17.
