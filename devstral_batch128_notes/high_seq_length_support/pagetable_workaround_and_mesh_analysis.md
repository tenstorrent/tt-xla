# page_table Row-Major Workaround & DP/TP Mesh Hang Analysis

**Scope:** read-only code investigation of the Devstral-2-123B DP+TP (mesh `[4,8]`) chunked-prefill
hang, and why Gemma-4-31B on qb2 `[1,4]` reportedly does not hit it.

**tt-mlir state at time of writing:** submodule HEAD `3abca42835`
(`third_party/tt-mlir/src/tt-mlir`), clean working tree (no local modifications).

---

## TL;DR

- The four operand-workaround **factories** and their `.td` interface **wiring** already exist at
  HEAD. The **only** missing piece is enabled-set membership: the four op names are **absent** from
  `enabledOpsForWorkaroundWithOptimizer` (verified by grep — no matches). So at
  `optimization_level >= 1` the rewriter never applies their operand workarounds, and the
  `page_table` operand stays TILE.
- **Re-apply = add exactly four `::getOperationName()` lines** to that set
  (`lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp`, initializer at lines
  545–591). Nothing else needs to change.
- **Q2 discriminator is one fact: the coworker's `optimization_level` and tt-mlir SHA.** The most
  likely, code-provable explanation is that the coworker's compiler never had the bug active
  (opt0 workarounds everything; or their tt-mlir tree still listed the ops). The tile-swizzle
  shape-dependent-symptom story is a secondary hypothesis that only applies *if* both ran the same
  buggy compiler.

---

## Q1 — What the row-major `page_table` operand workaround does, and how to re-apply it

### 1. The mechanism (three layers)

**Layer A — default layout assignment tilizes every operand.**
`lib/Dialect/TTNN/Transforms/TTNNLayout.cpp:243-245` creates each operand's desired layout with
`/*tiled=*/true` unconditionally:

```cpp
std::optional<Value> desiredLayout = createToLayoutOp(
    rewriter, newLoc, operand.get(), g_defaultMemorySpaceDevice,
    /*tiled=*/true);
```

Results are tiled too (`shouldTilizeResult(op)` returns `true` for everything except a couple of
special cases, `TTNNLayout.cpp:256-291`). So absent any workaround the `page_table` operand of a
paged op is laid out in **TILE** layout.

**Layer B — per-op operand workarounds declare the corrective layout/dtype.**
Each op's `getOperandsWorkarounds()` interface method (defined in
`include/ttmlir/Dialect/TTNN/IR/TTNNOps.td`) dispatches to a factory in
`lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp`:

| Op | `.td` wiring (line) | Factory (line) |
|---|---|---|
| `PagedFillCacheOp` | TTNNOps.td:1091 | TTNNWorkaroundsPass.cpp:500 |
| `PagedScaledDotProductAttentionDecodeOp` | TTNNOps.td:3906 | TTNNWorkaroundsPass.cpp:1157 |
| `ChunkedScaledDotProductAttentionOp` | TTNNOps.td:3939 | TTNNWorkaroundsPass.cpp:1208 |
| `PagedFlashMultiLatentAttentionDecodeOp` | TTNNOps.td:3977 | TTNNWorkaroundsPass.cpp:1241 |

**Layer C — the rewriter only fires for ops in the enabled set at opt>=1.**
`TTNNOperandsWorkaroundsRewriter::matchAndRewrite`
(`.../Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp:272-276`) bails immediately unless the op's
name is in `enabledOps`:

```cpp
if (!enabledOps->count(op.getOperation()->getName().getStringRef())) {
  return failure();
}
```

And `enabledOps` is selected by optimization level (same file, 503-508):

```cpp
std::set<mlir::StringRef> enabledOps;
if (optimizationLevel >= 1) {
  enabledOps = enabledOpsForWorkaroundWithOptimizer;   // curated small set
} else {
  enabledOps = utils::getAllTTNNDialectOps(&getContext());  // EVERY op
}
```

So at **opt0** every op (including the four paged ops) gets its workarounds; at **opt>=1** only the
curated set does. At HEAD the curated set (initializer at 545–591) does **not** contain any of the
four op names — confirmed by grep returning no matches for
`PagedFillCacheOp::getOperationName` / `ChunkedScaledDotProductAttentionOp::getOperationName` /
`PagedScaledDotProductAttentionDecodeOp::getOperationName` /
`PagedFlashMultiLatentAttentionDecodeOp::getOperationName` in that file.

### 2. Exactly which operands each factory touches, and what it forces

Quoting each factory. `TTNNOperandWorkarounds` with `tensorLayoutWorkaround = Layout::RowMajor`
forces the operand to ROW_MAJOR; adding `tensorDataTypeWorkaround = ...Int32` additionally forces
the dtype.

**`PagedFillCacheOp`** (`TTNNWorkaroundsPass.cpp:500-521`) — operand order is
`(cache, input/kv, page_table[, batch_idx])`:
```cpp
pageTableWorkarounds.tensorLayoutWorkaround = Layout::RowMajor;
// if op has 4 operands (batch_idx tensor present):
batchIdxTensorWorkarounds.tensorLayoutWorkaround = Layout::RowMajor;
...
.addInputOperandWorkaround(nullWorkarounds)          // cache      -> no change
.addInputOperandWorkaround(nullWorkarounds)          // input/kv   -> no change
.addInputOperandWorkaround(pageTableWorkarounds)     // page_table -> RowMajor
.addInputOperandWorkaround(batchIdxTensorWorkarounds)// batch_idx  -> RowMajor
```
→ **page_table: RowMajor (layout only)**, **batch_idx: RowMajor (layout only)** when present. No
dtype forced.

**`ChunkedScaledDotProductAttentionOp`** (`TTNNWorkaroundsPass.cpp:1208-1235`) — operand order
`(query, key, value, page_table, chunk_start_idx)`:
```cpp
rowMajorLayoutWorkaround.tensorLayoutWorkaround = Layout::RowMajor;
...
.addInputOperandWorkaround(emptyWorkaround)          // query
.addInputOperandWorkaround(emptyWorkaround)          // key
.addInputOperandWorkaround(emptyWorkaround)          // value
.addInputOperandWorkaround(rowMajorLayoutWorkaround) // page_table    -> RowMajor
.addInputOperandWorkaround(rowMajorLayoutWorkaround) // chunk_start_idx-> RowMajor
```
→ **page_table: RowMajor** and **chunk_start_idx: RowMajor** (layout only, no dtype). Q/K/V/output
untouched.

**`PagedScaledDotProductAttentionDecodeOp`** (`TTNNWorkaroundsPass.cpp:1157-1202`):
```cpp
rowMajorLayoutWorkaround.tensorLayoutWorkaround = Layout::RowMajor;
// Q, K, V -> emptyWorkaround (no change)
if (sdpaOp.getPageTable())    addInputOperandWorkaround(rowMajorLayoutWorkaround); // page_table -> RowMajor
// attention_mask -> empty
if (sdpaOp.getCurPosTensor()) addInputOperandWorkaround(rowMajorLayoutWorkaround); // cur_pos     -> RowMajor
// attention_sink -> empty; output -> empty
```
→ **page_table: RowMajor**, **cur_pos_tensor: RowMajor** (layout only, no dtype).

**`PagedFlashMultiLatentAttentionDecodeOp`** (`TTNNWorkaroundsPass.cpp:1241-1295`) — the only one
that also forces dtype:
```cpp
rowMajorInt32Workaround.tensorLayoutWorkaround = Layout::RowMajor;
rowMajorInt32Workaround.tensorDataTypeWorkaround = ttcore::DataType::Int32;   // <-- dtype too
// query, key -> empty; value (optional) -> empty
.addInputOperandWorkaround(rowMajorInt32Workaround)  // page_table -> RowMajor + Int32
// attention_mask -> empty
if (mlaOp.getCurPosTensor()) addInputOperandWorkaround(rowMajorLayoutWorkaround); // cur_pos -> RowMajor
// attention_sink -> empty; output -> empty
```
→ **page_table: RowMajor + Int32**, **cur_pos_tensor: RowMajor** (layout only).

> Precision note: do **not** generalize "RowMajor + Int32" to all four. Only the MLA-decode op forces
> the page_table dtype to Int32. The other three force **layout (RowMajor) only** and leave the dtype
> as-is (already int32 from the frontend).

For reference, the adjacent `PagedUpdateCacheOp` factory (`TTNNWorkaroundsPass.cpp:449-473`) does the
same thing (`updateIndex` and `page_table` -> RowMajor), and it *is* handled — but via a dedicated
always-on rewrite pattern (`PagedUpdateCacheOpRewritePattern`, added below opt-level 2 at
TTNNWorkaroundsPatterns.cpp:491-495), not via the enabled-set. That is why UpdateCache was never
implicated in this hang.

### 3. IR-level effect

`workaroundInputOperand` (`.../TTNNWorkaroundsPatterns.cpp:103-147`) computes
`applyWorkarounds(workaround, currentLayout)` (definition in `TTNNWorkaroundsPass.cpp`):

```cpp
results.tensorLayoutResult.targetValue =
    workaround.tensorLayoutWorkaround.value_or(inputLayoutAttr.getLayout());
results.tensorLayoutResult.previousValue = inputLayoutAttr.getLayout();
```

`isModified()` is true iff `target != previous`. If the operand is currently TILE (from Layer A) and
the workaround forces RowMajor, `target=RowMajor != previous=TILE` → modified → the rewriter
**inserts a `ttnn.to_layout` (untilize) op before the paged op**, rewiring the operand
(`op->setOperand(...)`, lines 124-144). For the MLA op the same `to_layout` also carries the Int32
dtype change.

**With the op in the enabled set (opt>=1):** compiler inserts
`page_table_rm = ttnn.to_layout(page_table, layout=RowMajor)` (untilize) immediately before the op,
and the paged op reads the ROW_MAJOR page_table — matching what the tt-metal `paged_fill_cache` /
paged-SDPA kernels expect.

**Without it (current HEAD, opt>=1):** the rewriter returns `failure()` at the enabled-set gate; no
`to_layout` is inserted; the page_table reaches the kernel in **TILE** layout. The tt-metal kernel
indexes it as a flat row-major int32 buffer, so it reads swizzled/garbage block indices → device
hang (see Q2 for why the *symptom* is mesh/shape-dependent).

> Empirical cross-check that "default is TILE": because adding the ops to the set demonstrably fixed
> the hang, the inserted workaround was **not** a no-op, i.e. `target != previous`, i.e. the operand
> really was TILE by default. This is a stronger proof than the TTNNLayout.cpp line alone, and it
> confirms the always-on `FillCacheInputPadRewritePattern<PagedFillCacheOp>`
> (TTNNWorkaroundsPatterns.cpp:449-450) does **not** touch page_table layout (else the workaround
> would be redundant).

### 4. Re-apply recipe (post-rebase)

Net statement per op: *adding op X's name to `enabledOpsForWorkaroundWithOptimizer` causes the
compiler, at opt>=1, to insert a `ttnn.to_layout` untilize on the operands listed below,
immediately before op X.*

Edit `lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp`, the
`enabledOpsForWorkaroundWithOptimizer` initializer (currently lines 545–591), adding:

```cpp
ttnn::PagedFillCacheOp::getOperationName(),                     // page_table (+batch_idx) -> RowMajor
ttnn::ChunkedScaledDotProductAttentionOp::getOperationName(),   // page_table, chunk_start_idx -> RowMajor
ttnn::PagedScaledDotProductAttentionDecodeOp::getOperationName(),// page_table, cur_pos -> RowMajor
ttnn::PagedFlashMultiLatentAttentionDecodeOp::getOperationName(),// page_table -> RowMajor+Int32, cur_pos -> RowMajor
```

No changes to the factories (already present at TTNNWorkaroundsPass.cpp:500/1157/1208/1241) or to the
`.td` wiring (TTNNOps.td:1091/3906/3939/3977) are needed — only the set membership. Confirm the exact
op class names still match the generated C++ if the ops were renamed in the rebase.

---

## Q2 — Why Gemma-4-31B `[1,4]` reportedly does not hang while Devstral `[4,8]` does

### The compiler bug is mesh-independent; only the symptom can be mesh/shape-dependent

The workaround-application gate (Layer C above) keys on the **op name**, not on mesh shape,
sharding, or tensor size. So at opt>=1 with the ops absent from the set, the `page_table` reaches the
kernel in TILE layout on **any** mesh — `[1,4]` and `[4,8]` alike. Therefore the *bug* cannot be what
differs; at most the *hang symptom* differs. This reframes Q2 as: either the coworker's compiler
didn't have the bug active, or it did and the symptom is masked on `[1,4]`.

### Code-provable mesh-path differences (tt-xla side)

Parallel-mode selection, `integrations/vllm_plugin/vllm_tt/model_runner.py:284-323`:

- **Gemma `[1,4]`**: `1 in mesh_shape` → `use_2d_mesh=False` and `explicit_2d_mesh=False`
  (line 292-295). With TP-only enabled → `parallel_mode = TENSOR_PARALLEL_ONLY_1D` (line 299).
  `dp_size` stays `1` (line 318-322 only bumps dp_size in DP modes).
- **Devstral `[4,8]`**: DP+TP both enabled → `parallel_mode = DATA_TENSOR_PARALLEL` (line 285),
  `dp_size = mesh_shape[0] = 4` (line 322).

Consequences that are directly code-provable:

1. **page_table sharding.** The `safe_mark_sharding(page_table, mesh, ("batch", None))` block only
   runs for `DATA_PARALLEL_ONLY`/`DATA_TENSOR_PARALLEL` (`model_runner.py:1447-1459`). On Devstral
   `[4,8]` the page_table is **batch-sharded** across 4 replicas → per-device leading dim = 32 users
   (128 total / 4). On Gemma `[1,4]` (1D TP) the page_table is **not** batch-sharded — it is
   replicated across the 4 TP devices with its full `max_num_seqs` leading dim.

2. **batch_idx rebase.** `batch_idxs = batch_idxs % local_batch` runs only when
   `dp_size > 1` (`attention_impls/attention.py:497-499`). Gemma (dp_size=1) skips it. **This is not
   the cause of the hang** — it only remaps global→local batch *rows* for the paged_fill_cache write
   target; it does not affect whether the block indices inside page_table are read correctly. Identity
   no-op at dp_size=1.

3. **page_table dtype/shape.** page_table is `torch.int32`, shape `[num_reqs, blocks_per_req]`
   (`model_runner.py:526-528`, `1348-1366`), `blocks_per_req = cdiv(max_model_len, block_size)`
   (line 389). Devstral high-seq-length → **large blocks_per_req and large block-id values**; Gemma
   short context + `max_num_seqs ~8` → small page_table, most rows zero-padded to a tile.

### Ranked explanations

**1. (Most likely — code-provable) The coworker's compiler never had the bug active.** Two variants,
both making page_table RowMajor on *any* mesh:
   - Coworker ran **`optimization_level == 0`**: `enabledOps = getAllTTNNDialectOps`
     (TTNNWorkaroundsPatterns.cpp:507) → the paged ops get their workarounds → page_table untilized →
     no hang.
   - Coworker's **tt-mlir tree still listed the four ops** in
     `enabledOpsForWorkaroundWithOptimizer` (i.e. before the rebase dropped them), even at opt>=1.

   *Action:* confirm the coworker's `optimization_level` and tt-mlir SHA. This is the boring, probable
   answer and should be the primary hypothesis.

**2. (Hypothesis — only if both ran the identical buggy compiler, opt>=1 + ops absent) Symptom is
shape/value-driven, not mesh-mode-driven.** The tile layout mostly **permutes valid block-ids among
positions** (a 32×32 int32 tile read as row-major scrambles which entry lands where) → the kernel
gets *wrong-but-in-range* block ids → **silent KV corruption, no fault**. To get an out-of-range NoC
address (→ hang) the row-major reader must land in **uninitialized tile padding** — which happens
when `blocks_per_req` is not tile-width-aligned, or when the swizzle pulls from zeroed/uninitialized
tail regions. The plausible discriminator is **Devstral's high sequence length → many blocks and large
block-id magnitudes**, making an out-of-range read likely; Gemma's tiny page_table (few blocks, mostly
zero padding → block id 0, the null/in-range block) tends to land in-range → corruption without a
fault. The 32-users-vs-8-users difference contributes (32 exactly fills one tile row; 8 leaves rows
8–31 zeroed) but seq length is the stronger lever.

   *Caveats to state to the coworker:* (a) "no hang" ≠ "correct" — on this branch Gemma `[1,4]` would
   likely be producing **silently corrupt KV**; they should check PCC/output quality, not just
   liveness. (b) The out-of-range mechanism is reasoned, not proven from kernel source here — it
   requires the row-major reader to hit tile padding.

**Bottom line:** The compiler bug (TILE page_table at opt>=1) is mesh-independent. The realistic reason
Gemma `[1,4]` "works" is #1 — a different opt level or a tt-mlir tree that still enabled the ops. Only
if that is ruled out does the shape-dependent-symptom story (#2) apply, and even then Gemma is most
likely silently corrupt rather than truly correct. Confirm opt level + tt-mlir SHA to break the tie.

---

## Key file:line index

- `third_party/tt-mlir/.../Transforms/TTNNLayout.cpp:243-245` — operands default to `tiled=true`.
- `third_party/tt-mlir/.../IR/TTNNWorkaroundsPass.cpp:500-521` — PagedFillCache factory (page_table, batch_idx → RowMajor).
- `...TTNNWorkaroundsPass.cpp:1157-1202` — PagedSDPADecode factory (page_table, cur_pos → RowMajor).
- `...TTNNWorkaroundsPass.cpp:1208-1235` — ChunkedSDPA factory (page_table, chunk_start_idx → RowMajor).
- `...TTNNWorkaroundsPass.cpp:1241-1295` — PagedFlashMLADecode factory (page_table → RowMajor+Int32, cur_pos → RowMajor).
- `...TTNNWorkaroundsPass.cpp:449-473` — PagedUpdateCache factory (handled separately via always-on pattern).
- `third_party/tt-mlir/.../Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp:272-276` — enabled-set gate.
- `...TTNNWorkaroundsPatterns.cpp:491-495` — PagedUpdateCache always-on pattern (opt<2).
- `...TTNNWorkaroundsPatterns.cpp:503-508` — opt-level selects enabled set (opt0=all ops, opt>=1=curated).
- `...TTNNWorkaroundsPatterns.cpp:545-591` — `enabledOpsForWorkaroundWithOptimizer` (the 4 ops are MISSING here).
- `...TTNNWorkaroundsPatterns.cpp:103-147` — `workaroundInputOperand` inserts the `to_layout`.
- `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td:1091/3906/3939/3977` — `.td` `getOperandsWorkarounds` wiring.
- `integrations/vllm_plugin/vllm_tt/model_runner.py:284-323` — parallel-mode / dp_size / use_2d_mesh selection.
- `model_runner.py:1447-1459` — page_table/cache_position/batch_idx `safe_mark_sharding` (DP modes only).
- `model_runner.py:526-528,1348-1366,389` — page_table int32 dtype and `[num_reqs, blocks_per_req]` shape.
- `integrations/vllm_plugin/vllm_tt/attention_impls/attention.py:497-511` — batch_idx `% local_batch` (dp_size>1) + paged_fill_cache calls.
- `attention.py:550-560` — chunked_scaled_dot_product_attention call (page_table, chunk_start_idx).
