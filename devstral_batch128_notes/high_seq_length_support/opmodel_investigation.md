# OpModel vs. row-major-workaround for the `paged_fill_cache` tile-`page_table` hang

Read-only investigation of `third_party/tt-mlir/src/tt-mlir` (base commit `183b2b45d8`,
which is *our* fix "[TTNN] Enable page_table row-major workaround for paged-cache + SDPA
index ops at opt>=1"). All paths below are relative to that tree unless noted.

## TL;DR verdict

**(b) — An OpModel COMPLEMENTS the workaround but does NOT and CANNOT fix this hang by itself.**

Two facts settle it:

1. **`paged_fill_cache` already HAS an OpModel** (interface impl `TTNNOpModelInterface.cpp:3345-3387`,
   backend `TTNNOpModel.cpp:4925-5007`). The question "would *adding* an OpModel fix it?" is moot —
   it exists and the hang still happened at opt>=1 before our workaround entry.

2. **An OpModel can only enforce an operand layout when tt-metal's `validate` *rejects* the illegal
   layout.** tt-metal's `paged_fill_cache` validate checks the page_table's *dtype* (INT32) and
   *memory layout* (INTERLEAVED) but has **no** `page_table.layout() == ROW_MAJOR` check
   (`paged_fill_cache_device_operation.cpp:42-44`). So the OpModel constraint query returns
   **success for a TILE page_table** → nothing for the optimizer to react to → the tile page_table
   survives → the kernel reads swizzled block indices → hang. Contrast SDPA, which *does*
   `TT_FATAL(page_table.layout() == Layout::ROW_MAJOR)` (`sdpa_device_operation.cpp:216`) and whose
   OpModel therefore *has teeth*.

The correct principled upstream fix is a **tt-metal** change (add the ROW_MAJOR FATAL to
`paged_fill_cache`, mirroring SDPA) which would give the *already-existing* OpModel teeth — not a
tt-mlir "add an OpModel" change. Even then, whether the workaround entry becomes removable is
**likely but unverified** (see Q5). The OpModel is worth keeping for L1 legalization/estimation, but
it is **orthogonal to the full-depth DRAM OOM** (the constraint API is L1-only).

---

## 1. What is an OpModel in tt-mlir?

**Interface declaration** — `include/ttmlir/Dialect/TTNN/Interfaces/TTNNOpModelInterface.td:11-45`.
`TTNN_OpModelInterface` (`OpModel`) exposes exactly two methods:

- `getOpRuntime(inputs, config) -> llvm::Expected<size_t>` — runtime in ns (cost).
- `getOpConstraints(inputs, config) -> llvm::Expected<op_model::OpConstraints>` — legality + L1 usage.

Both take `const std::vector<TTNNLayoutAttr>& inputs` (the per-operand layouts) and a single
`const OpConfig& config` (which carries the **output** layout, plus op-specific config like conv
config). Default impl for both is `return llvm::createStringError("Not Implemented")`, so an op
"has no OpModel" unless it overrides these.

**What `getOpConstraints` returns** (`.td:28-43`): a 4-tuple `OpConstraints` =
`{CB L1 peak alloc bytes, Tensor L1 peak alloc bytes, Output L1 buffer alloc bytes, actual output
TTNNLayoutAttr}`. Note this is **L1-centric** — it models on-chip buffer usage and the *output*
layout the op will actually produce. It does **not** model DRAM.

**Backing library** — `lib/OpModel/TTNN/TTNNOpModel.cpp`. Each op's impl builds `ttnn::TensorSpec`s
from the (shape, `TTNNLayoutAttr`) pairs via `detail::convertToTensorSpec(...)`, then issues a
graph-capture query against the real tt-metal op:
`QUERY_OP_CONSTRAINTS(op, device, ...)` = `::ttnn::graph::query_op_constraints(...)`
(`TTNNOpModel.cpp:52-56`), wrapped by `operation::getOpConstraints(...)` (`:140-176`) which maps a
non-`Success` graph status to an `llvm::Error` (`:108`).

**Answer to the precise question — can an OpModel declare "operand N is only legal in RowMajor"?**
Only *transitively*, and only if tt-metal enforces it. The interface has **no per-operand legality
declaration**. The optimizer instead *probes*: it hands the OpModel a candidate `inputs` vector
(the current operand layouts) and reads back success/error + the actual output layout. An operand
layout is "illegal" iff the tt-metal graph-capture query **errors** on it (typically a `TT_FATAL`
inside the device op's `validate_*`). There is no static per-operand tile/row-major constraint the
OpModel can assert independently of what tt-metal's `validate` chooses to reject.

## 2. Which passes consume it, and how they use operand-layout info

Pipeline order (`lib/Dialect/TTNN/Pipelines/TTNNPipelines.cpp`): the **workaround pass runs first**
(`createTTNNPipelineWorkaroundPass`, `:429`), then the analysis/optimizer passes
(`createTTNNPipelineAnalysisPasses`, `:100-184`). The optimizer passes are the only OpModel
consumers. `grep` for `getOpConstraints`/`getOpRuntime` callers →
`MemoryLayoutPropagation.cpp`, `L1SpillManagement.cpp`,
`OptimizerPasses/GreedyMemoryLayoutPropagation.cpp`,
`OptimizerPasses/OperationValidationAndFallback.cpp`,
`Validation/OpConstraintValidation.cpp` (the shared entry point).

Shared query path — `Validation/OpConstraintValidation.cpp:160-208`: `validateConstraints` casts the
op to `OpModel`, and if the op lacks the interface **and** is not `OpModelExempt` it
`reportFatalInternalError`s; `OpModelExempt` ops instead return `notImplemented` so the optimizer can
fall back gracefully (`:164-181`). Otherwise it calls `backend.getOpConstraints(inputLayouts, config)`
(`:204-205`). The `inputLayouts` are the **current in-IR operand layouts** — see
`utils::extractInputLayouts(op)` at both call sites below.

**(a) `RowMajorLayoutPropagation`** (`OptimizerPasses/RowMajorLayoutPropagation.cpp`): propagates
RowMajor forward from RowMajor function args. For each consumer it calls
`opStopsRowMajorPropagation(user, operandIdx)` (`:318-382`), which builds `inputLayouts` from IR
(`:349`), nulls the output layout to let the backend choose (`:353`), and calls `validateOperation`
(`:355`). **If the query errors → it STOPS propagating RowMajor into that operand** (`:357-364`); if
the op returns a tiled output it also stops (`:367-374`). Crucially it does **not** re-tilize an
operand on its own — it only decides whether to *keep* RowMajor flowing and sets the *user's result*
layout. It stops outright at `ToLayoutOp` and at in-place ops with no results (`:320-338`).

**(b) `OperationValidationAndFallback`** (`OptimizerPasses/OperationValidationAndFallback.cpp`):
this is the one pass that *can re-lay-out an operand*. It extracts current operand layouts
(`extractInputLayouts`, `:195`) and validates (`:206-208`). If the original query **succeeds**, it
leaves operands untouched and only fixes output-layout mismatches (`:217-251`). If the query
**fails**, it runs `tryFallbacks` (`:270`) / `tryConfigFallbacks`, which sweep per-operand
`{RowMajor, Tile}` × dtype combinations (`:397-431`, `:534-542`) and apply an
`InputOperandChange`/`applyInputOperandChange` (`:45-116`) — i.e. it inserts a ToLayout to change an
operand's layout. **But this only fires on a failing query.**

**(c) Greedy memory-layout / `MemoryLayoutPropagation`**: choose output layouts / memory configs
(sharding, L1 vs DRAM) and L1 spill decisions using the same constraint queries. They pick
*output/mem-config*, not operand tile-vs-row-major.

**Answer to the critical sub-question:** the base pass's unconditional operand tilization
(`TTNNLayout.cpp:243`, `createToLayoutOp(..., /*tiled=*/true)` for every operand) is **only**
overridden back to RowMajor by the optimizer when a downstream OpModel query **errors** on the tiled
operand — via RM-propagation *stopping* (keeping an already-RM arg RM) or via
`OperationValidationAndFallback::tryFallbacks` *re-laying* the operand. Neither triggers if the query
succeeds. So an OpModel enforces operand layout **only when tt-metal's validate rejects the illegal
layout.** That is the linchpin of this whole investigation.

## 3. Does `paged_fill_cache` have an OpModel today?

**Yes — a complete one, independent of our workaround.**

- Interface: `TTNNOpModelInterface.cpp:3345-3387` — `PagedFillCacheOp::getOpConstraints` /
  `getOpRuntime`. It threads all operand layouts through: `inputs[0]`=cache, `inputs[1]`=input,
  `inputs[2]`=**page_table**, `inputs[3]`=batch_idx (optional) → backend
  (`:3361-3364`, `:3383-3386`).
- Backend: `TTNNOpModel.cpp:4925-5007` — builds a `pageTableSpec` from `pageTableLayout` and queries
  `::ttnn::experimental::paged_fill_cache` (`:4956-4962` constraints, `:4999-5005` runtime).

So the page_table's layout **is** passed to the tt-metal constraint query. Nothing is "missing" on
the tt-mlir side. What is missing is on the tt-metal side (Q5).

## 4. Does chunked SDPA have an OpModel (post #9027), and is the workaround still needed?

**Yes**, chunked SDPA has an OpModel: interface `TTNNOpModelInterface.cpp:2331-2371`, backend
`TTNNOpModel.cpp:3098-3194`.

**What #9027 (`ad2012fdc6`) actually added** (`git show --stat`): the OpModel for
`ChunkedScaledDotProductAttentionOp` — interface impl + backend + a `TestOpModelLib` case + an
`chunked_sdpa_rm_layout_propagation.mlir` lit test. Its commit message is decisive: without the
OpModel, RM-Layout-Propagation hit `insertTiledFixup` (because the op looked
`OpModelExempt`-like / unmodeled, `RowMajorLayoutPropagation.cpp:285-287`), fed the op **tile**
inputs, and tt-metal crashed with:

```
TT_FATAL @ .../sdpa_device_operation.cpp:212: page_table.layout() == Layout::ROW_MAJOR
Page table must be row major
```

That FATAL still exists today at `sdpa_device_operation.cpp:216`. So SDPA's validate **rejects a tile
page_table**, which is exactly why adding the OpModel *worked* as a fix for SDPA: the constraint query
now errors on the tiled page_table, and the optimizer keeps/produces RowMajor instead of crashing.

**The whitelist "decisive check" is contaminated — do not use it naively.** Chunked SDPA *is*
currently in `enabledOpsForWorkaroundWithOptimizer` (`TTNNWorkaroundsPatterns.cpp:601`), but:

- #9027 **deliberately did NOT whitelist** it — its message names whitelisting as the *rejected
  alternative* and shipped OpModel-only.
- Chunked SDPA re-entered the whitelist **only via our own commit** `183b2b45d8`
  (`git show 183b2b45d8` shows `+ ttnn::ChunkedScaledDotProductAttentionOp::getOperationName()`,
  added alongside `PagedFillCacheOp`, `PagedScaledDotProductAttentionDecodeOp`,
  `PagedFlashMultiLatentAttentionDecodeOp`).

So chunked SDPA is the **positive control**, not evidence of orthogonality: it's the case where the
OpModel route *did* enforce operand layout (workaround entry was never needed by #9027). Our
whitelist entry for chunked SDPA is therefore **plausibly redundant** with its OpModel — at most a
perf hedge to stop the optimizer from ever generating the expensive failing tile query (same
rationale as the f32-narrowing SDPA/TopK comment at `TTNNWorkaroundsPatterns.cpp:556-561`). Worth a
one-line note; not worth deeper investigation.

The real principle (not the whitelist heuristic):

> **An OpModel enforces operand layout iff tt-metal's `validate` rejects the illegal layout.**
> SDPA has `TT_FATAL(page_table.layout()==ROW_MAJOR)` → OpModel has teeth → workaround not required.
> `paged_fill_cache` has no such check → OpModel is toothless for layout → workaround required.

## 5. VERDICT

**(b) COMPLEMENT.** For `paged_fill_cache` specifically, adding an OpModel does **not** fix the
tile-page_table hang, because:

1. It already exists (Q3), and the hang occurred anyway at opt>=1 before the workaround entry.
2. tt-metal's `paged_fill_cache` validate (`paged_fill_cache_device_operation.cpp:26-53`) checks:
   - input dtype FLOAT32/BFLOAT16/BFLOAT8/4 (`:33-36`),
   - input memory-layout INTERLEAVED (`:38-40`),
   - **page_table memory-layout INTERLEAVED** (`:41-43`),
   - **page_table dtype INT32** (`:44`),
   - and even `batch_idx` tensor `layout() == ROW_MAJOR` (`:139`) —
   but **never** `page_table.layout() == ROW_MAJOR`. So a TILE page_table passes validate. The
   graph-capture constraint query therefore returns **Success** for a tiled page_table →
   `RowMajorLayoutPropagation` has nothing to stop on, and `OperationValidationAndFallback` never
   enters its fallback branch (`OperationValidationAndFallback.cpp:217` success path) → the tile
   page_table from `TTNNLayout.cpp:243` survives to runtime → kernel reads swizzled block indices →
   device hang. This is the *same blind spot* in validate and in the runtime kernel.

Only the **row-major operand workaround** forces it: `createPagedFillCacheOpOperandsWorkarounds`
(`TTNNWorkaroundsPass.cpp:499-521`) sets `pageTableWorkarounds.tensorLayoutWorkaround =
Layout::RowMajor` on operand 2 (and batch_idx when present). At opt>=1 that only fires if the op is in
`enabledOpsForWorkaroundWithOptimizer` — our fix (`TTNNWorkaroundsPatterns.cpp:600`). This is the
correct, working fix. (Note: the workaround forces layout **only** — no dtype override — matching
metal's needs since the graph already supplies INT32.)

**Is an OpModel feasible / worth adding for other reasons?**
- Feasibility: already done. The tt-metal backend exposes both constraints and runtime for
  `::ttnn::experimental::paged_fill_cache` (`TTNNOpModel.cpp:4956-5005`).
- Value: L1 legalization + L1 usage estimation at opt>=1 — genuine but **orthogonal to the DRAM OOM**
  we hit at full depth. The `OpConstraints` tuple is L1-only (CB/tensor/output **L1** bytes,
  `.td:30-36`); it does not model DRAM residency, so it will not by itself explain or fix a DRAM OOM.

**The principled "better fix" (if the goal is to retire the workaround entry):** mirror SDPA — add
`TT_FATAL(page_table.layout() == Layout::ROW_MAJOR, ...)` to tt-metal's `paged_fill_cache`
`validate_on_program_cache_miss`. That gives the *already-existing* OpModel teeth: the constraint
query would then reject a tiled page_table, and the optimizer could enforce RowMajor. Two caveats,
stated not glossed:

1. **This is a tt-metal change, not a tt-mlir OpModel addition.** "Add an OpModel" is the wrong lever
   — the OpModel is present; what's missing is metal-side rejection.
2. **Whether the workaround entry then becomes removable is likely but UNVERIFIED.**
   `PagedFillCacheOp` is a `TTNN_InplaceOp` (no tensor results — `TTNNOps.td`, "TTNN_InplaceOp",
   operands = cache[MemWrite], input, page_table, optional batch_idx). RM-propagation *stops* at
   in-place ops (`RowMajorLayoutPropagation.cpp:331-338`), so it would never re-lay the page_table.
   The only pass that could is `OperationValidationAndFallback::tryFallbacks`, and only on a failing
   query. page_table is a read operand (not the MemWrite cache), so a fallback re-lay is mechanically
   plausible — but I did not confirm that `tryFallbacks` re-lays operands of an in-place op (its
   null-layout config path for no-result ops, `OperationValidationAndFallback.cpp:184-190`, needs
   checking). Until confirmed, treat the workaround as still required even after a metal FATAL is
   added.

### One-line recommendation
Keep the `enabledOpsForWorkaroundWithOptimizer` entry for `PagedFillCacheOp` — it is the actual fix.
An OpModel neither replaces it (already present, toothless without a metal-side ROW_MAJOR check) nor
addresses the DRAM OOM (L1-only). If you want to eventually drop the workaround, pursue the tt-metal
`page_table.layout()==ROW_MAJOR` FATAL (SDPA-style) and then re-verify whether
`OperationValidationAndFallback` can re-lay the in-place op's page_table operand.

---

### Key citations
- `include/ttmlir/Dialect/TTNN/Interfaces/TTNNOpModelInterface.td:11-45` — interface (2 methods, L1-only constraints).
- `lib/OpModel/TTNN/TTNNOpModel.cpp:52-56,108,140-176` — query macros + error mapping.
- `lib/Dialect/TTNN/Transforms/TTNNLayout.cpp:243` — base pass tilizes every operand unconditionally.
- `lib/Dialect/TTNN/Validation/OpConstraintValidation.cpp:160-208` — shared query entry.
- `lib/Dialect/TTNN/Transforms/OptimizerPasses/RowMajorLayoutPropagation.cpp:318-382` (stops on query error / tiled output; `:331-338` stops at in-place ops; `:285-287` insertTiledFixup).
- `lib/Dialect/TTNN/Transforms/OptimizerPasses/OperationValidationAndFallback.cpp:195-302` (only re-lays operands on a failing query), `:397-431,534-542` (RowMajor/Tile operand sweep).
- `lib/Dialect/TTNN/Interfaces/TTNNOpModelInterface.cpp:3345-3387` — PagedFillCache OpModel (exists).
- `lib/OpModel/TTNN/TTNNOpModel.cpp:4925-5007` — PagedFillCache backend query (page_table spec passed).
- `lib/Dialect/TTNN/Interfaces/TTNNOpModelInterface.cpp:2331-2371`, `lib/OpModel/TTNN/TTNNOpModel.cpp:3098-3194` — Chunked SDPA OpModel (added by #9027).
- `lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp:499-521` — PagedFillCache page_table RowMajor workaround; `:1207-1235` — Chunked SDPA page_table+chunk_start_idx RowMajor workaround.
- `lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp:544-604` — `enabledOpsForWorkaroundWithOptimizer` (our entries at `:600-603`); `:504-508` opt>=1 gating.
- `third_party/tt-metal/.../paged_cache/device/fill_cache/paged_fill_cache_device_operation.cpp:26-53,139` — validate: page_table INT32+INTERLEAVED, batch_idx ROW_MAJOR, **no page_table ROW_MAJOR check**.
- `third_party/tt-metal/.../transformer/sdpa/device/sdpa_device_operation.cpp:216` — `TT_FATAL(page_table.layout()==ROW_MAJOR)` (SDPA does check).
- `git show ad2012fdc6` (#9027, OpModel-only, whitelist rejected) and `git show 183b2b45d8` (our commit added the 4 paged/chunked entries to the whitelist).
