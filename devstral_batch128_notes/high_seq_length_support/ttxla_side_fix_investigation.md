# Can the DP+TP `paged_fill_cache` hang be fixed tt-xla-side (no tt-mlir enabled-set edit)?

**Verdict up front: NOT FEASIBLE from tt-xla alone. The clean, durable fix is to land the
enabled-set addition UPSTREAM in tt-mlir.** Every frontend lever is defeated by tt-mlir's
unconditional operand tilization (`TTNNLayout.cpp:243`), and the corrective row-major workaround
is gated by a compile-time static set with no runtime/option/env injection point.

All paths in the local tree; tt-mlir paths are under
`third_party/tt-mlir/src/tt-mlir/`.

---

## 1. What the workaround does, precisely (verified)

The operand workaround forces the **`page_table`** operand (and the optional **`batch_idx_tensor`**,
when the op has 4 operands) of `ttnn.paged_fill_cache` to **RowMajor layout**. It sets *only* the
layout — no dtype/buffer/memory-layout change.

- Factory: `TTNNOperandsWorkaroundsFactory::createPagedFillCacheOpOperandsWorkarounds(Operation *op)`
  — `lib/Dialect/TTNN/IR/TTNNWorkaroundsPass.cpp:499-521`.
  - operand 0 (`cache`): empty workaround (unchanged)
  - operand 1 (`input`/fill_value): empty
  - operand 2 (`page_table`): `tensorLayoutWorkaround = Layout::RowMajor` (line 505)
  - operand 3 (`batch_idx_tensor`, only when `getNumOperands()==4`): `Layout::RowMajor` (line 509)
- The op is bound to that factory unconditionally in its ODS definition:
  `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td:1089-1093`
  (`getOperandsWorkarounds()` → `createPagedFillCacheOpOperandsWorkarounds(getOperation())`).
  So the op *always knows how* to produce its corrective workaround — the factory is not the gate.

**The gate** is in the workaround pass. `TTNNWorkarounds::runOnOperation()`
(`lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp:500-521`) builds the enabled-op
set based on optimization level:

```cpp
if (optimizationLevel >= 1) {
  enabledOps = enabledOpsForWorkaroundWithOptimizer;   // static const set
} else {
  enabledOps = utils::getAllTTNNDialectOps(&getContext());  // opt0: EVERY op
}
```

The rewriter (`TTNNOperandsWorkaroundsRewriter::matchAndRewrite`, same file `:272-276`) bails out for
any op not in `enabledOps`:
```cpp
if (!enabledOps->count(op.getOperation()->getName().getStringRef())) return failure();
```

`enabledOpsForWorkaroundWithOptimizer` is a **compile-time `static const std::set<StringRef>`**
(`TTNNWorkaroundsPatterns.cpp:544-604`). Our current fix (tt-mlir local commit **`183b2b45d8`**,
"[TTNN] Enable page_table row-major workaround for paged-cache + SDPA index ops at opt>=1") adds four
ops to it (`:592-603`): `PagedFillCacheOp`, `ChunkedScaledDotProductAttentionOp`,
`PagedScaledDotProductAttentionDecodeOp`, `PagedFlashMultiLatentAttentionDecodeOp`.

Confirmed recap correct: at **opt0** the workaround fires for *all* ops so `page_table` is corrected
to RowMajor; at **opt>=1** only the static set fires, so without the addition the correction never
runs and `page_table` reaches the kernel TILE-tilized → misread block indices → hang.

---

## 2. Is it DP+TP-related? — No. It is opt-level + arch-specific; DP-sharding is incidental.

- **The layout/workaround logic contains zero mesh/parallelism/sharding conditionals.** The
  tilization (`TTNNLayout.cpp:243`), the enabled-set gate, and the operand workaround are all keyed
  purely on (a) optimization level and (b) op identity. Grep of the workaround/layout code shows no
  `mesh`/`shard`/`DP`/`replica` predicate anywhere in the decision.
- **The tile-vs-rowmajor misread is a property of the tt-metal `paged_fill_cache` kernel** reading
  the `page_table` buffer as row-major. It is Blackhole-specific (Wormhole tolerates the swizzle) —
  again a per-op, per-arch kernel property, not a mesh property. The in-source comment
  (`TTNNWorkaroundsPatterns.cpp:592-599`) states this: the kernel "reads the page_table as row-major;
  without the workaround, opt_level>=1 layout propagation leaves the page_table TILE, so the kernel
  reads swizzled/garbage block indices and the device hangs."
- **The page_table's DP batch-sharding is incidental.** In `model_runner.py:1447-1459` the
  `page_table`/`batch_idx`/`fill_page_table` are `mark_sharding(..., ("batch", None))` **only** in
  `DATA_PARALLEL_ONLY`/`DATA_TENSOR_PARALLEL` modes — but that only changes the per-device *shape*
  of the tensor. Each per-device shard still flows through the identical tilize→(missing correction)
  path. Sharding neither causes nor cures the misread.
- **Why it was *observed* on DP+TP chunked-prefill:** that is simply the path that first exercised
  `paged_fill_cache` at opt1 on Blackhole in anger (see the same comment: "Observed on the DP+TP
  chunked-prefill path: ... hangs at opt1 but not at opt0"). Observation locus ≠ cause.

**Conclusion: opt>=1 + Blackhole-arch + per-op. DP-sharding is incidental, not causal.**

---

## 3. THE KEY QUESTION — can tt-xla force `page_table` RowMajor without editing the enabled set?

### The load-bearing obstacle (applies to every sub-option below)

`TTNNLayoutRewriter::matchAndRewrite` (`TTNNLayout.cpp:220-254`) runs in the **base layout pass**
(optimization-independent) and, for **every operand of every TTIR op**, inserts a tilizing
`to_layout`:

```cpp
for (OpOperand &operand : op->getOpOperands()) {
  ...
  std::optional<Value> desiredLayout = createToLayoutOp(
      rewriter, newLoc, operand.get(), g_defaultMemorySpaceDevice, /*tiled=*/true);  // line 243-245
  if (desiredLayout) op->setOperand(operand.getOperandNumber(), *desiredLayout);
}
```

`tiled=true` is hard-coded for operands (only *results* consult `shouldTilizeResult`, `:256-260`,
`:273`). There is **no operand-level exception, no op exclusion list, no dtype/shape/sharding
predicate** — the only skips are operands already produced by a `ttir.BroadcastOp` or
`ttir.ToLayoutOp` (`:234-236`), both internal tt-mlir ops the frontend cannot emit.

Net effect: whatever layout the `page_table` has when it reaches `paged_fill_cache`, this pass
re-tilizes it. The **only** thing that puts it back to RowMajor is the gated workaround pass. This
single fact refutes every "annotate it from the frontend" idea.

**The `ttir.ToLayoutOp` skip (`:234-236`) is verified unreachable from the frontend.** `ttir::ToLayoutOp`
is created only inside tt-mlir passes/conversions — `TTIR/Transforms/HoistCPUOps`,
`TTNN/Transforms/TTNNLayout`, `Conversion/TTNNToTTIR`, `Conversion/TTIRToTTNN`, `Conversion/TTIRToD2M`
(grep of `lib/` for `create<ttir::ToLayoutOp>`). It is **never** created in the frontend entry
conversion `StableHLOToTTIR` (grep of that directory returns nothing), and the `stablehlo.custom_call`
target registry in `StableHLOToTTIRPatterns.cpp` has no `tt.to_layout` target — nor does tt-xla emit
one (`custom_ops.py` has no such op). So the frontend cannot make `page_table`'s defining op a
row-major `ttir.to_layout` to trigger the skip; the tilize is genuinely unconditional from any
frontend-reachable construct.

### 3a. How tt-xla lowers `paged_fill_cache` — can it carry an operand-layout hint? **NOT FEASIBLE**

- Frontend op: `torch.ops.tt.paged_fill_cache`, registered in
  `python_package/tt_torch/custom_ops.py:1107-1195`. On XLA it emits
  `stablehlo_custom_call([cache, fill_value, page_table, batch_idx], "tt.paged_fill_cache",
  [cache.shape], [cache.dtype])` with **no `frontend_attributes` at all** (`:1119-1126`) — contrast
  with sibling ops that do pass attrs. There is no layout attribute plumbed and no place the current
  API exposes one.
- The StableHLO→TTIR conversion (`StableHLOToTTIRPatterns.cpp`) maps the custom call 1:1 to
  `ttir.paged_fill_cache`; TTIR→TTNN (`TTIRToTTNN.cpp:759-780`,
  `PagedFillCacheOpConversionPattern`) maps it 1:1 to `ttnn.paged_fill_cache`. Neither reads or
  carries any per-operand layout attribute.
- The op definitions themselves carry **no layout operand attribute**:
  `TTIR_PagedFillCacheOp` (`TTIROps.td:3692-3702`) and the TTNN op (`TTNNOps.td:1084-1096`) declare
  only `cache`/`input`/`page_table`/`batch_idx_tensor` tensors. Layout is decided exclusively by the
  layout + workaround passes.
- Even if `stablehlo.custom_call` `operand_layouts`/`result_layouts` attributes were set from the
  frontend, tt-mlir discards them here and the `tiled=true` rewriter (§3 obstacle) would re-tilize
  regardless.

**Refutes** the idea of a surviving operand-layout hint. Confirms "the compiler owns the layout" for
this operand.

### 3b. Compiler-option knob to enable the workaround at opt>=1 — **NOT FEASIBLE**

- tt-xla passes options as a plain dict via `torch_xla.set_custom_compile_options(...)`
  (`custom_ops`/`codegen.py`; vLLM builds it in
  `platform.py:201-221 get_pjrt_compile_config()` and applies it at `model_runner.py:234`). These
  are forwarded as strings to the tt-mlir pipeline; there is **no tt-xla C++ code** that constructs
  or configures the workaround pass (grep of `src/` for optimization/workaround/pipeline options
  returns nothing — it is pure passthrough).
- The tt-mlir `ttnn-workaround` pass exposes exactly **three** options
  (`include/ttmlir/Dialect/TTNN/Transforms/Passes.td`, `def TTNNWorkarounds`, `:31-53`):
  `ttnn-enable-layout-workaround-pass` (bool), `ttnn-enable-decomposition-workaround-pass` (bool),
  `ttnn-optimization-level` (int). **None accepts an op list.** The enabled set is the hard-coded
  static const; there is no "enable-all-workarounds", no per-op enable, no env var, no pipeline flag
  that injects an op into `enabledOpsForWorkaroundWithOptimizer`.
- The only pipeline knob that would make the workaround fire for this op is forcing
  `optimization_level = 0` — which globally disables the optimizer/fusion/memory-layout passes
  (`TTNNPipelines.h:599-615`). That is a whole-graph regression, not a targeted tt-xla fix, and the
  option is global to the compile (one value in `get_pjrt_compile_config`), so it cannot be scoped to
  just the `paged_fill_cache` graph.

**Refutes** the "flip an existing knob" hope.

### 3c. Construct/annotate the tensor so tt-mlir keeps it RowMajor — **NOT FEASIBLE**

- Because `TTNNLayout.cpp:243` tilizes operands unconditionally (§3 obstacle), no dtype, shape, or
  sharding chosen by the frontend keeps the `page_table` row-major at the op boundary.
- The one frontend-reachable RowMajor hook, `shouldForceInputRowMajor`
  (`TTNNLayout.cpp:709-729`, driven by `ttcore.argument_type = "input"` which tt-xla *can* set via
  `torch.ops.tt.mark_argument_attributes`, `custom_ops.py:11-55`), only sets the **function
  argument's** initial layout. It does **not** propagate to the op operand: even a row-major func-arg
  `page_table` feeding `paged_fill_cache` directly is re-tilized by the operand rewriter before the
  op. (And `mark_sharding`/`tt.sharding_constraint` affect Shardy partitioning, not tile-vs-row-major
  layout.)

**Verifies the "unconditional tilization leaves no frontend hook" belief is correct.**

### 3d. Restructure the write to avoid the tile-sensitive read — **NOT FEASIBLE (as a real fix)**

- The paged KV cache *requires* `page_table` indirection; the non-paged `tt.fill_cache` /
  `tt.update_cache` ops (`custom_ops.py:915-960`, `:860-912`) write contiguous per-batch slots and
  have no page-table semantics, so they cannot express the block-indexed write. Swapping ops changes
  correctness, not just layout.
- Pre-converting `page_table` on host to a "row-major" layout is meaningless: host tensors are
  transferred to device and pass through the same `TTNNLayout` tilization. Host dtype/layout does not
  survive as a device tile-vs-row-major choice.
- The chunked-prefill read path has the identical problem for its own index tensors
  (`ChunkedScaledDotProductAttentionOp`, `PagedScaledDotProductAttentionDecodeOp`), which is why the
  same commit adds all four ops — restructuring one op would still leave the others hanging.

---

## 4. Recommendation

**A tt-xla-only fix is not realistically possible.** The layout of the `page_table` operand is owned
end-to-end by tt-mlir's layout + workaround passes:

1. `TTNNLayout.cpp:243` unconditionally tilizes every op operand (no frontend hook survives — refutes
   3a/3c/3d), and
2. the corrective RowMajor workaround is gated by a **compile-time static set** with no option/env/
   pass-argument injection point (refutes 3b).

The frontend can annotate names, argument types, sharding, and dtype — none of which reach the
tile-vs-row-major decision for this operand. The prior-session belief **"the compiler owns the
layout; there is no frontend hook"** is **confirmed correct**, and a runtime `to_layout` from tt-xla
is likewise not expressible (there is no StableHLO/torch op that lowers to a surviving `ttnn.to_layout`
row-major on this operand).

**The cleanest durable path is to land the enabled-set addition UPSTREAM in tt-mlir.** The fix already
exists as commit `183b2b45d8` (author ssaliceTT, 2026-07-20) but currently lives only on the personal
branch `origin/ssalice/devstral-07-20-2026`; it is **not an ancestor of `origin/main`**
(`git merge-base --is-ancestor` → false). That is exactly why it "keeps getting dropped on rebases":
any submodule uplift to a newer tt-mlir `main` that doesn't include this commit loses it.

Concretely:
- Open a tt-mlir PR adding the four ops to `enabledOpsForWorkaroundWithOptimizer`
  (`TTNNWorkaroundsPatterns.cpp:592-603`) — the change is already written and self-documenting.
- Once merged to tt-mlir `main`, bump the tt-xla `third_party/tt-mlir` submodule pointer to a commit
  that contains it. Rebases then carry it automatically.

This is a one-line-per-op, well-justified compiler-correctness change (mirrors the existing SDPA/TopK
entries at `:549-563`), and upstreaming is the only mechanism that makes it rebase-durable. If an
*interim* tt-xla-side stopgap is unavoidable before the PR lands, the only lever that works is setting
`optimization_level = 0` for the affected compile — a global perf regression, explicitly not
recommended as anything but a temporary unblock.
