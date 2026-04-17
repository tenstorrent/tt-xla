# Plan: `RewriteBatchParallelGatherPass` in tt-mlir

## Context

Running `pytest -svv tests/torch/ops/test_gather.py::test_gather_indices[32-4]` on a 2×4 mesh produces low PCC because the `tenstorrent.gather` composite body (PyTorch/XLA lowering of `torch.gather(x, 1, index)`) builds a 3-D point-gather with a **global batch iota** in index slot 0:

```mlir
%iota_b = stablehlo.iota dim = 0 : tensor<4xui32>
%bc_b   = stablehlo.broadcast_in_dim %iota_b, dims = [0] : (tensor<4xui32>) -> tensor<4x16x512x1xui32>
%idx    = stablehlo.reshape %user_index : (tensor<4x16x512xui32>) -> tensor<4x16x512x1xui32>
%iota_f = stablehlo.iota dim = 0 : tensor<512xui32>
%bc_f   = stablehlo.broadcast_in_dim %iota_f, dims = [2] : (tensor<512xui32>) -> tensor<4x16x512x1xui32>
%cat    = stablehlo.concatenate %bc_b, %idx, %bc_f, dim = 3 : tensor<4x16x512x3xui32>
%g      = "stablehlo.gather"(%data, %cat) <{
            dimension_numbers = #stablehlo.gather<
              collapsed_slice_dims = [0, 1, 2],
              start_index_map = [0, 1, 2],
              index_vector_dim = 3>,
            slice_sizes = array<i64: 1, 1, 1>}>
          : (tensor<4x32x512xbf16>, tensor<4x16x512x3xui32>) -> tensor<4x16x512xbf16>
```

With `%data` sharded `[{"_axis_0"},{},{}]`, `InsertExplicitReshardsPass` (`StableHLOPipelines.cpp:92`) sees `start_index_map = [0, …]` and demands dim 0 replicated. `ReshardToCollectivesPass` (line 99) lowers the reshard to `sdy.all_gather`, and that collective sits between the index-construction ops and the gather, breaking the `reoutline.group` contiguity required by `ReoutlineCompositePass` (line 112). The composite stays inlined and the test produces low PCC (see `verbose_gather_sharded_index.log` lines 987, 1102 for the two places the all_gather appears).

`torch.gather(x, 1, index)` is batch-parallel on dim 0 — each device only needs its local shard. The all_gather is semantically unnecessary. We add a tt-mlir pass that detects "pass-through" axes in the gather index vector (index column equals `iota(dim=0).broadcast(identity_dim)`) and moves them from `start_index_map` → `operand_batching_dims` / `start_indices_batching_dims`. Shardy treats batching_dims as paired parallel factors → no reshard inserted → composite re-outlines cleanly → TTIR legalization sees only the top-level composite, never the rewritten gather.

## Approach

Add `RewriteBatchParallelGatherPass` that runs **immediately after `FlattenCompositePass`** (before sharding propagation) and rewrites gather ops tagged with `reoutline.group` whose index-vector slots include iota pass-throughs aligned with sharded operand axes.

Rewrite (before → after):
```mlir
// BEFORE
%cat = stablehlo.concatenate %bc_b, %idx, %bc_f, dim = 3
     {reoutline.group = "composite_tenstorrent.gather.impl"} : tensor<4x16x512x3xui32>
%g   = stablehlo.gather(%data, %cat) <{
         collapsed_slice_dims = [0, 1, 2], start_index_map = [0, 1, 2],
         index_vector_dim = 3, slice_sizes = array<i64: 1, 1, 1>}>
       {reoutline.group = "composite_tenstorrent.gather.impl"}

// AFTER
%cat2 = stablehlo.concatenate %idx, %bc_f, dim = 3
      {reoutline.group = "composite_tenstorrent.gather.impl"} : tensor<4x16x512x2xui32>
%g    = stablehlo.gather(%data, %cat2) <{
          operand_batching_dims = [0], start_indices_batching_dims = [0],
          collapsed_slice_dims = [1, 2], start_index_map = [1, 2],
          index_vector_dim = 3, slice_sizes = array<i64: 1, 1, 1>}>
        {reoutline.group = "composite_tenstorrent.gather.impl"}
```

Dead `%iota_b`/`%bc_b` are erased so the reoutline group stays contiguous.

## Files to change

All paths below are under `third_party/tt-mlir/src/tt-mlir/`.

### 1. `include/ttmlir/Dialect/StableHLO/Transforms/Passes.td` — add TableGen def

Append (after the existing `ReoutlineCompositePass` at line 585, before `#endif`):

```tablegen
def RewriteBatchParallelGatherPass : Pass<"rewrite-batch-parallel-gather", "::mlir::ModuleOp"> {
  let summary = "Rewrite batch-parallel axes of stablehlo.gather into operand_batching_dims.";
  let description = [{
    For a `stablehlo.gather` inside a flattened composite body (carrying the
    `reoutline.group` attribute) whose `start_indices` is a concatenation of
    per-axis index columns, detects axes whose column is
    `broadcast_in_dim(iota(dim = 0))` along the identity dim of the index tensor
    (i.e. a "pass-through" axis, not actually being indexed) and moves them from
    `start_index_map` / `collapsed_slice_dims` into `operand_batching_dims` /
    `start_indices_batching_dims`. The corresponding concat slot is dropped.

    This keeps the operand sharding on the pass-through axis valid (Shardy treats
    batching_dims as paired parallel factors) and prevents InsertExplicitReshards
    from inserting a reshard that would later lower to an `sdy.all_gather` and
    break `ReoutlineCompositePass`.

    Only rewrites when:
      - The gather carries `reoutline.group` (composite-scoped).
      - The operand has a non-trivial Shardy sharding on at least one pass-through axis.
      - `start_indices` is defined by `stablehlo.concatenate` along `index_vector_dim`
        with one operand per entry in `start_index_map`.
      - Each pass-through axis is in `collapsed_slice_dims` with `slice_sizes[axis] == 1`.
      - The op's `operand_batching_dims` is currently empty (idempotent).
  }];

  let dependentDialects = [
    "::mlir::stablehlo::StablehloDialect",
    "::mlir::sdy::SdyDialect"
  ];
}
```

### 2. `lib/Dialect/StableHLO/Transforms/RewriteBatchParallelGather.cpp` — new file

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/StableHLOUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::tt::stablehlo {

#define GEN_PASS_DEF_REWRITEBATCHPARALLELGATHERPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

namespace {

// Returns the identity broadcast dim if `v` matches
// `broadcast_in_dim(iota(dim = 0), dims = [k])` and the iota axis size equals
// `expectedSize`. Returns std::nullopt otherwise.
static std::optional<int64_t>
matchIotaPassThrough(Value v, int64_t expectedSize) {
  auto bcast = v.getDefiningOp<mlir::stablehlo::BroadcastInDimOp>();
  if (!bcast) {
    return std::nullopt;
  }
  auto iota = bcast.getOperand().getDefiningOp<mlir::stablehlo::IotaOp>();
  if (!iota || iota.getIotaDimension() != 0) {
    return std::nullopt;
  }
  auto iotaTy = llvm::cast<RankedTensorType>(iota.getType());
  if (iotaTy.getRank() != 1 || iotaTy.getDimSize(0) != expectedSize) {
    return std::nullopt;
  }
  auto bdims = bcast.getBroadcastDimensions();
  if (bdims.size() != 1) {
    return std::nullopt;
  }
  return bdims[0];
}

// Clone every attribute from `src` onto `dst` except those named in `drop`.
static void copyAttributesExcept(Operation *src, Operation *dst,
                                 ArrayRef<StringRef> drop) {
  for (auto &attr : src->getAttrs()) {
    if (llvm::is_contained(drop, attr.getName().strref())) {
      continue;
    }
    // Do not clobber operand-segment-size style attributes already set by the
    // builder.
    if (dst->hasAttr(attr.getName())) {
      continue;
    }
    dst->setAttr(attr.getName(), attr.getValue());
  }
}

struct RewriteBatchParallelGatherPattern
    : public OpRewritePattern<mlir::stablehlo::GatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::stablehlo::GatherOp gatherOp,
                                PatternRewriter &rewriter) const override {
    // (a) Only rewrite inside a flattened composite group.
    auto groupAttr = gatherOp->getAttrOfType<StringAttr>(
        utils::kReoutlineGroupAttr);
    if (!groupAttr) {
      return failure();
    }

    // (b) Idempotent — skip if already rewritten.
    auto dn = gatherOp.getDimensionNumbers();
    if (!dn.getOperandBatchingDims().empty()) {
      return failure();
    }

    // (c) Operand must have a non-trivial Shardy sharding.
    auto meshOp = shardy_utils::getMeshOp(gatherOp->getParentOfType<ModuleOp>());
    if (!meshOp) {
      return failure();
    }
    auto operandSharding = shardy_utils::getOperandShardingAttr(
        gatherOp.getOperation()->getOpOperand(0), meshOp);
    if (!operandSharding ||
        shardy_utils::isFullyReplicatedTensor(operandSharding)) {
      return failure();
    }

    // (d) start_indices must be a concatenate along index_vector_dim with one
    //     operand per entry in start_index_map.
    auto concat = gatherOp.getStartIndices()
                      .getDefiningOp<mlir::stablehlo::ConcatenateOp>();
    if (!concat) {
      return failure();
    }
    if (static_cast<int64_t>(concat.getDimension()) !=
        dn.getIndexVectorDim()) {
      return failure();
    }
    auto startIndexMap = dn.getStartIndexMap();
    if (concat.getInputs().size() != startIndexMap.size()) {
      return failure();
    }

    auto operandTy =
        llvm::cast<RankedTensorType>(gatherOp.getOperand().getType());
    auto sliceSizes = gatherOp.getSliceSizes();
    auto collapsedSliceDims = dn.getCollapsedSliceDims();
    auto dimShardings = operandSharding.getDimShardings();

    // (e) Find pass-through axes.
    SmallVector<int64_t> keepConcatSlots;
    SmallVector<int64_t> newStartIndexMap;
    SmallVector<int64_t> newCollapsedSliceDims;
    SmallVector<int64_t> operandBatchingDims;
    SmallVector<int64_t> startIndicesBatchingDims;

    for (auto [slot, axis] : llvm::enumerate(startIndexMap)) {
      bool isBatchable = [&]() -> bool {
        if (!llvm::is_contained(collapsedSliceDims, axis)) {
          return false;
        }
        if (sliceSizes[axis] != 1) {
          return false;
        }
        if (axis >= static_cast<int64_t>(dimShardings.size())) {
          return false;
        }
        if (dimShardings[axis].getAxes().empty()) {
          return false;
        }
        auto bdim = matchIotaPassThrough(concat.getInputs()[slot],
                                         operandTy.getDimSize(axis));
        if (!bdim) {
          return false;
        }
        operandBatchingDims.push_back(axis);
        startIndicesBatchingDims.push_back(*bdim);
        return true;
      }();

      if (isBatchable) {
        continue;
      }
      keepConcatSlots.push_back(slot);
      newStartIndexMap.push_back(axis);
    }

    if (operandBatchingDims.empty() || keepConcatSlots.empty()) {
      return failure();
    }

    for (int64_t axis : collapsedSliceDims) {
      if (!llvm::is_contained(operandBatchingDims, axis)) {
        newCollapsedSliceDims.push_back(axis);
      }
    }

    // StableHLO requires operand_batching_dims (and start_indices_batching_dims)
    // to be sorted in ascending order; the paired ordering must stay aligned.
    SmallVector<std::pair<int64_t, int64_t>> pairs;
    for (auto [op, idx] :
         llvm::zip(operandBatchingDims, startIndicesBatchingDims)) {
      pairs.emplace_back(op, idx);
    }
    llvm::sort(pairs,
               [](const auto &a, const auto &b) { return a.first < b.first; });
    operandBatchingDims.clear();
    startIndicesBatchingDims.clear();
    for (auto &p : pairs) {
      operandBatchingDims.push_back(p.first);
      startIndicesBatchingDims.push_back(p.second);
    }

    // Build the trimmed concat.
    SmallVector<Value> newConcatInputs;
    for (int64_t slot : keepConcatSlots) {
      newConcatInputs.push_back(concat.getInputs()[slot]);
    }

    Value newIndices;
    if (newConcatInputs.size() == 1) {
      newIndices = newConcatInputs.front();
    } else {
      auto newConcat = rewriter.create<mlir::stablehlo::ConcatenateOp>(
          concat.getLoc(), newConcatInputs, concat.getDimension());
      copyAttributesExcept(concat, newConcat.getOperation(),
                            /*drop=*/{"dimension"});
      newIndices = newConcat.getResult();
    }

    // Build the new dimension numbers.
    auto newDn = mlir::stablehlo::GatherDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*offsetDims=*/dn.getOffsetDims(),
        /*collapsedSliceDims=*/newCollapsedSliceDims,
        /*operandBatchingDims=*/operandBatchingDims,
        /*startIndicesBatchingDims=*/startIndicesBatchingDims,
        /*startIndexMap=*/newStartIndexMap,
        /*indexVectorDim=*/dn.getIndexVectorDim());

    auto newGather = rewriter.create<mlir::stablehlo::GatherOp>(
        gatherOp.getLoc(), gatherOp.getType(), gatherOp.getOperand(),
        newIndices, newDn,
        rewriter.getDenseI64ArrayAttr(sliceSizes),
        gatherOp.getIndicesAreSortedAttr());

    copyAttributesExcept(
        gatherOp.getOperation(), newGather.getOperation(),
        /*drop=*/{"dimension_numbers", "slice_sizes", "indices_are_sorted"});

    rewriter.replaceOp(gatherOp, newGather.getResult());

    // Clean up dead iota/broadcast producers so the reoutline group stays
    // contiguous. Use `use_empty` after the replacement above.
    if (concat->use_empty()) {
      SmallVector<Operation *> deadConcatInputs;
      for (Value v : concat.getInputs()) {
        if (auto *op = v.getDefiningOp()) {
          deadConcatInputs.push_back(op);
        }
      }
      rewriter.eraseOp(concat);
      for (Operation *op : deadConcatInputs) {
        if (op->use_empty()) {
          // bc_b first, then iota_b.
          Value iotaVal;
          if (auto bc = dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op)) {
            iotaVal = bc.getOperand();
          }
          rewriter.eraseOp(op);
          if (iotaVal) {
            if (Operation *iotaOp = iotaVal.getDefiningOp()) {
              if (iotaOp->use_empty()) {
                rewriter.eraseOp(iotaOp);
              }
            }
          }
        }
      }
    }

    return success();
  }
};

struct RewriteBatchParallelGatherPass
    : public impl::RewriteBatchParallelGatherPassBase<
          RewriteBatchParallelGatherPass> {
  using impl::RewriteBatchParallelGatherPassBase<
      RewriteBatchParallelGatherPass>::RewriteBatchParallelGatherPassBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<RewriteBatchParallelGatherPattern>(&getContext());
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                      config))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::stablehlo
```

### 3. `lib/Dialect/StableHLO/Transforms/CMakeLists.txt` — append source

Add `RewriteBatchParallelGather.cpp` to the existing `add_mlir_dialect_library` / `add_library` source list (follow the pattern of `ReoutlineComposite.cpp` already present).

### 4. `lib/Dialect/StableHLO/Pipelines/StableHLOPipelines.cpp` — wire pass

Insert one line after the existing `createFlattenCompositePass()` call at line 62:

```cpp
  // Flatten all composite ops to make sharding propagation easier.
  pm.addPass(createFlattenCompositePass());

  // Rewrite batch-parallel axes of stablehlo.gather so that Shardy does not
  // insert unneeded reshards / collectives inside composite gather bodies.
  pm.addPass(createRewriteBatchParallelGatherPass());

  // Register custom sharding rules for unsupported ops in Shardy.
  pm.addPass(createRegisterCustomShardingRulePass());
```

Rationale: the pass needs `reoutline.group` annotations from `FlattenCompositePass` and must run before `UserPriorityPropagationPass` (line 81) / `InsertExplicitReshardsPass` (line 92) so propagation operates on the corrected gather form.

### 5. `lib/Dialect/StableHLO/Transforms/UpdateGlobalToLocalShapes.cpp` — guard against batching axes

At the top of the loop inside the `GatherOp` handler (currently around line 230, just after the `newSliceSizes` copy at line 220), add a guard so `operand_batching_dims` axes are treated as no-op (slice size already `1`, must not be shrunk by sharding factor):

```cpp
            auto operandBatchingDims =
                gatherOp.getDimensionNumbers().getOperandBatchingDims();

            // 4. For each dimension, update slice size if not in
            // start_index_map.
            for (auto [index, sliceSize] : llvm::enumerate(newSliceSizes)) {
              // Batching dims are paired parallel factors — slice_size is
              // already 1 per StableHLO spec and must stay 1 locally.
              if (llvm::is_contained(operandBatchingDims,
                                     static_cast<int64_t>(index))) {
                continue;
              }
              // If this dimension is collapsed, it must be 1.
              if (llvm::is_contained(collapsedSliceDims, index)) {
                ...
```

(The rest of the loop is unchanged.)

## Edge cases & guards

| Case | Behavior |
|---|---|
| No `reoutline.group` on gather | Skip — composite-scoped only. |
| Operand fully replicated | Skip — no all_gather would be inserted anyway. |
| `indexVectorDim != concat.dimension` | Skip — layout assumption broken. |
| Concat operand count ≠ `|start_index_map|` | Skip. |
| `slice_sizes[axis] != 1` for candidate | Skip that axis (batching spec requires 1). |
| Axis not in `collapsed_slice_dims` | Skip that axis. |
| Iota axis size ≠ `operand_shape[axis]` | Skip (not a true identity). |
| Candidate axis not sharded on operand | Skip (nothing to gain — all_gather wouldn't be inserted). |
| Every operand would become batched (concat empties) | Skip — degenerate. |
| Already has `operand_batching_dims` non-empty | Skip (idempotent). |
| Only leading iota is pass-through, trailing is not | Rewrite leading axis only, leave trailing in `start_index_map`. |

## Verification

### lit test (under `test/ttmlir/Dialect/StableHLO/Transforms/RewriteBatchParallelGather/batch_parallel_gather.mlir`)

```
// RUN: ttmlir-opt --rewrite-batch-parallel-gather --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @positive_sharded_batch
// CHECK: stablehlo.gather
// CHECK-SAME: operand_batching_dims = [0]
// CHECK-SAME: start_indices_batching_dims = [0]
// CHECK-SAME: start_index_map = [1, 2]
// CHECK-SAME: collapsed_slice_dims = [1, 2]
func.func @positive_sharded_batch(
    %data : tensor<4x32x512xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}, {}]>},
    %idx  : tensor<4x16x512x1xui32> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}, {}, {}]>})
    -> tensor<4x16x512xbf16> {
  %iota_b = stablehlo.iota dim = 0 {reoutline.group = "composite_tenstorrent.gather.impl"} : tensor<4xui32>
  %bc_b   = stablehlo.broadcast_in_dim %iota_b, dims = [0] {reoutline.group = "composite_tenstorrent.gather.impl"} : (tensor<4xui32>) -> tensor<4x16x512x1xui32>
  %iota_f = stablehlo.iota dim = 0 {reoutline.group = "composite_tenstorrent.gather.impl"} : tensor<512xui32>
  %bc_f   = stablehlo.broadcast_in_dim %iota_f, dims = [2] {reoutline.group = "composite_tenstorrent.gather.impl"} : (tensor<512xui32>) -> tensor<4x16x512x1xui32>
  %cat    = stablehlo.concatenate %bc_b, %idx, %bc_f, dim = 3 {reoutline.group = "composite_tenstorrent.gather.impl"} : (tensor<4x16x512x1xui32>, tensor<4x16x512x1xui32>, tensor<4x16x512x1xui32>) -> tensor<4x16x512x3xui32>
  %g      = "stablehlo.gather"(%data, %cat) <{
               dimension_numbers = #stablehlo.gather<
                 collapsed_slice_dims = [0, 1, 2],
                 start_index_map = [0, 1, 2],
                 index_vector_dim = 3>,
               slice_sizes = array<i64: 1, 1, 1>}>
             {reoutline.group = "composite_tenstorrent.gather.impl"}
             : (tensor<4x32x512xbf16>, tensor<4x16x512x3xui32>) -> tensor<4x16x512xbf16>
  return %g : tensor<4x16x512xbf16>
}

// -----

// CHECK-LABEL: func.func @negative_no_group
// CHECK-NOT: operand_batching_dims
func.func @negative_no_group(...) { ... }

// -----

// CHECK-LABEL: func.func @negative_replicated_operand
// CHECK-NOT: operand_batching_dims
func.func @negative_replicated_operand(...) { ... }

// -----

// CHECK-LABEL: func.func @partial_only_leading
// CHECK: operand_batching_dims = [0]
// CHECK-SAME: start_index_map = [1, 2]
// CHECK-NOT: start_index_map = [1]
func.func @partial_only_leading(...) { ... }

// -----

// CHECK-LABEL: func.func @idempotent
// CHECK-COUNT-1: stablehlo.gather
// CHECK-SAME: operand_batching_dims = [0]
func.func @idempotent(...) { /* feed already-rewritten IR back in */ }
```

### End-to-end

After rebuilding tt-mlir and tt-xla:

```bash
TTXLA_LOGGER_LEVEL=VERBOSE pytest -svv tests/torch/ops/test_gather.py::test_gather_indices[32-4] \
  2>&1 | tee verbose_gather_32_4_fixed.log
```

Success criteria:
- No `sdy.all_gather` or `stablehlo.all_gather` on the data tensor inside the manual-computation body post-`ReshardToCollectivesPass` (grep the dumped IR).
- `stablehlo.composite "tenstorrent.gather"` present in the `IR Dump After ReoutlineCompositePass` dump.
- PCC check passes.

Regression sweep:
```bash
pytest -svv tests/torch/ops/test_gather.py
```
covers other parametrizations `[128-1]` (batch_size=1, no sharding, predicate (c) skips, no change), `[512-32]` (larger shard, rewrite triggers), etc.

## Open risks

- **Shardy propagation of batching_dims**: if tt-mlir's pinned Shardy version lacks a sharding rule for gather-with-batching-dims, `UserPriorityPropagationPass` may not propagate operand sharding. Fallback: extend `RegisterCustomShardingRulePass` to register a rule declaring batching_dims as parallel factors. Visible from the E2E log as a re-inserted reshard even after our rewrite.
- **TTIR legalization of the rewritten gather**: `StableHLOToTTIR` rejects non-empty batching_dims (`test/ttmlir/Conversion/StableHLOToTTIR/gather/gather_to_embedding_negative.mlir`). This is fine ONLY because `ReoutlineCompositePass` re-outlines the rewritten gather into the private decomposition function and the top-level op becomes `stablehlo.composite "tenstorrent.gather"`, lowered via the composite path. If reoutling ever fails for an edge case, the rewritten gather would escape and legalization would error out — the failure mode is loud, not silent.
- **`UpdateGlobalToLocalShapes` correctness on batching axes**: covered by the one-line guard added in file 5 above; without it, `calculateUpdatedDim` would be called on `slice_size = 1` with a sharded axis and could shrink it incorrectly.

## Critical files (summary)

- `third_party/tt-mlir/src/tt-mlir/include/ttmlir/Dialect/StableHLO/Transforms/Passes.td` — add pass def.
- `third_party/tt-mlir/src/tt-mlir/lib/Dialect/StableHLO/Transforms/RewriteBatchParallelGather.cpp` — **new**.
- `third_party/tt-mlir/src/tt-mlir/lib/Dialect/StableHLO/Transforms/CMakeLists.txt` — register source.
- `third_party/tt-mlir/src/tt-mlir/lib/Dialect/StableHLO/Pipelines/StableHLOPipelines.cpp` — wire after `FlattenCompositePass`.
- `third_party/tt-mlir/src/tt-mlir/lib/Dialect/StableHLO/Transforms/UpdateGlobalToLocalShapes.cpp` — guard against batching axes in gather slice-size update.

Reference patterns: `lib/Dialect/StableHLO/Transforms/StableHLOFusing.cpp:40-85` (OpRewritePattern style), `lib/Dialect/StableHLO/Transforms/UpdateGlobalToLocalShapes.cpp:211-272` (gather attr rebuild), `lib/Dialect/StableHLO/Transforms/ReoutlineComposite.cpp` (group tracking).
