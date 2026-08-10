# Fixing a per-device causal-window bug in `indexer_score_dsa`'s tt-mlir decomposition

Complete record of a tt-mlir compiler bug found while restructuring the DeepSeek-V3.2
Sparse Attention (DSA) indexer/MLA test harness to use the real production
`torch.ops.tt.*` custom ops, and the fix for it.

**Base:** tt-mlir `53f4a3baed` on branch `hshah/all-dsa-ops`.
tt-xla-dsa `7caa516b02` on branch `hshah/dsa-vllm-latest`.
**Scope:** tt-mlir: 2 files changed (+99/-15), 1 new test file.
tt-xla-dsa: 2 files changed (+28).

```
tt-mlir:
 lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp                      | 101 +++++++++++++++---
 lib/Dialect/StableHLO/Transforms/RegisterCustomShardingRule.cpp                 |  13 +++
 test/ttmlir/Conversion/StableHLOToTTIR/transformer/indexer_score_dsa_sharded.mlir | new file
 2 files changed, 99 insertions(+), 15 deletions(-)

tt-xla-dsa:
 integrations/vllm_plugin/vllm_tt/layers/dsa_indexer.py |  12 ++
 python_package/tt_torch/custom_ops.py                  |  16 ++
 2 files changed, 28 insertions(+)
```

---

## 1. How this was found

While verifying that a PyTorch-level test harness for DeepSeek-V3.2's MLA attention and
DSA indexer genuinely exercises the same `torch.ops.tt.*` custom ops the real vLLM plugin
path uses (rather than a hand-rolled stand-in), the harness was restructured to call
`torch.ops.tt.indexer_score_dsa` → `torch.ops.tt.topk_large_indices` →
`torch.ops.tt.sparse_sdpa` directly, on the real 32-device Blackhole Galaxy mesh
(`mesh_shape=(4,8)`, axes `("batch","model")`).

A whole-model PCC comparison (CPU golden vs. compiled TT device) on a 2-layer
dense+MoE transformer showed the DSA sparse-attention path landing at **PCC 0.9119**,
markedly worse than the equivalent dense-attention path (**PCC 0.9671**) run through the
exact same harness. Comparing final-model-output PCC alone couldn't say whether the
scoring, the top-k selection, or something downstream was responsible, so a dedicated
comparator was built for just the indexer (mirroring
`tests/torch/ops/test_topk.py`'s `topk_both_comparator`): PCC on the raw
`indexer_score_dsa` score tensor, plus a separate check that gathers each side's own score
using its own top-k indices and compares the gathered values (not the raw indices, since
top-k ordering is not required to be identical).

That comparator isolated the fault immediately:

```
indexer_score_dsa PCC: 1.0                                    <- scoring itself is fine
golden valid-key-count per query row (32-row chunks): [128, 4224, 8192, 8192]
device valid-key-count per query row (32-row chunks): [128,  128,  128,  128]
```

Golden (CPU) matches the expected causal formula `min(index_topk, row+1) * batch_size`
exactly. The device number is **constant** at every sampled row -- every 32-row shard
(the mesh's `"model"` axis has 8 devices, so the padded 256-row query sequence splits into
8 shards of 32 rows each) was computing the *same* causal window row 0's shard would
compute. Row-by-row, 96 of 128 rows were wrong; only shard 0 (rows 0-31) was correct, and
only because shard 0's local row range happens to equal the global one.

## 2. Root cause

`indexer_score_dsa` shards its query sequence across a named mesh axis (`cluster_axis`) on
a multi-device mesh; each device's causal window should start at
`chunk_start_idx + rank * Sq`, where `rank` is that device's coordinate along
`cluster_axis` and `Sq` is the per-device chunk length. There are two independent lowering
paths for this op, and only one of them gets the rank right:

- **The promoted `ttnn.indexer_score_dsa` kernel** (tt-metal,
  `indexer_score_program_factory.cpp`) is instantiated **once per mesh coordinate**
  (`create_mesh_workload()`), and each per-coordinate program bakes in its own
  `chunk_start_idx + rank*Sq` as a runtime argument
  (`device_causal_geometry()` / `device_index_for()`,
  using `ttnn::ccl::get_linearized_index_from_physical_coord(coord, cluster_axis)`). This
  is correct, and has existing multi-device test coverage in
  `tests/ttnn/nightly/unit_tests/operations/experimental/test_indexer_score.py`.

- **`TTNNResolveComposites`'s decomposition fallback** -- used whenever the promoted
  kernel isn't available, which is the *default* at `optimization_level=0`
  (`composite-resolution=auto` degrades to `inline` unless the optimizer pass is enabled;
  see `TTNNPipelines.cpp`), and always for batch > 1 regardless of optimization level
  (`IndexerScoreDsaOp::verify()` rejects batch != 1) -- is a **single MLIR region shared
  across every device**, with no per-coordinate specialization. Before this fix,
  `buildIndexerScoreDsaDecompositionBody` (in
  `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`) built its causal-mask row
  index from a fresh, device-**local** `ttir.arange(0, Sq)`. Because this decomposition is
  synthesized during `StableHLOToTTIR` conversion, which runs *after* Shardy's
  `UpdateGlobalToLocalShapes` pass has already shrunk every sharded tensor's shape down to
  its local size, "local row 0" is indistinguishable from "global row 0" at that point --
  every device numbered its rows `0..Sq-1` regardless of which shard of the global
  sequence it actually held. `cluster_axis` was parsed off the custom call's
  `mhlo.frontend_attributes` and attached to the `ttcore.composite` (for the promoted
  path to use later), but the decomposition builder never received it and never used it.

The result: correct on a single device, correct on the promoted kernel, and silently
wrong -- every shard computing rank 0's causal window -- whenever the decomposition
fallback ran on a sharded query sequence. No existing tt-mlir test exercised this
combination (`cluster_axis` set *and* the decomposition path, rather than the promoted
kernel); the closest existing test,
`test/ttmlir/Dialect/TTNN/transformer/indexer_score_dsa_decomposition_bsz.mlir`, exercises
the decomposition but never sets `cluster_axis`.

## 3. The fix

### 3.1 Design

The fix needs the decomposition -- a single MLIR region -- to recover each device's rank
along `cluster_axis` and use it. There is no existing "give me my rank as an SSA value"
primitive in tt-mlir usable from a shared region (the closest things,
`d2m.mesh_position` / `ttkernel.experimental.get_my_logical_mesh_position`, are D2M/TTKernel
device-code-generation ops, not reachable from a TTIR/TTNN decomposition). Inventing one
would mean a new op threaded through TTIR, TTNN, the flatbuffer schema, the runtime, and
both EmitC/EmitPy emitters.

Instead, the fix reuses **`ttir.mesh_partition`**, an existing, already fully-plumbed op
(TTIR → TTNN → flatbuffer → runtime → EmitC/EmitPy, with existing lit and golden tests)
that does the semantically equivalent thing: given a **global** tensor, it returns just
this device's slice along a named `cluster_axis`. Its lowering
(`ttnn::mesh_partition`, tt-metal's `mesh_partition_program_factory.cpp`) derives the
per-device slice offset via the exact same rank convention
(`get_linearized_index(coord, mesh_view)` / `mesh_coordinate[cluster_axis]`) the promoted
indexer kernel already uses -- so results from the two paths agree.

tt-mlir already fixed the identical class of bug for sharded `stablehlo.iota`
(`ReplicateNonSplittableValues.cpp`: replicate the iota, then let Shardy's ordinary
`InsertExplicitReshards` lower the resulting reshard to `sdy.all_slice`, which itself
legalizes to `ttir.mesh_partition`). This fix applies the same idea by hand, directly at
the one call site that needs it:

1. Build a row-index `ttir.arange` over the **global** query-sequence length
   (`querySeqLen * numDevices`), narrowed to a small `[1,1,G,1]` column rather than the
   full `[B,Hi,G,T]` tensor.
2. `ttir.mesh_partition` it on the sequence dim, naming `cluster_axis`, down to this
   device's own `[1,1,Sq,1]` window.
3. Broadcast that column back out to the score's `[B,1,Sq,T]` shape.

This needs `numDevices` (how many shards the global sequence was split into) at the point
the decomposition is built. The op's own `cluster_axis` attribute names *which axis*, but
not *how many devices are on it*. Rather than looking up the module's mesh shape at
compile time (fragile: within the single `StableHLOToTTIR` conversion pass, the `sdy.mesh`
op and the `ttcore.meshes` module attribute that replaces it are populated by a sibling
pattern in the same greedy rewrite, so which one (if either) is available when the
indexer's own pattern runs is not guaranteed), the fix threads `num_devices` through as an
explicit frontend attribute from Python -- mirroring the `all_to_all_dispatch` /
`moe_expert_token_remap` composites, which already pass their own `num_devices` this way.

`numDevices == 1` (the default, and the value whenever `cluster_axis` is unset) reduces
exactly to the original local arange, so single-device and unsharded behavior is
unchanged.

### 3.2 `python_package/tt_torch/custom_ops.py`

Adds `num_devices` to the `indexer_score_dsa` custom op wrapper (and its `_fake` meta
registration), carried into `mhlo.frontend_attributes` alongside `chunk_start_idx` /
`cluster_axis`. `torch.library.custom_op` infers the op schema from the Python type
hints, so no separate schema string needs updating.

```diff
@@ -691,6 +691,7 @@ def indexer_score_dsa(
     weights: torch.Tensor,
     chunk_start_idx: int = 0,
     cluster_axis: Optional[int] = None,
+    num_devices: int = 1,
 ) -> torch.Tensor:
     """
     DeepSeek Sparse Attention (DSA) "lightning indexer" scorer, mirroring
@@ -718,6 +719,16 @@ def indexer_score_dsa(
             the rank derivation exact and leaves the other axes free.
         chunk_start_idx: absolute position of the first query token, i.e. the
                          number of already-cached tokens. Compile-time constant.
+        num_devices: number of devices along ``cluster_axis`` -- i.e. how many
+            equal chunks the (already locally-sized) ``query`` sequence dim was
+            split into. Only meaningful when ``cluster_axis`` is set; mirrors
+            ``all_to_all_dispatch``'s ``num_devices`` argument. Needed because
+            when the promoted TTNN kernel isn't available, tt-mlir's fallback
+            decomposition runs on already-per-device-local shapes and has no
+            other way to recover how many shards the global sequence was split
+            into, which it needs to reconstruct each device's true causal
+            window (``local_row + chunk_start_idx + rank*local_seq_len``)
+            instead of every device's window collapsing to rank 0's.
     Returns:
         Scores of shape [b, 1, sq, t] with the same dtype as ``query``.
     """
@@ -757,6 +768,9 @@ def indexer_score_dsa(
     assert cluster_axis is None or (
         isinstance(cluster_axis, int) and cluster_axis >= 0
     ), f"cluster_axis must be None or a non-negative int, got {cluster_axis}."
+    assert (
+        isinstance(num_devices, int) and num_devices >= 1
+    ), f"num_devices must be a positive int, got {num_devices}."

     if batch != 1:
         _warn_no_kernel("indexer_score_dsa", f"batch size is {batch}, must be 1")
@@ -776,6 +790,7 @@ def indexer_score_dsa(
                 else {
                     "chunk_start_idx": str(chunk_start_idx),
                     "cluster_axis": str(cluster_axis),
+                    "num_devices": str(num_devices),
                 }
             ),
         )
@@ -812,6 +827,7 @@ def indexer_score_dsa_fake(
     weights: torch.Tensor,
     chunk_start_idx: int = 0,
     cluster_axis: Optional[int] = None,
+    num_devices: int = 1,
 ) -> torch.Tensor:
     return torch.zeros(
         (query.shape[0], 1, query.shape[2], key.shape[2]),
```

### 3.3 `integrations/vllm_plugin/vllm_tt/layers/dsa_indexer.py`

The production caller (`TTIndexer._forward_prefill`) already computes everything needed;
it just wasn't passing it through. `cluster_axis` comes from `_prefill_seq_shard_plan`;
`num_devices` is that axis's size in the mesh.

```diff
@@ -569,6 +569,17 @@ class TTIndexer(nn.Module):
                 visible_count, self._mesh, seq_spec
             )

+        # Number of devices q's sequence dim was actually split into. Needed by
+        # tt-mlir's decomposition fallback (used whenever the promoted TTNN
+        # kernel isn't available): that fallback runs on already-per-device-
+        # local shapes and has no other way to recover how many shards the
+        # global sequence was split into, which it needs to reconstruct each
+        # device's true causal window instead of every device silently
+        # collapsing to rank 0's.
+        num_devices = (
+            self._mesh.mesh_shape[cluster_axis] if cluster_axis is not None else 1
+        )
+
         # Both DSA ops require batch == 1, so score/select one user at a time.
         per_user = []
         for u in range(users):
@@ -578,6 +589,7 @@ class TTIndexer(nn.Module):
                 weights=w_op[u : u + 1],
                 chunk_start_idx=0,
                 cluster_axis=cluster_axis,
+                num_devices=num_devices,
             )  # [1, 1, padded_len/model_size, padded_len]
             per_user.append(self._select(score, visible_count))
         indices = torch.cat(per_user, dim=0)  # [users, 1, padded_len, topk]
```

### 3.4 `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` -- the actual fix

`buildIndexerScoreDsaDecompositionBody` gains `clusterAxis` / `numDevices` parameters and
uses them to build a sharding-aware row index instead of the old device-local one:

```diff
@@ -9558,10 +9558,10 @@ namespace {
 //   masked to -inf where t > chunk_start_idx + s.
 //
 // q [B, Hi, Sq, D], k [B, 1, T, D], weights [B, Hi, Sq, 1] -> [B, 1, Sq, T].
-static Value
-buildIndexerScoreDsaDecompositionBody(ConversionPatternRewriter &rewriter,
-                                      Location loc, Value query, Value key,
-                                      Value weights, uint32_t chunkStartIdx) {
+static Value buildIndexerScoreDsaDecompositionBody(
+    ConversionPatternRewriter &rewriter, Location loc, Value query, Value key,
+    Value weights, uint32_t chunkStartIdx,
+    std::optional<uint32_t> clusterAxis, uint32_t numDevices) {
   auto queryType = mlir::cast<RankedTensorType>(query.getType());
   auto keyType = mlir::cast<RankedTensorType>(key.getType());
   ArrayRef<int64_t> qShape = queryType.getShape();
@@ -9641,13 +9641,59 @@ buildIndexerScoreDsaDecompositionBody(ConversionPatternRewriter &rewriter,
   // beyond bf16's exact-integer range (256) are not conflated --
   // chunk_start_idx pushes the compared magnitudes well past that for long DSA
   // contexts.
-  auto indexType = RankedTensorType::get({batch, 1, querySeqLen, keySeqLen},
-                                         rewriter.getI32Type(), encoding);
-  Value rowIdx = rewriter
-                     .create<ttir::ArangeOp>(loc, indexType, /*start=*/0,
-                                             /*end=*/querySeqLen, /*step=*/1,
-                                             /*arange_dimension=*/2)
-                     .getResult();
+  Type i32Type = rewriter.getI32Type();
+  auto indexType =
+      RankedTensorType::get({batch, 1, querySeqLen, keySeqLen}, i32Type, encoding);
+
+  // `query` (and hence `querySeqLen`) is already per-device-local: this
+  // decomposition is synthesized during StableHLOToTTIR conversion, which runs
+  // after Shardy's UpdateGlobalToLocalShapes has shrunk every sharded shape
+  // down to its local size. A plain arange(0, querySeqLen) therefore numbers
+  // every device's rows 0..querySeqLen-1 regardless of which shard of the
+  // global query sequence this device actually holds -- every device ends up
+  // computing rank 0's causal window. The promoted ttnn.indexer_score_dsa
+  // kernel avoids this because it is instantiated once per mesh coordinate
+  // (indexer_score_program_factory.cpp's create_mesh_workload) and bakes each
+  // device's true `chunk_start_idx + rank*Sq` in as a runtime arg; this shared
+  // MLIR region has no equivalent per-device specialization, so the rank has
+  // to be recovered explicitly via ttir.mesh_partition instead: partition a
+  // GLOBAL row-index arange the same way the query sequence itself was
+  // partitioned, and each device gets back exactly its own `[rank*Sq,
+  // (rank+1)*Sq)` window. When numDevices == 1 (no sequence sharding, or the
+  // composite's cluster_axis wasn't set) this reduces to the original local
+  // arange, so single-device behavior is unchanged.
+  Value rowIdx;
+  if (clusterAxis.has_value() && numDevices > 1) {
+    int64_t globalQuerySeqLen = querySeqLen * static_cast<int64_t>(numDevices);
+    auto globalColType =
+        RankedTensorType::get({1, 1, globalQuerySeqLen, 1}, i32Type, encoding);
+    Value globalRows = rewriter
+                            .create<ttir::ArangeOp>(loc, globalColType,
+                                                    /*start=*/0,
+                                                    /*end=*/globalQuerySeqLen,
+                                                    /*step=*/1,
+                                                    /*arange_dimension=*/2)
+                            .getResult();
+    auto localColType =
+        RankedTensorType::get({1, 1, querySeqLen, 1}, i32Type, encoding);
+    Value localRows =
+        rewriter
+            .create<ttir::MeshPartitionOp>(
+                loc, localColType, globalRows, rewriter.getSI32IntegerAttr(2),
+                rewriter.getUI32IntegerAttr(*clusterAxis))
+            .getResult();
+    rowIdx = rewriter
+                 .create<ttir::BroadcastOp>(
+                     loc, indexType, localRows,
+                     SmallVector<int64_t>{batch, 1, 1, keySeqLen})
+                 .getResult();
+  } else {
+    rowIdx = rewriter
+                 .create<ttir::ArangeOp>(loc, indexType, /*start=*/0,
+                                         /*end=*/querySeqLen, /*step=*/1,
+                                         /*arange_dimension=*/2)
+                 .getResult();
+  }
   Value colIdx = rewriter
                      .create<ttir::ArangeOp>(loc, indexType, /*start=*/0,
                                              /*end=*/keySeqLen, /*step=*/1,
```

The caller (`StableHLOToTTCoreIndexerScoreDsaOpConversionPattern::matchAndRewrite`) parses
the new `num_devices` frontend attribute the same way it already parses `chunk_start_idx`
and `cluster_axis`, and forwards both to the decomposition builder:

```diff
@@ -9733,18 +9779,42 @@ public:
     // sequence is sharded across every device. Naming the axis is what makes a
     // partial split (e.g. heads on one mesh axis, sequence on another) correct.
     IntegerAttr clusterAxisAttr;
+    std::optional<uint32_t> clusterAxis;
     if (auto frontendAttributes = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
             srcOp->getDiscardableAttr("mhlo.frontend_attributes"))) {
       if (auto clusterAxisStringAttr =
               frontendAttributes.getAs<mlir::StringAttr>("cluster_axis")) {
-        uint32_t clusterAxis = 0;
-        if (!llvm::to_integer(clusterAxisStringAttr.getValue(), clusterAxis)) {
+        uint32_t clusterAxisValue = 0;
+        if (!llvm::to_integer(clusterAxisStringAttr.getValue(),
+                              clusterAxisValue)) {
           return rewriter.notifyMatchFailure(
               srcOp, "cluster_axis attribute must be a non-negative integer. "
                      "Received \"" +
                          clusterAxisStringAttr.getValue() + "\".");
         }
-        clusterAxisAttr = rewriter.getUI32IntegerAttr(clusterAxis);
+        clusterAxisAttr = rewriter.getUI32IntegerAttr(clusterAxisValue);
+        clusterAxis = clusterAxisValue;
+      }
+    }
+
+    // num_devices is optional and defaults to 1 (no sequence sharding). Only
+    // consumed here, to let the decomposition fallback (below) reconstruct
+    // each device's causal window when the promoted ttnn.indexer_score_dsa
+    // kernel isn't used -- the promoted kernel derives its own per-device
+    // rank from the device's mesh coordinates and does not need this. Mirrors
+    // all_to_all_dispatch's num_devices attribute.
+    uint32_t numDevices = 1;
+    if (auto frontendAttributes = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
+            srcOp->getDiscardableAttr("mhlo.frontend_attributes"))) {
+      if (auto numDevicesStringAttr =
+              frontendAttributes.getAs<mlir::StringAttr>("num_devices")) {
+        if (!llvm::to_integer(numDevicesStringAttr.getValue(), numDevices) ||
+            numDevices < 1) {
+          return rewriter.notifyMatchFailure(
+              srcOp, "num_devices attribute must be a positive integer. "
+                     "Received \"" +
+                         numDevicesStringAttr.getValue() + "\".");
+        }
       }
     }

@@ -9777,7 +9847,8 @@ public:

       Value decompResult = buildIndexerScoreDsaDecompositionBody(
           rewriter, srcOp.getLoc(), entry->getArgument(0),
-          entry->getArgument(1), entry->getArgument(2), chunkStartIdx);
+          entry->getArgument(1), entry->getArgument(2), chunkStartIdx,
+          clusterAxis, numDevices);
       rewriter.create<mlir::func::ReturnOp>(srcOp.getLoc(), decompResult);
     }
```

`num_devices` is deliberately **not** added to the `ttcore.composite`'s carried
attributes (`compositeAttrList`): it is consumed only while building the decomposition
body, inside this same pattern match, so it has no reason to survive as a persistent
IR attribute -- the promoted kernel path doesn't need it at all.

#### Full source of the modified function, post-fix

```cpp
// q [B, Hi, Sq, D], k [B, 1, T, D], weights [B, Hi, Sq, 1] -> [B, 1, Sq, T].
static Value buildIndexerScoreDsaDecompositionBody(
    ConversionPatternRewriter &rewriter, Location loc, Value query, Value key,
    Value weights, uint32_t chunkStartIdx,
    std::optional<uint32_t> clusterAxis, uint32_t numDevices) {
  auto queryType = mlir::cast<RankedTensorType>(query.getType());
  auto keyType = mlir::cast<RankedTensorType>(key.getType());
  ArrayRef<int64_t> qShape = queryType.getShape();

  int64_t batch = qShape[0];
  int64_t numHeads = qShape[1];
  int64_t querySeqLen = qShape[2];
  int64_t headDim = qShape[3];
  int64_t keySeqLen = keyType.getShape()[2];

  Type elemType = queryType.getElementType();
  Attribute encoding = queryType.getEncoding();

  auto tensorType = [&](ArrayRef<int64_t> shape) {
    return RankedTensorType::get(shape, elemType, encoding);
  };

  // Fold the query heads into the sequence dim so a single batched matmul
  // against K's single kv-head works without broadcasting K across heads.
  Value qFold =
      ttir::utils::createReshapeOp(rewriter, loc, query,
                                   {batch, 1, numHeads * querySeqLen, headDim})
          .getResult();

  // K^T: [B, 1, T, D] -> [B, 1, D, T].
  Value keyT = rewriter
                   .create<ttir::PermuteOp>(
                       loc, tensorType({batch, 1, headDim, keySeqLen}), key,
                       rewriter.getDenseI64ArrayAttr({0, 1, 3, 2}))
                   .getResult();

  // QK^T (grouped form), then unfold heads: [B, Hi, Sq, T].
  Value qkFold =
      rewriter
          .create<ttir::MatmulOp>(
              loc, tensorType({batch, 1, numHeads * querySeqLen, keySeqLen}),
              qFold, keyT)
          .getResult();
  Value qk =
      ttir::utils::createReshapeOp(rewriter, loc, qkFold,
                                   {batch, numHeads, querySeqLen, keySeqLen})
          .getResult();

  // relu(QK^T).
  Value qkRelu =
      rewriter
          .create<ttir::ReluOp>(
              loc, tensorType({batch, numHeads, querySeqLen, keySeqLen}), qk)
          .getResult();

  // Multiply by the per-head gate weights, broadcast over the key dim.
  Value weightsBcast =
      rewriter
          .create<ttir::BroadcastOp>(
              loc, tensorType({batch, numHeads, querySeqLen, keySeqLen}),
              weights, SmallVector<int64_t>{1, 1, 1, keySeqLen})
          .getResult();
  Value weighted =
      rewriter
          .create<ttir::MultiplyOp>(
              loc, tensorType({batch, numHeads, querySeqLen, keySeqLen}),
              qkRelu, weightsBcast)
          .getResult();

  // Sum over the head dim: [B, 1, Sq, T].
  auto scoreType = tensorType({batch, 1, querySeqLen, keySeqLen});
  Value score =
      rewriter
          .create<ttir::SumOp>(loc, scoreType, weighted,
                               rewriter.getBoolAttr(/*keep_dim=*/true),
                               rewriter.getI32ArrayAttr({1}))
          .getResult();

  // Causal mask: visible iff key index t <= chunk_start_idx + query index s.
  // Future positions get an additive -inf. The index arithmetic and the
  // comparison run in i32 (not the query element type) so that positions
  // beyond bf16's exact-integer range (256) are not conflated --
  // chunk_start_idx pushes the compared magnitudes well past that for long DSA
  // contexts.
  Type i32Type = rewriter.getI32Type();
  auto indexType =
      RankedTensorType::get({batch, 1, querySeqLen, keySeqLen}, i32Type, encoding);

  // `query` (and hence `querySeqLen`) is already per-device-local: this
  // decomposition is synthesized during StableHLOToTTIR conversion, which runs
  // after Shardy's UpdateGlobalToLocalShapes has shrunk every sharded shape
  // down to its local size. A plain arange(0, querySeqLen) therefore numbers
  // every device's rows 0..querySeqLen-1 regardless of which shard of the
  // global query sequence this device actually holds -- every device ends up
  // computing rank 0's causal window. The promoted ttnn.indexer_score_dsa
  // kernel avoids this because it is instantiated once per mesh coordinate
  // (indexer_score_program_factory.cpp's create_mesh_workload) and bakes each
  // device's true `chunk_start_idx + rank*Sq` in as a runtime arg; this shared
  // MLIR region has no equivalent per-device specialization, so the rank has
  // to be recovered explicitly via ttir.mesh_partition instead: partition a
  // GLOBAL row-index arange the same way the query sequence itself was
  // partitioned, and each device gets back exactly its own `[rank*Sq,
  // (rank+1)*Sq)` window. When numDevices == 1 (no sequence sharding, or the
  // composite's cluster_axis wasn't set) this reduces to the original local
  // arange, so single-device behavior is unchanged.
  Value rowIdx;
  if (clusterAxis.has_value() && numDevices > 1) {
    int64_t globalQuerySeqLen = querySeqLen * static_cast<int64_t>(numDevices);
    auto globalColType =
        RankedTensorType::get({1, 1, globalQuerySeqLen, 1}, i32Type, encoding);
    Value globalRows = rewriter
                            .create<ttir::ArangeOp>(loc, globalColType,
                                                    /*start=*/0,
                                                    /*end=*/globalQuerySeqLen,
                                                    /*step=*/1,
                                                    /*arange_dimension=*/2)
                            .getResult();
    auto localColType =
        RankedTensorType::get({1, 1, querySeqLen, 1}, i32Type, encoding);
    Value localRows =
        rewriter
            .create<ttir::MeshPartitionOp>(
                loc, localColType, globalRows, rewriter.getSI32IntegerAttr(2),
                rewriter.getUI32IntegerAttr(*clusterAxis))
            .getResult();
    rowIdx = rewriter
                 .create<ttir::BroadcastOp>(
                     loc, indexType, localRows,
                     SmallVector<int64_t>{batch, 1, 1, keySeqLen})
                 .getResult();
  } else {
    rowIdx = rewriter
                 .create<ttir::ArangeOp>(loc, indexType, /*start=*/0,
                                         /*end=*/querySeqLen, /*step=*/1,
                                         /*arange_dimension=*/2)
                 .getResult();
  }
  Value colIdx = rewriter
                     .create<ttir::ArangeOp>(loc, indexType, /*start=*/0,
                                             /*end=*/keySeqLen, /*step=*/1,
                                             /*arange_dimension=*/3)
                     .getResult();
  Value chunkStartConst =
      rewriter
          .create<ttir::FullOp>(
              loc, indexType,
              rewriter.getI32IntegerAttr(static_cast<int32_t>(chunkStartIdx)))
          .getResult();
  Value threshold =
      rewriter.create<ttir::AddOp>(loc, indexType, rowIdx, chunkStartConst)
          .getResult();
  Value visibleBool =
      rewriter.create<ttir::GreaterEqualOp>(loc, indexType, threshold, colIdx)
          .getResult();
  Value zeros =
      rewriter
          .create<ttir::FullOp>(loc, scoreType, rewriter.getF32FloatAttr(0.0f))
          .getResult();
  Value negInf =
      rewriter
          .create<ttir::FullOp>(
              loc, scoreType,
              rewriter.getF32FloatAttr(-std::numeric_limits<float>::infinity()))
          .getResult();
  Value maskAdd =
      rewriter.create<ttir::WhereOp>(loc, scoreType, visibleBool, zeros, negInf)
          .getResult();
  return rewriter.create<ttir::AddOp>(loc, scoreType, score, maskAdd)
      .getResult();
}
```

### 3.5 Considered and REJECTED: `TTCore_NonCacheableTrait` on `MeshPartitionOp`

An earlier revision of this fix also added `TTCore_NonCacheableTrait` to
`TTIR_MeshPartitionOp` / `TTNN_MeshPartitionOp` (in `TTIROps.td` / `TTNNOps.td`).
The reasoning was: `mesh_partition` has zero operands that vary across devices (the
global arange feeding it is identical everywhere) but a result that genuinely differs
per device, so `ConstEvalHoist`'s `operandConstEval` check passes vacuously for the
zero-operand chain and a *host*-side const-eval hoist could collapse it to one value
shared by every device -- silently reintroducing this very bug through a different door.

**That trait was wrong on both counts and has been reverted.** It is not part of this
fix. Do not re-add it.

**It was redundant.** Upstream tt-mlir already keeps `mesh_partition` off the CPU:
`lib/Dialect/TTIR/Transforms/HoistCPUOps/CPUHoistConstEval.cpp:88-92` lists it as a
hoisting barrier --

```cpp
// Check if an op is a barrier for CPU hoisting - CCL and MeshShard ops must
// remain on device and split the subgraph into segments.
static bool isBarrierOp(mlir::Operation *op) {
  return mlir::isa<CCL>(op) || mlir::isa<MeshShardOp, MeshPartitionOp>(op);
}
```

-- added by upstream `ff99051e48` ("[TTNN] Enable multi-chip const-eval CPU-hoisting",
#8474). So the only genuinely dangerous hoist was already prevented. Note
`enable_const_eval_on_cpu` defaults to **true** in tt-xla
(`pjrt_implementation/inc/api/compile_options.h:110`), so this barrier is load-bearing
rather than hypothetical -- it just is not ours to add.

**It was actively harmful.** The trait's only consumer is `ConstEvalHoist.cpp:205`,
which excludes the op from **device-side** const-eval as well -- where hoisting is
perfectly safe (the cached value is a mesh tensor, so per-device values are preserved).
Because that pass tracks hoistable values in a union-find (`dsu.valueExists`), excluding
`mesh_partition` also un-hoists everything *downstream* of it, while the upstream prefix
still gets hoisted and cached. And `mesh_partition` is the **size-reducing** op -- it
slices a global tensor down to one device's shard -- so the const-eval boundary landed
*before* the slice and the persistent cached tensors became the pre-slice,
`num_devices`x larger versions.

Measured on the 4-layer slice at the full model's mesh/opt/quant config
(`test_dsa_v32_4layer_ccl_ir_diagnostic`, added for exactly this purpose):

| | trait ON | trait reverted |
|---|---|---|
| `mesh_partition` ops inside const-eval funcs | 0 of 75 | 69 of 75 |
| const-eval function count | 479 | 270 |
| const-eval cached bytes, graph g1 / g3 / g7 | 746.1 / 746.1 / 746.1 MB | 555.4 / 555.4 / 555.4 MB |
| **total const-eval cached bytes** | **2255.0 MB** | **1683.0 MB** |

i.e. **+572 MB of persistent DRAM at four layers**, scaling per-layer. Generated output
was bit-identical either way (same `token_ids`), so this was purely a memory regression.
At the real 61 layers it exhausted device DRAM and broke
`test_tensor_parallel_generation_deepseek_v32_full` -- see 4.4.

### 3.6 `lib/Dialect/StableHLO/Transforms/RegisterCustomShardingRule.cpp`

`getIndexerScoreDsaShardingRule`'s comment used to claim "each device's causal window
starts at chunk_start_idx + rank*Sq ... handled by the op itself" as an unqualified fact.
That was only ever true of the promoted kernel; updated to say so explicitly and point at
the fix.

```diff
@@ -549,6 +549,19 @@ getFlashMlaPrefillShardingRule(mlir::stablehlo::CustomCallOp op) {
 //       produces, matching the op's cluster_axis=None flat linearization) masks
 //       correctly on every device. Any other split would not -- see the caveat
 //       in docs/dsa_blackhole_tt-mlir_changes.md.
+//
+//       This "the op itself handles it" guarantee holds for the PROMOTED
+//       ttnn.indexer_score_dsa kernel (instantiated once per mesh coordinate,
+//       each baking in its own chunk_start_idx + rank*Sq). It does NOT
+//       automatically hold for TTNNResolveComposites' fallback decomposition,
+//       which is a single MLIR region shared across every device and, absent
+//       explicit correction, would compute every shard's causal window as if
+//       it were rank 0. buildIndexerScoreDsaDecompositionBody
+//       (StableHLOToTTIRPatterns.cpp) recovers the rank itself via
+//       ttir.mesh_partition over a global row-index arange, so the guarantee
+//       is restored on the decomposition path too -- but only when the caller
+//       passes num_devices > 1 alongside cluster_axis; num_devices == 1
+//       (the default) reduces to the original, offset-free local arange.
 //   - Key seq   (kNeedReplication, size T) : key dim 2, out dim 3. Still cannot
 //       be sharded: every query row scores against all T keys, and the op reads
 //       T to derive the per-rank window.
```

### 3.7 New test: `test/ttmlir/Conversion/StableHLOToTTIR/transformer/indexer_score_dsa_sharded.mlir`

The existing `indexer_score_dsa.mlir` / `indexer_score_dsa_decomposition_bsz.mlir` tests
never set `cluster_axis`, so neither exercised this path. This new file does, and
FileChecks for the `ttir.mesh_partition` + `ttir.broadcast` sequence in the synthesized
decomposition:

```mlir
// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// Query-sequence-sharded indexer_score_dsa: cluster_axis + num_devices are set
// (num_devices > 1), so the composite's synthesized decomposition -- the
// fallback TTNNResolveComposites inlines whenever the promoted
// ttnn.indexer_score_dsa kernel isn't used -- must recover each device's true
// causal-window offset itself, since it is a single MLIR region shared across
// every device with no per-device specialization of its own (unlike the
// promoted kernel, which is instantiated once per mesh coordinate). It does
// this via ttir.mesh_partition over a global row-index arange rather than a
// device-local one; see StableHLOToTTIRPatterns.cpp's
// buildIndexerScoreDsaDecompositionBody and the bug this fixes
// (docs/dsa_blackhole_tt-mlir_changes.md, RegisterCustomShardingRule.cpp's
// getIndexerScoreDsaShardingRule comment).

module @indexer_score_dsa_sharded attributes {} {
  // Local query seq = 32 (one of num_devices=4 shards of a global 128-row
  // query sequence sharded over cluster_axis=1); key stays at the full,
  // unsharded 128.
  func.func public @indexer_score_dsa_sharded(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x1x128x128xbf16>, %w: tensor<1x8x32x1xbf16>) -> tensor<1x1x32x128xbf16> {
    // CHECK-LABEL: @indexer_score_dsa_sharded
    // CHECK: "ttcore.composite"(%arg0, %arg1, %arg2)
    // CHECK-SAME: chunk_start_idx = 0 : ui32
    // CHECK-SAME: cluster_axis = 1 : ui32
    // CHECK-SAME: composite_name = "indexer_score_dsa"
    // CHECK-SAME: decomposition = @indexer_score_dsa_decomp
    %0 = stablehlo.custom_call @tt.indexer_score_dsa(%q, %k, %w) {api_version = 0 : i32, mhlo.frontend_attributes = {chunk_start_idx = "0", cluster_axis = "1", num_devices = "4"}} : (tensor<1x8x32x128xbf16>, tensor<1x1x128x128xbf16>, tensor<1x8x32x1xbf16>) -> tensor<1x1x32x128xbf16>
    return %0 : tensor<1x1x32x128xbf16>
  }

  // The synthesized decomposition still holds the full primitive lowering
  // (QK^T, relu, gate multiply, head-sum, causal mask) -- only the row-index
  // construction differs from the unsharded case.
  // CHECK: func.func private @indexer_score_dsa_decomp
  // CHECK: "ttir.matmul"
  // CHECK: "ttir.relu"
  // CHECK: "ttir.sum"
  //
  // Row index: a GLOBAL arange over all num_devices=4 shards worth of query
  // rows (4 * 32 = 128), narrowed to a single [1,1,128,1] column rather than
  // the full [B,Hi,128,T] tensor (kept small; broadcast happens after the
  // partition, not before it).
  // CHECK: "ttir.arange"{{.*}} -> tensor<1x1x128x1xi32>
  // Partitioned down to this device's own [1,1,32,1] window via
  // ttir.mesh_partition on dim 2, naming cluster_axis=1 -- the same axis and
  // convention (a device's coordinate along that axis IS its rank) the
  // promoted kernel's get_linearized_index_from_physical_coord uses, so the
  // decomposition and the kernel agree on which row range a given device
  // owns.
  // CHECK: "ttir.mesh_partition"
  // CHECK-SAME: cluster_axis = 1 : ui32
  // CHECK-SAME: dim = 2 : si32
  // Broadcast the per-device row column back out to the score's shape.
  // CHECK: "ttir.broadcast"
  // Key index: still a plain local arange -- key is replicated, never sharded.
  // CHECK: "ttir.arange"{{.*}} -> tensor<{{.*}}xi32>
  // CHECK: "ttir.ge"{{.*}}(tensor<{{.*}}xi32>, tensor<{{.*}}xi32>) -> tensor<{{.*}}xi32>
  // CHECK: "ttir.where"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}) {{.*}}: (tensor<{{.*}}xi32>, tensor<{{.*}}xbf16>, tensor<{{.*}}xbf16>) -> tensor<{{.*}}xbf16>
}
```

The `llvm-lit`-driven runner couldn't be invoked standalone in this environment (a
pre-existing `lit.cfg.py` / `lit.site.cfg.py` generation issue unrelated to this change),
so the fix was instead confirmed by running `ttmlir-opt --stablehlo-to-ttir-pipeline`
directly on this file and inspecting the output, reproduced in §4.

## 4. Verification

### 4.1 Compiler output matches the test's expectations exactly

```
$ ttmlir-opt --stablehlo-to-ttir-pipeline \
    test/ttmlir/Conversion/StableHLOToTTIR/transformer/indexer_score_dsa_sharded.mlir
```

Relevant lines of the synthesized decomposition:

```mlir
%8 = "ttir.arange"() <{arange_dimension = 2 : i64, end = 128 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<1x1x128x1xi32>
%9 = "ttir.mesh_partition"(%8) <{cluster_axis = 1 : ui32, dim = 2 : si32}> : (tensor<1x1x128x1xi32>) -> tensor<1x1x32x1xi32>
%10 = "ttir.broadcast"(%9) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<1x1x32x1xi32>) -> tensor<1x1x32x128xi32>
%11 = "ttir.arange"() <{arange_dimension = 3 : i64, end = 128 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<1x1x32x128xi32>
```

Global arange over all 128 rows -> `mesh_partition(cluster_axis=1, dim=2)` down to this
device's own 32 -> broadcast back out to the score shape -> unchanged key-side arange.
Exactly the sequence the test's `CHECK` lines assert.

### 4.2 Isolated indexer test, real 32-device Blackhole Galaxy hardware

`tests/torch/models/deepseek_v3_2_exp/test_deepseek_v3_2_exp.py::test_deepseek_indexer_dsa_topk_bh_galaxy`
(batch=128, seq_len=128, `index_topk=64`, mesh `(4,8)` axes `("batch","model")`) --
`_dsa_indexer_score_topk_comparator` checks raw-score PCC and gathered-selection cosine
similarity separately (see §1):

| | Before | After |
|---|---|---|
| `indexer_score_dsa` PCC | 1.0 | 1.0 |
| golden valid-key-count per row (32-row chunks) | `[128, 4224, 8192, 8192]` | `[128, 4224, 8192, 8192]` |
| device valid-key-count per row (32-row chunks) | `[128, 128, 128, 128]` | `[128, 4224, 8192, 8192]` |
| rows with a count mismatch | 96 / 128 | **0 / 128** |
| Result | **FAILED** | **PASSED** |

### 4.3 Full 2-layer (dense + MoE) model, sparse DSA path enabled, same hardware

`test_deepseek_v3_2_dense_moe_2layer_transformer_ttops_bh_galaxy` -- both layers' MLA
attention and DSA indexer routed through the real `torch.ops.tt.flash_mla_prefill` /
`sparse_sdpa` / `indexer_score_dsa` / `topk_large_indices` / `paged_fill_cache` ops
(`index_topk=64` forces the sparse path for this test's `seq_len=128`), whole-model
output PCC (strict default threshold, PCC >= 0.99):

| Config | PCC |
|---|---|
| Hand-rolled (non-`torch.ops.tt`) dense MLA, for reference | 0.9673 |
| Real ops, dense path (`flash_mla_prefill`) | 0.9671 |
| Real ops, sparse DSA path, **before this fix** | 0.9119 |
| Real ops, sparse DSA path, **after this fix** | **0.9672** |

All four configurations still fail the test's strict `>= 0.99` PCC gate (that gap is a
separate, generic bf16-precision effect present in every configuration, dense or sparse,
real ops or hand-rolled -- unrelated to this bug). What this fix demonstrates is that
sparse DSA attention now lands in the **same** PCC neighborhood as dense attention, rather
than ~0.05 below it: the entire gap was this bug, not something inherent to top-k
selection under bf16 noise (an earlier, since-discarded hypothesis for the same
observation).

### 4.4 The DRAM regression this fix briefly caused, and its resolution

Worth recording because the symptom pointed nowhere near the cause.

After the tt-mlir rebuild carrying this fix, the full 61-layer end-to-end test
(`test_tensor_parallel_generation_deepseek_v32_full`) stopped reaching generation at all
and instead died in vLLM's `_initialize_kv_caches`:

```
TT_FATAL: Out of Memory: Not enough space to allocate 16777216 B DRAM buffer across
8 banks, where each bank needs to store 2097152 B, but bank size is 4272341376 B
(allocated: 4269564032 B, free: 2777344 B, largest free block: 1294336 B)
```

Note this is fragmentation at ~99.9% occupancy, not a clean capacity wall: 2.6 MiB free
per bank but a 1.23 MiB largest contiguous block against a 2 MiB request. The failing
buffer is exactly the `tensor<2048x2048xsi32>` (= 16777216 B) that graph `g0` caches from
`@main_const_eval_0`.

It initially looked unrelated to the compiler, because the OOM first appeared on a run
whose only *intended* change was an unrelated Python-side experiment. Log forensics across
every full-model run separated the two cleanly:

| run | date | outcome |
|---|---|---|
| `full_e2e_new` | 08-06 | ran → incoherent |
| `full_e2e_v2` | 08-08 | ran → incoherent |
| `full_e2e_v3_opt_1` | 08-08 14:12 | ran → incoherent |
| `full_e2e_cluster_axis_1` | 08-09 16:39 | **OOM** |
| `full_e2e_routing_fix` | 08-10 03:05 | **OOM** |

The two OOM runs carried *different* Python-side changes but near-identical allocator
state (same 16 MiB request, same `largest free block: 1294336 B`, `allocated` differing by
0.47 MB), and the tt-mlir rebuild sat between 08-08 14:12 and 08-09 16:39. Since tt-mlir
HEAD was unchanged (`53f4a3baed`, 07-30), the rebuild's only source delta was this fix --
and of its parts, only the `TTCore_NonCacheableTrait` could touch this test's graph at all
(`indexer_score_dsa` is never emitted here: at `index_topk=2048` with a 128-token prefill
bucket `dsa_prefill_uses_sparse` is false, so the dense path runs and the decomposition
this document is about is inert).

Reverting the trait (3.5) removed 572 MB of persistent const-eval DRAM at four layers,
scaling per-layer, while leaving generated tokens bit-identical -- and 4.2's isolated
indexer check still passes with 0/128 row mismatches and score PCC 1.0, confirming the
trait was never what made the fix correct.

`test_dsa_v32_4layer_ccl_ir_diagnostic` was added alongside, so the const-eval cache
structure and CCL counts behind this class of regression can be measured from a ~10 min
4-layer run instead of a ~9 h full-model compile.

## 5. What this does and does not fix

**Fixed.** The decomposition fallback for `indexer_score_dsa` now computes the correct
per-device causal window on a query-sequence-sharded multi-device mesh, matching the
promoted kernel's semantics. This is the path every caller hits by default
(`optimization_level=0`) and the only path available whenever batch > 1, which is the
common case for any real batched inference.

**Not touched -- `sparse_sdpa` and `flash_mla_prefill`'s own decompositions.** Both are
lowered by neighboring functions in the same file
(`buildSparseSdpaDecompositionBody`, `buildFlashMlaPrefillDecompositionBody`) and were not
audited here beyond a quick read. `buildFlashMlaPrefillDecompositionBody` in particular
builds its own causal mask from local, unsharded aranges on both the row and column
dimensions; it is presently safe only because its Shardy sharding rule marks the key-seq
dimension `kNeedReplication` and query-seq sharding is not currently exercised for it --
but it carries the same latent defect and would need the identical treatment (or worse,
since both its dimensions would need the same treatment) if query-seq sharding is ever
enabled there.

**Not touched -- decode.** This fix is specific to the prefill-time
`_forward_prefill`/`indexer_score_dsa` call path, where the query sequence is genuinely
sharded across `cluster_axis`. Decode (`_forward_decode`) calls `indexer_score_dsa` with
`chunk_start_idx=max_seq_len-1` and no `cluster_axis` at all (single query token per user,
nothing to sequence-shard), so it is unaffected by both the bug and this fix.

**Not a performance fix.** The decomposition fallback remains substantially slower than
the promoted kernel regardless of this change -- this fix only makes the fallback
*correct* when it is the path actually taken (the default at `optimization_level=0`, or
always for batch > 1). Forcing kernel promotion (`composite-resolution=force-promote` /
raising `optimization_level`) is a separate, orthogonal lever for recovering performance,
and was not touched by this change.
