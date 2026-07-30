# Plumbing `cluster_axis` through tt-mlir for `indexer_score_dsa`

Complete record of the tt-mlir changes that let a caller name **which mesh axis carries
the DSA indexer's query-sequence shard**, instead of leaving the kernel to infer it from a
flat enumeration of every device holding `q`.

**Base:** tt-mlir `4c6f88a08e` on branch `hshah/all-dsa-ops`
(tt-metal submodule `f1f4ff75579`, `v0.75.0-dev20260717-29`).
**Scope:** 6 files, +70 / -5.

```
 include/ttmlir/Dialect/TTNN/IR/TTNNOps.td                            |  3 ++-
 include/ttmlir/Target/TTNN/operations/transformer.fbs                |  5 +++++
 lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp           | 25 ++++++++++++++++++++++
 lib/Dialect/TTNN/Transforms/TTNNResolveComposites.cpp                | 22 +++++++++++++++++--
 lib/Target/TTNN/TTNNToFlatbuffer.cpp                                 |  6 +++++-
 runtime/lib/ttnn/operations/transformer/indexer_score_dsa.cpp        | 14 +++++++++++-
 6 files changed, 70 insertions(+), 5 deletions(-)
```

The matching tt-xla changes (`tt.indexer_score_dsa` wrapper +
`TTIndexer._prefill_seq_shard_plan`) are **not** in this document; see
[`dsa_blackhole_tt-mlir_changes.md`](./dsa_blackhole_tt-mlir_changes.md) for the wider DSA
change set.

---

## 1. Why

`ttnn::experimental::indexer_score_dsa` models its query as sequence-parallel. Both the
program factory and the validator derive a per-device row offset from a device *rank*:

```cpp
// indexer_score_program_factory.cpp:80 — this device's q-row 0, in tiles
return {(args.chunk_start_idx + (device_index + tp_index) * Sq) / TW, 0u, 0u};

// indexer_score_device_operation.cpp:42-48 — the worst-case device, for validation
return attrs.chunk_start_idx + (max_linearized_rank(q, attrs.sp_axis()) + tp_rank) * Sq;
```

With `cluster_axis` unset, `max_linearized_rank` falls back to the **position of the coord
in `q.device_storage().get_coords()`** — a flat `0..N-1` enumeration over every device
(`ccl_common.cpp:137-145`). That is only the sequence-shard index when the sequence is
sharded across *all* devices.

The failure mode when it is not: tt-mlir's Shardy rule for this op marks the head factor
`kReduction`, so Shardy is free to shard heads on one mesh axis and the sequence on
another. On a `[2, 4]` mesh that yields **4 distinct sequence shards across 8 devices** —
devices `(0,b)` and `(1,b)` hold the *same* query rows but receive flat ranks `b` and
`b+4`, so identical rows get different causal windows. The arithmetic still completes (the
head all-reduce is unaffected), so the result is silently wrong rather than an error,
except in the `T == S` prefill case where `max_cs + Sq <= T` happens to catch it.

Naming the axis makes the rank exact: with `cluster_axis` set,
`get_linearized_index_from_physical_coord` returns `physical_coord[cluster_axis]`
(`ccl_common.cpp:133`), so devices sharing sequence rows share a rank, and the other axes
are free to carry heads, batch, or replication.

---

## 2. Data flow

```
tt.indexer_score_dsa custom_call
  mhlo.frontend_attributes = {chunk_start_idx = "0", cluster_axis = "1"}
        │
        │  StableHLOToTTIRPatterns.cpp  (§3.3)
        ▼
ttcore.composite "indexer_score_dsa"
  composite_attributes = {chunk_start_idx = 0 : ui32, cluster_axis = 1 : ui32}
        │
        │  TTNNResolveComposites.cpp  (§3.4)  — validate + build
        ▼
ttnn.indexer_score_dsa
  <{chunk_start_idx = 0 : ui32, cluster_axis = 1 : ui32}>        (TTNNOps.td, §3.1)
        │
        │  TTNNToFlatbuffer.cpp  (§3.5)
        ▼
flatbuffer IndexerScoreDsaOp { …, cluster_axis: uint32 = null }  (transformer.fbs, §3.2)
        │
        │  runtime/…/indexer_score_dsa.cpp  (§3.6)
        ▼
ttnn::experimental::indexer_score_dsa(…, clusterAxis)            // 9th parameter
```

`cluster_axis` is **optional at every layer**, and absent everywhere means "flat
enumeration" — i.e. byte-identical behaviour to before this change for any caller that
does not set it.

---

## 3. The diffs

### 3.1 `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td`

`OptionalAttr` rather than `DefaultValuedAttr`: there is no sensible default axis, and
"unset" has to stay distinguishable from "axis 0".

```diff
@@ -4129,9 +4129,10 @@ def TTNN_IndexerScoreDsaOp : TTNN_Op<"indexer_score_dsa"> {

     let arguments = (ins AnyRankedTensor:$query,
                          AnyRankedTensor:$key,
                          AnyRankedTensor:$weights,
-                         DefaultValuedAttr<UI32Attr, "0">:$chunk_start_idx);
+                         DefaultValuedAttr<UI32Attr, "0">:$chunk_start_idx,
+                         OptionalAttr<UI32Attr>:$cluster_axis);

     let results = (outs AnyRankedTensor:$result);

     let hasVerifier = 1;
```

> **Note on formatting.** `git clang-format` reflows this `let arguments` block into a
> style unlike the other ~40 ops in the file. `.td` is *not* in pre-commit's clang-format
> scope (`.pre-commit-config.yaml:11` — `types_or: [c++, c]`), so the aligned form above is
> correct and the reflow should be reverted if a tool applies it.

The generated builder takes `/*optional*/::mlir::IntegerAttr`, **not**
`std::optional<uint32_t>` — see §4.

### 3.2 `include/ttmlir/Target/TTNN/operations/transformer.fbs`

```diff
@@ -98,8 +98,13 @@ table IndexerScoreDsaOp {
   key: tt.target.ttnn.TensorRef;
   weights: tt.target.ttnn.TensorRef;
   chunk_start_idx: uint32;
   out: tt.target.ttnn.TensorRef;
+  // Mesh axis carrying the query sequence shard. Unset = the op falls back to a
+  // flat row-major enumeration over ALL of q's devices, which is only correct
+  // when the sequence is sharded across every device. Appended after `out` so
+  // existing flatbuffers keep their field ids.
+  cluster_axis: uint32 = null;
 }

 table SparseSdpaOp {
   query: tt.target.ttnn.TensorRef;
```

Two deliberate choices:

* `= null` gives an **optional scalar**, matching the existing convention in this schema
  (`creation.fbs:11-12`, `cumsum.fbs:10`). Read side is `if (op->field())` + deref; write
  side is `flatbuffers::Optional<T>`.
* **Appended after `out`**, not inserted next to `chunk_start_idx`. Field order defines
  flatbuffer field ids, so inserting mid-table renumbers `out` and any previously-written
  binary would misparse. The build carries a persistent compile cache, so that risk is
  real rather than theoretical.

### 3.3 `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`

Recovers the attribute from the custom call and attaches it to the composite. Mirrors the
existing `chunk_start_idx` handling, including the string→integer parse failure path.

```diff
@@ -9727,8 +9727,28 @@ public:
       }
     }
     IntegerAttr chunkStartIdxAttr = rewriter.getUI32IntegerAttr(chunkStartIdx);

+    // cluster_axis is optional. Absent leaves the kernel on its flat row-major
+    // enumeration over all of q's devices, which is only correct when the query
+    // sequence is sharded across every device. Naming the axis is what makes a
+    // partial split (e.g. heads on one mesh axis, sequence on another) correct.
+    IntegerAttr clusterAxisAttr;
+    if (auto frontendAttributes = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
+            srcOp->getDiscardableAttr("mhlo.frontend_attributes"))) {
+      if (auto clusterAxisStringAttr =
+              frontendAttributes.getAs<mlir::StringAttr>("cluster_axis")) {
+        uint32_t clusterAxis = 0;
+        if (!llvm::to_integer(clusterAxisStringAttr.getValue(), clusterAxis)) {
+          return rewriter.notifyMatchFailure(
+              srcOp, "cluster_axis attribute must be a non-negative integer. "
+                     "Received \"" +
+                         clusterAxisStringAttr.getValue() + "\".");
+        }
+        clusterAxisAttr = rewriter.getUI32IntegerAttr(clusterAxis);
+      }
+    }
+
     RankedTensorType outputType = mlir::cast<RankedTensorType>(
         getTypeConverter()->convertType(srcOp.getResult(0).getType()));

     // Synthesize the private decomposition function (inlined fallback).
```

```diff
@@ -9763,8 +9783,13 @@ public:

     SmallVector<NamedAttribute> compositeAttrList;
     compositeAttrList.push_back(
         rewriter.getNamedAttr("chunk_start_idx", chunkStartIdxAttr));
+    // Only carried when the caller named an axis; absent means "flat".
+    if (clusterAxisAttr) {
+      compositeAttrList.push_back(
+          rewriter.getNamedAttr("cluster_axis", clusterAxisAttr));
+    }

     rewriter.replaceOpWithNewOp<ttcore::CompositeOp>(
         srcOp, TypeRange{outputType}, ValueRange(compositeInputs),
         rewriter.getStringAttr("indexer_score_dsa"),
```

The attribute is added to `compositeAttrList` **only when present**, so an unset
`cluster_axis` produces a composite byte-identical to the pre-change one.

### 3.4 `lib/Dialect/TTNN/Transforms/TTNNResolveComposites.cpp`

New recover helper, placed beside `getIndexerScoreDsaChunkStartIdx`:

```diff
@@ -122,8 +122,22 @@ getIndexerScoreDsaChunkStartIdx(ttcore::CompositeOp compositeOp) {
                                  chunkStartIdxAttr.getValue().getZExtValue())
                            : 0;
 }

+// Recover the optional cluster_axis attribute from an "indexer_score_dsa"
+// composite. Returns a NULL attribute when absent, which is what the op's
+// OptionalAttr builder parameter expects; absent leaves the kernel on its flat
+// row-major enumeration over all of q's devices, which is only correct when the
+// query sequence is sharded across every device.
+static mlir::IntegerAttr
+getIndexerScoreDsaClusterAxis(ttcore::CompositeOp compositeOp) {
+  DictionaryAttr attrs = compositeOp.getCompositeAttributes().value_or(nullptr);
+  if (!attrs) {
+    return {};
+  }
+  return attrs.getAs<mlir::IntegerAttr>("cluster_axis");
+}
+
 // Attributes recovered from a "sparse_sdpa" composite, shared by its validate,
 // build and promotion-guard callbacks.
 struct SparseSdpaCompositeArgs {
   uint32_t vDim;
```

Threaded through **both** registry callbacks. Doing only `build` would be a latent bug:
`validate` constructs the op in an isolated module to run the verifier and the op-model
query, so a mismatch there would validate a *different* op than the one built.

```diff
@@ -266,22 +280,26 @@ static void registerBuiltinComposites() {
          OpBuilder &builder) -> OpValidationResult {
         TT_assert(compositeOp.getInputs().size() == 3u);

         uint32_t chunkStartIdx = getIndexerScoreDsaChunkStartIdx(compositeOp);
+        mlir::IntegerAttr clusterAxis =
+            getIndexerScoreDsaClusterAxis(compositeOp);
         SmallVector<Type> resultTypes(compositeOp.getResultTypes());
         IsolatedIRValidationWrapper validator(compositeOp.getContext());
         return validator.validateOp<IndexerScoreDsaOp>(
             compositeOp.getOperation(), compositeOp.getLoc(), resultTypes,
             compositeOp.getInputs()[0], compositeOp.getInputs()[1],
-            compositeOp.getInputs()[2], chunkStartIdx);
+            compositeOp.getInputs()[2], chunkStartIdx, clusterAxis);
       },
       // Build
       [](ttcore::CompositeOp compositeOp, OpBuilder &builder) -> Operation * {
         uint32_t chunkStartIdx = getIndexerScoreDsaChunkStartIdx(compositeOp);
+        mlir::IntegerAttr clusterAxis =
+            getIndexerScoreDsaClusterAxis(compositeOp);
         return builder.create<IndexerScoreDsaOp>(
             compositeOp.getLoc(), compositeOp.getResultTypes(),
             compositeOp.getInputs()[0], compositeOp.getInputs()[1],
-            compositeOp.getInputs()[2], chunkStartIdx);
+            compositeOp.getInputs()[2], chunkStartIdx, clusterAxis);
       },
       // Promotion guard: ttnn.experimental.indexer_score_dsa is
       // Blackhole-only. On any other architecture, veto promotion so the
       // composite falls back to inlining its decomposition instead of
```

### 3.5 `lib/Target/TTNN/TTNNToFlatbuffer.cpp`

```diff
@@ -3553,14 +3553,18 @@ createOp(FlatbufferObjectCache &cache, IndexerScoreDsaOp op) {
       getOperandThroughDPSOps(op.getKey()));
   auto weights = cache.at<::tt::target::ttnn::TensorRef>(
       getOperandThroughDPSOps(op.getWeights()));
   auto chunkStartIdx = op.getChunkStartIdx();
+  ::flatbuffers::Optional<uint32_t> clusterAxis;
+  if (auto axis = op.getClusterAxis()) {
+    clusterAxis = *axis;
+  }
   auto out =
       cache.getOrCreateNoSharding(op.getResult(), tensorValueToFlatbuffer,
                                   /*local_shape*/ std::nullopt);

   return ::tt::target::ttnn::CreateIndexerScoreDsaOp(
-      *cache.fbb, query, key, weights, chunkStartIdx, out);
+      *cache.fbb, query, key, weights, chunkStartIdx, out, clusterAxis);
 }

 ::flatbuffers::Offset<::tt::target::ttnn::SparseSdpaOp>
 createOp(FlatbufferObjectCache &cache, SparseSdpaOp op) {
```

`clusterAxis` is default-constructed (i.e. absent) unless the attribute is present, and is
passed **last** because the field was appended last in the schema (§3.2).

### 3.6 `runtime/lib/ttnn/operations/transformer/indexer_score_dsa.cpp`

```diff
@@ -19,11 +19,23 @@ void run(const ::tt::target::ttnn::IndexerScoreDsaOp *op,
   const ::ttnn::Tensor &key = tensorPool.getTTNNTensorAndValidate(op->key());
   const ::ttnn::Tensor &weights =
       tensorPool.getTTNNTensorAndValidate(op->weights());

+  // Mesh axis carrying the query sequence shard. Unset leaves the op on its
+  // flat row-major enumeration over all of q's devices, which is only correct
+  // when the sequence is sharded across every device -- naming the axis is what
+  // makes a partial split (e.g. heads on one axis, sequence on another)
+  // correct.
+  std::optional<uint32_t> clusterAxis = std::nullopt;
+  if (op->cluster_axis()) {
+    clusterAxis = *op->cluster_axis();
+  }
+
   // program_config and compute_kernel_config fall back to the ttnn defaults.
   ::ttnn::Tensor out = ::ttnn::experimental::indexer_score_dsa(
-      query, key, weights, op->chunk_start_idx());
+      query, key, weights, op->chunk_start_idx(),
+      /*program_config=*/{}, /*compute_kernel_config=*/std::nullopt,
+      /*cache_batch_idx=*/std::nullopt, /*kv_len=*/std::nullopt, clusterAxis);

   tensorPool.insertTTNNTensorAndValidate(op->out(), out);
 }
 } // namespace tt::runtime::ttnn::operations::transformer
```

`cluster_axis` is the **9th** parameter of the ttnn entry point, so the four intervening
defaults have to be spelled out:

```cpp
// ttnn/cpp/ttnn/operations/experimental/indexer_score/device/indexer_score_device_operation.hpp:85
ttnn::Tensor indexer_score_dsa(
    const ttnn::Tensor& q, const ttnn::Tensor& k, const ttnn::Tensor& weights,
    std::optional<uint32_t> chunk_start_idx = std::nullopt,
    const IndexerScoreProgramConfig& program_config = {},
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    std::optional<uint32_t> cache_batch_idx = std::nullopt,
    std::optional<uint32_t> kv_len = std::nullopt,
    std::optional<uint32_t> cluster_axis = std::nullopt,        // <-- 9th
    std::optional<uint32_t> seq_subshard_axis = std::nullopt,
    std::optional<uint32_t> block_cyclic_sp_axis = std::nullopt,
    std::optional<uint32_t> block_cyclic_chunk_local = std::nullopt);
```

Naming them with `/*param=*/` comments keeps the call readable and makes it obvious that
`cache_batch_idx` / `kv_len` / the block-cyclic pair are still unplumbed.

---

## 4. Build gotcha: the ODS builder wants an `IntegerAttr`

The first attempt passed `std::optional<uint32_t>` and failed:

```
lib/Dialect/TTNN/Transforms/TTNNResolveComposites.cpp
/opt/ttmlir-toolchain/include/mlir/IR/Builders.h:508:5: error: no matching function for call to 'build'
```

`OptionalAttr<UI32Attr>` generates builders taking the attribute type, with no
`std::optional` unwrapped overload (`build/include/ttmlir/Dialect/TTNN/IR/TTNNOps.h.inc:124-135`):

```cpp
static void build(::mlir::OpBuilder &, ::mlir::OperationState &, ::mlir::TypeRange,
                  ::mlir::Value query, ::mlir::Value key, ::mlir::Value weights,
                  uint32_t chunk_start_idx,
                  /*optional*/::mlir::IntegerAttr cluster_axis);
```

Hence the recover helper returns `mlir::IntegerAttr` (null when absent) rather than
`std::optional<uint32_t>`. The `std::optional` conversion happens once, in the runtime
(§3.6), where the ttnn API genuinely wants it.

---

## 5. Verification

Built clean, then on `bh-rb-01-…` (8 × p150, mesh `[2, 4]`, `optimization_level: 1`):

```
pytest tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py -k deepseek_v32
  ..._3l[dense]                      PASSED
  ..._3l[sparse-topk128]             PASSED
  ..._3l[sparse-topk128-unaligned]   PASSED
  ..._3l[sparse-topk2048]            SKIPPED   (pre-existing Wormhole-OOM skip)
  ..._dsa[mesh_shape0]               PASSED    <- A/B numerics: sparse == dense
  ..._dsa[mesh_shape1]               FAILED    (pre-existing, see below)
= 1 failed, 4 passed, 1 skipped in 260.51s =
```

`mesh_shape1` is `[1, 4]` and fails before any DSA code runs:

```
ValueError: mesh_shape (1, 4) has product 4, which does not match the device count 8
```

That param is marked `bhqb` (4-chip quietbox) and cannot run on an 8-chip host. With
devices restricted so validation passes, it then hits a separate pre-existing Shardy
assert (`reshard_to_collectives.cc:394: CollectiveInserter::insert(): Assertion 'isDone()'
failed`). Neither cause is related to this change.

The attribute reaching the promoted kernel, from the exported TTNN IR:

```mlir
"ttnn.indexer_score_dsa"(%155, %161, %160)
  <{chunk_start_idx = 0 : ui32, cluster_axis = 1 : ui32}>
  : (tensor<1x64x32x128xbf16, #ttnn_layout130>,
     tensor<1x1x128x128xbf16, #ttnn_layout133>,
     tensor<1x64x32x1xbf16,   #ttnn_layout132>) -> tensor<1x1x32x128xbf16>
```

`cluster_axis = 1` is the `model` axis of the `("batch", "model")` mesh, and `Sq = 32`
is `seq_len 128 / model_size 4` — the sequence sharded on that axis alone. Promotion
counts were unchanged at 3 sites each for `indexer_score_dsa` / `topk_large_indices` /
`sparse_sdpa` per DSA graph, with zero decomposition markers.

`..._dsa[mesh_shape0]` is the load-bearing check: it is the only DSA test that asserts
numerics rather than op emission, and it verifies exactly what this change alters — which
axis carries the sequence, hence what causal window each rank computes.

---

## 6. What this does and does not fix

**Fixed.** A partial sequence split is now expressible and correct. The tt-xla caller
shards the query on the model axis alone and names it, so the batch axis is free; before
this, correctness relied on the caller consuming *every* mesh axis so that the flat rank
coincided with the sequence-shard index.

**Not fixed — the head factor is untested.** The Shardy rule still marks heads
`kReduction`, so head-sharding remains legal. It should now be *correct* rather than
dangerous, but no head-sharded configuration has been run; treat it as untested-legal.

**Not fixed — decode.** Sparse decode still cannot use the kernel. `Sq = 1` violates
`Sq % TILE_HEIGHT == 0`, and a replicated single-row query needs `ring == 1`, which needs a
**size-1** mesh axis to name — impossible on `[2, 4]`. `cluster_axis` is a prerequisite for
that route, not a solution to it. See
[`dsa_blackhole_tt-metal_changes.md`](./dsa_blackhole_tt-metal_changes.md) §2.5 for the
tt-metal ask (a genuine "sequence replicated" mode).

**Not plumbed.** `seq_subshard_axis`, `block_cyclic_sp_axis`, `block_cyclic_chunk_local`,
`cache_batch_idx`, `kv_len` — all still `std::nullopt`. The block-cyclic pair is what a
chunked-prefill SP-striped cache would need.
