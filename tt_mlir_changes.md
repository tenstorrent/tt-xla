# tt-mlir local changes (for porting to a tt-mlir branch)

Records every edit made to `third_party/tt-mlir/src/tt-mlir` in this workspace so
they can be re-applied on a clean tt-mlir branch.

- **tt-mlir baseline commit:** `260d4c495fd5bebbfda3d48bc2e575514ec396f0`
- **Source root:** `third_party/tt-mlir/src/tt-mlir` (paths below are relative to it)
- **Install prefix (what the tt-xla plugin links):** `third_party/tt-mlir/install`

A verbatim unified diff is at the bottom (`git diff` of the source tree); the
sections above explain each change.

---

## Change set 1 — thread `sliding_window_size` through `paged_flash_mla_decode`

**Goal:** make the `sliding_window_size` frontend attribute emitted by tt-xla's
`torch.ops.tt.paged_flash_mla_decode` reach the ttnn kernel
`paged_flash_multi_latent_attention_decode` (which already accepts it). Enables
DeepSeek-V4 SWA on the **decode** path (prefill uses a mask and needs no change).

**Nature:** pure attribute plumbing — one optional `uint32` threaded through the
compiler + runtime. No new op, kernel, or math. Copies the fully-plumbed sibling
`PagedScaledDotProductAttentionDecodeOp`, which already has `sliding_window_size`
at every layer.

**Files edited (9):** the 7 below are the runtime/flatbuffer path; two more
(`lib/Conversion/TTNNToEmitC/TTNNToEmitC.cpp`,
`lib/Conversion/TTNNToEmitPy/TTNNToEmitPy.cpp`) thread the attribute through the
EmitC/EmitPy codegen paths too (they already had a `sliding_window_size` slot
hardcoded to `std::nullopt`; now they read `op.getSlidingWindowSize()`), so the
change is complete across all lowerings.

1. `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` — in
   `StableHLOToTTIRPagedFlashMLADecodeOpConversionPattern`: read the
   `sliding_window_size` frontend string attr → `UI32IntegerAttr`, pass it to the
   TTIR op builder (last arg).
2. `include/ttmlir/Dialect/TTIR/IR/TTIROps.td` —
   `TTIR_PagedFlashMultiLatentAttentionDecodeOp`: add
   `OptionalAttr<UI32Attr>:$sliding_window_size` (after `$scale`).
3. `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp` —
   `PagedFlashMultiLatentAttentionDecodeOpConversionPattern`: forward
   `adaptor.getSlidingWindowSizeAttr()` to the TTNN op builder (last arg).
4. `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td` —
   `TTNN_PagedFlashMultiLatentAttentionDecodeOp`: add
   `OptionalAttr<UI32Attr>:$sliding_window_size` (after `$scale`).
5. `include/ttmlir/Target/TTNN/operations/transformer.fbs` — table
   `PagedFlashMultiLatentAttentionDecodeOp`: add `sliding_window_size: uint32 = null;`
   (after `scale`, before `out`).
6. `lib/Target/TTNN/TTNNToFlatbuffer.cpp` — `createOp(... PagedFlashMultiLatentAttentionDecodeOp)`:
   emit `toFlatbuffer(cache, op.getSlidingWindowSize())` and pass into the
   `Create...` call (after `scale`, before `out`).
7. `runtime/lib/ttnn/operations/transformer/paged_flash_multi_latent_attention_decode.cpp` —
   replace the hardcoded `std::optional<uint32_t> slidingWindowSize = std::nullopt;`
   with a read of `op->sliding_window_size()`.

(Detailed before/after per file follows; verbatim diff at the end.)

---

## Verbatim diff (`git diff` of third_party/tt-mlir/src/tt-mlir @ 260d4c49)

```diff
diff --git a/include/ttmlir/Dialect/TTIR/IR/TTIROps.td b/include/ttmlir/Dialect/TTIR/IR/TTIROps.td
index 6cbfa97f7..ef0bea668 100644
--- a/include/ttmlir/Dialect/TTIR/IR/TTIROps.td
+++ b/include/ttmlir/Dialect/TTIR/IR/TTIROps.td
@@ -6199,7 +6199,8 @@ def TTIR_PagedFlashMultiLatentAttentionDecodeOp : TTIR_NamedOp<"paged_flash_mult
                        Optional<AnyRankedTensor>:$attention_mask,
                        Optional<AnyRankedTensor>:$cur_pos_tensor,
                        Optional<AnyRankedTensor>:$attention_sink,
-                       OptionalAttr<F32Attr>:$scale);
+                       OptionalAttr<F32Attr>:$scale,
+                       OptionalAttr<UI32Attr>:$sliding_window_size);

   let results = (outs AnyRankedTensor:$result);

diff --git a/include/ttmlir/Dialect/TTNN/IR/TTNNOps.td b/include/ttmlir/Dialect/TTNN/IR/TTNNOps.td
index f4b4f7e06..f06226391 100644
--- a/include/ttmlir/Dialect/TTNN/IR/TTNNOps.td
+++ b/include/ttmlir/Dialect/TTNN/IR/TTNNOps.td
@@ -3913,7 +3913,8 @@ def TTNN_PagedFlashMultiLatentAttentionDecodeOp : TTNN_Op<"paged_flash_multi_lat
                        Optional<AnyRankedTensor>:$attention_mask,
                        Optional<AnyRankedTensor>:$cur_pos_tensor,
                        Optional<AnyRankedTensor>:$attention_sink,
-                       OptionalAttr<F32Attr>:$scale);
+                       OptionalAttr<F32Attr>:$scale,
+                       OptionalAttr<UI32Attr>:$sliding_window_size);

   let results = (outs AnyRankedTensor:$result);

diff --git a/include/ttmlir/Target/TTNN/operations/transformer.fbs b/include/ttmlir/Target/TTNN/operations/transformer.fbs
index feb3eebbc..4aa9ed4aa 100644
--- a/include/ttmlir/Target/TTNN/operations/transformer.fbs
+++ b/include/ttmlir/Target/TTNN/operations/transformer.fbs
@@ -156,6 +156,7 @@ table PagedFlashMultiLatentAttentionDecodeOp {
   cur_pos_tensor: tt.target.ttnn.TensorRef;
   attention_sink: tt.target.ttnn.TensorRef;
   scale: float = null;
+  sliding_window_size: uint32 = null;
   out: tt.target.ttnn.TensorRef;
   memcfg: tt.target.ttnn.MemoryConfig;
 }
diff --git a/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp b/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp
index 9238bb6ec..6105e943c 100644
--- a/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp
+++ b/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp
@@ -8804,13 +8804,30 @@ public:
       attentionSink = operands[operandIndex++];
     }

+    auto slidingWindowSizeStringAttr =
+        frontendAttributes.getAs<mlir::StringAttr>("sliding_window_size");
+    IntegerAttr slidingWindowSizeAttr = nullptr;
+    if (slidingWindowSizeStringAttr) {
+      uint32_t slidingWindowSize;
+      if (!llvm::to_integer(slidingWindowSizeStringAttr.getValue(),
+                            slidingWindowSize)) {
+        return rewriter.notifyMatchFailure(
+            srcOp, llvm::Twine("sliding_window_size attribute string must be "
+                               "convertible to a non-negative integer. "
+                               "Received \"") +
+                       slidingWindowSizeStringAttr.getValue() + "\".");
+      }
+      slidingWindowSizeAttr = rewriter.getUI32IntegerAttr(slidingWindowSize);
+    }
+
     RankedTensorType outputType = cast<RankedTensorType>(
         getTypeConverter()->convertType(srcOp.getResult(0).getType()));

     rewriter.replaceOpWithNewOp<
         mlir::tt::ttir::PagedFlashMultiLatentAttentionDecodeOp>(
         srcOp, outputType, query, key, value, headDimVAttr, pageTable,
-        isCausalAttr, attentionMask, curPosTensor, attentionSink, scaleAttr);
+        isCausalAttr, attentionMask, curPosTensor, attentionSink, scaleAttr,
+        slidingWindowSizeAttr);

     return success();
   }
diff --git a/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp b/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp
index 3defdd9b9..c3b53decf 100644
--- a/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp
+++ b/lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp
@@ -3240,7 +3240,7 @@ public:
         static_cast<uint32_t>(adaptor.getHeadDimV()), adaptor.getPageTable(),
         adaptor.getIsCausal(), adaptor.getAttentionMask(),
         adaptor.getCurPosTensor(), adaptor.getAttentionSink(),
-        adaptor.getScaleAttr());
+        adaptor.getScaleAttr(), adaptor.getSlidingWindowSizeAttr());
     return success();
   }
 };
diff --git a/lib/Conversion/TTNNToEmitC/TTNNToEmitC.cpp b/lib/Conversion/TTNNToEmitC/TTNNToEmitC.cpp
index 4c3d00641..d4b5f4400 100644
--- a/lib/Conversion/TTNNToEmitC/TTNNToEmitC.cpp
+++ b/lib/Conversion/TTNNToEmitC/TTNNToEmitC.cpp
@@ -3644,7 +3644,7 @@ public:
         emitter.emit(srcOp.getCurPosTensor()),
         emitter.emit(srcOp.getAttentionSink()),
         emitter.emit(srcOp.getScale()),
-        emitter.emit(/*slidingWindowSize=*/std::nullopt),
+        emitter.emit(srcOp.getSlidingWindowSize()),
         emitter.emit(srcOp.getMemoryConfigAttr()),
         emitter.emit(/*program_config=*/std::nullopt),
         emitter.emit(/*compute_kernel_config=*/std::nullopt),
diff --git a/lib/Conversion/TTNNToEmitPy/TTNNToEmitPy.cpp b/lib/Conversion/TTNNToEmitPy/TTNNToEmitPy.cpp
index beae5d0a6..1250d9ffb 100644
--- a/lib/Conversion/TTNNToEmitPy/TTNNToEmitPy.cpp
+++ b/lib/Conversion/TTNNToEmitPy/TTNNToEmitPy.cpp
@@ -4711,7 +4711,7 @@ public:
         emitter.emit(srcOp.getCurPosTensor(), "cur_pos_tensor"),
         emitter.emit(srcOp.getAttentionSink(), "attention_sink"),
         emitter.emit(srcOp.getScale(), "scale"),
-        emitter.emit(std::nullopt, "sliding_window_size"),
+        emitter.emit(srcOp.getSlidingWindowSize(), "sliding_window_size"),
         emitter.emit(srcOp.getMemoryConfig(), "memory_config"),
     };
     // NOLINTEND(clang-analyzer-cplusplus.NewDelete)
diff --git a/lib/Target/TTNN/TTNNToFlatbuffer.cpp b/lib/Target/TTNN/TTNNToFlatbuffer.cpp
index 4797071ea..189c8abe1 100644
--- a/lib/Target/TTNN/TTNNToFlatbuffer.cpp
+++ b/lib/Target/TTNN/TTNNToFlatbuffer.cpp
@@ -3417,6 +3417,8 @@ createOp(FlatbufferObjectCache &cache,
       cache, op.getScale()
                  ? std::make_optional(op.getScale().value().convertToFloat())
                  : std::nullopt);
+  ::flatbuffers::Optional<uint32_t> slidingWindowSize =
+      toFlatbuffer(cache, op.getSlidingWindowSize());
   auto memoryConfig = toFlatbuffer(cache, op.getMemoryConfigAttr());
   auto out =
       cache.getOrCreateNoSharding(op.getResult(), tensorValueToFlatbuffer,
@@ -3424,7 +3426,8 @@ createOp(FlatbufferObjectCache &cache,

   return ::tt::target::ttnn::CreatePagedFlashMultiLatentAttentionDecodeOp(
       *cache.fbb, query, key, value, headDimV, pageTable, isCausal,
-      attentionMask, curPosTensor, attentionSink, scale, out, memoryConfig);
+      attentionMask, curPosTensor, attentionSink, scale, slidingWindowSize, out,
+      memoryConfig);
 }

 ::flatbuffers::Offset<::tt::target::ttnn::ScaledDotProductAttentionOp>
diff --git a/runtime/lib/ttnn/operations/transformer/paged_flash_multi_latent_attention_decode.cpp b/runtime/lib/ttnn/operations/transformer/paged_flash_multi_latent_attention_decode.cpp
index e752299c4..e3ba962b0 100644
--- a/runtime/lib/ttnn/operations/transformer/paged_flash_multi_latent_attention_decode.cpp
+++ b/runtime/lib/ttnn/operations/transformer/paged_flash_multi_latent_attention_decode.cpp
@@ -47,7 +47,7 @@ static void runPagedFlashMultiLatentAttentionDecodeOp(
   }

   std::optional<float> scale = op->scale();
-  std::optional<uint32_t> slidingWindowSize = std::nullopt;
+  std::optional<uint32_t> slidingWindowSize = op->sliding_window_size();

   auto programConfig =
       std::make_optional<::ttnn::operations::transformer::SDPAProgramConfig>();
```
