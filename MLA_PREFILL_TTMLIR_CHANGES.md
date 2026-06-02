# tt-mlir changes to run `test_mla_prefill_impl_deepseek_v3`

This document details **all** changes made inside the `tt-mlir` submodule to get the
following test to run and pass:

```
pytest -svv "tests/integrations/vllm_plugin/test_mla_prefill_impl.py::test_mla_prefill_impl_deepseek_v3[1-64]"
```

Two independent bugs blocked the test, both fixed here:

1. **Custom-call fix** — a StableHLO sharding rule wrongly rejected the canonical MLA
   prefill input (`num_kv_heads == 1`) with a hard compile error.
2. **L1 fix** — the `permute+reshape` head-merge was fused into a `ttnn.concatenate_heads`
   op whose per-core circular buffer overflowed L1 for DeepSeek-V3's 16384-wide attention
   output, crashing at runtime.

All paths below are **relative to the tt-mlir checkout**:
`/localdev/hshah/tt-xla/third_party/tt-mlir/src/tt-mlir`
(tt-mlir submodule HEAD at time of change: `e018513cd`; tt-xla branch: `hshah/mla-vllm-new`).

> ⚠️ These edits live in the `tt-mlir` **submodule** and are currently uncommitted there.
> They must be committed in tt-mlir and the submodule pointer bumped in tt-xla.

---

## Summary of changes

| # | File | Lines (current) | Kind | What |
|---|------|-----------------|------|------|
| 1 | `lib/Dialect/StableHLO/Transforms/RegisterCustomShardingRule.cpp` | removed old `421–427`; edited head factor at `463–476` | Source | Remove `kvHeads == 1` rejection; replicate the single latent KV head under head sharding |
| 2 | `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` | added include at `6`; guard at `2181–2211` | Source | L1-fit guard on the `concatenate_heads` fusion |
| 3 | `test/ttmlir/Dialect/StableHLO/register_custom_sharding_rule/custom_op_flash_mla_prefill.mlir` | `138–159` | Test | Convert the "MQA rejected" case into an accepted latent-KV head-parallel case |
| 4 | `test/ttmlir/Dialect/TTIR/fusing/concatenate_heads_l1_guard.mlir` | new file (38 lines) | Test | New lit test pinning the L1 guard (negative + positive) |

---

## Change 1 — Custom-call fix (`RegisterCustomShardingRule.cpp`)

**File:** `lib/Dialect/StableHLO/Transforms/RegisterCustomShardingRule.cpp`
**Function:** `getFlashMlaPrefillShardingRule(mlir::stablehlo::CustomCallOp op)` (begins at **line 315**)

### Why

MLA compresses K/V into a **single shared latent head**, so `num_kv_heads == 1` is the
*canonical MLA prefill* form. This is confirmed by:

- the TTNN op's own tests (`tests/.../sdpa/test_mla_prefill.py`) — every parametrization uses `nkv=1`;
- the op verifiers `ttir::FlashMlaPrefillOp::verify` / `ttnn::FlashMlaPrefillOp::verify`, which only require `qHeads % kvHeads == 0` and explicitly say "GQA/MQA/MLA";
- the frontend `integrations/vllm_plugin/vllm_tt/attention_mla.py`, which builds the key as `[b, 1, S, L+R]`.

A check added earlier on this branch (commit `6de2c7929`, "make sharding rules MHA specific
for MLA prefill") `emitError`'d on `kvHeads == 1`, contradicting all of the above. The error
seen in the test was:

```
loc("custom-call.22"): error: flash_mla_prefill (MLA prefill) expects MHA inputs but got
num_kv_heads == 1 (MQA); MQA is the decode form and must not reach the prefill op
```

### What changed

**(a) Deleted the rejection block.** It previously sat immediately after the
`qHeads % kvHeads != 0` check (that check is now at **lines 414–419**); the deleted block was
at the old lines **421–427**. After deletion, `int64_t maskBatch = sdy::kNullDim;` is now at
**line 421**.

Removed:
```cpp
  if (kvHeads == 1) {
    op.getOperation()->emitError()
        << "flash_mla_prefill (MLA prefill) expects MHA inputs but got "
           "num_kv_heads == 1 (MQA); MQA is the decode form and must not reach "
           "the prefill op";
    return mlir::sdy::OpShardingRuleAttr();
  }
```

**(b) Fixed the Heads sharding factor** so the single latent KV head is *replicated* (not
split) when query heads are sharded — it cannot be divided across devices. Now at
**lines 463–476**:

```cpp
  // Heads (dim 1): kPassThrough, factor size qHeads. Out always carries the
  // full qHeads. Mask heads are always 1 so the mask sits out of this factor
  // (kNullDim).
  //
  // MLA's compressed latent K/V is a single shared head (kvHeads == 1) that is
  // broadcast across every query head — it cannot be split, and must stay
  // replicated when the query heads are sharded. So leave K/V's head dim out of
  // the factor (kNullDim) in that case; only Q/Out participate. When K/V
  // materialize heads (kvHeads == qHeads, the per-head form) or in the grouped
  // case, map K/V dim 1 into the shared factor and let Shardy split it
  // proportionally (same trick as getSDPAShardingRule).
  int64_t kvHeadDim = (kvHeads == 1) ? sdy::kNullDim : 1;
  builder.addFactor(makeOpDims(1, kvHeadDim, kvHeadDim, sdy::kNullDim), {1},
                    qHeads, sdy::FactorType::kPassThrough);
```

Previously (single line, no `kvHeads==1` handling):
```cpp
  builder.addFactor(makeOpDims(1, 1, 1, sdy::kNullDim), {1}, qHeads,
                    sdy::FactorType::kPassThrough);
```

The `kvHeads == qHeads` (materialized) path is byte-for-byte unchanged (`kvHeadDim == 1`).

---

## Change 2 — L1 fix (`TTIRFusing.cpp`)

**File:** `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp`
**Pattern:** `ConcatenateHeadsUpdatePattern` (class at **line 2098**), method
`isFusable(PermuteOp, ReshapeOp)` (at **line 2170**).

### Why

The MLA forward's head-merge (`attention_mla.py`, the
`out.transpose(1,2).reshape(users*S, N*V)` step) is matched by `ConcatenateHeadsUpdatePattern`
(`permute([0,2,1,3]) + reshape` → `ttir.concatenate_heads`) and lowered to ttnn's
`nlp_concat_heads`.

That kernel sizes its per-core source circular buffer to the **full concatenated hidden
width**, double-buffered
(`third_party/tt-metal/.../nlp_concat_heads/device/nlp_concat_heads_program_factory.cpp:34,130-132`):

```
cb_src0 = 2 * (num_heads * head_dim / TILE_WIDTH) tiles
```

Every core allocates this regardless of core count. For DeepSeek-V3 MLA
(`128 heads × 128 head_dim = 16384` hidden, bf16): `2 × 512 × 2048 B ≈ 2 MB`, exceeding the
~1.43 MB usable L1. The op cannot be compiled and crashes at runtime:

```
Statically allocated circular buffers on core range [0-0 - 0-1] grow to 2200896 B
which is beyond max L1 size of 1499136 B
```

(vLLM's `TTNNOperationValidationAndFallback` would catch this, but it only runs when the
optimizer is enabled — which it is not in the default PJRT flow — hence the hard crash.)

### What changed

**(a) Added an include** at **line 6**:
```cpp
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
```

**(b) Added an L1-fit guard** at the top of `isFusable`, **lines 2181–2211** (right after the
`inputShape.size() != 4` check and before the permutation check at line 2212). It reconstructs
the kernel's `cb_src0` size from `num_heads × head_dim` and the element size, reads usable L1
from the chip desc (the system desc is attached by `TTCoreRegisterDevicePass`, which runs
*before* fusing), and declines the fusion when the CB won't fit. The op then stays as the
generic `permute + reshape`, whose transpose lowering streams a few tiles at a time and stays
within L1. When no system desc is present (device-less lit pipelines) the guard is skipped,
preserving existing behavior.

```cpp
    // ttnn.concatenate_heads (nlp_concat_heads, interleaved) sizes its per-core
    // source circular buffer to the FULL concatenated hidden width and double-
    // buffers it: cb_src0 = 2 * (num_heads * head_size / TILE_WIDTH) tiles.
    // Every core allocates this regardless of the core count, so for a large
    // hidden (e.g. DeepSeek-V3 MLA: 128 heads x 128 head_dim = 16384) the CB
    // exceeds usable L1 and the program fails to compile. Decline the fusion in
    // that case and leave the generic permute+reshape, whose transpose lowering
    // streams a few tiles at a time and stays within L1. The system desc is
    // attached before this pass runs (TTCoreRegisterDevicePass); if it is
    // absent (e.g. a device-less test pipeline) skip the check and fuse.
    if (auto moduleOp = reshapeOp->getParentOfType<mlir::ModuleOp>()) {
      if (auto systemDesc =
              moduleOp->getAttrOfType<mlir::tt::ttcore::SystemDescAttr>(
                  mlir::tt::ttcore::SystemDescAttr::name)) {
        uint64_t elemSizeBytes = mlir::tt::ttcore::getElementSizeBytes(
            permuteOp.getInput().getType().getElementType());
        constexpr uint64_t kTileWidth = mlir::tt::ttnn::TILE_WIDTH;
        uint64_t hidden = static_cast<uint64_t>(inputShape[INPUT_NUM_HEADS]) *
                          inputShape[INPUT_HEAD_SIZE];
        uint64_t perTensorTiles =
            ttmlir::utils::alignUp(hidden, kTileWidth) / kTileWidth;
        // TT tiles are square (TILE_WIDTH x TILE_WIDTH).
        uint64_t tileSizeBytes = kTileWidth * kTileWidth * elemSizeBytes;
        uint64_t cbBytes = 2 * perTensorTiles * tileSizeBytes;
        uint64_t usableL1 = systemDesc.getChipDescs()[0].getUsableL1Size();
        if (cbBytes > usableL1) {
          return false;
        }
      }
    }
```

---

## Change 3 — Lit test update (`custom_op_flash_mla_prefill.mlir`)

**File:** `test/ttmlir/Dialect/StableHLO/register_custom_sharding_rule/custom_op_flash_mla_prefill.mlir`
**Lines:** the final `// -----` section, now **138–159**.

The module `FlashMlaPrefill_MQA_Rejected` (which asserted the now-removed error via
`expected-error`) was converted into `FlashMlaPrefill_Sharding_LatentKV_HeadParallel`, an
**accepted** test: Q's 128 heads shard to 64 while the single latent KV head stays replicated,
with no collective. This both documents the corrected semantics and pins Change 1.

---

## Change 4 — New lit test (`concatenate_heads_l1_guard.mlir`)

**File (new):** `test/ttmlir/Dialect/TTIR/fusing/concatenate_heads_l1_guard.mlir` (38 lines)

Pins the L1 guard from Change 2 using a registered `wormhole_b0` device:

- **Negative:** `128 heads × 128 head_dim = 16384` hidden → does **not** fuse (the
  `permute`/`reshape` remain, `concatenate_heads` is absent).
- **Positive:** `24 heads × 128 head_dim = 3072` hidden → still fuses to `concatenate_heads`,
  confirming the guard does not over-restrict normal models.

---

## Build / deploy

The two source files compile into `libMLIRStableHLOTransforms.a` and `libMLIRTTIRTransforms.a`,
which link into `libTTMLIRCompiler.so`. Rebuild and install (from
`third_party/tt-mlir/src/tt-mlir/build`):

```bash
ninja TTMLIRCompiler
cmake --install . --component SharedLib   # -> third_party/tt-mlir/install/lib/libTTMLIRCompiler.so
```

The PJRT plugin (`build/pjrt_implementation/src/pjrt_plugin_tt.so`, symlinked into
`python_package/pjrt_plugin_tt/`) dynamically loads `libTTMLIRCompiler.so` from the install
prefix via its RUNPATH, so **no plugin relink is required** — the rebuilt `.so` is picked up
directly.

---

## Verification

- **Target pytest** — all three parametrizations pass:
  `test_mla_prefill_impl_deepseek_v3[1-64]`, `[2-64]`, `[1-128]` → `3 passed`.
  No custom-call error, no `concatenate_heads`/L1 error.
- **Lit tests** (run via `ttmlir-opt` + `FileCheck`):
  - `custom_op_flash_mla_prefill.mlir` — PASS (Change 3).
  - `concatenate_heads_l1_guard.mlir` — PASS (Change 4).
  - `concatenate_heads_fusing.mlir` (no device → guard skipped, still fuses) — PASS.
  - `concatenate_heads_positive.mlir` (full `--ttir-to-ttnn-backend-pipeline` with a device,
    hidden=3072 → still fuses) — PASS (no regression).
  - `concatenate_heads_negative.mlir` (TTIR op verifier) — PASS.

---

# Appendix — Full code

## A.1 `lib/Dialect/StableHLO/Transforms/RegisterCustomShardingRule.cpp` (unified diff)

```diff
diff --git a/lib/Dialect/StableHLO/Transforms/RegisterCustomShardingRule.cpp b/lib/Dialect/StableHLO/Transforms/RegisterCustomShardingRule.cpp
index 3155d02b8..2877f7998 100644
--- a/lib/Dialect/StableHLO/Transforms/RegisterCustomShardingRule.cpp
+++ b/lib/Dialect/StableHLO/Transforms/RegisterCustomShardingRule.cpp
@@ -418,14 +418,6 @@ getFlashMlaPrefillShardingRule(mlir::stablehlo::CustomCallOp op) {
     return mlir::sdy::OpShardingRuleAttr();
   }
 
-  if (kvHeads == 1) {
-    op.getOperation()->emitError()
-        << "flash_mla_prefill (MLA prefill) expects MHA inputs but got "
-           "num_kv_heads == 1 (MQA); MQA is the decode form and must not reach "
-           "the prefill op";
-    return mlir::sdy::OpShardingRuleAttr();
-  }
-
   int64_t maskBatch = sdy::kNullDim;
   if (hasAttentionMask) {
     ArrayRef<int64_t> mShape = mType.getShape();
@@ -468,10 +460,20 @@ getFlashMlaPrefillShardingRule(mlir::stablehlo::CustomCallOp op) {
   builder.addFactor(makeOpDims(0, 0, 0, maskBatch), {0}, B,
                     sdy::FactorType::kPassThrough);
 
-  // Heads (dim 1): kPassThrough, factor size qHeads. Mask heads are always 1
-  // so the mask sits out of this factor (kNullDim).
-  builder.addFactor(makeOpDims(1, 1, 1, sdy::kNullDim), {1}, qHeads,
-                    sdy::FactorType::kPassThrough);
+  // Heads (dim 1): kPassThrough, factor size qHeads. Out always carries the
+  // full qHeads. Mask heads are always 1 so the mask sits out of this factor
+  // (kNullDim).
+  //
+  // MLA's compressed latent K/V is a single shared head (kvHeads == 1) that is
+  // broadcast across every query head — it cannot be split, and must stay
+  // replicated when the query heads are sharded. So leave K/V's head dim out of
+  // the factor (kNullDim) in that case; only Q/Out participate. When K/V
+  // materialize heads (kvHeads == qHeads, the per-head form) or in the grouped
+  // case, map K/V dim 1 into the shared factor and let Shardy split it
+  // proportionally (same trick as getSDPAShardingRule).
+  int64_t kvHeadDim = (kvHeads == 1) ? sdy::kNullDim : 1;
+  builder.addFactor(makeOpDims(1, kvHeadDim, kvHeadDim, sdy::kNullDim), {1},
+                    qHeads, sdy::FactorType::kPassThrough);
 
   // Sequence (dim 2): kNeedReplication, shared across Q/K/V/Out/mask.
   builder.addFactor(makeOpDims(2, 2, 2, 2), {2}, S,
```

## A.2 `lib/Dialect/TTIR/Transforms/TTIRFusing.cpp` (unified diff)

```diff
diff --git a/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp b/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp
index 179ae2985..c123d6258 100644
--- a/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp
+++ b/lib/Dialect/TTIR/Transforms/TTIRFusing.cpp
@@ -3,6 +3,7 @@
 // SPDX-License-Identifier: Apache-2.0
 
 #include "ttmlir/Asserts.h"
+#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
 #include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
 #include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
 #include "ttmlir/Dialect/TTIR/Utils/Utils.h"
@@ -2177,6 +2178,37 @@ private:
       return false;
     }
 
+    // ttnn.concatenate_heads (nlp_concat_heads, interleaved) sizes its per-core
+    // source circular buffer to the FULL concatenated hidden width and double-
+    // buffers it: cb_src0 = 2 * (num_heads * head_size / TILE_WIDTH) tiles.
+    // Every core allocates this regardless of the core count, so for a large
+    // hidden (e.g. DeepSeek-V3 MLA: 128 heads x 128 head_dim = 16384) the CB
+    // exceeds usable L1 and the program fails to compile. Decline the fusion in
+    // that case and leave the generic permute+reshape, whose transpose lowering
+    // streams a few tiles at a time and stays within L1. The system desc is
+    // attached before this pass runs (TTCoreRegisterDevicePass); if it is
+    // absent (e.g. a device-less test pipeline) skip the check and fuse.
+    if (auto moduleOp = reshapeOp->getParentOfType<mlir::ModuleOp>()) {
+      if (auto systemDesc =
+              moduleOp->getAttrOfType<mlir::tt::ttcore::SystemDescAttr>(
+                  mlir::tt::ttcore::SystemDescAttr::name)) {
+        uint64_t elemSizeBytes = mlir::tt::ttcore::getElementSizeBytes(
+            permuteOp.getInput().getType().getElementType());
+        constexpr uint64_t kTileWidth = mlir::tt::ttnn::TILE_WIDTH;
+        uint64_t hidden = static_cast<uint64_t>(inputShape[INPUT_NUM_HEADS]) *
+                          inputShape[INPUT_HEAD_SIZE];
+        uint64_t perTensorTiles =
+            ttmlir::utils::alignUp(hidden, kTileWidth) / kTileWidth;
+        // TT tiles are square (TILE_WIDTH x TILE_WIDTH).
+        uint64_t tileSizeBytes = kTileWidth * kTileWidth * elemSizeBytes;
+        uint64_t cbBytes = 2 * perTensorTiles * tileSizeBytes;
+        uint64_t usableL1 = systemDesc.getChipDescs()[0].getUsableL1Size();
+        if (cbBytes > usableL1) {
+          return false;
+        }
+      }
+    }
+
     // Check if the permutation is {0, 2, 1, 3}.
     llvm::ArrayRef<int64_t> permutation = permuteOp.getPermutation();
     llvm::SmallVector<int64_t> expectedPermutation = {
```

## A.3 `test/ttmlir/Dialect/StableHLO/register_custom_sharding_rule/custom_op_flash_mla_prefill.mlir` (unified diff)

```diff
diff --git a/test/ttmlir/Dialect/StableHLO/register_custom_sharding_rule/custom_op_flash_mla_prefill.mlir b/test/ttmlir/Dialect/StableHLO/register_custom_sharding_rule/custom_op_flash_mla_prefill.mlir
index e681c2c84..518efc678 100644
--- a/test/ttmlir/Dialect/StableHLO/register_custom_sharding_rule/custom_op_flash_mla_prefill.mlir
+++ b/test/ttmlir/Dialect/StableHLO/register_custom_sharding_rule/custom_op_flash_mla_prefill.mlir
@@ -135,10 +135,25 @@ module @FlashMlaPrefill_Sharding_HeadDimV_NeedsAllGather attributes {mhlo.cross_
 
 // -----
 
-// MQA -- a single shared KV head (kvHeads == 1) is the MLA *decode* form.
-module @FlashMlaPrefill_MQA_Rejected attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
+// Latent KV (the canonical MLA prefill form): the compressed K/V is a single
+// shared head (kvHeads == 1) broadcast across all query heads, with value
+// omitted (V is the leading head_dim_v features of K). This is exactly what the
+// frontend emits and what ttnn.transformer.flash_mla_prefill consumes (its
+// op tests only ever use num_kv_heads == 1).
+//   Hq         = 128   (num_attention_heads)
+//   Hkv        = 1     (shared latent KV head)
+//   dh_qk      = 576   (kv_lora_rank 512 + qk_rope_head_dim 64)
+//   head_dim_v = 512   (kv_lora_rank)
+// Head (tensor) parallel splits the 128 query heads (-> 64) on Q/Out, while the
+// single latent KV head stays replicated (it sits out of the head factor), so
+// no collective is required.
+// CHECK-LABEL: module @FlashMlaPrefill_Sharding_LatentKV_HeadParallel
+module @FlashMlaPrefill_Sharding_LatentKV_HeadParallel attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
   func.func @main(%query: tensor<1x128x2048x576xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}, {}, {}]>"}, mhlo.sharding = "{devices=[1,2,1,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "query"}, %key: tensor<1x1x2048x576xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}, {}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "key"}) -> tensor<1x128x2048x512xbf16> {
-    // expected-error @+1 {{flash_mla_prefill (MLA prefill) expects MHA inputs but got num_kv_heads == 1 (MQA)}}
+    // CHECK-NOT: stablehlo.all_gather
+    // CHECK: stablehlo.custom_call @tt.flash_mla_prefill
+    // CHECK-SAME: tensor<1x64x2048x576xbf16>, tensor<1x1x2048x576xbf16>
+    // CHECK-SAME: -> tensor<1x64x2048x512xbf16>
     %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "512", is_causal = "True", has_value = "False", has_attention_mask = "False"}} : (tensor<1x128x2048x576xbf16>, tensor<1x1x2048x576xbf16>) -> tensor<1x128x2048x512xbf16>
     return %0 : tensor<1x128x2048x512xbf16>
   }
```

## A.4 `test/ttmlir/Dialect/TTIR/fusing/concatenate_heads_l1_guard.mlir` (full new file)

```mlir
// RUN: ttmlir-opt -split-input-file --ttcore-register-device="mock-system-desc-arch=wormhole_b0" --ttir-fusing %s | FileCheck %s

// The permute([0,2,1,3]) + reshape head-merge fuses into ttir.concatenate_heads
// only when the resulting nlp_concat_heads op fits in L1. That op sizes its
// per-core source circular buffer to the FULL concatenated hidden width and
// double-buffers it (cb_src0 = 2 * num_heads * head_size / TILE_WIDTH tiles),
// so a large hidden cannot be compiled. These tests pin the L1-aware guard in
// ConcatenateHeadsUpdatePattern (a wormhole_b0 device is registered so the
// system desc / usable L1 is available to the fusing pass).

// NEGATIVE: 128 heads x 128 head_dim = 16384 hidden. The double-buffered bf16
// CB is ~2 MB > the ~1.43 MB usable L1, so the fusion must NOT fire — the
// generic permute + reshape (transpose streams a few tiles at a time) is kept.
module @ConcatHeads_TooLargeForL1 {
  // CHECK-LABEL: func.func @too_large
  func.func @too_large(%arg0: tensor<1x128x64x128xbf16>) -> tensor<64x16384xbf16> {
    // CHECK-NOT: ttir.concatenate_heads
    // CHECK: ttir.permute
    // CHECK: ttir.reshape
    %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x128x64x128xbf16>) -> tensor<1x64x128x128xbf16>
    %1 = "ttir.reshape"(%0) <{shape = [64 : i32, 16384 : i32]}> : (tensor<1x64x128x128xbf16>) -> tensor<64x16384xbf16>
    return %1 : tensor<64x16384xbf16>
  }
}

// -----

// POSITIVE: 24 heads x 128 head_dim = 3072 hidden. The CB easily fits in L1, so
// the fusion fires as usual even with a device registered.
module @ConcatHeads_FitsInL1 {
  // CHECK-LABEL: func.func @fits
  func.func @fits(%arg0: tensor<1x24x32x128xbf16>) -> tensor<1x32x3072xbf16> {
    // CHECK: ttir.concatenate_heads
    %0 = "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x24x32x128xbf16>) -> tensor<1x32x24x128xbf16>
    %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 32 : i32, 3072 : i32]}> : (tensor<1x32x24x128xbf16>) -> tensor<1x32x3072xbf16>
    return %1 : tensor<1x32x3072xbf16>
  }
}
```
