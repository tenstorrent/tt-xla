# SDPA Fold-Scale Workaround Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a TTNN-level rewrite pattern that detects a residual scalar multiply on Q and/or K feeding `ttnn.scaled_dot_product_attention` (through optional `typecast`/`permute`) and folds the scalar into the SDPA's `scale` attribute, removing the multiply.

**Architecture:** New `OpRewritePattern<ScaledDotProductAttentionOp>` in `lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/`, registered in the existing `--ttnn-workaround` pass. The constant-extraction helpers currently private to `SDPAFusing` are hoisted into a shared `Dialect/TTNN/Utils/SDPAUtils` module so both the TTIR-time fusing path and the new TTNN-time workaround read the same constant-detection logic.

**Tech Stack:** C++17, MLIR (LLVM 20), tt-mlir TTNN dialect, llvm-lit + FileCheck.

**Spec:** `docs/superpowers/specs/2026-05-06-sdpa-fold-scale-workaround-design.md`

---

## File Map

**Create:**
- `third_party/tt-mlir/src/tt-mlir/include/ttmlir/Dialect/TTNN/Utils/SDPAUtils.h`
- `third_party/tt-mlir/src/tt-mlir/lib/Dialect/TTNN/Utils/SDPAUtils.cpp`
- `third_party/tt-mlir/src/tt-mlir/include/ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionFoldScaleRewritePattern.h`
- `third_party/tt-mlir/src/tt-mlir/lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionFoldScaleRewritePattern.cpp`
- `third_party/tt-mlir/src/tt-mlir/test/ttmlir/Dialect/TTNN/Transforms/Workarounds/sdpa_fold_scale.mlir`

**Modify:**
- `third_party/tt-mlir/src/tt-mlir/lib/Dialect/TTNN/Utils/CMakeLists.txt` — add `SDPAUtils.cpp`
- `third_party/tt-mlir/src/tt-mlir/lib/Dialect/TTNN/Transforms/CMakeLists.txt` — add new workaround source
- `third_party/tt-mlir/src/tt-mlir/include/ttmlir/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.h` — drop two private method decls
- `third_party/tt-mlir/src/tt-mlir/lib/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.cpp` — delete the two helpers, include shared header, switch call sites to free functions
- `third_party/tt-mlir/src/tt-mlir/lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp` — `#include` new pattern header, add to `patterns.add<...>` list

**Working directory for all commands below:** `/root/tt-xla/third_party/tt-mlir/src/tt-mlir`

**Activate the env once before any build/test command:** `source env/activate`

---

## Task 1: Hoist constant-extraction helpers into shared `SDPAUtils`

The two helpers in `Fusing/SDPAFusingPattern.cpp` (`extractConstant` lines 126-159 and `extractMultiplyWithConstant` lines 161-172) are exactly what the new workaround needs. The existing pattern is gated on `TTMLIR_ENABLE_OPMODEL` (see `lib/Dialect/TTNN/Transforms/CMakeLists.txt` `OPMODEL_SRCS`). The new workaround is *not* gated, so the shared utility must live in always-built code (`lib/Dialect/TTNN/Utils/`).

**Files:**
- Create: `include/ttmlir/Dialect/TTNN/Utils/SDPAUtils.h`
- Create: `lib/Dialect/TTNN/Utils/SDPAUtils.cpp`
- Modify: `lib/Dialect/TTNN/Utils/CMakeLists.txt`
- Modify: `include/ttmlir/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.h`
- Modify: `lib/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.cpp`

- [ ] **Step 1.1: Create `SDPAUtils.h`**

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_SDPAUTILS_H
#define TTMLIR_DIALECT_TTNN_UTILS_SDPAUTILS_H

#include "mlir/IR/Value.h"

#include <optional>
#include <utility>

namespace mlir::tt::ttnn::utils {

// Returns the float value of a tensor that is constant-broadcast from a
// scalar. Looks through one outer ttnn.typecast on `v`. The constant must
// originate from either a `ttnn.full` op (FillValue is a FloatAttr) or from
// a `ttcore.load_cached` op whose callee returns a `ttnn.full` value.
//
// Returns std::nullopt for any other producer.
std::optional<float> extractScalarConstant(Value v);

// If `v` is the result of a `ttnn.multiply` whose other input is a scalar
// constant (per `extractScalarConstant`), returns `{nonScalarInput, scalar}`.
// Otherwise returns `{v, nullopt}`.
//
// The returned `nonScalarInput` is the multiply's input that is NOT the
// scalar, which can be substituted for the multiply's result with no shape
// change because multiply with a 1x1x1x1 broadcast preserves the larger
// operand's shape.
std::pair<Value, std::optional<float>> extractMultiplyWithScalarConstant(Value v);

} // namespace mlir::tt::ttnn::utils

#endif // TTMLIR_DIALECT_TTNN_UTILS_SDPAUTILS_H
```

- [ ] **Step 1.2: Create `SDPAUtils.cpp`** (function bodies moved verbatim from `SDPAFusingPattern.cpp` lines 126-172)

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/SDPAUtils.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::tt::ttnn::utils {

std::optional<float> extractScalarConstant(Value v) {
  v = ttmlir::utils::lookThrough<TypecastOp>(v);

  if (auto fullOp = v.getDefiningOp<FullOp>()) {
    if (auto attr = mlir::dyn_cast<FloatAttr>(fullOp.getFillValue())) {
      return attr.getValue().convertToFloat();
    }
  }

  if (auto loadCached = v.getDefiningOp<ttcore::LoadCachedOp>()) {
    auto callee = loadCached.getCallee();
    auto moduleOp = loadCached->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      return std::nullopt;
    }

    auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(callee);
    if (!funcOp) {
      return std::nullopt;
    }

    std::optional<float> result;
    funcOp.walk([&](FullOp fullOp) {
      if (auto attr = mlir::dyn_cast<FloatAttr>(fullOp.getFillValue())) {
        result = attr.getValue().convertToFloat();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return result;
  }

  return std::nullopt;
}

std::pair<Value, std::optional<float>>
extractMultiplyWithScalarConstant(Value v) {
  if (auto mulOp = v.getDefiningOp<MultiplyOp>()) {
    if (auto s = extractScalarConstant(mulOp.getRhs())) {
      return {mulOp.getLhs(), s};
    }
    if (auto s = extractScalarConstant(mulOp.getLhs())) {
      return {mulOp.getRhs(), s};
    }
  }
  return {v, std::nullopt};
}

} // namespace mlir::tt::ttnn::utils
```

- [ ] **Step 1.3: Add `SDPAUtils.cpp` to `lib/Dialect/TTNN/Utils/CMakeLists.txt`**

Insert one new line in alphabetical order (after `OptimizerUtils.cpp`):

```cmake
add_mlir_dialect_library(TTMLIRTTNNUtils
  Conv2dConfigParams.cpp
  D2MOptimizerUtils.cpp
  OptimizerOverrides.cpp
  OptimizerUtils.cpp
  PassOverrides.cpp
  SDPAUtils.cpp
  TransformUtils.cpp
  Utils.cpp
  VerificationUtils.cpp


  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/TTNN

  LINK_LIBS
  PRIVATE
  MLIRTTCoreDialect
)
```

- [ ] **Step 1.4: Drop the two private method decls from `SDPAFusingPattern.h`**

In `include/ttmlir/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.h`, delete these three lines (originally at the "Constant Extraction" comment, lines 41-44 in the snapshot):

```cpp
  // Constant Extraction
  std::optional<float> extractConstant(Value v) const;
  std::pair<Value, std::optional<float>>
  extractMultiplyWithConstant(Value v) const;
```

- [ ] **Step 1.5: Delete the two helper bodies in `SDPAFusingPattern.cpp` and update call sites**

In `lib/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.cpp`:

(a) Add the include near the existing TTNN includes (alphabetical):
```cpp
#include "ttmlir/Dialect/TTNN/Utils/SDPAUtils.h"
```

(b) Delete the two member-function bodies (the entire blocks for `SDPAFusing::extractConstant` lines 126-159 and `SDPAFusing::extractMultiplyWithConstant` lines 161-172, including the two preceding section comment header at lines 122-124).

(c) Replace the four internal call sites that referenced `this->extractConstant(...)` or `extractMultiplyWithConstant(...)` with the free-function versions in `mlir::tt::ttnn::utils::`:

| Old call | New call |
|---|---|
| `extractConstant(mulOp.getRhs())` (line 272) | `mlir::tt::ttnn::utils::extractScalarConstant(mulOp.getRhs())` |
| `extractConstant(mulOp.getLhs())` (line 275) | `mlir::tt::ttnn::utils::extractScalarConstant(mulOp.getLhs())` |
| `extractConstant(divOp.getRhs())` (line 282) | `mlir::tt::ttnn::utils::extractScalarConstant(divOp.getRhs())` |
| `extractMultiplyWithConstant(skipped)` (line 309 in `analyzeQ`) | `mlir::tt::ttnn::utils::extractMultiplyWithScalarConstant(skipped)` |
| `extractMultiplyWithConstant(v)` (line 335 in `analyzeK`) | `mlir::tt::ttnn::utils::extractMultiplyWithScalarConstant(v)` |

Use grep to be sure: `grep -n 'extractConstant\|extractMultiplyWithConstant' lib/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.cpp` after edits should return nothing.

- [ ] **Step 1.6: Build**

```bash
source env/activate
cmake --build build
```

Expected: build succeeds. If it doesn't, the most likely failure modes are (a) missing include path on `SDPAUtils.h`, (b) ambiguous `extractScalarConstant` due to `using namespace` collision (qualify with `mlir::tt::ttnn::utils::`).

- [ ] **Step 1.7: Verify existing TTIR-time SDPA fusing tests still pass**

```bash
source env/activate
llvm-lit -v test/ttmlir/Dialect/TTNN/optimizer/ttnn_fusing/scaled_dot_product_attention
```

Expected: all tests pass. The refactor is behavior-preserving.

- [ ] **Step 1.8: Commit**

```bash
git -C third_party/tt-mlir/src/tt-mlir add \
  include/ttmlir/Dialect/TTNN/Utils/SDPAUtils.h \
  lib/Dialect/TTNN/Utils/SDPAUtils.cpp \
  lib/Dialect/TTNN/Utils/CMakeLists.txt \
  include/ttmlir/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.h \
  lib/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.cpp
git -C third_party/tt-mlir/src/tt-mlir commit -m "Hoist SDPA constant-extraction helpers into shared TTNN utility

Moves extractConstant and extractMultiplyWithConstant from
SDPAFusing private methods to free functions in
mlir::tt::ttnn::utils, so the upcoming TTNN-level workaround for
folding residual scalar multiplies into SDPA scale can share the
same constant detection. No behavior change."
```

---

## Task 2: Pattern skeleton — Q-side fold with `scale=1.0` (TDD red→green)

Write the simplest case first: a Q operand whose chain is `multiply(q, full(c)) → typecast → SDPA(...)`, SDPA scale unset (or 1.0), expect rewrite to remove multiply and set `scale = c`.

**Files:**
- Create: `test/ttmlir/Dialect/TTNN/Transforms/Workarounds/sdpa_fold_scale.mlir`
- Create: `include/ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionFoldScaleRewritePattern.h`
- Create: `lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionFoldScaleRewritePattern.cpp`
- Modify: `lib/Dialect/TTNN/Transforms/CMakeLists.txt`
- Modify: `lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp`

- [ ] **Step 2.1: Write the failing lit test (Q-side, scale=1.0)**

Create `test/ttmlir/Dialect/TTNN/Transforms/Workarounds/sdpa_fold_scale.mlir`:

```mlir
// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#l_qkv = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 6 * 8 + d1 * 8 + d2, d3), <1x1>, memref<48x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#l_scalar = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module attributes {} {
  // CHECK-LABEL: q_side_multiply_only
  func.func @q_side_multiply_only(
      %q: tensor<1x6x256x128xbf16, #l_qkv>,
      %k: tensor<1x6x256x128xbf16, #l_qkv>,
      %v: tensor<1x6x256x128xbf16, #l_qkv>,
      %device: !ttcore.device) -> tensor<1x6x256x128xbf16, #l_qkv> {
    %scalar = "ttnn.full"(%device) <{fill_value = 8.838834e-02 : f32, shape = #ttnn.shape<1x1x1x1>}> : (!ttcore.device) -> tensor<1x1x1x1xf32, #l_scalar>
    %q_scaled = "ttnn.multiply"(%q, %scalar) : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x1x1x1xf32, #l_scalar>) -> tensor<1x6x256x128xbf16, #l_qkv>

    // CHECK-NOT: ttnn.multiply
    // CHECK: %[[OUT:.*]] = "ttnn.scaled_dot_product_attention"(%arg0, %arg1, %arg2)
    // CHECK-SAME: scale = 8.838834{{[0-9]*}}e-02 : f32
    %out = "ttnn.scaled_dot_product_attention"(%q_scaled, %k, %v)
        <{is_causal = false, scale = 1.000000e+00 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}>
        : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>) -> tensor<1x6x256x128xbf16, #l_qkv>
    return %out : tensor<1x6x256x128xbf16, #l_qkv>
  }
}
```

- [ ] **Step 2.2: Run the test — expect FAIL**

```bash
source env/activate
llvm-lit -v test/ttmlir/Dialect/TTNN/Transforms/Workarounds/sdpa_fold_scale.mlir
```

Expected: test FAILs because the pattern doesn't exist yet — `ttnn.multiply` is still present and `scale` is still `1.0`. This confirms the test is wired up to the right pass.

- [ ] **Step 2.3: Create the pattern header**

`include/ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionFoldScaleRewritePattern.h`:

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONFOLDSCALEREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONFOLDSCALEREWRITEPATTERN_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

// Folds residual scalar multiplies on the Q and/or K operands of a
// ttnn.scaled_dot_product_attention into the SDPA's `scale` attribute.
// Looks through optional one-use ttnn.typecast and ttnn.permute ops between
// the multiply and the SDPA. Mathematically equivalent: SDPA(c*Q, K, V, s) ==
// SDPA(Q, K, V, s*c). Combines existing scale with detected scalars.
class ScaledDotProductAttentionFoldScaleRewritePattern
    : public OpRewritePattern<ScaledDotProductAttentionOp> {
public:
  using OpRewritePattern<ScaledDotProductAttentionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScaledDotProductAttentionOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_SCALEDDOTPRODUCTATTENTIONFOLDSCALEREWRITEPATTERN_H
```

- [ ] **Step 2.4: Create the pattern implementation (Q-side only for now)**

`lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionFoldScaleRewritePattern.cpp`:

```cpp
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionFoldScaleRewritePattern.h"

#include "ttmlir/Dialect/TTNN/Utils/SDPAUtils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

namespace {

struct FoldHit {
  MultiplyOp multiplyOp; // the multiply to bypass
  Value bypassValue;     // value that should replace multiplyOp's result
  float scalar;          // the scalar to fold into SDPA's scale
};

// Walks back from `v` along the def-use chain through any one-use ttnn.typecast
// or ttnn.permute ops, looking for a one-use ttnn.multiply whose other input
// is a scalar constant. Returns the found multiply and scalar, or nullopt.
static std::optional<FoldHit> findUpstreamScaleMultiply(Value v) {
  Value cur = v;
  while (Operation *defOp = cur.getDefiningOp()) {
    if (isa<TypecastOp, PermuteOp>(defOp)) {
      if (!defOp->hasOneUse()) {
        return std::nullopt;
      }
      cur = defOp->getOperand(0);
      continue;
    }

    if (auto mulOp = dyn_cast<MultiplyOp>(defOp)) {
      if (!mulOp->hasOneUse()) {
        return std::nullopt;
      }
      auto [bypass, scalar] =
          mlir::tt::ttnn::utils::extractMultiplyWithScalarConstant(mulOp);
      if (!scalar) {
        return std::nullopt;
      }
      return FoldHit{mulOp, bypass, *scalar};
    }

    return std::nullopt;
  }
  return std::nullopt;
}

} // namespace

LogicalResult
ScaledDotProductAttentionFoldScaleRewritePattern::matchAndRewrite(
    ScaledDotProductAttentionOp op, PatternRewriter &rewriter) const {
  auto qHit = findUpstreamScaleMultiply(op.getQuery());

  if (!qHit) {
    return failure();
  }

  float currentScale =
      op.getScale().has_value() ? *op.getScale() : 1.0f;
  float newScale = currentScale * qHit->scalar;

  // Bypass the multiply (RAUW its result with its non-scalar input). Safe
  // because multiply with a 1x1x1x1 broadcast preserves the larger operand's
  // shape, so output type == non-scalar-input type.
  rewriter.replaceOp(qHit->multiplyOp, qHit->bypassValue);

  rewriter.modifyOpInPlace(op, [&] {
    op.setScaleAttr(rewriter.getF32FloatAttr(newScale));
  });

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
```

- [ ] **Step 2.5: Add the pattern source to `lib/Dialect/TTNN/Transforms/CMakeLists.txt`**

Add one line in alphabetical order under `Workarounds/Decomposition/`:

```cmake
        Workarounds/Decomposition/ScaledDotProductAttentionDecodeAttentionSinkRewritePattern.cpp
        Workarounds/Decomposition/ScaledDotProductAttentionDecodeBroadcastMaskRewritePattern.cpp
        Workarounds/Decomposition/ScaledDotProductAttentionFoldScaleRewritePattern.cpp
        Workarounds/Decomposition/ScaledDotProductAttentionPadTileDimsRewritePattern.cpp
```

Also ensure `LINK_LIBS PRIVATE` includes `TTMLIRTTNNUtils` (it already includes other libs; SDPAUtils lives there). Check the existing `LINK_LIBS` block of the `add_mlir_dialect_library(MLIRTTNNTransforms ...)` call. If `TTMLIRTTNNUtils` isn't already in `LINK_LIBS PRIVATE`, add it:

```cmake
        LINK_LIBS
        PUBLIC
        ...
        PRIVATE
        TTMLIRTTIRToTTNN
        TTMLIRTTNNUtils
        MLIRTTNNValidation
        ...
```

(Inspect first via `grep -n TTMLIRTTNN.*Utils third_party/tt-mlir/src/tt-mlir/lib/Dialect/TTNN/Transforms/CMakeLists.txt` — if it's already linked transitively through another lib, no edit needed. Otherwise add it.)

- [ ] **Step 2.6: Register the pattern in `TTNNWorkaroundsPatterns.cpp`**

Add include near the other SDPA-related includes (alphabetical):

```cpp
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionFoldScaleRewritePattern.h"
```

In the `patterns.add<...>` block (around line 607 of `TTNNWorkaroundsPatterns.cpp`), add the new pattern alongside its peers:

```cpp
              workarounds::decomposition::ScaledDotProductAttentionFoldScaleRewritePattern,
              ScaledDotProductAttentionPadTileDimsRewritePattern,
```

Match the namespace prefix style used by the surrounding lines (the existing SDPA workarounds in that file may already be `using` their namespace — match exactly).

- [ ] **Step 2.7: Build**

```bash
source env/activate
cmake --build build
```

Expected: build succeeds.

- [ ] **Step 2.8: Run the lit test — expect PASS**

```bash
source env/activate
llvm-lit -v test/ttmlir/Dialect/TTNN/Transforms/Workarounds/sdpa_fold_scale.mlir
```

Expected: PASS. The `q_side_multiply_only` test should now show `// CHECK-NOT: ttnn.multiply` succeeds (multiply removed) and `scale = 8.838834e-02 : f32` is set.

- [ ] **Step 2.9: Commit**

```bash
git -C third_party/tt-mlir/src/tt-mlir add \
  include/ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionFoldScaleRewritePattern.h \
  lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionFoldScaleRewritePattern.cpp \
  lib/Dialect/TTNN/Transforms/CMakeLists.txt \
  lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp \
  test/ttmlir/Dialect/TTNN/Transforms/Workarounds/sdpa_fold_scale.mlir
git -C third_party/tt-mlir/src/tt-mlir commit -m "Add SDPA fold-scale workaround (Q-side, single test)

Detects ttnn.multiply(Q, scalar) feeding ttnn.scaled_dot_product_attention
through optional one-use typecast/permute, bypasses the multiply and
combines the scalar into the SDPA scale attribute. Initial commit
covers Q-side only; K-side and combinations follow."
```

---

## Task 3: K-side support (TDD red→green)

Add the literal DiT pattern: `multiply → permute → typecast → SDPA(_, this, _)` at the K position.

**Files:**
- Modify: `test/ttmlir/Dialect/TTNN/Transforms/Workarounds/sdpa_fold_scale.mlir`
- Modify: `lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionFoldScaleRewritePattern.cpp`

- [ ] **Step 3.1: Add the failing K-side test case**

Append to `sdpa_fold_scale.mlir`, inside the same `module attributes {}` block (or as a sibling module — both work):

```mlir
  // CHECK-LABEL: k_side_dit_pattern
  func.func @k_side_dit_pattern(
      %q: tensor<1x6x256x128xbf16, #l_qkv>,
      %k_pre_perm: tensor<1x256x6x128xf32, #l_qkv_f32>,
      %v: tensor<1x6x256x128xbf16, #l_qkv>,
      %device: !ttcore.device) -> tensor<1x6x256x128xbf16, #l_qkv> {
    %scalar = "ttnn.full"(%device) <{fill_value = 8.838834e-02 : f32, shape = #ttnn.shape<1x1x1x1>}> : (!ttcore.device) -> tensor<1x1x1x1xf32, #l_scalar>
    %k_scaled = "ttnn.multiply"(%k_pre_perm, %scalar) : (tensor<1x256x6x128xf32, #l_qkv_f32>, tensor<1x1x1x1xf32, #l_scalar>) -> tensor<1x256x6x128xf32, #l_qkv_f32>
    %k_perm = "ttnn.permute"(%k_scaled) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x256x6x128xf32, #l_qkv_f32>) -> tensor<1x6x256x128xf32, #l_qkv_f32_permuted>
    %k_bf16 = "ttnn.typecast"(%k_perm) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x6x256x128xf32, #l_qkv_f32_permuted>) -> tensor<1x6x256x128xbf16, #l_qkv>

    // CHECK-NOT: ttnn.multiply
    // CHECK: %[[OUT:.*]] = "ttnn.scaled_dot_product_attention"
    // CHECK-SAME: scale = 8.838834{{[0-9]*}}e-02 : f32
    %out = "ttnn.scaled_dot_product_attention"(%q, %k_bf16, %v)
        <{is_causal = false, scale = 1.000000e+00 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}>
        : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>) -> tensor<1x6x256x128xbf16, #l_qkv>
    return %out : tensor<1x6x256x128xbf16, #l_qkv>
  }
```

Add the layouts at the top of the file (next to `#l_qkv` and `#l_scalar`):

```mlir
#l_qkv_f32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 * 6 + d1 * 6 + d2, d3), <1x1>, memref<256x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#l_qkv_f32_permuted = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 6 * 8 + d1 * 8 + d2, d3), <1x1>, memref<48x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
```

(If layout encoding errors arise from the test, check existing `sdpa_pad_sequence_dim_workaround.mlir` for layout encodings that work for SDPA inputs and adapt; the operational test is whether the rewrite fires, not whether the layouts are physically realizable.)

- [ ] **Step 3.2: Run — expect FAIL**

```bash
llvm-lit -v test/ttmlir/Dialect/TTNN/Transforms/Workarounds/sdpa_fold_scale.mlir
```

Expected: `q_side_multiply_only` PASSes, `k_side_dit_pattern` FAILs (multiply still present, scale still 1.0).

- [ ] **Step 3.3: Extend pattern to analyze K**

In `ScaledDotProductAttentionFoldScaleRewritePattern.cpp`, replace the `matchAndRewrite` body with the version that handles both Q and K independently:

```cpp
LogicalResult
ScaledDotProductAttentionFoldScaleRewritePattern::matchAndRewrite(
    ScaledDotProductAttentionOp op, PatternRewriter &rewriter) const {
  auto qHit = findUpstreamScaleMultiply(op.getQuery());
  auto kHit = findUpstreamScaleMultiply(op.getKey());

  if (!qHit && !kHit) {
    return failure();
  }

  float currentScale =
      op.getScale().has_value() ? *op.getScale() : 1.0f;
  float qScalar = qHit ? qHit->scalar : 1.0f;
  float kScalar = kHit ? kHit->scalar : 1.0f;
  float newScale = currentScale * qScalar * kScalar;

  if (qHit) {
    rewriter.replaceOp(qHit->multiplyOp, qHit->bypassValue);
  }
  if (kHit) {
    rewriter.replaceOp(kHit->multiplyOp, kHit->bypassValue);
  }

  rewriter.modifyOpInPlace(op, [&] {
    op.setScaleAttr(rewriter.getF32FloatAttr(newScale));
  });

  return success();
}
```

- [ ] **Step 3.4: Build, run — expect PASS for both tests**

```bash
cmake --build build
llvm-lit -v test/ttmlir/Dialect/TTNN/Transforms/Workarounds/sdpa_fold_scale.mlir
```

Expected: both `q_side_multiply_only` and `k_side_dit_pattern` PASS.

- [ ] **Step 3.5: Commit**

```bash
git -C third_party/tt-mlir/src/tt-mlir add \
  test/ttmlir/Dialect/TTNN/Transforms/Workarounds/sdpa_fold_scale.mlir \
  lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionFoldScaleRewritePattern.cpp
git -C third_party/tt-mlir/src/tt-mlir commit -m "SDPA fold-scale workaround: handle K-side multiply

Adds the literal Wan2.2 DiT TTNN pattern (multiply -> permute ->
typecast -> SDPA at the K input position). Q and K sides are
analyzed independently and their scalars combined into SDPA's scale."
```

---

## Task 4: Both-sides combine + non-1.0 existing scale

The implementation already supports both. This task adds two more lit cases proving it.

**Files:**
- Modify: `test/ttmlir/Dialect/TTNN/Transforms/Workarounds/sdpa_fold_scale.mlir`

- [ ] **Step 4.1: Add the both-sides test case**

Append to `sdpa_fold_scale.mlir`:

```mlir
  // CHECK-LABEL: both_sides_combine
  // CHECK-COUNT-0: ttnn.multiply
  // CHECK: %[[OUT:.*]] = "ttnn.scaled_dot_product_attention"
  // CHECK-SAME: scale = 2.500000e-01 : f32
  func.func @both_sides_combine(
      %q: tensor<1x6x256x128xbf16, #l_qkv>,
      %k: tensor<1x6x256x128xbf16, #l_qkv>,
      %v: tensor<1x6x256x128xbf16, #l_qkv>,
      %device: !ttcore.device) -> tensor<1x6x256x128xbf16, #l_qkv> {
    %s_q = "ttnn.full"(%device) <{fill_value = 5.000000e-01 : f32, shape = #ttnn.shape<1x1x1x1>}> : (!ttcore.device) -> tensor<1x1x1x1xf32, #l_scalar>
    %s_k = "ttnn.full"(%device) <{fill_value = 5.000000e-01 : f32, shape = #ttnn.shape<1x1x1x1>}> : (!ttcore.device) -> tensor<1x1x1x1xf32, #l_scalar>
    %q_scaled = "ttnn.multiply"(%q, %s_q) : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x1x1x1xf32, #l_scalar>) -> tensor<1x6x256x128xbf16, #l_qkv>
    %k_scaled = "ttnn.multiply"(%k, %s_k) : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x1x1x1xf32, #l_scalar>) -> tensor<1x6x256x128xbf16, #l_qkv>
    %out = "ttnn.scaled_dot_product_attention"(%q_scaled, %k_scaled, %v)
        <{is_causal = false, scale = 1.000000e+00 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}>
        : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>) -> tensor<1x6x256x128xbf16, #l_qkv>
    return %out : tensor<1x6x256x128xbf16, #l_qkv>
  }
```

(0.5 × 0.5 × 1.0 = 0.25.)

- [ ] **Step 4.2: Add the existing-non-1.0-scale test case**

```mlir
  // CHECK-LABEL: existing_scale_combines
  // CHECK-NOT: ttnn.multiply
  // CHECK: %[[OUT:.*]] = "ttnn.scaled_dot_product_attention"
  // CHECK-SAME: scale = 2.500000e-01 : f32
  func.func @existing_scale_combines(
      %q: tensor<1x6x256x128xbf16, #l_qkv>,
      %k: tensor<1x6x256x128xbf16, #l_qkv>,
      %v: tensor<1x6x256x128xbf16, #l_qkv>,
      %device: !ttcore.device) -> tensor<1x6x256x128xbf16, #l_qkv> {
    %scalar = "ttnn.full"(%device) <{fill_value = 5.000000e-01 : f32, shape = #ttnn.shape<1x1x1x1>}> : (!ttcore.device) -> tensor<1x1x1x1xf32, #l_scalar>
    %q_scaled = "ttnn.multiply"(%q, %scalar) : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x1x1x1xf32, #l_scalar>) -> tensor<1x6x256x128xbf16, #l_qkv>
    %out = "ttnn.scaled_dot_product_attention"(%q_scaled, %k, %v)
        <{is_causal = false, scale = 5.000000e-01 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}>
        : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>) -> tensor<1x6x256x128xbf16, #l_qkv>
    return %out : tensor<1x6x256x128xbf16, #l_qkv>
  }
```

(Existing 0.5 × 0.5 = 0.25.)

- [ ] **Step 4.3: Run — expect PASS for both new cases (no implementation change needed)**

```bash
llvm-lit -v test/ttmlir/Dialect/TTNN/Transforms/Workarounds/sdpa_fold_scale.mlir
```

Expected: all four tests PASS. If `existing_scale_combines` fails because of float-printing differences (e.g. `2.500000e-01` vs `0.25`), tighten the FileCheck pattern using `{{2.5[0]*e-01|2.500000e-01}}` or just `0.25`.

- [ ] **Step 4.4: Commit**

```bash
git -C third_party/tt-mlir/src/tt-mlir add \
  test/ttmlir/Dialect/TTNN/Transforms/Workarounds/sdpa_fold_scale.mlir
git -C third_party/tt-mlir/src/tt-mlir commit -m "SDPA fold-scale workaround: tests for both-sides and non-1.0 scale combine"
```

---

## Task 5: Negative test cases (no rewrite)

Verify the pattern correctly *declines* to fire when its preconditions are violated.

**Files:**
- Modify: `test/ttmlir/Dialect/TTNN/Transforms/Workarounds/sdpa_fold_scale.mlir`

- [ ] **Step 5.1: Add negative case — multiply has multiple uses**

```mlir
  // CHECK-LABEL: multiply_multi_use
  // The multiply feeds both SDPA and a dangling return; the workaround must
  // not fire because bypassing would corrupt the second consumer.
  // CHECK: ttnn.multiply
  // CHECK: scale = 1.000000e+00 : f32
  func.func @multiply_multi_use(
      %q: tensor<1x6x256x128xbf16, #l_qkv>,
      %k: tensor<1x6x256x128xbf16, #l_qkv>,
      %v: tensor<1x6x256x128xbf16, #l_qkv>,
      %device: !ttcore.device)
      -> (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>) {
    %scalar = "ttnn.full"(%device) <{fill_value = 5.000000e-01 : f32, shape = #ttnn.shape<1x1x1x1>}> : (!ttcore.device) -> tensor<1x1x1x1xf32, #l_scalar>
    %q_scaled = "ttnn.multiply"(%q, %scalar) : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x1x1x1xf32, #l_scalar>) -> tensor<1x6x256x128xbf16, #l_qkv>
    %out = "ttnn.scaled_dot_product_attention"(%q_scaled, %k, %v)
        <{is_causal = false, scale = 1.000000e+00 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}>
        : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>) -> tensor<1x6x256x128xbf16, #l_qkv>
    return %out, %q_scaled : tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>
  }
```

- [ ] **Step 5.2: Add negative case — non-constant multiplier**

```mlir
  // CHECK-LABEL: multiply_non_constant
  // CHECK: ttnn.multiply
  // CHECK: scale = 1.000000e+00 : f32
  func.func @multiply_non_constant(
      %q: tensor<1x6x256x128xbf16, #l_qkv>,
      %k: tensor<1x6x256x128xbf16, #l_qkv>,
      %v: tensor<1x6x256x128xbf16, #l_qkv>,
      %dyn: tensor<1x1x1x1xf32, #l_scalar>) -> tensor<1x6x256x128xbf16, #l_qkv> {
    %q_scaled = "ttnn.multiply"(%q, %dyn) : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x1x1x1xf32, #l_scalar>) -> tensor<1x6x256x128xbf16, #l_qkv>
    %out = "ttnn.scaled_dot_product_attention"(%q_scaled, %k, %v)
        <{is_causal = false, scale = 1.000000e+00 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}>
        : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>) -> tensor<1x6x256x128xbf16, #l_qkv>
    return %out : tensor<1x6x256x128xbf16, #l_qkv>
  }
```

- [ ] **Step 5.3: Add negative case — no upstream multiply**

```mlir
  // CHECK-LABEL: no_upstream_multiply
  // CHECK-NOT: ttnn.multiply
  // CHECK: scale = 1.000000e+00 : f32
  func.func @no_upstream_multiply(
      %q: tensor<1x6x256x128xbf16, #l_qkv>,
      %k: tensor<1x6x256x128xbf16, #l_qkv>,
      %v: tensor<1x6x256x128xbf16, #l_qkv>) -> tensor<1x6x256x128xbf16, #l_qkv> {
    %out = "ttnn.scaled_dot_product_attention"(%q, %k, %v)
        <{is_causal = false, scale = 1.000000e+00 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}>
        : (tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>, tensor<1x6x256x128xbf16, #l_qkv>) -> tensor<1x6x256x128xbf16, #l_qkv>
    return %out : tensor<1x6x256x128xbf16, #l_qkv>
  }
```

- [ ] **Step 5.4: Run — expect all PASS**

```bash
llvm-lit -v test/ttmlir/Dialect/TTNN/Transforms/Workarounds/sdpa_fold_scale.mlir
```

Expected: all 7 cases PASS.

- [ ] **Step 5.5: Commit**

```bash
git -C third_party/tt-mlir/src/tt-mlir add \
  test/ttmlir/Dialect/TTNN/Transforms/Workarounds/sdpa_fold_scale.mlir
git -C third_party/tt-mlir/src/tt-mlir commit -m "SDPA fold-scale workaround: negative tests

Verifies the pattern declines on multi-use multiply, non-constant
multiplier, and no upstream multiply."
```

---

## Task 6: End-to-end verification on Wan 2.2 DiT

Confirm the original 8 problematic sites in the WanDiT trace are now folded.

- [ ] **Step 6.1: Rebuild tt-mlir and the tt-xla plugin against the new tt-mlir**

From `/root/tt-xla`:

```bash
source venv/activate
cmake --build build
```

Expected: tt-xla rebuilds against the updated tt-mlir submodule, plugin .so is updated.

- [ ] **Step 6.2: Re-run the WanDiT component test and capture TTNN IR**

From `/root/tt-xla`:

```bash
TTMLIR_LOGGER_LEVEL=DEBUG pytest -svv tests/torch/models/wan2_2/test_wan_dit.py::test_wan_dit_480p_sharded > dit_after_fold.log 2>&1
```

(Set `MAX_BLOCKS = 2` in test_wan_dit.py if not already, to keep compile time short.)

- [ ] **Step 6.3: Verify the fold fired**

```bash
grep -E 'ttnn.scaled_dot_product_attention' dit_after_fold.log | grep -c 'scale = 1.000000e+00 : f32'
```

Expected: `0` self-attention sites with `scale = 1.0` remain. Cross-attn sites (already fused with `scale = 8.838834e-02`) should be unchanged — verify by:

```bash
grep -c 'scale = 8.838834' dit_after_fold.log
```

Expected: number of SDPA call sites ≥ 4 (cross-attn × 2 + self-attn × 2 in a 2-block model).

Also confirm the residual multiplies named in `fusion_todo.yml` are gone:

```bash
grep -E 'ttnn.multiply.*1x1x1x1' dit_after_fold.log | grep -c 'tensor<1x.*x6x128'
```

Expected: 0 (the pre-SDPA scalar multiplies are folded). Other multiplies in the model are unaffected.

- [ ] **Step 6.4: Confirm PCC unchanged (correctness)**

The pytest output should print a `PCC:` line. Expected: still ≥ 0.99 (the rewrite is mathematically exact; PCC drift from this change is at the noise floor).

- [ ] **Step 6.5: No commit needed; report results**

If all assertions hold, the workaround is verified. If any assertion fails, capture the failing site's IR and return to Task 3 / Task 5 to extend the pattern (e.g. add reshape look-through if a model needs it).

---

## Self-review

**Spec coverage:**
- "Both Q-side and K-side, combine if both" → Tasks 2 + 3 + 4
- "Always combine with existing scale" → Task 4
- "Look through typecast and permute, single-use guards" → Task 2 (impl) + Task 5 (multi-use guard verified)
- "Reuse existing constant-extraction helpers via shared utility" → Task 1
- "Workarounds dir, registered in TTNNWorkaroundsPatterns.cpp" → Task 2
- "Lit tests for all positive and negative cases" → Tasks 2-5
- "End-to-end verification on WanDiT" → Task 6

All sections covered.

**Type/method consistency:** `findUpstreamScaleMultiply`, `FoldHit { multiplyOp, bypassValue, scalar }`, `extractMultiplyWithScalarConstant`, `extractScalarConstant` — names and signatures used identically across header, implementation, and call sites.

**Placeholders:** none.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-06-sdpa-fold-scale-workaround.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session via executing-plans, batch with checkpoints.

Which approach?
