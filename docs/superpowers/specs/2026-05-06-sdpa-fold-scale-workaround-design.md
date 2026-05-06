# Design: TTNN workaround — fold residual scalar multiply into SDPA `scale`

**Date:** 2026-05-06
**Status:** Approved (pending implementation plan)
**Author:** ppadjin@tenstorrent.com (assisted by Claude)

## Problem

The Wan 2.2 DiT lowers its self-attention scaling pattern to TTNN as:

```mlir
%211 = "ttnn.multiply"(%210, %20) : (tensor<1x8190x6x128xf32>, tensor<1x1x1x1xf32>) -> tensor<1x8190x6x128xf32>
%215 = "ttnn.permute"(%211) <{permutation = array<i64: 0, 2, 1, 3>}> : (...) -> tensor<1x6x8190x128xf32>
%217 = "ttnn.typecast"(%215) <{dtype = bf16}> : (...) -> tensor<1x6x8190x128xbf16>
%219 = "ttnn.scaled_dot_product_attention"(%216, %217, %218)
       <{is_causal = false, scale = 1.000000e+00 : f32}>
       : (tensor<1x6x8190x128xbf16>, tensor<1x6x8190x128xbf16>, tensor<1x6x8190x128xbf16>) -> tensor<1x6x8190x128xbf16>
```

The pre-SDPA `ttnn.multiply` applies `1/sqrt(d_k)` to one of the SDPA inputs while SDPA itself runs with `scale = 1.0`. This is a leftover from imperfect TTIR-time fusion: the existing `SDPAFusing` pattern (in `lib/Dialect/TTNN/Transforms/Fusing/SDPAFusingPattern.cpp`) builds SDPA from a `Q·K^T` matmul and folds pre-scales then, but by the time the IR reaches TTNN with SDPA already constructed, an orphan multiply remains. There are eight such sites in the WanDiT trace (`fusion_todo.yml`).

The two cross-attention sites are unaffected — they already have `scale = 0.0883883387` baked in and no upstream multiply.

## Goal

Add a TTNN-level workaround pattern that, given an existing `ttnn.scaled_dot_product_attention` op, detects a residual scalar multiply on Q and/or K (through optional `typecast`/`permute`) and folds the scalar into the SDPA's `scale` attribute, removing the multiply.

## Mathematical justification

For any scalars `c_q`, `c_k`:

```
SDPA(c_q · Q, c_k · K, V, scale=s)
  = softmax(s · (c_q · Q) · (c_k · K)^T) · V
  = softmax((s · c_q · c_k) · Q · K^T) · V
  = SDPA(Q, K, V, scale = s · c_q · c_k)
```

So combining is exact (modulo a single floating-point multiply, computed in f32 inside SDPA's `scale` parameter — at least as precise as the bf16/f32 multiply we are removing).

## Scope

**In scope:**
- Prefill SDPA: `ttnn.scaled_dot_product_attention` (op def in `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td:3826`).
- Both Q (operand 0) and K (operand 1) sides; combine if both present.
- Look-through ops between the multiply and SDPA: `ttnn.typecast`, `ttnn.permute` (any permutation). Each looked-through op must have `hasOneUse()`.
- Combining with an already-non-1.0 SDPA `scale` attribute (multiply scalars together).

**Out of scope:**
- Decode SDPA (`ScaledDotProductAttentionDecodeOp`) — separate op, not present in the DiT case.
- V-side multiplies — V is post-softmax-product and a scalar there does not commute into `scale`.
- Look-through `reshape`, `repeat_interleave`, etc. — can be added incrementally if a model needs them.
- Non-broadcast multiplications (anything other than a uniform scalar).

## Design

### Location

New files:
- `include/ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionFoldScaleRewritePattern.h`
- `lib/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionFoldScaleRewritePattern.cpp`

Naming follows the existing siblings in the same directory:
- `ScaledDotProductAttentionPadTileDimsRewritePattern`
- `ScaledDotProductAttentionDecodeAttentionSinkRewritePattern`
- `ScaledDotProductAttentionDecodeBroadcastMaskRewritePattern`

The pattern is registered in `lib/Dialect/TTNN/Transforms/Workarounds/TTNNWorkaroundsPatterns.cpp` alongside the others (currently around line 607). Build registration in `lib/Dialect/TTNN/Transforms/CMakeLists.txt`.

### Pattern class

```cpp
class ScaledDotProductAttentionFoldScaleRewritePattern
    : public OpRewritePattern<ScaledDotProductAttentionOp> {
public:
  using OpRewritePattern<ScaledDotProductAttentionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScaledDotProductAttentionOp op,
                                PatternRewriter &rewriter) const override;
};
```

### Algorithm

For an `ScaledDotProductAttentionOp op`:

1. **Per-side analysis** (helper `findUpstreamScaleMultiply(Value v) -> std::optional<std::pair<MultiplyOp, float>>`):
   Walk backward from `v` along the def-use chain. At each step:
   - If the producer is `ttnn.typecast` or `ttnn.permute` and `hasOneUse()`, consume it (advance to its input) and continue.
   - If the producer is `ttnn.multiply` and `hasOneUse()`, call `extractMultiplyWithScalarConstant` (the shared utility). On a hit, return `(multiplyOp, scalar)`. On a miss, stop.
   - Otherwise stop.

   The walk only crosses look-through ops *before* finding the multiply; we don't continue past the multiply. (The `extractScalarConstant` helper itself already looks through one outer typecast on the constant side, so e.g. `multiply(x, typecast(full(c)))` is handled.)

2. **Combine and rewrite:**
   - Run the analysis on Q (operand 0) and K (operand 1) independently → `qHit`, `kHit`.
   - If both are `std::nullopt`, return `failure()`.
   - Compute `new_scale = op.getScale().value_or(1.0) * (qHit ? qHit.scalar : 1.0) * (kHit ? kHit.scalar : 1.0)`.
   - For each hit `(multiplyOp, _)`: call `rewriter.replaceOp(multiplyOp, multiplyOp.getNonScalarInput())`. Because `ttnn.multiply` of a tensor with a `1x1x1x1` broadcast preserves the larger operand's shape, the multiply's output type equals its non-scalar input's type, so RAUW is type-safe; the downstream typecast/permute chain retargets cleanly.
   - Update SDPA's `scale` attribute via `rewriter.modifyOpInPlace(op, [&] { op.setScaleAttr(rewriter.getF32FloatAttr(new_scale)); })`.
   - Return `success()`.

   Dead constants (the `FullOp` / `LoadCachedOp` feeding the multiply) become unused and are cleaned up by the greedy driver / standard DCE.

### Constant-extraction utility refactor (small)

`SDPAFusing::extractConstant` (lines 126-159 of `Fusing/SDPAFusingPattern.cpp`) and `SDPAFusing::extractMultiplyWithConstant` (lines 161-172) already do exactly what we need. They handle:
- The constant on either lhs or rhs of the multiply.
- An outer typecast wrapping the constant.
- The constant being a `ttnn.full` op.
- The constant being inside a `ttcore.load_cached` const-eval'd function.

These are currently private member functions of `SDPAFusing`. Hoist them to free functions in a new shared header:

- `include/ttmlir/Dialect/TTNN/Utils/SDPAUtils.h`:
  ```cpp
  namespace mlir::tt::ttnn::utils {
    std::optional<float> extractScalarConstant(Value v);
    std::pair<Value, std::optional<float>> extractMultiplyWithScalarConstant(Value v);
  }
  ```
- `lib/Dialect/TTNN/Utils/SDPAUtils.cpp`: implementation moved verbatim from the existing methods.

Update `Fusing/SDPAFusingPattern.cpp` to delegate its existing `extractConstant` / `extractMultiplyWithConstant` to the shared helpers (preserves all existing behavior, just one indirection).

This refactor is intentionally surgical — only the two helpers move, no other behavior changes. It keeps both SDPA scale-folding paths (TTIR-time fusing and TTNN-time workaround) reading from the same constant-detection logic and prevents drift.

### Pass invocation

The new pattern lives inside the existing `ttnn-workaround` pass (`TTNNWorkaroundsPass`, declared in `include/ttmlir/Dialect/TTNN/Transforms/Passes.td`). No new pass is created. Adding a pattern there is the standard way TTNN workarounds get hooked into the pipeline.

### Testing

New lit test file: `test/ttmlir/Dialect/TTNN/Transforms/Workarounds/sdpa_fold_scale.mlir`. Run via `llvm-lit`. Test cases:

| # | Case | Expected |
|---|---|---|
| 1 | Q-side multiply with `FullOp(c_q)` scalar, SDPA scale=1.0 | scale=c_q, multiply gone |
| 2 | K-side multiply (literal DiT pattern: multiply→permute→typecast→SDPA), scale=1.0 | scale=c_k, multiply gone |
| 3 | Both Q and K multiplies, scale=1.0 | scale=c_q*c_k, both multiplies gone |
| 4 | K-side multiply, SDPA already has scale=0.5 | scale=0.5*c_k |
| 5 | Multiply has 2 users | no rewrite |
| 6 | Permute between multiply and SDPA has 2 users | no rewrite |
| 7 | Multiply rhs is a non-constant tensor | no rewrite |
| 8 | SDPA with no upstream multiplies | no rewrite |
| 9 | LoadCachedOp-wrapped scalar (const-eval) | scale folded |

Each positive test uses `// CHECK-NOT: ttnn.multiply` and `// CHECK: scale = ` with the expected value. Negative tests use `// CHECK: ttnn.multiply` to assert the multiply survived.

### Build / verification

- Build: `cmake --build build --target check-ttmlir` (runs lit suite, must pass).
- Manual end-to-end: re-run wan_dit on the DiT and capture TTNN IR. The two self-attn sites referenced in `fusion_todo.yml` (`#loc1493/1495` and `#loc1594/1596`) should no longer have `ttnn.multiply` feeding into SDPA, and SDPA's `scale` should match `1/sqrt(128) ≈ 0.0883883`. Cross-attn sites should be unchanged.

## Risks

- **Numerical equivalence:** the rewrite is exact in real arithmetic. Floating-point reassociation introduces at most one fewer multiply per SDPA. The SDPA scale path computes in f32, which is at least as precise as the original f32/bf16 multiply being removed.
- **Multi-use bypass corruption:** mitigated by `hasOneUse()` checks on both the multiply and every looked-through op.
- **Unintended cross-attn rewrite:** cross-attn sites have no upstream multiply chain matching the pattern; the pattern returns `failure()` and leaves them alone.
- **Refactor scope creep:** the `SDPAUtils` extraction is intentionally narrow — only two helper functions move, with the existing `SDPAFusing` keeping the same external API by delegating internally. Adds one new header/cpp pair, no behavioral change to existing TTIR-time fusing.

## Open questions

None remaining as of approval. All design choices are committed:
- Both sides ✓
- Always combine ✓
- Look-through: typecast + permute only ✓
- Workarounds dir, single new pattern file, registered in `TTNNWorkaroundsPatterns.cpp` ✓
- Constant-extraction helper refactor into shared utility ✓
