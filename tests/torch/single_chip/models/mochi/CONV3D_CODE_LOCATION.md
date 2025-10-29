# Exact Code Location Where Conv3D Fails

## Summary

Conv3D operations fail during **TTIR → TTNN lowering** because:
1. All TTIR operations are marked as **illegal by default**
2. Only **Conv2D** has a conversion pattern
3. Conv3D has **no matching pattern**, so it remains illegal
4. The legalization pass fails

---

## The Call Stack

```
PyTorch Conv3d
    ↓
XLA Convolution
    ↓
StableHLO convolution (3 spatial dims)
    ↓
TTIR convolution (3 spatial dims) ← Successfully created
    ↓
TTIR → TTNN lowering  ← FAILS HERE
    ↓
    ✗ No pattern matches Conv3d
    ✗ Operation remains illegal
    ✗ Pass fails
```

---

## Key Code Locations

### 1. TTIR Operations Marked Illegal (Pass Setup)

**File:** `third_party/tt-mlir/src/tt-mlir/lib/Conversion/TTIRToTTNN/TTIRToTTNNPass.cpp`

**Lines 38-61:**
```cpp
struct ConvertTTIRToTTNNPass
    : public ttir::impl::ConvertTTIRToTTNNBase<ConvertTTIRToTTNNPass> {
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<ttnn::TTNNDialect>();
    target.addLegalDialect<quant::QuantDialect>();

    // ⚠️ KEY LINE: ALL TTIR operations are ILLEGAL by default
    target.addIllegalDialect<ttir::TTIRDialect>();

    target.addLegalOp<ttcore::DeviceOp>();
    target.addLegalOp<ttcore::OptimizationBarrierOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(&getContext());
    populateTTIRToTTNNPatterns(&getContext(), patterns, typeConverter);

    // Apply full conversion - all ops must be legalized
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();  // ← Conv3d fails here
      return;
    }
  }
};
```

**What happens:**
- Line 44: `target.addIllegalDialect<ttir::TTIRDialect>()` marks ALL TTIR ops as illegal
- Line 53: Loads conversion patterns from `populateTTIRToTTNNPatterns()`
- Line 58: Tries to convert all operations
- If any op remains illegal (no pattern matched), the pass fails

---

### 2. Conv2D Pattern (Only 2D Supported!)

**File:** `third_party/tt-mlir/src/tt-mlir/lib/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.cpp`

**Lines 342-377: Conv2D Pattern Definition**
```cpp
struct ConvolutionToConv2dPattern : public ConvolutionDecompositionPattern {
public:
  using ConvolutionDecompositionPattern::ConvolutionDecompositionPattern;

  // ⚠️ KEY: ONLY 2 SPATIAL DIMENSIONS SUPPORTED
  constexpr static uint32_t NUM_SPATIAL_DIMS = 2;
  constexpr static uint32_t SPATIAL_DIM_HEIGHT = 0;
  constexpr static uint32_t SPATIAL_DIM_WIDTH = 1;

  // NHWC layout (no Time dimension!)
  static inline const std::vector<int64_t> conv2dLayout = {
      ConvolutionDimension::BATCH,
      SPATIAL_DIM_HEIGHT,
      SPATIAL_DIM_WIDTH,
      ConvolutionDimension::FEATURE,
  };

  // OIHW kernel layout
  static inline const std::vector<int64_t> conv2dKernelLayout = {
      ConvolutionKernelDimension::OUTPUT_FEATURES,
      ConvolutionKernelDimension::INPUT_FEATURES,
      SPATIAL_DIM_HEIGHT,
      SPATIAL_DIM_WIDTH,
  };

  LogicalResult
  matchAndRewrite(ttir::ConvolutionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // ⚠️ CHECK: Must have exactly 2 spatial dimensions
    if (!(isSupportedConv(op) && isNDimensional(op, NUM_SPATIAL_DIMS))) {
      return failure();  // ← Conv3d fails this check!
    }

    // ... rest of conversion code (never reached for Conv3d)
  }
};
```

**What happens:**
- Line 346: `NUM_SPATIAL_DIMS = 2` (hardcoded)
- Line 375: Checks if convolution has exactly 2 spatial dims
- Conv3d has **3 spatial dims** (T, H, W), so check fails
- Pattern returns `failure()`, doesn't match

---

### 3. Dimension Check Function

**File:** Same file as above

**Lines 123-126:**
```cpp
static bool isNDimensional(ttir::ConvolutionOp op, uint32_t numSpatialDims) {
  // ⚠️ Checks number of spatial dimensions
  return op.getConvolutionLayout().getInputSpatialDimensions().size() ==
         numSpatialDims;
}
```

**For our Conv3d:**
```cpp
// Our Conv3d from test:
input_spatial_dimensions = [2, 3, 4]  // indices in [B, C, T, H, W]
                         //           T=2, H=3, W=4

// Check:
inputSpatialDimensions.size() == 3  // Three spatial dims!
numSpatialDims == 2                 // Pattern expects 2

// Result:
3 == 2  → false  → failure()
```

---

### 4. Pattern Registration

**File:** `third_party/tt-mlir/src/tt-mlir/lib/Conversion/TTIRToTTIRDecomposition/TTIRToTTIRDecomposition.cpp`

**Lines 2764-2785:**
```cpp
void populateTTIRToTTIRDecompositionPatterns(MLIRContext *ctx,
                                             RewritePatternSet &patterns,
                                             TypeConverter &typeConverter) {
  patterns.add<PoolingToPool2dPattern>(typeConverter, ctx);
  patterns.add<PoolingToFullOp>(typeConverter, ctx);
  patterns.add<IndexToSliceConversionPattern>(typeConverter, ctx);
  patterns.add<Legalize1DConvolutionPattern>(typeConverter, ctx);

  // ⚠️ ONLY Conv2D pattern is registered!
  patterns.add<ConvolutionToConv2dPattern>(typeConverter, ctx);

  patterns.add<GatherToEmbeddingConversionPattern>(typeConverter, ctx);
  // ... more patterns
}
```

**What's missing:**
- No `ConvolutionToConv3dPattern`
- No decomposition from 3D → 2D
- No fallback for unsupported dimensions

---

## Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  TTIR Module (after StableHLO → TTIR conversion)           │
│                                                             │
│  func @main(...) {                                          │
│    %0 = ttir.convolution(...) {                            │
│      input_spatial_dimensions = [2, 3, 4]  ← 3 dims!      │
│      ...                                                    │
│    }                                                        │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────────────────────────┐
        │  ConvertTTIRToTTNNPass::run()   │
        └───────────────────────────────────┘
                        ↓
        ┌───────────────────────────────────┐
        │  target.addIllegalDialect<TTIR>() │
        │  ↑                                │
        │  All TTIR ops are ILLEGAL         │
        └───────────────────────────────────┘
                        ↓
        ┌───────────────────────────────────┐
        │  Load conversion patterns         │
        │  - ConvolutionToConv2dPattern     │
        │  - Other patterns...              │
        │  (NO Conv3d pattern!)             │
        └───────────────────────────────────┘
                        ↓
        ┌───────────────────────────────────┐
        │  Try to match ttir.convolution    │
        │  with ConvolutionToConv2dPattern  │
        └───────────────────────────────────┘
                        ↓
        ┌───────────────────────────────────┐
        │  isNDimensional(op, 2)?           │
        │  → op has 3 dims                  │
        │  → 3 == 2? NO!                    │
        │  → return failure()               │
        └───────────────────────────────────┘
                        ↓
        ┌───────────────────────────────────┐
        │  No other pattern matches         │
        │  → ttir.convolution stays ILLEGAL │
        └───────────────────────────────────┘
                        ↓
        ┌───────────────────────────────────┐
        │  applyFullConversion() fails      │
        │  → signalPassFailure()            │
        └───────────────────────────────────┘
                        ↓
                    ✗ ERROR
      "failed to legalize operation 'ttir.convolution'
       that was explicitly marked illegal"
```

---

## Exact Error Flow

### Error Message Breakdown

```
loc("Conv3d[conv3d]/forward(single_ops.py:33)/aten__convolution_overrideable"):
error: failed to legalize operation 'ttir.convolution' that was explicitly marked illegal
```

**Decoded:**
- `Conv3d[conv3d]` - The PyTorch module name
- `single_ops.py:33` - Line in test file
- `aten__convolution_overrideable` - Original PyTorch op
- `ttir.convolution` - TTIR operation that failed
- `explicitly marked illegal` - From `target.addIllegalDialect<ttir::TTIRDialect>()`

### Log Location

```
2025-10-27 16:15:23.390 (   2.408s) [        32A22000]     client_instance.cc:471      1| ClientInstance::getOrCreateOptimizerSubmesh - creating optimizer submesh
loc("Conv3d[conv3d]/forward(single_ops.py:33)/aten__convolution_overrideable"): error: failed to legalize operation 'ttir.convolution' that was explicitly marked illegal
2025-10-27 16:15:23.394 (   2.412s) [        32A22000]      module_builder.cc:856    ERR| Failed to convert from TTIR to TTNN module
```

**Call stack:**
1. `module_builder.cc:856` - Tries to build TTNN module
2. TTIR → TTNN pass runs
3. ttir.convolution is marked illegal (line 44 of pass)
4. No pattern converts it
5. Pass fails, error bubbles up

---

## Why This Design?

### MLIR Convention Pattern

This is standard MLIR lowering design:

1. **Mark everything illegal** - Forces explicit conversion
2. **Add patterns** - Each pattern legalizes specific ops
3. **Verify completeness** - If any op remains, fail loudly

**Benefits:**
- Catches unhandled operations early
- Forces developers to be explicit
- No silent failures

**Downside:**
- Need explicit support for every operation variant
- No automatic fallbacks

---

## The Missing Pieces

### What Would Be Needed for Conv3D Support

#### Option 1: Add Conv3D → TTNN Direct Lowering
```cpp
// New file: Conv3dToTTNN.cpp
struct ConvolutionToConv3dPattern : public ConvolutionDecompositionPattern {
  constexpr static uint32_t NUM_SPATIAL_DIMS = 3;  // T, H, W

  LogicalResult matchAndRewrite(...) {
    if (!(isSupportedConv(op) && isNDimensional(op, NUM_SPATIAL_DIMS))) {
      return failure();
    }
    // Convert to ttnn.conv3d (doesn't exist!)
  }
};
```
**Problem:** TTNN doesn't have conv3d operations!

#### Option 2: Add Conv3D → Conv2D Decomposition
```cpp
struct ConvolutionToConv2dDecompositionPattern : ... {
  LogicalResult matchAndRewrite(ttir::ConvolutionOp op, ...) {
    if (!isNDimensional(op, 3)) return failure();

    // Split 3D conv into multiple 2D convs
    // Process each time step separately
    // Combine results
  }
};
```
**Challenge:** Must handle temporal dimension carefully

#### Option 3: Add Point-wise Conv3D → Linear
```cpp
struct PointwiseConv3dToLinearPattern : ... {
  LogicalResult matchAndRewrite(ttir::ConvolutionOp op, ...) {
    if (!isNDimensional(op, 3)) return failure();

    // Check if kernel is (1,1,1)
    if (!isPointwise(op)) return failure();

    // Convert to ttir.linear (channel mixing)
  }
};
```
**Good for:** Our specific case (kernel=1,1,1)

---

## Files Modified for Solution

To fix Conv3D, you would need to modify:

1. **Add pattern file:**
   - New: `third_party/tt-mlir/src/tt-mlir/lib/Conversion/TTIRToTTIRDecomposition/Conv3dDecomposition.cpp`

2. **Register pattern:**
   - Modify: `TTIRToTTIRDecomposition.cpp` line 2771
   - Add: `patterns.add<ConvolutionToConv3dPattern>(typeConverter, ctx);`

3. **Implement conversion:**
   - Write logic to handle 3 spatial dimensions
   - Either decompose to 2D or convert directly

---

## Summary

**Exact failure location:**
- **File:** `TTIRToTTNNPass.cpp`
- **Line:** 44 (`target.addIllegalDialect<ttir::TTIRDialect>()`)
- **Effect:** Marks all TTIR ops illegal, including ttir.convolution

**Why Conv3D specifically fails:**
- **File:** `TTIRToTTIRDecomposition.cpp`
- **Line:** 346 (`constexpr static uint32_t NUM_SPATIAL_DIMS = 2`)
- **Line:** 375 (`if (!(isSupportedConv(op) && isNDimensional(op, NUM_SPATIAL_DIMS)))`)
- **Effect:** Conv2D pattern only matches 2D, Conv3D (3 dims) doesn't match

**Result:** No pattern legalizes ttir.convolution with 3 spatial dims → Pass fails

---

## Next Steps

Now that we know exactly where and why it fails, we can:

1. ✅ **Understand the problem** - DONE
2. ⏭️ **Implement workaround** - Add Conv3D decomposition pattern
3. ⏭️ **Test solution** - Verify Mochi decoder works

The code is well-structured for adding new patterns - we just need to implement one!
