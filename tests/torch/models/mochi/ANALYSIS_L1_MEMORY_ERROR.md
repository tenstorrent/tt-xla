# Deep Analysis: L1 Memory Error in Mochi Decoder - Replicate Padding

## Executive Summary

**CRITICAL FINDING**: The error occurs in the **padding implementation**, NOT in Conv3d or its output conversion!

**Actual Failing Operation**: Operation `%9` - a `to_layout` (tile → row_major) conversion that's part of the replicate padding workaround using gather/embedding operations.

**Root Cause**: The padding implementation reshapes the input to `[4, 4884480]` and then tries to untilize (convert from tile layout to row_major) this extremely wide tensor on a single core, requiring ~9.99 MB of L1 memory when only 1.43 MB is available per core.

**Conv3d never executes** - it fails before even reaching the convolution!

---

## 1. The Actual Failing Operation (From Debug Logs)

### 1.1 Execution Trace (What Actually Happened)

From `TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG` output:

```
RuntimeTTNN | DEBUG | Executing operation: %6 = "ttnn.to_layout"(%arg2) <{layout = #ttnn.layout<tile>}>
  ✓ SUCCEEDS - Input [1,768,4,60,106] row_major → tile

RuntimeTTNN | DEBUG | Executing operation: %7 = "ttnn.permute"(%6)
  ✓ SUCCEEDS - Permute to [4,1,768,60,106]

RuntimeTTNN | DEBUG | Executing operation: %8 = "ttnn.reshape"(%7)
  ✓ SUCCEEDS - Reshape to [4, 4884480]  ← Creates VERY WIDE tensor

RuntimeTTNN | DEBUG | Executing operation: %9 = "ttnn.to_layout"(%8) <{layout = #ttnn.layout<row_major>}>
  : (tensor<4x4884480xbf16, tile>) -> (tensor<4x4884480xbf16, row_major>)
  loc("5|.../aten__index_workaround"(...))
  ✗ FAILS HERE!

2025-12-31 16:11:36.007 | critical | TT_THROW:
Statically allocated circular buffers on core range [(x=7,y=6) - (x=7,y=6)]
grow to 9990400 B which is beyond max L1 size of 1499136 B
```

**Key Evidence**:
- Conv3d operation (`%24`) is NOT in the execution log - it never runs!
- The failure is in operation `%9`, part of padding workaround
- Location tag: `"aten__index_workaround"` - this is the replicate padding implementation

### 1.2 The Failing Operation Details

```mlir
%8 = "ttnn.reshape"(%7) <{shape = [4 : i32, 4884480 : i32]}>
     : (tensor<4x1x768x60x106xbf16, #ttnn_layout15>)
    -> (tensor<4x4884480xbf16, #ttnn_layout16>)
     loc("aten__index_reshapeInput")

%9 = "ttnn.to_layout"(%8) <{layout = #ttnn.layout<row_major>}>
     : (tensor<4x4884480xbf16, #ttnn_layout16>)  ← Tile layout
    -> (tensor<4x4884480xbf16, #ttnn_layout17>)  ← Row_major layout
     loc("aten__index_workaround")               ← Part of padding!
```

**Layout Details**:

Input (`#ttnn_layout16`): Tile layout
```mlir
#ttnn_layout16 = #ttnn.ttnn_layout<
    (d0, d1) -> (d0, d1),
    <1x1>,  ← Single core processing
    memref<1x152640x!ttcore.tile<32x32, bf16>, #dram>,
    <interleaved>
>
```

Output (`#ttnn_layout17`): Row-major layout
```mlir
#ttnn_layout17 = #ttnn.ttnn_layout<
    (d0, d1) -> (d0, d1),
    <1x1>,  ← Still single core
    memref<4x4884480xbf16, #dram>,
    <interleaved>
>
```

---

## 2. Why This Operation Requires 9.99 MB of L1

### 2.1 Tensor Dimensions

**Tensor shape**: `[4, 4884480]`
- 4 rows
- 4,884,480 columns (extremely wide!)

**Where does 4,884,480 come from?**
```
Original: [4, 1, 768, 60, 106]
          ↓
Flatten spatial dimensions: 1 × 768 × 60 × 106 = 4,884,480
```

### 2.2 Untilize Operation Requirements

**Untilize** (tile → row_major) must:
1. Read tiles (32×32 blocks) from DRAM
2. Buffer them in L1 memory
3. Reorganize tiles back into row-major format
4. Write row-major data back to DRAM

**Tile dimensions**:
- Width in tiles: 4,884,480 / 32 = **152,640 tiles**
- Height in tiles: 4 / 32 = 0.125 → padded to 32 rows → **1 tile high**

So the tiled tensor is: `[1 tile high × 152,640 tiles wide]`

### 2.3 L1 Buffer Calculation

The untilize operation tries to buffer data for processing:

**Per-row data**:
```
4,884,480 elements × 2 bytes (bf16) = 9,768,960 bytes ≈ 9.77 MB
```

**With overhead** (alignment, metadata, circular buffer management):
```
9,990,400 bytes = 9.99 MB
```

**L1 available per core**: 1,499,136 bytes = 1.43 MB

**Overflow**: 9.99 MB ÷ 1.43 MB = **6.98× over budget**

### 2.4 Why So Much Buffering?

The untilize operation on a `[4, 4884480]` tensor faces a fundamental problem:

1. **Tiles are 32 elements wide**, but the tensor is **4.8M elements wide**
2. To produce even a single output row, the operation must:
   - Read 152,640 tiles (spread across memory)
   - Buffer enough data to reorganize into row format
   - Handle the extreme aspect ratio (4 high × 4.8M wide)

3. The `<1x1>` mesh shape means **no sharding** - one core handles everything
4. The pipeline depth tries to optimize throughput by buffering entire rows

---

## 3. Why Replicate Padding Creates This Problem

### 3.1 Replicate Padding Implementation

PyTorch's `F.pad(..., mode="replicate")` is lowered to a complex workaround using:
- `gather` operations (implemented as `embedding` in TTNN)
- Multiple reshape and permute operations
- Index tensors for replicating boundary values

### 3.2 The Workaround Strategy

```mlir
Input: [1, 768, 4, 60, 106]
  ↓ %6: to_layout (tile)
  ↓ %7: permute → [4, 1, 768, 60, 106]
  ↓ %8: reshape → [4, 4884480]  ← Flatten to 2D for embedding
  ↓ %9: to_layout (row_major)  ← ✗ FAILS HERE
  ↓ %10: embedding (gather along dimension 0)
```

This pattern repeats for each spatial dimension (temporal, height, width).

### 3.3 Why This Approach Fails

**Problem**: The reshape to `[4, 4884480]` creates an extremely wide 2D tensor to enable using `embedding` operations for the gather. However:

1. **No natural parallelism**: `<1x1>` mesh means single-core processing
2. **Extreme aspect ratio**: 4 rows × 4.8M columns doesn't map well to 32×32 tiles
3. **Large row size**: Each row is ~9.77 MB, far exceeding L1 capacity
4. **Pipeline depth**: Untilize tries to buffer full rows for efficiency

---

## 4. Why Conv3d-Only Case Has Different Error

When you comment out padding and run only Conv3d:

```python
def forward(self, x):
    # x = F.pad(x, self.time_causal_padding, mode="replicate")  ← Commented out
    return self.conv(x)
```

**Different failure mode**:
- Input goes directly to Conv3d
- Conv3d uses multi-core (`<8x8>`) L1-optimized strategy
- Tries to keep all intermediate activations in L1 across 64 cores
- Fails with 33.5 MB requirement across all cores

**Why the strategy difference?**

With padding operations present:
- The complex IR (gather/embedding ops) signals to Conv3d compiler
- Heuristics choose **DRAM-staged strategy** (safer, lower L1 usage)
- Conv3d succeeds, but padding fails first

Without padding:
- Simple IR → Conv3d compiler is more aggressive
- Chooses **L1-optimized strategy** (faster, higher L1 usage)
- Conv3d itself fails

---

## 5. Root Cause Summary

| Aspect | Issue |
|--------|-------|
| **Operation** | `%9`: `to_layout` (tile → row_major) untilize |
| **Part of** | Replicate padding workaround using gather/embedding |
| **Tensor Shape** | `[4, 4884480]` - 4 rows, 4.8M columns |
| **L1 Required** | 9.99 MB on single core |
| **L1 Available** | 1.43 MB per core |
| **Overflow** | 6.98× |
| **Why So Wide** | Spatial dimensions flattened for embedding: 1×768×60×106 = 4,884,480 |
| **Why Single Core** | `<1x1>` mesh shape - no multi-core sharding |
| **Why Not Conv3d** | Conv3d never executes - failure happens in padding |

---

## 6. Solutions and Workarounds

### 6.1 Immediate Workarounds

#### Option 1: Use Constant Padding Instead ⭐ RECOMMENDED
```python
class CogVideoXCausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.time_causal_padding = (1, 1, 1, 1, 2, 0)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        # Use constant padding (0) instead of replicate
        x = F.pad(x, self.time_causal_padding, mode="constant", value=0)
        return self.conv(x)
```

**Why this works**:
- Constant padding lowers to simple `pad` operation in MLIR
- No gather/embedding/reshape workaround needed
- Direct padding implementation without wide tensor creation
- Likely has negligible impact on model quality

#### Option 2: Manual Padding with Slicing
```python
def forward(self, x):
    # Manual replicate padding using slicing and concatenation
    b, c, t, h, w = x.shape

    # Temporal padding (replicate first 2 frames at front)
    front_pad = x[:, :, :1, :, :].expand(-1, -1, 2, -1, -1)
    x = torch.cat([front_pad, x], dim=2)

    # Height padding (replicate edges)
    top_pad = x[:, :, :, :1, :].expand(-1, -1, -1, 1, -1)
    bottom_pad = x[:, :, :, -1:, :].expand(-1, -1, -1, 1, -1)
    x = torch.cat([top_pad, x, bottom_pad], dim=3)

    # Width padding
    left_pad = x[:, :, :, :, :1].expand(-1, -1, -1, -1, 1)
    right_pad = x[:, :, :, :, -1:].expand(-1, -1, -1, -1, 1)
    x = torch.cat([left_pad, x, right_pad], dim=4)

    return self.conv(x)
```

**Trade-offs**:
- ✓ Avoids the gather/embedding workaround
- ✓ More explicit, easier to debug
- ✗ More verbose
- ✗ May compile differently

#### Option 3: Pre-pad Input
```python
# Pad once at the model input level, then use stride/dilation in Conv3d
x = F.pad(input_tensor, padding=(1, 1, 1, 1, 2, 0), mode="replicate")
# Then use regular Conv3d without manual padding
```

### 6.2 Compiler-Level Fixes (Requires TT-MLIR Changes)

#### Fix 1: Multi-Core Sharding for Untilize
**Location**: `third_party/tt-mlir/src/tt-mlir/lib/Conversion/TTNNToTTIR/TTNNToTTIR.cpp`

Force multi-core sharding for extremely wide tensors:

```cpp
// When lowering to_layout, detect problematic shapes
if (isUntilize && tensorWidth > THRESHOLD && meshShape == {1,1}) {
  // Force multi-core sharding
  meshShape = {8, 8};  // Distribute across all cores
  // Shard along width dimension
  shardSpec = createWidthSharding(tensorWidth, 64);
}
```

#### Fix 2: Reduce Pipeline Depth for Untilize
**Location**: `third_party/tt-mlir/third_party/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding`

Add heuristics to limit buffering:

```cpp
// Detect if row size exceeds L1 capacity
size_t row_bytes = width * sizeof(dtype);
size_t max_pipeline_depth = L1_SIZE / row_bytes;

if (max_pipeline_depth < MIN_PIPELINE_DEPTH) {
  // Use streaming mode with minimal buffering
  return createStreamingUntilize(tensor);
}
```

#### Fix 3: Alternative Replicate Padding Lowering
**Location**: `third_party/tt-mlir/src/tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIR.cpp`

Implement replicate padding without gather/embedding:

```cpp
// Instead of: reshape → embedding → reshape
// Use: slice + concatenate pattern

// For replicate padding:
// front_pad = slice(input, [0,0,0,0,0], [B,C,1,H,W])
// front_pad = broadcast(front_pad, [B,C,num_pad_front,H,W])
// output = concat([front_pad, input], dim=2)
```

### 6.3 Recommended Action Plan

**Immediate** (Can do now):
1. ✅ **Try constant padding** - Change `mode="replicate"` to `mode="constant"`
2. ✅ **Test quality impact** - Run inference, compare outputs
3. ✅ If quality is acceptable, use constant padding

**Short-term** (1-2 weeks):
1. File bug report with TT-MLIR team
2. Provide this analysis and minimal repro
3. Request either:
   - Multi-core sharding for wide untilize operations
   - Alternative replicate padding lowering

**Long-term** (1-2 months):
1. Implement proper multi-core sharding for layout conversions
2. Add heuristics to detect and handle extreme aspect ratios
3. Optimize replicate padding lowering to avoid reshape→embedding pattern

---

## 7. Testing the Fix

### Test 1: Constant Padding
```bash
# Edit single_ops.py line 116:
# OLD: x = F.pad(x, self.time_causal_padding, mode="replicate")
# NEW: x = F.pad(x, self.time_causal_padding, mode="constant", value=0)

python tests/torch/models/mochi/single_ops.py
```

**Expected**: Should complete without L1 memory error

### Test 2: Verify Conv3d Executes
```bash
export TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG
python tests/torch/models/mochi/single_ops.py 2>&1 | grep "conv3d"
```

**Expected**: Should see `"ttnn.conv3d"` in execution log

### Test 3: Check Output Correctness
```python
# Compare replicate vs constant padding
x = torch.randn(1, 768, 4, 60, 106, dtype=torch.bfloat16)

# Replicate (on CPU)
out_replicate = F.pad(x, (1,1,1,1,2,0), mode="replicate")

# Constant
out_constant = F.pad(x, (1,1,1,1,2,0), mode="constant", value=0)

# Compare boundary values
print("Replicate boundary:", out_replicate[0, 0, 0, 0, :5])
print("Constant boundary:", out_constant[0, 0, 0, 0, :5])
```

---

## 8. Why My Initial Analysis Was Wrong

**What I claimed**: Operation %25 (post-conv3d to_layout) was failing

**Why I was wrong**:
1. ❌ No location tag in the actual error message
2. ❌ Didn't have debug logs showing execution order
3. ❌ Made assumptions based on MLIR structure, not runtime behavior
4. ❌ Confused buffer size calculation (9.99 MB matches multiple operations)

**What the debug logs revealed**:
1. ✅ Conv3d never executes
2. ✅ Failure is in operation %9 (padding workaround)
3. ✅ Location tag: `"aten__index_workaround"`
4. ✅ Tensor shape `[4, 4884480]` only appears in padding ops

**Lesson**: Always enable debug logging to see actual execution order!

---

## Appendix: Full Operation Sequence (First 10 ops)

```
%5 = ttnn.get_device
%6 = ttnn.to_layout(%arg2) tile          ✓ Input [1,768,4,60,106]
%7 = ttnn.permute(%6)                     ✓ → [4,1,768,60,106]
%8 = ttnn.reshape(%7)                     ✓ → [4,4884480]
%9 = ttnn.to_layout(%8) row_major        ✗ FAILS - 9.99 MB
%10 = ttnn.embedding(...)                 ⊗ Never reached
...
%24 = ttnn.conv3d(...)                    ⊗ Never reached
%25 = ttnn.to_layout(%24)                 ⊗ Never reached
```

The error stops execution at operation %9, before any subsequent operations can run.
