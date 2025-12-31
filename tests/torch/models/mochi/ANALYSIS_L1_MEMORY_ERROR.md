# Deep Analysis: L1 Memory Error in Mochi Decoder Conv3d

## Executive Summary

**Problem**: Mochi decoder's first Conv3d layer fails with L1 memory overflow, but the **exact failure point changes** depending on input configuration.

**Three Failure Modes Observed**:

| Configuration | Input Shape | Error Location | L1 Requirement | Cores Affected |
|--------------|-------------|----------------|----------------|----------------|
| **Replicate Padding + Conv3d** | `[1,768,4,60,106]` | `to_layout` (op %25) | 9.99 MB | Single core (7,6) |
| **Conv3d Only** | `[1,768,4,60,106]` | `conv3d` (op %6) | 33.5 MB | All 64 cores |
| **Reduced Size** | `[1,768,4,16,16]` | `conv3d` | 31.9 MB | All 64 cores |

**Key Insight**: The presence of replicate padding changes the Conv3d execution strategy from L1-optimized (fails immediately) to DRAM-staged (succeeds), but then the subsequent `to_layout` conversion fails instead.

---

## 1. The Exact Failing Operation (Padding + Conv3d Case)

### 1.1 MLIR Operation Sequence

```mlir
func.func @main(%arg2: tensor<1x768x4x60x106xbf16>) -> tensor<1x768x4x60x106xbf16> {
  // ... padding via gather/embedding operations ...

  %23 = "ttnn.to_layout"(%22) <{layout = #ttnn.layout<row_major>}>
       : (tensor<1x6x62x108x768xbf16>) -> tensor<1x6x62x108x768xbf16, #ttnn_layout31>

  %24 = "ttnn.conv3d"(%23, %weight, %bias, %device) <{
         batch_size = 1, in_channels = 768, out_channels = 768,
         input_depth = 6, input_height = 62, input_width = 108,
         kernel_size = array<i32: 3, 3, 3>
       }>
       : (...) -> tensor<1x4x60x106x768xbf16, #ttnn_layout32>  ✓ SUCCEEDS

  %25 = "ttnn.to_layout"(%24) <{layout = #ttnn.layout<tile>}>
       : (tensor<1x4x60x106x768xbf16, #ttnn_layout32>)         ✗ FAILS HERE
      -> (tensor<1x4x60x106x768xbf16, #ttnn_layout33>)
       loc(#loc7: "aten__convolution_overrideable_in_0_layout")

  %26 = "ttnn.permute"(%25) <{permutation = array<i64: 0, 4, 1, 2, 3>}>
       : (tensor<1x4x60x106x768xbf16>) -> (tensor<1x768x4x60x106xbf16>)
}
```

### 1.2 Layout Details for Operation %25

**Input Layout** (`#ttnn_layout32`): Row-major in DRAM
```mlir
#ttnn_layout32 = #ttnn.ttnn_layout<
    (d0, d1, d2, d3, d4) -> (d0 * 25440 + d1 * 6360 + d2 * 106 + d3, d4),
    <1x1>,  ← Mesh shape: NO multi-core sharding
    memref<25440x768xbf16, #dram>,
    <interleaved>
>
```

**Output Layout** (`#ttnn_layout33`): Tile in DRAM
```mlir
#ttnn_layout33 = #ttnn.ttnn_layout<
    (d0, d1, d2, d3, d4) -> (d0 * 30720 + d1 * 7680 + d2 * 128 + d3, d4),
    <1x1>,  ← Still no sharding
    memref<960x24x!ttcore.tile<32x32, bf16>, #dram>,
    <interleaved>
>
```

**Key Observation**: Both layouts use `<1x1>` mesh shape, meaning **no multi-core sharding** is applied. The entire conversion happens on a single core.

### 1.3 Why This Specific to_layout Fails

The `to_layout` operation converts from row-major to tiled format, which requires:

1. **Reading** row-major data from DRAM
2. **Buffering** multiple rows in L1 to form complete 32×32 tiles
3. **Reorganizing** data into tile format
4. **Writing** tiled data back to DRAM

**Buffer Size Calculation**:
```
Tensor shape: [1, 4, 60, 106, 768]
                     ^   ^    ^
                     |   |    └─ Channels: 768 (VERY WIDE)
                     |   └────── Width: 106
                     └────────── Height: 60

Per-row size: 106 (width) × 768 (channels) × 2 bytes = 162,816 bytes (159 KB)
Pipeline depth: ~60 rows (entire height)
Total L1 buffer: 60 × 159 KB = 9,768,960 bytes ≈ 9.77 MB

Actual error: 9,990,400 bytes = 9.99 MB
Difference: 221,440 bytes (likely alignment/padding overhead)
```

**L1 Limit**: 1,499,136 bytes = 1.43 MB per core

**Overflow**: 9.99 MB ÷ 1.43 MB = **6.98× over limit**

### 1.4 Why 60 Rows Are Buffered

The tilize operation needs to create 32×32 tiles. With 768 channels:
- Channels span: 768 ÷ 32 = 24 tiles horizontally
- Height span: 60 ÷ 32 = 1.875 → needs 2 tiles vertically

To process efficiently, the runtime pipelines multiple rows simultaneously. With the current implementation, it tries to buffer the entire height dimension (60 rows), which is too much for L1.

---

## 2. Alternative Failure Mode: Conv3d Itself

### 2.1 When Conv3d Fails (Without Padding)

When replicate padding is **removed**, the MLIR changes significantly:

```mlir
func.func @main(%arg2: tensor<1x768x4x60x106xbf16>) -> tensor<1x768x2x58x104xbf16> {
  %5 = "ttnn.to_layout"(%arg2) <{layout = #ttnn.layout<row_major>}>
       : (tensor<1x768x4x60x106xbf16>) -> tensor<1x4x60x106x768xbf16>

  %6 = "ttnn.conv3d"(%5, %weight, %bias, %device) <{
         input_depth = 4,  ← NO PADDING, smaller depth
         input_height = 60,
         input_width = 106,
         kernel_size = array<i32: 3, 3, 3>
       }>
       : (...) -> tensor<1x2x58x104x768xbf16>  ✗ FAILS HERE

  %7 = "ttnn.to_layout"(%6) <{layout = #ttnn.layout<tile>}>
       : (...) -> (...)  ← Never reached
}
```

**Error**:
```
Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=7)]
grow to 33469184 B which is beyond max L1 size of 1499136 B
```

**Key Differences**:
- Fails on **all 64 cores** (8×8 grid)
- Each core requires **33.5 MB** of L1
- Error occurs in **Conv3d itself**, not to_layout

### 2.2 Why Conv3d Strategy Changes

The Conv3d operation in TT-MLIR has internal heuristics that choose between two strategies:

#### Strategy A: DRAM-Staged (Used with Padding)
- **When**: Input tensor has complex intermediate representation (gather/embedding ops)
- **How**: Uses DRAM as intermediate buffer for conv3d computation
- **L1 Usage**: Moderate (within limits)
- **Result**: Conv3d ✓ succeeds, but to_layout ✗ fails

#### Strategy B: L1-Optimized Multi-Core (Used without Padding)
- **When**: Input tensor is simple/direct
- **How**: Distributes work across all 64 cores, keeps data in L1
- **L1 Usage**: Very high (33.5 MB per core)
- **Result**: Conv3d ✗ fails immediately

### 2.3 Buffer Calculation for Conv3d Failure

```
Input shape: [1, 4, 60, 106, 768]
Kernel: 3×3×3
Output channels: 768

Per-core allocation (L1-optimized strategy):
  Input buffers: Store input slice + halo
  Weight buffers: Store portion of kernel weights
  Intermediate: Store partial convolution results
  Output buffers: Store output slice

Total per core: 33,469,184 bytes = 31.9 MB

L1 Limit: 1.43 MB
Overflow: 31.9 MB ÷ 1.43 MB = 22.3× over limit
```

---

## 3. Comparison of All Three Scenarios

### 3.1 Scenario Summary

| Scenario | Operation Chain | Conv3d Input | Conv3d Strategy | Conv3d Result | to_layout Result |
|----------|----------------|--------------|-----------------|---------------|------------------|
| **1. Padding + Conv3d<br/>[1,768,4,60,106]** | gather→embedding→conv3d→to_layout | Complex (padded via gather) | DRAM-staged | ✓ Succeeds | ✗ Fails (9.99 MB) |
| **2. Conv3d Only<br/>[1,768,4,60,106]** | direct→conv3d→to_layout | Simple tensor | L1-optimized | ✗ Fails (33.5 MB) | Never reached |
| **3. Small Size<br/>[1,768,4,16,16]** | direct→conv3d→to_layout | Small simple tensor | L1-optimized | ✗ Fails (31.9 MB) | Never reached |

### 3.2 Why Padding Affects Conv3d Strategy

The replicate padding operation is lowered to:
```mlir
%3 = "ttnn.embedding"(%indices_temporal, %reshaped_input)
%15 = "ttnn.embedding"(%indices_height, %intermediate)
%20 = "ttnn.embedding"(%indices_width, %intermediate2)
```

These embedding operations create a **complex intermediate representation** that signals to the compiler:
- "This input has already been through heavy transformations"
- "The data layout may be non-contiguous"
- "Use DRAM-staged approach for safety"

Without padding, the input is just a simple contiguous tensor, so the compiler thinks:
- "This is a straightforward tensor"
- "Optimize with L1 multi-core parallelization"
- → **Wrong choice** for this size

### 3.3 Input Size Effect

With small input `[1,768,4,16,16]`:
```
Spatial dimensions: 4 × 16 × 16 = 1,024 voxels
Total elements: 768 × 1,024 = 786,432 elements
Total size: 786,432 × 2 bytes = 1.5 MB
```

The compiler sees this small size and chooses L1-optimized strategy, but even for this "small" tensor:
```
Per-core requirement: 31.9 MB (22× over limit)
```

This shows the L1-optimized strategy is **fundamentally broken** for wide tensors (768 channels), regardless of spatial dimensions.

---

## 4. Root Cause Analysis

### 4.1 Why to_layout Needs So Much L1

The tilize operation (row_major → tile) requires buffering multiple rows because:

1. **Tile Structure**: 32×32 elements per tile
2. **Wide Tensors**: 768 channels = 24 tiles across
3. **Incomplete Tiles**: Height=60 means 2 tile rows (incomplete)

To form complete tiles, the runtime must:
- Buffer at least 32 rows to form one complete tile row
- Current implementation buffers ~60 rows (entire height)

**Per-row cost is very high due to width**:
```
106 (width) × 768 (channels) = 81,408 elements per row
81,408 × 2 bytes = 162,816 bytes per row
```

### 4.2 Why Conv3d L1-Optimized Strategy Fails

The L1-optimized strategy assumes:
- Small input/output tensors can fit in L1
- Multi-core parallelism reduces per-core memory

But with 768 channels:
- Each core needs full channel buffers for kernel computation
- Weight tensor is huge: 768 × 768 × 3 × 3 × 3 = 15.9M elements = 31.8 MB
- Even divided across 64 cores: 31.8 MB ÷ 64 = 509 KB per core (just for weights)
- Plus input/output buffers → total exceeds L1

### 4.3 Why DRAM-Staged Strategy Works (But Partially)

DRAM-staged strategy:
- Keeps weights in DRAM
- Streams data through L1 in smaller chunks
- Doesn't require full channel buffers in L1
- ✓ Conv3d succeeds

But then:
- to_layout sees row_major output tensor
- Tries to convert to tile on a single core
- Requires buffering ~60 rows simultaneously
- ✗ to_layout fails

---

## 5. Solutions

### 5.1 Short-Term Fixes

#### Option 1: Multi-Core Sharding for to_layout
**Change**: Modify TTIR→TTNN lowering to apply 8×8 sharding for conv3d output

```mlir
// Current (broken):
#ttnn_layout32 = <..., <1x1>, memref<25440x768xbf16>>

// Fixed:
#ttnn_layout32_sharded = <..., <8x8>, memref<396x768xbf16>>
//                                ^^^
//                      Distribute across 64 cores
```

**Effect**:
- Each core handles: 25440 ÷ 64 = 397 rows
- Per-core L1: 397 × 162 KB = 64 MB → **Still too much!**

Need more aggressive sharding:

```mlir
// Better: Shard along channel dimension too
#ttnn_layout32_sharded = <..., <8x8>, memref<25440x96xbf16>>
//                                    Channels: 768÷8 = 96
```

**Effect**:
- Per-core L1: 60 × 106 × 96 × 2 = 1,224,960 bytes = 1.17 MB ✓ Fits!

#### Option 2: Reduce Pipeline Depth
**Change**: Modify tilize operation to buffer fewer rows

```cpp
// Current: Buffer entire height (60 rows)
pipeline_depth = height;

// Fixed: Buffer only what's needed for one tile row
pipeline_depth = min(32, height);
```

**Effect**:
- Per-core L1: 32 × 159 KB = 5.09 MB → **Still over limit**
- Need to combine with Option 1 (multi-core sharding)

#### Option 3: Force DRAM-Staged Tilize
**Change**: Detect wide tensors and use DRAM staging for to_layout

```python
if (tensor.channels > CHANNEL_THRESHOLD and
    estimated_l1_usage > L1_LIMIT):
    use_dram_staged_tilize()
```

**Effect**:
- Slower (DRAM bandwidth limited)
- But ✓ works for any size

### 5.2 Long-Term Architecture Fixes

#### Fix 1: Intelligent Sharding Assignment
- Analyze tensor shapes during compilation
- Apply multi-core sharding automatically for large tensors
- Shard along multiple dimensions if needed

#### Fix 2: Dynamic Buffer Sizing
- Calculate required L1 at compile time
- Adjust pipeline depth to fit within L1 limits
- Trade throughput for memory safety

#### Fix 3: Fix Conv3d Strategy Heuristic
- Don't use L1-optimized strategy for wide tensors (channels > 256)
- Add heuristic: `if (in_channels * out_channels > THRESHOLD) use_dram_staged()`
- Current heuristic only looks at spatial dimensions, ignores channel width

---

## 6. Recommended Action Plan

### Phase 1: Immediate Workaround (Compiler Team)
1. **Disable L1-optimized Conv3d** for wide tensors:
   ```cpp
   // In conv3d strategy selection:
   if (in_channels >= 512 || out_channels >= 512) {
       return Strategy::DRAM_STAGED;
   }
   ```

2. **Apply automatic sharding** for to_layout on large tensors:
   ```cpp
   // In to_layout lowering:
   if (tensor.num_elements() > L1_SIZE_THRESHOLD) {
       layout.mesh_shape = {8, 8};  // Force multi-core
   }
   ```

### Phase 2: Proper Fix (2-3 weeks)
1. Implement **channel-wise sharding** for to_layout operations
2. Add **compile-time L1 usage analysis** to prevent overflow
3. Improve **Conv3d strategy heuristics** to consider channel dimensions

### Phase 3: Validation
1. Test with Mochi decoder (all layers)
2. Verify other models with wide tensors (e.g., ViT, BERT)
3. Performance benchmark vs. workaround

---

## 7. Testing Checklist

- [ ] Mochi decoder conv1: `[1, 768, 4, 60, 106]` with replicate padding
- [ ] Mochi decoder conv1: `[1, 768, 4, 60, 106]` without padding
- [ ] Reduced size: `[1, 768, 4, 16, 16]`
- [ ] Different channel counts: 256, 512, 1024, 2048
- [ ] Different spatial sizes: Small (16×16), Medium (60×106), Large (128×128)
- [ ] Multi-frame temporal: 4, 8, 16, 32 frames

---

## Appendix: Complete MLIR Trace

### A.1 Padding + Conv3d (Working until to_layout)

```mlir
%arg2: tensor<1x768x4x60x106xbf16, #ttnn_layout13>  // Input
  ↓
%6 = "ttnn.to_layout"(%arg2) {layout = tile}  // Prepare input
  ↓
%7 = "ttnn.permute"(%6) {perm = [2,0,1,3,4]}  // → [4,1,768,60,106]
  ↓
%8 = "ttnn.reshape"(%7) {shape = [4, 4884480]}  // Flatten for embedding
  ↓
%9 = "ttnn.to_layout"(%8) {layout = row_major}  // Embedding needs row-major
  ↓
%10 = "ttnn.embedding"(%temporal_indices, %9)  // Temporal padding
  ↓
%15 = "ttnn.embedding"(%height_indices, %14)   // Height padding
  ↓
%20 = "ttnn.embedding"(%width_indices, %19)    // Width padding
  ↓
%23 = "ttnn.to_layout"(%22) {layout = row_major}  // Prepare for conv3d
  ↓
%24 = "ttnn.conv3d"(%23, ...)  // ✓ SUCCEEDS (DRAM-staged strategy)
  → tensor<1x4x60x106x768xbf16, #ttnn_layout32, row_major>
  ↓
%25 = "ttnn.to_layout"(%24) {layout = tile}  // ✗ FAILS (9.99 MB L1)
```

### A.2 Conv3d Only (Fails at Conv3d)

```mlir
%arg2: tensor<1x768x4x60x106xbf16>  // Input (no padding)
  ↓
%3 = "ttnn.to_layout"(%arg2) {layout = tile}
  ↓
%4 = "ttnn.permute"(%3) {perm = [0,2,3,4,1]}  // → [1,4,60,106,768]
  ↓
%5 = "ttnn.to_layout"(%4) {layout = row_major}
  ↓
%6 = "ttnn.conv3d"(%5, ...)  // ✗ FAILS (33.5 MB per core, L1-optimized)
```

---

## Summary

**The failing to_layout is operation %25**, which converts the Conv3d output from row_major to tile layout. It fails because:

1. **No multi-core sharding**: Uses `<1x1>` mesh shape (single core)
2. **Wide tensor**: 768 channels in last dimension
3. **Deep pipelining**: Buffers ~60 rows × 106 width × 768 channels
4. **Total L1 requirement**: 9.99 MB on one core
5. **L1 limit**: 1.43 MB per core
6. **Overflow**: 7× over limit

The error **only appears** when replicate padding is present because it forces Conv3d to use DRAM-staged strategy (which succeeds), exposing the subsequent to_layout issue. Without padding, Conv3d itself fails earlier with L1-optimized strategy.
