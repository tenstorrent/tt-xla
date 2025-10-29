# All Conv3D Operations in Mochi Decoder

## Summary

The Mochi decoder uses **TWO types** of Conv3D operations:

1. **Point-wise Conv3d** - `kernel_size=(1,1,1)` - Used ONCE at the input
2. **Spatial-temporal Conv3d** - `kernel_size=(3,3,3)` - Used MANY times in ResNet blocks

---

## Complete Inventory

### 1. Input Convolution (Point-wise)

**Location:** `MochiDecoder3D.__init__()`

```python
self.conv_in = nn.Conv3d(
    in_channels=12,           # Latent channels
    out_channels=768,         # block_out_channels[-1]
    kernel_size=(1, 1, 1)     # ⚠️ Point-wise (no spatial/temporal kernel)
)
```

**Quantity:** 1 operation

**Purpose:** Channel projection from 12 latent channels to 768 feature channels

**Can be replaced?** ✅ YES - This is just a learned linear combination per spatial-temporal location

---

### 2. ResNet Block Convolutions (Spatial-temporal)

**Location:** `MochiResnetBlock3D.__init__()`

```python
self.conv1 = CogVideoXCausalConv3d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=3,              # ⚠️ Becomes (3, 3, 3)
    stride=1,
    pad_mode="replicate"
)

self.conv2 = CogVideoXCausalConv3d(
    in_channels=out_channels,
    out_channels=out_channels,
    kernel_size=3,              # ⚠️ Becomes (3, 3, 3)
    stride=1,
    pad_mode="replicate"
)
```

**Quantity:** 2 per ResNet block

**Purpose:**
- Extract spatial-temporal features
- Process 3×3×3 neighborhoods (temporal + spatial context)
- Maintain temporal causality (causal padding)

**Can be replaced?** ⚠️ DIFFICULT - Need proper 3D decomposition

---

## Decoder Architecture Breakdown

```
MochiDecoder3D
├─ conv_in: Conv3d(12, 768, kernel=(1,1,1))           ← 1× Conv3d
│
├─ block_in: MochiMidBlock3D
│   └─ resnets: 3× MochiResnetBlock3D
│       ├─ conv1: CogVideoXCausalConv3d(kernel=3)     ← 3× Conv3d
│       └─ conv2: CogVideoXCausalConv3d(kernel=3)     ← 3× Conv3d
│
├─ up_blocks: ModuleList
│   ├─ [0] MochiUpBlock3D
│   │   └─ resnets: 6× MochiResnetBlock3D
│   │       ├─ conv1: CogVideoXCausalConv3d(kernel=3) ← 6× Conv3d
│   │       └─ conv2: CogVideoXCausalConv3d(kernel=3) ← 6× Conv3d
│   │
│   ├─ [1] MochiUpBlock3D
│   │   └─ resnets: 4× MochiResnetBlock3D
│   │       ├─ conv1: CogVideoXCausalConv3d(kernel=3) ← 4× Conv3d
│   │       └─ conv2: CogVideoXCausalConv3d(kernel=3) ← 4× Conv3d
│   │
│   └─ [2] MochiUpBlock3D
│       └─ resnets: 3× MochiResnetBlock3D
│           ├─ conv1: CogVideoXCausalConv3d(kernel=3) ← 3× Conv3d
│           └─ conv2: CogVideoXCausalConv3d(kernel=3) ← 3× Conv3d
│
└─ block_out: MochiMidBlock3D
    └─ resnets: 3× MochiResnetBlock3D
        ├─ conv1: CogVideoXCausalConv3d(kernel=3)     ← 3× Conv3d
        └─ conv2: CogVideoXCausalConv3d(kernel=3)     ← 3× Conv3d
```

---

## Count Summary

```
┌─────────────────────────────────────────────────────────────┐
│               Conv3D Operation Count                        │
└─────────────────────────────────────────────────────────────┘

Point-wise (kernel=1,1,1):
  conv_in:                                    1 operation

Spatial-temporal (kernel=3,3,3):
  block_in (3 ResNet × 2 conv):               6 operations
  up_blocks[0] (6 ResNet × 2 conv):          12 operations
  up_blocks[1] (4 ResNet × 2 conv):           8 operations
  up_blocks[2] (3 ResNet × 2 conv):           6 operations
  block_out (3 ResNet × 2 conv):              6 operations
                                        ─────────────────
                                 Total:      39 operations

GRAND TOTAL: 40 Conv3D operations in decoder
```

---

## Kernel Size Breakdown

| Kernel Size | Count | Percentage | Type |
|-------------|-------|------------|------|
| (1, 1, 1)   | 1     | 2.5%       | Point-wise |
| (3, 3, 3)   | 39    | 97.5%      | Spatial-temporal |
| **Total**   | **40**| **100%**   | |

**Key Finding:** Only 2.5% of Conv3D operations are point-wise!

---

## CogVideoXCausalConv3d Details

### What it does:

```python
# Internally wraps CogVideoXSafeConv3d which inherits from nn.Conv3d
class CogVideoXCausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, ...):
        # Convert scalar to 3-tuple
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        # Create internal Conv3d
        self.conv = CogVideoXSafeConv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=(stride, 1, 1),     # Only temporal stride
            dilation=(1, 1, 1),
            padding=0                   # Causal padding handled separately
        )
```

### Key Features:

1. **Causal Padding:** Pads temporally before convolution (only past frames, not future)
2. **Memory Safety:** Chunks large tensors along temporal dimension if > 2GB
3. **Temporal Stride Only:** Spatial dimensions always have stride=1

### For kernel_size=3:

```
Actual kernel: (3, 3, 3)
├─ Temporal:   3 frames (past, current, future after causal padding)
├─ Height:     3 pixels
└─ Width:      3 pixels

Total receptive field: 3 × 3 × 3 = 27 positions
```

---

## Implications for TT Backend Support

### Easy Case (2.5% of ops):

**conv_in with kernel=(1,1,1):**

```python
# Original:
conv3d = nn.Conv3d(12, 768, kernel_size=(1, 1, 1))
out = conv3d(x)  # [B, 12, T, H, W] → [B, 768, T, H, W]

# Equivalent replacement:
linear = nn.Linear(12, 768)
x_perm = x.permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]
out_perm = linear(x_perm)
out = out_perm.permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]
```

**Workaround:** Replace with `nn.Linear` + permute operations

---

### Hard Case (97.5% of ops):

**CogVideoXCausalConv3d with kernel=(3,3,3):**

These operations:
- Process 3×3×3 neighborhoods
- Extract spatial-temporal features
- Maintain temporal causality
- Cannot be trivially replaced with Linear

**Potential approaches:**

#### Option 1: Decompose into 1D temporal + 2D spatial

```python
# Instead of: Conv3d(C, C, kernel=(3,3,3))
# Use:
temporal_conv = Conv1d(C, C, kernel=3, padding=1)  # Temporal
spatial_conv = Conv2d(C, C, kernel=(3,3), padding=1)  # Spatial

# Process:
# 1. Reshape: [B, C, T, H, W] → [B*H*W, C, T]
# 2. Temporal conv: [B*H*W, C, T] → [B*H*W, C, T]
# 3. Reshape back: [B*H*W, C, T] → [B, C, T, H, W]
# 4. Reshape: [B, C, T, H, W] → [B*T, C, H, W]
# 5. Spatial conv: [B*T, C, H, W] → [B*T, C, H, W]
# 6. Reshape back: [B*T, C, H, W] → [B, C, T, H, W]
```

**Challenge:** Not mathematically equivalent to 3D conv (different computation order)

#### Option 2: Multiple 2D convolutions per time step

```python
# Process each time step separately with larger receptive field
# Use padding to include neighboring time steps
```

**Challenge:** More complex, needs careful temporal handling

#### Option 3: Request Conv3D support from TT team

**Best long-term solution:** Add Conv3d → Conv2d decomposition in tt-mlir

---

## Weight Statistics

For a single `CogVideoXCausalConv3d(C_in=128, C_out=128, kernel=3)`:

```
Parameters:
  Weights: 128 × 128 × 3 × 3 × 3 = 442,368 params
  Bias:    128 params
  Total:   442,496 params

Memory (bf16): 442,496 × 2 bytes = ~885 KB per layer
```

For all 39 operations (varying channel counts):
- Total parameters: ~100-200M (significant portion of 240M decoder)
- These are the computationally expensive operations

---

## Test Strategy

### Phase 1: Test Point-wise Conv3d (EASY)
✅ Already done - `single_ops.py` reproduces the failure

### Phase 2: Test Linear Replacement
```python
# Test if replacing conv_in with Linear works
class MochiConv3dAsLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(12, 768)

    def forward(self, x):
        # x: [B, 12, T, H, W]
        x = x.permute(0, 2, 3, 4, 1)  # [B, T, H, W, 12]
        x = self.linear(x)             # [B, T, H, W, 768]
        x = x.permute(0, 4, 1, 2, 3)  # [B, 768, T, H, W]
        return x
```

### Phase 3: Test Spatial-temporal Conv3d (HARD)
```python
# Test CogVideoXCausalConv3d with kernel=3
# Will need Conv3d decomposition or TT backend support
```

---

## Recommended Path Forward

### Short-term (Get decoder running):

1. **Replace conv_in** (1 operation):
   - Point-wise Conv3d → Linear + permute
   - Should work on TT backend
   - Validate output matches original

2. **Block on ResNet blocks** (39 operations):
   - These will still fail
   - Cannot proceed without:
     a) Conv3d decomposition in tt-mlir, OR
     b) Conv3d support in TT backend

### Long-term (Full support):

1. **Add Conv3d decomposition pass** in tt-mlir:
   - Detect Conv3d operations
   - Decompose to Conv1d + Conv2d
   - Preserve semantics as much as possible

2. **Or: Native Conv3d support** in TTNN:
   - Implement ttnn.conv3d operation
   - Add lowering patterns
   - Full solution but more work

---

## Files to Modify for Workaround

### For testing Linear replacement:

1. **Create wrapper:**
   ```
   tests/torch/single_chip/models/mochi/conv3d_workaround.py
   ```

2. **Test isolated conv_in:**
   ```python
   # Load decoder
   # Replace decoder.conv_in with Linear version
   # Test forward pass
   ```

3. **Validate outputs match:**
   ```python
   # Compare original Conv3d vs Linear replacement
   # Should be mathematically identical for kernel=(1,1,1)
   ```

---

## Conclusion

**The Mochi decoder has 40 Conv3d operations:**
- 1 point-wise (can be replaced with Linear)
- 39 spatial-temporal (need proper Conv3d support)

**To run the decoder on TT backend, we need BOTH:**
1. Replace the point-wise conv_in (workaround)
2. Add Conv3d decomposition or native support (proper fix)

**The 39 spatial-temporal Conv3d operations are the real blocker.** They represent the core computational work of the decoder and cannot be easily worked around.
