# Decomposing Conv3D with kernel=(3,3,3)

## The Problem

We need to replace:
```python
conv3d = nn.Conv3d(C, C, kernel_size=3, stride=1, padding=1)
```

With operations that TT backend supports (Conv2d, Conv1d, Linear).

---

## Mathematical Background

### What Conv3D Does

A 3D convolution with kernel=(3,3,3) computes:

```
output[b, c_out, t, h, w] = Σ Σ Σ Σ Σ
                            c_in kt kh kw
  weight[c_out, c_in, kt, kh, kw] × input[b, c_in, t+kt-1, h+kh-1, w+kw-1]
```

For each output position, it:
- Looks at a 3×3×3 neighborhood in input
- Across all input channels
- Computes weighted sum
- Produces one output value

**Key:** It captures **joint spatio-temporal correlations** - how time, height, and width interact together.

---

## Decomposition Approaches

### ❌ Option 1: Exact Decomposition (Impossible)

**Bad news:** There is NO exact decomposition of a general 3D convolution into 2D operations.

**Why?**
- 3D conv has C_out × C_in × 3 × 3 × 3 = C_out × C_in × 27 parameters
- Can learn arbitrary 27-dimensional filters
- Captures complex spatio-temporal interactions

**Example of what 3D conv can do that decomposed cannot:**
```
Detect: "A pixel moving diagonally up-right over 3 frames"
- Requires joint (t, h, w) kernel
- Cannot be factored into separate temporal and spatial operations
```

### ✅ Option 2: Factorized Approximation (Practical)

**Good news:** We can approximate with **sequential 1D temporal + 2D spatial** convolutions.

This is the approach used in many successful video models:
- P3D (Pseudo-3D) Networks
- R(2+1)D (ResNet 2+1D)
- Mixed-Separable Convolutions

---

## Proposed Decomposition: (1+2)D Approach

### Concept

Replace one Conv3d(C, C, kernel=3,3,3) with:
1. **Temporal Conv1d**: kernel=3 (processes time)
2. **Spatial Conv2d**: kernel=3×3 (processes space)

```
Input [B, C, T, H, W]
    ↓
Conv1d(C, C, kernel=3, padding=1)  ← Temporal
    ↓ [B, C, T, H, W]
Reshape: [B*T, C, H, W]
    ↓
Conv2d(C, C, kernel=3×3, padding=1)  ← Spatial
    ↓ [B*T, C, H, W]
Reshape: [B, C, T, H, W]
```

### Implementation

```python
import torch
import torch.nn as nn

class Conv3dDecomposed(nn.Module):
    """
    Approximates Conv3d(C, C, kernel=3, stride=1, padding=1)
    using Conv1d + Conv2d
    """
    def __init__(self, channels, bias=True):
        super().__init__()

        # Temporal convolution: processes time dimension
        # Input: [B, C, T, H, W] → need to reshape for Conv1d
        # Conv1d expects: [B, C, T]
        self.temporal_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False  # Bias only on spatial conv
        )

        # Spatial convolution: processes height and width
        # Input: [B, C, H, W]
        self.spatial_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape

        # === TEMPORAL CONVOLUTION ===
        # Reshape to process time dimension with Conv1d
        # Conv1d needs: [B, C, T]
        # We have multiple spatial positions, so:
        # [B, C, T, H, W] → [B*H*W, C, T]
        x_temp = x.permute(0, 3, 4, 1, 2)  # [B, H, W, C, T]
        x_temp = x_temp.reshape(B * H * W, C, T)  # [B*H*W, C, T]

        # Apply temporal convolution
        x_temp = self.temporal_conv(x_temp)  # [B*H*W, C, T]

        # Reshape back to 5D
        x_temp = x_temp.reshape(B, H, W, C, T)  # [B, H, W, C, T]
        x = x_temp.permute(0, 3, 4, 1, 2)  # [B, C, T, H, W]

        # === SPATIAL CONVOLUTION ===
        # Reshape to process spatial dimensions with Conv2d
        # Conv2d needs: [B, C, H, W]
        # We have multiple time steps, so:
        # [B, C, T, H, W] → [B*T, C, H, W]
        x_spat = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        x_spat = x_spat.reshape(B * T, C, H, W)  # [B*T, C, H, W]

        # Apply spatial convolution
        x_spat = self.spatial_conv(x_spat)  # [B*T, C, H, W]

        # Reshape back to 5D
        x_spat = x_spat.reshape(B, T, C, H, W)  # [B, T, C, H, W]
        x = x_spat.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

        return x
```

---

## Parameter Comparison

### Original Conv3D:
```
Parameters = C_out × C_in × kt × kh × kw
           = C × C × 3 × 3 × 3
           = C² × 27

Example: C=128
Parameters = 128² × 27 = 442,368
```

### Decomposed (1+2)D:
```
Temporal Conv1d: C × C × 3 = C² × 3
Spatial Conv2d:  C × C × 3 × 3 = C² × 9
Total:           C² × (3 + 9) = C² × 12

Example: C=128
Parameters = 128² × 12 = 196,608

Reduction: 27 → 12 = 2.25× fewer parameters!
```

**Trade-off:**
- ✅ Fewer parameters (2.25× reduction)
- ✅ Faster computation
- ❌ Less expressive (cannot capture full 3D correlations)

---

## Why This Approximation Works

### 1. Temporal and Spatial Features are Somewhat Separable

In video:
- **Temporal features**: Motion, velocity, acceleration
- **Spatial features**: Textures, edges, objects

Many video patterns can be captured by processing time and space separately.

### 2. Proven in Research

This decomposition is used successfully in:

**R(2+1)D** (CVPR 2018):
- Replaces 3D convolutions with 2D spatial + 1D temporal
- Achieves comparable accuracy to full 3D CNNs
- Faster training and inference

**P3D** (ICCV 2017):
- Pseudo-3D networks
- Decomposes 3D convolutions
- State-of-art results on video recognition

### 3. Can Be Trained End-to-End

Even though it's an approximation, models can be:
- Initialized with decomposed weights (see below)
- Fine-tuned end-to-end
- Learn to compensate for approximation

---

## Converting Pre-trained Weights

### Challenge

Mochi's pre-trained weights are for full Conv3d. How do we initialize the decomposed version?

### Approach 1: SVD Decomposition

Use Singular Value Decomposition to factor the 3D kernel:

```python
def decompose_conv3d_weights(conv3d_weight):
    """
    Decompose Conv3d weights into Conv1d + Conv2d weights

    Args:
        conv3d_weight: [C_out, C_in, 3, 3, 3]

    Returns:
        temporal_weight: [C_out, C_in, 3]
        spatial_weight: [C_out, C_in, 3, 3]
    """
    C_out, C_in, kt, kh, kw = conv3d_weight.shape

    # Reshape to 2D matrix: [C_out × C_in, 3 × 3 × 3]
    W = conv3d_weight.reshape(C_out * C_in, kt * kh * kw)

    # SVD: W ≈ U @ S @ V^T
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)

    # Take top k singular values (rank-k approximation)
    k = min(C_out * C_in, kt * kh * kw)
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    # Reconstruct as temporal × spatial
    # This is a rough approximation; better methods exist

    # Temporal part: average over spatial dimensions
    temporal_kernel = conv3d_weight.mean(dim=(3, 4))  # [C_out, C_in, 3]

    # Spatial part: average over temporal dimension
    spatial_kernel = conv3d_weight.mean(dim=2)  # [C_out, C_in, 3, 3]

    return temporal_kernel, spatial_kernel
```

### Approach 2: Tucker Decomposition

More principled tensor decomposition:

```python
# Using tensorly library
import tensorly as tl
from tensorly.decomposition import tucker

def tucker_decompose_conv3d(conv3d_weight):
    """
    Use Tucker decomposition for Conv3d weights
    """
    # Tucker decomposition of 5D tensor
    core, factors = tucker(conv3d_weight, rank=[C_out, C_in, 1, 3, 3])

    # Extract temporal and spatial components
    # (Implementation details depend on desired factorization)
    pass
```

### Approach 3: Simple Averaging (Practical)

For quick initialization:

```python
def simple_decompose(conv3d_module):
    """
    Simple weight decomposition for initialization
    """
    weight = conv3d_module.weight.data  # [C_out, C_in, 3, 3, 3]

    # Temporal weights: average over spatial dims
    temporal_weight = weight.mean(dim=(3, 4))  # [C_out, C_in, 3]

    # Spatial weights: average over temporal dim
    spatial_weight = weight.mean(dim=2)  # [C_out, C_in, 3, 3]

    # Renormalize to preserve overall scale
    temporal_weight = temporal_weight * 1.5  # Heuristic scaling
    spatial_weight = spatial_weight * 1.5

    return temporal_weight, spatial_weight
```

---

## Handling Causal Convolution

CogVideoXCausalConv3d uses **causal padding** (only past frames, not future).

### Original:
```python
# Causal padding: pad before time, not after
pad = (0, 0, 0, 0, 2, 0)  # (W, W, H, H, T_before, T_after)
x = F.pad(x, pad, mode='replicate')
x = self.conv3d(x)  # No padding in conv itself
```

### Decomposed with Causality:
```python
class CausalConv3dDecomposed(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # Temporal: causal Conv1d (padding=2, then truncate)
        self.temporal_conv = nn.Conv1d(
            channels, channels,
            kernel_size=3,
            padding=0,  # Manual causal padding
            bias=False
        )

        # Spatial: standard Conv2d
        self.spatial_conv = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=1,
            bias=True
        )

    def forward(self, x):
        B, C, T, H, W = x.shape

        # === TEMPORAL (CAUSAL) ===
        x_temp = x.permute(0, 3, 4, 1, 2).reshape(B*H*W, C, T)

        # Causal padding: pad left (past), not right (future)
        x_temp = F.pad(x_temp, (2, 0), mode='replicate')  # [B*H*W, C, T+2]
        x_temp = self.temporal_conv(x_temp)  # [B*H*W, C, T]

        x_temp = x_temp.reshape(B, H, W, C, T)
        x = x_temp.permute(0, 3, 4, 1, 2)

        # === SPATIAL ===
        x_spat = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        x_spat = self.spatial_conv(x_spat)
        x_spat = x_spat.reshape(B, T, C, H, W)
        x = x_spat.permute(0, 2, 1, 3, 4)

        return x
```

---

## Testing the Decomposition

### Test Script

```python
import torch
import torch.nn as nn

def test_decomposition():
    """
    Compare Conv3d vs Decomposed version
    """
    C = 64
    B, T, H, W = 2, 8, 32, 32

    # Create input
    x = torch.randn(B, C, T, H, W)

    # Original Conv3d
    conv3d = nn.Conv3d(C, C, kernel_size=3, padding=1)

    # Decomposed version
    conv_decomposed = Conv3dDecomposed(C)

    # Forward pass
    with torch.no_grad():
        out_original = conv3d(x)
        out_decomposed = conv_decomposed(x)

    print(f"Original output: {out_original.shape}")
    print(f"Decomposed output: {out_decomposed.shape}")
    print(f"Output range - Original: [{out_original.min():.3f}, {out_original.max():.3f}]")
    print(f"Output range - Decomposed: [{out_decomposed.min():.3f}, {out_decomposed.max():.3f}]")

    # They won't match exactly (different weights), but shapes should match
    assert out_original.shape == out_decomposed.shape
    print("✓ Shapes match!")

    # Check if it runs without errors
    print("✓ Decomposition works!")

if __name__ == "__main__":
    test_decomposition()
```

---

## Implementation Plan for Mochi

### Step 1: Replace conv_in (Point-wise)

```python
# Already solved - use Linear
```

### Step 2: Replace CogVideoXCausalConv3d (Spatial-temporal)

```python
class CogVideoXCausalConv3dDecomposed(nn.Module):
    """
    Drop-in replacement for CogVideoXCausalConv3d
    Uses Conv1d + Conv2d instead of Conv3d
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, pad_mode="replicate"):
        super().__init__()

        # Only support kernel_size=3 for now
        assert kernel_size == 3, "Only kernel_size=3 supported"
        assert stride == 1, "Only stride=1 supported"

        self.pad_mode = pad_mode

        # Temporal (causal)
        self.temporal_conv = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=3,
            stride=1,
            padding=0,  # Manual causal padding
            bias=False
        )

        # Spatial
        self.spatial_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

    def forward(self, x, cache=None):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape

        # Temporal convolution (causal)
        x_t = x.permute(0, 3, 4, 1, 2).reshape(B*H*W, C, T)
        x_t = F.pad(x_t, (2, 0), mode=self.pad_mode)
        x_t = self.temporal_conv(x_t)
        x_t = x_t.reshape(B, H, W, C, T).permute(0, 3, 4, 1, 2)

        # Spatial convolution
        x_s = x_t.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
        x_s = self.spatial_conv(x_s)
        x_s = x_s.reshape(B, T, -1, H, W).permute(0, 2, 1, 3, 4)

        return x_s, cache
```

### Step 3: Monkey-patch Decoder

```python
def convert_decoder_to_decomposed(decoder):
    """
    Replace all Conv3d in decoder with decomposed versions
    """
    import diffusers.models.autoencoders.autoencoder_kl_mochi as mochi_module

    # Replace the class definition
    mochi_module.CogVideoXCausalConv3d = CogVideoXCausalConv3dDecomposed

    # Reload decoder
    # (Or manually replace each conv layer)
```

---

## Limitations and Trade-offs

### What We Lose:

1. **Expressive Power:**
   - Cannot capture full 3D correlations
   - Some video patterns may be missed

2. **Exact Equivalence:**
   - Outputs will differ from original model
   - Need to validate quality

3. **Pre-trained Weights:**
   - Cannot directly use original weights
   - Need approximation or fine-tuning

### What We Gain:

1. **TT Backend Compatibility:**
   - Uses only Conv1d + Conv2d
   - Should work on TT hardware

2. **Efficiency:**
   - 2.25× fewer parameters
   - Faster computation

3. **Proven Approach:**
   - Used successfully in video models
   - R(2+1)D, P3D show this works

---

## Alternative: Unfold + Linear

Another approach is to use `unfold` to extract patches:

```python
class Conv3dAsUnfoldLinear(nn.Module):
    """
    Implement Conv3d using unfold + linear
    Most general but may be slower
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Linear layer: (in_channels * k^3) → out_channels
        self.linear = nn.Linear(
            in_channels * kernel_size ** 3,
            out_channels
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        k = self.kernel_size

        # Manually extract 3D patches
        patches = []
        for t in range(T):
            for h in range(H):
                for w in range(W):
                    # Extract k×k×k patch centered at (t,h,w)
                    # With padding
                    patch = extract_patch(x, t, h, w, k)  # [B, C*k^3]
                    patches.append(patch)

        # Stack all patches
        patches = torch.stack(patches, dim=1)  # [B, T*H*W, C*k^3]

        # Apply linear
        out = self.linear(patches)  # [B, T*H*W, C_out]

        # Reshape back
        out = out.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3)

        return out
```

This is more general but potentially slower and uses more memory.

---

## Recommendation

**Use the (1+2)D decomposition approach:**

1. ✅ Proven in research (R(2+1)D, P3D)
2. ✅ Efficient (2.25× parameter reduction)
3. ✅ Uses only Conv1d + Conv2d (TT compatible)
4. ⚠️ Requires validation of output quality
5. ⚠️ May need fine-tuning for best results

**Next steps:**
1. Implement `Conv3dDecomposed` and `CausalConv3dDecomposed`
2. Test on TT backend (Conv1d and Conv2d support)
3. Replace Mochi decoder convolutions
4. Validate output quality vs original
5. Fine-tune if needed

---

## References

- **R(2+1)D**: "A Closer Look at Spatiotemporal Convolutions for Action Recognition" (CVPR 2018)
- **P3D**: "Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks" (ICCV 2017)
- **Separable Convolutions**: "Rethinking the Inception Architecture for Computer Vision" (CVPR 2016)
