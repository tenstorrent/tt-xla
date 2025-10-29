# Mochi Decoder Conv3D Issue Analysis

## Problem Summary

**Error:**
```
loc("Conv3d[conv_in]/forward(autoencoder_kl_mochi.py:623)/aten__convolution_overrideable"): error: failed to legalize operation 'ttir.convolution' that was explicitly marked illegal
```

**Root Cause:** The TT backend does not support 3D convolution operations (`nn.Conv3d`), which are essential to the Mochi decoder architecture.

---

## Detailed Analysis

### 1. Conv3d Usage in Mochi Decoder

#### Primary Conv3d (Failing):
```python
# In MochiDecoder3D.__init__()
self.conv_in = nn.Conv3d(
    in_channels=12,           # Latent channels
    out_channels=768,         # block_out_channels[-1]
    kernel_size=(1, 1, 1),    # Point-wise convolution
    stride=(1, 1, 1),
    padding=(0, 0, 0)
)

# In forward():
hidden_states = self.conv_in(hidden_states)  # [B, 12, t, h, w] → [B, 768, t, h, w]
```

**This is the first operation in the decoder and it's failing immediately.**

#### Secondary Conv3d (Also uses Conv3d):
All other convolutions use `CogVideoXCausalConv3d`, which internally:
```python
CogVideoXCausalConv3d
  └─> CogVideoXSafeConv3d (inherits from nn.Conv3d)
      └─> nn.Conv3d (base PyTorch 3D convolution)
```

So **ALL** convolution operations in the decoder are 3D convolutions.

### 2. Why Conv3d?

Video data has 3 dimensions: `[C, T, H, W]`
- `T` = temporal (time/frames)
- `H, W` = spatial (height, width)

3D convolutions process all three dimensions simultaneously:
- Temporal patterns: motion, changes over time
- Spatial patterns: objects, textures within frames
- Spatio-temporal patterns: moving objects

### 3. TT Backend Limitation

The error message indicates:
```
failed to legalize operation 'ttir.convolution' that was explicitly marked illegal
```

This means the TTIR → TTNN conversion **explicitly marks 3D convolutions as unsupported**.

---

## Investigation Plan

### Step 1: ✓ Confirm Conv3d Parameters
**Status:** COMPLETE

**Findings:**
- First Conv3d: `(12 → 768, kernel=(1,1,1))`
- Point-wise convolution (no spatial/temporal kernel)
- Used for channel projection only
- All subsequent blocks also use Conv3d variants

### Step 2: 🔄 Research TT Backend Support
**Status:** IN PROGRESS

**Questions to answer:**
1. Does TT backend support Conv2d?
2. Are there any Conv3d workarounds in tt-xla?
3. What's the recommended approach for video models?

### Step 3: ⏸ Check Conv3d Decomposition
**Status:** PENDING

**Options to explore:**

#### Option A: Decompose Conv3d into Conv2d
For a Conv3d with kernel `(kt, kh, kw)`:
```python
# Pseudo-code
# Original: Conv3d(C_in, C_out, kernel=(kt, kh, kw))

# Decomposed:
# Step 1: Temporal conv (1D along time)
temporal_conv = Conv1d(C_in, C_temp, kernel=kt)
# Step 2: Spatial conv (2D on H×W)
spatial_conv = Conv2d(C_temp, C_out, kernel=(kh, kw))
```

**Pros:**
- Conv2d likely supported by TT backend
- Maintains similar computation
- Can be mathematically equivalent

**Cons:**
- Requires model surgery
- Need to handle all CogVideoXCausalConv3d instances
- May break pre-trained weights

#### Option B: Replace with Linear Layers
For point-wise Conv3d (kernel=1):
```python
# Original:
conv = nn.Conv3d(12, 768, kernel_size=(1, 1, 1))
out = conv(x)  # [B, 12, T, H, W] → [B, 768, T, H, W]

# Equivalent:
linear = nn.Linear(12, 768)
# Reshape: [B, 12, T, H, W] → [B, T, H, W, 12]
x = x.permute(0, 2, 3, 4, 1)
out = linear(x)  # [B, T, H, W, 12] → [B, T, H, W, 768]
# Reshape back: [B, T, H, W, 768] → [B, 768, T, H, W]
out = out.permute(0, 4, 1, 2, 3)
```

**Pros:**
- Works for kernel_size=(1,1,1) cases
- No new operations needed
- Can extract/convert weights

**Cons:**
- Only works for point-wise convolutions
- Need to handle stride/padding for general case
- Doesn't solve CogVideoXCausalConv3d with larger kernels

#### Option C: Implement Custom Conv3d via Conv2d
Process temporal dimension separately:
```python
class Conv3dAs2D(nn.Module):
    """Implement Conv3d using Conv2d operations frame-by-frame"""
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3)):
        self.kt, self.kh, self.kw = kernel_size
        # Temporal convolution (1D)
        self.temporal_conv = nn.Conv1d(in_channels, out_channels, self.kt, padding=self.kt//2)
        # Spatial convolution (2D)
        self.spatial_conv = nn.Conv2d(out_channels, out_channels, (self.kh, self.kw), padding=(self.kh//2, self.kw//2))

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        # Temporal conv
        x = rearrange(x, 'b c t h w -> (b h w) c t')
        x = self.temporal_conv(x)
        x = rearrange(x, '(b h w) c t -> b c t h w', b=B, h=H, w=W)
        # Spatial conv (frame by frame)
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.spatial_conv(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', b=B, t=T)
        return x
```

**Pros:**
- Can handle arbitrary kernel sizes
- Uses only Conv1d/Conv2d
- More flexible

**Cons:**
- Not exact equivalent (different computation order)
- Need to convert pre-trained weights carefully
- More complex implementation

### Step 4: ⏸ Create Minimal Test Case
**Status:** PENDING

Create a minimal reproduction:
```python
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

xr.set_device_type("TT")
device = xm.xla_device()

# Minimal Conv3d test
conv3d = nn.Conv3d(12, 768, kernel_size=(1, 1, 1))
conv3d = conv3d.to(device)
conv3d = torch.compile(conv3d, backend="tt")

x = torch.randn(1, 12, 3, 32, 32).to(device)
out = conv3d(x)  # Should fail here
```

---

## Recommended Next Steps

### Immediate Actions:

1. **Test Conv2d support:**
   ```python
   # Test if Conv2d works on TT backend
   conv2d = nn.Conv2d(12, 768, kernel_size=(1, 1))
   conv2d = torch.compile(conv2d, backend="tt")
   ```

2. **Check tt-xla documentation/tests:**
   ```bash
   cd /localdev/vkovinic/tt-xla
   grep -r "Conv3d" tests/
   grep -r "conv3d" src/
   ```

3. **Create Conv3d → Linear wrapper for kernel=(1,1,1):**
   - This is the quickest workaround for `conv_in`
   - Can be done without modifying diffusers
   - Test if this allows decoder to proceed further

### Long-term Solutions:

1. **Implement Conv3d support in TT backend** (ideal but requires backend changes)
2. **Create custom Mochi decoder with Conv2d** (requires retraining)
3. **Decompose all Conv3d operations** (complex but feasible)

---

## Code Locations

### Failing Code:
- **File:** `diffusers/models/autoencoders/autoencoder_kl_mochi.py`
- **Class:** `MochiDecoder3D`
- **Line:** ~623 (in forward method)
- **Operation:** `self.conv_in(hidden_states)`

### Related Classes:
- `CogVideoXCausalConv3d` - autoencoder_kl_cogvideox.py
- `CogVideoXSafeConv3d` - autoencoder_kl_cogvideox.py (inherits nn.Conv3d)
- `MochiResnetBlock3D` - autoencoder_kl_mochi.py (uses CogVideoXCausalConv3d)
- `MochiUpBlock3D` - autoencoder_kl_mochi.py (uses CogVideoXCausalConv3d)

### Test Files:
- `/localdev/vkovinic/tt-xla/tests/torch/single_chip/models/mochi/decoder.py`
- `/localdev/vkovinic/tt-xla/tests/torch/single_chip/models/mochi/encoder.py`

---

## Technical Details

### Conv3d Dimensions:
```
Input:  [B, C_in,  T, H, W]
Kernel: [C_out, C_in, kt, kh, kw]
Output: [B, C_out, T', H', W']

Where:
  T' = (T + 2*pad_t - kt) / stride_t + 1
  H' = (H + 2*pad_h - kh) / stride_h + 1
  W' = (W + 2*pad_w - kw) / stride_w + 1
```

### For conv_in specifically:
```
Input:  [1, 12, 3, 32, 32]
Kernel: [768, 12, 1, 1, 1]
Output: [1, 768, 3, 32, 32]

Parameters: 768 * 12 * 1 * 1 * 1 + 768 (bias) = 9,984 params
```

This is a **point-wise convolution** - essentially a learned linear projection applied to each spatial-temporal location independently.

---

## Questions for Team

1. **Backend Support:**
   - Is Conv3d support planned for TT backend?
   - What's the timeline if it's on the roadmap?

2. **Alternative Approaches:**
   - Are there other video models working on TT?
   - What approach did they use?

3. **Testing:**
   - Should we focus on encoder (also uses Conv3d)?
   - Or work on transformer first (might not use Conv3d)?

---

## Status: 🔴 BLOCKED
**Blocker:** TT backend does not support Conv3d operations
**Next Action:** Investigate Conv2d decomposition or Linear replacement for point-wise Conv3d
