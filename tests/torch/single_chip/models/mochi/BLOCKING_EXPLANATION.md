# Blocking Parameters Explanation for Mochi Conv3D Tests

## What are Blocking Parameters?

Blocking parameters `(C_in_block, C_out_block, T_out_block, H_out_block, W_out_block)` control how the convolution computation is tiled/chunked on the Tenstorrent hardware for optimal performance.

## Constraints

### 1. **Channel Blocking** (`C_in_block`, `C_out_block`)
- Must be **multiples of 32** (hardware alignment requirement)
- Must **divide** the respective channel count
- **Larger is generally better** (more work per kernel launch)
- **Prefer 128** when possible for best efficiency

### 2. **Spatial Blocking** (`T_out_block`, `H_out_block`, `W_out_block`)
- Must be **powers of 2** (1, 2, 4, 8, 16, 32, ...)
- Must **divide** the respective output dimension
- **Total patches constraint**: `T_out_block * H_out_block * W_out_block ≤ 64`
- **Special constraint**: If `C_in_block == 128` OR `C_out_block == 128`, then patches ≤ 32

### 3. **Why These Constraints?**
- **Multiple of 32**: SRAM/L1 alignment on Tenstorrent hardware
- **Powers of 2**: Simplifies hardware indexing and memory access patterns
- **Patch limit**: Hardware resource constraints (registers, local memory)
- **128-channel limit**: Larger channel blocks need more resources, limiting spatial blocking

## Blocking Decisions for Mochi Decoder

| Config | Input Shape | Channels | Output Dims | Blocking | Patches | Reasoning |
|--------|------------|----------|-------------|----------|---------|-----------|
| conv_in | (1,12,7,60,106) | 12→768 | (7,60,106) | **(32,128,1,4,2)** | 8 | C_in=12 pads to 32; chose C_out=128 for efficiency; max spatial (4×2=8) |
| conv_768 | (1,768,9,62,108) | 768→768 | (7,60,106) | **(128,128,1,4,2)** | 8 | Both channels at 128 (max efficiency); limited to 32 patches but only 8 possible |
| conv_512 | (1,512,23,122,214) | 512→512 | (21,120,212) | **(128,128,1,8,4)** | 32 | Max channel blocks (128); max spatial at 32 patches (8×4) |
| conv_256_t22 | (1,256,22,242,426) | 256→256 | (20,240,424) | **(128,128,1,4,8)** | 32 | Max channel blocks; balanced H×W (4×8=32) |
| conv_256_t24 | (1,256,24,242,426) | 256→256 | (22,240,424) | **(128,128,1,4,8)** | 32 | Same as above; similar output dims |
| conv_128_t15 | (1,128,15,482,850) | 128→128 | (13,480,848) | **(128,128,1,2,16)** | 32 | Max channels; prefer W blocking (larger dim) |
| conv_128_t17 | (1,128,17,482,850) | 128→128 | (15,480,848) | **(128,128,1,2,16)** | 32 | Same as above |
| conv_128_t16 | (1,128,16,482,850) | 128→128 | (14,480,848) | **(128,128,1,2,16)** | 32 | Same as above |

## Strategy Used

### Priority Order:
1. **Maximize channel blocking** - Use 128 when possible (most work per block)
2. **Maximize spatial patches** - Within the constraint (32 or 64)
3. **Balance spatial dimensions** - Prefer blocking larger dimensions more

### Why This Approach?

**Channel blocking has the biggest performance impact:**
- More channels per block = fewer kernel launches
- 128 is the sweet spot (hardware optimized for this size)

**Spatial blocking helps but is constrained:**
- Can't exceed 32 patches when using 128-channel blocks
- Balance between dimensions based on their size

## Example: conv_768 (Most Common - 18 occurrences)

```
Input: (1, 768, 9, 62, 108)
Output after conv: (1, 768, 7, 60, 106)  # kernel 3x3x3, stride 1x1x1

Blocking: (128, 128, 1, 4, 2)
```

**Why this blocking?**
1. **C_in_block = 128**: 768/128 = 6 blocks (768 is divisible by 128, and 128 is max efficient)
2. **C_out_block = 128**: Same reasoning
3. **T_out_block = 1**: Output depth is 7, only 1, 2, or 4 work; chose 1 to allow more H×W
4. **H_out_block = 4**: 60/4 = 15 blocks (60 is divisible by 1, 2, 4)
5. **W_out_block = 2**: 106/2 = 53 blocks (106 is only divisible by 1, 2)
6. **Total patches = 1×4×2 = 8** ✅ (well under the limit of 32 for 128-channel blocks)

**Alternative considered but rejected:**
- `(128, 128, 1, 2, 2)` = 4 patches: Works but uses fewer patches (less parallelism)
- `(96, 96, 1, 4, 2)` = 8 patches: Smaller channel blocks = more kernel launches
- `(128, 128, 1, 2, 1)` = 2 patches: Too small, poor utilization

## Performance Expectations

**High-impact configs** (optimize these first):
1. **conv_768** - 18 occurrences! Most critical for overall performance
2. **conv_512** - 8 occurrences
3. **conv_256** variants - 6 occurrences each
4. **conv_128** variants - 6 occurrences each

**Why some configs have fewer patches:**
- Output dimensions don't divide evenly by larger powers of 2
- Example: W=106 only divides by 1, 2 (not 4, 8, 16...)
- Can't force larger blocks without violating divisibility constraint

## Testing the Configurations

Run the test:
```bash
pytest -v /localdev/vkovinic/tt-metal/tests/ttnn/nightly/unit_tests/operations/conv/test_conv3d.py::test_conv3d_mochi_shapes
```

If a configuration fails or performs poorly:
1. Check the output dims calculation
2. Try smaller patch counts (reduce H_out_block or W_out_block)
3. Try smaller channel blocks (96 or 64 instead of 128)
4. Use the sweep test to find optimal blocking empirically

## Notes on CI Skipping

Some tests skip on CI due to memory constraints:
```python
if is_ci_env and (out_channels == 128 or out_channels == 256):
    pytest.skip("Skipping test for 128/256 out channels on CI due to host OOM")
```

This affects configs 4-8 (all 256 and 128 channel convolutions).
